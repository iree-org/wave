"""OpenRouter LLM client and iterative scheduling loop for Conductor."""

import json
import os
import re
import sys
import time
from collections.abc import Callable
from typing import Any

import requests

API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL: str = "https://openrouter.ai/api/v1"
DEFAULT_MODEL: str = "deepseek/deepseek-v3.2"

_REQUEST_TIMEOUT = 120
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0


def _default_log(msg: str) -> None:
    """Default logger: print to stderr without trailing newline."""
    print(msg, file=sys.stderr, end="", flush=True)


def _noop_log(_msg: str) -> None:
    pass


# Valid move command pattern: move/swap with tag operands.
_MOVE_RE = re.compile(
    r"^(move\s+\S+\s+(?:before|after)\s+\S+|swap\s+\S+\s+\S+)$", re.IGNORECASE
)

SYSTEM_PROMPT = """\
You are an expert GPU instruction scheduler for AMD CDNA/RDNA architectures.

You will receive WaveASM MLIR IR with tagged instructions (loc("tag_name")).
Your job is to reorder instructions to:
1. Hide memory latency by interleaving loads with independent compute.
2. Reduce register pressure so the linear-scan allocator succeeds.
3. Minimize the number of s_waitcnt and s_nop instructions inserted by later passes.

Key latencies:
- Global loads (buffer_load): ~100 cycles.
- LDS loads (ds_read): ~20 cycles.
- MFMA (16x16): ~32 cycles, (32x32): ~64 cycles.

Rules:
- Issue move commands, one per line.
- Commands: "move TAG_A after TAG_B", "move TAG_A before TAG_B", "swap TAG_A TAG_B".
- Do NOT output anything else — no explanations, no markdown, just commands.
- Moves that break SSA dominance will be rejected; you will see the error.
- Pinned ops (s_endpgm, s_barrier, condition) cannot be moved.

Strategy tips:
- Move global loads earlier to start memory fetches sooner.
- Interleave LDS reads between MFMAs to hide LDS latency behind MFMA execution.
- Keep address computations close to their consumers.
- Avoid clustering same-type ops (all loads together, all MFMAs together).\
"""


def _chat(
    messages: list[dict[str, Any]],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    reasoning_effort: str | None = None,
    log: Callable[[str], None] = _default_log,
) -> str:
    """Send a chat completion request to OpenRouter. Returns content text."""
    if not API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Export it before running the LLM loop."
        )

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if reasoning_effort is not None:
        payload["reasoning"] = {"enabled": True, "effort": reasoning_effort}

    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=payload,
                stream=True,
                timeout=_REQUEST_TIMEOUT,
            )
            if resp.status_code >= 500:
                if attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_BACKOFF * (attempt + 1)
                    log(
                        f"\n  [retry] server {resp.status_code}, waiting {wait:.0f}s...\n"
                    )
                    time.sleep(wait)
                    continue
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"OpenRouter API error {resp.status_code}: {resp.text}"
                )

            # Stream and accumulate content + reasoning.
            content_chunks: list[str] = []
            reasoning_chunks: list[str] = []
            usage = None
            in_reasoning = False

            for line in resp.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                data = line[6:]
                if data == b"[DONE]":
                    break
                chunk = json.loads(data)
                if "usage" in chunk:
                    usage = chunk["usage"]
                choices = chunk.get("choices")
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                # Reasoning tokens (two OpenRouter formats).
                for detail in delta.get("reasoning_details", []):
                    text = detail.get("text", "")
                    if text:
                        if not in_reasoning:
                            log("\n  [thinking] ")
                            in_reasoning = True
                        log(text)
                        reasoning_chunks.append(text)
                rc = delta.get("reasoning_content", "")
                if rc:
                    if not in_reasoning:
                        log("\n  [thinking] ")
                        in_reasoning = True
                    log(rc)
                    reasoning_chunks.append(rc)

                # Content tokens.
                token = delta.get("content", "")
                if token:
                    if in_reasoning:
                        log("\n  [/thinking]\n")
                        in_reasoning = False
                    content_chunks.append(token)

            if in_reasoning:
                log("\n  [/thinking]\n")

            content = "".join(content_chunks)
            if usage:
                pt = usage.get("prompt_tokens", "?")
                ct = usage.get("completion_tokens", "?")
                log(f"\n  [tokens] prompt={pt} completion={ct}\n")
            return content

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ChunkedEncodingError,
        ):
            if attempt < _MAX_RETRIES - 1:
                wait = _RETRY_BACKOFF * (attempt + 1)
                log(f"\n  [retry] connection error, waiting {wait:.0f}s...\n")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("Unreachable")


def parse_commands(text: str) -> list[str]:
    """Parse move commands from LLM response. Returns list of command strings."""
    commands = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip markdown code fences if the model wraps output.
        if line.startswith("```"):
            continue
        if _MOVE_RE.match(line):
            commands.append(line)
    return commands


def format_prompt(
    tagged_ir: str,
    metrics: dict,
    round_num: int,
    error: str | None = None,
    target: str = "gfx942",
) -> str:
    """Format the user prompt for a scheduling round."""
    parts = [
        f"=== WaveASM Scheduling Round {round_num} ===",
        f"TARGET: {target} (wave64, 512 vgpr, 106 sgpr, 512 agpr)",
        "LATENCY: vmem=100, lds=20, mfma_16x16=32, mfma_32x32=64",
        "",
        "--- IR (tagged) ---",
        tagged_ir.strip(),
        "",
        "--- Metrics ---",
    ]
    for k, v in metrics.items():
        parts.append(f"  {k}: {v}")

    if error:
        parts.extend(["", "--- Error from previous round ---", error])

    parts.extend(
        [
            "",
            "GOAL: Minimize register pressure and hide memory latency.",
            "Respond with move commands, one per line.",
        ]
    )
    return "\n".join(parts)


def run_scheduling_loop(
    conductor,
    max_rounds: int = 5,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    reasoning_effort: str | None = "high",
    log: Callable[[str], None] = _default_log,
) -> dict:
    """
    Run the iterative LLM scheduling loop.

    Returns dict with keys: metrics, commands, rounds, baseline_metrics.
    """
    log("Computing baseline metrics...\n")
    baseline = conductor.baseline()
    log(f"  baseline: {baseline}\n")

    tagged_ir = conductor.tag()
    log(f"  tagged IR: {len(tagged_ir)} chars\n")

    best_metrics = dict(baseline)
    best_commands: list[str] = []
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    error: str | None = None

    for round_num in range(1, max_rounds + 1):
        log(f"\n--- Round {round_num}/{max_rounds} ---\n")

        prompt = format_prompt(
            tagged_ir, best_metrics, round_num, error=error, target=conductor.target
        )
        messages.append({"role": "user", "content": prompt})

        log("  Querying LLM...\n")
        response = _chat(
            messages,
            model=model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            log=log,
        )
        messages.append({"role": "assistant", "content": response})
        log(f"  Response:\n{response}\n")

        commands = parse_commands(response)
        if not commands:
            log("  No valid commands parsed, stopping.\n")
            break

        error = None
        try:
            metrics = conductor.evaluate(commands)
            log(f"  metrics: {metrics}\n")

            if _is_better(metrics, best_metrics):
                log("  Improvement found!\n")
                best_metrics = metrics
                best_commands = commands
            else:
                log("  No improvement, reverting.\n")
                error = (
                    f"Round {round_num} regressed metrics. "
                    f"Previous best: {best_metrics}, this round: {metrics}. "
                    "Moves reverted."
                )
        except RuntimeError as e:
            error_msg = str(e)
            log(f"  Error: {error_msg}\n")
            error = error_msg

    return {
        "metrics": best_metrics,
        "commands": best_commands,
        "rounds": round_num,
        "baseline_metrics": baseline,
    }


def _is_better(new: dict, old: dict) -> bool:
    """Compare metrics: lower VGPRs > fewer waitcnts > fewer nops > fewer instructions."""
    for key in ("peak_vgpr", "s_waitcnt", "s_nop", "total_instructions"):
        nv = new.get(key, 0)
        ov = old.get(key, 0)
        if nv < ov:
            return True
        if nv > ov:
            return False
    return False
