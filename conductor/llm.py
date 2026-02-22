"""OpenRouter LLM client and iterative scheduling loop for Conductor."""

import json
import os
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

Message = dict[str, Any]


def _default_log(msg: str) -> None:
    """Default logger: print to stderr without trailing newline."""
    print(msg, file=sys.stderr, end="", flush=True)


def _noop_log(_msg: str) -> None:
    pass


SYSTEM_PROMPT = """\
You are an expert GPU instruction scheduler for AMD CDNA/RDNA architectures.

You will receive WaveASM MLIR IR with tagged instructions (loc("tag_name")).
Your job is to reorder instructions to hide memory latency and reduce register pressure.

Key latencies: global loads ~100 cycles, LDS loads ~20 cycles, MFMA 16x16 ~32 cycles.

You have an `evaluate_moves` tool. Call it with a list of move command strings.
Each command is one of:
  "move TAG_A after TAG_B"
  "move TAG_A before TAG_B"
  "swap TAG_A TAG_B"

The tool will apply the moves, compile, and return metrics.

Constraints:
- Moves that break SSA dominance will be rejected.
- Pinned ops (s_endpgm, s_barrier, condition) cannot be moved.

Work incrementally: try 1-3 moves per tool call, read the resulting metrics, \
then decide your next moves. You can call the tool multiple times.\
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_moves",
            "description": (
                "Apply a list of move/swap commands to the tagged IR, "
                "compile through the post-scheduling pipeline, and return "
                "assembly metrics. Commands are applied in order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "moves": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of move commands, e.g. "
                            '["move tag_A after tag_B", "swap tag_C tag_D"].'
                        ),
                    }
                },
                "required": ["moves"],
            },
        },
    }
]


def _chat(
    messages: list[Message],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    reasoning_effort: str | None = None,
    tools: list[dict] | None = None,
    log: Callable[[str], None] = _default_log,
) -> Message:
    """Send a streaming chat completion request. Returns the full response message."""
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
    if tools:
        payload["tools"] = tools

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

            # Stream and accumulate content, reasoning, and tool calls.
            content_chunks: list[str] = []
            reasoning_chunks: list[str] = []
            tool_calls_by_index: dict[int, dict[str, Any]] = {}
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

                # Tool call deltas.
                for tc_delta in delta.get("tool_calls", []):
                    idx = tc_delta["index"]
                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = {
                            "id": tc_delta.get("id", ""),
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    tc = tool_calls_by_index[idx]
                    func_delta = tc_delta.get("function", {})
                    if func_delta.get("name"):
                        tc["function"]["name"] += func_delta["name"]
                    if func_delta.get("arguments"):
                        tc["function"]["arguments"] += func_delta["arguments"]

            if in_reasoning:
                log("\n  [/thinking]\n")

            result: Message = {"role": "assistant", "content": "".join(content_chunks)}
            if tool_calls_by_index:
                result["tool_calls"] = [
                    tool_calls_by_index[i] for i in sorted(tool_calls_by_index)
                ]
            if usage:
                pt = usage.get("prompt_tokens", "?")
                ct = usage.get("completion_tokens", "?")
                log(f"\n  [tokens] prompt={pt} completion={ct}\n")
            return result

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


def format_initial_prompt(
    tagged_ir: str,
    baseline_metrics: dict,
    target: str = "gfx942",
) -> str:
    """Format the initial user message with IR and baseline metrics."""
    parts = [
        f"TARGET: {target} (wave64, 512 vgpr, 106 sgpr, 512 agpr)",
        "LATENCY: vmem=100, lds=20, mfma_16x16=32, mfma_32x32=64",
        "",
        "--- Tagged IR ---",
        tagged_ir.strip(),
        "",
        "--- Baseline Metrics ---",
    ]
    for k, v in baseline_metrics.items():
        parts.append(f"  {k}: {v}")
    parts.extend(
        [
            "",
            "GOAL: Minimize register pressure and hide memory latency.",
            "Use the evaluate_moves tool to try reorderings.",
        ]
    )
    return "\n".join(parts)


def run_scheduling_loop(
    conductor,
    max_rounds: int = 10,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    reasoning_effort: str | None = "medium",
    log: Callable[[str], None] = _default_log,
) -> dict:
    """
    Run the iterative LLM scheduling loop with tool use.

    The LLM reasons in natural language and calls evaluate_moves to test
    scheduling ideas. Conversation history (including reasoning) is preserved
    across rounds.

    Returns dict with keys: metrics, commands, rounds, baseline_metrics.
    """
    log("Computing baseline metrics...\n")
    baseline = conductor.baseline()
    log(f"  baseline: {baseline}\n")

    tagged_ir = conductor.tag()
    log(f"  tagged IR: {len(tagged_ir)} chars\n")

    best_metrics = dict(baseline)
    best_commands: list[str] = []

    messages: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": format_initial_prompt(
                tagged_ir, baseline, target=conductor.target
            ),
        },
    ]

    for round_num in range(1, max_rounds + 1):
        log(f"\n--- Round {round_num}/{max_rounds} ---\n")

        response = _chat(
            messages,
            model=model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tools=TOOLS,
            log=log,
        )
        messages.append(response)

        content = response.get("content", "")
        if content:
            log(f"  [model] {content}\n")

        tool_calls = response.get("tool_calls")
        if not tool_calls:
            log("  No tool call, model is done.\n")
            break

        # Process each tool call.
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                tool_result = {"error": "Malformed JSON arguments."}
                log(f"  [tool] {name}: malformed args\n")
            else:
                if name == "evaluate_moves":
                    moves = args.get("moves", [])
                    log(f"  [tool] evaluate_moves({moves})\n")
                    try:
                        metrics = conductor.evaluate(moves)
                        log(f"  [result] {metrics}\n")
                        if _is_better(metrics, best_metrics):
                            log("  Improvement!\n")
                            best_metrics = metrics
                            best_commands = moves
                        tool_result = {
                            "metrics": metrics,
                            "improved": _is_better(metrics, best_metrics)
                            or metrics == best_metrics,
                        }
                    except RuntimeError as e:
                        tool_result = {"error": str(e)}
                        log(f"  [error] {e}\n")
                else:
                    tool_result = {"error": f"Unknown tool: {name}"}
                    log(f"  [tool] unknown: {name}\n")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(tool_result),
                }
            )

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
