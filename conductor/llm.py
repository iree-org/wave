"""OpenRouter LLM client and iterative scheduling loop for Conductor."""

import json
import os
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import requests

API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL: str = "https://openrouter.ai/api/v1"
DEFAULT_MODEL: str = "deepseek/deepseek-v3.2"

_REQUEST_TIMEOUT = 120
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0

Message = dict[str, Any]


# --- Monotonic API usage counters (thread-safe, per-model). ---


@dataclass
class Counters:
    """API usage snapshot."""

    tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


_counters_lock = threading.Lock()
_counters: defaultdict[str, Counters] = defaultdict(Counters)


def _record_usage(model: str, usage: dict[str, Any]) -> None:
    """Accumulate token and cost from an API response."""
    tokens = int(usage.get("total_tokens", 0))
    input_tokens = int(usage.get("prompt_tokens", 0))
    output_tokens = int(usage.get("completion_tokens", 0))
    cost = usage.get("cost")
    with _counters_lock:
        c = _counters[model]
        c.tokens += tokens
        c.input_tokens += input_tokens
        c.output_tokens += output_tokens
        if cost is not None:
            c.cost += float(cost)


class Stats:
    """Context manager that captures API usage over a scope.

    Snapshots the monotonic per-model counters on entry. The ``counters``
    property returns an aggregate delta; ``per_model`` returns per-model deltas.
    """

    def __init__(self) -> None:
        self._start: dict[str, Counters] = {}

    def __enter__(self) -> "Stats":
        with _counters_lock:
            self._start = {
                m: Counters(c.tokens, c.input_tokens, c.output_tokens, c.cost)
                for m, c in _counters.items()
            }
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass

    @property
    def counters(self) -> Counters:
        """Aggregate delta across all models since entering the context."""
        with _counters_lock:
            tok = sum(c.tokens for c in _counters.values()) - sum(
                c.tokens for c in self._start.values()
            )
            inp = sum(c.input_tokens for c in _counters.values()) - sum(
                c.input_tokens for c in self._start.values()
            )
            out = sum(c.output_tokens for c in _counters.values()) - sum(
                c.output_tokens for c in self._start.values()
            )
            cst = sum(c.cost for c in _counters.values()) - sum(
                c.cost for c in self._start.values()
            )
        return Counters(tokens=tok, input_tokens=inp, output_tokens=out, cost=cst)

    @property
    def per_model(self) -> defaultdict[str, Counters]:
        """Per-model deltas since entering the context."""
        with _counters_lock:
            result: defaultdict[str, Counters] = defaultdict(Counters)
            for model, current in _counters.items():
                start = self._start.get(model, Counters())
                delta = Counters(
                    tokens=current.tokens - start.tokens,
                    input_tokens=current.input_tokens - start.input_tokens,
                    output_tokens=current.output_tokens - start.output_tokens,
                    cost=current.cost - start.cost,
                )
                if delta.tokens or delta.cost:
                    result[model] = delta
            return result


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
then decide your next moves. You can call the tool multiple times.

When you are satisfied with the schedule or have no more ideas, call the `done` \
tool to finish.\
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
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": (
                "Call this when you are finished scheduling. "
                "Provide a short summary of what you tried."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of scheduling attempts.",
                    }
                },
                "required": ["summary"],
            },
        },
    },
]


_TRANSIENT_ERRORS = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.ReadTimeout,
    requests.exceptions.ConnectTimeout,
)


def _with_retry(
    func: Callable[..., Any],
    log: Callable[[str], None],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call *func* with retries on transient network errors and 5xx."""
    for attempt in range(_MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except requests.HTTPError as exc:
            resp = exc.response
            if resp is not None and resp.status_code >= 500:
                if attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_BACKOFF * (attempt + 1)
                    log(
                        f"\n  [retry] server {resp.status_code}, waiting {wait:.0f}s...\n"
                    )
                    time.sleep(wait)
                    continue
            raise
        except _TRANSIENT_ERRORS:
            if attempt < _MAX_RETRIES - 1:
                wait = _RETRY_BACKOFF * (attempt + 1)
                log(f"\n  [retry] connection error, waiting {wait:.0f}s...\n")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Unreachable")


def _stream_request(
    payload: dict[str, Any],
    on_token: Callable[[str], None] | None,
    on_thinking: Callable[[str], None] | None,
) -> Message:
    """Execute a streaming chat request and assemble the response message."""
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=payload,
        stream=True,
        timeout=_REQUEST_TIMEOUT,
    )
    if resp.status_code >= 400:
        raise requests.HTTPError(
            f"{resp.status_code} {resp.reason}: {resp.text}",
            response=resp,
        )

    content_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] | None = None

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
        reasoning_texts: list[str] = []
        for detail in delta.get("reasoning_details", []):
            text = detail.get("text", "")
            if text:
                reasoning_texts.append(text)
        rc = delta.get("reasoning_content", "")
        if rc:
            reasoning_texts.append(rc)
        for text in reasoning_texts:
            if on_thinking is not None:
                on_thinking(text)
            reasoning_chunks.append(text)

        # Content tokens.
        token = delta.get("content", "")
        if token:
            if on_token is not None:
                on_token(token)
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

    result: Message = {"role": "assistant", "content": "".join(content_chunks)}
    if reasoning_chunks:
        result["reasoning"] = "".join(reasoning_chunks)
    if tool_calls_by_index:
        result["tool_calls"] = [
            tool_calls_by_index[i] for i in sorted(tool_calls_by_index)
        ]
    if usage is not None:
        result["usage"] = usage
    return result


def chat(
    messages: list[Message],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    reasoning_effort: str | None = None,
    tools: list[dict] | None = None,
    log: Callable[[str], None] = _default_log,
) -> Message:
    """Send a chat completion request. Returns the full response message dict.

    Handles payload construction, retries on transient errors, usage
    recording, and streaming log output.
    """
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
        payload["tools"] = [dict(t) for t in tools]

    # Streaming callbacks that manage [thinking]/[/thinking] delimiters.
    in_reasoning = False

    def on_thinking(text: str) -> None:
        nonlocal in_reasoning
        if not in_reasoning:
            log("\n  [thinking] ")
            in_reasoning = True
        log(text)

    def on_token(text: str) -> None:
        nonlocal in_reasoning
        if in_reasoning:
            log("\n  [/thinking]\n")
            in_reasoning = False

    result: Message = _with_retry(_stream_request, log, payload, on_token, on_thinking)
    if in_reasoning:
        log("\n  [/thinking]\n")

    if "usage" in result:
        _record_usage(model, result["usage"])
        pt = result["usage"].get("prompt_tokens", "?")
        ct = result["usage"].get("completion_tokens", "?")
        log(f"\n  [tokens] prompt={pt} completion={ct}\n")
    return result


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

    Returns dict with keys: metrics, commands, rounds, baseline_metrics, usage.
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

    with Stats() as stats:
        for round_num in range(1, max_rounds + 1):
            log(f"\n--- Round {round_num}/{max_rounds} ---\n")

            response = chat(
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
                # Model should always call a tool (evaluate_moves or done).
                # If it didn't, nudge it to use the proper tool interface.
                log("  [retry] No tool call, nudging model...\n")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You must use the evaluate_moves tool to test "
                            "scheduling ideas, or call done when finished. "
                            "Do not write tool calls as text."
                        ),
                    }
                )
                continue

            # Process each tool call.
            done = False
            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    tool_result = {"error": "Malformed JSON arguments."}
                    log(f"  [tool] {name}: malformed args\n")
                else:
                    if name == "done":
                        summary = args.get("summary", "")
                        log(f"  [done] {summary}\n")
                        tool_result = {"status": "ok"}
                        done = True
                    elif name == "evaluate_moves":
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

            if done:
                break

    usage = stats.counters
    log(
        f"\n=== Usage ===\n"
        f"  tokens: {usage.tokens} (in={usage.input_tokens} out={usage.output_tokens})\n"
        f"  cost: ${usage.cost:.4f}\n"
    )

    return {
        "metrics": best_metrics,
        "commands": best_commands,
        "rounds": round_num,
        "baseline_metrics": baseline,
        "usage": usage,
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
