"""Iterative LLM scheduling loop for Conductor."""

import difflib
import json
import sys
from collections.abc import Callable

from conductor.providers.openrouter import (
    DEFAULT_MODEL,
    Message,
    Stats,
    chat,
)
from conductor.tools import Param, ToolRegistry


def _default_log(msg: str) -> None:
    """Default logger: print to stderr without trailing newline."""
    print(msg, file=sys.stderr, end="", flush=True)


SYSTEM_PROMPT = """\
You are an expert GPU instruction scheduler for AMD CDNA architectures.

You will receive WaveASM MLIR IR with tagged instructions (loc("tag_name")).
Your job is to reorder instructions to hide memory latency and reduce register pressure.

Latencies (cycles):
  global_load/buffer_load: ~100    LDS (ds_read/ds_write): ~20
  MFMA F16 16x16x16: 16           MFMA F16 32x32x8: 32
  MFMA F8/F4 16x16x128: 16-32     MFMA F8/F4 32x32x64: 32-64
  scaled_mfma (MXFP4): same as above
  VALU: 1 (transcendentals: 2)    SALU: 1

Scheduling strategy:
- Issue global loads as early as possible, defer s_waitcnt to just before use.
- Interleave LDS reads with MFMA: ~20 cycles of MFMA hides one LDS read.
- Fill cycles between dependent MFMAs with independent loads or VALU.
- MFMA accumulator chains (same opcode, SrcC=prev vDst) need 0 extra NOPs.
- Fewer s_waitcnt and s_nop in the final assembly = better schedule.
- Lower peak VGPRs = higher occupancy (key breakpoints: 128, 96, 80, 64).

You have an `evaluate_moves` tool. Call it with a list of move command strings.
Each command is one of:
  "move TAG_A after TAG_B"
  "move TAG_A before TAG_B"
  "swap TAG_A TAG_B"

The tool will apply the moves, compile, and return metrics + updated IR.

Constraints:
- All moves must stay within the same basic block. Never move across blocks.
- Moves must preserve SSA dominance: a value must be defined before all its uses.
  Before proposing any move, trace the SSA def-use chains for the instruction
  you want to move. Check: (1) every operand (%val) it consumes is still
  defined above it after the move, and (2) every result it produces is still
  above all its users after the move. If either check fails, the move is
  illegal — pick a different one.
- Pinned ops (s_endpgm, s_barrier, condition) cannot be moved.

Work incrementally: try 1-3 moves per tool call, read the resulting metrics, \
then decide your next moves. You can call the tool multiple times.

DO NOT try to build too long of a plan at once. Try 1-3 moves and see what changes.

Keep your reasoning brief. For each thinking step, write at most one short \
sentence. Do not re-analyze the entire IR — focus only on the specific move \
you are considering and its immediate SSA neighbors.

When you are satisfied with the schedule or have no more ideas, call the `done` \
tool to finish.\
"""


def format_initial_prompt(
    tagged_ir: str,
    baseline_metrics: dict,
    target: str = "gfx942",
) -> str:
    """Format the initial user message with IR and baseline metrics."""
    parts = [
        f"TARGET: {target} (wave64, 256 vgpr + 256 agpr, 102 sgpr)",
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
            "GOAL: Reduce s_waitcnt and s_nop count (better latency hiding),",
            "then reduce peak VGPRs (higher occupancy).",
            "Use the evaluate_moves tool to try reorderings.",
        ]
    )
    return "\n".join(parts)


def run_scheduling_loop(
    conductor,
    max_rounds: int = 10,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    reasoning_effort: str | None = "high",
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

    initial_ir = conductor.tag()
    current_ir = initial_ir
    log(f"  --- Tagged IR ---\n{current_ir.strip()}\n  --- End IR ---\n")

    best_metrics = dict(baseline)
    all_commands: list[str] = []
    finished = False

    # Build tool registry with closures over loop state.
    registry = ToolRegistry()

    def _evaluate_moves(moves: list[str], summary: str = "") -> str:
        nonlocal best_metrics, all_commands, current_ir
        if summary:
            log(f"  [reason] {summary}\n")
        log(f"  [tool] evaluate_moves({moves})\n")
        try:
            reordered_ir = conductor.apply_moves(current_ir, moves)
        except RuntimeError as e:
            log(f"  [error] {e}\n")
            return json.dumps({"error": str(e)})
        try:
            asm = conductor.compile_to_asm(reordered_ir)
            metrics = conductor.get_metrics(asm)
        except RuntimeError as e:
            log(f"  [error] {e}\n")
            return json.dumps({"error": str(e)})
        log(f"  [result] {metrics}\n")
        # Moves succeeded — update state.
        current_ir = reordered_ir
        all_commands.extend(moves)
        improved = _is_better(metrics, best_metrics)
        if improved:
            log("  Improvement!\n")
            best_metrics = metrics
        result = {
            "metrics": metrics,
            "improved": improved or metrics == best_metrics,
            "updated_ir": reordered_ir.strip(),
        }
        if summary:
            result["summary"] = summary
        return json.dumps(result)

    def _done(summary: str) -> str:
        nonlocal finished
        log(f"  [done] {summary}\n")
        finished = True
        return json.dumps({"status": "ok"})

    registry.add(
        name="evaluate_moves",
        description=(
            "Apply a list of move/swap commands to the tagged IR, "
            "compile through the post-scheduling pipeline, and return "
            "assembly metrics. Commands are applied in order."
        ),
        params=[
            Param(
                name="moves",
                description=(
                    "List of move commands, e.g. "
                    '["move tag_A after tag_B", "swap tag_C tag_D"].'
                ),
                type="array",
                items={"type": "string"},
            ),
            Param(
                name="summary",
                description=(
                    "Brief explanation of why these moves should help "
                    "(e.g. 'hide LDS latency by interleaving ds_read with mfma')."
                ),
            ),
        ],
        func=_evaluate_moves,
    )
    registry.add(
        name="done",
        description=(
            "Call this when you are finished scheduling. "
            "Provide a short summary of what you tried."
        ),
        params=[
            Param(name="summary", description="Brief summary of scheduling attempts."),
        ],
        func=_done,
    )

    messages: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": format_initial_prompt(
                current_ir, baseline, target=conductor.target
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
                tools=registry.definitions(),
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

            for tc in tool_calls:
                result = registry.execute(
                    tc["function"]["name"],
                    tc["function"]["arguments"],
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    }
                )

            if finished:
                break

    usage = stats.counters
    log(
        f"\n=== Usage ===\n"
        f"  tokens: {usage.tokens} (in={usage.input_tokens} out={usage.output_tokens})\n"
        f"  cost: ${usage.cost:.4f}\n"
    )

    ir_diff = _context_diff(initial_ir, current_ir)
    if ir_diff:
        log(f"\n=== IR Diff (original → final) ===\n{ir_diff}\n")
    else:
        log("\n=== No IR changes ===\n")

    return {
        "metrics": best_metrics,
        "commands": all_commands,
        "rounds": round_num,
        "baseline_metrics": baseline,
        "usage": usage,
    }


def _context_diff(before: str, after: str, n: int = 10) -> str:
    """Return a unified diff of changed lines with ±n lines of context."""
    a = before.splitlines(keepends=True)
    b = after.splitlines(keepends=True)
    diff = difflib.unified_diff(a, b, fromfile="before", tofile="after", n=n)
    return "".join(diff)


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
