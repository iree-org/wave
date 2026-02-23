# Conductor: LLM-Guided Instruction Scheduling for WaveASM

## Context

The WaveASM C++ backend has **no instruction scheduling pass**. Instructions are
emitted in the exact order they arrive from the upstream Python frontend. While
the Python frontend does tile-level scheduling (software pipelining,
double-buffering, stage assignment), it does not optimize **within** instruction
groups. This leaves performance on the table:

- LDS reads (20-cycle latency) are clustered together, then MFMAs (32+ cycles)
  are clustered together. Interleaving them would hide LDS latency behind MFMA
  execution.
- Global loads (100-cycle latency) are not interleaved with independent compute.
- Register pressure from the instruction order can cause the linear-scan
  allocator to fail (it has no spilling).

The target latency data (`getMFMALatency()`, `getGlobalLoadLatency()`,
`getLDSLoadLatency()`) already exists in `WaveASMAttrs.td` but is **unused by
any optimization pass**.

## Core Idea

No intermediate DSL. The LLM sees **real WaveASM MLIR textual IR** with named
location tags on each instruction. It issues **move commands** ("move A after
B"). A deterministic engine validates structural invariants (dominance,
dependency), applies the moves, runs the rest of the WaveASM MLIR compilation
pipeline, and reports **metrics** back to the LLM. The loop repeats until the
LLM is satisfied or a budget is exhausted.

```
WaveASM IR (post-CSE/peephole, virtual registers)
    |
    v
[Tag] ── attach NameLoc("I0"), ("I1"), ... to each op
    |
    v
[Print] ── module.print() ──> textual MLIR + header with target/latency info
    |
    v
[LLM] ── reads IR, issues move commands ──> "move I5 after I2"
    |                                        "move I8 before I12"
    |                                        "swap I3 I7"
    v
[Executor] ── parse commands
           ── validate (dominance, pinned ops)
           ── apply via Operation::moveBefore()
           ── anchor pseudo-ops (pack/extract/const)
    |
    v
[Compile] ── run LinearScan + InsertWaitcnt + HazardMitigation on clone
    |
    v
[Metrics] ── collect from WaveASM pipeline passes ──> report to LLM
    |
    v
[LLM] ── sees metrics, decides: more moves or done
    |
    v
(repeat up to N rounds, then emit final assembly)
```

The key insight: **the LLM works on the actual IR**, not an abstraction. It can
see every instruction, every SSA value, every type. The system handles all the
mechanical work (validation, compilation, metric collection) while the LLM
handles the creative work (what to move where).

## Instruction Tagging

Before presenting IR to the LLM, each operation in schedulable regions gets a
stable `NameLoc` tag:

```cpp
// During tagging pass.
int counter = 0;
for (auto &op : block.getOperations()) {
  auto tag = NameLoc::get(builder.getStringAttr("I" + std::to_string(counter++)));
  op.setLoc(tag);
}
```

In textual MLIR this appears as:

```mlir
%0 = waveasm.buffer_load_dwordx4 %srd, %off offset:0
       : !waveasm.sreg<4>, !waveasm.vreg -> !waveasm.vreg<4>  loc("I0")
%1 = waveasm.buffer_load_dwordx4 %srd, %off offset:64
       : !waveasm.sreg<4>, !waveasm.vreg -> !waveasm.vreg<4>  loc("I1")
%2 = waveasm.ds_write_b128 %0, %addr
       : !waveasm.vreg<4>, !waveasm.vreg                      loc("I2")
...
%10 = waveasm.v_mfma_f32_16x16x16_f16 %a, %b, %acc
       : !waveasm.vreg, !waveasm.vreg, !waveasm.vreg<4>
       -> !waveasm.vreg<4>                                     loc("I10")
```

Tags are stable across rounds — they follow the operation, not the position.

## LLM Input

Each round, the LLM receives:

```
=== WaveASM Scheduling Round {N} ===
TARGET: gfx942 (wave64, 512 vgpr, 106 sgpr, 512 agpr)
LATENCY: vmem=100, lds=20, mfma_16x16=32, mfma_32x32=64

--- IR (loop body) ---
{textual MLIR of the loop body, with loc("Ixx") tags}

--- Metrics (from previous round, or baseline) ---
peak_vgpr: 180
peak_sgpr: 42
peak_agpr: 128
nops_inserted: 3
waitcnts_inserted: 12
total_instructions: 87

--- Warnings from previous round ---
(none, or e.g.:
  "warn: move I8 after I12 — moved LDS read after LDS write, same base address
   info: move I3 after I7 — address computation moved away from consumer I4")

--- Error from previous round (if any) ---
(none, or e.g.:
  "Applied successfully: move I5 after I1, move I8 before I3
   Failed: swap I6 I9 — would break dominance of %2 (defined by I6, used by I7)
   All moves reverted.")

GOAL: Minimize register pressure and hide memory latency.
Respond with move commands, one per line.
```

## LLM Output (Move Commands)

The LLM responds with simple text commands:

```
move I5 after I1
move I8 before I3
swap I6 I9
```

### Command Set

| Command | Semantics |
|---------|-----------|
| `move Ix after Iy` | Move op tagged Ix to immediately after op tagged Iy. |
| `move Ix before Iy` | Move op tagged Ix to immediately before op tagged Iy. |
| `swap Ix Iy` | Exchange positions of ops Ix and Iy. |
| `done` | LLM is satisfied with current schedule. |

Three move commands + a stop signal. No DSL to learn.

## Validation

Move commands are applied sequentially. Each is validated **before** application:

1. **Tag resolution**: Both tags must exist.
2. **Pinned ops**: `ConditionOp` (loop terminator), `s_barrier`, `s_endpgm`
   cannot be moved.
3. **Dominance check (pre-flight)**: Simulate the move and verify every use of
   every SSA value defined by the moved op is still dominated by its def. Also
   check that all operands of the moved op are still dominated.
4. **Region boundary**: Cannot move ops across region boundaries (into/out of
   `LoopOp` or `IfOp`).

On the first invalid move, the system **aborts the entire round**: all moves
applied so far in this round are reverted, and the LLM receives a report
showing which moves succeeded before the failure and which move was invalid
(with the reason). For example:

```
--- Error ---
Applied successfully: move I5 after I1, move I8 before I3
Failed: swap I6 I9 — would break dominance of %2 (defined by I6, used by I7)
All moves reverted.
```

This gives the LLM a clear signal about what went wrong and a complete picture
of which commands were valid up to the failure point.

### Warnings

Some moves are structurally valid (dominance holds) but **potentially
dangerous**. These are applied but reported as warnings with severity levels,
so the LLM can decide whether to keep or reconsider them.

| Severity | Example | Meaning |
|----------|---------|---------|
| `info` | Moving address computation away from its consumer. | Unusual but harmless. |
| `warn` | Moving a memory read after a memory write. | May be valid (different addresses) but could introduce a WAR hazard. |
| `warn` | Moving an op past a barrier. | Barrier ordering is usually intentional. |
| `critical` | Moving a memory write after another write to the same buffer. | Likely WAW hazard; almost certainly wrong unless offsets are provably disjoint. |

The system performs lightweight alias analysis where possible (e.g. comparing
buffer SRD operands and constant offsets) to reduce false warnings. When it
cannot prove safety, it warns and lets the LLM proceed — the metrics from the
compilation round will reveal whether the move was actually beneficial.

Warnings are reported alongside metrics in the next round:

```
--- Warnings ---
warn: move I8 after I12 — moved LDS read (I8) after LDS write (I12), same base address
info: move I3 after I7 — address computation moved away from consumer I4
```

## Metrics Collection

After applying a round of moves, the system runs the downstream WaveASM MLIR
pipeline on a **clone** of the IR and collects metrics:

```cpp
struct SchedulingMetrics {
  int64_t peakVGPRs;
  int64_t peakSGPRs;
  int64_t peakAGPRs;
  int64_t nopsInserted;       // from HazardMitigation.
  int64_t waitcntsInserted;   // from InsertWaitcnt.
  int64_t totalInstructions;  // post-pipeline instruction count.
};
```

Sources (currently internal to passes, need to be exposed):
- **Peak registers**: `computeMaxPressure()` from `Liveness.h`, or
  `AllocationStats` from `LinearScanRegAlloc`.
- **NOP count**: `HazardMitigation` tracks `numNopsInserted` internally.
- **Waitcnt count**: `InsertWaitcnt` tracks tickets internally.
- **Total instructions**: count ops after all passes.

Additional metrics can be added as needed (e.g. estimated cycle count, LDS bank
conflict potential, critical path length).

The pipeline runs on a cloned module so the original IR is preserved for the
next round of moves.

### Optional: GPU Profiling Integration

When a GPU is available and profiling is enabled, the system can run the
compiled kernel through `rocprof` and feed per-instruction profiling data back
to the LLM. `rocprof` reports hit count and min/max/mean latency for each
assembly instruction. Since the assembly emitter preserves instruction order
and the tags map to assembly locations, profiling data can be correlated back
to the tagged IR.

The profiling section is appended to the LLM input when available:

```
--- Profiling (rocprof, previous run) ---
I0  buffer_load_dwordx4   hits=1024  lat_min=88  lat_mean=102  lat_max=145
I1  buffer_load_dwordx4   hits=1024  lat_min=90  lat_mean=105  lat_max=148
I5  ds_read_b128          hits=1024  lat_min=18  lat_mean=21   lat_max=34
I10 v_mfma_f32_16x16x16   hits=1024  lat_min=32  lat_mean=33   lat_max=35
I15 s_waitcnt             hits=1024  lat_min=0   lat_mean=45   lat_max=120
```

This lets the LLM see actual stall patterns — e.g. a `s_waitcnt` with high
mean latency indicates the preceding loads are not sufficiently hidden. This
data is strictly optional; the system works without it using only the static
pipeline metrics.

## Pipeline Integration

```
TranslateFromMLIR
  → ScopedCSE
  → Peephole
  → MemoryOffsetOpt
  → Canonicalizer + ScopedCSE
  → [NEW] Conductor (tag + LLM loop + apply final schedule)
  → LinearScan
  → InsertWaitcnt
  → HazardMitigation
  → EmitAssembly
```

The Conductor pass:
1. Tags all ops with `NameLoc`.
2. Runs baseline metrics (clone → compile → collect).
3. Enters the LLM loop (up to N rounds).
4. Applies the best schedule to the real IR.
5. Hands off to LinearScan.

## Iterative Loop Detail

```
baseline_metrics = compile_and_measure(clone(IR))
best_metrics = baseline_metrics
best_state = snapshot(IR)

for round in 1..max_rounds:
    text = print_ir(IR) + format_metrics(best_metrics) + format_error(error)
    commands = llm.query(text)

    if commands == ["done"]:
        break

    error = null
    applied = []
    for cmd in commands:
        result = validate_and_apply(IR, cmd)
        if result.is_error:
            error = { applied: applied, failed: cmd, reason: result.message }
            restore(IR, best_state)  // revert entire round.
            break
        applied.append(cmd)

    if error:
        continue  // next round, LLM sees the error report.

    anchor_pseudo_ops(IR)  // move pack/extract/const before earliest user.
    new_metrics = compile_and_measure(clone(IR))

    if is_better(new_metrics, best_metrics):
        best_metrics = new_metrics
        best_state = snapshot(IR)
    else:
        restore(IR, best_state)  // revert if regression.
        error = { applied: applied, reason: "round regressed metrics" }

apply(IR, best_state)
```

`is_better()` compares metrics lexicographically: lower peak VGPRs > fewer
waitcnts > fewer nops. This ordering is configurable.

## Handling Structured Control Flow

- **`LoopOp` body**: The primary scheduling target. Block arguments are always
  available (dominate the entire body). `ConditionOp` is pinned at end.
- **`IfOp`**: Treated as an atomic unit. The LLM can move the entire `IfOp` but
  not its internal ops. The tag is on the `IfOp` itself.
- **`ProgramOp` body (prologue)**: Tagged and schedulable as a straight-line
  block, but typically less benefit (SRD setup).
- **Nested regions**: Tags are scoped per region. The LLM sees one region at a
  time (typically the innermost loop body).

## Parallel Search with Checkpoints

Individual LLM calls have high latency but the API supports many concurrent
requests. Instead of running a single sequential loop, launch **P parallel
agents**, each exploring a different scheduling trajectory from the same
starting IR.

```
           ┌─ Agent 1 ─── round 1 ─── round 2 ─── ...
           │
tagged IR ─┼─ Agent 2 ─── round 1 ─── round 2 ─── ...
           │
           ├─ Agent 3 ─── round 1 ─── round 2 ─── ...
           │
           └─ Agent P ─── round 1 ─── round 2 ─── ...
                              │
                          checkpoint
```

At **checkpoints** (every K rounds), collect metrics from all agents, rank
them, and keep the **top-k**. The surviving agents continue from their current
state; the rest are terminated. New agents can be spawned from the top-k states
to refill the pool, optionally with varied system prompts (e.g. "prioritize
register pressure" vs "prioritize latency hiding").

```
for checkpoint in 1..num_checkpoints:
    // all agents run K rounds in parallel.
    results = await all_agents(K rounds each)

    // rank by metrics, keep top-k.
    ranked = sort(results, by=metrics)
    survivors = ranked[:top_k]

    // respawn from survivors to refill pool.
    agents = []
    for state in survivors:
        agents.append(continue_agent(state))
        agents.append(fork_agent(state, varied_prompt))  // optional diversity.

best = survivors[0]
```

This turns the scheduling search into a **beam search** over move sequences.
The validation + metrics infrastructure is unchanged — each agent is just an
independent instance of the iterative loop. The only coordination is at
checkpoints where we compare metrics and prune.

Parallelism is free in terms of wall-clock time (bounded by the slowest agent
per checkpoint) and the compile-and-measure step is CPU-local and fast (~ms
per clone).

## Caching

Final move sequences are cached by SHA-256 of the **tagged IR text** (before
any moves):

```
~/.waveasm-cache/conductor/
  <ir-hash>.json → {
    moves: ["move I5 after I1", "swap I6 I9"],
    metrics: { peakVGPRs: 160, nops: 1, waitcnts: 8 },
    rounds: 3,
    timestamp: "..."
  }
```

On cache hit, the moves are replayed directly (still validated) without querying
the LLM.

## Key Design Properties

1. **No abstraction gap.** The LLM sees actual MLIR IR, not a lossy summary.
2. **Zero new languages.** Three move commands, trivial to parse and validate.
3. **Fast feedback.** Invalid moves abort the round immediately with a clear
   error report; the LLM learns from mistakes without silent corruption.
4. **Iterative refinement.** The LLM observes metrics after each round and
   adjusts, rather than producing a one-shot schedule.
5. **Closed loop with hardware.** Optional `rocprof` integration feeds real
   per-instruction latencies back, grounding the LLM in measured data.
6. **Transparent.** Every move is logged. Easy to inspect, replay, and debug.
7. **Extensible metrics.** Adding a new metric is just a field in
   `SchedulingMetrics` — no protocol changes needed.
