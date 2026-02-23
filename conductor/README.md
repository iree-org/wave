# Conductor

LLM-guided instruction scheduling for WaveASM. Takes pre-scheduling WaveASM IR,
lets an LLM reorder instructions via move/swap commands, then compiles and
measures the result. See
[CONDUCTOR_DESIGN.md](../wave_lang/kernel/wave/asm/wave_asm/include/waveasm/CONDUCTOR_DESIGN.md)
for the full design rationale.

## Pipeline

```
Python frontend (GEMM kernel)
  → Wave compiler → MLIR
  → waveasm-translate (pre-scheduling: CSE, peephole, mem-offset-opt)
  → Tag instructions (attach loc("tag_name") to each op)
  → LLM loop: present IR + metrics → get move commands → apply → compile → measure
  → Final assembly
```

## Modules

| File | Purpose |
|------|---------|
| `extract_ir.py` | Captures MLIR from a Wave kernel, runs pre-scheduling pipeline, computes baseline metrics. |
| `conductor.py` | `Conductor` class: tag, apply_moves, compile_to_asm, evaluate, baseline. CLI entry point. |
| `llm.py` | OpenRouter client, prompt formatting, command parsing, iterative scheduling loop. |

## Prerequisites

- `waveasm-translate` and `waveasm-conductor` binaries (build from `wave_lang/kernel/wave/asm/wave_asm/`).
- Python `requests` package.
- `OPENROUTER_API_KEY` env var for LLM mode.

## Usage

All commands should be run from the wave project root with `WAVE_CACHE_ON=0`.

### Baseline metrics (no LLM)

```bash
python -m conductor.conductor --metrics
```

### Show tagged IR

```bash
python -m conductor.conductor --tag-only
```

### Apply manual moves

```bash
python -m conductor.conductor \
  --moves "swap buffer_load_dwordx4_0 v_mfma_f32_16x16x16_f16_0" \
  --metrics
```

### Read moves from a file

```bash
python -m conductor.conductor --moves-file moves.txt --metrics
```

### LLM-guided scheduling

```bash
OPENROUTER_API_KEY=sk-... python -m conductor.conductor \
  --llm --max-rounds 5 --model google/gemini-2.5-flash-preview --metrics
```

### Extract pre-scheduling IR only

```bash
python -m conductor.extract_ir -o /tmp/conductor_ir.mlir
python -m conductor.extract_ir --metrics
```

## Move commands

| Command | Example |
|---------|---------|
| `move TAG before TAG` | `move v_add_u32_1 before v_add_u32_0` |
| `move TAG after TAG` | `move buffer_load_dwordx4_0 after ds_read_b128_2` |
| `swap TAG TAG` | `swap v_mfma_f32_16x16x16_f16_0 ds_read_b128_1` |

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | (none) | Required for `--llm` mode. |
| `WAVE_CACHE_ON` | `1` | Set to `0` to disable caching during development. |
| `WAVE_DEFAULT_ARCH` | `gfx942` | Target GPU architecture. |
| `WAVEASM_TRANSLATE` | (auto-detect) | Override path to `waveasm-translate` binary. |
| `WAVEASM_CONDUCTOR` | (auto-detect) | Override path to `waveasm-conductor` binary. |

## Programmatic usage

```python
from conductor import Conductor
from conductor.extract_ir import capture_kernel_mlir, run_pre_scheduling_pipeline
from conductor.llm import run_scheduling_loop

mlir_text, wg_size = capture_kernel_mlir()
waveasm_ir = run_pre_scheduling_pipeline(mlir_text, wg_size)

c = Conductor(waveasm_ir, wg_size)

# Baseline.
print(c.baseline())

# Manual moves.
print(c.evaluate(["swap buffer_load_dwordx4_0 v_mfma_f32_16x16x16_f16_0"]))

# LLM loop.
result = run_scheduling_loop(c, max_rounds=5, model="google/gemini-2.5-flash-preview")
print(result["metrics"], result["commands"])
```
