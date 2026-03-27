"""
MXFP4 Scaled GEMM Scheduling for GFX950 (MI350)

Double-buffered MXFP4 GEMM with 4-wave and 8-wave configurations.
Uses get_tagged_mxfp4_gemm (templates) + get_mxfp4_dbuf_schedule (schedules).

Usage:
    python 7.1_schedule.py --test test_dbuf_4wave_mxfp_gemm
    python 7.1_schedule.py --test test_dbuf_8wave_mxfp_gemm
    python 7.1_schedule.py --test test_dbuf_8wave_mxfp_gemm --debug
    python 7.1_schedule.py --list_tests
"""

import os
import torch
import wave_lang.kernel.lang as tkl

from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.templates import (
    get_tagged_mxfp4_gemm,
    get_tagged_mxfp4_gemm_preshuffle_b,
    get_tagged_mxfp4_gemm_preshuffle_scales,
    get_tagged_mxfp4_gemm_preshuffle_scales_and_B,
)
from wave_lang.kernel.wave.schedules import (
    get_mxfp4_dbuf_schedule,
    get_mxfp4_dbuf_pingpong_schedule,
    get_mxfp4_dbuf_mixed_pingpong_schedule,
    get_mxfp4_asymmetric_schedule,
    get_mxfp4_dbuf_mixed_pingpong_shuffle_schedule,
    get_mxfp4_dbuf_pingpong_schedule_Bshuffled,
    get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds,
)
from wave_lang.kernel.wave.utils.mxfp_utils import (
    generate_gemm_afp4wfp4_inputs,
    torchScaledGemmMXFP4,
    b_preshuffle,
    e8m0_shuffle,
)
from wave_lang.kernel.lang.global_symbols import (
    GLOBAL_ADDRESS_SPACE,
    SHARED_ADDRESS_SPACE,
)
from utils import parse_args, list_tests, run_test
import re


def coalesce_buffer_stores_dwordx4(asm_text):
    """Post-process assembly to merge 4 consecutive buffer_store_dword into buffer_store_dwordx4.

    Detects groups of 4 buffer_store_dword with same SRD+voffset and
    offsets {X, X+4, X+8, X+12}.  Replaces each store with a v_mov_b32
    copy to v0-v3, then emits one buffer_store_dwordx4 v[0:3].
    """
    store_re = re.compile(
        r"^  buffer_store_dword (v\d+), (v\d+), (s\[\d+:\d+\]), 0 offen(?: offset:(\d+))?$"
    )

    lines = asm_text.split("\n")
    out = []
    i = 0
    coalesced_count = 0
    while i < len(lines):
        m0 = store_re.match(lines[i])
        if m0:
            group = [(i, m0)]
            j = i + 1
            while j < len(lines) and len(group) < 4:
                mj = store_re.match(lines[j])
                if mj:
                    group.append((j, mj))
                j += 1

            if len(group) == 4:
                srds = [g.group(3) for _, g in group]
                voffs = [g.group(2) for _, g in group]
                offsets = [int(g.group(4)) if g.group(4) else 0 for _, g in group]
                base = offsets[0]
                if (
                    all(s == srds[0] for s in srds)
                    and all(v == voffs[0] for v in voffs)
                    and offsets == [base, base + 4, base + 8, base + 12]
                ):
                    store_line_indices = {idx for idx, _ in group}
                    slot = 0
                    for k in range(i, group[-1][0] + 1):
                        if k in store_line_indices:
                            data_reg = group[slot][1].group(1)
                            out.append(f"  v_mov_b32 v{slot}, {data_reg}")
                            slot += 1
                        else:
                            out.append(lines[k])
                    merged = (
                        f"  buffer_store_dwordx4 v[0:3], {voffs[0]}, {srds[0]}, 0 offen"
                    )
                    if base > 0:
                        merged += f" offset:{base}"
                    out.append(merged)
                    coalesced_count += 1
                    i = group[-1][0] + 1
                    continue

        out.append(lines[i])
        i += 1

    print(
        f"[asm_transform] Coalesced {coalesced_count} groups of 4 stores -> buffer_store_dwordx4"
    )
    return "\n".join(out)


def convert_first_eliminate_cndmask(asm_text):
    """Replace per-tile swap+cndmask+cvt+store with convert-first+swap+dwordx4.

    The current epilogue swaps individual f32 values, then uses v_cndmask_b32
    to select own vs partner data (8 cndmask per tile = 384 total).  This
    transform converts to packed bf16 FIRST, swaps the packed dwords (1 swap
    instead of 4), re-reads AGPRs for own data, and stores via dwordx4.
    Eliminates all data-select cndmask and halves the swap count.
    """
    read_re = re.compile(r"^\s*v_accvgpr_read_b32\s+(v\d+),\s+a(\d+)")
    nop_re = re.compile(r"^s_nop 1\s*$")
    swap_re = re.compile(r"^\s*v_permlane16_swap_b32\s+")
    store_off12_re = re.compile(
        r"^\s*buffer_store_dword\s+v\d+,\s+(v\d+),\s+(s\[\d+:\d+\]),\s+0 offen\s+offset:12\s*$"
    )
    sub4_re = re.compile(r"^\s*v_sub_u32\s+v\d+,\s+v\d+,\s+4\s*$")
    cmp_ne_v244_re = re.compile(r"^\s*v_cmp_ne_u32\s+vcc,\s+v244,\s+0")
    cndmask_re = re.compile(r"^\s*v_cndmask_b32\s+(v\d+),")
    lshlrev1_re = re.compile(r"^\s*v_lshlrev_b32\s+v\d+,\s+1,")

    lines = asm_text.split("\n")
    out = []
    tile_count = 0
    vcc_emitted = False
    i = 0

    while i < len(lines):
        if i + 11 < len(lines) and read_re.match(lines[i]):
            agprs = []
            ok = True
            for k in range(4):
                base = i + k * 3
                m = read_re.match(lines[base])
                if not (
                    m
                    and nop_re.match(lines[base + 1])
                    and swap_re.match(lines[base + 2])
                ):
                    ok = False
                    break
                agprs.append(int(m.group(2)))
            if not ok:
                out.append(lines[i])
                i += 1
                continue

            swap_end = i + 12

            j = swap_end
            srd = None
            while j < len(lines):
                sm = store_off12_re.match(lines[j])
                if sm:
                    srd = sm.group(2)
                    break
                j += 1

            if srd is None:
                out.append(lines[i])
                i += 1
                continue

            tile_end = j + 1
            tile_count += 1

            middle = lines[swap_end:tile_end]
            preserved = []
            mi = 0
            while mi < len(middle):
                s = middle[mi].strip()

                # Offset-select pattern: v_sub_u32 .., 4 -> v_cmp_ne -> v_cndmask
                if sub4_re.match(s):
                    preserved.append(middle[mi])
                    if mi + 2 < len(middle):
                        preserved.append(middle[mi + 1])
                        preserved.append(middle[mi + 2])
                        mi += 3
                    else:
                        mi += 1
                    continue

                # Lane-mask creation: v_cndmask_b32 v244, ...
                cm = cndmask_re.match(s)
                if cm and cm.group(1) == "v244":
                    preserved.append(middle[mi])
                    mi += 1
                    continue

                if (
                    cmp_ne_v244_re.match(s)
                    or (cm and cm.group(1) != "v244")
                    or s.startswith("v_accvgpr_read_b32")
                    or s.startswith("v_cvt_pk_bf16_f32")
                    or s.startswith("buffer_store_dword")
                    or lshlrev1_re.match(s)
                ):
                    mi += 1
                    continue

                preserved.append(middle[mi])
                mi += 1

            a0, a1, a2, a3 = agprs
            # Preserved lines (lane mask, offset select, addr comp, SRD)
            # must come first so v244 and v253 are set before we use them.
            out.extend(preserved)
            if not vcc_emitted:
                out.append("  v_cmp_ne_u32 vcc, v244, 0")
                vcc_emitted = True
            out.extend(
                [
                    f"  v_accvgpr_read_b32 v0, a{a0}",
                    f"  v_accvgpr_read_b32 v1, a{a1}",
                    f"  v_cvt_pk_bf16_f32 v2, v0, v1",
                    f"  v_accvgpr_read_b32 v0, a{a2}",
                    f"  v_accvgpr_read_b32 v1, a{a3}",
                    f"  v_cvt_pk_bf16_f32 v3, v0, v1",
                    "  v_mov_b32 v8, v2",
                    "  v_mov_b32 v9, v3",
                    "s_nop 1",
                    "    v_permlane16_swap_b32 v4, v2",
                    "s_nop 1",
                    "    v_permlane16_swap_b32 v5, v3",
                    "  v_cndmask_b32 v0, v4, v8",
                    "  v_cndmask_b32 v1, v5, v9",
                    "  v_cndmask_b32 v2, v8, v4",
                    "  v_cndmask_b32 v3, v9, v5",
                    "  v_lshlrev_b32 v245, 1, v253",
                    f"  buffer_store_dwordx4 v[0:3], v245, {srd}, 0 offen",
                ]
            )

            i = tile_end
            continue

        out.append(lines[i])
        i += 1

    return "\n".join(out)


def lds_epilogue_transform(asm_text):
    """Replace epilogue with ds_bpermute cross-lane exchange (no permlane_swap).

    Uses ds_bpermute_b32 to read packed bf16 data from partner lanes via the
    LDS crossbar (no LDS memory access). Both halves of each swap-pair read
    from the same LOW and HIGH partner lanes, producing identical v[0:3].
    Eliminates all permlane_swap, cndmask, s_nop, and AGPR re-reads.

    Per-tile: 4 accvgpr_read + 2 cvt + 4 bpermute + 1 waitcnt + 1 lshlrev +
    1 store = 13 instructions (vs 25 cvt-first, 45 orig).
    """
    read_re = re.compile(r"^\s*v_accvgpr_read_b32\s+(v\d+),\s+a(\d+)")
    nop_re = re.compile(r"^s_nop 1\s*$")
    swap_re = re.compile(r"^\s*v_permlane16_swap_b32\s+")
    store_off12_re = re.compile(
        r"^\s*buffer_store_dword\s+v\d+,\s+(v\d+),\s+(s\[\d+:\d+\]),\s+0 offen\s+offset:12\s*$"
    )
    sub4_re = re.compile(r"^\s*v_sub_u32\s+v\d+,\s+v\d+,\s+4\s*$")
    cmp_ne_v244_re = re.compile(r"^\s*v_cmp_ne_u32\s+vcc,\s+v244,\s+0")
    cndmask_re = re.compile(r"^\s*v_cndmask_b32\s+(v\d+),")
    lshlrev1_re = re.compile(r"^\s*v_lshlrev_b32\s+v\d+,\s+1,")

    lines = asm_text.split("\n")

    out = []
    tile_count = 0
    lds_setup_emitted = False
    i = 0

    while i < len(lines):
        if i + 11 < len(lines) and read_re.match(lines[i]):
            agprs = []
            ok = True
            for k in range(4):
                base = i + k * 3
                m = read_re.match(lines[base])
                if not (
                    m
                    and nop_re.match(lines[base + 1])
                    and swap_re.match(lines[base + 2])
                ):
                    ok = False
                    break
                agprs.append(int(m.group(2)))
            if not ok:
                out.append(lines[i])
                i += 1
                continue

            swap_end = i + 12

            j = swap_end
            srd = None
            while j < len(lines):
                sm = store_off12_re.match(lines[j])
                if sm:
                    srd = sm.group(2)
                    break
                j += 1

            if srd is None:
                out.append(lines[i])
                i += 1
                continue

            tile_end = j + 1
            tile_count += 1

            middle = lines[swap_end:tile_end]
            preserved = []
            mi = 0
            while mi < len(middle):
                s = middle[mi].strip()

                if sub4_re.match(s):
                    preserved.append(middle[mi])
                    if mi + 2 < len(middle):
                        preserved.append(middle[mi + 1])
                        preserved.append(middle[mi + 2])
                        mi += 3
                    else:
                        mi += 1
                    continue

                cm = cndmask_re.match(s)
                if cm and cm.group(1) == "v244":
                    preserved.append(middle[mi])
                    mi += 1
                    continue

                if (
                    cmp_ne_v244_re.match(s)
                    or (cm and cm.group(1) != "v244")
                    or s.startswith("v_accvgpr_read_b32")
                    or s.startswith("v_cvt_pk_bf16_f32")
                    or s.startswith("buffer_store_dword")
                    or lshlrev1_re.match(s)
                ):
                    mi += 1
                    continue

                preserved.append(middle[mi])
                mi += 1

            if not lds_setup_emitted:
                out.extend(
                    [
                        "  ;; LDS epilogue: bpermute-based cross-lane exchange",
                        "  v_and_b32 v4, 4294967279, v238",
                        "  v_lshlrev_b32 v4, 2, v4",
                        "  v_or_b32 v5, v238, 16",
                        "  v_lshlrev_b32 v5, 2, v5",
                    ]
                )
                lds_setup_emitted = True

            out.extend(preserved)

            a0, a1, a2, a3 = agprs
            out.extend(
                [
                    f"  v_accvgpr_read_b32 v0, a{a0}",
                    f"  v_accvgpr_read_b32 v1, a{a1}",
                    "  v_cvt_pk_bf16_f32 v2, v0, v1",
                    f"  v_accvgpr_read_b32 v0, a{a2}",
                    f"  v_accvgpr_read_b32 v1, a{a3}",
                    "  v_cvt_pk_bf16_f32 v3, v0, v1",
                    "  ds_bpermute_b32 v0, v4, v2",
                    "  ds_bpermute_b32 v1, v4, v3",
                    "  ds_bpermute_b32 v2, v5, v2",
                    "  ds_bpermute_b32 v3, v5, v3",
                    "  s_waitcnt lgkmcnt(0)",
                    "  v_lshlrev_b32 v245, 1, v253",
                    f"  buffer_store_dwordx4 v[0:3], v245, {srd}, 0 offen",
                ]
            )

            i = tile_end
            continue

        out.append(lines[i])
        i += 1

    print(
        f"[lds_epilogue] Transformed {tile_count} tiles: bpermute exchange, "
        f"no swap/cndmask/LDS-memory"
    )
    return "\n".join(out)


def bpermute_masked_epilogue_transform(asm_text):
    """Like lds_epilogue_transform but with exec masking to eliminate duplicate stores.

    In the transposed output, each pair of lanes (i, i^16) writes the same data to
    the same address (benign duplicate). This transform masks the exec register so
    only one lane per pair executes the buffer_store, halving memory traffic.

    Uses s[20:21] to save exec and s[22:23] for the masked exec (these SGPRs hold
    input SRDs that are dead in the epilogue).
    """
    read_re = re.compile(r"^\s*v_accvgpr_read_b32\s+(v\d+),\s+a(\d+)")
    nop_re = re.compile(r"^s_nop 1\s*$")
    swap_re = re.compile(r"^\s*v_permlane16_swap_b32\s+")
    store_off12_re = re.compile(
        r"^\s*buffer_store_dword\s+v\d+,\s+(v\d+),\s+(s\[\d+:\d+\]),\s+0 offen\s+offset:12\s*$"
    )
    sub4_re = re.compile(r"^\s*v_sub_u32\s+v\d+,\s+v\d+,\s+4\s*$")
    cmp_ne_v244_re = re.compile(r"^\s*v_cmp_ne_u32\s+vcc,\s+v244,\s+0")
    cndmask_re = re.compile(r"^\s*v_cndmask_b32\s+(v\d+),")
    lshlrev1_re = re.compile(r"^\s*v_lshlrev_b32\s+v\d+,\s+1,")

    lines = asm_text.split("\n")
    out = []
    tile_count = 0
    lds_setup_emitted = False
    exec_mask_emitted = False
    i = 0

    while i < len(lines):
        if i + 11 < len(lines) and read_re.match(lines[i]):
            agprs = []
            ok = True
            for k in range(4):
                base = i + k * 3
                m = read_re.match(lines[base])
                if not (
                    m
                    and nop_re.match(lines[base + 1])
                    and swap_re.match(lines[base + 2])
                ):
                    ok = False
                    break
                agprs.append(int(m.group(2)))
            if not ok:
                out.append(lines[i])
                i += 1
                continue

            swap_end = i + 12
            j = swap_end
            srd = None
            while j < len(lines):
                sm = store_off12_re.match(lines[j])
                if sm:
                    srd = sm.group(2)
                    break
                j += 1

            if srd is None:
                out.append(lines[i])
                i += 1
                continue

            tile_end = j + 1
            tile_count += 1

            middle = lines[swap_end:tile_end]
            preserved = []
            mi = 0
            while mi < len(middle):
                s = middle[mi].strip()

                if sub4_re.match(s):
                    preserved.append(middle[mi])
                    if mi + 2 < len(middle):
                        preserved.append(middle[mi + 1])
                        preserved.append(middle[mi + 2])
                        mi += 3
                    else:
                        mi += 1
                    continue

                cm = cndmask_re.match(s)
                if cm and cm.group(1) == "v244":
                    preserved.append(middle[mi])
                    mi += 1
                    continue

                if (
                    cmp_ne_v244_re.match(s)
                    or (cm and cm.group(1) != "v244")
                    or s.startswith("v_accvgpr_read_b32")
                    or s.startswith("v_cvt_pk_bf16_f32")
                    or s.startswith("buffer_store_dword")
                    or lshlrev1_re.match(s)
                ):
                    mi += 1
                    continue

                preserved.append(middle[mi])
                mi += 1

            if not lds_setup_emitted:
                out.extend(
                    [
                        "  ;; bpermute epilogue with exec-masked stores",
                        "  v_and_b32 v4, 4294967279, v238",
                        "  v_lshlrev_b32 v4, 2, v4",
                        "  v_or_b32 v5, v238, 16",
                        "  v_lshlrev_b32 v5, 2, v5",
                    ]
                )
                lds_setup_emitted = True

            out.extend(preserved)

            if not exec_mask_emitted:
                out.extend(
                    [
                        "  ;; exec mask: only low lanes (0-15, 32-47) store",
                        "  s_mov_b64 s[20:21], exec",
                        "  v_cmp_ne_u32 vcc, v244, 0",
                        "  s_and_b64 s[22:23], s[20:21], vcc",
                    ]
                )
                exec_mask_emitted = True

            a0, a1, a2, a3 = agprs
            out.extend(
                [
                    f"  v_accvgpr_read_b32 v0, a{a0}",
                    f"  v_accvgpr_read_b32 v1, a{a1}",
                    "  v_cvt_pk_bf16_f32 v2, v0, v1",
                    f"  v_accvgpr_read_b32 v0, a{a2}",
                    f"  v_accvgpr_read_b32 v1, a{a3}",
                    "  v_cvt_pk_bf16_f32 v3, v0, v1",
                    "  ds_bpermute_b32 v0, v4, v2",
                    "  ds_bpermute_b32 v1, v4, v3",
                    "  ds_bpermute_b32 v2, v5, v2",
                    "  ds_bpermute_b32 v3, v5, v3",
                    "  s_waitcnt lgkmcnt(0)",
                    "  v_lshlrev_b32 v245, 1, v253",
                    "  s_mov_b64 exec, s[22:23]",
                    f"  buffer_store_dwordx4 v[0:3], v245, {srd}, 0 offen",
                    "  s_mov_b64 exec, s[20:21]",
                ]
            )

            i = tile_end
            continue

        out.append(lines[i])
        i += 1

    return "\n".join(out)


def bpermute_pipelined_epilogue_transform(asm_text):
    """Bpermute epilogue with software pipelining to hide ds_bpermute latency.

    Uses two alternating register sets: even tiles in v[0:3], odd tiles in
    v[6:9].  Tile N+1's AGPR read+cvt_pk overlaps with tile N's in-flight
    ds_bpermute, hiding the LDS crossbar latency behind VALU work.

    Byte offsets pre-saved in v10 (even) / v11 (odd) so the deferred store
    uses the correct address after v253 has been updated for the next tile.
    """
    read_re = re.compile(r"^\s*v_accvgpr_read_b32\s+(v\d+),\s+a(\d+)")
    nop_re = re.compile(r"^s_nop 1\s*$")
    swap_re = re.compile(r"^\s*v_permlane16_swap_b32\s+")
    store_off12_re = re.compile(
        r"^\s*buffer_store_dword\s+v\d+,\s+(v\d+),\s+(s\[\d+:\d+\]),\s+0 offen\s+offset:12\s*$"
    )
    sub4_re = re.compile(r"^\s*v_sub_u32\s+v\d+,\s+v\d+,\s+4\s*$")
    cmp_ne_v244_re = re.compile(r"^\s*v_cmp_ne_u32\s+vcc,\s+v244,\s+0")
    cndmask_re = re.compile(r"^\s*v_cndmask_b32\s+(v\d+),")
    lshlrev1_re = re.compile(r"^\s*v_lshlrev_b32\s+v\d+,\s+1,")

    lines = asm_text.split("\n")
    out = []
    tile_count = 0
    lds_setup_emitted = False
    exec_mask_emitted = False
    first_srd = None
    pending_store = None
    i = 0

    while i < len(lines):
        if i + 11 < len(lines) and read_re.match(lines[i]):
            agprs = []
            ok = True
            for k in range(4):
                base = i + k * 3
                m = read_re.match(lines[base])
                if not (
                    m
                    and nop_re.match(lines[base + 1])
                    and swap_re.match(lines[base + 2])
                ):
                    ok = False
                    break
                agprs.append(int(m.group(2)))
            if not ok:
                out.append(lines[i])
                i += 1
                continue

            swap_end = i + 12
            j = swap_end
            srd = None
            while j < len(lines):
                sm = store_off12_re.match(lines[j])
                if sm:
                    srd = sm.group(2)
                    break
                j += 1

            if srd is None:
                out.append(lines[i])
                i += 1
                continue

            if first_srd is None:
                first_srd = srd

            tile_end = j + 1
            tile_count += 1

            middle = lines[swap_end:tile_end]
            preserved = []
            mi = 0
            while mi < len(middle):
                s = middle[mi].strip()
                if sub4_re.match(s):
                    preserved.append(middle[mi])
                    if mi + 2 < len(middle):
                        preserved.append(middle[mi + 1])
                        preserved.append(middle[mi + 2])
                        mi += 3
                    else:
                        mi += 1
                    continue
                cm = cndmask_re.match(s)
                if cm and cm.group(1) == "v244":
                    preserved.append(middle[mi])
                    mi += 1
                    continue
                if (
                    cmp_ne_v244_re.match(s)
                    or (cm and cm.group(1) != "v244")
                    or s.startswith("v_accvgpr_read_b32")
                    or s.startswith("v_cvt_pk_bf16_f32")
                    or s.startswith("buffer_store_dword")
                    or lshlrev1_re.match(s)
                ):
                    mi += 1
                    continue
                preserved.append(middle[mi])
                mi += 1

            is_even = (tile_count - 1) % 2 == 0
            if is_even:
                rd0, rd1, cv0, cv1 = "v0", "v1", "v2", "v3"
                store_reg, addr_reg = "v[0:3]", "v10"
            else:
                rd0, rd1, cv0, cv1 = "v6", "v7", "v8", "v9"
                store_reg, addr_reg = "v[6:9]", "v11"

            if not lds_setup_emitted:
                out.extend(
                    [
                        "  ;; pipelined bpermute epilogue with exec-masked stores",
                        "  v_and_b32 v4, 4294967279, v238",
                        "  v_lshlrev_b32 v4, 2, v4",
                        "  v_or_b32 v5, v238, 16",
                        "  v_lshlrev_b32 v5, 2, v5",
                    ]
                )
                lds_setup_emitted = True

            out.extend(preserved)

            if not exec_mask_emitted:
                out.extend(
                    [
                        "  ;; exec mask: only low lanes (0-15, 32-47) store",
                        "  s_mov_b64 s[20:21], exec",
                        "  v_cmp_ne_u32 vcc, v244, 0",
                        "  s_and_b64 s[22:23], s[20:21], vcc",
                    ]
                )
                exec_mask_emitted = True

            a0, a1, a2, a3 = agprs

            out.append(f"  v_lshlrev_b32 {addr_reg}, 1, v253")

            out.extend(
                [
                    f"  v_accvgpr_read_b32 {rd0}, a{a0}",
                    f"  v_accvgpr_read_b32 {rd1}, a{a1}",
                    f"  v_cvt_pk_bf16_f32 {cv0}, {rd0}, {rd1}",
                    f"  v_accvgpr_read_b32 {rd0}, a{a2}",
                    f"  v_accvgpr_read_b32 {rd1}, a{a3}",
                    f"  v_cvt_pk_bf16_f32 {cv1}, {rd0}, {rd1}",
                ]
            )

            if pending_store is not None:
                ps_reg, pa_reg = pending_store
                out.extend(
                    [
                        "  s_waitcnt lgkmcnt(0)",
                        "  s_mov_b64 exec, s[22:23]",
                        f"  buffer_store_dwordx4 {ps_reg}, {pa_reg}, {first_srd}, 0 offen",
                        "  s_mov_b64 exec, s[20:21]",
                    ]
                )

            out.extend(
                [
                    f"  ds_bpermute_b32 {rd0}, v4, {cv0}",
                    f"  ds_bpermute_b32 {rd1}, v4, {cv1}",
                    f"  ds_bpermute_b32 {cv0}, v5, {cv0}",
                    f"  ds_bpermute_b32 {cv1}, v5, {cv1}",
                ]
            )

            pending_store = (store_reg, addr_reg)

            i = tile_end
            continue

        if pending_store is not None and lines[i].strip() == "s_endpgm":
            ps_reg, pa_reg = pending_store
            out.extend(
                [
                    "  s_waitcnt lgkmcnt(0)",
                    "  s_mov_b64 exec, s[22:23]",
                    f"  buffer_store_dwordx4 {ps_reg}, {pa_reg}, {first_srd}, 0 offen",
                    "  s_mov_b64 exec, s[20:21]",
                ]
            )
            pending_store = None

        out.append(lines[i])
        i += 1

    return "\n".join(out)


def _run_mxfp_gemm(gemm, shape):
    """Run compiled GEMM kernel and verify against reference."""
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    x, w = x.cuda(), w.cuda()
    x_scales, w_scales = x_scales.cuda(), w_scales.cuda()
    out = torch.zeros(x.shape[0], w.shape[1], dtype=torch.float32).cuda()

    gemm(x, x_scales, w.T.contiguous(), w_scales, out)
    torch.testing.assert_close(
        torch_out, out.cpu(), check_dtype=False, check_device=False
    )


def _run_mxfp_gemm_preshuffle(
    gemm,
    shape,
    output_dtype=torch.float32,
    swap_inputs=False,
    **kwargs,
):
    """Run compiled GEMM kernel, verify against reference.

    When swap_inputs is True, the kernel computes C^T = B x A^T (with A=X, B=W)
    and writes C [M, N] directly via transpose_output + coalesced epilogue.
    When swap_inputs is False (baseline), uses standard input order.
    """
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    w_t = w.T.contiguous()

    if swap_inputs:
        kern_a = w_t.cuda()
        kern_a_scale = e8m0_shuffle(w_scales).cuda()
        kern_b = b_preshuffle(x).cuda()
        kern_b_scale = e8m0_shuffle(x_scales).cuda()
        out = torch.zeros(shape[0], shape[1], dtype=output_dtype).cuda()
        gemm(kern_a, kern_a_scale, kern_b, kern_b_scale, out)
        result = out.cpu()
    else:
        kern_a = x.cuda()
        kern_b = b_preshuffle(w_t).cuda()
        kern_a_scale = e8m0_shuffle(x_scales).cuda()
        kern_b_scale = e8m0_shuffle(w_scales).cuda()
        out = torch.zeros(shape[0], shape[1], dtype=output_dtype).cuda()
        gemm(kern_a, kern_a_scale, kern_b, kern_b_scale, out)
        result = out.cpu()

    if os.environ.get("WAVE_DEBUG_COMPARE"):
        ref = torch_out.to(torch.float32).cpu()
        got = result.to(torch.float32).cpu()
        mismatch = ~torch.isclose(ref, got, atol=1e-1, rtol=0.1)
        M, N = ref.shape
        wrong_idx = torch.nonzero(mismatch)
        if wrong_idx.numel() > 0:
            print(f"  Mismatch count: {mismatch.sum().item()} / {mismatch.numel()}")
            m_wrong = wrong_idx[:, 0]
            m_mod8 = m_wrong % 8
            print(
                f"  Mismatch M%8 distribution: {torch.bincount(m_mod8, minlength=8).tolist()}"
            )
            # Check if got values at wrong M match ref at shifted M
            print("  === Shift analysis (checking if got[m,n] == ref[m+shift,n]) ===")
            for shift in [-4, -3, -2, -1, 1, 2, 3, 4]:
                match_count = 0
                total = 0
                for i in range(min(1000, wrong_idx.shape[0])):
                    mi, ni = wrong_idx[i].tolist()
                    ms = mi + shift
                    if 0 <= ms < M:
                        total += 1
                        if torch.isclose(
                            got[mi, ni : ni + 1],
                            ref[ms, ni : ni + 1],
                            atol=1e-1,
                            rtol=0.1,
                        ).item():
                            match_count += 1
                if total > 0:
                    print(
                        f"    shift={shift:+d}: {match_count}/{total} ({100*match_count/total:.0f}%)"
                    )
            # Print samples with neighboring values
            print("  === Sample values (M=0..7 at N=0) ===")
            for m in range(min(8, M)):
                r = ref[m, 0].item()
                g = got[m, 0].item()
                ok = (
                    "OK"
                    if torch.isclose(
                        ref[m, 0:1], got[m, 0:1], atol=1e-1, rtol=0.1
                    ).item()
                    else "WRONG"
                )
                print(f"    M={m} ref={r:.4f} got={g:.4f} {ok}")
        else:
            print("  No mismatches (within tolerance)")

    torch.testing.assert_close(torch_out, result, check_dtype=False, check_device=False)


def _get_8wave_shape_from_block(block):
    """Choose an 8-wave shape (4x2 or 2x4) from block M/N dims.

    If either block M or N is 32, force that corresponding wave dimension to 2.
    """
    m_blk, n_blk = block[0], block[1]
    if m_blk == 32 and n_blk == 32:
        raise ValueError(
            "Cannot satisfy both M and N=32 with an 8-wave shape constrained to (4, 2) or (2, 4)."
        )
    if m_blk == 32:
        return (2, 4)
    if n_blk == 32:
        return (4, 2)
    return (4, 2)


def test_dbuf_4wave_mxfp_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)
):
    """Double-buffered MXFP4 GEMM, 4 waves, no stagger."""
    gemm, options = get_tagged_mxfp4_gemm(shape, block, wave_shape=(2, 2))
    schedule = get_mxfp4_dbuf_schedule(use_stagger=False)

    options.print_ir_after = "all" if is_debug else []
    options.print_mlir_file = "gemm_mxfp4_dbuf_4wave.mlir"
    options.print_mlir = True
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm(gemm, shape)
    print("MXFP GEMM double-buffer 4-wave test passed!")


def test_dbuf_8wave_pingpong_mxfp_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256), dynamic=False
):
    """Double-buffered MXFP4 GEMM, 8 waves, ping-pong with stagger.
    A&B scales are preshuffled and read from global memory directly to VGPRs.
    A and B are read from global memory directly to LDS.

    Note: for dynamic mode, keep block MxN at or below 128x256 or 256x128
    to avoid exceeding shared-memory limits.
    """
    wave_shape = _get_8wave_shape_from_block(block)
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales(
        shape, block, wave_shape=wave_shape
    )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True

    if dynamic:
        options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
        for sym in options.dynamic_symbols:
            del options.subs[sym]

    schedule = get_mxfp4_dbuf_pingpong_schedule(use_stagger=True, shape=shape)

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, only_scale=True)
    mode = "dynamic" if dynamic else "static"
    print(
        f"MXFP GEMM double-buffer 8-wave ping pong with scale shuffling ({mode}) test passed!"
    )


def test_dbuf_8wave_pingpong_mxfp_gemm_Bshuffle(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256), dynamic=False
):
    """Double-buffered MXFP4 GEMM, 8 waves, ping-pong with stagger.
    A&B scales are preshuffled and read from global memory directly to VGPRs.
    Same for B data. However, prefetching shuffled B directly to VGPR consumes too many VGPRs and causes spilling.
    A is read from global memory directly to LDS.
    """
    wave_shape = _get_8wave_shape_from_block(block)
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales_and_B(
        shape, block, wave_shape=wave_shape
    )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True

    if dynamic:
        options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
        for sym in options.dynamic_symbols:
            del options.subs[sym]
    schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled(use_stagger=True, shape=shape)

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True)
    mode = "dynamic" if dynamic else "static"
    print(
        f"MXFP GEMM double-buffer 8-wave ping pong with scale and B shuffling and B->VGPR ({mode}) test passed!"
    )


def test_dbuf_8wave_pingpong_mxfp_gemm_Bshuffle_lds(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256), dynamic=False
):
    """Double-buffered MXFP4 GEMM, 8 waves, ping-pong with stagger.
    A&B scales are preshuffled and read from global memory directly to VGPRs.
    B data is preshuffled and loaded to LDS (shared memory), not directly to VGPRs.
    A data is read from global memory directly to LDS.
    """
    wave_shape = _get_8wave_shape_from_block(block)
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales_and_B(
        shape,
        block,
        wave_shape=wave_shape,
        b_address_space=SHARED_ADDRESS_SPACE,
    )
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = False
    options.linearize_shared_access = True

    if dynamic:
        options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
        for sym in options.dynamic_symbols:
            del options.subs[sym]
    schedule = get_mxfp4_dbuf_pingpong_schedule_Bshuffled_lds(
        use_stagger=True, shape=shape
    )

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True)
    mode = "dynamic" if dynamic else "static"
    print(
        f"MXFP GEMM double-buffer 8-wave ping pong with scales and B shuffling and B->LDS ({mode}) test passed!"
    )


def test_dbuf_8wave_mixed_pingpong_mxfp_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)
):
    """Double-buffered MXFP4 GEMM, 8 waves, with stagger.

    A variant of the ping-pong schedule that hides the latency of the extra
    WorkgroupBarrier required for large shapes. With staggering, the two
    clusters of waves write to LDS at different times.
    When the bus becomes congested, memory operations loaded by the later cluster may not arrive
    in LDS before the other cluster attempts to read from it. In this case,
    we add a second workgroup barrier to fix the timing and prevent incorrect output results.

    This schedule overlaps that barrier with useful work by splitting LDS loads:
      - "Safe" loads: rows this wave wrote itself — readable immediately after
        memory_counter_wait, before the global WorkgroupBarrier.
      - "Dependent" loads: rows written by other waves — deferred until after
        the global WorkgroupBarrier.

    This lets the MFMAs on the safe operands start firing as soon as the
    barrier releases, effectively hiding the second barrier's latency behind
    the early loads and compute.
    """
    gemm, options = get_tagged_mxfp4_gemm(shape, block, wave_shape=(4, 2))
    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    schedule = get_mxfp4_dbuf_mixed_pingpong_schedule(use_stagger=True)

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm(gemm, shape)
    print("MXFP GEMM double-buffer 8-wave mixed ping pong test passed!")


def test_dbuf_8wave_mixed_pingpong_shuffle_mxfp_gemm(
    is_debug=False, shape=(16384, 16384, 16384), block=(256, 256, 256)
):
    """Like :func:`test_dbuf_8wave_mixed_pingpong_mxfp_gemm` but with A_scale & B_scale
    preshuffled and prefetched to VGPRs.

    Note: preshuffling B and loading it directly to VGPRs combined with prefetching
    consumes too many VGPRs and causes spilling.
    """

    gemm, options = get_tagged_mxfp4_gemm_preshuffle_scales(
        shape, block, wave_shape=(4, 2)
    )

    options.specialize = True
    options.use_buffer_ops = True
    options.minimize_shared_allocs = True
    schedule = get_mxfp4_dbuf_mixed_pingpong_shuffle_schedule(use_stagger=True)

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, only_scale=True)
    print("MXFP GEMM double-buffer 8-wave mixed ping pong with shuffling test passed!")


def test_dbuf_4wave_mxfp_asymmetric_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(256, 256, 256)
):
    """Asymmetric-prefetch MXFP4 GEMM: A through LDS (2x prefetch), B direct from global."""
    gemm, options = get_tagged_mxfp4_gemm(
        shape, block, wave_shape=(1, 4), b_address_space=GLOBAL_ADDRESS_SPACE
    )
    options.print_mlir_file = "gemm_mxfp4_dbuf_4wave_asymmetric.mlir"
    options.print_mlir = True
    options.dump_binaries = "build/binaries"
    options.dump_intermediates = "build/intermediates"
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.use_buffer_ops = True
    options.use_water_backend = True
    schedule = get_mxfp4_asymmetric_schedule()

    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm(gemm, shape)
    print("MXFP GEMM asymmetric-prefetch 4-wave test passed!")


def test_dbuf_4wave_mxfp_preshuffle_b_gemm(
    is_debug=False,
    shape=(1024, 1024, 8192),
    block=(128, 256, 256),
    eliminate_epilogue=True,
):
    """Asymmetric MXFP4 GEMM with preshuffled B data and B scales."""
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(shape, block, wave_shape=(1, 4))
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.use_buffer_ops = True
    options.eliminate_epilogue = eliminate_epilogue
    options.dump_intermediates = "build/intermediates/"
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )

    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True)
    print("MXFP GEMM preshuffle-B 4-wave test passed!")


def test_dbuf_4wave_mxfp_asymmetric_gemm_cpp(
    is_debug=False, shape=(1024, 1024, 8192), block=(128, 256, 256)
):
    """Asymmetric MXFP4 GEMM using C++ WaveASM backend (no preshuffle)."""
    gemm, options = get_tagged_mxfp4_gemm(
        shape, block, wave_shape=(1, 4), b_address_space=GLOBAL_ADDRESS_SPACE
    )
    options.backend = "asm"
    options.wave_runtime = True
    options.dump_intermediates = "build/intermediates"
    schedule = get_mxfp4_asymmetric_schedule()
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm(gemm, shape)
    print("MXFP GEMM asymmetric 4-wave (WaveASM backend) test passed!")


def test_dbuf_4wave_mxfp_preshuffle_b_gemm_cpp(
    is_debug=False,
    shape=(512, 1024, 8192),  # 4*T0, 4*T1, 8192
    block=(128, 256, 256),
    eliminate_epilogue=True,
):
    """Preshuffle-B MXFP4 GEMM using C++ WaveASM backend."""
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape, block, wave_shape=(2, 2), reorder_workgroups=True
    )
    options.backend = "asm"
    options.use_buffer_ops = True
    options.wave_runtime = True
    options.use_wave_asm_backend = True
    options.dump_intermediates = "build/intermediates"
    options.eliminate_epilogue = eliminate_epilogue
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True)
    print(
        f"MXFP GEMM preshuffle-B 4-wave (WaveASM) epilogue elimination={eliminate_epilogue} PASSED"
    )


def test_dbuf_4wave_mxfp_dynamic_preshuffle_b_gemm(
    is_debug=False,
    shape=(1024, 6144, 8192),
    block=(256, 192, 256),
    eliminate_epilogue=True,
):
    """Preshuffle-B MXFP4 GEMM with dynamic M, N, K."""
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape,
        block,
        wave_shape=(1, 4),
        reorder_workgroups=True,
        output_dtype=tkl.bf16,
        transpose_output=True,
    )
    # Make M, N, K dynamic so the compiler does not specialize on problem size.
    dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    for sym in dynamic_symbols:
        del options.subs[sym]
    options.dynamic_symbols = dynamic_symbols
    options.use_buffer_ops = True
    options.backend = "llvm"
    options.wave_runtime = True
    options.eliminate_epilogue = eliminate_epilogue
    options.dump_intermediates = "build/intermediates/"
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(
        gemm, shape, all=True, output_dtype=torch.bfloat16, transpose_output=True
    )
    print("MXFP GEMM preshuffle-B 4-wave dynamic M, N, K (LLVM backend) test passed!")


def test_dbuf_4wave_mxfp_dynamic_preshuffle_b_gemm_asm(
    is_debug=False,
    shape=(1024, 6144, 8192),
    block=(256, 192, 256),
    eliminate_epilogue=True,
):
    """Preshuffle-B MXFP4 GEMM with dynamic M, N, K."""
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape, block, wave_shape=(2, 2), reorder_workgroups=True
    )
    # Make M, N, K dynamic so the compiler does not specialize on problem size.
    dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    for sym in dynamic_symbols:
        del options.subs[sym]
    options.dynamic_symbols = dynamic_symbols
    options.use_buffer_ops = True
    options.backend = "asm"
    options.use_wave_asm_backend = True
    options.wave_runtime = True
    options.eliminate_epilogue = eliminate_epilogue
    options.dump_intermediates = "build/intermediates/waveasm_256x192x256_baseline/"
    options.print_mlir_file = "gemm_mxfp4_dbuf_4wave_asymmetric.mlir"
    options.print_mlir = True
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    with open(
        "build/intermediates/waveasm_256x192x256_baseline/gemm_mxfp4_dbuf_4wave_asymmetric.mlir",
        "w",
    ) as f:
        f.write(gemm.asm)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True)
    print(
        "MXFP GEMM preshuffle-B 4-wave dynamic M, N, K (WaveASM backend) test passed!"
    )


def test_dbuf_4wave_mxfp_dynamic_preshuffle_b_gemm_asm_bf16(
    is_debug=False,
    shape=(6400, 3072, 7168),
    block=(256, 192, 256),
    eliminate_epilogue=True,
):
    """Preshuffle-B MXFP4 GEMM with dynamic M, N, K and bf16 output (WaveASM)."""
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape,
        block,
        wave_shape=(2, 2),
        reorder_workgroups=True,
        output_dtype=tkl.bf16,
    )
    dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    for sym in dynamic_symbols:
        del options.subs[sym]
    options.dynamic_symbols = dynamic_symbols
    options.use_buffer_ops = True
    options.backend = "asm"
    options.use_wave_asm_backend = True
    options.wave_runtime = True
    options.eliminate_epilogue = eliminate_epilogue
    options.dump_intermediates = (
        "build/intermediates/waveasm_256x192x256_bf16_baseline/"
    )
    options.print_mlir_file = "gemm_mxfp4_dbuf_4wave_asymmetric_bf16.mlir"
    options.print_mlir = True
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    with open(
        "build/intermediates/waveasm_256x192x256_bf16_baseline/gemm_mxfp4_dbuf_4wave_asymmetric_bf16.mlir",
        "w",
    ) as f:
        f.write(gemm.asm)

    _run_mxfp_gemm_preshuffle(gemm, shape, all=True, output_dtype=torch.bfloat16)
    print(
        "MXFP GEMM preshuffle-B 4-wave dynamic M, N, K bf16 (WaveASM backend) test passed!"
    )


def test_dbuf_4wave_mxfp_dynamic_preshuffle_b_gemm_asm_bf16_coalesced(
    is_debug=False,
    shape=(6400, 3072, 7168),
    block=(256, 192, 256),
    eliminate_epilogue=True,
):
    """Preshuffle-B MXFP4 GEMM bf16 with coalesced epilogue stores via permlane swap (WaveASM)."""
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape,
        block,
        wave_shape=(2, 2),
        reorder_workgroups=True,
        output_dtype=tkl.bf16,
        transpose_output=True,
    )
    dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    for sym in dynamic_symbols:
        del options.subs[sym]
    options.dynamic_symbols = dynamic_symbols
    options.use_buffer_ops = True
    options.backend = "asm"
    options.use_wave_asm_backend = True
    options.wave_runtime = True
    options.eliminate_epilogue = eliminate_epilogue
    options.coalesce_epilogue_stores = True
    options.dump_intermediates = (
        "build/intermediates/waveasm_256x192x256_bf16_coalesced/"
    )
    options.print_mlir_file = "gemm_mxfp4_dbuf_4wave_asymmetric_bf16_coalesced.mlir"
    options.print_mlir = True
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    with open(
        "build/intermediates/waveasm_256x192x256_bf16_coalesced/gemm_mxfp4_dbuf_4wave_asymmetric_bf16_coalesced.mlir",
        "w",
    ) as f:
        f.write(gemm.asm)

    _run_mxfp_gemm_preshuffle(
        gemm, shape, all=True, output_dtype=torch.bfloat16, transpose_output=True
    )
    print(
        "MXFP GEMM preshuffle-B 4-wave bf16 coalesced epilogue (WaveASM backend) test passed!"
    )


def test_dbuf_4wave_mxfp_dynamic_preshuffle_b_gemm_asm_bf16_coalesced_dwordx4(
    is_debug=False,
    shape=(6400, 3072, 7168),
    block=(256, 192, 256),
    eliminate_epilogue=True,
):
    """Same as bf16_coalesced but with post-asm dwordx4 store coalescing."""
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape,
        block,
        wave_shape=(2, 2),
        reorder_workgroups=True,
        output_dtype=tkl.bf16,
        transpose_output=True,
    )
    dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    for sym in dynamic_symbols:
        del options.subs[sym]
    options.dynamic_symbols = dynamic_symbols
    options.use_buffer_ops = True
    options.backend = "asm"
    options.use_wave_asm_backend = True
    options.wave_runtime = True
    options.eliminate_epilogue = eliminate_epilogue
    options.coalesce_epilogue_stores = True
    options.asm_transform = coalesce_buffer_stores_dwordx4
    options.dump_intermediates = (
        "build/intermediates/waveasm_256x192x256_bf16_coalesced_dwordx4/"
    )
    options.print_mlir_file = (
        "gemm_mxfp4_dbuf_4wave_asymmetric_bf16_coalesced_dwordx4.mlir"
    )
    options.print_mlir = True
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(
        gemm, shape, all=True, output_dtype=torch.bfloat16, transpose_output=True
    )
    print("MXFP GEMM bf16 coalesced dwordx4 epilogue (WaveASM backend) test passed!")


def test_dbuf_4wave_mxfp_dynamic_preshuffle_b_gemm_asm_bf16_cvt_first(
    is_debug=False,
    shape=(6400, 3072, 7168),
    block=(256, 192, 256),
    eliminate_epilogue=True,
):
    """Same as bf16_coalesced but with convert-first epilogue (no cndmask)."""
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape,
        block,
        wave_shape=(2, 2),
        reorder_workgroups=True,
        output_dtype=tkl.bf16,
        transpose_output=True,
    )
    dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    for sym in dynamic_symbols:
        del options.subs[sym]
    options.dynamic_symbols = dynamic_symbols
    options.use_buffer_ops = True
    options.backend = "asm"
    options.use_wave_asm_backend = True
    options.wave_runtime = True
    options.eliminate_epilogue = eliminate_epilogue
    options.coalesce_epilogue_stores = True
    options.asm_transform = convert_first_eliminate_cndmask
    options.dump_intermediates = (
        "build/intermediates/waveasm_256x192x256_bf16_cvt_first/"
    )
    options.print_mlir_file = "gemm_mxfp4_dbuf_4wave_asymmetric_bf16_cvt_first.mlir"
    options.print_mlir = True
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(
        gemm, shape, output_dtype=torch.bfloat16, swap_inputs=True
    )
    print("MXFP GEMM bf16 convert-first epilogue (WaveASM backend) test passed!")


def test_dbuf_4wave_mxfp_dynamic_preshuffle_b_gemm_asm_bf16_transpose_only(
    is_debug=False,
    shape=(6400, 3072, 7168),
    block=(256, 192, 256),
    eliminate_epilogue=True,
):
    """bf16 with transpose_output=True but NO coalesce_epilogue_stores (simple stores)."""
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape,
        block,
        wave_shape=(2, 2),
        reorder_workgroups=True,
        output_dtype=tkl.bf16,
        transpose_output=True,
    )
    dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    for sym in dynamic_symbols:
        del options.subs[sym]
    options.dynamic_symbols = dynamic_symbols
    options.use_buffer_ops = True
    options.backend = "asm"
    options.use_wave_asm_backend = True
    options.wave_runtime = True
    options.eliminate_epilogue = eliminate_epilogue
    options.dump_intermediates = (
        "build/intermediates/waveasm_256x192x256_bf16_transpose_only/"
    )
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(
        gemm, shape, all=True, output_dtype=torch.bfloat16, transpose_output=True
    )
    print("MXFP GEMM bf16 transpose-only (no coalesce) test passed!")


def test_dbuf_4wave_mxfp_dynamic_preshuffle_b_gemm_asm_bf16_lds_epilogue(
    is_debug=False,
    shape=(6400, 3072, 7168),
    block=(256, 192, 256),
    eliminate_epilogue=True,
):
    """bf16 pipelined bpermute epilogue with exec-masked stores."""
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape,
        block,
        wave_shape=(2, 2),
        reorder_workgroups=True,
        output_dtype=tkl.bf16,
        transpose_output=True,
    )
    dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    for sym in dynamic_symbols:
        del options.subs[sym]
    options.dynamic_symbols = dynamic_symbols
    options.use_buffer_ops = True
    options.backend = "asm"
    options.use_wave_asm_backend = True
    options.wave_runtime = True
    options.eliminate_epilogue = eliminate_epilogue
    options.coalesce_epilogue_stores = True
    options.asm_transform = bpermute_pipelined_epilogue_transform
    options.dump_intermediates = (
        "build/intermediates/waveasm_256x192x256_bf16_lds_epilogue/"
    )
    options.print_mlir_file = "gemm_mxfp4_dbuf_4wave_asymmetric_bf16_lds_epilogue.mlir"
    options.print_mlir = True
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=eliminate_epilogue, is_bscale_shuffled=True
    )
    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle(
        gemm, shape, output_dtype=torch.bfloat16, swap_inputs=True
    )
    print("MXFP GEMM bf16 pipelined bpermute epilogue (WaveASM backend) test passed!")


def _compile_bf16_kernel(
    block,
    *,
    coalesce=False,
    dwordx4=False,
    convert_first=False,
    lds_epilogue=False,
    bpermute_masked=False,
    bpermute_pipelined=False,
    swap_inputs=False,
    transpose_only=False,
):
    """Compile a bf16 kernel once (M,N,K dynamic). Returns (kernel, mode_str).

    mode_str is one of: False (baseline), True (transpose), "swap" (input swap).
    """
    shape_placeholder = (block[0] * 4, block[1] * 4, block[2] * 4)
    use_transpose = (
        coalesce
        or lds_epilogue
        or bpermute_masked
        or bpermute_pipelined
        or swap_inputs
        or convert_first
        or transpose_only
    )
    kwargs = dict(
        wave_shape=(2, 2),
        reorder_workgroups=True,
        output_dtype=tkl.bf16,
    )
    if use_transpose:
        kwargs["transpose_output"] = True
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape_placeholder, block, **kwargs
    )
    for sym in [tkl.sym.M, tkl.sym.N, tkl.sym.K]:
        del options.subs[sym]
    options.dynamic_symbols = [tkl.sym.M, tkl.sym.N, tkl.sym.K]
    options.use_buffer_ops = True
    options.backend = "asm"
    options.use_wave_asm_backend = True
    options.wave_runtime = True
    options.eliminate_epilogue = True
    if (
        coalesce
        or lds_epilogue
        or bpermute_masked
        or bpermute_pipelined
        or swap_inputs
        or convert_first
    ):
        options.coalesce_epilogue_stores = True
    if bpermute_pipelined:
        options.asm_transform = bpermute_pipelined_epilogue_transform
    elif convert_first:
        options.asm_transform = convert_first_eliminate_cndmask
    elif swap_inputs:
        options.asm_transform = bpermute_masked_epilogue_transform
    elif bpermute_masked:
        options.asm_transform = bpermute_masked_epilogue_transform
    elif lds_epilogue:
        options.asm_transform = lds_epilogue_transform
    elif dwordx4:
        options.asm_transform = coalesce_buffer_stores_dwordx4
    schedule = get_mxfp4_asymmetric_schedule(
        eliminate_epilogue=True,
        is_bscale_shuffled=True,
    )
    options = set_default_run_config(options)
    mode = "swap" if swap_inputs else use_transpose
    return wave_compile(options, gemm, schedule), mode


def _time_kernel(gemm, shape, warmup=2, iters=5, swap_inputs=False, **kwargs):
    """Time a compiled GEMM kernel on the given shape. Returns median us."""
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    w_t = w.T.contiguous()

    if swap_inputs:
        kern_a = w_t.cuda()
        kern_a_scale = e8m0_shuffle(w_scales).cuda()
        kern_b = b_preshuffle(x).cuda()
        kern_b_scale = e8m0_shuffle(x_scales).cuda()
    else:
        kern_a = x.cuda()
        kern_b = b_preshuffle(w_t).cuda()
        kern_a_scale = e8m0_shuffle(x_scales).cuda()
        kern_b_scale = e8m0_shuffle(w_scales).cuda()
    out = torch.zeros(shape[0], shape[1], dtype=torch.bfloat16).cuda()

    for _ in range(warmup):
        gemm(kern_a, kern_a_scale, kern_b, kern_b_scale, out)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        gemm(kern_a, kern_a_scale, kern_b, kern_b_scale, out)
        end_events[i].record()
    torch.cuda.synchronize()

    times = sorted([s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)])
    return times[len(times) // 2]


def test_benchmark_bf16_shapes(is_debug=False, **kwargs):
    """Benchmark bf16 baseline vs masked vs pipelined (exec-masked stores)."""
    shapes = [
        (6400, 3072, 7168),
        (4608, 7680, 6656),
    ]
    block = (256, 192, 256)
    warmup = 10
    iters = 200
    rounds = 50

    print("Compiling bf16 baseline (no transpose)...")
    baseline_kernel, _ = _compile_bf16_kernel(block)
    print("Compiling bf16 bpermute-masked (exec-masked stores)...")
    masked_kernel, _ = _compile_bf16_kernel(block, bpermute_masked=True)
    print("Compiling bf16 bpermute-pipelined (SW pipelined + exec-masked)...")
    pipelined_kernel, _ = _compile_bf16_kernel(block, bpermute_pipelined=True)

    hdr = (
        f"{'Shape (M,N,K)':<30} {'baseline':>10} {'masked':>10} "
        f"{'pipelined':>10} {'pipe/base':>10}"
    )
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for shape in shapes:
        bt, mt, pt = [], [], []
        for _ in range(rounds):
            bt.append(
                _time_kernel(
                    baseline_kernel,
                    shape,
                    transpose_output=False,
                    warmup=warmup,
                    iters=iters,
                )
            )
            mt.append(
                _time_kernel(
                    masked_kernel,
                    shape,
                    transpose_output=True,
                    warmup=warmup,
                    iters=iters,
                )
            )
            pt.append(
                _time_kernel(
                    pipelined_kernel,
                    shape,
                    transpose_output=True,
                    warmup=warmup,
                    iters=iters,
                )
            )
        tb = sorted(bt)[len(bt) // 2]
        tm = sorted(mt)[len(mt) // 2]
        tp = sorted(pt)[len(pt) // 2]
        print(
            f"{str(shape):<30} {tb:>8.1f}us {tm:>8.1f}us "
            f"{tp:>8.1f}us {tb/tp:>9.3f}x"
        )
    print()


if __name__ == "__main__":
    args = parse_args()

    if args.list_tests:
        list_tests(globals())
        exit(0)

    if not args.test:
        print("Error: --test argument is required")
        print("Use --list_tests to see available tests")
        exit(1)

    success = run_test(
        args.test,
        globals(),
        args.debug,
        args.repeat,
        args.shape,
        args.block,
        args.eliminate_epilogue,
    )
    exit(0 if success else 1)
