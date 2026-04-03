// RUN: waveasm-translate %s

// Test basic program structure with pure SSA ops
waveasm.program @simple_kernel
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 256 : i64, sgprs = 104 : i64} {

  // Define precolored ABI registers
  %tid = waveasm.precolored.vreg 0 : !waveasm.vreg

  // Pure SSA instructions - each instruction is its own op
  %sum = waveasm.v_add_u32 %tid, %tid : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

  // Comment
  waveasm.comment "Basic kernel body"

  // End program
  waveasm.s_endpgm
}
