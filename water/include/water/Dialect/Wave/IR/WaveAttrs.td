// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_WAVEATTRS
#define WATER_DIALECT_WAVE_WAVEATTRS

include "water/Dialect/Wave/IR/WaveDialect.td"
include "mlir/IR/AttrTypeBase.td"

def WaveSymbolAttr : AttrDef<WaveDialect, "WaveSymbol"> {
  let mnemonic = "symbol";
  let description = [{
    Base symbol used in symbolic expressions in the Wave dialect. This is
    different from the SymbolRef hierarchy of attributes since it refers to
    symbols, aka variables, in mathematical expressions as opposed to symbols
    identifying global objects such as functions and global variables in
    programs. Wave expression symbols are not part of the symbol table.
  }];

  let parameters = (ins StringRefParameter<"name of the symbol">:$name);
  let assemblyFormat = "`<` $name `>`";

  let extraClassDeclaration = [{
    void printAsSymbolRef(::llvm::raw_ostream &os) const {
      os << "@" << getName();
    }
  }];
}

#endif // WATER_DIALECT_WAVE_WAVEATTRS
