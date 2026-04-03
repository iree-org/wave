// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveUtils.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Casting.h"
#include <optional>

using namespace mlir;

SmallVector<int64_t>
wave::getUncollapsedVectorShape(llvm::ArrayRef<wave::WaveSymbolAttr> shape,
                                DictionaryAttr indexDict,
                                wave::WaveHyperparameterAttr hyper) {
  return llvm::map_to_vector(shape, [&](wave::WaveSymbolAttr symbol) {
    Attribute entry = indexDict.get(symbol.getName());
    // The entry may be missing from the index and we shouldn't crash. Just
    // treat it as dynamic meaning we cannot statically evaluate it to a
    // constant.
    if (!entry)
      return ShapedType::kDynamic;
    auto mapAttr = cast<wave::WaveIndexMappingAttr>(entry);
    if (!mapAttr.getStep())
      return ShapedType::kDynamic;
    std::optional<SmallVector<int64_t>> folded =
        wave::evaluateMapWithHyperparams(mapAttr.getStep(),
                                         mapAttr.getSymbols(), hyper);
    if (!folded)
      return ShapedType::kDynamic;
    assert(folded->size() == 1 && "expected single-result map");
    return (*folded)[0];
  });
}

std::optional<int64_t>
wave::getPositionOfVectorizedDim(llvm::ArrayRef<wave::WaveSymbolAttr> shape,
                                 DictionaryAttr indexDict,
                                 wave::WaveHyperparameterAttr hyper) {
  int64_t bestIdx = -1;
  std::optional<int64_t> bestSize; // largest constant size seen so far
  for (auto [i, size] :
       llvm::enumerate(getUncollapsedVectorShape(shape, indexDict, hyper))) {
    if (ShapedType::isDynamic(size))
      return std::nullopt;
    if (!bestSize || size >= *bestSize) {
      bestSize = size;
      bestIdx = i;
    }
  }
  assert(bestIdx != -1);
  return bestIdx;
}

std::optional<llvm::SmallVector<int64_t>>
wave::resolveSymbolNames(llvm::ArrayRef<Attribute> symbols,
                         wave::WaveHyperparameterAttr hyper) {
  if (llvm::any_of(symbols, llvm::IsaPred<WaveIndexSymbolAttr>))
    return std::nullopt;

  if (!hyper)
    return std::nullopt;

  // Collect concrete values for each symbol in stored order.
  llvm::SmallVector<int64_t> symVals;
  symVals.reserve(symbols.size());
  for (Attribute attr : symbols) {
    wave::WaveSymbolAttr symbol = cast<wave::WaveSymbolAttr>(attr);
    auto value = hyper.getSymbolValue(symbol.getName());
    if (!value)
      return std::nullopt;
    symVals.push_back(*value);
  }
  return symVals;
}

std::optional<SmallVector<int64_t>>
wave::evaluateMapWithHyperparams(AffineMap map, ArrayRef<Attribute> symbols,
                                 wave::WaveHyperparameterAttr hyperparams) {
  SmallVector<AffineExpr> symReplacements;
  symReplacements.reserve(map.getNumSymbols());
  for (unsigned i = 0, e = map.getNumSymbols(); i < e; ++i) {
    if (llvm::none_of(map.getResults(), [i](AffineExpr expr) {
          return expr.isFunctionOfSymbol(i);
        })) {
      symReplacements.push_back(AffineExpr());
      continue;
    }

    auto symbol = dyn_cast<wave::WaveSymbolAttr>(symbols[i]);
    if (!symbol)
      return std::nullopt;

    std::optional<int64_t> value =
        hyperparams ? hyperparams.getSymbolValue(symbol.getName())
                    : std::nullopt;
    if (!value)
      return std::nullopt;
    symReplacements.push_back(getAffineConstantExpr(*value, map.getContext()));
  }

  SmallVector<int64_t> out;
  out.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults()) {
    AffineExpr sub = expr.replaceSymbols(symReplacements);
    sub = simplifyAffineExpr(sub, map.getNumDims(), map.getNumSymbols());
    if (auto c = llvm::dyn_cast<AffineConstantExpr>(sub)) {
      out.push_back(c.getValue());
      continue;
    }

    return std::nullopt;
  }
  return out;
}

LogicalResult wave::computeWavesPerBlockFromConstraints(
    const llvm::SmallDenseMap<wave::WaveSymbolAttr,
                              wave::WorkgroupConstraintAttr>
        &workgroupConstraints,
    const llvm::SmallDenseMap<wave::WaveSymbolAttr, wave::WaveConstraintAttr>
        &waveConstraints,
    wave::WaveHyperparameterAttr hyperparams,
    SmallVectorImpl<unsigned> &wavesPerBlock) {
  // Default to 1 wave per block for each dimension, this may be recomputed
  // later if the corresponding constraints are provided.
  wavesPerBlock.assign(/*NumElts=*/3, /*Elt=*/1);

  for (auto &&[symbol, waveConstraint] : waveConstraints) {
    auto wgIt = workgroupConstraints.find(symbol);
    if (wgIt == workgroupConstraints.end())
      return failure();

    wave::WorkgroupConstraintAttr wgConstraint = wgIt->second;

    std::optional<llvm::SmallVector<int64_t>> wgEvaluated =
        wave::evaluateMapWithHyperparams(
            wgConstraint.getTileSize().getMap(),
            wgConstraint.getTileSize().getSymbols(), hyperparams);
    if (!wgEvaluated || wgEvaluated->size() != 1)
      return failure();

    std::optional<llvm::SmallVector<int64_t>> waveEvaluated =
        wave::evaluateMapWithHyperparams(
            waveConstraint.getTileSize().getMap(),
            waveConstraint.getTileSize().getSymbols(), hyperparams);
    if (!waveEvaluated || waveEvaluated->size() != 1)
      return failure();

    int64_t workgroupSize = wgEvaluated->front();
    int64_t waveSize = waveEvaluated->front();

    if (waveSize <= 0 || workgroupSize % waveSize != 0)
      return failure();

    int64_t numWaves = workgroupSize / waveSize;
    unsigned wgDim =
        static_cast<unsigned>(wgConstraint.getWorkgroupDim().getValue());
    wavesPerBlock[wgDim] = static_cast<unsigned>(numWaves);
  }

  return success();
}

/// Dependency graph over hyperparameter symbols used for cycle detection via
/// scc_iterator.  A synthetic root node (null symbol) fans out to every
/// expr_list entry so that a single traversal covers all components.
struct HyperparamDepGraph {
  /// Adjacency list: symbol -> symbols it depends on.
  llvm::DenseMap<wave::WaveSymbolAttr, llvm::SmallVector<wave::WaveSymbolAttr>>
      deps;
  /// All expr_list symbols, also the edge list of the synthetic root.
  llvm::SmallVector<wave::WaveSymbolAttr> exprListKeys;

  /// A node carries a back-pointer to the graph so that child_begin/child_end
  /// can look up the adjacency list without external state.
  struct Node {
    const HyperparamDepGraph *graph;
    wave::WaveSymbolAttr sym;
    bool operator==(const Node &o) const {
      return graph == o.graph && sym == o.sym;
    }
    bool operator!=(const Node &o) const { return !(*this == o); }
  };

  Node root() const { return {this, {}}; }

  llvm::ArrayRef<wave::WaveSymbolAttr>
  children(wave::WaveSymbolAttr sym) const {
    if (!sym)
      return exprListKeys;
    auto it = deps.find(sym);
    if (it == deps.end())
      return {};
    return it->second;
  }
};

namespace llvm {

/// Adaptor that wraps a pointer into a WaveSymbolAttr adjacency list and
/// produces Node values carrying the graph back-pointer.
struct HyperparamChildIterator
    : iterator_adaptor_base<
          HyperparamChildIterator, const wave::WaveSymbolAttr *,
          std::random_access_iterator_tag, HyperparamDepGraph::Node,
          std::ptrdiff_t, const HyperparamDepGraph::Node *,
          HyperparamDepGraph::Node> {
  const HyperparamDepGraph *graph = nullptr;
  HyperparamChildIterator() = default;
  HyperparamChildIterator(const wave::WaveSymbolAttr *it,
                          const HyperparamDepGraph *g)
      : iterator_adaptor_base(it), graph(g) {}
  HyperparamDepGraph::Node operator*() const { return {graph, *I}; }
};

template <> struct DenseMapInfo<HyperparamDepGraph::Node> {
  static HyperparamDepGraph::Node getEmptyKey() {
    return {nullptr, DenseMapInfo<wave::WaveSymbolAttr>::getEmptyKey()};
  }
  static HyperparamDepGraph::Node getTombstoneKey() {
    return {nullptr, DenseMapInfo<wave::WaveSymbolAttr>::getTombstoneKey()};
  }
  static unsigned getHashValue(const HyperparamDepGraph::Node &n) {
    return DenseMapInfo<wave::WaveSymbolAttr>::getHashValue(n.sym);
  }
  static bool isEqual(const HyperparamDepGraph::Node &a,
                      const HyperparamDepGraph::Node &b) {
    return a.graph == b.graph && a.sym == b.sym;
  }
};

template <> struct GraphTraits<const HyperparamDepGraph *> {
  using NodeRef = HyperparamDepGraph::Node;
  using ChildIteratorType = HyperparamChildIterator;

  static NodeRef getEntryNode(const HyperparamDepGraph *g) { return g->root(); }
  static ChildIteratorType child_begin(NodeRef node) {
    return {node.graph->children(node.sym).begin(), node.graph};
  }
  static ChildIteratorType child_end(NodeRef node) {
    return {node.graph->children(node.sym).end(), node.graph};
  }
};

} // namespace llvm

LogicalResult wave::verifyHyperparameterAcyclicity(
    wave::WaveHyperparameterAttr hyperparams, MLIRContext *ctx,
    llvm::function_ref<InFlightDiagnostic()> emitError) {
  HyperparamDepGraph graph;
  for (const NamedAttribute &entry : hyperparams.getMapping()) {
    auto exprList = llvm::dyn_cast<wave::WaveExprListAttr>(entry.getValue());
    if (!exprList)
      continue;
    wave::WaveSymbolAttr key =
        wave::WaveSymbolAttr::get(ctx, entry.getName().getValue());
    graph.exprListKeys.push_back(key);
    auto &edges = graph.deps[key];
    for (Attribute symAttr : exprList.getSymbols())
      edges.push_back(llvm::cast<wave::WaveSymbolAttr>(symAttr));
  }

  const HyperparamDepGraph *graphPtr = &graph;
  for (auto scci = llvm::scc_begin(graphPtr); !scci.isAtEnd(); ++scci) {
    if (!scci.hasCycle())
      continue;
    const auto &scc = *scci;
    llvm::SmallVector<StringRef> names;
    for (const HyperparamDepGraph::Node &n : scc)
      if (n.sym)
        names.push_back(n.sym.getName());
    return emitError() << "hyperparameter dependency cycle: "
                       << llvm::join(names, ", ");
  }
  return llvm::success();
}

LogicalResult
wave::collectWaveConstraints(Operation *top,
                             DenseMap<Operation *, Attribute> &constraints) {
  auto *waveDialect = top->getContext()->getLoadedDialect<wave::WaveDialect>();
  auto walkResult = top->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto attr = op->getAttrOfType<ArrayAttr>(
            wave::WaveDialect::kWaveConstraintsAttrName)) {
      constraints[op] = attr;
      return WalkResult::skip();
    }
    if (op->getDialect() == waveDialect) {
      op->emitError()
          << "wave dialect operation without constraints on an ancestor";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();
  return success();
}
