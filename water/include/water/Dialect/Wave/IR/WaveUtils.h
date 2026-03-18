// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_UTILS_H

#define WATER_DIALECT_WAVE_UTILS_H

#include "water/Dialect/Wave/IR/WaveAttrs.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"

namespace wave {

/// Dependency graph over hyperparameter symbols used for cycle detection via
/// scc_iterator.  A synthetic root node (null symbol) fans out to every
/// expr_list entry so that a single traversal covers all components.
struct HyperparamDepGraph {
  /// Adjacency list: symbol -> symbols it depends on.
  llvm::DenseMap<WaveSymbolAttr, llvm::SmallVector<WaveSymbolAttr>> deps;
  /// All expr_list symbols, also the edge list of the synthetic root.
  llvm::SmallVector<WaveSymbolAttr> exprListKeys;

  /// A node carries a back-pointer to the graph so that child_begin/child_end
  /// can look up the adjacency list without external state.
  struct Node {
    const HyperparamDepGraph *graph;
    WaveSymbolAttr sym;
    bool operator==(const Node &o) const {
      return graph == o.graph && sym == o.sym;
    }
    bool operator!=(const Node &o) const { return !(*this == o); }
  };

  Node root() const { return {this, {}}; }

  llvm::ArrayRef<WaveSymbolAttr> children(WaveSymbolAttr sym) const {
    if (!sym)
      return exprListKeys;
    auto it = deps.find(sym);
    if (it == deps.end())
      return {};
    return it->second;
  }
};

} // namespace wave

namespace llvm {

/// Adaptor that wraps a pointer into a WaveSymbolAttr adjacency list and
/// produces Node values carrying the graph back-pointer.
struct HyperparamChildIterator
    : iterator_adaptor_base<
          HyperparamChildIterator, const wave::WaveSymbolAttr *,
          std::random_access_iterator_tag, wave::HyperparamDepGraph::Node,
          std::ptrdiff_t, const wave::HyperparamDepGraph::Node *,
          wave::HyperparamDepGraph::Node> {
  const wave::HyperparamDepGraph *graph = nullptr;
  HyperparamChildIterator() = default;
  HyperparamChildIterator(const wave::WaveSymbolAttr *it,
                          const wave::HyperparamDepGraph *g)
      : iterator_adaptor_base(it), graph(g) {}
  wave::HyperparamDepGraph::Node operator*() const { return {graph, *I}; }
};

template <> struct DenseMapInfo<wave::HyperparamDepGraph::Node> {
  static wave::HyperparamDepGraph::Node getEmptyKey() {
    return {nullptr, DenseMapInfo<wave::WaveSymbolAttr>::getEmptyKey()};
  }
  static wave::HyperparamDepGraph::Node getTombstoneKey() {
    return {nullptr, DenseMapInfo<wave::WaveSymbolAttr>::getTombstoneKey()};
  }
  static unsigned getHashValue(const wave::HyperparamDepGraph::Node &n) {
    return DenseMapInfo<wave::WaveSymbolAttr>::getHashValue(n.sym);
  }
  static bool isEqual(const wave::HyperparamDepGraph::Node &a,
                      const wave::HyperparamDepGraph::Node &b) {
    return a.graph == b.graph && a.sym == b.sym;
  }
};

template <> struct GraphTraits<const wave::HyperparamDepGraph *> {
  using NodeRef = wave::HyperparamDepGraph::Node;
  using ChildIteratorType = HyperparamChildIterator;

  static NodeRef getEntryNode(const wave::HyperparamDepGraph *g) {
    return g->root();
  }
  static ChildIteratorType child_begin(NodeRef node) {
    return {node.graph->children(node.sym).begin(), node.graph};
  }
  static ChildIteratorType child_end(NodeRef node) {
    return {node.graph->children(node.sym).end(), node.graph};
  }
};
} // namespace llvm

namespace wave {
/// Return the position of the dimension that is vectorized based on the index
/// sequence. The dimension with the largest step is considered to be
/// vectorized. In case of a tie, take the dimension that is farther in the
/// index dictionary, which is secretly a list. Return failure when the index
/// sequence step cannot be evaluated statically.
std::optional<int64_t>
getPositionOfVectorizedDim(llvm::ArrayRef<wave::WaveSymbolAttr> shape,
                           mlir::DictionaryAttr indexDict,
                           wave::WaveHyperparameterAttr hyper);

// Return the vector shape implied by the index sequence and hyperparameteters,
// i.e., the step expression of the index sequence evaluated using the
// hyperparameter values. The step may be indicated as ShapedType::kDynamic if
// it cannot be fully evaluated.
llvm::SmallVector<int64_t>
getUncollapsedVectorShape(llvm::ArrayRef<wave::WaveSymbolAttr> shape,
                          mlir::DictionaryAttr indexDict,
                          wave::WaveHyperparameterAttr hyper);

/// Resolve named Wave symbols to concrete integer values using the
/// hyperparameter table.
std::optional<llvm::SmallVector<int64_t>>
resolveSymbolNames(llvm::ArrayRef<mlir::Attribute> symbols,
                   wave::WaveHyperparameterAttr hyper);

/// Substitute named symbol values used in the affine map by the constant values
/// defined in the hyperparameter list then evaluate the expressions to get
/// concrete integer results. Return nullopt if the substitution doesn't yield
/// constant results, in particular, if some symbols are not defined.
std::optional<llvm::SmallVector<int64_t>>
evaluateMapWithHyperparams(mlir::AffineMap map,
                           llvm::ArrayRef<mlir::Attribute> symbols,
                           wave::WaveHyperparameterAttr hyperparams);

/// Compute waves per block from wave constraints and workgroup constraints.
/// Returns failure if the computation fails.
llvm::LogicalResult computeWavesPerBlockFromConstraints(
    const llvm::SmallDenseMap<wave::WaveSymbolAttr,
                              wave::WorkgroupConstraintAttr>
        &workgroupConstraints,
    const llvm::SmallDenseMap<wave::WaveSymbolAttr, wave::WaveConstraintAttr>
        &waveConstraints,
    wave::WaveHyperparameterAttr hyperparams,
    llvm::SmallVectorImpl<unsigned> &wavesPerBlock);

/// Permute the shape according to the mapping.
void permuteShape(llvm::ArrayRef<wave::WaveSymbolAttr> shape,
                  mlir::AffineMap map, bool inverse,
                  llvm::SmallVectorImpl<wave::WaveSymbolAttr> &permutedShape);

/// Verify that derived hyperparameter symbols (expr_list values) form a DAG.
/// Returns success if there are no cycles, or emits a diagnostic naming the
/// cycle participants.
llvm::LogicalResult verifyHyperparameterAcyclicity(
    wave::WaveHyperparameterAttr hyperparams, mlir::MLIRContext *ctx,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError);

} // namespace wave

namespace llvm {
// Combine two potentially failing ChangeResults: if any of them failed, the
// result of the combination is also failure.
llvm::FailureOr<mlir::ChangeResult> static inline
operator|(llvm::FailureOr<mlir::ChangeResult> lhs,
          FailureOr<mlir::ChangeResult> rhs) {
  if (llvm::failed(lhs) || llvm::failed(rhs))
    return llvm::failure();
  return *lhs | *rhs;
}
} // namespace llvm

#endif // WATER_DIALECT_WAVE_IR_WAVEUTILS_H
