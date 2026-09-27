// Copyright 2026 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "src/compiler/turboshaft/duplication-optimization-reducer.h"

#include <array>

#include "test/unittests/compiler/turboshaft/reducer-test.h"

namespace v8::internal::compiler::turboshaft {

#include "src/compiler/turboshaft/define-assembler-macros.inc"

class DuplicationOptimizationReducerTest : public ReducerTest {};

namespace {

// Returns the branch on the comparison (the other branch is on a parameter).
const BranchOp& GetBranchOnComparison(TestInstance& test) {
  const BranchOp* result = nullptr;
  for (const Operation& op : test.graph().AllOperations()) {
    const BranchOp* branch = op.TryCast<BranchOp>();
    if (branch == nullptr ||
        test.graph().Get(branch->condition()).Is<ParameterOp>()) {
      continue;
    }
    CHECK_NULL(result);
    result = branch;
  }
  CHECK_NOT_NULL(result);
  return *result;
}

constexpr std::array kParameterReps = {RegisterRepresentation::Word32(),
                                       RegisterRepresentation::Word32(),
                                       RegisterRepresentation::Word32()};

}  // namespace

// A comparison whose only use is a branch in a later block is re-emitted right
// before that branch, so that the InstructionSelector can combine them.
TEST_F(DuplicationOptimizationReducerTest, SinkComparisonIntoBranchBlock) {
  auto test = CreateFromGraph(base::VectorOf(kParameterReps), [](auto& Asm) {
    V<Word32> a = Asm.template GetParameter<Word32>(0);
    V<Word32> b = Asm.template GetParameter<Word32>(1);
    V<Word32> c = Asm.template GetParameter<Word32>(2);
    V<Word32> comparison = __ Word32Equal(a, b);
    // An unrelated branch, so that the branch on {comparison} is in
    // another block.
    IF (c) {
      __ Return(__ Word32Constant(2));
    }
    IF (comparison) {
      __ Return(__ Word32Constant(1));
    } ELSE {
      __ Return(__ Word32Constant(0));
    }
  });

  test.Run<DuplicationOptimizationReducer>();

  const BranchOp& branch = GetBranchOnComparison(test);
  const Graph& graph = test.graph();
  const ComparisonOp* condition =
      graph.Get(branch.condition()).TryCast<ComparisonOp>();
  ASSERT_NE(condition, nullptr);
  EXPECT_EQ(ComparisonOp::Kind::kEqual, condition->kind);
  EXPECT_EQ(graph.BlockOf(branch.condition()),
            graph.BlockOf(graph.Index(branch)));
}

// A comparison that is already in the block of the branch using it is left
// alone.
TEST_F(DuplicationOptimizationReducerTest, KeepComparisonInBranchBlock) {
  auto test = CreateFromGraph(base::VectorOf(kParameterReps), [](auto& Asm) {
    V<Word32> a = Asm.template GetParameter<Word32>(0);
    V<Word32> b = Asm.template GetParameter<Word32>(1);
    V<Word32> c = Asm.template GetParameter<Word32>(2);
    IF (c) {
      __ Return(__ Word32Constant(2));
    }
    IF (__ Word32Equal(a, b)) {
      __ Return(__ Word32Constant(1));
    } ELSE {
      __ Return(__ Word32Constant(0));
    }
  });

  test.Run<DuplicationOptimizationReducer>();

  EXPECT_EQ(1u, test.CountOp(Opcode::kComparison));
  const BranchOp& branch = GetBranchOnComparison(test);
  const Graph& graph = test.graph();
  EXPECT_EQ(graph.BlockOf(branch.condition()),
            graph.BlockOf(graph.Index(branch)));
}

#include "src/compiler/turboshaft/undef-assembler-macros.inc"

}  // namespace v8::internal::compiler::turboshaft
