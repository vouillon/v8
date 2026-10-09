// Copyright 2026 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "src/compiler/turboshaft/wasm-bounds-check-elimination-reducer.h"

#include "src/compiler/turboshaft/assembler.h"
#include "src/compiler/turboshaft/copying-phase.h"
#include "src/compiler/turboshaft/operations.h"
#include "src/compiler/turboshaft/representations.h"
#include "src/compiler/turboshaft/wasm-load-elimination-reducer.h"
#include "src/flags/flags.h"
#include "test/common/flag-utils.h"
#include "test/unittests/compiler/turboshaft/reducer-test.h"

namespace v8::internal::compiler::turboshaft {

#include "src/compiler/turboshaft/define-assembler-macros.inc"

class WasmBoundsCheckEliminationReducerTest : public ReducerTest {
 public:
  WasmBoundsCheckEliminationReducerTest()
      : flag_bounds_check_elimination_(
            &v8_flags.turboshaft_wasm_bounds_check_elimination, true),
        flag_load_elimination_(&v8_flags.turboshaft_wasm_load_elimination,
                               true) {}

  // The graph built by {builder} gets two arrays {a} and {b}, an index
  // {i} and a condition {c} as parameters.
  template <typename Builder>
  TestInstance CreateTest(const Builder& builder) {
    std::initializer_list<RegisterRepresentation> parameter_reps{
        RegisterRepresentation::Tagged(), RegisterRepresentation::Tagged(),
        RegisterRepresentation::Word32(), RegisterRepresentation::Word32()};
    return CreateFromGraph(
        base::VectorOf(parameter_reps),
        [&builder](auto& Asm) {
          builder(Asm, V<WasmArrayNullable>::Cast(Asm.GetParameter(0)),
                  V<WasmArrayNullable>::Cast(Asm.GetParameter(1)),
                  Asm.template GetParameter<Word32>(2),
                  Asm.template GetParameter<Word32>(3));
          __ Return(__ Word32Constant(0));
        },
        true);
  }

  // Like {CreateTest}, with a fifth parameter {s}, a struct whose only
  // field is an array, for tests that also run load elimination.
  template <typename Builder>
  TestInstance CreateTestWithStruct(const Builder& builder) {
    module_ = std::make_unique<wasm::WasmModule>();
    wasm::ArrayType* array_type =
        zone()->New<wasm::ArrayType>(wasm::kWasmI32, true);
    module_->AddArrayTypeForTesting(array_type, wasm::kNoSuperType, true,
                                    SharedFlag{false});
    wasm::ValueType array_ref = wasm::ValueType::RefNull(
        {0}, SharedFlag{false}, wasm::RefTypeKind::kArray);
    wasm::StructType::Builder<Zone> struct_builder(zone(), 1, false,
                                                   SharedFlag{false});
    struct_builder.AddField(array_ref, true);
    struct_type_ = struct_builder.Build();
    module_->AddStructTypeForTesting(struct_type_, wasm::kNoSuperType, true,
                                     SharedFlag{false});
    wasm::ValueType struct_ref = wasm::ValueType::RefNull(
        {1}, SharedFlag{false}, wasm::RefTypeKind::kStruct);
    std::initializer_list<RegisterRepresentation> parameter_reps{
        RegisterRepresentation::Tagged(), RegisterRepresentation::Tagged(),
        RegisterRepresentation::Word32(), RegisterRepresentation::Word32(),
        RegisterRepresentation::Tagged()};
    return CreateFromGraph(
        base::VectorOf(parameter_reps),
        [&builder](auto& Asm) {
          builder(Asm, V<WasmArrayNullable>::Cast(Asm.GetParameter(0)),
                  V<WasmArrayNullable>::Cast(Asm.GetParameter(1)),
                  Asm.template GetParameter<Word32>(2),
                  Asm.template GetParameter<Word32>(3),
                  V<WasmStructNullable>::Cast(Asm.GetParameter(4)));
          __ Return(__ Word32Constant(0));
        },
        module_.get(),
        wasm::FunctionSig::Build(zone(), {wasm::kWasmI32},
                                 {array_ref, array_ref, wasm::kWasmI32,
                                  wasm::kWasmI32, struct_ref}));
  }

  // The array field of {s} (see {CreateTestWithStruct}).
  template <typename Asm_t>
  V<WasmArrayNullable> ArrayField(Asm_t& Asm, V<WasmStructNullable> s) {
    return V<WasmArrayNullable>::Cast(__ StructGet(
        s, struct_type_, {1}, 0, true, CheckForNull::kWithNullCheck, {}));
  }

  // Emits the bounds check for {array[index + offset]} the way the
  // graph builder does. No index means a constant index.
  template <typename Asm_t>
  static void BoundsCheck(Asm_t& Asm, V<WasmArrayNullable> array,
                          OptionalV<Word32> index, int32_t offset) {
    V<Word32> length = __ ArrayLength(array, compiler::kWithNullCheck);
    V<Word32> full_index = __ Word32Constant(offset);
    if (index.has_value()) {
      full_index =
          offset == 0 ? index.value() : __ Word32Add(index.value(), full_index);
    }
    __ TrapIfNot(__ Uint32LessThan(full_index, length),
                 TrapId::kTrapArrayOutOfBounds);
  }

  // Emits a branch on {condition}, and the code generated by {body} in
  // the true branch.
  template <typename Asm_t, typename Body>
  static void If(Asm_t& Asm, V<Word32> condition, const Body& body) {
    Block* if_true = __ NewBlock();
    Block* if_false = __ NewBlock();
    Block* merge = __ NewBlock();
    __ Branch(condition, if_true, if_false);
    __ Bind(if_true);
    body();
    __ Goto(merge);
    __ Bind(if_false);
    __ Goto(merge);
    __ Bind(merge);
  }

  // Emits a branch to an unreachable block when {condition} holds, and
  // continues otherwise. This is how wasm_of_ocaml raises an exception
  // for an OCaml bounds check.
  template <typename Asm_t>
  static void FailIf(Asm_t& Asm, V<Word32> condition) {
    Block* failure = __ NewBlock();
    Block* success = __ NewBlock();
    __ Branch(condition, failure, success);
    __ Bind(failure);
    __ Unreachable();
    __ Bind(success);
  }

  // {a.length - reduction}.
  template <typename Asm_t>
  static V<Word32> ReducedLength(Asm_t& Asm, V<WasmArrayNullable> array,
                                 int32_t reduction) {
    return __ Word32Sub(__ ArrayLength(array, compiler::kWithNullCheck),
                        __ Word32Constant(reduction));
  }

  static void Run(TestInstance& test) {
    test.Run<WasmBoundsCheckEliminationReducer>();
  }

  // Runs the pass with load elimination, as in the WasmGCOptimize phase.
  static void RunWithLoadElimination(TestInstance& test) {
    test.Run<WasmBoundsCheckEliminationReducer, WasmLoadEliminationReducer>();
  }

 private:
  const FlagScope<bool> flag_bounds_check_elimination_;
  const FlagScope<bool> flag_load_elimination_;
  std::unique_ptr<wasm::WasmModule> module_;
  wasm::StructType* struct_type_ = nullptr;
};

TEST_F(WasmBoundsCheckEliminationReducerTest, RedundantConstantIndex) {
  // a[1]; a[0];
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, {}, 1);
    BoundsCheck(Asm, a, {}, 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, RedundantBetweenChecks) {
  // a[i]; a[i+2]; a[i+1];
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, i, 0);
    BoundsCheck(Asm, a, i, 2);
    BoundsCheck(Asm, a, i, 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       NonNegativeIndexMakesCheckRedundant) {
  // a[i]; b[i+1]; b[i];
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, i, 0);
    BoundsCheck(Asm, b, i, 1);
    BoundsCheck(Asm, b, i, 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NonNegativeIndexLearntLater) {
  // b[i+5]; a[i]; b[i+1];
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, b, i, 5);
    BoundsCheck(Asm, a, i, 0);
    BoundsCheck(Asm, b, i, 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       NonNegativeIndexDoesNotCoverSmallerOffsets) {
  // a[i+2]; b[i+1]; b[i];
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, i, 2);
    BoundsCheck(Asm, b, i, 1);
    BoundsCheck(Asm, b, i, 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 3u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       NonNegativeIndexFromUnsignedComparison) {
  // if (i < 100) { a[i+1]; a[i]; }
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    If(Asm, __ Uint32LessThan(i, __ Word32Constant(100)), [&]() {
      BoundsCheck(Asm, a, i, 1);
      BoundsCheck(Asm, a, i, 0);
    });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       NoNonNegativeIndexFromLargeBound) {
  // if (i < 0x80000001) { a[i+1]; a[i]; }
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    If(Asm, __ Uint32LessThan(i, __ Word32Constant(0x80000001)), [&]() {
      BoundsCheck(Asm, a, i, 1);
      BoundsCheck(Asm, a, i, 0);
    });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NoFactsFromNonDominatingBranch) {
  // if (c) { a[i]; } a[i];
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    If(Asm, c, [&]() { BoundsCheck(Asm, a, i, 0); });
    BoundsCheck(Asm, a, i, 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, FactsFromDominatingBranch) {
  // if (i < a.length) { a[i]; }
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> length = __ ArrayLength(a, compiler::kWithNullCheck);
    If(Asm, __ Uint32LessThan(i, length), [&]() { BoundsCheck(Asm, a, i, 0); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, OCamlBoundsCheck) {
  // a[0]; if (a.length - 1 <= i) fail; a[i+1];
  // The first access shows that the length is at least 1, so the OCaml
  // bounds check shows that a[i+1] is within bounds.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, {}, 0);
    FailIf(Asm, __ Uint32LessThanOrEqual(ReducedLength(Asm, a, 1), i));
    BoundsCheck(Asm, a, i, 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, OCamlBoundsCheckAsAddition) {
  // a[0]; if (a.length + (-1) <= i) fail; a[i+1];
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, {}, 0);
    V<Word32> length = __ ArrayLength(a, compiler::kWithNullCheck);
    FailIf(Asm, __ Uint32LessThanOrEqual(
                    __ Word32Add(length, __ Word32Constant(-1)), i));
    BoundsCheck(Asm, a, i, 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, OCamlBoundsCheckUnknownLength) {
  // if (a.length - 1 <= i) fail; a[i+1];
  // The array might be empty, in which case a.length - 1 wraps around
  // and the OCaml bounds check does not show anything.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    FailIf(Asm, __ Uint32LessThanOrEqual(ReducedLength(Asm, a, 1), i));
    BoundsCheck(Asm, a, i, 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, OCamlBoundsChecksInSequence) {
  // a[i+1] after the OCaml bounds check of a.(i) shows that the length is
  // at least 1, so the Wasm bounds checks of a.(j) and a.(k) are
  // redundant.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    FailIf(Asm, __ Uint32LessThanOrEqual(ReducedLength(Asm, a, 1), i));
    BoundsCheck(Asm, a, i, 1);
    FailIf(Asm, __ Uint32LessThanOrEqual(ReducedLength(Asm, a, 1), c));
    BoundsCheck(Asm, a, c, 1);
    V<Word32> k = __ Word32Add(i, c);
    FailIf(Asm, __ Uint32LessThanOrEqual(ReducedLength(Asm, a, 1), k));
    BoundsCheck(Asm, a, k, 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, ReducedLengthCoversRange) {
  // a[1]; if (i < a.length - 2) { a[i]; a[i+1]; a[i+2]; }
  // a[1] shows that the length is at least 2.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, {}, 1);
    If(Asm, __ Uint32LessThan(i, ReducedLength(Asm, a, 2)), [&]() {
      BoundsCheck(Asm, a, i, 0);
      BoundsCheck(Asm, a, i, 1);
      BoundsCheck(Asm, a, i, 2);
    });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, ReducedLengthTooLarge) {
  // a[0]; if (i < a.length - 2) { a[i+2]; }
  // a[0] only shows that the length is at least 1, so a.length - 2 may
  // wrap around.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, {}, 0);
    If(Asm, __ Uint32LessThan(i, ReducedLength(Asm, a, 2)),
       [&]() { BoundsCheck(Asm, a, i, 2); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, MinLengthCoversConstantIndices) {
  // if (2 < a.length) { a[2]; a[0]; a[1]; }
  // The length of a is at least 3.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> length = __ ArrayLength(a, compiler::kWithNullCheck);
    If(Asm, __ Uint32LessThan(__ Word32Constant(2), length), [&] {
      BoundsCheck(Asm, a, {}, 2);
      BoundsCheck(Asm, a, {}, 0);
      BoundsCheck(Asm, a, {}, 1);
    });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       SameArrayThroughNonNullAssertion) {
  // a[i]; (a!)[i];
  // The lengths of a and of its non-null assertion are the same.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, i, 0);
    V<WasmArrayNullable> non_null = V<WasmArrayNullable>::Cast(
        __ AssertNotNull(a, wasm::ValueType::RefNull(wasm::kWasmArrayRef),
                         TrapId::kTrapNullDereference));
    BoundsCheck(Asm, non_null, i, 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, SameArrayThroughTypeAnnotation) {
  // a[i]; (a annotated with a type)[i];
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, i, 0);
    V<WasmArrayNullable> annotated =
        __ AnnotateWasmType(a, wasm::ValueType::RefNull(wasm::kWasmArrayRef));
    BoundsCheck(Asm, annotated, i, 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, SameArrayThroughLoadElimination) {
  // x = s.f; x[i]; y = s.f; y[i];
  // Load elimination replaces y by x.
  auto test = CreateTestWithStruct(
      [this](auto& Asm, auto a, auto b, auto i, auto c, auto s) {
        BoundsCheck(Asm, ArrayField(Asm, s), i, 0);
        BoundsCheck(Asm, ArrayField(Asm, s), i, 0);
      });
  RunWithLoadElimination(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       NoNonNegativeIndexFromWrappedLength) {
  // if (i <u a.length - 1) { b[i+2]; b[i]; }
  // The length of a may be 0, and a.length - 1 then wraps around, so the
  // condition does not show that i is non-negative: with i = -2, b[i+2]
  // may be within bounds, but not b[i].
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    If(Asm, __ Uint32LessThan(i, ReducedLength(Asm, a, 1)), [&] {
      BoundsCheck(Asm, b, i, 2);
      BoundsCheck(Asm, b, i, 0);
    });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NestedConstantAdditions) {
  // a[(i+1)+1]; a[i+2]; a[(i+3)-1]; as produced for instance by loop
  // unrolling: all three indices are i+2.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> j = __ Word32Add(i, __ Word32Constant(1));
    BoundsCheck(Asm, a, j, 1);
    BoundsCheck(Asm, a, i, 2);
    V<Word32> k = __ Word32Sub(__ Word32Add(i, __ Word32Constant(3)),
                               __ Word32Constant(1));
    BoundsCheck(Asm, a, k, 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

// The sign extension of a 31-bit integer, as done by wasm_of_ocaml.
template <typename Asm_t>
static V<Word32> SignExtend31(Asm_t& Asm, V<Word32> value, int shift = 1) {
  return __ Word32ShiftRightArithmetic(
      __ Word32ShiftLeft(value, __ Word32Constant(shift)),
      __ Word32Constant(shift));
}

TEST_F(WasmBoundsCheckEliminationReducerTest, DuplicatedComputations) {
  // a[0]; if (a.length - 1 <= sext(i - c)) fail; a[sext(i - c) + 1];
  // where {sext(i - c)} is computed twice. Both computations have the
  // same value, so the OCaml bounds check covers the Wasm one.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, {}, 0);
    FailIf(Asm,
           __ Uint32LessThanOrEqual(ReducedLength(Asm, a, 1),
                                    SignExtend31(Asm, __ Word32Sub(i, c))));
    BoundsCheck(Asm, a, SignExtend31(Asm, __ Word32Sub(i, c)), 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, CommutedComputations) {
  // a[0]; if (a.length - 1 <= i * c) fail; a[c * i + 1];
  // Both products have the same value.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, {}, 0);
    FailIf(Asm, __ Uint32LessThanOrEqual(ReducedLength(Asm, a, 1),
                                         __ Word32Mul(i, c)));
    BoundsCheck(Asm, a, __ Word32Mul(c, i), 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, DifferentComputations) {
  // a[0]; if (a.length - 1 <= sext(i)) fail; a[sext2(i) + 1];
  // where {sext2} shifts by 2 instead of 1: the values differ.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, {}, 0);
    FailIf(Asm, __ Uint32LessThanOrEqual(ReducedLength(Asm, a, 1),
                                         SignExtend31(Asm, i)));
    BoundsCheck(Asm, a, SignExtend31(Asm, i, 2), 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

// {trunc(x + offset)}, for a 64-bit {x}, as computed by languages with
// 64-bit integers for the index of {a[x + offset]}.
template <typename Asm_t>
static V<Word32> TruncatedIndex(Asm_t& Asm, V<Word64> x, int64_t offset) {
  return __ TruncateWord64ToWord32(
      offset == 0 ? x : __ Word64Add(x, __ Word64Constant(offset)));
}

TEST_F(WasmBoundsCheckEliminationReducerTest, TruncatedAdditions) {
  // a[trunc(x)]; a[trunc(x + 2)]; a[trunc(x + 1)];
  // As for a[i]; a[i+2]; a[i+1]: the last check is redundant.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word64> x = __ ChangeUint32ToUint64(i);
    BoundsCheck(Asm, a, TruncatedIndex(Asm, x, 0), 0);
    BoundsCheck(Asm, a, TruncatedIndex(Asm, x, 2), 0);
    BoundsCheck(Asm, a, TruncatedIndex(Asm, x, 1), 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, TruncatedSubtractions) {
  // a[trunc(x)]; a[trunc(x + 2)]; a[trunc(x - (-1))];
  // The last index is trunc(x) + 1, between the first two.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word64> x = __ ChangeUint32ToUint64(i);
    BoundsCheck(Asm, a, TruncatedIndex(Asm, x, 0), 0);
    BoundsCheck(Asm, a, TruncatedIndex(Asm, x, 2), 0);
    BoundsCheck(Asm, a,
                __ TruncateWord64ToWord32(
                    __ Word64Sub(x, __ Word64Constant(int64_t{-1}))),
                0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, TruncatedAdditionsNoBase) {
  // a[trunc(x + 2)]; a[trunc(x + 1)];
  // No operation computes trunc(x), so the indices are not decomposed,
  // and the second check stays.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word64> x = __ ChangeUint32ToUint64(i);
    BoundsCheck(Asm, a, TruncatedIndex(Asm, x, 2), 0);
    BoundsCheck(Asm, a, TruncatedIndex(Asm, x, 1), 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

// {i31.get_s(x)}, as compiled by the graph builder: a bitcast of the
// reference, and a shift.
template <typename Asm_t>
static V<Word32> I31GetS(Asm_t& Asm, V<Object> x) {
  return __ Word32ShiftRightArithmeticShiftOutZeros(
      __ TruncateWordPtrToWord32(__ BitcastTaggedToWordPtr(x)),
      kSmiTagSize + kSmiShiftSize);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, SmiBitcastsShared) {
  // x = b!, for b of type (ref null i31); a[0];
  // if (a.length - 1 <= i31.get_s(x)) fail; a[i31.get_s(x) + 1];
  // A non-null i31 reference is a Smi, so the two bitcasts of x, and
  // the two shifts, have the same value.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Object> x =
        __ AssertNotNull(b, wasm::ValueType::RefNull(wasm::kWasmI31Ref),
                         TrapId::kTrapNullDereference);
    BoundsCheck(Asm, a, {}, 0);
    FailIf(Asm,
           __ Uint32LessThanOrEqual(ReducedLength(Asm, a, 1), I31GetS(Asm, x)));
    BoundsCheck(Asm, a, I31GetS(Asm, x), 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       SmiBitcastsSharedThroughAnnotation) {
  // x = b annotated with type (ref i31); a[0];
  // if (a.length - 1 <= i31.get_s(x)) fail; a[i31.get_s(x) + 1];
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Object> object = b;
    V<Object> x =
        __ AnnotateWasmType(object, wasm::ValueType::Ref(wasm::kWasmI31Ref));
    BoundsCheck(Asm, a, {}, 0);
    FailIf(Asm,
           __ Uint32LessThanOrEqual(ReducedLength(Asm, a, 1), I31GetS(Asm, x)));
    BoundsCheck(Asm, a, I31GetS(Asm, x), 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NullableI31BitcastsNotShared) {
  // x = b annotated with type (ref null i31); a[0];
  // if (a.length - 1 <= i31.get_s(x)) fail; a[i31.get_s(x) + 1];
  // x may be null, a heap object.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Object> object = b;
    V<Object> x = __ AnnotateWasmType(
        object, wasm::ValueType::RefNull(wasm::kWasmI31Ref));
    BoundsCheck(Asm, a, {}, 0);
    FailIf(Asm,
           __ Uint32LessThanOrEqual(ReducedLength(Asm, a, 1), I31GetS(Asm, x)));
    BoundsCheck(Asm, a, I31GetS(Asm, x), 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, DifferentShiftsNotShared) {
  // a[i >>> c]; a[i >> c];
  // The shifts have the same inputs, but are of different kinds.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, __ Word32ShiftRightLogical(i, c), 0);
    BoundsCheck(Asm, a, __ Word32ShiftRightArithmetic(i, c), 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, SubtractionsNotCommuted) {
  // a[i - c]; a[c - i];
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, __ Word32Sub(i, c), 0);
    BoundsCheck(Asm, a, __ Word32Sub(c, i), 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

static const wasm::ArrayType kI32ArrayType(wasm::kWasmI32, true);

TEST_F(WasmBoundsCheckEliminationReducerTest,
       AllocatedWithLengthOfAnotherArray) {
  // a[i]; x = new T[a.length]; x[i];
  // Load elimination replaces x.length by a.length, so that the lengths
  // of a and x have the same canonical value.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, i, 0);
    V<WasmArrayNullable> x = __ WasmAllocateArray(
        V<Map>::Cast(b), __ ArrayLength(a, compiler::kWithNullCheck),
        &kI32ArrayType, SharedFlag{false});
    BoundsCheck(Asm, x, i, 0);
  });
  RunWithLoadElimination(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, HeapObjectBitcastsNotShared) {
  // a[0]; if (a.length - 1 <= i31.get_s(b)) fail; a[i31.get_s(b) + 1];
  // b is not known to be a Smi: its address may change between the two
  // bitcasts.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, {}, 0);
    FailIf(Asm,
           __ Uint32LessThanOrEqual(ReducedLength(Asm, a, 1), I31GetS(Asm, b)));
    BoundsCheck(Asm, a, I31GetS(Asm, b), 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

#include "src/compiler/turboshaft/undef-assembler-macros.inc"

}  // namespace v8::internal::compiler::turboshaft
