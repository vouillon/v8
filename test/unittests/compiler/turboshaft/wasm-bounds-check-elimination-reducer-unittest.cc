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

  // The graph built by {builder} gets two arrays {a} and {b}, and two
  // integers {i} and {c} (an index, or a condition) as parameters.
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
  // at least 1, so the Wasm bounds checks of a.(c) and a.(i + c) are
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

TEST_F(WasmBoundsCheckEliminationReducerTest, NonStrictConstantBound) {
  // if (i <=u 0x7fffffff) { a[i+1]; a[i]; }
  // i is non-negative, so the first check covers the second one.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    If(Asm, __ Uint32LessThanOrEqual(i, __ Word32Constant(0x7fffffff)), [&] {
      BoundsCheck(Asm, a, i, 1);
      BoundsCheck(Asm, a, i, 0);
    });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NonStrictConstantBoundTooLarge) {
  // if (i <=u 0x80000000) { a[i+1]; a[i]; }
  // i may be 0x80000000, which is negative.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    If(Asm, __ Uint32LessThanOrEqual(i, __ Word32Constant(0x80000000)), [&] {
      BoundsCheck(Asm, a, i, 1);
      BoundsCheck(Asm, a, i, 0);
    });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, KnownOffsetsAreExtended) {
  // a[i+2]; a[i]; a[i+1];
  // The range [0, 2] of offsets of i within bounds covers the last check.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, i, 2);
    BoundsCheck(Asm, a, i, 0);
    BoundsCheck(Asm, a, i, 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       MinLengthFromIndexCoversConstantIndex) {
  // a[i]; a[0];
  // The first check shows that the length is at least 1.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, i, 0);
    BoundsCheck(Asm, a, {}, 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
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

static constexpr wasm::ArrayType kI32ArrayType(wasm::kWasmI32, true);

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

TEST_F(WasmBoundsCheckEliminationReducerTest, SmiBitcastsSharedThroughCast) {
  // x = (b cast to (ref null i31))!, the assertion being typed anyref;
  // a[0]; if (a.length - 1 <= i31.get_s(x)) fail; a[i31.get_s(x) + 1];
  // The cast allows null, but the non-null assertion does not.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Object> object = b;
    V<Object> cast =
        __ WasmTypeCast(object, OptionalV<Map>::Nullopt(),
                        {wasm::ValueType::RefNull(wasm::kWasmAnyRef),
                         wasm::ValueType::RefNull(wasm::kWasmI31Ref)});
    V<Object> x =
        __ AssertNotNull(cast, wasm::ValueType::RefNull(wasm::kWasmAnyRef),
                         TrapId::kTrapNullDereference);
    BoundsCheck(Asm, a, {}, 0);
    FailIf(Asm,
           __ Uint32LessThanOrEqual(ReducedLength(Asm, a, 1), I31GetS(Asm, x)));
    BoundsCheck(Asm, a, I31GetS(Asm, x), 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, TruncationInSiblingBranch) {
  // if (c) a[trunc(x)]; a[trunc(x + 1)]; a[trunc(x + 1)];
  // The base of the last two accesses is trunc(x), found in the branch,
  // which does not leak facts after it: only the last check is
  // redundant.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word64> x = __ ChangeUint32ToUint64(i);
    If(Asm, c, [&] { BoundsCheck(Asm, a, TruncatedIndex(Asm, x, 0), 0); });
    BoundsCheck(Asm, a, TruncatedIndex(Asm, x, 1), 0);
    BoundsCheck(Asm, a, TruncatedIndex(Asm, x, 1), 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, Truncated64BitConstant) {
  // a[trunc(x + 1)]; a[trunc(x + 2^32 + 1)];
  // Both indices are the same 32-bit value.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word64> x = __ ChangeUint32ToUint64(i);
    BoundsCheck(Asm, a, TruncatedIndex(Asm, x, 0), 0);
    BoundsCheck(Asm, a, TruncatedIndex(Asm, x, 1), 0);
    BoundsCheck(Asm, a, TruncatedIndex(Asm, x, (int64_t{1} << 32) + 1), 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, DeeplyNestedAdditions) {
  // a[i+8]; a[((i+1)+1)...+1] (8 additions);
  // The nested additions are folded up to a depth of 8.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, i, 8);
    V<Word32> index = i;
    for (int k = 0; k < 8; k++) {
      index = __ Word32Add(index, __ Word32Constant(1));
    }
    BoundsCheck(Asm, a, index, 0);
  });
  Run(test);
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

TEST_F(WasmBoundsCheckEliminationReducerTest, OrOfConditions) {
  // if ((a.length <= i) | (b.length <= i)) fail; a[i]; b[i];
  // When the or does not hold, neither condition holds.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> a_length = __ ArrayLength(a, compiler::kWithNullCheck);
    V<Word32> b_length = __ ArrayLength(b, compiler::kWithNullCheck);
    FailIf(Asm, __ Word32BitwiseOr(__ Uint32LessThanOrEqual(a_length, i),
                                   __ Uint32LessThanOrEqual(b_length, i)));
    BoundsCheck(Asm, a, i, 0);
    BoundsCheck(Asm, b, i, 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, OrOfConditionsHolding) {
  // if ((i < a.length) | c) a[i];
  // The or holding does not show that one of its conditions holds.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> a_length = __ ArrayLength(a, compiler::kWithNullCheck);
    If(Asm, __ Word32BitwiseOr(__ Uint32LessThan(i, a_length), c),
       [&] { BoundsCheck(Asm, a, i, 0); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, OrOfConditionsHoldingNegated) {
  // if ((a.length <= i) | c) a[i];
  // The or holding does not show that its conditions do not hold.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> a_length = __ ArrayLength(a, compiler::kWithNullCheck);
    If(Asm, __ Word32BitwiseOr(__ Uint32LessThanOrEqual(a_length, i), c),
       [&] { BoundsCheck(Asm, a, i, 0); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, AndOfConditions) {
  // if ((i < a.length) & (i < b.length)) { a[i]; b[i]; }
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> a_length = __ ArrayLength(a, compiler::kWithNullCheck);
    V<Word32> b_length = __ ArrayLength(b, compiler::kWithNullCheck);
    If(Asm,
       __ Word32BitwiseAnd(__ Uint32LessThan(i, a_length),
                           __ Uint32LessThan(i, b_length)),
       [&] {
         BoundsCheck(Asm, a, i, 0);
         BoundsCheck(Asm, b, i, 0);
       });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, AndOfValues) {
  // if ((i < a.length) & c) a[i];
  // {c} is not a comparison, but when the and is not 0, neither operand
  // is, so the comparison holds.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> a_length = __ ArrayLength(a, compiler::kWithNullCheck);
    If(Asm, __ Word32BitwiseAnd(__ Uint32LessThan(i, a_length), c),
       [&] { BoundsCheck(Asm, a, i, 0); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, AndNotHolding) {
  // if ((a.length <=u i) & c) fail; a[i];
  // When the and does not hold (is 0), nothing is known about its
  // operands: (a.length <=u i) may hold, with c == 0.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> a_length = __ ArrayLength(a, compiler::kWithNullCheck);
    FailIf(Asm, __ Word32BitwiseAnd(__ Uint32LessThanOrEqual(a_length, i), c));
    BoundsCheck(Asm, a, i, 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, AndNotHoldingIsNotSplit) {
  // if ((i <u a.length) & c) fail; a[i];
  // The and may not hold because c is 0, while i <u a.length holds or
  // not.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> a_length = __ ArrayLength(a, compiler::kWithNullCheck);
    FailIf(Asm, __ Word32BitwiseAnd(__ Uint32LessThan(i, a_length), c));
    BoundsCheck(Asm, a, i, 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NegatedCondition) {
  // if ((i < a.length) == 0) fail; a[i];
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> a_length = __ ArrayLength(a, compiler::kWithNullCheck);
    FailIf(Asm, __ Word32Equal(__ Uint32LessThan(i, a_length), 0));
    BoundsCheck(Asm, a, i, 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, SignedComparison) {
  // b[i]; if (a.length - 2 <=s i) fail; a[i]; a[i+1]; a[i+2];
  // b[i] shows that i is non-negative, so the signed comparison shows
  // that i + 2 < a.length, without knowing the length of a.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, b, i, 0);
    FailIf(Asm, __ Int32LessThanOrEqual(ReducedLength(Asm, a, 2), i));
    BoundsCheck(Asm, a, i, 0);
    BoundsCheck(Asm, a, i, 1);
    BoundsCheck(Asm, a, i, 2);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, SignedComparisonWithUnknownSign) {
  // if (a.length - 2 <=s i) fail; a[i];
  // i may be negative.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    FailIf(Asm, __ Int32LessThanOrEqual(ReducedLength(Asm, a, 2), i));
    BoundsCheck(Asm, a, i, 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       SignedComparisonAboveCheckedOffset) {
  // a[i]; if (!(i + 5 <s a.length)) fail; a[i+5];
  // i + 5 is non-negative, since i is within bounds and array lengths
  // are small.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, i, 0);
    V<Word32> a_length = __ ArrayLength(a, compiler::kWithNullCheck);
    FailIf(Asm, __ Int32LessThanOrEqual(a_length,
                                        __ Word32Add(i, __ Word32Constant(5))));
    BoundsCheck(Asm, a, i, 5);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, SignedLessThanOrEqual) {
  // b[i]; if (i <=s a.length - 2) a[i+2];
  // i + 2 may be the length of a.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, b, i, 0);
    If(Asm, __ Int32LessThanOrEqual(i, ReducedLength(Asm, a, 2)),
       [&] { BoundsCheck(Asm, a, i, 2); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       SignedComparisonFarAboveCheckedOffset) {
  // a[i]; if (!(i + 0x7fffffff <s a.length)) fail; a[i+0x7fffffff];
  // i + 0x7fffffff may be negative.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, i, 0);
    V<Word32> a_length = __ ArrayLength(a, compiler::kWithNullCheck);
    FailIf(Asm, __ Int32LessThanOrEqual(
                    a_length, __ Word32Add(i, __ Word32Constant(0x7fffffff))));
    BoundsCheck(Asm, a, i, 0x7fffffff);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, SignedComparisonMinLength) {
  // b[i]; if (a.length - 1 <=s i) fail; if (a.length - 3 <= c) fail;
  // a[c+3];
  // The signed comparison shows that the length of a is at least 2, not
  // 3: a.length - 3 may wrap around, and a[c+3] may be out of bounds.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, b, i, 0);
    FailIf(Asm, __ Int32LessThanOrEqual(ReducedLength(Asm, a, 1), i));
    FailIf(Asm, __ Uint32LessThanOrEqual(ReducedLength(Asm, a, 3), c));
    BoundsCheck(Asm, a, c, 3);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       NonNegativeIndexFromSignedComparison) {
  // if (i >= 0) { a[i+1]; a[i]; }
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    If(Asm, __ Int32LessThanOrEqual(__ Word32Constant(0), i), [&]() {
      BoundsCheck(Asm, a, i, 1);
      BoundsCheck(Asm, a, i, 0);
    });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NotEqualToLength) {
  // a[i]; if (i + 1 == a.length) fail; a[i+1];
  // i + 1 is at most the length, and not equal to it.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, i, 0);
    V<Word32> length = __ ArrayLength(a, compiler::kWithNullCheck);
    FailIf(Asm, __ Word32Equal(__ Word32Add(i, __ Word32Constant(1)), length));
    BoundsCheck(Asm, a, i, 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, EqualToLength) {
  // a[i]; if (i + 1 == a.length) a[i+1];
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, i, 0);
    V<Word32> length = __ ArrayLength(a, compiler::kWithNullCheck);
    If(Asm, __ Word32Equal(__ Word32Add(i, __ Word32Constant(1)), length),
       [&] { BoundsCheck(Asm, a, i, 1); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NotEqualToLengthTooFar) {
  // a[i]; if (i + 2 == a.length) fail; a[i+1];
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, i, 0);
    V<Word32> length = __ ArrayLength(a, compiler::kWithNullCheck);
    FailIf(Asm, __ Word32Equal(__ Word32Add(i, __ Word32Constant(2)), length));
    BoundsCheck(Asm, a, i, 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NotEqualToReducedLength) {
  // a[i+1]; if (i == a.length - 2) fail; a[i+2];
  // i + 2 is at most the length, and not equal to it.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, i, 1);
    FailIf(Asm, __ Word32Equal(i, ReducedLength(Asm, a, 2)));
    BoundsCheck(Asm, a, i, 2);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LengthNotEqualToIndex) {
  // a[i]; if (a.length == i + 1) fail; a[i+1];
  // As {NotEqualToLength}, with the length on the left.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, i, 0);
    V<Word32> length = __ ArrayLength(a, compiler::kWithNullCheck);
    FailIf(Asm, __ Word32Equal(length, __ Word32Add(i, __ Word32Constant(1))));
    BoundsCheck(Asm, a, i, 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NonStrictReducedLength) {
  // b[i]; if (i <=s a.length - 2) a[i+1];
  // i <= a.length - 2 is i < a.length - 1, which covers i and i + 1.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, b, i, 0);
    If(Asm, __ Int32LessThanOrEqual(i, ReducedLength(Asm, a, 2)),
       [&] { BoundsCheck(Asm, a, i, 1); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       NonStrictUnsignedReducedLengthUnknownLength) {
  // if (i <=u a.length - 1) a[i];
  // The array might be empty, in which case a.length - 1 wraps around.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    If(Asm, __ Uint32LessThanOrEqual(i, ReducedLength(Asm, a, 1)),
       [&] { BoundsCheck(Asm, a, i, 0); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       NonStrictUnsignedReducedLengthKnownLength) {
  // a[0]; if (i <=u a.length - 1) a[i];
  // The first access shows that the array is not empty.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, {}, 0);
    If(Asm, __ Uint32LessThanOrEqual(i, ReducedLength(Asm, a, 1)),
       [&] { BoundsCheck(Asm, a, i, 0); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, ConstantAtMostLength) {
  // if (2 <=s a.length) { a[1]; a[0]; }
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> a_length = __ ArrayLength(a, compiler::kWithNullCheck);
    If(Asm, __ Int32LessThanOrEqual(__ Word32Constant(2), a_length), [&] {
      BoundsCheck(Asm, a, {}, 1);
      BoundsCheck(Asm, a, {}, 0);
    });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, ConstantAtMostLengthTooLarge) {
  // if (2 <=s a.length) a[2];
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> a_length = __ ArrayLength(a, compiler::kWithNullCheck);
    If(Asm, __ Int32LessThanOrEqual(__ Word32Constant(2), a_length),
       [&] { BoundsCheck(Asm, a, {}, 2); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LengthNotZero) {
  // if (a.length != 0) a[0];
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> a_length = __ ArrayLength(a, compiler::kWithNullCheck);
    If(Asm,
       __ Word32Equal(__ Word32Equal(a_length, __ Word32Constant(0)),
                      __ Word32Constant(0)),
       [&] { BoundsCheck(Asm, a, {}, 0); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, BranchOnLength) {
  // if (a.length) a[0];
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> a_length = __ ArrayLength(a, compiler::kWithNullCheck);
    If(Asm, a_length, [&] { BoundsCheck(Asm, a, {}, 0); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LengthNotEqualToMinLength) {
  // a[0]; if (a.length != 1) a[1];
  // The length is at least 1, and not 1.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, {}, 0);
    V<Word32> a_length = __ ArrayLength(a, compiler::kWithNullCheck);
    If(Asm,
       __ Word32Equal(__ Word32Equal(a_length, __ Word32Constant(1)),
                      __ Word32Constant(0)),
       [&] { BoundsCheck(Asm, a, {}, 1); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LengthNotEqualAboveMinLength) {
  // a[0]; if (a.length != 2) a[1];
  // The length may be 1.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, {}, 0);
    V<Word32> a_length = __ ArrayLength(a, compiler::kWithNullCheck);
    If(Asm,
       __ Word32Equal(__ Word32Equal(a_length, __ Word32Constant(2)),
                      __ Word32Constant(0)),
       [&] { BoundsCheck(Asm, a, {}, 1); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, ReducedLengthNotEqual) {
  // a[0]; if (a.length - 1 != 0) a[1];
  // The length is at least 1, and not 1.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, {}, 0);
    If(Asm,
       __ Word32Equal(__ Word32Equal(ReducedLength(Asm, a, 1), 0),
                      __ Word32Constant(0)),
       [&] { BoundsCheck(Asm, a, {}, 1); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       ReducedLengthNotEqualAboveMinLength) {
  // a[0]; if (a.length - 1 != 1) a[1];
  // The length may be 1.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, {}, 0);
    If(Asm,
       __ Word32Equal(
           __ Word32Equal(ReducedLength(Asm, a, 1), __ Word32Constant(1)),
           __ Word32Constant(0)),
       [&] { BoundsCheck(Asm, a, {}, 1); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       SignedConditionCoversConstantIndices) {
  // b[i]; if (i <s a.length - 2) { a[2]; a[0]; a[1]; }
  // The length of a is at least 3.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, b, i, 0);
    If(Asm, __ Int32LessThan(i, ReducedLength(Asm, a, 2)), [&] {
      BoundsCheck(Asm, a, {}, 2);
      BoundsCheck(Asm, a, {}, 0);
      BoundsCheck(Asm, a, {}, 1);
    });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

// A 64-bit value whose low 32 bits are {low} and high 32 bits are {high}.
template <typename Asm_t>
static V<Word64> Wide(Asm_t& Asm, V<Word32> low, V<Word32> high) {
  return __ Word64BitwiseOr(
      __ ChangeUint32ToUint64(low),
      __ Word64ShiftLeft(__ ChangeUint32ToUint64(high), 32));
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NarrowedOCamlBoundsCheck) {
  // a[0]; if (zext(a.length - 1) <=u x) fail; a[trunc(x) + 1];
  // The OCaml bounds check of a 64-bit index {x} shows that {x} is less
  // than 2^32, and that {trunc(x) < a.length - 1}.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word64> x = Wide(Asm, i, c);
    BoundsCheck(Asm, a, {}, 0);
    FailIf(Asm, __ Uint64LessThanOrEqual(
                    __ ChangeUint32ToUint64(ReducedLength(Asm, a, 1)), x));
    BoundsCheck(Asm, a, __ TruncateWord64ToWord32(x), 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NarrowedSubtraction) {
  // if (zext(a.length) <=u zext(i) - 1) fail; a[i+1];
  // The condition shows that i - 1 is within bounds, not i + 1.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> length = __ ArrayLength(a, compiler::kWithNullCheck);
    FailIf(Asm, __ Uint64LessThanOrEqual(
                    __ ChangeUint32ToUint64(length),
                    __ Word64Sub(__ ChangeUint32ToUint64(i),
                                 __ Word64Constant(uint64_t{1}))));
    BoundsCheck(Asm, a, i, 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, ReducedLengthNotEqualPositive) {
  // a[1]; if (a.length - 1 != 1) a[2];
  // The length is at least 2, and not 2.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, {}, 1);
    If(Asm,
       __ Word32Equal(
           __ Word32Equal(ReducedLength(Asm, a, 1), __ Word32Constant(1)),
           __ Word32Constant(0)),
       [&] { BoundsCheck(Asm, a, {}, 2); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NarrowedAfterTruncation) {
  // b[trunc(x)]; if (zext(a.length) <=u x + 1) fail; a[trunc(x) + 1];
  // trunc(x) is canonicalized before the narrowed comparison, which reads
  // x + 1 as trunc(x) + 1.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word64> x = Wide(Asm, i, c);
    V<Word32> truncated = __ TruncateWord64ToWord32(x);
    BoundsCheck(Asm, b, truncated, 0);
    V<Word32> length = __ ArrayLength(a, compiler::kWithNullCheck);
    FailIf(Asm, __ Uint64LessThanOrEqual(
                    __ ChangeUint32ToUint64(length),
                    __ Word64Add(x, __ Word64Constant(uint64_t{1}))));
    BoundsCheck(Asm, a, truncated, 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

// if (lo op i) { a[i+1]; a[i]; } for a signed comparison {op}.
template <typename Asm_t>
static void SignedLowerBound(Asm_t& Asm, V<WasmArrayNullable> a, V<Word32> i,
                             int32_t lo, bool strict) {
  using Test = WasmBoundsCheckEliminationReducerTest;
  V<Word32> lo_value = __ Word32Constant(lo);
  Test::If(Asm,
           strict ? __ Int32LessThan(lo_value, i)
                  : __ Int32LessThanOrEqual(lo_value, i),
           [&] {
             Test::BoundsCheck(Asm, a, i, 1);
             Test::BoundsCheck(Asm, a, i, 0);
           });
}

TEST_F(WasmBoundsCheckEliminationReducerTest, SignedLowerBoundEdges) {
  // -1 <s i shows that i is non-negative, so that a[i+1] covers a[i].
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    SignedLowerBound(Asm, a, i, -1, true);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
  // -1 <=s i and -2 <s i do not.
  auto test2 = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    SignedLowerBound(Asm, a, i, -1, false);
  });
  Run(test2);
  ASSERT_EQ(test2.CountOp(Opcode::kTrapIf), 2u);
  auto test3 = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    SignedLowerBound(Asm, a, i, -2, true);
  });
  Run(test3);
  ASSERT_EQ(test3.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NarrowedConstant) {
  // if (zext(a.length) <=u 3) fail; a[3];
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> length = __ ArrayLength(a, compiler::kWithNullCheck);
    FailIf(Asm, __ Uint64LessThanOrEqual(__ ChangeUint32ToUint64(length),
                                         __ Word64Constant(uint64_t{3})));
    BoundsCheck(Asm, a, {}, 3);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NarrowedNonStrictComparison) {
  // a[0]; if (x >u zext(a.length - 1)) fail; a[trunc(x)];
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word64> x = Wide(Asm, i, c);
    BoundsCheck(Asm, a, {}, 0);
    FailIf(Asm, __ Uint64LessThan(
                    __ ChangeUint32ToUint64(ReducedLength(Asm, a, 1)), x));
    BoundsCheck(Asm, a, __ TruncateWord64ToWord32(x), 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NarrowedExtendedIndex) {
  // if (zext(a.length) <=u zext(i) + 1) fail; a[i+1];
  // This needs no min length, unlike {i <u a.length - 1}.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> length = __ ArrayLength(a, compiler::kWithNullCheck);
    FailIf(Asm, __ Uint64LessThanOrEqual(
                    __ ChangeUint32ToUint64(length),
                    __ Word64Add(__ ChangeUint32ToUint64(i),
                                 __ Word64Constant(uint64_t{1}))));
    BoundsCheck(Asm, a, i, 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NarrowedSignExtendedIndex) {
  // if (zext(a.length) <=u sext(i) + 1) fail; a[i+1];
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> length = __ ArrayLength(a, compiler::kWithNullCheck);
    FailIf(Asm, __ Uint64LessThanOrEqual(
                    __ ChangeUint32ToUint64(length),
                    __ Word64Add(__ ChangeInt32ToInt64(i),
                                 __ Word64Constant(uint64_t{1}))));
    BoundsCheck(Asm, a, i, 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NarrowedSignedComparison) {
  // b[trunc(x)]; if (zext(a.length) <=s x) fail; a[trunc(x)];
  // A negative {x} passes the signed test, whatever {trunc(x)} is (and
  // the first access shows that it is non-negative).
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word64> x = Wide(Asm, i, c);
    BoundsCheck(Asm, b, __ TruncateWord64ToWord32(x), 0);
    V<Word32> length = __ ArrayLength(a, compiler::kWithNullCheck);
    FailIf(Asm, __ Int64LessThanOrEqual(__ ChangeUint32ToUint64(length), x));
    BoundsCheck(Asm, a, __ TruncateWord64ToWord32(x), 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NarrowedWithoutZeroExtension) {
  // if (sext(a.length) <=u x) fail; a[trunc(x)];
  // Only a zero-extended right side is recognized.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word64> x = Wide(Asm, i, c);
    V<Word32> length = __ ArrayLength(a, compiler::kWithNullCheck);
    FailIf(Asm, __ Uint64LessThanOrEqual(__ ChangeInt32ToInt64(length), x));
    BoundsCheck(Asm, a, __ TruncateWord64ToWord32(x), 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NarrowedNotHolding) {
  // if (x <u zext(5)) fail; a[4]; with x = zext(a.length) + 2^32
  // {zext(5) <=u x} holds for any length, and says nothing about
  // {trunc(x)}, which is {a.length}: only the normalized relation, whose
  // right side is {x}, may be narrowed.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> length = __ ArrayLength(a, compiler::kWithNullCheck);
    V<Word64> x = __ Word64Add(__ ChangeUint32ToUint64(length),
                               __ Word64Constant(uint64_t{1} << 32));
    FailIf(Asm,
           __ Uint64LessThan(x, __ ChangeUint32ToUint64(__ Word32Constant(5))));
    BoundsCheck(Asm, a, {}, 4);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NarrowedTrapIsKept) {
  // a[i]; trap if !(x <u zext(a.length)); with trunc(x) = i
  // The trap holds for {x >= 2^32} although {trunc(x) < a.length}.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word64> x = Wide(Asm, i, c);
    BoundsCheck(Asm, a, __ TruncateWord64ToWord32(x), 0);
    V<Word32> length = __ ArrayLength(a, compiler::kWithNullCheck);
    __ TrapIfNot(__ Uint64LessThan(x, __ ChangeUint32ToUint64(length)),
                 TrapId::kTrapArrayOutOfBounds);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       NarrowedComparisonCreatesNoTruncation) {
  // if (zext(a.length) <=u x) fail;
  // a[trunc(x + 1)]; a[trunc(x + 2)]; a[trunc(x + 3)];
  // No operation computes trunc(x), so it cannot be the base of the
  // accesses (a DCHECK in {DecomposeIndex} checks that the base of a
  // 32-bit index is a 32-bit value).
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word64> x = Wide(Asm, i, c);
    V<Word32> length = __ ArrayLength(a, compiler::kWithNullCheck);
    FailIf(Asm, __ Uint64LessThanOrEqual(__ ChangeUint32ToUint64(length), x));
    BoundsCheck(Asm, a, TruncatedIndex(Asm, x, 1), 0);
    BoundsCheck(Asm, a, TruncatedIndex(Asm, x, 2), 0);
    BoundsCheck(Asm, a, TruncatedIndex(Asm, x, 3), 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 3u);
}

// Loops over {k}, starting at {init} and adding {step}, with {exit}
// testing whether to leave the loop before {body}, and after {before}.
template <typename Asm_t, typename Exit, typename Body,
          typename Before = void (*)(V<Word32>)>
static void Loop(
    Asm_t& Asm, int32_t init, int32_t step, const Exit& exit, const Body& body,
    const Before& before = [](V<Word32>) {}) {
  LoopLabel<Word32> loop(&Asm);
  Label<> done(&Asm);
  GOTO(loop, __ Word32Constant(init));
  BIND_LOOP(loop, k) {
    before(k);
    GOTO_IF(exit(k), done);
    body(k);
    GOTO(loop, __ Word32Add(k, __ Word32Constant(step)));
  }
  BIND(done);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LoopWithSignedBound) {
  // for (k = 0; k <s a.length - 2; k += 3) { a[k]; a[k+1]; a[k+2]; }
  // k is non-negative, since it starts at 0, and its increments cannot
  // overflow, so the signed comparison covers the three accesses.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    Loop(
        Asm, 0, 3,
        [&](V<Word32> k) {
          return __ Int32LessThanOrEqual(ReducedLength(Asm, a, 2), k);
        },
        [&](V<Word32> k) {
          BoundsCheck(Asm, a, k, 0);
          BoundsCheck(Asm, a, k, 1);
          BoundsCheck(Asm, a, k, 2);
        });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LoopWithNegativeStart) {
  // for (k = -1; k <s a.length - 1; k += 1) { a[k+1]; a[k]; }
  // k may be negative.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    Loop(
        Asm, -1, 1,
        [&](V<Word32> k) {
          return __ Int32LessThanOrEqual(ReducedLength(Asm, a, 1), k);
        },
        [&](V<Word32> k) {
          BoundsCheck(Asm, a, k, 1);
          BoundsCheck(Asm, a, k, 0);
        });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LoopWithNegativeStep) {
  // for (k = 0; k <s a.length - 1; k -= 1) { a[k+1]; a[k]; }
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    Loop(
        Asm, 0, -1,
        [&](V<Word32> k) {
          return __ Int32LessThanOrEqual(ReducedLength(Asm, a, 1), k);
        },
        [&](V<Word32> k) {
          BoundsCheck(Asm, a, k, 1);
          BoundsCheck(Asm, a, k, 0);
        });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

template <typename Asm_t>
static void LoopWithConstantBound(Asm_t& Asm, V<WasmArrayNullable> a,
                                  int32_t bound, bool is_signed = true) {
  // for (k = 0; ; k += 3) {
  //   if (!(a.length - 1 <=s k)) { a[k+1]; a[k]; }
  //   if (!(k < bound)) break;
  // }
  // The accesses come before the exit test, so that only the induction
  // shows that k is non-negative.
  using Test = WasmBoundsCheckEliminationReducerTest;
  Loop(
      Asm, 0, 3,
      [&](V<Word32> k) {
        return is_signed ? __ Int32LessThanOrEqual(__ Word32Constant(bound), k)
                         : __ Uint32LessThanOrEqual(__ Word32Constant(bound),
                                                    k);
      },
      [&](V<Word32> k) {},
      [&](V<Word32> k) {
        Label<> next(&Asm);
        GOTO_IF(__ Int32LessThanOrEqual(Test::ReducedLength(Asm, a, 1), k),
                next);
        Test::BoundsCheck(Asm, a, k, 1);
        Test::BoundsCheck(Asm, a, k, 0);
        GOTO(next);
        BIND(next);
      });
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LoopWithSmallConstantBound) {
  // With k < 0x7ffffffd, k + 3 cannot overflow.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    LoopWithConstantBound(Asm, a, 0x7ffffffd);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       LoopWithSmallUnsignedConstantBound) {
  // With k <u 0x7ffffffd, k + 3 cannot overflow.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    LoopWithConstantBound(Asm, a, 0x7ffffffd, false);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       LoopWithLargeUnsignedConstantBound) {
  // With k <u 0x7ffffffe, k + 3 may overflow (as a signed integer).
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    LoopWithConstantBound(Asm, a, 0x7ffffffe, false);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LoopWithLargeConstantBound) {
  // With k < 0x7ffffffe, k + 3 may overflow.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    LoopWithConstantBound(Asm, a, 0x7ffffffe);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

// for (k = 0; ; k++) {
//   if (!(a.length - 1 <=s k)) { a[k+1]; a[k]; }
//   if (!(k <u a.length - reduction)) break;
// }
template <typename Asm_t>
static void LoopWithUnsignedLengthBound(Asm_t& Asm, V<WasmArrayNullable> a,
                                        int32_t reduction) {
  using Test = WasmBoundsCheckEliminationReducerTest;
  Loop(
      Asm, 0, 1,
      [&](V<Word32> k) {
        V<Word32> bound = reduction == 0
                              ? __ ArrayLength(a, compiler::kWithNullCheck)
                              : Test::ReducedLength(Asm, a, reduction);
        return __ Uint32LessThanOrEqual(bound, k);
      },
      [&](V<Word32> k) {},
      [&](V<Word32> k) {
        Label<> next(&Asm);
        GOTO_IF(__ Int32LessThanOrEqual(Test::ReducedLength(Asm, a, 1), k),
                next);
        Test::BoundsCheck(Asm, a, k, 1);
        Test::BoundsCheck(Asm, a, k, 0);
        GOTO(next);
        BIND(next);
      });
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LoopWithUnsignedLengthBound) {
  // With k <u a.length, k + 1 cannot overflow.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    LoopWithUnsignedLengthBound(Asm, a, 0);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       LoopWithUnsignedReducedLengthBound) {
  // a.length - 1 is 0xffffffff for an empty array, and k + 1 may then
  // overflow.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    LoopWithUnsignedLengthBound(Asm, a, 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

// for (k = 0; ; k += step) {
//   if (!(a.length - 1 <=s k)) { a[k+1]; a[k]; }
//   if (a.length <s k) break;
// }
// The accesses come before the exit test, so that only the induction
// shows that k is non-negative.
template <typename Asm_t>
static void LoopWithInclusiveLengthBound(Asm_t& Asm, V<WasmArrayNullable> a,
                                         int32_t step) {
  using Test = WasmBoundsCheckEliminationReducerTest;
  Loop(
      Asm, 0, step,
      [&](V<Word32> k) {
        return __ Int32LessThan(__ ArrayLength(a, compiler::kWithNullCheck), k);
      },
      [&](V<Word32> k) {},
      [&](V<Word32> k) {
        Label<> next(&Asm);
        GOTO_IF(__ Int32LessThanOrEqual(Test::ReducedLength(Asm, a, 1), k),
                next);
        Test::BoundsCheck(Asm, a, k, 1);
        Test::BoundsCheck(Asm, a, k, 0);
        GOTO(next);
        BIND(next);
      });
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       LoopWithInclusiveLengthBoundAndLargeStep) {
  // With k <= a.length < 2^30, k + 2^30 cannot overflow.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    LoopWithInclusiveLengthBound(Asm, a, 1 << 30);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       LoopWithInclusiveLengthBoundAndTooLargeStep) {
  // With k <= a.length, k + 2^30 + 1 may overflow.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    LoopWithInclusiveLengthBound(Asm, a, (1 << 30) + 1);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

// for (k = init; k != hi; k++) body, where the loop is only entered if
// {guard} holds.
template <typename Asm_t, typename Guard, typename Body>
static void NotEqualLoop(Asm_t& Asm, V<Word32> hi, int32_t init, int32_t step,
                         const Guard& guard, const Body& body) {
  Label<> done(&Asm);
  GOTO_IF_NOT(guard(), done);
  LoopLabel<Word32> loop(&Asm);
  GOTO(loop, __ Word32Constant(init));
  BIND_LOOP(loop, k) {
    body(k);
    for (int32_t n = 0; n < step; n++) {
      GOTO_IF(__ Word32Equal(__ Word32Add(k, __ Word32Constant(n)), hi), done);
    }
    GOTO(loop, __ Word32Add(k, __ Word32Constant(step)));
  }
  BIND(done);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NotEqualLoop) {
  // hi = a.length - 1; if (0 <= hi) for (k = 0; ; k++) {
  //   a[k]; if (k == hi) break;
  // }
  // As for an OCaml loop {for k = 0 to Array.length a - 1}: k is at
  // most hi in the loop.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> hi = ReducedLength(Asm, a, 1);
    NotEqualLoop(
        Asm, hi, 0, 1,
        [&] { return __ Int32LessThanOrEqual(__ Word32Constant(0), hi); },
        [&](V<Word32> k) { BoundsCheck(Asm, a, k, 0); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NotEqualLoopPastBound) {
  // if (0 <= hi) for (k = 0; ; k++) { a[k]; a[k+1]; if (k == hi) break; }
  // a[k+1] is out of bounds when k is hi.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> hi = ReducedLength(Asm, a, 1);
    NotEqualLoop(
        Asm, hi, 0, 1,
        [&] { return __ Int32LessThanOrEqual(__ Word32Constant(0), hi); },
        [&](V<Word32> k) {
          BoundsCheck(Asm, a, k, 0);
          BoundsCheck(Asm, a, k, 1);
        });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NotEqualLoopWithoutGuard) {
  // Without the guard, k may start above hi, and go past it.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> hi = ReducedLength(Asm, a, 1);
    NotEqualLoop(
        Asm, hi, 0, 1, [&] { return c; },
        [&](V<Word32> k) { BoundsCheck(Asm, a, k, 0); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NotEqualLoopAboveGuard) {
  // if (0 <= hi) for (k = 1; ; k++): k may start above hi.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> hi = ReducedLength(Asm, a, 1);
    NotEqualLoop(
        Asm, hi, 1, 1,
        [&] { return __ Int32LessThanOrEqual(__ Word32Constant(0), hi); },
        [&](V<Word32> k) { BoundsCheck(Asm, a, k, 0); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NotEqualLoopExcludedValue) {
  // if (0 <= hi && hi != 0) for (k = 1; ; k++), as after peeling the
  // first iteration of a loop from 0: k starts at most at hi.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> hi = ReducedLength(Asm, a, 1);
    Label<> done(&Asm);
    GOTO_IF(__ Int32LessThan(hi, __ Word32Constant(0)), done);
    NotEqualLoop(
        Asm, hi, 1, 1, [&] { return hi; },
        [&](V<Word32> k) { BoundsCheck(Asm, a, k, 0); });
    GOTO(done);
    BIND(done);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NotEqualLoopAboveExcludedValue) {
  // if (0 <= hi && hi != 0) for (k = 2; ; k++): hi may be 1.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> hi = ReducedLength(Asm, a, 1);
    Label<> done(&Asm);
    GOTO_IF(__ Int32LessThan(hi, __ Word32Constant(0)), done);
    NotEqualLoop(
        Asm, hi, 2, 1, [&] { return hi; },
        [&](V<Word32> k) { BoundsCheck(Asm, a, k, 0); });
    GOTO(done);
    BIND(done);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

// if (3 <= hi && hi != 3) for (k = init; ; k++) { a[k]; if (k == hi) break; }
// (A test {hi != 0} is a branch on {hi}.)
template <typename Asm_t>
static void NotEqualLoopExcludedConstant(Asm_t& Asm, V<WasmArrayNullable> a,
                                         int32_t init) {
  using Test = WasmBoundsCheckEliminationReducerTest;
  V<Word32> hi = Test::ReducedLength(Asm, a, 1);
  Label<> done(&Asm);
  GOTO_IF(__ Int32LessThan(hi, __ Word32Constant(3)), done);
  NotEqualLoop(
      Asm, hi, init, 1,
      [&] {
        return __ Word32Equal(__ Word32Equal(hi, __ Word32Constant(3)),
                              __ Word32Constant(0));
      },
      [&](V<Word32> k) { Test::BoundsCheck(Asm, a, k, 0); });
  GOTO(done);
  BIND(done);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NotEqualLoopExcludedConstant) {
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    NotEqualLoopExcludedConstant(Asm, a, 4);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       NotEqualLoopAboveExcludedConstant) {
  // hi may be 4.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    NotEqualLoopExcludedConstant(Asm, a, 5);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NotEqualLoopWhenZero) {
  // if (0 <= hi && !hi) for (k = 1; ; k++): k starts above hi.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> hi = ReducedLength(Asm, a, 1);
    Label<> done(&Asm);
    GOTO_IF(__ Int32LessThan(hi, __ Word32Constant(0)), done);
    // The loop is entered on the false edge of a branch on hi.
    GOTO_IF(hi, done);
    NotEqualLoop(
        Asm, hi, 1, 1, [&] { return c; },
        [&](V<Word32> k) { BoundsCheck(Asm, a, k, 0); });
    GOTO(done);
    BIND(done);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, UnrolledNotEqualLoop) {
  // if (0 <= hi) for (k = 0; ; k += 2) {
  //   a[k]; if (k == hi) break; if (k + 1 == hi) break;
  // }
  // As after unrolling: k still does not go past hi.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> hi = ReducedLength(Asm, a, 1);
    NotEqualLoop(
        Asm, hi, 0, 2,
        [&] { return __ Int32LessThanOrEqual(__ Word32Constant(0), hi); },
        [&](V<Word32> k) { BoundsCheck(Asm, a, k, 0); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, SteppingOverNotEqual) {
  // if (0 <= hi) for (k = 0; ; k += 2) { a[k]; if (k == hi) break; }
  // k may step over hi.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> hi = ReducedLength(Asm, a, 1);
    Label<> done(&Asm);
    GOTO_IF_NOT(__ Int32LessThanOrEqual(__ Word32Constant(0), hi), done);
    Loop(
        Asm, 0, 2, [&](V<Word32> k) { return __ Word32Constant(0); },
        [&](V<Word32> k) {
          BoundsCheck(Asm, a, k, 0);
          GOTO_IF(__ Word32Equal(k, hi), done);
        });
    GOTO(done);
    BIND(done);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

// hi = (c ? a.length : other) - 1; if (0 <= hi) for (k = 0; ; k++) {
//   a[k]; if (k == hi) break;
// }
template <typename Asm_t, typename Other>
static void LoopWithPhiBound(Asm_t& Asm, V<WasmArrayNullable> a, V<Word32> c,
                             const Other& other) {
  using Test = WasmBoundsCheckEliminationReducerTest;
  Label<Word32> merge(&Asm);
  IF (c) {
    GOTO(merge, __ ArrayLength(a, compiler::kWithNullCheck));
  } ELSE {
    GOTO(merge, other());
  }
  BIND(merge, length);
  V<Word32> hi = __ Word32Sub(length, __ Word32Constant(1));
  NotEqualLoop(
      Asm, hi, 0, 1,
      [&] { return __ Int32LessThanOrEqual(__ Word32Constant(0), hi); },
      [&](V<Word32> k) { Test::BoundsCheck(Asm, a, k, 0); });
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LoopWithPhiBound) {
  // As for the length of an OCaml array that may be a float array or
  // empty: on the path where the length is 0, hi is -1, and the loop is
  // not entered, so hi is a.length - 1 in the loop.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    LoopWithPhiBound(Asm, a, c, [&] { return __ Word32Constant(0); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LoopWithPhiOfLengths) {
  // hi may be b.length - 1.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    LoopWithPhiBound(
        Asm, a, c, [&] { return __ ArrayLength(b, compiler::kWithNullCheck); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LoopWithPhiOfLargeConstant) {
  // hi may be 5 - 1, and the loop may then be entered.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    LoopWithPhiBound(Asm, a, c, [&] { return __ Word32Constant(5); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LoopWithPhiOfSmallConstant) {
  // hi may be 1 - 1 = 0, and the loop may then be entered.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    LoopWithPhiBound(Asm, a, c, [&] { return __ Word32Constant(1); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LoopWithPhiOfLengthsReversed) {
  // hi = (c ? b.length : a.length) - 1; if (0 <= hi) for (k = 0; ; k++) {
  //   a[k]; if (k == hi) break;
  // }
  // hi may be b.length - 1.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    Label<Word32> merge(&Asm);
    IF (c) {
      GOTO(merge, __ ArrayLength(b, compiler::kWithNullCheck));
    } ELSE {
      GOTO(merge, __ ArrayLength(a, compiler::kWithNullCheck));
    }
    BIND(merge, length);
    V<Word32> hi = __ Word32Sub(length, __ Word32Constant(1));
    NotEqualLoop(
        Asm, hi, 0, 1,
        [&] { return __ Int32LessThanOrEqual(__ Word32Constant(0), hi); },
        [&](V<Word32> k) { BoundsCheck(Asm, a, k, 0); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LoopWithLoopPhiBound) {
  // for (arr = a, len = 0; c; arr = b, len = arr.length) {
  //   hi = len - 1; if (0 <= hi) for (k = 0; ; k++) {
  //     arr[k]; if (k == hi) break;
  //   }
  // }
  // len is 0 only in the first iteration, but it is then the length of
  // the array of the previous iteration, not of arr.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    LoopLabel<WasmArrayNullable, Word32> outer(&Asm);
    Label<> done(&Asm);
    GOTO(outer, a, __ Word32Constant(0));
    BIND_LOOP(outer, arr, len) {
      GOTO_IF_NOT(c, done);
      V<Word32> hi = __ Word32Sub(len, __ Word32Constant(1));
      NotEqualLoop(
          Asm, hi, 0, 1,
          [&] { return __ Int32LessThanOrEqual(__ Word32Constant(0), hi); },
          [&](V<Word32> k) { BoundsCheck(Asm, arr, k, 0); });
      GOTO(outer, b, __ ArrayLength(arr, compiler::kWithNullCheck));
    }
    BIND(done);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

// hi = (c ? a.length : 0) - 1;
// if (i) { if (0 <= hi) for (k = 0; ; k++) { a[k]; if (k == hi) break; } }
// a[c]; if (c == hi) fail; a[c+1];
// hi is a.length - 1 in the loop, but not after it: on the path that
// does not go through the loop, it may be -1. The loop is in the
// {loop_first} branch.
template <typename Asm_t>
static void LengthAliasAfterLoop(Asm_t& Asm, V<WasmArrayNullable> a,
                                 V<Word32> i, V<Word32> c, bool loop_first) {
  using Test = WasmBoundsCheckEliminationReducerTest;
  Label<Word32> merge(&Asm);
  IF (c) {
    GOTO(merge, __ ArrayLength(a, compiler::kWithNullCheck));
  } ELSE {
    GOTO(merge, __ Word32Constant(0));
  }
  BIND(merge, length);
  V<Word32> hi = __ Word32Sub(length, __ Word32Constant(1));
  auto loop = [&] {
    NotEqualLoop(
        Asm, hi, 0, 1,
        [&] { return __ Int32LessThanOrEqual(__ Word32Constant(0), hi); },
        [&](V<Word32> k) { Test::BoundsCheck(Asm, a, k, 0); });
  };
  Label<> after(&Asm);
  IF (i) {
    if (loop_first) loop();
    GOTO(after);
  } ELSE {
    if (!loop_first) loop();
    GOTO(after);
  }
  BIND(after);
  Test::BoundsCheck(Asm, a, c, 0);
  Test::FailIf(Asm, __ Word32Equal(c, hi));
  Test::BoundsCheck(Asm, a, c, 1);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LengthAliasAfterLoop) {
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    LengthAliasAfterLoop(Asm, a, i, c, true);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LengthAliasAfterLoopInElse) {
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    LengthAliasAfterLoop(Asm, a, i, c, false);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

// hi = (c ? a.length : 0) - 1;
// if (0 <= hi) { for (k = 0; ; k++) { a[k]; if (k == hi) break; } }
// a[i]; if (i == hi) fail; a[i+1];
// After the loop, hi may be -1, when the loop was not entered. With
// {loop_first}, the loop is in the first branch of the test on hi.
template <typename Asm_t>
static void LengthAliasAfterLoopGuard(Asm_t& Asm, V<WasmArrayNullable> a,
                                      V<Word32> i, V<Word32> c,
                                      bool loop_first) {
  using Test = WasmBoundsCheckEliminationReducerTest;
  Label<Word32> merge(&Asm);
  IF (c) {
    GOTO(merge, __ ArrayLength(a, compiler::kWithNullCheck));
  } ELSE {
    GOTO(merge, __ Word32Constant(0));
  }
  BIND(merge, length);
  V<Word32> hi = __ Word32Sub(length, __ Word32Constant(1));
  auto loop = [&] {
    LoopLabel<Word32> loop(&Asm);
    Label<> exit(&Asm);
    GOTO(loop, __ Word32Constant(0));
    BIND_LOOP(loop, k) {
      Test::BoundsCheck(Asm, a, k, 0);
      GOTO_IF(__ Word32Equal(k, hi), exit);
      GOTO(loop, __ Word32Add(k, __ Word32Constant(1)));
    }
    BIND(exit);
  };
  V<Word32> negative = __ Int32LessThan(hi, __ Word32Constant(0));
  Label<> after(&Asm);
  IF (loop_first ? __ Word32Equal(negative, __ Word32Constant(0)) : negative) {
    if (loop_first) loop();
    GOTO(after);
  } ELSE {
    if (!loop_first) loop();
    GOTO(after);
  }
  BIND(after);
  Test::BoundsCheck(Asm, a, i, 0);
  Test::FailIf(Asm, __ Word32Equal(i, hi));
  Test::BoundsCheck(Asm, a, i, 1);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LengthAliasAfterLoopGuard) {
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    LengthAliasAfterLoopGuard(Asm, a, i, c, true);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LengthAliasAfterLoopGuardInElse) {
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    LengthAliasAfterLoopGuard(Asm, a, i, c, false);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LengthAliasAtLoopExit) {
  // hi = (c ? a.length : 0) - 1; if (0 <= hi) for (k = 0; ; k++) {
  //   a[k]; if (k == hi) { a[i]; if (i == hi) fail; a[i+1]; break; }
  // }
  // The loop exit is dominated by the loop header, so hi is still
  // a.length - 1 there.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    Label<Word32> merge(&Asm);
    IF (c) {
      GOTO(merge, __ ArrayLength(a, compiler::kWithNullCheck));
    } ELSE {
      GOTO(merge, __ Word32Constant(0));
    }
    BIND(merge, length);
    V<Word32> hi = __ Word32Sub(length, __ Word32Constant(1));
    Label<> done(&Asm);
    GOTO_IF_NOT(__ Int32LessThanOrEqual(__ Word32Constant(0), hi), done);
    LoopLabel<Word32> loop(&Asm);
    GOTO(loop, __ Word32Constant(0));
    BIND_LOOP(loop, k) {
      BoundsCheck(Asm, a, k, 0);
      IF (__ Word32Equal(k, hi)) {
        BoundsCheck(Asm, a, i, 0);
        FailIf(Asm, __ Word32Equal(i, hi));
        BoundsCheck(Asm, a, i, 1);
        GOTO(done);
      }
      GOTO(loop, __ Word32Add(k, __ Word32Constant(1)));
    }
    BIND(done);
  });
  Run(test);
  // Only a[i] is checked.
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

// for (k = init; ; k -= step) { body(k); if (exit(k)) break; }, where
// the loop is only entered if {guard} holds.
template <typename Asm_t, typename Guard, typename Exit, typename Body>
static void DecreasingLoop(Asm_t& Asm, V<Word32> init, int32_t step,
                           const Guard& guard, const Exit& exit,
                           const Body& body) {
  Label<> done(&Asm);
  GOTO_IF_NOT(guard(), done);
  LoopLabel<Word32> loop(&Asm);
  GOTO(loop, init);
  BIND_LOOP(loop, k) {
    body(k);
    GOTO_IF(exit(k), done);
    GOTO(loop, __ Word32Sub(k, __ Word32Constant(step)));
  }
  BIND(done);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, DecreasingLoop) {
  // hi = a.length - 1; if (0 <=s hi) for (k = hi; ; k--) {
  //   a[k]; if (k - 1 <s 0) break;
  // }
  // As for {for (k = a.length - 1; k >= 0; k--)}: k is non-negative,
  // and at most a.length - 1.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> hi = ReducedLength(Asm, a, 1);
    DecreasingLoop(
        Asm, hi, 1,
        [&] { return __ Int32LessThanOrEqual(__ Word32Constant(0), hi); },
        [&](V<Word32> k) {
          return __ Int32LessThan(__ Word32Sub(k, __ Word32Constant(1)),
                                  __ Word32Constant(0));
        },
        [&](V<Word32> k) { BoundsCheck(Asm, a, k, 0); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, DecreasingLoopTestBefore) {
  // if (0 <=s hi) for (k = hi; ; k--) { a[k]; if (k <=s 0) break; }
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> hi = ReducedLength(Asm, a, 1);
    DecreasingLoop(
        Asm, hi, 1,
        [&] { return __ Int32LessThanOrEqual(__ Word32Constant(0), hi); },
        [&](V<Word32> k) {
          return __ Int32LessThanOrEqual(k, __ Word32Constant(0));
        },
        [&](V<Word32> k) { BoundsCheck(Asm, a, k, 0); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, DecreasingLoopWithoutGuard) {
  // for (k = a.length - 1; ; k--) { a[k]; if (k - 1 <s 0) break; }
  // k is -1 if the array is empty.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> hi = ReducedLength(Asm, a, 1);
    DecreasingLoop(
        Asm, hi, 1, [&] { return c; },
        [&](V<Word32> k) {
          return __ Int32LessThan(__ Word32Sub(k, __ Word32Constant(1)),
                                  __ Word32Constant(0));
        },
        [&](V<Word32> k) { BoundsCheck(Asm, a, k, 0); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, DecreasingLoopTestTooEarly) {
  // if (0 <=s hi) for (k = hi; ; k--) { a[k]; if (k <s 0) break; }
  // k becomes -1 after k = 0.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> hi = ReducedLength(Asm, a, 1);
    DecreasingLoop(
        Asm, hi, 1,
        [&] { return __ Int32LessThanOrEqual(__ Word32Constant(0), hi); },
        [&](V<Word32> k) { return __ Int32LessThan(k, __ Word32Constant(0)); },
        [&](V<Word32> k) { BoundsCheck(Asm, a, k, 0); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, DecreasingLoopUnsignedTest) {
  // if (0 <=s hi) for (k = hi; ; k--) { a[k]; if (k - 1 <u 1) break; }
  // With hi = 0, k - 1 is 0xffffffff, and k becomes -1.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> hi = ReducedLength(Asm, a, 1);
    DecreasingLoop(
        Asm, hi, 1,
        [&] { return __ Int32LessThanOrEqual(__ Word32Constant(0), hi); },
        [&](V<Word32> k) {
          return __ Uint32LessThan(__ Word32Sub(k, __ Word32Constant(1)),
                                   __ Word32Constant(1));
        },
        [&](V<Word32> k) { BoundsCheck(Asm, a, k, 0); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, DecreasingLoopFromMinusOne) {
  // if (-1 <=s hi) for (k = hi; ; k--) { a[k]; if (k - 1 <s 0) break; }
  // The guard does not show that k starts non-negative.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> hi = ReducedLength(Asm, a, 1);
    DecreasingLoop(
        Asm, hi, 1,
        [&] { return __ Int32LessThanOrEqual(__ Word32Constant(-1), hi); },
        [&](V<Word32> k) {
          return __ Int32LessThan(__ Word32Sub(k, __ Word32Constant(1)),
                                  __ Word32Constant(0));
        },
        [&](V<Word32> k) { BoundsCheck(Asm, a, k, 0); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, DecreasingLoopPastBound) {
  // if (0 <=s hi) for (k = hi; ; k--) { a[k+1]; if (k - 1 <s 0) break; }
  // a[k+1] is out of bounds when k is hi.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> hi = ReducedLength(Asm, a, 1);
    DecreasingLoop(
        Asm, hi, 1,
        [&] { return __ Int32LessThanOrEqual(__ Word32Constant(0), hi); },
        [&](V<Word32> k) {
          return __ Int32LessThan(__ Word32Sub(k, __ Word32Constant(1)),
                                  __ Word32Constant(0));
        },
        [&](V<Word32> k) { BoundsCheck(Asm, a, k, 1); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, DecreasingNotEqualLoop) {
  // hi = a.length - 2; if (0 <=s hi) for (k = hi; ; k--) {
  //   a[k+1]; if (k == 0) break;
  // }
  // As for an OCaml loop {for k = Array.length a - 1 downto 0}, with
  // the OCaml array stored from index 1 of the Wasm array.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> hi = ReducedLength(Asm, a, 2);
    DecreasingLoop(
        Asm, hi, 1,
        [&] { return __ Int32LessThanOrEqual(__ Word32Constant(0), hi); },
        [&](V<Word32> k) { return __ Word32Equal(k, __ Word32Constant(0)); },
        [&](V<Word32> k) { BoundsCheck(Asm, a, k, 1); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       DecreasingNotEqualLoopAboveGuard) {
  // if (0 <=s hi) for (k = hi; ; k--) { a[k]; if (k == 1) break; }
  // With hi = 0, k goes past 1.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> hi = ReducedLength(Asm, a, 1);
    DecreasingLoop(
        Asm, hi, 1,
        [&] { return __ Int32LessThanOrEqual(__ Word32Constant(0), hi); },
        [&](V<Word32> k) { return __ Word32Equal(k, __ Word32Constant(1)); },
        [&](V<Word32> k) { BoundsCheck(Asm, a, k, 0); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, DecreasingNotEqualLoopUnrolled) {
  // if (0 <=s hi) for (k = hi; ; k -= 2) {
  //   a[k]; if (k == 0) break; if (k - 1 == 0) break;
  // }
  // As after unrolling, with an exit test for each value of k.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> hi = ReducedLength(Asm, a, 1);
    Label<> done(&Asm);
    GOTO_IF_NOT(__ Int32LessThanOrEqual(__ Word32Constant(0), hi), done);
    LoopLabel<Word32> loop(&Asm);
    GOTO(loop, hi);
    BIND_LOOP(loop, k) {
      BoundsCheck(Asm, a, k, 0);
      GOTO_IF(__ Word32Equal(k, __ Word32Constant(0)), done);
      GOTO_IF(__ Word32Equal(__ Word32Sub(k, __ Word32Constant(1)),
                             __ Word32Constant(0)),
              done);
      GOTO(loop, __ Word32Sub(k, __ Word32Constant(2)));
    }
    BIND(done);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

// hi = a.length - 1; if (0 <=s hi) { a[hi]; if (hi != 0 || !nonzero) {
//   for (k = hi - 1; ; k--) { a[k+1]; a[k]; if (k == 0) break; }
// } }
// As for an OCaml loop {for k = Array.length a - 1 downto 0} whose first
// iteration was peeled. With {nonzero}, k starts at hi - 1 >= 0.
template <typename Asm_t>
static void PeeledDecreasingLoop(Asm_t& Asm, V<WasmArrayNullable> a,
                                 bool nonzero) {
  using Test = WasmBoundsCheckEliminationReducerTest;
  V<Word32> hi = Test::ReducedLength(Asm, a, 1);
  Label<> done(&Asm);
  GOTO_IF(__ Int32LessThan(hi, __ Word32Constant(0)), done);
  Test::BoundsCheck(Asm, a, hi, 0);
  if (nonzero) GOTO_IF(__ Word32Equal(hi, __ Word32Constant(0)), done);
  LoopLabel<Word32> loop(&Asm);
  GOTO(loop, __ Word32Sub(hi, __ Word32Constant(1)));
  BIND_LOOP(loop, k) {
    Test::BoundsCheck(Asm, a, k, 1);
    Test::BoundsCheck(Asm, a, k, 0);
    GOTO_IF(__ Word32Equal(k, __ Word32Constant(0)), done);
    GOTO(loop, __ Word32Sub(k, __ Word32Constant(1)));
  }
  BIND(done);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, PeeledDecreasingNotEqualLoop) {
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    PeeledDecreasingLoop(Asm, a, true);
  });
  Run(test);
  // Only the access of the peeled iteration remains.
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       PeeledDecreasingNotEqualLoopFromZero) {
  // Without the test {hi != 0}, k may start at -1.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    PeeledDecreasingLoop(Asm, a, false);
  });
  Run(test);
  ASSERT_GE(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       DecreasingNotEqualLoopUnrolledMissingTest) {
  // if (0 <=s hi) for (k = hi; ; k -= 2) { a[k]; if (k == 0) break; }
  // With an odd hi, k goes past 0.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> hi = ReducedLength(Asm, a, 1);
    DecreasingLoop(
        Asm, hi, 2,
        [&] { return __ Int32LessThanOrEqual(__ Word32Constant(0), hi); },
        [&](V<Word32> k) { return __ Word32Equal(k, __ Word32Constant(0)); },
        [&](V<Word32> k) { BoundsCheck(Asm, a, k, 0); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LoopBoundShowsMinLength) {
  // if (0 <= hi) for (k = 0; ; k++) { a[0]; a[1]; a[k]; if (k == hi) break; }
  // with hi = a.length - 1: the length is at least 1 in the loop, but
  // may be 1.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> hi = ReducedLength(Asm, a, 1);
    NotEqualLoop(
        Asm, hi, 0, 1,
        [&] { return __ Int32LessThanOrEqual(__ Word32Constant(0), hi); },
        [&](V<Word32> k) {
          BoundsCheck(Asm, a, {}, 0);
          BoundsCheck(Asm, a, {}, 1);
          BoundsCheck(Asm, a, k, 0);
        });
  });
  Run(test);
  // Only a[1] is checked.
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, DecreasingLoopShowsMinLength) {
  // if (0 <=s hi) for (k = hi; ; k--) { a[0]; a[1]; a[k];
  //   if (k - 1 <s 0) break; }
  // with hi = a.length - 1.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> hi = ReducedLength(Asm, a, 1);
    DecreasingLoop(
        Asm, hi, 1,
        [&] { return __ Int32LessThanOrEqual(__ Word32Constant(0), hi); },
        [&](V<Word32> k) {
          return __ Int32LessThan(__ Word32Sub(k, __ Word32Constant(1)),
                                  __ Word32Constant(0));
        },
        [&](V<Word32> k) {
          BoundsCheck(Asm, a, {}, 0);
          BoundsCheck(Asm, a, {}, 1);
          BoundsCheck(Asm, a, k, 0);
        });
  });
  Run(test);
  // Only a[1] is checked.
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, FactsOfAnEarlierPhi) {
  // len = a.length; hi = (c ? len : 0) - 1; if (0 <= hi)
  // for (k = 0, j = len - 1; ; k++, j--) {
  //   a[j]; if (k == hi) break; if (j == 0) break;
  // }
  // The bound of k is an alias of a.length - 1 in the loop, which shows
  // that the length is at least 1, so that j starts non-negative.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<Word32> len = __ ArrayLength(a, compiler::kWithNullCheck);
    Label<Word32> merge(&Asm);
    IF (c) {
      GOTO(merge, len);
    } ELSE {
      GOTO(merge, __ Word32Constant(0));
    }
    BIND(merge, length);
    V<Word32> hi = __ Word32Sub(length, __ Word32Constant(1));
    V<Word32> j0 = __ Word32Sub(len, __ Word32Constant(1));
    Label<> done(&Asm);
    GOTO_IF_NOT(__ Int32LessThanOrEqual(__ Word32Constant(0), hi), done);
    LoopLabel<Word32, Word32> loop(&Asm);
    GOTO(loop, __ Word32Constant(0), j0);
    BIND_LOOP(loop, k, j) {
      BoundsCheck(Asm, a, j, 0);
      GOTO_IF(__ Word32Equal(k, hi), done);
      GOTO_IF(__ Word32Equal(j, __ Word32Constant(0)), done);
      GOTO(loop, __ Word32Add(k, __ Word32Constant(1)),
           __ Word32Sub(j, __ Word32Constant(1)));
    }
    BIND(done);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, AliasOfAnEarlierPhi) {
  // hi = (c ? a.length : 0) - 1; if (0 <= hi)
  // for (k = 0, j = hi; ; k++, j--) {
  //   a[j]; if (k == hi) break; if (j == 0) break;
  // }
  // The bound of k is an alias of a.length - 1 in the loop, which j
  // starts at.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    Label<Word32> merge(&Asm);
    IF (c) {
      GOTO(merge, __ ArrayLength(a, compiler::kWithNullCheck));
    } ELSE {
      GOTO(merge, __ Word32Constant(0));
    }
    BIND(merge, length);
    V<Word32> hi = __ Word32Sub(length, __ Word32Constant(1));
    Label<> done(&Asm);
    GOTO_IF_NOT(__ Int32LessThanOrEqual(__ Word32Constant(0), hi), done);
    LoopLabel<Word32, Word32> loop(&Asm);
    GOTO(loop, __ Word32Constant(0), hi);
    BIND_LOOP(loop, k, j) {
      BoundsCheck(Asm, a, j, 0);
      GOTO_IF(__ Word32Equal(k, hi), done);
      GOTO_IF(__ Word32Equal(j, __ Word32Constant(0)), done);
      GOTO(loop, __ Word32Add(k, __ Word32Constant(1)),
           __ Word32Sub(j, __ Word32Constant(1)));
    }
    BIND(done);
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, NarrowedLoopBound) {
  // for (k = 0; zext(k) <u zext(a.length); k++) a[k];
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    using Test = WasmBoundsCheckEliminationReducerTest;
    Loop(
        Asm, 0, 1,
        [&](V<Word32> k) {
          V<Word32> length = __ ArrayLength(a, compiler::kWithNullCheck);
          return __ Uint64LessThanOrEqual(__ ChangeUint32ToUint64(length),
                                          __ ChangeUint32ToUint64(k));
        },
        [&](V<Word32> k) { Test::BoundsCheck(Asm, a, k, 0); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

// The bounds check {index + offset <u length}, for an arbitrary
// {length}.
template <typename Asm_t>
static void BoundsCheckWithLength(Asm_t& Asm, V<Word32> length, V<Word32> index,
                                  int32_t offset) {
  V<Word32> full_index =
      offset == 0 ? index : __ Word32Add(index, __ Word32Constant(offset));
  __ TrapIfNot(__ Uint32LessThan(full_index, length),
               TrapId::kTrapArrayOutOfBounds);
}

// Allocates an array of length {length} (the map is not used by the
// analysis). With load elimination, the length of the new array is then
// {length}.
template <typename Asm_t>
static V<WasmArrayNullable> AllocateArray(Asm_t& Asm, V<WasmArrayNullable> any,
                                          V<Word32> length) {
  return __ WasmAllocateArray(V<Map>::Cast(any), length, &kI32ArrayType,
                              SharedFlag{false});
}

TEST_F(WasmBoundsCheckEliminationReducerTest, AllocationLength) {
  // x = new T[c]; a[i]; if (i <s c - 1) { x[i+1]; x[i]; }
  // c is an array length after the allocation, so the condition covers
  // both accesses.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<WasmArrayNullable> x = AllocateArray(Asm, a, c);
    BoundsCheck(Asm, a, i, 0);
    If(Asm, __ Int32LessThan(i, __ Word32Sub(c, __ Word32Constant(1))), [&] {
      BoundsCheck(Asm, x, i, 1);
      BoundsCheck(Asm, x, i, 0);
    });
  });
  RunWithLoadElimination(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, AllocationLengthRecomputed) {
  // x = new T[i + c]; a[i]; if (i <s (i + c) - 1) { x[i+1]; x[i]; }
  // The length is computed again for the condition.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<WasmArrayNullable> x = AllocateArray(Asm, a, __ Word32Add(i, c));
    BoundsCheck(Asm, a, i, 0);
    V<Word32> length = __ Word32Add(i, c);
    If(Asm, __ Int32LessThan(i, __ Word32Sub(length, __ Word32Constant(1))),
       [&] {
         BoundsCheck(Asm, x, i, 1);
         BoundsCheck(Asm, x, i, 0);
       });
  });
  RunWithLoadElimination(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, AllocationShowsNonNegativeIndex) {
  // new T[c]; if (i <u c) { a[i+2]; a[i]; }
  // c is an array length, so i is non-negative, and the first check of a
  // covers the second one.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    AllocateArray(Asm, a, c);
    If(Asm, __ Uint32LessThan(i, c), [&] {
      BoundsCheck(Asm, a, i, 2);
      BoundsCheck(Asm, a, i, 0);
    });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LengthWithoutAllocation) {
  // if (i <u c) { a[i+2]; a[i]; }
  // Without the allocation, c may be 0xffffffff, and i may be -2.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    If(Asm, __ Uint32LessThan(i, c), [&] {
      BoundsCheck(Asm, a, i, 2);
      BoundsCheck(Asm, a, i, 0);
    });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, AllocationKeepsMinLength) {
  // a[5]; new T[a.length]; if (i <u a.length - 3) a[i+3];
  // The allocation does not lower the min length of a.length, which
  // shows that a.length - 3 does not wrap around.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, {}, 5);
    AllocateArray(Asm, a, __ ArrayLength(a, compiler::kWithNullCheck));
    If(Asm, __ Uint32LessThan(i, ReducedLength(Asm, a, 3)),
       [&] { BoundsCheck(Asm, a, i, 3); });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, AllocationInBranch) {
  // if (i) new T[c]; if (i <u c) { a[i+2]; a[i]; }
  // c is only known to be an array length in the branch.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    If(Asm, i, [&] { AllocateArray(Asm, a, c); });
    If(Asm, __ Uint32LessThan(i, c), [&] {
      BoundsCheck(Asm, a, i, 2);
      BoundsCheck(Asm, a, i, 0);
    });
  });
  Run(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, LengthBeforeAllocation) {
  // a[i]; if (i <s c - 1) { x = new T[c]; x[i+1]; x[i]; }
  // The condition is tested before c is known to be an array length, so
  // only the check of x[i+1] covers the one of x[i] (as i is
  // non-negative).
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    BoundsCheck(Asm, a, i, 0);
    If(Asm, __ Int32LessThan(i, __ Word32Sub(c, __ Word32Constant(1))), [&] {
      V<WasmArrayNullable> x = AllocateArray(Asm, a, c);
      BoundsCheck(Asm, x, i, 1);
      BoundsCheck(Asm, x, i, 0);
    });
  });
  RunWithLoadElimination(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 2u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, AllocationLengthLoop) {
  // x = new T[c]; for (k = 0; k <s c; k++) x[k];
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<WasmArrayNullable> x = AllocateArray(Asm, a, c);
    Loop(
        Asm, 0, 1, [&](V<Word32> k) { return __ Int32LessThanOrEqual(c, k); },
        [&](V<Word32> k) { BoundsCheck(Asm, x, k, 0); });
  });
  RunWithLoadElimination(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, ConstantAllocationLength) {
  // x = new T[3]; x[2]; x[0];
  // The length is known to be 3.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<WasmArrayNullable> x = AllocateArray(Asm, a, __ Word32Constant(3));
    BoundsCheck(Asm, x, {}, 2);
    BoundsCheck(Asm, x, {}, 0);
  });
  RunWithLoadElimination(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest,
       ConstantAllocationLengthTooLarge) {
  // x = new T[3]; x[2]; x[3];
  // The last access is out of bounds.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<WasmArrayNullable> x = AllocateArray(Asm, a, __ Word32Constant(3));
    BoundsCheck(Asm, x, {}, 2);
    BoundsCheck(Asm, x, {}, 3);
  });
  RunWithLoadElimination(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 1u);
}

TEST_F(WasmBoundsCheckEliminationReducerTest, ConstantLengthCondition) {
  // x = new T[3]; if (i <u 3) x[i];
  // The condition is a bounds check against the constant length.
  auto test = CreateTest([](auto& Asm, auto a, auto b, auto i, auto c) {
    V<WasmArrayNullable> x = AllocateArray(Asm, a, __ Word32Constant(3));
    If(Asm, __ Uint32LessThan(i, __ Word32Constant(3)),
       [&] { BoundsCheck(Asm, x, i, 0); });
  });
  RunWithLoadElimination(test);
  ASSERT_EQ(test.CountOp(Opcode::kTrapIf), 0u);
}

#include "src/compiler/turboshaft/undef-assembler-macros.inc"

}  // namespace v8::internal::compiler::turboshaft
