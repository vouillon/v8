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

#include "src/compiler/turboshaft/undef-assembler-macros.inc"

}  // namespace v8::internal::compiler::turboshaft
