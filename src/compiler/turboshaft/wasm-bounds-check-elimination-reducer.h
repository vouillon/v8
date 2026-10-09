// Copyright 2026 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_COMPILER_TURBOSHAFT_WASM_BOUNDS_CHECK_ELIMINATION_REDUCER_H_
#define V8_COMPILER_TURBOSHAFT_WASM_BOUNDS_CHECK_ELIMINATION_REDUCER_H_

#if !V8_ENABLE_WEBASSEMBLY
#error This header should only be included if WebAssembly is enabled.
#endif  // !V8_ENABLE_WEBASSEMBLY

#include <algorithm>
#include <optional>

#include "src/compiler/turboshaft/analyzer-iterator.h"
#include "src/compiler/turboshaft/assembler.h"
#include "src/compiler/turboshaft/graph.h"
#include "src/compiler/turboshaft/loop-finder.h"
#include "src/compiler/turboshaft/snapshot-table.h"
#include "src/compiler/turboshaft/utils.h"
#include "src/compiler/turboshaft/wasm-load-elimination-reducer.h"
#include "src/zone/zone-containers.h"
#include "src/zone/zone.h"

namespace v8::internal::compiler::turboshaft {

#include "src/compiler/turboshaft/define-assembler-macros.inc"

// General overview
//
// The analysis tracks previous bounds checks for array accesses in
// Wasm GC code, considering both explicit traps and branch conditions.
// It handles accesses of the forms:
//      a[offset]
//      a[base +/- offset]
// where `offset` is an integer constant.
//
// Redundant bounds check traps that are covered by earlier checks are
// removed. For example, given the sequence:
//     a[1] = 11;
//     a[0] = 10;
// the bounds check for the second access can be safely eliminated. Similarly,
// for the sequence:
//     a[i] = 11;
//     a[i+2] = 12;
//     a[i+1] = 11;
// the last bounds check can be removed, since the index is guaranteed
// to lie between the previous two indices.
//
// The analysis also tracks which offsets {n} are such that {base + n}
// is known to be non-negative (as a signed 32-bit integer). This does
// not depend on any particular array: since array lengths are less
// than 2^31, any successful bounds check {base + n < a.length}
// implies that {base + n} is non-negative. A single bounds check can
// then cover a whole range of offsets of another array. For example,
// given the sequence:
//     a[i] = 10;
//     b[i+1] = 11;
//     b[i] = 12;
// the first access shows that {i} is non-negative, so the second one
// shows that both {i} and {i+1} are within the bounds of {b}, and the
// last bounds check can be removed.
//
// Conditions of the form {base + n < a.length - r}, for a small
// constant {r}, are also taken into account, once {a.length} is known
// to be at least {r} (otherwise, {a.length - r} may wrap around). Such
// a condition shows that the offsets from {n} to {n + r} are within
// bounds. For example, wasm_of_ocaml stores a header in the first
// element of the arrays it uses for OCaml arrays, so the OCaml bounds
// check of an access {a.(i)} is {i < a.length - 1}, which shows that
// the Wasm access {a[i+1]} is within bounds, as long as some previous
// access to {a} showed that its length is at least 1. A non-strict
// condition {base + n <= a.length - r} is the same as
// {base + n < a.length - (r - 1)} when {r >= 1}.
//
// The analysis also tracks a lower bound on the length of each array.
// It shows that the constant indices below it are within bounds, and
// that {a.length - r} does not wrap around when it is at least {r}. It
// comes from bounds checks, from conditions {n <= a.length - r} for a
// constant {n}, and from tests {a.length - r != n} when the length is
// known to be at least {n + r} (for instance, {a.length != 0} shows that
// the length is at least 1). A constant length is its own lower bound.
//
// Signed conditions {base + n <s a.length - r}, as produced by languages
// with signed integers (Java, Kotlin, Dart...), are taken into account
// when {base + n} is known to be non-negative: the condition then shows
// that {a.length - r} is positive, and that {base + n <u a.length - r}.
// Since array lengths are less than 2^30, {base + n} is non-negative
// when some {base + m} is within bounds and {n - m} is at most 2^30.
// Languages with 64-bit integers may compare a 64-bit index with the
// zero-extended array length: {X <u zext(l)} (or {X <=u zext(l)}) shows
// that {X} is less than 2^32, so that it is the same comparison of
// {trunc(X)} with {l}. Such comparisons are taken into account as the
// latter, reading {X} as its truncation, which is {x} for {zext(x)} or
// {sext(x)}, and {trunc(Y) + n} for {Y + n}, so that {zext(i) + 1 <u
// zext(a.length)} is the bounds check {i + 1 < a.length}. They only
// show facts, and are never eliminated: when {X} is 2^32 or more, the
// comparison fails while {trunc(X) <u l} may hold.
// Conditions of branches and traps combining other conditions are also
// taken into account: when {x | y} does not hold (is 0), neither {x} nor
// {y} holds, when {x & y} holds (is not 0), neither {x} nor {y} is 0 (so
// that comparisons, whose values are 0 or 1, hold), and {x == 0} holds
// when {x} does not.
//
// Facts are about values: a bounds check {base + n < l} depends on the
// value {l}, not on the array whose length it is. Bounds checks are
// thus grouped by the canonical value of the length (see
// {CanonicalValue}), which is the same for all the ArrayLength
// operations of an array, and for arrays that load elimination knows to
// have the same length, as an array allocated with the length of
// another one: in {b = new T[a.length]; for (i = 0; i < a.length; i++)
// b[i] = a[i];}, the condition covers the accesses to both arrays. A
// condition {i < n} is also recognized as a bounds check after an
// allocation of length {n}, as in {b = new T[n]; for (i = 0; i < n; i++)
// b[i] = ...}.
// Bases are canonical values too, so that the same value computed twice
// (as the index of an OCaml bounds check and of the access it protects)
// is the same base, and nested additions of constants, as produced by
// loop unrolling, are folded into a single offset. An index
// {trunc(Y + n)}, as computed by languages with 64-bit integers, is
// {trunc(Y) + n} when an operation computing {trunc(Y)} already exists
// (for the index of a previous access {a[trunc(Y)]}).
//
// Loop variables are shown to be non-negative by induction. A loop phi
// {i = phi(c0, i + c)}, with constants {c0 >= 0} and {c > 0}, is
// non-negative in the whole loop (and after it) if its back edge is only
// taken when {i < x} or {i <= x}, for a bound {x} small enough that
// {i + c} cannot overflow: an array length minus a constant (for an
// unsigned comparison, the array length itself), or a constant at most
// {2^31 - c}. Loops written {for (i = c0; i != x; i++)}, as OCaml {for}
// loops are compiled, are handled too, when the loop is only entered if
// {c0 <= x}, for a loop-invariant {x}: {i} is then at most {x} in the
// loop, and when {x} is an array length minus {r >= 1}, the offsets 0
// to {r - 1} of {i} are within bounds. After loop unrolling, the step is
// {c > 1} and the back edge is taken when {i + k != x} for all {k} from
// 0 to {c - 1}, which works the same way. In the unrolled copies, a
// test {i + k != a.length - r} then shows that {i + k + r} is within
// bounds when {i + k + r - 1} is. When {x} is a phi plus a constant, the
// constant inputs of the phi for which the loop is not entered are
// ruled out: if a single input remains, {x} is this input plus the
// constant in the loop. For instance, with {x = phi(a.length, 0) - 1}
// (the length of an array that may be empty, minus 1), the loop is not
// entered when {x} is -1, so {x} is {a.length - 1} in the loop, which
// is recorded in {known_length_aliases_}.
//
// Decreasing loop phis {i = phi(i0, i - c)} are non-negative in the
// whole loop when {i0} is non-negative when entering the loop (which may
// be shown by conditions on {x} when {i0} is {x - k}, as for a loop whose
// first iteration was peeled), and the back edge is only taken when
// {i - c} is non-negative: when {lo <= i + d} for constants with
// {lo - d >= c}, as for an exit test {i >= 0} after
// decrementing {i}, or {i > 0} before, or when {i + k != y} for all {k}
// below {c}, for a constant {y} with {0 <= y <= i0} when entering the
// loop, as for an OCaml loop {for i = i0 downto y}. As {i} then
// decreases without wrapping around, it is at most {i0}, and when {i0}
// is {a.length - r} with {r >= 1}, the offsets 0 to {r - 1} of {i} are
// within bounds, as in {for (i = a.length - 1; i >= 0; i--)}.
//
// Facts are recorded at the point where a condition is known to hold:
// right after a trap, or at the start of a branch target, at loop
// headers for facts shown by induction (see {ProcessLoopHeader}), and
// right after an allocation for its length (see {RecordKnownLength}). They
// are only used in blocks dominated by that point (see {BeginBlock}),
// where they still hold (see {CanonicalValue}, and {LengthAliasMap} for
// length aliases).
//
// Limits
//
// Array lengths are less than {kMaxArrayLength} = 2^30 (as checked by a
// static_assert on {WasmArray::MaxLength}):
// - Since they are less than 2^31, an index within bounds is
//   non-negative as a signed integer, and ranges of offsets that span
//   less than 2^31 values can be compared in modular arithmetic (see
//   {OffsetRange}).
// - Since they are less than 2^30, a non-negative index at most an array
//   length can be incremented by a step of up to 2^30 without
//   overflowing, which loop induction relies on, and an index at most
//   2^30 above an index within bounds is non-negative.
// The constants {r} in conditions {x < a.length - r} are at most
// {kMaxReduction} = 2^16. This limit is arbitrary, since only small
// constants are useful. It keeps the range of offsets from {n} to
// {n + r} shown by such a condition valid, and it ensures that
// {a.length - r} is negative (as a signed integer) rather than a large
// positive value when {a.length < r}.
//
// A value {n} is also known to be an array length after an allocation
// {WasmAllocateArray} of length {n} (as for {array.new} or
// {array.new_fixed}), which traps when {n} exceeds the maximum length of
// the array type (the check is emitted when the operation is lowered)
// ([allocation-length]): {n} is then at most {WasmArray::MaxLength}, and
// so less than 2^30, which is all that the analysis assumes about array
// lengths. For the same reason, a constant {c < 2^30} is used as an
// array length, whose min length is {c} ([constant-length]): load
// elimination replaces the length of an array allocated with a constant
// length by this constant, which is then the key of its bounds checks
// (see {LengthKey}).

// Key for grouping bounds checks by base and array length. The length
// is the canonical value of an array length (see {LengthKey}): a bounds
// check {base + n < l} only depends on the value {l}, which may be the
// length of several arrays.
struct BoundsCheckKey {
  OpIndex base;  // Invalid means that we do not have a base.
  OpIndex length;

  bool operator==(const BoundsCheckKey& other) const = default;

  template <typename H>
  friend H AbslHashValue(H h, const BoundsCheckKey& key) {
    return H::combine(std::move(h), key.base, key.length);
  }
};

// Identifies a specific bounds check by its key (base + length) and
// offset.
struct BoundsCheck {
  BoundsCheckKey key;
  uint32_t offset;

  bool operator==(const BoundsCheck& other) const = default;
};

// Represents a contiguous range of 32-bit offsets [lower, upper],
// correctly handling modular arithmetic.
//
// This ensures that bounds checks are properly handled even when the
// range wraps around the 32-bit modulus. Allows checking whether a
// given offset is inside the range, below it, above it, or otherwise
// out of range.
//
// Correctness is supported by Z3 proofs (see below class definition).
class OffsetRange {
 public:
  OffsetRange(uint32_t lower, uint32_t upper) : lower_(lower), upper_(upper) {
    DCHECK(IsValid());
  }

  explicit OffsetRange(uint32_t single) : OffsetRange(single, single) {}

  // Returns the inclusive lower bound.
  uint32_t lower() const { return lower_; }

  // Returns the inclusive upper bound.
  uint32_t upper() const { return upper_; }

  // Classifies a given offset relative to the range.
  enum class RelativePosition {
    kInside,     // Offset is within [lower, upper]
    kBelow,      // Offset is less than lower
    kAbove,      // Offset is greater than upper
    kOutOfRange  // Offset is invalid for this range
  };

  RelativePosition Classify(uint32_t offset) const {
    if (IsValidRange(lower_, offset) && IsValidRange(offset, upper_)) {
      return RelativePosition::kInside;
    }
    if (IsValidRange(upper_, offset) && IsValidRange(lower_, offset)) {
      return RelativePosition::kAbove;
    }
    if (IsValidRange(offset, lower_) && IsValidRange(offset, upper_)) {
      return RelativePosition::kBelow;
    }
    return RelativePosition::kOutOfRange;
  }

  bool Contains(uint32_t offset) const {
    return Classify(offset) == RelativePosition::kInside;
  }

  // Returns the smallest valid range containing both {this} and
  // {other}, if there is one. It starts at the lower bound of one of
  // the two ranges. It contains the bounds of both ranges, and so all
  // their offsets ([range-containment]).
  std::optional<OffsetRange> Hull(const OffsetRange& other) const {
    std::optional<OffsetRange> result;
    std::pair<const OffsetRange*, const OffsetRange*> candidates[] = {
        {this, &other}, {&other, this}};
    for (auto [first, second] : candidates) {
      // Offsets relative to the lower bound of {first}.
      uint32_t second_lower = second->lower_ - first->lower_;
      uint32_t second_upper = second->upper_ - first->lower_;
      // {second} should not wrap around the lower bound of {first}.
      if (second_lower > second_upper) continue;
      uint32_t size = std::max(first->upper_ - first->lower_, second_upper);
      // The hull would span 2^31 values or more, and would not be a valid
      // range. This cannot happen for two ranges of offsets within the
      // bounds of the same array for the same base (as array lengths, and
      // constants used as lengths, are less than 2^30, see
      // [constant-length]), nor for two ranges of non-negative offsets of
      // the same base. But it can in unreachable code, where facts may
      // contradict each other, so this check is needed.
      if (static_cast<int32_t>(size) < 0) continue;
      if (!result.has_value() || size < result->upper_ - result->lower_) {
        result = OffsetRange(first->lower_, first->lower_ + size);
      }
    }
    return result;
  }

  // Checks whether the range [lower, upper] is valid in modular
  // 32-bit arithmetic. Valid means the range spans at most 2^31
  // values, ensuring {upper} hasn't wrapped around past {lower} in
  // the circular address space.
  static bool IsValidRange(uint32_t lower, uint32_t upper) {
    return static_cast<int32_t>(upper - lower) >= 0;
  }

  bool operator==(const OffsetRange& other) const = default;

 private:
  bool IsValid() const { return IsValidRange(lower_, upper_); }
  uint32_t lower_;
  uint32_t upper_;
};

// The correctness of the analysis is supported by the following Z3
// proofs, which are referred to by their names in brackets (except
// [allocation-length] and [constant-length], which refer to assumptions
// stated in "Limits" above). Preceded by
// the shared definitions below, each one is a complete SMT-LIB query: it
// asserts some assumptions and the negation of a conclusion, and Z3
// reports it unsatisfiable (unsat). The shared definitions are:
//
//   (define-fun valid-range ((n1 (_ BitVec 32)) (n2 (_ BitVec 32))) Bool
//     (bvsge (bvsub n2 n1) #x00000000))
//   (define-fun non-negative ((v (_ BitVec 32))) Bool
//     (bvsge v #x00000000))
//
// Array lengths {l} are assumed to be non-negative as signed integers
// (less than 2^31), or less than 2^30 where needed (see "Limits" above).
//
// [covered] A bounds check is redundant when its offset is covered by
// a range of offsets within bounds:
//
//   (declare-const l (_ BitVec 32))
//   (declare-const x (_ BitVec 32))
//   (declare-const n (_ BitVec 32))
//   (declare-const n1 (_ BitVec 32))
//   (declare-const n2 (_ BitVec 32))
//   ; Assumptions
//   (assert (bvsge l #x00000000)) ; the array length l not too large
//   (assert (bvult (bvadd x n1) l)) ; x + n1 within bounds
//   (assert (bvult (bvadd x n2) l)) ; x + n2 within bounds
//   (assert (valid-range n1 n2))
//   ; We check that n is covered by the range n1..n2
//   (assert (valid-range n1 n))
//   (assert (valid-range n n2))
//   ; Then x + n is within bounds
//   (assert (not (bvult (bvadd x n) l)))
//   (check-sat)
//
// As a consequence, the hull of two ranges of offsets within bounds is
// also within bounds, provided that it is a valid range. It is valid
// when both ranges are within bounds for the same {x}, since their
// indices are then all within [0, l).
//
// [range-containment] A valid range containing both bounds of a valid
// range contains all its offsets. So the hull of two ranges contains
// both of them, and a range only grows when replaced by its hull with
// another one:
//
//   (declare-const n (_ BitVec 32))
//   (declare-const n1 (_ BitVec 32))
//   (declare-const n2 (_ BitVec 32))
//   (declare-const m1 (_ BitVec 32))
//   (declare-const m2 (_ BitVec 32))
//   (assert (valid-range n1 n2))
//   ; n is in the range n1..n2
//   (assert (valid-range n1 n))
//   (assert (valid-range n n2))
//   ; Both bounds of n1..n2 are in the range m1..m2
//   (assert (valid-range m1 m2))
//   (assert (valid-range m1 n1))
//   (assert (valid-range n1 m2))
//   (assert (valid-range m1 n2))
//   (assert (valid-range n2 m2))
//   ; Then n is in the range m1..m2
//   (assert (not (and (valid-range m1 n) (valid-range n m2))))
//   (check-sat)
//
// [check-non-negative] A bounds check shows that the index is
// non-negative:
//
//   (declare-const l (_ BitVec 32))
//   (declare-const x (_ BitVec 32))
//   (declare-const n (_ BitVec 32))
//   (assert (bvsge l #x00000000))
//   (assert (bvult (bvadd x n) l))
//   (assert (not (non-negative (bvadd x n))))
//   (check-sat)
//
// [non-negative-index] So do comparisons with small enough bounds, or
// large enough ones for signed comparisons: {x <u b} with {b <= 2^31},
// {x <=u b} with {b < 2^31} (in particular, {b} can be an array length
// minus a constant that does not wrap around), {b <s x} with {b >= -1}
// and {b <=s x} with {b >= 0}:
//
//   (declare-const x (_ BitVec 32))
//   (declare-const b (_ BitVec 32))
//   (assert (or (and (bvult x b) (bvule b #x80000000))
//               (and (bvule x b) (bvule b #x7fffffff))
//               (and (bvslt b x) (bvsge b #xffffffff))
//               (and (bvsle b x) (bvsge b #x00000000))))
//   (assert (not (non-negative x)))
//   (check-sat)
//
// [non-negative-range] Non-negative offsets form ranges:
//
//   (declare-const x (_ BitVec 32))
//   (declare-const n (_ BitVec 32))
//   (declare-const n1 (_ BitVec 32))
//   (declare-const n2 (_ BitVec 32))
//   (assert (non-negative (bvadd x n1)))
//   (assert (non-negative (bvadd x n2)))
//   (assert (valid-range n1 n2))
//   (assert (valid-range n1 n))
//   (assert (valid-range n n2))
//   (assert (not (non-negative (bvadd x n))))
//   (check-sat)
//
// [non-negative-and-check] If x + n is non-negative, a bounds check at
// offset m covers the whole range n..m:
//
//   (declare-const l (_ BitVec 32))
//   (declare-const x (_ BitVec 32))
//   (declare-const n (_ BitVec 32))
//   (declare-const m (_ BitVec 32))
//   (declare-const k (_ BitVec 32))
//   (assert (bvsge l #x00000000))
//   (assert (non-negative (bvadd x n)))
//   (assert (bvult (bvadd x m) l))
//   (assert (valid-range n m))
//   (assert (valid-range n k))
//   (assert (valid-range k m))
//   (assert (not (bvult (bvadd x k) l)))
//   (check-sat)
//
// [reduced-length] If the array length l is at least r, then
// x + n < l - r shows that all the offsets from n to n + r are within
// bounds (without this assumption on l, it does not: l - r may wrap
// around):
//
//   (declare-const l (_ BitVec 32))
//   (declare-const x (_ BitVec 32))
//   (declare-const n (_ BitVec 32))
//   (declare-const r (_ BitVec 32))
//   (declare-const k (_ BitVec 32))
//   (assert (bvsge l #x00000000))
//   (assert (bvuge l r))
//   (assert (bvult (bvadd x n) (bvsub l r)))
//   (assert (valid-range n k))
//   (assert (valid-range k (bvadd n r)))
//   (assert (not (bvult (bvadd x k) l)))
//   (check-sat)
//
// [reduced-length-min-length] It also shows that x + n is
// non-negative, and that l is at least r + 1:
//
//   (declare-const l (_ BitVec 32))
//   (declare-const x (_ BitVec 32))
//   (declare-const n (_ BitVec 32))
//   (declare-const r (_ BitVec 32))
//   (assert (bvsge l #x00000000))
//   (assert (bvuge l r))
//   (assert (bvult (bvadd x n) (bvsub l r)))
//   (assert (not (and (non-negative (bvadd x n))
//                     (bvuge l (bvadd r #x00000001)))))
//   (check-sat)
//
// [constant-index-min-length] For a constant index n, less than 2^31,
// l is at least n + r + 1:
//
//   (declare-const l (_ BitVec 32))
//   (declare-const n (_ BitVec 32))
//   (declare-const r (_ BitVec 32))
//   (assert (bvsge l #x00000000))
//   (assert (bvuge l r))
//   (assert (non-negative n))
//   (assert (bvult n (bvsub l r)))
//   (assert (not (bvuge l (bvadd (bvadd n r) #x00000001))))
//   (check-sat)
//
// [min-length-constant-index] A constant index below a lower bound of
// the length is within bounds:
//
//   (declare-const l (_ BitVec 32))
//   (declare-const m (_ BitVec 32))
//   (declare-const n (_ BitVec 32))
//   (assert (bvsge l #x00000000))
//   (assert (bvuge l m))
//   (assert (bvult n m))
//   (assert (not (bvult n l)))
//   (check-sat)
//
// [signed-check] A signed condition {x + n <s l - r} with {x + n}
// non-negative (and a small constant {r}) shows the unsigned condition,
// and that {l - r} does not wrap around:
//
//   (declare-const l (_ BitVec 32))
//   (declare-const x (_ BitVec 32))
//   (declare-const n (_ BitVec 32))
//   (declare-const r (_ BitVec 32))
//   (assert (bvsge l #x00000000))
//   (assert (bvule r #x00010000))
//   (assert (non-negative (bvadd x n)))
//   (assert (bvslt (bvadd x n) (bvsub l r)))
//   (assert (not (and (bvult (bvadd x n) (bvsub l r))
//                     (bvuge l (bvadd r #x00000001)))))
//   (check-sat)
//
// [non-negative-near-bounds] With array lengths less than 2^30, if
// {x + m} is within bounds, then {x + m + d} is non-negative for {d} at
// most 2^30:
//
//   (declare-const l (_ BitVec 32))
//   (declare-const x (_ BitVec 32))
//   (declare-const m (_ BitVec 32))
//   (declare-const d (_ BitVec 32))
//   (assert (bvult l #x40000000))
//   (assert (bvult (bvadd x m) l))
//   (assert (bvule d #x40000000))
//   (assert (not (non-negative (bvadd (bvadd x m) d))))
//   (check-sat)
//
// [not-equal-length] A test {x + m + 1 != l} extends a known range
// ending at {x + m}:
//
//   (declare-const x (_ BitVec 32))
//   (declare-const m (_ BitVec 32))
//   (declare-const l (_ BitVec 32))
//   (assert (bvsge l #x00000000))
//   (assert (bvult (bvadd x m) l))
//   (assert (not (= (bvadd (bvadd x m) #x00000001) l)))
//   (assert (not (bvult (bvadd (bvadd x m) #x00000001) l)))
//   (check-sat)
//
// [non-strict-length] A non-strict condition {x <= l - r} implies
// {x < l - (r - 1)} for {r >= 1}, when the comparison is signed ({s}),
// or when {l} is at least {r}:
//
//   (declare-const l (_ BitVec 32))
//   (declare-const x (_ BitVec 32))
//   (declare-const r (_ BitVec 32))
//   (declare-const s Bool)
//   (assert (bvsge l #x00000000))
//   (assert (bvuge r #x00000001))
//   (assert (bvule r #x00010000))
//   (assert (or (and s (bvsle x (bvsub l r)))
//               (and (not s) (bvuge l r) (bvule x (bvsub l r)))))
//   (assert (not (ite s (bvslt x (bvsub l (bvsub r #x00000001)))
//                       (bvult x (bvsub l (bvsub r #x00000001))))))
//   (check-sat)
//
// [non-strict-constant] For a constant {n >= 1}, {n <= y} implies
// {n - 1 < y}, signed ({s}) or unsigned:
//
//   (declare-const n (_ BitVec 32))
//   (declare-const y (_ BitVec 32))
//   (declare-const s Bool)
//   (assert (or (and s (bvsge n #x00000001) (bvsle n y))
//               (and (not s) (bvuge n #x00000001) (bvule n y))))
//   (assert (not (ite s (bvslt (bvsub n #x00000001) y)
//                       (bvult (bvsub n #x00000001) y))))
//   (check-sat)
//
// [not-equal-min-length] A test {l - r != n}, when {l} is known to be at
// least {n + r}, shows that {l} is at least {n + r + 1}:
//
//   (declare-const l (_ BitVec 32))
//   (declare-const n (_ BitVec 32))
//   (declare-const r (_ BitVec 32))
//   (assert (bvsge l #x00000000))
//   (assert (bvule r #x00010000))
//   (assert (bvule n #x7fffffff))
//   (assert (bvule (bvadd n r) #x7fffffff))
//   (assert (bvuge l (bvadd n r)))
//   (assert (not (= (bvsub l r) n)))
//   (assert (not (bvuge l (bvadd (bvadd n r) #x00000001))))
//   (check-sat)
//
// [induction-constant] Induction for loop phis {i = phi(c0, i + c)}: if
// {i} is non-negative and the back edge is only taken when {i < x} (or
// {i <= x}), then {i + c} is non-negative for a constant bound {x} at
// most {2^31 - c} (at most {2^31 - 1 - c} for {i <= x}), with a signed
// or unsigned comparison. For a signed comparison, a negative {x} is
// allowed: the back edge is then never taken.
//
//   (declare-const i (_ BitVec 32))
//   (declare-const x (_ BitVec 32))
//   (declare-const c (_ BitVec 32))
//   (assert (non-negative i))
//   (assert (bvugt c #x00000000))
//   (assert (bvule c #x40000000))
//   (assert (or (and (bvult i x) (bvule x (bvsub #x80000000 c)))
//               (and (bvslt i x) (bvsle x (bvsub #x80000000 c)))
//               (and (bvule i x) (bvule x (bvsub #x7fffffff c)))
//               (and (bvsle i x) (bvsle x (bvsub #x7fffffff c)))))
//   (assert (not (non-negative (bvadd i c))))
//   (check-sat)
//
// [induction-length] The same holds for a bound {l - r}, where {l} is
// an array length, with a signed comparison, and for the bound {l}
// with an unsigned comparison ({i < x} implies {i <= x}):
//
//   (declare-const i (_ BitVec 32))
//   (declare-const l (_ BitVec 32))
//   (declare-const r (_ BitVec 32))
//   (declare-const c (_ BitVec 32))
//   (assert (bvult l #x40000000))
//   (assert (bvule r #x00010000))
//   (assert (non-negative i))
//   (assert (or (bvsle i (bvsub l r)) (bvule i l)))
//   (assert (bvugt c #x00000000))
//   (assert (bvule c #x40000000))
//   (assert (not (non-negative (bvadd i c))))
//   (check-sat)
//
// [induction-not-equal] For {i != x} loops, {0 <= i <= x} is preserved
// by each increment of 1, which cannot overflow (and so by an unrolled
// step {c}, with {i + k != x} for all {k} below {c}):
//
//   (declare-const i (_ BitVec 32))
//   (declare-const x (_ BitVec 32))
//   (assert (non-negative i))
//   (assert (bvsle i x))
//   (assert (not (= i x)))
//   (assert (not (and (non-negative (bvadd i #x00000001))
//                     (bvsle (bvadd i #x00000001) x))))
//   (check-sat)
//
// [induction-not-equal-bounds] and {0 <= i <= l - r} shows that the
// offsets 0 to {r - 1} of {i} are within bounds:
//
//   (declare-const i (_ BitVec 32))
//   (declare-const l (_ BitVec 32))
//   (declare-const r (_ BitVec 32))
//   (declare-const k (_ BitVec 32))
//   (assert (bvult l #x40000000))
//   (assert (bvuge r #x00000001))
//   (assert (bvule r #x00010000))
//   (assert (non-negative i))
//   (assert (bvsle i (bvsub l r)))
//   (assert (bvult k r))
//   (assert (not (bvult (bvadd i k) l)))
//   (check-sat)
//
// [induction-decreasing] Induction for decreasing loop phis
// {i = phi(i0, i - c)}: if {0 <= i <= i0} and the back edge is only
// taken when {lo <= i + d} (or {lo < i + d}), signed, with
// {-2^30 <= d <= 0} and {lo - d >= c} (or {lo + 1 - d >= c}) as
// integers, then {0 <= i - c <= i0}:
//
//   (declare-const i (_ BitVec 32))
//   (declare-const i0 (_ BitVec 32))
//   (declare-const c (_ BitVec 32))
//   (declare-const d (_ BitVec 32))
//   (declare-const lo (_ BitVec 32))
//   (assert (non-negative i))
//   (assert (bvsle i i0))
//   (assert (bvugt c #x00000000))
//   (assert (bvule c #x40000000))
//   (assert (bvsge d #xc0000000))
//   (assert (bvsle d #x00000000))
//   (assert (or (and (bvsle lo (bvadd i d))
//                    (bvsge (bvsub ((_ sign_extend 32) lo)
//                                  ((_ sign_extend 32) d))
//                           ((_ zero_extend 32) c)))
//               (and (bvslt lo (bvadd i d))
//                    (bvsge (bvsub (bvadd ((_ sign_extend 32) lo)
//                                         #x0000000000000001)
//                                  ((_ sign_extend 32) d))
//                           ((_ zero_extend 32) c)))))
//   (assert (not (and (non-negative (bvsub i c)) (bvsle (bvsub i c) i0))))
//   (check-sat)
//
// [induction-decreasing-not-equal] For decreasing {i != y} loops,
// {0 <= y <= i <= i0} is preserved by a decrement {c} when the back edge
// is only taken if {i != y + k} for all {k} below {c}, that is, if
// {i - y} is not below {c}:
//
//   (declare-const i (_ BitVec 32))
//   (declare-const i0 (_ BitVec 32))
//   (declare-const y (_ BitVec 32))
//   (declare-const c (_ BitVec 32))
//   (assert (non-negative y))
//   (assert (bvsle y i))
//   (assert (bvsle i i0))
//   (assert (bvugt c #x00000000))
//   (assert (bvule c #x40000000))
//   (assert (not (bvult (bvsub i y) c)))
//   (assert (not (and (bvsle y (bvsub i c)) (bvsle (bvsub i c) i0))))
//   (check-sat)
//
// [entry-lower-bound-offset] If {x >= lo} (signed) and
// {-2^30 <= o < 0} with {lo + o >= 0} as integers, then {x + o} does
// not wrap around, and is at least {lo + o}:
//
//   (declare-const x (_ BitVec 32))
//   (declare-const lo (_ BitVec 32))
//   (declare-const o (_ BitVec 32))
//   (assert (bvsge x lo))
//   (assert (bvslt o #x00000000))
//   (assert (bvsge o #xc0000000))
//   (assert (bvsge (bvadd ((_ sign_extend 32) lo) ((_ sign_extend 32) o))
//                  #x0000000000000000))
//   (assert (not (and (non-negative (bvadd x o))
//                     (bvsge (bvadd x o) (bvadd lo o)))))
//   (check-sat)
//
// [truncated-addition] The truncation to 32 bits of a 64-bit addition
// (or subtraction) is the addition of the truncations:
//
//   (declare-const x (_ BitVec 64))
//   (declare-const c (_ BitVec 64))
//   (assert (not (and (= ((_ extract 31 0) (bvadd x c))
//                        (bvadd ((_ extract 31 0) x) ((_ extract 31 0) c)))
//                     (= ((_ extract 31 0) (bvsub x c))
//                        (bvsub ((_ extract 31 0) x) ((_ extract 31 0) c))))))
//   (check-sat)
//
// [narrowed-comparison] A 64-bit unsigned comparison of {X} with a
// zero-extended {y} is the 32-bit comparison of the truncation of {X}
// with {y}, and the truncation of an extension of {x} is {x}:
//
//   (declare-const X (_ BitVec 64))
//   (declare-const y (_ BitVec 32))
//   (declare-const x (_ BitVec 32))
//   (assert (not (and (=> (bvult X ((_ zero_extend 32) y))
//                         (bvult ((_ extract 31 0) X) y))
//                     (=> (bvule X ((_ zero_extend 32) y))
//                         (bvule ((_ extract 31 0) X) y))
//                     (= ((_ extract 31 0) ((_ zero_extend 32) x)) x)
//                     (= ((_ extract 31 0) ((_ sign_extend 32) x)) x))))
//   (check-sat)

// Maps keys of type {K} to values of type {V}. Supports snapshotting
// for control flow merge points.
template <typename K, typename V>
class KeyedSnapshotTable : public SnapshotTable<std::optional<V>> {
 public:
  using Base = SnapshotTable<std::optional<V>>;

  explicit KeyedSnapshotTable(Zone* zone) : Base(zone), key_table_(zone) {}

  std::optional<V> Get(const K& key) const {
    auto it = key_table_.find(key);
    if (it == key_table_.end()) return std::nullopt;
    return Base::Get(it->second);
  }

  void Set(const K& key, const V& value) {
    auto [it, inserted] = key_table_.try_emplace(key);
    if (inserted) {
      // The initial value of a key is visible in all snapshots, so we
      // create keys without a value, and then set the value in the
      // current snapshot only.
      it->second = Base::NewKey(std::nullopt);
    }
    Base::Set(it->second, value);
  }

 private:
  ZoneAbslFlatHashMap<K, typename Base::Key> key_table_;
};

// An array length minus a constant, {length - reduction}, where
// {length} is an ArrayLength operation, a value known to be the length
// of an array (see {IsKnownLength}), or a constant below 2^30 (with a
// reduction of 0).
struct ReducedLength {
  OpIndex length;
  uint32_t reduction;

  bool operator==(const ReducedLength& other) const = default;
};

// Maps (base, length) pairs to their known safe offset ranges.
using BoundsCheckMap = KeyedSnapshotTable<BoundsCheckKey, OffsetRange>;

// Maps bases to the range of offsets {n} such that {base + n} is
// known to be non-negative.
using NonNegativeOffsetMap = KeyedSnapshotTable<OpIndex, OffsetRange>;

// Maps array lengths (by canonical value) to a lower bound. A length
// has an entry, possibly 0, when it is known to be an array length,
// for instance the length of an allocated array (see {IsKnownLength}).
using MinLengthMap = KeyedSnapshotTable<OpIndex, uint32_t>;

// Maps values to an array length minus a constant that they are equal
// to, when this is only known in part of the graph (see
// {ProcessLoopHeader}). The ArrayLength operation may be an input of a
// merge phi, and then does not dominate the uses of the alias. Facts
// learnt from it (bounds checks and min lengths of the length) are
// still sound: they hold when they are recorded (see
// {TryResolveMergeBound}), and they remain true in the blocks dominated
// by the point where they are recorded, although their leaves (see
// {CanonicalValue}) do not dominate it. These leaves are not defined in
// blocks dominated by the merge block of the alias, which dominates
// that point, so they are not computed again before such blocks are
// reached.
using LengthAliasMap = KeyedSnapshotTable<OpIndex, ReducedLength>;

class WasmBoundsCheckEliminationAnalyzer {
 public:
  // {load_elimination} is optional. When provided, values replaced by
  // load elimination are considered the same as their replacement.
  WasmBoundsCheckEliminationAnalyzer(
      const Graph& graph, Zone* phase_zone,
      const WasmLoadEliminationAnalyzer* load_elimination)
      : graph_(graph),
        phase_zone_(phase_zone),
        load_elimination_(load_elimination),
        redundant_traps_(phase_zone),
        canonical_values_(phase_zone),
        values_by_structure_(phase_zone),
        truncations_(phase_zone),
        known_bounds_checks_(phase_zone),
        known_non_negative_offsets_(phase_zone),
        known_min_lengths_(phase_zone),
        known_length_aliases_(phase_zone),
        block_to_snapshot_mapping_(graph.block_count(), phase_zone),
        predecessor_bounds_check_snapshots_(phase_zone),
        predecessor_non_negative_snapshots_(phase_zone),
        predecessor_min_length_snapshots_(phase_zone),
        predecessor_length_alias_snapshots_(phase_zone) {}

  void Run() {
    // On wasm_of_ocaml programs, about one operation in 10 gets a canonical
    // value, and one in 30 a structural key: reserving that much avoids
    // most of the rehashing as the maps grow.
    canonical_values_.reserve(graph_.op_id_count() / 8);
    values_by_structure_.reserve(graph_.op_id_count() / 32);
    LoopFinder loop_finder(phase_zone_, &graph_, LoopFinder::Config{});
    AnalyzerIterator iterator(phase_zone_, graph_, loop_finder);

    while (iterator.HasNext()) {
      const Block* block = iterator.Next();
      ProcessBlock(*block);
    }
  }

  // Whether {graph} contains any array bounds check. If not, there is no
  // need to run the analysis.
  static bool HasArrayBoundsChecks(const Graph& graph) {
    for (const Operation& op : graph.AllOperations()) {
      const TrapIfOp* trap_if = op.TryCast<TrapIfOp>();
      if (trap_if && trap_if->trap_id == TrapId::kTrapArrayOutOfBounds) {
        return true;
      }
    }
    return false;
  }

  bool IsRedundantTrap(OpIndex index) const {
    return redundant_traps_.contains(index);
  }

 private:
  // Snapshots of the tables at the end of a block.
  struct Snapshot {
    BoundsCheckMap::Snapshot bounds_checks;
    NonNegativeOffsetMap::Snapshot non_negative_offsets;
    MinLengthMap::Snapshot min_lengths;
    // Only once some length alias has been recorded.
    std::optional<LengthAliasMap::Snapshot> length_aliases;
  };

  // A condition that holds, normalized to {left kind right}, where
  // {kind} is a less-than comparison ({a < b} that does not hold is
  // {b <= a}), or to {left == right}, or {left != right} if
  // {not_equal}.
  struct Relation {
    ComparisonOp::Kind kind;
    bool not_equal;
    OpIndex left;
    OpIndex right;
    // A 64-bit comparison {left < zext(right)} or {left <= zext(right)}
    // (unsigned), read as the 32-bit comparison of the truncation of
    // {left} (a 64-bit value) with {right} (see {Normalize}).
    bool narrowed = false;

    bool is_signed() const {
      return kind == ComparisonOp::Kind::kSignedLessThan ||
             kind == ComparisonOp::Kind::kSignedLessThanOrEqual;
    }
    bool is_strict() const {
      return kind == ComparisonOp::Kind::kSignedLessThan ||
             kind == ComparisonOp::Kind::kUnsignedLessThan;
    }
  };

  // A condition {index < a.length - reduction}, where {reduction} is a
  // constant (usually 0), and the comparison is signed or unsigned.
  struct BoundsCheckCondition {
    BoundsCheck bounds_check;
    uint32_t reduction;
    bool is_signed;
  };

  // An index decomposed as {base + offset}. {base} is the canonical
  // value of {base_value} (see {CanonicalValue}), an operation computing
  // the base, which is only used to recognize array lengths, and is not
  // necessarily available where the index is (see {ExistingTruncation}).
  // For the left side of a narrowed relation (see {DecomposeIndex}),
  // {base_value} may be a 64-bit operation whose truncation is the base,
  // so that {base} is its truncation key (see {TruncationKey}).
  struct BaseAndOffset {
    OpIndex base;
    OpIndex base_value;
    uint32_t offset;
  };

  // A loop phi {i = phi(init, i + step)}, where {value} is the
  // canonical value of {i}.
  struct InductionVariable {
    OpIndex value;
    uint32_t init;
    uint32_t step;
  };

  // A bound {i <= x} of an induction variable in the loop, where {x} is
  // at least {x_lower_bound} when entering the loop.
  struct LoopBound {
    OpIndex x;
    int64_t x_lower_bound;
  };

  // Identifies an operation whose value only depends on its inputs (see
  // {CanonicalValue}) by its opcode, options and the canonical values of
  // its inputs.
  struct StructuralKey {
    Opcode opcode;
    uint64_t options;
    uint64_t constant;
    OpIndex left;
    OpIndex right;

    bool operator==(const StructuralKey& other) const = default;

    template <typename H>
    friend H AbslHashValue(H h, const StructuralKey& key) {
      return H::combine(std::move(h), key.opcode, key.options, key.constant,
                        key.left, key.right);
    }
  };

  void ProcessBlock(const Block& block);
  void ProcessTrapIf(OpIndex op_idx, const TrapIfOp& trap_if);
  void ProcessBranch(const Block* block);

  void BeginBlock(const Block* block);
  void FinishBlock(const Block* block);

  OpIndex ResolveAliases(OpIndex object) const;
  OpIndex ResolveReplacements(OpIndex value) const;

  void ProcessCondition(OpIndex trap_if, OpIndex condition, bool holds);
  void ProcessComparison(OpIndex trap_if, OpIndex condition, bool holds);
  void ProcessNotEqual(const Relation& relation);
  void ProcessNotEqualToConstant(const ReducedLength& length, uint32_t n);
  void ProcessBoundsCheck(OpIndex trap_if, const BoundsCheck& bounds_check,
                          uint32_t reduction);

  void ProcessLoopHeader(const Block* header);
  std::optional<InductionVariable> TryMatchInductionVariable(
      OpIndex index, const PhiOp& phi) const;
  template <typename Conditions>
  bool HasSmallBound(const InductionVariable& induction,
                     const Conditions& conditions) const;
  bool IsSmallBound(OpIndex x, uint32_t step, bool is_signed,
                    bool inclusive) const;
  template <typename Conditions, typename Values>
  void ProcessDecreasingInduction(OpIndex index, const PhiOp& phi,
                                  const Conditions& conditions,
                                  const Values& non_zero, const Block* forward);
  template <typename Conditions, typename Values, typename InitLower>
  bool HasDecreasingLowerBound(OpIndex value, uint32_t decrement,
                               const Conditions& conditions,
                               const Values& non_zero,
                               const InitLower& init_lower) const;
  template <typename Conditions>
  std::optional<LoopBound> FindNotEqualBound(const InductionVariable& induction,
                                             const Conditions& conditions,
                                             const Block* forward) const;
  std::optional<int64_t> EntryLowerBound(OpIndex x, OpIndex x_value,
                                         const Block* forward) const;
  std::optional<int64_t> EntryLowerBoundOfStart(OpIndex init,
                                                const Block* forward) const;
  void RecordLoopBound(const InductionVariable& induction,
                       const LoopBound& bound);
  std::optional<ReducedLength> TryResolveMergeBound(const LoopBound& bound);

  // Calls {f(condition, holds)} for the conditions combined by
  // {condition} (with and, or, and comparisons to 0) whose value is
  // known when {condition} holds, or does not hold if not {holds}. These
  // conditions are comparisons, or other values, which hold when they
  // are not 0.
  template <typename F>
  void ForEachCondition(OpIndex condition, bool holds, const F& f,
                        int depth = 0) const;
  // Calls {f(condition, holds)} for the conditions of the branches
  // leading to {block} from {stop} (excluded) in the dominator tree.
  template <typename F>
  void ForEachDominatingCondition(const Block* block, const Block* stop,
                                  const F& f) const;
  std::optional<Relation> Normalize(OpIndex condition, bool holds) const;
  bool Dominates(const Block* dominator, const Block* block) const;

  static std::optional<OffsetRange> ExtendDownToNonNegative(
      const OffsetRange& range, const std::optional<OffsetRange>& non_negative);
  std::optional<OffsetRange> KnownOffsets(
      const BoundsCheckKey& key,
      const std::optional<OffsetRange>& non_negative) const;
  std::optional<OffsetRange> NonNegativeOffsets(OpIndex base) const;
  bool IsKnownNonNegative(const BoundsCheckKey& key, uint32_t offset) const;
  void RecordNonNegativeOffset(OpIndex base, uint32_t offset);
  uint32_t MinLength(OpIndex length) const;
  void RecordMinLength(OpIndex length, uint64_t min_length);
  bool IsKnownLength(OpIndex value) const;
  void RecordKnownLength(OpIndex value);

  std::optional<BoundsCheckCondition> TryExtractBoundsCheckCondition(
      const Relation& relation) const;
  std::optional<ReducedLength> TryExtractArrayLength(OpIndex length) const;
  OpIndex LengthKey(OpIndex length) const;
  bool IsArrayLengthWithoutWrapAround(OpIndex length) const;
  std::optional<BaseAndOffset> TryExtractNonNegativeIndex(
      const Relation& relation) const;
  BaseAndOffset ExtractBaseAndOffset(OpIndex index, bool wide = false) const;
  BaseAndOffset DecomposeIndex(OpIndex index, bool wide = false) const;
  OpIndex CanonicalValue(OpIndex value, int depth = 0) const;
  bool IsKnownSmi(OpIndex object) const;
  std::optional<uint32_t> TryExtractI32Const(OpIndex expr) const;
  std::optional<uint32_t> TryExtractI64ConstLow(OpIndex expr) const;
  static uint64_t ChangeOptions(const ChangeOp& change);
  static uint64_t ChangeOptions(ChangeOp::Kind kind,
                                ChangeOp::Assumption assumption,
                                RegisterRepresentation from,
                                RegisterRepresentation to);
  OpIndex TruncationKey(OpIndex x) const;
  OpIndex ExistingTruncation(OpIndex x) const;

  const Graph& graph_;
  Zone* phase_zone_;
  const WasmLoadEliminationAnalyzer* load_elimination_;

  // Set of traps identified as redundant by the analysis.
  ZoneAbslFlatHashSet<OpIndex> redundant_traps_;

  // Memoized canonical values, and the canonical value of each
  // structure (the first operation found with this structure).
  mutable ZoneAbslFlatHashMap<OpIndex, OpIndex> canonical_values_;
  mutable ZoneAbslFlatHashMap<StructuralKey, OpIndex> values_by_structure_;
  // The first TruncateWord64ToWord32 operation found for each canonical
  // value of its input (see {ExistingTruncation}), which may be in any
  // block. The canonical value
  // of the truncation may instead be a 64-bit operation (see
  // {TruncationKey}).
  mutable ZoneAbslFlatHashMap<OpIndex, OpIndex> truncations_;

  // Summary of all the bounds check information collected so far.
  BoundsCheckMap known_bounds_checks_;
  NonNegativeOffsetMap known_non_negative_offsets_;
  MinLengthMap known_min_lengths_;
  LengthAliasMap known_length_aliases_;
  // Whether some length alias has been recorded.
  bool has_length_aliases_ = false;
  // Whether {known_length_aliases_} has an open snapshot for the current
  // block.
  bool length_aliases_open_ = false;

  FixedBlockSidetable<std::optional<Snapshot>> block_to_snapshot_mapping_;

  // The predecessor snapshots are used as temporary vectors when
  // starting to process a block. We store them as members to avoid
  // reallocation.
  ZoneVector<BoundsCheckMap::Snapshot> predecessor_bounds_check_snapshots_;
  ZoneVector<NonNegativeOffsetMap::Snapshot>
      predecessor_non_negative_snapshots_;
  ZoneVector<MinLengthMap::Snapshot> predecessor_min_length_snapshots_;
  ZoneVector<LengthAliasMap::Snapshot> predecessor_length_alias_snapshots_;
};

template <class Next>
class WasmBoundsCheckEliminationReducer : public Next {
 public:
  TURBOSHAFT_REDUCER_BOILERPLATE(WasmBoundsCheckElimination)

  void Analyze() {
    // Load elimination is analyzed first, so that we can take its
    // replacements into account.
    Next::Analyze();
    if (v8_flags.turboshaft_wasm_bounds_check_elimination &&
        WasmBoundsCheckEliminationAnalyzer::HasArrayBoundsChecks(
            __ input_graph())) {
      const WasmLoadEliminationAnalyzer* load_elimination = nullptr;
      if constexpr (reducer_list_contains<ReducerList,
                                          WasmLoadEliminationReducer>::value) {
        load_elimination = __ GetWasmLoadEliminationAnalyzer();
      }
      analyzer_.emplace(__ input_graph(), __ phase_zone(), load_elimination);
      analyzer_->Run();
    }
  }

  OpIndex REDUCE_INPUT_GRAPH(TrapIf)(OpIndex ig_index,
                                     const TrapIfOp& trap_if) {
    if (analyzer_.has_value() && analyzer_->IsRedundantTrap(ig_index)) {
      if (v8_flags.turboshaft_verify_wasm_bounds_check_elimination) {
        VerifyRedundantBoundsCheck(trap_if);
      }
      return OpIndex::Invalid();
    }
    return Next::ReduceInputGraphTrapIf(ig_index, trap_if);
  }

 private:
  // Only constructed when the optimization is enabled.
  std::optional<WasmBoundsCheckEliminationAnalyzer> analyzer_;

  void VerifyRedundantBoundsCheck(const TrapIfOp& trap_if) {
    V<Word32> condition = __ MapToNewGraph(trap_if.condition());
    // The trap fires when the condition is false if the trap is
    // negated, and when it is true otherwise.
    V<Word32> in_bounds =
        trap_if.negated ? condition : __ Word32Equal(condition, 0);
    IF_NOT (LIKELY(in_bounds)) {
      __ WasmCallRuntime(
          __ phase_zone(), Runtime::kAbort,
          {__ TagSmi(static_cast<int>(
              AbortReason::kTurboshaftWasmBoundsCheckEliminationError))},
          __ NoContextConstant());
      __ Unreachable();
    }
  }
};

#include "src/compiler/turboshaft/undef-assembler-macros.inc"

}  // namespace v8::internal::compiler::turboshaft

#endif  // V8_COMPILER_TURBOSHAFT_WASM_BOUNDS_CHECK_ELIMINATION_REDUCER_H_
