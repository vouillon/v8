// Copyright 2025 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_COMPILER_TURBOSHAFT_WASM_BOUNDS_CHECK_ELIMINATION_REDUCER_H_
#define V8_COMPILER_TURBOSHAFT_WASM_BOUNDS_CHECK_ELIMINATION_REDUCER_H_

#if !V8_ENABLE_WEBASSEMBLY
#error This header should only be included if WebAssembly is enabled.
#endif  // !V8_ENABLE_WEBASSEMBLY

#include <algorithm>
#include <optional>

#include "src/common/scoped-modification.h"
#include "src/compiler/turboshaft/analyzer-iterator.h"
#include "src/compiler/turboshaft/assembler.h"
#include "src/compiler/turboshaft/graph.h"
#include "src/compiler/turboshaft/loop-finder.h"
#include "src/compiler/turboshaft/phase.h"
#include "src/compiler/turboshaft/snapshot-table.h"
#include "src/compiler/turboshaft/utils.h"
#include "src/compiler/turboshaft/wasm-load-elimination-reducer.h"
#include "src/zone/zone-containers.h"
#include "src/zone/zone.h"

#ifdef DEBUG
#define TRACE(x)                                                 \
  do {                                                           \
    if (v8_flags.turboshaft_trace_wasm_bounds_check_elimination) \
      StdoutStream() << x << std::endl;                          \
  } while (false)
#else
#define TRACE(x)
#endif

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
// Conditions of the form {base + n < a.length - c}, for a small
// constant {c}, are also taken into account, once {a.length} is known
// to be at least {c} (otherwise, {a.length - c} may wrap around). Such
// a condition shows that the offsets from {n} to {n + c} are within
// bounds. Any successful bounds check of {a} shows that its length is
// at least 1. For example, wasm_of_ocaml stores a header in the first
// element of the arrays it uses for OCaml arrays, so the OCaml bounds
// check of an access {a.(i)} is {i < a.length - 1}, which shows that
// the Wasm access {a[i+1]} is within bounds, as long as some previous
// access to {a} showed that its length is at least 1.
//
// Signed conditions {base + n <s a.length - c}, as produced by languages
// with signed integers (Java, Kotlin, Dart...), are taken into account
// when {base + n} is known to be non-negative: the condition then shows
// that {a.length - c} is positive, and that {base + n <u a.length - c}.
// Since array lengths are less than 2^30, {base + n} is non-negative
// when some {base + m} is within bounds and {n - m} is at most 2^30. Branch
// conditions combining comparisons are also taken into account: when
// {x | y} does not hold, neither {x} nor {y} holds, and when {x & y}
// holds for two comparisons (whose values are 0 or 1), both hold.
//
// Loop variables are shown to be non-negative by induction. A loop phi
// {i = phi(c0, i + c)}, with constants {c0 >= 0} and {c > 0}, is
// non-negative in the whole loop (and after it) if its back edge is only
// taken when {i < x}, for a bound {x} small enough that {i + c} cannot
// overflow: an array length minus a constant, or a constant at most
// {2^31 - c}. Loops written {for (i = c0; i != x; i++)}, as OCaml {for}
// loops are compiled, are handled too, when the loop is only entered if
// {c0 <= x}, for a loop-invariant {x}: {i} is then at most {x} in the
// loop, and when {x} is an array length minus {r >= 1}, the offsets 0
// to {r - 1} of {i} are within bounds. After loop unrolling, the step is
// {c > 1} and the back edge is taken when {i + k != x} for all {k} from
// 0 to {c - 1}, which works the same way. In the unrolled copies, a
// test {x + k != a.length - r} then shows that {x + k + r} is within
// bounds when {x + k + r - 1} is. When {x} is a phi plus a constant, the
// constant inputs of the phi for which the loop is not entered are
// ruled out: if a single input remains, {x} is this input plus the
// constant in the loop. For instance, with {x = phi(a.length, 0) - 1}
// (the length of an array that may be empty, minus 1), the loop is not
// entered when {x} is -1, so {x} is {a.length - 1} in the loop, which
// is recorded in {known_length_aliases_}.
//
// When some bounds checks are later followed by stronger checks in
// the same block, the instructions in between can be duplicated into
// two paths:
// 1. A fast path that executes without the redundant bounds checks. The
//    guard protecting this fast path consists of the stronger checks.
// 2. A fallback path that preserves the original bounds checks, ensuring
//    correct semantics if the guard fails.
// This approach guarantees that the program's semantics are preserved,
// even when the duplicated instructions have observable side-effects,
// and that traps are reported at the correct location in the source code.
//
// For example, consider:
//     a[0] = 10;
//     a[1] = 11;
// This can be rewritten as:
//     if (1 < a.length) {
//         // Fast path: redundant bounds checks removed
//         a[0] = 10;
//         a[1] = 11;
//     } else {
//         // Fallback path: execute with original bounds checks
//         a[0] = 10;
//         a[1] = 11;
//         // The above code is guaranteed to trap if out of bounds
//         UNREACHABLE();
//     }
//
// The guard checks the smallest and the largest offsets of the
// bounds checks removed from the fast path, unless one of these two
// checks is implied by what is already known when entering the
// sequence. In the example above, the check {0 < a.length} is implied
// by {1 < a.length}. But for a dynamic index, the sequence
//     a[i] = 10;
//     a[i+1] = 11;
// needs a guard {i < a.length && i + 1 < a.length}, unless {i} is
// already known to be non-negative, in which case {i + 1 < a.length}
// is enough. On 64-bit targets, the two checks are done with a single
// 64-bit comparison {zext(i) + 1 < zext(a.length)}.

// Key for grouping bounds checks by base and array
struct BoundsCheckKey {
  OpIndex base;  // Invalid means that we do not have a base.
  OpIndex array;

  bool operator==(const BoundsCheckKey& other) const = default;

  bool operator<(const BoundsCheckKey& other) const {
    return std::tie(base, array) < std::tie(other.base, other.array);
  }

  template <typename H>
  friend H AbslHashValue(H h, const BoundsCheckKey& key) {
    return H::combine(std::move(h), key.base, key.array);
  }
};

// Identifies a specific bounds check by its key (base + array) and offset.
struct BoundsCheck {
  BoundsCheckKey key;
  uint32_t offset;

  bool operator<(const BoundsCheck& other) const {
    return std::tie(key, offset) < std::tie(other.key, other.offset);
  }
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
  explicit OffsetRange(uint32_t lower, uint32_t upper)
      : lower_(lower), upper_(upper) {
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
  // the two ranges.
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
// The correctness of this implementation is supported by the
// following Z3 proofs. They share these definitions:
//
//   (define-fun valid-range ((n1 (_ BitVec 32)) (n2 (_ BitVec 32))) Bool
//     (bvsge (bvsub n2 n1) #x00000000))
//   (define-fun non-negative ((v (_ BitVec 32))) Bool
//     (bvsge v #x00000000))
//
// A bounds check is redundant when its offset is covered by the
// range:
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
// also within bounds, provided that it is a valid range.
//
// A redundant bounds check remains redundant when the upper bound is
// updated:
//
//   (declare-const n (_ BitVec 32))
//   (declare-const n1 (_ BitVec 32))
//   (declare-const n2 (_ BitVec 32))
//   (declare-const n3 (_ BitVec 32))
//   (assert (valid-range n1 n2))
//   ; Assumption: n is covered by the range n1..n2
//   (assert (valid-range n1 n))
//   (assert (valid-range n n2))
//   ; What we check before updating the upper bound
//   (assert (valid-range n2 n3))
//   (assert (valid-range n1 n3))
//   ; Then n is covered by the range n1..n3
//   (assert (not (valid-range n n3)))
//   (check-sat)
//
// A bounds check shows that the index is non-negative:
//
//   (declare-const l (_ BitVec 32))
//   (declare-const x (_ BitVec 32))
//   (declare-const n (_ BitVec 32))
//   (assert (bvsge l #x00000000))
//   (assert (bvult (bvadd x n) l))
//   (assert (not (non-negative (bvadd x n))))
//   (check-sat)
//
// Non-negative offsets form ranges:
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
// If x + n is non-negative, a bounds check at offset m covers the
// whole range n..m:
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
// If the array length l is at least c, then x + n < l - c shows that
// all the offsets from n to n + c are within bounds (without this
// assumption on l, it does not: l - c may wrap around):
//
//   (declare-const l (_ BitVec 32))
//   (declare-const x (_ BitVec 32))
//   (declare-const n (_ BitVec 32))
//   (declare-const c (_ BitVec 32))
//   (declare-const k (_ BitVec 32))
//   (assert (bvsge l #x00000000))
//   (assert (bvuge l c))
//   (assert (bvult (bvadd x n) (bvsub l c)))
//   (assert (valid-range n k))
//   (assert (valid-range k (bvadd n c)))
//   (assert (not (bvult (bvadd x k) l)))
//   (check-sat)
//
// It also shows that x + n is non-negative, and that l is at least
// c + 1:
//
//   (declare-const l (_ BitVec 32))
//   (declare-const x (_ BitVec 32))
//   (declare-const n (_ BitVec 32))
//   (declare-const c (_ BitVec 32))
//   (assert (bvsge l #x00000000))
//   (assert (bvuge l c))
//   (assert (bvult (bvadd x n) (bvsub l c)))
//   (assert (not (and (non-negative (bvadd x n))
//                     (bvuge l (bvadd c #x00000001)))))
//   (check-sat)
//
// A guard for the range lo..hi only needs to check hi when some x + n
// with n <= lo is known to be non-negative (in particular, when it is
// known to be within bounds): this is the previous proof with m = hi.
// Symmetrically, it only needs to check lo when x + n is known to be
// within bounds for some n >= hi:
//
//   (declare-const l (_ BitVec 32))
//   (declare-const x (_ BitVec 32))
//   (declare-const n (_ BitVec 32))
//   (declare-const lo (_ BitVec 32))
//   (declare-const hi (_ BitVec 32))
//   (declare-const k (_ BitVec 32))
//   (assert (bvsge l #x00000000))
//   (assert (bvult (bvadd x n) l))
//   (assert (bvult (bvadd x lo) l))
//   (assert (valid-range lo hi))
//   (assert (valid-range hi n))
//   (assert (valid-range lo n))
//   (assert (valid-range lo k))
//   (assert (valid-range k hi))
//   (assert (not (bvult (bvadd x k) l)))
//   (check-sat)
//
// On 64-bit targets, a guard checking both bounds of the range lo..hi
// is the single comparison zext(x + lo) + (hi - lo) < zext(l), with
// 64-bit zero extensions. It shows that all the offsets of the range
// are within bounds:
//
//   (define-fun zext ((v (_ BitVec 32))) (_ BitVec 64)
//     ((_ zero_extend 32) v))
//   (declare-const l (_ BitVec 32))
//   (declare-const x (_ BitVec 32))
//   (declare-const lo (_ BitVec 32))
//   (declare-const hi (_ BitVec 32))
//   (declare-const k (_ BitVec 32))
//   (assert (bvsge l #x00000000))
//   (assert (valid-range lo hi))
//   (assert (bvult (bvadd (zext (bvadd x lo)) (zext (bvsub hi lo)))
//                  (zext l)))
//   (assert (valid-range lo k))
//   (assert (valid-range k hi))
//   (assert (not (bvult (bvadd x k) l)))
//   (check-sat)
//
// It holds whenever both bounds are within bounds, so it never sends
// to the fallback code a sequence whose checks would all succeed:
//
//   (declare-const l (_ BitVec 32))
//   (declare-const x (_ BitVec 32))
//   (declare-const lo (_ BitVec 32))
//   (declare-const hi (_ BitVec 32))
//   (assert (bvsge l #x00000000))
//   (assert (valid-range lo hi))
//   (assert (bvult (bvadd x lo) l))
//   (assert (bvult (bvadd x hi) l))
//   (assert (not (bvult (bvadd (zext (bvadd x lo)) (zext (bvsub hi lo)))
//                       (zext l))))
//   (check-sat)

//
// A signed condition {x + n <s l - c} with {x + n} non-negative (and a
// small constant {c}) shows the unsigned condition, and that {l - c}
// does not wrap around:
//
//   (declare-const l (_ BitVec 32))
//   (declare-const x (_ BitVec 32))
//   (declare-const n (_ BitVec 32))
//   (declare-const c (_ BitVec 32))
//   (assert (bvsge l #x00000000))
//   (assert (bvule c #x00010000))
//   (assert (non-negative (bvadd x n)))
//   (assert (bvslt (bvadd x n) (bvsub l c)))
//   (assert (not (and (bvult (bvadd x n) (bvsub l c))
//                     (bvuge l (bvadd c #x00000001)))))
//   (check-sat)
//
// With array lengths less than 2^30, if {x + m} is within bounds, then
// {x + m + d} is non-negative for {d} at most 2^30:
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
// Induction for loop phis {i = phi(c0, i + c)}: if {i} is non-negative
// and the back edge is only taken when {i < x}, then {i + c} is
// non-negative for a constant bound at most {2^31 - c} (signed or
// unsigned comparison):
//
//   (declare-const i (_ BitVec 32))
//   (declare-const x (_ BitVec 32))
//   (declare-const c (_ BitVec 32))
//   (assert (non-negative i))
//   (assert (bvult i x)) ; or (bvslt i x)
//   (assert (bvugt c #x00000000))
//   (assert (bvule c #x40000000))
//   (assert (bvule x (bvadd (bvsub #x7fffffff c) #x00000001)))
//   (assert (not (non-negative (bvadd i c))))
//   (check-sat)
//
// and for a bound {l - r}, where {l} is an array length, with a signed
// comparison {i < l - r} or {i <= l - r}, or for the bound {l} with an
// unsigned comparison (the inclusive comparison needs {l} less than
// 2^30):
//
//   (declare-const i (_ BitVec 32))
//   (declare-const l (_ BitVec 32))
//   (declare-const r (_ BitVec 32))
//   (declare-const c (_ BitVec 32))
//   (assert (bvult l #x40000000))
//   (assert (bvult r #x40000000))
//   (assert (non-negative i))
//   (assert (or (bvsle i (bvsub l r)) (bvule i l)))
//   (assert (bvugt c #x00000000))
//   (assert (bvule c #x40000000))
//   (assert (not (non-negative (bvadd i c))))
//   (check-sat)
//
// For {i != x} loops, {i <= x} is preserved by each increment of 1
// (and so by an unrolled step {c}, with {i + k != x} for all {k} below
// {c}):
//
//   (declare-const i (_ BitVec 32))
//   (declare-const x (_ BitVec 32))
//   (assert (bvsle i x))
//   (assert (not (= i x)))
//   (assert (not (bvsle (bvadd i #x00000001) x)))
//   (check-sat)
//
// and {0 <= i <= l - r} shows that the offsets 0 to {r - 1} of {i} are
// within bounds:
//
//   (declare-const i (_ BitVec 32))
//   (declare-const l (_ BitVec 32))
//   (declare-const r (_ BitVec 32))
//   (declare-const k (_ BitVec 32))
//   (assert (bvult l #x40000000))
//   (assert (bvuge r #x00000001))
//   (assert (bvule r #x40000000))
//   (assert (non-negative i))
//   (assert (bvsle i (bvsub l r)))
//   (assert (bvult k r))
//   (assert (not (bvult (bvadd i k) l)))
//   (check-sat)
//
// A test {x + m + 1 != l} extends a known range ending at {x + m}:
//
//   (declare-const x (_ BitVec 32))
//   (declare-const m (_ BitVec 32))
//   (declare-const l (_ BitVec 32))
//   (assert (bvsge l #x00000000))
//   (assert (bvult (bvadd x m) l))
//   (assert (not (= (bvadd (bvadd x m) #x00000001) l)))
//   (assert (not (bvult (bvadd (bvadd x m) #x00000001) l)))
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

// Maps (base, array) pairs to their known safe offset ranges.
using BoundsCheckMap = KeyedSnapshotTable<BoundsCheckKey, OffsetRange>;

// Maps bases to the range of offsets {n} such that {base + n} is
// known to be non-negative.
using NonNegativeOffsetMap = KeyedSnapshotTable<OpIndex, OffsetRange>;

// Maps arrays to a lower bound on their length.
using MinLengthMap = KeyedSnapshotTable<OpIndex, uint32_t>;

// Maps values to an array length {l} and a constant {c} such that the
// value is {l - c}, when this is only known in part of the graph (see
// {ProcessLoopHeader}).
using LengthAliasMap =
    KeyedSnapshotTable<OpIndex, std::pair<OpIndex, uint32_t>>;

// Represents a sequence of instructions that could be duplicated:
// - Fast path: execute without bounds checks (checks moved earlier)
// - Fallback path: execute with original bounds checks intact
struct FallbackInstructionSequence {
  BoundsCheckKey key;
  // The base value used by the guard. {key.base} identifies a value up
  // to duplicated computations (see {CanonicalValue}), and the
  // operation it refers to may not be available where the guard is
  // emitted, while this one is computed before the start of the
  // sequence.
  OpIndex base_value;
  // The array length used by the guard, computed before the start of
  // the sequence.
  OpIndex array_length;

  // The smallest range containing the offsets of the covered traps.
  // Its bounds are offsets of covered traps.
  OffsetRange offsets;

  // What is known when entering the sequence: the offsets within
  // bounds, and the offsets {n} for which {key.base + n} is
  // non-negative.
  std::optional<OffsetRange> known_offsets;
  std::optional<OffsetRange> non_negative_offsets;

  // Range of instructions to clone [start_index, end_index), and their
  // positions in the block.
  OpIndex start_index;
  OpIndex end_index;
  uint32_t start_position;
  uint32_t end_position;

  // Traps that become redundant on the fast path (they remain in
  // fallback code), sorted by index.
  ZoneVector<OpIndex> covered_traps;

  // Heuristic: only generate fallback code if the instruction
  // sequence is shorter than this factor times the number of
  // eliminated bounds checks. This ensures an at most linear growth
  // of the code size.
  static constexpr uint32_t instruction_budget_per_trap = 20;

  // Heuristic: only generate fallback code if it eliminates at least
  // this number of bounds checks, once the checks of the guard are
  // taken into account. Sequences that eliminate a single check save
  // little at run time, while the duplicated code makes compilation
  // noticeably slower (mostly in later phases, such as late load
  // elimination).
  //
  // The exception is a sequence whose guard checks both bounds with a
  // single comparison (on 64-bit targets), which is accepted when it
  // eliminates a single check. Such sequences are typical
  // of byte accesses to an array with no known bound, as in
  //     a[i] = v; a[i+1] = v >> 8;
  // while sequences eliminating a single check in code produced by
  // wasm_of_ocaml almost always have a known bound, and are frequent
  // enough to make compilation noticeably slower.
  static constexpr int min_eliminated_checks = 2;

  // On 64-bit targets, a guard checking both bounds is a single
  // comparison, done on 64 bits (see {EmitTwoSidedBoundsCheck}).
  static constexpr bool single_comparison_two_sided_guard = Is64();

  // Which bounds of {offsets} the guard needs to check.
  struct Guard {
    bool check_lower;
    bool check_upper;
    // The number of comparisons of the guard.
    int check_count() const {
      if (check_lower && check_upper && single_comparison_two_sided_guard) {
        return 1;
      }
      return check_lower + check_upper;
    }
  };

  FallbackInstructionSequence(Zone* zone, const BoundsCheckKey& key,
                              OpIndex base_value, OpIndex array_length,
                              OffsetRange offsets,
                              std::optional<OffsetRange> known_offsets,
                              std::optional<OffsetRange> non_negative_offsets,
                              OpIndex start_index, OpIndex end_index,
                              uint32_t start_position, uint32_t end_position,
                              std::initializer_list<OpIndex> covered)
      : key(key),
        base_value(base_value),
        array_length(array_length),
        offsets(offsets),
        known_offsets(known_offsets),
        non_negative_offsets(non_negative_offsets),
        start_index(start_index),
        end_index(end_index),
        start_position(start_position),
        end_position(end_position),
        covered_traps(covered, zone) {
    DCHECK_LT(start_index, end_index);
    DCHECK_LT(start_position, end_position);
    std::sort(covered_traps.begin(), covered_traps.end());
  }

  Guard ComputeGuard() const {
    return ComputeGuard(offsets, known_offsets, non_negative_offsets);
  }

  uint32_t instruction_count() const { return end_position - start_position; }

  bool ShouldKeep() const {
    return IsProfitable(instruction_count(), covered_traps.size(),
                        ComputeGuard());
  }

  // Whether {other}, which ends after {this}, can be merged into
  // {this}: they must overlap, and the merged sequence must be
  // profitable.
  bool CanCoalesce(const FallbackInstructionSequence& other) const {
    DCHECK_EQ(key, other.key);
    DCHECK_LE(end_index, other.end_index);
    if (end_index < other.start_index) return false;
    const FallbackInstructionSequence& first =
        start_index <= other.start_index ? *this : other;
    // Both ranges are within the current known range, so their hull
    // is valid.
    std::optional<OffsetRange> merged_offsets = offsets.Hull(other.offsets);
    if (!merged_offsets.has_value()) return false;
    size_t trap_count = covered_traps.size();
    for (OpIndex trap : other.covered_traps) {
      if (!std::binary_search(covered_traps.begin(), covered_traps.end(),
                              trap)) {
        trap_count++;
      }
    }
    return IsProfitable(other.end_position - first.start_position, trap_count,
                        ComputeGuard(*merged_offsets, first.known_offsets,
                                     first.non_negative_offsets));
  }

  void Coalesce(const FallbackInstructionSequence& other) {
    DCHECK(CanCoalesce(other));
    offsets = *offsets.Hull(other.offsets);
    if (other.start_index < start_index) {
      // The guard moves to the start of {other}, so it can only rely
      // on what is known there, and on values computed before it.
      base_value = other.base_value;
      array_length = other.array_length;
      known_offsets = other.known_offsets;
      non_negative_offsets = other.non_negative_offsets;
      start_index = other.start_index;
      start_position = other.start_position;
    }
    end_index = other.end_index;
    end_position = other.end_position;
    for (OpIndex trap : other.covered_traps) {
      auto it =
          std::lower_bound(covered_traps.begin(), covered_traps.end(), trap);
      if (it == covered_traps.end() || *it != trap) {
        covered_traps.insert(it, 1, trap);
      }
    }
  }

  // Delete copy operations to avoid accidental copies
  FallbackInstructionSequence(const FallbackInstructionSequence&) = delete;
  FallbackInstructionSequence& operator=(const FallbackInstructionSequence&) =
      delete;
  // Move operations are allowed
  FallbackInstructionSequence(FallbackInstructionSequence&&) noexcept = default;
  FallbackInstructionSequence& operator=(
      FallbackInstructionSequence&&) noexcept = default;

 private:
  static Guard ComputeGuard(const OffsetRange& offsets,
                            const std::optional<OffsetRange>& known,
                            const std::optional<OffsetRange>& non_negative) {
    bool lower_implied =
        (known && ImpliesLowerBound(*known, offsets)) ||
        (non_negative && ImpliesLowerBound(*non_negative, offsets));
    bool upper_implied = known && ImpliesUpperBound(*known, offsets);
    // When both checks are implied, we still check the upper bound
    // (which is enough given the lower bound) rather than proving that
    // the guard can be omitted.
    return {!lower_implied, lower_implied || !upper_implied};
  }

  // Whether some offset {n <= offsets.lower()} from {facts} is such
  // that {base + n} is non-negative, so that only the upper bound of
  // {offsets} needs to be checked. Offsets within bounds are
  // non-negative, so {facts} can be either kind of range.
  static bool ImpliesLowerBound(const OffsetRange& facts,
                                const OffsetRange& offsets) {
    OffsetRange::RelativePosition position = facts.Classify(offsets.lower());
    // Either {n = offsets.lower()}, or {n = facts.upper()}.
    return position == OffsetRange::RelativePosition::kInside ||
           (position == OffsetRange::RelativePosition::kAbove &&
            OffsetRange::IsValidRange(facts.upper(), offsets.upper()));
  }

  // Whether some offset {n >= offsets.upper()} from {known} is within
  // bounds, so that only the lower bound of {offsets} needs to be
  // checked.
  static bool ImpliesUpperBound(const OffsetRange& known,
                                const OffsetRange& offsets) {
    OffsetRange::RelativePosition position = known.Classify(offsets.upper());
    // Either {n = offsets.upper()}, or {n = known.lower()}.
    return position == OffsetRange::RelativePosition::kInside ||
           (position == OffsetRange::RelativePosition::kBelow &&
            OffsetRange::IsValidRange(offsets.lower(), known.lower()));
  }

  // Whether the sequence removes more bounds checks than the guard
  // adds, for a limited amount of duplicated code.
  static bool IsProfitable(uint32_t instruction_count, size_t trap_count,
                           Guard guard) {
    int eliminated = static_cast<int>(trap_count) - guard.check_count();
    int min_eliminated = guard.check_lower && guard.check_upper &&
                                 single_comparison_two_sided_guard
                             ? 1
                             : min_eliminated_checks;
    return eliminated >= min_eliminated &&
           instruction_count <=
               instruction_budget_per_trap * static_cast<uint32_t>(eliminated);
  }
};

class WasmBoundsCheckEliminationAnalyzer {
 public:
  // {load_elimination} is optional. When provided, values replaced by
  // load elimination are considered the same as their replacement.
  WasmBoundsCheckEliminationAnalyzer(
      const Graph& graph, Zone* phase_zone,
      WasmLoadEliminationAnalyzer* load_elimination)
      : graph_(graph),
        phase_zone_(phase_zone),
        load_elimination_(load_elimination),
        redundant_traps_(phase_zone),
        canonical_values_(phase_zone),
        values_by_structure_(phase_zone),
        fallback_sequence_starts_(phase_zone),
        known_bounds_checks_(phase_zone),
        known_non_negative_offsets_(phase_zone),
        known_min_lengths_(phase_zone),
        known_length_aliases_(phase_zone),
        alias_lengths_(phase_zone),
        block_to_snapshot_mapping_(graph.block_count(), phase_zone),
        last_trap_bounds_checks_(phase_zone),
        array_lengths_(phase_zone),
        planned_fallbacks_by_key_(phase_zone),
        predecessor_bounds_check_snapshots_(phase_zone),
        predecessor_non_negative_snapshots_(phase_zone),
        predecessor_min_length_snapshots_(phase_zone),
        predecessor_length_alias_snapshots_(phase_zone) {}

  void Run() {
    LoopFinder loop_finder(phase_zone_, &graph_, LoopFinder::Config{});
    AnalyzerIterator iterator(phase_zone_, graph_, loop_finder);

    while (iterator.HasNext()) {
      const Block* block = iterator.Next();
      ProcessBlock(*block);
    }

#if DEBUG
    PrintStats();
#endif
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

  const FallbackInstructionSequence* FindFallbackCode(OpIndex index) const {
    auto it = fallback_sequence_starts_.find(index);
    if (it == fallback_sequence_starts_.end()) return nullptr;
    return &it->second;
  }

  void FinalizeFallbackSequence(const FallbackInstructionSequence& seq) {
    for (auto trap_if : seq.covered_traps) {
      redundant_traps_.insert(trap_if);
    }
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

  // A condition {index < a.length - reduction}, where {reduction} is a
  // constant (usually 0).
  struct BoundsCheckCondition {
    BoundsCheck bounds_check;
    // The base as computed by the index of the condition, while
    // {bounds_check.key.base} is its canonical value.
    OpIndex base_value;
    OpIndex array_length;
    uint32_t reduction;
  };

  // A trap that may start a fallback sequence, with what is known
  // right before it.
  struct TrapInfo {
    OpIndex trap_if;
    // The base as computed by the index of the trap.
    OpIndex base_value;
    // The trap, or its condition when the condition is right before
    // the trap.
    OpIndex start_index;
    uint32_t start_position;
    std::optional<OffsetRange> known_offsets;
    std::optional<OffsetRange> non_negative_offsets;
  };

  // An index decomposed as {base + offset}. {base} is the canonical
  // value of {base_value} (see {CanonicalValue}).
  struct BaseAndOffset {
    OpIndex base;
    OpIndex base_value;
    uint32_t offset;
  };

  // Identifies a pure operation by its opcode, options and the
  // canonical values of its inputs.
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

  void ProcessCondition(OpIndex trap_if, OpIndex condition, bool inverted,
                        int depth = 0);
  void ProcessLoopHeader(const Block* header);
  // Calls {f(comparison, holds)} for the comparisons combined by
  // {condition} (with and, or, and comparisons to 0) whose value is
  // known when {condition} holds, or does not hold if not {holds}.
  template <typename F>
  void ForEachComparison(OpIndex condition, bool holds, const F& f,
                         int depth = 0) const;
  // Calls {f(comparison, holds)} for the comparisons of the branches
  // leading to {block} from {stop} (excluded) in the dominator tree.
  template <typename F>
  void ForEachDominatingComparison(const Block* block, const Block* stop,
                                   const F& f) const;
  bool Dominates(const Block* dominator, const Block* block) const;
  void ProcessBoundsCheck(OpIndex trap_if, OpIndex condition,
                          const BoundsCheck& bounds_check, OpIndex base_value,
                          OpIndex array_length, uint32_t extent);

  void RecordFallbackSequence(const BoundsCheckKey& key, uint32_t prev_offset,
                              OpIndex trap_if, uint32_t offset);
  void UpdateKnownBoundsChecks(const BoundsCheckKey& key, uint32_t offset,
                               const OffsetRange& offsets, OpIndex trap_if,
                               OpIndex condition, OpIndex base_value,
                               const std::optional<OffsetRange>& known_before,
                               const std::optional<OffsetRange>& non_negative);

  void RegisterFallbackSequence(FallbackInstructionSequence&& sequence);

  std::optional<OffsetRange> KnownOffsets(
      const BoundsCheckKey& key,
      const std::optional<OffsetRange>& non_negative) const;
  std::optional<OffsetRange> NonNegativeOffsets(OpIndex base) const;
  void RecordNonNegativeOffset(OpIndex base, uint32_t offset);
  uint32_t MinLength(OpIndex array) const;
  void RecordMinLength(OpIndex array, uint64_t length);

  std::optional<BoundsCheckCondition> TryExtractBoundChecksCondition(
      OpIndex index, bool inverted) const;
  std::optional<BoundsCheckCondition> TryExtractSignedBoundsCheckCondition(
      OpIndex condition, bool inverted) const;
  std::optional<std::pair<OpIndex, uint32_t>> TryExtractArrayLength(
      OpIndex length) const;
  bool IsArrayLengthWithoutWrapAround(OpIndex length) const;
  std::optional<std::pair<OpIndex, uint32_t>> TryExtractNonNegativeIndex(
      OpIndex condition, bool inverted) const;
  BaseAndOffset ExtractBaseAndOffset(OpIndex index) const;
  OpIndex CanonicalValue(OpIndex value, int depth = 0) const;
  bool IsKnownSmi(OpIndex object) const;
  std::optional<uint32_t> TryExtractI32Const(OpIndex expr) const;

  const Graph& graph_;
  Zone* phase_zone_;
  WasmLoadEliminationAnalyzer* load_elimination_;

  // Set of traps identified as redundant by the analysis.
  ZoneAbslFlatHashSet<OpIndex> redundant_traps_;

  // Memoized canonical values, and the canonical value of each
  // structure (the first operation found with this structure).
  mutable ZoneAbslFlatHashMap<OpIndex, OpIndex> canonical_values_;
  mutable ZoneAbslFlatHashMap<StructuralKey, OpIndex> values_by_structure_;

  // Fallback sequences keyed by their start index.
  ZoneAbslFlatHashMap<OpIndex, FallbackInstructionSequence>
      fallback_sequence_starts_;

  // Summary of all the bounds check information collected so far.
  BoundsCheckMap known_bounds_checks_;
  NonNegativeOffsetMap known_non_negative_offsets_;
  MinLengthMap known_min_lengths_;
  LengthAliasMap known_length_aliases_;
  // The array lengths used by {known_length_aliases_}.
  ZoneAbslFlatHashSet<OpIndex> alias_lengths_;
  // Whether {known_length_aliases_} has an open snapshot for the current
  // block.
  bool length_aliases_open_ = false;

  FixedBlockSidetable<std::optional<Snapshot>> block_to_snapshot_mapping_;

  // Information about the current block:

  // Position of the current operation in the block.
  uint32_t current_position_ = 0;

  // The block being processed.
  const Block* current_block_ = nullptr;

  // Previous bounds check traps that can start a fallback sequence.
  ZoneAbslBTreeMap<BoundsCheck, TrapInfo> last_trap_bounds_checks_;

  // Array lengths used in bounds checks within this block.
  ZoneAbslBTreeMap<OpIndex, OpIndex> array_lengths_;

  ZoneAbslBTreeMap<BoundsCheckKey, ZoneVector<FallbackInstructionSequence>>
      planned_fallbacks_by_key_;

  // The predecessor snapshots are used as temporary vectors when
  // starting to process a block. We store them as members to avoid
  // reallocation.
  ZoneVector<BoundsCheckMap::Snapshot> predecessor_bounds_check_snapshots_;
  ZoneVector<NonNegativeOffsetMap::Snapshot>
      predecessor_non_negative_snapshots_;
  ZoneVector<MinLengthMap::Snapshot> predecessor_min_length_snapshots_;
  ZoneVector<LengthAliasMap::Snapshot> predecessor_length_alias_snapshots_;

#if DEBUG
  void PrintStats();
  size_t total_trap_count_ = 0;
#endif
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
      WasmLoadEliminationAnalyzer* load_elimination = nullptr;
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
    if (analyzer_.has_value()) {
      MaybeInsertFallbackSequence(ig_index);
      if (analyzer_->IsRedundantTrap(ig_index)) {
        if (v8_flags.turboshaft_verify_wasm_bounds_check_elimination) {
          VerifyRedundantBoundsCheck(trap_if);
        }
        return OpIndex::Invalid();
      }
    }
    return Next::ReduceInputGraphTrapIf(ig_index, trap_if);
  }

  OpIndex REDUCE_INPUT_GRAPH(Comparison)(OpIndex ig_index,
                                         const ComparisonOp& comparison) {
    if (analyzer_.has_value()) {
      MaybeInsertFallbackSequence(ig_index);
    }
    return Next::ReduceInputGraphComparison(ig_index, comparison);
  }

 private:
  // Only constructed when the optimization is enabled.
  std::optional<WasmBoundsCheckEliminationAnalyzer> analyzer_;

  bool in_fallback_code_ = false;

  // Inserts fallback code if the analyzer planned it here.
  void MaybeInsertFallbackSequence(OpIndex ig_index) {
    if (in_fallback_code_) return;

    if (auto* sequence = analyzer_->FindFallbackCode(ig_index)) {
      TRACE("Insert fallback sequence: [" << sequence->start_index << ", "
                                          << sequence->end_index << ")");
      EmitFallbackSequence(*sequence);
      analyzer_->FinalizeFallbackSequence(*sequence);
    }
  }

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

  void EmitFallbackSequence(const FallbackInstructionSequence& seq) {
    ScopedModification<bool> set_true(&in_fallback_code_, true);

    // The cloned range must not separate an operation that can throw
    // from the DidntThrow that follows it: it starts at a trap or at its
    // condition, and ends right after a trap.
    DCHECK(__ input_graph().Get(seq.start_index).template Is<TrapIfOp>() ||
           __ input_graph().Get(seq.start_index).template Is<ComparisonOp>());
    DCHECK(__ input_graph()
               .Get(__ input_graph().PreviousIndex(seq.end_index))
               .template Is<TrapIfOp>());

    const Block* current_input_block = __ current_input_block();
    Label<> fallback_code(this);
    Label<> done(this);
    FallbackInstructionSequence::Guard guard = seq.ComputeGuard();
    if (guard.check_lower && guard.check_upper &&
        FallbackInstructionSequence::single_comparison_two_sided_guard) {
      EmitTwoSidedBoundsCheck(seq, fallback_code);
    } else {
      if (guard.check_upper) {
        EmitBoundsCheck(seq, seq.offsets.upper(), fallback_code);
      }
      if (guard.check_lower) {
        EmitBoundsCheck(seq, seq.offsets.lower(), fallback_code);
      }
    }
    GOTO(done);
    BIND(fallback_code);
    __ CloneAndInlineTrappingInstructions(seq.start_index, seq.end_index,
                                          current_input_block);
    BIND(done);
  }

  // Emits a bounds check that compares {base} + {offset} against the
  // array length and jumps to {fallback_code} if the index would go
  // out of bounds.
  void EmitBoundsCheck(const FallbackInstructionSequence& seq, uint32_t offset,
                       Label<>& fallback_code) {
    V<Word32> index = __ Word32Constant(offset);
    DCHECK_EQ(seq.key.base.valid(), seq.base_value.valid());
    if (seq.base_value.valid()) {
      index = __ Word32Add(__ MapToNewGraph(seq.base_value), index);
    }
    V<Word32> length = __ MapToNewGraph(seq.array_length);
    GOTO_IF_NOT(LIKELY(__ Uint32LessThan(index, length)), fallback_code);
  }

  // Emits a single comparison checking that all the offsets of
  // {seq.offsets} are within bounds, and jumps to {fallback_code}
  // otherwise: with {lo} and {hi} the bounds of the range,
  //     zext(base + lo) + (hi - lo) < zext(length)
  // computed on 64 bits, so that it cannot wrap around (see the proof
  // in the header).
  void EmitTwoSidedBoundsCheck(const FallbackInstructionSequence& seq,
                               Label<>& fallback_code) {
    DCHECK(FallbackInstructionSequence::single_comparison_two_sided_guard);
    uint32_t lower = seq.offsets.lower();
    uint32_t size = seq.offsets.upper() - lower;
    V<Word32> start = __ Word32Constant(lower);
    DCHECK_EQ(seq.key.base.valid(), seq.base_value.valid());
    if (seq.base_value.valid()) {
      start = __ Word32Add(__ MapToNewGraph(seq.base_value), start);
    }
    V<Word64> end = __ Word64Add(__ ChangeUint32ToUint64(start),
                                 __ Word64Constant(uint64_t{size}));
    V<Word32> length = __ MapToNewGraph(seq.array_length);
    GOTO_IF_NOT(LIKELY(__ Uint64LessThan(end, __ ChangeUint32ToUint64(length))),
                fallback_code);
  }
};

#include "src/compiler/turboshaft/undef-assembler-macros.inc"

}  // namespace v8::internal::compiler::turboshaft

#undef TRACE

#endif  // V8_COMPILER_TURBOSHAFT_WASM_BOUNDS_CHECK_ELIMINATION_REDUCER_H_
