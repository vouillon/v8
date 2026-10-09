// Copyright 2026 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "src/compiler/turboshaft/wasm-bounds-check-elimination-reducer.h"

#include "src/compiler/turboshaft/opmasks.h"
#include "src/wasm/wasm-objects.h"

namespace v8::internal::compiler::turboshaft {

#ifdef DEBUG
#define TRACE(x)                                                 \
  do {                                                           \
    if (v8_flags.turboshaft_trace_wasm_bounds_check_elimination) \
      StdoutStream() << x << std::endl;                          \
  } while (false)
#else
#define TRACE(x)
#endif

// See "Limits" in the header. {MaxLength(1)} is the largest possible
// array length (for the smallest elements, of 1 byte).
static constexpr uint32_t kMaxArrayLength = 1u << 30;
static_assert(v8::internal::WasmArray::MaxLength(1) < kMaxArrayLength);
static constexpr uint32_t kMaxReduction = 1u << 16;

void WasmBoundsCheckEliminationAnalyzer::ProcessBlock(const Block& block) {
  BeginBlock(&block);
  for (OpIndex op_idx : graph_.OperationIndices(block)) {
    if (const TrapIfOp* trap_if = graph_.Get(op_idx).TryCast<TrapIfOp>();
        trap_if && !ShouldSkipOptimizationStep()) {
      ProcessTrapIf(op_idx, *trap_if);
    }
  }
  FinishBlock(&block);
}

void WasmBoundsCheckEliminationAnalyzer::ProcessTrapIf(
    OpIndex op_idx, const TrapIfOp& trap_if) {
  // Other traps can still tell us something about the values they
  // check, but they are never eliminated.
  bool is_bounds_check = trap_if.trap_id == TrapId::kTrapArrayOutOfBounds;
  if (is_bounds_check) TRACE("ProcessTrapIf(" << op_idx << ")");

  // Past the trap, the condition holds if the trap is negated, and
  // does not hold otherwise.
  ProcessComparison(is_bounds_check ? op_idx : OpIndex::Invalid(),
                    trap_if.condition(), trap_if.negated);
}

void WasmBoundsCheckEliminationAnalyzer::BeginBlock(const Block* block) {
  // Collect the snapshots of all predecessors.
  predecessor_bounds_check_snapshots_.clear();
  predecessor_non_negative_snapshots_.clear();
  predecessor_min_length_snapshots_.clear();
  for (const Block* p : block->PredecessorsIterable()) {
    std::optional<Snapshot> pred_snapshots =
        block_to_snapshot_mapping_[p->index()];
    // When we visit a loop header, its back edge hasn't been visited
    // yet, so we ignore it: what is known at the header is what is
    // known at the end of its forward predecessor. This remains true in
    // later iterations. The leaves of these facts (see {CanonicalValue})
    // are defined in dominators of the header, and so are not computed
    // again in the loop, and facts are never propagated along back
    // edges.
    DCHECK_IMPLIES(!pred_snapshots.has_value(),
                   block->IsLoop() && block->LastPredecessor() == p);
    if (!pred_snapshots.has_value()) {
      continue;
    }
    predecessor_bounds_check_snapshots_.push_back(
        pred_snapshots->bounds_checks);
    predecessor_non_negative_snapshots_.push_back(
        pred_snapshots->non_negative_offsets);
    predecessor_min_length_snapshots_.push_back(pred_snapshots->min_lengths);
  }
  // Without merge functions, the new snapshots only contain what is
  // known in all predecessors, that is, in the common ancestor of the
  // predecessor snapshots. By induction, the ancestors of the snapshot
  // of a block are snapshots of blocks that dominate it, so this common
  // ancestor is the snapshot of a block that dominates all the (forward)
  // predecessors, and thus the current block (for a loop header, as it
  // dominates its back edge, any path to it enters it first from its
  // forward predecessor): facts known at the start of a block were
  // recorded in its dominators, as {CanonicalValue} requires.
  known_bounds_checks_.StartNewSnapshot(
      base::VectorOf(predecessor_bounds_check_snapshots_));
  known_non_negative_offsets_.StartNewSnapshot(
      base::VectorOf(predecessor_non_negative_snapshots_));
  known_min_lengths_.StartNewSnapshot(
      base::VectorOf(predecessor_min_length_snapshots_));

  if (block->IsBranchTarget() && !ShouldSkipOptimizationStep()) {
    // The current block is a branch target, so we see if the branch
    // condition can be used as a bounds check.
    ProcessBranch(block);
  }
}

void WasmBoundsCheckEliminationAnalyzer::FinishBlock(const Block* block) {
  block_to_snapshot_mapping_[block->index()] =
      Snapshot{known_bounds_checks_.Seal(), known_non_negative_offsets_.Seal(),
               known_min_lengths_.Seal()};
}

void WasmBoundsCheckEliminationAnalyzer::ProcessBranch(const Block* block) {
  // The current block is a branch target. Track the branch condition
  // for bounds check analysis.
  DCHECK_EQ(block->PredecessorCount(), 1);
  const Operation& op = block->LastPredecessor()->LastOperation(graph_);
  const BranchOp* branch = op.TryCast<BranchOp>();
  if (!branch) return;
  DCHECK_EQ(block, any_of(branch->if_true, branch->if_false));

  TRACE("ProcessBranch (" << block->index() << ")");

  ProcessComparison(/*Not a TrapIf*/ OpIndex::Invalid(), branch->condition(),
                    block == branch->if_true);
}

// Records what we learn from {condition}, which holds if {holds}, and
// does not hold otherwise. For an array bounds check trap, {trap_if} is
// the trap. For a branch or another trap, it is OpIndex::Invalid().
void WasmBoundsCheckEliminationAnalyzer::ProcessComparison(OpIndex trap_if,
                                                           OpIndex condition,
                                                           bool holds) {
  std::optional<Relation> relation = Normalize(condition, holds);
  if (!relation.has_value() || relation->is_signed()) return;
  if (auto decoded = TryExtractBoundsCheckCondition(*relation)) {
    const auto& [bounds_check, reduction] = *decoded;
    const auto& [key, offset] = bounds_check;
    // Unless the length is known to be at least {r}, {a.length - r} may
    // wrap around, and we learn nothing.
    if (reduction == 0 || MinLength(key.length) >= reduction) {
      // Only a comparison with the length itself is an actual bounds
      // check, which can be eliminated. Other conditions are only used
      // for what they show, even for a trap.
      ProcessBoundsCheck(reduction == 0 ? trap_if : OpIndex::Invalid(),
                         bounds_check, reduction);
      // With {index < a.length - reduction}, the length is at least
      // {reduction + 1}, and more for a constant index
      // ([reduced-length-min-length], [constant-index-min-length]).
      uint64_t index_bound = key.base.valid() ? 0 : offset;
      if (index_bound <= kMaxInt) {
        RecordMinLength(key.length, index_bound + reduction + 1);
      }
    }
  }
  // This must be done after processing the bounds check: a trap must not
  // show that its own index is non-negative.
  if (auto non_negative_index = TryExtractNonNegativeIndex(*relation)) {
    RecordNonNegativeOffset(non_negative_index->base,
                            non_negative_index->offset);
  }
}

void WasmBoundsCheckEliminationAnalyzer::ProcessBoundsCheck(
    OpIndex trap_if, const BoundsCheck& bounds_check, uint32_t reduction) {
  const auto& [key, offset] = bounds_check;
  // Only plain bounds checks can be eliminated.
  DCHECK_IMPLIES(trap_if.valid(), reduction == 0);

  // The condition shows that the offsets from {offset} to {last} are
  // within bounds ([reduced-length]).
  uint32_t last = offset + reduction;
  DCHECK(OffsetRange::IsValidRange(offset, last));
  OffsetRange direct(offset, last);

  // A constant index is non-negative when within bounds. If {offset} or
  // {last} is at least 2^31, the condition cannot hold (the trap always
  // traps, or the branch is never taken).
  if (!key.base.valid() && !(OffsetRange::IsValidRange(0, offset) &&
                             OffsetRange::IsValidRange(0, last))) {
    return;
  }

  std::optional<OffsetRange> non_negative = NonNegativeOffsets(key.base);
  std::optional<OffsetRange> known = KnownOffsets(key, non_negative);

  if (known.has_value() && known->Contains(offset) && known->Contains(last)) {
    TRACE("  Redundant bounds check: offsets ["
          << offset << ", " << last << "] within [" << known->lower() << ", "
          << known->upper() << "]");
    if (trap_if.valid()) {
      redundant_traps_.insert(trap_if);
    }
    return;
  }

  // Offsets within bounds after this check: {direct}, extended down to
  // the non-negative offsets below it ([non-negative-and-check]).
  OffsetRange checked =
      ExtendDownToNonNegative(direct, non_negative).value_or(direct);

  // When the condition holds, {known} and {checked} are both within
  // bounds, so their hull is valid ([covered]). If it is not, the
  // condition cannot hold given what we know.
  std::optional<OffsetRange> offsets =
      known.has_value() ? known->Hull(checked) : checked;
  if (!offsets.has_value()) {
    TRACE("  Out-of-range check: offsets ["
          << offset << ", " << last << "] outside [" << known->lower() << ", "
          << known->upper() << "]");
    return;
  }
  TRACE("  Known offsets for base " << key.base << " and length " << key.length
                                    << ": [" << offsets->lower() << ", "
                                    << offsets->upper() << "]");
  known_bounds_checks_.Set(key, *offsets);
}

std::optional<WasmBoundsCheckEliminationAnalyzer::Relation>
WasmBoundsCheckEliminationAnalyzer::Normalize(OpIndex condition,
                                              bool holds) const {
  using Kind = ComparisonOp::Kind;
  const ComparisonOp* comparison =
      graph_.Get(condition).TryCast<ComparisonOp>();
  if (!comparison || comparison->rep != RegisterRepresentation::Word32()) {
    return std::nullopt;
  }
  // Equality tests show nothing about bounds.
  if (comparison->kind == Kind::kEqual) return std::nullopt;
  if (holds) {
    return Relation{comparison->kind, comparison->left(), comparison->right()};
  }
  switch (comparison->kind) {
    case Kind::kEqual:
      UNREACHABLE();
    // !(a < b) is (b <= a), and !(a <= b) is (b < a).
    case Kind::kSignedLessThan:
      return Relation{Kind::kSignedLessThanOrEqual, comparison->right(),
                      comparison->left()};
    case Kind::kSignedLessThanOrEqual:
      return Relation{Kind::kSignedLessThan, comparison->right(),
                      comparison->left()};
    case Kind::kUnsignedLessThan:
      return Relation{Kind::kUnsignedLessThanOrEqual, comparison->right(),
                      comparison->left()};
    case Kind::kUnsignedLessThanOrEqual:
      return Relation{Kind::kUnsignedLessThan, comparison->right(),
                      comparison->left()};
  }
  UNREACHABLE();
}

// If {base + n} is known to be non-negative for some {n} below
// {range}, whose offsets are within bounds, then all the offsets from
// {n} to {range.upper()} are within bounds ([non-negative-and-check]).
// Returns this larger range for the lowest such {n}, if valid. It is
// always valid when the facts hold: {base + n} is then at least 0 and
// {base + range.upper()} less than 2^30. But it may not be in
// unreachable code, where facts may contradict each other, so the second
// check is needed.
std::optional<OffsetRange>
WasmBoundsCheckEliminationAnalyzer::ExtendDownToNonNegative(
    const OffsetRange& range, const std::optional<OffsetRange>& non_negative) {
  if (!non_negative.has_value() ||
      !OffsetRange::IsValidRange(non_negative->lower(), range.lower()) ||
      !OffsetRange::IsValidRange(non_negative->lower(), range.upper())) {
    return std::nullopt;
  }
  return OffsetRange(non_negative->lower(), range.upper());
}

// Returns the offsets known to be within bounds for {key}, including
// the ones shown by non-negative offsets below them, even if we learnt
// that these offsets are non-negative after the bounds checks, and, for
// a constant index, the ones below the min length of the array
// ([min-length-constant-index]).
std::optional<OffsetRange> WasmBoundsCheckEliminationAnalyzer::KnownOffsets(
    const BoundsCheckKey& key,
    const std::optional<OffsetRange>& non_negative) const {
  std::optional<OffsetRange> known = known_bounds_checks_.Get(key);
  if (!key.base.valid()) {
    // Both ranges are within bounds, so their hull is valid ([covered]).
    if (uint32_t min_length = MinLength(key.length); min_length > 0) {
      OffsetRange below_min_length(0, min_length - 1);
      known = known.has_value() ? known->Hull(below_min_length).value_or(*known)
                                : below_min_length;
    }
  }
  if (!known.has_value()) return known;
  return ExtendDownToNonNegative(*known, non_negative).value_or(*known);
}

std::optional<OffsetRange>
WasmBoundsCheckEliminationAnalyzer::NonNegativeOffsets(OpIndex base) const {
  // Without a base, the index is the offset itself.
  if (!base.valid()) return OffsetRange(0, kMaxInt);
  return known_non_negative_offsets_.Get(base);
}

void WasmBoundsCheckEliminationAnalyzer::RecordNonNegativeOffset(
    OpIndex base, uint32_t offset) {
  std::optional<OffsetRange> known = known_non_negative_offsets_.Get(base);
  // Keep the previous range if the new offset is too far away from it
  // ([non-negative-range]).
  std::optional<OffsetRange> offsets = known.has_value()
                                           ? known->Hull(OffsetRange(offset))
                                           : OffsetRange(offset);
  if (!offsets.has_value() || offsets == known) return;
  TRACE("  Non-negative offsets for " << base << ": [" << offsets->lower()
                                      << ", " << offsets->upper() << "]");
  known_non_negative_offsets_.Set(base, *offsets);
}

uint32_t WasmBoundsCheckEliminationAnalyzer::MinLength(OpIndex length) const {
  return known_min_lengths_.Get(length).value_or(0);
}

void WasmBoundsCheckEliminationAnalyzer::RecordMinLength(OpIndex length,
                                                         uint64_t min_length) {
  std::optional<uint32_t> known = known_min_lengths_.Get(length);
  if (known.has_value() && min_length <= *known) return;
  // Array lengths are less than 2^31, so a condition implying a larger
  // length cannot hold, and we are in unreachable code.
  if (min_length > kMaxInt) return;
  TRACE("  Min length for " << length << ": " << min_length);
  known_min_lengths_.Set(length, static_cast<uint32_t>(min_length));
}

// Attempts to decode a bounds check condition of the form:
//     n <u a.length - r   or   base + n <u a.length - r
// where {r} is a constant (usually 0).
std::optional<WasmBoundsCheckEliminationAnalyzer::BoundsCheckCondition>
WasmBoundsCheckEliminationAnalyzer::TryExtractBoundsCheckCondition(
    const Relation& relation) const {
  if (relation.kind != ComparisonOp::Kind::kUnsignedLessThan) {
    return std::nullopt;
  }
  std::optional<ReducedLength> length = TryExtractArrayLength(relation.right);
  if (!length) return std::nullopt;
  auto [base, base_value, offset] = ExtractBaseAndOffset(relation.left);
  return BoundsCheckCondition{
      BoundsCheck(BoundsCheckKey(base, LengthKey(length->length)), offset),
      length->reduction};
}

// Recognizes {a.length - r} for a small constant {r} (possibly written
// {a.length + (-r)}), where {a.length} is an ArrayLength operation.
std::optional<ReducedLength>
WasmBoundsCheckEliminationAnalyzer::TryExtractArrayLength(
    OpIndex length) const {
  BaseAndOffset b = DecomposeIndex(length);
  if (!b.base_value.valid() || !graph_.Get(b.base_value).Is<ArrayLengthOp>()) {
    return std::nullopt;
  }
  uint32_t reduction = 0u - b.offset;
  // Only small reductions are useful (see "Limits" in the header).
  if (reduction > kMaxReduction) return std::nullopt;
  return ReducedLength{b.base_value, reduction};
}

// The key of an array length: its canonical value, which is the same for
// all the ArrayLength operations of an array, and for the length given
// to {array.new} when load elimination replaced the length of the new
// array by it.
OpIndex WasmBoundsCheckEliminationAnalyzer::LengthKey(OpIndex length) const {
  return CanonicalValue(length);
}

// Whether {length} is {a.length - r}, with {a.length} known to be at
// least {r}, so that it does not wrap around and is less than 2^31.
bool WasmBoundsCheckEliminationAnalyzer::IsArrayLengthWithoutWrapAround(
    OpIndex length) const {
  std::optional<ReducedLength> array_length = TryExtractArrayLength(length);
  return array_length.has_value() &&
         (array_length->reduction == 0 ||
          MinLength(LengthKey(array_length->length)) >=
              array_length->reduction);
}

// Attempts to decode a condition showing that {base + n} is
// non-negative ([non-negative-index]):
//     base + n < b    or   base + n <= b   (unsigned)
// where {b} is an array length (possibly minus a constant, if the
// length is known to be at least that constant), or a small enough
// constant.
std::optional<WasmBoundsCheckEliminationAnalyzer::BaseAndOffset>
WasmBoundsCheckEliminationAnalyzer::TryExtractNonNegativeIndex(
    const Relation& relation) const {
  DCHECK(!relation.is_signed());
  // The index is at most {b}, which must be less than 2^31, or at most
  // {b - 1} for a strict comparison. Array lengths are less than 2^31,
  // and so are reduced array lengths that do not wrap around.
  if (!IsArrayLengthWithoutWrapAround(relation.right)) {
    std::optional<uint32_t> bound = TryExtractI32Const(relation.right);
    if (!bound.has_value()) return std::nullopt;
    uint32_t limit = relation.is_strict() ? 1u << 31 : kMaxInt;
    if (*bound > limit) return std::nullopt;
  }
  BaseAndOffset result = ExtractBaseAndOffset(relation.left);
  if (!result.base.valid()) return std::nullopt;
  return result;
}

// Extracts the base and the constant offset of an index expression; the
// base is the canonical value of the decomposed base.
WasmBoundsCheckEliminationAnalyzer::BaseAndOffset
WasmBoundsCheckEliminationAnalyzer::ExtractBaseAndOffset(OpIndex index) const {
  BaseAndOffset result = DecomposeIndex(index);
  if (result.base_value.valid()) {
    result.base = CanonicalValue(result.base_value);
  }
  return result;
}

// Decomposes {index} as {base_value + offset}, without computing the
// canonical value of the base, which is left invalid.
WasmBoundsCheckEliminationAnalyzer::BaseAndOffset
WasmBoundsCheckEliminationAnalyzer::DecomposeIndex(OpIndex index) const {
  // Nested additions of constants, as produced for instance by loop
  // unrolling, are folded into a single offset, so that
  // {(base + n) + m} and {base + (n + m)} have the same base. We are
  // using modular arithmetic, so a subtraction can be replaced by the
  // addition of the opposite. The depth is bounded to keep the cost
  // linear.
  static constexpr int kMaxDecompositionDepth = 8;
  uint32_t offset = 0;
  for (int depth = 0;; depth++) {
    // An array length is kept as the base, even when load elimination
    // replaced it (for instance, by the length given to {array.new}), so
    // that conditions on it are recognized as bounds checks (see
    // {TryExtractArrayLength}). Its canonical value is still the one of
    // its replacement.
    if (graph_.Get(index).Is<ArrayLengthOp>()) break;
    index = ResolveReplacements(index);
    // a[n]
    if (auto constant = TryExtractI32Const(index)) {
      return {OpIndex::Invalid(), OpIndex::Invalid(), offset + *constant};
    }
    if (depth == kMaxDecompositionDepth) break;
    // a[trunc(base64 + n)]: languages with 64-bit integers compute the
    // index of {a[i + 1]} as {trunc(i + 1)}, which is {trunc(i) + 1}
    // ([truncated-addition]). The base is then an operation computing
    // the truncation of a value with the same canonical value as {i},
    // which must already have been seen (as for the index of a previous
    // access {a[i]}), possibly elsewhere: it is only used through its
    // canonical value (see {CanonicalValue}).
    if (const ChangeOp* truncation =
            graph_.Get(index).TryCast<Opmask::kTruncateWord64ToWord32>()) {
      const WordBinopOp* op =
          graph_.Get(ResolveReplacements(truncation->input()))
              .TryCast<WordBinopOp>();
      if (op == nullptr || op->rep != WordRepresentation::Word64() ||
          (op->kind != WordBinopOp::Kind::kAdd &&
           op->kind != WordBinopOp::Kind::kSub)) {
        break;
      }
      std::optional<uint32_t> constant = TryExtractI64ConstLow(op->right());
      if (!constant.has_value()) break;
      OpIndex base = ExistingTruncation(op->left());
      if (!base.valid()) break;
      offset += op->kind == WordBinopOp::Kind::kAdd ? *constant : -*constant;
      index = base;
      continue;
    }
    // a[base + n] / a[base - n]
    const WordBinopOp* op = graph_.Get(index).TryCast<WordBinopOp>();
    if (op == nullptr || op->rep != WordRepresentation::Word32()) break;
    std::optional<uint32_t> constant = TryExtractI32Const(op->right());
    if (!constant.has_value()) break;
    if (op->kind == WordBinopOp::Kind::kAdd) {
      offset += *constant;
    } else if (op->kind == WordBinopOp::Kind::kSub) {
      offset -= *constant;
    } else {
      break;
    }
    index = op->left();
  }
  // Default: a[base]
  return {OpIndex::Invalid(), index, offset};
}

// Returns a canonical representative of {value}: constants, and
// operations whose value only depends on their inputs (arithmetic,
// shifts, conversions, the length of an array, and the bitcast of a
// Smi or of a word) with the same options and inputs with the same canonical
// values compute the same value, and so have the same canonical value. Value
// numbering, which runs before this phase, merges most duplicated
// computations, but not those on values that load elimination finds
// equal in this phase, nor those on bitcasts of a reference to a heap
// object, which it never merges (see the TaggedBitcast case below), as
// for an {i31.get_s} of the same reference done once for an OCaml bounds
// check and once for the access it protects. Casts, non-null assertions
// and type annotations return their input, and values replaced by load
// elimination are the same as their replacement, which dominates them,
// is equal to them, and is what the output graph computes.
//
// Facts about canonical values are sound because a fact recorded at a
// point {p} is only used at points {q} dominated by {p} (see
// {BeginBlock}), and none of its leaves (the operations whose canonical
// value is not structural: phis, loads, parameters...) is defined at a
// point strictly dominated by {p}: they are defined in dominators of
// {p}. Such a leaf is then not executed again between the last
// execution of {p} and {q}: as {p} does not dominate it, there is a path
// to it that avoids {p}, which, followed by the execution from the leaf
// to {q}, would reach {q} without going through {p}. By induction on
// canonical values, a value used at {q} (and so computed in a dominator
// of {q}) is equal to its canonical value, computed from the current
// values of its leaves, and so two values used at {q} with the same
// canonical value are equal. This holds with the representatives of
// structural keys, and with the depth limit below, whose cut-off values
// are leaves, computed in dominators of the values that use them. The
// base of a truncated addition found by {ExistingTruncation} (see
// {DecomposeIndex}), which may be computed in another branch, is only
// used through its canonical value, the truncation of the canonical
// value of the index's input: its leaves are those of that input. Phis
// never get a structural key: two phis of different merges with the
// same inputs are different values (a phi that load elimination
// replaces by the value of all its inputs is that value, which
// dominates it).
OpIndex WasmBoundsCheckEliminationAnalyzer::CanonicalValue(OpIndex value,
                                                           int depth) const {
  // Bound the depth of the recursion. Deeper values are their own
  // canonical value, which is always correct. They are not memoized, so
  // that they may get another canonical value when reached at a lower
  // depth: two operations computing the same value may then get
  // different canonical values, which only loses precision.
  static constexpr int kMaxCanonicalDepth = 16;
  value = ResolveReplacements(value);
  if (auto it = canonical_values_.find(value); it != canonical_values_.end()) {
    return it->second;
  }
  if (depth == kMaxCanonicalDepth) return value;
  const Operation& op = graph_.Get(value);
  std::optional<StructuralKey> key;
  switch (op.opcode) {
    case Opcode::kWasmTypeCast:
    case Opcode::kAssertNotNull:
    case Opcode::kWasmTypeAnnotation: {
      // {ResolveAliases} goes through at least this operation.
      OpIndex object = ResolveAliases(value);
      DCHECK_NE(object, value);
      OpIndex result = CanonicalValue(object, depth + 1);
      canonical_values_.emplace(value, result);
      return result;
    }
    case Opcode::kConstant: {
      const ConstantOp& constant = op.Cast<ConstantOp>();
      if (constant.kind != ConstantOp::Kind::kWord32 &&
          constant.kind != ConstantOp::Kind::kWord64) {
        break;
      }
      key = StructuralKey{op.opcode, static_cast<uint64_t>(constant.kind),
                          constant.integral(), OpIndex::Invalid(),
                          OpIndex::Invalid()};
      break;
    }
    case Opcode::kWordBinop: {
      const WordBinopOp& binop = op.Cast<WordBinopOp>();
      OpIndex left = CanonicalValue(binop.left(), depth + 1);
      OpIndex right = CanonicalValue(binop.right(), depth + 1);
      // {a + b} and {b + a} compute the same value.
      if (WordBinopOp::IsCommutative(binop.kind) && right < left) {
        std::swap(left, right);
      }
      key = StructuralKey{op.opcode,
                          static_cast<uint64_t>(binop.kind) |
                              static_cast<uint64_t>(binop.rep.value()) << 8,
                          0, left, right};
      break;
    }
    case Opcode::kShift: {
      const ShiftOp& shift = op.Cast<ShiftOp>();
      key = StructuralKey{op.opcode,
                          static_cast<uint64_t>(shift.kind) |
                              static_cast<uint64_t>(shift.rep.value()) << 8,
                          0, CanonicalValue(shift.left(), depth + 1),
                          CanonicalValue(shift.right(), depth + 1)};
      break;
    }
    case Opcode::kChange: {
      const ChangeOp& change = op.Cast<ChangeOp>();
      OpIndex input = CanonicalValue(change.input(), depth + 1);
      key = StructuralKey{op.opcode, ChangeOptions(change), 0, input,
                          OpIndex::Invalid()};
      if (change.Is<Opmask::kTruncateWord64ToWord32>()) {
        truncations_.emplace(input, value);
      }
      break;
    }
    case Opcode::kArrayLength:
      // The length of an array never changes, so the ArrayLength
      // operations of the same array compute the same value.
      key = StructuralKey{
          op.opcode, 0, 0,
          CanonicalValue(op.Cast<ArrayLengthOp>().array(), depth + 1),
          OpIndex::Invalid()};
      break;
    case Opcode::kTaggedBitcast: {
      const TaggedBitcastOp& bitcast = op.Cast<TaggedBitcastOp>();
      // The bitcast of a reference to a heap object is its address, which
      // a moving GC can change, so two bitcasts of the same reference may
      // give different values. This is not the case for a Smi, such as a
      // non-null i31 reference, which {i31.get} converts this way, nor for
      // a bitcast from a word, whose bits only depend on the word.
      if (bitcast.from.IsTaggedOrCompressed() &&
          bitcast.kind != TaggedBitcastOp::Kind::kSmi &&
          !IsKnownSmi(bitcast.input())) {
        break;
      }
      key = StructuralKey{op.opcode,
                          static_cast<uint64_t>(bitcast.kind) |
                              static_cast<uint64_t>(bitcast.from.value()) << 8 |
                              static_cast<uint64_t>(bitcast.to.value()) << 16,
                          0, CanonicalValue(bitcast.input(), depth + 1),
                          OpIndex::Invalid()};
      break;
    }
    default:
      break;
  }
  OpIndex result = value;
  if (key.has_value()) {
    result = values_by_structure_.emplace(*key, value).first->second;
  }
  canonical_values_.emplace(value, result);
  return result;
}

// Whether {object} is known to be a Smi: a non-null i31 reference, as
// shown by a cast, a type annotation or a null check.
bool WasmBoundsCheckEliminationAnalyzer::IsKnownSmi(OpIndex object) const {
  bool non_null = false;
  while (true) {
    object = ResolveReplacements(object);
    const Operation& op = graph_.Get(object);
    wasm::ValueType type;
    if (const WasmTypeCastOp* cast = op.TryCast<WasmTypeCastOp>()) {
      type = cast->config.to;
      object = cast->object();
    } else if (const AssertNotNullOp* check = op.TryCast<AssertNotNullOp>()) {
      type = check->type;
      non_null = true;
      object = check->object();
    } else if (const WasmTypeAnnotationOp* annotation =
                   op.TryCast<WasmTypeAnnotationOp>()) {
      type = annotation->type;
      object = annotation->value();
    } else {
      return false;
    }
    if (type.is_reference_to(wasm::GenericKind::kI31) &&
        (non_null || type.is_non_nullable())) {
      return true;
    }
  }
}

// The options of a Change, in the structural key of its canonical value.
uint64_t WasmBoundsCheckEliminationAnalyzer::ChangeOptions(
    const ChangeOp& change) {
  return static_cast<uint64_t>(change.kind) |
         static_cast<uint64_t>(change.assumption) << 8 |
         static_cast<uint64_t>(change.from.value()) << 16 |
         static_cast<uint64_t>(change.to.value()) << 24;
}

// A TruncateWord64ToWord32 operation of {x} (or of a value with the same
// canonical value), if one has been seen already (see {CanonicalValue}).
// All the truncations of {x} have the same canonical value (up to the
// depth limit of {CanonicalValue}, which only loses precision).
OpIndex WasmBoundsCheckEliminationAnalyzer::ExistingTruncation(
    OpIndex x) const {
  auto it = truncations_.find(CanonicalValue(x));
  return it == truncations_.end() ? OpIndex::Invalid() : it->second;
}

// The low 32 bits of a 64-bit constant.
std::optional<uint32_t>
WasmBoundsCheckEliminationAnalyzer::TryExtractI64ConstLow(OpIndex expr) const {
  if (const ConstantOp* constant = graph_.Get(expr).TryCast<ConstantOp>();
      constant && constant->kind == ConstantOp::Kind::kWord64) {
    return static_cast<uint32_t>(constant->integral());
  }
  return std::nullopt;
}

std::optional<uint32_t> WasmBoundsCheckEliminationAnalyzer::TryExtractI32Const(
    OpIndex expr) const {
  if (const ConstantOp* constant = graph_.Get(expr).TryCast<ConstantOp>();
      constant && constant->kind == ConstantOp::Kind::kWord32) {
    return constant->word32();
  }
  return std::nullopt;
}

// Values replaced by load elimination are equal to their replacement.
OpIndex WasmBoundsCheckEliminationAnalyzer::ResolveReplacements(
    OpIndex value) const {
  if (load_elimination_ == nullptr) return value;
  while (true) {
    OpIndex replacement = load_elimination_->Replacement(value);
    // A struct.set is "replaced" by itself when it is unreachable.
    if (!replacement.valid() || replacement == value) return value;
    value = replacement;
  }
}

// Casts, non-null assertions and type annotations return their input.
OpIndex WasmBoundsCheckEliminationAnalyzer::ResolveAliases(
    OpIndex object) const {
  while (true) {
    object = ResolveReplacements(object);
    const Operation* op = &graph_.Get(object);
    switch (op->opcode) {
      case Opcode::kWasmTypeCast:
        object = op->Cast<WasmTypeCastOp>().object();
        break;
      case Opcode::kAssertNotNull:
        object = op->Cast<AssertNotNullOp>().object();
        break;
      case Opcode::kWasmTypeAnnotation:
        object = op->Cast<WasmTypeAnnotationOp>().value();
        break;
      default:
        return object;
    }
  }
}

#undef TRACE

}  // namespace v8::internal::compiler::turboshaft
