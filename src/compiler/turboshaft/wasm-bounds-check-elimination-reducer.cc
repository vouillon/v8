// Copyright 2026 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "src/compiler/turboshaft/wasm-bounds-check-elimination-reducer.h"

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
    // later iterations: these facts were recorded in dominators of the
    // header, and facts are never propagated along back edges.
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
  // recorded in its dominators (see "General overview" in the header).
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
  auto [base, offset] = DecomposeIndex(relation.left);
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
  if (!b.base.valid() || !graph_.Get(b.base).Is<ArrayLengthOp>()) {
    return std::nullopt;
  }
  uint32_t reduction = 0u - b.offset;
  // Only small reductions are useful (see "Limits" in the header).
  if (reduction > kMaxReduction) return std::nullopt;
  return ReducedLength{b.base, reduction};
}

// The key of the length {length}, an ArrayLength operation: its array,
// through casts, non-null assertions, type annotations and replacements
// by load
// elimination, so that all the ArrayLength operations of an array have
// the same key. The length of an array never changes.
OpIndex WasmBoundsCheckEliminationAnalyzer::LengthKey(OpIndex length) const {
  return ResolveAliases(graph_.Get(length).Cast<ArrayLengthOp>().array());
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
  BaseAndOffset result = DecomposeIndex(relation.left);
  if (!result.base.valid()) return std::nullopt;
  return result;
}

// Decomposes {index} as {base + offset}, where {base} is the operation
// computing the base, which is also its key.
WasmBoundsCheckEliminationAnalyzer::BaseAndOffset
WasmBoundsCheckEliminationAnalyzer::DecomposeIndex(OpIndex index) const {
  // An array length is kept as the base, even when load elimination
  // replaced it (for instance, by the length given to {array.new}), so
  // that conditions on it are recognized as bounds checks (see
  // {TryExtractArrayLength}).
  if (graph_.Get(index).Is<ArrayLengthOp>()) return {index, 0};
  index = ResolveReplacements(index);
  // a[n]
  if (auto constant = TryExtractI32Const(index)) {
    return {OpIndex::Invalid(), *constant};
  }
  // a[base + n] / a[base - n]. We are using modular arithmetic, so a
  // subtraction can be replaced by the addition of the opposite.
  if (const WordBinopOp* op = graph_.Get(index).TryCast<WordBinopOp>();
      op != nullptr && op->rep == WordRepresentation::Word32() &&
      (op->kind == WordBinopOp::Kind::kAdd ||
       op->kind == WordBinopOp::Kind::kSub)) {
    if (std::optional<uint32_t> constant = TryExtractI32Const(op->right())) {
      OpIndex base = op->left();
      if (!graph_.Get(base).Is<ArrayLengthOp>()) {
        base = ResolveReplacements(base);
      }
      return {base,
              op->kind == WordBinopOp::Kind::kAdd ? *constant : -*constant};
    }
  }
  // Default: a[base]
  return {index, 0};
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
