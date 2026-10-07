// Copyright 2026 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "src/compiler/turboshaft/wasm-bounds-check-elimination-reducer.h"

#include "src/base/small-vector.h"
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
  current_position_ = 0;
  for (OpIndex op_idx : graph_.OperationIndices(block)) {
    if (const TrapIfOp* trap_if = graph_.Get(op_idx).TryCast<TrapIfOp>();
        trap_if && !ShouldSkipOptimizationStep()) {
      ProcessTrapIf(op_idx, *trap_if);
    }
    current_position_++;
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
  ProcessCondition(is_bounds_check ? op_idx : OpIndex::Invalid(),
                   trap_if.condition(), trap_if.negated);
}

void WasmBoundsCheckEliminationAnalyzer::BeginBlock(const Block* block) {
  current_block_ = block;
  // Collect the snapshots of all predecessors.
  predecessor_bounds_check_snapshots_.clear();
  predecessor_non_negative_snapshots_.clear();
  predecessor_min_length_snapshots_.clear();
  predecessor_length_alias_snapshots_.clear();
  bool all_predecessors_have_length_aliases = true;
  for (const Block* p : block->PredecessorsIterable()) {
    std::optional<Snapshot> pred_snapshots =
        block_to_snapshot_mapping_[p->index()];
    // When we visit a loop header, its back edge hasn't been visited
    // yet, so we ignore it: what is known at the header is what is
    // known at the end of its forward predecessor. This remains true in
    // later iterations. These facts are about values computed before the
    // loop, which do not change in the loop (see {CanonicalValue}), and
    // facts are never propagated along back edges.
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
    if (pred_snapshots->length_aliases.has_value()) {
      predecessor_length_alias_snapshots_.push_back(
          *pred_snapshots->length_aliases);
    } else {
      all_predecessors_have_length_aliases = false;
    }
  }
  // Without merge functions, the new snapshots only contain what is
  // known in all predecessors, that is, in the common ancestor of the
  // predecessor snapshots. By induction, the ancestors of the snapshot
  // of a block are snapshots of blocks that dominate it, so this common
  // ancestor is the snapshot of a block that dominates all the (forward)
  // predecessors, and thus the current block: facts known at the start
  // of a block were recorded in its dominators, as {CanonicalValue}
  // requires.
  known_bounds_checks_.StartNewSnapshot(
      base::VectorOf(predecessor_bounds_check_snapshots_));
  known_non_negative_offsets_.StartNewSnapshot(
      base::VectorOf(predecessor_non_negative_snapshots_));
  known_min_lengths_.StartNewSnapshot(
      base::VectorOf(predecessor_min_length_snapshots_));
  // Length aliases are rare, so their table is only used once one has
  // been recorded. A predecessor without a snapshot of this table then
  // has no alias (and the current block is not dominated by the loop
  // header that recorded them).
  DCHECK(!length_aliases_open_);
  if (!alias_lengths_.empty()) {
    if (all_predecessors_have_length_aliases) {
      known_length_aliases_.StartNewSnapshot(
          base::VectorOf(predecessor_length_alias_snapshots_));
    } else {
      known_length_aliases_.StartNewSnapshot();
    }
    length_aliases_open_ = true;
  }

  if (block->IsLoop() && !ShouldSkipOptimizationStep()) {
    ProcessLoopHeader(block);
  }

  if (block->IsBranchTarget() && !ShouldSkipOptimizationStep()) {
    // The current block is a branch target, so we see if the branch
    // condition can be used as a bounds check.
    ProcessBranch(block);
  }
}

void WasmBoundsCheckEliminationAnalyzer::FinishBlock(const Block* block) {
  // Commit all profitable fallback sequences for this block.
  for (auto& [_, v] : planned_fallbacks_by_key_) {
    for (FallbackInstructionSequence& s : v) {
      if (!s.ShouldKeep()) {
        TRACE("  Discard fallback sequence: ["
              << s.start_index() << ", " << s.end_index()
              << "), not profitable (" << s.instruction_count()
              << " instructions, " << s.covered_traps().size() << " traps, "
              << s.ComputeGuard().check_count() << " guard checks)");
        continue;
      }
      TRACE("Commit fallback sequence: ["
            << s.start_index() << ", " << s.end_index() << "), "
            << s.covered_traps().size() << " traps");
      OpIndex start_index = s.start_index();
      auto [it, inserted] =
          fallback_sequence_starts_.try_emplace(start_index, std::move(s));
      if (!inserted) {
        // Sequences for different keys start at different traps, but
        // two sequences for the same key can start at the same trap
        // when they could not be coalesced. The second one is simply
        // not used: its traps are only removed once a sequence covering
        // them has been emitted.
        TRACE("  Discard fallback sequence starting at "
              << start_index << ": another sequence starts there");
      }
    }
  }
  planned_fallbacks_by_key_.clear();

  last_trap_bounds_checks_.clear();
  array_lengths_.clear();

  std::optional<LengthAliasMap::Snapshot> length_aliases;
  if (length_aliases_open_) {
    length_aliases = known_length_aliases_.Seal();
    length_aliases_open_ = false;
  }
  block_to_snapshot_mapping_[block->index()] =
      Snapshot{known_bounds_checks_.Seal(), known_non_negative_offsets_.Seal(),
               known_min_lengths_.Seal(), length_aliases};
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

  ProcessCondition(/*Not a TrapIf*/ OpIndex::Invalid(), branch->condition(),
                   block == branch->if_true);
}

// Records what we learn from {condition}, which holds if {holds}, and
// does not hold otherwise. For an array bounds check trap, {trap_if} is
// the trap. For a branch or another trap, it is OpIndex::Invalid().
void WasmBoundsCheckEliminationAnalyzer::ProcessCondition(OpIndex trap_if,
                                                          OpIndex condition,
                                                          bool holds) {
  ForEachCondition(condition, holds, [&](OpIndex part, bool part_holds) {
    // Only a trap whose condition is a single comparison can be
    // eliminated.
    ProcessComparison(part == condition ? trap_if : OpIndex::Invalid(), part,
                      part_holds);
  });
}

void WasmBoundsCheckEliminationAnalyzer::ProcessComparison(OpIndex trap_if,
                                                           OpIndex condition,
                                                           bool holds) {
  std::optional<Relation> relation = Normalize(condition, holds);
  if (!relation.has_value()) return;
  if (relation->not_equal) {
    ProcessNotEqual(*relation);
    return;
  }
  if (auto decoded = TryExtractBoundsCheckCondition(*relation)) {
    const auto& [bounds_check, base_value, array_length, reduction, is_signed] =
        *decoded;
    const auto& [key, offset] = bounds_check;
    // A signed comparison {x + n < a.length - r}, where {x + n} is known
    // to be non-negative, shows that {a.length - r} is positive, and so
    // that it does not wrap around, and that {x + n < a.length - r} as
    // unsigned integers ([signed-check]). For an unsigned comparison,
    // unless the length is known to be at least {r}, {a.length - r} may
    // wrap around, and we learn nothing.
    if (is_signed ? IsKnownNonNegative(key, offset)
                  : reduction == 0 || MinLength(key.array) >= reduction) {
      if (is_signed) TRACE("  Signed bounds check");
      // Only an unsigned comparison with the length itself is an actual
      // bounds check, which can be eliminated. Other conditions are only
      // used for what they show, even for a trap.
      ProcessBoundsCheck(
          !is_signed && reduction == 0 ? trap_if : OpIndex::Invalid(),
          bounds_check, base_value, array_length, reduction);
      // With {index < a.length - reduction}, the length is at least
      // {reduction + 1}, and more for a constant index
      // ([reduced-length-min-length], [constant-index-min-length]).
      uint64_t index_bound = key.base.valid() ? 0 : offset;
      if (index_bound <= kMaxInt) {
        RecordMinLength(key.array, index_bound + reduction + 1);
      }
    }
  }
  // This must be done after processing the bounds check: what is known
  // before a trap is used by the guard of a fallback sequence starting
  // at it, and the fast path does not include the trap. So this trap
  // must not show that its own index is non-negative.
  if (auto non_negative_index = TryExtractNonNegativeIndex(*relation)) {
    RecordNonNegativeOffset(non_negative_index->base,
                            non_negative_index->offset);
  }
}

// {x + k != a.length - r}: if {x + k + r - 1} is known to be within
// bounds, then so is {x + k + r} ([not-equal-length]).
void WasmBoundsCheckEliminationAnalyzer::ProcessNotEqual(
    const Relation& relation) {
  for (auto [index, length] : {std::pair{relation.left, relation.right},
                               std::pair{relation.right, relation.left}}) {
    std::optional<ReducedLength> array_length = TryExtractArrayLength(length);
    if (!array_length) continue;
    auto [base, base_value, offset] = ExtractBaseAndOffset(index);
    BoundsCheckKey key(base, ArrayOfLength(array_length->length));
    uint32_t last = offset + array_length->reduction - 1;
    std::optional<OffsetRange> known =
        KnownOffsets(key, NonNegativeOffsets(base));
    if (!known.has_value() || !known->Contains(last) ||
        known->Contains(last + 1)) {
      continue;
    }
    TRACE("  Not equal to the array length");
    ProcessBoundsCheck(OpIndex::Invalid(), BoundsCheck(key, last + 1),
                       base_value, array_length->length, 0);
    return;
  }
}

void WasmBoundsCheckEliminationAnalyzer::ProcessBoundsCheck(
    OpIndex trap_if, const BoundsCheck& bounds_check, OpIndex base_value,
    OpIndex array_length, uint32_t reduction) {
  const auto& [key, offset] = bounds_check;
  // Only plain bounds checks can be eliminated.
  DCHECK_IMPLIES(trap_if.valid(), reduction == 0);

  // Record the array length, so that we can use it when emitting new
  // bounds checks. Array length instructions can trap on null, so it
  // is simpler to keep using the very first one rather than inserting
  // a new one. A length known from {known_length_aliases_} may not be
  // available in the current block, and is then not recorded.
  if (!alias_lengths_.contains(array_length) ||
      Dominates(&graph_.Get(graph_.BlockOf(array_length)), current_block_)) {
    array_lengths_.try_emplace(key.array, array_length);
  }

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
  TRACE("  Known offsets for base " << key.base << " and array " << key.array
                                    << ": [" << offsets->lower() << ", "
                                    << offsets->upper() << "]");

  if (known.has_value() && trap_if.valid()) {
    // The trap that established the bound of {known} on the side of
    // {offset} can start a fallback sequence. For a trap, {reduction} is
    // 0, and {offset} is not in {known}, so it is above or below it.
    OffsetRange::RelativePosition position = known->Classify(offset);
    DCHECK(position == OffsetRange::RelativePosition::kAbove ||
           position == OffsetRange::RelativePosition::kBelow);
    uint32_t prev_offset = position == OffsetRange::RelativePosition::kAbove
                               ? known->upper()
                               : known->lower();
    RecordFallbackSequence(key, prev_offset, trap_if, offset);
  }
  UpdateKnownBoundsChecks(key, offset, *offsets, trap_if, base_value, known,
                          non_negative);
}

// Record a fallback sequence covering the previous trap at
// {previous_offset} and the current trap at {offset}.
void WasmBoundsCheckEliminationAnalyzer::RecordFallbackSequence(
    const BoundsCheckKey& key, uint32_t previous_offset, OpIndex trap_if,
    uint32_t offset) {
  DCHECK(trap_if.valid());

  auto last_trap_it =
      last_trap_bounds_checks_.find(BoundsCheck(key, previous_offset));
  if (last_trap_it == last_trap_bounds_checks_.end()) {
    // No previous trap found in this block. The prior bound comes
    // from a trap or a branch in a dominating block, or from a
    // non-negative offset. The fallback code must cover instructions
    // up to {trap_if} and thus remain entirely within this block.
    return;
  }
  const TrapInfo& previous = last_trap_it->second;

  // The guard is emitted at the start of the sequence, so it needs an
  // array length computed before it. The length recorded for the array
  // is usually the one used by the previous trap, or an earlier one. But
  // the previous trap may have used a length known from
  // {known_length_aliases_}, which is not recorded if not available in
  // this block, and a later condition may then have recorded another
  // length.
  auto array_length_it = array_lengths_.find(key.array);
  if (array_length_it == array_lengths_.end()) return;
  OpIndex array_length = array_length_it->second;
  if (graph_.BlockIndexOf(array_length) == current_block_->index() &&
      array_length >= previous.start_index) {
    return;
  }

  // Both offsets are within the known range, so their hull is valid.
  std::optional<OffsetRange> offsets =
      OffsetRange(previous_offset).Hull(OffsetRange(offset));
  DCHECK(offsets.has_value());

  OpIndex fallback_end_index = graph_.NextIndex(trap_if);

  TRACE("  Record fallback sequence:");
  TRACE("    Previous trap: " << previous.trap_if);
  TRACE("    Current trap: " << trap_if);
  TRACE("    Instruction range: [" << previous.start_index << ", "
                                   << fallback_end_index << ")");

  FallbackInstructionSequence new_sequence(
      phase_zone_, key, previous.base_value, array_length, *offsets,
      previous.known_offsets, previous.non_negative_offsets,
      previous.start_index, fallback_end_index, previous.start_position,
      current_position_ + 1, {previous.trap_if, trap_if});
  RegisterFallbackSequence(std::move(new_sequence));
}

// Registers a fallback sequence for the key contained in the sequence.
// Coalesces it with previous sequences if possible.
void WasmBoundsCheckEliminationAnalyzer::RegisterFallbackSequence(
    FallbackInstructionSequence&& sequence) {
  auto [it, inserted] =
      planned_fallbacks_by_key_.try_emplace(sequence.key(), phase_zone_);
  auto& sequences = it->second;

  sequences.emplace_back(std::move(sequence));

  // Coalesce with previous sequences while possible. Coalescing can
  // move the start of the sequence earlier, so it may then overlap
  // with more sequences.
  while (sequences.size() >= 2) {
    auto& previous_sequence = sequences[sequences.size() - 2];
    auto& current_sequence = sequences.back();
    if (!previous_sequence.CanCoalesce(current_sequence)) break;
    previous_sequence.Coalesce(current_sequence);
    TRACE("  Coalesced with previous sequence:");
    TRACE("    New range: [" << previous_sequence.start_index() << ", "
                             << previous_sequence.end_index() << ")");
    TRACE("    Traps: " << previous_sequence.covered_traps().size());
    sequences.pop_back();
  }
}

void WasmBoundsCheckEliminationAnalyzer::UpdateKnownBoundsChecks(
    const BoundsCheckKey& key, uint32_t offset, const OffsetRange& offsets,
    OpIndex trap_if, OpIndex base_value,
    const std::optional<OffsetRange>& known_before,
    const std::optional<OffsetRange>& non_negative_before) {
  // Records the updated bounds check range.
  known_bounds_checks_.Set(key, offsets);

  if (!trap_if.valid()) return;

  // We register a possible place to insert some fallback code. We
  // will also clone the condition when it is right before the trap,
  // since we don't want to keep it on the main sequence of execution.
  OpIndex condition = graph_.Get(trap_if).Cast<TrapIfOp>().condition();
  bool starts_at_condition = graph_.PreviousIndex(trap_if) == condition;
  // The operation before the first one of a block is the terminator of
  // another block, which is not a condition.
  DCHECK_IMPLIES(starts_at_condition, current_position_ > 0);
  OpIndex start_index = starts_at_condition ? condition : trap_if;
  uint32_t start_position = current_position_ - starts_at_condition;
  last_trap_bounds_checks_.insert_or_assign(
      BoundsCheck(key, offset),
      TrapInfo{trap_if, base_value, start_index, start_position, known_before,
               non_negative_before});
}

// If {base + n} is known to be non-negative for some {n} below
// {range}, whose offsets are within bounds, then all the offsets from
// {n} to {range.upper()} are within bounds ([non-negative-and-check]).
// Returns this larger range for the lowest such {n}, if valid.
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
// that these offsets are non-negative after the bounds checks.
std::optional<OffsetRange> WasmBoundsCheckEliminationAnalyzer::KnownOffsets(
    const BoundsCheckKey& key,
    const std::optional<OffsetRange>& non_negative) const {
  std::optional<OffsetRange> known = known_bounds_checks_.Get(key);
  if (!known.has_value()) return known;
  return ExtendDownToNonNegative(*known, non_negative).value_or(*known);
}

std::optional<OffsetRange>
WasmBoundsCheckEliminationAnalyzer::NonNegativeOffsets(OpIndex base) const {
  // Without a base, the index is the offset itself.
  if (!base.valid()) return OffsetRange(0, kMaxInt);
  return known_non_negative_offsets_.Get(base);
}

// Whether {key.base + offset} is known to be non-negative.
bool WasmBoundsCheckEliminationAnalyzer::IsKnownNonNegative(
    const BoundsCheckKey& key, uint32_t offset) const {
  std::optional<OffsetRange> non_negative = NonNegativeOffsets(key.base);
  if (non_negative.has_value() && non_negative->Contains(offset)) return true;
  if (!key.base.valid()) return false;
  // If {base + m} is within the bounds of an array, {base + m + d} is
  // non-negative for {0 <= d <= kMaxArrayLength}
  // ([non-negative-near-bounds]).
  std::optional<OffsetRange> known = known_bounds_checks_.Get(key);
  return known.has_value() && offset - known->upper() <= kMaxArrayLength;
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

uint32_t WasmBoundsCheckEliminationAnalyzer::MinLength(OpIndex array) const {
  return known_min_lengths_.Get(array).value_or(0);
}

void WasmBoundsCheckEliminationAnalyzer::RecordMinLength(OpIndex array,
                                                         uint64_t length) {
  if (length <= MinLength(array)) return;
  // Array lengths are less than 2^31, so a condition implying a larger
  // length cannot hold, and we are in unreachable code.
  if (length > kMaxInt) return;
  TRACE("  Min length for " << array << ": " << length);
  known_min_lengths_.Set(array, static_cast<uint32_t>(length));
}

// Attempts to decode a bounds check condition of the form:
//     n < a.length - r   or   base + n < a.length - r
// (signed or unsigned), where {r} is a constant (usually 0).
std::optional<WasmBoundsCheckEliminationAnalyzer::BoundsCheckCondition>
WasmBoundsCheckEliminationAnalyzer::TryExtractBoundsCheckCondition(
    const Relation& relation) const {
  if (relation.kind != ComparisonOp::Kind::kUnsignedLessThan &&
      relation.kind != ComparisonOp::Kind::kSignedLessThan) {
    return std::nullopt;
  }
  std::optional<ReducedLength> length = TryExtractArrayLength(relation.right);
  if (!length) return std::nullopt;
  auto [base, base_value, offset] = ExtractBaseAndOffset(relation.left);
  return BoundsCheckCondition{
      BoundsCheck(BoundsCheckKey(base, ArrayOfLength(length->length)), offset),
      base_value, length->length, length->reduction, relation.is_signed()};
}

template <typename F>
void WasmBoundsCheckEliminationAnalyzer::ForEachCondition(OpIndex condition,
                                                          bool holds,
                                                          const F& f,
                                                          int depth) const {
  // {a | b} does not hold when neither {a} nor {b} holds, {a & b} of two
  // comparisons (which are 0 or 1) holds when both hold, and {a == 0}
  // holds when {a} does not.
  static constexpr int kMaxDepth = 4;
  if (depth < kMaxDepth) {
    const Operation& op = graph_.Get(condition);
    if (const WordBinopOp* binop = op.TryCast<WordBinopOp>();
        binop && binop->rep == WordRepresentation::Word32()) {
      if (binop->kind == WordBinopOp::Kind::kBitwiseOr && !holds) {
        ForEachCondition(binop->left(), false, f, depth + 1);
        ForEachCondition(binop->right(), false, f, depth + 1);
        return;
      }
      if (binop->kind == WordBinopOp::Kind::kBitwiseAnd && holds &&
          graph_.Get(binop->left()).Is<ComparisonOp>() &&
          graph_.Get(binop->right()).Is<ComparisonOp>()) {
        ForEachCondition(binop->left(), true, f, depth + 1);
        ForEachCondition(binop->right(), true, f, depth + 1);
        return;
      }
    }
    if (const ComparisonOp* comparison = op.TryCast<ComparisonOp>();
        comparison && comparison->rep == RegisterRepresentation::Word32() &&
        comparison->kind == ComparisonOp::Kind::kEqual) {
      if (auto constant = TryExtractI32Const(comparison->right());
          constant.has_value() && *constant == 0) {
        ForEachCondition(comparison->left(), !holds, f, depth + 1);
        return;
      }
    }
  }
  f(condition, holds);
}

template <typename F>
void WasmBoundsCheckEliminationAnalyzer::ForEachDominatingCondition(
    const Block* block, const Block* stop, const F& f) const {
  for (const Block* b = block; b != nullptr && b != stop;
       b = b->GetDominator()) {
    if (!b->IsBranchTarget()) continue;
    const BranchOp* branch =
        b->LastPredecessor()->LastOperation(graph_).TryCast<BranchOp>();
    if (!branch) continue;
    ForEachCondition(branch->condition(), b == branch->if_true, f);
  }
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
  if (holds) {
    return Relation{comparison->kind, false, comparison->left(),
                    comparison->right()};
  }
  switch (comparison->kind) {
    case Kind::kEqual:
      return Relation{Kind::kEqual, true, comparison->left(),
                      comparison->right()};
    // !(a < b) is (b <= a), and !(a <= b) is (b < a).
    case Kind::kSignedLessThan:
      return Relation{Kind::kSignedLessThanOrEqual, false, comparison->right(),
                      comparison->left()};
    case Kind::kSignedLessThanOrEqual:
      return Relation{Kind::kSignedLessThan, false, comparison->right(),
                      comparison->left()};
    case Kind::kUnsignedLessThan:
      return Relation{Kind::kUnsignedLessThanOrEqual, false,
                      comparison->right(), comparison->left()};
    case Kind::kUnsignedLessThanOrEqual:
      return Relation{Kind::kUnsignedLessThan, false, comparison->right(),
                      comparison->left()};
  }
  UNREACHABLE();
}

bool WasmBoundsCheckEliminationAnalyzer::Dominates(const Block* dominator,
                                                   const Block* block) const {
  for (const Block* b = block; b != nullptr; b = b->GetDominator()) {
    if (b == dominator) return true;
  }
  return false;
}

// Loop induction. A loop phi {i = phi(c0, i + c)}, where {c0 >= 0} and
// {c > 0} are constants, is non-negative in the whole loop when the back
// edge is only taken if {i < x} or {i <= x}, for a bound {x} small
// enough that {i + c} cannot overflow (see {IsSmallBound}). It is also
// non-negative when the back edge is only taken if {i + k != x} for all
// {k} below {c} (for a small {c}), for a loop-invariant {x}, and the
// loop is only entered if {c0 <= x}: {i} is then at most {x} in the
// whole loop ([induction-not-equal]), which is a bounds check fact when
// {x} is an array length minus a constant (see {RecordLoopBound}).
//
// These facts are recorded at the header, so they must hold whenever the
// header is executed, which is shown by induction on the iterations.
// When the loop is entered, {i} is {c0}, which is non-negative, and at
// most {x} for a not-equal bound (see {EntryLowerBound}). When the back
// edge is taken, the conditions collected below held when they were
// tested: they are the branch conditions of the dominators of the back
// edge within the loop (branch targets have a single predecessor). The
// value of {i} they tested is the current one, since the header, where
// {i} is defined, is not executed again before the back edge. This
// value satisfied the facts (induction hypothesis), and the lemmas show
// that {i + c}, the value of {i} at the next iteration, does too. For a
// small bound, the lemmas hold for any value of {x}, so {x} can change
// in the loop. For a not-equal bound, {x} must be the same as when
// entering the loop, so it must be computed before the loop (see
// {FindNotEqualBound}). The facts then remain true in the blocks
// dominated by the header, including after the loop, as long as {i} is
// not computed again, that is, until the header is executed again (see
// {CanonicalValue}).
void WasmBoundsCheckEliminationAnalyzer::ProcessLoopHeader(
    const Block* header) {
  if (header->PredecessorCount() != 2) return;
  const Block* back_edge = header->LastPredecessor();
  const Block* forward = back_edge->NeighboringPredecessor();
  // The conditions that hold whenever the back edge is taken.
  base::SmallVector<Relation, 8> conditions;
  ForEachDominatingCondition(
      back_edge, header, [&](OpIndex condition, bool holds) {
        if (auto relation = Normalize(condition, holds)) {
          conditions.push_back(*relation);
        }
      });
  if (conditions.empty()) return;
  for (OpIndex index : graph_.OperationIndices(*header)) {
    const PhiOp* phi = graph_.Get(index).TryCast<PhiOp>();
    if (!phi) continue;
    std::optional<InductionVariable> induction =
        TryMatchInductionVariable(index, *phi);
    if (!induction.has_value()) continue;
    std::optional<LoopBound> bound =
        FindNotEqualBound(*induction, conditions, forward);
    if (!bound.has_value() && !HasSmallBound(*induction, conditions)) {
      continue;
    }
    TRACE("  Loop induction: "
          << index << " is non-negative ("
          << (bound.has_value() ? "not-equal bound" : "small bound") << ")");
    RecordNonNegativeOffset(induction->value, 0);
    if (bound.has_value()) RecordLoopBound(*induction, *bound);
  }
}

// Recognizes {i = phi(c0, i + c)}, where {c0 >= 0} and {0 < c <= 2^30}
// are constants.
std::optional<WasmBoundsCheckEliminationAnalyzer::InductionVariable>
WasmBoundsCheckEliminationAnalyzer::TryMatchInductionVariable(
    OpIndex index, const PhiOp& phi) const {
  if (phi.rep != RegisterRepresentation::Word32() || phi.input_count != 2) {
    return std::nullopt;
  }
  std::optional<uint32_t> init = TryExtractI32Const(phi.input(0));
  if (!init.has_value() || static_cast<int32_t>(*init) < 0) {
    return std::nullopt;
  }
  OpIndex value = CanonicalValue(index);
  BaseAndOffset step = ExtractBaseAndOffset(phi.input(1));
  if (step.base != value || step.offset == 0 || step.offset > kMaxArrayLength) {
    return std::nullopt;
  }
  return InductionVariable{value, *init, step.offset};
}

// Whether the back edge is only taken if {i < x} or {i <= x}, for a
// bound {x} small enough that {i + c} cannot overflow.
template <typename Conditions>
bool WasmBoundsCheckEliminationAnalyzer::HasSmallBound(
    const InductionVariable& induction, const Conditions& conditions) const {
  for (const Relation& relation : conditions) {
    if (relation.kind == ComparisonOp::Kind::kEqual) continue;
    BaseAndOffset left = ExtractBaseAndOffset(relation.left);
    if (left.base != induction.value || left.offset != 0) continue;
    if (IsSmallBound(relation.right, induction.step, relation.is_signed(),
                     !relation.is_strict())) {
      return true;
    }
  }
  return false;
}

// Whether {i < x} (or {i <= x} if {inclusive}), for a non-negative
// {i}, shows that {i + step} does not overflow ([induction-constant],
// [induction-length]).
bool WasmBoundsCheckEliminationAnalyzer::IsSmallBound(OpIndex x, uint32_t step,
                                                      bool is_signed,
                                                      bool inclusive) const {
  if (auto constant = TryExtractI32Const(x)) {
    // {i < x <= 2^31 - step} (or {i <= x <= 2^31 - 1 - step}), so that
    // {i + step <= 2^31 - 1}. A negative {x} is fine for a signed
    // comparison: the back edge is then never taken.
    uint32_t limit = uint32_t{kMaxInt} - step + (inclusive ? 0 : 1);
    return is_signed
               ? static_cast<int32_t>(*constant) <= static_cast<int32_t>(limit)
               : *constant <= limit;
  }
  // {x = a.length - r} is less than 2^30 as a signed integer. As an
  // unsigned integer, it may wrap around unless {r} is 0.
  std::optional<ReducedLength> length = TryExtractArrayLength(x);
  return length.has_value() && (is_signed || length->reduction == 0);
}

// Looks for a loop-invariant {x} such that the back edge is only taken
// if {i + k != x} for all {k} below the step, as after unrolling a loop
// with a step of 1, and such that the loop is only entered if
// {c0 <= x}. Then, {i} never goes past {x} ([induction-not-equal]).
template <typename Conditions>
std::optional<WasmBoundsCheckEliminationAnalyzer::LoopBound>
WasmBoundsCheckEliminationAnalyzer::FindNotEqualBound(
    const InductionVariable& induction, const Conditions& conditions,
    const Block* forward) const {
  static constexpr uint32_t kMaxUnrolledStep = 16;
  if (induction.step > kMaxUnrolledStep) return std::nullopt;
  // For each candidate {x} (by canonical value): an operation computing
  // it, and the offsets {k} with {i + k != x}, as a bit mask.
  struct Candidate {
    OpIndex x;
    OpIndex x_value;
    uint32_t offsets;
  };
  base::SmallVector<Candidate, 4> candidates;
  for (const Relation& relation : conditions) {
    if (!relation.not_equal) continue;
    for (auto [index, other] : {std::pair{relation.left, relation.right},
                                std::pair{relation.right, relation.left}}) {
      BaseAndOffset i = ExtractBaseAndOffset(index);
      if (i.base != induction.value) continue;
      if (i.offset < induction.step) {
        OpIndex x = CanonicalValue(other);
        auto it = std::find_if(candidates.begin(), candidates.end(),
                               [&](const Candidate& c) { return c.x == x; });
        if (it == candidates.end()) {
          candidates.push_back({x, other, 0});
          it = candidates.end() - 1;
        }
        it->offsets |= 1u << i.offset;
      }
      break;
    }
  }
  uint32_t all_offsets = (1u << induction.step) - 1;
  for (const Candidate& candidate : candidates) {
    if (candidate.offsets != all_offsets) continue;
    // {x} must be computed before the loop, so that it is the same in
    // the whole loop. In practice, {EntryLowerBound} only finds a lower
    // bound for a value computed before the loop, but this is not
    // obvious, so we check it.
    if (!TryExtractI32Const(candidate.x_value).has_value() &&
        !Dominates(&graph_.Get(graph_.BlockOf(candidate.x_value)), forward)) {
      continue;
    }
    std::optional<int64_t> lower =
        EntryLowerBound(candidate.x, candidate.x_value, forward);
    if (lower.has_value() && static_cast<int32_t>(induction.init) <= *lower) {
      return LoopBound{candidate.x_value, *lower};
    }
  }
  return std::nullopt;
}

// A lower bound of {x} (the canonical value of {x_value}) when entering
// the loop from {forward}: {k <= x} or {k < x} (signed), raised past the
// values that {x} is known to differ from ({x != k}, or a branch on
// {x}, which shows that {x != 0}).
std::optional<int64_t> WasmBoundsCheckEliminationAnalyzer::EntryLowerBound(
    OpIndex x, OpIndex x_value, const Block* forward) const {
  std::optional<int64_t> lower;
  if (auto constant = TryExtractI32Const(x_value)) {
    lower = static_cast<int32_t>(*constant);
  }
  base::SmallVector<int64_t, 4> excluded;
  ForEachDominatingCondition(
      forward, nullptr, [&](OpIndex condition, bool holds) {
        std::optional<Relation> relation = Normalize(condition, holds);
        if (!relation.has_value()) {
          if (holds && CanonicalValue(condition) == x) excluded.push_back(0);
          return;
        }
        if (relation->kind == ComparisonOp::Kind::kEqual) {
          if (!relation->not_equal) return;
          OpIndex other = CanonicalValue(relation->left) == x ? relation->right
                          : CanonicalValue(relation->right) == x
                              ? relation->left
                              : OpIndex::Invalid();
          if (!other.valid()) return;
          if (auto k = TryExtractI32Const(other)) {
            excluded.push_back(static_cast<int32_t>(*k));
          }
          return;
        }
        if (!relation->is_signed() || CanonicalValue(relation->right) != x) {
          return;
        }
        auto k = TryExtractI32Const(relation->left);
        if (!k.has_value()) return;
        int64_t bound =
            static_cast<int32_t>(*k) + (relation->is_strict() ? 1 : 0);
        if (!lower.has_value() || bound > *lower) lower = bound;
      });
  if (!lower.has_value()) return std::nullopt;
  int64_t bound = *lower;
  while (std::find(excluded.begin(), excluded.end(), bound) != excluded.end()) {
    bound++;
  }
  return bound;
}

// With {0 <= i <= x} in the loop, if {x} is {a.length - r} with
// {r >= 1}, the offsets 0 to {r - 1} of {i} are within bounds
// ([induction-not-equal-bounds]).
void WasmBoundsCheckEliminationAnalyzer::RecordLoopBound(
    const InductionVariable& induction, const LoopBound& bound) {
  std::optional<ReducedLength> length = TryExtractArrayLength(bound.x);
  if (!length.has_value()) length = TryResolveMergeBound(bound);
  if (!length.has_value() || length->reduction == 0) return;
  OpIndex array = ArrayOfLength(length->length);
  uint32_t last = length->reduction - 1;
  TRACE("  Loop induction: offsets [0, " << last << "] of " << induction.value
                                         << " within bounds of " << array);
  BoundsCheckKey key(induction.value, array);
  // The loop phi is defined in the loop header, so nothing is known yet
  // about it.
  DCHECK(!known_bounds_checks_.Get(key).has_value());
  known_bounds_checks_.Set(key, OffsetRange(0, last));
  RecordMinLength(array, uint64_t{last} + 1);
}

// With {x = phi(...) + offset}, where the phi is a merge (not a loop
// phi), and {x} at least {x_lower_bound} when entering the loop, the
// constant inputs {k} of the phi with {k + offset < x_lower_bound} are
// ruled out in the loop. If a single input remains, {x} is this input
// plus {offset} in the loop. For instance, for the length of an array
// that may be empty (a constant 0 on the other path) minus 1, and a loop
// that is only entered if it is non-negative. If {x} is then an array
// length minus a constant, this is recorded as a length alias, which
// holds in the blocks dominated by the loop header.
//
// The alias holds in a block {b} dominated by the header, although the
// length {l} in the remaining input does not dominate {b}. Consider an
// execution reaching {b}. The merge block {m} dominates {forward} (it
// dominates {x}, which does), which dominates {b} (the header's other
// predecessor is the back edge). The last execution of {m} before {b}
// is thus followed by an execution of {forward}, and {m} is not
// executed again in between (any path from {m} to the header goes
// through {forward}). When {forward} was last executed, the conditions
// dominating it held, so {x} was at least {x_lower_bound}, and that
// execution of {m} took the edge of the remaining input, which then
// had the value of {l} plus a constant. Neither {l} nor its array is
// computed again before {b}. Otherwise, as {m} is not a loop header, it
// does not dominate the block {d} where they are computed ({d} dominates
// the predecessor of {m} for the remaining input). There would then be
// a path to {d} that avoids {m}, followed by a path from {d} to {b}
// that avoids {m}, while {m} dominates {b}. So in {b}, {x} is {l} minus
// a constant, for the current values of {l} and of its array, which is
// what facts about the array refer to.
std::optional<ReducedLength>
WasmBoundsCheckEliminationAnalyzer::TryResolveMergeBound(
    const LoopBound& bound) {
  BaseAndOffset x = ExtractBaseAndOffset(bound.x);
  if (!x.base.valid()) return std::nullopt;
  const PhiOp* merge = graph_.Get(x.base).TryCast<PhiOp>();
  if (!merge || graph_.Get(graph_.BlockOf(x.base)).IsLoop()) {
    return std::nullopt;
  }
  OpIndex remaining = OpIndex::Invalid();
  for (OpIndex input : merge->inputs()) {
    if (auto k = TryExtractI32Const(input)) {
      int64_t candidate = static_cast<int32_t>(*k + x.offset);
      if (candidate < bound.x_lower_bound) continue;
    }
    if (remaining.valid()) return std::nullopt;
    remaining = input;
  }
  if (!remaining.valid()) return std::nullopt;
  BaseAndOffset input = DecomposeIndex(remaining);
  if (!input.base_value.valid() ||
      !graph_.Get(input.base_value).Is<ArrayLengthOp>()) {
    return std::nullopt;
  }
  ReducedLength length{input.base_value, 0u - (input.offset + x.offset)};
  if (length.reduction > kMaxReduction) return std::nullopt;
  if (!length_aliases_open_) {
    // The first alias: no block had any before.
    known_length_aliases_.StartNewSnapshot();
    length_aliases_open_ = true;
  }
  TRACE("  Length alias: " << bound.x << " is " << length.length << " - "
                           << length.reduction << " in the loop");
  known_length_aliases_.Set(CanonicalValue(bound.x), length);
  alias_lengths_.insert(length.length);
  return length;
}

// Recognizes {a.length - r} for a small constant {r} (possibly written
// {a.length + (-r)}, or with nested additions of constants), and values
// known from {known_length_aliases_} to be equal to such an expression.
std::optional<ReducedLength>
WasmBoundsCheckEliminationAnalyzer::TryExtractArrayLength(
    OpIndex length) const {
  BaseAndOffset b = DecomposeIndex(length);
  if (b.base_value.valid() && graph_.Get(b.base_value).Is<ArrayLengthOp>()) {
    uint32_t reduction = 0u - b.offset;
    // Only small reductions are useful (see "Limits" in the header).
    if (reduction > kMaxReduction) return std::nullopt;
    return ReducedLength{b.base_value, reduction};
  }
  if (length_aliases_open_) {
    return known_length_aliases_.Get(CanonicalValue(length));
  }
  return std::nullopt;
}

OpIndex WasmBoundsCheckEliminationAnalyzer::ArrayOfLength(
    OpIndex length) const {
  return ResolveAliases(graph_.Get(length).Cast<ArrayLengthOp>().array());
}

// Whether {length} is {a.length - r}, with {a.length} known to be at
// least {r}, so that it does not wrap around and is less than 2^31.
bool WasmBoundsCheckEliminationAnalyzer::IsArrayLengthWithoutWrapAround(
    OpIndex length) const {
  std::optional<ReducedLength> array_length = TryExtractArrayLength(length);
  return array_length.has_value() &&
         (array_length->reduction == 0 ||
          MinLength(ArrayOfLength(array_length->length)) >=
              array_length->reduction);
}

// Attempts to decode a condition showing that {base + n} is
// non-negative ([non-negative-index]):
//     base + n < b    or   base + n <= b   (unsigned)
// where {b} is an array length (possibly minus a constant, if the
// length is known to be at least that constant), or a small enough
// constant, or
//     c < base + n    or   c <= base + n   (signed)
// where {c} is a large enough constant.
std::optional<WasmBoundsCheckEliminationAnalyzer::BaseAndOffset>
WasmBoundsCheckEliminationAnalyzer::TryExtractNonNegativeIndex(
    const Relation& relation) const {
  using Kind = ComparisonOp::Kind;
  OpIndex index;
  switch (relation.kind) {
    case Kind::kEqual:
      return std::nullopt;
    case Kind::kUnsignedLessThan:
    case Kind::kUnsignedLessThanOrEqual: {
      // The index is at most {b}, which must be less than 2^31, or at
      // most {b - 1} for a strict comparison. Array lengths are less
      // than 2^31, and so are reduced array lengths that do not wrap
      // around.
      if (!IsArrayLengthWithoutWrapAround(relation.right)) {
        std::optional<uint32_t> bound = TryExtractI32Const(relation.right);
        if (!bound.has_value()) return std::nullopt;
        uint32_t limit = relation.is_strict() ? 1u << 31 : kMaxInt;
        if (*bound > limit) return std::nullopt;
      }
      index = relation.left;
      break;
    }
    case Kind::kSignedLessThan:
    case Kind::kSignedLessThanOrEqual: {
      // The index is at least {c + 1}, respectively {c}, which must be
      // non-negative.
      std::optional<uint32_t> bound = TryExtractI32Const(relation.left);
      if (!bound.has_value()) return std::nullopt;
      int32_t limit = relation.is_strict() ? -1 : 0;
      if (static_cast<int32_t>(*bound) < limit) return std::nullopt;
      index = relation.right;
      break;
    }
  }

  BaseAndOffset result = ExtractBaseAndOffset(index);
  if (!result.base.valid()) return std::nullopt;
  return result;
}

// Extract base expression and constant offset from an index expression.
WasmBoundsCheckEliminationAnalyzer::BaseAndOffset
WasmBoundsCheckEliminationAnalyzer::ExtractBaseAndOffset(OpIndex index) const {
  BaseAndOffset result = DecomposeIndex(index);
  if (result.base_value.valid())
    result.base = CanonicalValue(result.base_value);
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
  static constexpr int kMaxDepth = 8;
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
    if (depth == kMaxDepth) break;
    // a[base + n] / a[n + base] / a[base - n]
    const WordBinopOp* op = graph_.Get(index).TryCast<WordBinopOp>();
    if (op == nullptr || op->rep != WordRepresentation::Word32()) break;
    if (std::optional<uint32_t> constant = TryExtractI32Const(op->right())) {
      if (op->kind == WordBinopOp::Kind::kAdd) {
        offset += *constant;
      } else if (op->kind == WordBinopOp::Kind::kSub) {
        offset -= *constant;
      } else {
        break;
      }
      index = op->left();
    } else if (std::optional<uint32_t> left_constant =
                   TryExtractI32Const(op->left());
               left_constant.has_value() &&
               op->kind == WordBinopOp::Kind::kAdd) {
      // Constants are not always on the right yet: this phase runs before
      // machine optimizations, except in functions with unrolled loops.
      offset += *left_constant;
      index = op->right();
    } else {
      break;
    }
  }
  // Default: a[base]
  return {OpIndex::Invalid(), index, offset};
}

// Returns a canonical representative of {value}: pure operations
// (arithmetic, shifts, conversions) with the same options and inputs
// with the same canonical values compute the same value, and so have
// the same canonical value. Code producers often compute the same
// value several times: for instance, wasm_of_ocaml sign-extends a
// 31-bit integer both for an OCaml bounds check and for the index of
// the array access it protects, and an {i31.get_s} of the same
// reference computes the same integer. Casts and non-null assertions
// return their input, and values replaced by load elimination are the
// same as their replacement, which dominates them, is equal to them,
// and is what the output graph computes.
//
// Facts about canonical values are sound because a fact recorded at a
// point {p} is only used at points {q} dominated by {p}. The operations
// whose canonical value is not structural (phis, loads, parameters...)
// used at {p} are defined in dominators of {p}, and so are not executed
// again between the last execution of {p} and {q} (otherwise, there
// would be a path to {q} avoiding {p}). A value used at {q} with the
// same canonical value is computed from these same operations, and so
// is equal to the value used at {p}. In particular, phis are never
// looked through.
OpIndex WasmBoundsCheckEliminationAnalyzer::CanonicalValue(OpIndex value,
                                                           int depth) const {
  // Bound the depth of the recursion. Deeper values are their own
  // canonical value, which is always correct.
  static constexpr int kMaxDepth = 16;
  value = ResolveReplacements(value);
  if (auto it = canonical_values_.find(value); it != canonical_values_.end()) {
    return it->second;
  }
  if (depth == kMaxDepth) return value;
  const Operation& op = graph_.Get(value);
  std::optional<StructuralKey> key;
  switch (op.opcode) {
    case Opcode::kWasmTypeCast:
    case Opcode::kAssertNotNull:
    case Opcode::kWasmTypeAnnotation: {
      OpIndex object = ResolveAliases(value);
      if (object == value) break;
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
      key = StructuralKey{op.opcode,
                          static_cast<uint64_t>(change.kind) |
                              static_cast<uint64_t>(change.assumption) << 8 |
                              static_cast<uint64_t>(change.from.value()) << 16 |
                              static_cast<uint64_t>(change.to.value()) << 24,
                          0, CanonicalValue(change.input(), depth + 1),
                          OpIndex::Invalid()};
      break;
    }
    case Opcode::kTaggedBitcast: {
      const TaggedBitcastOp& bitcast = op.Cast<TaggedBitcastOp>();
      // The bitcast of a reference to a heap object is its address, which
      // a moving GC can change, so two bitcasts of the same reference may
      // give different values. This is not the case for a Smi, such as a
      // non-null i31 reference, which {i31.get} converts this way.
      if (bitcast.from == RegisterRepresentation::Tagged() &&
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
