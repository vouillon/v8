// Copyright 2025 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "src/compiler/turboshaft/wasm-bounds-check-elimination-reducer.h"

#include <iomanip>
#include <map>
#include <set>

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

// This code assumes that array lengths fit within signed 31-bit integers
// for modular arithmetic. MaxLength(1) returns the largest possible array
// length (using element size of 1 byte, the smallest possible element).
static_assert(v8::internal::WasmArray::MaxLength(1) < (1u << 31));
// Loop induction relies on array lengths being much smaller, so that
// adding a step to an index bounded by an array length cannot overflow.
static_assert(v8::internal::WasmArray::MaxLength(1) <= (1 << 30));
static constexpr uint32_t kMaxArrayLength = 1u << 30;

void WasmBoundsCheckEliminationAnalyzer::ProcessBlock(const Block& block) {
  BeginBlock(&block);
  current_position_ = 0;
  for (OpIndex op_idx : graph_.OperationIndices(block)) {
    const Operation& op = graph_.Get(op_idx);
    if (op.opcode == Opcode::kTrapIf && !ShouldSkipOperation(op) &&
        !ShouldSkipOptimizationStep()) {
      ProcessTrapIf(op_idx, op.Cast<TrapIfOp>());
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
  if (is_bounds_check) {
    TRACE("ProcessTrapIf(" << op_idx << ")");
#if DEBUG
    total_trap_count_++;
#endif
  }

  // Past the trap, the condition holds if the trap is negated, and
  // does not hold otherwise.
  ProcessCondition(is_bounds_check ? op_idx : OpIndex::Invalid(),
                   trap_if.condition(), !trap_if.negated);
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
    // When we visit a loop, the loop header hasn't been visited yet,
    // so we ignore it.
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
  // predecessor snapshots.
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
              << s.start_index << ", " << s.end_index << "), not profitable ("
              << s.instruction_count() << " instructions, "
              << s.covered_traps.size() << " traps, "
              << s.ComputeGuard().check_count() << " guard checks)");
        continue;
      }
      TRACE("Commit fallback sequence: ["
            << s.start_index << ", " << s.end_index << "), "
            << s.covered_traps.size() << " traps");
      OpIndex start_index = s.start_index;
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
                   block == branch->if_false);
}

// Records what we learn from {condition}, which holds if not {inverted},
// and does not hold otherwise. For a TrapIf, {trap_if} is the trap. For a
// Branch or a trap that cannot be eliminated, {trap_if} is
// OpIndex::Invalid().
void WasmBoundsCheckEliminationAnalyzer::ProcessCondition(OpIndex trap_if,
                                                          OpIndex condition,
                                                          bool inverted,
                                                          int depth) {
  // Combined conditions: {a | b} does not hold when neither {a} nor {b}
  // holds, {a & b} of two comparisons (which are 0 or 1) holds when both
  // hold, and {a == 0} holds when {a} does not.
  static constexpr int kMaxDepth = 4;
  if (depth < kMaxDepth) {
    const Operation& op = graph_.Get(condition);
    if (const WordBinopOp* binop = op.TryCast<WordBinopOp>();
        binop && binop->rep == WordRepresentation::Word32()) {
      if (binop->kind == WordBinopOp::Kind::kBitwiseOr && inverted) {
        ProcessCondition(OpIndex::Invalid(), binop->left(), true, depth + 1);
        ProcessCondition(OpIndex::Invalid(), binop->right(), true, depth + 1);
        return;
      }
      if (binop->kind == WordBinopOp::Kind::kBitwiseAnd && !inverted &&
          graph_.Get(binop->left()).Is<ComparisonOp>() &&
          graph_.Get(binop->right()).Is<ComparisonOp>()) {
        ProcessCondition(OpIndex::Invalid(), binop->left(), false, depth + 1);
        ProcessCondition(OpIndex::Invalid(), binop->right(), false, depth + 1);
        return;
      }
    }
    if (const ComparisonOp* comparison = op.TryCast<ComparisonOp>();
        comparison && comparison->kind == ComparisonOp::Kind::kEqual &&
        comparison->rep == RegisterRepresentation::Word32()) {
      if (auto constant = TryExtractI32Const(comparison->right());
          constant.has_value() && *constant == 0) {
        ProcessCondition(OpIndex::Invalid(), comparison->left(), !inverted,
                         depth + 1);
        return;
      }
    }
  }
  if (const ComparisonOp* comparison =
          graph_.Get(condition).TryCast<ComparisonOp>();
      comparison && inverted &&
      comparison->kind == ComparisonOp::Kind::kEqual &&
      comparison->rep == RegisterRepresentation::Word32()) {
    // {x + k != a.length - r}: if {x + k + r - 1} is known to be within
    // bounds, then so is {x + k + r} (see the proofs in the header).
    for (auto [index, length] :
         {std::pair{comparison->left(), comparison->right()},
          std::pair{comparison->right(), comparison->left()}}) {
      std::optional<std::pair<OpIndex, uint32_t>> array_length =
          TryExtractArrayLength(length);
      if (!array_length) continue;
      const auto& [length_op, reduction] = *array_length;
      auto [base, base_value, offset] = ExtractBaseAndOffset(index);
      OpIndex array =
          ResolveAliases(graph_.Get(length_op).Cast<ArrayLengthOp>().array());
      BoundsCheckKey key(base, array);
      uint32_t last = offset + reduction - 1;
      std::optional<OffsetRange> known =
          KnownOffsets(key, NonNegativeOffsets(base));
      if (!known.has_value() || !known->Contains(last) ||
          known->Contains(last + 1)) {
        continue;
      }
      TRACE("  Not equal to the array length");
      ProcessBoundsCheck(OpIndex::Invalid(), condition,
                         BoundsCheck(key, last + 1), base_value, length_op, 0);
      return;
    }
  }
  if (auto signed_decoded =
          TryExtractSignedBoundsCheckCondition(condition, inverted)) {
    // A signed comparison {x + n < a.length - c}, where {x + n} is known
    // to be non-negative, shows that {a.length - c} is positive, and so
    // that it does not wrap around, and that {x + n < a.length - c} as
    // unsigned integers (see the proofs in the header).
    const auto& [bounds_check, base_value, array_length, reduction] =
        *signed_decoded;
    const auto& [key, offset] = bounds_check;
    TRACE("  Signed bounds check");
    ProcessBoundsCheck(OpIndex::Invalid(), condition, bounds_check, base_value,
                       array_length, reduction);
    uint64_t index_bound = key.base.valid() ? 0 : offset;
    if (index_bound <= kMaxInt) {
      RecordMinLength(key.array, index_bound + reduction + 1);
    }
  } else if (auto decoded =
                 TryExtractBoundChecksCondition(condition, inverted)) {
    const auto& [bounds_check, base_value, array_length, reduction] = *decoded;
    const auto& [key, offset] = bounds_check;
    // Unless the length is known to be at least {reduction},
    // {a.length - reduction} may wrap around, and we learn nothing.
    uint32_t min_length = MinLength(key.array);
    if (reduction == 0 || min_length >= reduction) {
      // When {reduction} is not 0, this is not an actual bounds check, so
      // it is only used for what it shows, even for a trap.
      ProcessBoundsCheck(reduction == 0 ? trap_if : OpIndex::Invalid(),
                         condition, bounds_check, base_value, array_length,
                         reduction);
      // With {index < a.length - reduction}, the length is at least
      // {reduction + 1}, and more for a constant index.
      uint64_t index_bound = key.base.valid() ? 0 : offset;
      if (index_bound <= kMaxInt) {
        RecordMinLength(key.array, index_bound + reduction + 1);
      }
    }
  }
  // This is done after processing the bounds check, since what is
  // known before the bounds check is used when emitting fallback
  // code.
  if (auto non_negative_index =
          TryExtractNonNegativeIndex(condition, inverted)) {
    const auto& [base, offset] = *non_negative_index;
    RecordNonNegativeOffset(base, offset);
  }
}

void WasmBoundsCheckEliminationAnalyzer::ProcessBoundsCheck(
    OpIndex trap_if, OpIndex condition, const BoundsCheck& bounds_check,
    OpIndex base_value, OpIndex array_length, uint32_t extent) {
  const auto& [key, offset] = bounds_check;
  // Only plain bounds checks can be eliminated.
  DCHECK_IMPLIES(trap_if.valid(), extent == 0);

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
  // within bounds (see the proofs in the header).
  uint32_t last = offset + extent;
  DCHECK(OffsetRange::IsValidRange(offset, last));
  OffsetRange direct(offset, last);

  // For constant indices, check if [0, last] would be valid.
  // If not, this offset is out-of-bounds and will always trap.
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

  // Offsets within bounds after this check: {direct}, and every offset
  // from {n} to {last} if {key.base + n} is non-negative, for {n} below
  // {offset}.
  OffsetRange checked = direct;
  if (non_negative.has_value()) {
    OffsetRange::RelativePosition position = non_negative->Classify(offset);
    if ((position == OffsetRange::RelativePosition::kInside ||
         position == OffsetRange::RelativePosition::kAbove) &&
        OffsetRange::IsValidRange(non_negative->lower(), last)) {
      checked = OffsetRange(non_negative->lower(), last);
    }
  }

  if (!known.has_value()) {
    // First bounds check for this base and array.
    TRACE("  New bounds check: offset="
          << offset << " extent=" << extent << " base=" << key.base
          << " array=" << key.array << " -> [" << checked.lower() << ", "
          << checked.upper() << "]");
    UpdateKnownBoundsChecks(key, offset, checked, trap_if, condition,
                            base_value, known, non_negative);
    return;
  }

  // The hull of {known} and {direct} might not be valid, in which case
  // this condition cannot hold given what we know: a bounds check will
  // always trap, and a branch will never be taken.
  std::optional<OffsetRange> offsets = known->Hull(checked);
  if (!offsets.has_value()) offsets = known->Hull(direct);
  if (!offsets.has_value()) {
    TRACE("  Out-of-range check: offsets ["
          << offset << ", " << last << "] outside [" << known->lower() << ", "
          << known->upper() << "]");
    return;
  }
  TRACE("  Extend range: [" << known->lower() << ", " << known->upper()
                            << "] -> [" << offsets->lower() << ", "
                            << offsets->upper() << "]");

  if (trap_if.valid()) {
    // The trap that established the bound of {known} on the side of
    // {offset} can start a fallback sequence. For a trap, {extent} is 0,
    // and {offset} is not in {known}, so it is above or below it.
    OffsetRange::RelativePosition position = known->Classify(offset);
    DCHECK(position == OffsetRange::RelativePosition::kAbove ||
           position == OffsetRange::RelativePosition::kBelow);
    uint32_t prev_offset = position == OffsetRange::RelativePosition::kAbove
                               ? known->upper()
                               : known->lower();
    RecordFallbackSequence(key, prev_offset, trap_if, offset);
  }
  UpdateKnownBoundsChecks(key, offset, *offsets, trap_if, condition, base_value,
                          known, non_negative);
}

// Record a fallback sequence covering the previous trap at
// {previous_offset} and the current trap at {offset}.
void WasmBoundsCheckEliminationAnalyzer::RecordFallbackSequence(
    const BoundsCheckKey& key, uint32_t previous_offset, OpIndex trap_if,
    uint32_t offset) {
  if (!trap_if.valid()) return;

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

  auto array_length_it = array_lengths_.find(key.array);
  // The guard needs an array length available in this block.
  if (array_length_it == array_lengths_.end()) return;
  OpIndex array_length = array_length_it->second;

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
      planned_fallbacks_by_key_.try_emplace(sequence.key, phase_zone_);
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
    TRACE("    New range: [" << previous_sequence.start_index << ", "
                             << previous_sequence.end_index << ")");
    TRACE("    Traps: " << previous_sequence.covered_traps.size());
    sequences.pop_back();
  }
}

void WasmBoundsCheckEliminationAnalyzer::UpdateKnownBoundsChecks(
    const BoundsCheckKey& key, uint32_t offset, const OffsetRange& offsets,
    OpIndex trap_if, OpIndex condition, OpIndex base_value,
    const std::optional<OffsetRange>& known_before,
    const std::optional<OffsetRange>& non_negative_before) {
  // Records the updated bounds check range.
  known_bounds_checks_.Set(key, offsets);

  if (!trap_if.valid()) return;

  // We register a possible place to insert some fallback code. We
  // will also clone the condition when it is right before the trap,
  // since we don't want to keep it on the main sequence of execution.
  bool starts_at_condition = graph_.PreviousIndex(trap_if) == condition;
  OpIndex start_index = starts_at_condition ? condition : trap_if;
  uint32_t start_position = current_position_ - starts_at_condition;
  last_trap_bounds_checks_.insert_or_assign(
      BoundsCheck(key, offset),
      TrapInfo{trap_if, base_value, start_index, start_position, known_before,
               non_negative_before});
}

// Returns the offsets known to be within bounds for {key}. If
// {key.base + n} is non-negative and some offset {m >= n} is within
// bounds, then all offsets from {n} to {m} are within bounds, even if
// we learnt that {key.base + n} is non-negative after checking {m}.
std::optional<OffsetRange> WasmBoundsCheckEliminationAnalyzer::KnownOffsets(
    const BoundsCheckKey& key,
    const std::optional<OffsetRange>& non_negative) const {
  std::optional<OffsetRange> known = known_bounds_checks_.Get(key);
  if (!known.has_value() || !non_negative.has_value()) return known;
  OffsetRange::RelativePosition position =
      non_negative->Classify(known->upper());
  if (position != OffsetRange::RelativePosition::kInside &&
      position != OffsetRange::RelativePosition::kAbove) {
    return known;
  }
  // Both ranges end at {known->upper()}, so their hull is valid.
  return known->Hull(OffsetRange(non_negative->lower(), known->upper()))
      .value_or(*known);
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
  // Keep the previous range if the new offset is too far away from it.
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
//     n < a.length - c   or   base + n < a.length - c
// where {c} is a constant (usually 0). If the condition cannot be
// interpreted as a bounds check, returns std::nullopt. Otherwise,
// returns a {BoundsCheck} describing the base, array, and offset,
// along with the index of the array length instruction, and {c}.
std::optional<WasmBoundsCheckEliminationAnalyzer::BoundsCheckCondition>
WasmBoundsCheckEliminationAnalyzer::TryExtractBoundChecksCondition(
    OpIndex condition, bool inverted) const {
  const ComparisonOp* comparison =
      graph_.Get(condition).TryCast<ComparisonOp>();
  if (!comparison) return std::nullopt;

  const auto expected_kind = inverted
                                 ? ComparisonOp::Kind::kUnsignedLessThanOrEqual
                                 : ComparisonOp::Kind::kUnsignedLessThan;
  if (comparison->kind != expected_kind) return std::nullopt;

  // Select operands depending on polarity.
  OpIndex index = inverted ? comparison->right() : comparison->left();
  OpIndex length = inverted ? comparison->left() : comparison->right();

  std::optional<std::pair<OpIndex, uint32_t>> array_length =
      TryExtractArrayLength(length);
  if (!array_length) return std::nullopt;
  const auto& [length_op, reduction] = *array_length;

  auto [base, base_value, offset] = ExtractBaseAndOffset(index);
  OpIndex array =
      ResolveAliases(graph_.Get(length_op).Cast<ArrayLengthOp>().array());
  return BoundsCheckCondition{BoundsCheck(BoundsCheckKey(base, array), offset),
                              base_value, length_op, reduction};
}

// Attempts to decode a signed condition {n < a.length - c} or
// {base + n < a.length - c} where the index is known to be
// non-negative, which then implies the unsigned condition.
std::optional<WasmBoundsCheckEliminationAnalyzer::BoundsCheckCondition>
WasmBoundsCheckEliminationAnalyzer::TryExtractSignedBoundsCheckCondition(
    OpIndex condition, bool inverted) const {
  const ComparisonOp* comparison =
      graph_.Get(condition).TryCast<ComparisonOp>();
  if (!comparison || comparison->rep != RegisterRepresentation::Word32()) {
    return std::nullopt;
  }
  const auto expected_kind = inverted
                                 ? ComparisonOp::Kind::kSignedLessThanOrEqual
                                 : ComparisonOp::Kind::kSignedLessThan;
  if (comparison->kind != expected_kind) return std::nullopt;
  OpIndex index = inverted ? comparison->right() : comparison->left();
  OpIndex length = inverted ? comparison->left() : comparison->right();
  std::optional<std::pair<OpIndex, uint32_t>> array_length =
      TryExtractArrayLength(length);
  if (!array_length) return std::nullopt;
  const auto& [length_op, reduction] = *array_length;
  auto [base, base_value, offset] = ExtractBaseAndOffset(index);
  OpIndex array =
      ResolveAliases(graph_.Get(length_op).Cast<ArrayLengthOp>().array());
  std::optional<OffsetRange> non_negative = NonNegativeOffsets(base);
  bool is_non_negative = non_negative && non_negative->Contains(offset);
  if (!is_non_negative && base.valid()) {
    // If {base + m} is within the bounds of an array, {base + m + d} is
    // non-negative for {0 <= d <= kMaxArrayLength}, since array lengths
    // are at most {kMaxArrayLength} (see the proofs in the header).
    if (std::optional<OffsetRange> known =
            known_bounds_checks_.Get(BoundsCheckKey(base, array))) {
      is_non_negative = offset - known->upper() <= kMaxArrayLength;
    }
  }
  if (!is_non_negative) return std::nullopt;
  return BoundsCheckCondition{BoundsCheck(BoundsCheckKey(base, array), offset),
                              base_value, length_op, reduction};
}

template <typename F>
void WasmBoundsCheckEliminationAnalyzer::ForEachComparison(OpIndex condition,
                                                           bool holds,
                                                           const F& f,
                                                           int depth) const {
  static constexpr int kMaxDepth = 4;
  const Operation& op = graph_.Get(condition);
  if (depth < kMaxDepth) {
    if (const WordBinopOp* binop = op.TryCast<WordBinopOp>();
        binop && binop->rep == WordRepresentation::Word32()) {
      if (binop->kind == WordBinopOp::Kind::kBitwiseOr && !holds) {
        ForEachComparison(binop->left(), false, f, depth + 1);
        ForEachComparison(binop->right(), false, f, depth + 1);
        return;
      }
      if (binop->kind == WordBinopOp::Kind::kBitwiseAnd && holds &&
          graph_.Get(binop->left()).Is<ComparisonOp>() &&
          graph_.Get(binop->right()).Is<ComparisonOp>()) {
        ForEachComparison(binop->left(), true, f, depth + 1);
        ForEachComparison(binop->right(), true, f, depth + 1);
        return;
      }
    }
  }
  if (const ComparisonOp* comparison = op.TryCast<ComparisonOp>();
      comparison && comparison->rep == RegisterRepresentation::Word32()) {
    if (depth < kMaxDepth && comparison->kind == ComparisonOp::Kind::kEqual) {
      if (auto constant = TryExtractI32Const(comparison->right());
          constant.has_value() && *constant == 0) {
        ForEachComparison(comparison->left(), !holds, f, depth + 1);
        return;
      }
    }
    f(*comparison, holds);
  }
}

template <typename F>
void WasmBoundsCheckEliminationAnalyzer::ForEachDominatingComparison(
    const Block* block, const Block* stop, const F& f) const {
  for (const Block* b = block; b != nullptr && b != stop;
       b = b->GetDominator()) {
    if (!b->IsBranchTarget()) continue;
    const BranchOp* branch =
        b->LastPredecessor()->LastOperation(graph_).TryCast<BranchOp>();
    if (!branch) continue;
    ForEachComparison(branch->condition(), b == branch->if_true, f);
  }
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
// edge is only taken if {i < x} (signed or unsigned), for a bound {x}
// small enough that {i + c} cannot overflow (an array length minus a
// constant, or a constant). With {c = 1}, it is also non-negative when
// the back edge is only taken if {i != x}, for a loop-invariant {x}, and
// the loop is only entered if {c0 <= x}: {i} is then at most {x} in the
// whole loop, which is a bounds check fact when {x} is an array length
// minus a constant (see the proofs in the header).
void WasmBoundsCheckEliminationAnalyzer::ProcessLoopHeader(
    const Block* header) {
  if (header->PredecessorCount() != 2) return;
  const Block* back_edge = header->LastPredecessor();
  const Block* forward = back_edge->NeighboringPredecessor();
  for (OpIndex index : graph_.OperationIndices(*header)) {
    const PhiOp* phi = graph_.Get(index).TryCast<PhiOp>();
    if (!phi) continue;
    if (phi->rep != RegisterRepresentation::Word32() || phi->input_count != 2) {
      continue;
    }
    std::optional<uint32_t> init = TryExtractI32Const(phi->input(0));
    if (!init.has_value() || static_cast<int32_t>(*init) < 0) continue;
    OpIndex value = CanonicalValue(index);
    BaseAndOffset step = ExtractBaseAndOffset(phi->back_edge());
    if (step.base != value) continue;
    uint32_t c = step.offset;
    if (c == 0 || c > kMaxArrayLength) continue;

    // Whether {x}, for {i < x} (or {i <= x} if {inclusive}), bounds {i}
    // enough for {i + c} not to overflow.
    auto small_bound = [&](OpIndex x, bool is_signed, bool inclusive) {
      if (auto constant = TryExtractI32Const(x)) {
        // {i < x <= 2^31 - c} (or {i <= x <= 2^31 - 1 - c}), so that
        // {i + c <= 2^31 - 1}. A negative {x} is fine for a signed
        // comparison: the back edge is then never taken.
        uint32_t limit = uint32_t{kMaxInt} - c + (inclusive ? 0 : 1);
        return is_signed ? static_cast<int32_t>(*constant) <=
                               static_cast<int32_t>(limit)
                         : *constant <= limit;
      }
      BaseAndOffset bound = ExtractBaseAndOffset(x);
      if (!bound.base.valid() || !graph_.Get(bound.base).Is<ArrayLengthOp>()) {
        return false;
      }
      // {x = a.length + bound.offset}: signed, a non-positive offset
      // keeps {x} below the maximal length; unsigned, {x} could wrap
      // around unless the offset is 0.
      return is_signed ? static_cast<int32_t>(bound.offset) <= 0 &&
                             static_cast<int32_t>(bound.offset) >
                                 -static_cast<int32_t>(kMaxArrayLength)
                       : bound.offset == 0;
    };
    auto is_value = [&](OpIndex v) {
      BaseAndOffset b = ExtractBaseAndOffset(v);
      return b.base == value && b.offset == 0;
    };
    auto is_loop_invariant = [&](OpIndex v) {
      return TryExtractI32Const(v).has_value() ||
             Dominates(&graph_.Get(graph_.BlockOf(v)), forward);
    };

    bool non_negative = false;
    OpIndex upper = OpIndex::Invalid();  // {i <= upper} in the loop.
    int64_t upper_lower_bound = 0;       // {upper_lower_bound <= upper}.
    // For the comparisons {i + k != x}: the offsets {k} for each {x}.
    std::map<OpIndex, std::set<uint32_t>> not_equal_offsets;
    std::map<OpIndex, OpIndex> not_equal_values;
    ForEachDominatingComparison(
        back_edge, header, [&](const ComparisonOp& cmp, bool holds) {
          using Kind = ComparisonOp::Kind;
          // Normalize to {left kind right}, which holds, or {left != right}.
          OpIndex left = cmp.left(), right = cmp.right();
          Kind kind = cmp.kind;
          bool not_equal = false;
          if (!holds) {
            if (kind == Kind::kEqual) {
              not_equal = true;
            } else {
              std::swap(left, right);
              kind =
                  kind == Kind::kSignedLessThan ? Kind::kSignedLessThanOrEqual
                  : kind == Kind::kSignedLessThanOrEqual ? Kind::kSignedLessThan
                  : kind == Kind::kUnsignedLessThan
                      ? Kind::kUnsignedLessThanOrEqual
                      : Kind::kUnsignedLessThan;
            }
          } else if (kind == Kind::kEqual) {
            return;
          }
          if (not_equal) {
            // {i + k != x}: record {k} for {x}.
            BaseAndOffset l = ExtractBaseAndOffset(left);
            BaseAndOffset r = ExtractBaseAndOffset(right);
            if (l.base == value) {
              not_equal_offsets[CanonicalValue(right)].insert(l.offset);
              not_equal_values.try_emplace(CanonicalValue(right), right);
            } else if (r.base == value) {
              not_equal_offsets[CanonicalValue(left)].insert(r.offset);
              not_equal_values.try_emplace(CanonicalValue(left), left);
            }
            return;
          }
          if (!is_value(left)) return;
          bool is_signed = kind == Kind::kSignedLessThan ||
                           kind == Kind::kSignedLessThanOrEqual;
          bool inclusive = kind == Kind::kSignedLessThanOrEqual ||
                           kind == Kind::kUnsignedLessThanOrEqual;
          if (small_bound(right, is_signed, inclusive)) non_negative = true;
        });
    // With {i + k != x} for all {k} from 0 to {c - 1} on the path to the
    // back edge (as after unrolling a loop with a step of 1), {i} never
    // goes past {x} if it starts at most at {x}.
    static constexpr uint32_t kMaxUnrolledStep = 16;
    for (const auto& [x, offsets] : not_equal_offsets) {
      if (upper.valid() || c > kMaxUnrolledStep) break;
      bool covered = true;
      for (uint32_t k = 0; k < c; k++) covered &= offsets.contains(k);
      OpIndex x_value = not_equal_values[x];
      if (!covered || !is_loop_invariant(x_value)) continue;
      // The loop must only be entered if {c0 <= x}: look for a lower
      // bound of {x} ({k <= x} or {k < x}), raised by the values that
      // {x} is known to differ from.
      std::optional<int64_t> lower;
      std::set<int64_t> excluded;
      if (auto constant = TryExtractI32Const(x_value)) {
        lower = static_cast<int32_t>(*constant);
      }
      using Kind = ComparisonOp::Kind;
      auto guard = [&](const ComparisonOp& cmp, bool holds) {
        Kind kind = cmp.kind;
        OpIndex l = cmp.left(), r = cmp.right();
        if (kind == Kind::kEqual) {
          if (holds) return;
          OpIndex other = CanonicalValue(l) == x   ? r
                          : CanonicalValue(r) == x ? l
                                                   : OpIndex::Invalid();
          if (!other.valid()) return;
          if (auto k = TryExtractI32Const(other)) {
            excluded.insert(static_cast<int32_t>(*k));
          }
          return;
        }
        if (!holds) {
          if (kind == Kind::kSignedLessThan) {
            std::swap(l, r);
            kind = Kind::kSignedLessThanOrEqual;
          } else if (kind == Kind::kSignedLessThanOrEqual) {
            std::swap(l, r);
            kind = Kind::kSignedLessThan;
          } else {
            return;
          }
        }
        if (CanonicalValue(r) != x) return;
        auto k = TryExtractI32Const(l);
        if (!k.has_value()) return;
        int64_t bound = static_cast<int32_t>(*k);
        if (kind == Kind::kSignedLessThan) {
          bound += 1;
        } else if (kind != Kind::kSignedLessThanOrEqual) {
          return;
        }
        if (!lower.has_value() || bound > *lower) lower = bound;
      };
      for (const Block* b = forward; b != nullptr; b = b->GetDominator()) {
        if (!b->IsBranchTarget()) continue;
        const BranchOp* branch =
            b->LastPredecessor()->LastOperation(graph_).TryCast<BranchOp>();
        if (!branch) continue;
        bool holds = b == branch->if_true;
        // A branch on {x} itself shows that {x != 0} when taken.
        if (holds && CanonicalValue(branch->condition()) == x) {
          excluded.insert(0);
          continue;
        }
        ForEachComparison(branch->condition(), holds, guard);
      }
      bool entered = false;
      if (lower.has_value()) {
        int64_t bound = *lower;
        while (excluded.contains(bound)) bound++;
        entered = static_cast<int32_t>(*init) <= bound;
        upper_lower_bound = bound;
      }
      if (entered) {
        non_negative = true;
        upper = x_value;
      }
    }
    if (!non_negative) continue;
    TRACE("  Loop induction: " << index << " is non-negative");
    RecordNonNegativeOffset(value, 0);
    if (!upper.valid()) continue;
    // {i <= a.length - r} with {r >= 1}: the offsets 0 to {r - 1} of {i}
    // are within bounds.
    BaseAndOffset bound = ExtractBaseAndOffset(upper);
    if (!bound.base.valid()) continue;
    if (const PhiOp* merge = graph_.Get(bound.base).TryCast<PhiOp>();
        merge && !graph_.Get(graph_.BlockOf(bound.base)).IsLoop()) {
      // {upper = phi(...) + offset}, loop-invariant: the loop is only
      // entered with a value of {upper} at least {upper_lower_bound},
      // which rules out the constant inputs {k} with
      // {k + offset < upper_lower_bound}. If a single input remains,
      // {upper} is that input plus {offset} in the loop. For instance,
      // the length of an array that may be empty (a constant 0 on the
      // other path) minus 1, for a loop that is only entered if it is
      // non-negative.
      OpIndex remaining = OpIndex::Invalid();
      bool single = true;
      for (OpIndex input : merge->inputs()) {
        if (auto k = TryExtractI32Const(input)) {
          int64_t candidate = static_cast<int32_t>(*k + bound.offset);
          if (candidate < upper_lower_bound) continue;
        }
        if (remaining.valid()) single = false;
        remaining = input;
      }
      if (!single || !remaining.valid()) continue;
      BaseAndOffset input = ExtractBaseAndOffset(remaining);
      bound = BaseAndOffset{input.base, input.base_value,
                            input.offset + bound.offset};
      if (!bound.base.valid()) continue;
      // In the loop, {upper} is then {a.length - r}.
      if (graph_.Get(bound.base).Is<ArrayLengthOp>()) {
        uint32_t reduction = 0u - bound.offset;
        if (reduction <= (1u << 16)) {
          if (!length_aliases_open_) {
            // The first alias: no block had any before.
            known_length_aliases_.StartNewSnapshot();
            length_aliases_open_ = true;
          }
          known_length_aliases_.Set(CanonicalValue(upper),
                                    std::pair{bound.base, reduction});
          alias_lengths_.insert(bound.base);
        }
      }
    }
    const ArrayLengthOp* length =
        graph_.Get(bound.base).TryCast<ArrayLengthOp>();
    if (!length) continue;
    int32_t offset = static_cast<int32_t>(bound.offset);
    if (offset >= 0 || offset < -static_cast<int32_t>(kMaxArrayLength)) {
      continue;
    }
    OpIndex array = ResolveAliases(length->array());
    uint32_t last = static_cast<uint32_t>(-offset) - 1;
    TRACE("  Loop induction: offsets [0, " << last << "] of " << index
                                           << " within bounds of " << array);
    BoundsCheckKey key(value, array);
    std::optional<OffsetRange> known = known_bounds_checks_.Get(key);
    std::optional<OffsetRange> range = OffsetRange(0, last);
    if (known.has_value()) range = known->Hull(*range);
    if (range.has_value()) known_bounds_checks_.Set(key, *range);
    RecordMinLength(array, uint64_t{last} + 1);
  }
}

// Recognizes {a.length} and {a.length - c} (possibly written
// {a.length + (-c)}) for a small constant {c}. Returns the array length
// instruction and {c}.
std::optional<std::pair<OpIndex, uint32_t>>
WasmBoundsCheckEliminationAnalyzer::TryExtractArrayLength(
    OpIndex length) const {
  if (graph_.Get(length).Is<ArrayLengthOp>()) return std::pair{length, 0u};
  if (length_aliases_open_) {
    if (auto alias = known_length_aliases_.Get(CanonicalValue(length))) {
      return *alias;
    }
  }
  const WordBinopOp* op = graph_.Get(length).TryCast<WordBinopOp>();
  if (!op || op->rep != WordRepresentation::Word32() ||
      !graph_.Get(op->left()).Is<ArrayLengthOp>()) {
    return std::nullopt;
  }
  std::optional<uint32_t> constant = TryExtractI32Const(op->right());
  if (!constant) return std::nullopt;
  uint32_t reduction;
  switch (op->kind) {
    case WordBinopOp::Kind::kSub:
      reduction = *constant;
      break;
    case WordBinopOp::Kind::kAdd:
      reduction = 0u - *constant;
      break;
    default:
      return std::nullopt;
  }
  // Only small reductions are useful. This also ensures that the range
  // of offsets shown by the condition is valid.
  static constexpr uint32_t kMaxReduction = 1 << 16;
  if (reduction > kMaxReduction) return std::nullopt;
  return std::pair{op->left(), reduction};
}

// Whether {length} is {a.length - c}, with {a.length} known to be at
// least {c}, so that it does not wrap around and is less than 2^31.
bool WasmBoundsCheckEliminationAnalyzer::IsArrayLengthWithoutWrapAround(
    OpIndex length) const {
  std::optional<std::pair<OpIndex, uint32_t>> array_length =
      TryExtractArrayLength(length);
  if (!array_length) return false;
  const auto& [length_op, reduction] = *array_length;
  return reduction == 0 ||
         MinLength(ResolveAliases(
             graph_.Get(length_op).Cast<ArrayLengthOp>().array())) >= reduction;
}

// Attempts to decode a condition showing that {base + n} is
// non-negative:
//     base + n < b    or   base + n <= b   (unsigned)
// where {b} is an array length (possibly minus a constant, if the
// length is known to be at least that constant), or a small enough
// constant, or
//     c < base + n    or   c <= base + n   (signed)
// where {c} is a large enough constant. Returns {base} and {n}.
std::optional<std::pair<OpIndex, uint32_t>>
WasmBoundsCheckEliminationAnalyzer::TryExtractNonNegativeIndex(
    OpIndex condition, bool inverted) const {
  const ComparisonOp* comparison =
      graph_.Get(condition).TryCast<ComparisonOp>();
  if (!comparison || comparison->rep != RegisterRepresentation::Word32()) {
    return std::nullopt;
  }

  // We normalize the condition to {left kind right}, which holds.
  using Kind = ComparisonOp::Kind;
  Kind kind = comparison->kind;
  OpIndex left = comparison->left();
  OpIndex right = comparison->right();
  if (inverted) {
    // !(a < b) is (b <= a), and !(a <= b) is (b < a).
    std::swap(left, right);
    switch (kind) {
      case Kind::kEqual:
        return std::nullopt;
      case Kind::kSignedLessThan:
        kind = Kind::kSignedLessThanOrEqual;
        break;
      case Kind::kSignedLessThanOrEqual:
        kind = Kind::kSignedLessThan;
        break;
      case Kind::kUnsignedLessThan:
        kind = Kind::kUnsignedLessThanOrEqual;
        break;
      case Kind::kUnsignedLessThanOrEqual:
        kind = Kind::kUnsignedLessThan;
        break;
    }
  }

  OpIndex index;
  switch (kind) {
    case Kind::kEqual:
      return std::nullopt;
    case Kind::kUnsignedLessThan:
    case Kind::kUnsignedLessThanOrEqual: {
      // The index is at most {b}, which must be less than 2^31, or at
      // most {b - 1} for a strict comparison. Array lengths are less
      // than 2^31, and so are reduced array lengths that do not wrap
      // around.
      if (!IsArrayLengthWithoutWrapAround(right)) {
        std::optional<uint32_t> bound = TryExtractI32Const(right);
        if (!bound.has_value()) return std::nullopt;
        uint32_t limit = kind == Kind::kUnsignedLessThan ? 1u << 31 : kMaxInt;
        if (*bound > limit) return std::nullopt;
      }
      index = left;
      break;
    }
    case Kind::kSignedLessThan:
    case Kind::kSignedLessThanOrEqual: {
      // The index is at least {c + 1}, respectively {c}, which must be
      // non-negative.
      std::optional<uint32_t> bound = TryExtractI32Const(left);
      if (!bound.has_value()) return std::nullopt;
      int32_t limit = kind == Kind::kSignedLessThan ? -1 : 0;
      if (static_cast<int32_t>(*bound) < limit) return std::nullopt;
      index = right;
      break;
    }
  }

  auto [base, base_value, offset] = ExtractBaseAndOffset(index);
  if (!base.valid()) return std::nullopt;
  return std::pair{base, offset};
}

// Extract base expression and constant offset from an index expression.
WasmBoundsCheckEliminationAnalyzer::BaseAndOffset
WasmBoundsCheckEliminationAnalyzer::ExtractBaseAndOffset(OpIndex index) const {
  // Nested additions of constants, as produced for instance by loop
  // unrolling, are folded into a single offset, so that
  // {(base + n) + m} and {base + (n + m)} have the same base. We are
  // using modular arithmetic, so a subtraction can be replaced by the
  // addition of the opposite. The depth is bounded to keep the cost
  // linear.
  static constexpr int kMaxDepth = 8;
  uint32_t offset = 0;
  for (int depth = 0;; depth++) {
    index = ResolveReplacements(index);
    // a[n]
    if (auto constant = TryExtractI32Const(index)) {
      return {OpIndex::Invalid(), OpIndex::Invalid(), offset + *constant};
    }
    if (depth == kMaxDepth) break;
    // a[base + n] / a[base - n]
    const WordBinopOp* op = graph_.Get(index).TryCast<WordBinopOp>();
    if (op == nullptr) break;
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
  return {CanonicalValue(index), index, offset};
}

// Returns a canonical representative of {value}: pure operations
// (arithmetic, shifts, conversions) with the same options and inputs
// with the same canonical values compute the same value, and so have
// the same canonical value. Code producers often compute the same
// value several times: for instance, wasm_of_ocaml sign-extends a
// 31-bit integer both for an OCaml bounds check and for the index of
// the array access it protects, and an {i31.get_s} of the same
// reference computes the same integer. Casts and non-null assertions
// return their input.
//
// Values that are computed from the same operations are only equal
// when these operations were executed the same number of times. Loop
// phis are the only operations whose values may change without
// leaving a loop iteration, and facts are never propagated along
// back edges, so the facts established about a value in an iteration
// are only used in the same iteration.
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
      key = StructuralKey{op.opcode,
                          static_cast<uint64_t>(binop.kind) |
                              static_cast<uint64_t>(binop.rep.value()) << 8,
                          0, CanonicalValue(binop.left(), depth + 1),
                          CanonicalValue(binop.right(), depth + 1)};
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

#if DEBUG
void WasmBoundsCheckEliminationAnalyzer::PrintStats() {
  if (v8_flags.turboshaft_trace_wasm_bounds_check_elimination) {
    size_t redundant_trap_count = redundant_traps_.size();
    size_t covered_trap_count = 0;
    size_t instruction_count = 0;
    for (auto& [idx, seq] : fallback_sequence_starts_) {
      covered_trap_count += seq.covered_traps.size();
      instruction_count += seq.instruction_count();
    }

    size_t eliminated_count = redundant_trap_count + covered_trap_count;
    double elimination_rate =
        total_trap_count_ > 0 ? (100.0 * eliminated_count / total_trap_count_)
                              : 0.0;
    double avg_fallback_size =
        covered_trap_count > 0
            ? (static_cast<double>(instruction_count) / covered_trap_count)
            : 0.0;

    if (eliminated_count == 0) {
      return;
    }

    TRACE("== Bounds Check Elimination Stats ==");
    TRACE("  Traps: " << eliminated_count << "/" << total_trap_count_
                      << " eliminated (" << std::fixed << std::setprecision(1)
                      << elimination_rate << "%)");
    TRACE("    Directly redundant:  " << redundant_trap_count);
    if (fallback_sequence_starts_.size() > 0) {
      TRACE("    Covered by fallback: " << covered_trap_count << " ("
                                        << fallback_sequence_starts_.size()
                                        << " sequences)");
      TRACE("Fallback: " << instruction_count << " instructions "
                         << "(avg " << std::fixed << std::setprecision(1)
                         << avg_fallback_size << " per trap)");
    }
  }
}
#endif

#undef TRACE

}  // namespace v8::internal::compiler::turboshaft
