// Copyright 2025 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "src/compiler/turboshaft/wasm-bounds-check-elimination-reducer.h"

#include <iomanip>

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
  // Collect the snapshots of all predecessors.
  predecessor_bounds_check_snapshots_.clear();
  predecessor_non_negative_snapshots_.clear();
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
  }
  // Without merge functions, the new snapshots only contain what is
  // known in all predecessors, that is, in the common ancestor of the
  // predecessor snapshots.
  known_bounds_checks_.StartNewSnapshot(
      base::VectorOf(predecessor_bounds_check_snapshots_));
  known_non_negative_offsets_.StartNewSnapshot(
      base::VectorOf(predecessor_non_negative_snapshots_));

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

  block_to_snapshot_mapping_[block->index()] =
      Snapshot{known_bounds_checks_.Seal(), known_non_negative_offsets_.Seal()};
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
                                                          bool inverted) {
  if (auto decoded_condition =
          TryExtractBoundChecksCondition(condition, inverted)) {
    const auto& [bounds_check, array_length] = *decoded_condition;
    ProcessBoundsCheck(trap_if, condition, bounds_check, array_length);
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
    OpIndex array_length) {
  const auto& [key, offset] = bounds_check;

  // Record the array length, so that we can use it when emitting new
  // bounds checks. Array length instructions can trap on null, so it
  // is simpler to keep using the very first one rather than inserting
  // a new one.
  array_lengths_.try_emplace(key.array, array_length);

  // For constant indices, check if [0, offset] would be valid.
  // If not, this offset is out-of-bounds and will always trap.
  if (!key.base.valid() && !OffsetRange::IsValidRange(0, offset)) {
    return;
  }

  std::optional<OffsetRange> non_negative = NonNegativeOffsets(key.base);
  std::optional<OffsetRange> known = KnownOffsets(key, non_negative);

  if (known.has_value() && known->Contains(offset)) {
    TRACE("  Redundant bounds check: offset "
          << offset << " within [" << known->lower() << ", " << known->upper()
          << "]");
    if (trap_if.valid()) {
      redundant_traps_.insert(trap_if);
    }
    return;
  }

  // Offsets within bounds after this check: {offset}, and every offset
  // from {n} to {offset} if {key.base + n} is non-negative (see the
  // proofs in the header).
  OffsetRange checked(offset);
  if (non_negative.has_value()) {
    OffsetRange::RelativePosition position = non_negative->Classify(offset);
    if (position == OffsetRange::RelativePosition::kInside ||
        position == OffsetRange::RelativePosition::kAbove) {
      checked = OffsetRange(non_negative->lower(), offset);
    }
  }

  if (!known.has_value()) {
    // First bounds check for this base and array.
    TRACE("  New bounds check: offset="
          << offset << " base=" << key.base << " array=" << key.array << " -> ["
          << checked.lower() << ", " << checked.upper() << "]");
    UpdateKnownBoundsChecks(key, offset, checked, trap_if, condition, known,
                            non_negative);
    return;
  }

  OffsetRange::RelativePosition position = known->Classify(offset);
  if (position == OffsetRange::RelativePosition::kOutOfRange) {
    // Given the previous bounds check, the current bounds check
    // will always trap.
    TRACE("  Out-of-range check: offset "
          << offset << " outside [" << known->lower() << ", " << known->upper()
          << "]");
    return;
  }
  DCHECK(position == OffsetRange::RelativePosition::kAbove ||
         position == OffsetRange::RelativePosition::kBelow);

  // The hull of {known} and {offset} is valid since {offset} is above
  // or below {known}, but the hull with all of {checked} might not be.
  std::optional<OffsetRange> offsets = known->Hull(checked);
  if (!offsets.has_value()) offsets = known->Hull(OffsetRange(offset));
  DCHECK(offsets.has_value());
  TRACE("  Extend range: [" << known->lower() << ", " << known->upper()
                            << "] -> [" << offsets->lower() << ", "
                            << offsets->upper() << "]");

  // The trap that established the bound of {known} on the side of
  // {offset} can start a fallback sequence.
  uint32_t prev_offset = position == OffsetRange::RelativePosition::kAbove
                             ? known->upper()
                             : known->lower();
  RecordFallbackSequence(key, prev_offset, trap_if, offset);
  UpdateKnownBoundsChecks(key, offset, *offsets, trap_if, condition, known,
                          non_negative);
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
  DCHECK_NE(array_length_it, array_lengths_.end());
  OpIndex array_length = array_length_it->second;

  FallbackInstructionSequence new_sequence(
      phase_zone_, key, array_length, *offsets, previous.known_offsets,
      previous.non_negative_offsets, previous.start_index, fallback_end_index,
      previous.start_position, current_position_ + 1,
      {previous.trap_if, trap_if});
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
    OpIndex trap_if, OpIndex condition,
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
      BoundsCheck(key, offset), TrapInfo{trap_if, start_index, start_position,
                                         known_before, non_negative_before});
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

// Attempts to decode a bounds check condition of the form:
//     n < a.length   or   base + n < a.length
// If the condition cannot be interpreted as a bounds check, returns
// std::nullopt. Otherwise, returns a {BoundsCheck} describing the base,
// array, and offset, along with the index of the array length
// instruction.
std::optional<std::pair<BoundsCheck, OpIndex>>
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

  const ArrayLengthOp* array_length =
      graph_.Get(length).TryCast<ArrayLengthOp>();
  if (!array_length) return std::nullopt;

  auto [base, offset] = ExtractBaseAndOffset(index);
  OpIndex array = ResolveAliases(array_length->array());
  return std::pair{BoundsCheck(BoundsCheckKey(base, array), offset), length};
}

// Attempts to decode a condition showing that {base + n} is
// non-negative:
//     base + n < b    or   base + n <= b   (unsigned)
// where {b} is an array length or a small enough constant, or
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
      // than 2^31.
      if (!graph_.Get(right).Is<ArrayLengthOp>()) {
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

  auto [base, offset] = ExtractBaseAndOffset(index);
  if (!base.valid()) return std::nullopt;
  return std::pair{base, offset};
}

// Extract base expression and constant offset from an index expression.
std::pair<OpIndex, uint32_t>
WasmBoundsCheckEliminationAnalyzer::ExtractBaseAndOffset(OpIndex index) const {
  index = ResolveReplacements(index);
  // a[n]
  if (auto offset = TryExtractI32Const(index)) {
    return {OpIndex::Invalid(), *offset};
  }
  // a[base + n] / a[base - n]
  if (const WordBinopOp* op = graph_.Get(index).TryCast<WordBinopOp>()) {
    if (auto offset = TryExtractI32Const(op->right()); offset.has_value()) {
      switch (op->kind) {
        case WordBinopOp::Kind::kAdd:
          return {ResolveReplacements(op->left()), *offset};
        case WordBinopOp::Kind::kSub:
          // We are using modular arithmetic, so we can always replace
          // a subtraction by the addition of the opposite
          return {ResolveReplacements(op->left()), 0u - *offset};
        default:
          break;
      }
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
