// Copyright 2026 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Flags: --allow-natives-syntax --liftoff --no-wasm-lazy-compilation
// Flags: --wasm-tiering-budget=1000000000
// Flags: --turboshaft-wasm-bounds-check-elimination
// Flags: --turboshaft-verify-wasm-bounds-check-elimination

// Randomly generated loops, for the parts of bounds check elimination
// that reason about loops: induction on loop phis, bounds {i <= x}
// shown by {i != x} exits (possibly unrolled), lower bounds of {x} when
// entering the loop, and bounds that are a merge phi of an array length
// and constants, and decreasing loops {i = phi(i0, i - step)}, as
// {for (i = a.length - 1; i >= 0; i--)} or OCaml loops
// {for i = i0 downto lo}. The accesses in the loops also use duplicated
// computations, 64-bit indices, and arrays allocated by the function.
// Each function first runs with Liftoff, which gives the expected
// results, and then with optimized code: the results, the position of
// traps and the contents of the arrays must be the same.
//
// For stress testing: d8 ... -- <seed> [function count] [verbose|stats]
// With "verbose", each function is printed before its optimized runs.
// With "stats", the outcomes of the runs are counted.

d8.file.execute('test/mjsunit/wasm/wasm-module-builder.js');

const kSeed = (typeof arguments != 'undefined' && arguments.length > 0)
    ? Number(arguments[0]) : 0x5eed1007;
const kFunctionCount = (typeof arguments != 'undefined' &&
                        arguments.length > 1)
    ? Number(arguments[1]) : 60;
const kMode = typeof arguments != 'undefined' ? arguments[2] : undefined;
const kVerbose = kMode == 'verbose';
const kRunsPerFunction = 16;

let seed = kSeed | 0 || 1;
function random(n) {
  // xorshift32
  seed ^= seed << 13;
  seed ^= seed >>> 17;
  seed ^= seed << 5;
  return (seed >>> 0) % n;
}
function pick(list) {
  return list[random(list.length)];
}
function chance(n) {
  return random(n) == 0;
}

// Locals: parameters a, b (arrays), p, q, then i (loop variable), x (its
// bound), j (inner loop variable), count and jcount (iterations left in
// the loops, which count down and are not used as indices), acc
// (result), c (an array allocated by the function), and
// s, t, u (structs: u is s or t), and w (a 64-bit integer).
const kLocals = {a: 0, b: 1, p: 2, q: 3, i: 4, x: 5, j: 6, count: 7,
                 jcount: 8, acc: 9, c: 10, s: 11, t: 12, u: 13, w: 14};

// Expressions (all i32).
const constant = (v) => ({k: 'const', v: v | 0});
const local = (n) => ({k: 'local', n});
const len = (arr) => ({k: 'len', arr});
const binop = (op, l, r) => ({k: 'binop', op, l, r});
const cmp = (op, l, r) => ({k: 'cmp', op, l, r});
const eqz = (e) => ({k: 'eqz', e});

// Statements.
const set = (n, e) => ({k: 'set', n, e});
const brIf = (label, c) => ({k: 'brif', label, c});

// Negations of comparisons: !(l op r) is (l neg r).
const kNegated = {
  lt_s: 'ge_s', lt_u: 'ge_u', le_s: 'gt_s', le_u: 'gt_u',
  gt_s: 'le_s', gt_u: 'le_u', ge_s: 'lt_s', ge_u: 'lt_u',
  eq: 'ne', ne: 'eq',
};
// (l op r) is (r swapped l).
const kSwapped = {
  lt_s: 'gt_s', lt_u: 'gt_u', le_s: 'ge_s', le_u: 'ge_u',
  gt_s: 'lt_s', gt_u: 'lt_u', ge_s: 'le_s', ge_u: 'le_u',
  eq: 'eq', ne: 'ne',
};

// The comparison {l op r}, written in various ways.
function relation(op, l, r) {
  switch (random(4)) {
    case 0: return cmp(kSwapped[op], r, l);
    case 1: return eqz(cmp(kNegated[op], l, r));
    default: return cmp(op, l, r);
  }
}

// The negation of {l op r}, written in various ways.
function negatedRelation(op, l, r) {
  return chance(3) ? eqz(relation(op, l, r))
                   : relation(kNegated[op], l, r);
}

// {base + d}, written in various ways. A constant on the left is rare:
// the pipeline moves constants to the right before the analysis (in loop
// peeling), so that the analysis only reads {base + d}, and runs with
// loop peeling disabled would not test much with it.
function index(base, d) {
  d |= 0;
  switch (random(16)) {
    case 0: return binop('add', constant(d), base);
    case 1: case 2: return binop('sub', base, constant(-d));
    case 3: case 4: {
      let d1 = random(5) - 2;
      return binop('add', binop('add', base, constant(d1)),
                   constant(d - d1));
    }
    default:
      return d == 0 && chance(2) ? base : binop('add', base, constant(d));
  }
}

// {arr.length - r}, written in various ways.
function lengthMinus(arr, r) {
  r |= 0;
  if (r == 0 && chance(2)) return len(arr);
  return chance(3) ? binop('add', len(arr), constant(-r))
                   : binop('sub', len(arr), constant(r));
}

function randomReduction() {
  return pick([0, 0, 1, 1, 1, 2, 3, 0x10000, 0x10001, -1, 0x7fffffff]);
}

// A small offset, rarely negative, as most accesses in loops start from
// {i}.
function smallOffset() {
  return pick([0, 0, 0, 0, 1, 1, 1, 2, 2, 3, -1, -2]);
}

function randomOffset() {
  return chance(16) ? pick([0x7fffffff, -0x80000000, 0x40000000, -0x40000000])
                    : smallOffset();
}

// A loop bound {x} of any kind.
function wildBound(step) {
  switch (random(6)) {
    case 0:
      return constant(pick([0, 1, 2, 3, 5, 8, -1, 0x7fffffff, 0x7ffffffe,
                            0x80000000 - step, 0x7fffffff - step,
                            -0x80000000, 0x40000000]));
    case 1:
      return lengthMinus('a', randomReduction());
    case 2:
      return lengthMinus('b', randomReduction());
    case 3:
      return local('p');
    default:
      return phiBound(pick([0, 1, 3, -1, 0x7fffffff]),
                      pick([1, 0, 2, -1]));
  }
}

// {(cond ? a.length : k) - r}, or the converse.
function phiBound(k, r) {
  let cond = pick([local('q'), {k: 'isnull', arr: 'b'},
                   cmp('lt_s', local('p'), constant(3))]);
  let phi = chance(4) ? {k: 'select', c: cond, t: constant(k), e: len('a')}
                      : {k: 'select', c: cond, t: len('a'), e: constant(k)};
  return r == 0 && chance(2) ? phi : binop('sub', phi, constant(r));
}

function access(arr, idx) {
  return chance(2) ? {k: 'store', arr, idx, value: random(1000) + 1}
                   : {k: 'load', arr, idx};
}

// Accesses to {arr} at offsets near {center} of {base}.
function burst(arr, base, center) {
  let statements = [];
  let size = 1 + random(4);
  for (let n = 0; n < size; n++) {
    statements.push(access(chance(5) ? pick(['a', 'b']) : arr,
                           index(base, center + pick([0, 0, 1, 1, 2, -1]))));
  }
  return statements;
}

// Statements using the variable {iv}. {labels.out} is the label to exit
// the current loop, if any. With {guarded_only}, accesses are guarded by
// conditions, so that they do not always trap for a large {iv}.
// The array mostly accessed by {randomStatements}, when set.
let preferredArray = null;

function randomStatements(iv, labels, depth, guarded_only = false) {
  let statements = [];
  let count = 1 + random(3);
  let arr = preferredArray ?? (chance(4) ? 'b' : 'a');
  for (let n = 0; n < count; n++) {
    switch (guarded_only ? pick([4, 5, 9]) : random(12)) {
      case 0: case 1: case 2:
        statements.push(...burst(arr, local(iv), smallOffset()));
        break;
      case 3:
        statements.push(access(pick(['a', 'a', 'b']),
                               index(local(iv), randomOffset())));
        break;
      case 4: case 5: {
        // if (iv + d < bound) { accesses around iv + d } else { ... }
        let d = smallOffset();
        let r = pick([0, 0, 1, 1, 2]);
        let op = pick(['lt_s', 'lt_s', 'lt_u', 'le_s', 'le_u']);
        let bound = pick([lengthMinus(arr, r), lengthMinus(arr, r),
                          local('x'), index(local('x'), random(3) - 1)]);
        let then = burst(arr, local(iv), d + random(r + 2));
        let otherwise = [];
        if (guarded_only) {
          // Nothing that traps or leaves the loop.
        } else if (labels.out && chance(3)) {
          otherwise.push({k: 'br', label: labels.out});
        } else if (chance(2)) {
          otherwise.push(...burst(arr, local(iv), d));
        }
        let c = relation(op, index(local(iv), d), bound);
        statements.push(chance(2) ? {k: 'if', c, then, else: otherwise}
                                  : {k: 'if', c: eqz(c), then: otherwise,
                                     else: then});
        break;
      }
      case 6:
        if (labels.out) {
          statements.push(brIf(labels.out, chance(2)
              ? cmp('eq', local(iv), local('p'))
              : relation('lt_s', local('p'), index(local(iv), random(3)))));
        }
        break;
      case 7:
        if (depth == 0) {
          // for (j = 0, jcount = 6; j < bound && jcount != 0; j++, jcount--)
          //   { accesses around j }
          let bound = pick([local('i'), lengthMinus(arr, pick([0, 1, 2])),
                            local('x'), constant(3)]);
          let op = pick(['lt_s', 'lt_u', 'le_s', 'ne']);
          let body = randomStatements('j', {out: 'jout'}, depth + 1);
          statements.push(
              set('j', constant(pick([0, 0, 1]))),
              set('jcount', constant(6)),
              {k: 'block', label: 'jout', body: [{
                k: 'loop', label: 'jloop', body: [
                  brIf('jout', eqz(local('jcount'))),
                  set('jcount', index(local('jcount'), -1)),
                  brIf('jout', negatedRelation(op, local('j'), bound)),
                  ...body,
                  set('j', index(local('j'), 1)),
                  {k: 'br', label: 'jloop'},
                ]}]});
        }
        break;
      case 8:
        statements.push(...burst(arr, local('x'), smallOffset() - 2));
        break;
      case 9:
        statements.push(set('acc', binop('xor', local('acc'),
                                         constant(random(100)))));
        break;
      default:
        statements.push(...burst(arr, local(iv), 0));
        break;
    }
  }
  return statements;
}

// A function from {randomLoopFunction}. When the bound {x} is a merge
// phi, the loop may be in a branch, after which {x} is used: the length
// alias for {x} only holds in the loop and the blocks it dominates, not
// after the merge, where {x} may come from the constant input of the phi.
function randomFunction() {
  let state = {};
  let statements = randomLoopFunction(state);
  if (!state.phi || chance(2)) return statements;
  let [setX, ...loop] = statements;
  return [
    setX,
    // Accesses showing a minimal length of {a}, which conditions on {x}
    // (seen as {a.length - r}) need.
    ...burst('a', constant(0), 1),
    // The loop is in either branch: the analysis visits one before the
    // other.
    chance(2)
        ? {k: 'if', c: pick([local('q'), cmp('lt_s', local('p'), constant(3))]),
           then: burst('a', local('p'), 0), else: loop}
        : {k: 'if', c: pick([local('q'), cmp('lt_s', local('p'), constant(3))]),
           then: loop, else: burst('a', local('p'), 0)},
    // With {x} from the constant input, {x} is negative, so {k <u x}
    // holds: it shows that {k} is within bounds only if {x} is the array
    // length minus a constant.
    ...(() => {
      let k = random(6);
      return [{k: 'if', c: relation(pick(['lt_u', 'le_u', 'lt_s']),
                                    constant(k), local('x')),
               then: [access('a', constant(k))], else: []}];
    })(),
    ...randomStatements('i', {}, 0),
  ];
}

// Ways for a function to differ from a loop that the analysis should
// recognize. Each one is applied with probability 1/10.
const kMutations = ['init', 'step', 'bound', 'op', 'inline', 'guard',
                    'netest', 'phi'];

// A function with a loop over {i = phi(init, i + step)}, followed by
// statements using {i} after the loop. The loop is of one of these kinds:
// - while: {i < x} or {i <= x} (signed or unsigned) is tested at the top
//   of each iteration;
// - ne: {i + k == x} exits the loop, for each {k} below the step (as an
//   OCaml for loop, unrolled {step} times), and the loop is only entered
//   if {x >= init};
// - dowhile: the back edge is taken when the incremented {i} is below
//   {x}.
// The bound {x} is a constant, an array length minus a constant, or for
// ne loops, a merge phi of an array length and a constant, minus a
// constant, which is only the array length in the loop. Some while loops
// start close to 2^31, with a constant bound close to the largest one for
// which {i + step} cannot overflow, or with an array length minus a
// constant, compared in any way: {i} then overflows within a few
// iterations when the bound does not prevent it.
// {state.phi} is set when the bound is a merge phi.
function randomLoopFunction(state) {
  let mutations = new Set(kMutations.filter(() => chance(10)));
  let kind = pick(['while', 'while', 'ne', 'ne', 'ne', 'dowhile']);
  let high = kind == 'while' && chance(6);
  let unroll = kind == 'ne' ? pick([1, 1, 1, 2, 2, 3, 4]) : 1;
  let step = kind == 'ne' ? unroll : pick([1, 1, 1, 2, 3]);
  if (mutations.has('step')) {
    step = kind == 'ne'
        ? pick([unroll + 1, unroll - 1, -1])
        : pick([-1, 0x40000000, 0x40000001, 0x3fffffff, 0x7fffffff]);
  }
  let init = mutations.has('init')
      ? pick([-1, 0x7ffffffe, 0x7ffffff0, -0x80000000])
      : pick([0, 0, 0, 1, 2]);

  // The bound, and the comparisons that show that {i} does not overflow.
  let x, ops;
  let r = pick([0, 1, 1, 2, 3]);
  switch (mutations.has('bound') ? 'wild'
          : high ? pick(['highConst', 'highLen'])
          : pick(['len', 'len', 'len', 'const',
                  ...(kind == 'ne' ? ['phi', 'phi'] : [])])) {
    case 'highConst': {
      let c = pick([0x80000000 - step, 0x7fffffff - step, 0x80000001 - step,
                    0x7fffffff]) | 0;
      x = constant(c);
      if (!mutations.has('init')) init = (c - step * random(4)) | 0;
      ops = ['lt_s', 'le_s', 'lt_u', 'le_u'];
      break;
    }
    case 'highLen':
      x = lengthMinus(chance(4) ? 'b' : 'a', r);
      if (!mutations.has('init')) init = 0x7ffffff0 + random(15);
      // Unsigned comparisons with {a.length - r} do not show anything
      // when {r > 0}.
      ops = ['lt_s', 'le_s', 'lt_u', 'le_u'];
      break;
    case 'wild':
      x = wildBound(step);
      ops = ['lt_s'];
      break;
    case 'len':
      x = lengthMinus(chance(4) ? 'b' : 'a', r);
      // {a.length - r} may wrap around as an unsigned integer.
      ops = r == 0 ? ['lt_s', 'le_s', 'lt_u', 'le_u'] : ['lt_s', 'le_s'];
      break;
    case 'const':
      x = constant(pick([3, 5, 8, 100]));
      ops = ['lt_s', 'le_s', 'lt_u', 'le_u'];
      break;
    case 'phi':
      // When the phi is {k}, {x} is {k - r}, below the lower bound {init}
      // when entering the loop, unless mutated.
      x = mutations.has('phi')
          ? phiBound(pick([3, 0x7fffffff, 1]), pick([0, -1, 1]))
          : phiBound(0, pick([1, 1, 2]));
      ops = ['lt_s'];
      state.phi = true;
      break;
  }
  let op = mutations.has('op')
      ? pick(['lt_s', 'lt_u', 'le_s', 'le_u', 'ne', 'gt_s', 'ge_u'])
      : pick(ops);
  let bound = mutations.has('inline') ? x : local('x');

  // The loop. Its iterations are capped, at its top or before the
  // increment of {i}.
  let cap = pick([2, 5, 12, 40]);
  let body = [];
  let capCheck = [brIf('out', eqz(local('count'))),
                  set('count', index(local('count'), -1))];
  let capAtTop = !chance(4);
  if (capAtTop) body.push(...capCheck);
  if (kind == 'while') {
    body.push(brIf('out', negatedRelation(op, local('i'), bound)));
  }
  let skippedTest = mutations.has('netest') ? random(unroll + 1) : -1;
  for (let k = 0; k < unroll; k++) {
    let statements = randomStatements('i', {out: 'out'}, 0, high);
    if (k > 0) {
      // Accesses at {i + k}, as in an unrolled copy of the loop body.
      statements = statements.map(s => shiftStatement(s, k));
    }
    if (kind != 'ne') {
      body.push(...statements);
      continue;
    }
    let tested = k == skippedTest ? k + 1 : k;
    let test = brIf('out', relation('eq', index(local('i'), tested), bound));
    if (chance(4)) {
      body.push(test, ...statements);
    } else {
      body.push(...statements, test);
    }
  }
  if (!capAtTop) body.push(...capCheck);
  body.push(set('i', chance(16) ? binop('add', constant(step), local('i'))
                               : binop('add', local('i'), constant(step))));
  if (kind == 'dowhile') {
    body.push(brIf('loop', relation(op, local('i'), bound)));
  } else {
    body.push({k: 'br', label: 'loop'});
  }
  let loop = [
    set('i', constant(init)),
    set('count', constant(cap)),
    {k: 'block', label: 'out',
     body: [{k: 'loop', label: 'loop', body}]},
    ...randomStatements('i', {}, 0, high),
  ];

  // The entry guard: the loop is only entered if {x >= g}, possibly with
  // {x != g - 1} and {x >= g - 1}. When {g - 1} is 0, {x != 0} can also be
  // shown by a branch on {x}.
  let statements = [set('x', x)];
  let guard = !high && (kind == 'ne' || chance(2));
  let g = init;
  let excluded = chance(4);
  let unsigned = false;
  let enterOnZero = false;
  if (mutations.has('guard')) {
    switch (random(5)) {
      case 0: guard = false; break;
      case 1: unsigned = true; break;
      case 2: g = (init + 1) | 0; break;
      case 3: excluded = false; g = (init - 1) | 0; break;
      case 4: excluded = false; enterOnZero = true; g = (init - 1) | 0; break;
    }
  }
  if (!guard) return [...statements, ...loop];
  if (enterOnZero) {
    // The loop is only entered if {x == 0}.
    statements.push({k: 'if', c: local('x'), then: [{k: 'return'}],
                     else: []});
  }
  if (excluded) {
    // x >= g - 1 && x != g - 1, in various ways.
    let below = (g - 1) | 0;
    let forms = [
      {k: 'if', c: relation('eq', local('x'), constant(below)),
       then: [{k: 'return'}], else: []},
      {k: 'if', c: relation('ne', local('x'), constant(below)),
       then: [], else: [{k: 'return'}]},
    ];
    if (below == 0) {
      forms.push({k: 'if', c: local('x'), then: [], else: [{k: 'return'}]},
                 {k: 'if', c: eqz(local('x')), then: [{k: 'return'}],
                  else: []});
    }
    statements.push(pick(forms));
    g = below;
  }
  if (unsigned) {
    statements.push({k: 'if', c: relation('lt_u', local('x'), constant(g)),
                     then: [{k: 'return'}], else: []});
  } else if (chance(3)) {
    statements.push({k: 'if', c: relation('le_s', constant(g), local('x')),
                     then: loop, else: []});
    return statements;
  } else {
    statements.push({k: 'if', c: relation('lt_s', local('x'), constant(g)),
                     then: [{k: 'return'}], else: []});
  }
  statements.push(...loop);
  return statements;
}

// Ways for a decreasing loop to differ from a loop that the analysis
// should recognize. Each one is applied with probability 1/5: some of
// them only lead to a wrong result when combined with particular inputs
// (for instance, an array shorter than the constant subtracted from its
// length).
const kDecreasingMutations = ['init', 'step', 'lo', 'op', 'guard', 'netest'];

// A function with a loop over {i = phi(i0, i - step)}, followed by
// statements using {i} after the loop. The loop is of one of these kinds:
// - ge: the back edge is taken when the decremented {i} is at least {lo}
//   (as for {for (i = i0; i >= lo; i--)});
// - gt: {i > lo} is tested at the top of each iteration, before {i} is
//   decremented, so that {lo} must be at least {step - 1};
// - ne: {i - k == lo} exits the loop, for each {k} below the step (as an
//   OCaml loop {for i = i0 downto lo}, unrolled {step} times).
// The start {i0} is an array length minus a constant, or a small constant,
// and the loop is only entered if {i0 >= lo}, for a small {lo >= 0}.
function randomDecreasingFunction() {
  let mutations = new Set(kDecreasingMutations.filter(() => chance(5)));
  let kind = pick(['ge', 'gt', 'ne', 'ne']);
  let unroll = kind == 'ne' ? pick([1, 1, 1, 2, 2, 3]) : 1;
  let step = kind == 'ne' ? unroll : kind == 'ge' ? pick([1, 1, 1, 2, 3])
                                                  : pick([1, 1, 2]);
  let lo = kind == 'gt' ? step - 1 + random(2) : pick([0, 0, 0, 1]);
  if (mutations.has('step')) {
    step = kind == 'ne' ? pick([unroll + 1, unroll - 1, -1])
                        : kind == 'gt' ? lo + 2 : pick([-1, 0x40000001]);
  }
  if (mutations.has('lo')) lo = kind == 'gt' ? step - 2 : -1;
  let i0 = mutations.has('init')
      ? constant(pick([-1, 0x7fffffff, -0x80000000, 0x40000000]))
      : chance(4) ? constant(pick([0, 1, 3, 8]))
                  : lengthMinus(chance(4) ? 'b' : 'a', pick([0, 1, 1, 2, 3]));

  // The loop. Its iterations are capped, at its top or before the
  // decrement of {i}.
  let cap = pick([2, 5, 12, 40]);
  let body = [];
  let capCheck = [brIf('out', eqz(local('count'))),
                  set('count', index(local('count'), -1))];
  let capAtTop = !chance(4);
  if (capAtTop) body.push(...capCheck);
  if (kind == 'gt') {
    let op = mutations.has('op') ? pick(['ge_s', 'gt_u', 'ne']) : 'gt_s';
    body.push(brIf('out', negatedRelation(op, local('i'), constant(lo))));
  }
  let skippedTest = mutations.has('netest') ? random(unroll + 1) : -1;
  for (let k = 0; k < unroll; k++) {
    let statements = randomStatements('i', {out: 'out'}, 0);
    if (k > 0) {
      // Accesses at {i - k}, as in an unrolled copy of the loop body.
      statements = statements.map(s => shiftStatement(s, -k));
    }
    if (kind != 'ne') {
      body.push(...statements);
      continue;
    }
    let tested = k == skippedTest ? k + 1 : k;
    let test = brIf('out', relation('eq', index(local('i'), -tested),
                                    constant(lo)));
    if (chance(4)) {
      body.push(test, ...statements);
    } else {
      body.push(...statements, test);
    }
  }
  if (!capAtTop) body.push(...capCheck);
  body.push(set('i', index(local('i'), -step)));
  if (kind == 'ge') {
    // {lo <= i}, or {lo - 1 < i}, or when mutated, a comparison that
    // lets {i} go below {lo}.
    let [op, bound] = mutations.has('op')
        ? pick([['le_u', lo], ['lt_s', lo - 2], ['ne', lo - 1]])
        : pick([['le_s', lo], ['lt_s', lo - 1]]);
    body.push(brIf('loop', relation(op, constant(bound), local('i'))));
  } else {
    body.push({k: 'br', label: 'loop'});
  }
  let loop = [
    set('i', local('x')),
    set('count', constant(cap)),
    {k: 'block', label: 'out',
     body: [{k: 'loop', label: 'loop', body}]},
    ...randomStatements('i', {}, 0),
  ];

  // The entry guard: the loop is only entered if {x >= g}.
  let statements = [set('x', i0)];
  let g = lo;
  let guard = true;
  let unsigned = false;
  if (mutations.has('guard')) {
    switch (random(3)) {
      case 0: guard = false; break;
      case 1: unsigned = true; break;
      case 2: g = lo - 1; break;
    }
  }
  if (!guard) return [...statements, ...loop];
  if (unsigned) {
    statements.push({k: 'if', c: relation('lt_u', local('x'), constant(g)),
                     then: [{k: 'return'}], else: []});
  } else if (chance(3)) {
    statements.push({k: 'if', c: relation('le_s', constant(g), local('x')),
                     then: loop, else: []});
    return statements;
  } else {
    statements.push({k: 'if', c: relation('lt_s', local('x'), constant(g)),
                     then: [{k: 'return'}], else: []});
  }
  statements.push(...loop);
  return statements;
}

// Ways for a loop over an allocated array to differ from a loop whose
// accesses are within bounds. Each one is applied with probability 1/5.
const kAllocationMutations = ['bound', 'other', 'op', 'init', 'size'];

// A function that allocates an array {c} of length {n}, followed by a loop
// {for (i = 0; i < bound; i++)} accessing mostly {c}, where {bound} is
// {n} (computed again in some cases), {c.length}, or the length of the
// array that {n} is the length of, as in fill and copy loops:
// - fill: {n} is an array length minus a constant, a small parameter, or
//   a constant;
// - copy: {n} is {a.length} (or {b.length}), and the loop also reads the
//   source array;
// - fixed: {c} is allocated with {array.new_fixed}, with a constant
//   length.
// Statements using {c} follow the loop.
function randomAllocationFunction() {
  let mutations = new Set(kAllocationMutations.filter(() => chance(5)));
  let kind = pick(['fill', 'fill', 'copy', 'fixed']);
  let source = chance(4) ? 'b' : 'a';
  let n, bounds;
  switch (kind) {
    case 'fill':
      switch (random(4)) {
        case 0: n = lengthMinus(source, pick([0, 1, 2])); break;
        case 1: n = binop('and', local('p'), constant(15)); break;
        case 2: n = local('p'); break;  // May trap: too large or negative.
        default: n = constant(pick([0, 1, 3, 8])); break;
      }
      bounds = [local('x'), local('x'), len('c')];
      break;
    case 'copy':
      n = len(source);
      bounds = [len(source), len(source), local('x'), len('c')];
      break;
    case 'fixed':
      n = constant(pick([0, 1, 2, 3, 5]));
      bounds = [n, n, len('c')];
      break;
  }
  let bound = pick(bounds);
  if (mutations.has('bound')) bound = index(bound, 1);
  if (mutations.has('other')) bound = len(source == 'a' ? 'b' : 'a');
  let size = mutations.has('size') ? index(local('x'), -1) : local('x');
  let op = mutations.has('op') ? pick(['le_s', 'le_u'])
                               : pick(['lt_s', 'lt_u']);
  let init = mutations.has('init') ? -1 : 0;

  let saved = preferredArray;
  preferredArray = chance(4) ? source : 'c';
  let body = [brIf('out', eqz(local('count'))),
              set('count', index(local('count'), -1)),
              brIf('out', negatedRelation(op, local('i'), bound)),
              ...randomStatements('i', {out: 'out'}, 0)];
  if (kind == 'copy') {
    body.push(access(source, local('i')));
  }
  body.push(set('i', index(local('i'), 1)), {k: 'br', label: 'loop'});
  let after = randomStatements('i', {}, 0);
  preferredArray = saved;
  return [
    set('x', n),
    kind == 'fixed' && !mutations.has('size')
        ? {k: 'newfixed', count: n.v}
        : {k: 'new', len: size},
    set('i', constant(init)),
    set('count', constant(pick([2, 5, 12, 40]))),
    {k: 'block', label: 'out', body: [{k: 'loop', label: 'loop', body}]},
    ...after,
  ];
}

// A base expression for {randomCanonicalFunction}, over the parameters
// {p} and {q}, or over {iv} (a loop variable) in loops.
function randomBaseExpression(iv) {
  let operand = () => pick([local(iv), local(iv), local('q'),
                            constant(pick([3, 7, 15, -16]))]);
  switch (random(6)) {
    case 0: case 1:
      return binop(pick(['and', 'or', 'xor', 'sub', 'mul']), local(iv),
                   operand());
    case 2:
      return binop(pick(['shl', 'shr_s', 'shr_u']), local(iv),
                   chance(2) ? constant(1 + random(3)) : local('q'));
    case 3:
      return {k: 'select',
              c: pick([local('q'), cmp('lt_s', local(iv), constant(3)),
                       {k: 'isnull', arr: 'b'}]),
              t: local(iv), e: local('q')};
    case 4:
      return chance(2) ? {k: 'aget', arr: 'b', idx: constant(random(3))}
                       : {k: 'sget'};
    default:
      return {k: 'high', signed: chance(2), e: local(iv)};
  }
}

const kSameKind = [['and', 'or', 'xor'], ['shl', 'shr_s', 'shr_u']];
const kCommutative = ['and', 'or', 'xor', 'mul', 'add'];

// The expression {e} computed again: the same value, possibly written
// differently (commuted), or, if {near_miss}, a close but different one
// (another operation of the same kind, swapped operands, another
// condition, the other extension). An element of an array or the field
// of a struct is loaded again, and may have changed in between.
function recompute(e, near_miss) {
  switch (e.k) {
    case 'binop': {
      if (!near_miss) {
        return kCommutative.includes(e.op) && chance(2)
            ? binop(e.op, e.r, e.l) : binop(e.op, e.l, e.r);
      }
      let kinds = kSameKind.find(kinds => kinds.includes(e.op));
      if (kinds && chance(3)) {
        return binop(pick(kinds.filter(op => op != e.op)), e.l, e.r);
      }
      return e.op == 'sub' || e.op == 'shl' ? binop(e.op, e.r, e.l)
          : binop(e.op == 'mul' ? 'add' : 'sub', e.l, e.r);
    }
    case 'select':
      if (!near_miss) return {...e};
      return chance(2) ? {...e, t: e.e, e: e.t}
                       : {...e, c: eqz(e.c)};
    case 'high':
      return {...e, signed: near_miss ? !e.signed : e.signed};
    default:
      return {...e};
  }
}

// 64-bit expressions: {e} extended to 64 bits, the local {w}, binary
// operations, and constants (BigInts).
const extend = (e, signed) => ({k: 'extend', e, signed});
const w64 = () => ({k: 'w64'});
const binop64 = (op, l, r) => ({k: 'binop64', op, l, r});
const const64 = (v) => ({k: 'const64', v: BigInt.asIntN(64, BigInt(v))});
const wrap = (e) => ({k: 'wrap', e});

// {trunc(w + d)}, written in various ways, where the constant may have
// high bits, which the truncation discards ([truncated-addition]).
function truncatedIndex(d) {
  let c = BigInt(d) + (chance(6) ? BigInt(pick([1, -1, 3])) << 32n : 0n);
  switch (random(8)) {
    case 0: return wrap(binop64('add', const64(c), w64()));
    case 1: case 2: return wrap(binop64('sub', w64(), const64(-c)));
    default:
      return d == 0 && chance(2) ? wrap(w64())
                                 : wrap(binop64('add', w64(), const64(c)));
  }
}

// Statements accessing {arr} at truncations of 64-bit additions to {w}:
// {trunc(w + d)} is {trunc(w) + d}, when an operation computing
// {trunc(w)} exists before. Near misses ({trunc(d - w)}, {trunc(w * 2)})
// are not such additions.
function truncatedStatements(arr) {
  let result = [access(arr, wrap(w64()))];
  for (let n = 1 + random(4); n > 0; n--) {
    switch (random(5)) {
      case 0: case 1: {
        let d = smallOffset();
        let r = pick([0, 0, 1, 2]);
        let then = [];
        for (let k = 1 + random(3); k > 0; k--) {
          then.push(access(arr, truncatedIndex(d + random(r + 2))));
        }
        result.push({k: 'if', c: relation(pick(['lt_u', 'lt_s']),
                                          truncatedIndex(d),
                                          lengthMinus(arr, r)),
                     then, else: []});
        break;
      }
      case 2:
        result.push(access(arr, truncatedIndex(smallOffset())));
        break;
      case 3:
        result.push(access(arr, wrap(chance(2)
            ? binop64('sub', const64(smallOffset()), w64())
            : binop64('mul', w64(), const64(2)))));
        break;
      default:
        // A loop in a single block, whose exit condition computes
        // {trunc(w)} after accesses at {trunc(w + d)}: the analysis
        // sees the condition at the loop header (because of the
        // induction variable {jcount}), before the accesses, although
        // the operation computing {trunc(w)} comes after them.
        let accesses = [];
        for (let k = 2 + random(3), d = 1; k > 0; k--, d += random(2)) {
          accesses.push(access(arr, truncatedIndex(d)));
        }
        // The induction variable is {jcount}, not {i}, which may be the
        // variable of an enclosing loop.
        result.push(
            set('jcount', constant(0)),
            {k: 'loop', label: 'tloop', body: [
              ...accesses,
              set('j', relation('lt_s', wrap(w64()), len(arr))),
              {k: 'set64', e: binop64('add', w64(), const64(1))},
              set('jcount', index(local('jcount'), 1)),
              brIf('tloop', local('j')),
            ]});
        break;
    }
  }
  return result;
}

// A 64-bit value {X}, and a 32-bit value {e} equal to its truncation,
// which accesses use: {w} or an extension of a 32-bit value, possibly
// plus a constant.
function randomWide() {
  let d = smallOffset();
  if (chance(3)) {
    // A constant with high bits makes {X} differ from the extension of
    // its truncation, so that only unsigned comparisons with zero-extended
    // values can be narrowed.
    let c = BigInt(d) + (chance(3) ? BigInt(pick([1, -1])) << 32n : 0n);
    return chance(2) ? {X: w64(), e: wrap(w64())}
                     : {X: binop64('add', w64(), const64(c)),
                        e: index(wrap(w64()), d)};
  }
  let v = pick([local('p'), local('q'),
                binop('and', local('p'), constant(15))]);
  let x = extend(v, chance(2));
  switch (random(3)) {
    case 0: return {X: x, e: v};
    case 1: return {X: binop64('add', x, const64(d)), e: index(v, d)};
    default: return {X: binop64('sub', x, const64(-d)), e: index(v, d)};
  }
}

// A 64-bit comparison {X < zext(a.length - r)} (or {<=}), unsigned, as
// languages with 64-bit integers compare indices with lengths: it shows
// that {X} is less than 2^32, and so is the same as the comparison of
// its truncation {e} ([narrowed-comparison]). Accesses at {e + k} follow
// when it holds, in a branch, or after leaving a block when it does not
// hold (as wasm_of_ocaml does with {X >= zext(a.length - 1)}). Near misses:
// signed comparisons, and sign-extended lengths.
// An array {c} allocated with a truncated 64-bit length {trunc(n)} in a
// branch, while the other branch computes {trunc(n)} for an access. Both
// truncations have the same canonical value, but only the one in the
// current branch is available where the length is needed: by the guard
// of a fallback sequence of accesses to {c}, after a condition
// {x < trunc(n - r)} that shows that their first offsets are within
// bounds (the decomposition of {trunc(n - r)} uses the first truncation
// of {n} seen). The test {trunc(n) != 0} keeps {trunc(n - r)} from
// wrapping around, so that an unsigned condition shows something. {n} is
// {w | m}, which machine optimization does not fold with {-r}, and of
// which no truncation comes before (it would then be used, and would be
// available).
function truncatedLengthStatements(arr) {
  let n = binop64('or', w64(), const64(pick([1, 4, 16])));
  let r = pick([1, 1, 2]);
  let x = pick([local('p'), local('q'), binop('and', local('p'), constant(7))]);
  let accesses = [];
  for (let k = 0, count = r + 3 + random(2); k < count; k++) {
    accesses.push(access('c', index(x, k)));
  }
  let allocating = [
    {k: 'new', len: wrap(n)},
    {k: 'if', c: wrap(n), then: [
      {k: 'if', c: relation(pick(['lt_u', 'lt_u', 'lt_s']), x,
                            wrap(binop64('add', n, const64(-r)))),
       then: accesses, else: []},
    ], else: []},
  ];
  let other = [access(arr, wrap(n))];
  let c = pick([local('q'), cmp('lt_s', local('p'), constant(3))]);
  return [chance(2) ? {k: 'if', c, then: other, else: allocating}
                    : {k: 'if', c, then: allocating, else: other}];
}

function narrowedStatements(arr, in_loop) {
  let {X, e} = randomWide();
  if (chance(4)) {
    // {X < zext(c)} with a constant {c} below 2^31 shows that {trunc(X)}
    // is non-negative, so that a signed check {trunc(X) < a.length} is
    // a bounds check. A sign-extended constant {c >= 2^31} is a near miss:
    // {X} can then be 2^32 or more.
    let c = pick([0x7fffffff, 0x80000000, 0x80000001, 8]);
    let d = smallOffset();
    return [{k: 'if',
             c: {k: 'cmp64', op: pick(['lt_u', 'le_u']), l: X,
                 r: extend(constant(c), chance(2))},
             then: [{k: 'if', c: relation('lt_s', index(e, d), len(arr)),
                     then: [access(arr, index(e, d))], else: []}],
             else: []}];
  }
  let r = pick([0, 0, 1, 2]);
  let bound = extend(lengthMinus(arr, r), chance(6));
  let op = chance(6) ? pick(['lt_s', 'le_s']) : pick(['lt_u', 'le_u']);
  let accesses = [];
  for (let k = 1 + random(3); k > 0; k--) {
    accesses.push(access(arr, index(e, random(r + 1))));
  }
  // An access at {e} before the comparison shows that {e} is non-negative,
  // so that a signed comparison of {e} is a bounds check: a signed
  // comparison of {X}, which may differ from {e} in its high bits, is not.
  let before = chance(2) ? [access(arr, e)] : [];
  return [...before, ...narrowedCheck(X, bound, op, accesses, in_loop)];
}

// The comparison {X op bound}, followed by {accesses} when it holds. In a
// loop, it may also be an exit test, which then dominates the back edge,
// so that the conditions of the loop include it.
function narrowedCheck(X, bound, op, accesses, in_loop) {
  if (in_loop && chance(3)) {
    return [brIf('out', {k: 'cmp64', op: kNegated[op], l: X, r: bound}),
            ...accesses];
  }
  switch (random(3)) {
    case 0:
      return [{k: 'if', c: {k: 'cmp64', op, l: X, r: bound},
               then: accesses, else: []}];
    case 1:
      return [{k: 'if', c: {k: 'cmp64', op: kSwapped[op], l: bound, r: X},
               then: accesses, else: []}];
    default:
      // Leaving a block rather than returning, which would skip the rest
      // of the function too often.
      return [{k: 'block', label: 'checked', body: [
                brIf('checked', {k: 'cmp64', op: kNegated[op], l: X,
                                 r: bound}),
                ...accesses]}];
  }
}

// A function that computes a few base expressions several times (see
// {randomBaseExpression}), and accesses arrays at small offsets of them,
// in guarded accesses, merges and loops, possibly with stores in between.
// The analysis identifies equal computations (see {CanonicalValue}), so
// facts shown for one of them are used for the others, which must
// compute the same value. Near misses (see {recompute}) must not be
// identified.
function randomCanonicalFunction() {
  let bases = [randomBaseExpression('p')];
  if (chance(2)) bases.push(randomBaseExpression('p'));
  let near_miss = () => chance(4);
  let use = () => recompute(pick(bases), near_miss());
  let arr = chance(4) ? 'b' : 'a';
  // Statements using the base expressions, in a loop over {i} when
  // {in_loop}.
  function statements(depth, in_loop) {
    let result = [];
    let count = 2 + random(4);
    for (let n = 0; n < count; n++) {
      if (chance(8)) {
        result.push(...truncatedStatements(arr));
        continue;
      }
      if (chance(8)) {
        result.push(...truncatedLengthStatements(arr));
        continue;
      }
      // As in wasm_of_ocaml's bounds checks, which are 64-bit.
      if (chance(5)) {
        result.push(...narrowedStatements(arr, in_loop));
        continue;
      }
      if (chance(6)) {
        // A condition combining a bounds check with another condition:
        // when {(e + d >= a.length) | c} does not hold, {e + d} is within
        // bounds, and when {(e + d < a.length) & c} holds, too.
        let d = smallOffset();
        let length = lengthMinus(arr, pick([0, 0, 1]));
        let other = pick([local('q'), eqz(local('q')),
                          cmp('lt_s', local('p'), constant(random(8)))]);
        let c = chance(2)
            ? binop('or', cmp('ge_u', index(use(), d), length), other)
            : binop('and', cmp('lt_u', index(use(), d), length),
                    cmp('lt_s', local('p'), constant(random(8))));
        result.push({k: 'if', c,
                     then: burst(arr, use(), d), else: burst(arr, use(), d)});
        continue;
      }
      switch (random(depth < 2 ? 9 : 6)) {
        case 0: case 1: {
          // if (e + d < a.length - r) { accesses around e' + d }
          let d = smallOffset();
          let r = pick([0, 0, 1, 2]);
          let c = relation(pick(['lt_u', 'lt_u', 'lt_s', 'le_s']),
                           index(use(), d), lengthMinus(arr, r));
          let then = [];
          for (let k = 1 + random(3); k > 0; k--) {
            then.push(access(chance(5) ? 'b' : arr,
                             index(use(), d + random(r + 2))));
          }
          let otherwise = chance(2) ? [] : [access(arr, index(use(), d))];
          result.push(chance(4) ? {k: 'if', c: eqz(c), then: otherwise,
                                   else: then}
                                : {k: 'if', c, then, else: otherwise});
          break;
        }
        case 2:
          result.push(access(chance(4) ? 'b' : arr,
                             index(use(), smallOffset())));
          break;
        case 3:
          // A store that may change an element or a field loaded by a
          // base expression ({a} and {b} may be the same array, and {u}
          // is {s} or {t}), with a small value, which can be an index.
          result.push(chance(2)
              ? {k: 'store', arr: pick(['a', 'b']), idx: constant(random(3)),
                 value: random(8)}
              : {k: 'sset', n: pick(['s', 't']), value: constant(random(8))});
          break;
        case 4:
          if (in_loop) {
            // Accesses at a base expression of the loop variable, so that
            // facts from one iteration are about other values in the next
            // one.
            let e = randomBaseExpression('i');
            result.push(access(arr, index(e, smallOffset())),
                        access(arr, index(recompute(e, near_miss()),
                                          smallOffset())));
          } else {
            result.push(set('acc', binop('xor', local('acc'),
                                         constant(random(100)))));
          }
          break;
        case 5:
          result.push(...burst(arr, use(), smallOffset()));
          break;
        case 6: case 7:
          // A merge: what is known after it is what both branches show.
          result.push({k: 'if', c: pick([local('q'),
                                         cmp('lt_s', local('p'), use())]),
                       then: statements(depth + 1, in_loop),
                       else: statements(depth + 1, in_loop)});
          break;
        default:
        {
          // for (i = 0; i < 4; i++) { ... }, with {x} as the variable of a
          // nested loop, so that it does not reset the one of the
          // enclosing loop.
          let v = in_loop ? 'x' : 'i';
          result.push(
              set(v, constant(0)),
              {k: 'block', label: 'out', body: [{
                k: 'loop', label: 'loop', body: [
                  brIf('out', cmp('ge_s', local(v), constant(4))),
                  ...statements(depth + 1, true),
                  set(v, index(local(v), 1)),
                  {k: 'br', label: 'loop'},
                ]}]});
          break;
        }
      }
    }
    return result;
  }
  // The structs: {s.f} and {t.f} are small values, and {u} is one of
  // them, so that loads of {u.f} cannot be replaced by the values stored.
  return [
    {k: 'snew', n: 's', value: binop('and', local('p'), constant(7))},
    {k: 'snew', n: 't', value: constant(random(8))},
    {k: 'salias', c: local('q')},
    {k: 'set64', e: extend(pick([local('p'), local('q'),
                                 binop('and', local('p'), constant(15))]),
                           chance(2))},
    ...statements(0, false),
  ];
}

// Replaces {i + d} by {i + d + k} in the accesses and conditions of {s}.
function shiftStatement(s, k) {
  function shift(e) {
    if (e === undefined) return e;
    switch (e.k) {
      case 'local':
        return e.n == 'i' ? index(e, k) : e;
      case 'binop': case 'cmp':
        return {...e, l: shift(e.l), r: shift(e.r)};
      case 'eqz':
        return {...e, e: shift(e.e)};
      default:
        return e;
    }
  }
  switch (s.k) {
    case 'load': case 'store':
      return {...s, idx: shift(s.idx)};
    case 'if':
      return {...s, c: shift(s.c), then: s.then.map(t => shiftStatement(t, k)),
              else: s.else.map(t => shiftStatement(t, k))};
    case 'brif':
      return {...s, c: shift(s.c)};
    default:
      // Inner loops and other statements are kept as they are.
      return s;
  }
}

// Code generation.
const kBinops = {add: kExprI32Add, sub: kExprI32Sub, xor: kExprI32Xor,
                 and: kExprI32And, or: kExprI32Ior, mul: kExprI32Mul,
                 shl: kExprI32Shl, shr_s: kExprI32ShrS, shr_u: kExprI32ShrU};
const kComparisons = {
  lt_s: kExprI32LtS, lt_u: kExprI32LtU, le_s: kExprI32LeS, le_u: kExprI32LeU,
  gt_s: kExprI32GtS, gt_u: kExprI32GtU, ge_s: kExprI32GeS, ge_u: kExprI32GeU,
  eq: kExprI32Eq, ne: kExprI32Ne,
};

function emitExpression(e, array) {
  switch (e.k) {
    case 'const':
      return wasmI32Const(e.v);
    case 'local':
      return [kExprLocalGet, kLocals[e.n]];
    case 'len':
      return [kExprLocalGet, kLocals[e.arr], kGCPrefix, kExprArrayLen];
    case 'isnull':
      return [kExprLocalGet, kLocals[e.arr], kExprRefIsNull];
    case 'binop':
      return [...emitExpression(e.l, array), ...emitExpression(e.r, array),
              kBinops[e.op]];
    case 'cmp':
      return [...emitExpression(e.l, array), ...emitExpression(e.r, array),
              kComparisons[e.op]];
    case 'eqz':
      return [...emitExpression(e.e, array), kExprI32Eqz];
    case 'select':
      return [...emitExpression(e.c, array), kExprIf, kWasmI32,
              ...emitExpression(e.t, array), kExprElse,
              ...emitExpression(e.e, array), kExprEnd];
    case 'aget':
      return [kExprLocalGet, kLocals[e.arr], ...emitExpression(e.idx, array),
              kGCPrefix, kExprArrayGet, array];
    case 'sget':
      return [kExprLocalGet, kLocals.u, kGCPrefix, kExprStructGet, struct, 0];
    case 'wrap':
      return [...emitExpression64(e.e, array), kExprI32ConvertI64];
    case 'cmp64':
      return [...emitExpression64(e.l, array), ...emitExpression64(e.r, array),
              kComparisons64[e.op]];
    case 'high':
      // The high half of {e} extended to 64 bits.
      return [...emitExpression(e.e, array),
              e.signed ? kExprI64SConvertI32 : kExprI64UConvertI32,
              ...wasmI64Const(32), kExprI64ShrU, kExprI32ConvertI64];
  }
  throw new Error(`unknown expression ${e.k}`);
}

const kBinops64 = {
  add: kExprI64Add, sub: kExprI64Sub, mul: kExprI64Mul, or: kExprI64Ior,
};
const kComparisons64 = {
  lt_s: kExprI64LtS, lt_u: kExprI64LtU, le_s: kExprI64LeS, le_u: kExprI64LeU,
  gt_s: kExprI64GtS, gt_u: kExprI64GtU, ge_s: kExprI64GeS, ge_u: kExprI64GeU,
};

function emitExpression64(e, array) {
  switch (e.k) {
    case 'extend':
      return [...emitExpression(e.e, array),
              e.signed ? kExprI64SConvertI32 : kExprI64UConvertI32];
    case 'w64':
      return [kExprLocalGet, kLocals.w];
    case 'binop64':
      return [...emitExpression64(e.l, array), ...emitExpression64(e.r, array),
              kBinops64[e.op]];
    case 'const64':
      return wasmI64Const(e.v);
  }
  throw new Error(`unknown 64-bit expression ${e.k}`);
}

function emitStatements(statements, array, labels) {
  let code = [];
  let depth = (label) => {
    let n = labels.lastIndexOf(label);
    assertTrue(n >= 0, label);
    return labels.length - 1 - n;
  };
  for (let s of statements) {
    switch (s.k) {
      case 'set':
        code.push(...emitExpression(s.e, array), kExprLocalSet, kLocals[s.n]);
        break;
      case 'load':
        code.push(kExprLocalGet, kLocals.acc, ...wasmI32Const(31), kExprI32Mul,
                  kExprLocalGet, kLocals[s.arr],
                  ...emitExpression(s.idx, array),
                  kGCPrefix, kExprArrayGet, array,
                  kExprI32Add, kExprLocalSet, kLocals.acc);
        break;
      case 'store':
        code.push(kExprLocalGet, kLocals[s.arr],
                  ...emitExpression(s.idx, array), ...wasmI32Const(s.value),
                  kGCPrefix, kExprArraySet, array);
        break;
      case 'if':
        code.push(...emitExpression(s.c, array), kExprIf, kWasmVoid,
                  ...emitStatements(s.then, array, [...labels, '']),
                  kExprElse,
                  ...emitStatements(s.else, array, [...labels, '']),
                  kExprEnd);
        break;
      case 'brif':
        code.push(...emitExpression(s.c, array), kExprBrIf, depth(s.label));
        break;
      case 'br':
        code.push(kExprBr, depth(s.label));
        break;
      case 'return':
        code.push(kExprLocalGet, kLocals.acc, kExprReturn);
        break;
      case 'new':
        code.push(...emitExpression(s.len, array), kGCPrefix,
                  kExprArrayNewDefault, array, kExprLocalSet, kLocals.c);
        break;
      case 'newfixed':
        for (let n = 0; n < s.count; n++) code.push(...wasmI32Const(n));
        code.push(kGCPrefix, kExprArrayNewFixed, array, s.count,
                  kExprLocalSet, kLocals.c);
        break;
      case 'set64':
        code.push(...emitExpression64(s.e, array), kExprLocalSet, kLocals.w);
        break;
      case 'snew':
        code.push(...emitExpression(s.value, array), kGCPrefix, kExprStructNew,
                  struct, kExprLocalSet, kLocals[s.n]);
        break;
      case 'sset':
        code.push(kExprLocalGet, kLocals[s.n],
                  ...emitExpression(s.value, array),
                  kGCPrefix, kExprStructSet, struct, 0);
        break;
      case 'salias':
        code.push(...emitExpression(s.c, array),
                  kExprIf, kWasmRefNull, struct,
                  kExprLocalGet, kLocals.s, kExprElse, kExprLocalGet, kLocals.t,
                  kExprEnd, kExprLocalSet, kLocals.u);
        break;
      case 'block':
      case 'loop':
        code.push(s.k == 'block' ? kExprBlock : kExprLoop, kWasmVoid,
                  ...emitStatements(s.body, array, [...labels, s.label]),
                  kExprEnd);
        break;
      default:
        throw new Error(`unknown statement ${s.k}`);
    }
  }
  return code;
}

// Pseudo-code, for debugging.
const kOperators = {
  add: '+', sub: '-', xor: '^', and: '&', or: '|', mul: '*', shl: '<<',
  shr_s: '>>s', shr_u: '>>u', lt_s: '<s', lt_u: '<u', le_s: '<=s',
  le_u: '<=u', gt_s: '>s', gt_u: '>u', ge_s: '>=s', ge_u: '>=u', eq: '==',
  ne: '!=',
};

function printExpression(e) {
  switch (e.k) {
    case 'const': return `${e.v}`;
    case 'local': return e.n;
    case 'len': return `${e.arr}.length`;
    case 'isnull': return `${e.arr} == null`;
    case 'binop': case 'cmp':
      return `(${printExpression(e.l)} ${kOperators[e.op]} ` +
             `${printExpression(e.r)})`;
    case 'eqz': return `!${printExpression(e.e)}`;
    case 'select':
      return `(${printExpression(e.c)} ? ${printExpression(e.t)} : ` +
             `${printExpression(e.e)})`;
    case 'aget': return `${e.arr}[${printExpression(e.idx)}]`;
    case 'sget': return 'u.f';
    case 'wrap': return `trunc(${printExpression64(e.e)})`;
    case 'cmp64':
      return `(${printExpression64(e.l)} ${kOperators[e.op]} ` +
             `${printExpression64(e.r)})`;
    case 'high':
      return `high${e.signed ? '_s' : '_u'}(${printExpression(e.e)})`;
  }
}

function printExpression64(e) {
  switch (e.k) {
    case 'extend':
      return `extend${e.signed ? '_s' : '_u'}(${printExpression(e.e)})`;
    case 'w64': return 'w';
    case 'binop64':
      return `(${printExpression64(e.l)} ${kOperators[e.op]} ` +
             `${printExpression64(e.r)})`;
    case 'const64': return `${e.v}L`;
  }
}

function printStatements(statements, indent = '  ') {
  let lines = [];
  for (let s of statements) {
    switch (s.k) {
      case 'set':
        lines.push(`${indent}${s.n} = ${printExpression(s.e)};`);
        break;
      case 'load':
        lines.push(`${indent}acc = acc * 31 + ${s.arr}[` +
                   `${printExpression(s.idx)}];`);
        break;
      case 'store':
        lines.push(`${indent}${s.arr}[${printExpression(s.idx)}] = ` +
                   `${s.value};`);
        break;
      case 'if':
        lines.push(`${indent}if ${printExpression(s.c)} {`,
                   ...printStatements(s.then, indent + '  '),
                   `${indent}} else {`,
                   ...printStatements(s.else, indent + '  '),
                   `${indent}}`);
        break;
      case 'brif':
        lines.push(`${indent}if ${printExpression(s.c)} br ${s.label};`);
        break;
      case 'br':
        lines.push(`${indent}br ${s.label};`);
        break;
      case 'return':
        lines.push(`${indent}return acc;`);
        break;
      case 'new':
        lines.push(`${indent}c = new T[${printExpression(s.len)}];`);
        break;
      case 'newfixed':
        lines.push(`${indent}c = new_fixed T[${s.count}];`);
        break;
      case 'set64':
        lines.push(`${indent}w = ${printExpression64(s.e)};`);
        break;
      case 'snew':
        lines.push(`${indent}${s.n} = new S(${printExpression(s.value)});`);
        break;
      case 'sset':
        lines.push(`${indent}${s.n}.f = ${printExpression(s.value)};`);
        break;
      case 'salias':
        lines.push(`${indent}u = ${printExpression(s.c)} ? s : t;`);
        break;
      case 'block':
      case 'loop':
        lines.push(`${indent}${s.k} ${s.label} {`,
                   ...printStatements(s.body, indent + '  '),
                   `${indent}}`);
        break;
    }
  }
  return lines;
}

// Inputs.
const kLengths = [0, 1, 2, 3, 4, 5, 8];
const kParams = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0x7fffffff,
                 0x7ffffffe, -0x80000000, 0x40000000];

function randomInputs() {
  let inputs = {
    lengths: [pick(kLengths), pick(kLengths)],
    p: pick(kParams),
    // Mostly a flag, otherwise an operand of base expressions.
    q: chance(2) ? random(2) : pick(kParams),
    same: chance(8),
    nulls: [chance(32), chance(16)],
  };
  return inputs;
}

let builder = new WasmModuleBuilder();
let array = builder.addArray(kWasmI32);
let struct = builder.addStruct([makeField(kWasmI32, true)]);
builder.addFunction('make', makeSig([kWasmI32], [wasmRefType(array)]))
  .addBody([kExprLocalGet, 0, kGCPrefix, kExprArrayNewDefault, array])
  .exportFunc();
builder.addFunction('get',
                    makeSig([wasmRefNullType(array), kWasmI32], [kWasmI32]))
  .addBody([kExprLocalGet, 0, kExprLocalGet, 1,
            kGCPrefix, kExprArrayGet, array])
  .exportFunc();
builder.addFunction('set',
                    makeSig([wasmRefNullType(array), kWasmI32, kWasmI32], []))
  .addBody([kExprLocalGet, 0, kExprLocalGet, 1, kExprLocalGet, 2,
            kGCPrefix, kExprArraySet, array])
  .exportFunc();
let sig = makeSig([wasmRefNullType(array), wasmRefNullType(array),
                   kWasmI32, kWasmI32], [kWasmI32]);
let functions = [];
for (let f = 0; f < kFunctionCount; f++) {
  let statements = chance(4) ? randomDecreasingFunction()
                 : chance(4) ? randomCanonicalFunction()
                 : chance(3) ? randomAllocationFunction()
                 : randomFunction();
  functions.push(statements);
  builder.addFunction(`f${f}`, sig)
    .addLocals(kWasmI32, 6)
    .addLocals(wasmRefNullType(array), 1)
    .addLocals(wasmRefNullType(struct), 3)
    .addLocals(kWasmI64, 1)
    .addBody([...emitStatements(statements, array, []),
              kExprLocalGet, kLocals.acc])
    .exportFunc();
}
let instance = builder.instantiate();
let exports = instance.exports;

// Runs function {f} on {inputs}, and returns its result or the message
// and the position of the trap, followed by the contents of the arrays.
function run(f, inputs) {
  let arrays = inputs.lengths.map((length, n) => {
    if (inputs.nulls[n]) return null;
    let a = exports.make(length);
    // Small values, which can be indices.
    for (let k = 0; k < length; k++) exports.set(a, k, (3 * k + n) % 7);
    return a;
  });
  if (inputs.same) arrays[1] = arrays[0];
  let result;
  try {
    result = exports[`f${f}`](arrays[0], arrays[1], inputs.p, inputs.q);
  } catch (e) {
    if (!(e instanceof WebAssembly.RuntimeError)) throw e;
    let frame = e.stack.split('\n').find(line => line.includes('wasm://'));
    result = `${e.message} at ${frame.trim()}`;
  }
  let contents = arrays.map((a, n) => {
    if (a === null) return null;
    let values = [];
    for (let k = 0; k < inputs.lengths[inputs.same ? 0 : n]; k++) {
      values.push(exports.get(a, k));
    }
    return values;
  });
  return JSON.stringify([result, contents]);
}

let inputs = [];
let expected = [];
for (let f = 0; f < kFunctionCount; f++) {
  assertTrue(%IsLiftoffFunction(exports[`f${f}`]));
  inputs.push([]);
  expected.push([]);
  for (let n = 0; n < kRunsPerFunction; n++) {
    inputs[f].push(randomInputs());
    expected[f].push(run(f, inputs[f][n]));
  }
}
if (kMode == 'stats') {
  let counts = {};
  for (let results of expected) {
    for (let result of results) {
      let [value] = JSON.parse(result);
      let kind = typeof value == 'number' ? 'result' : value.split(' at ')[0];
      counts[kind] = (counts[kind] ?? 0) + 1;
    }
  }
  print(JSON.stringify(counts));
}
for (let f = 0; f < kFunctionCount; f++) {
  if (kVerbose) {
    print(`f${f}:`);
    print(printStatements(functions[f]).join('\n'));
  }
  %WasmTierUpFunction(exports[`f${f}`]);
  assertTrue(%IsTurboFanFunction(exports[`f${f}`]));
  for (let n = 0; n < kRunsPerFunction; n++) {
    let actual = run(f, inputs[f][n]);
    if (actual != expected[f][n]) {
      print(`seed ${kSeed}, f${f}, inputs ${JSON.stringify(inputs[f][n])}:`);
      print(printStatements(functions[f]).join('\n'));
    }
    assertEquals(expected[f][n], actual, `seed ${kSeed} f${f} run ${n}`);
  }
}
