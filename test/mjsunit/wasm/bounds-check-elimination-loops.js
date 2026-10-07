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
// and constants. Each function first runs with Liftoff, which gives the
// expected results, and then with optimized code: the results, the
// position of traps and the contents of the arrays must be the same.
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
// the loops, which count down so that they are not induction variables
// themselves), acc (result).
const kLocals = {a: 0, b: 1, p: 2, q: 3, i: 4, x: 5, j: 6, count: 7,
                 jcount: 8, acc: 9};

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
// (l op r) is (r swapped r).
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

// {base + d}, written in various ways.
function index(base, d) {
  d |= 0;
  switch (random(8)) {
    case 0: return binop('add', constant(d), base);
    case 1: return binop('sub', base, constant(-d));
    case 2: {
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
function randomStatements(iv, labels, depth, guarded_only = false) {
  let statements = [];
  let count = 1 + random(3);
  let arr = chance(4) ? 'b' : 'a';
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
function randomFunction() {
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
  body.push(set('i', chance(4) ? binop('add', constant(step), local('i'))
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
const kBinops = {add: kExprI32Add, sub: kExprI32Sub, xor: kExprI32Xor};
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
  }
  throw new Error(`unknown expression ${e.k}`);
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
  add: '+', sub: '-', xor: '^', lt_s: '<s', lt_u: '<u', le_s: '<=s',
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
    q: random(2),
    same: chance(8),
    nulls: [chance(32), chance(16)],
  };
  return inputs;
}

let builder = new WasmModuleBuilder();
let array = builder.addArray(kWasmI32);
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
  let statements = randomFunction();
  functions.push(statements);
  builder.addFunction(`f${f}`, sig)
    .addLocals(kWasmI32, 6)
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
    for (let k = 0; k < length; k++) exports.set(a, k, 100 * (n + 1) + k);
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
