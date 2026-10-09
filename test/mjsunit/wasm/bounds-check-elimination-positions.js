// Copyright 2026 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Flags: --allow-natives-syntax --liftoff --no-wasm-lazy-compilation
// Flags: --turboshaft-wasm-bounds-check-elimination
// Flags: --turboshaft-verify-wasm-bounds-check-elimination

// Bounds checks covered by fallback code must trap at the same position
// as without the optimization. Each function first runs with Liftoff,
// which gives the expected results, and then with optimized code.

d8.file.execute('test/mjsunit/wasm/wasm-module-builder.js');

let builder = new WasmModuleBuilder();
let array = builder.addArray(kWasmI32);
// Non-nullable elements, so that the bounds check of aa[i][1] is its
// only operation that can trap.
let outer = builder.addArray(wasmRefType(array));
// A JS function counting its calls.
let tick = builder.addImport('m', 'tick', kSig_v_v);
builder.addFunction('make', makeSig([kWasmI32], [wasmRefType(array)]))
  .addBody([
    kExprLocalGet, 0,
    kGCPrefix, kExprArrayNewDefault, array,
  ])
  .exportFunc();
builder.addFunction('makeOuter',
                    makeSig([kWasmI32, wasmRefType(array)],
                            [wasmRefType(outer)]))
  .addBody([
    kExprLocalGet, 1,
    kExprLocalGet, 0,
    kGCPrefix, kExprArrayNew, outer,
  ])
  .exportFunc();
builder.addFunction('setOuter',
                    makeSig([wasmRefNullType(outer), kWasmI32,
                             wasmRefType(array)], []))
  .addBody([
    kExprLocalGet, 0,
    kExprLocalGet, 1,
    kExprLocalGet, 2,
    kGCPrefix, kExprArraySet, outer,
  ])
  .exportFunc();

// a[i + offset], as an i32.
function load(offset) {
  return [
    kExprLocalGet, 0,
    kExprLocalGet, 1,
    ...wasmI32Const(offset),
    kExprI32Add,
    kGCPrefix, kExprArrayGet, array,
  ];
}

// a[i] + a[i+1] + a[i+2] + a[i+3]: the four bounds checks are covered
// by fallback code.
builder.addFunction('loads',
                    makeSig([wasmRefNullType(array), kWasmI32], [kWasmI32]))
  .addBody([
    ...load(0), ...load(1), kExprI32Add,
    ...load(2), kExprI32Add, ...load(3), kExprI32Add,
  ])
  .exportFunc();

// a[i+2] + a[i+1] + a[i] + a[i+3]: the guard checks both bounds.
builder.addFunction('loadsBelow',
                    makeSig([wasmRefNullType(array), kWasmI32], [kWasmI32]))
  .addBody([
    ...load(2), ...load(1), kExprI32Add,
    ...load(0), kExprI32Add, ...load(3), kExprI32Add,
  ])
  .exportFunc();

// if (i + 7 <u a.length) a[i+5] + a[i+3] + a[i+1] else -1: the
// condition shows the upper bound of the sequence, so the guard only
// checks the lower bound.
builder.addFunction('loadsDownward',
                    makeSig([wasmRefNullType(array), kWasmI32], [kWasmI32]))
  .addBody([
    kExprLocalGet, 1,
    ...wasmI32Const(7),
    kExprI32Add,
    kExprLocalGet, 0,
    kGCPrefix, kExprArrayLen,
    kExprI32LtU,
    kExprIf, kWasmI32,
      ...load(5), ...load(3), kExprI32Add, ...load(1), kExprI32Add,
    kExprElse,
      ...wasmI32Const(-1),
    kExprEnd,
  ])
  .exportFunc();

// aa[i+k], as a reference to an inner array.
function loadOuter(offset) {
  return [
    kExprLocalGet, 0,
    kExprLocalGet, 1,
    ...wasmI32Const(offset),
    kExprI32Add,
    kGCPrefix, kExprArrayGet, outer,
  ];
}

// aa[i][1] + len(aa[i+1]) + len(aa[i+2]): the bounds check of aa[i][1]
// comes between the bounds checks of aa covered by fallback code, and
// may trap first.
builder.addFunction('nested',
                    makeSig([wasmRefNullType(outer), kWasmI32], [kWasmI32]))
  .addBody([
    ...loadOuter(0),
    ...wasmI32Const(1),
    kGCPrefix, kExprArrayGet, array,
    ...loadOuter(1), kGCPrefix, kExprArrayLen, kExprI32Add,
    ...loadOuter(2), kGCPrefix, kExprArrayLen, kExprI32Add,
  ])
  .exportFunc();

// a[i] + b[j] + a[i+1] + b[j+1] + a[i+2] + b[j+2]: the fallback code for
// b starts within the fallback sequence for a, and each one contains
// checks of the other array.
function loadAt(local, offset) {
  return [
    kExprLocalGet, local,
    kExprLocalGet, local + 1,
    ...wasmI32Const(offset),
    kExprI32Add,
    kGCPrefix, kExprArrayGet, array,
  ];
}
builder.addFunction('interleaved',
                    makeSig([wasmRefNullType(array), kWasmI32,
                             wasmRefNullType(array), kWasmI32], [kWasmI32]))
  .addBody([
    ...loadAt(0, 0), ...loadAt(2, 0), kExprI32Add,
    ...loadAt(0, 1), kExprI32Add, ...loadAt(2, 1), kExprI32Add,
    ...loadAt(0, 2), kExprI32Add, ...loadAt(2, 2), kExprI32Add,
  ])
  .exportFunc();

builder.addFunction('get',
                    makeSig([wasmRefNullType(array), kWasmI32], [kWasmI32]))
  .addBody([
    kExprLocalGet, 0,
    kExprLocalGet, 1,
    kGCPrefix, kExprArrayGet, array,
  ])
  .exportFunc();

// a[i + offset] = value.
function store(offset, value) {
  return [
    kExprLocalGet, 0,
    kExprLocalGet, 1,
    ...wasmI32Const(offset),
    kExprI32Add,
    ...wasmI32Const(value),
    kGCPrefix, kExprArraySet, array,
  ];
}

// log[k] = value, for a log array whose checks always succeed.
function storeLog(k, value) {
  return [
    kExprLocalGet, 2,
    ...wasmI32Const(k),
    ...wasmI32Const(value),
    kGCPrefix, kExprArraySet, array,
  ];
}

// a[i] = 1; a[i+1] = 2; a[i+2] = 3; a[i+3] = 4: the fallback code repeats
// the stores, and must leave the same elements written when it traps.
builder.addFunction('stores',
                    makeSig([wasmRefNullType(array), kWasmI32,
                             wasmRefNullType(array)], []))
  .addBody([...store(0, 1), ...store(1, 2), ...store(2, 3), ...store(3, 4)])
  .exportFunc();

// a[i+2] = 3; a[i+1] = 2; a[i] = 1; a[i+3] = 4: the guard checks both
// bounds.
builder.addFunction('storesBelow',
                    makeSig([wasmRefNullType(array), kWasmI32,
                             wasmRefNullType(array)], []))
  .addBody([...store(2, 3), ...store(1, 2), ...store(0, 1), ...store(3, 4)])
  .exportFunc();

// Each store to a is followed by a store to another array, so that the
// stores done before a trap are observable on two objects.
builder.addFunction('storesLogged',
                    makeSig([wasmRefNullType(array), kWasmI32,
                             wasmRefNullType(array)], []))
  .addBody([
    ...store(0, 1), ...storeLog(0, 101),
    ...store(1, 2), ...storeLog(1, 102),
    ...store(2, 3), ...storeLog(2, 103),
    ...store(3, 4), ...storeLog(3, 104),
  ])
  .exportFunc();

// a[wrap(x + k)], for a 64-bit x, as an i32: the indices of the
// accesses are truncated additions.
function load64(offset) {
  return [
    kExprLocalGet, 0,
    kExprLocalGet, 1,
    ...wasmI64Const(offset),
    kExprI64Add,
    kExprI32ConvertI64,
    kGCPrefix, kExprArrayGet, array,
  ];
}

// a[wrap(x)] + a[wrap(x+1)] + a[wrap(x+2)] + a[wrap(x+3)].
builder.addFunction('loads64',
                    makeSig([wasmRefNullType(array), kWasmI64], [kWasmI32]))
  .addBody([
    ...load64(0), ...load64(1), kExprI32Add,
    ...load64(2), kExprI32Add, ...load64(3), kExprI32Add,
  ])
  .exportFunc();

// a[i] + tick() + a[i+1] + a[i+2] + a[i+3]: the fallback code repeats
// the call, which must happen as many times as without the optimization
// before a trap.
builder.addFunction('loadsAroundCall',
                    makeSig([wasmRefNullType(array), kWasmI32], [kWasmI32]))
  .addBody([
    ...load(0), kExprCallFunction, tick, ...load(1), kExprI32Add,
    ...load(2), kExprI32Add, ...load(3), kExprI32Add,
  ])
  .exportFunc();

// The same in a try, so that the accesses follow the exception edge of
// the call.
builder.addFunction('loadsInTry',
                    makeSig([wasmRefNullType(array), kWasmI32], [kWasmI32]))
  .addBody([
    kExprTry, kWasmI32,
      ...load(0), kExprCallFunction, tick, ...load(1), kExprI32Add,
      ...load(2), kExprI32Add, ...load(3), kExprI32Add,
    kExprCatchAll,
      ...wasmI32Const(-1),
    kExprEnd,
  ])
  .exportFunc();

let ticks = 0;
let instance = builder.instantiate({m: {tick: () => { ticks++; }}});
let exports = instance.exports;

// Runs {f}, and returns its result, or the message and the position of
// the top frame of the exception it throws.
function run(f) {
  try {
    return f();
  } catch (e) {
    let frame = e.stack.split('\n').find(line => line.includes('wasm://'));
    return `${e.message} at ${frame.trim()}`;
  }
}

const kLength = 6;
let a = exports.make(kLength);
let aa = exports.makeOuter(kLength, a);
// Inner arrays of decreasing lengths, so that aa[i][1] traps for the
// largest values of i, for which the guard of the fallback code on aa
// fails.
for (let k = 0; k < kLength; k++) {
  exports.setOuter(aa, k, exports.make(kLength - 1 - k));
}
const kIndices = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 0x7fffffff,
                  -0x80000000];
let b = exports.make(kLength - 2);

function contents(x, length) {
  return Array.from({length}, (_, k) => exports.get(x, k)).join(',');
}

// Runs a store function on fresh arrays, and returns its result together
// with the contents of the arrays afterwards.
function runStores(f, i) {
  let x = exports.make(kLength);
  let log = exports.make(4);
  let result = run(() => f(x, i, log));
  return `${result}; a = ${contents(x, kLength)}; log = ${contents(log, 4)}`;
}

let cases = [];
for (let i of kIndices) {
  cases.push(['loads', () => exports.loads(a, i)]);
  cases.push(['loadsBelow', () => exports.loadsBelow(a, i)]);
  cases.push(['loadsDownward', () => exports.loadsDownward(a, i)]);
  cases.push(['nested', () => exports.nested(aa, i)]);
  for (let name of ['stores', 'storesBelow', 'storesLogged']) {
    cases.push([name, () => runStores(exports[name], i)]);
  }
  for (let j of kIndices) {
    cases.push(['interleaved', () => exports.interleaved(a, i, b, j)]);
  }
  for (let name of ['loadsAroundCall', 'loadsInTry']) {
    cases.push([name, () => {
      ticks = 0;
      return `${run(() => exports[name](a, i))}; ticks = ${ticks}`;
    }]);
  }
}
const kIndices64 = [-4n, -1n, 0n, 2n, 3n, 6n, (1n << 32n) - 2n, 1n << 32n,
                    (1n << 32n) + 1n, -(1n << 32n), (1n << 63n) - 1n];
for (let x of kIndices64) {
  cases.push(['loads64', () => exports.loads64(a, x)]);
}
cases.push(['loads', () => exports.loads(null, 0)]);
cases.push(['loads64', () => exports.loads64(null, 0n)]);

let expected = cases.map(([name, f]) => run(f));
for (let name of ['loads', 'loadsBelow', 'loadsDownward', 'nested',
                  'interleaved', 'stores', 'storesBelow', 'storesLogged',
                  'loads64', 'loadsAroundCall', 'loadsInTry']) {
  %WasmTierUpFunction(exports[name]);
}
for (let k = 0; k < cases.length; k++) {
  assertEquals(expected[k], run(cases[k][1]), cases[k][0]);
}
