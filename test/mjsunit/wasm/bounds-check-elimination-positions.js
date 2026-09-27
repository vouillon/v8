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

let instance = builder.instantiate();
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
let cases = [];
for (let i of kIndices) {
  cases.push(['loads', () => exports.loads(a, i)]);
  cases.push(['loadsBelow', () => exports.loadsBelow(a, i)]);
  cases.push(['nested', () => exports.nested(aa, i)]);
  for (let j of kIndices) {
    cases.push(['interleaved', () => exports.interleaved(a, i, b, j)]);
  }
}

let expected = cases.map(([name, f]) => run(f));
for (let name of ['loads', 'loadsBelow', 'nested', 'interleaved']) {
  %WasmTierUpFunction(exports[name]);
}
for (let k = 0; k < cases.length; k++) {
  assertEquals(expected[k], run(cases[k][1]), cases[k][0]);
}
