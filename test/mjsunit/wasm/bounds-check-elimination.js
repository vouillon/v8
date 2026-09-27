// Copyright 2026 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Flags: --no-liftoff --no-wasm-lazy-compilation
// Flags: --turboshaft-wasm-bounds-check-elimination
// Flags: --turboshaft-verify-wasm-bounds-check-elimination

d8.file.execute('test/mjsunit/wasm/wasm-module-builder.js');

const kOutOfBounds = 'array element access out of bounds';
const kNull = 'dereferencing a null pointer';

function addArrayHelpers(builder, array) {
  builder.addFunction('make', makeSig([kWasmI32], [wasmRefType(array)]))
    .addBody([
      kExprLocalGet, 0,
      kGCPrefix, kExprArrayNewDefault, array,
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
}

function contents(instance, a, length) {
  let result = [];
  for (let k = 0; k < length; k++) result.push(instance.exports.get(a, k));
  return result;
}

// a[index] = value, where {index} is the parameter {param} plus {offset}.
function store(array, param, offset, value) {
  return [
    kExprLocalGet, 0,
    kExprLocalGet, param,
    ...wasmI32Const(offset),
    kExprI32Add,
    ...wasmI32Const(value),
    kGCPrefix, kExprArraySet, array,
  ];
}

(function TestNoFactsFromNonDominatingBranch() {
  print(arguments.callee.name);
  let builder = new WasmModuleBuilder();
  let array = builder.addArray(kWasmI32);
  addArrayHelpers(builder, array);
  let sig = makeSig([wasmRefNullType(array), kWasmI32, kWasmI32], []);
  // if (c) a[i]; a[i] = 42;
  builder.addFunction('afterIf', sig)
    .addBody([
      kExprLocalGet, 2,
      kExprIf, kWasmVoid,
        kExprLocalGet, 0,
        kExprLocalGet, 1,
        kGCPrefix, kExprArrayGet, array,
        kExprDrop,
      kExprEnd,
      kExprLocalGet, 0,
      kExprLocalGet, 1,
      ...wasmI32Const(42),
      kGCPrefix, kExprArraySet, array,
    ])
    .exportFunc();
  // if (i < a.length) a[i] = 1; else a[i] = 2;
  builder.addFunction('inElse', sig)
    .addBody([
      kExprLocalGet, 1,
      kExprLocalGet, 0,
      kGCPrefix, kExprArrayLen,
      kExprI32LtU,
      kExprIf, kWasmVoid,
        ...store(array, 1, 0, 1),
      kExprElse,
        ...store(array, 1, 0, 2),
      kExprEnd,
    ])
    .exportFunc();
  let instance = builder.instantiate();
  let a = instance.exports.make(4);
  instance.exports.afterIf(a, 3, 1);
  instance.exports.afterIf(a, 2, 0);
  assertTraps(kTrapArrayOutOfBounds, () => instance.exports.afterIf(a, 4, 1));
  assertTraps(kTrapArrayOutOfBounds, () => instance.exports.afterIf(a, 4, 0));
  assertTraps(kTrapArrayOutOfBounds, () => instance.exports.afterIf(a, -1, 0));
  instance.exports.inElse(a, 1);
  assertTraps(kTrapArrayOutOfBounds, () => instance.exports.inElse(a, 4));
  assertTraps(kTrapArrayOutOfBounds, () => instance.exports.inElse(a, -1));
  assertEquals([0, 1, 42, 42], contents(instance, a, 4));
})();

(function TestFallbackGuards() {
  print(arguments.callee.name);
  let builder = new WasmModuleBuilder();
  let array = builder.addArray(kWasmI32);
  addArrayHelpers(builder, array);
  let sig = makeSig([wasmRefNullType(array), kWasmI32], []);
  let sequences = {
    // Each sequence of accesses is a list of offsets. On 64-bit targets,
    // pairs of accesses with no known bound get fallback code.
    below: [1, 0],
    above: [0, 1],
    middle: [0, 2, 1],
    chainBelow: [7, 5, 3, 1],
    chainAbove: [1, 3, 5, 7],
    consecutive: [0, 1, 2, 3, 4],
    zigzag: [2, 3, 1, 4, 0],
    wrapAround: [-2, -1, 0, 1],
    large: [0x7ffffffd, 0x7ffffffe, 0x7fffffff, -0x80000000],
    // With a single-comparison guard checking both bounds (on 64-bit
    // targets), three accesses are enough for a fallback sequence.
    three: [0, 1, 2],
    threeBelow: [2, 1, 0],
    threeWrapAround: [-1, 0, 1],
    threeLarge: [0x7ffffffe, 0x7fffffff, -0x80000000],
  };
  for (let [name, offsets] of Object.entries(sequences)) {
    builder.addFunction(name, sig)
      .addBody(offsets.flatMap((offset, k) => store(array, 1, offset, k + 1)))
      .exportFunc();
  }
  let instance = builder.instantiate();
  const length = 8;
  for (let [name, offsets] of Object.entries(sequences)) {
    for (let i of [-6, -5, -2, -1, 0, 1, 3, 4, 5, 6, 7, 8, 0x7fffffff,
                   -0x80000000, 0x80000001]) {
      let a = instance.exports.make(length);
      // Stores happen in order until the first out-of-bounds access.
      let expected = new Array(length).fill(0);
      let traps = false;
      for (let [k, offset] of offsets.entries()) {
        let index = (i + offset) >>> 0;
        if (index >= length) {
          traps = true;
          break;
        }
        expected[index] = k + 1;
      }
      let run = () => instance.exports[name](a, i);
      if (traps) {
        assertTraps(kTrapArrayOutOfBounds, run);
      } else {
        run();
      }
      assertEquals(expected, contents(instance, a, length), `${name}(${i})`);
    }
  }
})();

(function TestDuplicatedComputations() {
  print(arguments.callee.name);
  let builder = new WasmModuleBuilder();
  let array = builder.addArray(kWasmI32);
  addArrayHelpers(builder, array);
  let sig = makeSig([wasmRefNullType(array), kWasmI32, kWasmI32], []);
  // 2 * i, computed anew each time.
  let twice = [kExprLocalGet, 1, ...wasmI32Const(2), kExprI32Mul];
  // if (c) a[2 * i]; a[2 * i] = 1; a[2 * i + 1] = 2; ... a[2 * i + 3] = 4;
  // The four stores after the if use a fallback sequence, whose guard
  // must use the value of 2 * i computed after the if, and not the one
  // computed in the if, which has the same structure.
  let stores = [];
  for (let k = 0; k < 4; k++) {
    stores.push(kExprLocalGet, 0, ...twice);
    if (k > 0) stores.push(...wasmI32Const(k), kExprI32Add);
    stores.push(...wasmI32Const(k + 1), kGCPrefix, kExprArraySet, array);
  }
  builder.addFunction('guardAfterIf', sig)
    .addBody([
      kExprLocalGet, 2,
      kExprIf, kWasmVoid,
        kExprLocalGet, 0,
        ...twice,
        kGCPrefix, kExprArrayGet, array,
        kExprDrop,
      kExprEnd,
      ...stores,
    ])
    .exportFunc();
  let instance = builder.instantiate();
  const length = 8;
  for (let c of [0, 1]) {
    for (let i of [-1, 0, 1, 2, 3, 4, 0x40000000, 0x7fffffff]) {
      let a = instance.exports.make(length);
      let expected = new Array(length).fill(0);
      let traps = c && (2 * i) >>> 0 >= length;
      for (let k = 0; k < 4 && !traps; k++) {
        let index = (2 * i + k) >>> 0;
        if (index >= length) {
          traps = true;
        } else {
          expected[index] = k + 1;
        }
      }
      let run = () => instance.exports.guardAfterIf(a, i, c);
      if (traps) {
        assertTraps(kTrapArrayOutOfBounds, run);
      } else {
        run();
      }
      assertEquals(expected, contents(instance, a, length), `${c} ${i}`);
    }
  }
})();

(function TestNonNegativeIndexAcrossArrays() {
  print(arguments.callee.name);
  let builder = new WasmModuleBuilder();
  let array = builder.addArray(kWasmI32);
  addArrayHelpers(builder, array);
  let sig = makeSig([wasmRefNullType(array), wasmRefNullType(array),
                     kWasmI32], []);
  function storeB(offset, value) {
    return [
      kExprLocalGet, 1,
      kExprLocalGet, 2,
      ...wasmI32Const(offset),
      kExprI32Add,
      ...wasmI32Const(value),
      kGCPrefix, kExprArraySet, array,
    ];
  }
  function storeA(offset, value) {
    return [
      kExprLocalGet, 0,
      kExprLocalGet, 2,
      ...wasmI32Const(offset),
      kExprI32Add,
      ...wasmI32Const(value),
      kGCPrefix, kExprArraySet, array,
    ];
  }
  // a[i] = 1; b[i+1] = 2; b[i] = 3;
  builder.addFunction('redundant', sig)
    .addBody([...storeA(0, 1), ...storeB(1, 2), ...storeB(0, 3)])
    .exportFunc();
  // a[i] = 1; b[i] = 2; b[i+1] = 3;
  builder.addFunction('oneSidedGuard', sig)
    .addBody([...storeA(0, 1), ...storeB(0, 2), ...storeB(1, 3)])
    .exportFunc();
  // a[i+2] = 1; b[i+1] = 2; b[i] = 3; (i+2 >= 0 says nothing about i)
  builder.addFunction('notCovered', sig)
    .addBody([...storeA(2, 1), ...storeB(1, 2), ...storeB(0, 3)])
    .exportFunc();
  // if (i >= 0) { b[i+1] = 2; b[i] = 3; }
  builder.addFunction('signedCheck', sig)
    .addBody([
      kExprLocalGet, 2,
      ...wasmI32Const(0),
      kExprI32GeS,
      kExprIf, kWasmVoid,
        ...storeB(1, 2), ...storeB(0, 3),
      kExprEnd,
    ])
    .exportFunc();
  let instance = builder.instantiate();
  let model = {
    redundant: [['a', 0, 1], ['b', 1, 2], ['b', 0, 3]],
    oneSidedGuard: [['a', 0, 1], ['b', 0, 2], ['b', 1, 3]],
    notCovered: [['a', 2, 1], ['b', 1, 2], ['b', 0, 3]],
    signedCheck: [['b', 1, 2], ['b', 0, 3]],
  };
  const lengthA = 10;
  const lengthB = 4;
  for (let [name, accesses] of Object.entries(model)) {
    for (let i of [-3, -2, -1, 0, 1, 2, 3, 4, 9, 10, 0x7fffffff]) {
      let arrays = {
        a: instance.exports.make(lengthA),
        b: instance.exports.make(lengthB),
      };
      let expected = {
        a: new Array(lengthA).fill(0),
        b: new Array(lengthB).fill(0),
      };
      let traps = false;
      if (name != 'signedCheck' || i >= 0) {
        for (let [array, offset, value] of accesses) {
          let index = (i + offset) >>> 0;
          if (index >= expected[array].length) {
            traps = true;
            break;
          }
          expected[array][index] = value;
        }
      }
      let run = () => instance.exports[name](arrays.a, arrays.b, i);
      if (traps) {
        assertTraps(kTrapArrayOutOfBounds, run);
      } else {
        run();
      }
      assertEquals(expected.a, contents(instance, arrays.a, lengthA));
      assertEquals(expected.b, contents(instance, arrays.b, lengthB));
    }
  }
})();

(function TestNonNegativeIndexEdgeCases() {
  print(arguments.callee.name);
  let builder = new WasmModuleBuilder();
  let array = builder.addArray(kWasmI32);
  addArrayHelpers(builder, array);
  let sig = makeSig([wasmRefNullType(array), wasmRefNullType(array),
                     kWasmI32], []);
  function access(array_param, offset, value) {
    return [
      kExprLocalGet, array_param,
      kExprLocalGet, 2,
      ...wasmI32Const(offset),
      kExprI32Add,
      ...wasmI32Const(value),
      kGCPrefix, kExprArraySet, array,
    ];
  }
  function guarded(condition, accesses) {
    return [kExprLocalGet, 2, ...condition, kExprIf, kWasmVoid,
            ...accesses.flat(), kExprEnd];
  }
  let cases = {
    // These conditions do not show that i is non-negative.
    ltU: {
      condition: i => (i >>> 0) < 0xffffffff,
      code: guarded([...wasmI32Const(-1), kExprI32LtU],
                    [access(1, 2, 1), access(1, 1, 2)]),
    },
    gtS: {
      condition: i => i > -2,
      code: guarded([...wasmI32Const(-2), kExprI32GtS],
                    [access(1, 1, 1), access(1, 0, 2)]),
    },
    geS: {
      condition: i => i >= -1,
      code: guarded([...wasmI32Const(-1), kExprI32GeS],
                    [access(1, 1, 1), access(1, 0, 2)]),
    },
    // i + 0x55555555 is non-negative, but this says nothing about the
    // range from i - 0x55555556 to i, which is on the other side of the
    // circle of 32-bit offsets.
    farNonNegative: {
      condition: i => ((i + 0x55555555) >>> 0) < 0x80000000,
      code: guarded([...wasmI32Const(0x55555555), kExprI32Add,
                     ...wasmI32Const(0x80000000 | 0), kExprI32LtU],
                    [access(1, 0, 1), access(1, -0x55555556, 2)]),
    },
    // Offsets that are too far apart for i >= 0 to help.
    farOffsets: {
      condition: i => true,
      code: [access(0, 0, 1), access(1, 0x7fffffff, 2), access(1, -2, 3)]
                .flat(),
    },
  };
  let accesses = {
    ltU: [[1, 2, 1], [1, 1, 2]],
    gtS: [[1, 1, 1], [1, 0, 2]],
    geS: [[1, 1, 1], [1, 0, 2]],
    farNonNegative: [[1, 0, 1], [1, -0x55555556, 2]],
    farOffsets: [[0, 0, 1], [1, 0x7fffffff, 2], [1, -2, 3]],
  };
  for (let [name, {code}] of Object.entries(cases)) {
    builder.addFunction(name, sig).addBody(code).exportFunc();
  }
  let instance = builder.instantiate();
  const length = 4;
  for (let [name, {condition}] of Object.entries(cases)) {
    for (let i of [-3, -2, -1, 0, 1, 2, 3, 4, 0x7fffffff, -0x80000000]) {
      let arrays = [instance.exports.make(length),
                    instance.exports.make(length)];
      let expected = [new Array(length).fill(0), new Array(length).fill(0)];
      let traps = false;
      if (condition(i)) {
        for (let [array, offset, value] of accesses[name]) {
          let index = (i + offset) >>> 0;
          if (index >= length) {
            traps = true;
            break;
          }
          expected[array][index] = value;
        }
      }
      let run = () => instance.exports[name](...arrays, i);
      if (traps) {
        assertTraps(kTrapArrayOutOfBounds, run);
      } else {
        run();
      }
      assertEquals(expected[0], contents(instance, arrays[0], length));
      assertEquals(expected[1], contents(instance, arrays[1], length));
    }
  }
})();

(function TestOCamlBoundsChecks() {
  print(arguments.callee.name);
  // wasm_of_ocaml stores a header in the first element of the arrays
  // it uses for OCaml arrays, and checks an access a.(i) with
  // {i < a.length - 1} before accessing a[i+1]. This check only shows
  // that a[i+1] is within bounds if the length is known to be at least 1.
  let builder = new WasmModuleBuilder();
  let array = builder.addArray(kWasmI32);
  addArrayHelpers(builder, array);
  let sig = makeSig([wasmRefNullType(array), kWasmI32, kWasmI32], []);
  // if (i < a.length - c) a[i + offset] = value;
  function guarded(index, reduction, offset, value) {
    return [
      kExprLocalGet, index,
      kExprLocalGet, 0,
      kGCPrefix, kExprArrayLen,
      ...wasmI32Const(reduction), kExprI32Sub,
      kExprI32LtU,
      kExprIf, kWasmVoid,
        ...store(array, index, offset, value),
      kExprEnd,
    ];
  }
  // if (a.length - c <= i) skip; a[i + offset] = value;
  function skipUnless(index, reduction, offset, value) {
    return [
      kExprBlock, kWasmVoid,
        kExprLocalGet, 0,
        kGCPrefix, kExprArrayLen,
        ...wasmI32Const(reduction), kExprI32Sub,
        kExprLocalGet, index,
        kExprI32LeU,
        kExprBrIf, 0,
        ...store(array, index, offset, value),
      kExprEnd,
    ];
  }
  let cases = {
    // Nothing is known about the length: a.length - 1 may wrap around.
    unknownLength: [[guarded, 1, 1, 0, 1]],
    unknownLengthSkip: [[skipUnless, 1, 1, 0, 1]],
    // a[0] shows that the length is at least 1.
    knownLength: [['store', 0, 1], [guarded, 1, 1, 0, 2]],
    knownLengthSkip: [['store', 0, 1], [skipUnless, 1, 1, 0, 2]],
    // The first OCaml access shows that the length is at least 1 for
    // the second one.
    sequence: [[skipUnless, 1, 1, 0, 1], [skipUnless, 2, 1, 0, 2]],
    // a[0] only shows that the length is at least 1, not 2.
    reductionTooLarge: [['store', 0, 1], [guarded, 1, 2, 1, 2]],
    reductionOk: [['store', 1, 1], [guarded, 1, 2, 0, 2], [guarded, 1, 2, 2, 3]],
  };
  // Parameters: the array, then two indices i and j.
  for (let [name, steps] of Object.entries(cases)) {
    let code = [];
    for (let [kind, ...rest] of steps) {
      if (kind === 'store') {
        let [index, value] = rest;
        code.push(kExprLocalGet, 0, ...wasmI32Const(index),
                  ...wasmI32Const(value), kGCPrefix, kExprArraySet, array);
      } else {
        let [index, reduction, offset, value] = rest;
        code.push(...kind(index, reduction, offset, value));
      }
    }
    builder.addFunction(name, sig).addBody(code).exportFunc();
  }
  let instance = builder.instantiate();
  for (let [name, steps] of Object.entries(cases)) {
    for (let length of [0, 1, 2, 3, 5]) {
      for (let i of [-2, -1, 0, 1, 2, 3, 4, 5]) {
        for (let j of [-1, 0, 1, 3]) {
          let a = instance.exports.make(length);
          let expected = new Array(length).fill(0);
          let traps = false;
          let params = [null, i, j];
          for (let [kind, ...rest] of steps) {
            let index, value;
            if (kind === 'store') {
              [index, value] = rest;
            } else {
              let [param, reduction, offset] = rest;
              value = rest[3];
              // The condition, with the same modular arithmetic as Wasm.
              let bound = (length - reduction) >>> 0;
              if (!((params[param] >>> 0) < bound)) continue;
              index = params[param] + offset;
            }
            index >>>= 0;
            if (index >= length) {
              traps = true;
              break;
            }
            expected[index] = value;
          }
          let run = () => instance.exports[name](a, i, j);
          if (traps) {
            assertTraps(kTrapArrayOutOfBounds, run);
          } else {
            run();
          }
          assertEquals(expected, contents(instance, a, length),
                       `${name}(length=${length}, i=${i}, j=${j})`);
        }
      }
    }
  }
})();

// Randomly generated functions, compared against a JS model. For stress
// testing, a different seed can be passed with: d8 ... -- <seed>
const kRandomSeed = (typeof arguments != 'undefined' && arguments.length > 0)
    ? Number(arguments[0]) : 0x2545f491;

(function TestLoops() {
  print(arguments.callee.name);
  // Loops over k, from {init} by {step}, with various exit conditions,
  // accessing a[k + d] for each offset d of {accesses}. The loops stop
  // after {kMaxIterations} iterations in any case.
  const kMaxIterations = 20;
  let builder = new WasmModuleBuilder();
  let array = builder.addArray(kWasmI32);
  addArrayHelpers(builder, array);
  const kA = 0, kK = 1, kCount = 2, kAcc = 3, kHi = 4;
  let length = [kExprLocalGet, kA, kGCPrefix, kExprArrayLen];
  let k = (d) => d == 0 ? [kExprLocalGet, kK]
                        : [kExprLocalGet, kK, ...wasmI32Const(d), kExprI32Add];
  // Exits: {test} is true to leave the loop at the top of an iteration;
  // with {ne}, the loop is left after the accesses when k == hi, and it
  // is only entered if {guard} (hi >= guard) holds.
  let exits = {
    ltSLength0: {test: [...length, ...k(0), kExprI32LeS],
                 model: (k, n) => !(k < n)},
    ltSLength2: {test: [...length, ...wasmI32Const(2), kExprI32Sub, ...k(0),
                        kExprI32LeS],
                 model: (k, n) => !(k < ((n - 2) | 0))},
    ltSmall: {test: [...wasmI32Const(6), ...k(0), kExprI32LeS],
              model: (k, n) => !(k < 6)},
    ltHuge: {test: [...wasmI32Const(0x7ffffffe), ...k(0), kExprI32LeS],
             model: (k, n) => !(k < 0x7ffffffe)},
    ltULength: {test: [...length, ...k(0), kExprI32LeU],
                model: (k, n) => !((k >>> 0) < n)},
    ltUHuge: {test: [...wasmI32Const(0x7fffffff), ...k(0), kExprI32LeU],
              model: (k, n) => !((k >>> 0) < 0x7fffffff)},
    ltUMedium: {test: [...wasmI32Const(0x7ffffffe), ...k(0), kExprI32LeU],
                model: (k, n) => !((k >>> 0) < 0x7ffffffe)},
    // k - 0x20 <s 0x7fffffe0 holds up to k = 0x7fffffff.
    ltSOffset: {test: [...wasmI32Const(0x7fffffe0), ...k(-0x20),
                       kExprI32LeS],
                model: (k, n) => !(((k - 0x20) | 0) < 0x7fffffe0)},
    // a.length - 1 wraps around for an empty array.
    ltULength1: {test: [...length, ...wasmI32Const(1), kExprI32Sub, ...k(0),
                        kExprI32LeU],
                 model: (k, n) => !((k >>> 0) < ((n - 1) >>> 0))},
    leSLength: {test: [...k(0), ...length, kExprI32GtS],
                model: (k, n) => !(k <= n)},
    leULength: {test: [...k(0), ...length, kExprI32GtU],
                model: (k, n) => !((k >>> 0) <= n)},
    neLength1: {ne: 1, guard: 0}, neLength2: {ne: 2, guard: 0},
    neLength1Guard1: {ne: 1, guard: 1},
    neLength1NoGuard: {ne: 1},
  };
  let cases = [];
  for (let init of [-1, 0, 1, 2]) {
    for (let step of [-1, 1, 2, 3]) {
      for (let [exitName, exit] of Object.entries(exits)) {
        for (let accesses of [[0], [0, 1], [2, 1, 0], [-1, 0], [1]]) {
          cases.push({init, step, exitName, exit, accesses,
                      store: cases.length % 2 == 0});
        }
      }
    }
  }
  // Starting close to 2^31, so that k can wrap around within the maximal
  // number of iterations when the bound does not prevent it.
  // Without accesses in the loop, which would trap first, so that the
  // access after the loop uses what is known about k once it wrapped.
  // Loop unrolling makes the loop variable take one value every few
  // steps, so a few starting values are tried for it to wrap around.
  for (let step of [1, 2, 3]) {
    for (let exitName of ['ltHuge', 'ltSmall', 'ltUHuge', 'ltUMedium',
                          'ltULength1', 'ltSOffset', 'neLength1NoGuard']) {
      for (let init = 0x7ffffff0; init < 0x7ffffffc; init++) {
        cases.push({init, step, exitName, exit: exits[exitName],
                    accesses: init == 0x7ffffff0 ? [0] : [], store: false});
      }
    }
  }
  // Large steps: they cannot overflow for an index at most an array
  // length (less than 2^30) when they are at most 2^30.
  for (let step of [0x40000000, 0x40000001, 0x7fffffff]) {
    for (let exitName of ['leSLength', 'leULength', 'ltSLength0',
                          'ltULength']) {
      for (let init of [0, 1]) {
        cases.push({init, step, exitName, exit: exits[exitName],
                    accesses: [0], store: false});
      }
    }
  }
  let sig = makeSig([wasmRefNullType(array)], [kWasmI32]);
  // if (k <s a.length) acc += a[k] + 1, which uses what is known about k
  // at the top of the loop, or after it.
  let useK = [...k(0), ...length, kExprI32LtS,
              kExprIf, kWasmVoid,
                kExprLocalGet, kAcc, ...wasmI32Const(1), kExprI32Add,
                kExprLocalGet, kA, ...k(0), kGCPrefix, kExprArrayGet, array,
                kExprI32Add, kExprLocalSet, kAcc,
              kExprEnd];
  for (let [n, c] of cases.entries()) {
    let body = [];
    for (let [m, d] of c.accesses.entries()) {
      if (c.store) {
        body.push(kExprLocalGet, kA, ...k(d), ...wasmI32Const(m + 1),
                  kGCPrefix, kExprArraySet, array);
      } else {
        body.push(kExprLocalGet, kAcc, ...wasmI32Const(31), kExprI32Mul,
                  kExprLocalGet, kA, ...k(d), kGCPrefix, kExprArrayGet, array,
                  kExprI32Add, kExprLocalSet, kAcc);
      }
    }
    let code = [];
    if (c.exit.ne !== undefined) {
      // hi = a.length - ne; if (hi >= guard) { loop }
      code.push(...length, ...wasmI32Const(c.exit.ne), kExprI32Sub,
                kExprLocalSet, kHi);
      if (c.exit.guard !== undefined) {
        code.push(kExprLocalGet, kHi, ...wasmI32Const(c.exit.guard),
                  kExprI32LtS, kExprIf, kWasmVoid,
                    kExprLocalGet, kAcc, kExprReturn,
                  kExprEnd);
      }
    }
    code.push(...wasmI32Const(c.init), kExprLocalSet, kK,
              kExprBlock, kWasmVoid,
                kExprLoop, kWasmVoid,
                  ...useK,
                  // if (count++ == kMaxIterations) break;
                  kExprLocalGet, kCount, ...wasmI32Const(kMaxIterations),
                  kExprI32Eq, kExprBrIf, 1,
                  kExprLocalGet, kCount, ...wasmI32Const(1), kExprI32Add,
                  kExprLocalSet, kCount,
                  ...(c.exit.test ? [...c.exit.test, kExprBrIf, 1] : []),
                  ...body,
                  ...(c.exit.ne !== undefined
                      ? [...k(0), kExprLocalGet, kHi, kExprI32Eq,
                         kExprBrIf, 1]
                      : []),
                  ...k(c.step), kExprLocalSet, kK,
                  kExprBr, 0,
                kExprEnd,
              kExprEnd,
              ...useK,
              kExprLocalGet, kAcc);
    builder.addFunction(`loop${n}`, sig)
      .addLocals(kWasmI32, 4)
      .addBody(code)
      .exportFunc();
  }
  let instance = builder.instantiate();
  // The model.
  function run(c, a) {
    let n = a.length, acc = 0, count = 0, hi;
    if (c.exit.ne !== undefined) {
      hi = (n - c.exit.ne) | 0;
      if (c.exit.guard !== undefined && hi < c.exit.guard) return acc;
    }
    let k = c.init;
    let useK = () => {
      if (k < n) {
        if ((k >>> 0) >= n) throw new Error(kOutOfBounds);
        acc = (acc + 1 + a[k]) | 0;
      }
    };
    while (true) {
      useK();
      if (count == kMaxIterations) break;
      count++;
      if (c.exit.model && c.exit.model(k, n)) break;
      for (let [m, d] of c.accesses.entries()) {
        let index = (k + d) | 0;
        if ((index >>> 0) >= n) throw new Error(kOutOfBounds);
        if (c.store) {
          a[index] = m + 1;
        } else {
          acc = (Math.imul(acc, 31) + a[index]) | 0;
        }
      }
      if (c.exit.ne !== undefined && k == hi) break;
      k = (k + c.step) | 0;
    }
    useK();
    return acc;
  }
  for (let [n, c] of cases.entries()) {
    for (let len of [0, 1, 2, 3, 5, 8]) {
      let a = instance.exports.make(len);
      let model = new Array(len).fill(0);
      for (let m = 0; m < len; m++) {
        instance.exports.set?.(a, m, m);
      }
      let expected;
      try {
        expected = run(c, model);
      } catch (e) {
        expected = e;
      }
      let name = `${c.exitName} init ${c.init} step ${c.step} ` +
                 `accesses ${c.accesses} length ${len}`;
      if (expected instanceof Error) {
        assertTraps(kTrapArrayOutOfBounds,
                    () => instance.exports[`loop${n}`](a), name);
      } else {
        assertEquals(expected, instance.exports[`loop${n}`](a), name);
      }
      assertEquals(model, contents(instance, a, len), name);
    }
  }
})();

(function TestLoopsWithPhiBound() {
  print(arguments.callee.name);
  // for (i = 0; i <= hi; i++) body, where hi is the length of an array
  // minus 1, or {other} - 1 when the array has another type (as for the
  // length of an OCaml float array, where {other} is 0 for an empty
  // array), and the loop is only entered if hi >= 0.
  let builder = new WasmModuleBuilder();
  let floats = builder.addArray(kWasmF64);
  let ints = builder.addArray(kWasmI32);
  builder.addFunction('makeFloats', makeSig([kWasmI32], [kWasmEqRef]))
    .addBody([
      ...wasmF64Const(1.5), kExprLocalGet, 0,
      kGCPrefix, kExprArrayNew, floats,
    ])
    .exportFunc();
  builder.addFunction('makeInts', makeSig([kWasmI32], [kWasmEqRef]))
    .addBody([
      ...wasmI32Const(7), kExprLocalGet, 0,
      kGCPrefix, kExprArrayNew, ints,
    ])
    .exportFunc();
  const kA = 0, kHi = 1, kI = 2, kAcc = 3;
  let asFloats = [kExprLocalGet, kA, kGCPrefix, kExprRefCast, floats];
  let cases = [];
  for (let other of [0, 1, 3]) {
    for (let accesses of [[0], [0, 1], [1], [-1]]) {
      cases.push({other, accesses});
    }
  }
  let sig = makeSig([kWasmEqRef], [kWasmF64]);
  for (let [n, c] of cases.entries()) {
    let body = [];
    for (let d of c.accesses) {
      body.push(kExprLocalGet, kAcc, ...asFloats, kExprLocalGet, kI,
                ...wasmI32Const(d), kExprI32Add,
                kGCPrefix, kExprArrayGet, floats, kExprF64Add,
                kExprLocalSet, kAcc);
    }
    builder.addFunction(`loop${n}`, sig)
      .addLocals(kWasmI32, 2).addLocals(kWasmF64, 1)
      .addBody([
        // hi = (a is a float array ? a.length : other) - 1
        kExprLocalGet, kA, kGCPrefix, kExprRefTest, floats,
        kExprIf, kWasmI32,
          ...asFloats, kGCPrefix, kExprArrayLen,
        kExprElse,
          ...wasmI32Const(c.other),
        kExprEnd,
        ...wasmI32Const(1), kExprI32Sub, kExprLocalSet, kHi,
        kExprLocalGet, kHi, ...wasmI32Const(0), kExprI32GeS,
        kExprIf, kWasmVoid,
          kExprLoop, kWasmVoid,
            ...body,
            kExprLocalGet, kI, kExprLocalGet, kHi, kExprI32Ne,
            kExprIf, kWasmVoid,
              kExprLocalGet, kI, ...wasmI32Const(1), kExprI32Add,
              kExprLocalSet, kI,
              kExprBr, 1,
            kExprEnd,
          kExprEnd,
        kExprEnd,
        kExprLocalGet, kAcc,
      ])
      .exportFunc();
  }
  let instance = builder.instantiate();
  for (let [n, c] of cases.entries()) {
    for (let isFloats of [true, false]) {
      for (let length of [0, 1, 2, 3, 8]) {
        let a = isFloats ? instance.exports.makeFloats(length)
                         : instance.exports.makeInts(length);
        // The model.
        let expected = 0;
        let hi = (isFloats ? length : c.other) - 1;
        let traps = false;
        let cast = false;
        for (let i = 0; i <= hi && !traps && !cast; i++) {
          for (let d of c.accesses) {
            if (!isFloats) {
              cast = true;
              break;
            }
            if (((i + d) >>> 0) >= length) {
              traps = true;
              break;
            }
            expected += 1.5;
          }
        }
        let run = () => instance.exports[`loop${n}`](a);
        let name = `other ${c.other} accesses ${c.accesses} ` +
                   `floats ${isFloats} length ${length}`;
        if (cast) {
          assertTraps(kTrapIllegalCast, run, name);
        } else if (traps) {
          assertTraps(kTrapArrayOutOfBounds, run, name);
        } else {
          assertEquals(expected, run(), name);
        }
      }
    }
  }
})();

(function TestRandomAccessSequences() {
  print(arguments.callee.name);

  let seed = kRandomSeed | 0 || 1;
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
  function randomUint32() {
    return (random(0x10000) * 0x10000 + random(0x10000)) >>> 0;
  }

  // Function parameters: three arrays, two indices and a flag.
  const kArrays = [0, 1, 2];
  const kI = 3;
  const kJ = 4;
  const kFlag = 5;
  // Locals: an accumulator, a loop variable, and a loop counter.
  const kAcc = 6;
  const kK = 7;
  const kCount = 8;

  // Offsets are chosen close to a few anchors per function. The anchors
  // can be far apart on the circle of 32-bit offsets (half or a third of
  // it, for instance), so that the modular arithmetic of the analysis is
  // exercised. Index values are chosen so that the accesses close to one
  // of the anchors are within bounds.
  const kFarAnchors = [0x55555555, 0xaaaaaaab, 0x40000000, 0xc0000000,
                       0x7fffffff, 0x80000000, 0x80000001, 0xffffffff];
  const kUnsignedBounds = [0, 1, 3, 8, 10, 0x3fffffff, 0x7fffffff,
                           0x80000000, 0x80000001, 0xfffffffe, 0xffffffff];
  const kSignedBounds = [-2, -1, 0, 1, 2, 0x7fffffff, -0x80000000];
  const kIndices = [0, 1, 2, 3, 4, 5, 7, 8, -1, -2, -3, 0x7fffffff,
                    -0x80000000, 0x40000000, 0x3fffffff];

  function randomAnchors() {
    let first = random(4) == 0 ? randomUint32() : 0;
    let anchors = [first];
    let count = pick([0, 0, 1, 1, 2]);
    for (let n = 0; n < count; n++) {
      anchors.push(random(4) == 0 ? randomUint32()
                                  : (first + pick(kFarAnchors)) >>> 0);
    }
    return anchors;
  }

  function randomOffset(anchors) {
    if (random(16) == 0) return randomUint32() | 0;
    return (pick(anchors) + random(9) - 4) | 0;
  }

  // A value for {i} or {j}. Bugs tend to show up when an index used by
  // the function is at the edge of the bounds, so we pick values for
  // which some offset {n} of the function gives an index {i + n} close
  // to 0, or else accesses close to one of the anchors are within
  // bounds.
  function randomIndexValue(offsets, anchors) {
    switch (random(8)) {
      case 0:
        return pick(kIndices);
      case 1:
      case 2:
        return (random(12) - 3 - pick(anchors)) | 0;
      default:
        return (random(7) - 3 - pick(offsets)) | 0;
    }
  }

  // The offsets used with {i}, {j} or {k} in {statements}.
  function collectOffsets(statements, offsets = []) {
    let add = (index) => {
      if (index && index.base != 'none') offsets.push(index.offset);
    };
    let addCondition = (condition) => {
      if (!condition) return;
      add(condition.index);
      addCondition(condition.condition);
      addCondition(condition.left);
      addCondition(condition.right);
    };
    for (let s of statements) {
      add(s.index);
      addCondition(s.condition);
      for (let nested of [s.then, s.else, s.body]) {
        if (nested) collectOffsets(nested, offsets);
      }
    }
    return offsets;
  }

  function randomIndex(bases, anchors) {
    return {base: pick(bases), offset: randomOffset(anchors),
            sub: random(2) == 0};
  }

  function randomUnsignedBound() {
    return random(8) == 0 ? randomUint32() : pick(kUnsignedBounds);
  }

  function randomSignedBound() {
    return random(8) == 0 ? randomUint32() | 0 : pick(kSignedBounds);
  }

  // A condition, on {index} if given.
  function randomCondition(bases, anchors, index, depth = 0) {
    let allow_flag = index === undefined;
    index ??= randomIndex(bases, anchors);
    switch (allow_flag ? random(13) : 1 + random(12)) {
      case 0:
        return {kind: 'flag'};
      case 1:
        return {kind: 'bounds', array: pick(kArrays), index};
      case 2:
        return {kind: 'boundsGt', array: pick(kArrays), index};
      case 3:
        return {kind: 'outOfBounds', array: pick(kArrays), index};
      case 4:
        return {kind: 'geS', bound: randomSignedBound(), index};
      case 5:
        return {kind: 'gtS', bound: randomSignedBound(), index};
      case 6:
        return {kind: 'ltS', bound: randomSignedBound(), index};
      case 7:
        // index < a.length - c, as in wasm_of_ocaml's bounds checks.
        return {kind: 'boundsMinus', array: pick(kArrays), index,
                reduction: 1 + random(2)};
      case 8:
        return {kind: 'outOfBoundsMinus', array: pick(kArrays), index,
                reduction: 1 + random(2)};
      case 9:
        // Signed comparisons, as used by Java, Kotlin or Dart for their
        // loops.
        return {kind: random(2) == 0 ? 'boundsMinusS' : 'outOfBoundsMinusS',
                array: pick(kArrays), index, reduction: random(3)};
      case 10:
      case 11:
        if (depth == 0) {
          // Combined conditions.
          let other = random(2) == 0 ? index : randomIndex(bases, anchors);
          switch (random(3)) {
            case 0:
              return {kind: 'not',
                      condition: randomCondition(bases, anchors, index, 1)};
            case 1:
              return {kind: 'or',
                      left: randomCondition(bases, anchors, index, 1),
                      right: randomCondition(bases, anchors, other, 1)};
            default:
              return {kind: 'and',
                      left: randomCondition(bases, anchors, index, 1),
                      right: randomCondition(bases, anchors, other, 1)};
          }
        }
        // Fall through.
      default:
        return {kind: 'ltU', bound: randomUnsignedBound(), index};
    }
  }

  function randomAccess(array, index) {
    return random(2) == 0
        ? {kind: 'store', array, index, value: random(1000) + 1}
        : {kind: 'load', array, index};
  }

  // A burst of accesses with the same base and nearby offsets, in any
  // order, which fallback code can cover. They are mostly to the same
  // array, with some accesses to other arrays or with offsets close to
  // other anchors mixed in.
  function randomBurst(array, base, anchor, anchors) {
    let statements = [];
    let size = 2 + random(6);
    for (let m = 0; m < size; m++) {
      let center = random(6) == 0 ? pick(anchors) : anchor;
      statements.push(randomAccess(random(4) == 0 ? pick(kArrays) : array, {
        base, offset: (center + random(9) - 4) | 0, sub: random(2) == 0,
      }));
    }
    return statements;
  }

  function randomStatements(depth, bases, anchors, count) {
    let statements = [];
    for (let n = 0; n < count; n++) {
      let r = random(24);
      if (r < 11) {
        statements.push(randomAccess(pick(kArrays),
                                     randomIndex(bases, anchors)));
      } else if (r < 15) {
        statements.push(...randomBurst(pick(kArrays), pick(bases),
                                       pick(anchors), anchors));
      } else if (r < 18 && depth < 2) {
        // A condition on an index, and a burst of accesses with nearby
        // indices where the condition holds, so that what the analysis
        // learns from the condition matters.
        let base = pick(bases);
        let anchor = (pick(anchors) + random(9) - 4) | 0;
        let condition = randomCondition(
            bases, anchors, {base, offset: anchor, sub: random(2) == 0});
        let array = condition.array ?? condition.left?.array ??
                    condition.condition?.array ?? pick(kArrays);
        let then = randomBurst(array, base, anchor,
                               anchors);
        let otherwise = randomStatements(depth + 1, bases, anchors, random(2));
        statements.push(random(2) == 0
            ? {kind: 'if', condition, then, else: otherwise}
            : {kind: 'if', condition, then: otherwise, else: then});
      } else if (r < 19) {
        // Some unrelated computation, to make fallback code larger.
        statements.push({kind: 'filler', size: random(30)});
      } else if (r < 22 && depth < 2) {
        statements.push({
          kind: 'if',
          condition: randomCondition(bases, anchors),
          then: randomStatements(depth + 1, bases, anchors, random(5)),
          else: randomStatements(depth + 1, bases, anchors, random(3)),
        });
      } else if (depth < 1) {
        statements.push({
          kind: 'loop',
          body: randomStatements(depth + 1, [...bases, 'k'], anchors,
                                 1 + random(4)),
        });
      }
    }
    return statements;
  }

  // Code generation.
  function emitIndex(index) {
    let code;
    if (index.base == 'none') return wasmI32Const(index.offset);
    code = [kExprLocalGet, {i: kI, j: kJ, k: kK}[index.base]];
    if (index.offset == 0 && index.sub) return code;
    if (index.sub) {
      return [...code, ...wasmI32Const(-index.offset | 0), kExprI32Sub];
    }
    return [...code, ...wasmI32Const(index.offset | 0), kExprI32Add];
  }

  function emitCondition(condition) {
    let length = (array) => [kExprLocalGet, array, kGCPrefix, kExprArrayLen];
    switch (condition.kind) {
      case 'flag':
        return [kExprLocalGet, kFlag];
      case 'bounds':
        return [...emitIndex(condition.index), ...length(condition.array),
                kExprI32LtU];
      case 'boundsGt':
        return [...length(condition.array), ...emitIndex(condition.index),
                kExprI32GtU];
      case 'outOfBounds':
        return [...emitIndex(condition.index), ...length(condition.array),
                kExprI32GeU];
      case 'boundsMinus':
        return [...emitIndex(condition.index), ...length(condition.array),
                ...wasmI32Const(condition.reduction), kExprI32Sub,
                kExprI32LtU];
      case 'outOfBoundsMinus':
        return [...length(condition.array),
                ...wasmI32Const(condition.reduction), kExprI32Sub,
                ...emitIndex(condition.index), kExprI32LeU];
      case 'boundsMinusS':
        return [...emitIndex(condition.index), ...length(condition.array),
                ...wasmI32Const(condition.reduction), kExprI32Sub,
                kExprI32LtS];
      case 'outOfBoundsMinusS':
        return [...length(condition.array),
                ...wasmI32Const(condition.reduction), kExprI32Sub,
                ...emitIndex(condition.index), kExprI32LeS];
      case 'not':
        return [...emitCondition(condition.condition), kExprI32Eqz];
      case 'or':
        return [...emitCondition(condition.left),
                ...emitCondition(condition.right), kExprI32Ior];
      case 'and':
        return [...emitCondition(condition.left),
                ...emitCondition(condition.right), kExprI32And];
      case 'geS':
        return [...emitIndex(condition.index), ...wasmI32Const(condition.bound),
                kExprI32GeS];
      case 'gtS':
        return [...emitIndex(condition.index), ...wasmI32Const(condition.bound),
                kExprI32GtS];
      case 'ltS':
        return [...emitIndex(condition.index), ...wasmI32Const(condition.bound),
                kExprI32LtS];
      case 'ltU':
        return [...emitIndex(condition.index),
                ...wasmI32Const(condition.bound | 0), kExprI32LtU];
    }
  }

  function emitStatements(statements, array_type) {
    let code = [];
    for (let s of statements) {
      switch (s.kind) {
        case 'store':
          code.push(kExprLocalGet, s.array, ...emitIndex(s.index),
                    ...wasmI32Const(s.value),
                    kGCPrefix, kExprArraySet, array_type);
          break;
        case 'load':
          code.push(kExprLocalGet, kAcc, ...wasmI32Const(31), kExprI32Mul,
                    kExprLocalGet, s.array, ...emitIndex(s.index),
                    kGCPrefix, kExprArrayGet, array_type,
                    kExprI32Add, kExprLocalSet, kAcc);
          break;
        case 'filler':
          for (let n = 0; n < s.size; n++) {
            code.push(kExprLocalGet, kAcc, ...wasmI32Const(n + 3), kExprI32Xor,
                      kExprLocalSet, kAcc);
          }
          break;
        case 'if':
          code.push(...emitCondition(s.condition), kExprIf, kWasmVoid,
                    ...emitStatements(s.then, array_type), kExprElse,
                    ...emitStatements(s.else, array_type), kExprEnd);
          break;
        case 'loop':
          // for (k = i, count = 0; count < 3; k++, count++) body
          code.push(kExprLocalGet, kI, kExprLocalSet, kK,
                    ...wasmI32Const(0), kExprLocalSet, kCount,
                    kExprLoop, kWasmVoid,
                      ...emitStatements(s.body, array_type),
                      kExprLocalGet, kK, ...wasmI32Const(1), kExprI32Add,
                      kExprLocalSet, kK,
                      kExprLocalGet, kCount, ...wasmI32Const(1), kExprI32Add,
                      kExprLocalTee, kCount,
                      ...wasmI32Const(3), kExprI32LtU,
                      kExprBrIf, 0,
                    kExprEnd);
          break;
      }
    }
    return code;
  }

  // The model.
  class Trap {
    constructor(message) {
      this.message = message;
    }
  }

  function evalIndex(index, env) {
    let base = index.base == 'none' ? 0 : env[index.base];
    return (base + index.offset) | 0;
  }

  function checkedIndex(array, index) {
    if (array === null) throw new Trap(kNull);
    if ((index >>> 0) >= array.length) throw new Trap(kOutOfBounds);
    return index >>> 0;
  }

  function evalCondition(condition, env) {
    let index = condition.index ? evalIndex(condition.index, env) : 0;
    let length = (array) => {
      if (env.arrays[array] === null) throw new Trap(kNull);
      return env.arrays[array].length;
    };
    switch (condition.kind) {
      case 'flag':
        return env.flag != 0;
      case 'bounds':
      case 'boundsGt':
        return (index >>> 0) < length(condition.array);
      case 'outOfBounds':
        return (index >>> 0) >= length(condition.array);
      case 'boundsMinus':
        return (index >>> 0) <
               ((length(condition.array) - condition.reduction) >>> 0);
      case 'outOfBoundsMinus':
        return ((length(condition.array) - condition.reduction) >>> 0) <=
               (index >>> 0);
      case 'geS':
        return index >= condition.bound;
      case 'gtS':
        return index > condition.bound;
      case 'ltS':
        return index < condition.bound;
      case 'ltU':
        return (index >>> 0) < (condition.bound >>> 0);
      case 'boundsMinusS':
        return index < ((length(condition.array) - condition.reduction) | 0);
      case 'outOfBoundsMinusS':
        return ((length(condition.array) - condition.reduction) | 0) <= index;
      case 'not':
        return !evalCondition(condition.condition, env);
      case 'or':
      case 'and': {
        // Both conditions are evaluated.
        let left = evalCondition(condition.left, env);
        let right = evalCondition(condition.right, env);
        return condition.kind == 'or' ? left || right : left && right;
      }
    }
  }

  function evalStatements(statements, env) {
    for (let s of statements) {
      switch (s.kind) {
        case 'store': {
          let array = env.arrays[s.array];
          array[checkedIndex(array, evalIndex(s.index, env))] = s.value;
          break;
        }
        case 'load': {
          let array = env.arrays[s.array];
          let value = array[checkedIndex(array, evalIndex(s.index, env))];
          env.acc = (Math.imul(env.acc, 31) + value) | 0;
          break;
        }
        case 'filler':
          for (let n = 0; n < s.size; n++) env.acc ^= n + 3;
          break;
        case 'if':
          evalStatements(evalCondition(s.condition, env) ? s.then : s.else,
                         env);
          break;
        case 'loop':
          env.k = env.i;
          for (let count = 0; count < 3; count++) {
            evalStatements(s.body, env);
            env.k = (env.k + 1) | 0;
          }
          break;
      }
    }
  }

  const kFunctionCount = 200;
  const kRunsPerFunction = 24;
  const kLengths = [0, 1, 2, 5, 8];

  let builder = new WasmModuleBuilder();
  let array = builder.addArray(kWasmI32);
  addArrayHelpers(builder, array);
  let sig = makeSig([wasmRefNullType(array), wasmRefNullType(array),
                     wasmRefNullType(array), kWasmI32, kWasmI32, kWasmI32],
                    [kWasmI32]);
  let functions = [];
  for (let f = 0; f < kFunctionCount; f++) {
    let anchors = randomAnchors();
    let statements = randomStatements(0, ['i', 'i', 'j', 'none'], anchors,
                                      3 + random(10));
    let offsets = collectOffsets(statements);
    if (offsets.length == 0) offsets = anchors;
    functions.push({statements, anchors, offsets});
    builder.addFunction(`f${f}`, sig)
      .addLocals(kWasmI32, 3)
      .addBody([...emitStatements(statements, array), kExprLocalGet, kAcc])
      .exportFunc();
  }
  let instance = builder.instantiate();

  for (let f = 0; f < kFunctionCount; f++) {
    for (let run = 0; run < kRunsPerFunction; run++) {
      let lengths = kArrays.map(() => pick(kLengths));
      let arrays = lengths.map(length => instance.exports.make(length));
      let env = {
        arrays: lengths.map(length => new Array(length).fill(0)),
        i: randomIndexValue(functions[f].offsets, functions[f].anchors),
        j: randomIndexValue(functions[f].offsets, functions[f].anchors),
        flag: random(2),
        acc: 0,
      };
      // Occasionally pass the same array twice, or a null array.
      if (random(8) == 0) {
        arrays[1] = arrays[0];
        env.arrays[1] = env.arrays[0];
      }
      if (random(16) == 0) {
        arrays[2] = null;
        env.arrays[2] = null;
      }
      let expected;
      try {
        evalStatements(functions[f].statements, env);
        expected = env.acc;
      } catch (e) {
        if (!(e instanceof Trap)) throw e;
        expected = e;
      }
      let actual;
      try {
        actual = instance.exports[`f${f}`](...arrays, env.i, env.j, env.flag);
      } catch (e) {
        if (!(e instanceof WebAssembly.RuntimeError)) throw e;
        actual = new Trap(e.message);
      }
      let description = `seed ${kRandomSeed}: f${f}(i=${env.i}, j=${env.j}, ` +
          `flag=${env.flag}, lengths=${lengths})`;
      if (expected instanceof Trap) {
        assertInstanceof(actual, Trap, description);
        assertEquals(expected.message, actual.message, description);
      } else {
        assertEquals(expected, actual, description);
      }
      for (let n = 0; n < arrays.length; n++) {
        if (arrays[n] === null) continue;
        assertEquals(env.arrays[n],
                     contents(instance, arrays[n], env.arrays[n].length),
                     `${description}: array ${n}`);
      }
    }
  }
})();
