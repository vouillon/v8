// Copyright 2026 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Flags: --wasm-imported-strings-utf8

// Exercises the boundaries between the inline fast paths of the
// wasm:text-decoder / wasm:text-encoder imports (short pure-ASCII inputs) and
// the runtime paths (everything else), and the SIMD paths in the runtime.

d8.file.execute("test/mjsunit/wasm/wasm-module-builder.js");

let builder = new WasmModuleBuilder();
builder.startRecGroup();
let kArrayI8 = builder.addArray(kWasmI8, {final: true});
builder.endRecGroup();
let a8 = wasmRefNullType(kArrayI8);
let a8nn = wasmRefType(kArrayI8);
let kRefExtern = wasmRefType(kWasmExternRef);

let kDecode = builder.addImport(
    'wasm:text-decoder', 'decodeStringFromUTF8Array',
    makeSig([a8, kWasmI32, kWasmI32], [kRefExtern]));
let kMeasure = builder.addImport(
    'wasm:text-encoder', 'measureStringAsUTF8', kSig_i_r);
let kEncodeInto = builder.addImport(
    'wasm:text-encoder', 'encodeStringIntoUTF8Array',
    makeSig([kWasmExternRef, a8, kWasmI32], [kWasmI32]));
let kEncode = builder.addImport(
    'wasm:text-encoder', 'encodeStringToUTF8Array',
    makeSig([kWasmExternRef], [a8nn]));

builder.addFunction('decode', makeSig([a8, kWasmI32, kWasmI32], [kRefExtern]))
    .exportFunc()
    .addBody([
      kExprLocalGet, 0, kExprLocalGet, 1, kExprLocalGet, 2,
      kExprCallFunction, kDecode
    ]);
builder.addFunction('measure', kSig_i_r)
    .exportFunc()
    .addBody([kExprLocalGet, 0, kExprCallFunction, kMeasure]);
builder.addFunction('encode_into',
                    makeSig([kWasmExternRef, a8, kWasmI32], [kWasmI32]))
    .exportFunc()
    .addBody([
      kExprLocalGet, 0, kExprLocalGet, 1, kExprLocalGet, 2,
      kExprCallFunction, kEncodeInto
    ]);
builder.addFunction('encode', makeSig([kWasmExternRef], [a8nn]))
    .exportFunc()
    .addBody([kExprLocalGet, 0, kExprCallFunction, kEncode]);
builder.addFunction('new_array', makeSig([kWasmI32], [a8nn]))
    .exportFunc()
    .addBody([kExprLocalGet, 0, kGCPrefix, kExprArrayNewDefault, kArrayI8]);
builder.addFunction('array_len', makeSig([a8], [kWasmI32]))
    .exportFunc()
    .addBody([kExprLocalGet, 0, kGCPrefix, kExprArrayLen]);
builder.addFunction('array_get', makeSig([a8, kWasmI32], [kWasmI32]))
    .exportFunc()
    .addBody([
      kExprLocalGet, 0, kExprLocalGet, 1, kGCPrefix, kExprArrayGetU, kArrayI8
    ]);
builder.addFunction('array_set', makeSig([a8, kWasmI32, kWasmI32], []))
    .exportFunc()
    .addBody([
      kExprLocalGet, 0, kExprLocalGet, 1, kExprLocalGet, 2,
      kGCPrefix, kExprArraySet, kArrayI8
    ]);

let ex = builder.instantiate({}, {builtins: ['text-decoder', 'text-encoder']})
             .exports;

// -----------------------------------------------------------------------------
// Helpers.

function IsSurrogate(cp) { return 0xD800 <= cp && cp <= 0xDFFF; }

// Lossy UTF-8 encoding: isolated surrogates become U+FFFD.
function encodeUtf8Lossy(str) {
  let out = [];
  for (let c of str) {
    let cp = c.codePointAt(0);
    if (IsSurrogate(cp)) cp = 0xFFFD;
    if (cp <= 0x7f) {
      out.push(cp);
    } else if (cp <= 0x7ff) {
      out.push(0xc0 | (cp >> 6), 0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
      out.push(0xe0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3f),
               0x80 | (cp & 0x3f));
    } else {
      out.push(0xf0 | (cp >> 18), 0x80 | ((cp >> 12) & 0x3f),
               0x80 | ((cp >> 6) & 0x3f), 0x80 | (cp & 0x3f));
    }
  }
  return out;
}

function replaceIsolatedSurrogates(str) {
  let out = '';
  for (let c of str) {
    out += IsSurrogate(c.codePointAt(0)) ? '�' : c;
  }
  return out;
}

function arrayToBytes(arr) {
  let n = ex.array_len(arr);
  let bytes = [];
  for (let i = 0; i < n; i++) bytes.push(ex.array_get(arr, i));
  return bytes;
}

function bytesToArray(bytes) {
  let arr = ex.new_array(bytes.length);
  for (let i = 0; i < bytes.length; i++) ex.array_set(arr, i, bytes[i]);
  return arr;
}

function checkRoundTrip(str) {
  let expectedBytes = encodeUtf8Lossy(str);
  assertEquals(expectedBytes.length, ex.measure(str), `measure(${str})`);

  let arr = ex.encode(str);
  assertEquals(expectedBytes, arrayToBytes(arr), `encode(${str})`);

  // encodeInto with an exactly sized array, and into the middle of a larger
  // one.
  let exact = ex.new_array(expectedBytes.length);
  assertEquals(expectedBytes.length, ex.encode_into(str, exact, 0));
  assertEquals(expectedBytes, arrayToBytes(exact), `encodeInto(${str})`);
  let padded = ex.new_array(expectedBytes.length + 5);
  assertEquals(expectedBytes.length, ex.encode_into(str, padded, 3));
  assertEquals([0, 0, 0, ...expectedBytes, 0, 0], arrayToBytes(padded));

  // Decoding the encoding gives the string back, with isolated surrogates
  // replaced.
  let expectedStr = replaceIsolatedSurrogates(str);
  assertEquals(expectedStr, ex.decode(arr, 0, expectedBytes.length),
               `decode(encode(${str}))`);
  assertEquals(expectedStr, ex.decode(padded, 3, 3 + expectedBytes.length));
}

// -----------------------------------------------------------------------------

(function TestAsciiLengths() {
  print(arguments.callee.name);
  // Covers the inline threshold (currently 32) and the 8-byte scan loop.
  let base = 'abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' +
             'abcdefghijklmnopqrstuvwxyz0123456789';
  for (let len = 0; len <= 80; len++) {
    checkRoundTrip(base.substring(0, len));
  }
})();

(function TestNonAsciiAtEveryPosition() {
  print(arguments.callee.name);
  let specials = [
    '\x80',       // Smallest non-ASCII.
    '\xe9',       // Latin-1.
    '\xff',       // Largest Latin-1: C3 BF.
    'Ā',     // Smallest non-Latin-1: C4 80.
    '߿',     // Largest two-byte sequence.
    'ࠀ',     // Smallest three-byte sequence.
    '中',     // CJK.
    '￿',     // Largest BMP.
    '\u{10000}',  // Smallest supplementary.
    '\u{1F600}',  // Emoji.
    '\u{10FFFF}', // Largest code point.
    '\ud800',     // Isolated lead surrogate.
    '\udfff',     // Isolated trail surrogate.
  ];
  for (let len of [1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 40, 64, 100]) {
    for (let special of specials) {
      for (let pos = 0; pos < len; pos += (len > 20 ? 7 : 1)) {
        let str = 'a'.repeat(pos) + special + 'b'.repeat(len - pos - 1);
        checkRoundTrip(str);
      }
    }
  }
})();

(function TestSurrogateCombinations() {
  print(arguments.callee.name);
  for (let str of [
    '𐀀',           // Proper pair.
    '\udc00\ud800',           // Swapped.
    '\ud800𐀀',     // Lead, then pair.
    '𐀀\udc00',     // Pair, then trail.
    'a\ud800', '\udc00a', 'a\ud800b\udc00c',
    '\ud800'.repeat(20), '\udc00'.repeat(20),
    ('ab𐀀').repeat(30),
  ]) {
    checkRoundTrip(str);
  }
})();

(function TestLongStrings() {
  print(arguments.callee.name);
  // Long enough to exercise the SIMD paths, with the non-ASCII part starting
  // at various alignments.
  for (let prefix of [0, 1, 7, 8, 15, 16, 17, 63, 64, 65, 200]) {
    for (let special of ['\xe9', '\xff', 'Ā', '中', '\u{1F600}',
                         '\ud800']) {
      checkRoundTrip('x'.repeat(prefix) + special.repeat(300) +
                     'y'.repeat(prefix));
      checkRoundTrip('x'.repeat(prefix) + special + 'y'.repeat(1000));
    }
  }
  checkRoundTrip('z'.repeat(5000));
})();

(function TestNonFlatStrings() {
  print(arguments.callee.name);
  let long = 'abcdefghijklmnopqrstuvwxyz'.repeat(4);
  let cons = long + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';  // ConsString.
  checkRoundTrip(cons);
  checkRoundTrip(cons.substring(3, 30));  // SlicedString.
  checkRoundTrip(cons.substring(3, 30) + '\xe9');
  let consNonAscii = long + '中中中中中中中';
  checkRoundTrip(consNonAscii);
  checkRoundTrip(consNonAscii.substring(100, 120));
})();

(function TestEncodeIntoBounds() {
  print(arguments.callee.name);
  for (let str of ['', 'a', 'abcdefgh', 'x'.repeat(31), 'x'.repeat(32),
                   'x'.repeat(33), 'x'.repeat(100), '\xe9\xe9', '中',
                   '\u{1F600}', 'ab\ud800']) {
    let len = encodeUtf8Lossy(str).length;
    if (len == 0) continue;
    // One byte short, at start 0 and further in: traps and leaves the array
    // untouched.
    for (let start of [0, 1, 5]) {
      let arr = ex.new_array(start + len - 1);
      assertThrows(() => ex.encode_into(str, arr, start),
                   WebAssembly.RuntimeError,
                   'array element access out of bounds');
      assertEquals(new Array(start + len - 1).fill(0), arrayToBytes(arr));
    }
    // Start beyond the end.
    let arr = ex.new_array(len);
    assertThrows(() => ex.encode_into(str, arr, len + 1),
                 WebAssembly.RuntimeError, 'array element access out of bounds');
    assertThrows(() => ex.encode_into(str, arr, -1),
                 WebAssembly.RuntimeError, 'array element access out of bounds');
    // Exactly fitting at the very end.
    let arr2 = ex.new_array(len + 4);
    assertEquals(len, ex.encode_into(str, arr2, 4));
    assertEquals([0, 0, 0, 0, ...encodeUtf8Lossy(str)], arrayToBytes(arr2));
  }
  // Empty string into an empty array, and at the end of an array.
  assertEquals(0, ex.encode_into('', ex.new_array(0), 0));
  assertEquals(0, ex.encode_into('', ex.new_array(3), 3));
  assertThrows(() => ex.encode_into('', ex.new_array(3), 4),
               WebAssembly.RuntimeError, 'array element access out of bounds');
})();

(function TestDecodeRanges() {
  print(arguments.callee.name);
  let bytes = [];
  for (let i = 0; i < 70; i++) bytes.push(0x41 + (i % 26));
  let arr = bytesToArray(bytes);
  let str = String.fromCharCode(...bytes);
  for (let start = 0; start <= 70; start += 3) {
    for (let end = start; end <= 70; end += (end - start < 40 ? 1 : 5)) {
      assertEquals(str.substring(start, end), ex.decode(arr, start, end));
    }
  }
  assertThrows(() => ex.decode(arr, 0, 71), WebAssembly.RuntimeError,
               'array element access out of bounds');
  assertThrows(() => ex.decode(arr, 5, 4), WebAssembly.RuntimeError,
               'array element access out of bounds');
  assertThrows(() => ex.decode(arr, -1, 0), WebAssembly.RuntimeError,
               'array element access out of bounds');
  // A non-ASCII byte just outside the decoded range must not matter.
  let bytes2 = bytes.slice();
  bytes2[10] = 0xe9;
  let arr2 = bytesToArray(bytes2);
  assertEquals(str.substring(0, 10), ex.decode(arr2, 0, 10));
  assertEquals(str.substring(11, 40), ex.decode(arr2, 11, 40));
  assertEquals('�', ex.decode(arr2, 10, 11));
  assertEquals(str.substring(0, 10) + '�' + str.substring(11, 20),
               ex.decode(arr2, 0, 20));
})();

(function TestDecodeInvalid() {
  print(arguments.callee.name);
  // [bytes, expected] pairs, following the WHATWG UTF-8 decoder (which is what
  // TextDecoder implements).
  let cases = [
    [[0x80], '�'],
    [[0xc0, 0x80], '��'],                  // Overlong.
    [[0xc1, 0xbf], '��'],                  // Overlong.
    [[0xc2], '�'],                              // Truncated.
    [[0xe2, 0x82], '�'],                        // Truncated.
    [[0xe2, 0x82, 0x41], '�A'],                 // Truncated then ASCII.
    [[0xf0, 0x9f, 0x98], '�'],                  // Truncated 4-byte.
    [[0xed, 0xa0, 0x80], '���'],      // Encoded surrogate.
    [[0xf4, 0x90, 0x80, 0x80], '����'],  // > U+10FFFF.
    [[0xf5, 0x80], '��'],
    [[0xff], '�'],
    [[0xc3, 0xbf], '\xff'],                          // Valid: U+00FF.
    [[0xc4, 0x80], 'Ā'],                        // Valid: U+0100.
    [[0xc3, 0xbf, 0xc4, 0x80], '\xffĀ'],
    [[0x41, 0xc3, 0xbf, 0x80], 'A\xff�'],
    [[0xe0, 0x80, 0x80], '���'],      // Overlong 3-byte.
    [[0xf0, 0x80, 0x80, 0x80], '����'],  // Overlong.
    [[0xef, 0xbf, 0xbf], '￿'],
    [[0xf4, 0x8f, 0xbf, 0xbf], '\u{10FFFF}'],
  ];
  for (let [bytes, expected] of cases) {
    // Alone, and after ASCII prefixes of various lengths (which move the
    // sequence across the alignment boundaries of the ASCII prefix scan and
    // past the inline threshold).
    for (let prefix of [0, 1, 3, 8, 9, 16, 31, 32, 33, 64]) {
      let all = [...new Array(prefix).fill(0x61), ...bytes];
      let arr = bytesToArray(all);
      assertEquals('a'.repeat(prefix) + expected,
                   ex.decode(arr, 0, all.length),
                   `prefix ${prefix} bytes ${bytes}`);
    }
    // Long tail after the sequence, to reach the SIMD paths in the runtime.
    let all = [...bytes, ...new Array(300).fill(0x62)];
    let arr = bytesToArray(all);
    assertEquals(expected + 'b'.repeat(300), ex.decode(arr, 0, all.length));
    // Long valid non-ASCII tail.
    let tail = [];
    for (let i = 0; i < 100; i++) tail.push(0xc3, 0xa9);
    all = [...bytes, ...tail];
    arr = bytesToArray(all);
    assertEquals(expected + '\xe9'.repeat(100), ex.decode(arr, 0, all.length));
  }
})();

(function TestDecodeLatin1Boundary() {
  print(arguments.callee.name);
  // Long strings where exactly one character decides between a one-byte and a
  // two-byte result.
  for (let pos of [0, 1, 50, 199]) {
    for (let [seq, ch] of [[[0xc3, 0xbf], '\xff'], [[0xc4, 0x80], 'Ā']]) {
      let bytes = [];
      let expected = '';
      for (let i = 0; i < 200; i++) {
        if (i == pos) {
          bytes.push(...seq);
          expected += ch;
        } else {
          bytes.push(0xc2, 0xa9);
          expected += '\xa9';
        }
      }
      let arr = bytesToArray(bytes);
      assertEquals(expected, ex.decode(arr, 0, bytes.length));
    }
  }
})();
