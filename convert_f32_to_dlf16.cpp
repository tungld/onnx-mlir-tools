/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- DLF16 Conversion -----------------------------===//
// Extracted from  NNP1 class in DLF16Conversion.hpp in onnx-mlir.
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>

// Macros for FPFormat.
#define SIGN(one, exp, frac) (one << exp << frac)
#define EXPO(one, exp, frac) (((one << exp) - 1) << frac)
#define EXPO_BIAS(one, exp, frac) ((one << (exp - 1)) - 1)
#define FRAC(one, frac) ((one << frac) - 1)

// Macros for conversion.
#define BFLAST(mask) ((mask) & (1 + ~(mask)))
#define BFGET(w, mask) (((w) & (mask)) / BFLAST(mask))
#define BFPUT(w, mask, value)                                                  \
  ((w) = ((w) & ~(mask)) | (((value)*BFLAST(mask)) & (mask)))

int main(int argc, char *argv[]) {
  // float x = 0.0123456789;
  // uint16_t y = 12585; // dlfloat16 of fp32 0.0123456789
  // fp32_to_dlf16(&x, &y, 1);
  // float z = 1;
  // dlf16_to_fp32(&y, &z, 1);
  // printf("x: %f\n", x);
  // printf("y in int: %i\n", y);
  // printf("z: %f\n", z);

  //--------------------------------------------------------------------------//
  // f32 to dlf16
  //--------------------------------------------------------------------------//
  float fp = atof(argv[1]); // input 0.0123456789
  // assert((fp == 0.0123456789f) && "please input 0.0123456789");
  uint16_t dlf16_item; // expected output: 12585

  // from constructor of parent FPFormat: `public FPFormat<uint16_t, 6, 9>`
  static constexpr unsigned DLF16_EXPONENT_BITS = 6;
  static constexpr unsigned DLF16_FRACTION_BITS = 9;
  static constexpr uint16_t DLF16_One = 1;
  static constexpr uint16_t DLF16_SIGN =
      SIGN(DLF16_One, DLF16_EXPONENT_BITS, DLF16_FRACTION_BITS);
  static constexpr uint16_t DLF16_EXPONENT =
      EXPO(DLF16_One, DLF16_EXPONENT_BITS, DLF16_FRACTION_BITS);
  static constexpr signed DLF16_EXPONENT_BIAS =
      EXPO_BIAS(DLF16_One, DLF16_EXPONENT_BITS, DLF16_FRACTION_BITS);
  static constexpr uint16_t DLF16_FRACTION =
      FRAC(DLF16_One, DLF16_FRACTION_BITS);
  static constexpr uint16_t DLF16_NINF = DLF16_EXPONENT | DLF16_FRACTION;

  // from consturtor of FP32:FPFormat<uint32_t, 8, 23>
  static constexpr uint32_t FP32_One = 1;
  static constexpr unsigned FP32_EXPONENT_BITS = 8;
  static constexpr unsigned FP32_FRACTION_BITS = 23;
  static constexpr uint32_t FP32_SIGN =
      SIGN(FP32_One, FP32_EXPONENT_BITS, FP32_FRACTION_BITS);
  static constexpr uint32_t FP32_EXPONENT =
      EXPO(FP32_One, FP32_EXPONENT_BITS, FP32_FRACTION_BITS);
  static constexpr signed FP32_EXPONENT_BIAS =
      EXPO_BIAS(FP32_One, FP32_EXPONENT_BITS, FP32_FRACTION_BITS);
  static constexpr uint32_t FP32_FRACTION = FRAC(FP32_One, FP32_FRACTION_BITS);

  static constexpr uint32_t FP32_DLF16_ROUND =
      1 << (FP32_FRACTION_BITS - DLF16_FRACTION_BITS - 1);

  static constexpr uint32_t FP32_DLF16_NMAX =
      (((1 << DLF16_EXPONENT_BITS) - 1 + FP32_EXPONENT_BIAS -
        DLF16_EXPONENT_BIAS)
       << FP32_FRACTION_BITS) |
      (((1 << DLF16_FRACTION_BITS) - 2)
       << (FP32_FRACTION_BITS - DLF16_FRACTION_BITS)) |
      (FP32_DLF16_ROUND - 1);

  // Conversion.
  uint32_t fp32;
  memcpy(&fp32, &fp, sizeof(fp32));

  signed nnp1_biased_exponent =
      BFGET(fp32, FP32_EXPONENT) - FP32_EXPONENT_BIAS + DLF16_EXPONENT_BIAS;
  uint32_t fraction = BFGET(fp32, FP32_FRACTION) + FP32_DLF16_ROUND;

  if (fraction > FP32_FRACTION) {
    fraction = 0;
    nnp1_biased_exponent++;
  }

  uint16_t uint = DLF16_SIGN * BFGET(fp32, FP32_SIGN);

  if (nnp1_biased_exponent >= 0) {
    if ((fp32 & ~FP32_SIGN) <= FP32_DLF16_NMAX) {
      BFPUT(uint, DLF16_EXPONENT, nnp1_biased_exponent);
      BFPUT(uint, DLF16_FRACTION,
            fraction >> (FP32_FRACTION_BITS - DLF16_FRACTION_BITS));
    } else {
      uint |= DLF16_NINF;
    }
  }

  // assert(uint == 12585);
  // printf("dlf16 in uint: %i\n", uint); // expected result: 12585
  return uint;
}
