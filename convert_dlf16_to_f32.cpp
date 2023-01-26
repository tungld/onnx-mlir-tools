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
  // dlf16 to f32
  //--------------------------------------------------------------------------//
  uint16_t dlf16_item = atoi(argv[1]);
  float fp = 1;

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

  if ((dlf16_item & ~DLF16_SIGN) == 0) {
    fp = ((dlf16_item & DLF16_SIGN) == 0) ? +0.0f : -0.0f;
  } else if ((dlf16_item & ~DLF16_SIGN) == DLF16_NINF) {
    fp = NAN;
  } else {
    uint32_t fp32 = FP32_SIGN * BFGET(dlf16_item, DLF16_SIGN);
    BFPUT(fp32, FP32_EXPONENT,
        BFGET(dlf16_item, DLF16_EXPONENT) - DLF16_EXPONENT_BIAS +
            FP32_EXPONENT_BIAS);
    BFPUT(fp32, FP32_FRACTION,
        BFGET(dlf16_item, DLF16_FRACTION)
            << (FP32_FRACTION_BITS - DLF16_FRACTION_BITS));
    memcpy(&fp, &fp32, sizeof(fp));
  }

  printf("fp: %f\n", fp); // expected result: 0.012344 
  return 1;
}
