// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2021
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#if defined(__MVS__)    // If z/OS, use XL C include and typedef
#include <builtins.h>   // needed for XL C vector ops
#else                   // If LoZ, use gcc include and typedef
#include <s390intrin.h> // needed for LoZ vector ops
#endif

typedef vector unsigned int vec_float32;
typedef vector unsigned short vec_int16;
typedef vector unsigned char vec_char8;

vec_char8 selection_vector = {0,  1,  4,  5,  8,  9,  12, 13,
                              16, 17, 20, 21, 24, 25, 28, 29};
static vec_int16 zero_vector16 = {0, 0, 0, 0, 0, 0, 0, 0};

typedef uint16_t float_bit16;

typedef union uint32_float_u {
  uint32_t u;
  float f;
} uint32_float_u;

/***********************************************************************
 * aiu_vec_round_from_fp32 routines
 *
 * Converts 2 vectors (4 elements each) of 32-bit floating point
 * numbers to 1 vector of 16-bit DLFLOAT numbers (8 numbers total)
 *
 * Input: 2 vectors of 4 FP32 data elements to convert
 * Output: vector of 8 DLFloat16 floats
 *
 * Note:  There is also a non-inlined wrapper function as well.
 **********************************************************************/

static vec_int16 inline aiu_vec_round_from_fp32_inline(vec_float32 a,
                                                       vec_float32 b) {
  vec_int16 out;

#ifndef ZDNN_CONFIG_NO_NNPA
#if defined(__MVS__)
  /*
       Invoke the VCRNF
                  "*     VCRNF VReg0,VRegL,VRegR,mask2,0        \n\t"
       Note that registers are hardcoded (vs using %0 notation) to ensure
       that the hardcoded instruction (E60120020075) has expected regs free
    */
  // clang-format off
  __asm volatile("      VL    1,%[in_vector_left]            \n\t"
                 "      VL    2,%[in_vector_right]           \n\t"
                 "      DC    XL6'E60120020075'              \n\t"
                 "      DS    0H                             \n\t"
                 "      VST   0,%[out_vector]                \n\t"
                 : /* Outputs - out_vector */
                 [out_vector] "=m"(out) //(out_vector)
                 :                        /* Inputs                        */
                 [mask2] "i"(2),        /* 2 = Internal NNPA format (DLF)  */
                 [mask0] "i"(0),        /* 0 = FP32                        */
                 [in_vector_left] "m"(a), /* data */
                 [in_vector_right] "m"(b) /* data */
                 :  /* "%v0", "%v1", "%v2"   Clobbered */
  );
// clang-format on
#else
  // clang-format off
  __asm volatile(".insn vrr,0xe60000000075,%[out],%[in_hi],%[in_lo],0,2,0"
                : [ out ] "=v"(out)
                : [ in_hi ] "v"(a), [ in_lo ] "v"(b));
// clang-format on
#endif
#else
  // cast our 32 bit vectors as bytes, then select via vec_perm which
  // bytes to return. (Will be first 2 bytes of every float32)
  out = (vec_int16)vec_perm((vec_char8)a, (vec_char8)b, selection_vector);
#endif // ZDNN_CONFIG_NO_NNPA

  return out;
} // End aiu_vec_round_from_fp32

//  Common version of non-inlined aiu_vec_round_from_fp32
vec_int16 aiu_vec_round_from_fp32(vec_float32 a, vec_float32 b) {
  return aiu_vec_round_from_fp32_inline(a, b);
}

/***********************************************************************
 * aiu_vec_convert_from_fp16
 *
 * Converts 1 vector (8 elements) of 16-bit floating point
 * numbers to 1 vector of 16-bit DLFLOAT numbers (8 numbers total)
 *
 * Input: 1 vector of 8 FP16 data elements to convert
 * Output: vector of 8 DLFloat16 floats
 *
 * Note:  There is also a non-inlined wrapper function as well.
 **********************************************************************/

static vec_int16 inline aiu_vec_convert_from_fp16_inline(vec_int16 a) {
  vec_int16 out;

#ifndef ZDNN_CONFIG_NO_NNPA
#if defined(__MVS__)
  /*
      Invoke the VCNF
                 "*     VCNF VReg0,VReg1,,mask1,0        \n\t"
      Note that registers are hardcoded (vs using %0 notation) to ensure
      that the hardcoded instruction (E60100010055) has expected regs free
   */
  // clang-format off
  __asm volatile("      VL    1,%[in_vector]                 \n\t"
                 "      DC    XL6'E60100010055'              \n\t"
                 "      DS    0H                             \n\t"
                 "      VST   0,%[out_vector]                \n\t"
                 : /* Outputs - out_vector */
                 [out_vector] "=m"(out) //(out_vector)
                 :                      /* Inputs                        */
                 [mask1] "i"(1),        /* 1 = BFP tiny format (FP16)    */
                 [mask0] "i"(0),        /* 0 = NNP format                */
                 [in_vector] "m"(a)     /* data */
                 :  /* "%v0", "%v1"   Clobbered */
                 // clang-format on
  );
#else
  // clang-format off
    __asm volatile(".insn vrr,0xe60000000055,%[out],%[in_vec],0,0,1,0"
                : [ out ] "=v"(out)
                : [ in_vec ] "v"(a));
// clang-format on
#endif
#else
  // scaffolding: just copy the input 16-bit elements as is to output
  memcpy(&out, &a, sizeof(vec_int16));
#endif // ZDNN_CONFIG_NO_NNPA

  return out;
} // End aiu_vec_convert_from_fp16

//  Common wrapper version of non-inlined aiu_vec_convert_from_fp16
vec_int16 aiu_vec_convert_from_fp16(vec_int16 a) {
  return aiu_vec_convert_from_fp16_inline(a);
}

/***********************************************************************
 * aiu_vec_lengthen_to_fp32
 *
 * Converts 1 vector of 16-bit DLFLOAT numbers (8 numbers total) to
 * 2 vectors (4 elements each) of 32-bit floating point
 *
 * Input: 1 vector (input) of 8 DLFloat16 floats to convert
 *        2 vectors (output) of 4 FP32 data elements
 *
 * Note:  There is also a non-inlined wrapper function as well.
 **********************************************************************/

static void inline aiu_vec_lengthen_to_fp32_inline(vec_int16 a,
                                                   vec_float32 *out1,
                                                   vec_float32 *out2) {

#ifndef ZDNN_CONFIG_NO_NNPA
  vec_float32 work_float_1;
  vec_float32 work_float_2;

#if defined(__MVS__)
  /*
   *  Invoke the VCLFNx
   *  "*     VCLFN(H/L) VReg0,VReg2,mask0,mask2      \n\t"
   */
  // clang-format off
    __asm volatile("      VL   2,%[in_vector]            \n\t" // load VR with 8 DLFs
                   "      DC   XL6'E60200002056'         \n\t" //VCLFNH to VR0
                   "      DC   XL6'E6120000205E'         \n\t" //VCLFNL to VR1
                   "      DS   0H                        \n\t"
                   "      VST  0,%[out_vector_left]      \n\t" //store 1-4 FP32s to output
                   "      VST  1,%[out_vector_right]     \n\t" //store 5-8 FP32s to output
                   : /* Outputs - out_vector */
                   [ out_vector_left ] "=m"(work_float_1),  //(out_vector)
                   [ out_vector_right ] "=m"(work_float_2)  //(out_vector)
                   :                 /* Inputs                        */
                   [ in_vector ] "m" (a),        /* data */
                   [ mask2 ] "i"(2), /* 2 = Internal NNPA format (DLF)  */
                   [ mask0 ] "i"(0)  /* 0 = FP32                        */
                 :  /* "%v0", "%v1", "%v2"   Clobbered */
    );
  // clang-format on
  *out1 = work_float_1;
  *out2 = work_float_2;
#else
  // clang-format off
  __asm volatile(".insn vrr,0xe60000000056,%[out1],%[in_vec],0,2,0,0    \n\t"
                ".insn vrr,0xe6000000005E,%[out2],%[in_vec],0,2,0,0     \n\t"
                : [ out1 ] "=&v"(work_float_1), [ out2 ] "=v"(work_float_2)
                : [ in_vec ] "v"(a));
  // clang-format on

  *out1 = work_float_1;
  *out2 = work_float_2;
#endif
#else
  *out1 = (vec_float32)vec_mergeh(a, zero_vector16);
  *out2 = (vec_float32)vec_mergel(a, zero_vector16);
#endif // ZDNN_CONFIG_NO_NNPA

  return;
} // End aiu_vec_lengthen_to_fp32

//  Common wrapper version of non-inlined aiu_vec_lengthen_to_fp32
void aiu_vec_lengthen_to_fp32(vec_int16 a, vec_float32 *out1,
                              vec_float32 *out2) {
  aiu_vec_lengthen_to_fp32_inline(a, out1, out2);
}

/***********************************************************************
 * aiu_vec_convert_to_fp16
 *
 * Converts 1 vector (8 elements) of 16-bit DLFloat numbers
 * to 1 vector of 16-bit FP16 numbers (8 numbers total)
 *
 * Input: 1 vector of 8 DLFloat data elements to convert
 * Output: 1 vector of 8 FP16 elements
 *
 * Note:  There is also a non-inlined wrapper function as well.
 **********************************************************************/

static vec_int16 inline aiu_vec_convert_to_fp16_inline(vec_int16 a) {
  vec_int16 work_short_1;

#ifndef ZDNN_CONFIG_NO_NNPA
#if defined(__MVS__)
  /*
   *  Invoke the VCFN
   *  "*     VCFN VReg0,VReg2,mask0,mask1      \n\t"
   */
  // clang-format off
    __asm volatile("      VL   2,%[in_vector]            \n\t" // load VR with 8 DLFs
                   "      DC   XL6'E6020000105D'         \n\t" //VCFN to VR0
                   "      DS   0H                        \n\t"
                   "      VST  0,%[out_vector]           \n\t" //store 8 FP16s to output
                   : /* Outputs - out_vector */
                   [ out_vector ] "=m"(work_short_1)
                   :                 /* Inputs                        */
                   [ in_vector ] "m" (a),        /* data */
                   [ mask1 ] "i"(1), /* 1 = FP16  */
                   [ mask0 ] "i"(0)  /* 0 = Internal NNPA format (DLF) */
                 :  /* "%v0", "%v2"   Clobbered */
    );
  // clang-format on
#else
  // clang-format off
  __asm volatile(".insn vrr,0xe6000000005D,%[out_vec],%[in_vec],0,1,0,0  \n\t"
                : [out_vec] "=v"(work_short_1)
                : [in_vec] "v"(a));
  // clang-format on
#endif
#else
  // scaffolding: just copy the input 16-bit elements as is to output
  memcpy(&work_short_1, &a, sizeof(vec_int16));
#endif // #ifndef ZDNN_CONFIG_NO_NNPA
  return work_short_1;
}

//  Common wrapper version of non-inlined aiu_vec_convert_to_fp16
vec_int16 aiu_vec_convert_to_fp16(vec_int16 a) {
  return aiu_vec_convert_to_fp16_inline(a);
}
// End of ASM functions

/***********************************************************************
 *  cnvt_1 functions - These functions invoke the aiu_vec functions to
 *  convert one value.  Highly inefficient.
 **********************************************************************/
/*  cnvt_1_fp32_to_dlf16 */
uint16_t cnvt_1_fp32_to_dlf16(float a) {

  vec_int16 aiu_op_output_dfloat; // vector output from aiu_vec_round...
  /* Copy value to work area, use AIU op routine to convert value from fp32
     to dlfloat in pseudo vector (array), then copy the 1 converted entry
     into the expected data area */

  uint32_float_u tempfp32array[8] = {0};
  memcpy(tempfp32array, &a,
         sizeof(float)); /* used as input to aiu_vec_round conversion */

  aiu_op_output_dfloat = aiu_vec_round_from_fp32(
      *((vec_float32 *)&tempfp32array[0]),
      *((vec_float32 *)&tempfp32array[4])); /* Convert from fp32 to
                                               dlfloat with rounding */
  return (uint16_t)aiu_op_output_dfloat[0]; // return first value from vector
}

float cnvt_1_dlf16_to_fp32(uint16_t a) {

  vec_float32 aiu_op_output_fp32[2]; // vector output from aiu_vec_lengthen

  /* Copy value to work area, use AIU op routine to convert value from
     dlfloat to fp32 in pseudo vector (array), then copy the 1 converted
     entry into the expected data area */
  float_bit16 tempshortarray[8] = {a}; // used as input to aiu_vec_lengthen...
                                       // conversion
  aiu_vec_lengthen_to_fp32(*((vec_int16 *)&tempshortarray[0]),
                           aiu_op_output_fp32,
                           aiu_op_output_fp32 + 1); /* Convert from dlfloat to
                                                       fp32 with lengthening */
  return (*(uint32_float_u *)aiu_op_output_fp32)
      .f; /* return first value from vector output */
}

int main(int argc, char *argv[]) {
  float a = atof(argv[1]);
  uint16_t b = cnvt_1_fp32_to_dlf16(a);
  float c = cnvt_1_dlf16_to_fp32(b);
  printf("c: %f\n", c);

  return 1;
}
