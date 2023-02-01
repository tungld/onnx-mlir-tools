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

int main(int argc, char *argv[]) {
  // float a = atof(argv[1]);
  // uint16_t b = cnvt_1_fp32_to_dlf16(a);
  // float c = cnvt_1_dlf16_to_fp32(b);
  // printf("c: %f\n", c);

  uint16_t a = atoi(argv[1]);

  vec_float32 aiu_op_output_fp32[2];
  float_bit16 tempshortarray[8] = {a};

  vec_int16 aa = *((vec_int16 *)&tempshortarray[0]);
  vec_float32 *out1 = aiu_op_output_fp32;                       
  vec_float32 *out2 = aiu_op_output_fp32 + 1;

  vec_float32 work_float_1;
  vec_float32 work_float_2;
  // clang-format off
  __asm volatile(".insn vrr,0xe60000000056,%[out1],%[in_vec],0,2,0,0    \n\t"
                ".insn vrr,0xe6000000005E,%[out2],%[in_vec],0,2,0,0     \n\t"
                : [ out1 ] "=&v"(work_float_1), [ out2 ] "=v"(work_float_2)
                : [ in_vec ] "v"(a));
  // clang-format on

  *out1 = work_float_1;
  *out2 = work_float_2;

  float c = (*(uint32_float_u *)out1).f; /* return first value from vector output */
  printf("c: %f\n", c);

  return 1;
}
