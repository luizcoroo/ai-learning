#ifndef TENSOR_H
#define TENSOR_H

#include "config.h"
#include <stdint.h>

#define MAX_DIMS 8

typedef struct {
  int sizes[MAX_DIMS];
  int strides[MAX_DIMS];
  int data_len;
  int rank;
} DataDesc;

typedef struct {
  float_t *data;
  DataDesc desc;
} FTensor;

typedef struct {
  uint_t *data;
  DataDesc desc;
} UTensor;

FTensor ftensor_init(float_t *data, const int *sizes, int rank);
UTensor utensor_init(uint_t *data, const int *sizes, int rank);

void ftensor_log(FTensor a);
void ftensor_neg(FTensor a);
void ftensor_add(FTensor a, FTensor b);
void ftensor_subexp(FTensor a, FTensor b);
void ftensor_mul(FTensor a, FTensor b);
void ftensor_div(FTensor a, FTensor b);
void ftensor_describe(FTensor a);

FTensor ftensor_sum(float_t *a_data, FTensor b, const int *dims, int n);
FTensor ftensor_max(float_t *a_data, FTensor b, const int *dims, int n);
FTensor ftensor_min(float_t *a_data, FTensor b, const int *dims, int n);
void ftensor_assign(FTensor a, float_t x);

FTensor ftensor_onehotenc(float_t *a_data, UTensor b);
float_t ftensor_crossentropysum(UTensor a, FTensor b);

FTensor ftensor_matmul(float_t *a_data, FTensor b, FTensor c);
FTensor utensor_matmul(float_t *a_data, UTensor b, FTensor c);
FTensor utensor_matmuladd(float_t *a_data, UTensor b, FTensor c, FTensor d);

FTensor ftensor_softmax(FTensor a, float_t *tmp_data);

void tensor_align_to(DataDesc *d, int rank);
void tensor_unalign(DataDesc *a);
void tensor_broadcast_to(DataDesc *a, const int *sizes, int rank);
void tensor_unbroadcast(DataDesc *a);
void tensor_squeeze_all(DataDesc *a);

#endif
