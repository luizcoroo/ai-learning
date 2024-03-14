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

UTensor utensor_transpose(UTensor a, int i, int j);

FTensor ftensor_umatmuladd(float_t *a_data, UTensor b, FTensor c, FTensor d);
FTensor ftensor_logsoftmax(FTensor a, float_t *tmp_data);

FTensor ftensor_max(float_t *a_data, FTensor b, const int *dims, int n);
FTensor ftensor_logsumexp(float_t *a_data, FTensor b, const int *dims, int n);
FTensor ftensor_sum(float_t *a_data, FTensor b, const int *dims, int n);

float_t ftensor_crossentropysum(FTensor b, UTensor a);

void ftensor_exp(FTensor a);
void ftensor_onehotdiff(FTensor a, UTensor b);

void ftensor_sub(FTensor a, FTensor b);

void ftensor_describe(FTensor a);
void utensor_describe(UTensor a);

void tensor_align_to(DataDesc *d, int rank);
void tensor_unalign(DataDesc *a);
void tensor_broadcast_to(DataDesc *a, const int *sizes, int rank);
void tensor_unbroadcast(DataDesc *a);
void tensor_squeeze_all(DataDesc *a);

#endif
