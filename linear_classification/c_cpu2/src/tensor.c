#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "tensor.h"

int incr_pos_at_dim(int *pos, int *idxs, const DataDesc *desc, int dim);

FTensor ftensor_init(float_t *data, const int *sizes, int rank) {
  FTensor view = {
      .data = data,
      .desc.rank = rank,
  };
  memcpy(view.desc.sizes, sizes, sizeof(int) * rank);

  int stride = 1;
  for (int i = rank; i > 0; i--) {
    view.desc.strides[i - 1] = stride;
    stride *= view.desc.sizes[i - 1];
  }

  view.desc.data_len = stride;
  return view;
}

UTensor utensor_init(uint_t *data, const int *sizes, int rank) {
  UTensor view = {
      .data = data,
      .desc.rank = rank,
  };
  memcpy(view.desc.sizes, sizes, sizeof(int) * rank);

  int stride = 1;
  for (int i = rank; i > 0; i--) {
    view.desc.strides[i - 1] = stride;
    stride *= view.desc.sizes[i - 1];
  }

  view.desc.data_len = stride;
  return view;
}

UTensor utensor_transpose(UTensor a, int i, int j) {
  UTensor at = {
      .data = a.data,
      .desc.data_len = a.desc.data_len,
      .desc.rank = a.desc.rank,
  };

  memcpy(at.desc.sizes, a.desc.sizes, sizeof(int) * a.desc.rank);
  memcpy(at.desc.strides, a.desc.strides, sizeof(int) * a.desc.rank);

  int tmp = at.desc.sizes[i];
  at.desc.sizes[i] = at.desc.sizes[j];
  at.desc.sizes[j] = tmp;

  tmp = at.desc.strides[i];
  at.desc.strides[i] = at.desc.strides[j];
  at.desc.strides[j] = tmp;

  return at;
}

FTensor ftensor_umatmuladd(float_t *a_data, UTensor b, FTensor c, FTensor d) {
  int dim = b.desc.rank - 2;

  tensor_align_to(&c.desc, b.desc.rank);
  tensor_broadcast_to(&c.desc, b.desc.sizes, dim);

  int N = b.desc.sizes[dim];
  int K = c.desc.sizes[dim];
  int M = c.desc.sizes[dim + 1];
  assert(b.desc.sizes[dim + 1] == K);

  int a_sizes[MAX_DIMS];
  memcpy(a_sizes, b.desc.sizes, sizeof(int) * dim);
  a_sizes[dim] = N;
  a_sizes[dim + 1] = M;

  FTensor a = ftensor_init(a_data, a_sizes, b.desc.rank);
  tensor_align_to(&d.desc, a.desc.rank);
  tensor_broadcast_to(&d.desc, a.desc.sizes, a.desc.rank);

  int pos_a = 0, idxs_a[MAX_DIMS] = {0};
  int pos_b = 0, idxs_b[MAX_DIMS] = {0};
  int pos_c = 0, idxs_c[MAX_DIMS] = {0};
  int pos_d = 0, idxs_d[MAX_DIMS] = {0};

  do {
    for (int i = 0; i < N; i++) {
      int a_i = i * a.desc.strides[dim];
      int b_i = i * b.desc.strides[dim];
      int d_i = i * d.desc.strides[dim];
      for (int j = 0; j < M; j++) {
        int a_j = j * a.desc.strides[dim + 1];
        int c_j = j * c.desc.strides[dim + 1];
        int d_j = j * d.desc.strides[dim + 1];

        int a_total = pos_a + a_i + a_j;
        int d_total = pos_d + d_i + d_j;
        a.data[a_total] = d.data[d_total];

        for (int k = 0; k < K; k++) {
          int b_k = k * b.desc.strides[dim + 1];
          int c_k = k * c.desc.strides[dim];

          int b_total = pos_b + b_i + b_k;
          int c_total = pos_c + c_k + c_j;
          a.data[a_total] += b.data[b_total] * c.data[c_total];
        }
      }
    }
  } while (incr_pos_at_dim(&pos_a, idxs_a, &a.desc, dim) &&
           incr_pos_at_dim(&pos_b, idxs_b, &b.desc, dim) &&
           incr_pos_at_dim(&pos_c, idxs_c, &c.desc, dim) &&
           incr_pos_at_dim(&pos_d, idxs_d, &d.desc, dim));

  return a;
}

FTensor ftensor_logsoftmax(FTensor a, float_t *tmp_data) {
  ftensor_sub(a, ftensor_max(tmp_data, a, (int[]){a.desc.rank - 1}, 1));
  ftensor_sub(a, ftensor_logsumexp(tmp_data, a, (int[]){a.desc.rank - 1}, 1));
  return a;
}

FTensor ftensor_max(float_t *a_data, FTensor b, const int *dims, int n) {
  int a_sizes[MAX_DIMS];
  memcpy(a_sizes, b.desc.sizes, sizeof(int) * b.desc.rank);
  for (int i = 0; i < n; i++)
    a_sizes[dims[i]] = 1;

  FTensor a = ftensor_init(a_data, a_sizes, b.desc.rank);
  tensor_broadcast_to(&a.desc, b.desc.sizes, b.desc.rank);
  for (int i = 0; i < a.desc.data_len; i++)
    a.data[i] = -FLT_MAX;

  int pos_a = 0, idxs_a[MAX_DIMS] = {0};
  int pos_b = 0, idxs_b[MAX_DIMS] = {0};

  do {
    a.data[pos_a] = fmax(a.data[pos_a], b.data[pos_b]);
  } while (incr_pos_at_dim(&pos_a, idxs_a, &a.desc, a.desc.rank) &&
           incr_pos_at_dim(&pos_b, idxs_b, &b.desc, b.desc.rank));

  tensor_unbroadcast(&a.desc);

  return a;
}

FTensor ftensor_logsumexp(float_t *a_data, FTensor b, const int *dims, int n) {
  int a_sizes[MAX_DIMS];
  memcpy(a_sizes, b.desc.sizes, sizeof(int) * b.desc.rank);
  for (int i = 0; i < n; i++)
    a_sizes[dims[i]] = 1;

  FTensor a = ftensor_init(a_data, a_sizes, b.desc.rank);
  tensor_broadcast_to(&a.desc, b.desc.sizes, b.desc.rank);
  for (int i = 0; i < a.desc.data_len; i++)
    a.data[i] = 0.0;

  int pos_a = 0, idxs_a[MAX_DIMS] = {0};
  int pos_b = 0, idxs_b[MAX_DIMS] = {0};

  do {
    a.data[pos_a] += exp(b.data[pos_b]);
  } while (incr_pos_at_dim(&pos_a, idxs_a, &a.desc, a.desc.rank) &&
           incr_pos_at_dim(&pos_b, idxs_b, &b.desc, b.desc.rank));

  tensor_unbroadcast(&a.desc);

  for (int i = 0; i < a.desc.data_len; i++)
    a.data[i] = log(a.data[i]);

  return a;
}

FTensor ftensor_sum(float_t *a_data, FTensor b, const int *dims, int n) {
  int a_sizes[MAX_DIMS];
  memcpy(a_sizes, b.desc.sizes, sizeof(int) * b.desc.rank);
  for (int i = 0; i < n; i++)
    a_sizes[dims[i]] = 1;

  FTensor a = ftensor_init(a_data, a_sizes, b.desc.rank);
  tensor_broadcast_to(&a.desc, b.desc.sizes, b.desc.rank);
  for (int i = 0; i < a.desc.data_len; i++)
    a.data[i] = 0.0;

  int pos_a = 0, idxs_a[MAX_DIMS] = {0};
  int pos_b = 0, idxs_b[MAX_DIMS] = {0};

  do {
    a.data[pos_a] += b.data[pos_b];
  } while (incr_pos_at_dim(&pos_a, idxs_a, &a.desc, a.desc.rank) &&
           incr_pos_at_dim(&pos_b, idxs_b, &b.desc, b.desc.rank));

  tensor_unbroadcast(&a.desc);

  return a;
}

float_t ftensor_crossentropysum(FTensor a, UTensor b) {
  int dim = a.desc.rank - 1;
  tensor_align_to(&b.desc, a.desc.rank);
  tensor_broadcast_to(&b.desc, a.desc.sizes, dim);

  float_t sum = 0;
  int pos_a = 0, idxs_a[MAX_DIMS] = {0};
  int pos_b = 0, idxs_b[MAX_DIMS] = {0};

  do {
    sum += a.data[pos_a + b.data[pos_b]];
  } while (incr_pos_at_dim(&pos_a, idxs_a, &a.desc, dim) &&
           incr_pos_at_dim(&pos_b, idxs_b, &b.desc, dim));

  return sum;
}

void ftensor_exp(FTensor a) {
  for (int i = 0; i < a.desc.data_len; i++)
    a.data[i] = exp(a.data[i]);
}

void ftensor_onehotdiff(FTensor a, UTensor b) {
  int dim = a.desc.rank;
  tensor_align_to(&b.desc, a.desc.rank);
  tensor_broadcast_to(&b.desc, a.desc.sizes, dim);

  int pos_a = 0, idxs_a[MAX_DIMS] = {0};
  int pos_b = 0, idxs_b[MAX_DIMS] = {0};

  do {
    a.data[pos_a + b.data[pos_b]] -= 1.0;
  } while (incr_pos_at_dim(&pos_a, idxs_a, &a.desc, dim - 1) &&
           incr_pos_at_dim(&pos_b, idxs_b, &b.desc, dim - 1));
}

void ftensor_sub(FTensor a, FTensor b) {
  tensor_align_to(&b.desc, a.desc.rank);
  tensor_broadcast_to(&b.desc, a.desc.sizes, a.desc.rank);

  int pos_a = 0, idxs_a[MAX_DIMS] = {0};
  int pos_b = 0, idxs_b[MAX_DIMS] = {0};

  do {
    a.data[pos_a] -= b.data[pos_b];
  } while (incr_pos_at_dim(&pos_a, idxs_a, &a.desc, a.desc.rank) &&
           incr_pos_at_dim(&pos_b, idxs_b, &b.desc, b.desc.rank));
}

void ftensor_describe(FTensor a) {
  printf("sizes: %d", a.desc.sizes[0]);
  for (int i = 1; i < a.desc.rank; i++)
    printf(", %d", a.desc.sizes[i]);
  printf("\nstrides: %d", a.desc.strides[0]);
  for (int i = 1; i < a.desc.rank; i++)
    printf(", %d", a.desc.strides[i]);
  printf("\ndata_len: %d", a.desc.data_len);
  printf("\ndata: \n");
  if (a.desc.rank == 0) {
    printf("%4e\n", (double)a.data[0]);
    return;
  }
  printf("[");
  for (int i = 1; i < a.desc.rank; i++)
    printf("[");
  int pos = 0, idxs[MAX_DIMS] = {0}, dim;
  do {
    if (idxs[a.desc.rank - 1] > 0)
      printf(" ");
    printf("%.4e", (double)a.data[pos]);
    dim = incr_pos_at_dim(&pos, idxs, &a.desc, a.desc.rank);
    for (int i = dim; i < a.desc.rank; i++)
      printf("]");
    if (dim > 0 && dim < a.desc.rank) {
      printf("\n");
      int i = 0;
      for (; i < dim; i++)
        printf(" ");
      for (; i < a.desc.rank; i++)
        printf("[");
    }
  } while (dim > 0);
  printf("\n");
}

void utensor_describe(UTensor a) {
  printf("sizes: %d", a.desc.sizes[0]);
  for (int i = 1; i < a.desc.rank; i++)
    printf(", %d", a.desc.sizes[i]);
  printf("\nstrides: %d", a.desc.strides[0]);
  for (int i = 1; i < a.desc.rank; i++)
    printf(", %d", a.desc.strides[i]);
  printf("\ndata_len: %d", a.desc.data_len);
  printf("\ndata: \n");
  if (a.desc.rank == 0) {
    printf("%3ld\n", (long)a.data[0]);
    return;
  }
  printf("[");
  for (int i = 1; i < a.desc.rank; i++)
    printf("[");
  int pos = 0, idxs[MAX_DIMS] = {0}, dim;
  do {
    if (idxs[a.desc.rank - 1] > 0)
      printf(" ");
    printf("%3ld", (long)a.data[pos]);
    dim = incr_pos_at_dim(&pos, idxs, &a.desc, a.desc.rank);
    for (int i = dim; i < a.desc.rank; i++)
      printf("]");
    if (dim > 0 && dim < a.desc.rank) {
      printf("\n");
      int i = 0;
      for (; i < dim; i++)
        printf(" ");
      for (; i < a.desc.rank; i++)
        printf("[");
    }
  } while (dim > 0);
  printf("\n");
}

void tensor_align_to(DataDesc *a, int rank) {
  int rank_diff = rank - a->rank;
  for (int i = a->rank; i > 0; --i) {
    a->sizes[i - 1 + rank_diff] = a->sizes[i - 1];
    a->strides[i - 1 + rank_diff] = a->strides[i - 1];
  }

  for (int i = 0; i < rank_diff; i++) {
    a->sizes[i] = 1;
    a->strides[i] = a->data_len;
  }

  a->rank = rank;
}

void tensor_unalign(DataDesc *a) {
  int rank_diff = 0;
  while (rank_diff < a->rank && a->sizes[rank_diff] == 1)
    rank_diff += 1;

  for (int i = rank_diff; i < a->rank; ++i) {
    a->sizes[i - rank_diff] = a->sizes[i];
    a->strides[i - rank_diff] = a->strides[i];
  }

  a->rank -= rank_diff;
}

void tensor_squeeze_all(DataDesc *a) {
  int i = 0, j = 0;
  for (; j < a->rank; i++, j++) {
    if (a->sizes[j] == 1)
      while (j < a->rank && a->sizes[j] == 1)
        j++;

    a->sizes[i] = a->sizes[j];
    a->strides[i] = a->strides[j];
  }
  a->rank -= j - i;
}

void tensor_unbroadcast(DataDesc *a) {
  for (int i = 0; i < a->rank; ++i) {
    if (a->strides[i] == 0) {
      a->strides[i] = 1;
      a->sizes[i] = 1;
    }
  }
}

void tensor_broadcast_to(DataDesc *a, const int *sizes, int rank) {
  for (int i = 0; i < rank; i++) {
    if (a->sizes[i] == 1 && sizes[i] > 1) {
      a->sizes[i] = sizes[i];
      a->strides[i] = 0;
    }
  }
}

int incr_pos_at_dim(int *pos, int *idxs, const DataDesc *desc, int dim) {
  while (dim > 0) {
    *pos += desc->strides[dim - 1];
    idxs[dim - 1] += 1;
    if (idxs[dim - 1] != desc->sizes[dim - 1])
      break;

    *pos -= idxs[dim - 1] * desc->strides[dim - 1];
    idxs[dim - 1] = 0;
    dim -= 1;
  }

  return dim;
}
