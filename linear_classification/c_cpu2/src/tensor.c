#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "tensor.h"

int incr_pos_idxs_at_dim(int *pos, int *idxs, int dim, TensorView view);
int incr_pos_idxs_at_dim_int(int *pos, int *idxs, int dim, TensorViewInt view);

TensorView tensor_view(float *data, const int *sizes, int rank) {
  TensorView view = {
      .data = data,
      .rank = rank,
  };

  memcpy(view.sizes, sizes, sizeof(int) * rank);

  int stride = 1;
  for (int i = rank; i > 0; i--) {
    view.strides[i - 1] = stride;
    stride *= view.sizes[i - 1];
  }

  view.data_len = stride;

  return view;
}

TensorViewInt tensorint_view(unsigned char *data, const int *sizes, int rank) {
  TensorViewInt view = {
      .data = data,
      .rank = rank,
  };

  memcpy(view.sizes, sizes, sizeof(int) * rank);

  int stride = 1;
  for (int i = rank; i > 0; i--) {
    view.strides[i - 1] = stride;
    stride *= view.sizes[i - 1];
  }

  view.data_len = stride;

  return view;
}

TensorView tensor_at(TensorView view, const int *indexes, int rank) {
  int idx = 0;
  for (int i = 0; i < rank; i++)
    idx += indexes[i] * view.strides[i];

  TensorView new_view = {
      .data = view.data + idx,
      .data_len = view.strides[rank - 1],
      .rank = view.rank - rank,
  };

  memcpy(new_view.sizes, view.sizes + rank, sizeof(int) * new_view.rank);
  memcpy(new_view.strides, view.strides + rank, sizeof(int) * new_view.rank);
  return new_view;
}

void tensor_align_to(TensorView *a, int rank) {
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

void tensor_unalign(TensorView *a) {
  int rank_diff = 0;
  while (rank_diff < a->rank && a->sizes[rank_diff] == 1)
    rank_diff += 1;

  for (int i = rank_diff; i < a->rank; ++i) {
    a->sizes[i - rank_diff] = a->sizes[i];
    a->strides[i - rank_diff] = a->strides[i];
  }

  a->rank -= rank_diff;
}

void tensor_squeeze_all(TensorView *a) {
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

void tensor_unbroadcast(TensorView *a) {
  for (int i = 0; i < a->rank; ++i) {
    if (a->strides[i] == 0) {
      a->strides[i] = 1;
      a->sizes[i] = 1;
    }
  }
}

void tensor_broadcast_to(TensorView *a, const int *sizes, int rank) {
  for (int i = 0; i < rank; i++) {
    if (a->sizes[i] == 1 && sizes[i] > 1) {
      a->sizes[i] = sizes[i];
      a->strides[i] = 0;
    }
  }
}

void tensor_exp(TensorView a) {
  for (int i = 0; i < a.data_len; i++)
    a.data[i] = expf(a.data[i]);
}

void tensor_log(TensorView a) {
  for (int i = 0; i < a.data_len; i++)
    a.data[i] = logf(a.data[i]);
}

void tensor_neg(TensorView a) {
  for (int i = 0; i < a.data_len; i++)
    a.data[i] = -a.data[i];
}

void tensor_add(TensorView a, TensorView b) {
  tensor_align_to(&b, a.rank);
  tensor_broadcast_to(&b, a.sizes, a.rank);

  int pos_a = 0;
  int pos_b = 0;
  int idxs_a[MAX_DIMS] = {0};
  int idxs_b[MAX_DIMS] = {0};

  do {
    a.data[pos_a] += b.data[pos_b];
  } while (incr_pos_idxs_at_dim(&pos_a, idxs_a, a.rank, a) &&
           incr_pos_idxs_at_dim(&pos_b, idxs_b, b.rank, b));
}

void tensor_sub(TensorView a, TensorView b) {
  tensor_align_to(&b, a.rank);
  tensor_broadcast_to(&b, a.sizes, a.rank);

  int pos_a = 0;
  int pos_b = 0;
  int idxs_a[MAX_DIMS] = {0};
  int idxs_b[MAX_DIMS] = {0};

  do {
    a.data[pos_a] -= b.data[pos_b];
  } while (incr_pos_idxs_at_dim(&pos_a, idxs_a, a.rank, a) &&
           incr_pos_idxs_at_dim(&pos_b, idxs_b, b.rank, b));
}

void tensor_mul(TensorView a, TensorView b) {
  tensor_align_to(&b, a.rank);
  tensor_broadcast_to(&b, a.sizes, a.rank);

  int pos_a = 0;
  int pos_b = 0;
  int idxs_a[MAX_DIMS] = {0};
  int idxs_b[MAX_DIMS] = {0};

  do {
    a.data[pos_a] *= b.data[pos_b];
  } while (incr_pos_idxs_at_dim(&pos_a, idxs_a, a.rank, a) &&
           incr_pos_idxs_at_dim(&pos_b, idxs_b, b.rank, b));
}


void tensor_div(TensorView a, TensorView b) {
  tensor_align_to(&b, a.rank);
  tensor_broadcast_to(&b, a.sizes, a.rank);

  int pos_a = 0;
  int pos_b = 0;
  int idxs_a[MAX_DIMS] = {0};
  int idxs_b[MAX_DIMS] = {0};

  do {
    a.data[pos_a] /= b.data[pos_b];
  } while (incr_pos_idxs_at_dim(&pos_a, idxs_a, a.rank, a) &&
           incr_pos_idxs_at_dim(&pos_b, idxs_b, b.rank, b));
}

TensorView tensor_sum(float *a_data, TensorView b, const int *dims, int n) {
  int a_sizes[MAX_DIMS];
  memcpy(a_sizes, b.sizes, sizeof(int) * b.rank);
  for (int i = 0; i < n; i++)
    a_sizes[dims[i]] = 1;

  TensorView a = tensor_view(a_data, a_sizes, b.rank);
  tensor_broadcast_to(&a, b.sizes, b.rank);

  int pos_a = 0;
  int pos_b = 0;
  int idxs_a[MAX_DIMS] = {0};
  int idxs_b[MAX_DIMS] = {0};

  for (int i = 0; i < a.data_len; i++)
    a.data[i] = 0;

  do {
    a.data[pos_a] += b.data[pos_b];
  } while (incr_pos_idxs_at_dim(&pos_a, idxs_a, a.rank, a) &&
           incr_pos_idxs_at_dim(&pos_b, idxs_b, b.rank, b));

  tensor_unbroadcast(&a);

  return a;
}

TensorView tensor_max(float *a_data, TensorView b, const int *dims, int n) {
  int a_sizes[MAX_DIMS];
  memcpy(a_sizes, b.sizes, sizeof(int) * b.rank);
  for (int i = 0; i < n; i++)
    a_sizes[dims[i]] = 1;

  TensorView a = tensor_view(a_data, a_sizes, b.rank);
  tensor_broadcast_to(&a, b.sizes, b.rank);

  int pos_a = 0;
  int pos_b = 0;
  int idxs_a[MAX_DIMS] = {0};
  int idxs_b[MAX_DIMS] = {0};

  for (int i = 0; i < a.data_len; i++)
    a.data[i] = -FLT_MAX;

  do {
    a.data[pos_a] = fmax(a.data[pos_a], b.data[pos_b]);
  } while (incr_pos_idxs_at_dim(&pos_a, idxs_a, a.rank, a) &&
           incr_pos_idxs_at_dim(&pos_b, idxs_b, b.rank, b));

  tensor_unbroadcast(&a);

  return a;
}

TensorView tensor_matmul(float *a_data, TensorView b, TensorView c) {
  int dim = b.rank - 2;
  tensor_align_to(&c, b.rank);
  tensor_broadcast_to(&c, b.sizes, dim);

  int N = b.sizes[dim];
  int K = c.sizes[dim];
  int M = c.sizes[dim + 1];
  assert(b.sizes[dim + 1] == K);

  int a_sizes[MAX_DIMS];
  memcpy(a_sizes, b.sizes, sizeof(int) * dim);
  a_sizes[dim] = N;
  a_sizes[dim + 1] = M;
  TensorView a = tensor_view(a_data, a_sizes, b.rank);

  int pos_a = 0;
  int pos_b = 0;
  int pos_c = 0;
  int idxs_a[MAX_DIMS] = {0};
  int idxs_b[MAX_DIMS] = {0};
  int idxs_c[MAX_DIMS] = {0};

  memset(a.data, 0, sizeof(float) * a.data_len);

  do {
    for (int i = 0; i < N; i++)
      for (int j = 0; j < M; j++)
        for (int k = 0; k < K; k++)
          a.data[pos_a + i * a.strides[dim] + j * a.strides[dim + 1]] +=
              b.data[pos_b + i * b.strides[dim] + k * b.strides[dim + 1]] *
              c.data[pos_c + k * c.strides[dim] + j * c.strides[dim + 1]];
  } while (incr_pos_idxs_at_dim(&pos_a, idxs_a, dim, a) &&
           incr_pos_idxs_at_dim(&pos_b, idxs_b, dim, b) &&
           incr_pos_idxs_at_dim(&pos_c, idxs_c, dim, c));

  return a;
}

TensorView tensorint_matmul(float *a_data, TensorViewInt b, TensorView c) {
  int dim = b.rank - 2;
  tensor_align_to(&c, b.rank);
  tensor_broadcast_to(&c, b.sizes, dim);

  int N = b.sizes[dim];
  int K = c.sizes[dim];
  int M = c.sizes[dim + 1];
  assert(b.sizes[dim + 1] == K);

  int a_sizes[MAX_DIMS];
  memcpy(a_sizes, b.sizes, sizeof(int) * dim);
  a_sizes[dim] = N;
  a_sizes[dim + 1] = M;
  TensorView a = tensor_view(a_data, a_sizes, b.rank);

  int pos_a = 0;
  int pos_b = 0;
  int pos_c = 0;
  int idxs_a[MAX_DIMS] = {0};
  int idxs_b[MAX_DIMS] = {0};
  int idxs_c[MAX_DIMS] = {0};

  memset(a.data, 0, sizeof(float) * a.data_len);

  do {
    for (int i = 0; i < N; i++)
      for (int j = 0; j < M; j++)
        for (int k = 0; k < K; k++)
          a.data[pos_a + i * a.strides[dim] + j * a.strides[dim + 1]] +=
              b.data[pos_b + i * b.strides[dim] + k * b.strides[dim + 1]] *
              c.data[pos_c + k * c.strides[dim] + j * c.strides[dim + 1]];
  } while (incr_pos_idxs_at_dim(&pos_a, idxs_a, dim, a) &&
           incr_pos_idxs_at_dim_int(&pos_b, idxs_b, dim, b) &&
           incr_pos_idxs_at_dim(&pos_c, idxs_c, dim, c));

  return a;
}

void tensor_describe(TensorView view) {
  printf("sizes: %d", view.sizes[0]);
  for (int i = 1; i < view.rank; i++)
    printf(", %d", view.sizes[i]);

  printf("\nstrides: %d", view.strides[0]);
  for (int i = 1; i < view.rank; i++)
    printf(", %d", view.strides[i]);

  printf("\ndata_len: %d", view.data_len);

  printf("\ndata: \n");
  if (view.rank == 0) {
    printf("%4.1f\n", view.data[0]);
    return;
  }

  printf("[");
  for (int i = 1; i < view.rank; i++)
    printf("[");

  int idxs[MAX_DIMS] = {0};
  int pos = 0;
  int dim;
  do {
    if (idxs[view.rank - 1] > 0)
      printf(" ");
    printf("%f", view.data[pos]);

    dim = incr_pos_idxs_at_dim(&pos, idxs, view.rank, view);
    for (int i = dim; i < view.rank; i++)
      printf("]");

    if (dim > 0 && dim < view.rank) {
      printf("\n");
      int i = 0;
      for (; i < dim; i++)
        printf(" ");
      for (; i < view.rank; i++)
        printf("[");
    }
  } while (dim > 0);

  printf("\n");
}

int incr_pos_idxs_at_dim(int *pos, int *idxs, int dim, TensorView view) {
  while (dim > 0) {
    *pos += view.strides[dim - 1];
    idxs[dim - 1] += 1;
    if (idxs[dim - 1] != view.sizes[dim - 1])
      break;

    *pos -= idxs[dim - 1] * view.strides[dim - 1];
    idxs[dim - 1] = 0;
    dim -= 1;
  }

  return dim;
}

int incr_pos_idxs_at_dim_int(int *pos, int *idxs, int dim, TensorViewInt view) {
  while (dim > 0) {
    *pos += view.strides[dim - 1];
    idxs[dim - 1] += 1;
    if (idxs[dim - 1] != view.sizes[dim - 1])
      break;

    *pos -= idxs[dim - 1] * view.strides[dim - 1];
    idxs[dim - 1] = 0;
    dim -= 1;
  }

  return dim;
}

// int check_is_alignable(TensorView a, TensorView b) {
//   int rank_diff = b.rank - a.rank;
//   for (int i = a.rank; i > 0; i--) {
//     int a_size = a.sizes[i - 1];
//     int b_size = b.sizes[i - 1 + rank_diff];
//     if (a_size != 1 && b_size != 1 && a_size != b_size)
//       return 0;
//   }
//
//   return 1;
// }
//
// int check_is_expandable(TensorView a, TensorView b) {
//   if (a.rank != b.rank)
//     return 0;
//
//   for (int i = 0; i < b.rank; i++)
//     if (a.sizes[i] != 1 && b.sizes[i] != 1 && a.sizes[i] != b.sizes[i])
//       return 0;
//
//   return 1;
// }
