
#include <math.h>
#include <stdlib.h>

#include "util.h"

float randnf() {
  float u = ((double)rand() / (RAND_MAX)) * 2 - 1;
  float v = ((double)rand() / (RAND_MAX)) * 2 - 1;
  float r = u * u + v * v;
  if (r == 0 || r > 1)
    return randnf();
  float c = sqrt(-2 * log(r) / r);
  return u * c;
}

void shuffleiarr(int *arr, int n) {
  for (int i = 0; i < n - 1; i++) {
    size_t k = i + rand() / (RAND_MAX / (n - i) + 1);

    int tmp = arr[k];
    arr[k] = arr[i];
    arr[i] = tmp;
  }
}
