#include "util.h"

#include <math.h>
#include <stdlib.h>

double rand_normal() {
  double u = ((double)rand() / (RAND_MAX)) * 2 - 1;
  double v = ((double)rand() / (RAND_MAX)) * 2 - 1;
  double r = u * u + v * v;
  if (r == 0 || r > 1)
    return rand_normal();
  double c = sqrt(-2 * log(r) / r);
  return u * c;
}
