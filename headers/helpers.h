#ifndef HELPERS_H_
#define HELPERS_H_

__host__ __device__ int mod(int x, int m);

__host__ __device__ int sign(int x);

void populateMatrix(int n, int* matrix);

int validate(int n, int* matrixA, int* matrixB);

#endif // !HELPERS_H_
