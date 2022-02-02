#ifndef KERNEL_H_
#define KERNEL_H_

__global__ void gpuOneThreadEvolution(int n, int* state_a, int* state_b);

__global__ void gpuOneThreadBlockEvolution(int n, int* state_a, int* state_b, int moment_block_width);

__global__ void gpuSharedBlockEvolution(int n, int* state_a, int* state_b, int moment_block_width, int shmem_width, int limit);



#endif

