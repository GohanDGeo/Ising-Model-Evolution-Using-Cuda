#ifndef ISINGEVOLUTION_H_
#define ISINGEVOLUTION_H_

float gpuSharedBlockIsing(int* state_a, int n, int k, int bytes, int moment_block_width, int thread_width);

float gpuOneThreadBlockIsing(int* state_a, int n, int k, int bytes, int moment_block_width, int thread_width);

float gpuOneThreadIsing(int* state_a, int n, int k, int bytes, int thread_width);

double sequentialIsingEvolution(int* state_a, int n, int k, int bytes);

#endif