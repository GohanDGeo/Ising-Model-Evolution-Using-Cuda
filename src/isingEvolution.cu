//Author: Giorgos Koutroumpis,
//ECE AUTH, Parallel and Distributed Systems 2022,
//AEM: 9668,
//Contact: geokonkou@ece.auth.gr

//Contains the driver function for each implementation of the project.

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "../headers/isingEvolution.h"
#include "../headers/kernel.h"
#include "../headers/helpers.h"

//Executes v3 of the project. It simulates the ising evolution of a nxn matrix, for k steps.
//Each thread calculates a bxb block of moments. Shared memory is used.
//@args:
//state_a ->  the input nxn matrix
//n -> width of the matrix state_a
//k -> number of steps to run ising evolution for
//bytes -> the size of the matrix state_a in bytes
//moment_block_width -> the width of the block of moments each thread calculates (bxb)
//thread_width -> the width of the thread block
float gpuSharedBlockIsing(int* state_a, int n, int k, int bytes, int moment_block_width, int thread_width)
{
	//Initialize cuda events to calculate time of v3
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//The width of the shared memory matrix. In extreme cases, were the width would be greater
	//than the matrix's width + 2, set width to matrix width + 2.
	//Since this would only happen for small n, this is trivial. But better safe...
	int shmem_width = min(thread_width * moment_block_width + 2, n + 2);

	//The size of the shared memory matrix (width x width), in bytes
	int shmem_size = (shmem_width * shmem_width) * sizeof(int);

	//The width of the grid of thread blocks. Padding added so there are always enough threads
	int blocks = (n + thread_width * moment_block_width - 1) / (thread_width * moment_block_width);

	//The max number of thread blocks needed horizontally/vertically
	int limit = (n + moment_block_width - 1) / moment_block_width;
	
	//Setup kernel launch parameters
	dim3 THREADS(thread_width, thread_width);
	dim3 BLOCKS(blocks, blocks);

	//Copy state_a to the device
	int* state_a_d;
	cudaMalloc(&state_a_d, bytes);
	cudaMemcpy(state_a_d, state_a, bytes, cudaMemcpyHostToDevice);

	//Create a matrix to hold the next state of the matrix
	int* state_b_d;
	cudaMalloc(&state_b_d, bytes);

	//A temporary pointer used for swapping
	int* temp_ptr;

	//Point to start counting runtime
	cudaEventRecord(start);

	//Iterate for k steps
	for (int iter = 0; iter < k; iter++)
	{
		//Run the kernel for the ising evolution using shared memory
		gpuSharedBlockEvolution << <BLOCKS, THREADS, shmem_size >> > (n, state_a_d, state_b_d, moment_block_width, shmem_width, limit);
		//Wait for all blocks to finish
		cudaDeviceSynchronize();

		//Swap the pointers for state a and state b
		temp_ptr = state_a_d;
		state_a_d = state_b_d;
		state_b_d = temp_ptr;
	}

	//Stop counting runtime
	cudaEventRecord(stop);

	//Copy state_a back to the CPU
	cudaMemcpy(state_a, state_a_d, bytes, cudaMemcpyDeviceToHost);

	//Free the device matrices
	cudaFree(state_a_d);
	cudaFree(state_b_d);

	//Calculate runtime
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	//printf("\n\033[1;33mv3\033[0m Implementation Runtime: %fms\n", milliseconds);
	printf("\nv3 Implementation Runtime: %fms\n", milliseconds);

	return milliseconds;
}


//Executes v2 of the project. It simulates the ising evolution of a nxn matrix, for k steps.
//Each thread calculates a bxb block of moments
//@args:
//state_a ->  the input nxn matrix
//n -> width of the matrix state_a
//k -> number of steps to run ising evolution for
//bytes -> the size of the matrix state_a in bytes
//moment_block_width -> the width of the block of moments each thread calculates (bxb)
//thread_width -> the width of the thread block
float gpuOneThreadBlockIsing(int* state_a, int n, int k, int bytes, int moment_block_width, int thread_width)
{
	//Initialize cuda events to calculate time of v2
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//The width of the grid of thread blocks. Padding added so there are always enough threads
	int blocks = (n + thread_width * moment_block_width - 1) / (thread_width * moment_block_width);

	//Setup kernel launch parameters
	dim3 THREADS(thread_width, thread_width);
	dim3 BLOCKS(blocks, blocks);

	//Copy state_a to the device
	int* state_a_d;
	cudaMalloc(&state_a_d, bytes);
	cudaMemcpy(state_a_d, state_a, bytes, cudaMemcpyHostToDevice);

	//Create a matrix to hold the next state of the matrix
	int* state_b_d;
	cudaMalloc(&state_b_d, bytes);

	//A temporary pointer used for swapping
	int* temp_ptr;

	//Point to start counting runtime
	cudaEventRecord(start);

	//Iterate for k steps
	for (int iter = 0; iter < k; iter++)
	{
		//Run the kernel for the ising evolution
		gpuOneThreadBlockEvolution <<<BLOCKS, THREADS >>>(n, state_a_d, state_b_d, moment_block_width);
		//Wait for all blocks to finish
		cudaDeviceSynchronize();

		//Swap the pointers for state a and state b
		temp_ptr = state_a_d;
		state_a_d = state_b_d;
		state_b_d = temp_ptr;
	}

	//Stop counting runtime
	cudaEventRecord(stop);

	//Copy state_a back to the CPU
	cudaMemcpy(state_a, state_a_d, bytes, cudaMemcpyDeviceToHost);

	//Free the device matrices
	cudaFree(state_a_d);
	cudaFree(state_b_d);

	//Calculate runtime
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	//printf("\n\033[1;35mv2\033[0m Implementation Runtime: %fms\n", milliseconds);
	printf("\nv2 Implementation Runtime: %fms\n", milliseconds);

	return milliseconds;
}

//Executes v1 of the project. It simulates the ising evolution of a nxn matrix, for k steps.
//Each thread calculates the spin of one moment
//@args:
//state_a ->  the input nxn matrix
//n -> width of the matrix state_a
//k -> number of steps to run ising evolution for
//bytes -> the size of the matrix state_a in bytes
//thread_width -> the width of the thread block
float gpuOneThreadIsing(int* state_a, int n, int k, int bytes, int thread_width)
{
	//Initialize cuda events to calculate time of v1
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//The width of the grid of thread blocks. Padding added so there are always enough threads
	int blocks = (n + thread_width - 1) / thread_width;

	//Setup kernel launch parameters
	dim3 THREADS(thread_width, thread_width);
	dim3 BLOCKS(blocks, blocks);

	//Copy state_a to the device
	int* state_a_d;
	cudaMalloc(&state_a_d, bytes);
	cudaMemcpy(state_a_d, state_a, bytes, cudaMemcpyHostToDevice);

	//Create a matrix to hold the next state of the matrix
	int* state_b_d;
	cudaMalloc(&state_b_d, bytes);

	//A temporary pointer used for swapping
	int* temp_ptr;

	//Point to start counting runtime
	cudaEventRecord(start);

	//Iterate for k steps
	for (int iter = 0; iter < k; iter++)
	{
		//Run the kernel for the ising evolution
		gpuOneThreadEvolution << <BLOCKS, THREADS >> > (n, state_a_d, state_b_d);
		//Wait for all blocks to finish
		cudaDeviceSynchronize();

		//Swap the pointers for state a and state b
		temp_ptr = state_a_d;
		state_a_d = state_b_d;
		state_b_d = temp_ptr;
	}

	//Stop counting runtime
	cudaEventRecord(stop);

	//Copy state_a back to the CPU
	cudaMemcpy(state_a, state_a_d, bytes, cudaMemcpyDeviceToHost);

	//Free the device matrices
	cudaFree(state_a_d);
	cudaFree(state_b_d);

	//Calculate runtime
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	//printf("\n\033[1;36mv1\033[0m Implementation Runtime: %fms\n", milliseconds);
	printf("\nv1 Implementation Runtime: %fms\n", milliseconds);

	return milliseconds;
}


//Executes v0 of the project. It simulates the ising evolution of a nxn matrix, for k steps.
//Each moment's spin is calculated sequentially
//@args:
//state_a ->  the input nxn matrix
//n -> width of the matrix state_a
//k -> number of steps to run ising evolution for
//bytes -> the size of the matrix state_a in bytes
double sequentialIsingEvolution(int* state_a, int n, int k, int bytes)
{
	//Initialize variables for counting runtime
	clock_t time;
	double cpu_time_used;

	//Create a matrix to hold the next state of the matrix
	int* state_b = (int*)malloc(bytes);

	//A temporary pointer used for swapping
	int* temp_ptr;

	//Start counting rutime
	time = clock();

	//Iterate for k steps
	for (int iter = 0; iter < k; iter++)
	{
		//Go through each moment, one at a time, and calculate its spin.
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
				state_b[i * n + j] = sign(state_a[mod(i - 1, n) * n + j] + state_a[mod(i + 1, n) * n + j] + state_a[i * n + j] + state_a[i * n + mod(j - 1, n)] + state_a[i * n + mod(j + 1, n)]);
		}

		//Swap the two states
		temp_ptr = state_a;
		state_a = state_b;
		state_b = temp_ptr;
	}
	
	//Stop counting runtime
	time = clock() - time;

	//If k is an odd number, temp_ptr holds the original pointer to state_a,
	//so it must be copied back to state_a
	if (k % 2 == 1)
		memcpy(temp_ptr, state_a, bytes);

	//Calculate runtime
	cpu_time_used = ((double)(time)) / (CLOCKS_PER_SEC / 1000);

	//printf("\n\033[1mv0\033[0m Implementation Runtime: %fms\n", cpu_time_used);
	printf("\nv0 Implementation Runtime: %fms\n", cpu_time_used);

	return cpu_time_used;
}
