//Author: Giorgos Koutroumpis,
//ECE AUTH, Parallel and Distributed Systems 2022,
//AEM: 9668,
//Contact: geokonkou@ece.auth.gr

//Contains the CUDA kernels for each parallel implementation of the project.

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "../headers/kernel.h"
#include "../headers/helpers.h"

//Kernel for v1 implementation of ising evolution.
//Each thread calculates one moment's spin
//@args:
//n -> width of ising matrix
//state_a -> current state of spins matrix
//state_b -> matrix to store the new state of the spins
__global__ void gpuOneThreadEvolution(int n, int* state_a, int* state_b)
{
	//Get the width of the thread blocks
	int block_dim = blockDim.x;

	//Calculate the thread's coordinates. 
	//Each thread is mapped to the moment with the same coordinates
	int i = blockIdx.y * block_dim + threadIdx.y;
	int j = blockIdx.x * block_dim + threadIdx.x;

	//Since most times more threads than moments are spawned,
	//check if the thread actually needs to do work
	if (i < n && j < n)
	{
		state_b[i * n + j] = sign(state_a[mod(i - 1, n) * n + j] + state_a[mod(i + 1, n) * n + j] + state_a[i * n + j] + state_a[i * n + mod(j - 1, n)] + state_a[i * n + mod(j + 1, n)]);
	}
}

//Kernel for v2 implementation of ising evolution.
//Each thread calculates the spin of a block of moments (moment_block_width x moment_block_width or bxb)
//@args:
//n -> width of ising matrix
//state_a -> current state of spins matrix
//state_b -> matrix to store the new state of the spins
//moment_block_width -> the widht of the moment block each thread calculates (bxb)
__global__ void gpuOneThreadBlockEvolution(int n, int* state_a, int* state_b, int moment_block_width)
{
	//Get the width of the thread blocks
	int block_dim = blockDim.x;

	//Calculate the thread's global coordinates
	int iT = blockIdx.y * block_dim + threadIdx.y;
	int jT = blockIdx.x * block_dim + threadIdx.x;

	//Each thread calculates moment_block_width * moment_block_width moments
	for (int k = 0; k < moment_block_width * moment_block_width; k++)
	{
		//Get the respective moment's coordinates that the thread must calculate
		int i = iT * moment_block_width + (k / moment_block_width);
		int j = jT * moment_block_width + (k % moment_block_width);

		//Only calculate spin if the moment is within the matrix' bounds
		//(Since usually more threads than moments are spawned, must check
		if (i < n && j < n)
			state_b[i * n + j] = sign(state_a[mod(i - 1, n) * n + j] + state_a[mod(i + 1, n) * n + j] + state_a[i * n + j] + state_a[i * n + mod(j - 1, n)] + state_a[i * n + mod(j + 1, n)]);
	}
}

//Kernel for v3 implementation of ising evolution.
//Each thread calculates the spin of a block of moments (moment_block_width x moment_block_width or bxb) using shared memory
//@args:
//n -> width of ising matrix
//state_a -> current state of spins matrix
//state_b -> matrix to store the new state of the spins
//moment_block_width -> the widht of the moment block each thread calculates (bxb)
//shmem_width -> the width of the shared memory matrix
//limit -> the max number of thread blocks needed horizontally/vertically
__global__ void gpuSharedBlockEvolution(int n, int* state_a, int* state_b, int moment_block_width, int shmem_width, int limit)
{
	//Initialize shared memory matrix s_a
	extern __shared__ int s_a[];

	//Get thread's and block's info for easy reuse, as they are costly to access
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int bx = blockIdx.x;
	int by = blockIdx.y;

	//Get the width of the thread blocks
	int block_dim = blockDim.x;

	//Calculate the thread's global coordinates
	int jT = block_dim * bx + tx;
	int iT = block_dim * by + ty;

	//The edge of the block
	int m = moment_block_width * block_dim;

	//Check if the thread actually has to do work
	if (iT < limit && jT < limit)
	{
		//Each thread loads its moments (some threads load the edge moments, too) to the shared matrix
		for (int k = 0; k < moment_block_width * moment_block_width; k++)
		{
			//Local coordinates within thread
			int x = (k % moment_block_width);
			int y = (k / moment_block_width);

			//Local coordinates within thread-block
			int iM = iT * moment_block_width + y;
			int jM = jT * moment_block_width + x;

			//Check if the moment is out of bounds
			if (iM < n && jM < n)
			{
				//Moment coordinates within the block (row-major)
				int jMB = moment_block_width * tx + x;
				int iMB = moment_block_width * ty + y;

				//Padding for shared matrix
				int i = iMB + 1;
				int j = jMB + 1;
				
				//Load moment to shared memory
				s_a[i * shmem_width + j] = state_a[iM * n + jM];

				//If moment is at the edge of the thread-block or the matrix, load the respective 
				//edge neighbors
				if ((jMB == 0))
					s_a[i * shmem_width + (j - 1)] = state_a[iM * n + mod(jM - 1, n)];

				else if ((jMB + 1) == m || (jM + 1) == n)
					s_a[i * shmem_width + (j + 1)] = state_a[iM * n + mod(jM + 1, n)];

				if ((iMB == 0))
					s_a[(i - 1) * shmem_width + j] = state_a[mod(iM - 1, n) * n + jM];

				else if ((iMB + 1) == m || (iM + 1) == n)
					s_a[(i + 1) * shmem_width + j] = state_a[mod(iM + 1, n) * n + jM];
			}
		}
	}
	//Wait for all threads to finish loading to shared matrix
	__syncthreads();

	//Calculate each moment's spin, like in v2, but now getting the spins from the shared memory, instead of global
	if (iT < limit && jT < limit)
	{
		for (int k = 0; k < moment_block_width * moment_block_width; k++)
		{
			int x = (k % moment_block_width);
			int y = (k / moment_block_width);

			int iM = iT * moment_block_width + y;
			int jM = jT * moment_block_width + x;

			if (iM < n && jM < n)
			{
				//Moment coordinates within the block (row-major)
				int jMB = moment_block_width * tx + x;
				int iMB = moment_block_width * ty + y;

				//Padding for shared matrix
				int i = iMB + 1;
				int j = jMB + 1;

				state_b[iM * n + jM] = sign(s_a[(i - 1) * shmem_width + j] + s_a[(i + 1) * shmem_width + j] + s_a[i * shmem_width + j] + s_a[i * shmem_width + (j - 1)] + s_a[i * shmem_width + (j + 1)]);
			}
		}
	}
}


