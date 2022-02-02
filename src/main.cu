//Author: Giorgos Koutroumpis,
//ECE AUTH, Parallel and Distributed Systems 2022,
//AEM: 9668,
//Contact: geokonkou@ece.auth.gr

//The project's driver code. For a given n and k, it simulates the ising evolution of a nxn ising matrix, for k steps.
//Command line arguments: argv1: n, argv2: k

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "../headers/isingEvolution.h"
#include "../headers/helpers.h"

#define SHMEM_LIMIT (48000) //The limit of the Shared Memory in most GPUs (48kB)
#define THREAD_WIDTH (8)	//The width of the thread block
#define B (4)				//The width of the bxb block that v2, v3 use

//Driver code for the project. 
//Command line arguments: argv1: n, argv2: k
int main(int argc, char* argv[])
{
	//Default n size for the nxn matrix
	int n = 5000;		

	//Default number of steps 
	int k = 10;

	//Initialize seed for populating the matrix with random values
	srand(time(NULL));

	//b value for bxb block of v2, v3
	int moment_block_width = B;

	//The width of the thread block
	int thread_width = THREAD_WIDTH;

	//The matrix that holds the inital generated matrix
	int* state_a;

	//The matrix that will hold the result of the sequential implementation
	int* seq_a;

	//The matrix that will hold the result of each parallel implentation
	int* par_a;

	//Check for command line arguments. 
	//If one argument is provided, it is the size n
	if (argc == 2)
		n = atoi(argv[1]);
	//if two arguments are provided, the first is n, the second is k
	else if (argc == 3)
	{
		n = atoi(argv[1]);
		k = atoi(argv[2]);
	}

	//Start execution of each version for provided n, k values
	printf("\nExecuting for n = %d, k = %d\n\n", n, k);

	//The size of each matrix
	int bytes = n * n * sizeof(int);

	//Allocate memory for each matrix
	state_a = (int*)malloc(bytes);
	seq_a = (int*)malloc(bytes);
	par_a = (int*)malloc(bytes);

	//Populate the matrix with random -1 and 1 values
	populateMatrix(n, state_a);

	//Copy the matrix for use with the sequential implementation
	memcpy(seq_a, state_a, bytes);

	//Run the sequential implementation 
	double v0_time = sequentialIsingEvolution(seq_a, n, k, bytes);

	//Copy the matrix for use with v1
	memcpy(par_a, state_a, bytes);
	//Run v1
	float v1_time = gpuOneThreadIsing(par_a, n, k, bytes, thread_width);

	//Validate that the result is the same as the sequential implementation
	printf("v1 Validation:");
	validate(n, seq_a, par_a);

	//For v2, v3 limit b to so that the shared memory size will not exceed the limit
	if (((moment_block_width * thread_width + 2) * (moment_block_width * thread_width + 2) * sizeof(int)) >= SHMEM_LIMIT)
	{
		moment_block_width = sqrt(SHMEM_LIMIT / sizeof(int)) / (thread_width + 2);
	}

	//Copy the matrix for use with v2
	memcpy(par_a, state_a, bytes);
	//Run v2
	float v2_time = gpuOneThreadBlockIsing(par_a, n, k, bytes, moment_block_width, thread_width);

	//Validate that the result is the same as the sequential implementation (and with v1, as long as v1 is validated)
	printf("v2 Validation:");
	validate(n, seq_a, par_a);

	//Copy the matrix for use with v3
	memcpy(par_a, state_a, bytes);
	//Run v3
	float v3_time = gpuSharedBlockIsing(par_a, n, k, bytes, moment_block_width, thread_width);

	//Validate that the result is the same as the sequential implementation (and with v1, v2, as long as v1, v2 are validated)
	printf("v3 Validation:");
	validate(n, seq_a, par_a);
	
	//Free memory
	free(state_a);
	free(par_a);
}
