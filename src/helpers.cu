//Author: Giorgos Koutroumpis,
//ECE AUTH, Parallel and Distributed Systems 2022,
//AEM: 9668,
//Contact: geokonkou@ece.auth.gr

//Helper functions for the project

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "../headers/helpers.h"

//Returns the mathematical result of x % m
//@args:
//x ->  left side of mod
//m ->  right side of mod
__host__ __device__ int mod(int x, int m)
{
	return (x % m + m) % m;
}

//Get the sign of the input number x
//@args:
//x -> the number to get the sign of
__host__ __device__ int sign(int x)
{
	return (x > 0) - (x < 0);
}

//Prints a nxn square matrix.
//@args:
//n -> width of matrix
//matrix -> 2D matrix in 1D form (row major)
__host__  void printMatrix(int n, int* matrix)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			printf("%2d ", matrix[i * n + j]);
		}
		printf("\n");
	}
	printf("\n\n");
}


//Generates a random integer number between rangeLow and rangeHigh (inclusive)
//@args:
//rangeLow -> lower number of range
//rangeHigh -> higher number of range
__host__  int uniform_distribution(int rangeLow, int rangeHigh)
{
	double myRand = rand() / (1.0 + RAND_MAX);
	int range = rangeHigh - rangeLow + 1;
	int myRand_scaled = (myRand * range) + rangeLow;
	return myRand_scaled;
}

//Populates a 2D matrix (1D form, row-major) with -1s and 1s.
//@args:
//n -> width of matrix
//matrix -> 2D matrix in 1D form (row major)
void populateMatrix(int n, int* matrix)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
		{
			matrix[i * n + j] = uniform_distribution(0, 1) == 1 ? 1 : -1;
		}
}

//Compares if all elements of matrixA and matrixB are equal.
//@args:
//matrixA -> 2D matrix in 1D form (row major)
//matrixB -> 2D matrix in 1D form (row major)
int validate(int n, int* matrixA, int* matrixB)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (matrixA[i * n + j] != matrixB[i * n + j])
			{
				//printf("\nCheck \033[1;31mfailed\033[0m!\n\n");
				printf("\nCheck failed!\n\n");
				return -1;
			}
		}
	}
	//printf("\nCheck \033[1;32msuccess\033[0m!\n\n");
	printf("\nCheck success!\n\n");
	return 0;
}
