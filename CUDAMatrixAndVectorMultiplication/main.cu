#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernel.h"

int main(int argc, char* argv[])
{

	float* hA;
	float* dA;

	float* hX;
	float* dX;

	float* hC;
	float* dC;

	int  N_thread = 32;
	int vectorSize = N_thread *1;
	int matrixSize = vectorSize * vectorSize;
	int N_blocks;
	int i;
	int j;
	unsigned int vectorMem_size = sizeof(float) * vectorSize;
	unsigned int matrixMem_size = sizeof(float) * matrixSize;

	hA = (float*)malloc(matrixMem_size);
	hX = (float*)malloc(vectorMem_size);
	hC = (float*)malloc(vectorMem_size);

	cudaError_t err;

	err = cudaMalloc((void**)&dA, matrixMem_size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Cannot allocate GPU memory: %s\n", cudaGetErrorString(err));
		return 1;
	}

	err = cudaMalloc((void**)&dX, vectorMem_size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Cannot allocate GPU memory: %s\n", cudaGetErrorString(err));
		return 1;
	}

	err = cudaMalloc((void**)&dC, vectorMem_size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Cannot allocate GPU memory: %s\n", cudaGetErrorString(err));
		return 1;
	}

	for (i = 0; i < vectorSize; i++) {
		for (int j = 0; j < vectorSize; j++) {
			hA[i * vectorSize + j] = j;
			printf("A[%d,%d] = %.5f\n", i,j, hA[i * vectorSize + j]);
		}
		hX[i] = i;
		hC[i] = 0.0f;
		printf("x[%d] = %.5f\n", i, hX[i]);
	}

	N_blocks = matrixSize / N_thread;

	cudaMemcpy(dA , hA , matrixMem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dX , hX , vectorMem_size, cudaMemcpyHostToDevice);

	function << < N_blocks, N_thread >> > (dA , dX , dC , vectorSize);

	cudaMemcpy(hC, dC, vectorMem_size, cudaMemcpyDeviceToHost);

	for (int idx = 0; idx < vectorSize; idx++) {
		printf("c[%d] = %.5f\n", idx, hC[idx]);
	}

	free(hA);
	free(hX);
	free(hC);

	cudaFree(dA);
	cudaFree(dX);
	cudaFree(dC);


	return 0;
}