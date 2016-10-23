/*
* This code is taken more or less entirely from the NVIDIA CUDA SDK.
* This software contains source code provided by NVIDIA Corporation.
*
*/


//Includes for IntelliSense 
#define _SIZE_T_DEFINED
#ifndef __CUDACC__
#define __CUDACC__
#endif
#ifndef __cplusplus
#define __cplusplus
#endif

#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>


extern "C" {
	// Device code
	__global__ void VecAdd(const float* A, const float* B, float* C, int N)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		if (i < N)
			C[i] = A[i] + B[i];
	}
}