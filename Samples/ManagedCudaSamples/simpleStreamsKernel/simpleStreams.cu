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
	__global__ void init_array(int *g_data, int *factor, int num_iterations)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		for (int i = 0; i<num_iterations; i++)
			g_data[idx] += *factor;	// non-coalesced on purpose, to burn time
	}
}