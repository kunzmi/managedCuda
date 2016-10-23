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
	texture<float, 2, cudaReadModeElementType> texRefPL;
	texture<float, 2, cudaReadModeElementType> texRefArray;
	// Device code

	// -------
	// kernels
	// -------
	//
	// NB: (1) The second argument "pitch" is in elements, not bytes
	//     (2) normalized coordinates are used (required for wrap address mode)

	__global__ void shiftPitchLinear(float* odata, int pitch, int width, int height,
		int shiftX, int shiftY)
	{
		int xid = blockIdx.x * blockDim.x + threadIdx.x;
		int yid = blockIdx.y * blockDim.y + threadIdx.y;

		odata[yid*pitch + xid] = tex2D(texRefPL,
			(xid + shiftX) / (float)width,
			(yid + shiftY) / (float)height);
	}

	__global__ void shiftArray(float* odata, int pitch, int width, int height,
		int shiftX, int shiftY)
	{
		int xid = blockIdx.x * blockDim.x + threadIdx.x;
		int yid = blockIdx.y * blockDim.y + threadIdx.y;

		odata[yid*pitch + xid] = tex2D(texRefArray,
			(xid + shiftX) / (float)width,
			(yid + shiftY) / (float)height);
	}

}