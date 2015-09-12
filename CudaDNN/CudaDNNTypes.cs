//	Copyright (c) 2015, Michael Kunz. All rights reserved.
//	http://kunzmi.github.io/managedCuda
//
//	This file is part of ManagedCuda.
//
//	ManagedCuda is free software: you can redistribute it and/or modify
//	it under the terms of the GNU Lesser General Public License as 
//	published by the Free Software Foundation, either version 2.1 of the 
//	License, or (at your option) any later version.
//
//	ManagedCuda is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//	GNU Lesser General Public License for more details.
//
//	You should have received a copy of the GNU Lesser General Public
//	License along with this library; if not, write to the Free Software
//	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
//	MA 02110-1301  USA, http://www.gnu.org/licenses/.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Text;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace ManagedCuda.CudaDNN
{
	/// <summary>
	/// Constants for LRN, #define in cudnn.h
	/// </summary>
	public static struct LRNConstants
	{
		/// <summary>
		/// minimum allowed lrnN
		/// </summary>
		public const double MinN = 1;
		/// <summary>
		/// maximum allowed lrnN
		/// </summary>
		public const double MaxN = 16;
		/// <summary>
		/// minimum allowed lrnK
		/// </summary>
		public const double MinK = 1e-5;
		/// <summary>
		/// minimum allowed lrnBeta
		/// </summary>
		public const double MinBeta = 0.01;
	}

	#region struct
	/// <summary>
	/// 
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudnnConvolutionFwdAlgoPerf
	{
		/// <summary>
		/// 
		/// </summary>
		public cudnnConvolutionFwdAlgo algo;
		/// <summary>
		/// 
		/// </summary>
		public cudnnStatus status;
		/// <summary>
		/// 
		/// </summary>
		public float time;
		/// <summary>
		/// 
		/// </summary>
		public SizeT memory;
	}

	/// <summary>
	/// 
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudnnConvolutionBwdFilterAlgoPerf
	{
		/// <summary>
		/// 
		/// </summary>
		public cudnnConvolutionBwdFilterAlgo algo;
		/// <summary>
		/// 
		/// </summary>
		public cudnnStatus status;
		/// <summary>
		/// 
		/// </summary>
		public float time;
		/// <summary>
		/// 
		/// </summary>
		public SizeT memory;
	}

	/// <summary>
	/// 
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudnnConvolutionBwdDataAlgoPerf
	{
		/// <summary>
		/// 
		/// </summary>
		public cudnnConvolutionBwdDataAlgo algo;
		/// <summary>
		/// 
		/// </summary>
		public cudnnStatus status;
		/// <summary>
		/// 
		/// </summary>
		public float time;
		/// <summary>
		/// 
		/// </summary>
		public SizeT memory;
	}
	#endregion

	#region struct as types
	[StructLayout(LayoutKind.Sequential)]
	public struct cudnnHandle
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Pointer;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct cudnnTensorDescriptor
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Pointer;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct cudnnConvolutionDescriptor
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Pointer;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct cudnnPoolingDescriptor
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Pointer;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct cudnnFilterDescriptor
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Pointer;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct cudnnLRNDescriptor
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Pointer;
	}
	#endregion

	#region enums

	/*
	* CUDNN return codes
	*/
	public enum cudnnStatus
	{
		Success = 0,
		NotInitialized = 1,
		AllocFailed = 2,
		BadParam = 3,
		InternalError = 4,
		InvalidValue = 5,
		ArchMismatch = 6,
		MappingError = 7,
		ExecutionFailed = 8,
		NotSupported = 9,
		LicenseError = 10
	}

	/*
	* CUDNN data type
	*/
	public enum cudnnDataType
	{
		Float = 0,
		Double = 1,
		Half = 2
	}

	public enum cudnnTensorFormat
	{
		NCHW = 0,   /* row major (wStride = 1, hStride = w) */
		NHWC = 1    /* feature maps interleaved ( cStride = 1 )*/
	}

	public enum cudnnAddMode
	{
		Image = 0,       /* add one image to every feature maps of each input */
		SameHW = 0,

		FeatureMap = 1,   /* add a set of feature maps to a batch of inputs : tensorBias has n=1 , same nb feature than Src/dest */
		SameCHW = 1,

		SameC = 2,   /* add a tensor of size 1,c,1,1 to every corresponding point of n,c,h,w input */

		FullTensor = 3    /* add 2 tensors with same n,c,h,w */
	}

	/*
	 *  convolution mode
	 */
	public enum cudnnConvolutionMode
	{
		Convolution = 0,
		CrossCorrelation = 1
	}



	/* helper function to provide the convolution algo that fit best the requirement */
	public enum cudnnConvolutionFwdPreference
	{
		NoWorkspace = 0,
		PreferFastest = 1,
		SpecifyWorkspaceLimit = 2,
	}

	public enum cudnnConvolutionFwdAlgo
	{
		ImplicitGEMM = 0,
		ImplicitPrecompGEMM = 1,
		GEMM = 2,
		Direct = 3,
		FFT = 4
	}



	/*
	 *  softmax algorithm
	 */
	public enum cudnnSoftmaxAlgorithm
	{
		Fast = 0,        /* straightforward implementation */
		Accurate = 1,         /* subtract max from every point to avoid overflow */
		Log = 2
	}

	public enum cudnnSoftmaxMode
	{
		Instace = 0,   /* compute the softmax over all C, H, W for each N */
		Channel = 1     /* compute the softmax over all C for each H, W, N */
	}

	/*
	 *  pooling mode
	 */
	public enum cudnnPoolingMode
	{
		Max = 0,
		AverageCountIncludePadding = 1, // count for average includes padded values
		AverageCountExcludePadding = 2 // count for average does not include padded values
	};


	/*
	 * activation mode
	 */
	public enum cudnnActivationMode
	{
		Sigmoid = 0,
		Relu = 1,
		Tanh = 2
	}

	
	public enum cudnnConvolutionBwdFilterPreference
	{
		NoWorkspace = 0,
		PreferFastest = 1,
		SpecifyWorkspaceLimit = 2
	}
	
	public enum cudnnConvolutionBwdFilterAlgo
	{
		Algo0 = 0,  // non-deterministic
		Algo1 = 1,
		AlgoFFT = 2
	}

	public enum cudnnConvolutionBwdDataPreference
	{
		NoWorkspace = 0,
		PreferFastest = 1,
		SpecifyWorkspaceLimit = 2
	}

	public enum cudnnConvolutionBwdDataAlgo
	{
		Algo0 = 0,  // non-deterministic
		Algo1 = 1,
		AlgoFFT = 2
	}

	/// <summary>
	/// LRN layer mode, currently only cross-channel is supported (across the tensor's dimA[1] dimension)
	/// </summary>
	public enum cudnnLRNMode
	{
		CrossChannelDim1 = 0
	} 

	public enum cudnnDivNormMode
	{
		PrecomputedMeans = 0
	} 

         
	#endregion
}
