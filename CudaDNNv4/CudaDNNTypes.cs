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

namespace ManagedCuda.CudaDNNv4
{
	/// <summary>
	/// Constants for LRN, #define in cudnn.h
	/// </summary>
	public struct LRNConstants
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

    public struct BNConstants
    {
        public const double MinEpsilon = 1e-5;
    }

	#region struct
	/// <summary>
	/// cudnnConvolutionFwdAlgoPerf is a structure containing performance results
	/// returned by cudnnFindConvolutionForwardAlgorithm().
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudnnConvolutionFwdAlgoPerf
	{
		/// <summary>
		/// The algorithm run to obtain the associated performance metrics.
		/// </summary>
		public cudnnConvolutionFwdAlgo algo;
		/// <summary>
		/// If any error occurs during the workspace allocation or timing of cudnnConvolutionForward(),
		/// this status will represent that error. Otherwise, this status will be the return status of
		/// cudnnConvolutionForward().<para/>
		/// - CUDNN_STATUS_ALLOC_FAILED if any error occured during workspace allocation or deallocation.<para/>
		/// - CUDNN_STATUS_EXECUTION_FAILED if any error occured during timing calculations.<para/>
		/// - Otherwise, this will be the return status of cudnnConvolutionForward().<para/>
		/// </summary>
		public cudnnStatus status;
		/// <summary>
		/// The execution time of cudnnConvolutionForward() (in milliseconds).
		/// </summary>
		public float time;
		/// <summary>
		/// The workspace size (in bytes).
		/// </summary>
		public SizeT memory;
	}

	/// <summary>
	/// cudnnConvolutionBwdFilterAlgoPerf is a structure containing performance
	/// results returned by cudnnFindConvolutionBackwardFilterAlgorithm().
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudnnConvolutionBwdFilterAlgoPerf
	{
		/// <summary>
		/// The algorithm run to obtain the associated performance metrics.
		/// </summary>
		public cudnnConvolutionBwdFilterAlgo algo;
		/// <summary>
		/// If any error occurs during the workspace allocation or timing of
		/// cudnnConvolutionBackwardFilter_v3(), this status will represent that error. Otherwise,
		/// this status will be the return status of cudnnConvolutionBackwardFilter_v3().<para/>
		/// - CUDNN_STATUS_ALLOC_FAILED if any error occured during workspace allocation or deallocation.<para/>
		/// - CUDNN_STATUS_EXECUTION_FAILED if any error occured during timing calculations.<para/>
		/// - Otherwise, this will be the return status of cudnnConvolutionBackwardFilter_v3().<para/>
		/// </summary>
		public cudnnStatus status;
		/// <summary>
		/// The execution time of cudnnConvolutionBackwardFilter_v3() (in milliseconds).
		/// </summary>
		public float time;
		/// <summary>
		/// The workspace size (in bytes).
		/// </summary>
		public SizeT memory;
	}

	/// <summary>
	/// cudnnConvolutionBwdDataAlgoPerf is a structure containing performance results
	/// returned by cudnnFindConvolutionBackwardDataAlgorithm().
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudnnConvolutionBwdDataAlgoPerf
	{
		/// <summary>
		/// The algorithm run to obtain the associated performance metrics.
		/// </summary>
		public cudnnConvolutionBwdDataAlgo algo;
		/// <summary>
		/// If any error occurs during the workspace allocation or timing of
		/// cudnnConvolutionBackwardData_v3(), this status will represent that error. Otherwise,
		/// this status will be the return status of cudnnConvolutionBackwardData_v3().<para/>
		/// - CUDNN_STATUS_ALLOC_FAILED if any error occured during workspace allocation or deallocation.<para/>
		/// - CUDNN_STATUS_EXECUTION_FAILED if any error occured during timing calculations.<para/>
		/// - Otherwise, this will be the return status of cudnnConvolutionBackwardData_v3().
		/// </summary>
		public cudnnStatus status;
		/// <summary>
		/// The execution time of cudnnConvolutionBackwardData_v3() (in milliseconds).
		/// </summary>
		public float time;
		/// <summary>
		/// The workspace size (in bytes).
		/// </summary>
		public SizeT memory;
	}
	#endregion

	#region struct as types
	/// <summary>
	/// cudnnHandle is a pointer to an opaque structure holding the cuDNN library context.<para/>
	/// The cuDNN library context must be created using cudnnCreate() and the returned
	/// handle must be passed to all subsequent library function calls. The context should be
	/// destroyed at the end using cudnnDestroy(). The context is associated with only one
	/// GPU device, the current device at the time of the call to cudnnCreate(). However
	/// multiple contexts can be created on the same GPU device.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudnnHandle
	{
		private IntPtr Pointer;
	}

	/// <summary>
	/// cudnnCreateTensorDescriptor is a pointer to an opaque structure holding the
	/// description of a generic n-D dataset.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudnnTensorDescriptor
	{
		private IntPtr Pointer;
	}

	/// <summary>
	/// cudnnConvolutionDescriptor is a pointer to an opaque structure holding the
	/// description of a convolution operation.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudnnConvolutionDescriptor
	{
		/// <summary>
		/// 
		/// </summary>
		private IntPtr Pointer;
	}

	/// <summary>
	/// cudnnPoolingDescriptor is a pointer to an opaque structure holding
	/// the description of a pooling operation.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudnnPoolingDescriptor
	{
		private IntPtr Pointer;
	}

	/// <summary>
	/// cudnnFilterDescriptor is a pointer to an opaque structure holding the description
	/// of a filter dataset.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cudnnFilterDescriptor
	{
		private IntPtr Pointer;
	}

    /// <summary>
    /// cudnnLRNDescriptor is a pointer to an opaque structure holding the description
    /// of a local response normalization operation.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
	public struct cudnnLRNDescriptor
	{
		private IntPtr Pointer;
	}

    /// <summary>
    /// cudnnActivationDescriptor is a pointer to an opaque structure holding the description
    /// of a activation operation.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cudnnActivationDescriptor
    {
        private IntPtr Pointer;
    }

    #endregion

    #region enums

    /// <summary>
    /// CUDNN return codes
    /// </summary>
    public enum cudnnStatus
	{
		/// <summary>
		/// The operation completed successfully.
		/// </summary>
		Success = 0,
		/// <summary>
		/// The cuDNN library was not initialized properly.<para/>
		/// This error is usually returned when a call to
		/// cudnnCreate() fails or when cudnnCreate()
		/// has not been called prior to calling another cuDNN
		/// routine. In the former case, it is usually due
		/// to an error in the CUDA Runtime API called by
		/// cudnnCreate() or by an error in the hardware
		/// setup.
		/// </summary>
		NotInitialized = 1,
		/// <summary>
		/// Resource allocation failed inside the cuDNN
		/// library. This is usually caused by an internal
		/// cudaMalloc() failure.<para/>
		/// To correct: prior to the function call, deallocate
		/// previously allocated memory as much as possible.
		/// </summary>
		AllocFailed = 2,
		/// <summary>
		/// An incorrect value or parameter was passed to the
		/// function.<para/>
		/// To correct: ensure that all the parameters being
		/// passed have valid values.
		/// </summary>
		BadParam = 3,
		/// <summary>
		/// An internal cuDNN operation failed.
		/// </summary>
		InternalError = 4,
		/// <summary>
		/// 
		/// </summary>
		InvalidValue = 5,
		/// <summary>
		/// The function requires a feature absent from
		/// the current GPU device. Note that cuDNN only
		/// supports devices with compute capabilities greater
		/// than or equal to 3.0.<para/>
		/// To correct: compile and run the application on a
		/// device with appropriate compute capability.
		/// </summary>
		ArchMismatch = 6,
		/// <summary>
		/// An access to GPU memory space failed, which is
		/// usually caused by a failure to bind a texture.<para/>
		/// To correct: prior to the function call, unbind any
		/// previously bound textures.<para/>
		/// Otherwise, this may indicate an internal error/bug
		/// in the library.
		/// </summary>
		MappingError = 7,
		/// <summary>
		/// The GPU program failed to execute. This is usually
		/// caused by a failure to launch some cuDNN kernel
		/// on the GPU, which can occur for multiple reasons.<para/>
		/// To correct: check that the hardware, an
		/// appropriate version of the driver, and the cuDNN
		/// library are correctly installed.<para/>
		/// Otherwise, this may indicate a internal error/bug
		/// in the library.
		/// </summary>
		ExecutionFailed = 8,
		/// <summary>
		/// The functionality requested is not presently supported by cuDNN.
		/// </summary>
		NotSupported = 9,
		/// <summary>
		/// The functionality requested requires some license
		/// and an error was detected when trying to check
		/// the current licensing. This error can happen if
		/// the license is not present or is expired or if the
		/// environment variable NVIDIA_LICENSE_FILE is not
		/// set properly.
		/// </summary>
		LicenseError = 10
	}

	/// <summary>
	/// cudnnDataType is an enumerated type indicating the data type to which a tensor
	/// descriptor or filter descriptor refers.
	/// </summary>
	public enum cudnnDataType
	{
		/// <summary>
		/// The data is 32-bit single-precision floating point (float).
		/// </summary>
		Float = 0,
		/// <summary>
		/// The data is 64-bit double-precision floating point (double).
		/// </summary>
		Double = 1,
		/// <summary>
		/// The data is 16-bit floating point.
		/// </summary>
		Half = 2
	}

    /// </summary>
	/// cudnnNanPropagation is an enumerated type for the NanPropagation flag.
	/// </summary>
	public enum cudnnNanPropagation
    {
        /// </summary>
        /// Selects the not propagate NaN option.
        /// </summary>
        NotPropagateNan = 0,
        /// </summary>
        /// Selects the propagate NaN option.
        /// </summary>
        PropagateNan = 1
	}

    /// <summary>
    /// cudnnTensorFormat is an enumerated type used by
    /// cudnnSetTensor4dDescriptor() to create a tensor with a pre-defined layout.
    /// </summary>
    public enum cudnnTensorFormat
	{
		/// <summary>
		/// This tensor format specifies that the data is laid out in the following order: image, features map,
		/// rows, columns. The strides are implicitly defined in such a way that the data are contiguous in
		/// memory with no padding between images, feature maps, rows, and columns; the columns are the
		/// inner dimension and the images are the outermost dimension.
		/// </summary>
		NCHW = 0,   /* row major (wStride = 1, hStride = w) */
		/// <summary>
		/// This tensor format specifies that the data is laid out in the following order: image, rows, columns,
		/// features maps. The strides are implicitly defined in such a way that the data are contiguous in memory
		/// with no padding between images, rows, columns, and features maps; the feature maps are the
		/// inner dimension and the images are the outermost dimension.
		/// </summary>
		NHWC = 1    /* feature maps interleaved ( cStride = 1 )*/
	}

	/// <summary>
	/// cudnnAddMode is an enumerated type used by cudnnAddTensor() to specify how a
	/// bias tensor is added to an input/output tensor.
	/// </summary>
	public enum cudnnAddMode
	{
		/// <summary>
		/// In this mode, the bias tensor is defined as one
		/// image with one feature map. This image will be
		/// added to every feature map of every image of the
		/// input/output tensor.
		/// </summary>
		Image = 0,       /* add one image to every feature maps of each input */
		/// <summary>
		/// In this mode, the bias tensor is defined as one
		/// image with one feature map. This image will be
		/// added to every feature map of every image of the
		/// input/output tensor.
		/// </summary>
		SameHW = 0,

		/// <summary>
		/// In this mode, the bias tensor is defined as one
		/// image with multiple feature maps. This image
		/// will be added to every image of the input/output
		/// tensor.
		/// </summary>
		FeatureMap = 1,   /* add a set of feature maps to a batch of inputs : tensorBias has n=1 , same nb feature than Src/dest */

		/// <summary>
		/// In this mode, the bias tensor is defined as one
		/// image with multiple feature maps. This image
		/// will be added to every image of the input/output
		/// tensor.
		/// </summary>
		SameCHW = 1,

		/// <summary>
		/// In this mode, the bias tensor is defined as one
		/// image with multiple feature maps of dimension
		/// 1x1; it can be seen as an vector of feature maps.
		/// Each feature map of the bias tensor will be added
		/// to the corresponding feature map of all height-bywidth
		/// pixels of every image of the input/output
		/// tensor.
		/// </summary>
		SameC = 2,   /* add a tensor of size 1,c,1,1 to every corresponding point of n,c,h,w input */

		/// <summary>
		/// In this mode, the bias tensor has the same
		/// dimensions as the input/output tensor. It will be
		/// added point-wise to the input/output tensor.
		/// </summary>
		FullTensor = 3    /* add 2 tensors with same n,c,h,w */
	}

	/// <summary>
	/// cudnnConvolutionMode is an enumerated type used by
	/// cudnnSetConvolutionDescriptor() to configure a convolution descriptor.
	/// </summary>
	public enum cudnnConvolutionMode
	{
		/// <summary>
		/// In this mode, a convolution operation will be done
		/// when applying the filter to the images.
		/// </summary>
		Convolution = 0,
		/// <summary>
		/// In this mode, a cross-correlation operation will be
		/// done when applying the filter to the images.
		/// </summary>
		CrossCorrelation = 1
	}



	/// <summary>
	/// cudnnConvolutionFwdPreference is an enumerated type used by
	/// cudnnGetConvolutionForwardAlgorithm() to help the choice of the algorithm used
	/// for the forward convolution.
	/// </summary>
	public enum cudnnConvolutionFwdPreference
	{
		/// <summary>
		/// In this configuration, the routine cudnnGetConvolutionForwardAlgorithm() is
		/// guaranteed to return an algorithm that does not require any extra workspace to be provided by the
		/// user.
		/// </summary>
		NoWorkspace = 0,
		/// <summary>
		/// In this configuration, the routine cudnnGetConvolutionForwardAlgorithm() will
		/// return the fastest algorithm regardless how much workspace is needed to execute it.
		/// </summary>
		PreferFastest = 1,
		/// <summary>
		/// In this configuration, the routine cudnnGetConvolutionForwardAlgorithm() will
		/// return the fastest algorithm that fits within the memory limit that the user provided.
		/// </summary>
		SpecifyWorkspaceLimit = 2,
	}

	/// <summary>
	/// cudnnConvolutionFwdAlgo is an enumerated type that exposes the different
	/// algorithms available to execute the forward convolution operation.
	/// </summary>
	public enum cudnnConvolutionFwdAlgo
	{
		/// <summary>
		/// This algorithm expresses the convolution as a matrix product without actually explicitly form the
		/// matrix that holds the input tensor data.
		/// </summary>
		ImplicitGEMM = 0,
		/// <summary>
		/// This algorithm expresses the convolution as a matrix product without actually explicitly form
		/// the matrix that holds the input tensor data, but still needs some memory workspace to precompute
		/// some indices in order to facilitate the implicit construction of the matrix that holds the input
		/// tensor data
		/// </summary>
		ImplicitPrecompGEMM = 1,
		/// <summary>
		/// This algorithm expresses the convolution as an explicit matrix product. A significant memory
		/// workspace is needed to store the matrix that holds the input tensor data.
		/// </summary>
		GEMM = 2,
		/// <summary>
		/// This algorithm expresses the convolution as a direct convolution (e.g without implicitly or
		/// explicitly doing a matrix multiplication).
		/// </summary>
		Direct = 3,
		/// <summary>
		/// This algorithm uses a Fast-Fourier Transform approach to compute the convolution. A
		/// significant memory workspace is needed to store intermediate results.
		/// </summary>
		FFT = 4,
        /// <summary>
        /// This algorithm uses a Fast-Fourier Transform approach but splits the inputs into 32x32 tiles. A
        /// significant memory workspace is needed to store intermediate results but significantly less than
        /// FFT for big size images.
        /// </summary>
        FFTWithTiling = 5
    }



	/// <summary>
	/// cudnnSoftmaxAlgorithm is used to select an implementation of the softmax
	/// function used in cudnnSoftmaxForward() and cudnnSoftmaxBackward().
	/// </summary>
	public enum cudnnSoftmaxAlgorithm
	{
		/// <summary>
		/// This implementation applies the straightforward softmax operation.
		/// </summary>
		Fast = 0,        /* straightforward implementation */
		/// <summary>
		/// This implementation scales each point of the softmax input domain by its maximum value to
		/// avoid potential floating point overflows in the softmax evaluation.
		/// </summary>
		Accurate = 1,         /* subtract max from every point to avoid overflow */
		/// <summary>
		/// This entry performs the Log softmax operation, avoiding overflows by scaling each point in the
		/// input domain as in CUDNN_SOFTMAX_ACCURATE
		/// </summary>
		Log = 2
	}

	/// <summary>
	/// cudnnSoftmaxMode is used to select over which data the cudnnSoftmaxForward()
	/// and cudnnSoftmaxBackward() are computing their results.
	/// </summary>
	public enum cudnnSoftmaxMode
	{
		/// <summary>
		/// The softmax operation is computed per image (N) across the dimensions C,H,W.
		/// </summary>
		Instance = 0,   /* compute the softmax over all C, H, W for each N */
		/// <summary>
		/// The softmax operation is computed per spatial location (H,W) per image (N) across the dimension C.
		/// </summary>
		Channel = 1     /* compute the softmax over all C for each H, W, N */
	}

	/// <summary>
	/// cudnnPoolingMode is an enumerated type passed to cudnnSetPoolingDescriptor() to select the pooling method to be used by
	/// cudnnPoolingForward() and cudnnPoolingBackward().
	/// </summary>
	public enum cudnnPoolingMode
	{
		/// <summary>
		/// The maximum value inside the pooling window will be used.
		/// </summary>
		Max = 0,
		/// <summary>
		/// The values inside the pooling window will be averaged. The number of padded values will be
		/// taken into account when computing the average value.
		/// </summary>
		AverageCountIncludePadding = 1, // count for average includes padded values
		/// <summary>
		/// The values inside the pooling window will be averaged. The number of padded values will not
		/// be taken into account when computing the average value.
		/// </summary>
		AverageCountExcludePadding = 2 // count for average does not include padded values
	};


	/// <summary>
	/// cudnnActivationMode is an enumerated type used to select the neuron activation
	/// function used in cudnnActivationForward() and cudnnActivationBackward().
	/// </summary>
	public enum cudnnActivationMode
	{
		/// <summary>
		/// Selects the sigmoid function.
		/// </summary>
		Sigmoid = 0,
		/// <summary>
		/// Selects the rectified linear function.
		/// </summary>
		Relu = 1,
		/// <summary>
		/// Selects the hyperbolic tangent function.
		/// </summary>
		Tanh = 2,
        /// <summary>
        /// Selects the clipped rectified linear function.
        /// </summary>
        ClippedRelu = 3
    }

	/// <summary>
	/// cudnnConvolutionBwdFilterPreference is an enumerated type used by
	/// cudnnGetConvolutionBackwardFilterAlgorithm() to help the choice of the
	/// algorithm used for the backward filter convolution.
	/// </summary>
	public enum cudnnConvolutionBwdFilterPreference
	{
		/// <summary>
		/// In this configuration, the routine cudnnGetConvolutionBackwardFilterAlgorithm()
		/// is guaranteed to return an algorithm that does not require any extra workspace to be provided by the user.
		/// </summary>
		NoWorkspace = 0,
		/// <summary>
		/// In this configuration, the routine cudnnGetConvolutionBackwardFilterAlgorithm()
		/// will return the fastest algorithm regardless how much workspace is needed to execute it.
		/// </summary>
		PreferFastest = 1,
		/// <summary>
		/// In this configuration, the routine cudnnGetConvolutionBackwardFilterAlgorithm()
		/// will return the fastest algorithm that fits within the memory limit that the user provided.
		/// </summary>
		SpecifyWorkspaceLimit = 2
	}
	
	/// <summary>
	/// cudnnConvolutionBwdFilterAlgo is an enumerated type that exposes the different
	/// algorithms available to execute the backward filter convolution operation.
	/// </summary>
	public enum cudnnConvolutionBwdFilterAlgo
	{
		/// <summary>
		/// This algorithm expresses the convolution as a sum of matrix product without actually explicitly form
		/// the matrix that holds the input tensor data. The sum is done using atomic adds operation, thus the
		/// results are non-deterministic.
		/// </summary>
		Algo0 = 0,  // non-deterministic
		/// <summary>
		/// This algorithm expresses the convolution as a matrix product without actually explicitly form
		/// the matrix that holds the input tensor data. The results are deterministic.
		/// </summary>
		Algo1 = 1,
		/// <summary>
		/// This algorithm uses a Fast-Fourier Transform approach to compute the convolution. A
		/// significant memory workspace is needed to store intermediate results. The results are
		/// deterministic.
		/// </summary>
		AlgoFFT = 2,
		/// <summary>
		/// This algorithm is similar to CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 but
		/// uses some small workspace to precomputes some indices. The results are also non-deterministic.
		/// </summary>
		Algo3 = 3   // non-deterministic, algo0 with workspace

	}

	/// <summary>
	/// cudnnConvolutionBwdDataPreference is an enumerated type used by
	/// cudnnGetConvolutionBackwardDataAlgorithm() to help the choice of the
	/// algorithm used for the backward data convolution.
	/// </summary>
	public enum cudnnConvolutionBwdDataPreference
	{
		/// <summary>
		/// In this configuration, the routine cudnnGetConvolutionBackwardDataAlgorithm()
		/// is guaranteed to return an algorithm that does not require any extra workspace to be provided by the
		/// user.
		/// </summary>
		NoWorkspace = 0,
		/// <summary>
		/// In this configuration, the routine cudnnGetConvolutionBackwardDataAlgorithm()
		/// will return the fastest algorithm regardless how much workspace is needed to execute it.
		/// </summary>
		PreferFastest = 1,
		/// <summary>
		/// In this configuration, the routine cudnnGetConvolutionBackwardDataAlgorithm()
		/// will return the fastest algorithm that fits within the memory limit that the user provided.
		/// </summary>
		SpecifyWorkspaceLimit = 2
	}

	/// <summary>
	/// cudnnConvolutionBwdDataAlgo is an enumerated type that exposes the different
	/// algorithms available to execute the backward data convolution operation.
	/// </summary>
	public enum cudnnConvolutionBwdDataAlgo
	{
		/// <summary>
		/// This algorithm expresses the convolution as a sum of matrix product without actually explicitly form
		/// the matrix that holds the input tensor data. The sum is done using atomic adds operation, thus the
		/// results are non-deterministic.
		/// </summary>
		Algo0 = 0,  // non-deterministic
		/// <summary>
		/// This algorithm expresses the convolution as a matrix product without actually explicitly form
		/// the matrix that holds the input tensor data. The results are deterministic.
		/// </summary>
		Algo1 = 1,
		/// <summary>
		/// This algorithm uses a Fast-Fourier Transform approach to compute the convolution. A
		/// significant memory workspace is needed to store intermediate results. The results are
		/// deterministic.
		/// </summary>
		AlgoFFT = 2
	}

	/// <summary>
	/// cudnnLRNMode is an enumerated type used to specify the mode of operation in cudnnLRNCrossChannelForward() and cudnnLRNCrossChannelBackward().
	/// </summary>
	public enum cudnnLRNMode
	{
		/// <summary>
		/// LRN co mputation is performed across tensor's dimension dimA[1].
		/// </summary>
		CrossChannelDim1 = 0
	} 

	/// <summary>
	/// cudnnDivNormMode is an enumerated type used to specify the
	/// mode of operation in cudnnDivisiveNormalizationForward() and
	/// cudnnDivisiveNormalizationBackward().
	/// </summary>
	public enum cudnnDivNormMode
	{
		/// <summary>
		/// The means tensor data pointer is expected to
		/// contain means or other kernel convolution values
		/// precomputed by the user. The means pointer
		/// can also be NULL, in that case it's considered
		/// to be filled with zeroes. This is equivalent to
		/// spatial LRN. Note that in the backward pass
		/// the means are treated as independent inputs
		/// and the gradient over means is computed
		/// independently. In this mode to yield a net gradient
		/// over the entire LCN computational graph the
		/// destDiffMeans result should be backpropagated
		/// through the user's means layer (which can
		/// be impelemented using average pooling) and
		/// added to the destDiffData tensor produced by
		/// cudnnDivisiveNormalizationBackward.
		/// </summary>
		PrecomputedMeans = 0
	}

    /// <summary>
    /// cudnnBatchNormMode_t is an enumerated type used to specify the mode of operation in 
    /// cudnnBatchNormalizationForwardInference(), cudnnBatchNormalizationForwardTraining(), 
    /// cudnnBatchNormalizationBackward() and cudnnDeriveBNTensorDescriptor() routines. 
    /// </summary>
    public enum cudnnBatchNormMode
    {
        /// <summary>
        /// Normalization is performed per-activation. This mode is intended to be used after nonconvolutional
        /// network layers. In this mode bnBias and bnScale tensor dimensions are 1xCxHxW.
        /// </summary>
        BatchNormPerActivation = 0,

        /// <summary>
        /// Normalization is performed over N+spatial dimensions. This mode is intended for use after
        /// convolutional layers (where spatial invariance is desired). In this mode bnBias, bnScale tensor
        /// dimensions are 1xCx1x1.
        /// </summary>
        BatchNormSpatial = 1
    }

    #endregion
    }
