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

	/// <summary>
	/// Constant values for BN
	/// </summary>
    public struct BNConstants
    {
		/// <summary>
		/// MinEpsilon = 1e-5
		/// </summary>
        public const double MinEpsilon = 1e-5;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="sev"></param>
    /// <param name="udata"></param>
    /// <param name="dbg"></param>
    /// <param name="msg"></param>
    public delegate void cudnnCallback(cudnnSeverity sev, IntPtr udata, ref cudnnDebug dbg, [MarshalAs(UnmanagedType.LPStr)] string msg);


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

        /// <summary>
        /// The determinism of the algorithm.
        /// </summary>
        public cudnnDeterminism determinism;

        /// <summary>
        /// 
        /// </summary>
        public cudnnMathType mathType;
        int reserved1;
        int reserved2;
        int reserved3;
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

        /// <summary>
        /// The determinism of the algorithm.
        /// </summary>
        public cudnnDeterminism determinism;

        /// <summary>
        /// 
        /// </summary>
        public cudnnMathType mathType;
        int reserved1;
        int reserved2;
        int reserved3;
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

        /// <summary>
        /// The determinism of the algorithm.
        /// </summary>
        public cudnnDeterminism determinism;

        /// <summary>
        /// 
        /// </summary>
        public cudnnMathType mathType;
        int reserved1;
        int reserved2;
        int reserved3;
    }


	/// <summary>
	/// </summary>
	[StructLayout(LayoutKind.Explicit)]
	public struct cudnnAlgorithm
    {
        /// <summary>
        /// 
        /// </summary>
        [FieldOffset(0)]
        public cudnnConvolutionFwdAlgo convFwdAlgo;
        /// <summary>
        /// 
        /// </summary>
        [FieldOffset(0)]
        public cudnnConvolutionBwdFilterAlgo convBwdFilterAlgo;
        /// <summary>
        /// 
        /// </summary>
        [FieldOffset(0)]
        public cudnnConvolutionBwdDataAlgo convBwdDataAlgo;
        /// <summary>
        /// 
        /// </summary>
        [FieldOffset(0)]
        public cudnnRNNAlgo RNNAlgo;
        /// <summary>
        /// 
        /// </summary>
        [FieldOffset(0)]
        public cudnnCTCLossAlgo CTCLossAlgo;
    }


    /// <summary>
    /// 
    /// </summary>
    public struct cudnnDebug
    {
        /// <summary>
        /// 
        /// </summary>
        public uint cudnn_version;
        /// <summary>
        /// 
        /// </summary>
        public cudnnStatus cudnnStatus;
        /// <summary>
        /// epoch time in seconds
        /// </summary>
        public uint time_sec;              
        /// <summary>
        /// microseconds part of epoch time
        /// </summary>
        public uint time_usec;             
        /// <summary>
        /// time since start in seconds
        /// </summary>
        public uint time_delta;           
        /// <summary>
        /// cudnn handle 
        /// </summary>
        public cudnnHandle handle;           
        /// <summary>
        /// cuda stream ID
        /// </summary>
        public CUstream stream;           
        /// <summary>
        /// process ID
        /// </summary>
        public ulong pid;      
        /// <summary>
        /// thread ID
        /// </summary>
        public ulong tid;       
        /// <summary>
        /// CUDA device ID
        /// </summary>
        public int cudaDeviceId;             
        int reserved0;               /* reserved for future use */
        int reserved1;               /* reserved for future use */
        int reserved2;               /* reserved for future use */
        int reserved3;               /* reserved for future use */
        int reserved4;               /* reserved for future use */
        int reserved5;               /* reserved for future use */
        int reserved6;               /* reserved for future use */
        int reserved7;               /* reserved for future use */
        int reserved8;               /* reserved for future use */
        int reserved9;               /* reserved for future use */
        int reserved10;               /* reserved for future use */
        int reserved11;               /* reserved for future use */
        int reserved12;               /* reserved for future use */
        int reserved13;               /* reserved for future use */
        int reserved14;               /* reserved for future use */
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

    /// <summary>
    /// cudnnSpatialTransformerDescriptor_t is a pointer to an opaque structure 
    /// holding the description of a spatial transformation operation. 
    /// cudnnCreateSpatialTransformerDescriptor() is used to create one instance, 
    /// cudnnSetSpatialTransformerNdDescriptor() is used to initialize this instance, 
    /// cudnnDestroySpatialTransformerDescriptor() is used to destroy this instance.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cudnnSpatialTransformerDescriptor
    {
        private IntPtr Pointer;
    }

    /// <summary>
    /// cudnnOpTensorDescriptor is a pointer to an opaque structure holding the 
    /// description of a tensor operation, used as a parameter to cudnnOpTensor(). 
    /// cudnnCreateOpTensorDescriptor() is used to create one instance, and 
    /// cudnnSetOpTensorDescriptor() must be used to initialize this instance.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cudnnOpTensorDescriptor
    {
        private IntPtr Pointer;
    }

    /// <summary>
    /// cudnnDropoutDescriptor_t is a pointer to an opaque structure holding the description of a dropout operation. 
    /// cudnnCreateDropoutDescriptor() is used to create one instance, cudnnSetDropoutDescriptor() is be used to 
    /// initialize this instance, cudnnDestroyDropoutDescriptor() is be used to destroy this instance.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cudnnDropoutDescriptor
    {
        private IntPtr Pointer;
    }

    /// <summary>
    /// cudnnRNNDescriptor_t is a pointer to an opaque structure holding the description of an RNN operation. 
    /// cudnnCreateRNNDescriptor() is used to create one instance, and cudnnSetRNNDescriptor() must be used to 
    /// initialize this instance.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cudnnRNNDescriptor
    {
        private IntPtr Pointer;
    }

    /// <summary>
    /// cudnnReduceTensorDescriptor_t is a pointer to an opaque structure
    /// holding the description of a tensor reduction operation, used as a parameter to
    /// cudnnReduceTensor(). cudnnCreateReduceTensorDescriptor() is used to create
    /// one instance, and cudnnSetReduceTensorDescriptor() must be used to initialize this instance.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cudnnReduceTensorDescriptor
    {
        private IntPtr Pointer;
    }

    /// <summary>
    /// cudnnPersistentRNNPlan_t is a pointer to an opaque structure holding a plan to
    /// execute a dynamic persistent RNN.cudnnCreatePersistentRNNPlan() is used to
    /// create and initialize one instance.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cudnnPersistentRNNPlan
    {
        private IntPtr Pointer;
    }

    /// <summary>
    /// Forward definition in this version only
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cudnnRuntimeTag
    {
        private IntPtr Pointer;
    }

    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cudnnCTCLossDescriptor
    {
        private IntPtr Pointer;
    }

    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cudnnAlgorithmDescriptor
    {
        private IntPtr Pointer;
    }

    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cudnnAlgorithmPerformance
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
		LicenseError = 10,
        /// <summary>
        /// 
        /// </summary>
        RuntimePrerequisiteMissing = 11,
        /// <summary>
        /// 
        /// </summary>
        RuntimInProgress = 12,
        /// <summary>
        /// 
        /// </summary>
        RuntimeFPOverflow = 13
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
		Half = 2,
        /// <summary>
        /// The data is 8-bit signed integer.
        /// </summary>
        Int8 = 3,
        /// <summary>
        /// The data is 32-bit signed integer.
        /// </summary>
        Int32 = 4,
        /// <summary>
        /// The data is 32-bit element composed of 4 8-bit signed integer. This data type
        /// is only supported with tensor format CUDNN_TENSOR_NCHW_VECT_C.
        /// </summary>
        Int8x4 = 5,
        /// <summary>
        /// The data is 8-bit unsigned integer.
        /// </summary>
        UInt8 = 6,
        /// <summary>
        /// The data is 32-bit element composed of 4 8-bit unsigned integer. This data type
        /// is only supported with tensor format CUDNN_TENSOR_NCHW_VECT_C.
        /// </summary>
        UInt8x4 = 7,
    }

    /// <summary>
	/// cudnnNanPropagation is an enumerated type for the NanPropagation flag.
	/// </summary>
	public enum cudnnNanPropagation
    {
        /// <summary>
        /// Selects the not propagate NaN option.
        /// </summary>
        NotPropagateNan = 0,
        /// <summary>
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
		NHWC = 1,    /* feature maps interleaved ( cStride = 1 )*/
        /// <summary>
        /// This tensor format specifies that the data is laid out in the following order: batch size, feature
        /// maps, rows, columns. However, each element of the tensor is a vector of multiple feature
        /// maps. The length of the vector is carried by the data type of the tensor. The strides are
        /// implicitly defined in such a way that the data are contiguous in memory with no padding
        /// between images, feature maps, rows, and columns; the columns are the inner dimension
        /// and the images are the outermost dimension. This format is only supported with tensor data type
        /// CUDNN_DATA_INT8x4.
        /// </summary>
        NCHW_VECT_C = 2    /* each image point is vector of element of C : the length of the vector is carried by the data type*/
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
        FFTWithTiling = 5,

        /// <summary>
        /// This algorithm uses a Winograd Transform approach to compute the convolution. A reasonably 
        /// sized workspace is needed to store intermediate results.
        /// </summary>
        Winograd = 6
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
        ClippedRelu = 3,
        /// <summary>
        /// Selects the exponential linear function
        /// </summary>
        Elu = 4,
        /// <summary>
        /// Selects the identity function, intended for bypassing the activation step in cudnnConvolutionBiasActivationForward() (need to use CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM). Does not work with cudnnActivationForward() or cudnnActivationBackward().
        /// </summary>
        Identity = 5
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
		Algo3 = 3,   // non-deterministic, algo0 with workspace
        /// <summary>
        /// Not implemented
        /// </summary>
        AlgoWinograd = 4,  // not implemented
        /// <summary>
        /// This algorithm uses the Winograd Transform
        /// approach to compute the convolution. Significant
        /// workspace may be needed to store intermediate
        /// results. The results are deterministic.
        /// </summary>
        AlgoWinogradNonFused = 5,
        /// <summary>
        /// This algorithm uses the Fast-Fourier Transform
        /// approach to compute the convolution but splits
        /// the input tensor into tiles. Significant workspace
        /// may be needed to store intermediate results. The
        /// results are deterministic.
        /// </summary>
        AlgoFFTTiling = 6

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
		AlgoFFT = 2,

        /// <summary>
        /// This algorithm uses a Winograd Transform approach to compute the convolution. 
        /// A reasonably sized workspace is needed to store intermediate results. The results are deterministic.
        /// </summary>
        Winograd = 3
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
    /// cudnnBatchNormMode is an enumerated type used to specify the mode of operation in 
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
        BatchNormSpatial = 1,
        /// <summary>
        /// bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized over Nx1xHxW subtensors). 
        /// May be faster than CUDNN_BATCHNORM_SPATIAL but imposes some limits on the range of values
        /// </summary>
        BatchNormSpatialPersitent = 2
    }

    /// <summary>
    /// cudnnOpTensorOp is an enumerated type used to indicate the tensor operation to be used 
    /// by the cudnnOpTensor() routine. This enumerated type is used as a field for the 
    /// cudnnOpTensorDescriptor descriptor.
    /// </summary>
    public enum cudnnOpTensorOp
    {
        /// <summary>
        /// The operation to be performed is addition.
        /// </summary>
        OpTensorAdd = 0,
        /// <summary>
        /// The operation to be performed is multiplication.
        /// </summary>
        OpTensorMul = 1,
        /// <summary>
        /// The operation to be performed is a minimum comparison.
        /// </summary>
        OpTensorMin = 2,
        /// <summary>
        /// The operation to be performed is a maximum comparison.
        /// </summary>
        OpTensorMax = 3,
        /// <summary>
        /// 
        /// </summary>
        OpTensorSqrt = 4,
        /// <summary>
        /// 
        /// </summary>
        OpTensorNot = 5,
    }

    /// <summary>
    /// cudnnSamplerType_t is an enumerated type passed to cudnnSetSpatialTransformerNdDescriptor() to 
    /// select the sampler type to be used by cudnnSpatialTfSamplerForward() and cudnnSpatialTfSamplerBackward().
    /// </summary>
    public enum cudnnSamplerType
    {
        /// <summary>
        /// Selects the bilinear sampler.
        /// </summary>
        SamplerBilinear = 0
    }

    /// <summary>
    /// cudnnRNNMode_t is an enumerated type used to specify the type of network used in the 
    /// cudnnRNNForwardInference(), cudnnRNNForwardTraining(), cudnnRNNBackwardData() and 
    /// cudnnRNNBackwardWeights() routines.
    /// </summary>
    public enum cudnnRNNMode
    {
        /// <summary>
        /// A single-gate recurrent neural network with a ReLU activation function. In the forward pass the output ht for a 
        /// given iteration can be computed from the recurrent input ht-1 and the previous layer input xt given matrices 
        /// W, R and biases bW, bR from the following equation:
        /// h_t = ReLU(W_i x_t + R_i h_(t-1) + b_Wi + b_Ri) 
        /// Where ReLU(x) = max(x, 0). 
        /// </summary>
        RNNRelu = 0, // Stock RNN with ReLu activation
        /// <summary>
        /// A single-gate recurrent neural network with a tanh activation function. In the forward pass the output ht 
        /// for a given iteration can be computed from the recurrent input ht-1 and the previous layer input xt given 
        /// matrices W, R and biases bW, bR from the following equation:
        /// h_t = tanh(W_i x_t + R_i h_(t-1) + b_Wi + b_Ri) 
        /// Where tanh is the hyperbolic tangent function.
        /// </summary>
        RNNTanh = 1, // Stock RNN with tanh activation
        /// <summary>
        /// A four-gate Long Short-Term Memory network with no peephole connections. In the forward pass the output ht 
        /// and cell output c_t for a given iteration can be computed from the recurrent input h_(t-1), the cell input c_(t-1)
        /// and the previous layer input x_t given matrices W, R and biases b_W, b_R from the following equations: 
        /// i_t = σ(W_i x_t + R_i h_(t-1) + b_Wi + b_Ri) 
        /// f_t = σ(W_f x_t + R_f h_(t-1) + b_Wf + b_Rf) 
        /// o_t = σ(W_o x_t + R_o h_(t-1) + b_Wo + b_Ro)
        /// c_'t = tanh(W_c x_t + R_c h_(t-1) + b_Wc + b_Rc) 
        /// c_t = f_t◦c_'(t-1) + i_t◦c_'t 
        /// h_t = o_t◦tanh(c_t)
        /// Where σ is the sigmoid operator: σ(x) = 1 / (1 + e^-x), ◦ represents a point-wise multiplication 
        /// and tanh is the hyperbolic tangent function. i_t, f_t, o_t, c_'t represent the input, forget, output 
        /// and new gates respectively. 
        /// </summary>
        LSTM = 2,     // LSTM with no peephole connections
        /// <summary>
        /// A three-gate network consisting of Gated Recurrent Units. In the forward pass the output ht 
        /// for a given iteration can be computed from the recurrent input ht-1 and the previous layer input 
        /// xt given matrices W, R and biases bW, bR from the following equations:
        /// i_t = σ(W_i x_t + R_i h_(t-1) + b_Wi + b_Ru)
        /// r_t = σ(W_r x_t + R_r h_(t-1) + b_Wr + b_Rr)
        /// h_'t = tanh(W_h x_t + r_t◦R_h h_(t-1) + b_Wh + b_Rh) 
        /// h_t = (1 - i_t◦h_'t) + i_t◦h_(t-1)
        /// Where σ is the sigmoid operator: σ(x) = 1 / (1 + e^-x), ◦ represents a point-wise multiplication 
        /// and tanh is the hyperbolic tangent function. i_t, r_t, h_'t represent the input, reset, new gates respectively.
        /// </summary>
        GRU = 3       // Using h' = tanh(r * Uh(t-1) + Wx) and h = (1 - z) * h' + z * h(t-1);
    }

    /// <summary>
    /// cudnnDirectionMode_t is an enumerated type used to specify the recurrence pattern in the cudnnRNNForwardInference(), 
    /// cudnnRNNForwardTraining(), cudnnRNNBackwardData() and cudnnRNNBackwardWeights() routines.
    /// </summary>
    public enum cudnnDirectionMode
    {
        /// <summary>
        /// The network iterates recurrently from the first input to the last.
        /// </summary>
        Unidirectional = 0,
        /// <summary>
        /// Each layer of the the network iterates recurrently from the first input to the last and separately 
        /// from the last input to the first. The outputs of the two are concatenated at each iteration giving 
        /// the output of the layer.
        /// </summary>
        Bidirectional = 1      // Using output concatination at each step. Do we also want to support output sum?
    }

    /// <summary>
    /// cudnnRNNInputMode_t is an enumerated type used to specify the behavior of the first 
    /// layer in the cudnnRNNForwardInference(), cudnnRNNForwardTraining(), 
    /// cudnnRNNBackwardData() and cudnnRNNBackwardWeights() routines.
    /// </summary>
    public enum cudnnRNNInputMode
    {
        /// <summary>
        /// A biased matrix multiplication is performed at the input of the first recurrent layer.
        /// </summary>
        LinearInput = 0,
        /// <summary>
        /// No operation is performed at the input of the first recurrent layer. If CUDNN_SKIP_INPUT 
        /// is used the leading dimension of the input tensor must be equal to the hidden state size 
        /// of the network.
        /// </summary>
        SkipInput = 1
    }

    /// <summary>
    /// 
    /// </summary>
    public enum libraryPropertyType
    {
        /// <summary>
        /// 
        /// </summary>
        MajorVersion,
        /// <summary>
        /// 
        /// </summary>
        MinorVersion,
        /// <summary>
        /// 
        /// </summary>
        PatchLevel
    }

    /// <summary>
    /// cudnnDeterminism_t is an enumerated type used to indicate if the computed results
    /// are deterministic(reproducible). See section 2.5 (Reproducibility) for more details on
    /// determinism.
    /// </summary>
    public enum cudnnDeterminism
    {
        /// <summary>
        /// Results are not guaranteed to be reproducible
        /// </summary>
        NonDeterministic = 0,
        /// <summary>
        /// Results are guaranteed to be reproducible
        /// </summary>
        Deterministic = 1,
    }

    /// <summary>
    /// cudnnReduceTensorOp is an enumerated type used to indicate the tensor operation
    /// to be used by the cudnnReduceTensor() routine.This enumerated type is used as a
    /// field for the cudnnReduceTensorDescriptor_t descriptor.
    /// </summary>
    public enum cudnnReduceTensorOp
    {
        /// <summary>
        /// The operation to be performed is addition
        /// </summary>
        Add = 0,
        /// <summary>
        /// The operation to be performed is multiplication
        /// </summary>
        Mul = 1,
        /// <summary>
        /// The operation to be performed is a minimum comparison
        /// </summary>
        Min = 2,
        /// <summary>
        /// The operation to be performed is a maximum comparison
        /// </summary>
        Max = 3,
        /// <summary>
        /// The operation to be performed is a maximum comparison of absolute values
        /// </summary>
        AMax = 4,
        /// <summary>
        /// The operation to be performed is averaging
        /// </summary>
        Avg = 5,
        /// <summary>
        /// The operation to be performed is addition of absolute values
        /// </summary>
        Norm1 = 6,
        /// <summary>
        /// The operation to be performed is a square root of sum of squares
        /// </summary>
        Norm2 = 7,
        /// <summary>
        /// 
        /// </summary>
        MulNoZeros = 8,
    }

    /// <summary>
    /// cudnnReduceTensorIndices_t is an enumerated type used to indicate whether
    /// indices are to be computed by the cudnnReduceTensor() routine.This enumerated
    /// type is used as a field for the cudnnReduceTensorDescriptor_t descriptor.
    /// </summary>
    public enum cudnnReduceTensorIndices
    {
        /// <summary>
        /// Do not compute indices
        /// </summary>
        NoIndices = 0,
        /// <summary>
        /// Compute indices. The resulting indices are relative, and flattened.
        /// </summary>
        FlattenedIndices = 1,
    }

    /// <summary>
    /// cudnnIndicesType_t is an enumerated type used to indicate the data type for the
    /// indices to be computed by the cudnnReduceTensor() routine. This enumerated type is
    /// used as a field for the cudnnReduceTensorDescriptor_t descriptor.
    /// </summary>
    public enum cudnnIndicesType
    {
        /// <summary>
        /// Compute unsigned int indices
        /// </summary>
        Indices32Bit = 0,
        /// <summary>
        /// Compute unsigned long long indices
        /// </summary>
        Indices64Bit = 1,
        /// <summary>
        /// Compute unsigned short indices
        /// </summary>
        Indices16Bit = 2,
        /// <summary>
        /// Compute unsigned char indices
        /// </summary>
        Indices8Bit = 3,
    }

    /// <summary>
    /// cudnnRNNAlgo_t is an enumerated type used to specify the algorithm used
    /// in the cudnnRNNForwardInference(), cudnnRNNForwardTraining(),
    /// cudnnRNNBackwardData() and cudnnRNNBackwardWeights() routines.
    /// </summary>
    public enum cudnnRNNAlgo
    {
        /// <summary>
        /// Each RNN layer is executed as a sequence of operations. This
        /// algorithm is expected to have robust performance across a wide
        /// range of network parameters.
        /// </summary>
        Standard = 0,
        /// <summary>
        /// The recurrent parts of the network are executed using a persistent
        /// kernel approach. This method is expected to be fast when the first
        /// dimension of the input tensor is small (ie. a small minibatch).
        /// CUDNN_RNN_ALGO_PERSIST_STATIC is only supported on devices
        /// with compute capability >= 6.0.
        /// </summary>
        PersistStatic = 1,
        /// <summary>
        /// The recurrent parts of the network are executed using a persistent
        /// kernel approach. This method is expected to be fast when the first
        /// dimension of the input tensor is small (ie. a small minibatch). When
        /// using CUDNN_RNN_ALGO_PERSIST_DYNAMIC persistent kernels are
        /// prepared at runtime and are able to optimized using the specific
        /// parameters of the network and active GPU.As such, when using
        /// CUDNN_RNN_ALGO_PERSIST_DYNAMIC a one-time plan preparation
        /// stage must be executed.These plans can then be reused in repeated
        /// calls with the same model parameters.<para/>
        /// The limits on the maximum number of hidden units
        /// supported when using CUDNN_RNN_ALGO_PERSIST_DYNAMIC
        /// are significantly higher than the limits when using
        /// CUDNN_RNN_ALGO_PERSIST_STATIC, however throughput is likely
        /// to significantly reduce when exceeding the maximums supported by
        /// CUDNN_RNN_ALGO_PERSIST_STATIC.In this regime this method will
        /// still outperform CUDNN_RNN_ALGO_STANDARD for some cases.<para/>
        /// CUDNN_RNN_ALGO_PERSIST_DYNAMIC is only supported on devices
        /// with compute capability >= 6.0 on Linux machines.
        /// </summary>
        PersistDynamic = 2,
        /// <summary>
        /// 
        /// </summary>
        Count = 3
    }

    /// <summary>
    /// 
    /// </summary>
    public enum cudnnErrQueryMode
    {
        /// <summary>
        /// 
        /// </summary>
        RawCode = 0,
        /// <summary>
        /// 
        /// </summary>
        NonBlocking = 1,
        /// <summary>
        /// 
        /// </summary>
        Blocking = 2
    }

    /// <summary>
    /// CUDNN math type
    /// </summary>
    public enum cudnnMathType
    {
        /// <summary>
        /// 
        /// </summary>
        Default = 0,
        /// <summary>
        /// 
        /// </summary>
        TensorOP = 1
    }

    /// <summary>
    /// 
    /// </summary>
    public enum cudnnCTCLossAlgo
    {
        /// <summary>
        /// 
        /// </summary>
        Deterministic = 0,
        /// <summary>
        /// 
        /// </summary>
        NonDeterministic = 1
    }

    /// <summary>
    /// 
    /// </summary>
    public enum cudnnSeverity
    {
        /// <summary>
        /// 
        /// </summary>
        Fatal = 0,
        /// <summary>
        /// 
        /// </summary>
        Error = 1,
        /// <summary>
        /// 
        /// </summary>
        Warning = 2,
        /// <summary>
        /// 
        /// </summary>
        Info = 3,
    }

    /// <summary>
    /// Message masks to be used with cudnnSetCallback()  
    /// </summary>
    [Flags]
    public enum MessageMask
    {
        /// <summary>
        /// 
        /// </summary>
        Error = 1 << (int)cudnnSeverity.Error,
        /// <summary>
        /// 
        /// </summary>
        Warning = 1 << (int)cudnnSeverity.Warning,
        /// <summary>
        /// 
        /// </summary>
        Info = 1 << (int)cudnnSeverity.Info        
    }

    #endregion
}
