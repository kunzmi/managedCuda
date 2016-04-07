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
using ManagedCuda.VectorTypes;

namespace ManagedCuda.CudaDNNv4
{
	/// <summary/>
	public static class CudaDNNNativeMethods
	{
		internal const string CUDNN_API_DLL_NAME = "cudnn64_4.dll";
		/// <summary>
		/// Gives the version of the wrapped api
		/// </summary>
		public static Version Version
		{
			get { return new Version(4, 0, 7); }
		}

		[DllImport(CUDNN_API_DLL_NAME, EntryPoint = "cudnnGetVersion")]
		internal static extern SizeT cudnnGetVersionInternal();
		/// <summary>
		/// This function returns the version number of the cuDNN Library. It returns the
		/// CUDNN_VERSION define present in the cudnn.h header file. Starting with release R2, the
		/// routine can be used to identify dynamically the current cuDNN Library used by the
		/// application. The define CUDNN_VERSION can be used to have the same application linked
		/// against different cuDNN versions using conditional compilation statements.
		/// </summary>
		public static Version cudnnGetVersion()
		{
			SizeT ver = cudnnGetVersionInternal();
			SizeT maj = ver / 100;
			SizeT min = ver % 100;
			return new Version(maj, min);
		}



		// human-readable error messages
		[DllImport(CUDNN_API_DLL_NAME, EntryPoint = "cudnnGetErrorString")]
		internal static extern IntPtr cudnnGetErrorStringInternal(cudnnStatus status);
		/// <summary>
		/// This function returns a human-readable character string describing the cudnnStatus enumerate passed as input parameter.
		/// </summary>
		public static string cudnnGetErrorString(cudnnStatus status)
		{
			IntPtr str = cudnnGetErrorStringInternal(status);
			return Marshal.PtrToStringAnsi(str);
		}
		
		/// <summary>
		/// This function initializes the cuDNN library and creates a handle to an opaque
		/// structure holding the cuDNN library context. It allocates hardware resources on
		/// the host and device and must be called prior to making any other cuDNN library
		/// calls. The cuDNN library context is tied to the current CUDA device. To use the
		/// library on multiple devices, one cuDNN handle needs to be created for each device.
		/// For a given device, multiple cuDNN handles with different configurations (e.g.,
		/// different current CUDA streams) may be created. Because cudnnCreate allocates
		/// some internal resources, the release of those resources by calling cudnnDestroy will
		/// implicitly call cudaDeviceSynchronize; therefore, the recommended best practice
		/// is to call cudnnCreate/cudnnDestroy outside of performance-critical code paths.
		/// For multithreaded applications that use the same device from different threads, the
		/// </summary>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnCreate(ref cudnnHandle handle);
		
		/// <summary>
		/// This function releases hardware resources used by the cuDNN library. This function
		/// is usually the last call with a particular handle to the cuDNN library. Because
		/// cudnnCreate allocates some internal resources, the release of those resources by
		/// calling cudnnDestroy will implicitly call cudaDeviceSynchronize; therefore,
		/// the recommended best practice is to call cudnnCreate/cudnnDestroy outside of
		/// performance-critical code paths.
		/// </summary>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnDestroy(cudnnHandle handle);
		
		/// <summary>
		/// This function sets the cuDNN library stream, which will be used to execute all
		/// subsequent calls to the cuDNN library functions with that particular handle. If the
		/// cuDNN library stream is not set, all kernels use the default (NULL) stream. In particular,
		/// this routine can be used to change the stream between kernel launches and then to reset
		/// the cuDNN library stream back to NULL.
		/// </summary>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSetStream(cudnnHandle handle, CUstream streamId);

		/// <summary>
		/// This function gets the cuDNN library stream, which is being used to execute all calls to
		/// the cuDNN library functions. If the cuDNN library stream is not set, all kernels use the
		/// default NULL stream.
		/// </summary>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetStream(cudnnHandle handle, ref CUstream streamId);




		/// <summary>
		/// This function creates a generic Tensor descriptor object by allocating the memory needed
		/// to hold its opaque structure. The data is initialized to be all zero.
		/// </summary>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnCreateTensorDescriptor( ref cudnnTensorDescriptor tensorDesc );

		/// <summary>
		/// This function initializes a previously created generic Tensor descriptor object into a
		/// 4D tensor. The strides of the four dimensions are inferred from the format parameter
		/// and set in such a way that the data is contiguous in memory with no padding between
		/// dimensions.
		/// </summary>
		/// <param name="tensorDesc">Handle to a previously created tensor descriptor.</param>
		/// <param name="format">Type of format.</param>
		/// <param name="dataType">Data type.</param>
		/// <param name="n">Number of images.</param>
		/// <param name="c">Number of feature maps per image.</param>
		/// <param name="h">Height of each feature map.</param>
		/// <param name="w">Width of each feature map.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSetTensor4dDescriptor(cudnnTensorDescriptor tensorDesc,
																cudnnTensorFormat  format,
																cudnnDataType dataType, // image data type
																int n,        // number of inputs (batch size)
																int c,        // number of input feature maps
																int h,        // height of input section
																int w         // width of input section
															);

		/// <summary>
		/// This function initializes a previously created generic Tensor descriptor object into a
		/// 4D tensor, similarly to cudnnSetTensor4dDescriptor but with the strides explicitly
		/// passed as parameters. This can be used to lay out the 4D tensor in any order or simply to
		/// define gaps between dimensions.
		/// </summary>
		/// <param name="tensorDesc">Handle to a previously created tensor descriptor.</param>
		/// <param name="dataType">Data type.</param>
		/// <param name="n">Number of images.</param>
		/// <param name="c">Number of feature maps per image.</param>
		/// <param name="h">Height of each feature map.</param>
		/// <param name="w">Width of each feature map.</param>
		/// <param name="nStride">Stride between two consecutive images.</param>
		/// <param name="cStride">Stride between two consecutive feature maps.</param>
		/// <param name="hStride">Stride between two consecutive rows.</param>
		/// <param name="wStride">Stride between two consecutive columns.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSetTensor4dDescriptorEx( cudnnTensorDescriptor tensorDesc,
																cudnnDataType dataType, // image data type
																int n,        // number of inputs (batch size)
																int c,        // number of input feature maps
																int h,        // height of input section
																int w,        // width of input section
																int nStride,
																int cStride,
																int hStride,
																int wStride
															  );
		
		/// <summary>
		/// This function queries the parameters of the previouly initialized Tensor4D descriptor object.
		/// </summary>
		/// <param name="tensorDesc">Handle to a previously insitialized tensor descriptor.</param>
		/// <param name="dataType">Data type.</param>
		/// <param name="n">Number of images.</param>
		/// <param name="c">Number of feature maps per image.</param>
		/// <param name="h">Height of each feature map.</param>
		/// <param name="w">Width of each feature map.</param>
		/// <param name="nStride">Stride between two consecutive images.</param>
		/// <param name="cStride">Stride between two consecutive feature maps.</param>
		/// <param name="hStride">Stride between two consecutive rows.</param>
		/// <param name="wStride">Stride between two consecutive columns.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetTensor4dDescriptor(   cudnnTensorDescriptor tensorDesc,
																ref cudnnDataType dataType, // image data type
																ref int n,        // number of inputs (batch size)
																ref int c,        // number of input feature maps
																ref int h,        // height of input section
																ref int w,        // width of input section
																ref int nStride,
																ref int cStride,
																ref int hStride,
																ref int wStride
															);
		
		/// <summary>
		/// This function initializes a previously created generic Tensor descriptor object.
		/// </summary>
		/// <param name="tensorDesc">Handle to a previously created tensor descriptor.</param>
		/// <param name="dataType">Data type.</param>
		/// <param name="nbDims">Dimension of the tensor.</param>
		/// <param name="dimA">Array of dimension nbDims that contain the size of the tensor for every dimension.</param>
		/// <param name="strideA">Array of dimension nbDims that contain the stride of the tensor for every dimension.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSetTensorNdDescriptor(cudnnTensorDescriptor tensorDesc,
															   cudnnDataType dataType,
															   int nbDims,
															   int[] dimA,
															   int[] strideA
															 );
		
		/// <summary>
		/// This function retrieves values stored in a previously initialized Tensor descriptor object.
		/// </summary>
		/// <param name="tensorDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="nbDimsRequested">Number of dimensions to extract from a given tensor descriptor. It is
		/// also the minimum size of the arrays dimA and strideA. If this number is
		/// greater than the resulting nbDims[0], only nbDims[0] dimensions will be
		/// returned.</param>
		/// <param name="dataType">Data type.</param>
		/// <param name="nbDims">Actual number of dimensions of the tensor will be returned in nbDims[0].</param>
		/// <param name="dimA">Array of dimension of at least nbDimsRequested that will be filled with
		/// the dimensions from the provided tensor descriptor.</param>
		/// <param name="strideA">Array of dimension of at least nbDimsRequested that will be filled with
		/// the strides from the provided tensor descriptor.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetTensorNdDescriptor(  cudnnTensorDescriptor tensorDesc,
															   int nbDimsRequested,
															   ref cudnnDataType dataType,
															   ref int nbDims,
															   int[] dimA,
															   int[] strideA
															 );



		/// <summary>
		/// This function destroys a previously created Tensor descriptor object.
		/// </summary>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnDestroyTensorDescriptor( cudnnTensorDescriptor tensorDesc );


		/// <summary>
		/// This function copies the scaled data from one tensor to another tensor with a different
		/// layout. Those descriptors need to have the same dimensions but not necessarily the
		/// same strides. The input and output tensors must not overlap in any way (i.e., tensors
		/// cannot be transformed in place). This function can be used to convert a tensor with an
		/// unsupported format to a supported one.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the source
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*srcValue + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="srcDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="srcData">Pointer to data of the tensor described by the srcDesc descriptor.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the source
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*srcValue + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="destDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="destData">Pointer to data of the tensor described by the destDesc descriptor.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnTransformTensor(   cudnnHandle                    handle,
														  ref float alpha,
														  cudnnTensorDescriptor    srcDesc,
														  CUdeviceptr srcData,
														  ref float beta,
														  cudnnTensorDescriptor    destDesc,
														  CUdeviceptr destData
														);

		/// <summary>
		/// This function copies the scaled data from one tensor to another tensor with a different
		/// layout. Those descriptors need to have the same dimensions but not necessarily the
		/// same strides. The input and output tensors must not overlap in any way (i.e., tensors
		/// cannot be transformed in place). This function can be used to convert a tensor with an
		/// unsupported format to a supported one.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the source
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*srcValue + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="srcDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="srcData">Pointer to data of the tensor described by the srcDesc descriptor.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the source
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*srcValue + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="destDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="destData">Pointer to data of the tensor described by the destDesc descriptor.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnTransformTensor(   cudnnHandle                    handle,
														  ref double alpha,
														  cudnnTensorDescriptor    srcDesc,
														  CUdeviceptr srcData,
														  ref double beta,
														  cudnnTensorDescriptor    destDesc,
														  CUdeviceptr destData
														);

		/// <summary>
		/// This function adds the scaled values of one bias tensor to another tensor. Each dimension
		/// of the bias tensor must match the coresponding dimension of the srcDest tensor or
		/// must be equal to 1. In the latter case, the same value from the bias tensor for thoses
		/// dimensions will be used to blend into the srcDest tensor.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the source
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*srcValue + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="biasDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="biasData">Pointer to data of the tensor described by the biasDesc descriptor.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the source
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*srcValue + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="srcDestDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="srcDestData">Pointer to data of the tensor described by the srcDestDesc descriptor.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnAddTensor(cudnnHandle                    handle,
                                            ref float alpha,
                                            cudnnTensorDescriptor biasDesc,
                                            CUdeviceptr biasData,
											ref float beta,
                                            cudnnTensorDescriptor srcDestDesc,
											CUdeviceptr srcDestData
                                          );

		/// <summary>
		/// This function adds the scaled values of one bias tensor to another tensor. Each dimension
		/// of the bias tensor must match the coresponding dimension of the srcDest tensor or
		/// must be equal to 1. In the latter case, the same value from the bias tensor for thoses
		/// dimensions will be used to blend into the srcDest tensor.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the source
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*srcValue + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="biasDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="biasData">Pointer to data of the tensor described by the biasDesc descriptor.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the source
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*srcValue + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="srcDestDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="srcDestData">Pointer to data of the tensor described by the srcDestDesc descriptor.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnAddTensor(cudnnHandle handle,
											ref double alpha,
											cudnnTensorDescriptor biasDesc,
											CUdeviceptr biasData,
											ref double beta,
											cudnnTensorDescriptor srcDestDesc,
											CUdeviceptr srcDestData
										  );


		/// <summary>
		/// This function sets all the elements of a tensor to a given value
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="srcDestDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="srcDestData">Pointer to data of the tensor described by the srcDestDesc descriptor.</param>
		/// <param name="value">Pointer in Host memory to a value that all elements of the tensor will be set to.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSetTensor( cudnnHandle                   handle,
												  cudnnTensorDescriptor   srcDestDesc,
												  CUdeviceptr srcDestData,
												  ref float value
												 );

        /// <summary>
        /// This function sets all the elements of a tensor to a given value
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="srcDestDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="srcDestData">Pointer to data of the tensor described by the srcDestDesc descriptor.</param>
        /// <param name="value">Pointer in Host memory to a value that all elements of the tensor will be set to.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSetTensor(cudnnHandle handle,
                                                  cudnnTensorDescriptor srcDestDesc,
                                                  CUdeviceptr srcDestData,
                                                  ref double value
                                                 );

        /// <summary>
        /// This function scale all the elements of a tensor by a give factor.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="srcDestDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="srcDestData">Pointer to data of the tensor described by the srcDestDesc descriptor.</param>
        /// <param name="alpha">Pointer in Host memory to a value that all elements of the tensor will be scaled with.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnScaleTensor(   cudnnHandle                    handle,
													  cudnnTensorDescriptor    srcDestDesc,
													  CUdeviceptr srcDestData,
													  ref float alpha
												  );

		/// <summary>
		/// This function scale all the elements of a tensor by a give factor.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="srcDestDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="srcDestData">Pointer to data of the tensor described by the srcDestDesc descriptor.</param>
		/// <param name="alpha">Pointer in Host memory to a value that all elements of the tensor will be scaled with.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnScaleTensor(   cudnnHandle                    handle,
													  cudnnTensorDescriptor    srcDestDesc,
													  CUdeviceptr srcDestData,
													  ref double alpha
												  );




		/// <summary>
		/// This function creates a filter descriptor object by allocating the memory needed to hold its opaque structure.
		/// </summary>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnCreateFilterDescriptor( ref cudnnFilterDescriptor filterDesc );

        /// <summary>
        /// This function initializes a previously created filter descriptor object into a 4D filter.
        /// Filters layout must be contiguous in memory.
        /// v4 version of the function also has the format parameter.
        /// </summary>
        /// <param name="filterDesc">Handle to a previously created filter descriptor.</param>
        /// <param name="dataType">Data type.</param>
        /// <param name="k">Number of output feature maps.</param>
        /// <param name="c">Number of input feature maps.</param>
        /// <param name="h">Height of each filter.</param>
        /// <param name="w">Width of each filter.</param>
        [DllImport(CUDNN_API_DLL_NAME, EntryPoint = "cudnnSetFilter4dDescriptor_v4")]
        public static extern cudnnStatus cudnnSetFilter4dDescriptor(cudnnFilterDescriptor filterDesc,
                                                               cudnnDataType dataType, // image data type
                                                               cudnnTensorFormat format, // layout format
                                                               int k,        // number of output feature maps
                                                               int c,        // number of input feature maps
                                                               int h,        // height of each input filter
                                                               int w         // width of  each input fitler
                                                          );

        /// <summary>
        /// This function queries the parameters of the previouly initialized filter descriptor object.
        /// v4 version of the function also has the format parameter.
        /// </summary>
        /// <param name="filterDesc">Handle to a previously created filter descriptor.</param>
        /// <param name="dataType">Data type.</param>
        /// <param name="format">Layout format.</param>
        /// <param name="k">Number of output feature maps.</param>
        /// <param name="c">Number of input feature maps.</param>
        /// <param name="h">Height of each filter.</param>
        /// <param name="w">Width of each filter.</param>
        [DllImport(CUDNN_API_DLL_NAME, EntryPoint = "cudnnGetFilter4dDescriptor_v4")]
        public static extern cudnnStatus cudnnGetFilter4dDescriptor(cudnnFilterDescriptor filterDesc,
                                                               ref cudnnDataType dataType, // image data type
                                                               ref cudnnTensorFormat format, // layout format
                                                               ref int k,        // number of output feature maps
                                                               ref int c,        // number of input feature maps
                                                               ref int h,        // height of each input filter
                                                               ref int w         // width of  each input fitler
                                                          );


        /// <summary>
        /// This function initializes a previously created filter descriptor object. Filters layout must
        /// be contiguous in memory.
        /// v4 version of the function also has the format parameter.
        /// </summary>
        /// <param name="filterDesc">Handle to a previously created filter descriptor.</param>
        /// <param name="dataType">Data type.</param>
        /// <param name="nbDims">Dimension of the filter.</param>
        /// <param name="filterDimA">Array of dimension nbDims containing the size of the filter for each dimension.</param>
        [DllImport(CUDNN_API_DLL_NAME, EntryPoint = "cudnnSetFilterNdDescriptor_v4")]
        public static extern cudnnStatus cudnnSetFilterNdDescriptor(cudnnFilterDescriptor filterDesc,
                                                               cudnnDataType dataType, // image data type
                                                               cudnnTensorFormat format, // layout format
                                                               int nbDims,
                                                               int[] filterDimA
                                                             );

        /// <summary>
        /// This function queries a previously initialized filter descriptor object.
        /// v4 version of the function also has the format parameter.
        /// </summary>
        /// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="nbDimsRequested">Dimension of the expected filter descriptor. It is also the minimum size of
        /// the arrays filterDimA in order to be able to hold the results</param>
        /// <param name="dataType">Data type.</param>
        /// <param name="nbDims">Actual dimension of the filter.</param>
        /// <param name="filterDimA">Array of dimension of at least nbDimsRequested that will be filled with
        /// the filter parameters from the provided filter descriptor.</param>
        [DllImport(CUDNN_API_DLL_NAME, EntryPoint = "cudnnGetFilterNdDescriptor_v4")]
        public static extern cudnnStatus cudnnGetFilterNdDescriptor(cudnnFilterDescriptor filterDesc,
                                                               int nbDimsRequested,
                                                               ref cudnnDataType dataType, // image data type
                                                               ref cudnnTensorFormat format, // layout format
                                                               ref int nbDims,
                                                               int[] filterDimA
                                                            );

        /// <summary>
        /// This function destroys a previously created Tensor4D descriptor object.
        /// </summary>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnDestroyFilterDescriptor( cudnnFilterDescriptor filterDesc );


		/// <summary>
		/// This function creates a convolution descriptor object by allocating the memory needed to
		/// hold its opaque structure
		/// </summary>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnCreateConvolutionDescriptor(ref cudnnConvolutionDescriptor convDesc );

		/// <summary>
		/// This function initializes a previously created convolution descriptor object into a 2D
		/// correlation. This function assumes that the tensor and filter descriptors corresponds
		/// to the forward convolution path and checks if their settings are valid. That same
		/// convolution descriptor can be reused in the backward path provided it corresponds to
		/// the same layer.
		/// </summary>
		/// <param name="convDesc">Handle to a previously created convolution descriptor.</param>
		/// <param name="pad_h">zero-padding height: number of rows of zeros implicitly concatenated
		/// onto the top and onto the bottom of input images.</param>
		/// <param name="pad_w">zero-padding width: number of columns of zeros implicitly concatenated
		/// onto the left and onto the right of input images.</param>
		/// <param name="u">Vertical filter stride.</param>
		/// <param name="v">Horizontal filter stride.</param>
		/// <param name="upscalex">Upscale the input in x-direction.</param>
		/// <param name="upscaley">Upscale the input in y-direction.</param>
		/// <param name="mode">Selects between CUDNN_CONVOLUTION and CUDNN_CROSS_CORRELATION.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSetConvolution2dDescriptor(  cudnnConvolutionDescriptor convDesc,
																	int pad_h,    // zero-padding height
																	int pad_w,    // zero-padding width
																	int u,        // vertical filter stride
																	int v,        // horizontal filter stride
																	int upscalex, // upscale the input in x-direction
																	int upscaley, // upscale the input in y-direction
																	cudnnConvolutionMode mode
																 );

        /// <summary>
        /// This function queries a previously initialized 2D convolution descriptor object.
        /// </summary>
        /// <param name="convDesc">Handle to a previously created convolution descriptor.</param>
        /// <param name="pad_h">zero-padding height: number of rows of zeros implicitly concatenated
        /// onto the top and onto the bottom of input images.</param>
        /// <param name="pad_w">zero-padding width: number of columns of zeros implicitly concatenated
        /// onto the left and onto the right of input images.</param>
        /// <param name="u">Vertical filter stride.</param>
        /// <param name="v">Horizontal filter stride.</param>
        /// <param name="upscalex">Upscale the input in x-direction.</param>
        /// <param name="upscaley">Upscale the input in y-direction.</param>
        /// <param name="mode">Selects between CUDNN_CONVOLUTION and CUDNN_CROSS_CORRELATION.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetConvolution2dDescriptor(   cudnnConvolutionDescriptor convDesc,
																	 ref int pad_h,    // zero-padding height
																	 ref int pad_w,    // zero-padding width
																	 ref int u,        // vertical filter stride
																	 ref int v,        // horizontal filter stride
																	 ref int upscalex, // upscale the input in x-direction
																	 ref int upscaley, // upscale the input in y-direction
																	 ref cudnnConvolutionMode mode
																  );

		/// <summary>
		/// This function returns the dimensions of the resulting 4D tensor of a 2D convolution,
		/// given the convolution descriptor, the input tensor descriptor and the filter descriptor
		/// This function can help to setup the output tensor and allocate the proper amount of
		/// memory prior to launch the actual convolution.<para/>
		/// Each dimension h and w of the output images is computed as followed:<para/>
		/// outputDim = 1 + (inputDim + 2*pad - filterDim)/convolutionStride;
		/// </summary>
		/// <param name="convDesc">Handle to a previously created convolution descriptor.</param>
		/// <param name="inputTensorDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="n">Number of output images.</param>
		/// <param name="c">Number of output feature maps per image.</param>
		/// <param name="h">Height of each output feature map.</param>
		/// <param name="w">Width of each output feature map.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetConvolution2dForwardOutputDim( cudnnConvolutionDescriptor convDesc,
																		 cudnnTensorDescriptor     inputTensorDesc,
																		 cudnnFilterDescriptor     filterDesc,
																		 ref int n,
																		 ref int c,
																		 ref int h,
																		 ref int w
																		);

		/// <summary>
		/// This function initializes a previously created generic convolution descriptor object into
		/// a n-D correlation. That same convolution descriptor can be reused in the backward path
		/// provided it corresponds to the same layer. The convolution computation will done in the
		/// specified dataType, which can be potentially different from the input/output tensors.
		/// </summary>
		/// <param name="convDesc">Handle to a previously created convolution descriptor.</param>
		/// <param name="arrayLength">Dimension of the convolution.</param>
		/// <param name="padA">Array of dimension arrayLength containing the zero-padding size
		/// for each dimension. For every dimension, the padding represents the
		/// number of extra zeros implicitly concatenated at the start and at the
		/// end of every element of that dimension.</param>
		/// <param name="filterStrideA">Array of dimension arrayLength containing the filter stride for each
		/// dimension. For every dimension, the fitler stride represents the number
		/// of elements to slide to reach the next start of the filtering window of
		/// the next point.</param>
		/// <param name="upscaleA">Array of dimension arrayLength containing the upscale factor for each dimension.</param>
		/// <param name="mode">Selects between CUDNN_CONVOLUTION and CUDNN_CROSS_CORRELATION.</param>
		/// <param name="dataType">Selects the datatype in which the computation will be done.</param>
		[DllImport(CUDNN_API_DLL_NAME)]  
		public static extern cudnnStatus cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor convDesc,
                                                              int arrayLength,             /* nbDims-2 size */  
                                                              int[] padA,                                          
                                                              int[] filterStrideA,         
                                                              int[] upscaleA,              
                                                              cudnnConvolutionMode mode,
                                                              cudnnDataType dataType   // convolution data type
                                                         );
                                           
        /// <summary>
		/// This function queries a previously initialized convolution descriptor object.
        /// </summary>
		/// <param name="convDesc">Handle to a previously created convolution descriptor.</param>
        /// <param name="arrayLengthRequested">Dimension of the expected convolution descriptor. It is also the
		/// minimum size of the arrays padA, filterStrideA and upsacleA in
		/// order to be able to hold the results</param>
		/// <param name="arrayLength">actual dimension of the convolution descriptor.</param>
        /// <param name="padA">Array of dimension of at least arrayLengthRequested that will be
		/// filled with the padding parameters from the provided convolution
		/// descriptor.</param>
        /// <param name="strideA">Array of dimension of at least arrayLengthRequested that will be
		/// filled with the filter stride from the provided convolution descriptor.</param>
        /// <param name="upscaleA">Array of dimension at least arrayLengthRequested that will be filled
		/// with the upscaling parameters from the provided convolution descriptor.</param>
		/// <param name="mode">convolution mode of the provided descriptor.</param>
		/// <param name="dataType">datatype of the provided descriptor.</param>
		[DllImport(CUDNN_API_DLL_NAME)]  
		public static extern cudnnStatus cudnnGetConvolutionNdDescriptor(cudnnConvolutionDescriptor convDesc,
                                                              int arrayLengthRequested,
                                                              ref int arrayLength,
                                                              int[] padA,                                        
                                                              int[] strideA,
                                                              int[] upscaleA,
                                                              ref cudnnConvolutionMode mode,
                                                              ref cudnnDataType dataType     // convolution data type
                                                         );




		/// <summary>
		/// This function returns the dimensions of the resulting n-D tensor of a nbDims-2-D
		/// convolution, given the convolution descriptor, the input tensor descriptor and the filter
		/// descriptor This function can help to setup the output tensor and allocate the proper
		/// amount of memory prior to launch the actual convolution.<para/>
		/// Each dimension of the (nbDims-2)-D images of the output tensor is computed as
		/// followed:<para/>
		/// outputDim = 1 + (inputDim + 2*pad - filterDim)/convolutionStride;
		/// </summary>
		/// <param name="convDesc">Handle to a previously created convolution descriptor.</param>
		/// <param name="inputTensorDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="nbDims">Dimension of the output tensor</param>
		/// <param name="tensorOuputDimA">Array of dimensions nbDims that contains on exit of this routine the sizes
		/// of the output tensor</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetConvolutionNdForwardOutputDim( cudnnConvolutionDescriptor convDesc,
																		 cudnnTensorDescriptor inputTensorDesc,
																		 cudnnFilterDescriptor filterDesc,
																		 int nbDims,
																		 int[] tensorOuputDimA
																		);

		/// <summary>
		/// This function destroys a previously created convolution descriptor object.
		/// </summary>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnDestroyConvolutionDescriptor( cudnnConvolutionDescriptor convDesc );


		/// <summary>
		/// This function attempts all cuDNN algorithms and outputs performance metrics to a
		/// user-allocated array of cudnnConvolutionFwdAlgoPerf_t. These metrics are written
		/// in sorted fashion where the first element has the lowest compute time.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="destDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="requestedAlgoCount">The maximum number of elements to be stored in perfResults.</param>
		/// <param name="returnedAlgoCount">The number of output elements stored in perfResults.</param>
		/// <param name="perfResults">A user-allocated array to store performance metrics sorted ascending by
		/// compute time.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnFindConvolutionForwardAlgorithm(cudnnHandle   handle,
                                                                 cudnnTensorDescriptor      srcDesc,
                                                                 cudnnFilterDescriptor      filterDesc,
                                                                 cudnnConvolutionDescriptor convDesc, 
                                                                 cudnnTensorDescriptor      destDesc,
                                                                 int                        requestedAlgoCount,
                                                                 ref int                    returnedAlgoCount,
                                                                 cudnnConvolutionFwdAlgoPerf[] perfResults                                                 
                                                                );


		/// <summary>
		/// This function serves as a heuristic for obtaining the best suited algorithm for
		/// cudnnConvolutionForward for the given layer specifications. Based on the input
		/// preference, this function will either return the fastest algorithm or the fastest algorithm
		/// within a given memory limit. For an exhaustive search for the fastest algorithm, please
		/// use cudnnFindConvolutionForwardAlgorithm.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="destDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="preference">Enumerant to express the preference criteria in terms of memory
		/// requirement and speed.</param>
		/// <param name="memoryLimitInbytes">It is used when enumerant preference is set to
		/// CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT to specify the
		/// maximum amount of GPU memory the user is willing to use as a workspace</param>
		/// <param name="algo">Enumerant that specifies which convolution algorithm should be used to
		/// compute the results according to the specified preference</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetConvolutionForwardAlgorithm( cudnnHandle                      handle,
																	   cudnnTensorDescriptor      srcDesc,
																	   cudnnFilterDescriptor      filterDesc,
																	   cudnnConvolutionDescriptor convDesc, 
																	   cudnnTensorDescriptor      destDesc,
																	   cudnnConvolutionFwdPreference    preference, 
																	   SizeT                             memoryLimitInbytes,
																	   ref cudnnConvolutionFwdAlgo         algo                                                  
																	 );        
                                                                                                           
		/// <summary>
		/// This function returns the amount of GPU memory workspace the user needs
		/// to allocate to be able to call cudnnConvolutionForward with the specified
		/// algorithm. The workspace allocated will then be passed to the routine
		/// cudnnConvolutionForward. The specified algorithm can be the result of the call to
		/// cudnnGetConvolutionForwardAlgorithm or can be chosen arbitrarily by the user.
		/// Note that not every algorithm is available for every configuration of the input tensor
		/// and/or every configuration of the convolution descriptor.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="destDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="algo">Enumerant that specifies the chosen convolution algorithm</param>
		/// <param name="sizeInBytes">Amount of GPU memory needed as workspace to be able to execute a
		/// forward convolution with the specified algo</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetConvolutionForwardWorkspaceSize( cudnnHandle                      handle, 
																		   cudnnTensorDescriptor      srcDesc,
																		   cudnnFilterDescriptor      filterDesc,
																		   cudnnConvolutionDescriptor convDesc,  
																		   cudnnTensorDescriptor      destDesc,
																		   cudnnConvolutionFwdAlgo          algo,
																		   ref SizeT                            sizeInBytes
																		);        


		/// <summary>
		/// This function executes convolutions or cross-correlations over src using the specified
		/// filters, returning results in dest. Scaling factors alpha and beta can be used to scale
		/// the input tensor and the output tensor respectively.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="srcDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="srcData">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="filterData">Data pointer to GPU memory associated with the filter descriptor filterDesc.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="algo">Enumerant that specifies which convolution algorithm shoud be used to compute the results</param>
		/// <param name="workSpace">Data pointer to GPU memory to a workspace needed to able to execute
		/// the specified algorithm. If no workspace is needed for a particular
		/// algorithm, that pointer can be nil</param>
		/// <param name="workSpaceSizeInBytes">Specifies the size in bytes of the provided workSpace</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="destDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="destData">Data pointer to GPU memory associated with the tensor descriptor
		/// destDesc that carries the result of the convolution.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionForward( cudnnHandle handle,
																  ref float alpha,
																  cudnnTensorDescriptor srcDesc,
																  CUdeviceptr srcData,
																  cudnnFilterDescriptor filterDesc,
																  CUdeviceptr filterData,
																  cudnnConvolutionDescriptor convDesc,
																  cudnnConvolutionFwdAlgo algo,
																  CUdeviceptr workSpace,
																  SizeT workSpaceSizeInBytes,            
																  ref float beta,
																  cudnnTensorDescriptor destDesc,
																  CUdeviceptr destData
														 );

		/// <summary>
		/// This function executes convolutions or cross-correlations over src using the specified
		/// filters, returning results in dest. Scaling factors alpha and beta can be used to scale
		/// the input tensor and the output tensor respectively.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="srcDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="srcData">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="filterData">Data pointer to GPU memory associated with the filter descriptor filterDesc.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="algo">Enumerant that specifies which convolution algorithm shoud be used to compute the results</param>
		/// <param name="workSpace">Data pointer to GPU memory to a workspace needed to able to execute
		/// the specified algorithm. If no workspace is needed for a particular
		/// algorithm, that pointer can be nil</param>
		/// <param name="workSpaceSizeInBytes">Specifies the size in bytes of the provided workSpace</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="destDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="destData">Data pointer to GPU memory associated with the tensor descriptor
		/// destDesc that carries the result of the convolution.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionForward( cudnnHandle handle,
																  ref double alpha,
																  cudnnTensorDescriptor srcDesc,
																  CUdeviceptr srcData,
																  cudnnFilterDescriptor filterDesc,
																  CUdeviceptr filterData,
																  cudnnConvolutionDescriptor convDesc,
																  cudnnConvolutionFwdAlgo algo,
																  CUdeviceptr workSpace,
																  SizeT workSpaceSizeInBytes,
																  ref double beta,
																  cudnnTensorDescriptor destDesc,
																  CUdeviceptr destData
														 );

		/// <summary>
		/// This function computes the convolution gradient with respect to the bias, which is the
		/// sum of every element belonging to the same feature map across all of the images of the
		/// input tensor. Therefore, the number of elements produced is equal to the number of
		/// features maps of the input tensor.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="srcData">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="destDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="destData">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionBackwardBias(cudnnHandle handle,
																	  ref float alpha,
																	  cudnnTensorDescriptor srcDesc,
																	  CUdeviceptr srcData,
																	  ref float beta,
																	  cudnnTensorDescriptor destDesc,
																	  CUdeviceptr destData
															  );
		/// <summary>
		/// This function computes the convolution gradient with respect to the bias, which is the
		/// sum of every element belonging to the same feature map across all of the images of the
		/// input tensor. Therefore, the number of elements produced is equal to the number of
		/// features maps of the input tensor.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="srcData">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="destDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="destData">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionBackwardBias(cudnnHandle handle,
																	  ref double alpha,
																	  cudnnTensorDescriptor srcDesc,
																	  CUdeviceptr srcData,
																	  ref double beta,
																	  cudnnTensorDescriptor destDesc,
																	  CUdeviceptr destData
															  );


		/// <summary>
		/// This function attempts all cuDNN algorithms for cudnnConvolutionBackwardFilter_v3 and outputs performance metrics to a user-
		/// allocated array of cudnnConvolutionBwdFilterAlgoPerf_t. These metrics are
		/// written in sorted fashion where the first element has the lowest compute time. 
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="diffDesc">Handle to the previously initialized input differential tensor descriptor.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="gradDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="requestedAlgoCount">The maximum number of elements to be stored in perfResults.</param>
		/// <param name="returnedAlgoCount">The number of output elements stored in perfResults.</param>
		/// <param name="perfResults">A user-allocated array to store performance metrics sorted ascending by compute time.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnFindConvolutionBackwardFilterAlgorithm( cudnnHandle     handle,
                                                                       cudnnTensorDescriptor          srcDesc,
                                                                       cudnnTensorDescriptor          diffDesc,
                                                                       cudnnConvolutionDescriptor     convDesc, 
                                                                       cudnnFilterDescriptor          gradDesc,
                                                                       int                              requestedAlgoCount,
                                                                       ref int                          returnedAlgoCount,
                                                                       cudnnConvolutionBwdFilterAlgoPerf[] perfResults   
                                                                     );
                                          
        /// <summary>
        /// This function serves as a heuristic for obtaining the best suited algorithm for
		/// cudnnConvolutionBackwardFilter for the given layer specifications. Based
		/// on the input preference, this function will either return the fastest algorithm or the
		/// fastest algorithm within a given memory limit. For an exhaustive search for the fastest
		/// algorithm, please use cudnnFindConvolutionBackwardFilterAlgorithm.
        /// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="diffDesc">Handle to the previously initialized input differential tensor descriptor.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="gradDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="preference">Enumerant to express the preference criteria in terms of memory requirement and speed.</param>
        /// <param name="memoryLimitInbytes">It is to specify the maximum amount of GPU memory the user is willing to 
		/// use as a workspace. This is currently a placeholder and is not used.</param>
        /// <param name="algo">Enumerant that specifies which convolution algorithm should be used to
		/// compute the results according to the specified preference</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetConvolutionBackwardFilterAlgorithm( cudnnHandle             handle,
                                                                      cudnnTensorDescriptor          srcDesc,
                                                                      cudnnTensorDescriptor          diffDesc,
                                                                      cudnnConvolutionDescriptor     convDesc, 
                                                                      cudnnFilterDescriptor          gradDesc,
                                                                      cudnnConvolutionBwdFilterPreference  preference,
                                                                      SizeT                                memoryLimitInbytes,
                                                                      ref cudnnConvolutionBwdFilterAlgo algo
                                                                     );



		/// <summary>
		/// This function returns the amount of GPU memory workspace the user needs
		/// to allocate to be able to call cudnnConvolutionBackwardFilter_v3 with the
		/// specified algorithm. The workspace allocated will then be passed to the routine
		/// cudnnConvolutionBackwardFilter. The specified algorithm can be the result
		/// of the call to cudnnGetConvolutionBackwardFilterAlgorithm or can be chosen
		/// arbitrarily by the user. Note that not every algorithm is available for every configuration
		/// of the input tensor and/or every configuration of the convolution descriptor.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="diffDesc">Handle to the previously initialized input differential tensor descriptor.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="gradDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="algo">Enumerant that specifies the chosen convolution algorithm
		/// sizeInBytes output Amount of GPU memory needed as workspace to be able to execute</param>
		/// <param name="sizeInBytes">Amount of GPU memory needed as workspace to be able to execute a
		/// forward convolution with the specified algo</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetConvolutionBackwardFilterWorkspaceSize( cudnnHandle          handle, 
																				  cudnnTensorDescriptor       srcDesc,
																				  cudnnTensorDescriptor       diffDesc,
																				  cudnnConvolutionDescriptor  convDesc,  
																				  cudnnFilterDescriptor       gradDesc,
																				  cudnnConvolutionBwdFilterAlgo     algo,
																				  ref SizeT                         sizeInBytes
																				);
                                  
		/// <summary>
		/// This function computes the convolution gradient with respect to filter coefficients using
		/// the specified algo, returning results in gradDesc.Scaling factors alpha and beta can be
		/// used to scale the input tensor and the output tensor respectively.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="srcDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="srcData">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="diffDesc">Handle to the previously initialized input differential tensor descriptor.</param>
		/// <param name="diffData">Data pointer to GPU memory associated with the input differential tensor descriptor diffDesc.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="algo">Enumerant that specifies which convolution algorithm shoud be used to compute the results</param>
		/// <param name="workSpace">Data pointer to GPU memory to a workspace needed to able to execute
		/// the specified algorithm. If no workspace is needed for a particular
		/// algorithm, that pointer can be nil</param>
		/// <param name="workSpaceSizeInBytes">Specifies the size in bytes of the provided workSpace</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="gradDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="gradData">Data pointer to GPU memory associated with the filter descriptor
		/// gradDesc that carries the result.</param>    
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionBackwardFilter( cudnnHandle                 handle,
																	 ref float alpha,
																	 cudnnTensorDescriptor       srcDesc,
																	 CUdeviceptr srcData,
																	 cudnnTensorDescriptor       diffDesc,
																	 CUdeviceptr diffData,
																	 cudnnConvolutionDescriptor  convDesc,
																	 cudnnConvolutionBwdFilterAlgo     algo,
																	 CUdeviceptr workSpace,
																	 SizeT                              workSpaceSizeInBytes,
																	 ref float beta,
																	 cudnnFilterDescriptor       gradDesc,
																	 CUdeviceptr gradData
																   );

        /// <summary>
        /// This function computes the convolution gradient with respect to filter coefficients using
        /// the specified algo, returning results in gradDesc.Scaling factors alpha and beta can be
        /// used to scale the input tensor and the output tensor respectively.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="srcDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="srcData">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="diffDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="diffData">Data pointer to GPU memory associated with the input differential tensor descriptor diffDesc.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="algo">Enumerant that specifies which convolution algorithm shoud be used to compute the results</param>
        /// <param name="workSpace">Data pointer to GPU memory to a workspace needed to able to execute
        /// the specified algorithm. If no workspace is needed for a particular
        /// algorithm, that pointer can be nil</param>
        /// <param name="workSpaceSizeInBytes">Specifies the size in bytes of the provided workSpace</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="gradDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="gradData">Data pointer to GPU memory associated with the filter descriptor
        /// gradDesc that carries the result.</param>    
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnConvolutionBackwardFilter(cudnnHandle handle,
                                                                     ref double alpha,
                                                                     cudnnTensorDescriptor srcDesc,
                                                                     CUdeviceptr srcData,
                                                                     cudnnTensorDescriptor diffDesc,
                                                                     CUdeviceptr diffData,
                                                                     cudnnConvolutionDescriptor convDesc,
                                                                     cudnnConvolutionBwdFilterAlgo algo,
                                                                     CUdeviceptr workSpace,
                                                                     SizeT workSpaceSizeInBytes,
                                                                     ref double beta,
                                                                     cudnnFilterDescriptor gradDesc,
                                                                     CUdeviceptr gradData
                                                                   );


        /// <summary>
        /// This function attempts all cuDNN algorithms for
        /// cudnnConvolutionBackwardData and outputs performance metrics to a user-
        /// allocated array of cudnnConvolutionBwdDataAlgoPerf_t. These metrics are written
        /// in sorted fashion where the first element has the lowest compute time.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="diffDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="gradDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="requestedAlgoCount">The maximum number of elements to be stored in perfResults.</param>
        /// <param name="returnedAlgoCount">The number of output elements stored in perfResults.</param>
        /// <param name="perfResults">A user-allocated array to store performance metrics sorted ascending by compute time.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnFindConvolutionBackwardDataAlgorithm( cudnnHandle handle,
                                                                     cudnnFilterDescriptor       filterDesc,
                                                                     cudnnTensorDescriptor       diffDesc,
                                                                     cudnnConvolutionDescriptor  convDesc, 
                                                                     cudnnTensorDescriptor       gradDesc,
                                                                     int                           requestedAlgoCount,
                                                                     ref int                               returnedAlgoCount,
                                                                     cudnnConvolutionBwdDataAlgoPerf[] perfResults  
                                                                   );
                                          
		/// <summary>
		/// This function serves as a heuristic for obtaining the best suited algorithm for
		/// cudnnConvolutionBackwardData for the given layer specifications. Based
		/// on the input preference, this function will either return the fastest algorithm or the
		/// fastest algorithm within a given memory limit. For an exhaustive search for the fastest
		/// algorithm, please use cudnnFindConvolutionBackwardDataAlgorithm.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="diffDesc">Handle to the previously initialized input differential tensor descriptor.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="gradDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="preference">Enumerant to express the preference criteria in terms of memory
		/// requirement and speed.</param>
		/// <param name="memoryLimitInbytes">It is to specify the maximum amount of GPU memory the user is willing to
		/// use as a workspace. This is currently a placeholder and is not used.</param>
		/// <param name="algo">Enumerant that specifies which convolution algorithm should be used to
		/// compute the results according to the specified preference</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetConvolutionBackwardDataAlgorithm( cudnnHandle handle,
																	   cudnnFilterDescriptor       filterDesc,
																	   cudnnTensorDescriptor       diffDesc,
																	   cudnnConvolutionDescriptor  convDesc, 
																	   cudnnTensorDescriptor       gradDesc,
																	   cudnnConvolutionBwdDataPreference preference, 
																	   SizeT                              memoryLimitInbytes,
																	   ref cudnnConvolutionBwdDataAlgo algo
																	 );

		/// <summary>
		/// This function returns the amount of GPU memory workspace the user needs
		/// to allocate to be able to call cudnnConvolutionBackwardData with the
		/// specified algorithm. The workspace allocated will then be passed to the routine
		/// cudnnConvolutionBackwardData. The specified algorithm can be the result of the
		/// call to cudnnGetConvolutionBackwardDataAlgorithm or can be chosen arbitrarily
		/// by the user. Note that not every algorithm is available for every configuration of the
		/// input tensor and/or every configuration of the convolution descriptor.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="diffDesc">Handle to the previously initialized input differential tensor descriptor.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="gradDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="algo">Enumerant that specifies the chosen convolution algorithm</param>
		/// <param name="sizeInBytes">Amount of GPU memory needed as workspace to be able to execute a forward convolution with the specified algo</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetConvolutionBackwardDataWorkspaceSize( cudnnHandle handle,
																		   cudnnFilterDescriptor      filterDesc,
																		   cudnnTensorDescriptor       diffDesc,
																		   cudnnConvolutionDescriptor convDesc,  
																		   cudnnTensorDescriptor       gradDesc,
																		   cudnnConvolutionBwdDataAlgo          algo,
																		   ref SizeT                            sizeInBytes
																		);        

        /// <summary>
        /// This function computes the convolution gradient with respect to the output tensor using
		/// the specified algo, returning results in gradDesc. Scaling factors alpha and beta can
		/// be used to scale the input tensor and the output tensor respectively.
        /// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="filterData">Data pointer to GPU memory associated with the filter descriptor filterDesc.</param>
		/// <param name="diffDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="diffData">Data pointer to GPU memory associated with the input differential tensor descriptor diffDesc.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="algo">Enumerant that specifies which backward data convolution algorithm shoud be used to compute the results</param>
        /// <param name="workSpace">Data pointer to GPU memory to a workspace needed to able to execute
		/// the specified algorithm. If no workspace is needed for a particular
		/// algorithm, that pointer can be nil</param>
		/// <param name="workSpaceSizeInBytes">Specifies the size in bytes of the provided workSpace</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="gradDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="gradData">Data pointer to GPU memory associated with the output tensor descriptor
		/// gradDesc that carries the result.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionBackwardData( cudnnHandle handle,
																 ref float alpha,
																 cudnnFilterDescriptor       filterDesc,
																 CUdeviceptr filterData,
																 cudnnTensorDescriptor       diffDesc,
																 CUdeviceptr diffData,
																 cudnnConvolutionDescriptor  convDesc,
																 cudnnConvolutionBwdDataAlgo           algo,
																 CUdeviceptr workSpace,
																 SizeT                              workSpaceSizeInBytes,
																 ref float beta,
																 cudnnTensorDescriptor       gradDesc,
																 CUdeviceptr gradData
															   );

		/// <summary>
		/// This function computes the convolution gradient with respect to the output tensor using
		/// the specified algo, returning results in gradDesc. Scaling factors alpha and beta can
		/// be used to scale the input tensor and the output tensor respectively.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="filterData">Data pointer to GPU memory associated with the filter descriptor filterDesc.</param>
		/// <param name="diffDesc">Handle to the previously initialized input differential tensor descriptor.</param>
		/// <param name="diffData">Data pointer to GPU memory associated with the input differential tensor descriptor diffDesc.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="algo">Enumerant that specifies which backward data convolution algorithm shoud be used to compute the results</param>
		/// <param name="workSpace">Data pointer to GPU memory to a workspace needed to able to execute
		/// the specified algorithm. If no workspace is needed for a particular
		/// algorithm, that pointer can be nil</param>
		/// <param name="workSpaceSizeInBytes">Specifies the size in bytes of the provided workSpace</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="gradDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="gradData">Data pointer to GPU memory associated with the output tensor descriptor
		/// gradDesc that carries the result.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionBackwardData( cudnnHandle handle,
																 ref double alpha,
																 cudnnFilterDescriptor       filterDesc,
																 CUdeviceptr filterData,
																 cudnnTensorDescriptor       diffDesc,
																 CUdeviceptr diffData,
																 cudnnConvolutionDescriptor  convDesc,
																 cudnnConvolutionBwdDataAlgo           algo,
																 CUdeviceptr workSpace,
																 SizeT                              workSpaceSizeInBytes,
																 ref double beta,
																 cudnnTensorDescriptor       gradDesc,
																 CUdeviceptr gradData
															   );


		/// <summary>
		/// 
		/// </summary>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnIm2Col(  cudnnHandle handle,
												cudnnTensorDescriptor srcDesc,
												CUdeviceptr srcData,
												cudnnFilterDescriptor filterDesc,                                        
												cudnnConvolutionDescriptor convDesc,
												CUdeviceptr colBuffer
											 );




		/* Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" */

		/// <summary>
		/// This routine computes the softmax function.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="algorithm">Enumerant to specify the softmax algorithm.</param>
		/// <param name="mode">Enumerant to specify the softmax mode.</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="srcData">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="destDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="destData">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSoftmaxForward(  cudnnHandle handle,
														cudnnSoftmaxAlgorithm algorithm,
														cudnnSoftmaxMode mode,
														ref float alpha,
														cudnnTensorDescriptor srcDesc,
														CUdeviceptr srcData,
														ref float beta,
														cudnnTensorDescriptor destDesc,
														CUdeviceptr destData
													 );
		/// <summary>
		/// This routine computes the softmax function.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="algorithm">Enumerant to specify the softmax algorithm.</param>
		/// <param name="mode">Enumerant to specify the softmax mode.</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="srcData">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="destDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="destData">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSoftmaxForward(  cudnnHandle handle,
														cudnnSoftmaxAlgorithm algorithm,
														cudnnSoftmaxMode mode,
														ref double alpha,
														cudnnTensorDescriptor srcDesc,
														CUdeviceptr srcData,
														ref double beta,
														cudnnTensorDescriptor destDesc,
														CUdeviceptr destData
													 );

		/// <summary>
		/// This routine computes the gradient of the softmax function.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="algorithm">Enumerant to specify the softmax algorithm.</param>
		/// <param name="mode">Enumerant to specify the softmax mode.</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="srcData">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="srcDiffDesc">Handle to the previously initialized input differential tensor descriptor.</param>
		/// <param name="srcDiffData">Data pointer to GPU memory associated with the tensor descriptor srcDiffData.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="destDiffDesc">Handle to the previously initialized output differential tensor descriptor.</param>
		/// <param name="destDiffData">Data pointer to GPU memory associated with the output tensor descriptor destDiffDesc.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSoftmaxBackward( cudnnHandle handle,
														cudnnSoftmaxAlgorithm algorithm,
														cudnnSoftmaxMode mode,
														ref float alpha,
														cudnnTensorDescriptor srcDesc,
														CUdeviceptr srcData,
														cudnnTensorDescriptor srcDiffDesc,
														CUdeviceptr srcDiffData,
														ref float beta,
														cudnnTensorDescriptor destDiffDesc,
														CUdeviceptr destDiffData
													  );

		/// <summary>
		/// This routine computes the gradient of the softmax function.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="algorithm">Enumerant to specify the softmax algorithm.</param>
		/// <param name="mode">Enumerant to specify the softmax mode.</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="srcData">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="srcDiffDesc">Handle to the previously initialized input differential tensor descriptor.</param>
		/// <param name="srcDiffData">Data pointer to GPU memory associated with the tensor descriptor srcDiffData.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="destDiffDesc">Handle to the previously initialized output differential tensor descriptor.</param>
		/// <param name="destDiffData">Data pointer to GPU memory associated with the output tensor descriptor destDiffDesc.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSoftmaxBackward( cudnnHandle handle,
														cudnnSoftmaxAlgorithm algorithm,
														cudnnSoftmaxMode mode,
														ref double alpha,
														cudnnTensorDescriptor srcDesc,
														CUdeviceptr srcData,
														cudnnTensorDescriptor srcDiffDesc,
														CUdeviceptr srcDiffData,
														ref double beta,
														cudnnTensorDescriptor destDiffDesc,
														CUdeviceptr destDiffData
													  );

		/// <summary>
		/// This function creates a pooling descriptor object by allocating the memory needed to hold its opaque structure
		/// </summary>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnCreatePoolingDescriptor( ref cudnnPoolingDescriptor poolingDesc);

        /// <summary>
        /// This function initializes a previously created generic pooling descriptor object into a 2D description.
        /// </summary>
        /// <param name="poolingDesc">Handle to a previously created pooling descriptor.</param>
        /// <param name="mode">Enumerant to specify the pooling mode.</param>
        /// <param name="maxpoolingNanOpt">Nan propagation option for max pooling.</param>
        /// <param name="windowHeight">Height of the pooling window.</param>
        /// <param name="windowWidth">Width of the pooling window.</param>
        /// <param name="verticalPadding">Size of vertical padding.</param>
        /// <param name="horizontalPadding">Size of horizontal padding</param>
        /// <param name="verticalStride">Pooling vertical stride.</param>
        /// <param name="horizontalStride">Pooling horizontal stride.</param>
        [DllImport(CUDNN_API_DLL_NAME, EntryPoint = "cudnnSetPooling2dDescriptor_v4")]
		public static extern cudnnStatus cudnnSetPooling2dDescriptor(  cudnnPoolingDescriptor poolingDesc,
																cudnnPoolingMode mode,
                                                                cudnnNanPropagation maxpoolingNanOpt,
                                                                int windowHeight,
																int windowWidth,
																int verticalPadding,
																int horizontalPadding,
																int verticalStride,
																int horizontalStride
														   );

        /// <summary>
        /// This function queries a previously created 2D pooling descriptor object.
        /// </summary>
        /// <param name="poolingDesc">Handle to a previously created pooling descriptor.</param>
        /// <param name="mode">Enumerant to specify the pooling mode.</param>
        /// <param name="maxpoolingNanOpt">Nan propagation option for max pooling.</param>
        /// <param name="windowHeight">Height of the pooling window.</param>
        /// <param name="windowWidth">Width of the pooling window.</param>
        /// <param name="verticalPadding">Size of vertical padding.</param>
        /// <param name="horizontalPadding">Size of horizontal padding.</param>
        /// <param name="verticalStride">Pooling vertical stride.</param>
        /// <param name="horizontalStride">Pooling horizontal stride.</param>
        [DllImport(CUDNN_API_DLL_NAME, EntryPoint = "cudnnGetPooling2dDescriptor_v4")]
		public static extern cudnnStatus cudnnGetPooling2dDescriptor(  cudnnPoolingDescriptor poolingDesc,
																ref cudnnPoolingMode mode,
                                                                ref cudnnNanPropagation maxpoolingNanOpt,
                                                                ref int windowHeight,
																ref int windowWidth,
																ref int verticalPadding,
																ref int horizontalPadding,
																ref int verticalStride,
																ref int horizontalStride
														   );

        /// <summary>
        /// This function initializes a previously created generic pooling descriptor object.
        /// </summary>
        /// <param name="poolingDesc">Handle to a previously created pooling descriptor.</param>
        /// <param name="mode">Enumerant to specify the pooling mode.</param>
        /// <param name="maxpoolingNanOpt">Nan propagation option for max pooling.</param>
        /// <param name="nbDims">Dimension of the pooling operation.</param>
        /// <param name="windowDimA">Array of dimension nbDims containing the window size for each dimension.</param>
        /// <param name="paddingA">Array of dimension nbDims containing the padding size for each dimension.</param>
        /// <param name="strideA">Array of dimension nbDims containing the striding size for each dimension.</param>
        [DllImport(CUDNN_API_DLL_NAME, EntryPoint = "cudnnSetPoolingNdDescriptor_v4")]
		public static extern cudnnStatus cudnnSetPoolingNdDescriptor(  cudnnPoolingDescriptor poolingDesc,
																cudnnPoolingMode mode,
                                                                cudnnNanPropagation maxpoolingNanOpt,
                                                                int nbDims,
																int[] windowDimA,
																int[] paddingA,
																int[] strideA
														   );

        /// <summary>
        /// This function queries a previously initialized generic pooling descriptor object.
        /// </summary>
        /// <param name="poolingDesc">Handle to a previously created pooling descriptor.</param>
        /// <param name="nbDimsRequested">Dimension of the expected pooling descriptor. It is also the minimum
        /// size of the arrays windowDimA, paddingA and strideA in order to be
        /// able to hold the results</param>
        /// <param name="mode">Enumerant to specify the pooling mode.</param>
        /// <param name="maxpoolingNanOpt">Nan propagation option for max pooling.</param>
        /// <param name="nbDims">Actual dimension of the pooling descriptor.</param>
        /// <param name="windowDimA">Array of dimension of at least nbDimsRequested that will be filled with
        /// the window parameters from the provided pooling descriptor.</param>
        /// <param name="paddingA">Array of dimension of at least nbDimsRequested that will be filled with
        /// the padding parameters from the provided pooling descriptor.</param>
        /// <param name="strideA">Array of dimension at least nbDimsRequested that will be filled with
        /// the stride parameters from the provided pooling descriptor.</param>
        [DllImport(CUDNN_API_DLL_NAME, EntryPoint = "cudnnGetPoolingNdDescriptor_v4")]
		public static extern cudnnStatus cudnnGetPoolingNdDescriptor(  cudnnPoolingDescriptor poolingDesc,
																int nbDimsRequested,
																ref cudnnPoolingMode mode,
                                                                ref cudnnNanPropagation maxpoolingNanOpt,
                                                                ref int nbDims,
																int[] windowDimA,
																int[] paddingA,
																int[] strideA
															 );

		/// <summary>
		/// This function provides the output dimensions of a tensor after Nd pooling has been applied
		/// </summary>
		/// <param name="poolingDesc">Handle to a previously inititalized pooling descriptor.</param>
		/// <param name="inputTensorDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="nbDims">Number of dimensions in which pooling is to be applied.</param>
		/// <param name="outputTensorDimA">Array of nbDims output dimensions</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetPoolingNdForwardOutputDim( cudnnPoolingDescriptor poolingDesc,
																	 cudnnTensorDescriptor inputTensorDesc,
																	 int nbDims,
																	 int[] outputTensorDimA);
		/// <summary>
		/// This function provides the output dimensions of a tensor after 2d pooling has been applied
		/// </summary>
		/// <param name="poolingDesc">Handle to a previously inititalized pooling descriptor.</param>
		/// <param name="inputTensorDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="outN">Number of images in the output</param>
		/// <param name="outC">Number of channels in the output</param>
		/// <param name="outH">Height of images in the output</param>
		/// <param name="outW">Width of images in the output</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetPooling2dForwardOutputDim( cudnnPoolingDescriptor poolingDesc,
																	 cudnnTensorDescriptor inputTensorDesc,
																	 ref int outN,
																	 ref int outC,
																	 ref int outH,
																	 ref int outW);


		/// <summary>
		/// This function destroys a previously created pooling descriptor object.
		/// </summary>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnDestroyPoolingDescriptor( cudnnPoolingDescriptor poolingDesc );

		/* Pooling functions: All of the form "output = alpha * Op(inputs) + beta * output" */

		/// <summary>
		/// This function computes pooling of input values (i.e., the maximum or average of several
		/// adjacent values) to produce an output with smaller height and/or width.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="poolingDesc">Handle to a previously initialized pooling descriptor.</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="srcData">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="destDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="destData">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnPoolingForward(  cudnnHandle handle,
														cudnnPoolingDescriptor poolingDesc,
														ref float alpha,
														cudnnTensorDescriptor srcDesc,
														CUdeviceptr srcData,
														ref float beta,
														cudnnTensorDescriptor destDesc,
														CUdeviceptr destData
													 );

		/// <summary>
		/// This function computes pooling of input values (i.e., the maximum or average of several
		/// adjacent values) to produce an output with smaller height and/or width.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="poolingDesc">Handle to a previously initialized pooling descriptor.</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="srcData">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="destDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="destData">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnPoolingForward(  cudnnHandle handle,
														cudnnPoolingDescriptor poolingDesc,
														ref double alpha,
														cudnnTensorDescriptor srcDesc,
														CUdeviceptr srcData,
														ref double beta,
														cudnnTensorDescriptor destDesc,
														CUdeviceptr destData
													 );

		/// <summary>
		/// This function computes the gradient of a pooling operation.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="poolingDesc">Handle to the previously initialized pooling descriptor.</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="srcData">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="srcDiffDesc">Handle to the previously initialized input differential tensor descriptor.</param>
		/// <param name="srcDiffData">Data pointer to GPU memory associated with the tensor descriptor srcDiffData.</param>
		/// <param name="destDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="destData">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="destDiffDesc">Handle to the previously initialized output differential tensor descriptor.</param>
		/// <param name="destDiffData">Data pointer to GPU memory associated with the output tensor descriptor destDiffDesc.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnPoolingBackward( cudnnHandle handle,
														cudnnPoolingDescriptor poolingDesc,
														ref float alpha,
														cudnnTensorDescriptor srcDesc,
														CUdeviceptr srcData,
														cudnnTensorDescriptor srcDiffDesc,
														CUdeviceptr srcDiffData,
														cudnnTensorDescriptor destDesc,
														CUdeviceptr destData,
														ref float beta,
														cudnnTensorDescriptor destDiffDesc,
														CUdeviceptr destDiffData
													  );
		/// <summary>
		/// This function computes the gradient of a pooling operation.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="poolingDesc">Handle to the previously initialized pooling descriptor.</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="srcData">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="srcDiffDesc">Handle to the previously initialized input differential tensor descriptor.</param>
		/// <param name="srcDiffData">Data pointer to GPU memory associated with the tensor descriptor srcDiffData.</param>
		/// <param name="destDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="destData">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="destDiffDesc">Handle to the previously initialized output differential tensor descriptor.</param>
		/// <param name="destDiffData">Data pointer to GPU memory associated with the output tensor descriptor destDiffDesc.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnPoolingBackward( cudnnHandle handle,
														cudnnPoolingDescriptor poolingDesc,
														ref double alpha,
														cudnnTensorDescriptor srcDesc,
														CUdeviceptr srcData,
														cudnnTensorDescriptor srcDiffDesc,
														CUdeviceptr srcDiffData,
														cudnnTensorDescriptor destDesc,
														CUdeviceptr destData,
														ref double beta,
														cudnnTensorDescriptor destDiffDesc,
														CUdeviceptr destDiffData
													  );


        /* Activation functions: All of the form "output = alpha * Op(inputs) + beta * output" */
        ///<summary>
        /// This function creates a activation descriptor object by allocating the memory needed to hold its opaque structure.
        /// </summary>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnCreateActivationDescriptor(
                            ref cudnnActivationDescriptor activationDesc);

        ///<summary>
        /// This function initializes then previously created activation descriptor object.
        /// </summary>
        /// <param name="activationDesc">Handle to the previously created activation descriptor object.</param>
        /// <param name="mode">Enumerant to specify the activation mode.</param>
        /// <param name="reluNanOpt">Nan propagation option for the relu.</param>
        /// <param name="reluCeiling">The ceiling for the clipped relu.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSetActivationDescriptor(
                                        cudnnActivationDescriptor activationDesc,
                                        cudnnActivationMode mode,
                                        cudnnNanPropagation reluNanOpt,
                                        double reluCeiling);

        /// <summary>
        /// This function queries the parameters of the previouly initialized activation descriptor object.
        /// </summary>
        /// <param name="activationDesc">Handle to the previously created activation descriptor object.</param>
        /// <param name="mode">Enumerant to specify the activation mode.</param>
        /// <param name="reluNanOpt">Nan propagation option for the relu.</param>
        /// <param name="reluCeiling">The ceiling for the clipped relu.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetActivationDescriptor(
                                cudnnActivationDescriptor activationDesc,
                                ref cudnnActivationMode              mode,
                                ref cudnnNanPropagation reluNanOpt,
                                ref double reluCeiling );

        /// <summary>
        /// This function destroys a previously created activation descriptor object.
        /// </summary>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnDestroyActivationDescriptor(
                                        cudnnActivationDescriptor activationDesc);

        /// <summary>
        /// This routine applies a specified neuron activation function element-wise over each input value.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="activationDesc">Handle to the previously created activation descriptor object.</param>
        /// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="srcData">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="destDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="destData">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        [DllImport(CUDNN_API_DLL_NAME, EntryPoint = "cudnnActivationForward_v4")]
		public static extern cudnnStatus cudnnActivationForward( cudnnHandle handle,
														  cudnnActivationDescriptor activationDesc,
														  ref float alpha,
														  cudnnTensorDescriptor srcDesc,
														  CUdeviceptr srcData,
														  ref float beta,
														  cudnnTensorDescriptor destDesc,
														  CUdeviceptr destData
														);
        /// <summary>
        /// This routine applies a specified neuron activation function element-wise over each input value.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="activationDesc">Handle to the previously created activation descriptor object.</param>
        /// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="srcData">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="destDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="destData">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        [DllImport(CUDNN_API_DLL_NAME, EntryPoint = "cudnnActivationForward_v4")]
        public static extern cudnnStatus cudnnActivationForward( cudnnHandle handle,
                                                          cudnnActivationDescriptor activationDesc,
                                                          ref double alpha,
														  cudnnTensorDescriptor srcDesc,
														  CUdeviceptr srcData,
														  ref double beta,
														  cudnnTensorDescriptor destDesc,
														  CUdeviceptr destData
														);

        /// <summary>
        /// This routine computes the gradient of a neuron activation function.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="activationDesc">Handle to the previously created activation descriptor object.</param>
        /// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="srcData">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="srcDiffDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="srcDiffData">Data pointer to GPU memory associated with the tensor descriptor srcDiffData.</param>
        /// <param name="destDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="destData">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="destDiffDesc">Handle to the previously initialized output differential tensor descriptor.</param>
        /// <param name="destDiffData">Data pointer to GPU memory associated with the output tensor descriptor destDiffDesc.</param>
        [DllImport(CUDNN_API_DLL_NAME, EntryPoint = "cudnnActivationBackward_v4")]
		public static extern cudnnStatus cudnnActivationBackward( cudnnHandle handle,
                                                           cudnnActivationDescriptor activationDesc,
                                                           ref float alpha,
														   cudnnTensorDescriptor srcDesc,
														   CUdeviceptr srcData,
														   cudnnTensorDescriptor srcDiffDesc,
														   CUdeviceptr srcDiffData,
														   cudnnTensorDescriptor destDesc,
														   CUdeviceptr destData,
														   ref float beta,
														   cudnnTensorDescriptor destDiffDesc,
														   CUdeviceptr destDiffData
														 );

        /// <summary>
        /// This routine computes the gradient of a neuron activation function.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="activationDesc">Handle to the previously created activation descriptor object.</param>
        /// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="srcData">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="srcDiffDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="srcDiffData">Data pointer to GPU memory associated with the tensor descriptor srcDiffData.</param>
        /// <param name="destDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="destData">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="destDiffDesc">Handle to the previously initialized output differential tensor descriptor.</param>
        /// <param name="destDiffData">Data pointer to GPU memory associated with the output tensor descriptor destDiffDesc.</param>
        [DllImport(CUDNN_API_DLL_NAME, EntryPoint = "cudnnActivationBackward_v4")]
        public static extern cudnnStatus cudnnActivationBackward( cudnnHandle handle,
                                                           cudnnActivationDescriptor activationDesc,
                                                           ref double alpha,
														   cudnnTensorDescriptor srcDesc,
														   CUdeviceptr srcData,
														   cudnnTensorDescriptor srcDiffDesc,
														   CUdeviceptr srcDiffData,
														   cudnnTensorDescriptor destDesc,
														   CUdeviceptr destData,
														   ref double beta,
														   cudnnTensorDescriptor destDiffDesc,
														   CUdeviceptr destDiffData
														 );


		/// <summary>
		/// Create an instance of LRN (Local Response Normalization) descriptor <para/>
		/// This function will set lrnN=5, lrnAlpha=1e-4, lrnBeta=0.75, lrnK=2.0 as defaults from Krizhevsky'12 ImageNet paper
		/// </summary>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnCreateLRNDescriptor(ref cudnnLRNDescriptor normDesc);

		// LRN uses a window [center-lookBehind, center+lookAhead], where
		// lookBehind = floor( (lrnN-1)/2 ), lookAhead = lrnN-lookBehind-1.
		// So for n=10, the window is [k-4...k...k+5] with a total of 10 samples.
		// Values of double parameters will be cast down to tensor data type.
		/// <summary>
		/// This function initializes a previously created LRN descriptor object.
		/// </summary>
		/// <param name="normDesc">Handle to a previously created LRN descriptor.</param>
		/// <param name="lrnN">Normalization window width in elements. LRN layer uses a window
		/// [center-lookBehind, center+lookAhead], where lookBehind =
		/// floor( (lrnN-1)/2 ), lookAhead = lrnN-lookBehind-1. So for n=10,
		/// the window is [k-4...k...k+5] with a total of 10 samples. For
		/// DivisiveNormalization layer the window has the same extents as above in
		/// all 'spatial' dimensions (dimA[2], dimA[3], dimA[4]). By default lrnN is set
		/// to 5 in cudnnCreateLRNDescriptor.</param>
		/// <param name="lrnAlpha">Value of the alpha variance scaling parameter in the normalization
		/// formula. Inside the library code this value is divided by the
		/// window width for LRN and by (window width)^#spatialDimensions
		/// for DivisiveNormalization. By default this value is set to 1e-4 in
		/// cudnnCreateLRNDescriptor.</param>
		/// <param name="lrnBeta">Value of the beta power parameter in the normalization formula. By
		/// default this value is set to 0.75 in cudnnCreateLRNDescriptor.</param>
		/// <param name="lrnK">Value of the k parameter in normalization formula. By default this value is set to 2.0.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSetLRNDescriptor(
									  cudnnLRNDescriptor   normDesc,
									  uint               lrnN,
									  double                 lrnAlpha,
									  double                 lrnBeta,
									  double                 lrnK);

		/// <summary>
		/// This function retrieves values stored in the previously initialized LRN descriptor object.
		/// </summary>
		/// <param name="normDesc">Handle to a previously created LRN descriptor.</param>
		/// <param name="lrnN">Pointers to receive values of parameters stored in the descriptor object.
		/// See cudnnSetLRNDescriptor for more details. Any of these pointers can be
		/// NULL (no value is returned for the corresponding parameter).</param>
		/// <param name="lrnAlpha">Pointers to receive values of parameters stored in the descriptor object.
		/// See cudnnSetLRNDescriptor for more details. Any of these pointers can be
		/// NULL (no value is returned for the corresponding parameter).</param>
		/// <param name="lrnBeta">Pointers to receive values of parameters stored in the descriptor object.
		/// See cudnnSetLRNDescriptor for more details. Any of these pointers can be
		/// NULL (no value is returned for the corresponding parameter).</param>
		/// <param name="lrnK">Pointers to receive values of parameters stored in the descriptor object.
		/// See cudnnSetLRNDescriptor for more details. Any of these pointers can be
		/// NULL (no value is returned for the corresponding parameter).</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetLRNDescriptor(
									  cudnnLRNDescriptor   normDesc,
									  ref uint              lrnN,
									  ref double                lrnAlpha,
									  ref double                lrnBeta,
									  ref double                lrnK);

		/// <summary>
		/// This function destroys a previously created LRN descriptor object.
		/// </summary>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnDestroyLRNDescriptor( cudnnLRNDescriptor lrnDesc );

		// LRN functions: of the form "output = alpha * normalize(srcData) + beta * destData"

		/// <summary>
		/// This function performs the forward LRN layer computation.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN library descriptor.</param>
		/// <param name="normDesc">Handle to a previously intialized LRN parameter descriptor.</param>
		/// <param name="lrnMode">LRN layer mode of operation. Currently only
		/// CUDNN_LRN_CROSS_CHANNEL_DIM1 is implemented. Normalization is
		/// performed along the tensor's dimA[1].</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the layer output
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
		/// for additional details.</param>
		/// <param name="srcDesc">Tensor descriptor objects for the input and output tensors.</param>
		/// <param name="srcData">Input tensor data pointer in device memory.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the layer output
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
		/// for additional details.</param>
		/// <param name="destDesc">Tensor descriptor objects for the input and output tensors.</param>
		/// <param name="destData">Output tensor data pointer in device memory.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnLRNCrossChannelForward(
									  cudnnHandle                    handle,
									  cudnnLRNDescriptor             normDesc,
									  cudnnLRNMode                   lrnMode,
									  ref float alpha,
									  cudnnTensorDescriptor    srcDesc,
									  CUdeviceptr srcData,
									  ref float beta,
									  cudnnTensorDescriptor    destDesc,
									  CUdeviceptr destData);
		/// <summary>
		/// This function performs the forward LRN layer computation.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN library descriptor.</param>
		/// <param name="normDesc">Handle to a previously intialized LRN parameter descriptor.</param>
		/// <param name="lrnMode">LRN layer mode of operation. Currently only
		/// CUDNN_LRN_CROSS_CHANNEL_DIM1 is implemented. Normalization is
		/// performed along the tensor's dimA[1].</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the layer output
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
		/// for additional details.</param>
		/// <param name="srcDesc">Tensor descriptor objects for the input and output tensors.</param>
		/// <param name="srcData">Input tensor data pointer in device memory.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the layer output
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
		/// for additional details.</param>
		/// <param name="destDesc">Tensor descriptor objects for the input and output tensors.</param>
		/// <param name="destData">Output tensor data pointer in device memory.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnLRNCrossChannelForward(
									  cudnnHandle                    handle,
									  cudnnLRNDescriptor             normDesc,
									  cudnnLRNMode                   lrnMode,
									  ref double alpha,
									  cudnnTensorDescriptor    srcDesc,
									  CUdeviceptr srcData,
									  ref double beta,
									  cudnnTensorDescriptor    destDesc,
									  CUdeviceptr destData);

		/// <summary>
		/// This function performs the backward LRN layer computation.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN library descriptor.</param>
		/// <param name="normDesc">Handle to a previously intialized LRN parameter descriptor.</param>
		/// <param name="lrnMode">LRN layer mode of operation. Currently only
		/// CUDNN_LRN_CROSS_CHANNEL_DIM1 is implemented. Normalization is
		/// performed along the tensor's dimA[1].</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the layer output
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
		/// for additional details.</param>
		/// <param name="srcDesc">Tensor descriptor and pointer in device memory for the bottom layer's
		/// data. (Bottom layer is the earlier layer in the computation graph during
		/// inference).</param>
		/// <param name="srcData">Tensor descriptor and pointer in device memory for the bottom layer's
		/// data. (Bottom layer is the earlier layer in the computation graph during
		/// inference).</param>
		/// <param name="srcDiffDesc">Tensor descriptor and pointer in device memory for the top layer's
		/// cumulative loss differential data (error backpropagation). (Top layer is the
		/// later layer in the computation graph during inference).</param>
		/// <param name="srcDiffData">Tensor descriptor and pointer in device memory for the top layer's
		/// cumulative loss differential data (error backpropagation). (Top layer is the
		/// later layer in the computation graph during inference).</param>
		/// <param name="destDesc">Tensor descriptor and pointer in device memory for the bottom layer's
		/// data. (Bottom layer is the earlier layer in the computation graph
		/// during inference). Note that these values are not modified during
		/// backpropagation.</param>
		/// <param name="destData">Tensor descriptor and pointer in device memory for the bottom layer's
		/// data. (Bottom layer is the earlier layer in the computation graph
		/// during inference). Note that these values are not modified during
		/// backpropagation.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the layer output
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
		/// for additional details.</param>
		/// <param name="destDiffDesc">Tensor descriptor and pointer in device memory for the bottom layer's
		/// cumulative loss differential data (error backpropagation). (Bottom layer is
		/// the earlier layer in the computation graph during inference).</param>
		/// <param name="destDiffData">Tensor descriptor and pointer in device memory for the bottom layer's
		/// cumulative loss differential data (error backpropagation). (Bottom layer is
		/// the earlier layer in the computation graph during inference).</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnLRNCrossChannelBackward(
									  cudnnHandle                    handle,
									  cudnnLRNDescriptor             normDesc,
									  cudnnLRNMode                   lrnMode,
									  ref float alpha,
									  cudnnTensorDescriptor    srcDesc,
									  CUdeviceptr srcData,
									  cudnnTensorDescriptor    srcDiffDesc,
									  CUdeviceptr srcDiffData,
									  cudnnTensorDescriptor    destDesc,
									  CUdeviceptr destData,
									  ref float beta,
									  cudnnTensorDescriptor    destDiffDesc,
									  CUdeviceptr destDiffData);
		/// <summary>
		/// This function performs the backward LRN layer computation.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN library descriptor.</param>
		/// <param name="normDesc">Handle to a previously intialized LRN parameter descriptor.</param>
		/// <param name="lrnMode">LRN layer mode of operation. Currently only
		/// CUDNN_LRN_CROSS_CHANNEL_DIM1 is implemented. Normalization is
		/// performed along the tensor's dimA[1].</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the layer output
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
		/// for additional details.</param>
		/// <param name="srcDesc">Tensor descriptor and pointer in device memory for the bottom layer's
		/// data. (Bottom layer is the earlier layer in the computation graph during
		/// inference).</param>
		/// <param name="srcData">Tensor descriptor and pointer in device memory for the bottom layer's
		/// data. (Bottom layer is the earlier layer in the computation graph during
		/// inference).</param>
		/// <param name="srcDiffDesc">Tensor descriptor and pointer in device memory for the top layer's
		/// cumulative loss differential data (error backpropagation). (Top layer is the
		/// later layer in the computation graph during inference).</param>
		/// <param name="srcDiffData">Tensor descriptor and pointer in device memory for the top layer's
		/// cumulative loss differential data (error backpropagation). (Top layer is the
		/// later layer in the computation graph during inference).</param>
		/// <param name="destDesc">Tensor descriptor and pointer in device memory for the bottom layer's
		/// data. (Bottom layer is the earlier layer in the computation graph
		/// during inference). Note that these values are not modified during
		/// backpropagation.</param>
		/// <param name="destData">Tensor descriptor and pointer in device memory for the bottom layer's
		/// data. (Bottom layer is the earlier layer in the computation graph
		/// during inference). Note that these values are not modified during
		/// backpropagation.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the layer output
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
		/// for additional details.</param>
		/// <param name="destDiffDesc">Tensor descriptor and pointer in device memory for the bottom layer's
		/// cumulative loss differential data (error backpropagation). (Bottom layer is
		/// the earlier layer in the computation graph during inference).</param>
		/// <param name="destDiffData">Tensor descriptor and pointer in device memory for the bottom layer's
		/// cumulative loss differential data (error backpropagation). (Bottom layer is
		/// the earlier layer in the computation graph during inference).</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnLRNCrossChannelBackward(
									  cudnnHandle                    handle,
									  cudnnLRNDescriptor             normDesc,
									  cudnnLRNMode                   lrnMode,
									  ref double alpha,
									  cudnnTensorDescriptor    srcDesc,
									  CUdeviceptr srcData,
									  cudnnTensorDescriptor    srcDiffDesc,
									  CUdeviceptr srcDiffData,
									  cudnnTensorDescriptor    destDesc,
									  CUdeviceptr destData,
									  ref double beta,
									  cudnnTensorDescriptor    destDiffDesc,
									  CUdeviceptr destDiffData);

		

		/// <summary>
		/// This function performs the forward DivisiveNormalization layer computation.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN library descriptor.</param>
		/// <param name="normDesc">Handle to a previously intialized LRN parameter descriptor. This descriptor
		/// is used for both LRN and DivisiveNormalization layers.</param>
		/// <param name="mode">DivisiveNormalization layer mode of operation. Currently only
		/// CUDNN_DIVNORM_PRECOMPUTED_MEANS is implemented. Normalization
		/// is performed using the means input tensor that is expected to be
		/// precomputed by the user.</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the layer output
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
		/// for additional details.</param>
		/// <param name="srcDesc">Tensor descriptor objects for the input and output tensors. Note that
		/// srcDesc is shared between srcData, srcMeansData, tempData, tempData2
		/// tensors.</param>
		/// <param name="srcData">Input tensor data pointer in device memory.</param>
		/// <param name="srcMeansData">Input means tensor data pointer in device memory. Note that this tensor
		/// can be NULL (in that case it's values are assumed to be zero during the
		/// computation). This tensor also doesn't have to contain means, these can
		/// be any values, a frequently used variation is a result of convolution with a
		/// normalized positive kernel (such as Gaussian).</param>
		/// <param name="tempData">Temporary tensors in device memory. These are used for computing
		/// intermediate values during the forward pass. These tensors do not have
		/// to be preserved as inputs from forward to the backward pass. Both use
		/// srcDesc as a descriptor.</param>
		/// <param name="tempData2">Temporary tensors in device memory. These are used for computing
		/// intermediate values during the forward pass. These tensors do not have
		/// to be preserved as inputs from forward to the backward pass. Both use
		/// srcDesc as a descriptor.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the layer output
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
		/// for additional details.</param>
		/// <param name="destDesc">Tensor descriptor objects for the input and output tensors. Note that
		/// srcDesc is shared between srcData, srcMeansData, tempData, tempData2
		/// tensors.</param>
		/// <param name="destData">Pointer in device memory to a tensor for the result of the forward DivisiveNormalization pass.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnDivisiveNormalizationForward(
									  cudnnHandle                    handle,
									  cudnnLRNDescriptor             normDesc,
									  cudnnDivNormMode               mode,
									  ref float alpha,
									  cudnnTensorDescriptor    srcDesc, // same desc for means, temp, temp2
									  CUdeviceptr srcData,
									  CUdeviceptr srcMeansData, // if NULL, means are assumed to be zero
									  CUdeviceptr tempData,
									  CUdeviceptr tempData2,
									  ref float beta,
									  cudnnTensorDescriptor    destDesc,
									  CUdeviceptr destData
									  );

		/// <summary>
		/// This function performs the forward DivisiveNormalization layer computation.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN library descriptor.</param>
		/// <param name="normDesc">Handle to a previously intialized LRN parameter descriptor. This descriptor
		/// is used for both LRN and DivisiveNormalization layers.</param>
		/// <param name="mode">DivisiveNormalization layer mode of operation. Currently only
		/// CUDNN_DIVNORM_PRECOMPUTED_MEANS is implemented. Normalization
		/// is performed using the means input tensor that is expected to be
		/// precomputed by the user.</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the layer output
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
		/// for additional details.</param>
		/// <param name="srcDesc">Tensor descriptor objects for the input and output tensors. Note that
		/// srcDesc is shared between srcData, srcMeansData, tempData, tempData2
		/// tensors.</param>
		/// <param name="srcData">Input tensor data pointer in device memory.</param>
		/// <param name="srcMeansData">Input means tensor data pointer in device memory. Note that this tensor
		/// can be NULL (in that case it's values are assumed to be zero during the
		/// computation). This tensor also doesn't have to contain means, these can
		/// be any values, a frequently used variation is a result of convolution with a
		/// normalized positive kernel (such as Gaussian).</param>
		/// <param name="tempData">Temporary tensors in device memory. These are used for computing
		/// intermediate values during the forward pass. These tensors do not have
		/// to be preserved as inputs from forward to the backward pass. Both use
		/// srcDesc as a descriptor.</param>
		/// <param name="tempData2">Temporary tensors in device memory. These are used for computing
		/// intermediate values during the forward pass. These tensors do not have
		/// to be preserved as inputs from forward to the backward pass. Both use
		/// srcDesc as a descriptor.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the layer output
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
		/// for additional details.</param>
		/// <param name="destDesc">Tensor descriptor objects for the input and output tensors. Note that
		/// srcDesc is shared between srcData, srcMeansData, tempData, tempData2
		/// tensors.</param>
		/// <param name="destData">Pointer in device memory to a tensor for the result of the forward DivisiveNormalization pass.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnDivisiveNormalizationForward(
									  cudnnHandle                    handle,
									  cudnnLRNDescriptor             normDesc,
									  cudnnDivNormMode               mode,
									  ref double alpha,
									  cudnnTensorDescriptor    srcDesc, // same desc for means, temp, temp2
									  CUdeviceptr srcData,
									  CUdeviceptr srcMeansData, // if NULL, means are assumed to be zero
									  CUdeviceptr tempData,
									  CUdeviceptr tempData2,
									  ref double beta,
									  cudnnTensorDescriptor    destDesc,
									  CUdeviceptr destData
									  );

		/// <summary>
		/// This function performs the backward DivisiveNormalization layer computation.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN library descriptor.</param>
		/// <param name="normDesc">Handle to a previously intialized LRN parameter descriptor (this descriptor
		/// is used for both LRN and DivisiveNormalization layers).</param>
		/// <param name="mode">DivisiveNormalization layer mode of operation. Currently only
		/// CUDNN_DIVNORM_PRECOMPUTED_MEANS is implemented. Normalization
		/// is performed using the means input tensor that is expected to be
		/// precomputed by the user.</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the layer output
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
		/// for additional details.</param>
		/// <param name="srcDesc">Tensor descriptor and pointers in device memory for the bottom layer's
		/// data and means. (Bottom layer is the earlier layer in the computation
		/// graph during inference). Note: the means tensor is expected to be
		/// precomputed by the user. It can also contain any valid values (not required
		/// to be actual means, and can be for instance a result of a convolution with
		/// a Gaussian kernel).</param>
		/// <param name="srcData">Tensor descriptor and pointers in device memory for the bottom layer's
		/// data and means. (Bottom layer is the earlier layer in the computation
		/// graph during inference). Note: the means tensor is expected to be
		/// precomputed by the user. It can also contain any valid values (not required
		/// to be actual means, and can be for instance a result of a convolution with
		/// a Gaussian kernel).</param>
		/// <param name="srcMeansData">Tensor descriptor and pointers in device memory for the bottom layer's
		/// data and means. (Bottom layer is the earlier layer in the computation
		/// graph during inference). Note: the means tensor is expected to be
		/// precomputed by the user. It can also contain any valid values (not required
		/// to be actual means, and can be for instance a result of a convolution with
		/// a Gaussian kernel).</param>
		/// <param name="srcDiffData">Tensor pointer in device memory for the top layer's cumulative loss
		/// differential data (error backpropagation). (Top layer is the later layer in
		/// the computation graph during inference).</param>
		/// <param name="tempData">Temporary tensors in device memory. These are used for computing
		/// intermediate values during the backward pass. These tensors do not have
		/// to be preserved from forward to backward pass. Both use srcDesc as a
		/// descriptor.</param>
		/// <param name="tempData2">Temporary tensors in device memory. These are used for computing
		/// intermediate values during the backward pass. These tensors do not have
		/// to be preserved from forward to backward pass. Both use srcDesc as a
		/// descriptor.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the layer output
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
		/// for additional details.</param>
		/// <param name="destDataDesc">Tensor descriptor for destDataDiff and destMeansDiff.</param>
		/// <param name="destDataDiff">Tensor pointers (in device memory) for the bottom layer's resulting
		/// differentials (data and means). Both share the same descriptor.</param>
		/// <param name="destMeansDiff">Tensor pointers (in device memory) for the bottom layer's resulting
		/// differentials (data and means). Both share the same descriptor.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnDivisiveNormalizationBackward(
									  cudnnHandle                    handle,
									  cudnnLRNDescriptor             normDesc,
									  cudnnDivNormMode               mode,
									  ref float alpha,
									  cudnnTensorDescriptor    srcDesc, // same desc for diff, means, temp, temp2
									  CUdeviceptr srcData,
									  CUdeviceptr srcMeansData, // if NULL, means are assumed to be zero
									  CUdeviceptr srcDiffData,
									  CUdeviceptr tempData,
									  CUdeviceptr tempData2,
									  ref float beta,
									  cudnnTensorDescriptor    destDataDesc, // same desc for dest, means, meansDiff
									  CUdeviceptr destDataDiff, // output data differential
									  CUdeviceptr destMeansDiff // output means differential, can be NULL
									  );


		/// <summary>
		/// This function performs the backward DivisiveNormalization layer computation.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN library descriptor.</param>
		/// <param name="normDesc">Handle to a previously intialized LRN parameter descriptor (this descriptor
		/// is used for both LRN and DivisiveNormalization layers).</param>
		/// <param name="mode">DivisiveNormalization layer mode of operation. Currently only
		/// CUDNN_DIVNORM_PRECOMPUTED_MEANS is implemented. Normalization
		/// is performed using the means input tensor that is expected to be
		/// precomputed by the user.</param>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the layer output
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
		/// for additional details.</param>
		/// <param name="srcDesc">Tensor descriptor and pointers in device memory for the bottom layer's
		/// data and means. (Bottom layer is the earlier layer in the computation
		/// graph during inference). Note: the means tensor is expected to be
		/// precomputed by the user. It can also contain any valid values (not required
		/// to be actual means, and can be for instance a result of a convolution with
		/// a Gaussian kernel).</param>
		/// <param name="srcData">Tensor descriptor and pointers in device memory for the bottom layer's
		/// data and means. (Bottom layer is the earlier layer in the computation
		/// graph during inference). Note: the means tensor is expected to be
		/// precomputed by the user. It can also contain any valid values (not required
		/// to be actual means, and can be for instance a result of a convolution with
		/// a Gaussian kernel).</param>
		/// <param name="srcMeansData">Tensor descriptor and pointers in device memory for the bottom layer's
		/// data and means. (Bottom layer is the earlier layer in the computation
		/// graph during inference). Note: the means tensor is expected to be
		/// precomputed by the user. It can also contain any valid values (not required
		/// to be actual means, and can be for instance a result of a convolution with
		/// a Gaussian kernel).</param>
		/// <param name="srcDiffData">Tensor pointer in device memory for the top layer's cumulative loss
		/// differential data (error backpropagation). (Top layer is the later layer in
		/// the computation graph during inference).</param>
		/// <param name="tempData">Temporary tensors in device memory. These are used for computing
		/// intermediate values during the backward pass. These tensors do not have
		/// to be preserved from forward to backward pass. Both use srcDesc as a
		/// descriptor.</param>
		/// <param name="tempData2">Temporary tensors in device memory. These are used for computing
		/// intermediate values during the backward pass. These tensors do not have
		/// to be preserved from forward to backward pass. Both use srcDesc as a
		/// descriptor.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the layer output
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
		/// for additional details.</param>
		/// <param name="destDataDesc">Tensor descriptor for destDataDiff and destMeansDiff.</param>
		/// <param name="destDataDiff">Tensor pointers (in device memory) for the bottom layer's resulting
		/// differentials (data and means). Both share the same descriptor.</param>
		/// <param name="destMeansDiff">Tensor pointers (in device memory) for the bottom layer's resulting
		/// differentials (data and means). Both share the same descriptor.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnDivisiveNormalizationBackward(
									  cudnnHandle                    handle,
									  cudnnLRNDescriptor             normDesc,
									  cudnnDivNormMode               mode,
									  ref double alpha,
									  cudnnTensorDescriptor    srcDesc, // same desc for diff, means, temp, temp2
									  CUdeviceptr srcData,
									  CUdeviceptr srcMeansData, // if NULL, means are assumed to be zero
									  CUdeviceptr srcDiffData,
									  CUdeviceptr tempData,
									  CUdeviceptr tempData2,
									  ref double beta,
									  cudnnTensorDescriptor    destDataDesc, // same desc for dest, means, meansDiff
									  CUdeviceptr destDataDiff, // output data differential
									  CUdeviceptr destMeansDiff // output means differential, can be NULL
									  );


        /*
        * Derives a tensor descriptor from layer data descriptor for BatchNormalization 
        * scale, invVariance, bnBias, bnScale tensors. Use this tensor desc for 
        * bnScaleBiasMeanVarDesc and bnScaleBiasDiffDesc in Batch Normalization forward and backward functions.
        */
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnDeriveBNTensorDescriptor(
                                        cudnnTensorDescriptor derivedBnDesc,
                                        cudnnTensorDescriptor xDesc,
                                        cudnnBatchNormMode mode );

/* Computes y = BN(x). Also accumulates moving averages of mean and inverse variances */
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnBatchNormalizationForwardTraining(
                                cudnnHandle handle,
                                cudnnBatchNormMode mode,

                                ref float alpha, // alpha[0] = result blend factor
                                ref float beta,  // beta[0] = dest layer blend factor

                                cudnnTensorDescriptor xDesc,
                                CUdeviceptr x,     // NxCxHxW
                                cudnnTensorDescriptor yDesc,
                                CUdeviceptr y,     // NxCxHxW

                                /* Shared desc for the next 6 tensors in the argument list.
                                   Data type to be set as follows:
                                   type = (typeOf(x) == double) ? double : float
                                   Dimensions for this descriptor depend on normalization mode
                                   - Spatial Normalization : tensors are expected to have dims 1xCx1x1
                                    (normalization is performed across NxHxW)
                                   - Per-Activation Normalization : tensors are expected to have dims of 1xCxHxW 
                                    (normalization is performed across N) */
                                cudnnTensorDescriptor bnScaleBiasMeanVarDesc,

                                // 'Gamma' and 'Beta' respectively in Ioffe and Szegedy's paper's notation
                                CUdeviceptr bnScale,
                                CUdeviceptr bnBias,

                                /* MUST use factor=1 in the very first call of a complete training cycle.
                                   Use a factor=1/(1+n) at N-th call to the function to get
                                   Cumulative Moving Average (CMA) behavior
                                   CMA[n] = (x[1]+...+x[n])/n
                                   Since CMA[n+1] = (n*CMA[n]+x[n+1])/(n+1) =
                                   ((n+1)*CMA[n]-CMA[n])/(n+1) + x[n+1]/(n+1) =
                                   CMA[n]*(1-1/(n+1)) + x[n+1]*1/(n+1) */
                                double exponentialAverageFactor,

                                /* Used in Training phase only. 
                                   runningMean = newMean*factor + runningMean*(1-factor) */
                                CUdeviceptr resultRunningMean,
                                /* Output in training mode, input in inference. Is the moving average
                                   of 1 / sqrt( epsilon + variance[x] ) */
                                CUdeviceptr resultRunningInvVariance,

                                /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
                                double epsilon,

                                /* Optionally save intermediate results from the forward pass here
                                   - can be reused to speed up backward pass. NULL if unused */
                                CUdeviceptr resultSaveMean,
                                CUdeviceptr resultSaveInvVariance );

        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnBatchNormalizationForwardTraining(
                        cudnnHandle handle,
                        cudnnBatchNormMode mode,

                        ref double alpha, // alpha[0] = result blend factor
                        ref double beta,  // beta[0] = dest layer blend factor

                        cudnnTensorDescriptor xDesc,
                        CUdeviceptr x,     // NxCxHxW
                        cudnnTensorDescriptor yDesc,
                        CUdeviceptr y,     // NxCxHxW

                        /* Shared desc for the next 6 tensors in the argument list.
                           Data type to be set as follows:
                           type = (typeOf(x) == double) ? double : float
                           Dimensions for this descriptor depend on normalization mode
                           - Spatial Normalization : tensors are expected to have dims 1xCx1x1
                            (normalization is performed across NxHxW)
                           - Per-Activation Normalization : tensors are expected to have dims of 1xCxHxW 
                            (normalization is performed across N) */
                        cudnnTensorDescriptor bnScaleBiasMeanVarDesc,

                        // 'Gamma' and 'Beta' respectively in Ioffe and Szegedy's paper's notation
                        CUdeviceptr bnScale,
                        CUdeviceptr bnBias,

                        /* MUST use factor=1 in the very first call of a complete training cycle.
                           Use a factor=1/(1+n) at N-th call to the function to get
                           Cumulative Moving Average (CMA) behavior
                           CMA[n] = (x[1]+...+x[n])/n
                           Since CMA[n+1] = (n*CMA[n]+x[n+1])/(n+1) =
                           ((n+1)*CMA[n]-CMA[n])/(n+1) + x[n+1]/(n+1) =
                           CMA[n]*(1-1/(n+1)) + x[n+1]*1/(n+1) */
                        double exponentialAverageFactor,

                        /* Used in Training phase only. 
                           runningMean = newMean*factor + runningMean*(1-factor) */
                        CUdeviceptr resultRunningMean,
                        /* Output in training mode, input in inference. Is the moving average
                           of 1 / sqrt( epsilon + variance[x] ) */
                        CUdeviceptr resultRunningInvVariance,

                        /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
                        double epsilon,

                        /* Optionally save intermediate results from the forward pass here
                           - can be reused to speed up backward pass. NULL if unused */
                        CUdeviceptr resultSaveMean,
                        CUdeviceptr resultSaveInvVariance);


        /*
        * Performs Batch Normalization during Inference: 
        * y[i] = bnScale[k]*(x[i]-estimatedMean[k])*estimatedInvVariance[k] + bnBias[k]
        * with bnScale, bnBias, runningMean, runningInvVariance tensors indexed
        * according to spatial or per-activation mode. Refer to cudnnBatchNormalizationForwardTraining
        * above for notes on function arguments.
        */
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnBatchNormalizationForwardInference(
                                        cudnnHandle handle,
                                        cudnnBatchNormMode mode,
                                        ref float alpha, // alpha[0] = result blend factor
                                        ref float beta,  // beta[0] = dest layer blend factor
                                        cudnnTensorDescriptor xDesc,
                                        CUdeviceptr x,     // NxCxHxW
                                        cudnnTensorDescriptor yDesc,
                                        CUdeviceptr y,     // NxCxHxW
                                        cudnnTensorDescriptor bnScaleBiasMeanVarDesc,
                                        CUdeviceptr bnScale,
                                        CUdeviceptr bnBias,
                                        CUdeviceptr estimatedMean,
                                        CUdeviceptr estimatedInvVariance,
                                        double epsilon);

        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnBatchNormalizationForwardInference(
                                        cudnnHandle handle,
                                        cudnnBatchNormMode mode,
                                        ref double alpha, // alpha[0] = result blend factor
                                        ref double beta,  // beta[0] = dest layer blend factor
                                        cudnnTensorDescriptor xDesc,
                                        CUdeviceptr x,     // NxCxHxW
                                        cudnnTensorDescriptor yDesc,
                                        CUdeviceptr y,     // NxCxHxW
                                        cudnnTensorDescriptor bnScaleBiasMeanVarDesc,
                                        CUdeviceptr bnScale,
                                        CUdeviceptr bnBias,
                                        CUdeviceptr estimatedMean,
                                        CUdeviceptr estimatedInvVariance,
                                        double epsilon );

        /* Performs backward pass of Batch Normalization layer. Returns x gradient,
        * bnScale gradient and bnBias gradient */
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnBatchNormalizationBackward(
                                        cudnnHandle handle,
                                        cudnnBatchNormMode mode,
                                        ref float alphaDataDiff,
                                        ref float betaDataDiff,
                                        ref float alphaParamDiff,
                                        ref float betaParamDiff,
                                        cudnnTensorDescriptor xDesc, // same desc for x, dx, dy
                                        CUdeviceptr x,
                                        cudnnTensorDescriptor dyDesc,
                                        CUdeviceptr dy,
                                        cudnnTensorDescriptor dxDesc,
                                        CUdeviceptr dx,
                                        /* Shared tensor desc for the 4 tensors below */
                                        cudnnTensorDescriptor dBnScaleBiasDesc,
                                        CUdeviceptr bnScale, // bnBias doesn't affect backpropagation
                                                             /* scale and bias diff are not backpropagated below this layer */
                                        CUdeviceptr dBnScaleResult,
                                        CUdeviceptr dBnBiasResult,
                                        /* Same epsilon as forward pass */
                                        double epsilon,

                                        /* Optionally cached intermediate results from
                                           forward pass */
                                        CUdeviceptr savedMean,
                                        CUdeviceptr savedInvVariance );


        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnBatchNormalizationBackward(
                                        cudnnHandle handle,
                                        cudnnBatchNormMode mode,
                                        ref double alphaDataDiff,
                                        ref double betaDataDiff,
                                        ref double alphaParamDiff,
                                        ref double betaParamDiff,
                                        cudnnTensorDescriptor xDesc, // same desc for x, dx, dy
                                        CUdeviceptr x,
                                        cudnnTensorDescriptor dyDesc,
                                        CUdeviceptr dy,
                                        cudnnTensorDescriptor dxDesc,
                                        CUdeviceptr dx,
                                        /* Shared tensor desc for the 4 tensors below */
                                        cudnnTensorDescriptor dBnScaleBiasDesc,
                                        CUdeviceptr bnScale, // bnBias doesn't affect backpropagation
                                                             /* scale and bias diff are not backpropagated below this layer */
                                        CUdeviceptr dBnScaleResult,
                                        CUdeviceptr dBnBiasResult,
                                        /* Same epsilon as forward pass */
                                        double epsilon,

                                        /* Optionally cached intermediate results from
                                           forward pass */
                                        CUdeviceptr savedMean,
                                        CUdeviceptr savedInvVariance);

    }
}
