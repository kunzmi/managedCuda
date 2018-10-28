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

namespace ManagedCuda.CudaDNN
{
	/// <summary/>
	public static class CudaDNNNativeMethods
	{
		internal const string CUDNN_API_DLL_NAME = "cudnn64_7.dll";
		/// <summary>
		/// Gives the version of the wrapped api
		/// </summary>
		public static Version Version
		{
			get { return new Version(7, 0, 5); }
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
			SizeT maj = ver / 1000;
			SizeT min = (ver % 1000) / 100;
            SizeT build = ver % 100;
			return new Version(maj, min, build);
		}

        /// <summary>
        /// 
        /// </summary>
        /// <param name="type"></param>
        /// <param name="value"></param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern void cudnnGetProperty(libraryPropertyType type, ref int value);

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
        /// 
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="rstatus"></param>
        /// <param name="mode"></param>
        /// <param name="tag"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnQueryRuntimeError(cudnnHandle handle, ref cudnnStatus rstatus, cudnnErrQueryMode mode, cudnnRuntimeTag tag);

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
        /// This function initializes a previously created generic Tensor descriptor object.
        /// </summary>
        /// <param name="tensorDesc">Handle to a previously created tensor descriptor.</param>
        /// <param name="format"></param>
        /// <param name="dataType">Data type.</param>
        /// <param name="nbDims">Dimension of the tensor.</param>
        /// <param name="dimA">Array of dimension nbDims that contain the size of the tensor for every dimension.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSetTensorNdDescriptorEx(
                                cudnnTensorDescriptor tensorDesc,
                                cudnnTensorFormat format,
                                cudnnDataType dataType,
                                int nbDims,
                                int[] dimA );


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
        /// This function returns the size of the tensor in memory in respect to the given descriptor.
        /// This function can be used to know the amount of GPU memory to be allocated to hold that tensor.
        /// </summary>
        /// <param name="tensorDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="size">Size in bytes needed to hold the tensor in GPU memory.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetTensorSizeInBytes(
                                cudnnTensorDescriptor tensorDesc,
                                ref SizeT size);


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
		/// <param name="xDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="x">Pointer to data of the tensor described by the srcDesc descriptor.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the source
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*srcValue + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="yDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="y">Pointer to data of the tensor described by the destDesc descriptor.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnTransformTensor(   cudnnHandle                    handle,
														  ref float alpha,
														  cudnnTensorDescriptor    xDesc,
														  CUdeviceptr x,
														  ref float beta,
														  cudnnTensorDescriptor    yDesc,
														  CUdeviceptr y
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
		/// <param name="xDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="x">Pointer to data of the tensor described by the srcDesc descriptor.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the source
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*srcValue + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="yDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="y">Pointer to data of the tensor described by the destDesc descriptor.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnTransformTensor(   cudnnHandle                    handle,
														  ref double alpha,
														  cudnnTensorDescriptor    xDesc,
														  CUdeviceptr x,
														  ref double beta,
														  cudnnTensorDescriptor    yDesc,
														  CUdeviceptr y
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
        /// <param name="aDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="a">Pointer to data of the tensor described by the biasDesc descriptor.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the source
        /// value with prior value in the destination tensor as follows: dstValue =
        /// alpha[0]*srcValue + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="cDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="c">Pointer to data of the tensor described by the srcDestDesc descriptor.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnAddTensor(cudnnHandle                    handle,
                                            ref float alpha,
                                            cudnnTensorDescriptor aDesc,
                                            CUdeviceptr a,
											ref float beta,
                                            cudnnTensorDescriptor cDesc,
											CUdeviceptr c
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
        /// <param name="aDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="a">Pointer to data of the tensor described by the biasDesc descriptor.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the source
        /// value with prior value in the destination tensor as follows: dstValue =
        /// alpha[0]*srcValue + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="cDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="cData">Pointer to data of the tensor described by the srcDestDesc descriptor.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnAddTensor(cudnnHandle handle,
                                            ref double alpha,
                                            cudnnTensorDescriptor aDesc,
                                            CUdeviceptr a,
                                            ref double beta,
                                            cudnnTensorDescriptor cDesc,
                                            CUdeviceptr cData
                                          );

		/// <summary>
		/// 
		/// </summary>
		/// <param name="opTensorDesc"></param>
		/// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnCreateOpTensorDescriptor(
                                        ref cudnnOpTensorDescriptor opTensorDesc);

		/// <summary>
		/// 
		/// </summary>
		/// <param name="opTensorDesc"></param>
		/// <param name="opTensorOp"></param>
		/// <param name="opTensorCompType"></param>
		/// <param name="opTensorNanOpt"></param>
		/// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSetOpTensorDescriptor(
                                        cudnnOpTensorDescriptor opTensorDesc,
                                        cudnnOpTensorOp opTensorOp,
                                        cudnnDataType opTensorCompType,
                                        cudnnNanPropagation opTensorNanOpt);

		/// <summary>
		/// 
		/// </summary>
		/// <param name="opTensorDesc"></param>
		/// <param name="opTensorOp"></param>
		/// <param name="opTensorCompType"></param>
		/// <param name="opTensorNanOpt"></param>
		/// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetOpTensorDescriptor(
                                cudnnOpTensorDescriptor opTensorDesc,
                                ref cudnnOpTensorOp                  opTensorOp,
                                ref cudnnDataType opTensorCompType,
                                ref cudnnNanPropagation              opTensorNanOpt );

		/// <summary>
		/// 
		/// </summary>
		/// <param name="opTensorDesc"></param>
		/// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnDestroyOpTensorDescriptor(
                                cudnnOpTensorDescriptor opTensorDesc);

        /// <summary>
        /// This function implements the equation C = op ( alpha1[0] * A, alpha2[0] * B ) + beta[0] * C, given 
        /// tensors A, B, and C and scaling factors alpha1, alpha2, and beta. The op to use is indicated by 
        /// the descriptor opTensorDesc. Currently-supported ops are listed by the cudnnOpTensorOp_t enum. 
        /// Each dimension of the input tensor A must match the corresponding dimension of the destination 
        /// tensor C, and each dimension of the input tensor B must match the corresponding dimension of the 
        /// destination tensor C or must be equal to 1. In the latter case, the same value from the input tensor 
        /// B for those dimensions will be used to blend into the C tensor. The data types of the input tensors 
        /// A and B must match. If the data type of the destination tensor C is double, then the data type of 
        /// the input tensors also must be double. If the data type of the destination tensor C is double, then 
        /// opTensorCompType in opTensorDesc must be double. Else opTensorCompType must be float. If the input 
        /// tensor B is the same tensor as the destination tensor C, then the input tensor A also must be the 
        /// same tensor as the destination tensor C.
        /// </summary>        
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="opTensorDesc">Handle to a previously initialized op tensor descriptor.</param>
        /// <param name="alpha1">Pointer to scaling factors(in host memory) used to blend the source value with prior value in the destination tensor as indicated by the above op equation.</param>
        /// <param name="aDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="A">Pointer to data of the tensors described by the aDesc descriptor.</param>
        /// <param name="alpha2">Pointer to scaling factors(in host memory) used to blend the source value with prior value in the destination tensor as indicated by the above op equation.</param>
        /// <param name="bDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="B">Pointer to data of the tensors described by the bDesc descriptor.</param>
        /// <param name="beta">Pointer to scaling factors(in host memory) used to blend the source value with prior value in the destination tensor as indicated by the above op equation.</param>
        /// <param name="cDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="C">Pointer to data of the tensor described by the cDesc descriptor.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnOpTensor(
                                        cudnnHandle handle,
                                cudnnOpTensorDescriptor opTensorDesc,
                                ref float alpha1,
                                cudnnTensorDescriptor aDesc,
                                CUdeviceptr A,
                                ref float alpha2,
                                cudnnTensorDescriptor bDesc,
                                CUdeviceptr B,
                                ref float beta,
                                cudnnTensorDescriptor cDesc,
                                CUdeviceptr C );

        /// <summary>
        /// This function implements the equation C = op ( alpha1[0] * A, alpha2[0] * B ) + beta[0] * C, given 
        /// tensors A, B, and C and scaling factors alpha1, alpha2, and beta. The op to use is indicated by 
        /// the descriptor opTensorDesc. Currently-supported ops are listed by the cudnnOpTensorOp_t enum. 
        /// Each dimension of the input tensor A must match the corresponding dimension of the destination 
        /// tensor C, and each dimension of the input tensor B must match the corresponding dimension of the 
        /// destination tensor C or must be equal to 1. In the latter case, the same value from the input tensor 
        /// B for those dimensions will be used to blend into the C tensor. The data types of the input tensors 
        /// A and B must match. If the data type of the destination tensor C is double, then the data type of 
        /// the input tensors also must be double. If the data type of the destination tensor C is double, then 
        /// opTensorCompType in opTensorDesc must be double. Else opTensorCompType must be float. If the input 
        /// tensor B is the same tensor as the destination tensor C, then the input tensor A also must be the 
        /// same tensor as the destination tensor C.
        /// </summary>        
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="opTensorDesc">Handle to a previously initialized op tensor descriptor.</param>
        /// <param name="alpha1">Pointer to scaling factors(in host memory) used to blend the source value with prior value in the destination tensor as indicated by the above op equation.</param>
        /// <param name="aDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="A">Pointer to data of the tensors described by the aDesc descriptor.</param>
        /// <param name="alpha2">Pointer to scaling factors(in host memory) used to blend the source value with prior value in the destination tensor as indicated by the above op equation.</param>
        /// <param name="bDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="B">Pointer to data of the tensors described by the bDesc descriptor.</param>
        /// <param name="beta">Pointer to scaling factors(in host memory) used to blend the source value with prior value in the destination tensor as indicated by the above op equation.</param>
        /// <param name="cDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="C">Pointer to data of the tensor described by the cDesc descriptor.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnOpTensor(
                                        cudnnHandle handle,
                                cudnnOpTensorDescriptor opTensorDesc,
                                ref double alpha1,
                                cudnnTensorDescriptor aDesc,
                                CUdeviceptr A,
                                ref double alpha2,
                                cudnnTensorDescriptor bDesc,
                                CUdeviceptr B,
                                ref double beta,
                                cudnnTensorDescriptor cDesc,
                                CUdeviceptr C);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="reduceTensorDesc"></param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnCreateReduceTensorDescriptor(
                                ref cudnnReduceTensorDescriptor reduceTensorDesc);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="reduceTensorDesc"></param>
        /// <param name="reduceTensorOp"></param>
        /// <param name="reduceTensorCompType"></param>
        /// <param name="reduceTensorNanOpt"></param>
        /// <param name="reduceTensorIndices"></param>
        /// <param name="reduceTensorIndicesType"></param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSetReduceTensorDescriptor(
                                        cudnnReduceTensorDescriptor reduceTensorDesc,
                                        cudnnReduceTensorOp reduceTensorOp,
                                        cudnnDataType reduceTensorCompType,
                                        cudnnNanPropagation reduceTensorNanOpt,
                                        cudnnReduceTensorIndices reduceTensorIndices,
                                        cudnnIndicesType reduceTensorIndicesType);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="reduceTensorDesc"></param>
        /// <param name="reduceTensorOp"></param>
        /// <param name="reduceTensorCompType"></param>
        /// <param name="reduceTensorNanOpt"></param>
        /// <param name="reduceTensorIndices"></param>
        /// <param name="reduceTensorIndicesType"></param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetReduceTensorDescriptor(
                                cudnnReduceTensorDescriptor reduceTensorDesc,
                                ref cudnnReduceTensorOp                  reduceTensorOp,
                                ref cudnnDataType reduceTensorCompType,
                                ref cudnnNanPropagation              reduceTensorNanOpt,
                                ref cudnnReduceTensorIndices reduceTensorIndices,
                                ref cudnnIndicesType                 reduceTensorIndicesType );

        /// <summary>
        /// 
        /// </summary>
        /// <param name="reduceTensorDesc"></param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnDestroyReduceTensorDescriptor(
                                cudnnReduceTensorDescriptor reduceTensorDesc);

        /* Helper function to return the minimum size of the index space to be passed to the reduction given the input and output tensors */
        /// <summary>
        /// Helper function to return the minimum size of the index space to be passed to the reduction given the input and output tensors
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="reduceTensorDesc"></param>
        /// <param name="aDesc"></param>
        /// <param name="cDesc"></param>
        /// <param name="sizeInBytes"></param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetReductionIndicesSize(
                                        cudnnHandle handle,
                                cudnnReduceTensorDescriptor reduceTensorDesc,
                                cudnnTensorDescriptor aDesc,
                                cudnnTensorDescriptor cDesc,
                                ref SizeT sizeInBytes );

        /// <summary>
        /// Helper function to return the minimum size of the workspace to be passed to the reduction given the input and output tensors
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="reduceTensorDesc"></param>
        /// <param name="aDesc"></param>
        /// <param name="cDesc"></param>
        /// <param name="sizeInBytes"></param>
        /// <returns></returns>
        /* Helper function to return the minimum size of the workspace to be passed to the reduction given the input and output tensors */
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetReductionWorkspaceSize(
                                cudnnHandle handle,
                                cudnnReduceTensorDescriptor reduceTensorDesc,
                                cudnnTensorDescriptor aDesc,
                                cudnnTensorDescriptor cDesc,
                                ref SizeT sizeInBytes );
        
        /* Tensor operation : C = reduce op( alpha * A ) + beta * C */
        /* The NaN propagation enum applies to only the min and max reduce ops; the other reduce ops propagate NaN as usual. */
        /* The indices space is ignored for reduce ops other than min or max. */
        /// <summary>
        /// This function reduces tensor A by implementing the equation C = alpha * reduce op ( A )
        /// + beta* C, given tensors A and C and scaling factors alpha and beta.The reduction op
        /// to use is indicated by the descriptor reduceTensorDesc.Currently-supported ops are
        /// listed by the cudnnReduceTensorOp_t enum.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="reduceTensorDesc">Handle to a previously initialized reduce tensor descriptor.</param>
        /// <param name="indices">Handle to a previously allocated space for writing indices.</param>
        /// <param name="indicesSizeInBytes">Size of the above previously allocated space.</param>
        /// <param name="workspace">Handle to a previously allocated space for the reduction implementation.</param>
        /// <param name="workspaceSizeInBytes">Size of the above previously allocated space.</param>
        /// <param name="alpha">Pointer to scaling factor (in host memory) used to blend the source value
        /// with prior value in the destination tensor as indicated by the above op equation.</param>
        /// <param name="aDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="A">Pointer to data of the tensor described by the aDesc descriptor.</param>
        /// <param name="beta">Pointer to scaling factor (in host memory) used to blend the source value
        /// with prior value in the destination tensor as indicated by the above op equation.</param>
        /// <param name="cDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="C">Pointer to data of the tensor described by the cDesc descriptor.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnReduceTensor(
                                cudnnHandle handle,
                                cudnnReduceTensorDescriptor reduceTensorDesc,
                                CUdeviceptr indices,
                                SizeT                              indicesSizeInBytes,
                                CUdeviceptr workspace,
                                SizeT                              workspaceSizeInBytes,
                                ref float alpha,
                                cudnnTensorDescriptor aDesc,
                                CUdeviceptr A,
                                ref float beta,
                                cudnnTensorDescriptor cDesc,
                                CUdeviceptr C );

        /// <summary>
        /// This function reduces tensor A by implementing the equation C = alpha * reduce op ( A )
        /// + beta* C, given tensors A and C and scaling factors alpha and beta.The reduction op
        /// to use is indicated by the descriptor reduceTensorDesc.Currently-supported ops are
        /// listed by the cudnnReduceTensorOp_t enum.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="reduceTensorDesc">Handle to a previously initialized reduce tensor descriptor.</param>
        /// <param name="indices">Handle to a previously allocated space for writing indices.</param>
        /// <param name="indicesSizeInBytes">Size of the above previously allocated space.</param>
        /// <param name="workspace">Handle to a previously allocated space for the reduction implementation.</param>
        /// <param name="workspaceSizeInBytes">Size of the above previously allocated space.</param>
        /// <param name="alpha">Pointer to scaling factor (in host memory) used to blend the source value
        /// with prior value in the destination tensor as indicated by the above op equation.</param>
        /// <param name="aDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="A">Pointer to data of the tensor described by the aDesc descriptor.</param>
        /// <param name="beta">Pointer to scaling factor (in host memory) used to blend the source value
        /// with prior value in the destination tensor as indicated by the above op equation.</param>
        /// <param name="cDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="C">Pointer to data of the tensor described by the cDesc descriptor.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnReduceTensor(
                                cudnnHandle handle,
                                cudnnReduceTensorDescriptor reduceTensorDesc,
                                CUdeviceptr indices,
                                SizeT indicesSizeInBytes,
                                CUdeviceptr workspace,
                                SizeT workspaceSizeInBytes,
                                ref double alpha,
                                cudnnTensorDescriptor aDesc,
                                CUdeviceptr A,
                                ref double beta,
                                cudnnTensorDescriptor cDesc,
                                CUdeviceptr C);



        /// <summary>
        /// This function sets all the elements of a tensor to a given value
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="yDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="y">Pointer to data of the tensor described by the srcDestDesc descriptor.</param>
        /// <param name="value">Pointer in Host memory to a value that all elements of the tensor will be set to.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSetTensor( cudnnHandle                   handle,
												  cudnnTensorDescriptor   yDesc,
												  CUdeviceptr y,
												  ref float value
												 );

        /// <summary>
        /// This function sets all the elements of a tensor to a given value
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="yDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="y">Pointer to data of the tensor described by the srcDestDesc descriptor.</param>
        /// <param name="value">Pointer in Host memory to a value that all elements of the tensor will be set to.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSetTensor(cudnnHandle handle,
                                                  cudnnTensorDescriptor yDesc,
                                                  CUdeviceptr y,
                                                  ref double value
                                                 );

        /// <summary>
        /// This function scale all the elements of a tensor by a give factor.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="yDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="y">Pointer to data of the tensor described by the srcDestDesc descriptor.</param>
        /// <param name="alpha">Pointer in Host memory to a value that all elements of the tensor will be scaled with.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnScaleTensor(   cudnnHandle                    handle,
													  cudnnTensorDescriptor    yDesc,
													  CUdeviceptr y,
													  ref float alpha
												  );

		/// <summary>
		/// This function scale all the elements of a tensor by a give factor.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="yDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="y">Pointer to data of the tensor described by the srcDestDesc descriptor.</param>
		/// <param name="alpha">Pointer in Host memory to a value that all elements of the tensor will be scaled with.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnScaleTensor(   cudnnHandle                    handle,
													  cudnnTensorDescriptor    yDesc,
													  CUdeviceptr y,
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
		/// <param name="format">Layout format.</param>
        /// <param name="k">Number of output feature maps.</param>
        /// <param name="c">Number of input feature maps.</param>
        /// <param name="h">Height of each filter.</param>
        /// <param name="w">Width of each filter.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
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
        [DllImport(CUDNN_API_DLL_NAME)]
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
		/// <param name="format">Layout format.</param>
        /// <param name="nbDims">Dimension of the filter.</param>
        /// <param name="filterDimA">Array of dimension nbDims containing the size of the filter for each dimension.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
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
		/// <param name="format">Layout format.</param>
        /// <param name="nbDims">Actual dimension of the filter.</param>
        /// <param name="filterDimA">Array of dimension of at least nbDimsRequested that will be filled with
        /// the filter parameters from the provided filter descriptor.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
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
        /// 
        /// </summary>
        /// <param name="convDesc"></param>
        /// <param name="mathType"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSetConvolutionMathType(cudnnConvolutionDescriptor convDesc, cudnnMathType mathType);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="convDesc"></param>
        /// <param name="mathType"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetConvolutionMathType(cudnnConvolutionDescriptor convDesc, ref cudnnMathType mathType);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="convDesc"></param>
        /// <param name="groupCount"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor convDesc, int groupCount);
        /// <summary>
        /// 
        /// </summary>
        /// <param name="convDesc"></param>
        /// <param name="groupCount"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetConvolutionGroupCount(cudnnConvolutionDescriptor convDesc, ref int groupCount);

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
        /// <param name="dilation_h">Filter height dilation.</param>
        /// <param name="dilation_w">Filter width dilation.</param>
        /// <param name="mode">Selects between CUDNN_CONVOLUTION and CUDNN_CROSS_CORRELATION.</param>
        /// <param name="dataType">Selects the datatype in which the computation will be done.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSetConvolution2dDescriptor(  cudnnConvolutionDescriptor convDesc,
																	int pad_h,    // zero-padding height
																	int pad_w,    // zero-padding width
																	int u,        // vertical filter stride
																	int v,        // horizontal filter stride
																	int dilation_h, // filter dilation in the vertical dimension
                                                                    int dilation_w, // filter dilation in the horizontal dimension
                                                                    cudnnConvolutionMode mode,
                                                                    cudnnDataType dataType
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
        /// <param name="dilation_h">Filter height dilation.</param>
        /// <param name="dilation_w">Filter width dilation.</param>
		/// <param name="mode">Selects between CUDNN_CONVOLUTION and CUDNN_CROSS_CORRELATION.</param>
		/// <param name="dataType">Data type.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetConvolution2dDescriptor(   cudnnConvolutionDescriptor convDesc,
																	 ref int pad_h,    // zero-padding height
																	 ref int pad_w,    // zero-padding width
																	 ref int u,        // vertical filter stride
																	 ref int v,        // horizontal filter stride
																	 ref int dilation_h, // filter dilation in the vertical dimension
                                                                     ref int dilation_w, // filter dilation in the horizontal dimension
                                                                     ref cudnnConvolutionMode mode,
                                                                     ref cudnnDataType dataType
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
        /// <param name="dilationA">Array of dimension arrayLength containing the dilation factor for each dimension.</param>
        /// <param name="mode">Selects between CUDNN_CONVOLUTION and CUDNN_CROSS_CORRELATION.</param>
        /// <param name="computeType">Selects the datatype in which the computation will be done.</param>
        [DllImport(CUDNN_API_DLL_NAME)]  
		public static extern cudnnStatus cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor convDesc,
                                                              int arrayLength,             /* nbDims-2 size */  
                                                              int[] padA,                                          
                                                              int[] filterStrideA,         
                                                              int[] dilationA,              
                                                              cudnnConvolutionMode mode,
                                                              cudnnDataType computeType   // convolution data type
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
        /// <param name="dilationA">Array of dimension at least arrayLengthRequested that will be filled
        /// with the dilation parameters from the provided convolution descriptor.</param>
        /// <param name="mode">convolution mode of the provided descriptor.</param>
        /// <param name="computeType">datatype of the provided descriptor.</param>
        [DllImport(CUDNN_API_DLL_NAME)]  
		public static extern cudnnStatus cudnnGetConvolutionNdDescriptor(cudnnConvolutionDescriptor convDesc,
                                                              int arrayLengthRequested,
                                                              ref int arrayLength,
                                                              int[] padA,                                        
                                                              int[] strideA,
                                                              int[] dilationA,
                                                              ref cudnnConvolutionMode mode,
                                                              ref cudnnDataType computeType     // convolution data type
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
        /// 
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="count"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle handle, ref int count);


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
        /// This function attempts all available cuDNN algorithms for cudnnConvolutionForward, using 
        /// user-allocated GPU memory, and outputs performance metrics to a user-allocated array of 
        /// cudnnConvolutionFwdAlgoPerf_t. These metrics are written in sorted fashion where the first 
        /// element has the lowest compute time. The workspace size should be the largest workspace you 
        /// can spare in device memory; the size of this workspace will determine the availablity of 
        /// the convolution algorithms.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor xDesc.</param>
        /// <param name="wDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor wDesc.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the tensor descriptor yDesc. The content of this tensor will be overwritten with arbitary values.</param>
        /// <param name="requestedAlgoCount">The maximum number of elements to be stored in perfResults.</param>
        /// <param name="returnedAlgoCount">The number of output elements stored in perfResults.</param>
        /// <param name="perfResults">A user-allocated array to store performance metrics sorted ascending by compute time.</param>
        /// <param name="workSpace">Data pointer to GPU memory that is a necessary workspace for some algorithms. The size of this workspace will determine the availability of algorithms. A nil pointer is considered a workSpace of 0 bytes.</param>
        /// <param name="workSpaceSizeInBytes">Specifies the size in bytes of the provided workSpace.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnFindConvolutionForwardAlgorithmEx(
                                        cudnnHandle handle,
                                        cudnnTensorDescriptor xDesc,
                                        CUdeviceptr x,
                                        cudnnFilterDescriptor wDesc,
                                        CUdeviceptr w,
                                        cudnnConvolutionDescriptor convDesc,
                                        cudnnTensorDescriptor yDesc,
                                        CUdeviceptr y,
                                        int requestedAlgoCount,
                                        ref int returnedAlgoCount,
                                        ref cudnnConvolutionFwdAlgoPerf      perfResults,
                                        CUdeviceptr workSpace,
                                        SizeT                              workSpaceSizeInBytes );


		/// <summary>
		/// This function serves as a heuristic for obtaining the best suited algorithm for
		/// cudnnConvolutionForward for the given layer specifications. Based on the input
		/// preference, this function will either return the fastest algorithm or the fastest algorithm
		/// within a given memory limit. For an exhaustive search for the fastest algorithm, please
		/// use cudnnFindConvolutionForwardAlgorithm.
		/// </summary>
		/// <param name="handle">Handle to a previously created cuDNN context.</param>
		/// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="preference">Enumerant to express the preference criteria in terms of memory
		/// requirement and speed.</param>
		/// <param name="memoryLimitInbytes">It is used when enumerant preference is set to
		/// CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT to specify the
		/// maximum amount of GPU memory the user is willing to use as a workspace</param>
		/// <param name="algo">Enumerant that specifies which convolution algorithm should be used to
		/// compute the results according to the specified preference</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetConvolutionForwardAlgorithm( cudnnHandle                      handle,
																	   cudnnTensorDescriptor      xDesc,
																	   cudnnFilterDescriptor      filterDesc,
																	   cudnnConvolutionDescriptor convDesc, 
																	   cudnnTensorDescriptor      yDesc,
																	   cudnnConvolutionFwdPreference    preference, 
																	   SizeT                             memoryLimitInbytes,
																	   ref cudnnConvolutionFwdAlgo         algo                                                  
																	 );


        /// <summary>
        /// 
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="srcDesc"></param>
        /// <param name="filterDesc"></param>
        /// <param name="convDesc"></param>
        /// <param name="destDesc"></param>
        /// <param name="requestedAlgoCount"></param>
        /// <param name="returnedAlgoCount"></param>
        /// <param name="perfResults"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle handle, cudnnTensorDescriptor srcDesc, cudnnFilterDescriptor filterDesc,
            cudnnConvolutionDescriptor convDesc, cudnnTensorDescriptor destDesc, int requestedAlgoCount, ref int returnedAlgoCount, cudnnConvolutionFwdAlgoPerf[] perfResults);

              
                                                                                                           
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
		/// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="algo">Enumerant that specifies the chosen convolution algorithm</param>
		/// <param name="sizeInBytes">Amount of GPU memory needed as workspace to be able to execute a
		/// forward convolution with the specified algo</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetConvolutionForwardWorkspaceSize( cudnnHandle                      handle, 
																		   cudnnTensorDescriptor      xDesc,
																		   cudnnFilterDescriptor      filterDesc,
																		   cudnnConvolutionDescriptor convDesc,  
																		   cudnnTensorDescriptor      yDesc,
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
		/// <param name="xDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="wDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="w">Data pointer to GPU memory associated with the filter descriptor filterDesc.</param>
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
		/// <param name="yDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="y">Data pointer to GPU memory associated with the tensor descriptor
		/// destDesc that carries the result of the convolution.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionForward( cudnnHandle handle,
																  ref float alpha,
																  cudnnTensorDescriptor xDesc,
																  CUdeviceptr x,
																  cudnnFilterDescriptor wDesc,
																  CUdeviceptr w,
																  cudnnConvolutionDescriptor convDesc,
																  cudnnConvolutionFwdAlgo algo,
																  CUdeviceptr workSpace,
																  SizeT workSpaceSizeInBytes,            
																  ref float beta,
																  cudnnTensorDescriptor yDesc,
																  CUdeviceptr y
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
		/// <param name="xDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="wDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="w">Data pointer to GPU memory associated with the filter descriptor filterDesc.</param>
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
		/// <param name="yDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="y">Data pointer to GPU memory associated with the tensor descriptor
		/// destDesc that carries the result of the convolution.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionForward( cudnnHandle handle,
																  ref double alpha,
																  cudnnTensorDescriptor xDesc,
																  CUdeviceptr x,
																  cudnnFilterDescriptor wDesc,
																  CUdeviceptr w,
																  cudnnConvolutionDescriptor convDesc,
																  cudnnConvolutionFwdAlgo algo,
																  CUdeviceptr workSpace,
																  SizeT workSpaceSizeInBytes,
																  ref double beta,
																  cudnnTensorDescriptor yDesc,
																  CUdeviceptr y
														 );

        /* Fused conv/bias/activation operation : y = Act( alpha1 * conv(x) + alpha2 * z + bias ) */
        /// <summary>
        /// This function applies a bias and then an activation to the convolutions or crosscorrelations
        /// of cudnnConvolutionForward(), returning results in y.The full computation
        /// follows the equation y = act(alpha1* conv(x) + alpha2* z + bias ).<para/>
        /// The routine cudnnGetConvolution2dForwardOutputDim or
        /// cudnnGetConvolutionNdForwardOutputDim can be used to determine the proper
        /// dimensions of the output tensor descriptor yDesc with respect to xDesc, convDesc and wDesc.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="alpha1">Pointers to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as described by the above equation.</param>
        /// <param name="xDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor xDesc.</param>
        /// <param name="wDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor wDesc.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="algo">Enumerant that specifies which convolution algorithm shoud be used to compute the results</param>
        /// <param name="workSpace">Data pointer to GPU memory to a workspace needed to able to execute
        /// the specified algorithm.If no workspace is needed for a particular
        /// algorithm, that pointer can be nil</param>
        /// <param name="workSpaceSizeInBytes">Specifies the size in bytes of the provided workSpace</param>
        /// <param name="alpha2">Pointers to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as described by the above equation.</param>
        /// <param name="zDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="z">Data pointer to GPU memory associated with the tensor descriptor zDesc.</param>
        /// <param name="biasDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="bias">Data pointer to GPU memory associated with the tensor descriptor biasDesc.</param>
        /// <param name="activationDesc">Handle to a previously initialized activation descriptor.</param>
        /// <param name="yDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the tensor descriptor yDesc
        /// that carries the result of the convolution.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnConvolutionBiasActivationForward(
                                        cudnnHandle handle,
                                ref float alpha1,
                                cudnnTensorDescriptor xDesc,
                                CUdeviceptr x,
                                cudnnFilterDescriptor wDesc,
                                CUdeviceptr w,
                                cudnnConvolutionDescriptor convDesc,
                                cudnnConvolutionFwdAlgo           algo,
                                CUdeviceptr workSpace,
                                SizeT                              workSpaceSizeInBytes,
                                ref float alpha2,
                                cudnnTensorDescriptor zDesc,
                                CUdeviceptr z,
                                cudnnTensorDescriptor biasDesc,
                                CUdeviceptr bias,
                                cudnnActivationDescriptor activationDesc,
                                cudnnTensorDescriptor yDesc,
                                CUdeviceptr y );

        /// <summary>
        /// This function applies a bias and then an activation to the convolutions or crosscorrelations
        /// of cudnnConvolutionForward(), returning results in y.The full computation
        /// follows the equation y = act(alpha1* conv(x) + alpha2* z + bias ).<para/>
        /// The routine cudnnGetConvolution2dForwardOutputDim or
        /// cudnnGetConvolutionNdForwardOutputDim can be used to determine the proper
        /// dimensions of the output tensor descriptor yDesc with respect to xDesc, convDesc and wDesc.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="alpha1">Pointers to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as described by the above equation.</param>
        /// <param name="xDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor xDesc.</param>
        /// <param name="wDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor wDesc.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="algo">Enumerant that specifies which convolution algorithm shoud be used to compute the results</param>
        /// <param name="workSpace">Data pointer to GPU memory to a workspace needed to able to execute
        /// the specified algorithm.If no workspace is needed for a particular
        /// algorithm, that pointer can be nil</param>
        /// <param name="workSpaceSizeInBytes">Specifies the size in bytes of the provided workSpace</param>
        /// <param name="alpha2">Pointers to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as described by the above equation.</param>
        /// <param name="zDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="z">Data pointer to GPU memory associated with the tensor descriptor zDesc.</param>
        /// <param name="biasDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="bias">Data pointer to GPU memory associated with the tensor descriptor biasDesc.</param>
        /// <param name="activationDesc">Handle to a previously initialized activation descriptor.</param>
        /// <param name="yDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the tensor descriptor yDesc
        /// that carries the result of the convolution.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnConvolutionBiasActivationForward(
                                        cudnnHandle handle,
                                ref double alpha1,
                                cudnnTensorDescriptor xDesc,
                                CUdeviceptr x,
                                cudnnFilterDescriptor wDesc,
                                CUdeviceptr w,
                                cudnnConvolutionDescriptor convDesc,
                                cudnnConvolutionFwdAlgo algo,
                                CUdeviceptr workSpace,
                                SizeT workSpaceSizeInBytes,
                                ref double alpha2,
                                cudnnTensorDescriptor zDesc,
                                CUdeviceptr z,
                                cudnnTensorDescriptor biasDesc,
                                CUdeviceptr bias,
                                cudnnActivationDescriptor activationDesc,
                                cudnnTensorDescriptor yDesc,
                                CUdeviceptr y);

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
        /// <param name="dyDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="dbDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="db">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionBackwardBias(cudnnHandle handle,
																	  ref float alpha,
																	  cudnnTensorDescriptor dyDesc,
																	  CUdeviceptr dy,
																	  ref float beta,
																	  cudnnTensorDescriptor dbDesc,
																	  CUdeviceptr db
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
        /// <param name="dyDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="dbDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="db">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionBackwardBias(cudnnHandle handle,
																	  ref double alpha,
																	  cudnnTensorDescriptor dyDesc,
																	  CUdeviceptr dy,
																	  ref double beta,
																	  cudnnTensorDescriptor dbDesc,
																	  CUdeviceptr db
															  );


        /// <summary>
        /// 
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="count"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle handle, ref int count);



        /// <summary>
        /// This function attempts all cuDNN algorithms for cudnnConvolutionBackwardFilter_v3 and outputs performance metrics to a user-
        /// allocated array of cudnnConvolutionBwdFilterAlgoPerf_t. These metrics are
        /// written in sorted fashion where the first element has the lowest compute time. 
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="dwDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="requestedAlgoCount">The maximum number of elements to be stored in perfResults.</param>
        /// <param name="returnedAlgoCount">The number of output elements stored in perfResults.</param>
        /// <param name="perfResults">A user-allocated array to store performance metrics sorted ascending by compute time.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnFindConvolutionBackwardFilterAlgorithm( cudnnHandle     handle,
                                                                       cudnnTensorDescriptor          xDesc,
                                                                       cudnnTensorDescriptor          dyDesc,
                                                                       cudnnConvolutionDescriptor     convDesc, 
                                                                       cudnnFilterDescriptor          dwDesc,
                                                                       int                              requestedAlgoCount,
                                                                       ref int                          returnedAlgoCount,
                                                                       cudnnConvolutionBwdFilterAlgoPerf[] perfResults   
                                                                     );

        /// <summary>
        /// This function attempts all cuDNN algorithms for cudnnConvolutionBackwardFilter, 
        /// using user-allocated GPU memory, and outputs performance metrics to a 
        /// user-allocated array of cudnnConvolutionBwdFilterAlgoPerf_t. These metrics are 
        /// written in sorted fashion where the first element has the lowest compute time. The 
        /// workspace size should be the largest workspace you can spare in device memory; the 
        /// size of this workspace will determine the availablity of convolution algorithms.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor. </param>
        /// <param name="x">Data pointer to GPU memory associated with the filter descriptor xDesc.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the tensor descriptor dyDesc.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="dwDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="dw">Data pointer to GPU memory associated with the filter descriptor dwDesc.The content of this tensor will be overwritten with arbitary values.</param>
        /// <param name="requestedAlgoCount">The maximum number of elements to be stored in perfResults.</param>
        /// <param name="returnedAlgoCount">The number of output elements stored in perfResults.</param>
        /// <param name="perfResults">A user-allocated array to store performance metrics sorted ascending by compute time.</param>
        /// <param name="workSpace">Data pointer to GPU memory that is a necessary workspace for some algorithms. The size of this workspace will determine the availabilty of algorithms. A nil pointer is considered a workSpace of 0 bytes.</param>
        /// <param name="workSpaceSizeInBytes">Specifies the size in bytes of the provided workSpace.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnFindConvolutionBackwardFilterAlgorithmEx(
                                        cudnnHandle handle,
                                        cudnnTensorDescriptor xDesc,
                                        CUdeviceptr x,
                                        cudnnTensorDescriptor dyDesc,
                                        CUdeviceptr dy,
                                        cudnnConvolutionDescriptor convDesc,
                                        cudnnFilterDescriptor dwDesc,
                                        CUdeviceptr dw,
                                        int requestedAlgoCount,
                                        ref int returnedAlgoCount,
                                        ref cudnnConvolutionBwdFilterAlgoPerf perfResults,
                                        CUdeviceptr workSpace,
                                        SizeT                               workSpaceSizeInBytes );

        /// <summary>
        /// This function serves as a heuristic for obtaining the best suited algorithm for
        /// cudnnConvolutionBackwardFilter for the given layer specifications. Based
        /// on the input preference, this function will either return the fastest algorithm or the
        /// fastest algorithm within a given memory limit. For an exhaustive search for the fastest
        /// algorithm, please use cudnnFindConvolutionBackwardFilterAlgorithm.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="dwDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="preference">Enumerant to express the preference criteria in terms of memory requirement and speed.</param>
        /// <param name="memoryLimitInbytes">It is to specify the maximum amount of GPU memory the user is willing to 
        /// use as a workspace. This is currently a placeholder and is not used.</param>
        /// <param name="algo">Enumerant that specifies which convolution algorithm should be used to
        /// compute the results according to the specified preference</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetConvolutionBackwardFilterAlgorithm( cudnnHandle             handle,
                                                                      cudnnTensorDescriptor          xDesc,
                                                                      cudnnTensorDescriptor          dyDesc,
                                                                      cudnnConvolutionDescriptor     convDesc, 
                                                                      cudnnFilterDescriptor          dwDesc,
                                                                      cudnnConvolutionBwdFilterPreference  preference,
                                                                      SizeT                                memoryLimitInbytes,
                                                                      ref cudnnConvolutionBwdFilterAlgo algo
                                                                     );



        /// <summary>
        /// 
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="srcDesc"></param>
        /// <param name="diffDesc"></param>
        /// <param name="convDesc"></param>
        /// <param name="gradDesc"></param>
        /// <param name="requestedAlgoCount"></param>
        /// <param name="returnedAlgoCount"></param>
        /// <param name="perfResults"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnnHandle handle, cudnnTensorDescriptor srcDesc, cudnnTensorDescriptor diffDesc,
                                cudnnConvolutionDescriptor convDesc, cudnnFilterDescriptor gradDesc, int requestedAlgoCount, ref int returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf[] perfResults);



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
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="gradDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="algo">Enumerant that specifies the chosen convolution algorithm
        /// sizeInBytes output Amount of GPU memory needed as workspace to be able to execute</param>
        /// <param name="sizeInBytes">Amount of GPU memory needed as workspace to be able to execute a
        /// forward convolution with the specified algo</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetConvolutionBackwardFilterWorkspaceSize( cudnnHandle          handle, 
																				  cudnnTensorDescriptor       xDesc,
																				  cudnnTensorDescriptor       dyDesc,
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
		/// <param name="xDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
		/// <param name="dy">Data pointer to GPU memory associated with the input differential tensor descriptor diffDesc.</param>
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
		/// <param name="dwDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="dw">Data pointer to GPU memory associated with the filter descriptor
		/// gradDesc that carries the result.</param>    
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionBackwardFilter( cudnnHandle                 handle,
																	 ref float alpha,
																	 cudnnTensorDescriptor       xDesc,
																	 CUdeviceptr x,
																	 cudnnTensorDescriptor       dyDesc,
																	 CUdeviceptr dy,
																	 cudnnConvolutionDescriptor  convDesc,
																	 cudnnConvolutionBwdFilterAlgo     algo,
																	 CUdeviceptr workSpace,
																	 SizeT                              workSpaceSizeInBytes,
																	 ref float beta,
																	 cudnnFilterDescriptor       dwDesc,
																	 CUdeviceptr dw
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
        /// <param name="xDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the input differential tensor descriptor diffDesc.</param>
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
        /// <param name="dwDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="dw">Data pointer to GPU memory associated with the filter descriptor
        /// gradDesc that carries the result.</param>    
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnConvolutionBackwardFilter(cudnnHandle handle,
                                                                     ref double alpha,
                                                                     cudnnTensorDescriptor xDesc,
                                                                     CUdeviceptr x,
                                                                     cudnnTensorDescriptor dyDesc,
                                                                     CUdeviceptr dy,
                                                                     cudnnConvolutionDescriptor convDesc,
                                                                     cudnnConvolutionBwdFilterAlgo algo,
                                                                     CUdeviceptr workSpace,
                                                                     SizeT workSpaceSizeInBytes,
                                                                     ref double beta,
                                                                     cudnnFilterDescriptor dwDesc,
                                                                     CUdeviceptr dw
                                                                   );

        /// <summary>
        /// 
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="count"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle handle, ref int count);



        /// <summary>
        /// This function attempts all cuDNN algorithms for
        /// cudnnConvolutionBackwardData and outputs performance metrics to a user-
        /// allocated array of cudnnConvolutionBwdDataAlgoPerf_t. These metrics are written
        /// in sorted fashion where the first element has the lowest compute time.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="wDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="dxDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="requestedAlgoCount">The maximum number of elements to be stored in perfResults.</param>
        /// <param name="returnedAlgoCount">The number of output elements stored in perfResults.</param>
        /// <param name="perfResults">A user-allocated array to store performance metrics sorted ascending by compute time.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnFindConvolutionBackwardDataAlgorithm( cudnnHandle handle,
                                                                     cudnnFilterDescriptor       wDesc,
                                                                     cudnnTensorDescriptor       dyDesc,
                                                                     cudnnConvolutionDescriptor  convDesc, 
                                                                     cudnnTensorDescriptor       dxDesc,
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
		/// <param name="wDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="dxDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="preference">Enumerant to express the preference criteria in terms of memory
		/// requirement and speed.</param>
		/// <param name="memoryLimitInbytes">It is to specify the maximum amount of GPU memory the user is willing to
		/// use as a workspace. This is currently a placeholder and is not used.</param>
		/// <param name="algo">Enumerant that specifies which convolution algorithm should be used to
		/// compute the results according to the specified preference</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetConvolutionBackwardDataAlgorithm( cudnnHandle handle,
																	   cudnnFilterDescriptor       wDesc,
																	   cudnnTensorDescriptor       dyDesc,
																	   cudnnConvolutionDescriptor  convDesc, 
																	   cudnnTensorDescriptor       dxDesc,
																	   cudnnConvolutionBwdDataPreference preference, 
																	   SizeT                              memoryLimitInbytes,
																	   ref cudnnConvolutionBwdDataAlgo algo
																	 );


        /// <summary>
        /// 
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="filterDesc"></param>
        /// <param name="diffDesc"></param>
        /// <param name="convDesc"></param>
        /// <param name="gradDesc"></param>
        /// <param name="requestedAlgoCount"></param>
        /// <param name="returnedAlgoCount"></param>
        /// <param name="perfResults"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnnHandle handle, cudnnFilterDescriptor filterDesc, cudnnTensorDescriptor diffDesc,
            cudnnConvolutionDescriptor convDesc, cudnnTensorDescriptor gradDesc, int requestedAlgoCount, ref int returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf[] perfResults);



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
		/// <param name="wDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="dxDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="algo">Enumerant that specifies the chosen convolution algorithm</param>
		/// <param name="sizeInBytes">Amount of GPU memory needed as workspace to be able to execute a forward convolution with the specified algo</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetConvolutionBackwardDataWorkspaceSize( cudnnHandle handle,
																		   cudnnFilterDescriptor      wDesc,
																		   cudnnTensorDescriptor       dyDesc,
																		   cudnnConvolutionDescriptor convDesc,  
																		   cudnnTensorDescriptor       dxDesc,
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
        /// <param name="wDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor filterDesc.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the input differential tensor descriptor diffDesc.</param>
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
        /// <param name="dxDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="dx">Data pointer to GPU memory associated with the output tensor descriptor
        /// gradDesc that carries the result.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionBackwardData( cudnnHandle handle,
																 ref float alpha,
																 cudnnFilterDescriptor       wDesc,
																 CUdeviceptr w,
																 cudnnTensorDescriptor       dyDesc,
																 CUdeviceptr dy,
																 cudnnConvolutionDescriptor  convDesc,
																 cudnnConvolutionBwdDataAlgo           algo,
																 CUdeviceptr workSpace,
																 SizeT                              workSpaceSizeInBytes,
																 ref float beta,
																 cudnnTensorDescriptor       dxDesc,
																 CUdeviceptr dx
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
        /// <param name="wDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor filterDesc.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the input differential tensor descriptor diffDesc.</param>
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
        /// <param name="dxDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="dx">Data pointer to GPU memory associated with the output tensor descriptor
        /// gradDesc that carries the result.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionBackwardData( cudnnHandle handle,
																 ref double alpha,
																 cudnnFilterDescriptor       wDesc,
																 CUdeviceptr w,
																 cudnnTensorDescriptor       dyDesc,
																 CUdeviceptr dy,
																 cudnnConvolutionDescriptor  convDesc,
																 cudnnConvolutionBwdDataAlgo           algo,
																 CUdeviceptr workSpace,
																 SizeT                              workSpaceSizeInBytes,
																 ref double beta,
																 cudnnTensorDescriptor       dxDesc,
																 CUdeviceptr dx
															   );


		/// <summary>
		/// 
		/// </summary>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnIm2Col(  cudnnHandle handle,
												cudnnTensorDescriptor xDesc,
												CUdeviceptr x,
												cudnnFilterDescriptor wDesc,                                        
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
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSoftmaxForward(  cudnnHandle handle,
														cudnnSoftmaxAlgorithm algorithm,
														cudnnSoftmaxMode mode,
														ref float alpha,
														cudnnTensorDescriptor xDesc,
														CUdeviceptr x,
														ref float beta,
														cudnnTensorDescriptor yDesc,
														CUdeviceptr y
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
		/// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSoftmaxForward(  cudnnHandle handle,
														cudnnSoftmaxAlgorithm algorithm,
														cudnnSoftmaxMode mode,
														ref double alpha,
														cudnnTensorDescriptor xDesc,
														CUdeviceptr x,
														ref double beta,
														cudnnTensorDescriptor yDesc,
														CUdeviceptr y
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
        /// <param name="yDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the tensor descriptor srcDiffData.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="dxDesc">Handle to the previously initialized output differential tensor descriptor.</param>
        /// <param name="dx">Data pointer to GPU memory associated with the output tensor descriptor destDiffDesc.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSoftmaxBackward( cudnnHandle handle,
														cudnnSoftmaxAlgorithm algorithm,
														cudnnSoftmaxMode mode,
														ref float alpha,
														cudnnTensorDescriptor yDesc,
														CUdeviceptr y,
														cudnnTensorDescriptor dyDesc,
														CUdeviceptr dy,
														ref float beta,
														cudnnTensorDescriptor dxDesc,
														CUdeviceptr dx
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
        /// <param name="yDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the tensor descriptor srcDiffData.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="dxDesc">Handle to the previously initialized output differential tensor descriptor.</param>
        /// <param name="dx">Data pointer to GPU memory associated with the output tensor descriptor destDiffDesc.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSoftmaxBackward( cudnnHandle handle,
														cudnnSoftmaxAlgorithm algorithm,
														cudnnSoftmaxMode mode,
														ref double alpha,
														cudnnTensorDescriptor yDesc,
														CUdeviceptr y,
														cudnnTensorDescriptor dyDesc,
														CUdeviceptr dy,
														ref double beta,
														cudnnTensorDescriptor dxDesc,
														CUdeviceptr dx
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
        [DllImport(CUDNN_API_DLL_NAME)]
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
        [DllImport(CUDNN_API_DLL_NAME)]
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
        [DllImport(CUDNN_API_DLL_NAME)]
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
        [DllImport(CUDNN_API_DLL_NAME)]
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
		/// <param name="n">Number of images in the output</param>
		/// <param name="c">Number of channels in the output</param>
		/// <param name="h">Height of images in the output</param>
		/// <param name="w">Width of images in the output</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetPooling2dForwardOutputDim( cudnnPoolingDescriptor poolingDesc,
																	 cudnnTensorDescriptor inputTensorDesc,
																	 ref int n,
																	 ref int c,
																	 ref int h,
																	 ref int w);


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
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnPoolingForward(  cudnnHandle handle,
														cudnnPoolingDescriptor poolingDesc,
														ref float alpha,
														cudnnTensorDescriptor xDesc,
														CUdeviceptr x,
														ref float beta,
														cudnnTensorDescriptor yDesc,
														CUdeviceptr y
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
		/// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnPoolingForward(  cudnnHandle handle,
														cudnnPoolingDescriptor poolingDesc,
														ref double alpha,
														cudnnTensorDescriptor xDesc,
														CUdeviceptr x,
														ref double beta,
														cudnnTensorDescriptor yDesc,
														CUdeviceptr y
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
        /// <param name="yDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the tensor descriptor srcDiffData.</param>
        /// <param name="xDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="dxDesc">Handle to the previously initialized output differential tensor descriptor.</param>
        /// <param name="dx">Data pointer to GPU memory associated with the output tensor descriptor destDiffDesc.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnPoolingBackward( cudnnHandle handle,
														cudnnPoolingDescriptor poolingDesc,
														ref float alpha,
														cudnnTensorDescriptor yDesc,
														CUdeviceptr y,
														cudnnTensorDescriptor dyDesc,
														CUdeviceptr dy,
														cudnnTensorDescriptor xDesc,
														CUdeviceptr x,
														ref float beta,
														cudnnTensorDescriptor dxDesc,
														CUdeviceptr dx
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
        /// <param name="yDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the tensor descriptor srcDiffData.</param>
        /// <param name="xDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="dxDesc">Handle to the previously initialized output differential tensor descriptor.</param>
        /// <param name="dx">Data pointer to GPU memory associated with the output tensor descriptor destDiffDesc.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnPoolingBackward( cudnnHandle handle,
														cudnnPoolingDescriptor poolingDesc,
														ref double alpha,
														cudnnTensorDescriptor yDesc,
														CUdeviceptr y,
														cudnnTensorDescriptor dyDesc,
														CUdeviceptr dy,
														cudnnTensorDescriptor xDesc,
														CUdeviceptr x,
														ref double beta,
														cudnnTensorDescriptor dxDesc,
														CUdeviceptr dx
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
        /// <param name="coef">floating point number to specify the clipping threashold when the activation
        /// mode is set to CUDNN_ACTIVATION_CLIPPED_RELU or to specify the alpha
        /// coefficient when the activation mode is set to CUDNN_ACTIVATION_ELU.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSetActivationDescriptor(
                                        cudnnActivationDescriptor activationDesc,
                                        cudnnActivationMode mode,
                                        cudnnNanPropagation reluNanOpt,
                                        double coef); /* ceiling for clipped RELU, alpha for ELU */

        /// <summary>
        /// This function queries the parameters of the previouly initialized activation descriptor object.
        /// </summary>
        /// <param name="activationDesc">Handle to the previously created activation descriptor object.</param>
        /// <param name="mode">Enumerant to specify the activation mode.</param>
        /// <param name="reluNanOpt">Nan propagation option for the relu.</param>
        /// <param name="coef">floating point number to specify the clipping threashold when the activation
        /// mode is set to CUDNN_ACTIVATION_CLIPPED_RELU or to specify the alpha
        /// coefficient when the activation mode is set to CUDNN_ACTIVATION_ELU.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetActivationDescriptor(
                                cudnnActivationDescriptor activationDesc,
                                ref cudnnActivationMode              mode,
                                ref cudnnNanPropagation reluNanOpt,
                                ref double coef); /* ceiling for clipped RELU, alpha for ELU */

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
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnActivationForward( cudnnHandle handle,
														  cudnnActivationDescriptor activationDesc,
														  ref float alpha,
														  cudnnTensorDescriptor xDesc,
														  CUdeviceptr x,
														  ref float beta,
														  cudnnTensorDescriptor yDesc,
														  CUdeviceptr y
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
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnActivationForward( cudnnHandle handle,
                                                          cudnnActivationDescriptor activationDesc,
                                                          ref double alpha,
														  cudnnTensorDescriptor xDesc,
														  CUdeviceptr x,
														  ref double beta,
														  cudnnTensorDescriptor yDesc,
														  CUdeviceptr y
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
        /// <param name="yDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the tensor descriptor srcDiffData.</param>
        /// <param name="xDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="dxDesc">Handle to the previously initialized output differential tensor descriptor.</param>
        /// <param name="dx">Data pointer to GPU memory associated with the output tensor descriptor destDiffDesc.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnActivationBackward( cudnnHandle handle,
                                                           cudnnActivationDescriptor activationDesc,
                                                           ref float alpha,
														   cudnnTensorDescriptor yDesc,
														   CUdeviceptr y,
														   cudnnTensorDescriptor dyDesc,
														   CUdeviceptr dy,
														   cudnnTensorDescriptor xDesc,
														   CUdeviceptr x,
														   ref float beta,
														   cudnnTensorDescriptor dxDesc,
														   CUdeviceptr dx
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
        /// <param name="yDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the tensor descriptor srcDiffData.</param>
        /// <param name="xDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="dxDesc">Handle to the previously initialized output differential tensor descriptor.</param>
        /// <param name="dx">Data pointer to GPU memory associated with the output tensor descriptor destDiffDesc.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnActivationBackward( cudnnHandle handle,
                                                           cudnnActivationDescriptor activationDesc,
                                                           ref double alpha,
														   cudnnTensorDescriptor yDesc,
														   CUdeviceptr y,
														   cudnnTensorDescriptor dyDesc,
														   CUdeviceptr dy,
														   cudnnTensorDescriptor xDesc,
														   CUdeviceptr x,
														   ref double beta,
														   cudnnTensorDescriptor dxDesc,
														   CUdeviceptr dx
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
		/// <param name="xDesc">Tensor descriptor objects for the input and output tensors.</param>
		/// <param name="x">Input tensor data pointer in device memory.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the layer output
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
		/// for additional details.</param>
		/// <param name="yDesc">Tensor descriptor objects for the input and output tensors.</param>
		/// <param name="y">Output tensor data pointer in device memory.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnLRNCrossChannelForward(
									  cudnnHandle                    handle,
									  cudnnLRNDescriptor             normDesc,
									  cudnnLRNMode                   lrnMode,
									  ref float alpha,
									  cudnnTensorDescriptor    xDesc,
									  CUdeviceptr x,
									  ref float beta,
									  cudnnTensorDescriptor    yDesc,
									  CUdeviceptr y);
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
		/// <param name="xDesc">Tensor descriptor objects for the input and output tensors.</param>
		/// <param name="x">Input tensor data pointer in device memory.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the layer output
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
		/// for additional details.</param>
		/// <param name="yDesc">Tensor descriptor objects for the input and output tensors.</param>
		/// <param name="y">Output tensor data pointer in device memory.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnLRNCrossChannelForward(
									  cudnnHandle                    handle,
									  cudnnLRNDescriptor             normDesc,
									  cudnnLRNMode                   lrnMode,
									  ref double alpha,
									  cudnnTensorDescriptor    xDesc,
									  CUdeviceptr x,
									  ref double beta,
									  cudnnTensorDescriptor    yDesc,
									  CUdeviceptr y);

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
        /// <param name="yDesc">Tensor descriptor and pointer in device memory for the bottom layer's
        /// data. (Bottom layer is the earlier layer in the computation graph during
        /// inference).</param>
        /// <param name="y">Tensor descriptor and pointer in device memory for the bottom layer's
        /// data. (Bottom layer is the earlier layer in the computation graph during
        /// inference).</param>
        /// <param name="dyDesc">Tensor descriptor and pointer in device memory for the top layer's
        /// cumulative loss differential data (error backpropagation). (Top layer is the
        /// later layer in the computation graph during inference).</param>
        /// <param name="dy">Tensor descriptor and pointer in device memory for the top layer's
        /// cumulative loss differential data (error backpropagation). (Top layer is the
        /// later layer in the computation graph during inference).</param>
        /// <param name="xDesc">Tensor descriptor and pointer in device memory for the bottom layer's
        /// data. (Bottom layer is the earlier layer in the computation graph
        /// during inference). Note that these values are not modified during
        /// backpropagation.</param>
        /// <param name="x">Tensor descriptor and pointer in device memory for the bottom layer's
        /// data. (Bottom layer is the earlier layer in the computation graph
        /// during inference). Note that these values are not modified during
        /// backpropagation.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the layer output
        /// value with prior value in the destination tensor as follows: dstValue =
        /// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
        /// for additional details.</param>
        /// <param name="dxDesc">Tensor descriptor and pointer in device memory for the bottom layer's
        /// cumulative loss differential data (error backpropagation). (Bottom layer is
        /// the earlier layer in the computation graph during inference).</param>
        /// <param name="dx">Tensor descriptor and pointer in device memory for the bottom layer's
        /// cumulative loss differential data (error backpropagation). (Bottom layer is
        /// the earlier layer in the computation graph during inference).</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnLRNCrossChannelBackward(
									  cudnnHandle                    handle,
									  cudnnLRNDescriptor             normDesc,
									  cudnnLRNMode                   lrnMode,
									  ref float alpha,
									  cudnnTensorDescriptor    yDesc,
									  CUdeviceptr y,
									  cudnnTensorDescriptor    dyDesc,
									  CUdeviceptr dy,
									  cudnnTensorDescriptor    xDesc,
									  CUdeviceptr x,
									  ref float beta,
									  cudnnTensorDescriptor    dxDesc,
									  CUdeviceptr dx);
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
        /// <param name="yDesc">Tensor descriptor and pointer in device memory for the bottom layer's
        /// data. (Bottom layer is the earlier layer in the computation graph during
        /// inference).</param>
        /// <param name="y">Tensor descriptor and pointer in device memory for the bottom layer's
        /// data. (Bottom layer is the earlier layer in the computation graph during
        /// inference).</param>
        /// <param name="dyDesc">Tensor descriptor and pointer in device memory for the top layer's
        /// cumulative loss differential data (error backpropagation). (Top layer is the
        /// later layer in the computation graph during inference).</param>
        /// <param name="dy">Tensor descriptor and pointer in device memory for the top layer's
        /// cumulative loss differential data (error backpropagation). (Top layer is the
        /// later layer in the computation graph during inference).</param>
        /// <param name="xDesc">Tensor descriptor and pointer in device memory for the bottom layer's
        /// data. (Bottom layer is the earlier layer in the computation graph
        /// during inference). Note that these values are not modified during
        /// backpropagation.</param>
        /// <param name="x">Tensor descriptor and pointer in device memory for the bottom layer's
        /// data. (Bottom layer is the earlier layer in the computation graph
        /// during inference). Note that these values are not modified during
        /// backpropagation.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the layer output
        /// value with prior value in the destination tensor as follows: dstValue =
        /// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
        /// for additional details.</param>
        /// <param name="dxDesc">Tensor descriptor and pointer in device memory for the bottom layer's
        /// cumulative loss differential data (error backpropagation). (Bottom layer is
        /// the earlier layer in the computation graph during inference).</param>
        /// <param name="dx">Tensor descriptor and pointer in device memory for the bottom layer's
        /// cumulative loss differential data (error backpropagation). (Bottom layer is
        /// the earlier layer in the computation graph during inference).</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnLRNCrossChannelBackward(
									  cudnnHandle                    handle,
									  cudnnLRNDescriptor             normDesc,
									  cudnnLRNMode                   lrnMode,
									  ref double alpha,
									  cudnnTensorDescriptor    yDesc,
									  CUdeviceptr y,
									  cudnnTensorDescriptor    dyDesc,
									  CUdeviceptr dy,
									  cudnnTensorDescriptor    xDesc,
									  CUdeviceptr x,
									  ref double beta,
									  cudnnTensorDescriptor    dxDesc,
									  CUdeviceptr dx);

		

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
		/// <param name="xDesc">Tensor descriptor objects for the input and output tensors. Note that
		/// srcDesc is shared between srcData, srcMeansData, tempData, tempData2
		/// tensors.</param>
		/// <param name="x">Input tensor data pointer in device memory.</param>
		/// <param name="means">Input means tensor data pointer in device memory. Note that this tensor
		/// can be NULL (in that case it's values are assumed to be zero during the
		/// computation). This tensor also doesn't have to contain means, these can
		/// be any values, a frequently used variation is a result of convolution with a
		/// normalized positive kernel (such as Gaussian).</param>
		/// <param name="temp">Temporary tensors in device memory. These are used for computing
		/// intermediate values during the forward pass. These tensors do not have
		/// to be preserved as inputs from forward to the backward pass. Both use
		/// srcDesc as a descriptor.</param>
		/// <param name="temp2">Temporary tensors in device memory. These are used for computing
		/// intermediate values during the forward pass. These tensors do not have
		/// to be preserved as inputs from forward to the backward pass. Both use
		/// srcDesc as a descriptor.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the layer output
		/// value with prior value in the destination tensor as follows: dstValue =
		/// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
		/// for additional details.</param>
		/// <param name="yDesc">Tensor descriptor objects for the input and output tensors. Note that
		/// srcDesc is shared between srcData, srcMeansData, tempData, tempData2
		/// tensors.</param>
		/// <param name="y">Pointer in device memory to a tensor for the result of the forward DivisiveNormalization pass.</param>
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnDivisiveNormalizationForward(
									  cudnnHandle                    handle,
									  cudnnLRNDescriptor             normDesc,
									  cudnnDivNormMode               mode,
									  ref float alpha,
									  cudnnTensorDescriptor    xDesc, // same desc for means, temp, temp2
									  CUdeviceptr x,
									  CUdeviceptr means, // if NULL, means are assumed to be zero
									  CUdeviceptr temp,
									  CUdeviceptr temp2,
									  ref float beta,
									  cudnnTensorDescriptor    yDesc,
									  CUdeviceptr y
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
        /// <param name="xDesc">Tensor descriptor objects for the input and output tensors. Note that
        /// srcDesc is shared between srcData, srcMeansData, tempData, tempData2
        /// tensors.</param>
        /// <param name="x">Input tensor data pointer in device memory.</param>
        /// <param name="means">Input means tensor data pointer in device memory. Note that this tensor
        /// can be NULL (in that case it's values are assumed to be zero during the
        /// computation). This tensor also doesn't have to contain means, these can
        /// be any values, a frequently used variation is a result of convolution with a
        /// normalized positive kernel (such as Gaussian).</param>
        /// <param name="temp">Temporary tensors in device memory. These are used for computing
        /// intermediate values during the forward pass. These tensors do not have
        /// to be preserved as inputs from forward to the backward pass. Both use
        /// srcDesc as a descriptor.</param>
        /// <param name="temp2">Temporary tensors in device memory. These are used for computing
        /// intermediate values during the forward pass. These tensors do not have
        /// to be preserved as inputs from forward to the backward pass. Both use
        /// srcDesc as a descriptor.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the layer output
        /// value with prior value in the destination tensor as follows: dstValue =
        /// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
        /// for additional details.</param>
        /// <param name="yDesc">Tensor descriptor objects for the input and output tensors. Note that
        /// srcDesc is shared between srcData, srcMeansData, tempData, tempData2
        /// tensors.</param>
        /// <param name="y">Pointer in device memory to a tensor for the result of the forward DivisiveNormalization pass.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnDivisiveNormalizationForward(
									  cudnnHandle                    handle,
									  cudnnLRNDescriptor             normDesc,
									  cudnnDivNormMode               mode,
									  ref double alpha,
									  cudnnTensorDescriptor    xDesc, // same desc for means, temp, temp2
									  CUdeviceptr x,
									  CUdeviceptr means, // if NULL, means are assumed to be zero
									  CUdeviceptr temp,
									  CUdeviceptr temp2,
									  ref double beta,
									  cudnnTensorDescriptor    yDesc,
									  CUdeviceptr y
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
        /// <param name="xDesc">Tensor descriptor and pointers in device memory for the bottom layer's
        /// data and means. (Bottom layer is the earlier layer in the computation
        /// graph during inference). Note: the means tensor is expected to be
        /// precomputed by the user. It can also contain any valid values (not required
        /// to be actual means, and can be for instance a result of a convolution with
        /// a Gaussian kernel).</param>
        /// <param name="x">Tensor descriptor and pointers in device memory for the bottom layer's
        /// data and means. (Bottom layer is the earlier layer in the computation
        /// graph during inference). Note: the means tensor is expected to be
        /// precomputed by the user. It can also contain any valid values (not required
        /// to be actual means, and can be for instance a result of a convolution with
        /// a Gaussian kernel).</param>
        /// <param name="means">Tensor descriptor and pointers in device memory for the bottom layer's
        /// data and means. (Bottom layer is the earlier layer in the computation
        /// graph during inference). Note: the means tensor is expected to be
        /// precomputed by the user. It can also contain any valid values (not required
        /// to be actual means, and can be for instance a result of a convolution with
        /// a Gaussian kernel).</param>
        /// <param name="dy">Tensor pointer in device memory for the top layer's cumulative loss
        /// differential data (error backpropagation). (Top layer is the later layer in
        /// the computation graph during inference).</param>
        /// <param name="temp">Temporary tensors in device memory. These are used for computing
        /// intermediate values during the backward pass. These tensors do not have
        /// to be preserved from forward to backward pass. Both use srcDesc as a
        /// descriptor.</param>
        /// <param name="temp2">Temporary tensors in device memory. These are used for computing
        /// intermediate values during the backward pass. These tensors do not have
        /// to be preserved from forward to backward pass. Both use srcDesc as a
        /// descriptor.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the layer output
        /// value with prior value in the destination tensor as follows: dstValue =
        /// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
        /// for additional details.</param>
        /// <param name="dXdMeansDesc">Tensor descriptor for destDataDiff and destMeansDiff.</param>
        /// <param name="dx">Tensor pointers (in device memory) for the bottom layer's resulting
        /// differentials (data and means). Both share the same descriptor.</param>
        /// <param name="dMeans">Tensor pointers (in device memory) for the bottom layer's resulting
        /// differentials (data and means). Both share the same descriptor.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnDivisiveNormalizationBackward(
									  cudnnHandle                    handle,
									  cudnnLRNDescriptor             normDesc,
									  cudnnDivNormMode               mode,
									  ref float alpha,
									  cudnnTensorDescriptor    xDesc, // same desc for diff, means, temp, temp2
									  CUdeviceptr x,
									  CUdeviceptr means, // if NULL, means are assumed to be zero
									  CUdeviceptr dy,
									  CUdeviceptr temp,
									  CUdeviceptr temp2,
									  ref float beta,
									  cudnnTensorDescriptor    dXdMeansDesc, // same desc for dest, means, meansDiff
									  CUdeviceptr dx, // output data differential
									  CUdeviceptr dMeans // output means differential, can be NULL
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
        /// <param name="xDesc">Tensor descriptor and pointers in device memory for the bottom layer's
        /// data and means. (Bottom layer is the earlier layer in the computation
        /// graph during inference). Note: the means tensor is expected to be
        /// precomputed by the user. It can also contain any valid values (not required
        /// to be actual means, and can be for instance a result of a convolution with
        /// a Gaussian kernel).</param>
        /// <param name="x">Tensor descriptor and pointers in device memory for the bottom layer's
        /// data and means. (Bottom layer is the earlier layer in the computation
        /// graph during inference). Note: the means tensor is expected to be
        /// precomputed by the user. It can also contain any valid values (not required
        /// to be actual means, and can be for instance a result of a convolution with
        /// a Gaussian kernel).</param>
        /// <param name="means">Tensor descriptor and pointers in device memory for the bottom layer's
        /// data and means. (Bottom layer is the earlier layer in the computation
        /// graph during inference). Note: the means tensor is expected to be
        /// precomputed by the user. It can also contain any valid values (not required
        /// to be actual means, and can be for instance a result of a convolution with
        /// a Gaussian kernel).</param>
        /// <param name="dy">Tensor pointer in device memory for the top layer's cumulative loss
        /// differential data (error backpropagation). (Top layer is the later layer in
        /// the computation graph during inference).</param>
        /// <param name="temp">Temporary tensors in device memory. These are used for computing
        /// intermediate values during the backward pass. These tensors do not have
        /// to be preserved from forward to backward pass. Both use srcDesc as a
        /// descriptor.</param>
        /// <param name="temp2">Temporary tensors in device memory. These are used for computing
        /// intermediate values during the backward pass. These tensors do not have
        /// to be preserved from forward to backward pass. Both use srcDesc as a
        /// descriptor.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the layer output
        /// value with prior value in the destination tensor as follows: dstValue =
        /// alpha[0]*resultValue + beta[0]*priorDstValue. Please refer to this section
        /// for additional details.</param>
        /// <param name="dXdMeansDesc">Tensor descriptor for destDataDiff and destMeansDiff.</param>
        /// <param name="dx">Tensor pointers (in device memory) for the bottom layer's resulting
        /// differentials (data and means). Both share the same descriptor.</param>
        /// <param name="dMeans">Tensor pointers (in device memory) for the bottom layer's resulting
        /// differentials (data and means). Both share the same descriptor.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnDivisiveNormalizationBackward(
									  cudnnHandle                    handle,
									  cudnnLRNDescriptor             normDesc,
									  cudnnDivNormMode               mode,
									  ref double alpha,
									  cudnnTensorDescriptor    xDesc, // same desc for diff, means, temp, temp2
									  CUdeviceptr x,
									  CUdeviceptr means, // if NULL, means are assumed to be zero
									  CUdeviceptr dy,
									  CUdeviceptr temp,
									  CUdeviceptr temp2,
									  ref double beta,
									  cudnnTensorDescriptor    dXdMeansDesc, // same desc for dest, means, meansDiff
									  CUdeviceptr dx, // output data differential
									  CUdeviceptr dMeans // output means differential, can be NULL
									  );


        /// <summary>
        /// Derives a tensor descriptor from layer data descriptor for BatchNormalization 
		/// scale, invVariance, bnBias, bnScale tensors. Use this tensor desc for 
		/// bnScaleBiasMeanVarDesc and bnScaleBiasDiffDesc in Batch Normalization forward and backward functions.
        /// </summary>
        /// <param name="derivedBnDesc"></param>
        /// <param name="xDesc"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnDeriveBNTensorDescriptor(
                                        cudnnTensorDescriptor derivedBnDesc,
                                        cudnnTensorDescriptor xDesc,
                                        cudnnBatchNormMode mode );

		/// <summary>
		/// Computes y = BN(x). Also accumulates moving averages of mean and inverse variances
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="mode"></param>
		/// <param name="alpha"></param>
		/// <param name="beta"></param>
		/// <param name="xDesc"></param>
		/// <param name="x"></param>
		/// <param name="yDesc"></param>
		/// <param name="y"></param>
		/// <param name="bnScaleBiasMeanVarDesc"></param>
		/// <param name="bnScale"></param>
		/// <param name="bnBias"></param>
		/// <param name="exponentialAverageFactor"></param>
		/// <param name="resultRunningMean"></param>
		/// <param name="resultRunningVariance"></param>
		/// <param name="epsilon"></param>
		/// <param name="resultSaveMean"></param>
		/// <param name="resultSaveVariance"></param>
		/// <returns></returns>
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
                                CUdeviceptr resultRunningVariance,

                                /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
                                double epsilon,

                                /* Optionally save intermediate results from the forward pass here
                                   - can be reused to speed up backward pass. NULL if unused */
                                CUdeviceptr resultSaveMean,
                                CUdeviceptr resultSaveVariance );

		/// <summary>
		/// 
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="mode"></param>
		/// <param name="alpha"></param>
		/// <param name="beta"></param>
		/// <param name="xDesc"></param>
		/// <param name="x"></param>
		/// <param name="yDesc"></param>
		/// <param name="y"></param>
		/// <param name="bnScaleBiasMeanVarDesc"></param>
		/// <param name="bnScale"></param>
		/// <param name="bnBias"></param>
		/// <param name="exponentialAverageFactor"></param>
		/// <param name="resultRunningMean"></param>
		/// <param name="resultRunningVariance"></param>
		/// <param name="epsilon"></param>
		/// <param name="resultSaveMean"></param>
		/// <param name="resultSaveVariance"></param>
		/// <returns></returns>
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
                        CUdeviceptr resultRunningVariance,

                        /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
                        double epsilon,

                        /* Optionally save intermediate results from the forward pass here
                           - can be reused to speed up backward pass. NULL if unused */
                        CUdeviceptr resultSaveMean,
                        CUdeviceptr resultSaveVariance);


        /// <summary>
        /// Performs Batch Normalization during Inference: 
		/// y[i] = bnScale[k]*(x[i]-estimatedMean[k])*estimatedInvVariance[k] + bnBias[k]
		/// with bnScale, bnBias, runningMean, runningInvVariance tensors indexed
		/// according to spatial or per-activation mode. Refer to cudnnBatchNormalizationForwardTraining
		/// above for notes on function arguments.
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="mode"></param>
        /// <param name="alpha"></param>
        /// <param name="beta"></param>
        /// <param name="xDesc"></param>
        /// <param name="x"></param>
        /// <param name="yDesc"></param>
        /// <param name="y"></param>
        /// <param name="bnScaleBiasMeanVarDesc"></param>
        /// <param name="bnScale"></param>
        /// <param name="bnBias"></param>
        /// <param name="estimatedMean"></param>
        /// <param name="estimatedVariance"></param>
        /// <param name="epsilon"></param>
        /// <returns></returns>
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
                                        CUdeviceptr estimatedVariance,
                                        double epsilon);

		/// <summary>
		/// 
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="mode"></param>
		/// <param name="alpha"></param>
		/// <param name="beta"></param>
		/// <param name="xDesc"></param>
		/// <param name="x"></param>
		/// <param name="yDesc"></param>
		/// <param name="y"></param>
		/// <param name="bnScaleBiasMeanVarDesc"></param>
		/// <param name="bnScale"></param>
		/// <param name="bnBias"></param>
		/// <param name="estimatedMean"></param>
		/// <param name="estimatedVariance"></param>
		/// <param name="epsilon"></param>
		/// <returns></returns>
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
                                        CUdeviceptr estimatedVariance,
                                        double epsilon );

        /// <summary>
        /// Performs backward pass of Batch Normalization layer. Returns x gradient, bnScale gradient and bnBias gradient
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="mode"></param>
        /// <param name="alphaDataDiff"></param>
        /// <param name="betaDataDiff"></param>
        /// <param name="alphaParamDiff"></param>
        /// <param name="betaParamDiff"></param>
        /// <param name="xDesc"></param>
        /// <param name="x"></param>
        /// <param name="dyDesc"></param>
        /// <param name="dy"></param>
        /// <param name="dxDesc"></param>
        /// <param name="dx"></param>
        /// <param name="dBnScaleBiasDesc"></param>
        /// <param name="bnScale"></param>
        /// <param name="dBnScaleResult"></param>
        /// <param name="dBnBiasResult"></param>
        /// <param name="epsilon"></param>
        /// <param name="savedMean"></param>
        /// <param name="savedInvVariance"></param>
        /// <returns></returns>
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

		/// <summary>
		/// 
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="mode"></param>
		/// <param name="alphaDataDiff"></param>
		/// <param name="betaDataDiff"></param>
		/// <param name="alphaParamDiff"></param>
		/// <param name="betaParamDiff"></param>
		/// <param name="xDesc"></param>
		/// <param name="x"></param>
		/// <param name="dyDesc"></param>
		/// <param name="dy"></param>
		/// <param name="dxDesc"></param>
		/// <param name="dx"></param>
		/// <param name="dBnScaleBiasDesc"></param>
		/// <param name="bnScale"></param>
		/// <param name="dBnScaleResult"></param>
		/// <param name="dBnBiasResult"></param>
		/// <param name="epsilon"></param>
		/// <param name="savedMean"></param>
		/// <param name="savedInvVariance"></param>
		/// <returns></returns>
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

        /// <summary>
        /// This function creates a generic spatial transformer descriptor object by allocating the memory needed to hold its opaque structure. 
        /// </summary>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnCreateSpatialTransformerDescriptor(
                                       ref cudnnSpatialTransformerDescriptor stDesc);

        /// <summary>
        /// This function destroys a previously created spatial transformer descriptor object. 
        /// </summary>
        /// <param name="stDesc">Previously created spatial transformer descriptor object.</param>
        /// <param name="samplerType">Enumerant to specify the sampler type.</param>
        /// <param name="dataType">Data type.</param>
        /// <param name="nbDims">Dimension of the transformed tensor.</param>
        /// <param name="dimA">Array of dimension nbDims containing the size of the transformed tensor for every dimension.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSetSpatialTransformerNdDescriptor(
                                        cudnnSpatialTransformerDescriptor stDesc,
                                        cudnnSamplerType samplerType,
                                        cudnnDataType dataType,
                                        int nbDims,
                                        int []dimA);

        /// <summary>
        /// This function destroys a previously created spatial transformer descriptor object. 
        /// </summary>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnDestroySpatialTransformerDescriptor(
                                         cudnnSpatialTransformerDescriptor stDesc);

        /// <summary>
        /// This function generates a grid of coordinates in the input tensor corresponding to each pixel from the output tensor.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="stDesc">Previously created spatial transformer descriptor object.</param>
        /// <param name="theta">Affine transformation matrix. It should be of size n*2*3 for a 2d transformation, where n is the number of images specified in stDesc.</param>
        /// <param name="grid">A grid of coordinates. It is of size n*h*w*2 for a 2d transformation, where n, h, w is specified in stDesc. In the 4th dimension, the first coordinate is x, and the second coordinate is y.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSpatialTfGridGeneratorForward(
                                         cudnnHandle handle,
                                         cudnnSpatialTransformerDescriptor stDesc,
                                         CUdeviceptr theta,
                                         CUdeviceptr grid);

        /// <summary>
        /// This function computes the gradient of a grid generation operation.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="stDesc">Previously created spatial transformer descriptor object.</param>
        /// <param name="dgrid">Data pointer to GPU memory contains the input differential data.</param>
        /// <param name="dtheta">Data pointer to GPU memory contains the output differential data.</param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSpatialTfGridGeneratorBackward(
                                         cudnnHandle handle,
                                         cudnnSpatialTransformerDescriptor stDesc,
                                         CUdeviceptr dgrid,
                                         CUdeviceptr dtheta);

        /// <summary>
        /// This function performs a sampler operation and generates the output tensor using the grid given by the grid generator.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="stDesc">Previously created spatial transformer descriptor object.</param>
        /// <param name="alpha">Pointer to scaling factor (in host memory) used to blend the source value with prior value in the destination tensor as follows: dstValue = alpha[0]*srcValue + beta[0]*priorDstValue.</param>
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor xDesc.</param>
        /// <param name="grid">A grid of coordinates generated by cudnnSpatialTfGridGeneratorForward.</param>
        /// <param name="beta">Pointer to scaling factor (in host memory) used to blend the source value with prior value in the destination tensor as follows: dstValue = alpha[0]*srcValue + beta[0]*priorDstValue.</param>
        /// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor yDesc.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSpatialTfSamplerForward(
                                         cudnnHandle handle,
                                         cudnnSpatialTransformerDescriptor stDesc,
                                         ref float alpha,
                                         cudnnTensorDescriptor xDesc,
                                         CUdeviceptr x,
                                         CUdeviceptr grid,
                                         ref float beta,
                                         cudnnTensorDescriptor                    yDesc,
                                         CUdeviceptr y);

		/// <summary>
		/// 
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="stDesc"></param>
		/// <param name="alpha"></param>
		/// <param name="xDesc"></param>
		/// <param name="x"></param>
		/// <param name="grid"></param>
		/// <param name="beta"></param>
		/// <param name="yDesc"></param>
		/// <param name="y"></param>
		/// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSpatialTfSamplerForward(
                                         cudnnHandle handle,
                                         cudnnSpatialTransformerDescriptor stDesc,
                                         ref double alpha,
                                         cudnnTensorDescriptor xDesc,
                                         CUdeviceptr x,
                                         CUdeviceptr grid,
                                         ref double beta,
                                         cudnnTensorDescriptor yDesc,
                                         CUdeviceptr y);

        /// <summary>
        /// This function computes the gradient of a sampling operation.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="stDesc">Previously created spatial transformer descriptor object.</param>
        /// <param name="alpha">Pointer to scaling factor (in host memory) used to blend the source value with prior value in the destination tensor as follows: dstValue = alpha[0]*srcValue + beta[0]*priorDstValue.</param>
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor xDesc.</param>
        /// <param name="beta">Pointer to scaling factor (in host memory) used to blend the source value with prior value in the destination tensor as follows: dstValue = alpha[0]*srcValue + beta[0]*priorDstValue.</param>
        /// <param name="dxDesc">Handle to the previously initialized output differential tensor descriptor.</param>
        /// <param name="dx">Data pointer to GPU memory associated with the output tensor descriptor dxDesc.</param>
        /// <param name="alphaDgrid">Pointer to scaling factor (in host memory) used to blend the gradient outputs dgrid with prior value in the destination pointer as follows: dstValue = alpha[0]*srcValue + beta[0]*priorDstValue.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the tensor descriptor dyDesc.</param>
        /// <param name="grid">A grid of coordinates generated by cudnnSpatialTfGridGeneratorForward.</param>
        /// <param name="betaDgrid">Pointer to scaling factor (in host memory) used to blend the gradient outputs dgrid with prior value in the destination pointer as follows: dstValue = alpha[0]*srcValue + beta[0]*priorDstValue.</param>
        /// <param name="dgrid">Data pointer to GPU memory contains the output differential data.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSpatialTfSamplerBackward(
                                         cudnnHandle handle,
                                         cudnnSpatialTransformerDescriptor stDesc,
                                         ref float alpha,
                                         cudnnTensorDescriptor xDesc,
                                         CUdeviceptr x,
                                         ref float beta,
                                         cudnnTensorDescriptor dxDesc,
                                         CUdeviceptr dx,
                                         CUdeviceptr alphaDgrid,
                                         cudnnTensorDescriptor dyDesc,
                                         CUdeviceptr dy,
                                         CUdeviceptr grid,
                                         CUdeviceptr betaDgrid,
                                         CUdeviceptr dgrid);

		/// <summary>
		/// 
		/// </summary>
		/// <param name="handle"></param>
		/// <param name="stDesc"></param>
		/// <param name="alpha"></param>
		/// <param name="xDesc"></param>
		/// <param name="x"></param>
		/// <param name="beta"></param>
		/// <param name="dxDesc"></param>
		/// <param name="dx"></param>
		/// <param name="alphaDgrid"></param>
		/// <param name="dyDesc"></param>
		/// <param name="dy"></param>
		/// <param name="grid"></param>
		/// <param name="betaDgrid"></param>
		/// <param name="dgrid"></param>
		/// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSpatialTfSamplerBackward(
                                         cudnnHandle handle,
                                         cudnnSpatialTransformerDescriptor stDesc,
                                         ref double alpha,
                                         cudnnTensorDescriptor xDesc,
                                         CUdeviceptr x,
                                         ref double beta,
                                         cudnnTensorDescriptor dxDesc,
                                         CUdeviceptr dx,
                                         CUdeviceptr alphaDgrid,
                                         cudnnTensorDescriptor dyDesc,
                                         CUdeviceptr dy,
                                         CUdeviceptr grid,
                                         CUdeviceptr betaDgrid,
                                         CUdeviceptr dgrid);

        /// <summary>
        /// This function creates a generic dropout descriptor object by allocating the memory needed to hold its opaque structure. 
        /// </summary>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnCreateDropoutDescriptor(ref cudnnDropoutDescriptor dropoutDesc);

        /// <summary>
        /// This function destroys a previously created dropout descriptor object. 
        /// </summary>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor dropoutDesc);

        /*helper function to determine size of the states to be passed to cudnnSetDropoutDescriptor */
        /// <summary>
        /// This function is used to query the amount of space required to store the states of the random number generators used by cudnnDropoutForward function.
        /// </summary>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnDropoutGetStatesSize(cudnnHandle handle, ref SizeT sizeInBytes);

        /*helper function to determine size of the reserve space to be passed to dropout forward/backward calls */
        /// <summary>
        /// This function is used to query the amount of reserve needed to run dropout with the input dimensions given by xDesc. 
        /// The same reserve space is expected to be passed to cudnnDropoutForward and cudnnDropoutBackward, and its contents is 
        /// expected to remain unchanged between cudnnDropoutForward and cudnnDropoutBackward calls. 
        /// </summary>
        /// <param name="xdesc">Handle to a previously initialized tensor descriptor, describing input to a dropout operation.</param>
        /// <param name="sizeInBytes">Amount of GPU memory needed as reserve space to be able to run dropout with an input tensor descriptor specified by xDesc.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor xdesc, ref SizeT sizeInBytes);

        /// <summary>
        /// This function initializes a previously created dropout descriptor object. If states argument is equal to 
        /// NULL, random number generator states won't be initialized, and only dropout value will be set. No other 
        /// function should be writing to the memory
        /// </summary>
        /// <param name="dropoutDesc">Previously created dropout descriptor object.</param>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="dropout">The probability with which the value from input would be propagated through the dropout layer.</param>
        /// <param name="states">Pointer to user-allocated GPU memory that will hold random number generator states.</param>
        /// <param name="stateSizeInBytes">Specifies size in bytes of the provided memory for the states.</param>
        /// <param name="seed">Seed used to initialize random number generator states.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSetDropoutDescriptor(cudnnDropoutDescriptor dropoutDesc,
                                                            cudnnHandle handle,
                                                            float dropout,
                                                            CUdeviceptr states,
                                                            SizeT stateSizeInBytes,
                                                            ulong seed);


        /// <summary>
        /// Restores the dropout descriptor to a previously saved-off state
        /// </summary>
        /// <param name="dropoutDesc"></param>
        /// <param name="handle"></param>
        /// <param name="dropout"></param>
        /// <param name="states"></param>
        /// <param name="stateSizeInBytes"></param>
        /// <param name="seed"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnRestoreDropoutDescriptor(cudnnDropoutDescriptor dropoutDesc, cudnnHandle handle,
                                                                float dropout, CUdeviceptr states, SizeT stateSizeInBytes, ulong seed);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dropoutDesc"></param>
        /// <param name="handle"></param>
        /// <param name="dropout"></param>
        /// <param name="states"></param>
        /// <param name="seed"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetDropoutDescriptor(cudnnDropoutDescriptor dropoutDesc, cudnnHandle handle, ref float dropout, ref CUdeviceptr states, ref ulong seed);


        /// <summary>
        /// This function performs forward dropout operation over x returning results in y. If dropout was 
        /// used as a parameter to cudnnSetDropoutDescriptor, the approximately dropout fraction of x values 
        /// will be replaces by 0, and the rest will be scaled by 1/(1-dropout) This function should not be 
        /// running concurrently with another cudnnDropoutForward function using the same states.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="dropoutDesc">Previously created dropout descriptor object.</param>
        /// <param name="xdesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="x">Pointer to data of the tensor described by the xDesc descriptor.</param>
        /// <param name="ydesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="y">Pointer to data of the tensor described by the yDesc descriptor.</param>
        /// <param name="reserveSpace">Pointer to user-allocated GPU memory used by this function. It is expected that contents of reserveSpace doe not change between cudnnDropoutForward and cudnnDropoutBackward calls.</param>
        /// <param name="reserveSpaceSizeInBytes">Specifies size in bytes of the provided memory for the reserve space.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnDropoutForward(cudnnHandle handle,
                                                      cudnnDropoutDescriptor dropoutDesc,
                                                      cudnnTensorDescriptor xdesc,
                                                      CUdeviceptr x,
                                                      cudnnTensorDescriptor ydesc,
                                                      CUdeviceptr y,
                                                      CUdeviceptr reserveSpace,
                                                      SizeT reserveSpaceSizeInBytes);

        /// <summary>
        /// This function performs backward dropout operation over dy returning results in dx. If during 
        /// forward dropout operation value from x was propagated to y then during backward operation value 
        /// from dy will be propagated to dx, otherwise, dx value will be set to 0.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="dropoutDesc">Previously created dropout descriptor object.</param>
        /// <param name="dydesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="dy">Pointer to data of the tensor described by the dyDesc descriptor.</param>
        /// <param name="dxdesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="dx">Pointer to data of the tensor described by the dxDesc descriptor.</param>
        /// <param name="reserveSpace">Pointer to user-allocated GPU memory used by this function. It is expected that reserveSpace was populated during a call to cudnnDropoutForward and has not been changed.</param>
        /// <param name="reserveSpaceSizeInBytes">Specifies size in bytes of the provided memory for the reserve space.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnDropoutBackward(cudnnHandle handle,
                                               cudnnDropoutDescriptor dropoutDesc,
                                               cudnnTensorDescriptor dydesc,
                                               CUdeviceptr dy,
                                               cudnnTensorDescriptor dxdesc,
                                               CUdeviceptr dx,
                                               CUdeviceptr reserveSpace,
                                               SizeT reserveSpaceSizeInBytes);

        /// <summary>
        /// This function creates a generic RNN descriptor object by allocating the memory 
        /// needed to hold its opaque structure.
        /// </summary>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnCreateRNNDescriptor(ref cudnnRNNDescriptor rnnDesc);

        /// <summary>
        /// This function destroys a previously created RNN descriptor object.
        /// </summary>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnDestroyRNNDescriptor(cudnnRNNDescriptor rnnDesc);



        /// <summary>
        /// 
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="rnnDesc"></param>
        /// <param name="count"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetRNNForwardInferenceAlgorithmMaxCount(
                                cudnnHandle handle,
                                cudnnRNNDescriptor rnnDesc,
                                ref int count);

        /// <summary>
        /// This function attempts all available cuDNN algorithms for cudnnRNNForwardInference, using user-allocated GPU memory, and outputs performance metrics to a user-allocated array of cudnnAlgorithmPerformance_t. These metrics are written in sorted fashion where the first element has the lowest compute time. 
        /// </summary>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnFindRNNForwardInferenceAlgorithmEx(cudnnHandle handle,
                                                    cudnnRNNDescriptor rnnDesc,
                                                    int seqLength,
                                                    cudnnTensorDescriptor[] xDesc,
                                                    CUdeviceptr x,
                                                    cudnnTensorDescriptor hxDesc,
                                                    CUdeviceptr hx,
                                                    cudnnTensorDescriptor cxDesc,
                                                    CUdeviceptr cx,
                                                    cudnnFilterDescriptor wDesc,
                                                    CUdeviceptr w,
                                                    cudnnTensorDescriptor[] yDesc,
                                                    CUdeviceptr y,
                                                    cudnnTensorDescriptor hyDesc,
                                                    CUdeviceptr hy,
                                                    cudnnTensorDescriptor cyDesc,
                                                    CUdeviceptr cy,
                                                    float findIntensity,
                                                    int requestedAlgoCount,
                                                    ref int returnedAlgoCount,
                                                    cudnnAlgorithmPerformance[] perfResults,
                                                    CUdeviceptr workspace,
                                                    SizeT workSpaceSizeInBytes);

        /// <summary>
        /// 
        /// </summary>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetRNNForwardTrainingAlgorithmMaxCount(
                                cudnnHandle handle,
                                cudnnRNNDescriptor rnnDesc,
                                ref int count);

        /// <summary>
        /// This function attempts all available cuDNN algorithms for cudnnRNNForwardTraining, using user-allocated GPU memory, and outputs performance metrics to a user-allocated array of cudnnAlgorithmPerformance_t. These metrics are written in sorted fashion where the first element has the lowest compute time. 
        /// </summary>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnFindRNNForwardTrainingAlgorithmEx(cudnnHandle handle,
                                                    cudnnRNNDescriptor rnnDesc,
                                                    int seqLength,
                                                    cudnnTensorDescriptor[] xDesc,
                                                    CUdeviceptr x,
                                                    cudnnTensorDescriptor hxDesc,
                                                    CUdeviceptr hx,
                                                    cudnnTensorDescriptor cxDesc,
                                                    CUdeviceptr cx,
                                                    cudnnFilterDescriptor wDesc,
                                                    CUdeviceptr w,
                                                    cudnnTensorDescriptor[] yDesc,
                                                    CUdeviceptr y,
                                                    cudnnTensorDescriptor hyDesc,
                                                    CUdeviceptr hy,
                                                    cudnnTensorDescriptor cyDesc,
                                                    CUdeviceptr cy,
                                                    float findIntensity,
                                                    int requestedAlgoCount,
                                                    ref int returnedAlgoCount,
                                                    cudnnAlgorithmPerformance[] perfResults,
                                                    CUdeviceptr workspace,
                                                    SizeT workSpaceSizeInBytes,
                                                    CUdeviceptr reserveSpace,
                                                    SizeT reserveSpaceSizeInBytes);

        /// <summary>
        /// 
        /// </summary>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetRNNBackwardDataAlgorithmMaxCount(
                                cudnnHandle handle,
                                cudnnRNNDescriptor rnnDesc,
                                ref int count);

        /// <summary>
        /// 
        /// </summary>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnFindRNNBackwardDataAlgorithmEx(cudnnHandle handle,
                                                cudnnRNNDescriptor rnnDesc,
                                                int seqLength,
                                                cudnnTensorDescriptor[] yDesc,
                                                CUdeviceptr y,
                                                cudnnTensorDescriptor[] dyDesc,
                                                CUdeviceptr dy,
                                                cudnnTensorDescriptor dhyDesc,
                                                CUdeviceptr dhy,
                                                cudnnTensorDescriptor dcyDesc,
                                                CUdeviceptr dcy,
                                                cudnnFilterDescriptor wDesc,
                                                CUdeviceptr w,
                                                cudnnTensorDescriptor hxDesc,
                                                CUdeviceptr hx,
                                                cudnnTensorDescriptor cxDesc,
                                                CUdeviceptr cx,
                                                cudnnTensorDescriptor[] dxDesc,
                                                CUdeviceptr dx,
                                                cudnnTensorDescriptor dhxDesc,
                                                CUdeviceptr dhx,
                                                cudnnTensorDescriptor dcxDesc,
                                                CUdeviceptr dcx,
                                                float findIntensity,
                                                int requestedAlgoCount,
                                                ref int returnedAlgoCount,
                                                cudnnAlgorithmPerformance[] perfResults,
                                                CUdeviceptr workspace,
                                                SizeT workSpaceSizeInBytes,
                                                CUdeviceptr reserveSpace,
                                                SizeT reserveSpaceSizeInBytes );

        /// <summary>
        /// 
        /// </summary>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetRNNBackwardWeightsAlgorithmMaxCount(
                                cudnnHandle handle,
                                cudnnRNNDescriptor rnnDesc,
                                ref int count);

        /// <summary>
        /// 
        /// </summary>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnFindRNNBackwardWeightsAlgorithmEx(cudnnHandle handle,
                                                   cudnnRNNDescriptor rnnDesc,
                                                   int seqLength,
                                                   cudnnTensorDescriptor[] xDesc,
                                                   CUdeviceptr x,
                                                   cudnnTensorDescriptor hxDesc,
                                                   CUdeviceptr hx,
                                                   cudnnTensorDescriptor[] yDesc,
                                                   CUdeviceptr y,
                                                   float findIntensity,
                                                   int requestedAlgoCount,
                                                   ref int returnedAlgoCount,
                                                   cudnnAlgorithmPerformance[] perfResults,
                                                   CUdeviceptr workspace,
                                                   SizeT workSpaceSizeInBytes, 
                                                   cudnnFilterDescriptor dwDesc,
                                                   CUdeviceptr dw,
                                                   CUdeviceptr reserveSpace,
                                                   SizeT reserveSpaceSizeInBytes );






        // Expensive. Creates the plan for the specific settings.
        /// <summary>
        /// This function creates a plan to execute persistent RNNs when using the
        /// CUDNN_RNN_ALGO_PERSIST_DYNAMIC algo.This plan is tailored to the current GPU
        /// and problem hyperparemeters. This function call is expected to be expensive in terms of
        /// runtime, and should be used infrequently.
        /// </summary>
        /// <param name="rnnDesc"></param>
        /// <param name="minibatch"></param>
        /// <param name="dataType"></param>
        /// <param name="plan"></param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnCreatePersistentRNNPlan(cudnnRNNDescriptor rnnDesc,
                                             int minibatch,
                                             cudnnDataType dataType,
                                             ref cudnnPersistentRNNPlan plan);

        /// <summary>
        /// This function sets the persistent RNN plan to be executed when using rnnDesc and
        /// CUDNN_RNN_ALGO_PERSIST_DYNAMIC algo.
        /// </summary>
        /// <param name="rnnDesc"></param>
        /// <param name="plan"></param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSetPersistentRNNPlan(cudnnRNNDescriptor rnnDesc,
                                          cudnnPersistentRNNPlan plan);

        /// <summary>
        /// This function destroys a previously created persistent RNN plan object.
        /// </summary>
        /// <param name="plan"></param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan plan);




        /// <summary>
        /// This function initializes a previously created RNN descriptor object.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN library descriptor.</param>
        /// <param name="rnnDesc">A previously created RNN descriptor.</param>
        /// <param name="hiddenSize">Size of the internal hidden state for each layer.</param>
        /// <param name="numLayers">Number of stacked layers.</param>
        /// <param name="dropoutDesc">Handle to a previously created and initialized dropout descriptor.
        /// Dropout will be applied between layers(eg.a single layer network will have no dropout applied).</param>
        /// <param name="inputMode">Specifies the behavior at the input to the first layer</param>
        /// <param name="direction">Specifies the recurrence pattern. (eg. bidirectional)</param>
        /// <param name="mode">Specifies the type of RNN to compute.</param>
        /// <param name="algo">Specifies which RNN algorithm should be used to compute the results.</param>
        /// <param name="dataType">Compute precision.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSetRNNDescriptor(cudnnHandle handle,
                                                        cudnnRNNDescriptor rnnDesc,
                                                int hiddenSize,
                                                int numLayers,
                                                cudnnDropoutDescriptor dropoutDesc, // Between layers, not between recurrent steps.
                                                cudnnRNNInputMode inputMode,
                                                cudnnDirectionMode direction, 
                                                cudnnRNNMode mode,
                                                cudnnRNNAlgo algo, 
                                                cudnnDataType dataType);

        /// <summary>
        /// The cudnnSetRNNProjectionLayers() function should be called after cudnnSetRNNDescriptor() to enable the "recurrent" and/or "output" projection in a recursive neural network
        /// </summary>
        /// <param name="handle"> Handle to a previously created cuDNN library descriptor</param>
        /// <param name="rnnDesc"> A previously created and initialized RNN descriptor. </param>
        /// <param name="recProjSize">The size of the LSTM cell output after the “recurrent” projection. This value should not be larger than hiddenSize programmed via cudnnSetRNNDescriptor().</param>
        /// <param name="outProjSize"> This parameter should be zero. </param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSetRNNProjectionLayers(cudnnHandle handle,
                                                cudnnRNNDescriptor rnnDesc,
                                                int recProjSize,
                                                int outProjSize);
        /// <summary>
        /// This function retrieves the current RNN “projection” parameters. By default the projection feature is disabled so invoking this function immediately after cudnnSetRNNDescriptor() will yield recProjSize equal to hiddenSize and outProjSize set to zero. The cudnnSetRNNProjectionLayers() method enables the RNN projection. 
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="rnnDesc"></param>
        /// <param name="recProjSize"></param>
        /// <param name="outProjSize"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetRNNProjectionLayers(cudnnHandle handle,
                                                cudnnRNNDescriptor rnnDesc,
                                                ref int recProjSize,
                                                ref int outProjSize);
        /// <summary>
        /// 
        /// </summary>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSetRNNAlgorithmDescriptor(cudnnHandle handle,
                                                        cudnnRNNDescriptor rnnDesc,
                                                        cudnnAlgorithmDescriptor algoDesc);





        /// <summary>
        /// 
        /// </summary>
        /// <param name="cudnnHandle"></param>
        /// <param name="rnnDesc"></param>
        /// <param name="hiddenSize"></param>
        /// <param name="numLayers"></param>
        /// <param name="dropoutDesc"></param>
        /// <param name="inputMode"></param>
        /// <param name="direction"></param>
        /// <param name="mode"></param>
        /// <param name="algo"></param>
        /// <param name="dataType"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetRNNDescriptor(cudnnHandle cudnnHandle,
                                                        cudnnRNNDescriptor rnnDesc,
                                                        ref int hiddenSize,
                                                        ref int numLayers,
                                                        ref cudnnDropoutDescriptor dropoutDesc,
                                                        ref cudnnRNNInputMode inputMode,
                                                        ref cudnnDirectionMode direction,
                                                        ref cudnnRNNMode mode,
                                                        ref cudnnRNNAlgo algo,
                                                        ref cudnnDataType dataType);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="desc"></param>
        /// <param name="math"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSetRNNMatrixMathType(cudnnRNNDescriptor desc, cudnnMathType math);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="rnnDesc"></param>
        /// <param name="mType"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetRNNMatrixMathType(cudnnRNNDescriptor rnnDesc, ref cudnnMathType mType);


        ///// <summary>
        ///// This function initializes a previously created RNN descriptor object.
        ///// </summary>
        ///// <param name="rnnDesc">A previously created RNN descriptor.</param>
        ///// <param name="hiddenSize">Size of the internal hidden state for each layer.</param>
        ///// <param name="seqLength">Number of iterations to unroll over.</param>
        ///// <param name="numLayers">Number of layers.</param>
        ///// <param name="dropoutDesc">Handle to a previously created and initialized dropout descriptor.</param>
        ///// <param name="inputMode">Specifies the behavior at the input to the first layer.</param>
        ///// <param name="direction">Specifies the recurrence pattern. (eg. bidirectional)</param>
        ///// <param name="mode">The type of RNN to compute.</param>
        ///// <param name="dataType">Math precision.</param>
        //[DllImport(CUDNN_API_DLL_NAME)]
        //public static extern cudnnStatus cudnnSetRNNDescriptor(cudnnRNNDescriptor rnnDesc,
        //                                                int hiddenSize,
        //                                                int seqLength,
        //                                                int numLayers,
        //                                                cudnnDropoutDescriptor dropoutDesc, // Between layers, not between recurrent steps.
        //                                                cudnnRNNInputMode inputMode,
        //                                                cudnnDirectionMode direction,
        //                                                cudnnRNNMode mode,
        //                                                cudnnDataType dataType);


        // dataType in the RNN descriptor is used to determine math precision
        // dataType in weight descriptors and input descriptors is used to describe storage
        /// <summary>
        /// This function is used to query the amount of work space required to execute the RNN 
        /// described by rnnDesc with inputs dimensions defined by xDesc. 
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN library descriptor.</param>
        /// <param name="rnnDesc">A previously initialized RNN descriptor.</param>
        /// <param name="seqLength">Number of iterations to unroll over.</param>
        /// <param name="xDesc">An array of tensor descriptors describing the input to each recurrent iteration.</param>
        /// <param name="sizeInBytes">Minimum amount of GPU memory needed as workspace to be able to execute an RNN with the specified descriptor and input tensors.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetRNNWorkspaceSize(cudnnHandle handle,
                                                    cudnnRNNDescriptor rnnDesc,
                                                    int seqLength,
                                                    cudnnTensorDescriptor[] xDesc,
                                                    ref SizeT                     sizeInBytes
                                                    );

        /// <summary>
        /// This function is used to query the amount of reserved space required for training the 
        /// RNN described by rnnDesc with inputs dimensions defined by xDesc. The same reserve 
        /// space must be passed to cudnnRNNForwardTraining, cudnnRNNBackwardData and cudnnRNNBackwardWeights.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN library descriptor.</param>
        /// <param name="rnnDesc">A previously initialized RNN descriptor.</param>
        /// <param name="seqLength">Number of iterations to unroll over.</param>
        /// <param name="xDesc">An array of tensor descriptors describing the input to each recurrent iteration.</param>
        /// <param name="sizeInBytes">Minimum amount of GPU memory needed as reserve space to be able to train an RNN with the specified descriptor and input tensors.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetRNNTrainingReserveSize(cudnnHandle handle,
                                                          cudnnRNNDescriptor rnnDesc,
                                                          int seqLength,
                                                          cudnnTensorDescriptor[] xDesc,
                                                          ref SizeT                     sizeInBytes
                                                    );

        /// <summary>
        /// This function is used to query the amount of parameter space required to execute the RNN described by 
        /// rnnDesc with inputs dimensions defined by xDesc. 
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN library descriptor.</param>
        /// <param name="rnnDesc">A previously initialized RNN descriptor.</param>
        /// <param name="xDesc">A fully packed tensor descriptor describing the input to one recurrent iteration.</param>
        /// <param name="sizeInBytes">Minimum amount of GPU memory needed as parameter space to be able to execute an RNN with the specified descriptor and input tensors.</param>
        /// <param name="dataType">The data type of the parameters.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetRNNParamsSize(cudnnHandle handle,
                                                 cudnnRNNDescriptor rnnDesc,
                                                 cudnnTensorDescriptor xDesc,
                                                 ref SizeT                     sizeInBytes,
                                                 cudnnDataType dataType);

        /// <summary>
        /// This function is used to obtain a pointer and descriptor for the matrix parameters in layer within 
        /// the RNN described by rnnDesc with inputs dimensions defined by xDesc. 
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN library descriptor.</param>
        /// <param name="rnnDesc">A previously initialized RNN descriptor.</param>
        /// <param name="layer">The layer to query.</param>
        /// <param name="xDesc">A fully packed tensor descriptor describing the input to one recurrent iteration.</param>
        /// <param name="wDesc">Handle to a previously initialized filter descriptor describing the weights for the RNN.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor wDesc.</param>
        /// <param name="linLayerID">
        /// The linear layer to obtain information about: 
        /// * If mode in rnnDesc was set to CUDNN_RNN_RELU or CUDNN_RNN_TANH a value of 0 references the matrix multiplication 
        /// applied to the input from the previous layer, a value of 1 references the matrix multiplication applied to the recurrent input.
        /// * If mode in rnnDesc was set to CUDNN_LSTM values of 0-3 reference matrix multiplications applied to the input from the 
        /// previous layer, value of 4-7 reference matrix multiplications applied to the recurrent input.
        ///     ‣ Values 0 and 4 reference the input gate. 
        ///     ‣ Values 1 and 5 reference the forget gate. 
        ///     ‣ Values 2 and 6 reference the new memory gate. 
        ///     ‣ Values 3 and 7 reference the output gate.
        /// * If mode in rnnDesc was set to CUDNN_GRU values of 0-2 reference matrix multiplications applied to the input 
        /// from the previous layer, value of 3-5 reference matrix multiplications applied to the recurrent input. 
        ///     ‣ Values 0 and 3 reference the reset gate. 
        ///     ‣ Values 1 and 4 reference the update gate. 
        ///     ‣ Values 2 and 5 reference the new memory gate.
        /// </param>
        /// <param name="linLayerMatDesc">Handle to a previously created filter descriptor.</param>
        /// <param name="linLayerMat">Data pointer to GPU memory associated with the filter descriptor linLayerMatDesc.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetRNNLinLayerMatrixParams(cudnnHandle handle,
                             cudnnRNNDescriptor rnnDesc,
                             int layer,
                             cudnnTensorDescriptor xDesc,
                             cudnnFilterDescriptor wDesc,
                             CUdeviceptr w,
                             int linLayerID,
                             cudnnFilterDescriptor linLayerMatDesc, 
                             CUdeviceptr linLayerMat // void **
                             );

        /// <summary>
        /// This function is used to obtain a pointer and descriptor for the bias parameters 
        /// in layer within the RNN described by rnnDesc with inputs dimensions defined by xDesc. 
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN library descriptor.</param>
        /// <param name="rnnDesc">A previously initialized RNN descriptor.</param>
        /// <param name="layer">The layer to query.</param>
        /// <param name="xDesc">A fully packed tensor descriptor describing the input to one recurrent iteration.</param>
        /// <param name="wDesc">Handle to a previously initialized filter descriptor describing the weights for the RNN.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor wDesc.</param>
        /// <param name="linLayerID">
        /// The linear layer to obtain information about: 
        /// * If mode in rnnDesc was set to CUDNN_RNN_RELU or CUDNN_RNN_TANH a value of 0 references 
        /// the bias applied to the input from the previous layer, a value of 1 references the bias 
        /// applied to the recurrent input.
        /// * If mode in rnnDesc was set to CUDNN_LSTM values of 0, 1, 2 and 3 reference bias applied to the input 
        /// from the previous layer, value of 4, 5, 6 and 7 reference bias applied to the recurrent input.
        ///     ‣ Values 0 and 4 reference the input gate. 
        ///     ‣ Values 1 and 5 reference the forget gate. 
        ///     ‣ Values 2 and 6 reference the new memory gate. 
        ///     ‣ Values 3 and 7 reference the output gate.
        /// * If mode in rnnDesc was set to CUDNN_GRU values of 0, 1 and 2 reference bias applied to the 
        /// input from the previous layer, value of 3, 4 and 5 reference bias applied to the recurrent input.
        ///     ‣ Values 0 and 3 reference the reset gate. 
        ///     ‣ Values 1 and 4 reference the update gate. 
        ///     ‣ Values 2 and 5 reference the new memory gate.</param>
        /// <param name="linLayerBiasDesc">Handle to a previously created filter descriptor.</param>
        /// <param name="linLayerBias">Data pointer to GPU memory associated with the filter descriptor linLayerMatDesc.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetRNNLinLayerBiasParams(cudnnHandle handle,
                             cudnnRNNDescriptor rnnDesc,
                             int layer,
                             cudnnTensorDescriptor xDesc,
                             cudnnFilterDescriptor wDesc,
                             CUdeviceptr w,
                             int linLayerID,
                             cudnnFilterDescriptor linLayerBiasDesc,
                             CUdeviceptr linLayerBias // void **
                             );

        /// <summary>
        /// This routine executes the recurrent neural network described by rnnDesc with inputs x, hx, cx, weights w and 
        /// outputs y, hy, cy. workspace is required for intermediate storage. This function does not store data required 
        /// for training; cudnnRNNForwardTraining should be used for that purpose. 
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="rnnDesc">A previously initialized RNN descriptor.</param>
        /// <param name="xDesc">An array of tensor descriptors describing the input to each recurrent iteration. 
        /// Each tensor descriptor must have the same first dimension. The second dimension of the tensors may 
        /// decrease from element n to element n+1 but may not increase. The tensor must be fully packed.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptors in the array xDesc. 
        /// The data are expected to be packed contiguously with the first element of iteration n+1 following 
        /// directly from the last element of iteration n.</param>
        /// <param name="hxDesc">Handle to a previously initialized tensor descriptor describing the initial hidden 
        /// state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second 
        /// dimension of the first tensor described in xDesc. The third dimension must match the numLayers 
        /// argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be 
        /// fully packed.</param>
        /// <param name="hx">Data pointer to GPU memory associated with the tensor descriptor hxDesc. If a NULL pointer 
        /// is passed, the initial hidden state of the network will be initialized to zero.</param>
        /// <param name="cxDesc">A fully packed tensor descriptor describing the initial cell state for
        /// LSTM networks.The first dimension of the tensor depends on the
        /// direction argument passed to the cudnnSetRNNDescriptor call
        /// used to initialize rnnDesc:
        /// ‣ If direction is CUDNN_UNIDIRECTIONAL the first
        ///   dimension should match the numLayers argument passed to
        ///   cudnnSetRNNDescriptor.
        /// ‣ If direction is CUDNN_BIDIRECTIONAL the first dimension
        ///   should match double the numLayers argument passed to
        ///   cudnnSetRNNDescriptor.
        /// The second dimension must match the first dimension of the
        /// tensors described in xDesc.The third dimension must match the
        /// hiddenSize argument passed to the cudnnSetRNNDescriptor call
        /// used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="cx">Data pointer to GPU memory associated with the tensor descriptor
        /// cxDesc.If a NULL pointer is passed, the initial cell state of the network will be initialized to zero.</param>
        /// <param name="wDesc">Handle to a previously initialized filter descriptor describing the weights for the RNN.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor wDesc.</param>
        /// <param name="yDesc">A fully packed tensor descriptor describing the final cell state for
        /// LSTM networks.The first dimension of the tensor depends on the
        /// direction argument passed to the cudnnSetRNNDescriptor call
        /// used to initialize rnnDesc:
        /// ‣ If direction is CUDNN_UNIDIRECTIONAL the first
        ///   dimension should match the numLayers argument passed to
        ///   cudnnSetRNNDescriptor.
        /// ‣ If direction is CUDNN_BIDIRECTIONAL the first dimension
        ///   should match double the numLayers argument passed to
        ///   cudnnSetRNNDescriptor.
        /// The second dimension must match the first dimension of the
        /// tensors described in xDesc.The third dimension must match the
        /// hiddenSize argument passed to the cudnnSetRNNDescriptor call
        /// used to initialize rnnDesc.The tensor must be fully packed.</param>
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor yDesc. The data 
        /// are expected to be packed contiguously with the first element of iteration n+1 following directly 
        /// from the last element of iteration n.</param>
        /// <param name="hyDesc">Handle to a previously initialized tensor descriptor describing the final hidden 
        /// state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second 
        /// dimension of the first tensor described in xDesc. The third dimension must match the numLayers 
        /// argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be 
        /// fully packed.</param>
        /// <param name="hy">Data pointer to GPU memory associated with the tensor descriptor hyDesc. If a NULL 
        /// pointer is passed, the final hidden state of the network will not be saved.</param>
        /// <param name="cyDesc">Handle to a previously initialized tensor descriptor describing the final cell 
        /// state for LSTM networks. The first dimension of the tensor must match the hiddenSize argument passed 
        /// to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second 
        /// dimension of the first tensor described in xDesc. The third dimension must match the numLayers argument 
        /// passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="cy">Data pointer to GPU memory associated with the tensor descriptor cyDesc. If 
        /// a NULL pointer is passed, the final cell state of the network will be not be saved.</param>
        /// <param name="workspace">Data pointer to GPU memory to be used as a workspace for this call.</param>
        /// <param name="workSpaceSizeInBytes">Specifies the size in bytes of the provided workspace.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnRNNForwardInference(cudnnHandle handle,
                                                    cudnnRNNDescriptor rnnDesc,
                                                    cudnnTensorDescriptor[] xDesc,
                                                    CUdeviceptr x,
                                                    cudnnTensorDescriptor hxDesc,
                                                    CUdeviceptr hx,
                                                    cudnnTensorDescriptor cxDesc,
                                                    CUdeviceptr cx,
                                                    cudnnFilterDescriptor wDesc,
                                                    CUdeviceptr w,
                                                    cudnnTensorDescriptor[] yDesc,
                                                    CUdeviceptr y,
                                                    cudnnTensorDescriptor hyDesc,
                                                    CUdeviceptr hy,
                                                    cudnnTensorDescriptor cyDesc,
                                                    CUdeviceptr cy,
                                                    CUdeviceptr workspace,
                                                    SizeT workSpaceSizeInBytes);


        /// <summary>
        /// This routine executes the recurrent neural network described by rnnDesc with inputs x, hx, cx, weights w 
        /// and outputs y, hy, cy. workspace is required for intermediate storage. reserveSpace stores data required 
        /// for training. The same reserveSpace data must be used for future calls to cudnnRNNBackwardData and 
        /// cudnnRNNBackwardWeights if these execute on the same input data. 
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="rnnDesc">A previously initialized RNN descriptor.</param>
        /// <param name="xDesc">An array of tensor descriptors describing the input to each recurrent iteration. Each 
        /// tensor descriptor must have the same first dimension. The second dimension of the tensors may decrease 
        /// from element n to element n+1 but may not increase. The tensor must be fully packed.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptors in the array xDesc.</param>
        /// <param name="hxDesc">Handle to a previously initialized tensor descriptor describing the initial hidden state 
        /// of the RNN. The first dimension of the tensor must match the hiddenSize argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second 
        /// dimension of the first tensor described in xDesc. The third dimension must match the numLayers argument 
        /// passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="hx">Data pointer to GPU memory associated with the tensor descriptor hxDesc. If a NULL pointer 
        /// is passed, the initial hidden state of the network will be initialized to zero.</param>
        /// <param name="cxDesc">Handle to a previously initialized tensor descriptor describing the initial 
        /// cell state for LSTM networks. The first dimension of the tensor must match the hiddenSize argument 
        /// passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match 
        /// the second dimension of the first tensor described in xDesc. The third dimension must match the numLayers 
        /// argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully 
        /// packed.</param>
        /// <param name="cx">Data pointer to GPU memory associated with the tensor descriptor cxDesc. If a NULL pointer is 
        /// passed, the initial cell state of the network will be initialized to zero.</param>
        /// <param name="wDesc">Handle to a previously initialized filter descriptor describing the weights for the RNN.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor wDesc.</param>
        /// <param name="yDesc">An array of tensor descriptors describing the output from each recurrent iteration. The first 
        /// dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor 
        /// call used to initialize rnnDesc: 
        /// * If direction is CUDNN_UNIDIRECTIONAL the first dimension should match the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// * If direction is CUDNN_BIDIRECTIONAL the first dimension should match double the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// The second dimension of the tensor n must match the second dimension of the tensor 
        /// n in xDesc. The tensor must be fully packed.</param>
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor yDesc.</param>
        /// <param name="hyDesc">Handle to a previously initialized tensor descriptor describing the final 
        /// hidden state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second dimension 
        /// of the first tensor described in xDesc. The third dimension must match the numLayers argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="hy">Data pointer to GPU memory associated with the tensor descriptor hyDesc. If a 
        /// NULL pointer is passed, the final hidden state of the network will not be saved.</param>
        /// <param name="cyDesc">Handle to a previously initialized tensor descriptor describing the final cell state 
        /// for LSTM networks. The first dimension of the tensor must match the hiddenSize argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second dimension 
        /// of the first tensor described in xDesc. The third dimension must match the numLayers argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="cy">Data pointer to GPU memory associated with the tensor descriptor cyDesc. If a NULL pointer is 
        /// passed, the final cell state of the network will be not be saved.</param>
        /// <param name="workspace">Data pointer to GPU memory to be used as a workspace for this call.</param>
        /// <param name="workSpaceSizeInBytes">Specifies the size in bytes of the provided workspace.</param>
        /// <param name="reserveSpace">Data pointer to GPU memory to be used as a reserve space for this call.</param>
        /// <param name="reserveSpaceSizeInBytes">Specifies the size in bytes of the provided reserveSpace.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnRNNForwardTraining(cudnnHandle handle,
                                                   cudnnRNNDescriptor rnnDesc,
                                                   cudnnTensorDescriptor[] xDesc,
                                                   CUdeviceptr x,
                                                   cudnnTensorDescriptor hxDesc,
                                                   CUdeviceptr hx,
                                                   cudnnTensorDescriptor cxDesc,
                                                   CUdeviceptr cx,
                                                   cudnnFilterDescriptor wDesc,
                                                   CUdeviceptr w,
                                                   cudnnTensorDescriptor[] yDesc,
                                                   CUdeviceptr y,
                                                   cudnnTensorDescriptor hyDesc,
                                                   CUdeviceptr hy,
                                                   cudnnTensorDescriptor cyDesc,
                                                   CUdeviceptr cy,
                                                   CUdeviceptr workspace,
                                                   SizeT workSpaceSizeInBytes,
                                                   CUdeviceptr reserveSpace,
                                                   SizeT reserveSpaceSizeInBytes);

        /// <summary>
        /// This routine executes the recurrent neural network described by rnnDesc with 
        /// output gradients dy, dhy, dhc, weights w and input gradients dx, dhx, dcx. 
        /// workspace is required for intermediate storage. The data in reserveSpace must have 
        /// previously been generated by cudnnRNNForwardTraining. The same reserveSpace data must 
        /// be used for future calls to cudnnRNNBackwardWeights if they execute on the same input data. 
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="rnnDesc">A previously initialized RNN descriptor.</param>
        /// <param name="yDesc">An array of tensor descriptors describing the output from each 
        /// recurrent iteration. The first dimension of the tensor depends on the direction 
        /// argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc:
        /// * If direction is CUDNN_UNIDIRECTIONAL the first dimension should match the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// * If direction is CUDNN_BIDIRECTIONAL the first dimension should match double the 
        /// hiddenSize argument passed to cudnnSetRNNDescriptor.
        /// The second dimension of the tensor n must match the second dimension of the tensor n in dyDesc. 
        /// The tensor must be fully packed.</param>
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor yDesc.</param>
        /// <param name="dyDesc">An array of tensor descriptors describing the gradient at the output from each 
        /// recurrent iteration. The first dimension of the tensor depends on the direction argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc: 
        /// * If direction is CUDNN_UNIDIRECTIONAL the first dimension should match the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// * If direction is CUDNN_BIDIRECTIONAL the first dimension should match double the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// The second dimension of the tensor n must match the second dimension of the tensor n in dxDesc. The 
        /// tensor must be fully packed.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the tensor descriptors in the array dyDesc.</param>
        /// <param name="dhyDesc">Handle to a previously initialized tensor descriptor describing the gradients at the 
        /// final hidden state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed 
        /// to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second 
        /// dimension of the first tensor described in dyDesc. The third dimension must match the numLayers argument 
        /// passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="dhy">Data pointer to GPU memory associated with the tensor descriptor dhyDesc. If a NULL pointer 
        /// is passed, the gradients at the final hidden state of the network will be initialized to zero.</param>
        /// <param name="dcyDesc">Handle to a previously initialized tensor descriptor describing the gradients at 
        /// the final cell state of the RNN. The first dimension of the tensor must match the hiddenSize argument 
        /// passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the 
        /// second dimension of the first tensor described in dyDesc. The third dimension must match the numLayers argument 
        /// passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="dcy">Data pointer to GPU memory associated with the tensor descriptor dcyDesc. If a NULL pointer 
        /// is passed, the gradients at the final cell state of the network will be initialized to zero.</param>
        /// <param name="wDesc">Handle to a previously initialized filter descriptor describing the weights for the RNN.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor wDesc.</param>
        /// <param name="hxDesc">Handle to a previously initialized tensor descriptor describing the initial hidden 
        /// state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second 
        /// dimension of the first tensor described in xDesc. The third dimension must match the numLayers 
        /// argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be 
        /// fully packed.</param>
        /// <param name="hx">Data pointer to GPU memory associated with the tensor descriptor hxDesc. If a NULL pointer is 
        /// passed, the initial hidden state of the network will be initialized to zero.</param>
        /// <param name="cxDesc">Handle to a previously initialized tensor descriptor describing the 
        /// initial cell state for LSTM networks. The first dimension of the tensor must match the 
        /// hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The 
        /// second dimension must match the second dimension of the first tensor described in xDesc. The 
        /// third dimension must match the numLayers argument passed to the cudnnSetRNNDescriptor call 
        /// used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="cx">Data pointer to GPU memory associated with the tensor descriptor cxDesc. 
        /// If a NULL pointer is passed, the initial cell state of the network will be initialized to zero.</param>
        /// <param name="dxDesc">An array of tensor descriptors describing the gradient at the input of each recurrent iteration. 
        /// Each tensor descriptor must have the same first dimension. The second dimension of the tensors may decrease from 
        /// element n to element n+1 but may not increase. The tensor must be fully packed.</param>
        /// <param name="dx">Data pointer to GPU memory associated with the tensor descriptors in the array dxDesc. </param>
        /// <param name="dhxDesc">Handle to a previously initialized tensor descriptor describing the gradient at the initial hidden 
        /// state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed to the cudnnSetRNNDescriptor 
        /// call used to initialize rnnDesc. The second dimension must match the second dimension of the first tensor described in xDesc. 
        /// The third dimension must match the numLayers argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. 
        /// The tensor must be fully packed.</param>
        /// <param name="dhx">Data pointer to GPU memory associated with the tensor descriptor dhxDesc. If a NULL pointer is passed, the 
        /// gradient at the hidden input of the network will not be set.</param>
        /// <param name="dcxDesc">Handle to a previously initialized tensor descriptor describing the gradient 
        /// at the initial cell state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed 
        /// to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second dimension 
        /// of the first tensor described in xDesc. The third dimension must match the numLayers argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="dcx">Data pointer to GPU memory associated with the tensor descriptor dcxDesc. If 
        /// a NULL pointer is passed, the gradient at the cell input of the network will not be set.</param>
        /// <param name="workspace">Data pointer to GPU memory to be used as a workspace for this call.</param>
        /// <param name="workSpaceSizeInBytes">Specifies the size in bytes of the provided workspace.</param>
        /// <param name="reserveSpace">Data pointer to GPU memory to be used as a reserve space for this call.</param>
        /// <param name="reserveSpaceSizeInBytes">Specifies the size in bytes of the provided reserveSpace.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnRNNBackwardData(cudnnHandle handle,
                                                cudnnRNNDescriptor rnnDesc,
                                                cudnnTensorDescriptor[] yDesc,
                                                CUdeviceptr y,
                                                cudnnTensorDescriptor[] dyDesc,
                                                CUdeviceptr dy,
                                                cudnnTensorDescriptor dhyDesc,
                                                CUdeviceptr dhy,
                                                cudnnTensorDescriptor dcyDesc,
                                                CUdeviceptr dcy,
                                                cudnnFilterDescriptor wDesc,
                                                CUdeviceptr w,
                                                cudnnTensorDescriptor hxDesc,
                                                CUdeviceptr hx,
                                                cudnnTensorDescriptor cxDesc,
                                                CUdeviceptr cx,
                                                cudnnTensorDescriptor[] dxDesc,
                                                CUdeviceptr dx,
                                                cudnnTensorDescriptor dhxDesc,
                                                CUdeviceptr dhx,
                                                cudnnTensorDescriptor dcxDesc,
                                                CUdeviceptr dcx,
                                                CUdeviceptr workspace,
                                                SizeT workSpaceSizeInBytes,
                                                CUdeviceptr reserveSpace,
                                                SizeT reserveSpaceSizeInBytes );

        /// <summary>
        /// This routine accumulates weight gradients dw from the recurrent neural network described 
        /// by rnnDesc with inputs x, hx, and outputs y. The mode of operation in this case is additive, 
        /// the weight gradients calculated will be added to those already existing in dw. workspace 
        /// is required for intermediate storage. The data in reserveSpace must have previously been 
        /// generated by cudnnRNNBackwardData.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="rnnDesc">A previously initialized RNN descriptor.</param>
        /// <param name="xDesc">An array of tensor descriptors describing the input to each recurrent iteration. 
        /// Each tensor descriptor must have the same first dimension. The second dimension of the tensors may 
        /// decrease from element n to element n+1 but may not increase. The tensor must be fully packed.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptors in the array xDesc.</param>
        /// <param name="hxDesc">Handle to a previously initialized tensor descriptor describing the initial hidden 
        /// state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second dimension
        /// of the first tensor described in xDesc. The third dimension must match the numLayers argument passed to 
        /// the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed. </param>
        /// <param name="hx">Data pointer to GPU memory associated with the tensor descriptor hxDesc. If 
        /// a NULL pointer is passed, the initial hidden state of the network will be initialized to zero.</param>
        /// <param name="yDesc">An array of tensor descriptors describing the output from each 
        /// recurrent iteration. The first dimension of the tensor depends on the direction 
        /// argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc:
        /// * If direction is CUDNN_UNIDIRECTIONAL the first dimension should match the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// * If direction is CUDNN_BIDIRECTIONAL the first dimension should match double the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// The second dimension of the tensor n must match the second dimension of the tensor n in dyDesc. 
        /// The tensor must be fully packed.</param>
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor yDesc.</param>
        /// <param name="workspace">Data pointer to GPU memory to be used as a workspace for this call.</param>
        /// <param name="workSpaceSizeInBytes">Specifies the size in bytes of the provided workspace.</param>
        /// <param name="dwDesc">Handle to a previously initialized filter descriptor describing the gradients of the weights for the RNN.</param>
        /// <param name="dw">Data pointer to GPU memory associated with the filter descriptor dwDesc.</param>
        /// <param name="reserveSpace">Data pointer to GPU memory to be used as a reserve space for this call.</param>
        /// <param name="reserveSpaceSizeInBytes">Specifies the size in bytes of the provided reserveSpace.</param>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnRNNBackwardWeights(cudnnHandle handle,
                                                   cudnnRNNDescriptor rnnDesc,
                                                   cudnnTensorDescriptor[] xDesc,
                                                   CUdeviceptr x,
                                                   cudnnTensorDescriptor hxDesc,
                                                   CUdeviceptr hx,
                                                   cudnnTensorDescriptor[] yDesc,
                                                   CUdeviceptr y,
                                                   CUdeviceptr workspace,
                                                   SizeT workSpaceSizeInBytes, 
                                                   cudnnFilterDescriptor dwDesc,
                                                   CUdeviceptr dw,
                                                   CUdeviceptr reserveSpace,
                                                   SizeT reserveSpaceSizeInBytes );







        /// <summary>
        /// Create an instance of a CTC (Connectionist Temporal Classification) loss descriptor
        /// </summary>
        /// <param name="ctcLossDesc"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnCreateCTCLossDescriptor(ref cudnnCTCLossDescriptor ctcLossDesc);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="ctcLossDesc"></param>
        /// <param name="compType"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSetCTCLossDescriptor(cudnnCTCLossDescriptor ctcLossDesc, cudnnDataType compType);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="ctcLossDesc"></param>
        /// <param name="compType"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetCTCLossDescriptor(cudnnCTCLossDescriptor ctcLossDesc, ref cudnnDataType compType);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="ctcLossDesc"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnDestroyCTCLossDescriptor(cudnnCTCLossDescriptor ctcLossDesc);

        /// <summary>
        /// return the ctc costs and gradients, given the probabilities and labels
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="probsDesc"></param>
        /// <param name="probs"></param>
        /// <param name="labels"></param>
        /// <param name="labelLengths"></param>
        /// <param name="inputLengths"></param>
        /// <param name="costs"></param>
        /// <param name="gradientsDesc"></param>
        /// <param name="gradients"></param>
        /// <param name="algo"></param>
        /// <param name="ctcLossDesc"></param>
        /// <param name="workspace"></param>
        /// <param name="workSpaceSizeInBytes"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnCTCLoss(cudnnHandle handle,
                                        cudnnTensorDescriptor probsDesc,     /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is the timing steps, N is the mini batch size, A is the alphabet size)  */
                                        CUdeviceptr probs,                          /* probabilities after softmax, in GPU memory */
                                        int[] labels,                          /* labels, in CPU memory */
                                        int[] labelLengths,                    /* the length of each label, in CPU memory */
                                        int[] inputLengths,                    /* the lengths of timing steps in each batch, in CPU memory */
                                        CUdeviceptr costs,                                /* the returned costs of CTC, in GPU memory */
                                        cudnnTensorDescriptor gradientsDesc, /* Tensor descriptor for gradients, the dimensions are T,N,A */
                                        CUdeviceptr gradients,                      /* the returned CTC gradients, in GPU memory, to compute costs only, set it to NULL */
                                        cudnnCTCLossAlgo algo,                     /* algorithm selected, supported now 0 and 1 */
                                        cudnnCTCLossDescriptor ctcLossDesc,
                                        CUdeviceptr workspace,                            /* pointer to the workspace, in GPU memory */
                                        SizeT workSpaceSizeInBytes);                /* the workspace size needed */

        /// <summary>
        /// return the workspace size needed for ctc
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="probsDesc"></param>
        /// <param name="gradientsDesc"></param>
        /// <param name="labels"></param>
        /// <param name="labelLengths"></param>
        /// <param name="inputLengths"></param>
        /// <param name="algo"></param>
        /// <param name="ctcLossDesc"></param>
        /// <param name="sizeInBytes"></param>
        /// <returns></returns>
        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetCTCLossWorkspaceSize(
                                cudnnHandle handle,
                                cudnnTensorDescriptor probsDesc,       /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is the timing steps, N is the mini batch size, A is the alphabet size) */
                                cudnnTensorDescriptor gradientsDesc,   /* Tensor descriptor for gradients, the dimensions are T,N,A. To compute costs only, set it to NULL */
                                int[] labels,         /* labels, in CPU memory */
                                int[] labelLengths,   /* the length of each label, in CPU memory */
                                int[] inputLengths,   /* the lengths of timing steps in each batch, in CPU memory */
                                cudnnCTCLossAlgo                  algo,            /* algorithm selected, supported now 0 and 1 */
                                cudnnCTCLossDescriptor ctcLossDesc,
                                ref SizeT sizeInBytes );   /* pointer to the returned workspace size */



        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnCreateAlgorithmDescriptor(
                                ref cudnnAlgorithmDescriptor algoDesc);

        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSetAlgorithmDescriptor(
                                        cudnnAlgorithmDescriptor algoDesc,
                                        cudnnAlgorithm algorithm);

        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetAlgorithmDescriptor(
                                cudnnAlgorithmDescriptor algoDesc,
                                ref cudnnAlgorithm algorithm);

        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnCopyAlgorithmDescriptor(
                                cudnnAlgorithmDescriptor src,
                                cudnnAlgorithmDescriptor dest);

        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnDestroyAlgorithmDescriptor(
                                cudnnAlgorithmDescriptor algoDesc);

        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnCreateAlgorithmPerformance(
                                        cudnnAlgorithmPerformance[] algoPerf,
                                        int numberToCreate);

        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSetAlgorithmPerformance(
                                        cudnnAlgorithmPerformance algoPerf,
                                        cudnnAlgorithmDescriptor algoDesc,
                                        cudnnStatus status,
                                        float time,
                                        SizeT memory);

        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetAlgorithmPerformance(
                                cudnnAlgorithmPerformance algoPerf,
                                ref cudnnAlgorithmDescriptor algoDesc,
                                ref cudnnStatus status,
                                ref float time,
                                ref SizeT memory );

        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnDestroyAlgorithmPerformance(
                                cudnnAlgorithmPerformance[] algoPerf,
                                int numberToDestroy);

        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetAlgorithmSpaceSize(
                                        cudnnHandle handle,
                                        cudnnAlgorithmDescriptor algoDesc,
                                        ref SizeT algoSpaceSizeInBytes);

        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSaveAlgorithm(
                                        cudnnHandle handle,
                                        cudnnAlgorithmDescriptor algoDesc,
                                        IntPtr algoSpace,
                                        SizeT algoSpaceSizeInBytes);

        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnRestoreAlgorithm(
                                        cudnnHandle handle,
                                        IntPtr algoSpace,
                                        SizeT algoSpaceSizeInBytes,
                                        cudnnAlgorithmDescriptor algoDesc);


        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnSetCallback(
                                MessageMask mask,
                                IntPtr udata,
                                cudnnCallback fptr);

        [DllImport(CUDNN_API_DLL_NAME)]
        public static extern cudnnStatus cudnnGetCallback(
                                        ref MessageMask mask,
                                        ref IntPtr udata,
                                        ref cudnnCallback fptr);


    }
}
