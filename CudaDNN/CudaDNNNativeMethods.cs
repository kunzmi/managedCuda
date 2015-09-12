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
		internal const string CUDNN_API_DLL_NAME = "cudnn64_70.dll";


		[DllImport(CUDNN_API_DLL_NAME, EntryPoint = "cudnnGetVersion")]
		internal static extern SizeT cudnnGetVersionInternal();
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
		public static string cudnnGetErrorString(cudnnStatus status)
		{
			IntPtr str = cudnnGetErrorStringInternal(status);
			return Marshal.PtrToStringAnsi(str);
		}
		
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnCreate(ref cudnnHandle handle);
		
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnDestroy(cudnnHandle handle);
		
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSetStream(cudnnHandle handle, CUstream streamId);

		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetStream(cudnnHandle handle, ref CUstream streamId);




		/* Create an instance of a generic Tensor descriptor */
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnCreateTensorDescriptor( ref cudnnTensorDescriptor tensorDesc );

		
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSetTensor4dDescriptor(cudnnTensorDescriptor tensorDesc,
																cudnnTensorFormat  format,
																cudnnDataType dataType, // image data type
																int n,        // number of inputs (batch size)
																int c,        // number of input feature maps
																int h,        // height of input section
																int w         // width of input section
															);

		
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
		
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSetTensorNdDescriptor(cudnnTensorDescriptor tensorDesc,
															   cudnnDataType dataType,
															   int nbDims,
															   int[] dimA,
															   int[] strideA
															 );
		
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetTensorNdDescriptor(  cudnnTensorDescriptor tensorDesc,
															   int nbDimsRequested,
															   ref cudnnDataType dataType,
															   ref int nbDims,
															   int[] dimA,
															   int[] strideA
															 );



		/* Destroy an instance of Tensor4d descriptor */
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnDestroyTensorDescriptor( cudnnTensorDescriptor tensorDesc );


		/* Tensor layout conversion helper (dest = alpha * src + beta * dest) */
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnTransformTensor(   cudnnHandle                    handle,
														  ref float alpha,
														  cudnnTensorDescriptor    srcDesc,
														  CUdeviceptr srcData,
														  ref float beta,
														  cudnnTensorDescriptor    destDesc,
														  CUdeviceptr destData
														);

		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnTransformTensor(   cudnnHandle                    handle,
														  ref double alpha,
														  cudnnTensorDescriptor    srcDesc,
														  CUdeviceptr srcData,
														  ref double beta,
														  cudnnTensorDescriptor    destDesc,
														  CUdeviceptr destData
														);



		/* Tensor Bias addition : srcDest = alpha * bias + beta * srcDestDesc  */
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnAddTensor(   cudnnHandle                    handle,
													cudnnAddMode                   mode,
													ref float alpha,
													cudnnTensorDescriptor    biasDesc,
													CUdeviceptr biasData,
													ref float beta,
													cudnnTensorDescriptor          srcDestDesc,
													CUdeviceptr srcDestData
												  );
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnAddTensor(   cudnnHandle                    handle,
													cudnnAddMode                   mode,
													ref double alpha,
													cudnnTensorDescriptor    biasDesc,
													CUdeviceptr biasData,
													ref double beta,
													cudnnTensorDescriptor          srcDestDesc,
													CUdeviceptr srcDestData
												  );

		/* Set all data points of a tensor to a given value : srcDest = value */
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSetTensor( cudnnHandle                   handle,
												  cudnnTensorDescriptor   srcDestDesc,
												  CUdeviceptr srcDestData,
												  CUdeviceptr value
												 );

		/* Set all data points of a tensor to a given value : srcDest = alpha * srcDest */
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnScaleTensor(   cudnnHandle                    handle,
													  cudnnTensorDescriptor    srcDestDesc,
													  CUdeviceptr srcDestData,
													  ref float alpha
												  );
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnScaleTensor(   cudnnHandle                    handle,
													  cudnnTensorDescriptor    srcDestDesc,
													  CUdeviceptr srcDestData,
													  ref double alpha
												  );




		/* Create an instance of FilterStruct */
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnCreateFilterDescriptor( ref cudnnFilterDescriptor filterDesc );

		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSetFilter4dDescriptor(  cudnnFilterDescriptor filterDesc,
															   cudnnDataType dataType, // image data type
															   int k,        // number of output feature maps
															   int c,        // number of input feature maps
															   int h,        // height of each input filter
															   int w         // width of  each input fitler
														  );

		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetFilter4dDescriptor(  cudnnFilterDescriptor filterDesc,
															   ref cudnnDataType dataType, // image data type
															   ref int k,        // number of output feature maps
															   ref int c,        // number of input feature maps
															   ref int h,        // height of each input filter
															   ref int w         // width of  each input fitler
														  );

		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSetFilterNdDescriptor(  cudnnFilterDescriptor filterDesc,
															   cudnnDataType dataType, // image data type
															   int nbDims,
															   int[] filterDimA
															 );

		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetFilterNdDescriptor(  cudnnFilterDescriptor filterDesc,
															   int nbDimsRequested,
															   ref cudnnDataType dataType, // image data type
															   ref int nbDims,
															   int[] filterDimA
															);

		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnDestroyFilterDescriptor( cudnnFilterDescriptor filterDesc );

		/* Create an instance of convolution descriptor */
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnCreateConvolutionDescriptor(ref cudnnConvolutionDescriptor convDesc );

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

		/* Helper function to return the dimensions of the output tensor given a convolution descriptor */
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetConvolution2dForwardOutputDim( cudnnConvolutionDescriptor convDesc,
																		 cudnnTensorDescriptor     inputTensorDesc,
																		 cudnnFilterDescriptor     filterDesc,
																		 ref int n,
																		 ref int c,
																		 ref int h,
																		 ref int w
																		);


		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSetConvolutionNdDescriptor( cudnnConvolutionDescriptor convDesc,
																   int arrayLength,             /* nbDims-2 size */  
																   int[] padA,                                          
																   int[] filterStrideA,         
																   int[] upscaleA,              
																   cudnnConvolutionMode mode
																 );

		[DllImport(CUDNN_API_DLL_NAME)]  
		public static extern cudnnStatus cudnnGetConvolutionNdDescriptor(cudnnConvolutionDescriptor convDesc,
																   int arrayLengthRequested,
																   ref int arrayLength,
																   int[] padA,                                        
																   int[] strideA,
																   int[] upscaleA,
																   ref cudnnConvolutionMode mode
																 );

		[DllImport(CUDNN_API_DLL_NAME)]  
		public static extern cudnnStatus cudnnSetConvolutionNdDescriptor_v3(cudnnConvolutionDescriptor convDesc,
                                                              int arrayLength,             /* nbDims-2 size */  
                                                              int[] padA,                                          
                                                              int[] filterStrideA,         
                                                              int[] upscaleA,              
                                                              cudnnConvolutionMode mode,
                                                              cudnnDataType dataType   // convolution data type
                                                         );
                                                         
		[DllImport(CUDNN_API_DLL_NAME)]  
		public static extern cudnnStatus cudnnGetConvolutionNdDescriptor_v3(cudnnConvolutionDescriptor convDesc,
                                                              int arrayLengthRequested,
                                                              ref int arrayLength,
                                                              int[] padA,                                        
                                                              int[] strideA,
                                                              int[] upscaleA,
                                                              ref cudnnConvolutionMode mode,
                                                              ref cudnnDataType dataType     // convolution data type
                                                         );




		/* Helper function to return the dimensions of the output tensor given a convolution descriptor */
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetConvolutionNdForwardOutputDim( cudnnConvolutionDescriptor convDesc,
																		 cudnnTensorDescriptor inputTensorDesc,
																		 cudnnFilterDescriptor filterDesc,
																		 int nbDims,
																		 int[] tensorOuputDimA
																		);

		/* Destroy an instance of convolution descriptor */
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnDestroyConvolutionDescriptor( cudnnConvolutionDescriptor convDesc );



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
                                                                                                           
		/*
		 *  convolution algorithm (which requires potentially some workspace)
		 */

		/* Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetConvolutionForwardWorkspaceSize( cudnnHandle                      handle, 
																		   cudnnTensorDescriptor      srcDesc,
																		   cudnnFilterDescriptor      filterDesc,
																		   cudnnConvolutionDescriptor convDesc,  
																		   cudnnTensorDescriptor      destDesc,
																		   cudnnConvolutionFwdAlgo          algo,
																		   ref SizeT                            sizeInBytes
																		);        


		/* Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */

		/* Function to perform the forward multiconvolution */
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

		/* Functions to perform the backward multiconvolution */
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionBackwardBias(cudnnHandle handle,
																	  ref float alpha,
																	  cudnnTensorDescriptor srcDesc,
																	  CUdeviceptr srcData,
																	  ref float beta,
																	  cudnnTensorDescriptor destDesc,
																	  CUdeviceptr destData
															  );
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionBackwardBias(cudnnHandle handle,
																	  ref double alpha,
																	  cudnnTensorDescriptor srcDesc,
																	  CUdeviceptr srcData,
																	  ref double beta,
																	  cudnnTensorDescriptor destDesc,
																	  CUdeviceptr destData
															  );



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



		/*
		 *  convolution algorithm (which requires potentially some workspace)
		 */

		 /* Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/ 
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetConvolutionBackwardFilterWorkspaceSize( cudnnHandle          handle, 
																				  cudnnTensorDescriptor       srcDesc,
																				  cudnnTensorDescriptor       diffDesc,
																				  cudnnConvolutionDescriptor  convDesc,  
																				  cudnnFilterDescriptor       gradDesc,
																				  cudnnConvolutionBwdFilterAlgo     algo,
																				  ref SizeT                         sizeInBytes
																				);
                                                       
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionBackwardFilter_v3( cudnnHandle                 handle,
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
                                             
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionBackwardFilter_v3( cudnnHandle                 handle,
																	 ref double alpha,
																	 cudnnTensorDescriptor       srcDesc,
																	 CUdeviceptr srcData,
																	 cudnnTensorDescriptor       diffDesc,
																	 CUdeviceptr diffData,
																	 cudnnConvolutionDescriptor  convDesc,
																	 cudnnConvolutionBwdFilterAlgo     algo,
																	 CUdeviceptr workSpace,
																	 SizeT                              workSpaceSizeInBytes,
																	 ref double beta,
																	 cudnnFilterDescriptor       gradDesc,
																	 CUdeviceptr gradData
																   );





		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionBackwardFilter( cudnnHandle handle,
																  ref float alpha,
																  cudnnTensorDescriptor srcDesc,
																  CUdeviceptr srcData,
																  cudnnTensorDescriptor diffDesc,
																  CUdeviceptr diffData,
																  cudnnConvolutionDescriptor convDesc,
																  ref float beta,
																  cudnnFilterDescriptor gradDesc,
																  CUdeviceptr gradData
																);
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionBackwardFilter( cudnnHandle handle,
																  ref double alpha,
																  cudnnTensorDescriptor srcDesc,
																  CUdeviceptr srcData,
																  cudnnTensorDescriptor diffDesc,
																  CUdeviceptr diffData,
																  cudnnConvolutionDescriptor convDesc,
																  ref double beta,
																  cudnnFilterDescriptor gradDesc,
																  CUdeviceptr gradData
																);




		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnFindConvolutionBackwardDataAlgorithm( cudnnHandle handle,
                                                                     cudnnFilterDescriptor       filterDesc,
                                                                     cudnnTensorDescriptor       diffDesc,
                                                                     cudnnConvolutionDescriptor  convDesc, 
                                                                     cudnnTensorDescriptor       gradDesc,
                                                                     int                           requestedAlgoCount,
                                                                     ref int                               *returnedAlgoCount,
                                                                     cudnnConvolutionBwdDataAlgoPerf[] perfResults  
                                                                   );
                                          
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

		 /* Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/ 
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetConvolutionBackwardDataWorkspaceSize( cudnnHandle handle,
																		   cudnnFilterDescriptor      filterDesc,
																		   cudnnTensorDescriptor       diffDesc,
																		   cudnnConvolutionDescriptor convDesc,  
																		   cudnnTensorDescriptor       gradDesc,
																		   cudnnConvolutionBwdDataAlgo          algo,
																		   ref SizeT                            sizeInBytes
																		);        

                                         
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionBackwardData_v3( cudnnHandle handle,
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
   
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionBackwardData_v3( cudnnHandle handle,
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




		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionBackwardData( cudnnHandle handle,
																 ref float alpha,
																 cudnnFilterDescriptor       filterDesc,
																 CUdeviceptr filterData,
																 cudnnTensorDescriptor       diffDesc,
																 CUdeviceptr diffData,
																 cudnnConvolutionDescriptor  convDesc,
																 ref float beta,
																 cudnnTensorDescriptor       gradDesc,
																 CUdeviceptr gradData
															   );
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnConvolutionBackwardData( cudnnHandle handle,
																 ref double alpha,
																 cudnnFilterDescriptor       filterDesc,
																 CUdeviceptr filterData,
																 cudnnTensorDescriptor       diffDesc,
																 CUdeviceptr diffData,
																 cudnnConvolutionDescriptor  convDesc,
																 ref double beta,
																 cudnnTensorDescriptor       gradDesc,
																 CUdeviceptr gradData
															   );

		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnIm2Col(  cudnnHandle handle,
												cudnnTensorDescriptor srcDesc,
												CUdeviceptr srcData,
												cudnnFilterDescriptor filterDesc,                                        
												cudnnConvolutionDescriptor convDesc,
												CUdeviceptr colBuffer
											 );




		/* Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" */

		/* Function to perform forward softmax */
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

		/* Function to perform backward softmax */
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



		/* Create an instance of pooling descriptor */
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnCreatePoolingDescriptor( ref cudnnPoolingDescriptor poolingDesc);

		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSetPooling2dDescriptor(  cudnnPoolingDescriptor poolingDesc,
																cudnnPoolingMode mode,
																int windowHeight,
																int windowWidth,
																int verticalPadding,
																int horizontalPadding,
																int verticalStride,
																int horizontalStride
														   );

		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetPooling2dDescriptor(  cudnnPoolingDescriptor poolingDesc,
																ref cudnnPoolingMode mode,
																ref int windowHeight,
																ref int windowWidth,
																ref int verticalPadding,
																ref int horizontalPadding,
																ref int verticalStride,
																ref int horizontalStride
														   );

		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSetPoolingNdDescriptor(  cudnnPoolingDescriptor poolingDesc,
																cudnnPoolingMode mode,
																int nbDims,
																int[] windowDimA,
																int[] paddingA,
																int[] strideA
														   );

		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetPoolingNdDescriptor(  cudnnPoolingDescriptor poolingDesc,
																int nbDimsRequested,
																ref cudnnPoolingMode mode,
																ref int nbDims,
																int[] windowDimA,
																int[] paddingA,
																int[] strideA
															 );

		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetPoolingNdForwardOutputDim( cudnnPoolingDescriptor poolingDesc,
																	 cudnnTensorDescriptor inputTensorDesc,
																	 int nbDims,
																	 int[] outputTensorDimA);

		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetPooling2dForwardOutputDim( cudnnPoolingDescriptor poolingDesc,
																	 cudnnTensorDescriptor inputTensorDesc,
																	 ref int outN,
																	 ref int outC,
																	 ref int outH,
																	 ref int outW);


		/* Destroy an instance of pooling descriptor */
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnDestroyPoolingDescriptor( cudnnPoolingDescriptor poolingDesc );

		/* Pooling functions: All of the form "output = alpha * Op(inputs) + beta * output" */

		/* Function to perform forward pooling */
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

		/* Function to perform backward pooling */
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

		/* Function to perform forward activation  */
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnActivationForward( cudnnHandle handle,
														  cudnnActivationMode mode,
														  ref float alpha,
														  cudnnTensorDescriptor srcDesc,
														  CUdeviceptr srcData,
														  ref float beta,
														  cudnnTensorDescriptor destDesc,
														  CUdeviceptr destData
														);
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnActivationForward( cudnnHandle handle,
														  cudnnActivationMode mode,
														  ref double alpha,
														  cudnnTensorDescriptor srcDesc,
														  CUdeviceptr srcData,
														  ref double beta,
														  cudnnTensorDescriptor destDesc,
														  CUdeviceptr destData
														);

		/* Function to perform backward activation  */
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnActivationBackward( cudnnHandle handle,
														   cudnnActivationMode mode,
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
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnActivationBackward( cudnnHandle handle,
														   cudnnActivationMode mode,
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
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnSetLRNDescriptor(
									  cudnnLRNDescriptor   normDesc,
									  uint               lrnN,
									  double                 lrnAlpha,
									  double                 lrnBeta,
									  double                 lrnK);

		// Retrieve the settings currently stored in an LRN layer descriptor
		// Any of the provided pointers can be NULL (no corresponding value will be returned)
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnGetLRNDescriptor(
									  cudnnLRNDescriptor   normDesc,
									  ref uint              lrnN,
									  ref double                lrnAlpha,
									  ref double                lrnBeta,
									  ref double                lrnK);

		// Destroy an instance of LRN descriptor
		[DllImport(CUDNN_API_DLL_NAME)]
		public static extern cudnnStatus cudnnDestroyLRNDescriptor( cudnnLRNDescriptor lrnDesc );

		// LRN functions: of the form "output = alpha * normalize(srcData) + beta * destData"

		// Function to perform LRN forward cross-channel computation
		// Values of double parameters will be cast down to tensor data type
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

		// Function to perform LRN cross-channel backpropagation
		// values of double parameters will be cast down to tensor data type
		// src is the front layer, dst is the back layer
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

		

		// LCN/divisive normalization functions: of the form "output = alpha * normalize(srcData) + beta * destData"
		// srcMeansData can be NULL to reproduce Caffe's LRN within-channel behavior
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
									  ref float betaData,
									  cudnnTensorDescriptor    destDataDesc, // same desc for dest, means, meansDiff
									  CUdeviceptr destDataDiff, // output data differential
									  CUdeviceptr destMeansDiff // output means differential, can be NULL
									  );
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
									  ref double betaData,
									  cudnnTensorDescriptor    destDataDesc, // same desc for dest, means, meansDiff
									  CUdeviceptr destDataDiff, // output data differential
									  CUdeviceptr destMeansDiff // output means differential, can be NULL
									  );

	}
}
