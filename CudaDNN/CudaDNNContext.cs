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
using System.Text;
using System.Diagnostics;
using ManagedCuda.BasicTypes;

namespace ManagedCuda.CudaDNN
{
	/// <summary>
	/// An opaque structure holding the cuDNN library context.<para/>
	/// The cuDNN library context must be created using cudnnCreate() and the returned
	/// handle must be passed to all subsequent library function calls. The context should be
	/// destroyed at the end using cudnnDestroy(). The context is associated with only one
	/// GPU device, the current device at the time of the call to cudnnCreate(). However
	/// multiple contexts can be created on the same GPU device.
	/// </summary>
	public class CudaDNNContext : IDisposable
	{
		private cudnnHandle _handle;
		private cudnnStatus res;
		private bool disposed;

		#region Contructors
		/// <summary>
		/// </summary>
		public CudaDNNContext()
		{
			_handle = new cudnnHandle();
			res = CudaDNNNativeMethods.cudnnCreate(ref _handle);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCreate", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaDNNContext()
		{
			Dispose(false);
		}
		#endregion

		#region Dispose
		/// <summary>
		/// Dispose
		/// </summary>
		public void Dispose()
		{
			Dispose(true);
			GC.SuppressFinalize(this);
		}

		/// <summary>
		/// For IDisposable
		/// </summary>
		/// <param name="fDisposing"></param>
		protected virtual void Dispose(bool fDisposing)
		{
			if (fDisposing && !disposed)
			{
				//Ignore if failing
				res = CudaDNNNativeMethods.cudnnDestroy(_handle);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDestroy", res));
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Stream

		/// <summary>
		/// This function sets the stream to be used by the cudnn library to execute its routines.
		/// </summary>
		/// <param name="stream">the stream to be used by the library.</param>
		public void SetStream(CudaStream stream)
		{
			res = CudaDNNNativeMethods.cudnnSetStream(_handle, stream.Stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetStream", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/// <summary>
		/// This function gets the stream to be used by the cudnn library to execute its routines.
		/// </summary>
		public CudaStream GetStream()
		{
			CUstream stream = new CUstream();
			res = CudaDNNNativeMethods.cudnnGetStream(_handle, ref stream);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetStream", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);

			return new CudaStream(stream);
		}
		#endregion

		/// <summary>
		/// Returns the inner handle.
		/// </summary>
		public cudnnHandle Handle
		{
			get { return _handle; }
		}

		#region floats
		/// <summary>
		/// This function copies the scaled data from one tensor to another tensor with a different
		/// layout. Those descriptors need to have the same dimensions but not necessarily the
		/// same strides. The input and output tensors must not overlap in any way (i.e., tensors
		/// cannot be transformed in place). This function can be used to convert a tensor with an
		/// unsupported format to a supported one.
		/// </summary>
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
		public void TransformTensor(float alpha,
											TensorDescriptor srcDesc,
											CudaDeviceVariable<float> srcData,
											float beta,
											TensorDescriptor destDesc,
											CudaDeviceVariable<float> destData
										)
		{
			res = CudaDNNNativeMethods.cudnnTransformTensor(_handle, ref alpha, srcDesc.Desc, srcData.DevicePointer, ref beta, destDesc.Desc, destData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnTransformTensor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}



		/// <summary>
		/// This function adds the scaled values of one bias tensor to another tensor. Each dimension
		/// of the bias tensor must match the coresponding dimension of the srcDest tensor or
		/// must be equal to 1. In the latter case, the same value from the bias tensor for thoses
		/// dimensions will be used to blend into the srcDest tensor.
		/// </summary>
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
		public void AddTensor(float alpha,
									TensorDescriptor biasDesc,
									CudaDeviceVariable<float> biasData,
									float beta,
									TensorDescriptor srcDestDesc,
									CudaDeviceVariable<float> srcDestData
									)
		{
			res = CudaDNNNativeMethods.cudnnAddTensor(_handle, ref alpha, biasDesc.Desc, biasData.DevicePointer, ref beta, srcDestDesc.Desc, srcDestData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnAddTensor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/// <summary>
		/// This function sets all the elements of a tensor to a given value
		/// </summary>
		/// <param name="srcDestDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="srcDestData">Pointer to data of the tensor described by the srcDestDesc descriptor.</param>
		/// <param name="value">Pointer in Host memory to a value that all elements of the tensor will be set to.</param>
		public void SetTensor(TensorDescriptor srcDestDesc,
									CudaDeviceVariable<float> srcDestData,
									CudaDeviceVariable<float> value
									)
		{
			res = CudaDNNNativeMethods.cudnnSetTensor(_handle, srcDestDesc.Desc, srcDestData.DevicePointer, value.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetTensor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/// <summary>
		/// This function scale all the elements of a tensor by a give factor.
		/// </summary>
		/// <param name="srcDestDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="srcDestData">Pointer to data of the tensor described by the srcDestDesc descriptor.</param>
		/// <param name="alpha">Pointer in Host memory to a value that all elements of the tensor will be scaled with.</param>
		public void ScaleTensor(TensorDescriptor srcDestDesc,
										CudaDeviceVariable<float> srcDestData,
										float alpha
									)
		{
			res = CudaDNNNativeMethods.cudnnScaleTensor(_handle, srcDestDesc.Desc, srcDestData.DevicePointer, ref alpha);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnScaleTensor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}



        /// <summary>
        /// This function attempts all cuDNN algorithms and outputs performance metrics to a
        /// user-allocated array of cudnnConvolutionFwdAlgoPerf_t. These metrics are written
        /// in sorted fashion where the first element has the lowest compute time.
        /// </summary>
        /// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="destDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="requestedAlgoCount">The maximum number of elements to be stored in perfResults.</param>
        /// <param name="returnedAlgoCount">The number of output elements stored in perfResults.</param>
        /// <param name="perfResults">A user-allocated array to store performance metrics sorted ascending by
        /// compute time.</param>
        public void FindConvolutionForwardAlgorithm(TensorDescriptor srcDesc,
                                                    FilterDescriptor filterDesc,
                                                    ConvolutionDescriptor convDesc,
                                                    TensorDescriptor destDesc,
                                                    int requestedAlgoCount,
                                                    ref int returnedAlgoCount,
                                                    cudnnConvolutionFwdAlgoPerf[] perfResults
                                                )
        {
            res = CudaDNNNativeMethods.cudnnFindConvolutionForwardAlgorithm(_handle, srcDesc.Desc, filterDesc.Desc, convDesc.Desc, destDesc.Desc, requestedAlgoCount, ref returnedAlgoCount, perfResults);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnFindConvolutionForwardAlgorithm", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

		/// <summary>
		/// This function serves as a heuristic for obtaining the best suited algorithm for
		/// cudnnConvolutionForward for the given layer specifications. Based on the input
		/// preference, this function will either return the fastest algorithm or the fastest algorithm
		/// within a given memory limit. For an exhaustive search for the fastest algorithm, please
		/// use cudnnFindConvolutionForwardAlgorithm.
		/// </summary>
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
		public void GetConvolutionForwardAlgorithm(TensorDescriptor srcDesc,
													FilterDescriptor filterDesc,
													ConvolutionDescriptor convDesc,
													TensorDescriptor destDesc,
													cudnnConvolutionFwdPreference preference,
													SizeT memoryLimitInbytes,
													ref cudnnConvolutionFwdAlgo algo
													)
		{
			res = CudaDNNNativeMethods.cudnnGetConvolutionForwardAlgorithm(_handle, srcDesc.Desc, filterDesc.Desc, convDesc.Desc, destDesc.Desc, preference, memoryLimitInbytes, ref algo);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionForwardAlgorithm", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/// <summary>
		/// This function returns the amount of GPU memory workspace the user needs
		/// to allocate to be able to call cudnnConvolutionForward with the specified
		/// algorithm. The workspace allocated will then be passed to the routine
		/// cudnnConvolutionForward. The specified algorithm can be the result of the call to
		/// cudnnGetConvolutionForwardAlgorithm or can be chosen arbitrarily by the user.
		/// Note that not every algorithm is available for every configuration of the input tensor
		/// and/or every configuration of the convolution descriptor.
		/// </summary>
		/// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="destDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="algo">Enumerant that specifies the chosen convolution algorithm</param>
		public SizeT GetConvolutionForwardWorkspaceSize(TensorDescriptor srcDesc,
														FilterDescriptor filterDesc,
														ConvolutionDescriptor convDesc,
														TensorDescriptor destDesc,
														cudnnConvolutionFwdAlgo algo
													)
		{
			SizeT sizeInBytes = 0;
			res = CudaDNNNativeMethods.cudnnGetConvolutionForwardWorkspaceSize(_handle, srcDesc.Desc, filterDesc.Desc, convDesc.Desc, destDesc.Desc, algo, ref sizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionForwardWorkspaceSize", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
			return sizeInBytes;
		}

		/// <summary>
		/// This function executes convolutions or cross-correlations over src using the specified
		/// filters, returning results in dest. Scaling factors alpha and beta can be used to scale
		/// the input tensor and the output tensor respectively.
		/// </summary>
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
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="destDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="destData">Data pointer to GPU memory associated with the tensor descriptor
		/// destDesc that carries the result of the convolution.</param>
		public void ConvolutionForward(float alpha,
										TensorDescriptor srcDesc,
										CudaDeviceVariable<float> srcData,
										FilterDescriptor filterDesc,
										CudaDeviceVariable<float> filterData,
										ConvolutionDescriptor convDesc,
										cudnnConvolutionFwdAlgo algo,
										CudaDeviceVariable<byte> workSpace,
										float beta,
										TensorDescriptor destDesc,
										CudaDeviceVariable<float> destData
									)
		{
			res = CudaDNNNativeMethods.cudnnConvolutionForward(_handle, ref alpha, srcDesc.Desc, srcData.DevicePointer, filterDesc.Desc, filterData.DevicePointer, convDesc.Desc, algo, workSpace.DevicePointer, workSpace.SizeInBytes, ref beta, destDesc.Desc, destData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnConvolutionForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/// <summary>
		/// This function computes the convolution gradient with respect to the bias, which is the
		/// sum of every element belonging to the same feature map across all of the images of the
		/// input tensor. Therefore, the number of elements produced is equal to the number of
		/// features maps of the input tensor.
		/// </summary>
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
		public void ConvolutionBackwardBias(float alpha,
											TensorDescriptor srcDesc,
											CudaDeviceVariable<float> srcData,
											float beta,
											TensorDescriptor destDesc,
											CudaDeviceVariable<float> destData
									)
		{
			res = CudaDNNNativeMethods.cudnnConvolutionBackwardBias(_handle, ref alpha, srcDesc.Desc, srcData.DevicePointer, ref beta, destDesc.Desc, destData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnConvolutionBackwardBias", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


	    /// <summary>
	    /// This function attempts all cuDNN algorithms for cudnnConvolutionBackwardFilter_v3 and outputs performance metrics to a user-
	    /// allocated array of cudnnConvolutionBwdFilterAlgoPerf_t. These metrics are
	    /// written in sorted fashion where the first element has the lowest compute time. 
	    /// </summary>
	    /// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
	    /// <param name="diffDesc">Handle to the previously initialized input differential tensor descriptor.</param>
	    /// <param name="convDesc">Previously initialized convolution descriptor.</param>
	    /// <param name="gradDesc">Handle to a previously initialized filter descriptor.</param>
	    /// <param name="requestedAlgoCount">The maximum number of elements to be stored in perfResults.</param>
	    /// <param name="returnedAlgoCount">The number of output elements stored in perfResults.</param>
	    /// <param name="perfResults">A user-allocated array to store performance metrics sorted ascending by compute time.</param>
	    public void FindConvolutionBackwardFilterAlgorithm(TensorDescriptor srcDesc,
	                                                        TensorDescriptor diffDesc,
	                                                        ConvolutionDescriptor convDesc,
	                                                        FilterDescriptor gradDesc,
	                                                        int requestedAlgoCount,
	                                                        ref int returnedAlgoCount,
	                                                        cudnnConvolutionBwdFilterAlgoPerf[] perfResults
	                                                        )
        {
            res = CudaDNNNativeMethods.cudnnFindConvolutionBackwardFilterAlgorithm(_handle, srcDesc.Desc, diffDesc.Desc, convDesc.Desc, gradDesc.Desc, requestedAlgoCount, ref returnedAlgoCount, perfResults);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnFindConvolutionBackwardFilterAlgorithm", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function serves as a heuristic for obtaining the best suited algorithm for
        /// cudnnConvolutionBackwardFilter_v3 for the given layer specifications. Based
        /// on the input preference, this function will either return the fastest algorithm or the
        /// fastest algorithm within a given memory limit. For an exhaustive search for the fastest
        /// algorithm, please use cudnnFindConvolutionBackwardFilterAlgorithm.
        /// </summary>
        /// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="diffDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="gradDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="preference">Enumerant to express the preference criteria in terms of memory requirement and speed.</param>
        /// <param name="memoryLimitInbytes">It is to specify the maximum amount of GPU memory the user is willing to 
        /// use as a workspace. This is currently a placeholder and is not used.</param>
        /// <param name="algo">Enumerant that specifies which convolution algorithm should be used to
        /// compute the results according to the specified preference</param>
        public void GetConvolutionBackwardFilterAlgorithm(TensorDescriptor srcDesc,
                                                            TensorDescriptor diffDesc,
                                                            ConvolutionDescriptor convDesc,
                                                            FilterDescriptor gradDesc,
                                                            cudnnConvolutionBwdFilterPreference preference,
                                                            SizeT memoryLimitInbytes,
                                                            ref cudnnConvolutionBwdFilterAlgo algo
                                                            )
        {
            res = CudaDNNNativeMethods.cudnnGetConvolutionBackwardFilterAlgorithm(_handle, srcDesc.Desc, diffDesc.Desc, convDesc.Desc, gradDesc.Desc, preference, memoryLimitInbytes, ref algo);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionBackwardFilterAlgorithm", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

		/// <summary>
		/// This function returns the amount of GPU memory workspace the user needs
		/// to allocate to be able to call cudnnConvolutionBackwardFilter_v3 with the
		/// specified algorithm. The workspace allocated will then be passed to the routine
		/// cudnnConvolutionBackwardFilter_v3. The specified algorithm can be the result
		/// of the call to cudnnGetConvolutionBackwardFilterAlgorithm or can be chosen
		/// arbitrarily by the user. Note that not every algorithm is available for every configuration
		/// of the input tensor and/or every configuration of the convolution descriptor.
		/// </summary>
		/// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="diffDesc">Handle to the previously initialized input differential tensor descriptor.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="gradDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="algo">Enumerant that specifies the chosen convolution algorithm
		/// sizeInBytes output Amount of GPU memory needed as workspace to be able to execute</param>
		/// <param name="sizeInBytes">Amount of GPU memory needed as workspace to be able to execute a
		/// forward convolution with the specified algo</param>
		public void GetConvolutionBackwardFilterWorkspaceSize(TensorDescriptor       srcDesc,
																	TensorDescriptor       diffDesc,
																	ConvolutionDescriptor  convDesc,  
																	FilterDescriptor       gradDesc,
																	cudnnConvolutionBwdFilterAlgo     algo,
																	ref SizeT                         sizeInBytes
																)
        {
            res = CudaDNNNativeMethods.cudnnGetConvolutionBackwardFilterWorkspaceSize(_handle, srcDesc.Desc, diffDesc.Desc, convDesc.Desc, gradDesc.Desc, algo, ref sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionBackwardFilterWorkspaceSize", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

		/// <summary>
		/// This function computes the convolution gradient with respect to filter coefficients using
		/// the specified algo, returning results in gradDesc.Scaling factors alpha and beta can be
		/// used to scale the input tensor and the output tensor respectively.
		/// </summary>
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
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="gradDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="gradData">Data pointer to GPU memory associated with the filter descriptor
		/// gradDesc that carries the result.</param> 
		public void ConvolutionBackwardFilter(float alpha,
												TensorDescriptor srcDesc,
												CudaDeviceVariable<float> srcData,
												TensorDescriptor diffDesc,
												CudaDeviceVariable<float> diffData,
												ConvolutionDescriptor convDesc,
												cudnnConvolutionBwdFilterAlgo algo,
												CudaDeviceVariable<byte> workSpace,
												float beta,
												FilterDescriptor gradDesc,
												CudaDeviceVariable<float> gradData
											)
		{
			res = CudaDNNNativeMethods.cudnnConvolutionBackwardFilter(_handle, ref alpha, srcDesc.Desc, srcData.DevicePointer, diffDesc.Desc, diffData.DevicePointer, convDesc.Desc, algo, workSpace.DevicePointer, workSpace.SizeInBytes, ref beta, gradDesc.Desc, gradData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnConvolutionBackwardFilter", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


        /// <summary>
        /// This function attempts all cuDNN algorithms for
        /// cudnnConvolutionBackwardData_v3 and outputs performance metrics to a user-
        /// allocated array of cudnnConvolutionBwdDataAlgoPerf_t. These metrics are written
        /// in sorted fashion where the first element has the lowest compute time.
        /// </summary>
        /// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="diffDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="gradDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="requestedAlgoCount">The maximum number of elements to be stored in perfResults.</param>
        /// <param name="returnedAlgoCount">The number of output elements stored in perfResults.</param>
        /// <param name="perfResults">A user-allocated array to store performance metrics sorted ascending by compute time.</param>
        public void FindConvolutionBackwardDataAlgorithm(FilterDescriptor filterDesc,
                                                            TensorDescriptor diffDesc,
                                                            ConvolutionDescriptor convDesc,
                                                            TensorDescriptor gradDesc,
                                                            int requestedAlgoCount,
                                                            ref int returnedAlgoCount,
                                                            cudnnConvolutionBwdDataAlgoPerf[] perfResults
                                                        )
        {
            res = CudaDNNNativeMethods.cudnnFindConvolutionBackwardDataAlgorithm(_handle, filterDesc.Desc, diffDesc.Desc, convDesc.Desc, gradDesc.Desc, requestedAlgoCount, ref returnedAlgoCount, perfResults);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnFindConvolutionBackwardDataAlgorithm", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function serves as a heuristic for obtaining the best suited algorithm for
        /// cudnnConvolutionBackwardData_v3 for the given layer specifications. Based
        /// on the input preference, this function will either return the fastest algorithm or the
        /// fastest algorithm within a given memory limit. For an exhaustive search for the fastest
        /// algorithm, please use cudnnFindConvolutionBackwardDataAlgorithm.
        /// </summary>
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
        public void GetConvolutionBackwardDataAlgorithm(FilterDescriptor filterDesc,
                                                        TensorDescriptor diffDesc,
                                                        ConvolutionDescriptor convDesc,
                                                        TensorDescriptor gradDesc,
                                                        cudnnConvolutionBwdDataPreference preference,
                                                        SizeT memoryLimitInbytes,
                                                        ref cudnnConvolutionBwdDataAlgo algo
                                                        )
        {
            res = CudaDNNNativeMethods.cudnnGetConvolutionBackwardDataAlgorithm(_handle, filterDesc.Desc, diffDesc.Desc, convDesc.Desc, gradDesc.Desc, preference, memoryLimitInbytes, ref algo);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionBackwardDataAlgorithm", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function returns the amount of GPU memory workspace the user needs
        /// to allocate to be able to call cudnnConvolutionBackwardData_v3 with the
        /// specified algorithm. The workspace allocated will then be passed to the routine
        /// cudnnConvolutionBackwardData_v3. The specified algorithm can be the result of the
        /// call to cudnnGetConvolutionBackwardDataAlgorithm or can be chosen arbitrarily
        /// by the user. Note that not every algorithm is available for every configuration of the
        /// input tensor and/or every configuration of the convolution descriptor.
        /// </summary>
        /// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="diffDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="gradDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="algo">Enumerant that specifies the chosen convolution algorithm</param>
        /// <param name="sizeInBytes">Amount of GPU memory needed as workspace to be able to execute a forward convolution with the specified algo</param>
        public void GetConvolutionBackwardDataWorkspaceSize(FilterDescriptor filterDesc,
                                                            TensorDescriptor diffDesc,
                                                            ConvolutionDescriptor convDesc,
                                                            TensorDescriptor gradDesc,
                                                            cudnnConvolutionBwdDataAlgo algo,
                                                            ref SizeT sizeInBytes
                                                        )
        {
            res = CudaDNNNativeMethods.cudnnGetConvolutionBackwardDataWorkspaceSize(_handle, filterDesc.Desc, diffDesc.Desc, convDesc.Desc, gradDesc.Desc, algo, ref sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionBackwardDataWorkspaceSize", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

		/// <summary>
		/// This function computes the convolution gradient with respect to the output tensor using
		/// the specified algo, returning results in gradDesc. Scaling factors alpha and beta can
		/// be used to scale the input tensor and the output tensor respectively.
		/// </summary>
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
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="gradDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="gradData">Data pointer to GPU memory associated with the output tensor descriptor
		/// gradDesc that carries the result.</param>
		public void ConvolutionBackwardData(float alpha,
											FilterDescriptor filterDesc,
											CudaDeviceVariable<float> filterData,
											TensorDescriptor diffDesc,
											CudaDeviceVariable<float> diffData,
											ConvolutionDescriptor convDesc,
											float beta,
											cudnnConvolutionBwdDataAlgo algo,
											CudaDeviceVariable<byte> workSpace,
											TensorDescriptor gradDesc,
											CudaDeviceVariable<float> gradData
										)
		{
			res = CudaDNNNativeMethods.cudnnConvolutionBackwardData(_handle, ref alpha, filterDesc.Desc, filterData.DevicePointer, diffDesc.Desc, diffData.DevicePointer, convDesc.Desc, algo, workSpace.DevicePointer, workSpace.SizeInBytes, ref beta, gradDesc.Desc, gradData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnConvolutionBackwardData", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="srcDesc"></param>
		/// <param name="srcData"></param>
		/// <param name="filterDesc"></param>
		/// <param name="convDesc"></param>
		/// <param name="colBuffer"></param>
		public void Im2Col(
							TensorDescriptor srcDesc,
							CudaDeviceVariable<float> srcData,
							FilterDescriptor filterDesc,
							ConvolutionDescriptor convDesc,
							CudaDeviceVariable<byte> colBuffer
							)
		{
			res = CudaDNNNativeMethods.cudnnIm2Col(_handle, srcDesc.Desc, srcData.DevicePointer, filterDesc.Desc, convDesc.Desc, colBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnIm2Col", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}




		/* Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" */

		/// <summary>
		/// This routine computes the softmax function.
		/// </summary>
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
		public void SoftmaxForward(cudnnSoftmaxAlgorithm algorithm,
									cudnnSoftmaxMode mode,
									float alpha,
									TensorDescriptor srcDesc,
									CudaDeviceVariable<float> srcData,
									float beta,
									TensorDescriptor destDesc,
									CudaDeviceVariable<float> destData
									)
		{
			res = CudaDNNNativeMethods.cudnnSoftmaxForward(_handle, algorithm, mode, ref alpha, srcDesc.Desc, srcData.DevicePointer, ref beta, destDesc.Desc, destData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSoftmaxForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/// <summary>
		/// This routine computes the gradient of the softmax function.
		/// </summary>
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
		public void SoftmaxBackward(cudnnSoftmaxAlgorithm algorithm,
									cudnnSoftmaxMode mode,
									float alpha,
									TensorDescriptor srcDesc,
									CudaDeviceVariable<float> srcData,
									TensorDescriptor srcDiffDesc,
									CudaDeviceVariable<float> srcDiffData,
									float beta,
									TensorDescriptor destDiffDesc,
									CudaDeviceVariable<float> destDiffData
									)
		{
			res = CudaDNNNativeMethods.cudnnSoftmaxBackward(_handle, algorithm, mode, ref alpha, srcDesc.Desc, srcData.DevicePointer, srcDiffDesc.Desc, srcDiffData.DevicePointer, ref beta, destDiffDesc.Desc, destDiffData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSoftmaxBackward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}







		/// <summary>
		/// This function computes pooling of input values (i.e., the maximum or average of several
		/// adjacent values) to produce an output with smaller height and/or width.
		/// </summary>
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
		public void PoolingForward(PoolingDescriptor poolingDesc,
									float alpha,
									TensorDescriptor srcDesc,
									CudaDeviceVariable<float> srcData,
									float beta,
									TensorDescriptor destDesc,
									CudaDeviceVariable<float> destData
									)
		{
			res = CudaDNNNativeMethods.cudnnPoolingForward(_handle, poolingDesc.Desc, ref alpha, srcDesc.Desc, srcData.DevicePointer, ref beta, destDesc.Desc, destData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnPoolingForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/// <summary>
		/// This function computes the gradient of a pooling operation.
		/// </summary>
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
		public void PoolingBackward(PoolingDescriptor poolingDesc,
									float alpha,
									TensorDescriptor srcDesc,
									CudaDeviceVariable<float> srcData,
									TensorDescriptor srcDiffDesc,
									CudaDeviceVariable<float> srcDiffData,
									TensorDescriptor destDesc,
									CudaDeviceVariable<float> destData,
									float beta,
									TensorDescriptor destDiffDesc,
									CudaDeviceVariable<float> destDiffData
									)
		{
			res = CudaDNNNativeMethods.cudnnPoolingBackward(_handle, poolingDesc.Desc, ref alpha, srcDesc.Desc, srcData.DevicePointer, srcDiffDesc.Desc, srcDiffData.DevicePointer, destDesc.Desc, destData.DevicePointer, ref beta, destDiffDesc.Desc, destDiffData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnPoolingBackward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


		/* Activation functions: All of the form "output = alpha * Op(inputs) + beta * output" */

		/// <summary>
		/// This routine applies a specified neuron activation function element-wise over each input value.
		/// </summary>
		/// <param name="mode">Enumerant to specify the activation mode.</param>
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
		public void ActivationForward(cudnnActivationMode mode,
										float alpha,
										TensorDescriptor srcDesc,
										CudaDeviceVariable<float> srcData,
										float beta,
										TensorDescriptor destDesc,
										CudaDeviceVariable<float> destData
									)
		{
			res = CudaDNNNativeMethods.cudnnActivationForward(_handle, mode, ref alpha, srcDesc.Desc, srcData.DevicePointer, ref beta, destDesc.Desc, destData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnActivationForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/// <summary>
		/// This routine computes the gradient of a neuron activation function.
		/// </summary>
		/// <param name="mode">Enumerant to specify the activation mode.</param>
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
		public void ActivationBackward(cudnnActivationMode mode,
										float alpha,
										TensorDescriptor srcDesc,
										CudaDeviceVariable<float> srcData,
										TensorDescriptor srcDiffDesc,
										CudaDeviceVariable<float> srcDiffData,
										TensorDescriptor destDesc,
										CudaDeviceVariable<float> destData,
										float beta,
										TensorDescriptor destDiffDesc,
										CudaDeviceVariable<float> destDiffData
										)
		{
			res = CudaDNNNativeMethods.cudnnActivationBackward(_handle, mode, ref alpha, srcDesc.Desc, srcData.DevicePointer, srcDiffDesc.Desc, srcDiffData.DevicePointer, destDesc.Desc, destData.DevicePointer, ref beta, destDiffDesc.Desc, destDiffData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnActivationForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		#endregion

		#region doubles
		/// <summary>
		/// This function copies the scaled data from one tensor to another tensor with a different
		/// layout. Those descriptors need to have the same dimensions but not necessarily the
		/// same strides. The input and output tensors must not overlap in any way (i.e., tensors
		/// cannot be transformed in place). This function can be used to convert a tensor with an
		/// unsupported format to a supported one.
		/// </summary>
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
		public void TransformTensor(double alpha,
											TensorDescriptor srcDesc,
											CudaDeviceVariable<double> srcData,
											double beta,
											TensorDescriptor destDesc,
											CudaDeviceVariable<double> destData
										)
		{
			res = CudaDNNNativeMethods.cudnnTransformTensor(_handle, ref alpha, srcDesc.Desc, srcData.DevicePointer, ref beta, destDesc.Desc, destData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnTransformTensor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}



		/// <summary>
		/// This function adds the scaled values of one bias tensor to another tensor. Each dimension
		/// of the bias tensor must match the coresponding dimension of the srcDest tensor or
		/// must be equal to 1. In the latter case, the same value from the bias tensor for thoses
		/// dimensions will be used to blend into the srcDest tensor.
		/// </summary>
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
		public void AddTensor(double alpha,
									TensorDescriptor biasDesc,
									CudaDeviceVariable<double> biasData,
									double beta,
									TensorDescriptor srcDestDesc,
									CudaDeviceVariable<double> srcDestData
									)
		{
			res = CudaDNNNativeMethods.cudnnAddTensor(_handle, ref alpha, biasDesc.Desc, biasData.DevicePointer, ref beta, srcDestDesc.Desc, srcDestData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnAddTensor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/// <summary>
		/// This function sets all the elements of a tensor to a given value
		/// </summary>
		/// <param name="srcDestDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="srcDestData">Pointer to data of the tensor described by the srcDestDesc descriptor.</param>
		/// <param name="value">Pointer in Host memory to a value that all elements of the tensor will be set to.</param>
		public void SetTensor(TensorDescriptor srcDestDesc,
									CudaDeviceVariable<double> srcDestData,
									CudaDeviceVariable<double> value
									)
		{
			res = CudaDNNNativeMethods.cudnnSetTensor(_handle, srcDestDesc.Desc, srcDestData.DevicePointer, value.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetTensor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/// <summary>
		/// This function scale all the elements of a tensor by a give factor.
		/// </summary>
		/// <param name="srcDestDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="srcDestData">Pointer to data of the tensor described by the srcDestDesc descriptor.</param>
		/// <param name="alpha">Pointer in Host memory to a value that all elements of the tensor will be scaled with.</param>
		public void ScaleTensor(TensorDescriptor srcDestDesc,
										CudaDeviceVariable<double> srcDestData,
										double alpha
									)
		{
			res = CudaDNNNativeMethods.cudnnScaleTensor(_handle, srcDestDesc.Desc, srcDestData.DevicePointer, ref alpha);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnScaleTensor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		

		/* Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */


		/// <summary>
		/// This function executes convolutions or cross-correlations over src using the specified
		/// filters, returning results in dest. Scaling factors alpha and beta can be used to scale
		/// the input tensor and the output tensor respectively.
		/// </summary>
		/// <param name="alpha">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="srcDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="srcData">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="filterData">Data pointer to GPU memory associated with the filter descriptor filterDesc.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="algo">Enumerant that specifies which convolution algorithm shoud be used to compute the results
		/// the specified algorithm. If no workspace is needed for a particular
		/// algorithm, that pointer can be nil</param>
		/// <param name="workSpace">Data pointer to GPU memory to a workspace needed to able to execute
		/// the specified algorithm. If no workspace is needed for a particular
		/// algorithm, that pointer can be nil</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="destDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="destData">Data pointer to GPU memory associated with the tensor descriptor
		/// destDesc that carries the result of the convolution.</param>
		public void ConvolutionForward(double alpha,
										TensorDescriptor srcDesc,
										CudaDeviceVariable<double> srcData,
										FilterDescriptor filterDesc,
										CudaDeviceVariable<double> filterData,
										ConvolutionDescriptor convDesc,
										cudnnConvolutionFwdAlgo algo,
										CudaDeviceVariable<byte> workSpace,
										double beta,
										TensorDescriptor destDesc,
										CudaDeviceVariable<double> destData
									)
		{
			res = CudaDNNNativeMethods.cudnnConvolutionForward(_handle, ref alpha, srcDesc.Desc, srcData.DevicePointer, filterDesc.Desc, filterData.DevicePointer, convDesc.Desc, algo, workSpace.DevicePointer, workSpace.SizeInBytes, ref beta, destDesc.Desc, destData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnConvolutionForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


		/// <summary>
		/// This function computes the convolution gradient with respect to the bias, which is the
		/// sum of every element belonging to the same feature map across all of the images of the
		/// input tensor. Therefore, the number of elements produced is equal to the number of
		/// features maps of the input tensor.
		/// </summary>
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
		public void ConvolutionBackwardBias(double alpha,
											TensorDescriptor srcDesc,
											CudaDeviceVariable<double> srcData,
											double beta,
											TensorDescriptor destDesc,
											CudaDeviceVariable<double> destData
									)
		{
			res = CudaDNNNativeMethods.cudnnConvolutionBackwardBias(_handle, ref alpha, srcDesc.Desc, srcData.DevicePointer, ref beta, destDesc.Desc, destData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnConvolutionBackwardBias", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}



		/// <summary>
		/// This function computes the convolution gradient with respect to filter coefficients using
		/// the specified algo, returning results in gradDesc.Scaling factors alpha and beta can be
		/// used to scale the input tensor and the output tensor respectively.
		/// </summary>
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
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="gradDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="gradData">Data pointer to GPU memory associated with the filter descriptor
		/// gradDesc that carries the result.</param> 
		public void ConvolutionBackwardFilter(double alpha,
												TensorDescriptor srcDesc,
												CudaDeviceVariable<double> srcData,
												TensorDescriptor diffDesc,
												CudaDeviceVariable<double> diffData,
												ConvolutionDescriptor convDesc,
												cudnnConvolutionBwdFilterAlgo algo,
												CudaDeviceVariable<byte> workSpace,
												double beta,
												FilterDescriptor gradDesc,
												CudaDeviceVariable<double> gradData
											)
		{
			res = CudaDNNNativeMethods.cudnnConvolutionBackwardFilter(_handle, ref alpha, srcDesc.Desc, srcData.DevicePointer, diffDesc.Desc, diffData.DevicePointer, convDesc.Desc, algo, workSpace.DevicePointer, workSpace.SizeInBytes, ref beta, gradDesc.Desc, gradData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnConvolutionBackwardFilter", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


		/// <summary>
		/// This function computes the convolution gradient with respect to the output tensor using
		/// the specified algo, returning results in gradDesc. Scaling factors alpha and beta can
		/// be used to scale the input tensor and the output tensor respectively.
		/// </summary>
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
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="gradDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="gradData">Data pointer to GPU memory associated with the output tensor descriptor
		/// gradDesc that carries the result.</param>
		public void ConvolutionBackwardData(double alpha,
											FilterDescriptor filterDesc,
											CudaDeviceVariable<double> filterData,
											TensorDescriptor diffDesc,
											CudaDeviceVariable<double> diffData,
											ConvolutionDescriptor convDesc,
											cudnnConvolutionBwdDataAlgo algo,
											CudaDeviceVariable<byte> workSpace,
											double beta,
											TensorDescriptor gradDesc,
											CudaDeviceVariable<double> gradData
										)
		{
			res = CudaDNNNativeMethods.cudnnConvolutionBackwardData(_handle, ref alpha, filterDesc.Desc, filterData.DevicePointer, diffDesc.Desc, diffData.DevicePointer, convDesc.Desc, algo, workSpace.DevicePointer, workSpace.SizeInBytes, ref beta, gradDesc.Desc, gradData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnConvolutionBackwardData", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="srcDesc"></param>
		/// <param name="srcData"></param>
		/// <param name="filterDesc"></param>
		/// <param name="convDesc"></param>
		/// <param name="colBuffer"></param>
		public void Im2Col(
							TensorDescriptor srcDesc,
							CudaDeviceVariable<double> srcData,
							FilterDescriptor filterDesc,
							ConvolutionDescriptor convDesc,
							CudaDeviceVariable<byte> colBuffer
							)
		{
			res = CudaDNNNativeMethods.cudnnIm2Col(_handle, srcDesc.Desc, srcData.DevicePointer, filterDesc.Desc, convDesc.Desc, colBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnIm2Col", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}




		/* Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" */

		/// <summary>
		/// This routine computes the softmax function.
		/// </summary>
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
		public void SoftmaxForward(cudnnSoftmaxAlgorithm algorithm,
									cudnnSoftmaxMode mode,
									double alpha,
									TensorDescriptor srcDesc,
									CudaDeviceVariable<double> srcData,
									double beta,
									TensorDescriptor destDesc,
									CudaDeviceVariable<double> destData
									)
		{
			res = CudaDNNNativeMethods.cudnnSoftmaxForward(_handle, algorithm, mode, ref alpha, srcDesc.Desc, srcData.DevicePointer, ref beta, destDesc.Desc, destData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSoftmaxForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/// <summary>
		/// This routine computes the gradient of the softmax function.
		/// </summary>
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
		public void SoftmaxBackward(cudnnSoftmaxAlgorithm algorithm,
									cudnnSoftmaxMode mode,
									double alpha,
									TensorDescriptor srcDesc,
									CudaDeviceVariable<double> srcData,
									TensorDescriptor srcDiffDesc,
									CudaDeviceVariable<double> srcDiffData,
									double beta,
									TensorDescriptor destDiffDesc,
									CudaDeviceVariable<double> destDiffData
									)
		{
			res = CudaDNNNativeMethods.cudnnSoftmaxBackward(_handle, algorithm, mode, ref alpha, srcDesc.Desc, srcData.DevicePointer, srcDiffDesc.Desc, srcDiffData.DevicePointer, ref beta, destDiffDesc.Desc, destDiffData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSoftmaxBackward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}







		/// <summary>
		/// This function computes pooling of input values (i.e., the maximum or average of several
		/// adjacent values) to produce an output with smaller height and/or width.
		/// </summary>
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
		public void PoolingForward(PoolingDescriptor poolingDesc,
									double alpha,
									TensorDescriptor srcDesc,
									CudaDeviceVariable<double> srcData,
									double beta,
									TensorDescriptor destDesc,
									CudaDeviceVariable<double> destData
									)
		{
			res = CudaDNNNativeMethods.cudnnPoolingForward(_handle, poolingDesc.Desc, ref alpha, srcDesc.Desc, srcData.DevicePointer, ref beta, destDesc.Desc, destData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnPoolingForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


		/// <summary>
		/// This function computes the gradient of a pooling operation.
		/// </summary>
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
		public void PoolingBackward(PoolingDescriptor poolingDesc,
									double alpha,
									TensorDescriptor srcDesc,
									CudaDeviceVariable<double> srcData,
									TensorDescriptor srcDiffDesc,
									CudaDeviceVariable<double> srcDiffData,
									TensorDescriptor destDesc,
									CudaDeviceVariable<double> destData,
									double beta,
									TensorDescriptor destDiffDesc,
									CudaDeviceVariable<double> destDiffData
									)
		{
			res = CudaDNNNativeMethods.cudnnPoolingBackward(_handle, poolingDesc.Desc, ref alpha, srcDesc.Desc, srcData.DevicePointer, srcDiffDesc.Desc, srcDiffData.DevicePointer, destDesc.Desc, destData.DevicePointer, ref beta, destDiffDesc.Desc, destDiffData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnPoolingBackward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


		/* Activation functions: All of the form "output = alpha * Op(inputs) + beta * output" */

		/// <summary>
		/// This routine applies a specified neuron activation function element-wise over each input value.
		/// </summary>
		/// <param name="mode">Enumerant to specify the activation mode.</param>
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
		public void ActivationForward(cudnnActivationMode mode,
										double alpha,
										TensorDescriptor srcDesc,
										CudaDeviceVariable<double> srcData,
										double beta,
										TensorDescriptor destDesc,
										CudaDeviceVariable<double> destData
									)
		{
			res = CudaDNNNativeMethods.cudnnActivationForward(_handle, mode, ref alpha, srcDesc.Desc, srcData.DevicePointer, ref beta, destDesc.Desc, destData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnActivationForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/// <summary>
		/// This routine computes the gradient of a neuron activation function.
		/// </summary>
		/// <param name="mode">Enumerant to specify the activation mode.</param>
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
		public void ActivationBackward(cudnnActivationMode mode,
										double alpha,
										TensorDescriptor srcDesc,
										CudaDeviceVariable<double> srcData,
										TensorDescriptor srcDiffDesc,
										CudaDeviceVariable<double> srcDiffData,
										TensorDescriptor destDesc,
										CudaDeviceVariable<double> destData,
										double beta,
										TensorDescriptor destDiffDesc,
										CudaDeviceVariable<double> destDiffData
										)
		{
			res = CudaDNNNativeMethods.cudnnActivationBackward(_handle, mode, ref alpha, srcDesc.Desc, srcData.DevicePointer, srcDiffDesc.Desc, srcDiffData.DevicePointer, destDesc.Desc, destData.DevicePointer, ref beta, destDiffDesc.Desc, destDiffData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnActivationForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		#endregion
	}
}
