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

namespace ManagedCuda.CudaDNNv4
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
									float value
									)
		{
			res = CudaDNNNativeMethods.cudnnSetTensor(_handle, srcDestDesc.Desc, srcDestData.DevicePointer, ref value);
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
        public void ActivationForward( ActivationDescriptor activationDesc,
                                        float alpha,
										TensorDescriptor srcDesc,
										CudaDeviceVariable<float> srcData,
										float beta,
										TensorDescriptor destDesc,
										CudaDeviceVariable<float> destData
									)
		{
			res = CudaDNNNativeMethods.cudnnActivationForward(_handle, activationDesc.Desc, ref alpha, srcDesc.Desc, srcData.DevicePointer, ref beta, destDesc.Desc, destData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnActivationForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

        /// <summary>
        /// This routine computes the gradient of a neuron activation function.
        /// </summary>
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
        public void ActivationBackward(ActivationDescriptor activationDesc,
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
			res = CudaDNNNativeMethods.cudnnActivationBackward(_handle, activationDesc.Desc, ref alpha, srcDesc.Desc, srcData.DevicePointer, srcDiffDesc.Desc, srcDiffData.DevicePointer, destDesc.Desc, destData.DevicePointer, ref beta, destDiffDesc.Desc, destDiffData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnActivationForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// Derives a tensor descriptor from layer data descriptor for BatchNormalization 
        /// scale, invVariance, bnBias, bnScale tensors.Use this tensor desc for 
        /// bnScaleBiasMeanVarDesc and bnScaleBiasDiffDesc in Batch Normalization forward and backward functions.
        /// </summary>
        public void DeriveBNTensorDescriptor(
                                        TensorDescriptor derivedBnDesc,
                                        TensorDescriptor xDesc,
                                        cudnnBatchNormMode mode)
        {
            res = CudaDNNNativeMethods.cudnnDeriveBNTensorDescriptor(derivedBnDesc.Desc, xDesc.Desc, mode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDeriveBNTensorDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }


        /// <summary>
        /// This function performs the forward BatchNormalization layer computation for the training phase. 
        /// This layer is based on the paper "Batch Normalization: Accelerating Deep Network Training by 
        /// Reducing Internal Covariate Shift", S. Ioffe, C. Szegedy, 2015.
        /// </summary>
        /// <param name="mode"> Mode of operation (spatial or per-activation). </param>
        /// <param name="alpha"> Pointer to scaling factors (in host memory) used to blend the layer output value with prior value in the destination tensor as follows: dstValue = alpha[0]*resultValue + beta[0]*priorDstValue. </param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the layer output value with prior value in the destination tensor as follows: dstValue = alpha[0]*resultValue + beta[0]*priorDstValue. </param>
        /// <param name="xDesc">Tensor descriptor layer's x data.</param>
        /// <param name="x">Pointer in device memory for the layer's x data.</param>
        /// <param name="yDesc">Tensor descriptor the layer's y data.</param>
        /// <param name="y">Pointer in device memory for the layer's y data.</param>
        /// <param name="bnScaleBiasMeanVarDesc">Shared tensor descriptor desc for all the 6 tensors below in the argument list. The dimensions for this tensor descriptor are dependent on the normalization mode.</param>
        /// <param name="bnScale">Pointer in device memory for the batch normalization scale parameters (in original paper scale is referred to as gamma).</param>
        /// <param name="bnBias">Pointers in device memory for the batch normalization bias parameters (in original paper bias is referred to as beta). Note that bnBias parameter can replace the previous layer's bias parameter for improved efficiency. </param>
        /// <param name="exponentialAverageFactor">Factor used in the moving average computation runningMean = newMean*factor + runningMean*(1-factor). Use a factor=1/(1+n) at Nth call to the function to get Cumulative Moving Average (CMA) behavior CMA[n] = (x[1]+...+x[n])/n. Since CMA[n+1] = (n*CMA[n]+x[n+1])/(n+1)= ((n+1)*CMA[n]-CMA[n])/(n+1) + x[n+1]/(n+1) = CMA[n]*(1-1/(n+1))+x[n +1]*1/(n+1)</param>
        /// <param name="resultRunningMean">Running mean tensor (it has the same descriptor as the bias and scale). If this tensor is initially uninitialized, it is required that exponentialAverageFactor=1 is used for the very first call of a complete training cycle. This is necessary to properly initialize the moving average. Both resultRunningMean and resultRunningInvVariance can be NULL but only at the same time.</param>
        /// <param name="resultRunningInvVariance">Running variance tensor (it has the same descriptor as the bias and scale). If this tensors is initially uninitialized, it is required that exponentialAverageFactor=1 is used for the very first call of a complete training cycle. This is necessary to properly initialize the moving average. Both resultRunningMean and resultRunningInvVariance can be NULL but only at the same time. The value stored in resultRunningInvVariance (or passed as an input in inference mode) is the moving average of the expression 1 / sqrt(eps+variance[x]) where variance is computed either over batch or spatial+batch dimensions depending on the mode. </param>
        /// <param name="epsilon">Epsilon value used in the batch normalization formula. Minimum allowed value is currently 1e-5. Same epsilon value should be used in forward and backward functions.</param>
        /// <param name="resultSaveMean">Optional cache to save intermediate results computed during the forward pass - these can then be reused to speed up the backward pass. For this to work correctly, the bottom layer data has to remain unchanged until the backward function is called. Note that both resultSaveMean and resultSaveInvVariance can be NULL but only at the same time. It is recommended to use this cache since memory overhead is relatively small because these tensors have a much lower product of dimensions than the data tensors.</param>
        /// <param name="resultSaveInvVariance">Optional cache to save intermediate results computed during the forward pass - these can then be reused to speed up the backward pass. For this to work correctly, the bottom layer data has to remain unchanged until the backward function is called. Note that both resultSaveMean and resultSaveInvVariance can be NULL but only at the same time. It is recommended to use this cache since memory overhead is relatively small because these tensors have a much lower product of dimensions than the data tensors.</param>
        public void BatchNormalizationForwardTraining(
                                cudnnBatchNormMode mode,

                                float alpha, // alpha[0] = result blend factor
                                float beta,  // beta[0] = dest layer blend factor

                                TensorDescriptor xDesc,
                                CudaDeviceVariable<float> x,     // NxCxHxW
                                TensorDescriptor yDesc,
                                CudaDeviceVariable<float> y,     // NxCxHxW

                                /* Shared desc for the next 6 tensors in the argument list.
                                   Data type to be set as follows:
                                   type = (typeOf(x) == double) ? double : float
                                   Dimensions for this descriptor depend on normalization mode
                                   - Spatial Normalization : tensors are expected to have dims 1xCx1x1
                                    (normalization is performed across NxHxW)
                                   - Per-Activation Normalization : tensors are expected to have dims of 1xCxHxW 
                                    (normalization is performed across N) */
                                TensorDescriptor bnScaleBiasMeanVarDesc,

                                // 'Gamma' and 'Beta' respectively in Ioffe and Szegedy's paper's notation
                                CudaDeviceVariable<float> bnScale,
                                CudaDeviceVariable<float> bnBias,

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
                                CudaDeviceVariable<float> resultRunningMean,
                                /* Output in training mode, input in inference. Is the moving average
                                   of 1 / sqrt( epsilon + variance[x] ) */
                                CudaDeviceVariable<float> resultRunningInvVariance,

                                /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
                                double epsilon,

                                /* Optionally save intermediate results from the forward pass here
                                   - can be reused to speed up backward pass. NULL if unused */
                                CudaDeviceVariable<float> resultSaveMean,
                                CudaDeviceVariable<float> resultSaveInvVariance)
        {
            res = CudaDNNNativeMethods.cudnnBatchNormalizationForwardTraining(
                _handle, mode, ref alpha, ref beta, xDesc.Desc, x.DevicePointer, yDesc.Desc, y.DevicePointer,
                bnScaleBiasMeanVarDesc.Desc, bnScale.DevicePointer, bnBias.DevicePointer, exponentialAverageFactor,
                resultRunningMean.DevicePointer, resultRunningInvVariance.DevicePointer, epsilon, resultSaveMean.DevicePointer, resultSaveInvVariance.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "BatchNormalizationForwardTraining", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function performs the forward BatchNormalization layer computation for the inference phase. 
        /// This layer is based on the paper "Batch Normalization: Accelerating Deep Network 
        /// Training by Reducing Internal Covariate Shift", S. Ioffe, C. Szegedy, 2015.
        /// </summary>
        /// <param name="mode"> Mode of operation (spatial or per-activation). </param>
        /// <param name="alpha"> Pointer to scaling factors (in host memory) used to blend the layer output value with prior value in the destination tensor as follows: dstValue = alpha[0]*resultValue + beta[0]*priorDstValue. </param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the layer output value with prior value in the destination tensor as follows: dstValue = alpha[0]*resultValue + beta[0]*priorDstValue. </param>
        /// <param name="xDesc">Tensor descriptor layer's x data.</param>
        /// <param name="x">Pointer in device memory for the layer's x data.</param>
        /// <param name="yDesc">Tensor descriptor the layer's y data.</param>
        /// <param name="y">Pointer in device memory for the layer's y data.</param>
        /// <param name="bnScaleBiasMeanVarDesc">Shared tensor descriptor desc for all the 4 tensors below in the argument list. The dimensions for this tensor descriptor are dependent on the normalization mode.</param>
        /// <param name="bnScale">Pointer in device memory for the batch normalization scale parameters (in original paper scale is referred to as gamma).</param>
        /// <param name="bnBias">Pointers in device memory for the batch normalization bias parameters (in original paper bias is referred to as beta). Note that bnBias parameter can replace the previous layer's bias parameter for improved efficiency. </param>
        /// <param name="estimatedMean">Mean tensor (has the same descriptor as the bias and scale). It is suggested that resultRunningMean from the cudnnBatchNormalizationForwardTraining call accumulated during the training phase be passed as input here.</param>
        /// <param name="estimatedInvVariance">Variance tensor (has the same descriptor as the bias and scale). It is suggested that resultRunningVariance from the cudnnBatchNormalizationForwardTraining call accumulated during the training phase be passed as input here.</param>
        /// <param name="epsilon">Epsilon value used in the batch normalization formula. Minimum allowed value is currently 1e-5. Same epsilon value should be used in forward and backward functions.</param>
        public void BatchNormalizationForwardInference(
                                        cudnnBatchNormMode mode,
                                        float alpha, // alpha[0] = result blend factor
                                        float beta,  // beta[0] = dest layer blend factor
                                        TensorDescriptor xDesc,
                                        CudaDeviceVariable<float> x,     // NxCxHxW
                                        TensorDescriptor yDesc,
                                        CudaDeviceVariable<float> y,     // NxCxHxW
                                        TensorDescriptor bnScaleBiasMeanVarDesc,
                                        CudaDeviceVariable<float> bnScale,
                                        CudaDeviceVariable<float> bnBias,
                                        CudaDeviceVariable<float> estimatedMean,
                                        CudaDeviceVariable<float> estimatedInvVariance,
                                        double epsilon)
        {
            res = CudaDNNNativeMethods.cudnnBatchNormalizationForwardInference(
                _handle, mode, ref alpha, ref beta, xDesc.Desc, x.DevicePointer, yDesc.Desc, y.DevicePointer,
                bnScaleBiasMeanVarDesc.Desc, bnScale.DevicePointer, bnBias.DevicePointer, estimatedMean.DevicePointer, estimatedInvVariance.DevicePointer, epsilon);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnBatchNormalizationForwardInference", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function performs the backward BatchNormalization layer computation.
        /// </summary>
        /// <param name="mode"> Mode of operation (spatial or per-activation). </param>
        /// <param name="alphaDataDiff">Pointer to scaling factors in host memory used to blend the gradient output dx with a prior value in the destination tensor as follows: dstValue = alpha[0]*resultValue + beta[0]*priorDstValue.</param>
        /// <param name="betaDataDiff">Pointer to scaling factors in host memory used to blend the gradient output dx with a prior value in the destination tensor as follows: dstValue = alpha[0]*resultValue + beta[0]*priorDstValue.</param>
        /// <param name="alphaParamDiff">Pointer to scaling factors (in host memory) used to blend the gradient outputs dBnScaleResult and dBnBiasResult with prior values in the destination tensor as follows: dstValue = alpha[0]*resultValue + beta[0]*priorDstValue.</param>
        /// <param name="betaParamDiff">Pointer to scaling factors (in host memory) used to blend the gradient outputs dBnScaleResult and dBnBiasResult with prior values in the destination tensor as follows: dstValue = alpha[0]*resultValue + beta[0]*priorDstValue.</param>
        /// <param name="xDesc">Tensor descriptor for the layer's x data.</param>
        /// <param name="x">Pointers in device memory for the layer's x data.</param>
        /// <param name="dyDesc">Tensor descriptor for the layer's backpropagated differential dy (inputs).</param>
        /// <param name="dy">Pointers in device memory for the layer's backpropagated differential dy (inputs).</param>
        /// <param name="dxDesc">Tensor descriptor for the layer's resulting differential with respect to x, dx (output).</param>
        /// <param name="dx">Pointer in device memory for the layer's resulting differential with respect to x, dx (output).</param>
        /// <param name="dBnScaleBiasDesc">Shared tensor descriptor for all the 5 tensors below in the argument list (bnScale, resultBnScaleDiff, resultBnBiasDiff, savedMean, savedInvVariance). The dimensions for this tensor descriptor are dependent on normalization mode. Note: The data type of this tensor descriptor must be 'float' for FP16 and FP32 input tensors, and 'double' for FP64 input tensors.</param>
        /// <param name="bnScale">Pointers in device memory for the batch normalization scale parameter (in original paper bias is referred to as gamma). Note that bnBias parameter is not needed for this layer's computation.</param>
        /// <param name="dBnScaleResult">Pointer in device memory for the resulting scale differentials computed by this routine. Note that scale and bias gradients are not backpropagated below this layer (since they are dead-end computation DAG nodes).</param>
        /// <param name="dBnBiasResult">Pointer in device memory for the resulting bias differentials computed by this routine. Note that scale and bias gradients are not backpropagated below this layer (since they are dead-end computation DAG nodes).</param>
        /// <param name="epsilon">Epsilon value used in the batch normalization formula. Minimum allowed value is currently 1e-5. Same epsilon value should be used in forward and backward functions.</param>
        /// <param name="savedMean">Optional cache parameter saved intermediate results computed during the forward pass. For this to work correctly, the layer's x and bnScale, bnBias data has to remain unchanged until the backward function is called. Note that both savedMean and savedInvVariance parameters can be NULL but only at the same time. It is recommended to use this cache since the memory overhead is relatively small.</param>
        /// <param name="savedInvVariance">Optional cache parameter saved intermediate results computed during the forward pass. For this to work correctly, the layer's x and bnScale, bnBias data has to remain unchanged until the backward function is called. Note that both savedMean and savedInvVariance parameters can be NULL but only at the same time. It is recommended to use this cache since the memory overhead is relatively small.</param>
        public void BatchNormalizationBackward(
                                        cudnnBatchNormMode mode,
                                        float alphaDataDiff,
                                        float betaDataDiff,
                                        float alphaParamDiff,
                                        float betaParamDiff,
                                        TensorDescriptor xDesc, // same desc for x, dx, dy
                                        CudaDeviceVariable<float> x,
                                        TensorDescriptor dyDesc,
                                        CudaDeviceVariable<float> dy,
                                        TensorDescriptor dxDesc,
                                        CudaDeviceVariable<float> dx,
                                        /* Shared tensor desc for the 5 tensors below */
                                        TensorDescriptor dBnScaleBiasDesc,
                                        CudaDeviceVariable<float> bnScale, // bnBias doesn't affect backpropagation
                                                                           /* scale and bias diff are not backpropagated below this layer */
                                        CudaDeviceVariable<float> dBnScaleResult,
                                        CudaDeviceVariable<float> dBnBiasResult,
                                        /* Same epsilon as forward pass */
                                        double epsilon,

                                        /* Optionally cached intermediate results from
                                           forward pass */
                                        CudaDeviceVariable<float> savedMean,
                                        CudaDeviceVariable<float> savedInvVariance)
        {
            res = CudaDNNNativeMethods.cudnnBatchNormalizationBackward(
                _handle, mode, ref alphaDataDiff, ref betaDataDiff, ref alphaParamDiff, ref betaParamDiff, 
                xDesc.Desc, x.DevicePointer, dyDesc.Desc, dy.DevicePointer, dxDesc.Desc, dx.DevicePointer,
                dBnScaleBiasDesc.Desc, bnScale.DevicePointer, dBnScaleResult.DevicePointer, dBnBiasResult.DevicePointer,
                epsilon, savedMean.DevicePointer, savedInvVariance.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "BatchNormalizationBackward", res));
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
                                    double value
                                    )
        {
            res = CudaDNNNativeMethods.cudnnSetTensor(_handle, srcDestDesc.Desc, srcDestData.DevicePointer, ref value);
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
        public void ActivationForward(cudnnActivationDescriptor activationDesc,
                                        double alpha,
										TensorDescriptor srcDesc,
										CudaDeviceVariable<double> srcData,
										double beta,
										TensorDescriptor destDesc,
										CudaDeviceVariable<double> destData
									)
		{
			res = CudaDNNNativeMethods.cudnnActivationForward(_handle, activationDesc, ref alpha, srcDesc.Desc, srcData.DevicePointer, ref beta, destDesc.Desc, destData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnActivationForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

        /// <summary>
        /// This routine computes the gradient of a neuron activation function.
        /// </summary>
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
        public void ActivationBackward(ActivationDescriptor activationDesc,
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
			res = CudaDNNNativeMethods.cudnnActivationBackward(_handle, activationDesc.Desc, ref alpha, srcDesc.Desc, srcData.DevicePointer, srcDiffDesc.Desc, srcDiffData.DevicePointer, destDesc.Desc, destData.DevicePointer, ref beta, destDiffDesc.Desc, destDiffData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnActivationForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		#endregion

		#region Type independent

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
		/// <returns>An array to store performance metrics sorted ascending by compute time.</returns>
		public cudnnConvolutionFwdAlgoPerf[] FindConvolutionForwardAlgorithm(TensorDescriptor srcDesc,
													FilterDescriptor filterDesc,
													ConvolutionDescriptor convDesc,
													TensorDescriptor destDesc,
													int requestedAlgoCount
												)
		{
			cudnnConvolutionFwdAlgoPerf[] temp = new cudnnConvolutionFwdAlgoPerf[requestedAlgoCount];
			int returnedAlgoCount = 0;
			res = CudaDNNNativeMethods.cudnnFindConvolutionForwardAlgorithm(_handle, srcDesc.Desc, filterDesc.Desc, convDesc.Desc, destDesc.Desc, requestedAlgoCount, ref returnedAlgoCount, temp);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnFindConvolutionForwardAlgorithm", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
			if (returnedAlgoCount <= 0) return null;

			cudnnConvolutionFwdAlgoPerf[] perfResults = new cudnnConvolutionFwdAlgoPerf[returnedAlgoCount];
			Array.Copy(temp, perfResults, returnedAlgoCount);
			return perfResults;
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
		/// <returns>Enumerant that specifies which convolution algorithm should be used to
		/// compute the results according to the specified preference</returns>
		public cudnnConvolutionFwdAlgo GetConvolutionForwardAlgorithm(TensorDescriptor srcDesc,
													FilterDescriptor filterDesc,
													ConvolutionDescriptor convDesc,
													TensorDescriptor destDesc,
													cudnnConvolutionFwdPreference preference,
													SizeT memoryLimitInbytes
													)
		{
			cudnnConvolutionFwdAlgo algo = new cudnnConvolutionFwdAlgo();
			res = CudaDNNNativeMethods.cudnnGetConvolutionForwardAlgorithm(_handle, srcDesc.Desc, filterDesc.Desc, convDesc.Desc, destDesc.Desc, preference, memoryLimitInbytes, ref algo);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionForwardAlgorithm", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
			return algo;
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
		/// This function attempts all cuDNN algorithms for cudnnConvolutionBackwardFilter_v3 and outputs performance metrics to a user-
		/// allocated array of cudnnConvolutionBwdFilterAlgoPerf_t. These metrics are
		/// written in sorted fashion where the first element has the lowest compute time. 
		/// </summary>
		/// <param name="srcDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="diffDesc">Handle to the previously initialized input differential tensor descriptor.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="gradDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="requestedAlgoCount">The maximum number of elements to be stored in perfResults.</param>
		/// <returns>An array to store performance metrics sorted ascending by compute time.</returns>
		public cudnnConvolutionBwdFilterAlgoPerf[] FindConvolutionBackwardFilterAlgorithm(TensorDescriptor srcDesc,
															TensorDescriptor diffDesc,
															ConvolutionDescriptor convDesc,
															FilterDescriptor gradDesc,
															int requestedAlgoCount
															)
		{
			cudnnConvolutionBwdFilterAlgoPerf[] temp = new cudnnConvolutionBwdFilterAlgoPerf[requestedAlgoCount];
			int returnedAlgoCount = 0;
			res = CudaDNNNativeMethods.cudnnFindConvolutionBackwardFilterAlgorithm(_handle, srcDesc.Desc, diffDesc.Desc, convDesc.Desc, gradDesc.Desc, requestedAlgoCount, ref returnedAlgoCount, temp);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnFindConvolutionBackwardFilterAlgorithm", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
			if (returnedAlgoCount <= 0) return null;

			cudnnConvolutionBwdFilterAlgoPerf[] perfResults = new cudnnConvolutionBwdFilterAlgoPerf[returnedAlgoCount];
			Array.Copy(temp, perfResults, returnedAlgoCount);
			return perfResults;
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
		/// <returns>Enumerant that specifies which convolution algorithm should be used to
		/// compute the results according to the specified preference</returns>
		public cudnnConvolutionBwdFilterAlgo GetConvolutionBackwardFilterAlgorithm(TensorDescriptor srcDesc,
															TensorDescriptor diffDesc,
															ConvolutionDescriptor convDesc,
															FilterDescriptor gradDesc,
															cudnnConvolutionBwdFilterPreference preference,
															SizeT memoryLimitInbytes
															)
		{
			cudnnConvolutionBwdFilterAlgo algo = new cudnnConvolutionBwdFilterAlgo();
			res = CudaDNNNativeMethods.cudnnGetConvolutionBackwardFilterAlgorithm(_handle, srcDesc.Desc, diffDesc.Desc, convDesc.Desc, gradDesc.Desc, preference, memoryLimitInbytes, ref algo);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionBackwardFilterAlgorithm", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
			return algo;
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
		/// <returns>Amount of GPU memory needed as workspace to be able to execute a
		/// forward convolution with the specified algo</returns>
		public SizeT GetConvolutionBackwardFilterWorkspaceSize(TensorDescriptor srcDesc,
																	TensorDescriptor diffDesc,
																	ConvolutionDescriptor convDesc,
																	FilterDescriptor gradDesc,
																	cudnnConvolutionBwdFilterAlgo algo
																)
		{
			SizeT sizeInBytes = new SizeT();
			res = CudaDNNNativeMethods.cudnnGetConvolutionBackwardFilterWorkspaceSize(_handle, srcDesc.Desc, diffDesc.Desc, convDesc.Desc, gradDesc.Desc, algo, ref sizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionBackwardFilterWorkspaceSize", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
			return sizeInBytes;
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
		/// <returns>An array to store performance metrics sorted ascending by compute time.</returns>
		public cudnnConvolutionBwdDataAlgoPerf[] FindConvolutionBackwardDataAlgorithm(FilterDescriptor filterDesc,
															TensorDescriptor diffDesc,
															ConvolutionDescriptor convDesc,
															TensorDescriptor gradDesc,
															int requestedAlgoCount
														)
		{
			cudnnConvolutionBwdDataAlgoPerf[] temp = new cudnnConvolutionBwdDataAlgoPerf[requestedAlgoCount];
			int returnedAlgoCount = 0;
			res = CudaDNNNativeMethods.cudnnFindConvolutionBackwardDataAlgorithm(_handle, filterDesc.Desc, diffDesc.Desc, convDesc.Desc, gradDesc.Desc, requestedAlgoCount, ref returnedAlgoCount, temp);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnFindConvolutionBackwardDataAlgorithm", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
			if (returnedAlgoCount <= 0) return null;

			cudnnConvolutionBwdDataAlgoPerf[] perfResults = new cudnnConvolutionBwdDataAlgoPerf[returnedAlgoCount];
			Array.Copy(temp, perfResults, returnedAlgoCount);
			return perfResults;
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
		/// <returns>Enumerant that specifies which convolution algorithm should be used to
		/// compute the results according to the specified preference</returns>
		public cudnnConvolutionBwdDataAlgo GetConvolutionBackwardDataAlgorithm(FilterDescriptor filterDesc,
														TensorDescriptor diffDesc,
														ConvolutionDescriptor convDesc,
														TensorDescriptor gradDesc,
														cudnnConvolutionBwdDataPreference preference,
														SizeT memoryLimitInbytes
														)
		{
			cudnnConvolutionBwdDataAlgo algo = new cudnnConvolutionBwdDataAlgo();
			res = CudaDNNNativeMethods.cudnnGetConvolutionBackwardDataAlgorithm(_handle, filterDesc.Desc, diffDesc.Desc, convDesc.Desc, gradDesc.Desc, preference, memoryLimitInbytes, ref algo);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionBackwardDataAlgorithm", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
			return algo;
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
		/// <returns>Amount of GPU memory needed as workspace to be able to execute a forward convolution with the specified algo</returns>
		public SizeT GetConvolutionBackwardDataWorkspaceSize(FilterDescriptor filterDesc,
															TensorDescriptor diffDesc,
															ConvolutionDescriptor convDesc,
															TensorDescriptor gradDesc,
															cudnnConvolutionBwdDataAlgo algo
														)
		{
			SizeT sizeInBytes = new SizeT();
			res = CudaDNNNativeMethods.cudnnGetConvolutionBackwardDataWorkspaceSize(_handle, filterDesc.Desc, diffDesc.Desc, convDesc.Desc, gradDesc.Desc, algo, ref sizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionBackwardDataWorkspaceSize", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
			return sizeInBytes;
		}

        /// <summary>
        /// This function performs the forward BatchNormalization layer computation for the training phase. 
        /// This layer is based on the paper "Batch Normalization: Accelerating Deep Network Training by 
        /// Reducing Internal Covariate Shift", S. Ioffe, C. Szegedy, 2015.
        /// </summary>
        /// <param name="mode"> Mode of operation (spatial or per-activation). </param>
        /// <param name="alpha"> Pointer to scaling factors (in host memory) used to blend the layer output value with prior value in the destination tensor as follows: dstValue = alpha[0]*resultValue + beta[0]*priorDstValue. </param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the layer output value with prior value in the destination tensor as follows: dstValue = alpha[0]*resultValue + beta[0]*priorDstValue. </param>
        /// <param name="xDesc">Tensor descriptor layer's x data.</param>
        /// <param name="x">Pointer in device memory for the layer's x data.</param>
        /// <param name="yDesc">Tensor descriptor the layer's y data.</param>
        /// <param name="y">Pointer in device memory for the layer's y data.</param>
        /// <param name="bnScaleBiasMeanVarDesc">Shared tensor descriptor desc for all the 6 tensors below in the argument list. The dimensions for this tensor descriptor are dependent on the normalization mode.</param>
        /// <param name="bnScale">Pointer in device memory for the batch normalization scale parameters (in original paper scale is referred to as gamma).</param>
        /// <param name="bnBias">Pointers in device memory for the batch normalization bias parameters (in original paper bias is referred to as beta). Note that bnBias parameter can replace the previous layer's bias parameter for improved efficiency. </param>
        /// <param name="exponentialAverageFactor">Factor used in the moving average computation runningMean = newMean*factor + runningMean*(1-factor). Use a factor=1/(1+n) at Nth call to the function to get Cumulative Moving Average (CMA) behavior CMA[n] = (x[1]+...+x[n])/n. Since CMA[n+1] = (n*CMA[n]+x[n+1])/(n+1)= ((n+1)*CMA[n]-CMA[n])/(n+1) + x[n+1]/(n+1) = CMA[n]*(1-1/(n+1))+x[n +1]*1/(n+1)</param>
        /// <param name="resultRunningMean">Running mean tensor (it has the same descriptor as the bias and scale). If this tensor is initially uninitialized, it is required that exponentialAverageFactor=1 is used for the very first call of a complete training cycle. This is necessary to properly initialize the moving average. Both resultRunningMean and resultRunningInvVariance can be NULL but only at the same time.</param>
        /// <param name="resultRunningInvVariance">Running variance tensor (it has the same descriptor as the bias and scale). If this tensors is initially uninitialized, it is required that exponentialAverageFactor=1 is used for the very first call of a complete training cycle. This is necessary to properly initialize the moving average. Both resultRunningMean and resultRunningInvVariance can be NULL but only at the same time. The value stored in resultRunningInvVariance (or passed as an input in inference mode) is the moving average of the expression 1 / sqrt(eps+variance[x]) where variance is computed either over batch or spatial+batch dimensions depending on the mode. </param>
        /// <param name="epsilon">Epsilon value used in the batch normalization formula. Minimum allowed value is currently 1e-5. Same epsilon value should be used in forward and backward functions.</param>
        /// <param name="resultSaveMean">Optional cache to save intermediate results computed during the forward pass - these can then be reused to speed up the backward pass. For this to work correctly, the bottom layer data has to remain unchanged until the backward function is called. Note that both resultSaveMean and resultSaveInvVariance can be NULL but only at the same time. It is recommended to use this cache since memory overhead is relatively small because these tensors have a much lower product of dimensions than the data tensors.</param>
        /// <param name="resultSaveInvVariance">Optional cache to save intermediate results computed during the forward pass - these can then be reused to speed up the backward pass. For this to work correctly, the bottom layer data has to remain unchanged until the backward function is called. Note that both resultSaveMean and resultSaveInvVariance can be NULL but only at the same time. It is recommended to use this cache since memory overhead is relatively small because these tensors have a much lower product of dimensions than the data tensors.</param>
        public void BatchNormalizationForwardTraining(
                                cudnnBatchNormMode mode,

                                double alpha, // alpha[0] = result blend factor
                                double beta,  // beta[0] = dest layer blend factor

                                TensorDescriptor xDesc,
                                CudaDeviceVariable<double> x,     // NxCxHxW
                                TensorDescriptor yDesc,
                                CudaDeviceVariable<double> y,     // NxCxHxW

                                /* Shared desc for the next 6 tensors in the argument list.
                                   Data type to be set as follows:
                                   type = (typeOf(x) == double) ? double : float
                                   Dimensions for this descriptor depend on normalization mode
                                   - Spatial Normalization : tensors are expected to have dims 1xCx1x1
                                    (normalization is performed across NxHxW)
                                   - Per-Activation Normalization : tensors are expected to have dims of 1xCxHxW 
                                    (normalization is performed across N) */
                                TensorDescriptor bnScaleBiasMeanVarDesc,

                                // 'Gamma' and 'Beta' respectively in Ioffe and Szegedy's paper's notation
                                CudaDeviceVariable<double> bnScale,
                                CudaDeviceVariable<double> bnBias,

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
                                CudaDeviceVariable<double> resultRunningMean,
                                /* Output in training mode, input in inference. Is the moving average
                                   of 1 / sqrt( epsilon + variance[x] ) */
                                CudaDeviceVariable<double> resultRunningInvVariance,

                                /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
                                double epsilon,

                                /* Optionally save intermediate results from the forward pass here
                                   - can be reused to speed up backward pass. NULL if unused */
                                CudaDeviceVariable<double> resultSaveMean,
                                CudaDeviceVariable<double> resultSaveInvVariance)
        {
            res = CudaDNNNativeMethods.cudnnBatchNormalizationForwardTraining(
                _handle, mode, ref alpha, ref beta, xDesc.Desc, x.DevicePointer, yDesc.Desc, y.DevicePointer,
                bnScaleBiasMeanVarDesc.Desc, bnScale.DevicePointer, bnBias.DevicePointer, exponentialAverageFactor,
                resultRunningMean.DevicePointer, resultRunningInvVariance.DevicePointer, epsilon, resultSaveMean.DevicePointer, resultSaveInvVariance.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "BatchNormalizationForwardTraining", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function performs the forward BatchNormalization layer computation for the inference phase. 
        /// This layer is based on the paper "Batch Normalization: Accelerating Deep Network 
        /// Training by Reducing Internal Covariate Shift", S. Ioffe, C. Szegedy, 2015.
        /// </summary>
        /// <param name="mode"> Mode of operation (spatial or per-activation). </param>
        /// <param name="alpha"> Pointer to scaling factors (in host memory) used to blend the layer output value with prior value in the destination tensor as follows: dstValue = alpha[0]*resultValue + beta[0]*priorDstValue. </param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the layer output value with prior value in the destination tensor as follows: dstValue = alpha[0]*resultValue + beta[0]*priorDstValue. </param>
        /// <param name="xDesc">Tensor descriptor layer's x data.</param>
        /// <param name="x">Pointer in device memory for the layer's x data.</param>
        /// <param name="yDesc">Tensor descriptor the layer's y data.</param>
        /// <param name="y">Pointer in device memory for the layer's y data.</param>
        /// <param name="bnScaleBiasMeanVarDesc">Shared tensor descriptor desc for all the 4 tensors below in the argument list. The dimensions for this tensor descriptor are dependent on the normalization mode.</param>
        /// <param name="bnScale">Pointer in device memory for the batch normalization scale parameters (in original paper scale is referred to as gamma).</param>
        /// <param name="bnBias">Pointers in device memory for the batch normalization bias parameters (in original paper bias is referred to as beta). Note that bnBias parameter can replace the previous layer's bias parameter for improved efficiency. </param>
        /// <param name="estimatedMean">Mean tensor (has the same descriptor as the bias and scale). It is suggested that resultRunningMean from the cudnnBatchNormalizationForwardTraining call accumulated during the training phase be passed as input here.</param>
        /// <param name="estimatedInvVariance">Variance tensor (has the same descriptor as the bias and scale). It is suggested that resultRunningVariance from the cudnnBatchNormalizationForwardTraining call accumulated during the training phase be passed as input here.</param>
        /// <param name="epsilon">Epsilon value used in the batch normalization formula. Minimum allowed value is currently 1e-5. Same epsilon value should be used in forward and backward functions.</param>
        public void BatchNormalizationForwardInference(
                                        cudnnBatchNormMode mode,
                                        double alpha, // alpha[0] = result blend factor
                                        double beta,  // beta[0] = dest layer blend factor
                                        TensorDescriptor xDesc,
                                        CudaDeviceVariable<double> x,     // NxCxHxW
                                        TensorDescriptor yDesc,
                                        CudaDeviceVariable<double> y,     // NxCxHxW
                                        TensorDescriptor bnScaleBiasMeanVarDesc,
                                        CudaDeviceVariable<double> bnScale,
                                        CudaDeviceVariable<double> bnBias,
                                        CudaDeviceVariable<double> estimatedMean,
                                        CudaDeviceVariable<double> estimatedInvVariance,
                                        double epsilon)
        {
            res = CudaDNNNativeMethods.cudnnBatchNormalizationForwardInference(
                _handle, mode, ref alpha, ref beta, xDesc.Desc, x.DevicePointer, yDesc.Desc, y.DevicePointer, 
                bnScaleBiasMeanVarDesc.Desc,bnScale.DevicePointer,bnBias.DevicePointer,estimatedMean.DevicePointer,estimatedInvVariance.DevicePointer, epsilon);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnBatchNormalizationForwardInference", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function performs the backward BatchNormalization layer computation.
        /// </summary>
        /// <param name="mode"> Mode of operation (spatial or per-activation). </param>
        /// <param name="alphaDataDiff">Pointer to scaling factors in host memory used to blend the gradient output dx with a prior value in the destination tensor as follows: dstValue = alpha[0]*resultValue + beta[0]*priorDstValue.</param>
        /// <param name="betaDataDiff">Pointer to scaling factors in host memory used to blend the gradient output dx with a prior value in the destination tensor as follows: dstValue = alpha[0]*resultValue + beta[0]*priorDstValue.</param>
        /// <param name="alphaParamDiff">Pointer to scaling factors (in host memory) used to blend the gradient outputs dBnScaleResult and dBnBiasResult with prior values in the destination tensor as follows: dstValue = alpha[0]*resultValue + beta[0]*priorDstValue.</param>
        /// <param name="betaParamDiff">Pointer to scaling factors (in host memory) used to blend the gradient outputs dBnScaleResult and dBnBiasResult with prior values in the destination tensor as follows: dstValue = alpha[0]*resultValue + beta[0]*priorDstValue.</param>
        /// <param name="xDesc">Tensor descriptor for the layer's x data.</param>
        /// <param name="x">Pointers in device memory for the layer's x data.</param>
        /// <param name="dyDesc">Tensor descriptor for the layer's backpropagated differential dy (inputs).</param>
        /// <param name="dy">Pointers in device memory for the layer's backpropagated differential dy (inputs).</param>
        /// <param name="dxDesc">Tensor descriptor for the layer's resulting differential with respect to x, dx (output).</param>
        /// <param name="dx">Pointer in device memory for the layer's resulting differential with respect to x, dx (output).</param>
        /// <param name="dBnScaleBiasDesc">Shared tensor descriptor for all the 5 tensors below in the argument list (bnScale, resultBnScaleDiff, resultBnBiasDiff, savedMean, savedInvVariance). The dimensions for this tensor descriptor are dependent on normalization mode. Note: The data type of this tensor descriptor must be 'float' for FP16 and FP32 input tensors, and 'double' for FP64 input tensors.</param>
        /// <param name="bnScale">Pointers in device memory for the batch normalization scale parameter (in original paper bias is referred to as gamma). Note that bnBias parameter is not needed for this layer's computation.</param>
        /// <param name="dBnScaleResult">Pointer in device memory for the resulting scale differentials computed by this routine. Note that scale and bias gradients are not backpropagated below this layer (since they are dead-end computation DAG nodes).</param>
        /// <param name="dBnBiasResult">Pointer in device memory for the resulting bias differentials computed by this routine. Note that scale and bias gradients are not backpropagated below this layer (since they are dead-end computation DAG nodes).</param>
        /// <param name="epsilon">Epsilon value used in the batch normalization formula. Minimum allowed value is currently 1e-5. Same epsilon value should be used in forward and backward functions.</param>
        /// <param name="savedMean">Optional cache parameter saved intermediate results computed during the forward pass. For this to work correctly, the layer's x and bnScale, bnBias data has to remain unchanged until the backward function is called. Note that both savedMean and savedInvVariance parameters can be NULL but only at the same time. It is recommended to use this cache since the memory overhead is relatively small.</param>
        /// <param name="savedInvVariance">Optional cache parameter saved intermediate results computed during the forward pass. For this to work correctly, the layer's x and bnScale, bnBias data has to remain unchanged until the backward function is called. Note that both savedMean and savedInvVariance parameters can be NULL but only at the same time. It is recommended to use this cache since the memory overhead is relatively small.</param>
        public void BatchNormalizationBackward(
                                        cudnnBatchNormMode mode,
                                        double alphaDataDiff,
                                        double betaDataDiff,
                                        double alphaParamDiff,
                                        double betaParamDiff,
                                        TensorDescriptor xDesc, // same desc for x, dx, dy
                                        CudaDeviceVariable<double> x,
                                        TensorDescriptor dyDesc,
                                        CudaDeviceVariable<double> dy,
                                        TensorDescriptor dxDesc,
                                        CudaDeviceVariable<double> dx,
                                        /* Shared tensor desc for the 5 tensors below */
                                        TensorDescriptor dBnScaleBiasDesc,
                                        CudaDeviceVariable<double> bnScale, // bnBias doesn't affect backpropagation
                                                                           /* scale and bias diff are not backpropagated below this layer */
                                        CudaDeviceVariable<double> dBnScaleResult,
                                        CudaDeviceVariable<double> dBnBiasResult,
                                        /* Same epsilon as forward pass */
                                        double epsilon,

                                        /* Optionally cached intermediate results from
                                           forward pass */
                                        CudaDeviceVariable<double> savedMean,
                                        CudaDeviceVariable<double> savedInvVariance)
        {
            res = CudaDNNNativeMethods.cudnnBatchNormalizationBackward(
                _handle, mode, ref alphaDataDiff, ref betaDataDiff, ref alphaParamDiff, ref betaParamDiff,
                xDesc.Desc, x.DevicePointer, dyDesc.Desc, dy.DevicePointer, dxDesc.Desc, dx.DevicePointer,
                dBnScaleBiasDesc.Desc, bnScale.DevicePointer, dBnScaleResult.DevicePointer, dBnBiasResult.DevicePointer,
                epsilon, savedMean.DevicePointer, savedInvVariance.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "BatchNormalizationBackward", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        #endregion
    }
}
