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
		/* Tensor layout conversion helper (dest = alpha * src + beta * dest) */
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



		/* Tensor Bias addition : srcDest = alpha * bias + beta * srcDestDesc  */
		public void AddTensor(cudnnAddMode mode,
									float alpha,
									TensorDescriptor biasDesc,
									CudaDeviceVariable<float> biasData,
									float beta,
									TensorDescriptor srcDestDesc,
									CudaDeviceVariable<float> srcDestData
									)
		{
			res = CudaDNNNativeMethods.cudnnAddTensor(_handle, mode, ref alpha, biasDesc.Desc, biasData.DevicePointer, ref beta, srcDestDesc.Desc, srcDestData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnAddTensor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/* Set all data points of a tensor to a given value : srcDest = value */
		public void SetTensor(TensorDescriptor srcDestDesc,
									CudaDeviceVariable<float> srcDestData,
									CudaDeviceVariable<float> value
									)
		{
			res = CudaDNNNativeMethods.cudnnSetTensor(_handle, srcDestDesc.Desc, srcDestData.DevicePointer, value.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetTensor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/* Set all data points of a tensor to a given value : srcDest = alpha * srcDest */
		public void ScaleTensor(TensorDescriptor srcDestDesc,
										CudaDeviceVariable<float> srcDestData,
										float alpha
									)
		{
			res = CudaDNNNativeMethods.cudnnScaleTensor(_handle, srcDestDesc.Desc, srcDestData.DevicePointer, ref alpha);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnScaleTensor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


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


		/* Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */

		/* Function to perform the forward multiconvolution */
		public void ConvolutionForward(float alpha,
										TensorDescriptor srcDesc,
										CudaDeviceVariable<float> srcData,
										FilterDescriptor filterDesc,
										CudaDeviceVariable<float> filterData,
										ConvolutionDescriptor convDesc,
										cudnnConvolutionFwdAlgo algo,
										CudaDeviceVariable<byte> workSpace,
										SizeT workSpaceSizeInBytes,
										float beta,
										TensorDescriptor destDesc,
										CudaDeviceVariable<float> destData
									)
		{
			res = CudaDNNNativeMethods.cudnnConvolutionForward(_handle, ref alpha, srcDesc.Desc, srcData.DevicePointer, filterDesc.Desc, filterData.DevicePointer, convDesc.Desc, algo, workSpace.DevicePointer, workSpaceSizeInBytes, ref beta, destDesc.Desc, destData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnConvolutionForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/* Functions to perform the backward multiconvolution */
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



		public void ConvolutionBackwardFilter(float alpha,
												TensorDescriptor srcDesc,
												CudaDeviceVariable<float> srcData,
												TensorDescriptor diffDesc,
												CudaDeviceVariable<float> diffData,
												ConvolutionDescriptor convDesc,
												float beta,
												FilterDescriptor gradDesc,
												CudaDeviceVariable<float> gradData
											)
		{
			res = CudaDNNNativeMethods.cudnnConvolutionBackwardFilter(_handle, ref alpha, srcDesc.Desc, srcData.DevicePointer, diffDesc.Desc, diffData.DevicePointer, convDesc.Desc, ref beta, gradDesc.Desc, gradData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnConvolutionBackwardFilter", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


		public void ConvolutionBackwardData(float alpha,
											FilterDescriptor filterDesc,
											CudaDeviceVariable<float> filterData,
											TensorDescriptor diffDesc,
											CudaDeviceVariable<float> diffData,
											ConvolutionDescriptor convDesc,
											float beta,
											TensorDescriptor gradDesc,
											CudaDeviceVariable<float> gradData
										)
		{
			res = CudaDNNNativeMethods.cudnnConvolutionBackwardData(_handle, ref alpha, filterDesc.Desc, filterData.DevicePointer, diffDesc.Desc, diffData.DevicePointer, convDesc.Desc, ref beta, gradDesc.Desc, gradData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnConvolutionBackwardData", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


		public void Im2Col(float alpha,
							TensorDescriptor srcDesc,
							CudaDeviceVariable<float> srcData,
							FilterDescriptor filterDesc,
							ConvolutionDescriptor convDesc,
							CudaDeviceVariable<byte> colBuffer
							)
		{
			res = CudaDNNNativeMethods.cudnnIm2Col(_handle, ref alpha, srcDesc.Desc, srcData.DevicePointer, filterDesc.Desc, convDesc.Desc, colBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnIm2Col", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}




		/* Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" */

		/* Function to perform forward softmax */
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

		/* Function to perform backward softmax */
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







		/* Function to perform forward pooling */
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

		/* Function to perform backward pooling */
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

		/* Function to perform forward activation  */
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

		/* Function to perform backward activation  */
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
		/* Tensor layout conversion helper (dest = alpha * src + beta * dest) */
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



		/* Tensor Bias addition : srcDest = alpha * bias + beta * srcDestDesc  */
		public void AddTensor(cudnnAddMode mode,
									double alpha,
									TensorDescriptor biasDesc,
									CudaDeviceVariable<double> biasData,
									double beta,
									TensorDescriptor srcDestDesc,
									CudaDeviceVariable<double> srcDestData
									)
		{
			res = CudaDNNNativeMethods.cudnnAddTensor(_handle, mode, ref alpha, biasDesc.Desc, biasData.DevicePointer, ref beta, srcDestDesc.Desc, srcDestData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnAddTensor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/* Set all data points of a tensor to a given value : srcDest = value */
		public void SetTensor(TensorDescriptor srcDestDesc,
									CudaDeviceVariable<double> srcDestData,
									CudaDeviceVariable<double> value
									)
		{
			res = CudaDNNNativeMethods.cudnnSetTensor(_handle, srcDestDesc.Desc, srcDestData.DevicePointer, value.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetTensor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/* Set all data points of a tensor to a given value : srcDest = alpha * srcDest */
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

		/* Function to perform the forward multiconvolution */
		public void ConvolutionForward(double alpha,
										TensorDescriptor srcDesc,
										CudaDeviceVariable<double> srcData,
										FilterDescriptor filterDesc,
										CudaDeviceVariable<double> filterData,
										ConvolutionDescriptor convDesc,
										cudnnConvolutionFwdAlgo algo,
										CudaDeviceVariable<byte> workSpace,
										SizeT workSpaceSizeInBytes,
										double beta,
										TensorDescriptor destDesc,
										CudaDeviceVariable<double> destData
									)
		{
			res = CudaDNNNativeMethods.cudnnConvolutionForward(_handle, ref alpha, srcDesc.Desc, srcData.DevicePointer, filterDesc.Desc, filterData.DevicePointer, convDesc.Desc, algo, workSpace.DevicePointer, workSpaceSizeInBytes, ref beta, destDesc.Desc, destData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnConvolutionForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/* Functions to perform the backward multiconvolution */
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



		public void ConvolutionBackwardFilter(double alpha,
												TensorDescriptor srcDesc,
												CudaDeviceVariable<double> srcData,
												TensorDescriptor diffDesc,
												CudaDeviceVariable<double> diffData,
												ConvolutionDescriptor convDesc,
												double beta,
												FilterDescriptor gradDesc,
												CudaDeviceVariable<double> gradData
											)
		{
			res = CudaDNNNativeMethods.cudnnConvolutionBackwardFilter(_handle, ref alpha, srcDesc.Desc, srcData.DevicePointer, diffDesc.Desc, diffData.DevicePointer, convDesc.Desc, ref beta, gradDesc.Desc, gradData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnConvolutionBackwardFilter", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


		public void ConvolutionBackwardData(double alpha,
											FilterDescriptor filterDesc,
											CudaDeviceVariable<double> filterData,
											TensorDescriptor diffDesc,
											CudaDeviceVariable<double> diffData,
											ConvolutionDescriptor convDesc,
											double beta,
											TensorDescriptor gradDesc,
											CudaDeviceVariable<double> gradData
										)
		{
			res = CudaDNNNativeMethods.cudnnConvolutionBackwardData(_handle, ref alpha, filterDesc.Desc, filterData.DevicePointer, diffDesc.Desc, diffData.DevicePointer, convDesc.Desc, ref beta, gradDesc.Desc, gradData.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnConvolutionBackwardData", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


		public void Im2Col(double alpha,
							TensorDescriptor srcDesc,
							CudaDeviceVariable<double> srcData,
							FilterDescriptor filterDesc,
							ConvolutionDescriptor convDesc,
							CudaDeviceVariable<byte> colBuffer
							)
		{
			res = CudaDNNNativeMethods.cudnnIm2Col(_handle, ref alpha, srcDesc.Desc, srcData.DevicePointer, filterDesc.Desc, convDesc.Desc, colBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnIm2Col", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}




		/* Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" */

		/* Function to perform forward softmax */
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

		/* Function to perform backward softmax */
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







		/* Function to perform forward pooling */
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

		/* Function to perform backward pooling */
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

		/* Function to perform forward activation  */
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

		/* Function to perform backward activation  */
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
