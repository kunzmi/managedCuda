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
	/// 
	/// </summary>
	public class LRNDescriptor : IDisposable
	{
		private cudnnLRNDescriptor _desc;
		private cudnnStatus res;
		private bool disposed;
		private cudnnHandle _handle;

		#region Contructors
		/// <summary>
		/// </summary>
		public LRNDescriptor(CudaDNNContext context)
		{
			_handle = context.Handle;
			_desc = new cudnnLRNDescriptor();
			res = CudaDNNNativeMethods.cudnnCreateLRNDescriptor(ref _desc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCreateLRNDescriptor", res));
			if (res != cudnnStatus.Success)
				throw new CudaDNNException(res);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~LRNDescriptor()
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
				res = CudaDNNNativeMethods.cudnnDestroyLRNDescriptor(_desc);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDestroyLRNDescriptor", res));
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		/// <summary>
		/// Returns the inner handle.
		/// </summary>
		public cudnnLRNDescriptor Desc
		{
			get { return _desc; }
		}


		/// <summary>
		/// This function initializes a previously created LRN descriptor object.
		/// </summary>
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
		public void SetLRNDescriptor(uint lrnN,
									  double lrnAlpha,
									  double lrnBeta,
									  double lrnK
												)
		{
			res = CudaDNNNativeMethods.cudnnSetLRNDescriptor(_desc, lrnN, lrnAlpha, lrnBeta, lrnK);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetLRNDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/// <summary>
		/// This function retrieves values stored in the previously initialized LRN descriptor object.
		/// </summary>
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
		public void GetLRNDescriptor(ref uint lrnN,
									  ref double lrnAlpha,
									  ref double lrnBeta,
									  ref double lrnK
												)
		{
			res = CudaDNNNativeMethods.cudnnGetLRNDescriptor(_desc, ref lrnN, ref lrnAlpha, ref lrnBeta, ref lrnK);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetLRNDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


		/// <summary>
		/// This function performs the forward LRN layer computation.
		/// </summary>
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
		public void cudnnLRNCrossChannelForward(
									  cudnnLRNMode lrnMode,
									  float alpha,
									  cudnnTensorDescriptor xDesc,
									  CUdeviceptr x,
									  float beta,
									  cudnnTensorDescriptor yDesc,
									  CUdeviceptr y)
		{
			res = CudaDNNNativeMethods.cudnnLRNCrossChannelForward(_handle, _desc, lrnMode, ref alpha, xDesc, x, ref beta, yDesc, y);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnLRNCrossChannelForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/// <summary>
		/// This function performs the forward LRN layer computation.
		/// </summary>
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
		public void cudnnLRNCrossChannelForward(
									  cudnnLRNMode lrnMode,
									  double alpha,
									  cudnnTensorDescriptor xDesc,
									  CUdeviceptr x,
									  double beta,
									  cudnnTensorDescriptor yDesc,
									  CUdeviceptr y)
		{
			res = CudaDNNNativeMethods.cudnnLRNCrossChannelForward(_handle, _desc, lrnMode, ref alpha, xDesc, x, ref beta, yDesc, y);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnLRNCrossChannelForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


        /// <summary>
        /// This function performs the backward LRN layer computation.
        /// </summary>
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
        public void cudnnLRNCrossChannelBackward(
                                      cudnnLRNMode lrnMode,
                                      ref float alpha,
                                      cudnnTensorDescriptor yDesc,
                                      CUdeviceptr y,
                                      cudnnTensorDescriptor dyDesc,
                                      CUdeviceptr dy,
                                      cudnnTensorDescriptor xDesc,
                                      CUdeviceptr x,
                                      ref float beta,
                                      cudnnTensorDescriptor dxDesc,
                                      CUdeviceptr dx)
        {
            res = CudaDNNNativeMethods.cudnnLRNCrossChannelBackward(_handle, _desc, lrnMode, ref alpha, yDesc, y, dyDesc, dy, xDesc, x, ref beta, dxDesc, dx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnLRNCrossChannelBackward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

        /// <summary>
        /// This function performs the backward LRN layer computation.
        /// </summary>
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
        public void cudnnLRNCrossChannelBackward(
									  cudnnLRNMode lrnMode,
									  ref double alpha,
									  cudnnTensorDescriptor yDesc,
									  CUdeviceptr y,
									  cudnnTensorDescriptor dyDesc,
									  CUdeviceptr dy,
									  cudnnTensorDescriptor xDesc,
									  CUdeviceptr x,
									  ref double beta,
									  cudnnTensorDescriptor dxDesc,
									  CUdeviceptr dx)
		{
			res = CudaDNNNativeMethods.cudnnLRNCrossChannelBackward(_handle, _desc, lrnMode, ref alpha, yDesc, y, dyDesc, dy, xDesc, x, ref beta, dxDesc, dx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnLRNCrossChannelBackward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


        /// <summary>
        /// This function performs the forward DivisiveNormalization layer computation.
        /// </summary>
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
        public void cudnnDivisiveNormalizationForward(
                                      cudnnDivNormMode mode,
                                      float alpha,
                                      cudnnTensorDescriptor xDesc, // same desc for means, temp, temp2
                                      CUdeviceptr x,
                                      CUdeviceptr means, // if NULL, means are assumed to be zero
                                      CUdeviceptr temp,
                                      CUdeviceptr temp2,
                                      float beta,
                                      cudnnTensorDescriptor yDesc,
                                      CUdeviceptr y)
        {
            res = CudaDNNNativeMethods.cudnnDivisiveNormalizationForward(_handle, _desc, mode, ref alpha, xDesc, x, means, temp, temp2, ref beta, yDesc, y);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDivisiveNormalizationForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

        /// <summary>
        /// This function performs the forward DivisiveNormalization layer computation.
        /// </summary>
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
        public void cudnnDivisiveNormalizationForward(
									  cudnnDivNormMode mode,
									  double alpha,
									  cudnnTensorDescriptor xDesc, // same desc for means, temp, temp2
									  CUdeviceptr x,
									  CUdeviceptr means, // if NULL, means are assumed to be zero
									  CUdeviceptr temp,
									  CUdeviceptr temp2,
									  double beta,
									  cudnnTensorDescriptor yDesc,
									  CUdeviceptr y)
		{
			res = CudaDNNNativeMethods.cudnnDivisiveNormalizationForward(_handle, _desc, mode, ref alpha, xDesc, x, means, temp, temp2, ref beta, yDesc, y);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDivisiveNormalizationForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}



        /// <summary>
        /// This function performs the backward DivisiveNormalization layer computation.
        /// </summary>
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
        public void cudnnDivisiveNormalizationBackward(
                                      cudnnDivNormMode mode,
                                      float alpha,
                                      cudnnTensorDescriptor xDesc, // same desc for diff, means, temp, temp2
                                      CUdeviceptr x,
                                      CUdeviceptr means, // if NULL, means are assumed to be zero
                                      CUdeviceptr dy,
                                      CUdeviceptr temp,
                                      CUdeviceptr temp2,
                                      float beta,
                                      cudnnTensorDescriptor dXdMeansDesc, // same desc for dest, means, meansDiff
                                      CUdeviceptr dx, // output data differential
                                      CUdeviceptr dMeans // output means differential, can be NULL
            )
        {
            res = CudaDNNNativeMethods.cudnnDivisiveNormalizationBackward(_handle, _desc, mode, ref alpha, xDesc, x, means, dy, temp, temp2, ref beta, dXdMeansDesc, dx, dMeans);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDivisiveNormalizationBackward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}
        /// <summary>
        /// This function performs the backward DivisiveNormalization layer computation.
        /// </summary>
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
        public void cudnnDivisiveNormalizationBackward(
									  cudnnDivNormMode mode,
									  double alpha,
									  cudnnTensorDescriptor xDesc, // same desc for diff, means, temp, temp2
									  CUdeviceptr x,
									  CUdeviceptr means, // if NULL, means are assumed to be zero
									  CUdeviceptr dy,
									  CUdeviceptr temp,
									  CUdeviceptr temp2,
									  double beta,
									  cudnnTensorDescriptor dXdMeansDesc, // same desc for dest, means, meansDiff
									  CUdeviceptr dx, // output data differential
									  CUdeviceptr dMeans // output means differential, can be NULL
            )
		{
			res = CudaDNNNativeMethods.cudnnDivisiveNormalizationBackward(_handle, _desc, mode, ref alpha, xDesc, x, means, dy, temp, temp2, ref beta, dXdMeansDesc, dx, dMeans);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDivisiveNormalizationBackward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}
	}
}
