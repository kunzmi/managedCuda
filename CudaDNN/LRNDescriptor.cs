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


		// LRN uses a window [center-lookBehind, center+lookAhead], where
		// lookBehind = floor( (lrnN-1)/2 ), lookAhead = lrnN-lookBehind-1.
		// So for n=10, the window is [k-4...k...k+5] with a total of 10 samples.
		// Values of double parameters will be cast down to tensor data type.
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
		
		// Retrieve the settings currently stored in an LRN layer descriptor
		// Any of the provided pointers can be NULL (no corresponding value will be returned)
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


		// LRN functions: of the form "output = alpha * normalize(srcData) + beta * destData"

		// Function to perform LRN forward cross-channel computation
		// Values of double parameters will be cast down to tensor data type
		public void cudnnLRNCrossChannelForward(
									  cudnnLRNMode lrnMode,
									  float alpha,
									  cudnnTensorDescriptor srcDesc,
									  CUdeviceptr srcData,
									  float beta,
									  cudnnTensorDescriptor destDesc,
									  CUdeviceptr destData)
		{
			res = CudaDNNNativeMethods.cudnnLRNCrossChannelForward(_handle, _desc, lrnMode, ref alpha, srcDesc, srcData, ref beta, destDesc, destData);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnLRNCrossChannelForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}
		public void cudnnLRNCrossChannelForward(
									  cudnnLRNMode lrnMode,
									  double alpha,
									  cudnnTensorDescriptor srcDesc,
									  CUdeviceptr srcData,
									  double beta,
									  cudnnTensorDescriptor destDesc,
									  CUdeviceptr destData)
		{
			res = CudaDNNNativeMethods.cudnnLRNCrossChannelForward(_handle, _desc, lrnMode, ref alpha, srcDesc, srcData, ref beta, destDesc, destData);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnLRNCrossChannelForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		// Function to perform LRN cross-channel backpropagation
		// values of double parameters will be cast down to tensor data type
		// src is the front layer, dst is the back layer

		public void cudnnLRNCrossChannelBackward(
									  cudnnLRNMode lrnMode,
									  ref float alpha,
									  cudnnTensorDescriptor srcDesc,
									  CUdeviceptr srcData,
									  cudnnTensorDescriptor srcDiffDesc,
									  CUdeviceptr srcDiffData,
									  cudnnTensorDescriptor destDesc,
									  CUdeviceptr destData,
									  ref float beta,
									  cudnnTensorDescriptor destDiffDesc,
									  CUdeviceptr destDiffData)
		{
			res = CudaDNNNativeMethods.cudnnLRNCrossChannelBackward(_handle, _desc, lrnMode, ref alpha, srcDesc, srcData, srcDiffDesc, srcDiffData, destDesc, destData, ref beta, destDiffDesc, destDiffData);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnLRNCrossChannelBackward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}
		public void cudnnLRNCrossChannelBackward(
									  cudnnLRNMode lrnMode,
									  ref double alpha,
									  cudnnTensorDescriptor srcDesc,
									  CUdeviceptr srcData,
									  cudnnTensorDescriptor srcDiffDesc,
									  CUdeviceptr srcDiffData,
									  cudnnTensorDescriptor destDesc,
									  CUdeviceptr destData,
									  ref double beta,
									  cudnnTensorDescriptor destDiffDesc,
									  CUdeviceptr destDiffData)
		{
			res = CudaDNNNativeMethods.cudnnLRNCrossChannelBackward(_handle, _desc, lrnMode, ref alpha, srcDesc, srcData, srcDiffDesc, srcDiffData, destDesc, destData, ref beta, destDiffDesc, destDiffData);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnLRNCrossChannelBackward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


		// LCN/divisive normalization functions: of the form "output = alpha * normalize(srcData) + beta * destData"
		// srcMeansData can be NULL to reproduce Caffe's LRN within-channel behavior
		public void cudnnDivisiveNormalizationForward(
									  cudnnDivNormMode mode,
									  float alpha,
									  cudnnTensorDescriptor srcDesc, // same desc for means, temp, temp2
									  CUdeviceptr srcData,
									  CUdeviceptr srcMeansData, // if NULL, means are assumed to be zero
									  CUdeviceptr tempData,
									  CUdeviceptr tempData2,
									  float beta,
									  cudnnTensorDescriptor destDesc,
									  CUdeviceptr destData)
		{
			res = CudaDNNNativeMethods.cudnnDivisiveNormalizationForward(_handle, _desc, mode, ref alpha, srcDesc, srcData, srcMeansData, tempData, tempData2, ref beta, destDesc, destData);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDivisiveNormalizationForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}
		public void cudnnDivisiveNormalizationForward(
									  cudnnDivNormMode mode,
									  double alpha,
									  cudnnTensorDescriptor srcDesc, // same desc for means, temp, temp2
									  CUdeviceptr srcData,
									  CUdeviceptr srcMeansData, // if NULL, means are assumed to be zero
									  CUdeviceptr tempData,
									  CUdeviceptr tempData2,
									  double beta,
									  cudnnTensorDescriptor destDesc,
									  CUdeviceptr destData)
		{
			res = CudaDNNNativeMethods.cudnnDivisiveNormalizationForward(_handle, _desc, mode, ref alpha, srcDesc, srcData, srcMeansData, tempData, tempData2, ref beta, destDesc, destData);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDivisiveNormalizationForward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}



		public void cudnnDivisiveNormalizationBackward(
									  cudnnDivNormMode mode,
									  float alpha,
									  cudnnTensorDescriptor srcDesc, // same desc for diff, means, temp, temp2
									  CUdeviceptr srcData,
									  CUdeviceptr srcMeansData, // if NULL, means are assumed to be zero
									  CUdeviceptr srcDiffData,
									  CUdeviceptr tempData,
									  CUdeviceptr tempData2,
									  float betaData,
									  cudnnTensorDescriptor destDataDesc, // same desc for dest, means, meansDiff
									  CUdeviceptr destDataDiff, // output data differential
									  CUdeviceptr destMeansDiff // output means differential, can be NULL
			)
		{
			res = CudaDNNNativeMethods.cudnnDivisiveNormalizationBackward(_handle, _desc, mode, ref alpha, srcDesc, srcData, srcMeansData, srcDiffData, tempData, tempData2, ref betaData, destDataDesc, destDataDiff, destMeansDiff);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDivisiveNormalizationBackward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}
		public void cudnnDivisiveNormalizationBackward(
									  cudnnDivNormMode mode,
									  double alpha,
									  cudnnTensorDescriptor srcDesc, // same desc for diff, means, temp, temp2
									  CUdeviceptr srcData,
									  CUdeviceptr srcMeansData, // if NULL, means are assumed to be zero
									  CUdeviceptr srcDiffData,
									  CUdeviceptr tempData,
									  CUdeviceptr tempData2,
									  double betaData,
									  cudnnTensorDescriptor destDataDesc, // same desc for dest, means, meansDiff
									  CUdeviceptr destDataDiff, // output data differential
									  CUdeviceptr destMeansDiff // output means differential, can be NULL
			)
		{
			res = CudaDNNNativeMethods.cudnnDivisiveNormalizationBackward(_handle, _desc, mode, ref alpha, srcDesc, srcData, srcMeansData, srcDiffData, tempData, tempData2, ref betaData, destDataDesc, destDataDiff, destMeansDiff);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDivisiveNormalizationBackward", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}
	}
}
