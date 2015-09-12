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
	public class TensorDescriptor : IDisposable
	{
		private cudnnTensorDescriptor _desc;
		private cudnnStatus res;
		private bool disposed;

		#region Contructors
		/// <summary>
		/// </summary>
		public TensorDescriptor()
		{
			_desc = new cudnnTensorDescriptor();
			res = CudaDNNNativeMethods.cudnnCreateTensorDescriptor(ref _desc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCreateTensorDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~TensorDescriptor()
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
				res = CudaDNNNativeMethods.cudnnDestroyTensorDescriptor(_desc);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDestroyTensorDescriptor", res));
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		/// <summary>
		/// Returns the inner handle.
		/// </summary>
		public cudnnTensorDescriptor Desc
		{
			get { return _desc; }
		}


		public void SetTensor4dDescriptor(cudnnTensorFormat format,
											cudnnDataType dataType, // image data type
											int n,        // number of inputs (batch size)
											int c,        // number of input feature maps
											int h,        // height of input section
											int w         // width of input section
										)
		{
			res = CudaDNNNativeMethods.cudnnSetTensor4dDescriptor(_desc, format, dataType, n, c, h, w);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetTensor4dDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}



		public void SetTensor4dDescriptorEx(cudnnDataType dataType, // image data type
											int n,        // number of inputs (batch size)
											int c,        // number of input feature maps
											int h,        // height of input section
											int w,        // width of input section
											int nStride,
											int cStride,
											int hStride,
											int wStride
											)
		{
			res = CudaDNNNativeMethods.cudnnSetTensor4dDescriptorEx(_desc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetTensor4dDescriptorEx", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


		public void GetTensor4dDescriptor(  ref cudnnDataType dataType, // image data type
											ref int n,        // number of inputs (batch size)
											ref int c,        // number of input feature maps
											ref int h,        // height of input section
											ref int w,        // width of input section
											ref int nStride,
											ref int cStride,
											ref int hStride,
											ref int wStride
										)
		{
			res = CudaDNNNativeMethods.cudnnGetTensor4dDescriptor(_desc, ref dataType, ref n, ref c, ref h, ref w, ref nStride, ref cStride, ref hStride, ref wStride);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetTensor4dDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


		public void SetTensorNdDescriptor(  cudnnDataType dataType,
											int nbDims,
											int[] dimA,
											int[] strideA
											)
		{
			res = CudaDNNNativeMethods.cudnnSetTensorNdDescriptor(_desc, dataType, nbDims, dimA, strideA);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetTensorNdDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


		public void GetTensorNdDescriptor(  int nbDimsRequested,
											ref cudnnDataType dataType,
											ref int nbDims,
											int[] dimA,
											int[] strideA
											)
		{
			res = CudaDNNNativeMethods.cudnnGetTensorNdDescriptor(_desc, nbDimsRequested, ref dataType, ref nbDims, dimA, strideA);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetTensorNdDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}
	}
}
