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

	public class ConvolutionDescriptor : IDisposable
	{
		private cudnnConvolutionDescriptor _desc;
		private cudnnStatus res;
		private bool disposed;

		#region Contructors
		/// <summary>
		/// </summary>
		public ConvolutionDescriptor()
		{
			_desc = new cudnnConvolutionDescriptor();
			res = CudaDNNNativeMethods.cudnnCreateConvolutionDescriptor(ref _desc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCreateConvolutionDescriptor", res));
			if (res != cudnnStatus.Success)
				throw new CudaDNNException(res);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~ConvolutionDescriptor()
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
				res = CudaDNNNativeMethods.cudnnDestroyConvolutionDescriptor(_desc);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDestroyConvolutionDescriptor", res));
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		/// <summary>
		/// Returns the inner handle.
		/// </summary>
		public cudnnConvolutionDescriptor Desc
		{
			get { return _desc; }
		}


		public void SetConvolution2dDescriptor(int pad_h,    // zero-padding height
												int pad_w,    // zero-padding width
												int u,        // vertical filter stride
												int v,        // horizontal filter stride
												int upscalex, // upscale the input in x-direction
												int upscaley, // upscale the input in y-direction
												cudnnConvolutionMode mode
												)
		{
			res = CudaDNNNativeMethods.cudnnSetConvolution2dDescriptor(_desc, pad_h, pad_w, u, v, upscalex, upscaley, mode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetConvolution2dDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


		public void GetConvolution2dDescriptor(ref int pad_h,    // zero-padding height
												ref int pad_w,    // zero-padding width
												ref int u,        // vertical filter stride
												ref int v,        // horizontal filter stride
												ref int upscalex, // upscale the input in x-direction
												ref int upscaley, // upscale the input in y-direction
												ref cudnnConvolutionMode mode
											)
		{
			res = CudaDNNNativeMethods.cudnnGetConvolution2dDescriptor(_desc, ref pad_h, ref pad_w, ref u, ref v, ref upscalex, ref upscaley, ref mode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolution2dDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/* Helper function to return the dimensions of the output tensor given a convolution descriptor */
		public void GetConvolution2dForwardOutputDim(TensorDescriptor inputTensorDesc,
													FilterDescriptor filterDesc,
													ref int n,
													ref int c,
													ref int h,
													ref int w
												)
		{
			res = CudaDNNNativeMethods.cudnnGetConvolution2dForwardOutputDim(_desc, inputTensorDesc.Desc, filterDesc.Desc, ref n, ref c, ref h, ref w);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolution2dForwardOutputDim", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


		public void SetConvolutionNdDescriptor(int arrayLength,             /* nbDims-2 size */
											int[] padA,
											int[] filterStrideA,
											int[] upscaleA,
											cudnnConvolutionMode mode
											)
		{
			res = CudaDNNNativeMethods.cudnnSetConvolutionNdDescriptor(_desc, arrayLength, padA, filterStrideA, upscaleA, mode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetConvolutionNdDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		public void GetConvolutionNdDescriptor(int arrayLengthRequested,
											ref int arrayLength,
											int[] padA,
											int[] strideA,
											int[] upscaleA,
											ref cudnnConvolutionMode mode
											)
		{
			res = CudaDNNNativeMethods.cudnnGetConvolutionNdDescriptor(_desc, arrayLengthRequested, ref arrayLength, padA, strideA, upscaleA, ref  mode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionNdDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


		/* Helper function to return the dimensions of the output tensor given a convolution descriptor */
		public void GetConvolutionNdForwardOutputDim(TensorDescriptor inputTensorDesc,
													FilterDescriptor filterDesc,
													int nbDims,
													int[] tensorOuputDimA
												)
		{
			res = CudaDNNNativeMethods.cudnnGetConvolutionNdForwardOutputDim(_desc, inputTensorDesc.Desc, filterDesc.Desc, nbDims, tensorOuputDimA);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionNdForwardOutputDim", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}
	}
}
