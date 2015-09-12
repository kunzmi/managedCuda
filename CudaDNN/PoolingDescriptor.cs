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

	public class PoolingDescriptor : IDisposable
	{
		private cudnnPoolingDescriptor _desc;
		private cudnnStatus res;
		private bool disposed;

		#region Contructors
		/// <summary>
		/// </summary>
		public PoolingDescriptor()
		{
			_desc = new cudnnPoolingDescriptor();
			res = CudaDNNNativeMethods.cudnnCreatePoolingDescriptor(ref _desc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCreatePoolingDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~PoolingDescriptor()
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
				res = CudaDNNNativeMethods.cudnnDestroyPoolingDescriptor(_desc);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDestroyPoolingDescriptor", res));
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		/// <summary>
		/// Returns the inner handle.
		/// </summary>
		public cudnnPoolingDescriptor Desc
		{
			get { return _desc; }
		}


		public void SetPooling2dDescriptor(cudnnPoolingMode mode,
																int windowHeight,
																int windowWidth,
																int verticalPadding,
																int horizontalPadding,
																int verticalStride,
																int horizontalStride
														   )
		{
			res = CudaDNNNativeMethods.cudnnSetPooling2dDescriptor(_desc, mode, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetPooling2dDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		public void GetPooling2dDescriptor(ref cudnnPoolingMode mode,
											ref int windowHeight,
											ref int windowWidth,
											ref int verticalPadding,
											ref int horizontalPadding,
											ref int verticalStride,
											ref int horizontalStride
										)
		{
			res = CudaDNNNativeMethods.cudnnGetPooling2dDescriptor(_desc, ref mode, ref windowHeight, ref windowWidth, ref verticalPadding, ref horizontalPadding, ref verticalStride, ref horizontalStride);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetPooling2dDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		public void SetPoolingNdDescriptor(cudnnPoolingMode mode,
											int nbDims,
											int[] windowDimA,
											int[] paddingA,
											int[] strideA
										)
		{
			res = CudaDNNNativeMethods.cudnnSetPoolingNdDescriptor(_desc, mode, nbDims, windowDimA, paddingA, strideA);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetPoolingNdDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		public void GetPoolingNdDescriptor(int nbDimsRequested,
											ref cudnnPoolingMode mode,
											ref int nbDims,
											int[] windowDimA,
											int[] paddingA,
											int[] strideA
											)
		{
			res = CudaDNNNativeMethods.cudnnGetPoolingNdDescriptor(_desc, nbDimsRequested, ref mode, ref nbDims, windowDimA, paddingA, strideA);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetPoolingNdDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		public void GetPoolingNdForwardOutputDim(TensorDescriptor inputTensorDesc,
																	 int nbDims,
																	 int[] outputTensorDimA)
		{
			res = CudaDNNNativeMethods.cudnnGetPoolingNdForwardOutputDim(_desc, inputTensorDesc.Desc, nbDims, outputTensorDimA);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetPoolingNdForwardOutputDim", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		public void GetPooling2dForwardOutputDim(TensorDescriptor inputTensorDesc,
																	 ref int outN,
																	 ref int outC,
																	 ref int outH,
																	 ref int outW)
		{
			res = CudaDNNNativeMethods.cudnnGetPooling2dForwardOutputDim(_desc, inputTensorDesc.Desc, ref outN, ref outC, ref outH, ref outW);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetPooling2dForwardOutputDim", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}
	}
}
