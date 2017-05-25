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
	/// An opaque structure holding
	/// the description of a pooling operation.
	/// </summary>
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


        /// <summary>
        /// This function initializes a previously created generic pooling descriptor object into a 2D description.
        /// </summary>
        /// <param name="mode">Enumerant to specify the pooling mode.</param>
        /// <param name="maxpoolingNanOpt">Nan propagation option for max pooling.</param>
        /// <param name="windowHeight">Height of the pooling window.</param>
        /// <param name="windowWidth">Width of the pooling window.</param>
        /// <param name="verticalPadding">Size of vertical padding.</param>
        /// <param name="horizontalPadding">Size of horizontal padding</param>
        /// <param name="verticalStride">Pooling vertical stride.</param>
        /// <param name="horizontalStride">Pooling horizontal stride.</param>
        public void SetPooling2dDescriptor(cudnnPoolingMode mode,
                                            cudnnNanPropagation maxpoolingNanOpt,
                                            int windowHeight,
										    int windowWidth,
										    int verticalPadding,
										    int horizontalPadding,
										    int verticalStride,
										    int horizontalStride
														   )
		{
			res = CudaDNNNativeMethods.cudnnSetPooling2dDescriptor(_desc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetPooling2dDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


        /// <summary>
        /// This function queries a previously created 2D pooling descriptor object.
        /// </summary>
        /// <param name="mode">Enumerant to specify the pooling mode.</param>
        /// <param name="maxpoolingNanOpt">Nan propagation option for max pooling.</param>
        /// <param name="windowHeight">Height of the pooling window.</param>
        /// <param name="windowWidth">Width of the pooling window.</param>
        /// <param name="verticalPadding">Size of vertical padding.</param>
        /// <param name="horizontalPadding">Size of horizontal padding.</param>
        /// <param name="verticalStride">Pooling vertical stride.</param>
        /// <param name="horizontalStride">Pooling horizontal stride.</param>
        public void GetPooling2dDescriptor(ref cudnnPoolingMode mode,
                                            ref cudnnNanPropagation maxpoolingNanOpt,
                                            ref int windowHeight,
											ref int windowWidth,
											ref int verticalPadding,
											ref int horizontalPadding,
											ref int verticalStride,
											ref int horizontalStride
										)
		{
			res = CudaDNNNativeMethods.cudnnGetPooling2dDescriptor(_desc, ref mode, ref maxpoolingNanOpt, ref windowHeight, ref windowWidth, ref verticalPadding, ref horizontalPadding, ref verticalStride, ref horizontalStride);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetPooling2dDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

        /// <summary>
        /// This function initializes a previously created generic pooling descriptor object.
        /// </summary>
        /// <param name="mode">Enumerant to specify the pooling mode.</param>
        /// <param name="maxpoolingNanOpt">Nan propagation option for max pooling.</param>
        /// <param name="nbDims">Dimension of the pooling operation.</param>
        /// <param name="windowDimA">Array of dimension nbDims containing the window size for each dimension.</param>
        /// <param name="paddingA">Array of dimension nbDims containing the padding size for each dimension.</param>
        /// <param name="strideA">Array of dimension nbDims containing the striding size for each dimension.</param>
        public void SetPoolingNdDescriptor(cudnnPoolingMode mode,
                                            cudnnNanPropagation maxpoolingNanOpt,
                                            int nbDims,
											int[] windowDimA,
											int[] paddingA,
											int[] strideA
										)
		{
			res = CudaDNNNativeMethods.cudnnSetPoolingNdDescriptor(_desc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetPoolingNdDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

        /// <summary>
        /// This function queries a previously initialized generic pooling descriptor object.
        /// </summary>
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
        public void GetPoolingNdDescriptor(int nbDimsRequested,
											ref cudnnPoolingMode mode,
                                            ref cudnnNanPropagation maxpoolingNanOpt,
                                            ref int nbDims,
											int[] windowDimA,
											int[] paddingA,
											int[] strideA
											)
		{
			res = CudaDNNNativeMethods.cudnnGetPoolingNdDescriptor(_desc, nbDimsRequested, ref mode, ref maxpoolingNanOpt, ref nbDims, windowDimA, paddingA, strideA);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetPoolingNdDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/// <summary>
		/// This function provides the output dimensions of a tensor after Nd pooling has been applied
		/// </summary>
		/// <param name="inputTensorDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="nbDims">Number of dimensions in which pooling is to be applied.</param>
		/// <param name="outputTensorDimA">Array of nbDims output dimensions</param>
		public void GetPoolingNdForwardOutputDim(TensorDescriptor inputTensorDesc,
																	 int nbDims,
																	 int[] outputTensorDimA)
		{
			res = CudaDNNNativeMethods.cudnnGetPoolingNdForwardOutputDim(_desc, inputTensorDesc.Desc, nbDims, outputTensorDimA);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetPoolingNdForwardOutputDim", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}

		/// <summary>
		/// This function provides the output dimensions of a tensor after 2d pooling has been applied
		/// </summary>
		/// <param name="inputTensorDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="n">Number of images in the output</param>
		/// <param name="c">Number of channels in the output</param>
		/// <param name="h">Height of images in the output</param>
		/// <param name="w">Width of images in the output</param>
		public void GetPooling2dForwardOutputDim(TensorDescriptor inputTensorDesc,
																	 ref int n,
																	 ref int c,
																	 ref int h,
																	 ref int w)
		{
			res = CudaDNNNativeMethods.cudnnGetPooling2dForwardOutputDim(_desc, inputTensorDesc.Desc, ref n, ref c, ref h, ref w);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetPooling2dForwardOutputDim", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}
	}
}
