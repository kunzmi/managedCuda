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
	/// An opaque structure holding the
	/// description of a convolution operation.
	/// </summary>
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


		/// <summary>
		/// This function initializes a previously created convolution descriptor object into a 2D
		/// correlation. This function assumes that the tensor and filter descriptors corresponds
		/// to the formard convolution path and checks if their settings are valid. That same
		/// convolution descriptor can be reused in the backward path provided it corresponds to
		/// the same layer.
		/// </summary>
		/// <param name="pad_h">zero-padding height: number of rows of zeros implicitly concatenated
		/// onto the top and onto the bottom of input images.</param>
		/// <param name="pad_w">zero-padding width: number of columns of zeros implicitly concatenated
		/// onto the left and onto the right of input images.</param>
		/// <param name="u">Vertical filter stride.</param>
		/// <param name="v">Horizontal filter stride.</param>
		/// <param name="upscalex">Upscale the input in x-direction.</param>
		/// <param name="upscaley">Upscale the input in y-direction.</param>
		/// <param name="mode">Selects between CUDNN_CONVOLUTION and CUDNN_CROSS_CORRELATION.</param>
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

		/// <summary>
		/// This function queries a previously initialized 2D convolution descriptor object.
		/// </summary>
		/// <param name="pad_h">zero-padding height: number of rows of zeros implicitly concatenated
		/// onto the top and onto the bottom of input images.</param>
		/// <param name="pad_w">zero-padding width: number of columns of zeros implicitly concatenated
		/// onto the left and onto the right of input images.</param>
		/// <param name="u">Vertical filter stride.</param>
		/// <param name="v">Horizontal filter stride.</param>
		/// <param name="upscalex">Upscale the input in x-direction.</param>
		/// <param name="upscaley">Upscale the input in y-direction.</param>
		/// <param name="mode">convolution mode.</param>
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

		/// <summary>
		/// This function returns the dimensions of the resulting 4D tensor of a 2D convolution,
		/// given the convolution descriptor, the input tensor descriptor and the filter descriptor
		/// This function can help to setup the output tensor and allocate the proper amount of
		/// memory prior to launch the actual convolution.<para/>
		/// Each dimension h and w of the output images is computed as followed:<para/>
		/// outputDim = 1 + (inputDim + 2*pad - filterDim)/convolutionStride;
		/// </summary>
		/// <param name="inputTensorDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="n">Number of output images.</param>
		/// <param name="c">Number of output feature maps per image.</param>
		/// <param name="h">Height of each output feature map.</param>
		/// <param name="w">Width of each output feature map.</param>
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


		/// <summary>
		/// This function initializes a previously created generic convolution descriptor object into
		/// a n-D correlation. That same convolution descriptor can be reused in the backward path
		/// provided it corresponds to the same layer. The convolution computation will done in the
		/// specified dataType, which can be potentially different from the input/output tensors.
		/// </summary>
		/// <param name="arrayLength">Dimension of the convolution.</param>
		/// <param name="padA">Array of dimension arrayLength containing the zero-padding size
		/// for each dimension. For every dimension, the padding represents the
		/// number of extra zeros implicitly concatenated at the start and at the
		/// end of every element of that dimension.</param>
		/// <param name="filterStrideA">Array of dimension arrayLength containing the filter stride for each
		/// dimension. For every dimension, the fitler stride represents the number
		/// of elements to slide to reach the next start of the filtering window of
		/// the next point.</param>
		/// <param name="upscaleA">Array of dimension arrayLength containing the upscale factor for each dimension.</param>
		/// <param name="mode">Selects between CUDNN_CONVOLUTION and CUDNN_CROSS_CORRELATION.</param>
		/// <param name="dataType">Selects the datatype in which the computation will be done.</param>
		public void SetConvolutionNdDescriptor(int arrayLength,             /* nbDims-2 size */
											int[] padA,
											int[] filterStrideA,
											int[] upscaleA,
											cudnnConvolutionMode mode, cudnnDataType dataType
											)
		{
			res = CudaDNNNativeMethods.cudnnSetConvolutionNdDescriptor(_desc, arrayLength, padA, filterStrideA, upscaleA, mode, dataType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetConvolutionNdDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}
		/// <summary>
		/// This function queries a previously initialized convolution descriptor object.
		/// </summary>
		/// <param name="arrayLengthRequested">Dimension of the expected convolution descriptor. It is also the
		/// minimum size of the arrays padA, filterStrideA and upsacleA in
		/// order to be able to hold the results</param>
		/// <param name="arrayLength">actual dimension of the convolution descriptor.</param>
		/// <param name="padA">Array of dimension of at least arrayLengthRequested that will be
		/// filled with the padding parameters from the provided convolution
		/// descriptor.</param>
		/// <param name="strideA">Array of dimension of at least arrayLengthRequested that will be
		/// filled with the filter stride from the provided convolution descriptor.</param>
		/// <param name="upscaleA">Array of dimension at least arrayLengthRequested that will be filled
		/// with the upscaling parameters from the provided convolution descriptor.</param>
		/// <param name="mode">convolution mode of the provided descriptor.</param>
		/// <param name="dataType">datatype of the provided descriptor.</param>
		public void GetConvolutionNdDescriptor(int arrayLengthRequested,
											ref int arrayLength,
											int[] padA,
											int[] strideA,
											int[] upscaleA,
											ref cudnnConvolutionMode mode, ref cudnnDataType dataType
											)
		{
			res = CudaDNNNativeMethods.cudnnGetConvolutionNdDescriptor(_desc, arrayLengthRequested, ref arrayLength, padA, strideA, upscaleA, ref  mode, ref dataType);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionNdDescriptor", res));
			if (res != cudnnStatus.Success) throw new CudaDNNException(res);
		}


		/// <summary>
		/// This function returns the dimensions of the resulting n-D tensor of a nbDims-2-D
		/// convolution, given the convolution descriptor, the input tensor descriptor and the filter
		/// descriptor This function can help to setup the output tensor and allocate the proper
		/// amount of memory prior to launch the actual convolution.<para/>
		/// Each dimension of the (nbDims-2)-D images of the output tensor is computed as
		/// followed:<para/>
		/// outputDim = 1 + (inputDim + 2*pad - filterDim)/convolutionStride;
		/// </summary>
		/// <param name="inputTensorDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="nbDims">Dimension of the output tensor</param>
		/// <param name="tensorOuputDimA">Array of dimensions nbDims that contains on exit of this routine the sizes
		/// of the output tensor</param>
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
