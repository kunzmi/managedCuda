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
    /// An opaque structure holding the
    /// description of a generic n-D dataset.
    /// </summary>
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

        /// <summary>
        /// This function initializes a previously created generic Tensor descriptor object into a
        /// 4D tensor. The strides of the four dimensions are inferred from the format parameter
        /// and set in such a way that the data is contiguous in memory with no padding between
        /// dimensions.
        /// </summary>
        /// <param name="format">Type of format.</param>
        /// <param name="dataType">Data type.</param>
        /// <param name="n">Number of images.</param>
        /// <param name="c">Number of feature maps per image.</param>
        /// <param name="h">Height of each feature map.</param>
        /// <param name="w">Width of each feature map.</param>
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


        /// <summary>
        /// This function initializes a previously created generic Tensor descriptor object into a
        /// 4D tensor, similarly to cudnnSetTensor4dDescriptor but with the strides explicitly
        /// passed as parameters. This can be used to lay out the 4D tensor in any order or simply to
        /// define gaps between dimensions.
        /// </summary>
        /// <param name="dataType">Data type.</param>
        /// <param name="n">Number of images.</param>
        /// <param name="c">Number of feature maps per image.</param>
        /// <param name="h">Height of each feature map.</param>
        /// <param name="w">Width of each feature map.</param>
        /// <param name="nStride">Stride between two consecutive images.</param>
        /// <param name="cStride">Stride between two consecutive feature maps.</param>
        /// <param name="hStride">Stride between two consecutive rows.</param>
        /// <param name="wStride">Stride between two consecutive columns.</param>
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


        /// <summary>
        /// This function queries the parameters of the previouly initialized Tensor4D descriptor object.
        /// </summary>
        /// <param name="dataType">Data type.</param>
        /// <param name="n">Number of images.</param>
        /// <param name="c">Number of feature maps per image.</param>
        /// <param name="h">Height of each feature map.</param>
        /// <param name="w">Width of each feature map.</param>
        /// <param name="nStride">Stride between two consecutive images.</param>
        /// <param name="cStride">Stride between two consecutive feature maps.</param>
        /// <param name="hStride">Stride between two consecutive rows.</param>
        /// <param name="wStride">Stride between two consecutive columns.</param>
        public void GetTensor4dDescriptor(ref cudnnDataType dataType, // image data type
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


        /// <summary>
        /// This function initializes a previously created generic Tensor descriptor object.
        /// </summary>
        /// <param name="dataType">Data type.</param>
        /// <param name="nbDims">Dimension of the tensor.</param>
        /// <param name="dimA">Array of dimension nbDims that contain the size of the tensor for every dimension.</param>
        /// <param name="strideA">Array of dimension nbDims that contain the stride of the tensor for every dimension.</param>
        public void SetTensorNdDescriptor(cudnnDataType dataType,
                                            int nbDims,
                                            int[] dimA,
                                            int[] strideA
                                            )
        {
            res = CudaDNNNativeMethods.cudnnSetTensorNdDescriptor(_desc, dataType, nbDims, dimA, strideA);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetTensorNdDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function initializes a previously created generic Tensor descriptor object.
        /// </summary>
        /// <param name="tensorDesc">Handle to a previously created tensor descriptor.</param>
        /// <param name="format"></param>
        /// <param name="dataType">Data type.</param>
        /// <param name="nbDims">Dimension of the tensor.</param>
        /// <param name="dimA">Array of dimension nbDims that contain the size of the tensor for every dimension.</param>
        public void SetTensorNdDescriptorEx(
                                cudnnTensorDescriptor tensorDesc,
                                cudnnTensorFormat format,
                                cudnnDataType dataType,
                                int nbDims,
                                int[] dimA)
        {
            res = CudaDNNNativeMethods.cudnnSetTensorNdDescriptorEx(_desc, format, dataType, nbDims, dimA);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetTensorNdDescriptorEx", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function returns the size of the tensor in memory in respect to the given descriptor.
        /// This function can be used to know the amount of GPU memory to be allocated to hold that tensor.
        /// </summary>
        /// <returns>Size in bytes needed to hold the tensor in GPU memory.</returns>
        public SizeT GetTensorSizeInBytes()
        {
            SizeT retVal = 0;
            res = CudaDNNNativeMethods.cudnnGetTensorSizeInBytes(_desc, ref retVal);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetTensorSizeInBytes", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
            return retVal;
        }

        /// <summary>
        /// This function retrieves values stored in a previously initialized Tensor descriptor object.
        /// </summary>
        /// <param name="nbDimsRequested">Number of dimensions to extract from a given tensor descriptor. It is
        /// also the minimum size of the arrays dimA and strideA. If this number is
        /// greater than the resulting nbDims[0], only nbDims[0] dimensions will be
        /// returned.</param>
        /// <param name="dataType">Data type.</param>
        /// <param name="nbDims">Actual number of dimensions of the tensor will be returned in nbDims[0].</param>
        /// <param name="dimA">Array of dimension of at least nbDimsRequested that will be filled with
        /// the dimensions from the provided tensor descriptor.</param>
        /// <param name="strideA">Array of dimension of at least nbDimsRequested that will be filled with
        /// the strides from the provided tensor descriptor.</param>
        public void GetTensorNdDescriptor(int nbDimsRequested,
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
