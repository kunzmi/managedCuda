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
        /// <param name="xDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="x">Pointer to data of the tensor described by the srcDesc descriptor.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the source
        /// value with prior value in the destination tensor as follows: dstValue =
        /// alpha[0]*srcValue + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="yDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="y">Pointer to data of the tensor described by the destDesc descriptor.</param>
        public void TransformTensor(float alpha,
                                            TensorDescriptor xDesc,
                                            CudaDeviceVariable<float> x,
                                            float beta,
                                            TensorDescriptor yDesc,
                                            CudaDeviceVariable<float> y
                                        )
        {
            res = CudaDNNNativeMethods.cudnnTransformTensor(_handle, ref alpha, xDesc.Desc, x.DevicePointer, ref beta, yDesc.Desc, y.DevicePointer);
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
        /// <param name="aDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="a">Pointer to data of the tensor described by the biasDesc descriptor.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the source
        /// value with prior value in the destination tensor as follows: dstValue =
        /// alpha[0]*srcValue + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="cDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="c">Pointer to data of the tensor described by the srcDestDesc descriptor.</param>
        public void AddTensor(float alpha,
                                    TensorDescriptor aDesc,
                                    CudaDeviceVariable<float> a,
                                    float beta,
                                    TensorDescriptor cDesc,
                                    CudaDeviceVariable<float> c
                                    )
        {
            res = CudaDNNNativeMethods.cudnnAddTensor(_handle, ref alpha, aDesc.Desc, a.DevicePointer, ref beta, cDesc.Desc, c.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnAddTensor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function sets all the elements of a tensor to a given value
        /// </summary>
        /// <param name="yDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="y">Pointer to data of the tensor described by the srcDestDesc descriptor.</param>
        /// <param name="value">Pointer in Host memory to a value that all elements of the tensor will be set to.</param>
        public void SetTensor(TensorDescriptor yDesc,
                                    CudaDeviceVariable<float> y,
                                    float value
                                    )
        {
            res = CudaDNNNativeMethods.cudnnSetTensor(_handle, yDesc.Desc, y.DevicePointer, ref value);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetTensor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function scale all the elements of a tensor by a give factor.
        /// </summary>
        /// <param name="yDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="y">Pointer to data of the tensor described by the srcDestDesc descriptor.</param>
        /// <param name="alpha">Pointer in Host memory to a value that all elements of the tensor will be scaled with.</param>
        public void ScaleTensor(TensorDescriptor yDesc,
                                        CudaDeviceVariable<float> y,
                                        float alpha
                                    )
        {
            res = CudaDNNNativeMethods.cudnnScaleTensor(_handle, yDesc.Desc, y.DevicePointer, ref alpha);
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
        /// <param name="xDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="wDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="w">Data pointer to GPU memory associated with the filter descriptor filterDesc.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="algo">Enumerant that specifies which convolution algorithm shoud be used to compute the results</param>
		/// <param name="workSpace">Data pointer to GPU memory to a workspace needed to able to execute
		/// the specified algorithm. If no workspace is needed for a particular
		/// algorithm, that pointer can be nil</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="yDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="y">Data pointer to GPU memory associated with the tensor descriptor
		/// destDesc that carries the result of the convolution.</param>
        public void ConvolutionForward(float alpha,
                                        TensorDescriptor xDesc,
                                        CudaDeviceVariable<float> x,
                                        FilterDescriptor wDesc,
                                        CudaDeviceVariable<float> w,
                                        ConvolutionDescriptor convDesc,
                                        cudnnConvolutionFwdAlgo algo,
                                        CudaDeviceVariable<byte> workSpace,
                                        float beta,
                                        TensorDescriptor yDesc,
                                        CudaDeviceVariable<float> y
                                    )
        {
            res = CudaDNNNativeMethods.cudnnConvolutionForward(_handle, ref alpha, xDesc.Desc, x.DevicePointer, wDesc.Desc, w.DevicePointer, convDesc.Desc, algo, workSpace.DevicePointer, workSpace.SizeInBytes, ref beta, yDesc.Desc, y.DevicePointer);
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
        /// <param name="dyDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="dbDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="db">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        public void ConvolutionBackwardBias(float alpha,
                                            TensorDescriptor dyDesc,
                                            CudaDeviceVariable<float> dy,
                                            float beta,
                                            TensorDescriptor dbDesc,
                                            CudaDeviceVariable<float> db
                                    )
        {
            res = CudaDNNNativeMethods.cudnnConvolutionBackwardBias(_handle, ref alpha, dyDesc.Desc, dy.DevicePointer, ref beta, dbDesc.Desc, db.DevicePointer);
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
        /// <param name="xDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
		/// <param name="dy">Data pointer to GPU memory associated with the input differential tensor descriptor diffDesc.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="algo">Enumerant that specifies which convolution algorithm shoud be used to compute the results</param>
		/// <param name="workSpace">Data pointer to GPU memory to a workspace needed to able to execute
		/// the specified algorithm. If no workspace is needed for a particular
		/// algorithm, that pointer can be nil</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="dwDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="dw">Data pointer to GPU memory associated with the filter descriptor
		/// gradDesc that carries the result.</param>
        public void ConvolutionBackwardFilter(float alpha,
                                                TensorDescriptor xDesc,
                                                CudaDeviceVariable<float> x,
                                                TensorDescriptor dyDesc,
                                                CudaDeviceVariable<float> dy,
                                                ConvolutionDescriptor convDesc,
                                                cudnnConvolutionBwdFilterAlgo algo,
                                                CudaDeviceVariable<byte> workSpace,
                                                float beta,
                                                FilterDescriptor dwDesc,
                                                CudaDeviceVariable<float> dw
                                            )
        {
            res = CudaDNNNativeMethods.cudnnConvolutionBackwardFilter(_handle, ref alpha, xDesc.Desc, x.DevicePointer, dyDesc.Desc, dy.DevicePointer, convDesc.Desc, algo, workSpace.DevicePointer, workSpace.SizeInBytes, ref beta, dwDesc.Desc, dw.DevicePointer);
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
        /// <param name="wDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor filterDesc.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the input differential tensor descriptor diffDesc.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="algo">Enumerant that specifies which backward data convolution algorithm shoud be used to compute the results</param>
        /// <param name="workSpace">Data pointer to GPU memory to a workspace needed to able to execute
        /// the specified algorithm. If no workspace is needed for a particular
        /// algorithm, that pointer can be nil</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="dxDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="dx">Data pointer to GPU memory associated with the output tensor descriptor
        /// gradDesc that carries the result.</param>
        public void ConvolutionBackwardData(float alpha,
                                            FilterDescriptor wDesc,
                                            CudaDeviceVariable<float> w,
                                            TensorDescriptor dyDesc,
                                            CudaDeviceVariable<float> dy,
                                            ConvolutionDescriptor convDesc,
                                            cudnnConvolutionBwdDataAlgo algo,
                                            CudaDeviceVariable<byte> workSpace,
                                            float beta,
                                            TensorDescriptor dxDesc,
                                            CudaDeviceVariable<float> dx
                                        )
        {
            res = CudaDNNNativeMethods.cudnnConvolutionBackwardData(_handle, ref alpha, wDesc.Desc, w.DevicePointer, dyDesc.Desc, dy.DevicePointer, convDesc.Desc, algo, workSpace.DevicePointer, workSpace.SizeInBytes, ref beta, dxDesc.Desc, dx.DevicePointer);
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
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        public void SoftmaxForward(cudnnSoftmaxAlgorithm algorithm,
                                    cudnnSoftmaxMode mode,
                                    float alpha,
                                    TensorDescriptor xDesc,
                                    CudaDeviceVariable<float> x,
                                    float beta,
                                    TensorDescriptor yDesc,
                                    CudaDeviceVariable<float> y
                                    )
        {
            res = CudaDNNNativeMethods.cudnnSoftmaxForward(_handle, algorithm, mode, ref alpha, xDesc.Desc, x.DevicePointer, ref beta, yDesc.Desc, y.DevicePointer);
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
        /// <param name="yDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the tensor descriptor srcDiffData.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="dxDesc">Handle to the previously initialized output differential tensor descriptor.</param>
        /// <param name="dx">Data pointer to GPU memory associated with the output tensor descriptor destDiffDesc.</param>
        public void SoftmaxBackward(cudnnSoftmaxAlgorithm algorithm,
                                    cudnnSoftmaxMode mode,
                                    float alpha,
                                    TensorDescriptor yDesc,
                                    CudaDeviceVariable<float> y,
                                    TensorDescriptor dyDesc,
                                    CudaDeviceVariable<float> dy,
                                    float beta,
                                    TensorDescriptor dxDesc,
                                    CudaDeviceVariable<float> dx
                                    )
        {
            res = CudaDNNNativeMethods.cudnnSoftmaxBackward(_handle, algorithm, mode, ref alpha, yDesc.Desc, y.DevicePointer, dyDesc.Desc, dy.DevicePointer, ref beta, dxDesc.Desc, dx.DevicePointer);
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
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        public void PoolingForward(PoolingDescriptor poolingDesc,
                                    float alpha,
                                    TensorDescriptor xDesc,
                                    CudaDeviceVariable<float> x,
                                    float beta,
                                    TensorDescriptor yDesc,
                                    CudaDeviceVariable<float> y
                                    )
        {
            res = CudaDNNNativeMethods.cudnnPoolingForward(_handle, poolingDesc.Desc, ref alpha, xDesc.Desc, x.DevicePointer, ref beta, yDesc.Desc, y.DevicePointer);
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
        /// <param name="yDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the tensor descriptor srcDiffData.</param>
        /// <param name="xDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="dxDesc">Handle to the previously initialized output differential tensor descriptor.</param>
        /// <param name="dx">Data pointer to GPU memory associated with the output tensor descriptor destDiffDesc.</param>
        public void PoolingBackward(PoolingDescriptor poolingDesc,
                                    float alpha,
                                    TensorDescriptor yDesc,
                                    CudaDeviceVariable<float> y,
                                    TensorDescriptor dyDesc,
                                    CudaDeviceVariable<float> dy,
                                    TensorDescriptor xDesc,
                                    CudaDeviceVariable<float> x,
                                    float beta,
                                    TensorDescriptor dxDesc,
                                    CudaDeviceVariable<float> dx
                                    )
        {
            res = CudaDNNNativeMethods.cudnnPoolingBackward(_handle, poolingDesc.Desc, ref alpha, yDesc.Desc, y.DevicePointer, dyDesc.Desc, dy.DevicePointer, xDesc.Desc, x.DevicePointer, ref beta, dxDesc.Desc, dx.DevicePointer);
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
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        public void ActivationForward(ActivationDescriptor activationDesc,
                                        float alpha,
                                        TensorDescriptor xDesc,
                                        CudaDeviceVariable<float> x,
                                        float beta,
                                        TensorDescriptor yDesc,
                                        CudaDeviceVariable<float> y
                                    )
        {
            res = CudaDNNNativeMethods.cudnnActivationForward(_handle, activationDesc.Desc, ref alpha, xDesc.Desc, x.DevicePointer, ref beta, yDesc.Desc, y.DevicePointer);
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
        /// <param name="yDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the tensor descriptor srcDiffData.</param>
        /// <param name="xDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="dxDesc">Handle to the previously initialized output differential tensor descriptor.</param>
        /// <param name="dx">Data pointer to GPU memory associated with the output tensor descriptor destDiffDesc.</param>
        public void ActivationBackward(ActivationDescriptor activationDesc,
                                        float alpha,
                                        TensorDescriptor yDesc,
                                        CudaDeviceVariable<float> y,
                                        TensorDescriptor dyDesc,
                                        CudaDeviceVariable<float> dy,
                                        TensorDescriptor xDesc,
                                        CudaDeviceVariable<float> x,
                                        float beta,
                                        TensorDescriptor dxDesc,
                                        CudaDeviceVariable<float> dx
                                        )
        {
            res = CudaDNNNativeMethods.cudnnActivationBackward(_handle, activationDesc.Desc, ref alpha, yDesc.Desc, y.DevicePointer, dyDesc.Desc, dy.DevicePointer, xDesc.Desc, x.DevicePointer, ref beta, dxDesc.Desc, dx.DevicePointer);
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
        /// <param name="resultRunningVariance">Running variance tensor (it has the same descriptor as the bias and scale). If this tensors is initially uninitialized, it is required that exponentialAverageFactor=1 is used for the very first call of a complete training cycle. This is necessary to properly initialize the moving average. Both resultRunningMean and resultRunningInvVariance can be NULL but only at the same time. The value stored in resultRunningInvVariance (or passed as an input in inference mode) is the moving average of the expression 1 / sqrt(eps+variance[x]) where variance is computed either over batch or spatial+batch dimensions depending on the mode. </param>
        /// <param name="epsilon">Epsilon value used in the batch normalization formula. Minimum allowed value is currently 1e-5. Same epsilon value should be used in forward and backward functions.</param>
        /// <param name="resultSaveMean">Optional cache to save intermediate results computed during the forward pass - these can then be reused to speed up the backward pass. For this to work correctly, the bottom layer data has to remain unchanged until the backward function is called. Note that both resultSaveMean and resultSaveInvVariance can be NULL but only at the same time. It is recommended to use this cache since memory overhead is relatively small because these tensors have a much lower product of dimensions than the data tensors.</param>
        /// <param name="resultSaveVariance">Optional cache to save intermediate results computed during the forward pass - these can then be reused to speed up the backward pass. For this to work correctly, the bottom layer data has to remain unchanged until the backward function is called. Note that both resultSaveMean and resultSaveInvVariance can be NULL but only at the same time. It is recommended to use this cache since memory overhead is relatively small because these tensors have a much lower product of dimensions than the data tensors.</param>
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
                                CudaDeviceVariable<float> resultRunningVariance,

                                /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
                                double epsilon,

                                /* Optionally save intermediate results from the forward pass here
                                   - can be reused to speed up backward pass. NULL if unused */
                                CudaDeviceVariable<float> resultSaveMean,
                                CudaDeviceVariable<float> resultSaveVariance)
        {
            res = CudaDNNNativeMethods.cudnnBatchNormalizationForwardTraining(
                _handle, mode, ref alpha, ref beta, xDesc.Desc, x.DevicePointer, yDesc.Desc, y.DevicePointer,
                bnScaleBiasMeanVarDesc.Desc, bnScale.DevicePointer, bnBias.DevicePointer, exponentialAverageFactor,
                resultRunningMean.DevicePointer, resultRunningVariance.DevicePointer, epsilon, resultSaveMean.DevicePointer, resultSaveVariance.DevicePointer);
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
        /// <param name="estimatedVariance">Variance tensor (has the same descriptor as the bias and scale). It is suggested that resultRunningVariance from the cudnnBatchNormalizationForwardTraining call accumulated during the training phase be passed as input here.</param>
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
                                        CudaDeviceVariable<float> estimatedVariance,
                                        double epsilon)
        {
            res = CudaDNNNativeMethods.cudnnBatchNormalizationForwardInference(
                _handle, mode, ref alpha, ref beta, xDesc.Desc, x.DevicePointer, yDesc.Desc, y.DevicePointer,
                bnScaleBiasMeanVarDesc.Desc, bnScale.DevicePointer, bnBias.DevicePointer, estimatedMean.DevicePointer, estimatedVariance.DevicePointer, epsilon);
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

        /// <summary>
        /// This function implements the equation C = op(alpha1[0] * A, alpha2[0] * B) + beta[0] * C, 
        /// given tensors A, B, and C and scaling factors alpha1, alpha2, and beta.The op to use is 
        /// indicated by the descriptor opTensorDesc.Currently-supported ops are listed by the 
        /// cudnnOpTensorOp_t enum. Each dimension of the input tensor A must match the corresponding 
        /// dimension of the destination tensor C, and each dimension of the input tensor B must match 
        /// the corresponding dimension of the destination tensor C or must be equal to 1. In the latter 
        /// case, the same value from the input tensor B for those dimensions will be used to blend into the 
        /// C tensor.The data types of the input tensors A and B must match. If the data type of the 
        /// destination tensor C is double, then the data type of the input tensors also must be double. If 
        /// the data type of the destination tensor C is double, then opTensorCompType in opTensorDesc must 
        /// be double. Else opTensorCompType must be float. If the input tensor B is the same tensor as 
        /// the destination tensor C, then the input tensor A also must be the same tensor as the 
        /// destination tensor C.
        /// </summary>
        /// <param name="op_desc">Handle to a previously initialized op tensor descriptor.</param>
        /// <param name="alpha1">Pointer to the scaling factor(in host memory) used to blend the source value with prior 
        /// value in the destination tensor as indicated by the above op equation.</param>
        /// <param name="a_desc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="a">Pointer to data of the tensor described by the a_desc.</param>
        /// <param name="alpha2">Pointer to the scaling factor(in host memory) used to blend the source value with prior 
        /// value in the destination tensor as indicated by the above op equation.</param>
        /// <param name="b_desc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="b">Pointer to data of the tensor described by the b_desc.</param>
        /// <param name="beta">Pointer to the scaling factor(in host memory) used to blend the source value with prior 
        /// value in the destination tensor as indicated by the above op equation.</param>
        /// <param name="c_desc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="c">Output pointer to data of the tensor described by the c_desc.</param>
        public void OpTensor(OpTensorDescriptor op_desc,
            float alpha1, TensorDescriptor a_desc, CudaDeviceVariable<float> a,
            float alpha2, TensorDescriptor b_desc, CudaDeviceVariable<float> b,
            float beta, TensorDescriptor c_desc, CudaDeviceVariable<float> c)
        {
            res = CudaDNNNativeMethods.cudnnOpTensor(_handle, op_desc.Desc,
                ref alpha1, a_desc.Desc, a.DevicePointer,
                ref alpha2, b_desc.Desc, b.DevicePointer,
                ref beta, c_desc.Desc, c.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "BatchNormalizationBackward", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }


        /// <summary>
        /// This function applies a bias and then an activation to the convolutions or crosscorrelations
        /// of cudnnConvolutionForward(), returning results in y.The full computation
        /// follows the equation y = act(alpha1* conv(x) + alpha2* z + bias ).<para/>
        /// The routine cudnnGetConvolution2dForwardOutputDim or
        /// cudnnGetConvolutionNdForwardOutputDim can be used to determine the proper
        /// dimensions of the output tensor descriptor yDesc with respect to xDesc, convDesc and wDesc.
        /// </summary>
        /// <param name="alpha1">Pointers to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as described by the above equation.</param>
        /// <param name="xDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor xDesc.</param>
        /// <param name="wDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor wDesc.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="algo">Enumerant that specifies which convolution algorithm shoud be used to compute the results</param>
        /// <param name="workSpace">Data pointer to GPU memory to a workspace needed to able to execute
        /// the specified algorithm.If no workspace is needed for a particular
        /// algorithm, that pointer can be nil</param>
        /// <param name="alpha2">Pointers to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as described by the above equation.</param>
        /// <param name="zDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="z">Data pointer to GPU memory associated with the tensor descriptor zDesc.</param>
        /// <param name="biasDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="bias">Data pointer to GPU memory associated with the tensor descriptor biasDesc.</param>
        /// <param name="activationDesc">Handle to a previously initialized activation descriptor.</param>
        /// <param name="yDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the tensor descriptor yDesc
        /// that carries the result of the convolution.</param>
        public void ConvolutionBiasActivationForward(
                                float alpha1,
                                TensorDescriptor xDesc,
                                CudaDeviceVariable<float> x,
                                FilterDescriptor wDesc,
                                CudaDeviceVariable<float> w,
                                ConvolutionDescriptor convDesc,
                                cudnnConvolutionFwdAlgo algo,
                                CudaDeviceVariable<byte> workSpace,
                                float alpha2,
                                TensorDescriptor zDesc,
                                CudaDeviceVariable<float> z,
                                TensorDescriptor biasDesc,
                                CudaDeviceVariable<float> bias,
                                ActivationDescriptor activationDesc,
                                TensorDescriptor yDesc,
                                CudaDeviceVariable<float> y)
        {
            res = CudaDNNNativeMethods.cudnnConvolutionBiasActivationForward(_handle, ref alpha1,
                xDesc.Desc, x.DevicePointer,
                wDesc.Desc, w.DevicePointer,
                convDesc.Desc, algo, workSpace.DevicePointer, workSpace.SizeInBytes,
                ref alpha2, zDesc.Desc, z.DevicePointer,
                biasDesc.Desc, bias.DevicePointer, activationDesc.Desc, 
                yDesc.Desc, y.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnConvolutionBiasActivationForward", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function reduces tensor A by implementing the equation C = alpha * reduce op ( A )
        /// + beta* C, given tensors A and C and scaling factors alpha and beta.The reduction op
        /// to use is indicated by the descriptor reduceTensorDesc.Currently-supported ops are
        /// listed by the cudnnReduceTensorOp_t enum.
        /// </summary>
        /// <param name="reduceTensorDesc">Handle to a previously initialized reduce tensor descriptor.</param>
        /// <param name="indices">Handle to a previously allocated space for writing indices.</param>
        /// <param name="workspace">Handle to a previously allocated space for the reduction implementation.</param>
        /// <param name="workspaceSizeInBytes">Size of the above previously allocated space.</param>
        /// <param name="alpha">Pointer to scaling factor (in host memory) used to blend the source value
        /// with prior value in the destination tensor as indicated by the above op equation.</param>
        /// <param name="aDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="A">Pointer to data of the tensor described by the aDesc descriptor.</param>
        /// <param name="beta">Pointer to scaling factor (in host memory) used to blend the source value
        /// with prior value in the destination tensor as indicated by the above op equation.</param>
        /// <param name="cDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="C">Pointer to data of the tensor described by the cDesc descriptor.</param>
        public void ReduceTensor(
                                ReduceTensorDescriptor reduceTensorDesc,
                                CudaDeviceVariable<uint> indices,
                                CudaDeviceVariable<byte> workspace,
                                SizeT workspaceSizeInBytes,
                                float alpha,
                                TensorDescriptor aDesc,
                                CudaDeviceVariable<float> A,
                                float beta,
                                TensorDescriptor cDesc,
                                CudaDeviceVariable<float> C)
        {
            res = CudaDNNNativeMethods.cudnnReduceTensor(_handle,
                reduceTensorDesc.Desc, indices.DevicePointer, indices.SizeInBytes,
                workspace.DevicePointer, workspace.SizeInBytes,
                ref alpha, aDesc.Desc, A.DevicePointer, 
                ref beta, cDesc.Desc, C.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnReduceTensor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }


        /// <summary>  
        /// This function performs forward dropout operation over x returning results in y. If dropout was   
        /// used as a parameter to cudnnSetDropoutDescriptor, the approximately dropout fraction of x values   
        /// will be replaces by 0, and the rest will be scaled by 1/(1-dropout) This function should not be   
        /// running concurrently with another cudnnDropoutForward function using the same states.  
        /// </summary>  
        /// <param name="dropoutDesc">Handle to a previously created dropout descriptor object.</param>  
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>  
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>  
        /// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>  
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>  
        /// <param name="reserveSpace">Data pointer to GPU memory used by this function. It is expected that contents of reserveSpace doe not change between cudnnDropoutForward and cudnnDropoutBackward calls.</param>  
        public void DropoutForward(DropoutDescriptor dropoutDesc,
                                   TensorDescriptor xDesc,
                                   CudaDeviceVariable<float> x,
                                   TensorDescriptor yDesc,
                                   CudaDeviceVariable<float> y,
                                   CudaDeviceVariable<byte> reserveSpace)
        {
            res = CudaDNNNativeMethods.cudnnDropoutForward(_handle, dropoutDesc.Desc, xDesc.Desc, x.DevicePointer, yDesc.Desc, y.DevicePointer, reserveSpace.DevicePointer, reserveSpace.SizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDropoutForward", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>  
        /// This function performs backward dropout operation over dy returning results in dx. If during   
        /// forward dropout operation value from x was propagated to y then during backward operation value   
        /// from dy will be propagated to dx, otherwise, dx value will be set to 0.  
        /// </summary>  
        /// <param name="dropoutDesc">Handle to a previously created dropout descriptor object.</param>  
        /// <param name="dyDesc">Handle to a previously initialized tensor descriptor.</param>  
        /// <param name="dy">Pointer to data of the tensor described by the dyDesc descriptor.</param>  
        /// <param name="dxDesc">Handle to a previously initialized tensor descriptor.</param>  
        /// <param name="dx">Pointer to data of the tensor described by the dxDesc descriptor.</param>  
        /// <param name="reserveSpace">Data pointer to GPU memory used by this function. It is expected that contents of reserveSpace doe not change between cudnnDropoutForward and cudnnDropoutBackward calls.</param>  
        public void DropoutBackward(DropoutDescriptor dropoutDesc,
                                    TensorDescriptor dyDesc,
                                    CudaDeviceVariable<float> dy,
                                    TensorDescriptor dxDesc,
                                    CudaDeviceVariable<float> dx,
                                    CudaDeviceVariable<byte> reserveSpace)
        {
            res = CudaDNNNativeMethods.cudnnDropoutBackward(_handle, dropoutDesc.Desc, dyDesc.Desc, dy.DevicePointer, dxDesc.Desc, dx.DevicePointer, reserveSpace.DevicePointer, reserveSpace.SizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDropoutBackward", res));
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
        /// <param name="xDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="x">Pointer to data of the tensor described by the srcDesc descriptor.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the source
        /// value with prior value in the destination tensor as follows: dstValue =
        /// alpha[0]*srcValue + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="yDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="y">Pointer to data of the tensor described by the destDesc descriptor.</param>
        public void TransformTensor(double alpha,
                                            TensorDescriptor xDesc,
                                            CudaDeviceVariable<double> x,
                                            double beta,
                                            TensorDescriptor yDesc,
                                            CudaDeviceVariable<double> y
                                        )
        {
            res = CudaDNNNativeMethods.cudnnTransformTensor(_handle, ref alpha, xDesc.Desc, x.DevicePointer, ref beta, yDesc.Desc, y.DevicePointer);
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
        /// <param name="aDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="a">Pointer to data of the tensor described by the biasDesc descriptor.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the source
        /// value with prior value in the destination tensor as follows: dstValue =
        /// alpha[0]*srcValue + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="cDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="c">Pointer to data of the tensor described by the srcDestDesc descriptor.</param>
        public void AddTensor(double alpha,
                                    TensorDescriptor aDesc,
                                    CudaDeviceVariable<double> a,
                                    double beta,
                                    TensorDescriptor cDesc,
                                    CudaDeviceVariable<double> c
                                    )
        {
            res = CudaDNNNativeMethods.cudnnAddTensor(_handle, ref alpha, aDesc.Desc, a.DevicePointer, ref beta, cDesc.Desc, c.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnAddTensor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function sets all the elements of a tensor to a given value
        /// </summary>
        /// <param name="yDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="y">Pointer to data of the tensor described by the srcDestDesc descriptor.</param>
        /// <param name="value">Pointer in Host memory to a value that all elements of the tensor will be set to.</param>
        public void SetTensor(TensorDescriptor yDesc,
                                    CudaDeviceVariable<double> y,
                                    double value
                                    )
        {
            res = CudaDNNNativeMethods.cudnnSetTensor(_handle, yDesc.Desc, y.DevicePointer, ref value);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetTensor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function scale all the elements of a tensor by a give factor.
        /// </summary>
        /// <param name="yDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="y">Pointer to data of the tensor described by the srcDestDesc descriptor.</param>
        /// <param name="alpha">Pointer in Host memory to a value that all elements of the tensor will be scaled with.</param>
        public void ScaleTensor(TensorDescriptor yDesc,
                                        CudaDeviceVariable<double> y,
                                        double alpha
                                    )
        {
            res = CudaDNNNativeMethods.cudnnScaleTensor(_handle, yDesc.Desc, y.DevicePointer, ref alpha);
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
        /// <param name="xDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="wDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="w">Data pointer to GPU memory associated with the filter descriptor filterDesc.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="algo">Enumerant that specifies which convolution algorithm shoud be used to compute the results</param>
		/// <param name="workSpace">Data pointer to GPU memory to a workspace needed to able to execute
		/// the specified algorithm. If no workspace is needed for a particular
		/// algorithm, that pointer can be nil</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="yDesc">Handle to a previously initialized tensor descriptor.</param>
		/// <param name="y">Data pointer to GPU memory associated with the tensor descriptor
		/// destDesc that carries the result of the convolution.</param>
        public void ConvolutionForward(double alpha,
                                        TensorDescriptor xDesc,
                                        CudaDeviceVariable<double> x,
                                        FilterDescriptor wDesc,
                                        CudaDeviceVariable<double> w,
                                        ConvolutionDescriptor convDesc,
                                        cudnnConvolutionFwdAlgo algo,
                                        CudaDeviceVariable<byte> workSpace,
                                        double beta,
                                        TensorDescriptor yDesc,
                                        CudaDeviceVariable<double> y
                                    )
        {
            res = CudaDNNNativeMethods.cudnnConvolutionForward(_handle, ref alpha, xDesc.Desc, x.DevicePointer, wDesc.Desc, w.DevicePointer, convDesc.Desc, algo, workSpace.DevicePointer, workSpace.SizeInBytes, ref beta, yDesc.Desc, y.DevicePointer);
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
        /// <param name="dyDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="dbDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="db">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        public void ConvolutionBackwardBias(double alpha,
                                            TensorDescriptor dyDesc,
                                            CudaDeviceVariable<double> dy,
                                            double beta,
                                            TensorDescriptor dbDesc,
                                            CudaDeviceVariable<double> db
                                    )
        {
            res = CudaDNNNativeMethods.cudnnConvolutionBackwardBias(_handle, ref alpha, dyDesc.Desc, dy.DevicePointer, ref beta, dbDesc.Desc, db.DevicePointer);
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
        /// <param name="xDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the input differential tensor descriptor diffDesc.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="algo">Enumerant that specifies which convolution algorithm shoud be used to compute the results</param>
        /// <param name="workSpace">Data pointer to GPU memory to a workspace needed to able to execute
        /// the specified algorithm. If no workspace is needed for a particular
        /// algorithm, that pointer can be nil</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="dwDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="dw">Data pointer to GPU memory associated with the filter descriptor
        /// gradDesc that carries the result.</param>    
        public void ConvolutionBackwardFilter(double alpha,
                                                TensorDescriptor xDesc,
                                                CudaDeviceVariable<double> x,
                                                TensorDescriptor dyDesc,
                                                CudaDeviceVariable<double> dy,
                                                ConvolutionDescriptor convDesc,
                                                cudnnConvolutionBwdFilterAlgo algo,
                                                CudaDeviceVariable<byte> workSpace,
                                                double beta,
                                                FilterDescriptor dwDesc,
                                                CudaDeviceVariable<double> dw
                                            )
        {
            res = CudaDNNNativeMethods.cudnnConvolutionBackwardFilter(_handle, ref alpha, xDesc.Desc, x.DevicePointer, dyDesc.Desc, dy.DevicePointer, convDesc.Desc, algo, workSpace.DevicePointer, workSpace.SizeInBytes, ref beta, dwDesc.Desc, dw.DevicePointer);
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
        /// <param name="wDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor filterDesc.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the input differential tensor descriptor diffDesc.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="algo">Enumerant that specifies which backward data convolution algorithm shoud be used to compute the results</param>
        /// <param name="workSpace">Data pointer to GPU memory to a workspace needed to able to execute
        /// the specified algorithm. If no workspace is needed for a particular
        /// algorithm, that pointer can be nil</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="dxDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="dx">Data pointer to GPU memory associated with the output tensor descriptor
        /// gradDesc that carries the result.</param>
        public void ConvolutionBackwardData(double alpha,
                                            FilterDescriptor wDesc,
                                            CudaDeviceVariable<double> w,
                                            TensorDescriptor dyDesc,
                                            CudaDeviceVariable<double> dy,
                                            ConvolutionDescriptor convDesc,
                                            cudnnConvolutionBwdDataAlgo algo,
                                            CudaDeviceVariable<byte> workSpace,
                                            double beta,
                                            TensorDescriptor dxDesc,
                                            CudaDeviceVariable<double> dx
                                        )
        {
            res = CudaDNNNativeMethods.cudnnConvolutionBackwardData(_handle, ref alpha, wDesc.Desc, w.DevicePointer, dyDesc.Desc, dy.DevicePointer, convDesc.Desc, algo, workSpace.DevicePointer, workSpace.SizeInBytes, ref beta, dxDesc.Desc, dx.DevicePointer);
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
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
		/// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
		/// result with prior value in the output layer as follows: dstValue =
		/// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
		/// additional details.</param>
		/// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>
		/// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        public void SoftmaxForward(cudnnSoftmaxAlgorithm algorithm,
                                    cudnnSoftmaxMode mode,
                                    double alpha,
                                    TensorDescriptor xDesc,
                                    CudaDeviceVariable<double> x,
                                    double beta,
                                    TensorDescriptor yDesc,
                                    CudaDeviceVariable<double> y
                                    )
        {
            res = CudaDNNNativeMethods.cudnnSoftmaxForward(_handle, algorithm, mode, ref alpha, xDesc.Desc, x.DevicePointer, ref beta, yDesc.Desc, y.DevicePointer);
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
        /// <param name="yDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the tensor descriptor srcDiffData.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="dxDesc">Handle to the previously initialized output differential tensor descriptor.</param>
        /// <param name="dx">Data pointer to GPU memory associated with the output tensor descriptor destDiffDesc.</param>
        public void SoftmaxBackward(cudnnSoftmaxAlgorithm algorithm,
                                    cudnnSoftmaxMode mode,
                                    double alpha,
                                    TensorDescriptor yDesc,
                                    CudaDeviceVariable<double> y,
                                    TensorDescriptor dyDesc,
                                    CudaDeviceVariable<double> dy,
                                    double beta,
                                    TensorDescriptor dxDesc,
                                    CudaDeviceVariable<double> dx
                                    )
        {
            res = CudaDNNNativeMethods.cudnnSoftmaxBackward(_handle, algorithm, mode, ref alpha, yDesc.Desc, y.DevicePointer, dyDesc.Desc, dy.DevicePointer, ref beta, dxDesc.Desc, dx.DevicePointer);
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
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        public void PoolingForward(PoolingDescriptor poolingDesc,
                                    double alpha,
                                    TensorDescriptor xDesc,
                                    CudaDeviceVariable<double> x,
                                    double beta,
                                    TensorDescriptor yDesc,
                                    CudaDeviceVariable<double> y
                                    )
        {
            res = CudaDNNNativeMethods.cudnnPoolingForward(_handle, poolingDesc.Desc, ref alpha, xDesc.Desc, x.DevicePointer, ref beta, yDesc.Desc, y.DevicePointer);
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
        /// <param name="yDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the tensor descriptor srcDiffData.</param>
        /// <param name="xDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="dxDesc">Handle to the previously initialized output differential tensor descriptor.</param>
        /// <param name="dx">Data pointer to GPU memory associated with the output tensor descriptor destDiffDesc.</param>
        public void PoolingBackward(PoolingDescriptor poolingDesc,
                                    double alpha,
                                    TensorDescriptor yDesc,
                                    CudaDeviceVariable<double> y,
                                    TensorDescriptor dyDesc,
                                    CudaDeviceVariable<double> dy,
                                    TensorDescriptor xDesc,
                                    CudaDeviceVariable<double> x,
                                    double beta,
                                    TensorDescriptor dxDesc,
                                    CudaDeviceVariable<double> dx
                                    )
        {
            res = CudaDNNNativeMethods.cudnnPoolingBackward(_handle, poolingDesc.Desc, ref alpha, yDesc.Desc, y.DevicePointer, dyDesc.Desc, dy.DevicePointer, xDesc.Desc, x.DevicePointer, ref beta, dxDesc.Desc, dx.DevicePointer);
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
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        public void ActivationForward(ActivationDescriptor activationDesc,
                                        double alpha,
                                        TensorDescriptor xDesc,
                                        CudaDeviceVariable<double> x,
                                        double beta,
                                        TensorDescriptor yDesc,
                                        CudaDeviceVariable<double> y
                                    )
        {
            res = CudaDNNNativeMethods.cudnnActivationForward(_handle, activationDesc.Desc, ref alpha, xDesc.Desc, x.DevicePointer, ref beta, yDesc.Desc, y.DevicePointer);
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
        /// <param name="yDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the tensor descriptor srcDiffData.</param>
        /// <param name="xDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>
        /// <param name="beta">Pointer to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as follows: dstValue =
        /// alpha[0]*result + beta[0]*priorDstValue. Please refer to this section for
        /// additional details.</param>
        /// <param name="dxDesc">Handle to the previously initialized output differential tensor descriptor.</param>
        /// <param name="dx">Data pointer to GPU memory associated with the output tensor descriptor destDiffDesc.</param>
        public void ActivationBackward(ActivationDescriptor activationDesc,
                                        double alpha,
                                        TensorDescriptor yDesc,
                                        CudaDeviceVariable<double> y,
                                        TensorDescriptor dyDesc,
                                        CudaDeviceVariable<double> dy,
                                        TensorDescriptor xDesc,
                                        CudaDeviceVariable<double> x,
                                        double beta,
                                        TensorDescriptor dxDesc,
                                        CudaDeviceVariable<double> dx
                                        )
        {
            res = CudaDNNNativeMethods.cudnnActivationBackward(_handle, activationDesc.Desc, ref alpha, yDesc.Desc, y.DevicePointer, dyDesc.Desc, dy.DevicePointer, xDesc.Desc, x.DevicePointer, ref beta, dxDesc.Desc, dx.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnActivationForward", res));
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
        /// <param name="resultRunningVariance">Running variance tensor (it has the same descriptor as the bias and scale). If this tensors is initially uninitialized, it is required that exponentialAverageFactor=1 is used for the very first call of a complete training cycle. This is necessary to properly initialize the moving average. Both resultRunningMean and resultRunningInvVariance can be NULL but only at the same time. The value stored in resultRunningInvVariance (or passed as an input in inference mode) is the moving average of the expression 1 / sqrt(eps+variance[x]) where variance is computed either over batch or spatial+batch dimensions depending on the mode. </param>
        /// <param name="epsilon">Epsilon value used in the batch normalization formula. Minimum allowed value is currently 1e-5. Same epsilon value should be used in forward and backward functions.</param>
        /// <param name="resultSaveMean">Optional cache to save intermediate results computed during the forward pass - these can then be reused to speed up the backward pass. For this to work correctly, the bottom layer data has to remain unchanged until the backward function is called. Note that both resultSaveMean and resultSaveInvVariance can be NULL but only at the same time. It is recommended to use this cache since memory overhead is relatively small because these tensors have a much lower product of dimensions than the data tensors.</param>
        /// <param name="resultSaveVariance">Optional cache to save intermediate results computed during the forward pass - these can then be reused to speed up the backward pass. For this to work correctly, the bottom layer data has to remain unchanged until the backward function is called. Note that both resultSaveMean and resultSaveInvVariance can be NULL but only at the same time. It is recommended to use this cache since memory overhead is relatively small because these tensors have a much lower product of dimensions than the data tensors.</param>
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
                                CudaDeviceVariable<double> resultRunningVariance,

                                /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
                                double epsilon,

                                /* Optionally save intermediate results from the forward pass here
                                   - can be reused to speed up backward pass. NULL if unused */
                                CudaDeviceVariable<double> resultSaveMean,
                                CudaDeviceVariable<double> resultSaveVariance)
        {
            res = CudaDNNNativeMethods.cudnnBatchNormalizationForwardTraining(
                _handle, mode, ref alpha, ref beta, xDesc.Desc, x.DevicePointer, yDesc.Desc, y.DevicePointer,
                bnScaleBiasMeanVarDesc.Desc, bnScale.DevicePointer, bnBias.DevicePointer, exponentialAverageFactor,
                resultRunningMean.DevicePointer, resultRunningVariance.DevicePointer, epsilon, resultSaveMean.DevicePointer, resultSaveVariance.DevicePointer);
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
        /// <param name="estimatedVariance">Variance tensor (has the same descriptor as the bias and scale). It is suggested that resultRunningVariance from the cudnnBatchNormalizationForwardTraining call accumulated during the training phase be passed as input here.</param>
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
                                        CudaDeviceVariable<double> estimatedVariance,
                                        double epsilon)
        {
            res = CudaDNNNativeMethods.cudnnBatchNormalizationForwardInference(
                _handle, mode, ref alpha, ref beta, xDesc.Desc, x.DevicePointer, yDesc.Desc, y.DevicePointer,
                bnScaleBiasMeanVarDesc.Desc, bnScale.DevicePointer, bnBias.DevicePointer, estimatedMean.DevicePointer, estimatedVariance.DevicePointer, epsilon);
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

        /// <summary>
        /// This function implements the equation C = op(alpha1[0] * A, alpha2[0] * B) + beta[0] * C, 
        /// given tensors A, B, and C and scaling factors alpha1, alpha2, and beta.The op to use is 
        /// indicated by the descriptor opTensorDesc.Currently-supported ops are listed by the 
        /// cudnnOpTensorOp_t enum. Each dimension of the input tensor A must match the corresponding 
        /// dimension of the destination tensor C, and each dimension of the input tensor B must match 
        /// the corresponding dimension of the destination tensor C or must be equal to 1. In the latter 
        /// case, the same value from the input tensor B for those dimensions will be used to blend into the 
        /// C tensor.The data types of the input tensors A and B must match. If the data type of the 
        /// destination tensor C is double, then the data type of the input tensors also must be double. If 
        /// the data type of the destination tensor C is double, then opTensorCompType in opTensorDesc must 
        /// be double. Else opTensorCompType must be float. If the input tensor B is the same tensor as 
        /// the destination tensor C, then the input tensor A also must be the same tensor as the 
        /// destination tensor C.
        /// </summary>
        /// <param name="op_desc">Handle to a previously initialized op tensor descriptor.</param>
        /// <param name="alpha1">Pointer to the scaling factor(in host memory) used to blend the source value with prior 
        /// value in the destination tensor as indicated by the above op equation.</param>
        /// <param name="a_desc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="a">Pointer to data of the tensor described by the a_desc.</param>
        /// <param name="alpha2">Pointer to the scaling factor(in host memory) used to blend the source value with prior 
        /// value in the destination tensor as indicated by the above op equation.</param>
        /// <param name="b_desc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="b">Pointer to data of the tensor described by the b_desc.</param>
        /// <param name="beta">Pointer to the scaling factor(in host memory) used to blend the source value with prior 
        /// value in the destination tensor as indicated by the above op equation.</param>
        /// <param name="c_desc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="c">Output pointer to data of the tensor described by the c_desc.</param>
        public void OpTensor(OpTensorDescriptor op_desc,
            double alpha1, TensorDescriptor a_desc, CudaDeviceVariable<double> a,
            double alpha2, TensorDescriptor b_desc, CudaDeviceVariable<double> b,
            double beta, TensorDescriptor c_desc, CudaDeviceVariable<double> c)
        {
            res = CudaDNNNativeMethods.cudnnOpTensor(_handle, op_desc.Desc,
                ref alpha1, a_desc.Desc, a.DevicePointer,
                ref alpha2, b_desc.Desc, b.DevicePointer,
                ref beta, c_desc.Desc, c.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "BatchNormalizationBackward", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }


        /// <summary>
        /// This function applies a bias and then an activation to the convolutions or crosscorrelations
        /// of cudnnConvolutionForward(), returning results in y.The full computation
        /// follows the equation y = act(alpha1* conv(x) + alpha2* z + bias ).<para/>
        /// The routine cudnnGetConvolution2dForwardOutputDim or
        /// cudnnGetConvolutionNdForwardOutputDim can be used to determine the proper
        /// dimensions of the output tensor descriptor yDesc with respect to xDesc, convDesc and wDesc.
        /// </summary>
        /// <param name="alpha1">Pointers to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as described by the above equation.</param>
        /// <param name="xDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor xDesc.</param>
        /// <param name="wDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor wDesc.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="algo">Enumerant that specifies which convolution algorithm shoud be used to compute the results</param>
        /// <param name="workSpace">Data pointer to GPU memory to a workspace needed to able to execute
        /// the specified algorithm.If no workspace is needed for a particular
        /// algorithm, that pointer can be nil</param>
        /// <param name="alpha2">Pointers to scaling factors (in host memory) used to blend the computation
        /// result with prior value in the output layer as described by the above equation.</param>
        /// <param name="zDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="z">Data pointer to GPU memory associated with the tensor descriptor zDesc.</param>
        /// <param name="biasDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="bias">Data pointer to GPU memory associated with the tensor descriptor biasDesc.</param>
        /// <param name="activationDesc">Handle to a previously initialized activation descriptor.</param>
        /// <param name="yDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="y">Data pointer to GPU memory associated with the tensor descriptor yDesc
        /// that carries the result of the convolution.</param>
        public void ConvolutionBiasActivationForward(
                                double alpha1,
                                TensorDescriptor xDesc,
                                CudaDeviceVariable<double> x,
                                FilterDescriptor wDesc,
                                CudaDeviceVariable<double> w,
                                ConvolutionDescriptor convDesc,
                                cudnnConvolutionFwdAlgo algo,
                                CudaDeviceVariable<byte> workSpace,
                                double alpha2,
                                TensorDescriptor zDesc,
                                CudaDeviceVariable<double> z,
                                TensorDescriptor biasDesc,
                                CudaDeviceVariable<double> bias,
                                ActivationDescriptor activationDesc,
                                TensorDescriptor yDesc,
                                CudaDeviceVariable<double> y)
        {
            res = CudaDNNNativeMethods.cudnnConvolutionBiasActivationForward(_handle, ref alpha1,
                xDesc.Desc, x.DevicePointer,
                wDesc.Desc, w.DevicePointer,
                convDesc.Desc, algo, workSpace.DevicePointer, workSpace.SizeInBytes,
                ref alpha2, zDesc.Desc, z.DevicePointer,
                biasDesc.Desc, bias.DevicePointer, activationDesc.Desc,
                yDesc.Desc, y.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnConvolutionBiasActivationForward", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function reduces tensor A by implementing the equation C = alpha * reduce op ( A )
        /// + beta* C, given tensors A and C and scaling factors alpha and beta.The reduction op
        /// to use is indicated by the descriptor reduceTensorDesc.Currently-supported ops are
        /// listed by the cudnnReduceTensorOp_t enum.
        /// </summary>
        /// <param name="reduceTensorDesc">Handle to a previously initialized reduce tensor descriptor.</param>
        /// <param name="indices">Handle to a previously allocated space for writing indices.</param>
        /// <param name="workspace">Handle to a previously allocated space for the reduction implementation.</param>
        /// <param name="workspaceSizeInBytes">Size of the above previously allocated space.</param>
        /// <param name="alpha">Pointer to scaling factor (in host memory) used to blend the source value
        /// with prior value in the destination tensor as indicated by the above op equation.</param>
        /// <param name="aDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="A">Pointer to data of the tensor described by the aDesc descriptor.</param>
        /// <param name="beta">Pointer to scaling factor (in host memory) used to blend the source value
        /// with prior value in the destination tensor as indicated by the above op equation.</param>
        /// <param name="cDesc">Handle to a previously initialized tensor descriptor.</param>
        /// <param name="C">Pointer to data of the tensor described by the cDesc descriptor.</param>
        public void ReduceTensor(
                                ReduceTensorDescriptor reduceTensorDesc,
                                CudaDeviceVariable<uint> indices,
                                CudaDeviceVariable<byte> workspace,
                                SizeT workspaceSizeInBytes,
                                double alpha,
                                TensorDescriptor aDesc,
                                CudaDeviceVariable<double> A,
                                double beta,
                                TensorDescriptor cDesc,
                                CudaDeviceVariable<double> C)
        {
            res = CudaDNNNativeMethods.cudnnReduceTensor(_handle,
                reduceTensorDesc.Desc, indices.DevicePointer, indices.SizeInBytes,
                workspace.DevicePointer, workspace.SizeInBytes,
                ref alpha, aDesc.Desc, A.DevicePointer,
                ref beta, cDesc.Desc, C.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnReduceTensor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }


        /// <summary>  
        /// This function performs forward dropout operation over x returning results in y. If dropout was   
        /// used as a parameter to cudnnSetDropoutDescriptor, the approximately dropout fraction of x values   
        /// will be replaces by 0, and the rest will be scaled by 1/(1-dropout) This function should not be   
        /// running concurrently with another cudnnDropoutForward function using the same states.  
        /// </summary>  
        /// <param name="dropoutDesc">Handle to a previously created dropout descriptor object.</param>  
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>  
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptor srcDesc.</param>  
        /// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>  
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor destDesc.</param>  
        /// <param name="reserveSpace">Data pointer to GPU memory used by this function. It is expected that contents of reserveSpace doe not change between cudnnDropoutForward and cudnnDropoutBackward calls.</param>  
        public void DropoutForward(DropoutDescriptor dropoutDesc,
                                   TensorDescriptor xDesc,
                                   CudaDeviceVariable<double> x,
                                   TensorDescriptor yDesc,
                                   CudaDeviceVariable<double> y,
                                   CudaDeviceVariable<byte> reserveSpace)
        {
            res = CudaDNNNativeMethods.cudnnDropoutForward(_handle, dropoutDesc.Desc, xDesc.Desc, x.DevicePointer, yDesc.Desc, y.DevicePointer, reserveSpace.DevicePointer, reserveSpace.SizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDropoutForward", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>  
        /// This function performs backward dropout operation over dy returning results in dx. If during   
        /// forward dropout operation value from x was propagated to y then during backward operation value   
        /// from dy will be propagated to dx, otherwise, dx value will be set to 0.  
        /// </summary>  
        /// <param name="dropoutDesc">Handle to a previously created dropout descriptor object.</param>  
        /// <param name="dyDesc">Handle to a previously initialized tensor descriptor.</param>  
        /// <param name="dy">Pointer to data of the tensor described by the dyDesc descriptor.</param>  
        /// <param name="dxDesc">Handle to a previously initialized tensor descriptor.</param>  
        /// <param name="dx">Pointer to data of the tensor described by the dxDesc descriptor.</param>  
        /// <param name="reserveSpace">Data pointer to GPU memory used by this function. It is expected that contents of reserveSpace doe not change between cudnnDropoutForward and cudnnDropoutBackward calls.</param>  
        public void DropoutBackward(DropoutDescriptor dropoutDesc,
                                    TensorDescriptor dyDesc,
                                    CudaDeviceVariable<double> dy,
                                    TensorDescriptor dxDesc,
                                    CudaDeviceVariable<double> dx,
                                    CudaDeviceVariable<byte> reserveSpace)
        {
            res = CudaDNNNativeMethods.cudnnDropoutBackward(_handle, dropoutDesc.Desc, dyDesc.Desc, dy.DevicePointer, dxDesc.Desc, dx.DevicePointer, reserveSpace.DevicePointer, reserveSpace.SizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDropoutBackward", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }
        #endregion

        #region Type independent
        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public int GetConvolutionForwardAlgorithmMaxCount()
        {
            int count = 0;
            res = CudaDNNNativeMethods.cudnnGetConvolutionForwardAlgorithmMaxCount(_handle, ref count);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionForwardAlgorithmMaxCount", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
            return count;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public int GetConvolutionBackwardFilterAlgorithmMaxCount()
        {
            int count = 0;
            res = CudaDNNNativeMethods.cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(_handle, ref count);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionBackwardFilterAlgorithmMaxCount", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
            return count;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public int GetConvolutionBackwardDataAlgorithmMaxCount()
        {
            int count = 0;
            res = CudaDNNNativeMethods.cudnnGetConvolutionBackwardDataAlgorithmMaxCount(_handle, ref count);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionBackwardDataAlgorithmMaxCount", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
            return count;
        }


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
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="preference">Enumerant to express the preference criteria in terms of memory
        /// requirement and speed.</param>
        /// <param name="memoryLimitInbytes">It is used when enumerant preference is set to
        /// CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT to specify the
        /// maximum amount of GPU memory the user is willing to use as a workspace</param>
        /// <returns>Enumerant that specifies which convolution algorithm should be used to
        /// compute the results according to the specified preference</returns>
        public cudnnConvolutionFwdAlgo GetConvolutionForwardAlgorithm(TensorDescriptor xDesc,
                                                    FilterDescriptor filterDesc,
                                                    ConvolutionDescriptor convDesc,
                                                    TensorDescriptor yDesc,
                                                    cudnnConvolutionFwdPreference preference,
                                                    SizeT memoryLimitInbytes
                                                    )
        {
            cudnnConvolutionFwdAlgo algo = new cudnnConvolutionFwdAlgo();
            res = CudaDNNNativeMethods.cudnnGetConvolutionForwardAlgorithm(_handle, xDesc.Desc, filterDesc.Desc, convDesc.Desc, yDesc.Desc, preference, memoryLimitInbytes, ref algo);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionForwardAlgorithm", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
            return algo;
        }

        /// <summary>
        /// This function serves as a heuristic for obtaining the best suited algorithm for
        /// cudnnConvolutionForward for the given layer specifications.This function will return
        /// all algorithms sorted by expected (based on internal heuristic) relative performance with
        /// fastest being index 0 of perfResults.For an exhaustive search for the fastest algorithm,
        /// please use cudnnFindConvolutionForwardAlgorithm.
        /// </summary>
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="requestedAlgoCount">The maximum number of elements to be stored in perfResults.</param>
        /// <returns>array to store performance metrics sorted ascending by compute time.</returns>
        public cudnnConvolutionFwdAlgoPerf[] GetConvolutionForwardAlgorithm(TensorDescriptor xDesc,
                                                    FilterDescriptor filterDesc,
                                                    ConvolutionDescriptor convDesc,
                                                    TensorDescriptor yDesc,
                                                    int requestedAlgoCount
                                                    )
        {
            cudnnConvolutionFwdAlgoPerf[] algos = new cudnnConvolutionFwdAlgoPerf[requestedAlgoCount];
            int returnedAlgoCount = 0;
            res = CudaDNNNativeMethods.cudnnGetConvolutionForwardAlgorithm_v7(_handle, xDesc.Desc, filterDesc.Desc, convDesc.Desc, yDesc.Desc, requestedAlgoCount, ref returnedAlgoCount, algos);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionForwardAlgorithm_v7", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);

            if (returnedAlgoCount != requestedAlgoCount)
            {
                cudnnConvolutionFwdAlgoPerf[] temp = new cudnnConvolutionFwdAlgoPerf[returnedAlgoCount];
                Array.Copy(algos, temp, returnedAlgoCount);
                algos = temp;
            }

            return algos;
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
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="yDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="algo">Enumerant that specifies the chosen convolution algorithm</param>
        public SizeT GetConvolutionForwardWorkspaceSize(TensorDescriptor xDesc,
                                                        FilterDescriptor filterDesc,
                                                        ConvolutionDescriptor convDesc,
                                                        TensorDescriptor yDesc,
                                                        cudnnConvolutionFwdAlgo algo
                                                    )
        {
            SizeT sizeInBytes = 0;
            res = CudaDNNNativeMethods.cudnnGetConvolutionForwardWorkspaceSize(_handle, xDesc.Desc, filterDesc.Desc, convDesc.Desc, yDesc.Desc, algo, ref sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionForwardWorkspaceSize", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
            return sizeInBytes;
        }

        /// <summary>
        /// This function attempts all cuDNN algorithms for cudnnConvolutionBackwardFilter_v3 and outputs performance metrics to a user-
        /// allocated array of cudnnConvolutionBwdFilterAlgoPerf_t. These metrics are
        /// written in sorted fashion where the first element has the lowest compute time. 
        /// </summary>
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
		/// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="dwDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="requestedAlgoCount">The maximum number of elements to be stored in perfResults.</param>
        /// <returns>An array to store performance metrics sorted ascending by compute time.</returns>
        public cudnnConvolutionBwdFilterAlgoPerf[] FindConvolutionBackwardFilterAlgorithm(TensorDescriptor xDesc,
                                                            TensorDescriptor dyDesc,
                                                            ConvolutionDescriptor convDesc,
                                                            FilterDescriptor dwDesc,
                                                            int requestedAlgoCount
                                                            )
        {
            cudnnConvolutionBwdFilterAlgoPerf[] temp = new cudnnConvolutionBwdFilterAlgoPerf[requestedAlgoCount];
            int returnedAlgoCount = 0;
            res = CudaDNNNativeMethods.cudnnFindConvolutionBackwardFilterAlgorithm(_handle, xDesc.Desc, dyDesc.Desc, convDesc.Desc, dwDesc.Desc, requestedAlgoCount, ref returnedAlgoCount, temp);
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
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="dwDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="preference">Enumerant to express the preference criteria in terms of memory requirement and speed.</param>
        /// <param name="memoryLimitInbytes">It is to specify the maximum amount of GPU memory the user is willing to 
        /// use as a workspace. This is currently a placeholder and is not used.</param>
        /// <returns>Enumerant that specifies which convolution algorithm should be used to
        /// compute the results according to the specified preference</returns>
        public cudnnConvolutionBwdFilterAlgo GetConvolutionBackwardFilterAlgorithm(TensorDescriptor xDesc,
                                                            TensorDescriptor dyDesc,
                                                            ConvolutionDescriptor convDesc,
                                                            FilterDescriptor dwDesc,
                                                            cudnnConvolutionBwdFilterPreference preference,
                                                            SizeT memoryLimitInbytes
                                                            )
        {
            cudnnConvolutionBwdFilterAlgo algo = new cudnnConvolutionBwdFilterAlgo();
            res = CudaDNNNativeMethods.cudnnGetConvolutionBackwardFilterAlgorithm(_handle, xDesc.Desc, dyDesc.Desc, convDesc.Desc, dwDesc.Desc, preference, memoryLimitInbytes, ref algo);
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
        /// <param name = "xDesc" > Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="gradDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="algo">Enumerant that specifies the chosen convolution algorithm
        /// sizeInBytes output Amount of GPU memory needed as workspace to be able to execute</param>
        /// <returns>Amount of GPU memory needed as workspace to be able to execute a
        /// forward convolution with the specified algo</returns>
        public SizeT GetConvolutionBackwardFilterWorkspaceSize(TensorDescriptor xDesc,
                                                                    TensorDescriptor dyDesc,
                                                                    ConvolutionDescriptor convDesc,
                                                                    FilterDescriptor gradDesc,
                                                                    cudnnConvolutionBwdFilterAlgo algo
                                                                )
        {
            SizeT sizeInBytes = new SizeT();
            res = CudaDNNNativeMethods.cudnnGetConvolutionBackwardFilterWorkspaceSize(_handle, xDesc.Desc, dyDesc.Desc, convDesc.Desc, gradDesc.Desc, algo, ref sizeInBytes);
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
        /// <param name="wDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="dxDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="requestedAlgoCount">The maximum number of elements to be stored in perfResults.</param>
        /// <returns>An array to store performance metrics sorted ascending by compute time.</returns>
        public cudnnConvolutionBwdDataAlgoPerf[] FindConvolutionBackwardDataAlgorithm(FilterDescriptor wDesc,
                                                            TensorDescriptor dyDesc,
                                                            ConvolutionDescriptor convDesc,
                                                            TensorDescriptor dxDesc,
                                                            int requestedAlgoCount
                                                        )
        {
            cudnnConvolutionBwdDataAlgoPerf[] temp = new cudnnConvolutionBwdDataAlgoPerf[requestedAlgoCount];
            int returnedAlgoCount = 0;
            res = CudaDNNNativeMethods.cudnnFindConvolutionBackwardDataAlgorithm(_handle, wDesc.Desc, dyDesc.Desc, convDesc.Desc, dxDesc.Desc, requestedAlgoCount, ref returnedAlgoCount, temp);
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
        /// <param name="wDesc">Handle to a previously initialized filter descriptor.</param>
		/// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
		/// <param name="convDesc">Previously initialized convolution descriptor.</param>
		/// <param name="dxDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="preference">Enumerant to express the preference criteria in terms of memory
        /// requirement and speed.</param>
        /// <param name="memoryLimitInbytes">It is to specify the maximum amount of GPU memory the user is willing to
        /// use as a workspace. This is currently a placeholder and is not used.</param>
        /// <returns>Enumerant that specifies which convolution algorithm should be used to
        /// compute the results according to the specified preference</returns>
        public cudnnConvolutionBwdDataAlgo GetConvolutionBackwardDataAlgorithm(FilterDescriptor wDesc,
                                                        TensorDescriptor dyDesc,
                                                        ConvolutionDescriptor convDesc,
                                                        TensorDescriptor dxDesc,
                                                        cudnnConvolutionBwdDataPreference preference,
                                                        SizeT memoryLimitInbytes
                                                        )
        {
            cudnnConvolutionBwdDataAlgo algo = new cudnnConvolutionBwdDataAlgo();
            res = CudaDNNNativeMethods.cudnnGetConvolutionBackwardDataAlgorithm(_handle, wDesc.Desc, dyDesc.Desc, convDesc.Desc, dxDesc.Desc, preference, memoryLimitInbytes, ref algo);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionBackwardDataAlgorithm", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
            return algo;
        }


        /// <summary>
        /// This function serves as a heuristic for obtaining the best suited algorithm for
        /// cudnnConvolutionBackwardFilter for the given layer specifications.This function
        /// will return all algorithms sorted by expected (based on internal heuristic) relative
        /// performance with fastest being index 0 of perfResults.For an exhaustive search for the
        /// fastest algorithm, please use cudnnFindConvolutionBackwardFilterAlgorithm.
        /// </summary>
        /// <param name="xDesc">Handle to the previously initialized input tensor descriptor.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="requestedAlgoCount">The maximum number of elements to be stored in perfResults.</param>
        /// <returns>array to store performance metrics sorted ascending by compute time.</returns>
        public cudnnConvolutionBwdFilterAlgoPerf[] GetConvolutionBackwardFilterAlgorithm(TensorDescriptor xDesc,
                                                        TensorDescriptor dyDesc,
                                                    FilterDescriptor filterDesc,
                                                    ConvolutionDescriptor convDesc,
                                                    int requestedAlgoCount
                                                    )
        {
            cudnnConvolutionBwdFilterAlgoPerf[] algos = new cudnnConvolutionBwdFilterAlgoPerf[requestedAlgoCount];
            int returnedAlgoCount = 0;
            res = CudaDNNNativeMethods.cudnnGetConvolutionBackwardFilterAlgorithm_v7(_handle, xDesc.Desc, dyDesc.Desc, convDesc.Desc, filterDesc.Desc, requestedAlgoCount, ref returnedAlgoCount, algos);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionBackwardFilterAlgorithm_v7", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);

            if (returnedAlgoCount != requestedAlgoCount)
            {
                cudnnConvolutionBwdFilterAlgoPerf[] temp = new cudnnConvolutionBwdFilterAlgoPerf[returnedAlgoCount];
                Array.Copy(algos, temp, returnedAlgoCount);
                algos = temp;
            }

            return algos;
        }


        /// <summary>
        /// This function serves as a heuristic for obtaining the best suited algorithm for
        /// cudnnConvolutionBackwardData for the given layer specifications.This function
        /// will return all algorithms sorted by expected (based on internal heuristic) relative
        /// performance with fastest being index 0 of perfResults.For an exhaustive search for the
        /// fastest algorithm, please use cudnnFindConvolutionBackwardDataAlgorithm.
        /// </summary>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="dxDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="requestedAlgoCount">The maximum number of elements to be stored in perfResults.</param>
        /// <param name="filterDesc">Handle to a previously initialized filter descriptor.</param>
        /// <returns>array to store performance metrics sorted ascending by compute time.</returns>
        public cudnnConvolutionBwdDataAlgoPerf[] GetConvolutionBackwardDataAlgorithm(
                                                    FilterDescriptor filterDesc, TensorDescriptor dyDesc,
                                                    ConvolutionDescriptor convDesc,TensorDescriptor dxDesc,
                                                    int requestedAlgoCount
                                                    )
        {
            cudnnConvolutionBwdDataAlgoPerf[] algos = new cudnnConvolutionBwdDataAlgoPerf[requestedAlgoCount];
            int returnedAlgoCount = 0;
            res = CudaDNNNativeMethods.cudnnGetConvolutionBackwardDataAlgorithm_v7(_handle, filterDesc.Desc, dyDesc.Desc, convDesc.Desc, dxDesc.Desc, requestedAlgoCount, ref returnedAlgoCount, algos);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionBackwardDataAlgorithm_v7", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);

            if (returnedAlgoCount != requestedAlgoCount)
            {
                cudnnConvolutionBwdDataAlgoPerf[] temp = new cudnnConvolutionBwdDataAlgoPerf[returnedAlgoCount];
                Array.Copy(algos, temp, returnedAlgoCount);
                algos = temp;
            }

            return algos;
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
        /// <param name="wDesc">Handle to a previously initialized filter descriptor.</param>
        /// <param name="dyDesc">Handle to the previously initialized input differential tensor descriptor.</param>
        /// <param name="convDesc">Previously initialized convolution descriptor.</param>
        /// <param name="dxDesc">Handle to the previously initialized output tensor descriptor.</param>
        /// <param name="algo">Enumerant that specifies the chosen convolution algorithm</param>
        public SizeT GetConvolutionBackwardDataWorkspaceSize(FilterDescriptor wDesc,
                                                            TensorDescriptor dyDesc,
                                                            ConvolutionDescriptor convDesc,
                                                            TensorDescriptor dxDesc,
                                                            cudnnConvolutionBwdDataAlgo algo
                                                        )
        {
            SizeT sizeInBytes = new SizeT();
            res = CudaDNNNativeMethods.cudnnGetConvolutionBackwardDataWorkspaceSize(_handle, wDesc.Desc, dyDesc.Desc, convDesc.Desc, dxDesc.Desc, algo, ref sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionBackwardDataWorkspaceSize", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
            return sizeInBytes;
        }


        /// <summary>
        /// Helper function to return the minimum size of the index space to be passed to the reduction given the input and output tensors
        /// </summary>
        /// <param name="reduceTensorDesc"></param>
        /// <param name="aDesc"></param>
        /// <param name="cDesc"></param>
        public SizeT GetReductionIndicesSize(
                                ReduceTensorDescriptor reduceTensorDesc,
                                TensorDescriptor aDesc,
                                TensorDescriptor cDesc)
        {
            SizeT sizeInBytes = new SizeT();
            res = CudaDNNNativeMethods.cudnnGetReductionIndicesSize(_handle, reduceTensorDesc.Desc, aDesc.Desc, cDesc.Desc, ref sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetReductionIndicesSize", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
            return sizeInBytes;
        }

        /// <summary>
        /// Helper function to return the minimum size of the workspace to be passed to the reduction given the input and output tensors
        /// </summary>
        /// <param name="reduceTensorDesc"></param>
        /// <param name="aDesc"></param>
        /// <param name="cDesc"></param>
        public SizeT GetReductionWorkspaceSize(
                                ReduceTensorDescriptor reduceTensorDesc,
                                TensorDescriptor aDesc,
                                TensorDescriptor cDesc)
        {
            SizeT sizeInBytes = new SizeT();
            res = CudaDNNNativeMethods.cudnnGetReductionWorkspaceSize(_handle, reduceTensorDesc.Desc, aDesc.Desc, cDesc.Desc, ref sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetReductionWorkspaceSize", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
            return sizeInBytes;
        }

        /// <summary>
        /// This function initializes a previously created RNN descriptor object.
        /// </summary>
        /// <param name="rnnDesc">A previously created RNN descriptor.</param>
        /// <param name="hiddenSize">Size of the internal hidden state for each layer.</param>
        /// <param name="numLayers">Number of stacked layers.</param>
        /// <param name="dropoutDesc">Handle to a previously created and initialized dropout descriptor.
        /// Dropout will be applied between layers(eg.a single layer network will have no dropout applied).</param>
        /// <param name="inputMode">Specifies the behavior at the input to the first layer</param>
        /// <param name="direction">Specifies the recurrence pattern. (eg. bidirectional)</param>
        /// <param name="mode">Specifies the type of RNN to compute.</param>
        /// <param name="algo">Specifies which RNN algorithm should be used to compute the results.</param>
        /// <param name="dataType">Compute precision.</param>
        public void SetRNNDescriptor(
                                                RNNDescriptor rnnDesc,
                                                int hiddenSize,
                                                int numLayers,
                                                DropoutDescriptor dropoutDesc, // Between layers, not between recurrent steps.
                                                cudnnRNNInputMode inputMode,
                                                cudnnDirectionMode direction,
                                                cudnnRNNMode mode,
                                                cudnnRNNAlgo algo,
                                                cudnnDataType dataType)
        {
            res = CudaDNNNativeMethods.cudnnSetRNNDescriptor(_handle, rnnDesc.Desc, hiddenSize, numLayers, dropoutDesc.Desc, inputMode, direction, mode, algo, dataType);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetRNNDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }



        /// <summary>  
        /// This function is used to query the amount of reserve needed to run dropout with the input dimensions given by xDesc.   
        /// The same reserve space is expected to be passed to cudnnDropoutForward and cudnnDropoutBackward, and its contents is   
        /// expected to remain unchanged between cudnnDropoutForward and cudnnDropoutBackward calls.   
        /// </summary>  
        /// <param name="xDesc">Handle to a previously initialized tensor descriptor, describing input to a dropout operation.</param>  
        public SizeT GetDropoutReserveSpaceSize(TensorDescriptor xDesc)
        {
            SizeT sizeInBytes = new SizeT();
            res = CudaDNNNativeMethods.cudnnDropoutGetReserveSpaceSize(xDesc.Desc, ref sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDropoutGetReserveSpaceSize", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
            return sizeInBytes;
        }

        /// <summary>  
        /// This function is used to query the amount of space required to store the states of the random number generators used by cudnnDropoutForward function.  
        /// </summary>  
        public SizeT GetDropoutStateSize()
        {
            SizeT sizeInBytes = new SizeT();
            res = CudaDNNNativeMethods.cudnnDropoutGetStatesSize(_handle, ref sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDropoutGetStatesSize", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
            return sizeInBytes;
        }

        //new in CUDNN 7

        /// <summary>
        /// cuDNN library functions perform extensive input argument checking before launching
        /// GPU kernels.The last step is to verify that the GPU kernel actually started. When
        /// a kernel fails to start, CUDNN_STATUS_EXECUTION_FAILED is returned by the
        /// corresponding API call. Typically, after a GPU kernel starts, no runtime checks are
        /// performed by the kernel itself -- numerical results are simply written to output buffers.<para/>
        /// When the CUDNN_BATCHNORM_SPATIAL_PERSISTENT mode is selected in cudnnBatchNormalizationForwardTraining or
        /// cudnnBatchNormalizationBackward, the algorithm may encounter numerical overflows
        /// where CUDNN_BATCHNORM_SPATIAL performs just fine albeit at a slower speed.<para/>
        /// The user can invoke cudnnQueryRuntimeError to make sure numerical overflows did
        /// not occur during the kernel execution.Those issues are reported by the kernel that
        /// performs computations.
        /// </summary>
        /// <param name="mode">Remote error query mode.</param>
        /// <returns>the user's error code</returns>
        public cudnnStatus QueryRuntimeError(cudnnErrQueryMode mode)
        {
            cudnnStatus rstatus = new cudnnStatus();
            cudnnRuntimeTag tag = new cudnnRuntimeTag();
            res = CudaDNNNativeMethods.cudnnQueryRuntimeError(_handle, ref rstatus, mode, tag);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnQueryRuntimeError", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
            return rstatus;
        }

        /// <summary>
        /// This function queries the fields of a previously initialized dropout descriptor.
        /// </summary>
        /// <param name="dropoutDesc">Previously initialized dropout descriptor.</param>
        /// <param name="droupout">The probability with which the value from input is set to 0 during the 
        /// dropout layer.</param>
        /// <param name="seed">Seed used to initialize random number generator states.</param>
        /// <returns>user-allocated GPU memory that holds random number generator states.</returns>
        public CudaDeviceVariable<byte> GetDropoutDescriptor(DropoutDescriptor dropoutDesc, ref float droupout, ref ulong seed)
        {
            CUdeviceptr states = new CUdeviceptr();
            res = CudaDNNNativeMethods.cudnnGetDropoutDescriptor(dropoutDesc.Desc, _handle, ref droupout, ref states, ref seed);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnQueryRuntimeError", res));
            CudaDeviceVariable<byte> ret = new CudaDeviceVariable<byte>(states, false);
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
            return ret;
        }

        /// <summary>
        /// This function restores a dropout descriptor to a previously saved-off state.
        /// </summary>
        /// <param name="dropoutDesc">Previously created dropout descriptor.</param>
        /// <param name="droupout">Probability with which the value from an input tensor is set to 0 when performing dropout.</param>
        /// <param name="seed">Seed used in prior call to cudnnSetDropoutDescriptor that initialized
        /// #states' buffer. Using a different seed from this has no effect. A change of seed, and subsequent update to random number generator states can be achieved by calling
        /// cudnnSetDropoutDescriptor.</param>
        /// <param name="states">Pointer to GPU memory that holds random number generator states initialized by a prior call to cudnnSetDropoutDescriptor.</param>
        public void RestoreDropoutDescriptor(DropoutDescriptor dropoutDesc, CudaDeviceVariable<byte> states, ref float droupout, ref ulong seed)
        {
            res = CudaDNNNativeMethods.cudnnRestoreDropoutDescriptor(dropoutDesc.Desc, _handle, droupout, states.DevicePointer, states.SizeInBytes, seed);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnRestoreDropoutDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }
        #endregion
    }
}
