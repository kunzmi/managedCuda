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
using System.Linq;
using System.Diagnostics;
using ManagedCuda.BasicTypes;

namespace ManagedCuda.CudaDNN
{
    /// <summary>
    /// 
    /// </summary>
    public class RNNDescriptor : IDisposable
    {
        private cudnnRNNDescriptor _desc;
        private cudnnStatus res;
        private bool disposed;
        private cudnnHandle _handle;

        #region Contructors
        /// <summary>
        /// </summary>
        public RNNDescriptor(CudaDNNContext context)
        {
            _handle = context.Handle;
            _desc = new cudnnRNNDescriptor();
            res = CudaDNNNativeMethods.cudnnCreateRNNDescriptor(ref _desc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCreateRNNDescriptor", res));
            if (res != cudnnStatus.Success)
                throw new CudaDNNException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~RNNDescriptor()
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
                res = CudaDNNNativeMethods.cudnnDestroyRNNDescriptor(_desc);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDestroyRNNDescriptor", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        /// <summary>
        /// Returns the inner handle.
        /// </summary>
        public cudnnRNNDescriptor Desc
        {
            get { return _desc; }
        }


        /// <summary>
        /// This function initializes a previously created RNN descriptor object.
        /// </summary>
        /// <param name="ctx">Handle to a previously created cuDNN library descriptor.</param>
        /// <param name="hiddenSize">Size of the internal hidden state for each layer.</param>
        /// <param name="numLayers">Number of layers.</param>
        /// <param name="dropoutDesc">Handle to a previously created and initialized dropout descriptor.</param>
        /// <param name="inputMode">Specifies the behavior at the input to the first layer.</param>
        /// <param name="direction">Specifies the recurrence pattern. (eg. bidirectional)</param>
        /// <param name="mode">The type of RNN to compute.</param>
        /// <param name="algo">Specifies which RNN algorithm should be used to compute the results.</param>
        /// <param name="dataType">Math precision.</param>
        public void SetRNNDescriptor(CudaDNNContext ctx,
                                                        int hiddenSize,
                                                        int numLayers,
                                                        DropoutDescriptor dropoutDesc, // Between layers, not between recurrent steps.
                                                        cudnnRNNInputMode inputMode,
                                                        cudnnDirectionMode direction,
                                                        cudnnRNNMode mode,
                                                cudnnRNNAlgo algo,
                                                        cudnnDataType dataType)
        {
            res = CudaDNNNativeMethods.cudnnSetRNNDescriptor(ctx.Handle, _desc, hiddenSize, numLayers, dropoutDesc.Desc, inputMode, direction, mode, algo, dataType);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetRNNDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function is used to query the amount of work space required to execute the RNN 
        /// described by rnnDesc with inputs dimensions defined by xDesc. 
        /// </summary>
        /// <param name="seqLength">Number of iterations to unroll over.</param>
        /// <param name="xDesc">An array of tensor descriptors describing the input to each recurrent iteration.</param>
        /// <param name="sizeInBytes">Minimum amount of GPU memory needed as workspace to be able to execute an RNN with the specified descriptor and input tensors.</param>
        public void GetRNNWorkspaceSize(int seqLength, TensorDescriptor[] xDesc, ref SizeT sizeInBytes)
        {
            var a1 = xDesc.Select(x => x.Desc).ToArray();
            res = CudaDNNNativeMethods.cudnnGetRNNWorkspaceSize(_handle, _desc, seqLength, a1, ref sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetRNNWorkspaceSize", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function is used to query the amount of reserved space required for training the 
        /// RNN described by rnnDesc with inputs dimensions defined by xDesc. The same reserve 
        /// space must be passed to cudnnRNNForwardTraining, cudnnRNNBackwardData and cudnnRNNBackwardWeights.
        /// </summary>
        /// <param name="seqLength">Number of iterations to unroll over.</param>
        /// <param name="xDesc">An array of tensor descriptors describing the input to each recurrent iteration.</param>
        /// <param name="sizeInBytes">Minimum amount of GPU memory needed as reserve space to be able to train an RNN with the specified descriptor and input tensors.</param>
        public void GetRNNTrainingReserveSize(int seqLength,
                                                          TensorDescriptor[] xDesc,
                                                          ref SizeT sizeInBytes
                                                    )
        {
            var a1 = xDesc.Select(x => x.Desc).ToArray();
            res = CudaDNNNativeMethods.cudnnGetRNNTrainingReserveSize(_handle, _desc, seqLength, a1, ref sizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetRNNTrainingReserveSize", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function is used to query the amount of parameter space required to execute the RNN described by 
        /// rnnDesc with inputs dimensions defined by xDesc. 
        /// </summary>
        /// <param name="xDesc">A fully packed tensor descriptor describing the input to one recurrent iteration.</param>
        /// <param name="sizeInBytes">Minimum amount of GPU memory needed as parameter space to be able to execute an RNN with the specified descriptor and input tensors.</param>
        /// <param name="dataType">The data type of the parameters.</param>
        public void cudnnGetRNNParamsSize(
                                                 TensorDescriptor xDesc,
                                                 ref SizeT sizeInBytes,
                                                 cudnnDataType dataType   )
        {
            res = CudaDNNNativeMethods.cudnnGetRNNParamsSize(_handle, _desc, xDesc.Desc, ref sizeInBytes, dataType);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetRNNParamsSize", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function is used to obtain a pointer and descriptor for the matrix parameters in layer within 
        /// the RNN described by rnnDesc with inputs dimensions defined by xDesc. 
        /// </summary>
        /// <param name="layer">The layer to query.</param>
        /// <param name="xDesc">A fully packed tensor descriptor describing the input to one recurrent iteration.</param>
        /// <param name="wDesc">Handle to a previously initialized filter descriptor describing the weights for the RNN.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor wDesc.</param>
        /// <param name="linLayerID">
        /// The linear layer to obtain information about: 
        /// * If mode in rnnDesc was set to CUDNN_RNN_RELU or CUDNN_RNN_TANH a value of 0 references the matrix multiplication 
        /// applied to the input from the previous layer, a value of 1 references the matrix multiplication applied to the recurrent input.
        /// * If mode in rnnDesc was set to CUDNN_LSTM values of 0-3 reference matrix multiplications applied to the input from the 
        /// previous layer, value of 4-7 reference matrix multiplications applied to the recurrent input.
        ///     ‣ Values 0 and 4 reference the input gate. 
        ///     ‣ Values 1 and 5 reference the forget gate. 
        ///     ‣ Values 2 and 6 reference the new memory gate. 
        ///     ‣ Values 3 and 7 reference the output gate.
        /// * If mode in rnnDesc was set to CUDNN_GRU values of 0-2 reference matrix multiplications applied to the input 
        /// from the previous layer, value of 3-5 reference matrix multiplications applied to the recurrent input. 
        ///     ‣ Values 0 and 3 reference the reset gate. 
        ///     ‣ Values 1 and 4 reference the update gate. 
        ///     ‣ Values 2 and 5 reference the new memory gate.
        /// </param>
        /// <param name="linLayerMatDesc">Handle to a previously created filter descriptor.</param>
        /// <param name="linLayerMat">Data pointer to GPU memory associated with the filter descriptor linLayerMatDesc.</param>
        public void GetRNNLinLayerMatrixParams(
                             int layer,
                             TensorDescriptor xDesc,
                             FilterDescriptor wDesc,
                             CudaDeviceVariable<float> w,
                             int linLayerID,
                             FilterDescriptor linLayerMatDesc,
                             CudaDeviceVariable<SizeT> linLayerMat // void **
                             )
        {
            res = CudaDNNNativeMethods.cudnnGetRNNLinLayerMatrixParams(_handle, _desc, layer, xDesc.Desc, wDesc.Desc, w.DevicePointer, linLayerID, linLayerMatDesc.Desc, linLayerMat.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetRNNLinLayerMatrixParams", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function is used to obtain a pointer and descriptor for the matrix parameters in layer within 
        /// the RNN described by rnnDesc with inputs dimensions defined by xDesc. 
        /// </summary>
        /// <param name="layer">The layer to query.</param>
        /// <param name="xDesc">A fully packed tensor descriptor describing the input to one recurrent iteration.</param>
        /// <param name="wDesc">Handle to a previously initialized filter descriptor describing the weights for the RNN.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor wDesc.</param>
        /// <param name="linLayerID">
        /// The linear layer to obtain information about: 
        /// * If mode in rnnDesc was set to CUDNN_RNN_RELU or CUDNN_RNN_TANH a value of 0 references the matrix multiplication 
        /// applied to the input from the previous layer, a value of 1 references the matrix multiplication applied to the recurrent input.
        /// * If mode in rnnDesc was set to CUDNN_LSTM values of 0-3 reference matrix multiplications applied to the input from the 
        /// previous layer, value of 4-7 reference matrix multiplications applied to the recurrent input.
        ///     ‣ Values 0 and 4 reference the input gate. 
        ///     ‣ Values 1 and 5 reference the forget gate. 
        ///     ‣ Values 2 and 6 reference the new memory gate. 
        ///     ‣ Values 3 and 7 reference the output gate.
        /// * If mode in rnnDesc was set to CUDNN_GRU values of 0-2 reference matrix multiplications applied to the input 
        /// from the previous layer, value of 3-5 reference matrix multiplications applied to the recurrent input. 
        ///     ‣ Values 0 and 3 reference the reset gate. 
        ///     ‣ Values 1 and 4 reference the update gate. 
        ///     ‣ Values 2 and 5 reference the new memory gate.
        /// </param>
        /// <param name="linLayerMatDesc">Handle to a previously created filter descriptor.</param>
        /// <param name="linLayerMat">Data pointer to GPU memory associated with the filter descriptor linLayerMatDesc.</param>
        public void GetRNNLinLayerMatrixParams(
                             int layer,
                             TensorDescriptor xDesc,
                             FilterDescriptor wDesc,
                             CudaDeviceVariable<double> w,
                             int linLayerID,
                             FilterDescriptor linLayerMatDesc,
                             CudaDeviceVariable<SizeT> linLayerMat // void **
                             )
        {
            res = CudaDNNNativeMethods.cudnnGetRNNLinLayerMatrixParams(_handle, _desc, layer, xDesc.Desc, wDesc.Desc, w.DevicePointer, linLayerID, linLayerMatDesc.Desc, linLayerMat.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetRNNLinLayerMatrixParams", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function is used to obtain a pointer and descriptor for the bias parameters 
        /// in layer within the RNN described by rnnDesc with inputs dimensions defined by xDesc. 
        /// </summary>
        /// <param name="layer">The layer to query.</param>
        /// <param name="xDesc">A fully packed tensor descriptor describing the input to one recurrent iteration.</param>
        /// <param name="wDesc">Handle to a previously initialized filter descriptor describing the weights for the RNN.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor wDesc.</param>
        /// <param name="linLayerID">
        /// The linear layer to obtain information about: 
        /// * If mode in rnnDesc was set to CUDNN_RNN_RELU or CUDNN_RNN_TANH a value of 0 references 
        /// the bias applied to the input from the previous layer, a value of 1 references the bias 
        /// applied to the recurrent input.
        /// * If mode in rnnDesc was set to CUDNN_LSTM values of 0, 1, 2 and 3 reference bias applied to the input 
        /// from the previous layer, value of 4, 5, 6 and 7 reference bias applied to the recurrent input.
        ///     ‣ Values 0 and 4 reference the input gate. 
        ///     ‣ Values 1 and 5 reference the forget gate. 
        ///     ‣ Values 2 and 6 reference the new memory gate. 
        ///     ‣ Values 3 and 7 reference the output gate.
        /// * If mode in rnnDesc was set to CUDNN_GRU values of 0, 1 and 2 reference bias applied to the 
        /// input from the previous layer, value of 3, 4 and 5 reference bias applied to the recurrent input.
        ///     ‣ Values 0 and 3 reference the reset gate. 
        ///     ‣ Values 1 and 4 reference the update gate. 
        ///     ‣ Values 2 and 5 reference the new memory gate.</param>
        /// <param name="linLayerBiasDesc">Handle to a previously created filter descriptor.</param>
        /// <param name="linLayerBias">Data pointer to GPU memory associated with the filter descriptor linLayerMatDesc.</param>
        public void GetRNNLinLayerBiasParams(
                             int layer,
                             TensorDescriptor xDesc,
                             FilterDescriptor wDesc,
                             CudaDeviceVariable<float> w,
                             int linLayerID,
                             FilterDescriptor linLayerBiasDesc,
                             CudaDeviceVariable<SizeT> linLayerBias // void **
                             )
        {
            res = CudaDNNNativeMethods.cudnnGetRNNLinLayerBiasParams(_handle, _desc, layer, xDesc.Desc, wDesc.Desc, w.DevicePointer, linLayerID, linLayerBiasDesc.Desc, linLayerBias.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetRNNLinLayerBiasParams", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function is used to obtain a pointer and descriptor for the bias parameters 
        /// in layer within the RNN described by rnnDesc with inputs dimensions defined by xDesc. 
        /// </summary>
        /// <param name="layer">The layer to query.</param>
        /// <param name="xDesc">A fully packed tensor descriptor describing the input to one recurrent iteration.</param>
        /// <param name="wDesc">Handle to a previously initialized filter descriptor describing the weights for the RNN.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor wDesc.</param>
        /// <param name="linLayerID">
        /// The linear layer to obtain information about: 
        /// * If mode in rnnDesc was set to CUDNN_RNN_RELU or CUDNN_RNN_TANH a value of 0 references 
        /// the bias applied to the input from the previous layer, a value of 1 references the bias 
        /// applied to the recurrent input.
        /// * If mode in rnnDesc was set to CUDNN_LSTM values of 0, 1, 2 and 3 reference bias applied to the input 
        /// from the previous layer, value of 4, 5, 6 and 7 reference bias applied to the recurrent input.
        ///     ‣ Values 0 and 4 reference the input gate. 
        ///     ‣ Values 1 and 5 reference the forget gate. 
        ///     ‣ Values 2 and 6 reference the new memory gate. 
        ///     ‣ Values 3 and 7 reference the output gate.
        /// * If mode in rnnDesc was set to CUDNN_GRU values of 0, 1 and 2 reference bias applied to the 
        /// input from the previous layer, value of 3, 4 and 5 reference bias applied to the recurrent input.
        ///     ‣ Values 0 and 3 reference the reset gate. 
        ///     ‣ Values 1 and 4 reference the update gate. 
        ///     ‣ Values 2 and 5 reference the new memory gate.</param>
        /// <param name="linLayerBiasDesc">Handle to a previously created filter descriptor.</param>
        /// <param name="linLayerBias">Data pointer to GPU memory associated with the filter descriptor linLayerMatDesc.</param>
        public void GetRNNLinLayerBiasParams(
                             int layer,
                             TensorDescriptor xDesc,
                             FilterDescriptor wDesc,
                             CudaDeviceVariable<double> w,
                             int linLayerID,
                             FilterDescriptor linLayerBiasDesc,
                             CudaDeviceVariable<SizeT> linLayerBias // void **
                             )
        {
            res = CudaDNNNativeMethods.cudnnGetRNNLinLayerBiasParams(_handle, _desc, layer, xDesc.Desc, wDesc.Desc, w.DevicePointer, linLayerID, linLayerBiasDesc.Desc, linLayerBias.DevicePointer);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetRNNLinLayerBiasParams", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This routine executes the recurrent neural network described by rnnDesc with inputs x, hx, cx, weights w and 
        /// outputs y, hy, cy. workspace is required for intermediate storage. This function does not store data required 
        /// for training; cudnnRNNForwardTraining should be used for that purpose. 
        /// </summary>
        /// <param name="xDesc">An array of tensor descriptors describing the input to each recurrent iteration. 
        /// Each tensor descriptor must have the same first dimension. The second dimension of the tensors may 
        /// decrease from element n to element n+1 but may not increase. The tensor must be fully packed.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptors in the array xDesc. 
        /// The data are expected to be packed contiguously with the first element of iteration n+1 following 
        /// directly from the last element of iteration n.</param>
        /// <param name="hxDesc">Handle to a previously initialized tensor descriptor describing the initial hidden 
        /// state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second 
        /// dimension of the first tensor described in xDesc. The third dimension must match the numLayers 
        /// argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be 
        /// fully packed.</param>
        /// <param name="hx">Data pointer to GPU memory associated with the tensor descriptor hxDesc. If a NULL pointer 
        /// is passed, the initial hidden state of the network will be initialized to zero.</param>
        /// <param name="cxDesc">Handle to a previously initialized tensor descriptor describing the initial cell 
        /// state for LSTM networks. The first dimension of the tensor must match the hiddenSize argument passed to 
        /// the cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second 
        /// dimension of the first tensor described in xDesc. The third dimension must match the numLayers argument 
        /// passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="cx">Data pointer to GPU memory associated with the tensor descriptor cxDesc. If a NULL 
        /// pointer is passed, the initial cell state of the network will be initialized to zero.</param>
        /// <param name="wDesc">Handle to a previously initialized filter descriptor describing the weights for the RNN.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor wDesc.</param>
        /// <param name="yDesc">An array of tensor descriptors describing the output from each recurrent iteration. 
        /// The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor 
        /// call used to initialize rnnDesc:
        /// * If direction is CUDNN_UNIDIRECTIONAL the first dimension should match the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// * If direction is CUDNN_BIDIRECTIONAL the first dimension should match double the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// The second dimension of the tensor n must match the second dimension of the tensor n in xDesc. 
        /// The tensor must be fully packed.</param>
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor yDesc. The data 
        /// are expected to be packed contiguously with the first element of iteration n+1 following directly 
        /// from the last element of iteration n.</param>
        /// <param name="hyDesc">Handle to a previously initialized tensor descriptor describing the final hidden 
        /// state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second 
        /// dimension of the first tensor described in xDesc. The third dimension must match the numLayers 
        /// argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be 
        /// fully packed.</param>
        /// <param name="hy">Data pointer to GPU memory associated with the tensor descriptor hyDesc. If a NULL 
        /// pointer is passed, the final hidden state of the network will not be saved.</param>
        /// <param name="cyDesc">Handle to a previously initialized tensor descriptor describing the final cell 
        /// state for LSTM networks. The first dimension of the tensor must match the hiddenSize argument passed 
        /// to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second 
        /// dimension of the first tensor described in xDesc. The third dimension must match the numLayers argument 
        /// passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="cy">Data pointer to GPU memory associated with the tensor descriptor cyDesc. If 
        /// a NULL pointer is passed, the final cell state of the network will be not be saved.</param>
        /// <param name="workspace">Data pointer to GPU memory to be used as a workspace for this call.</param>
        /// <param name="workSpaceSizeInBytes">Specifies the size in bytes of the provided workspace.</param>
        public void RNNForwardInference(
                                                    TensorDescriptor[] xDesc,
                                                    CudaDeviceVariable<float> x,
                                                    TensorDescriptor hxDesc,
                                                    CudaDeviceVariable<float> hx,
                                                    TensorDescriptor cxDesc,
                                                    CudaDeviceVariable<float> cx,
                                                    FilterDescriptor wDesc,
                                                    CudaDeviceVariable<float> w,
                                                    TensorDescriptor[] yDesc,
                                                    CudaDeviceVariable<float> y,
                                                    TensorDescriptor hyDesc,
                                                    CudaDeviceVariable<float> hy,
                                                    TensorDescriptor cyDesc,
                                                    CudaDeviceVariable<float> cy,
                                                    CudaDeviceVariable<byte> workspace,
                                                    SizeT workSpaceSizeInBytes)
        {
            var a1 = xDesc.Select(q => q.Desc).ToArray();
            var a2 = yDesc.Select(q => q.Desc).ToArray();
            res = CudaDNNNativeMethods.cudnnRNNForwardInference(
                _handle, _desc, a1, x.DevicePointer, hxDesc.Desc, hx.DevicePointer, cxDesc.Desc, cx.DevicePointer, wDesc.Desc, w.DevicePointer,
                a2, y.DevicePointer, hyDesc.Desc, hy.DevicePointer, cyDesc.Desc, cy.DevicePointer, workspace.DevicePointer, workSpaceSizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnRNNForwardInference", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This routine executes the recurrent neural network described by rnnDesc with inputs x, hx, cx, weights w and 
        /// outputs y, hy, cy. workspace is required for intermediate storage. This function does not store data required 
        /// for training; cudnnRNNForwardTraining should be used for that purpose. 
        /// </summary>
        /// <param name="xDesc">An array of tensor descriptors describing the input to each recurrent iteration. 
        /// Each tensor descriptor must have the same first dimension. The second dimension of the tensors may 
        /// decrease from element n to element n+1 but may not increase. The tensor must be fully packed.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptors in the array xDesc. 
        /// The data are expected to be packed contiguously with the first element of iteration n+1 following 
        /// directly from the last element of iteration n.</param>
        /// <param name="hxDesc">Handle to a previously initialized tensor descriptor describing the initial hidden 
        /// state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second 
        /// dimension of the first tensor described in xDesc. The third dimension must match the numLayers 
        /// argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be 
        /// fully packed.</param>
        /// <param name="hx">Data pointer to GPU memory associated with the tensor descriptor hxDesc. If a NULL pointer 
        /// is passed, the initial hidden state of the network will be initialized to zero.</param>
        /// <param name="cxDesc">Handle to a previously initialized tensor descriptor describing the initial cell 
        /// state for LSTM networks. The first dimension of the tensor must match the hiddenSize argument passed to 
        /// the cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second 
        /// dimension of the first tensor described in xDesc. The third dimension must match the numLayers argument 
        /// passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="cx">Data pointer to GPU memory associated with the tensor descriptor cxDesc. If a NULL 
        /// pointer is passed, the initial cell state of the network will be initialized to zero.</param>
        /// <param name="wDesc">Handle to a previously initialized filter descriptor describing the weights for the RNN.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor wDesc.</param>
        /// <param name="yDesc">An array of tensor descriptors describing the output from each recurrent iteration. 
        /// The first dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor 
        /// call used to initialize rnnDesc:
        /// * If direction is CUDNN_UNIDIRECTIONAL the first dimension should match the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// * If direction is CUDNN_BIDIRECTIONAL the first dimension should match double the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// The second dimension of the tensor n must match the second dimension of the tensor n in xDesc. 
        /// The tensor must be fully packed.</param>
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor yDesc. The data 
        /// are expected to be packed contiguously with the first element of iteration n+1 following directly 
        /// from the last element of iteration n.</param>
        /// <param name="hyDesc">Handle to a previously initialized tensor descriptor describing the final hidden 
        /// state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second 
        /// dimension of the first tensor described in xDesc. The third dimension must match the numLayers 
        /// argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be 
        /// fully packed.</param>
        /// <param name="hy">Data pointer to GPU memory associated with the tensor descriptor hyDesc. If a NULL 
        /// pointer is passed, the final hidden state of the network will not be saved.</param>
        /// <param name="cyDesc">Handle to a previously initialized tensor descriptor describing the final cell 
        /// state for LSTM networks. The first dimension of the tensor must match the hiddenSize argument passed 
        /// to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second 
        /// dimension of the first tensor described in xDesc. The third dimension must match the numLayers argument 
        /// passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="cy">Data pointer to GPU memory associated with the tensor descriptor cyDesc. If 
        /// a NULL pointer is passed, the final cell state of the network will be not be saved.</param>
        /// <param name="workspace">Data pointer to GPU memory to be used as a workspace for this call.</param>
        /// <param name="workSpaceSizeInBytes">Specifies the size in bytes of the provided workspace.</param>
        public void RNNForwardInference(
                                                    TensorDescriptor[] xDesc,
                                                    CudaDeviceVariable<double> x,
                                                    TensorDescriptor hxDesc,
                                                    CudaDeviceVariable<double> hx,
                                                    TensorDescriptor cxDesc,
                                                    CudaDeviceVariable<double> cx,
                                                    FilterDescriptor wDesc,
                                                    CudaDeviceVariable<double> w,
                                                    TensorDescriptor[] yDesc,
                                                    CudaDeviceVariable<double> y,
                                                    TensorDescriptor hyDesc,
                                                    CudaDeviceVariable<double> hy,
                                                    TensorDescriptor cyDesc,
                                                    CudaDeviceVariable<double> cy,
                                                    CudaDeviceVariable<byte> workspace,
                                                    SizeT workSpaceSizeInBytes)
        {
            var a1 = xDesc.Select(q => q.Desc).ToArray();
            var a2 = yDesc.Select(q => q.Desc).ToArray();
            res = CudaDNNNativeMethods.cudnnRNNForwardInference(
                _handle, _desc, a1, x.DevicePointer, hxDesc.Desc, hx.DevicePointer, cxDesc.Desc, cx.DevicePointer, wDesc.Desc, w.DevicePointer,
                a2, y.DevicePointer, hyDesc.Desc, hy.DevicePointer, cyDesc.Desc, cy.DevicePointer, workspace.DevicePointer, workSpaceSizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnRNNForwardInference", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This routine executes the recurrent neural network described by rnnDesc with inputs x, hx, cx, weights w 
        /// and outputs y, hy, cy. workspace is required for intermediate storage. reserveSpace stores data required 
        /// for training. The same reserveSpace data must be used for future calls to cudnnRNNBackwardData and 
        /// cudnnRNNBackwardWeights if these execute on the same input data. 
        /// </summary>
        /// <param name="xDesc">An array of tensor descriptors describing the input to each recurrent iteration. Each 
        /// tensor descriptor must have the same first dimension. The second dimension of the tensors may decrease 
        /// from element n to element n+1 but may not increase. The tensor must be fully packed.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptors in the array xDesc.</param>
        /// <param name="hxDesc">Handle to a previously initialized tensor descriptor describing the initial hidden state 
        /// of the RNN. The first dimension of the tensor must match the hiddenSize argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second 
        /// dimension of the first tensor described in xDesc. The third dimension must match the numLayers argument 
        /// passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="hx">Data pointer to GPU memory associated with the tensor descriptor hxDesc. If a NULL pointer 
        /// is passed, the initial hidden state of the network will be initialized to zero.</param>
        /// <param name="cxDesc">Handle to a previously initialized tensor descriptor describing the initial 
        /// cell state for LSTM networks. The first dimension of the tensor must match the hiddenSize argument 
        /// passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match 
        /// the second dimension of the first tensor described in xDesc. The third dimension must match the numLayers 
        /// argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully 
        /// packed.</param>
        /// <param name="cx">Data pointer to GPU memory associated with the tensor descriptor cxDesc. If a NULL pointer is 
        /// passed, the initial cell state of the network will be initialized to zero.</param>
        /// <param name="wDesc">Handle to a previously initialized filter descriptor describing the weights for the RNN.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor wDesc.</param>
        /// <param name="yDesc">An array of tensor descriptors describing the output from each recurrent iteration. The first 
        /// dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor 
        /// call used to initialize rnnDesc: 
        /// * If direction is CUDNN_UNIDIRECTIONAL the first dimension should match the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// * If direction is CUDNN_BIDIRECTIONAL the first dimension should match double the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// The second dimension of the tensor n must match the second dimension of the tensor 
        /// n in xDesc. The tensor must be fully packed.</param>
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor yDesc.</param>
        /// <param name="hyDesc">Handle to a previously initialized tensor descriptor describing the final 
        /// hidden state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second dimension 
        /// of the first tensor described in xDesc. The third dimension must match the numLayers argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="hy">Data pointer to GPU memory associated with the tensor descriptor hyDesc. If a 
        /// NULL pointer is passed, the final hidden state of the network will not be saved.</param>
        /// <param name="cyDesc">Handle to a previously initialized tensor descriptor describing the final cell state 
        /// for LSTM networks. The first dimension of the tensor must match the hiddenSize argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second dimension 
        /// of the first tensor described in xDesc. The third dimension must match the numLayers argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="cy">Data pointer to GPU memory associated with the tensor descriptor cyDesc. If a NULL pointer is 
        /// passed, the final cell state of the network will be not be saved.</param>
        /// <param name="workspace">Data pointer to GPU memory to be used as a workspace for this call.</param>
        /// <param name="workSpaceSizeInBytes">Specifies the size in bytes of the provided workspace.</param>
        /// <param name="reserveSpace">Data pointer to GPU memory to be used as a reserve space for this call.</param>
        /// <param name="reserveSpaceSizeInBytes">Specifies the size in bytes of the provided reserveSpace.</param>
        public void RNNForwardTraining(
                                                   TensorDescriptor[] xDesc,
                                                   CudaDeviceVariable<float> x,
                                                   TensorDescriptor hxDesc,
                                                   CudaDeviceVariable<float> hx,
                                                   TensorDescriptor cxDesc,
                                                   CudaDeviceVariable<float> cx,
                                                   FilterDescriptor wDesc,
                                                   CudaDeviceVariable<float> w,
                                                   TensorDescriptor[] yDesc,
                                                   CudaDeviceVariable<float> y,
                                                   TensorDescriptor hyDesc,
                                                   CudaDeviceVariable<float> hy,
                                                   TensorDescriptor cyDesc,
                                                   CudaDeviceVariable<float> cy,
                                                   CudaDeviceVariable<byte> workspace,
                                                   SizeT workSpaceSizeInBytes,
                                                   CudaDeviceVariable<byte> reserveSpace,
                                                   SizeT reserveSpaceSizeInBytes)
        {
            var a1 = xDesc.Select(q => q.Desc).ToArray();
            var a2 = yDesc.Select(q => q.Desc).ToArray();
            res = CudaDNNNativeMethods.cudnnRNNForwardTraining(
                _handle, _desc, a1, x.DevicePointer, hxDesc.Desc, hx.DevicePointer, cxDesc.Desc, cx.DevicePointer, wDesc.Desc, w.DevicePointer,
                a2, y.DevicePointer, hyDesc.Desc, hy.DevicePointer, cyDesc.Desc, cy.DevicePointer, workspace.DevicePointer, workSpaceSizeInBytes, reserveSpace.DevicePointer, reserveSpaceSizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnRNNForwardTraining", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This routine executes the recurrent neural network described by rnnDesc with inputs x, hx, cx, weights w 
        /// and outputs y, hy, cy. workspace is required for intermediate storage. reserveSpace stores data required 
        /// for training. The same reserveSpace data must be used for future calls to cudnnRNNBackwardData and 
        /// cudnnRNNBackwardWeights if these execute on the same input data. 
        /// </summary>
        /// <param name="xDesc">An array of tensor descriptors describing the input to each recurrent iteration. Each 
        /// tensor descriptor must have the same first dimension. The second dimension of the tensors may decrease 
        /// from element n to element n+1 but may not increase. The tensor must be fully packed.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptors in the array xDesc.</param>
        /// <param name="hxDesc">Handle to a previously initialized tensor descriptor describing the initial hidden state 
        /// of the RNN. The first dimension of the tensor must match the hiddenSize argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second 
        /// dimension of the first tensor described in xDesc. The third dimension must match the numLayers argument 
        /// passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="hx">Data pointer to GPU memory associated with the tensor descriptor hxDesc. If a NULL pointer 
        /// is passed, the initial hidden state of the network will be initialized to zero.</param>
        /// <param name="cxDesc">Handle to a previously initialized tensor descriptor describing the initial 
        /// cell state for LSTM networks. The first dimension of the tensor must match the hiddenSize argument 
        /// passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match 
        /// the second dimension of the first tensor described in xDesc. The third dimension must match the numLayers 
        /// argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully 
        /// packed.</param>
        /// <param name="cx">Data pointer to GPU memory associated with the tensor descriptor cxDesc. If a NULL pointer is 
        /// passed, the initial cell state of the network will be initialized to zero.</param>
        /// <param name="wDesc">Handle to a previously initialized filter descriptor describing the weights for the RNN.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor wDesc.</param>
        /// <param name="yDesc">An array of tensor descriptors describing the output from each recurrent iteration. The first 
        /// dimension of the tensor depends on the direction argument passed to the cudnnSetRNNDescriptor 
        /// call used to initialize rnnDesc: 
        /// * If direction is CUDNN_UNIDIRECTIONAL the first dimension should match the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// * If direction is CUDNN_BIDIRECTIONAL the first dimension should match double the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// The second dimension of the tensor n must match the second dimension of the tensor 
        /// n in xDesc. The tensor must be fully packed.</param>
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor yDesc.</param>
        /// <param name="hyDesc">Handle to a previously initialized tensor descriptor describing the final 
        /// hidden state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second dimension 
        /// of the first tensor described in xDesc. The third dimension must match the numLayers argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="hy">Data pointer to GPU memory associated with the tensor descriptor hyDesc. If a 
        /// NULL pointer is passed, the final hidden state of the network will not be saved.</param>
        /// <param name="cyDesc">Handle to a previously initialized tensor descriptor describing the final cell state 
        /// for LSTM networks. The first dimension of the tensor must match the hiddenSize argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second dimension 
        /// of the first tensor described in xDesc. The third dimension must match the numLayers argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="cy">Data pointer to GPU memory associated with the tensor descriptor cyDesc. If a NULL pointer is 
        /// passed, the final cell state of the network will be not be saved.</param>
        /// <param name="workspace">Data pointer to GPU memory to be used as a workspace for this call.</param>
        /// <param name="workSpaceSizeInBytes">Specifies the size in bytes of the provided workspace.</param>
        /// <param name="reserveSpace">Data pointer to GPU memory to be used as a reserve space for this call.</param>
        /// <param name="reserveSpaceSizeInBytes">Specifies the size in bytes of the provided reserveSpace.</param>
        public void RNNForwardTraining(
                                                   TensorDescriptor[] xDesc,
                                                   CudaDeviceVariable<double> x,
                                                   TensorDescriptor hxDesc,
                                                   CudaDeviceVariable<double> hx,
                                                   TensorDescriptor cxDesc,
                                                   CudaDeviceVariable<double> cx,
                                                   FilterDescriptor wDesc,
                                                   CudaDeviceVariable<double> w,
                                                   TensorDescriptor[] yDesc,
                                                   CudaDeviceVariable<double> y,
                                                   TensorDescriptor hyDesc,
                                                   CudaDeviceVariable<double> hy,
                                                   TensorDescriptor cyDesc,
                                                   CudaDeviceVariable<double> cy,
                                                   CudaDeviceVariable<byte> workspace,
                                                   SizeT workSpaceSizeInBytes,
                                                   CudaDeviceVariable<byte> reserveSpace,
                                                   SizeT reserveSpaceSizeInBytes)
        {
            var a1 = xDesc.Select(q => q.Desc).ToArray();
            var a2 = yDesc.Select(q => q.Desc).ToArray();
            res = CudaDNNNativeMethods.cudnnRNNForwardTraining(
                _handle, _desc, a1, x.DevicePointer, hxDesc.Desc, hx.DevicePointer, cxDesc.Desc, cx.DevicePointer, wDesc.Desc, w.DevicePointer,
                a2, y.DevicePointer, hyDesc.Desc, hy.DevicePointer, cyDesc.Desc, cy.DevicePointer, workspace.DevicePointer, workSpaceSizeInBytes, reserveSpace.DevicePointer, reserveSpaceSizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnRNNForwardTraining", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This routine executes the recurrent neural network described by rnnDesc with 
        /// output gradients dy, dhy, dhc, weights w and input gradients dx, dhx, dcx. 
        /// workspace is required for intermediate storage. The data in reserveSpace must have 
        /// previously been generated by cudnnRNNForwardTraining. The same reserveSpace data must 
        /// be used for future calls to cudnnRNNBackwardWeights if they execute on the same input data. 
        /// </summary>
        /// <param name="yDesc">An array of tensor descriptors describing the output from each 
        /// recurrent iteration. The first dimension of the tensor depends on the direction 
        /// argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc:
        /// * If direction is CUDNN_UNIDIRECTIONAL the first dimension should match the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// * If direction is CUDNN_BIDIRECTIONAL the first dimension should match double the 
        /// hiddenSize argument passed to cudnnSetRNNDescriptor.
        /// The second dimension of the tensor n must match the second dimension of the tensor n in dyDesc. 
        /// The tensor must be fully packed.</param>
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor yDesc.</param>
        /// <param name="dyDesc">An array of tensor descriptors describing the gradient at the output from each 
        /// recurrent iteration. The first dimension of the tensor depends on the direction argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc: 
        /// * If direction is CUDNN_UNIDIRECTIONAL the first dimension should match the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// * If direction is CUDNN_BIDIRECTIONAL the first dimension should match double the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// The second dimension of the tensor n must match the second dimension of the tensor n in dxDesc. The 
        /// tensor must be fully packed.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the tensor descriptors in the array dyDesc.</param>
        /// <param name="dhyDesc">Handle to a previously initialized tensor descriptor describing the gradients at the 
        /// final hidden state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed 
        /// to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second 
        /// dimension of the first tensor described in dyDesc. The third dimension must match the numLayers argument 
        /// passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="dhy">Data pointer to GPU memory associated with the tensor descriptor dhyDesc. If a NULL pointer 
        /// is passed, the gradients at the final hidden state of the network will be initialized to zero.</param>
        /// <param name="dcyDesc">Handle to a previously initialized tensor descriptor describing the gradients at 
        /// the final cell state of the RNN. The first dimension of the tensor must match the hiddenSize argument 
        /// passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the 
        /// second dimension of the first tensor described in dyDesc. The third dimension must match the numLayers argument 
        /// passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="dcy">Data pointer to GPU memory associated with the tensor descriptor dcyDesc. If a NULL pointer 
        /// is passed, the gradients at the final cell state of the network will be initialized to zero.</param>
        /// <param name="wDesc">Handle to a previously initialized filter descriptor describing the weights for the RNN.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor wDesc.</param>
        /// <param name="hxDesc">Handle to a previously initialized tensor descriptor describing the initial hidden 
        /// state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second 
        /// dimension of the first tensor described in xDesc. The third dimension must match the numLayers 
        /// argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be 
        /// fully packed.</param>
        /// <param name="hx">Data pointer to GPU memory associated with the tensor descriptor hxDesc. If a NULL pointer is 
        /// passed, the initial hidden state of the network will be initialized to zero.</param>
        /// <param name="cxDesc">Handle to a previously initialized tensor descriptor describing the 
        /// initial cell state for LSTM networks. The first dimension of the tensor must match the 
        /// hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The 
        /// second dimension must match the second dimension of the first tensor described in xDesc. The 
        /// third dimension must match the numLayers argument passed to the cudnnSetRNNDescriptor call 
        /// used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="cx">Data pointer to GPU memory associated with the tensor descriptor cxDesc. 
        /// If a NULL pointer is passed, the initial cell state of the network will be initialized to zero.</param>
        /// <param name="dxDesc">An array of tensor descriptors describing the gradient at the input of each recurrent iteration. 
        /// Each tensor descriptor must have the same first dimension. The second dimension of the tensors may decrease from 
        /// element n to element n+1 but may not increase. The tensor must be fully packed.</param>
        /// <param name="dx">Data pointer to GPU memory associated with the tensor descriptors in the array dxDesc. </param>
        /// <param name="dhxDesc">Handle to a previously initialized tensor descriptor describing the gradient at the initial hidden 
        /// state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed to the cudnnSetRNNDescriptor 
        /// call used to initialize rnnDesc. The second dimension must match the second dimension of the first tensor described in xDesc. 
        /// The third dimension must match the numLayers argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. 
        /// The tensor must be fully packed.</param>
        /// <param name="dhx">Data pointer to GPU memory associated with the tensor descriptor dhxDesc. If a NULL pointer is passed, the 
        /// gradient at the hidden input of the network will not be set.</param>
        /// <param name="dcxDesc">Handle to a previously initialized tensor descriptor describing the gradient 
        /// at the initial cell state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed 
        /// to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second dimension 
        /// of the first tensor described in xDesc. The third dimension must match the numLayers argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="dcx">Data pointer to GPU memory associated with the tensor descriptor dcxDesc. If 
        /// a NULL pointer is passed, the gradient at the cell input of the network will not be set.</param>
        /// <param name="workspace">Data pointer to GPU memory to be used as a workspace for this call.</param>
        /// <param name="workSpaceSizeInBytes">Specifies the size in bytes of the provided workspace.</param>
        /// <param name="reserveSpace">Data pointer to GPU memory to be used as a reserve space for this call.</param>
        /// <param name="reserveSpaceSizeInBytes">Specifies the size in bytes of the provided reserveSpace.</param>
        public void RNNBackwardData(
                                                TensorDescriptor[] yDesc,
                                                CudaDeviceVariable<float> y,
                                                TensorDescriptor[] dyDesc,
                                                CudaDeviceVariable<float> dy,
                                                TensorDescriptor dhyDesc,
                                                CudaDeviceVariable<float> dhy,
                                                TensorDescriptor dcyDesc,
                                                CudaDeviceVariable<float> dcy,
                                                FilterDescriptor wDesc,
                                                CudaDeviceVariable<float> w,
                                                TensorDescriptor hxDesc,
                                                CudaDeviceVariable<float> hx,
                                                TensorDescriptor cxDesc,
                                                CudaDeviceVariable<float> cx,
                                                TensorDescriptor[] dxDesc,
                                                CudaDeviceVariable<float> dx,
                                                TensorDescriptor dhxDesc,
                                                CudaDeviceVariable<float> dhx,
                                                TensorDescriptor dcxDesc,
                                                CudaDeviceVariable<float> dcx,
                                                CudaDeviceVariable<byte> workspace,
                                                SizeT workSpaceSizeInBytes,
                                                CudaDeviceVariable<byte> reserveSpace,
                                                SizeT reserveSpaceSizeInBytes)
        {
            var a1 = yDesc.Select(q => q.Desc).ToArray();
            var a2 = dyDesc.Select(q => q.Desc).ToArray();
            var a3 = dxDesc.Select(q => q.Desc).ToArray();
            res = CudaDNNNativeMethods.cudnnRNNBackwardData(
                _handle, _desc, a1, y.DevicePointer, a2, dy.DevicePointer, dhyDesc.Desc, dhy.DevicePointer, dcyDesc.Desc, dcy.DevicePointer, wDesc.Desc, w.DevicePointer, 
                hxDesc.Desc, hx.DevicePointer, cxDesc.Desc, cx.DevicePointer, a3, dx.DevicePointer, dhxDesc.Desc, dhx.DevicePointer, dcxDesc.Desc, dcx.DevicePointer,
                workspace.DevicePointer, workSpaceSizeInBytes, reserveSpace.DevicePointer, reserveSpaceSizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnRNNBackwardData", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This routine executes the recurrent neural network described by rnnDesc with 
        /// output gradients dy, dhy, dhc, weights w and input gradients dx, dhx, dcx. 
        /// workspace is required for intermediate storage. The data in reserveSpace must have 
        /// previously been generated by cudnnRNNForwardTraining. The same reserveSpace data must 
        /// be used for future calls to cudnnRNNBackwardWeights if they execute on the same input data. 
        /// </summary>
        /// <param name="yDesc">An array of tensor descriptors describing the output from each 
        /// recurrent iteration. The first dimension of the tensor depends on the direction 
        /// argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc:
        /// * If direction is CUDNN_UNIDIRECTIONAL the first dimension should match the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// * If direction is CUDNN_BIDIRECTIONAL the first dimension should match double the 
        /// hiddenSize argument passed to cudnnSetRNNDescriptor.
        /// The second dimension of the tensor n must match the second dimension of the tensor n in dyDesc. 
        /// The tensor must be fully packed.</param>
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor yDesc.</param>
        /// <param name="dyDesc">An array of tensor descriptors describing the gradient at the output from each 
        /// recurrent iteration. The first dimension of the tensor depends on the direction argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc: 
        /// * If direction is CUDNN_UNIDIRECTIONAL the first dimension should match the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// * If direction is CUDNN_BIDIRECTIONAL the first dimension should match double the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// The second dimension of the tensor n must match the second dimension of the tensor n in dxDesc. The 
        /// tensor must be fully packed.</param>
        /// <param name="dy">Data pointer to GPU memory associated with the tensor descriptors in the array dyDesc.</param>
        /// <param name="dhyDesc">Handle to a previously initialized tensor descriptor describing the gradients at the 
        /// final hidden state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed 
        /// to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second 
        /// dimension of the first tensor described in dyDesc. The third dimension must match the numLayers argument 
        /// passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="dhy">Data pointer to GPU memory associated with the tensor descriptor dhyDesc. If a NULL pointer 
        /// is passed, the gradients at the final hidden state of the network will be initialized to zero.</param>
        /// <param name="dcyDesc">Handle to a previously initialized tensor descriptor describing the gradients at 
        /// the final cell state of the RNN. The first dimension of the tensor must match the hiddenSize argument 
        /// passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the 
        /// second dimension of the first tensor described in dyDesc. The third dimension must match the numLayers argument 
        /// passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="dcy">Data pointer to GPU memory associated with the tensor descriptor dcyDesc. If a NULL pointer 
        /// is passed, the gradients at the final cell state of the network will be initialized to zero.</param>
        /// <param name="wDesc">Handle to a previously initialized filter descriptor describing the weights for the RNN.</param>
        /// <param name="w">Data pointer to GPU memory associated with the filter descriptor wDesc.</param>
        /// <param name="hxDesc">Handle to a previously initialized tensor descriptor describing the initial hidden 
        /// state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second 
        /// dimension of the first tensor described in xDesc. The third dimension must match the numLayers 
        /// argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be 
        /// fully packed.</param>
        /// <param name="hx">Data pointer to GPU memory associated with the tensor descriptor hxDesc. If a NULL pointer is 
        /// passed, the initial hidden state of the network will be initialized to zero.</param>
        /// <param name="cxDesc">Handle to a previously initialized tensor descriptor describing the 
        /// initial cell state for LSTM networks. The first dimension of the tensor must match the 
        /// hiddenSize argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The 
        /// second dimension must match the second dimension of the first tensor described in xDesc. The 
        /// third dimension must match the numLayers argument passed to the cudnnSetRNNDescriptor call 
        /// used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="cx">Data pointer to GPU memory associated with the tensor descriptor cxDesc. 
        /// If a NULL pointer is passed, the initial cell state of the network will be initialized to zero.</param>
        /// <param name="dxDesc">An array of tensor descriptors describing the gradient at the input of each recurrent iteration. 
        /// Each tensor descriptor must have the same first dimension. The second dimension of the tensors may decrease from 
        /// element n to element n+1 but may not increase. The tensor must be fully packed.</param>
        /// <param name="dx">Data pointer to GPU memory associated with the tensor descriptors in the array dxDesc. </param>
        /// <param name="dhxDesc">Handle to a previously initialized tensor descriptor describing the gradient at the initial hidden 
        /// state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed to the cudnnSetRNNDescriptor 
        /// call used to initialize rnnDesc. The second dimension must match the second dimension of the first tensor described in xDesc. 
        /// The third dimension must match the numLayers argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc. 
        /// The tensor must be fully packed.</param>
        /// <param name="dhx">Data pointer to GPU memory associated with the tensor descriptor dhxDesc. If a NULL pointer is passed, the 
        /// gradient at the hidden input of the network will not be set.</param>
        /// <param name="dcxDesc">Handle to a previously initialized tensor descriptor describing the gradient 
        /// at the initial cell state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed 
        /// to the cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second dimension 
        /// of the first tensor described in xDesc. The third dimension must match the numLayers argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed.</param>
        /// <param name="dcx">Data pointer to GPU memory associated with the tensor descriptor dcxDesc. If 
        /// a NULL pointer is passed, the gradient at the cell input of the network will not be set.</param>
        /// <param name="workspace">Data pointer to GPU memory to be used as a workspace for this call.</param>
        /// <param name="workSpaceSizeInBytes">Specifies the size in bytes of the provided workspace.</param>
        /// <param name="reserveSpace">Data pointer to GPU memory to be used as a reserve space for this call.</param>
        /// <param name="reserveSpaceSizeInBytes">Specifies the size in bytes of the provided reserveSpace.</param>
        public void RNNBackwardData(
                                                TensorDescriptor[] yDesc,
                                                CudaDeviceVariable<double> y,
                                                TensorDescriptor[] dyDesc,
                                                CudaDeviceVariable<double> dy,
                                                TensorDescriptor dhyDesc,
                                                CudaDeviceVariable<double> dhy,
                                                TensorDescriptor dcyDesc,
                                                CudaDeviceVariable<double> dcy,
                                                FilterDescriptor wDesc,
                                                CudaDeviceVariable<double> w,
                                                TensorDescriptor hxDesc,
                                                CudaDeviceVariable<double> hx,
                                                TensorDescriptor cxDesc,
                                                CudaDeviceVariable<double> cx,
                                                TensorDescriptor[] dxDesc,
                                                CudaDeviceVariable<double> dx,
                                                TensorDescriptor dhxDesc,
                                                CudaDeviceVariable<double> dhx,
                                                TensorDescriptor dcxDesc,
                                                CudaDeviceVariable<double> dcx,
                                                CudaDeviceVariable<byte> workspace,
                                                SizeT workSpaceSizeInBytes,
                                                CudaDeviceVariable<byte> reserveSpace,
                                                SizeT reserveSpaceSizeInBytes)
        {
            var a1 = yDesc.Select(q => q.Desc).ToArray();
            var a2 = dyDesc.Select(q => q.Desc).ToArray();
            var a3 = dxDesc.Select(q => q.Desc).ToArray();
            res = CudaDNNNativeMethods.cudnnRNNBackwardData(
                _handle, _desc, a1, y.DevicePointer, a2, dy.DevicePointer, dhyDesc.Desc, dhy.DevicePointer, dcyDesc.Desc, dcy.DevicePointer, wDesc.Desc, w.DevicePointer,
                hxDesc.Desc, hx.DevicePointer, cxDesc.Desc, cx.DevicePointer, a3, dx.DevicePointer, dhxDesc.Desc, dhx.DevicePointer, dcxDesc.Desc, dcx.DevicePointer,
                workspace.DevicePointer, workSpaceSizeInBytes, reserveSpace.DevicePointer, reserveSpaceSizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnRNNBackwardData", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }


        /// <summary>
        /// This routine accumulates weight gradients dw from the recurrent neural network described 
        /// by rnnDesc with inputs x, hx, and outputs y. The mode of operation in this case is additive, 
        /// the weight gradients calculated will be added to those already existing in dw. workspace 
        /// is required for intermediate storage. The data in reserveSpace must have previously been 
        /// generated by cudnnRNNBackwardData.
        /// </summary>
        /// <param name="xDesc">An array of tensor descriptors describing the input to each recurrent iteration. 
        /// Each tensor descriptor must have the same first dimension. The second dimension of the tensors may 
        /// decrease from element n to element n+1 but may not increase. The tensor must be fully packed.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptors in the array xDesc.</param>
        /// <param name="hxDesc">Handle to a previously initialized tensor descriptor describing the initial hidden 
        /// state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second dimension
        /// of the first tensor described in xDesc. The third dimension must match the numLayers argument passed to 
        /// the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed. </param>
        /// <param name="hx">Data pointer to GPU memory associated with the tensor descriptor hxDesc. If 
        /// a NULL pointer is passed, the initial hidden state of the network will be initialized to zero.</param>
        /// <param name="yDesc">An array of tensor descriptors describing the output from each 
        /// recurrent iteration. The first dimension of the tensor depends on the direction 
        /// argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc:
        /// * If direction is CUDNN_UNIDIRECTIONAL the first dimension should match the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// * If direction is CUDNN_BIDIRECTIONAL the first dimension should match double the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// The second dimension of the tensor n must match the second dimension of the tensor n in dyDesc. 
        /// The tensor must be fully packed.</param>
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor yDesc.</param>
        /// <param name="workspace">Data pointer to GPU memory to be used as a workspace for this call.</param>
        /// <param name="workSpaceSizeInBytes">Specifies the size in bytes of the provided workspace.</param>
        /// <param name="dwDesc">Handle to a previously initialized filter descriptor describing the gradients of the weights for the RNN.</param>
        /// <param name="dw">Data pointer to GPU memory associated with the filter descriptor dwDesc.</param>
        /// <param name="reserveSpace">Data pointer to GPU memory to be used as a reserve space for this call.</param>
        /// <param name="reserveSpaceSizeInBytes">Specifies the size in bytes of the provided reserveSpace.</param>
        public void RNNBackwardWeights(
                                                   TensorDescriptor[] xDesc,
                                                   CudaDeviceVariable<float> x,
                                                   TensorDescriptor hxDesc,
                                                   CudaDeviceVariable<float> hx,
                                                   TensorDescriptor[] yDesc,
                                                   CudaDeviceVariable<float> y,
                                                   CudaDeviceVariable<byte> workspace,
                                                   SizeT workSpaceSizeInBytes,
                                                   FilterDescriptor dwDesc,
                                                   CudaDeviceVariable<float> dw,
                                                   CudaDeviceVariable<byte> reserveSpace,
                                                   SizeT reserveSpaceSizeInBytes)
        {
            var a1 = xDesc.Select(q => q.Desc).ToArray();
            var a2 = yDesc.Select(q => q.Desc).ToArray();
            res = CudaDNNNativeMethods.cudnnRNNBackwardWeights(
                _handle, _desc, a1, x.DevicePointer, hxDesc.Desc, hx.DevicePointer, a2, y.DevicePointer, workspace.DevicePointer, workSpaceSizeInBytes, dwDesc.Desc, dw.DevicePointer, reserveSpace.DevicePointer, reserveSpaceSizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnRNNBackwardWeights", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }


        /// <summary>
        /// This routine accumulates weight gradients dw from the recurrent neural network described 
        /// by rnnDesc with inputs x, hx, and outputs y. The mode of operation in this case is additive, 
        /// the weight gradients calculated will be added to those already existing in dw. workspace 
        /// is required for intermediate storage. The data in reserveSpace must have previously been 
        /// generated by cudnnRNNBackwardData.
        /// </summary>
        /// <param name="xDesc">An array of tensor descriptors describing the input to each recurrent iteration. 
        /// Each tensor descriptor must have the same first dimension. The second dimension of the tensors may 
        /// decrease from element n to element n+1 but may not increase. The tensor must be fully packed.</param>
        /// <param name="x">Data pointer to GPU memory associated with the tensor descriptors in the array xDesc.</param>
        /// <param name="hxDesc">Handle to a previously initialized tensor descriptor describing the initial hidden 
        /// state of the RNN. The first dimension of the tensor must match the hiddenSize argument passed to the 
        /// cudnnSetRNNDescriptor call used to initialize rnnDesc. The second dimension must match the second dimension
        /// of the first tensor described in xDesc. The third dimension must match the numLayers argument passed to 
        /// the cudnnSetRNNDescriptor call used to initialize rnnDesc. The tensor must be fully packed. </param>
        /// <param name="hx">Data pointer to GPU memory associated with the tensor descriptor hxDesc. If 
        /// a NULL pointer is passed, the initial hidden state of the network will be initialized to zero.</param>
        /// <param name="yDesc">An array of tensor descriptors describing the output from each 
        /// recurrent iteration. The first dimension of the tensor depends on the direction 
        /// argument passed to the cudnnSetRNNDescriptor call used to initialize rnnDesc:
        /// * If direction is CUDNN_UNIDIRECTIONAL the first dimension should match the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// * If direction is CUDNN_BIDIRECTIONAL the first dimension should match double the hiddenSize 
        /// argument passed to cudnnSetRNNDescriptor.
        /// The second dimension of the tensor n must match the second dimension of the tensor n in dyDesc. 
        /// The tensor must be fully packed.</param>
        /// <param name="y">Data pointer to GPU memory associated with the output tensor descriptor yDesc.</param>
        /// <param name="workspace">Data pointer to GPU memory to be used as a workspace for this call.</param>
        /// <param name="workSpaceSizeInBytes">Specifies the size in bytes of the provided workspace.</param>
        /// <param name="dwDesc">Handle to a previously initialized filter descriptor describing the gradients of the weights for the RNN.</param>
        /// <param name="dw">Data pointer to GPU memory associated with the filter descriptor dwDesc.</param>
        /// <param name="reserveSpace">Data pointer to GPU memory to be used as a reserve space for this call.</param>
        /// <param name="reserveSpaceSizeInBytes">Specifies the size in bytes of the provided reserveSpace.</param>
        public void RNNBackwardWeights(
                                                   TensorDescriptor[] xDesc,
                                                   CudaDeviceVariable<double> x,
                                                   TensorDescriptor hxDesc,
                                                   CudaDeviceVariable<double> hx,
                                                   TensorDescriptor[] yDesc,
                                                   CudaDeviceVariable<double> y,
                                                   CudaDeviceVariable<byte> workspace,
                                                   SizeT workSpaceSizeInBytes,
                                                   FilterDescriptor dwDesc,
                                                   CudaDeviceVariable<double> dw,
                                                   CudaDeviceVariable<byte> reserveSpace,
                                                   SizeT reserveSpaceSizeInBytes)
        {
            var a1 = xDesc.Select(q => q.Desc).ToArray();
            var a2 = yDesc.Select(q => q.Desc).ToArray();
            res = CudaDNNNativeMethods.cudnnRNNBackwardWeights(
                _handle, _desc, a1, x.DevicePointer, hxDesc.Desc, hx.DevicePointer, a2, y.DevicePointer, workspace.DevicePointer, workSpaceSizeInBytes, dwDesc.Desc, dw.DevicePointer, reserveSpace.DevicePointer, reserveSpaceSizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnRNNBackwardWeights", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }


        /// <summary>
        /// This function sets the persistent RNN plan to be executed when using rnnDesc and
        /// CUDNN_RNN_ALGO_PERSIST_DYNAMIC algo.
        /// </summary>
        /// <param name="plan"></param>
        public void SetPersistentRNNPlan(PersistentRNNPlan plan)
        {
            res = CudaDNNNativeMethods.cudnnSetPersistentRNNPlan(_desc, plan.Plan);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetPersistentRNNPlan", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }



        /// <summary>
        /// The math type specified in a given RNN descriptor.
        /// </summary>
        public cudnnMathType MathType
        {
            //get
            //{
            //    cudnnMathType mathType = new cudnnMathType();
            //    res = CudaDNNNativeMethods.cudnnGetConvolutionMathType(_desc, ref mathType);
            //    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetConvolutionMathType", res));
            //    if (res != cudnnStatus.Success) throw new CudaDNNException(res);
            //    return mathType;
            //}
            set
            {
                res = CudaDNNNativeMethods.cudnnSetRNNMatrixMathType(_desc, value);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetRNNMatrixMathType", res));
                if (res != cudnnStatus.Success) throw new CudaDNNException(res);
            }
        }
    }
}
