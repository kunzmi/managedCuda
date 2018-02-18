//	Copyright (c) 2018, Michael Kunz. All rights reserved.
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
    public class CTCLossDescriptor : IDisposable
    {

        private cudnnCTCLossDescriptor _desc;
        private cudnnStatus res;
        private bool disposed;
        private cudnnHandle _handle;

        #region Contructors
        /// <summary>
        /// </summary>
        public CTCLossDescriptor(CudaDNNContext context)
        {
            _handle = context.Handle;
            _desc = new cudnnCTCLossDescriptor();
            res = CudaDNNNativeMethods.cudnnCreateCTCLossDescriptor(ref _desc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCreateCTCLossDescriptor", res));
            if (res != cudnnStatus.Success)
                throw new CudaDNNException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~CTCLossDescriptor()
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
                res = CudaDNNNativeMethods.cudnnDestroyCTCLossDescriptor(_desc);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDestroyCTCLossDescriptor", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        /// <summary>
        /// Returns the inner handle.
        /// </summary>
        public cudnnCTCLossDescriptor Desc
        {
            get { return _desc; }
        }


        /// <summary>
        /// This function initializes a previously created CTC Loss descriptor object.
        /// </summary>
        /// <param name="dataType">Math precision.</param>
        public void SetCTCLossDescriptor(cudnnDataType dataType)
        {
            res = CudaDNNNativeMethods.cudnnSetCTCLossDescriptor(_desc, dataType);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetCTCLossDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }


        /// <summary>
        /// This function returns the ctc costs and gradients, given the probabilities and labels.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="probsDesc">Handle to the previously initialized probabilities tensor descriptor.</param>
        /// <param name="probs">Pointer to a previously initialized probabilities tensor.</param>
        /// <param name="labels">Pointer to a previously initialized labels list.</param>
        /// <param name="labelLengths">Pointer to a previously initialized lengths list, to walk the above labels list.</param>
        /// <param name="inputLengths">Pointer to a previously initialized list of the lengths of the timing steps in each batch.</param>
        /// <param name="costs">Pointer to the computed costs of CTC.</param>
        /// <param name="gradientsDesc">Handle to a previously initialized gradients tensor descriptor.</param>
        /// <param name="gradients">Pointer to the computed gradients of CTC.</param>
        /// <param name="algo">Enumerant that specifies the chosen CTC loss algorithm.</param>
        /// <param name="workspace">Pointer to GPU memory of a workspace needed to able to execute the specified algorithm.</param>
        public void CTCLoss(CudaDNNContext handle,
            TensorDescriptor probsDesc,     /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is the timing steps, N is the mini batch size, A is the alphabet size)  */
                                        CudaDeviceVariable<float> probs,                          /* probabilities after softmax, in GPU memory */
                                        int[] labels,                          /* labels, in CPU memory */
                                        int[] labelLengths,                    /* the length of each label, in CPU memory */
                                        int[] inputLengths,                    /* the lengths of timing steps in each batch, in CPU memory */
                                        CudaDeviceVariable<float> costs,                                /* the returned costs of CTC, in GPU memory */
                                        TensorDescriptor gradientsDesc, /* Tensor descriptor for gradients, the dimensions are T,N,A */
                                        CudaDeviceVariable<float> gradients,                      /* the returned CTC gradients, in GPU memory, to compute costs only, set it to NULL */
                                        cudnnCTCLossAlgo algo,                     /* algorithm selected, supported now 0 and 1 */
                                        CudaDeviceVariable<byte> workspace                            /* pointer to the workspace, in GPU memory */
                                        )
        {
            res = CudaDNNNativeMethods.cudnnCTCLoss(handle.Handle, probsDesc.Desc, probs.DevicePointer, labels, labelLengths, inputLengths, costs.DevicePointer,
                gradientsDesc.Desc, gradients.DevicePointer, algo, _desc, workspace.DevicePointer, workspace.SizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCTCLoss", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }


        /// <summary>
        /// This function returns the ctc costs and gradients, given the probabilities and labels.
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="probsDesc">Handle to the previously initialized probabilities tensor descriptor.</param>
        /// <param name="probs">Pointer to a previously initialized probabilities tensor.</param>
        /// <param name="labels">Pointer to a previously initialized labels list.</param>
        /// <param name="labelLengths">Pointer to a previously initialized lengths list, to walk the above labels list.</param>
        /// <param name="inputLengths">Pointer to a previously initialized list of the lengths of the timing steps in each batch.</param>
        /// <param name="costs">Pointer to the computed costs of CTC.</param>
        /// <param name="gradientsDesc">Handle to a previously initialized gradients tensor descriptor.</param>
        /// <param name="gradients">Pointer to the computed gradients of CTC.</param>
        /// <param name="algo">Enumerant that specifies the chosen CTC loss algorithm.</param>
        /// <param name="workspace">Pointer to GPU memory of a workspace needed to able to execute the specified algorithm.</param>
        public void CTCLoss(CudaDNNContext handle,
            TensorDescriptor probsDesc,     /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is the timing steps, N is the mini batch size, A is the alphabet size)  */
                                        CudaDeviceVariable<double> probs,                          /* probabilities after softmax, in GPU memory */
                                        int[] labels,                          /* labels, in CPU memory */
                                        int[] labelLengths,                    /* the length of each label, in CPU memory */
                                        int[] inputLengths,                    /* the lengths of timing steps in each batch, in CPU memory */
                                        CudaDeviceVariable<double> costs,                                /* the returned costs of CTC, in GPU memory */
                                        TensorDescriptor gradientsDesc, /* Tensor descriptor for gradients, the dimensions are T,N,A */
                                        CudaDeviceVariable<double> gradients,                      /* the returned CTC gradients, in GPU memory, to compute costs only, set it to NULL */
                                        cudnnCTCLossAlgo algo,                     /* algorithm selected, supported now 0 and 1 */
                                        CudaDeviceVariable<byte> workspace                            /* pointer to the workspace, in GPU memory */
                                        )
        {
            res = CudaDNNNativeMethods.cudnnCTCLoss(handle.Handle, probsDesc.Desc, probs.DevicePointer, labels, labelLengths, inputLengths, costs.DevicePointer,
                gradientsDesc.Desc, gradients.DevicePointer, algo, _desc, workspace.DevicePointer, workspace.SizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCTCLoss", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }


        /// <summary>
        /// return the workspace size needed for ctc
        /// </summary>
        /// <param name="handle">Handle to a previously created cuDNN context.</param>
        /// <param name="probsDesc">Handle to the previously initialized probabilities tensor descriptor.</param>
        /// <param name="gradientsDesc">Handle to a previously initialized gradients tensor descriptor.</param>
        /// <param name="labels">Pointer to a previously initialized labels list.</param>
        /// <param name="labelLengths">Pointer to a previously initialized lengths list, to walk the above labels list.</param>
        /// <param name="inputLengths">Pointer to a previously initialized list of the lengths of the timing steps in each batch.</param>
        /// <param name="algo">Enumerant that specifies the chosen CTC loss algorithm</param>
        /// <returns>Amount of GPU memory needed as workspace to be able to execute the CTC
        /// loss computation with the specified algo.</returns>
        public SizeT CTCLoss(CudaDNNContext handle,
            TensorDescriptor probsDesc,     /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is the timing steps, N is the mini batch size, A is the alphabet size)  */
                                        int[] labels,                          /* labels, in CPU memory */
                                        int[] labelLengths,                    /* the length of each label, in CPU memory */
                                        int[] inputLengths,                    /* the lengths of timing steps in each batch, in CPU memory */
                                        TensorDescriptor gradientsDesc, /* Tensor descriptor for gradients, the dimensions are T,N,A */
                                        cudnnCTCLossAlgo algo                     /* algorithm selected, supported now 0 and 1 */
                                        )
        {
            SizeT size = new SizeT();
            res = CudaDNNNativeMethods.cudnnGetCTCLossWorkspaceSize(handle.Handle, probsDesc.Desc, gradientsDesc.Desc, labels, labelLengths, inputLengths,
                algo, _desc, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetCTCLossWorkspaceSize", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
            return size;
        }



    }
}
