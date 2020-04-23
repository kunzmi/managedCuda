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
    /// </summary>
    public class SeqDataDescriptor : IDisposable
    {
        private cudnnSeqDataDescriptor _desc;
        private cudnnStatus res;
        private bool disposed;

        #region Contructors
        /// <summary>
        /// </summary>
        public SeqDataDescriptor()
        {
            _desc = new cudnnSeqDataDescriptor();
            res = CudaDNNNativeMethods.cudnnCreateSeqDataDescriptor(ref _desc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCreateSeqDataDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~SeqDataDescriptor()
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
                res = CudaDNNNativeMethods.cudnnDestroySeqDataDescriptor(_desc);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDestroySeqDataDescriptor", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        /// <summary>
        /// Returns the inner handle.
        /// </summary>
        public cudnnSeqDataDescriptor Desc
        {
            get { return _desc; }
        }

        public void SetSeqDataDescriptor(cudnnDataType dataType, int nbDims,
            int[] dimA, cudnnSeqDataAxis[] axes,
            int[] seqLengthArray, float paddingFill)
        {
            res = CudaDNNNativeMethods.cudnnSetSeqDataDescriptor(_desc, dataType, nbDims,
            dimA, axes,
            seqLengthArray.Length, seqLengthArray, ref paddingFill);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetSeqDataDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }
        public void SetSeqDataDescriptor(cudnnDataType dataType, int nbDims,
            int[] dimA, cudnnSeqDataAxis[] axes,
            int[] seqLengthArray, double paddingFill)
        {
            res = CudaDNNNativeMethods.cudnnSetSeqDataDescriptor(_desc, dataType, nbDims,
            dimA, axes,
            seqLengthArray.Length, seqLengthArray, ref paddingFill);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetSeqDataDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }
        public void SetSeqDataDescriptor(cudnnDataType dataType, int nbDims,
            int[] dimA, cudnnSeqDataAxis[] axes,
            int[] seqLengthArray, IntPtr paddingFill)
        {
            res = CudaDNNNativeMethods.cudnnSetSeqDataDescriptor(_desc, dataType, nbDims,
            dimA, axes,
            seqLengthArray.Length, seqLengthArray, paddingFill);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetSeqDataDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }


        public void GetSeqDataDescriptor(ref cudnnDataType dataType, ref int nbDims,
            int[] dimA, cudnnSeqDataAxis[] axes,
            ref SizeT seqLengthArraySize, int[] seqLengthArray, ref float paddingFill)
        {
            res = CudaDNNNativeMethods.cudnnGetSeqDataDescriptor(_desc, ref dataType, ref nbDims,
            dimA.Length, dimA, axes,
            ref seqLengthArraySize, dimA.Length, seqLengthArray, ref paddingFill);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetSeqDataDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        public void GetSeqDataDescriptor(ref cudnnDataType dataType, ref int nbDims,
            int[] dimA, cudnnSeqDataAxis[] axes,
            ref SizeT seqLengthArraySize, int[] seqLengthArray, ref double paddingFill)
        {
            res = CudaDNNNativeMethods.cudnnGetSeqDataDescriptor(_desc, ref dataType, ref nbDims,
            dimA.Length, dimA, axes,
            ref seqLengthArraySize, dimA.Length, seqLengthArray, ref paddingFill);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetSeqDataDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }
        public void GetSeqDataDescriptor(ref cudnnDataType dataType, ref int nbDims,
           int[] dimA, cudnnSeqDataAxis[] axes,
           ref SizeT seqLengthArraySize, int[] seqLengthArray, IntPtr paddingFill)
        {
            res = CudaDNNNativeMethods.cudnnGetSeqDataDescriptor(_desc, ref dataType, ref nbDims,
            dimA.Length, dimA, axes,
            ref seqLengthArraySize, dimA.Length, seqLengthArray, paddingFill);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetSeqDataDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }
    }
}
