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
    public class RNNDataDescriptor : IDisposable
    {

        private cudnnRNNDataDescriptor _desc;
        private cudnnStatus res;
        private bool disposed;
        private cudnnHandle _handle;

        #region Contructors
        /// <summary>
        /// </summary>
        public RNNDataDescriptor(CudaDNNContext context)
        {
            _handle = context.Handle;
            _desc = new cudnnRNNDataDescriptor();
            res = CudaDNNNativeMethods.cudnnCreateRNNDataDescriptor(ref _desc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCreateRNNDataDescriptor", res));
            if (res != cudnnStatus.Success)
                throw new CudaDNNException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~RNNDataDescriptor()
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
                res = CudaDNNNativeMethods.cudnnDestroyRNNDataDescriptor(_desc);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDestroyRNNDataDescriptor", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        /// <summary>
        /// Returns the inner handle.
        /// </summary>
        public cudnnRNNDataDescriptor Desc
        {
            get { return _desc; }
        }



        /// <summary>
        /// 
        /// </summary>
        public void SetRNNDataDescriptor(cudnnDataType dataType,
            cudnnRNNDataLayout layout, int maxSeqLength, int batchSize,
            int vectorSize,
            int[] seqLengthArray, /* length of each sequence in the batch */
            float paddingFill)
        {
            res = CudaDNNNativeMethods.cudnnSetRNNDataDescriptor(_desc, dataType,
            layout, maxSeqLength, batchSize, vectorSize, seqLengthArray, ref paddingFill);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetRNNDataDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }
        /// <summary>
        /// 
        /// </summary>
        public void SetRNNDataDescriptor(cudnnDataType dataType,
            cudnnRNNDataLayout layout, int maxSeqLength, int batchSize,
            int vectorSize,
            int[] seqLengthArray, /* length of each sequence in the batch */
            double paddingFill)
        {
            res = CudaDNNNativeMethods.cudnnSetRNNDataDescriptor(_desc, dataType,
            layout, maxSeqLength, batchSize, vectorSize, seqLengthArray, ref paddingFill);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetRNNDataDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }
        /// <summary>
        /// 
        /// </summary>
        public void SetRNNDataDescriptor(cudnnDataType dataType,
            cudnnRNNDataLayout layout, int maxSeqLength, int batchSize,
            int vectorSize,
            int[] seqLengthArray, /* length of each sequence in the batch */
            IntPtr paddingFill)
        {
            res = CudaDNNNativeMethods.cudnnSetRNNDataDescriptor(_desc, dataType,
            layout, maxSeqLength, batchSize, vectorSize, seqLengthArray, paddingFill);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetRNNDataDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// 
        /// </summary>
        public void GetRNNDataDescriptor(ref cudnnDataType dataType,
            ref cudnnRNNDataLayout layout, ref int maxSeqLength, ref int batchSize,
            ref int vectorSize, int arrayLengthRequested, int[] seqLengthArray,
            ref float paddingFill)
        {
            res = CudaDNNNativeMethods.cudnnGetRNNDataDescriptor(_desc, ref dataType,
            ref layout, ref maxSeqLength, ref batchSize,
            ref vectorSize, arrayLengthRequested, seqLengthArray,
            ref paddingFill);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetRNNDataDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }
        /// <summary>
        /// 
        /// </summary>
        public void GetRNNDataDescriptor(ref cudnnDataType dataType,
            ref cudnnRNNDataLayout layout, ref int maxSeqLength, ref int batchSize,
            ref int vectorSize, int arrayLengthRequested, int[] seqLengthArray,
            ref double paddingFill)
        {
            res = CudaDNNNativeMethods.cudnnGetRNNDataDescriptor(_desc, ref dataType,
            ref layout, ref maxSeqLength, ref batchSize,
            ref vectorSize, arrayLengthRequested, seqLengthArray,
            ref paddingFill);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetRNNDataDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }
        /// <summary>
        /// 
        /// </summary>
        public void GetRNNDataDescriptor(ref cudnnDataType dataType,
            ref cudnnRNNDataLayout layout, ref int maxSeqLength, ref int batchSize,
            ref int vectorSize, int arrayLengthRequested, int[] seqLengthArray,
            IntPtr paddingFill)
        {
            res = CudaDNNNativeMethods.cudnnGetRNNDataDescriptor(_desc, ref dataType,
            ref layout, ref maxSeqLength, ref batchSize,
            ref vectorSize, arrayLengthRequested, seqLengthArray,
            paddingFill);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetRNNDataDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }



    }
}
