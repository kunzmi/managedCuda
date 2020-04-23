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
    public class TensorTransformDescriptor : IDisposable
    {
        private cudnnTensorTransformDescriptor _desc;
        private cudnnStatus res;
        private bool disposed;

        #region Contructors
        /// <summary>
        /// </summary>
        public TensorTransformDescriptor()
        {
            _desc = new cudnnTensorTransformDescriptor();
            res = CudaDNNNativeMethods.cudnnCreateTensorTransformDescriptor(ref _desc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCreateTensorTransformDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~TensorTransformDescriptor()
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
                res = CudaDNNNativeMethods.cudnnDestroyTensorTransformDescriptor(_desc);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDestroyTensorTransformDescriptor", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        /// <summary>
        /// Returns the inner handle.
        /// </summary>
        public cudnnTensorTransformDescriptor Desc
        {
            get { return _desc; }
        }

        public SizeT InitTransformDest(TensorDescriptor srcDesc, TensorDescriptor destDesc)
        {
            SizeT destSizeInBytes = new SizeT();
            res = CudaDNNNativeMethods.cudnnInitTransformDest(_desc, srcDesc.Desc, destDesc.Desc, ref destSizeInBytes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnInitTransformDest", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
            return destSizeInBytes;
        }

        /// <summary>
        /// Initialize a previously created tensor transform descriptor.
        /// </summary>
        public void SetTensorTransformDescriptor(uint nbDims,
            cudnnTensorFormat destFormat, int[] padBeforeA,
            int[] padAfterA, uint[] foldA,
            cudnnFoldingDirection direction)
        {
            res = CudaDNNNativeMethods.cudnnSetTensorTransformDescriptor(_desc, nbDims, destFormat, padBeforeA, padAfterA, foldA, direction);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetTensorTransformDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// Retrieves the values stored in a previously initialized tensor transform descriptor.
        /// </summary>
        public void GetTensorTransformDescriptor(uint nbDims,
            cudnnTensorFormat destFormat, int[] padBeforeA,
            int[] padAfterA, uint[] foldA,
            cudnnFoldingDirection direction)
        {
            res = CudaDNNNativeMethods.cudnnGetTensorTransformDescriptor(_desc, nbDims, ref destFormat, padBeforeA, padAfterA, foldA, ref direction);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetTensorTransformDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }
    }
}
