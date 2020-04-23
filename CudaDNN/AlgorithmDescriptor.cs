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
using System.Runtime.InteropServices;

namespace ManagedCuda.CudaDNN
{
    /// <summary>
    /// 
    /// </summary>
    public class AlgorithmDescriptor : IDisposable
    {
        private cudnnAlgorithmDescriptor _desc;
        private cudnnStatus res;
        private bool disposed;

        #region Contructors
        /// <summary>
        /// An opaque structure holding the description of an activation operation.
        /// </summary>
        public AlgorithmDescriptor()
        {
            _desc = new cudnnAlgorithmDescriptor();
            res = CudaDNNNativeMethods.cudnnCreateAlgorithmDescriptor(ref _desc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCreateAlgorithmDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }
        /// <summary>
        /// An opaque structure holding the description of an activation operation.
        /// </summary>
        public AlgorithmDescriptor(cudnnAlgorithmDescriptor desc)
        {
            _desc = desc;
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~AlgorithmDescriptor()
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
                res = CudaDNNNativeMethods.cudnnDestroyAlgorithmDescriptor(_desc);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDestroyAlgorithmDescriptor", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        /// <summary>
        /// Returns the inner handle.
        /// </summary>
        public cudnnAlgorithmDescriptor Desc
        {
            get { return _desc; }
        }


        public void SetAlgorithmDescriptor(cudnnAlgorithm algorithm)
        {
            res = CudaDNNNativeMethods.cudnnSetAlgorithmDescriptor(_desc, algorithm);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetAlgorithmDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }


        public cudnnAlgorithm GetAlgorithmDescriptor()
        {
            cudnnAlgorithm algo = new cudnnAlgorithm();
            res = CudaDNNNativeMethods.cudnnGetAlgorithmDescriptor(_desc, ref algo);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetAlgorithmDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
            return algo;
        }


        public void CopyAlgorithmDescriptor(AlgorithmDescriptor dest)
        {
            res = CudaDNNNativeMethods.cudnnCopyAlgorithmDescriptor(_desc, dest.Desc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCopyAlgorithmDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }
        public SizeT GetAlgorithmSpaceSize(CudaDNNContext ctx)
        {
            SizeT size = new SizeT();
            res = CudaDNNNativeMethods.cudnnGetAlgorithmSpaceSize(ctx.Handle, _desc, ref size);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetAlgorithmSpaceSize", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
            return size;
        }


        public void SaveAlgorithm(CudaDNNContext ctx, byte[] algoSpace)
        {
            GCHandle handle = GCHandle.Alloc(algoSpace, GCHandleType.Pinned);
            try
            {
                IntPtr ptr = handle.AddrOfPinnedObject();
                res = CudaDNNNativeMethods.cudnnSaveAlgorithm(ctx.Handle, _desc, ptr, algoSpace.Length);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSaveAlgorithm", res));
            }
            finally
            {
                handle.Free();
            }
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        public void RestoreAlgorithm(CudaDNNContext ctx, byte[] algoSpace)
        {
            GCHandle handle = GCHandle.Alloc(algoSpace, GCHandleType.Pinned);
            try
            {
                IntPtr ptr = handle.AddrOfPinnedObject();
                res = CudaDNNNativeMethods.cudnnRestoreAlgorithm(ctx.Handle, ptr, algoSpace.Length, _desc);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnRestoreAlgorithm", res));
            }
            finally
            {
                handle.Free();
            }
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }
    }
}
