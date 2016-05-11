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
    public class OpTensorDescriptor : IDisposable
    {
        private cudnnOpTensorDescriptor _desc;
        private cudnnStatus res;
        private bool disposed;
        private cudnnHandle _handle;

        #region Contructors
        /// <summary>
        /// </summary>
        public OpTensorDescriptor(CudaDNNContext context)
        {
            _handle = context.Handle;
            _desc = new cudnnOpTensorDescriptor();
            res = CudaDNNNativeMethods.cudnnCreateOpTensorDescriptor(ref _desc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCreateOpTensorDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~OpTensorDescriptor()
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
                res = CudaDNNNativeMethods.cudnnDestroyOpTensorDescriptor(_desc);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDestroyOpTensorDescriptor", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        /// <summary>
        /// Returns the inner handle.
        /// </summary>
        public cudnnOpTensorDescriptor Desc
        {
            get { return _desc; }
        }

		/// <summary>
		/// 
		/// </summary>
		/// <param name="opTensorOp"></param>
		/// <param name="opTensorCompType"></param>
		/// <param name="opTensorNanOpt"></param>
        public void SetOpTensorDescriptor(
                                        cudnnOpTensorOp opTensorOp,
                                        cudnnDataType opTensorCompType,
                                        cudnnNanPropagation opTensorNanOpt)
        {
            res = CudaDNNNativeMethods.cudnnSetOpTensorDescriptor(_desc, opTensorOp, opTensorCompType, opTensorNanOpt);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetOpTensorDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

		/// <summary>
		/// 
		/// </summary>
		/// <param name="opTensorOp"></param>
		/// <param name="opTensorCompType"></param>
		/// <param name="opTensorNanOpt"></param>
        public void GetOpTensorDescriptor(
                                        ref cudnnOpTensorOp opTensorOp,
                                        ref cudnnDataType opTensorCompType,
                                        ref cudnnNanPropagation opTensorNanOpt)
        {
            res = CudaDNNNativeMethods.cudnnGetOpTensorDescriptor(_desc, ref opTensorOp, ref opTensorCompType, ref opTensorNanOpt);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetOpTensorDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }
    }
}
