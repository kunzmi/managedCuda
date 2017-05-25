//	Copyright (c) 2017, Michael Kunz. All rights reserved.
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
    /// ReduceTensorDescriptor is a pointer to an opaque structure
    /// holding the description of a tensor reduction operation, used as a parameter to
    /// cudnnReduceTensor(). cudnnCreateReduceTensorDescriptor() is used to create
    /// one instance, and cudnnSetReduceTensorDescriptor() must be used to initialize this instance.
    /// </summary>
    public class ReduceTensorDescriptor : IDisposable
    {
        private cudnnReduceTensorDescriptor _desc;
        private cudnnStatus res;
        private bool disposed;

        #region Contructors
        /// <summary>
        /// </summary>
        public ReduceTensorDescriptor()
        {
            _desc = new cudnnReduceTensorDescriptor();
            res = CudaDNNNativeMethods.cudnnCreateReduceTensorDescriptor(ref _desc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCreateReduceTensorDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~ReduceTensorDescriptor()
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
                res = CudaDNNNativeMethods.cudnnDestroyReduceTensorDescriptor(_desc);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDestroyReduceTensorDescriptor", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        /// <summary>
        /// Returns the inner handle.
        /// </summary>
        public cudnnReduceTensorDescriptor Desc
        {
            get { return _desc; }
        }

        

        /// <summary>
        /// 
        /// </summary>
        /// <param name="reduceTensorOp"></param>
        /// <param name="reduceTensorCompType"></param>
        /// <param name="reduceTensorNanOpt"></param>
        /// <param name="reduceTensorIndices"></param>
        /// <param name="reduceTensorIndicesType"></param>
        public void SetReduceTensorDescriptor(
                                        cudnnReduceTensorOp reduceTensorOp,
                                        cudnnDataType reduceTensorCompType,
                                        cudnnNanPropagation reduceTensorNanOpt,
                                        cudnnReduceTensorIndices reduceTensorIndices,
                                        cudnnIndicesType reduceTensorIndicesType)
        {
            res = CudaDNNNativeMethods.cudnnSetReduceTensorDescriptor(_desc, reduceTensorOp, reduceTensorCompType,
                reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetReduceTensorDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="reduceTensorOp"></param>
        /// <param name="reduceTensorCompType"></param>
        /// <param name="reduceTensorNanOpt"></param>
        /// <param name="reduceTensorIndices"></param>
        /// <param name="reduceTensorIndicesType"></param>
        public void GetReduceTensorDescriptor(
                                        ref cudnnReduceTensorOp reduceTensorOp,
                                        ref cudnnDataType reduceTensorCompType,
                                        ref cudnnNanPropagation reduceTensorNanOpt,
                                        ref cudnnReduceTensorIndices reduceTensorIndices,
                                        ref cudnnIndicesType reduceTensorIndicesType)
        {
            res = CudaDNNNativeMethods.cudnnGetReduceTensorDescriptor(_desc, ref reduceTensorOp, ref reduceTensorCompType,
                ref reduceTensorNanOpt, ref reduceTensorIndices, ref reduceTensorIndicesType);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetReduceTensorDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

    }
}
