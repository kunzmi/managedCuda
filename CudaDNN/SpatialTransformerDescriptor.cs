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
    public class SpatialTransformerDescriptor : IDisposable
    {
        private cudnnSpatialTransformerDescriptor _desc;
        private cudnnStatus res;
        private bool disposed;
        private cudnnHandle _handle;

        #region Contructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="context"></param>
        public SpatialTransformerDescriptor(CudaDNNContext context)
        {
            _handle = context.Handle;
            _desc = new cudnnSpatialTransformerDescriptor();
            res = CudaDNNNativeMethods.cudnnCreateSpatialTransformerDescriptor(ref _desc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCreateSpatialTransformerDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~SpatialTransformerDescriptor()
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
                res = CudaDNNNativeMethods.cudnnDestroySpatialTransformerDescriptor(_desc);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDestroySpatialTransformerDescriptor", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        /// <summary>
        /// Returns the inner handle.
        /// </summary>
        public cudnnSpatialTransformerDescriptor Desc
        {
            get { return _desc; }
        }

        /// <summary>
        /// This function destroys a previously created spatial transformer descriptor object. 
        /// </summary>
        /// <param name="samplerType">Enumerant to specify the sampler type.</param>
        /// <param name="dataType">Data type.</param>
        /// <param name="nbDims">Dimension of the transformed tensor.</param>
        /// <param name="dimA">Array of dimension nbDims containing the size of the transformed tensor for every dimension.</param>
        public void SetSpatialTransformerNdDescriptor(
                                        cudnnSamplerType samplerType,
                                        cudnnDataType dataType,
                                        int nbDims,
                                        int[] dimA)
        {
            res = CudaDNNNativeMethods.cudnnSetSpatialTransformerNdDescriptor(_desc, samplerType, dataType, nbDims, dimA);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetSpatialTransformerNdDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }
    }
}
