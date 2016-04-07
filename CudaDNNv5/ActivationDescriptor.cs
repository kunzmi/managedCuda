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

namespace ManagedCuda.CudaDNNv5
{
    public class ActivationDescriptor : IDisposable
    {
        private cudnnActivationDescriptor _desc;
        private cudnnStatus res;
        private bool disposed;

        #region Contructors
        /// <summary>
        /// An opaque structure holding the description of an activation operation.
        /// </summary>
        public ActivationDescriptor()
        {
            _desc = new cudnnActivationDescriptor();
            res = CudaDNNNativeMethods.cudnnCreateActivationDescriptor(ref _desc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCreateTensorDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~ActivationDescriptor()
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
                res = CudaDNNNativeMethods.cudnnDestroyActivationDescriptor(_desc);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDestroyTensorDescriptor", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        /// <summary>
        /// Returns the inner handle.
        /// </summary>
        public cudnnActivationDescriptor Desc
        {
            get { return _desc; }
        }


        ///<summary>
        /// This function initializes then previously created activation descriptor object.
        /// </summary>
        /// <param name="activationDesc">Handle to the previously created activation descriptor object.</param>
        /// <param name="mode">Enumerant to specify the activation mode.</param>
        /// <param name="reluNanOpt">Nan propagation option for the relu.</param>
        /// <param name="reluCeiling">The ceiling for the clipped relu.</param>
		public void SetActivationDescriptor(cudnnActivationMode mode, 
                                    cudnnNanPropagation reluNanOpt, 
                                    double reluCeiling)
        {
            res = CudaDNNNativeMethods.cudnnSetActivationDescriptor(_desc, mode, reluNanOpt, reluCeiling);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetActivationDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// This function queries the parameters of the previouly initialized activation descriptor object.
        /// </summary>
        /// <param name="activationDesc">Handle to the previously created activation descriptor object.</param>
        /// <param name="mode">Enumerant to specify the activation mode.</param>
        /// <param name="reluNanOpt">Nan propagation option for the relu.</param>
        /// <param name="reluCeiling">The ceiling for the clipped relu.</param>
        public void GetActivationDescriptor( ref cudnnActivationMode mode,
                                ref cudnnNanPropagation reluNanOpt,
                                ref double reluCeiling)
        {
            res = CudaDNNNativeMethods.cudnnGetActivationDescriptor(_desc, ref mode, ref reluNanOpt, ref reluCeiling);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnGetActivationDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }
    }
}
