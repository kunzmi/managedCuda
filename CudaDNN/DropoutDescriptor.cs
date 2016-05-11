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
    public class DropoutDescriptor : IDisposable
    {
        private cudnnDropoutDescriptor _desc;
        private cudnnStatus res;
        private bool disposed;
        private cudnnHandle _handle;

        #region Contructors
        /// <summary>
        /// </summary>
        public DropoutDescriptor(CudaDNNContext context)
        {
            _handle = context.Handle;
            _desc = new cudnnDropoutDescriptor();
            res = CudaDNNNativeMethods.cudnnCreateDropoutDescriptor(ref _desc);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnCreateDropoutDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~DropoutDescriptor()
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
                res = CudaDNNNativeMethods.cudnnDestroyDropoutDescriptor(_desc);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnDestroyDropoutDescriptor", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        /// <summary>
        /// Returns the inner handle.
        /// </summary>
        public cudnnDropoutDescriptor Desc
        {
            get { return _desc; }
        }

        /// <summary>
        /// This function initializes a previously created dropout descriptor object. If states argument is equal to 
        /// NULL, random number generator states won't be initialized, and only dropout value will be set. No other 
        /// function should be writing to the memory
        /// </summary>
        /// <param name="dropout">The probability with which the value from input would be propagated through the dropout layer.</param>
        /// <param name="states">Pointer to user-allocated GPU memory that will hold random number generator states.</param>
        /// <param name="stateSizeInBytes">Specifies size in bytes of the provided memory for the states.</param>
        /// <param name="seed">Seed used to initialize random number generator states.</param>
        public void SetDropoutDescriptor(
                                                            float dropout,
                                                            CudaDeviceVariable<byte> states,
                                                            SizeT stateSizeInBytes,
                                                            ulong seed)
        {
            res = CudaDNNNativeMethods.cudnnSetDropoutDescriptor(_desc, _handle, dropout, states.DevicePointer, stateSizeInBytes, seed);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cudnnSetDropoutDescriptor", res));
            if (res != cudnnStatus.Success) throw new CudaDNNException(res);
        }
    }
}
