//	Copyright (c) 2012, Michael Kunz. All rights reserved.
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
using System.Collections.Generic;
using System.Text;
using ManagedCuda.BasicTypes;
using System.Diagnostics;

namespace ManagedCuda
{
    /// <summary>
    /// Wrapps a CUevent handle.
    /// </summary>
    public class CudaEvent : IDisposable
    {
        bool disposed;
        CUResult res;
        CUevent _event;

        #region Constructor
        /// <summary>
        /// Creates a new Event using <see cref="CUEventFlags.Default"/> 
        /// </summary>
        public CudaEvent()
            : this (CUEventFlags.Default)
        {
        }

        /// <summary>
        /// Creates a new Event
        /// </summary>
        /// <param name="flags">Parameters for event creation</param>
        public CudaEvent(CUEventFlags flags)
        {
            _event = new CUevent();

            res = DriverAPINativeMethods.Events.cuEventCreate(ref _event, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuEventCreate", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~CudaEvent()
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
                res = DriverAPINativeMethods.Events.cuEventDestroy_v2(_event);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuEventDestroy", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        #region Properties
        /// <summary>
        /// returns the wrapped CUevent handle
        /// </summary>
        public CUevent Event
        {
            get { return _event; }
            set { _event = value; }
        }
        #endregion

        #region Methods
        /// <summary>
        /// Records an event. If <c>stream</c> is non-zero, the event is recorded after all preceding operations in the stream have been
        /// completed; otherwise, it is recorded after all preceding operations in the CUDA context have been completed. Since
        /// operation is asynchronous, <see cref="Query"/> and/or <see cref="Synchronize"/> must be used to determine when the event
        /// has actually been recorded. <para/>
        /// If <see cref="Record()"/> has previously been called and the event has not been recorded yet, this function throws
        /// <see cref="CUResult.ErrorInvalidValue"/>.
        /// </summary>
        public void Record()
        {
            CUstream _stream = new CUstream();
            res = DriverAPINativeMethods.Events.cuEventRecord(_event, _stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuEventRecord", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }


        /// <summary>
        /// Records an event. If <c>stream</c> is non-zero, the event is recorded after all preceding operations in the stream have been
        /// completed; otherwise, it is recorded after all preceding operations in the CUDA context have been completed. Since
        /// operation is asynchronous, <see cref="Query"/> and/or <see cref="Synchronize"/> must be used to determine when the event
        /// has actually been recorded. <para/>
        /// If <see cref="Record()"/> has previously been called and the event has not been recorded yet, this function throws
        /// <see cref="CUResult.ErrorInvalidValue"/>.
        /// <param name="stream"></param>
        /// </summary>
        public void Record(CUstream stream)
        {
            res = DriverAPINativeMethods.Events.cuEventRecord(_event, stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuEventRecord", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }


        /// <summary>
        /// Waits until the event has actually been recorded. If <see cref="Record()"/> has been called on this event, the function returns
        /// <see cref="CUResult.ErrorInvalidValue"/>. Waiting for an event that was created with the <see cref="CUEventFlags.BlockingSync"/>
        /// flag will cause the calling CPU thread to block until the event has actually been recorded. <para/>
        /// If <see cref="Record()"/> has previously been called and the event has not been recorded yet, this function throws <see cref="CUResult.ErrorInvalidValue"/>.
        /// </summary>
        public void Synchronize()
        {
            res = DriverAPINativeMethods.Events.cuEventSynchronize(_event);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuEventSynchronize", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Returns true if the event has actually been recorded, or false if not. If
        /// <see cref="Record()"/> has not been called on this event, the function throws <see cref="CUResult.ErrorInvalidValue"/>.
        /// </summary>
        /// <returns></returns>
        public bool Query()
        {
            res = DriverAPINativeMethods.Events.cuEventQuery(_event);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuEventQuery", res));
            if (res != CUResult.Success && res != CUResult.ErrorNotReady) throw new CudaException(res);

            if (res == CUResult.Success) return true;
            return false; // --> ErrorNotReady
        }
        #endregion

        #region static
        /// <summary>
        /// Computes the elapsed time between two events (in milliseconds with a resolution of around 0.5 microseconds). If
        /// either event has not been recorded yet, this function throws <see cref="CUResult.ErrorNotReady"/>. If either event has been
        /// recorded with a non-zero stream, the result is undefined.
        /// </summary>
        /// <param name="eventStart"></param>
        /// <param name="eventEnd"></param>
        /// <returns></returns>
        public static float ElapsedTime(CudaEvent eventStart, CudaEvent eventEnd)
        {
            float time = 0;
            CUResult res = DriverAPINativeMethods.Events.cuEventElapsedTime(ref time, eventStart.Event, eventEnd.Event);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuEventElapsedTime", res));
            if (res != CUResult.Success) throw new CudaException(res);
            return time;
        }

        #endregion
    }
}
