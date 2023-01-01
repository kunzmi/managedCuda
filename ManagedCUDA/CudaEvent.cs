// Copyright (c) 2023, Michael Kunz and Artic Imaging SARL. All rights reserved.
// http://kunzmi.github.io/managedCuda
//
// This file is part of ManagedCuda.
//
// Commercial License Usage
//  Licensees holding valid commercial ManagedCuda licenses may use this
//  file in accordance with the commercial license agreement provided with
//  the Software or, alternatively, in accordance with the terms contained
//  in a written agreement between you and Artic Imaging SARL. For further
//  information contact us at managedcuda@articimaging.eu.
//  
// GNU General Public License Usage
//  Alternatively, this file may be used under the terms of the GNU General
//  Public License as published by the Free Software Foundation, either 
//  version 3 of the License, or (at your option) any later version.
//  
//  ManagedCuda is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with this program. If not, see <http://www.gnu.org/licenses/>.


using System;
using ManagedCuda.BasicTypes;
using System.Diagnostics;

namespace ManagedCuda
{
    /// <summary>
    /// Wrapps a CUevent handle.
    /// </summary>
    public class CudaEvent : IDisposable
    {
        private bool disposed;
        private CUResult res;
        private CUevent _event;

        #region Constructor
        /// <summary>
        /// Creates a new Event using <see cref="CUEventFlags.Default"/> 
        /// </summary>
        public CudaEvent()
            : this(CUEventFlags.Default)
        {
        }

        /// <summary>
        /// Creates a new Event using <see cref="CUEventFlags.Default"/> 
        /// </summary>
        internal CudaEvent(CUevent event_)
        {
            _event = event_;
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
        /// Records an event
        /// Captures in \p hEvent the contents of \p hStream at the time of this call.
        /// \p hEvent and \p hStream must be from the same context.
        /// Calls such as ::cuEventQuery() or ::cuStreamWaitEvent() will then
        /// examine or wait for completion of the work that was captured.Uses of
        /// \p hStream after this call do not modify \p hEvent. See note on default
        /// stream behavior for what is captured in the default case.
        /// ::cuEventRecordWithFlags() can be called multiple times on the same event and
        /// will overwrite the previously captured state.Other APIs such as
        /// ::cuStreamWaitEvent() use the most recently captured state at the time
        /// of the API call, and are not affected by later calls to
        /// ::cuEventRecordWithFlags(). Before the first call to::cuEventRecordWithFlags(), an
        /// event represents an empty set of work, so for example::cuEventQuery()
        /// would return ::CUDA_SUCCESS.
        /// </summary>
        public void Record(CUstream stream, CUEventRecordFlags flags)
        {
            res = DriverAPINativeMethods.Events.cuEventRecordWithFlags(_event, stream, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuEventRecordWithFlags", res));
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
