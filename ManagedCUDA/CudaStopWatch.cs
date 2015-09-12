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
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace ManagedCuda
{
    /// <summary>
    /// Measures via CUDA events the timespan between Start() and Stop() calls.
    /// </summary>
    public class CudaStopWatch : IDisposable
    {
        CudaEvent _start, _stop;
        CUstream _stream;
        bool disposed;

        #region Constructors
        /// <summary>
        /// 
        /// </summary>
        public CudaStopWatch()
        {
            _start = new CudaEvent();
			_stop = new CudaEvent();
            _stream = new CUstream();

        }

        /// <summary>
        /// 
        /// </summary>
        public CudaStopWatch(CUEventFlags flags)
        {
			_start = new CudaEvent(flags);
			_stop = new CudaEvent(flags);
            _stream = new CUstream();

		}
		/// <summary>
		/// 
		/// </summary>
		public CudaStopWatch(CUstream stream)
		{
			_start = new CudaEvent();
			_stop = new CudaEvent();
			_stream = stream;

		}

		/// <summary>
		/// 
		/// </summary>
		public CudaStopWatch(CUEventFlags flags, CUstream stream)
		{
			_start = new CudaEvent(flags);
			_stop = new CudaEvent(flags);
			_stream = stream;
		}       
        
        /// <summary>
        /// For dispose
        /// </summary>
        ~CudaStopWatch()
        {
            Dispose (false);            
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
        protected virtual void Dispose (bool fDisposing)
        {
           if (fDisposing && !disposed)
		   {
			   _start.Dispose();
			   _stop.Dispose();
               disposed = true;
           }
           if (!fDisposing && !disposed)
               Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        #region Methods
        /// <summary>
        /// Start measurement
        /// </summary>
        public void Start()
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());
			_start.Record(_stream);
        }

        /// <summary>
        /// Stop measurement
        /// </summary>
        public void Stop()
        {
			if (disposed) throw new ObjectDisposedException(this.ToString());
			_stop.Record(_stream);
        }

        /// <summary>
        /// Get elapsed time in milliseconds, sync on stop event
        /// </summary>
        /// <returns>Elapsed time in ms</returns>
        public float GetElapsedTime()
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());
			_stop.Synchronize();
			return CudaEvent.ElapsedTime(_start, _stop);
        }

		/// <summary>
		/// Get elapsed time in milliseconds, no sync on stop event
		/// </summary>
		/// <returns>Elapsed time in ms</returns>
		public float GetElapsedTimeNoSync()
		{
			if (disposed) throw new ObjectDisposedException(this.ToString());
			return CudaEvent.ElapsedTime(_start, _stop);
		}
        #endregion

		#region Properties
		/// <summary>
		/// Returns the inner start event
		/// </summary>
		public CudaEvent StartEvent
		{
			get { return _start; }
		}

		/// <summary>
		/// Returns the inner stop event
		/// </summary>
		public CudaEvent StopEvent
		{
			get { return _stop; }
		}

		/// <summary>
		/// Returns the inner stream
		/// </summary>
		public CUstream Stream
		{
			get { return _stream; }
		}
		#endregion
	}
}
