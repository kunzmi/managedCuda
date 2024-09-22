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


using ManagedCuda.BasicTypes;
using System;
using System.Diagnostics;

namespace ManagedCuda
{
    /// <summary>
    /// A green context handle. This handle can be used safely from only one CPU thread at a time.
    /// </summary>
    public class GreenContext : IDisposable
    {
        /// <summary>
        /// </summary>
        protected bool disposed;
        /// <summary>
        /// Indicates if this GreenContext instance created the wrapped green context and should be destroyed while disposing.
        /// </summary>
        protected bool _contextOwner;
        /// <summary>
        /// </summary>
        protected CUgreenCtx _ctx;
        /// <summary>
        /// </summary>
        protected CUdevice _device;

        #region Constructors
        /// <summary>
        /// Create a new instace of green context. The instance is not the owner of the provided ctx and it won't be destroyed when disposing.
        /// </summary>
        public GreenContext(CUgreenCtx ctx, CUdevice device)
        {
            _contextOwner = false;
            _ctx = ctx;
            _device = device;
        }

        /// <summary>
        /// Creates a green context with a specified set of resources.<para/>
        /// This API creates a green context with the resources specified in the descriptor \p desc and
        /// returns it in the handle represented by \p phCtx.This API will retain the primary context on device \p dev,
        /// which will is released when the green context is destroyed.It is advised to have the primary context active
        /// before calling this API to avoid the heavy cost of triggering primary context initialization and
        /// deinitialization multiple times.<para/>
        /// The API does not set the green context current. In order to set it current, you need to explicitly set it current
        /// by first converting the green context to a CUcontext using ::cuCtxFromGreenCtx and subsequently calling
        /// ::cuCtxSetCurrent / ::cuCtxPushCurrent.It should be noted that a green context can be current to only one
        /// thread at a time.There is no internal synchronization to make API calls accessing the same green context
        /// from multiple threads work.<para/>
        /// Note: The API is not supported on 32-bit platforms.
        /// </summary>
        /// <param name="desc">Descriptor generated via ::cuDevResourceGenerateDesc which contains the set of resources to be used</param>
        /// <param name="dev">Device on which to create the green context.</param>
        /// <param name="flags">One of the supported green context creation flags. \p CU_GREEN_CTX_DEFAULT_STREAM is required.</param>
        public GreenContext(CUdevResourceDesc desc, CUdevice dev, CUgreenCtxCreate_flags flags)
        {
            CUResult res;

            res = DriverAPINativeMethods.GreenContextAPI.cuGreenCtxCreate(ref _ctx, desc, dev, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGreenCtxCreate", res));
            if (res != CUResult.Success)
                throw new CudaException(res);

            _device = dev;
            _contextOwner = true;
        }

        /// <summary>
        /// Create a new instace of a cuda green context from the given CudaStream
        /// Note: doesn't throw an exception if the returned green context is NULL!
        /// </summary>
        /// <param name="stream">The stream to query</param>
        public GreenContext(CudaStream stream)
        {
            CUResult res;

            _contextOwner = false;
            _device = new CUdevice();
            CUcontext ctx = new CUcontext();//dummy

            res = DriverAPINativeMethods.Streams.cuStreamGetCtx(stream.Stream, ref ctx, ref _ctx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuStreamGetCtx", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }


        /// <summary>
        /// For dispose
        /// </summary>
        ~GreenContext()
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
        /// For IDisposable. <para/>
        /// Note: If this instance created the wrapped CUcontext, it will be destroyed and can't be accessed by other threads anymore. <para/>
        /// If this instance only was bound to an existing CUcontext, the wrapped CUcontext won't be destroyed.
        /// </summary>
        /// <param name="fDisposing"></param>
        protected virtual void Dispose(bool fDisposing)
        {
            if (fDisposing && !disposed)
            {
                if (_contextOwner)
                {
                    //Ignore if failing
                    CUResult res;
                    res = DriverAPINativeMethods.GreenContextAPI.cuGreenCtxDestroy(_ctx);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGreenCtxDestroy", res));
                }
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        /// <summary>
        /// Converts a green context into the primary context<para/>
        /// The API converts a green context into the primary context returned in \p pContext. It is important
        /// to note that the converted context \p pContext is a normal primary context but with
        /// the resources of the specified green context \p hCtx.Once converted, it can then
        /// be used to set the context current with::cuCtxSetCurrent or with any of the CUDA APIs
        /// that accept a CUcontext parameter.<para/>
        /// Users are expected to call this API before calling any CUDA APIs that accept a
        /// CUcontext. Failing to do so will result in the APIs returning::CUDA_ERROR_INVALID_CONTEXT.
        /// </summary>
        public PrimaryContext CtxFromGreenCtx()
        {
            CUcontext ctx = new CUcontext();

            CUResult res;
            res = DriverAPINativeMethods.GreenContextAPI.cuCtxFromGreenCtx(ref ctx, _ctx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuCtxFromGreenCtx", res));
            if (res != CUResult.Success)
                throw new CudaException(res);

            return new PrimaryContext(ctx, _device, _device.Pointer);

        }

        /// <summary>
        /// Get green context resources - Get the \p type resources available to the green context represented by \p hCtx
        /// </summary>
        /// <param name="type">Type of resource to retrieve</param>
        public CUdevResource GetDevResource(CUdevResourceType type)
        {
            CUdevResource value = new CUdevResource();
            CUResult res = DriverAPINativeMethods.GreenContextAPI.cuGreenCtxGetDevResource(_ctx, ref value, type);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGreenCtxGetDevResource", res));
            if (res != CUResult.Success) throw new CudaException(res);
            return value;
        }

        /// <summary>
        /// Records an event.
        /// <para/>
        /// Captures in \phEvent all the activities of the green context of \phCtx
        /// at the time of this call. \phEvent and \phCtx must be from the same
        /// CUDA context. Calls such as ::cuEventQuery() or ::cuGreenCtxWaitEvent() will
        /// then examine or wait for completion of the work that was captured. Uses of
        /// \p hCtx after this call do not modify \p hEvent.
        /// <para/>
        /// \note The API will return an error if the specified green context \p hCtx
        /// has a stream in the capture mode. In such a case, the call will invalidate
        /// all the conflicting captures.
        /// </summary>
        public void RecordEvent(CudaEvent cudaEvent)
        {
            CUResult res = DriverAPINativeMethods.GreenContextAPI.cuGreenCtxRecordEvent(_ctx, cudaEvent.Event);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGreenCtxRecordEvent", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Make a green context wait on an event
        /// <para/>
        /// Makes all future work submitted to green context \phCtx wait for all work
        /// captured in \phEvent. The synchronization will be performed on the device
        /// and will not block the calling CPU thread. See ::cuGreenCtxRecordEvent()
        /// for details on what is captured by an event.
        /// <para/>
        /// \note The API will return an error and invalidate the capture if the specified
        /// event \p hEvent is part of an ongoing capture sequence.
        /// </summary>
        public void WaitEvent(CudaEvent cudaEvent)
        {
            CUResult res = DriverAPINativeMethods.GreenContextAPI.cuGreenCtxWaitEvent(_ctx, cudaEvent.Event);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGreenCtxWaitEvent", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Create a stream for use in the green context
        /// 
        /// Creates a stream for use in the specified green context \p greenCtx and returns a handle in \p phStream.
        /// The stream can be destroyed by calling::cuStreamDestroy(). Note that the API ignores the context that
        /// is current to the calling thread and creates a stream in the specified green context \p greenCtx.
        /// 
        /// The supported values for \p flags are:
        /// - ::CU_STREAM_NON_BLOCKING: This must be specified. It indicates that work running in the created
        ///   stream may run concurrently with work in the default stream, and that
        ///   the created stream should perform no implicit synchronization with the default stream.
        /// 
        /// Specifying \p priority affects the scheduling priority of work in the stream. Priorities provide a
        /// hint to preferentially run work with higher priority when possible, but do not preempt
        /// already-running work or provide any other functional guarantee on execution order.
        /// \p priority follows a convention where lower numbers represent higher priorities.
        /// '0' represents default priority.The range of meaningful numerical priorities can
        /// be queried using ::cuCtxGetStreamPriorityRange. If the specified priority is
        /// outside the numerical range returned by::cuCtxGetStreamPriorityRange,
        /// it will automatically be clamped to the lowest or the highest number in the range.
        /// </summary>
        /// <param name="flags">Flags for stream creation. \p CU_STREAM_NON_BLOCKING must be specified.</param>
        /// <param name="priority">Stream priority. Lower numbers represent higher priorities. See::cuCtxGetStreamPriorityRange for more information about meaningful stream priorities that can be passed.</param>
        /// <returns></returns>
        public CudaStream CreateStream(CUStreamFlags flags, int priority)
        {
            CUstream stream = new CUstream();

            CUResult res = DriverAPINativeMethods.GreenContextAPI.cuGreenCtxStreamCreate(ref stream, _ctx, flags, priority);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGreenCtxStreamCreate", res));
            if (res != CUResult.Success) throw new CudaException(res);

            CudaStream cudaStream = new CudaStream(stream, true);
            return cudaStream;
        }
    }
}
