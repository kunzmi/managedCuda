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
    /// CudaMemoryPool
    /// </summary>
    public class CudaMemoryPool : IDisposable
    {
        private CUmemoryPool _memoryPool;
        private CUResult res;
        private bool disposed;
        private bool _isOwner;

        #region Constructors
        /// <summary>
        /// Creates a new CudaMemoryPool. 
        /// </summary>
        /// <param name="props"></param>
        public CudaMemoryPool(CUmemPoolProps props)
        {
            _memoryPool = new CUmemoryPool();

            res = DriverAPINativeMethods.MemoryManagement.cuMemPoolCreate(ref _memoryPool, ref props);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemPoolCreate", res));
            if (res != CUResult.Success) throw new CudaException(res);
            _isOwner = true;
        }

        /// <summary>
        /// imports a memory pool from a shared handle.<para/>
        /// Specific allocations can be imported from the imported pool with cuMemPoolImportPointer.<para/>
        /// note Imported memory pools do not support creating new allocations. As such imported memory pools 
        /// may not be used in cuDeviceSetMemPool or ::cuMemAllocFromPoolAsync calls.
        /// </summary>
        /// <param name="handle">OS handle of the pool to open</param>
        /// <param name="handleType">The type of handle being imported</param>
        /// <param name="flags">must be 0</param>
        public CudaMemoryPool(IntPtr handle, CUmemAllocationHandleType handleType, ulong flags)
        {
            _memoryPool = new CUmemoryPool();

            res = DriverAPINativeMethods.MemoryManagement.cuMemPoolImportFromShareableHandle(ref _memoryPool, handle, handleType, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemPoolImportFromShareableHandle", res));
            if (res != CUResult.Success) throw new CudaException(res);
            _isOwner = true;
        }

        /// <summary>
        /// Gets the current or default memory pool of the CUdevice. 
        /// </summary>
        /// <param name="device">The device to the memory pool from</param>
        /// <param name="isDefault">Get the default or the current memory pool</param>
        public CudaMemoryPool(CUdevice device, bool isDefault)
        {
            _memoryPool = new CUmemoryPool();
            if (isDefault)
            {
                res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetDefaultMemPool(ref _memoryPool, device);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDeviceGetDefaultMemPool", res));
                if (res != CUResult.Success) throw new CudaException(res);
                _isOwner = false;
            }
            else
            {
                res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetMemPool(ref _memoryPool, device);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDeviceGetMemPool", res));
                if (res != CUResult.Success) throw new CudaException(res);
                _isOwner = true;
            }
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~CudaMemoryPool()
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
                if (_isOwner)
                {
                    res = DriverAPINativeMethods.MemoryManagement.cuMemPoolDestroy(_memoryPool);
                    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemPoolDestroy", res));
                }
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        #region Methods
        /// <summary>
        /// Tries to release memory back to the OS<para/>
        /// Releases memory back to the OS until the pool contains fewer than minBytesToKeep
        /// reserved bytes, or there is no more memory that the allocator can safely release.<para/>
        /// The allocator cannot release OS allocations that back outstanding asynchronous allocations.<para/>
        /// The OS allocations may happen at different granularity from the user allocations.<para/>
        /// <para/>
        /// note: Allocations that have not been freed count as outstanding.<para/>
        /// note: Allocations that have been asynchronously freed but whose completion has
        /// not been observed on the host (eg.by a synchronize) can count as outstanding.
        /// </summary>
        /// <param name="minBytesToKeep">If the pool has less than minBytesToKeep reserved,
        /// the TrimTo operation is a no-op.Otherwise the pool will be guaranteed to have at least minBytesToKeep bytes reserved after the operation.</param>
        public void TrimTo(SizeT minBytesToKeep)
        {
            res = DriverAPINativeMethods.MemoryManagement.cuMemPoolTrimTo(_memoryPool, minBytesToKeep);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemPoolTrimTo", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }


        /// <summary>
        /// Import a memory pool allocation from another process.<para/>
        /// Returns in \p ptr_out a pointer to the imported memory.<para/>
        /// The imported memory must not be accessed before the allocation operation completes
        /// in the exporting process.The imported memory must be freed from all importing processes before
        /// being freed in the exporting process.The pointer may be freed with cuMemFree
        /// or cuMemFreeAsync.If cuMemFreeAsync is used, the free must be completed
        /// on the importing process before the free operation on the exporting process.<para/>
        /// note The cuMemFreeAsync api may be used in the exporting process before
        /// the cuMemFreeAsync operation completes in its stream as long as the
        /// cuMemFreeAsync in the exporting process specifies a stream with
        /// a stream dependency on the importing process's cuMemFreeAsync.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="shareData"></param>
        /// <returns></returns>
        public CudaDeviceVariable<T> ImportPointer<T>(CUmemPoolPtrExportData shareData) where T : struct
        {
            CUdeviceptr devPtr = new CUdeviceptr();
            res = DriverAPINativeMethods.MemoryManagement.cuMemPoolImportPointer(ref devPtr, _memoryPool, ref shareData);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemPoolImportPointer", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
            return new CudaDeviceVariable<T>(devPtr, false);
        }


        /// <summary>
        /// Allocates memory from a specified pool with stream ordered semantics.<para/>
        /// Inserts an allocation operation into \p hStream.<para/>
        /// A pointer to the allocated memory is returned immediately in *dptr.<para/>
        /// The allocation must not be accessed until the the allocation operation completes.<para/>
        /// The allocation comes from the specified memory pool.<para/>
        /// note<para/>
        /// -  The specified memory pool may be from a device different than that of the specified \p hStream.<para/>
        /// -  Basic stream ordering allows future work submitted into the same stream to use the allocation.
        /// Stream query, stream synchronize, and CUDA events can be used to guarantee that the allocation
        /// operation completes before work submitted in a separate stream runs. 
        /// </summary>
        /// <param name="bytesize">Number of bytes to allocate</param>
        /// <param name="hStream">The stream establishing the stream ordering semantic</param>
        public CudaDeviceVariable<T> MemAllocFromPoolAsync<T>(SizeT bytesize, CudaStream hStream) where T : struct
        {
            CUdeviceptr devPtr = new CUdeviceptr();
            res = DriverAPINativeMethods.MemoryManagement.cuMemAllocFromPoolAsync(ref devPtr, bytesize, _memoryPool, hStream.Stream);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemAllocFromPoolAsync", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
            return new CudaDeviceVariable<T>(devPtr, false, bytesize);
        }


        /// <summary>
        /// Returns the accessibility of a pool from a device<para/>
        /// Returns the accessibility of the pool's memory from the specified location.
        /// </summary>
        /// <param name="location">the location accessing the pool</param>
        public CUmemAccess_flags GetAccess(CUmemLocation location)
        {
            CUmemAccess_flags flags = new CUmemAccess_flags();
            res = DriverAPINativeMethods.MemoryManagement.cuMemPoolGetAccess(ref flags, _memoryPool, ref location);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemPoolGetAccess", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
            return flags;
        }

        /// <summary>
        /// Controls visibility of pools between devices
        /// </summary>
        public void SetAccess(CUmemAccessDesc[] accessDescs)
        {
            res = DriverAPINativeMethods.MemoryManagement.cuMemPoolSetAccess(_memoryPool, accessDescs, accessDescs.Length);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemPoolSetAccess", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Exports a memory pool to the requested handle type.<para/>
        /// Given an IPC capable mempool, create an OS handle to share the pool with another process.<para/>
        /// A recipient process can convert the shareable handle into a mempool with::cuMemPoolImportFromShareableHandle.
        /// Individual pointers can then be shared with the ::cuMemPoolExportPointer and ::cuMemPoolImportPointer APIs.
        /// The implementation of what the shareable handle is and how it can be transferred is defined by the requested
        /// handle type.<para/>
        /// note: To create an IPC capable mempool, create a mempool with a CUmemAllocationHandleType other than CU_MEM_HANDLE_TYPE_NONE.
        /// </summary>
        /// <param name="handleType">the type of handle to create</param>
        /// <param name="flags">must be 0</param>
        public IntPtr ExportToShareableHandle(CUmemAllocationHandleType handleType, ulong flags)
        {
            IntPtr ret = new IntPtr();
            res = DriverAPINativeMethods.MemoryManagement.cuMemPoolExportToShareableHandle(ref ret, _memoryPool, handleType, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemPoolExportToShareableHandle", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
            return ret;
        }


        /// <summary>
        /// Sets attributes of a memory pool<para/>
        /// Supported attributes are:<para/>
        /// - ::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD: (value type = cuuint64_t)<para/>
        /// Amount of reserved memory in bytes to hold onto before trying to release memory back to the OS.When more than the release
        /// threshold bytes of memory are held by the memory pool, the allocator will try to release memory back to the OS on the next 
        /// call to stream, event or context synchronize. (default 0)<para/>
        /// - ::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES: (value type = int)<para/>
        /// Allow::cuMemAllocAsync to use memory asynchronously freed
        /// in another stream as long as a stream ordering dependency
        /// of the allocating stream on the free action exists.
        /// Cuda events and null stream interactions can create the required
        /// stream ordered dependencies. (default enabled)<para/>
        /// - ::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC: (value type = int)<para/>
        /// Allow reuse of already completed frees when there is no dependency
        /// between the free and allocation. (default enabled)<para/>
        /// - ::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES: (value type = int)<para/>
        /// Allow::cuMemAllocAsync to insert new stream dependencies
        /// in order to establish the stream ordering required to reuse
        /// a piece of memory released by::cuMemFreeAsync(default enabled).
        /// </summary>
        /// <param name="attr">The attribute to modify</param>
        /// <param name="value">Pointer to the value to assign</param>
        public void SetAttribute(CUmemPool_attribute attr, int value)
        {
            res = DriverAPINativeMethods.MemoryManagement.cuMemPoolSetAttribute(_memoryPool, attr, ref value);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemPoolSetAttribute", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        /// <summary>
        /// Sets attributes of a memory pool<para/>
        /// Supported attributes are:<para/>
        /// - ::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD: (value type = cuuint64_t)<para/>
        /// Amount of reserved memory in bytes to hold onto before trying to release memory back to the OS.When more than the release
        /// threshold bytes of memory are held by the memory pool, the allocator will try to release memory back to the OS on the next 
        /// call to stream, event or context synchronize. (default 0)<para/>
        /// - ::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES: (value type = int)<para/>
        /// Allow::cuMemAllocAsync to use memory asynchronously freed
        /// in another stream as long as a stream ordering dependency
        /// of the allocating stream on the free action exists.
        /// Cuda events and null stream interactions can create the required
        /// stream ordered dependencies. (default enabled)<para/>
        /// - ::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC: (value type = int)<para/>
        /// Allow reuse of already completed frees when there is no dependency
        /// between the free and allocation. (default enabled)<para/>
        /// - ::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES: (value type = int)<para/>
        /// Allow::cuMemAllocAsync to insert new stream dependencies
        /// in order to establish the stream ordering required to reuse
        /// a piece of memory released by::cuMemFreeAsync(default enabled).
        /// </summary>
        /// <param name="attr">The attribute to modify</param>
        /// <param name="value">Pointer to the value to assign</param>
        public void SetAttribute(CUmemPool_attribute attr, ulong value)
        {
            res = DriverAPINativeMethods.MemoryManagement.cuMemPoolSetAttribute(_memoryPool, attr, ref value);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemPoolSetAttribute", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }

        /// <summary>
        /// Sets attributes of a memory pool<para/>
        /// Supported attributes are:<para/>
        /// - ::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD: (value type = cuuint64_t)<para/>
        /// Amount of reserved memory in bytes to hold onto before trying to release memory back to the OS.When more than the release
        /// threshold bytes of memory are held by the memory pool, the allocator will try to release memory back to the OS on the next 
        /// call to stream, event or context synchronize. (default 0)<para/>
        /// - ::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES: (value type = int)<para/>
        /// Allow::cuMemAllocAsync to use memory asynchronously freed
        /// in another stream as long as a stream ordering dependency
        /// of the allocating stream on the free action exists.
        /// Cuda events and null stream interactions can create the required
        /// stream ordered dependencies. (default enabled)<para/>
        /// - ::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC: (value type = int)<para/>
        /// Allow reuse of already completed frees when there is no dependency
        /// between the free and allocation. (default enabled)<para/>
        /// - ::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES: (value type = int)<para/>
        /// Allow::cuMemAllocAsync to insert new stream dependencies
        /// in order to establish the stream ordering required to reuse
        /// a piece of memory released by::cuMemFreeAsync(default enabled).
        /// </summary>
        /// <param name="attr">The attribute to modify</param>
        /// <param name="value">Pointer to the value to assign</param>
        public void GetAttribute(CUmemPool_attribute attr, ref int value)
        {
            res = DriverAPINativeMethods.MemoryManagement.cuMemPoolGetAttribute(_memoryPool, attr, ref value);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemPoolGetAttribute", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        /// <summary>
        /// Sets attributes of a memory pool<para/>
        /// Supported attributes are:<para/>
        /// - ::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD: (value type = cuuint64_t)<para/>
        /// Amount of reserved memory in bytes to hold onto before trying to release memory back to the OS.When more than the release
        /// threshold bytes of memory are held by the memory pool, the allocator will try to release memory back to the OS on the next 
        /// call to stream, event or context synchronize. (default 0)<para/>
        /// - ::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES: (value type = int)<para/>
        /// Allow::cuMemAllocAsync to use memory asynchronously freed
        /// in another stream as long as a stream ordering dependency
        /// of the allocating stream on the free action exists.
        /// Cuda events and null stream interactions can create the required
        /// stream ordered dependencies. (default enabled)<para/>
        /// - ::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC: (value type = int)<para/>
        /// Allow reuse of already completed frees when there is no dependency
        /// between the free and allocation. (default enabled)<para/>
        /// - ::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES: (value type = int)<para/>
        /// Allow::cuMemAllocAsync to insert new stream dependencies
        /// in order to establish the stream ordering required to reuse
        /// a piece of memory released by::cuMemFreeAsync(default enabled).
        /// </summary>
        /// <param name="attr">The attribute to modify</param>
        /// <param name="value">Pointer to the value to assign</param>
        public void GetAttribute(CUmemPool_attribute attr, ref ulong value)
        {
            res = DriverAPINativeMethods.MemoryManagement.cuMemPoolGetAttribute(_memoryPool, attr, ref value);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMemPoolGetAttribute", res));
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
        #endregion

        #region Properties
        /// <summary>
        /// Returns the wrapped CUarray
        /// </summary>
        public CUmemoryPool MemoryPool
        {
            get { return _memoryPool; }
        }

        /// <summary>
        /// If the wrapper class instance is the owner of a CUDA handle, it will be destroyed while disposing.
        /// </summary>
        public bool IsOwner
        {
            get { return _isOwner; }
        }
        #endregion
    }
}
