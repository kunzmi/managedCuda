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
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace ManagedCuda.BasicTypes
{
    #region Struct as Types
    /// <summary>
    /// CUDA array
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUarray
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }
    /// <summary>
    /// CUDA linker
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUlinkState
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }
    /// <summary>
    /// CUDA mipmapped array
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUmipmappedArray
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    /// <summary>
    /// Cuda context
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUcontext
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    /// <summary>
    /// Cuda device
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUdevice
    {
        /// <summary>
        /// 
        /// </summary>
        public int Pointer;

        /// <summary>
        /// Device that represents the CPU
        /// </summary>
        static CUdevice CPU
        {
            get
            {
                CUdevice cpu = new CUdevice();
                cpu.Pointer = -1;
                return cpu;
            }
        }

        /// <summary>
        /// Device that represents an invalid device
        /// </summary>
        static CUdevice Invalid
        {
            get
            {
                CUdevice invalid = new CUdevice();
                invalid.Pointer = -2;
                return invalid;
            }
        }

        /// <summary>
        /// Sets the current memory pool of a device<para/>
        /// The memory pool must be local to the specified device.
        /// ::cuMemAllocAsync allocates from the current mempool of the provided stream's device.
        /// By default, a device's current memory pool is its default memory pool.
        /// <para/>
        /// note Use ::cuMemAllocFromPoolAsync to specify asynchronous allocations from a device different than the one the stream runs on.
        /// </summary>
        public void SetMemoryPool(CudaMemoryPool memPool)
        {
            CUResult res = DriverAPINativeMethods.DeviceManagement.cuDeviceSetMemPool(this, memPool.MemoryPool);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDeviceSetMemPool", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Gets the current memory pool of the CUdevice. 
        /// </summary>
        public CudaMemoryPool GetMemoryPool()
        {
            return new CudaMemoryPool(this, false);
        }

        /// <summary>
        /// Gets the default memory pool of the CUdevice. 
        /// </summary>
        public CudaMemoryPool GetDefaultMemoryPool()
        {
            return new CudaMemoryPool(this, true);
        }

        /// <summary>
        /// Return an UUID for the device (11.4+)<para/>
        /// Returns 16-octets identifing the device \p dev in the structure
        /// pointed by the \p uuid.If the device is in MIG mode, returns its
        /// MIG UUID which uniquely identifies the subscribed MIG compute instance.
        /// Returns 16-octets identifing the device \p dev in the structure pointed by the \p uuid.
        /// </summary>
        public CUuuid Uuid
        {
            get
            {
                CUuuid uuid = new CUuuid();
                CUResult res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetUuid_v2(ref uuid, this);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDeviceGetUuid_v2", res));
                if (res != CUResult.Success) throw new CudaException(res);
                return uuid;
            }
        }

        /// <summary>
        /// Returns information about the execution affinity support of the device.<para/>
        /// Returns in \p *pi whether execution affinity type \p type is supported by device \p dev.<para/>
        /// The supported types are:<para/>
        /// - ::CU_EXEC_AFFINITY_TYPE_SM_COUNT: 1 if context with limited SMs is supported by the device,
        /// or 0 if not;
        /// </summary>
        public bool GetExecAffinitySupport(CUexecAffinityType type)
        {
            int pi = 0;
            CUResult res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetExecAffinitySupport(ref pi, type, this);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDeviceGetExecAffinitySupport", res));
            if (res != CUResult.Success) throw new CudaException(res);
            return pi > 0;
        }

        /// <summary>
        /// Free unused memory that was cached on the specified device for use with graphs back to the OS.<para/>
        /// Blocks which are not in use by a graph that is either currently executing or scheduled to execute are freed back to the operating system.
        /// </summary>
        public void GraphMemTrim()
        {
            CUResult res = DriverAPINativeMethods.GraphManagment.cuDeviceGraphMemTrim(this);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDeviceGraphMemTrim", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Set asynchronous allocation attributes related to graphs<para/>
        /// Valid attributes are:<para/>
        /// - ::CU_GRAPH_MEM_ATTR_USED_MEM_HIGH: High watermark of memory, in bytes, associated with graphs since the last time it was reset.High watermark can only be reset to zero.<para/>
        /// - ::CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH: High watermark of memory, in bytes, currently allocated for use by the CUDA graphs asynchronous allocator.
        /// </summary>
        public void SetGraphMemAttribute(CUgraphMem_attribute attr, ulong value)
        {

            CUResult res = DriverAPINativeMethods.GraphManagment.cuDeviceSetGraphMemAttribute(this, attr, ref value);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDeviceSetGraphMemAttribute", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Query asynchronous allocation attributes related to graphs<para/>
        /// Valid attributes are:<para/>
        /// - ::CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT: Amount of memory, in bytes, currently associated with graphs<para/>
        /// - ::CU_GRAPH_MEM_ATTR_USED_MEM_HIGH: High watermark of memory, in bytes, associated with graphs since the last time it was reset.High watermark can only be reset to zero.<para/>
        /// - ::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT: Amount of memory, in bytes, currently allocated for use by the CUDA graphs asynchronous allocator.<para/>
        /// - ::CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH: High watermark of memory, in bytes, currently allocated for use by the CUDA graphs asynchronous allocator.
        /// </summary>
        public ulong GetGraphMemAttribute(CUgraphMem_attribute attr)
        {
            ulong value = 0;
            CUResult res = DriverAPINativeMethods.GraphManagment.cuDeviceGetGraphMemAttribute(this, attr, ref value);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDeviceGetGraphMemAttribute", res));
            if (res != CUResult.Success) throw new CudaException(res);
            return value;
        }


        #region operators
        /// <summary>
        /// 
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static bool operator ==(CUdevice src, CUdevice value)
        {
            return src.Pointer == value.Pointer;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static bool operator !=(CUdevice src, CUdevice value)
        {
            return src.Pointer != value.Pointer;
        }
        #endregion

        #region Override Methods
        /// <summary>
        /// Returns true if both objects are of type CUdevice and if both Pointer member are equal.
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            if (!(obj is CUdevice)) return false;

            CUdevice value = (CUdevice)obj;

            return this.Pointer.Equals(value.Pointer);
        }

        /// <summary>
        /// Overrides object.GetHashCode()
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            return Pointer.GetHashCode();
        }

        /// <summary>
        /// override ToString()
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return Pointer.ToString();
        }
        #endregion
    }

    /// <summary>
    /// Pointer to CUDA device memory
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUdeviceptr
    {
        /// <summary>
        /// 
        /// </summary>
        public SizeT Pointer;

        #region operators
        /// <summary>
        /// 
        /// </summary>
        /// <param name="src"></param>
        /// <returns></returns>
        public static implicit operator ulong(CUdeviceptr src)
        {
            return src.Pointer;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="src"></param>
        /// <returns></returns>
        public static explicit operator CUdeviceptr(SizeT src)
        {
            CUdeviceptr udeviceptr = new CUdeviceptr();
            udeviceptr.Pointer = src;
            return udeviceptr;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static CUdeviceptr operator +(CUdeviceptr src, SizeT value)
        {
            CUdeviceptr udeviceptr = new CUdeviceptr();
            udeviceptr.Pointer = src.Pointer + value;
            return udeviceptr;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static CUdeviceptr operator -(CUdeviceptr src, SizeT value)
        {
            CUdeviceptr udeviceptr = new CUdeviceptr();
            udeviceptr.Pointer = src.Pointer - value;
            return udeviceptr;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static bool operator ==(CUdeviceptr src, CUdeviceptr value)
        {
            return src.Pointer == value.Pointer;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static bool operator !=(CUdeviceptr src, CUdeviceptr value)
        {
            return src.Pointer != value.Pointer;
        }
        #endregion

        #region Override Methods
        /// <summary>
        /// Returns true if both objects are of type CUdeviceptr and if both Pointer member is equal.
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            if (!(obj is CUdeviceptr)) return false;

            CUdeviceptr value = (CUdeviceptr)obj;

            return this.Pointer.Equals(value.Pointer);
        }

        /// <summary>
        /// Overrides object.GetHashCode()
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        /// <summary>
        /// override ToString()
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return Pointer.ToString();
        }

        #endregion

        #region constructors
        /// <summary>
        /// 
        /// </summary>
        /// <param name="pointer"></param>
        public CUdeviceptr(SizeT pointer)
        {
            Pointer = pointer;
        }
        #endregion

        #region GetAttributeMethods
        /// <summary>
        /// The <see cref="CUcontext"/> on which a pointer was allocated or registered
        /// </summary>
        public CUcontext AttributeContext
        {
            get
            {
                CUcontext ret = new CUcontext();
                CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.Context, this);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
                if (res != CUResult.Success) throw new CudaException(res);
                return ret;
            }
        }

        /// <summary>
        /// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
        /// </summary>
        public CUMemoryType AttributeMemoryType
        {
            get
            {
                CUMemoryType ret = new CUMemoryType();
                CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.MemoryType, this);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
                if (res != CUResult.Success) throw new CudaException(res);
                return ret;
            }
        }

        /// <summary>
        /// The address at which a pointer's memory may be accessed on the device <para/>
        /// Except in the exceptional disjoint addressing cases, the value returned will equal the input value.
        /// </summary>
        public CUdeviceptr AttributeDevicePointer
        {
            get
            {
                CUdeviceptr ret = new CUdeviceptr();
                CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.DevicePointer, this);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
                if (res != CUResult.Success) throw new CudaException(res);
                return ret;
            }
        }

        /// <summary>
        /// The address at which a pointer's memory may be accessed on the host 
        /// </summary>
        public IntPtr AttributeHostPointer
        {
            get
            {
                IntPtr ret = new IntPtr();
                CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.HostPointer, this);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
                if (res != CUResult.Success) throw new CudaException(res);
                return ret;
            }
        }

        /// <summary>
        /// A pair of tokens for use with the nv-p2p.h Linux kernel interface
        /// </summary>
        public CudaPointerAttributeP2PTokens AttributeP2PTokens
        {
            get
            {
                CudaPointerAttributeP2PTokens ret = new CudaPointerAttributeP2PTokens();
                CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.P2PTokens, this);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
                if (res != CUResult.Success) throw new CudaException(res);
                return ret;
            }
        }

        /// <summary>
        /// Synchronize every synchronous memory operation initiated on this region
        /// </summary>
        public bool AttributeSyncMemops
        {
            get
            {
                int ret = 0;
                CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.SyncMemops, this);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
                if (res != CUResult.Success) throw new CudaException(res);
                return ret != 0;
            }
            set
            {
                int val = value ? 1 : 0;
                CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerSetAttribute(ref val, CUPointerAttribute.SyncMemops, this);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerSetAttribute", res));
                if (res != CUResult.Success) throw new CudaException(res);
            }
        }

        /// <summary>
        /// A process-wide unique ID for an allocated memory region
        /// </summary>
        public ulong AttributeBufferID
        {
            get
            {
                ulong ret = 0;
                CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.BufferID, this);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
                if (res != CUResult.Success) throw new CudaException(res);
                return ret;
            }
        }

        /// <summary>
        /// Indicates if the pointer points to managed memory
        /// </summary>
        public bool AttributeIsManaged
        {
            get
            {
                int ret = 0;
                CUResult res = DriverAPINativeMethods.MemoryManagement.cuPointerGetAttribute(ref ret, CUPointerAttribute.IsManaged, this);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuPointerGetAttribute", res));
                if (res != CUResult.Success) throw new CudaException(res);
                return ret != 0;
            }
        }

        #endregion
    }


    /// <summary>
    /// Cuda event
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUevent
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    /// <summary>
    /// Cuda function / kernel
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUfunction
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;

        /// <summary>
        /// Returns a module handle<para/>
        /// Returns in \p *hmod the handle of the module that function \p hfunc
        /// is located in. The lifetime of the module corresponds to the lifetime of
        /// the context it was loaded in or until the module is explicitly unloaded.<para/>
        /// The CUDA runtime manages its own modules loaded into the primary context.
        /// If the handle returned by this API refers to a module loaded by the CUDA runtime,
        /// calling ::cuModuleUnload() on that module will result in undefined behavior.
        /// </summary>
        public CUmodule GetModule()
        {
            CUmodule temp = new CUmodule();
            CUResult res = DriverAPINativeMethods.FunctionManagement.cuFuncGetModule(ref temp, this);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuFuncGetModule", res));
            if (res != CUResult.Success) throw new CudaException(res);
            return temp;
        }
    }

    /// <summary>
    /// Cuda module
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUmodule
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    /// <summary>
    /// Cuda stream
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUstream
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;

        /// <summary>
        /// Returns the CUDA NULL stream (0)
        /// </summary>
        public static CUstream NullStream
        {
            get
            {
                CUstream s = new CUstream();
                s.Pointer = (IntPtr)0;
                return s;
            }
        }

        /// <summary>
        /// Stream handle that can be passed as a CUstream to use an implicit stream
        /// with legacy synchronization behavior.
        /// </summary>
        public static CUstream LegacyStream
        {
            get
            {
                CUstream s = new CUstream();
                s.Pointer = (IntPtr)1;
                return s;
            }
        }

        /// <summary>
        /// Stream handle that can be passed as a CUstream to use an implicit stream
        /// with per-thread synchronization behavior.
        /// </summary>
        public static CUstream StreamPerThread
        {
            get
            {
                CUstream s = new CUstream();
                s.Pointer = (IntPtr)2;
                return s;
            }
        }
    }

    /// <summary>
    /// CUDA texture reference
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUtexref
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    /// <summary>
    /// CUDA surface reference
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUsurfref
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    /// <summary>
    /// CUDA graphics interop resource (DirectX / OpenGL)
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUgraphicsResource
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    /// <summary>
    /// CUDA texture object
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUtexObject
    {
        /// <summary>
        /// 
        /// </summary>
        public ulong Pointer;
    }

    /// <summary>
    /// CUDA surface object
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUsurfObject
    {
        /// <summary>
        /// 
        /// </summary>
        public ulong Pointer;
    }

    /// <summary>
    /// CUDA definition of UUID
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUuuid
    {
        /// <summary>
        /// 
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16, ArraySubType = UnmanagedType.I1)]
        public byte[] bytes;
    }

    /// <summary>
    /// 8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Luid
    {
        /// <summary>
        /// 
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8, ArraySubType = UnmanagedType.I1)]
        public byte[] bytes;
    }

    /// <summary>
    /// Interprocess Handle for Events
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUipcEventHandle
    {
        /// <summary>
        /// 
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 64, ArraySubType = UnmanagedType.I1)]
        public byte[] reserved;
    }

    /// <summary>
    /// Interprocess Handle for Memory
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUipcMemHandle
    {
        /// <summary>
        /// 
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 64, ArraySubType = UnmanagedType.I1)]
        public byte[] reserved;
    }

    /// <summary>
    /// half precission floating point
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct half
    {
        ushort x;

        /// <summary>
        /// 
        /// </summary>
        public half(float f)
        {
            x = __float2half(f).x;
        }

        /// <summary>
        /// 
        /// </summary>
        public half(double d)
        {
            x = __double2half(d).x;
        }

        /// <summary>
        /// 
        /// </summary>
        public half(half h16)
        {
            x = h16.x;
        }

        private static ushort __internal_float2half(float f, ref uint sign, ref uint remainder)
        {
            float[] ftemp = new float[] { f };
            uint[] x = new uint[1];
            uint u = 0;
            uint result = 0;
            Buffer.BlockCopy(ftemp, 0, x, 0, sizeof(float));

            u = (x[0] & 0x7fffffffU);
            sign = ((x[0] >> 16) & 0x8000U);
            // NaN/+Inf/-Inf
            if (u >= 0x7f800000U)
            {
                remainder = 0U;
                result = ((u == 0x7f800000U) ? (sign | 0x7c00U) : 0x7fffU);
            }
            else if (u > 0x477fefffU)
            { // Overflows
                remainder = 0x80000000U;
                result = (sign | 0x7bffU);
            }
            else if (u >= 0x38800000U)
            { // Normal numbers
                remainder = u << 19;
                u -= 0x38000000U;
                result = (sign | (u >> 13));
            }
            else if (u < 0x33000001U)
            { // +0/-0
                remainder = u;
                result = sign;
            }
            else
            { // Denormal numbers
                uint exponent = u >> 23;
                uint shift = 0x7eU - exponent;
                uint mantissa = (u & 0x7fffffU);
                mantissa |= 0x800000U;
                remainder = mantissa << (32 - (int)shift);
                result = (sign | (mantissa >> (int)shift));
            }
            return (ushort)(result);
        }

        private static half __double2half(double x)
        {
            // Perform rounding to 11 bits of precision, convert value
            // to float and call existing float to half conversion.
            // By pre-rounding to 11 bits we avoid additional rounding
            // in float to half conversion.
            ulong absx;
            ulong[] ux = new ulong[1];
            double[] xa = new double[] { x };
            Buffer.BlockCopy(xa, 0, ux, 0, sizeof(double));

            absx = (ux[0] & 0x7fffffffffffffffUL);
            if ((absx >= 0x40f0000000000000UL) || (absx <= 0x3e60000000000000UL))
            {
                // |x| >= 2^16 or NaN or |x| <= 2^(-25)
                // double-rounding is not a problem
                return __float2half((float)x);
            }

            // here 2^(-25) < |x| < 2^16
            // prepare shifter value such that x + shifter
            // done in double precision performs round-to-nearest-even
            // and (x + shifter) - shifter results in x rounded to
            // 11 bits of precision. Shifter needs to have exponent of
            // x plus 53 - 11 = 42 and a leading bit in mantissa to guard
            // against negative values.
            // So need to have |x| capped to avoid overflow in exponent.
            // For inputs that are smaller than half precision minnorm
            // we prepare fixed shifter exponent.
            ulong shifterBits = ux[0] & 0x7ff0000000000000UL;
            if (absx >= 0x3f10000000000000UL)
            {   // |x| >= 2^(-14)
                // add 42 to exponent bits
                shifterBits += 42ul << 52;
            }

            else
            {   // 2^(-25) < |x| < 2^(-14), potentially results in denormal
                // set exponent bits to 42 - 14 + bias
                shifterBits = ((42ul - 14 + 1023) << 52);
            }
            // set leading mantissa bit to protect against negative inputs
            shifterBits |= 1ul << 51;
            ulong[] shifterBitsArr = new ulong[] { shifterBits };
            double[] shifter = new double[1];

            Buffer.BlockCopy(shifterBitsArr, 0, shifter, 0, sizeof(double));

            double xShiftRound = x + shifter[0];
            double[] xShiftRoundArr = new double[] { xShiftRound };

            // Prevent the compiler from optimizing away x + shifter - shifter
            // by doing intermediate memcopy and harmless bitwize operation
            ulong[] xShiftRoundBits = new ulong[1];

            Buffer.BlockCopy(xShiftRoundArr, 0, xShiftRoundBits, 0, sizeof(double));

            // the value is positive, so this operation doesn't change anything
            xShiftRoundBits[0] &= 0x7ffffffffffffffful;

            Buffer.BlockCopy(xShiftRoundBits, 0, xShiftRoundArr, 0, sizeof(double));

            double xRounded = xShiftRound - shifter[0];
            float xRndFlt = (float)xRounded;
            half res = __float2half(xRndFlt);
            return res;
        }

        private static half __float2half(float a)
        {
            half r = new half();
            uint sign = 0;
            uint remainder = 0;
            r.x = __internal_float2half(a, ref sign, ref remainder);
            if ((remainder > 0x80000000U) || ((remainder == 0x80000000U) && ((r.x & 0x1U) != 0U)))
            {
                r.x++;
            }

            return r;
        }

        /// <summary>
        /// 
        /// </summary>
        public override string ToString()
        {
            return x.ToString();
        }
    }

    /// <summary>
    /// two half precission floating point (x,y)
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct half2
    {
        uint x;
    }

    /// <summary>
    /// bfloat16 floating point
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct bfloat16
    {
        ushort x;

        /// <summary>
        /// 
        /// </summary>
        public bfloat16(float f)
        {
            x = __float2bfloat16(f).x;
        }

        /// <summary>
        /// 
        /// </summary>
        public bfloat16(double d)
        {
            x = __double2bfloat16(d).x;
        }

        /// <summary>
        /// 
        /// </summary>
        public bfloat16(bfloat16 bf16)
        {
            x = bf16.x;
        }

        private static ushort __internal_float2bfloat16(float f, ref uint sign, ref uint remainder)
        {
            uint[] x = new uint[1];
            float[] ftemp = new float[] { f };
            Buffer.BlockCopy(ftemp, 0, x, 0, sizeof(float));

            if ((x[0] & 0x7fffffffU) > 0x7f800000U)
            {
                sign = 0U;
                remainder = 0U;
                return 0x7fff;
            }
            sign = x[0] >> 31;
            remainder = x[0] << 16;
            return (ushort)(x[0] >> 16);
        }

        private static bfloat16 __double2bfloat16(double x)
        {
            float[] f = new float[] { (float)x };
            double d = (double)f[0];
            uint[] u = new uint[1];

            Buffer.BlockCopy(f, 0, u, 0, sizeof(float));

            if ((x > 0.0) && (d > x))
            {
                u[0]--;
            }
            if ((x < 0.0) && (d < x))
            {
                u[0]--;
            }
#pragma warning disable CS1718 // Comparison made to same variable --> check for NAN
            if ((d != x) && (x == x))
#pragma warning restore CS1718
            {
                u[0] |= 1U;
            }

            Buffer.BlockCopy(u, 0, f, 0, sizeof(float));

            return __float2bfloat16(f[0]);
        }

        private static bfloat16 __float2bfloat16(float a)
        {
            bfloat16 r = new bfloat16();
            uint sign = 0;
            uint remainder = 0;
            r.x = __internal_float2bfloat16(a, ref sign, ref remainder);
            if ((remainder > 0x80000000U) || ((remainder == 0x80000000U) && ((r.x & 0x1U) != 0U)))
            {
                r.x++;
            }

            return r;
        }

        /// <summary>
        /// 
        /// </summary>
        public override string ToString()
        {
            return x.ToString();
        }
    }

    /// <summary>
    /// two bfloat16 floating point (x,y)
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct bfloat162
    {
        uint x;
    }


    /// <summary>
    /// CUDA external memory
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUexternalMemory
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    /// <summary>
    /// CUDA external semaphore
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUexternalSemaphore
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    /// <summary>
    /// CUDA graph
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUgraph
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    /// <summary>
    /// CUDA graph node
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUgraphNode
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;

        #region Properties
        /// <summary>
        /// Returns the type of the Node
        /// </summary>
        public CUgraphNodeType Type
        {
            get
            {
                CUgraphNodeType type = new CUgraphNodeType();
                CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphNodeGetType(this, ref type);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphNodeGetType", res));
                if (res != CUResult.Success) throw new CudaException(res);

                return type;
            }
        }
        #endregion

        #region Methods
        /// <summary>
        /// Sets the parameters of host node nodeParams.
        /// </summary>
        /// <param name="nodeParams"></param>
        public void SetParameters(CudaHostNodeParams nodeParams)
        {
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphHostNodeSetParams(this, ref nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphHostNodeSetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Sets the parameters of kernel node nodeParams.
        /// </summary>
        /// <param name="nodeParams"></param>
        public void SetParameters(CudaKernelNodeParams nodeParams)
        {
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphKernelNodeSetParams(this, ref nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphKernelNodeSetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Sets the parameters of memcpy node nodeParams.
        /// </summary>
        /// <param name="nodeParams"></param>
        public void SetParameters(CUDAMemCpy3D nodeParams)
        {
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphMemcpyNodeSetParams(this, ref nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphMemcpyNodeSetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Sets the parameters of memset node nodeParams.
        /// </summary>
        /// <param name="nodeParams"></param>
        public void SetParameters(CudaMemsetNodeParams nodeParams)
        {
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphMemsetNodeSetParams(this, ref nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphMemsetNodeSetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Sets an external semaphore signal node's parameters.
        /// </summary>
        /// <param name="nodeParams"></param>
        public void SetParameters(CudaExtSemSignalNodeParams nodeParams)
        {
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphExternalSemaphoresSignalNodeSetParams(this, nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphExternalSemaphoresSignalNodeSetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Sets an external semaphore wait node's parameters.
        /// </summary>
        /// <param name="nodeParams"></param>
        public void SetParameters(CudaExtSemWaitNodeParams nodeParams)
        {
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphExternalSemaphoresWaitNodeSetParams(this, nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphExternalSemaphoresWaitNodeSetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Gets the parameters of host node.
        /// </summary>
        /// <param name="nodeParams"></param>
        public void GetParameters(ref CudaHostNodeParams nodeParams)
        {
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphHostNodeGetParams(this, ref nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphHostNodeGetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Gets the parameters of kernel node.
        /// </summary>
        /// <param name="nodeParams"></param>
        public void GetParameters(ref CudaKernelNodeParams nodeParams)
        {
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphKernelNodeGetParams(this, ref nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphKernelNodeGetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Gets the parameters of memcpy node.
        /// </summary>
        /// <param name="nodeParams"></param>
        public void GetParameters(ref CUDAMemCpy3D nodeParams)
        {
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphMemcpyNodeGetParams(this, ref nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphMemcpyNodeGetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Gets the parameters of memset node.
        /// </summary>
        /// <param name="nodeParams"></param>
        public void GetParameters(ref CudaMemsetNodeParams nodeParams)
        {
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphMemsetNodeGetParams(this, ref nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphMemsetNodeGetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Gets the external semaphore signal node's parameters.
        /// </summary>
        /// <param name="nodeParams"></param>
        public void GetParameters(CudaExtSemSignalNodeParams nodeParams)
        {
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphExternalSemaphoresSignalNodeGetParams(this, nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphExternalSemaphoresSignalNodeGetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Gets the external semaphore wait node's parameters.
        /// </summary>
        /// <param name="nodeParams"></param>
        public void GetParameters(CudaExtSemWaitNodeParams nodeParams)
        {
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphExternalSemaphoresWaitNodeGetParams(this, nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphExternalSemaphoresWaitNodeGetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Returns a memory alloc node's parameters
        /// </summary>
        /// <param name="nodeParams"></param>
        public void GetParameters(ref CUDA_MEM_ALLOC_NODE_PARAMS nodeParams)
        {
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphMemAllocNodeGetParams(this, ref nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphMemAllocNodeGetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Returns a memory free node's parameters
        /// </summary>
        /// <param name="nodeParams"></param>
        public void GetParameters(ref CUdeviceptr nodeParams)
        {
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphMemFreeNodeGetParams(this, ref nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphMemFreeNodeGetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Only for ChildGraphNodes
        /// </summary>
        /// <returns></returns>
        public CudaGraph GetGraph()
        {
            CUgraph graph = new CUgraph();
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphChildGraphNodeGetGraph(this, ref graph);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphChildGraphNodeGetGraph", res));
            if (res != CUResult.Success) throw new CudaException(res);

            return new CudaGraph(graph);
        }

        /// <summary>
        /// Returns a node's dependencies.
        /// </summary>
        /// <returns></returns>
        public CUgraphNode[] GetDependencies()
        {
            SizeT numNodes = new SizeT();
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphNodeGetDependencies(this, null, ref numNodes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphNodeGetDependencies", res));
            if (res != CUResult.Success) throw new CudaException(res);

            if (numNodes > 0)
            {
                CUgraphNode[] nodes = new CUgraphNode[numNodes];
                res = DriverAPINativeMethods.GraphManagment.cuGraphNodeGetDependencies(this, nodes, ref numNodes);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphNodeGetDependencies", res));
                if (res != CUResult.Success) throw new CudaException(res);

                return nodes;
            }

            return null;
        }

        /// <summary>
        /// Returns a node's dependent nodes
        /// </summary>
        public CUgraphNode[] GetDependentNodes()
        {
            SizeT numNodes = new SizeT();
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphNodeGetDependentNodes(this, null, ref numNodes);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphNodeGetDependentNodes", res));
            if (res != CUResult.Success) throw new CudaException(res);

            if (numNodes > 0)
            {
                CUgraphNode[] nodes = new CUgraphNode[numNodes];
                res = DriverAPINativeMethods.GraphManagment.cuGraphNodeGetDependentNodes(this, nodes, ref numNodes);
                Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphNodeGetDependentNodes", res));
                if (res != CUResult.Success) throw new CudaException(res);

                return nodes;
            }

            return null;
        }


        /// <summary>
        /// Copies attributes from source node to destination node.<para/>
        /// Copies attributes from source node \p src to destination node \p dst. Both node must have the same context.
        /// </summary>
        /// <param name="dst">Destination node</param>
        public void cuGraphKernelNodeCopyAttributes(CUgraphNode dst)
        {
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphKernelNodeCopyAttributes(dst, this);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphKernelNodeCopyAttributes", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Queries node attribute.<para/>
        /// Queries attribute \p attr from node \p hNode and stores it in corresponding member of \p value_out.
        /// </summary>
        /// <param name="attr"></param>
        public CUkernelNodeAttrValue GetAttribute(CUkernelNodeAttrID attr)
        {
            CUkernelNodeAttrValue value = new CUkernelNodeAttrValue();
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphKernelNodeGetAttribute(this, attr, ref value);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphKernelNodeGetAttribute", res));
            if (res != CUResult.Success) throw new CudaException(res);
            return value;
        }

        /// <summary>
        /// Sets node attribute.<para/>
        /// Sets attribute \p attr on node \p hNode from corresponding attribute of value.
        /// </summary>
        /// <param name="attr"></param>
        /// <param name="value"></param>
        public void SetAttribute(CUkernelNodeAttrID attr, CUkernelNodeAttrValue value)
        {
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphKernelNodeSetAttribute(this, attr, ref value);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphKernelNodeSetAttribute", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }


        /// <summary>
        /// Returns the event associated with an event record node
        /// </summary>
        public CudaEvent GetRecordEvent()
        {
            CUevent event_out = new CUevent();
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphEventRecordNodeGetEvent(this, ref event_out);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphEventRecordNodeGetEvent", res));
            if (res != CUResult.Success) throw new CudaException(res);
            return new CudaEvent(event_out);
        }

        /// <summary>
        /// Sets an event record node's event
        /// </summary>
        public void SetRecordEvent(CudaEvent event_)
        {
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphEventRecordNodeSetEvent(this, event_.Event);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphEventRecordNodeSetEvent", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }


        /// <summary>
        /// Returns the event associated with an event wait node
        /// </summary>
        public CudaEvent GetWaitEvent()
        {
            CUevent event_out = new CUevent();
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphEventWaitNodeGetEvent(this, ref event_out);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphEventWaitNodeGetEvent", res));
            if (res != CUResult.Success) throw new CudaException(res);
            return new CudaEvent(event_out);
        }

        /// <summary>
        /// Sets an event wait node's event
        /// </summary>
        public void SetWaitEvent(CudaEvent event_)
        {
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphEventWaitNodeSetEvent(this, event_.Event);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphEventWaitNodeSetEvent", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        #endregion
    }

    /// <summary>
    /// CUDA executable graph
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUgraphExec
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUmemGenericAllocationHandle
    {
        /// <summary>
        /// 
        /// </summary>
        public ulong Pointer;
    }

    /// <summary>
    /// CUDA memory pool
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUmemoryPool
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }

    /// <summary>
    /// CUDA user object for graphs
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUuserObject
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }
    #endregion

}
