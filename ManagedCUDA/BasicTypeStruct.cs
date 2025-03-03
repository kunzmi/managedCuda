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
using System.ComponentModel;
using System.Diagnostics;
using System.Net;
using System.Runtime.InteropServices;

namespace ManagedCuda.BasicTypes
{
    #region Structs
    /// <summary>
    /// Legacy device properties
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDeviceProperties
    {
        /// <summary>
        /// Maximum number of threads per block
        /// </summary>
        public int maxThreadsPerBlock;

        /// <summary>
        /// Maximum size of each dimension of a block
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3, ArraySubType = UnmanagedType.I4)]
        public int[] maxThreadsDim;

        /// <summary>
        /// Maximum size of each dimension of a grid
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3, ArraySubType = UnmanagedType.I4)]
        public int[] maxGridSize;

        /// <summary>
        /// Shared memory available per block in bytes
        /// </summary>
        public int sharedMemPerBlock;

        /// <summary>
        /// Constant memory available on device in bytes
        /// </summary>
        public int totalConstantMemory;

        /// <summary>
        /// Warp size in threads. Also called SIMD width.
        /// </summary>
        public int SIMDWidth;

        /// <summary>
        /// Maximum pitch in bytes allowed by the memory copy functions that involve memory regions allocated through
        /// <see cref="DriverAPINativeMethods.MemoryManagement.cuMemAllocPitch_v2"/>.
        /// </summary>
        public int memPitch;

        /// <summary>
        /// 32-bit registers available per block
        /// </summary>
        public int regsPerBlock;

        /// <summary>
        /// Clock frequency in kilohertz
        /// </summary>
        public int clockRate;

        /// <summary>
        /// Alignment requirement for textures. texture base addresses that are aligned to textureAlign bytes do not
        /// need an offset applied to texture fetches.
        /// </summary>
        public int textureAlign;
    }

    /// <summary>
    /// 2D memory copy parameters
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDAMemCpy2D
    {
        /// <summary>
        /// Source X in bytes
        /// </summary>
        public SizeT srcXInBytes;

        /// <summary>
        /// Source Y
        /// </summary>
        public SizeT srcY;

        /// <summary>
        /// Source memory type (host, device, array)
        /// </summary>
        public CUMemoryType srcMemoryType;

        /// <summary>
        /// Source host pointer
        /// </summary>
        public IntPtr srcHost;

        /// <summary>
        /// Source device pointer
        /// </summary>
        public CUdeviceptr srcDevice;

        /// <summary>
        /// Source array reference
        /// </summary>
        public CUarray srcArray;

        /// <summary>
        /// Source pitch (ignored when src is array)
        /// </summary>
        public SizeT srcPitch;

        /// <summary>
        /// Destination X in bytes
        /// </summary>
        public SizeT dstXInBytes;

        /// <summary>
        /// Destination Y
        /// </summary>
        public SizeT dstY;

        /// <summary>
        /// Destination memory type (host, device, array)
        /// </summary>
        public CUMemoryType dstMemoryType;

        /// <summary>
        /// Destination host pointer
        /// </summary>
        public IntPtr dstHost;

        /// <summary>
        /// Destination device pointer
        /// </summary>
        public CUdeviceptr dstDevice;

        /// <summary>
        /// Destination array reference
        /// </summary>
        public CUarray dstArray;

        /// <summary>
        /// Destination pitch (ignored when dst is array)
        /// </summary>
        public SizeT dstPitch;

        /// <summary>
        /// Width of 2D memory copy in bytes
        /// </summary>
        public SizeT WidthInBytes;

        /// <summary>
        /// Height of 2D memory copy
        /// </summary>
        public SizeT Height;
    }

    /// <summary>
    /// 3D memory copy parameters
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDAMemCpy3D
    {
        /// <summary>
        /// Source X in bytes
        /// </summary>
        public SizeT srcXInBytes;

        /// <summary>
        /// Source Y
        /// </summary>
        public SizeT srcY;

        /// <summary>
        /// Source Z
        /// </summary>
        public SizeT srcZ;

        /// <summary>
        /// Source LOD
        /// </summary>
        public SizeT srcLOD;

        /// <summary>
        /// Source memory type (host, device, array)
        /// </summary>
        public CUMemoryType srcMemoryType;


        /// <summary>
        /// Source host pointer
        /// </summary>
        public IntPtr srcHost;


        /// <summary>
        /// Source device pointer
        /// </summary>
        public CUdeviceptr srcDevice;

        /// <summary>
        /// Source array reference
        /// </summary>
        public CUarray srcArray;

        /// <summary>
        /// Must be NULL
        /// </summary>
        public IntPtr reserved0;

        /// <summary>
        /// Source pitch (ignored when src is array)
        /// </summary>
        public SizeT srcPitch;

        /// <summary>
        /// Source height (ignored when src is array; may be 0 if Depth==1)
        /// </summary>
        public SizeT srcHeight;

        /// <summary>
        /// Destination X in bytes
        /// </summary>
        public SizeT dstXInBytes;

        /// <summary>
        /// Destination Y
        /// </summary>
        public SizeT dstY;

        /// <summary>
        /// Destination Z
        /// </summary>
        public SizeT dstZ;

        /// <summary>
        /// Destination LOD
        /// </summary>
        public SizeT dstLOD;

        /// <summary>
        /// Destination memory type (host, device, array)
        /// </summary>
        public CUMemoryType dstMemoryType;

        /// <summary>
        /// Destination host pointer
        /// </summary>
        public IntPtr dstHost;

        /// <summary>
        /// Destination device pointer
        /// </summary>
        public CUdeviceptr dstDevice;

        /// <summary>
        /// Destination array reference
        /// </summary>
        public CUarray dstArray;

        /// <summary>
        /// Must be NULL
        /// </summary>
        public IntPtr reserved1;

        /// <summary>
        /// Destination pitch (ignored when dst is array)
        /// </summary>
        public SizeT dstPitch;

        /// <summary>
        /// Destination height (ignored when dst is array; may be 0 if Depth==1)
        /// </summary>
        public SizeT dstHeight;

        /// <summary>
        /// Width of 3D memory copy in bytes
        /// </summary>
        public SizeT WidthInBytes;

        /// <summary>
        /// Height of 3D memory copy
        /// </summary>
        public SizeT Height;

        /// <summary>
        /// Depth of 3D memory copy
        /// </summary>
        public SizeT Depth;
    }

    /// <summary>
    /// 3D memory copy parameters
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDAMemCpy3DPeer
    {
        /// <summary>
        /// Source X in bytes
        /// </summary>
        public SizeT srcXInBytes;

        /// <summary>
        /// Source Y
        /// </summary>
        public SizeT srcY;

        /// <summary>
        /// Source Z
        /// </summary>
        public SizeT srcZ;

        /// <summary>
        /// Source LOD
        /// </summary>
        public SizeT srcLOD;

        /// <summary>
        /// Source memory type (host, device, array)
        /// </summary>
        public CUMemoryType srcMemoryType;


        /// <summary>
        /// Source host pointer
        /// </summary>
        public IntPtr srcHost;


        /// <summary>
        /// Source device pointer
        /// </summary>
        public CUdeviceptr srcDevice;

        /// <summary>
        /// Source array reference
        /// </summary>
        public CUarray srcArray;

        /// <summary>
        /// Source context (ignored with srcMemoryType is array)
        /// </summary>
        public CUcontext srcContext;

        /// <summary>
        /// Source pitch (ignored when src is array)
        /// </summary>
        public SizeT srcPitch;

        /// <summary>
        /// Source height (ignored when src is array; may be 0 if Depth==1)
        /// </summary>
        public SizeT srcHeight;

        /// <summary>
        /// Destination X in bytes
        /// </summary>
        public SizeT dstXInBytes;

        /// <summary>
        /// Destination Y
        /// </summary>
        public SizeT dstY;

        /// <summary>
        /// Destination Z
        /// </summary>
        public SizeT dstZ;

        /// <summary>
        /// Destination LOD
        /// </summary>
        public SizeT dstLOD;

        /// <summary>
        /// Destination memory type (host, device, array)
        /// </summary>
        public CUMemoryType dstMemoryType;

        /// <summary>
        /// Destination host pointer
        /// </summary>
        public IntPtr dstHost;

        /// <summary>
        /// Destination device pointer
        /// </summary>
        public CUdeviceptr dstDevice;

        /// <summary>
        /// Destination array reference
        /// </summary>
        public CUarray dstArray;

        /// <summary>
        /// Destination context (ignored with dstMemoryType is array)
        /// </summary>
        public CUcontext dstContext;

        /// <summary>
        /// Destination pitch (ignored when dst is array)
        /// </summary>
        public SizeT dstPitch;

        /// <summary>
        /// Destination height (ignored when dst is array; may be 0 if Depth==1)
        /// </summary>
        public SizeT dstHeight;

        /// <summary>
        /// Width of 3D memory copy in bytes
        /// </summary>
        public SizeT WidthInBytes;

        /// <summary>
        /// Height of 3D memory copy
        /// </summary>
        public SizeT Height;

        /// <summary>
        /// Depth of 3D memory copy
        /// </summary>
        public SizeT Depth;
    }

    /// <summary>
    /// Array descriptor
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDAArrayDescriptor
    {
        /// <summary>
        /// Width of array
        /// </summary>
        public SizeT Width;

        /// <summary>
        /// Height of array
        /// </summary>
        public SizeT Height;

        /// <summary>
        /// Array format
        /// </summary>
        public CUArrayFormat Format;

        /// <summary>
        /// Channels per array element
        /// </summary>
        public uint NumChannels;
    }

    /// <summary>
    /// 3D array descriptor
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDAArray3DDescriptor
    {
        /// <summary>
        /// Width of 3D array
        /// </summary>
        public SizeT Width;

        /// <summary>
        /// Height of 3D array
        /// </summary>
        public SizeT Height;

        /// <summary>
        /// Depth of 3D array
        /// </summary>
        public SizeT Depth;

        /// <summary>
        /// Array format
        /// </summary>
        public CUArrayFormat Format;

        /// <summary>
        /// Channels per array element
        /// </summary>
        public uint NumChannels;

        /// <summary>
        /// Flags
        /// </summary>
        public CUDAArray3DFlags Flags;
    }

    /// <summary>
    /// Idea of a SizeT type from http://blogs.hoopoe-cloud.com/index.php/tag/cudanet/, entry from Tuesday, September 15th, 2009
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct SizeT
    {
        private UIntPtr value;
        /// <summary>
        /// 
        /// </summary>
        /// <param name="value"></param>
        public SizeT(int value)
        {
            this.value = new UIntPtr((uint)value);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="value"></param>
        public SizeT(uint value)
        {
            this.value = new UIntPtr(value);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="value"></param>
        public SizeT(long value)
        {
            this.value = new UIntPtr((ulong)value);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="value"></param>
        public SizeT(ulong value)
        {
            this.value = new UIntPtr(value);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="value"></param>
        public SizeT(UIntPtr value)
        {
            this.value = value;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="value"></param>
        public SizeT(IntPtr value)
        {
            this.value = new UIntPtr((ulong)value.ToInt64());
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="t"></param>
        /// <returns></returns>
        public static implicit operator int(SizeT t)
        {
            return (int)t.value.ToUInt32();
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="t"></param>
        /// <returns></returns>
        public static implicit operator uint(SizeT t)
        {
            return (t.value.ToUInt32());
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="t"></param>
        /// <returns></returns>
        public static implicit operator long(SizeT t)
        {
            return (long)t.value.ToUInt64();
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="t"></param>
        /// <returns></returns>
        public static implicit operator ulong(SizeT t)
        {
            return (t.value.ToUInt64());
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="t"></param>
        /// <returns></returns>
        public static implicit operator UIntPtr(SizeT t)
        {
            return t.value;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="t"></param>
        /// <returns></returns>
        public static implicit operator IntPtr(SizeT t)
        {
            return new IntPtr((long)t.value.ToUInt64());
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static implicit operator SizeT(int value)
        {
            return new SizeT(value);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static implicit operator SizeT(uint value)
        {
            return new SizeT(value);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static implicit operator SizeT(long value)
        {
            return new SizeT(value);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static implicit operator SizeT(ulong value)
        {
            return new SizeT(value);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static implicit operator SizeT(IntPtr value)
        {
            return new SizeT(value);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static implicit operator SizeT(UIntPtr value)
        {
            return new SizeT(value);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static bool operator !=(SizeT val1, SizeT val2)
        {
            return (val1.value != val2.value);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static bool operator ==(SizeT val1, SizeT val2)
        {
            return (val1.value == val2.value);
        }
        #region +
        /// <summary>
        /// Define operator + on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static SizeT operator +(SizeT val1, SizeT val2)
        {
            return new SizeT(val1.value.ToUInt64() + val2.value.ToUInt64());
        }
        /// <summary>
        /// Define operator + on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static SizeT operator +(SizeT val1, int val2)
        {
            return new SizeT(val1.value.ToUInt64() + (ulong)val2);
        }
        /// <summary>
        /// Define operator + on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static SizeT operator +(int val1, SizeT val2)
        {
            return new SizeT((ulong)val1 + val2.value.ToUInt64());
        }
        /// <summary>
        /// Define operator + on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static SizeT operator +(uint val1, SizeT val2)
        {
            return new SizeT((ulong)val1 + val2.value.ToUInt64());
        }
        /// <summary>
        /// Define operator + on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static SizeT operator +(SizeT val1, uint val2)
        {
            return new SizeT(val1.value.ToUInt64() + (ulong)val2);
        }
        #endregion
        #region -
        /// <summary>
        /// Define operator - on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static SizeT operator -(SizeT val1, SizeT val2)
        {
            return new SizeT(val1.value.ToUInt64() - val2.value.ToUInt64());
        }
        /// <summary>
        /// Define operator - on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static SizeT operator -(SizeT val1, int val2)
        {
            return new SizeT(val1.value.ToUInt64() - (ulong)val2);
        }
        /// <summary>
        /// Define operator - on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static SizeT operator -(int val1, SizeT val2)
        {
            return new SizeT((ulong)val1 - val2.value.ToUInt64());
        }
        /// <summary>
        /// Define operator - on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static SizeT operator -(SizeT val1, uint val2)
        {
            return new SizeT(val1.value.ToUInt64() - (ulong)val2);
        }
        /// <summary>
        /// Define operator - on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static SizeT operator -(uint val1, SizeT val2)
        {
            return new SizeT((ulong)val1 - val2.value.ToUInt64());
        }
        #endregion
        #region *
        /// <summary>
        /// Define operator * on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static SizeT operator *(SizeT val1, SizeT val2)
        {
            return new SizeT(val1.value.ToUInt64() * val2.value.ToUInt64());
        }
        /// <summary>
        /// Define operator * on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static SizeT operator *(SizeT val1, int val2)
        {
            return new SizeT(val1.value.ToUInt64() * (ulong)val2);
        }
        /// <summary>
        /// Define operator * on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static SizeT operator *(int val1, SizeT val2)
        {
            return new SizeT((ulong)val1 * val2.value.ToUInt64());
        }
        /// <summary>
        /// Define operator * on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static SizeT operator *(SizeT val1, uint val2)
        {
            return new SizeT(val1.value.ToUInt64() * (ulong)val2);
        }
        /// <summary>
        /// Define operator * on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static SizeT operator *(uint val1, SizeT val2)
        {
            return new SizeT((ulong)val1 * val2.value.ToUInt64());
        }
        #endregion
        #region /
        /// <summary>
        /// Define operator / on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static SizeT operator /(SizeT val1, SizeT val2)
        {
            return new SizeT(val1.value.ToUInt64() / val2.value.ToUInt64());
        }
        /// <summary>
        /// Define operator / on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static SizeT operator /(SizeT val1, int val2)
        {
            return new SizeT(val1.value.ToUInt64() / (ulong)val2);
        }
        /// <summary>
        /// Define operator / on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static SizeT operator /(int val1, SizeT val2)
        {
            return new SizeT((ulong)val1 / val2.value.ToUInt64());
        }
        /// <summary>
        /// Define operator / on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static SizeT operator /(SizeT val1, uint val2)
        {
            return new SizeT(val1.value.ToUInt64() / (ulong)val2);
        }
        /// <summary>
        /// Define operator / on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static SizeT operator /(uint val1, SizeT val2)
        {
            return new SizeT((ulong)val1 / val2.value.ToUInt64());
        }
        #endregion
        #region >
        /// <summary>
        /// Define operator &gt; on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static bool operator >(SizeT val1, SizeT val2)
        {
            return val1.value.ToUInt64() > val2.value.ToUInt64();
        }
        /// <summary>
        /// Define operator &gt; on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static bool operator >(SizeT val1, int val2)
        {
            return val1.value.ToUInt64() > (ulong)val2;
        }
        /// <summary>
        /// Define operator &gt; on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static bool operator >(int val1, SizeT val2)
        {
            return (ulong)val1 > val2.value.ToUInt64();
        }
        /// <summary>
        /// Define operator &gt; on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static bool operator >(SizeT val1, uint val2)
        {
            return val1.value.ToUInt64() > (ulong)val2;
        }
        /// <summary>
        /// Define operator &gt; on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static bool operator >(uint val1, SizeT val2)
        {
            return (ulong)val1 > val2.value.ToUInt64();
        }
        #endregion
        #region <
        /// <summary>
        /// Define operator &lt; on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static bool operator <(SizeT val1, SizeT val2)
        {
            return val1.value.ToUInt64() < val2.value.ToUInt64();
        }
        /// <summary>
        /// Define operator &lt; on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static bool operator <(SizeT val1, int val2)
        {
            return val1.value.ToUInt64() < (ulong)val2;
        }
        /// <summary>
        /// Define operator &lt; on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static bool operator <(int val1, SizeT val2)
        {
            return (ulong)val1 < val2.value.ToUInt64();
        }
        /// <summary>
        /// Define operator &lt; on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static bool operator <(SizeT val1, uint val2)
        {
            return val1.value.ToUInt64() < (ulong)val2;
        }
        /// <summary>
        /// Define operator &lt; on converted to ulong values to avoid fall back to int
        /// </summary>
        /// <param name="val1"></param>
        /// <param name="val2"></param>
        /// <returns></returns>
        public static bool operator <(uint val1, SizeT val2)
        {
            return (ulong)val1 < val2.value.ToUInt64();
        }
        #endregion
        /// <summary>
        /// 
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            if (!(obj is SizeT)) return false;
            SizeT o = (SizeT)obj;
            return this.value.Equals(o.value);
        }
        /// <summary>
        /// returns this.value.ToString()
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            if (IntPtr.Size == 4)
                return ((uint)this.value.ToUInt32()).ToString();
            else
                return ((ulong)this.value.ToUInt64()).ToString();
        }
        /// <summary>
        /// Returns this.value.GetHashCode()
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            return this.value.GetHashCode();
        }
    }

    /// <summary>
    /// Inner struct for CudaResourceDesc
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaResourceDescLinear
    {
        /// <summary>
        /// Device pointer
        /// </summary>
        public CUdeviceptr devPtr;
        /// <summary>
        /// Array format
        /// </summary>
        public CUArrayFormat format;
        /// <summary>
        /// Channels per array element
        /// </summary>
        public uint numChannels;
        /// <summary>
        /// Size in bytes
        /// </summary>
        public SizeT sizeInBytes;
    }

    /// <summary>
    /// Inner struct for CudaResourceDesc
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaResourceDescPitch2D
    {
        /// <summary>
        /// Device pointer
        /// </summary>
        public CUdeviceptr devPtr;
        /// <summary>
        /// Array format
        /// </summary>
        public CUArrayFormat format;
        /// <summary>
        /// Channels per array element
        /// </summary>
        public uint numChannels;
        /// <summary>
        /// Width of the array in elements
        /// </summary>
        public SizeT width;
        /// <summary>
        /// Height of the array in elements
        /// </summary>
        public SizeT height;
        /// <summary>
        /// Pitch between two rows in bytes
        /// </summary>
        public SizeT pitchInBytes;
    }

    /// <summary>
    /// Mimics the union "CUDA_RESOURCE_DESC.res" in cuda.h
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CudaResourceDescUnion
    {
        /// <summary>
        /// CUDA array
        /// </summary>
        [FieldOffset(0)]
        public CUarray hArray;

        /// <summary>
        /// CUDA mipmapped array
        /// </summary>
        [FieldOffset(0)]
        public CUmipmappedArray hMipmappedArray;

        /// <summary>
        /// Linear memory
        /// </summary>
        [FieldOffset(0)]
        public CudaResourceDescLinear linear;

        /// <summary>
        /// Linear pitched 2D memory
        /// </summary>
        [FieldOffset(0)]
        public CudaResourceDescPitch2D pitch2D;

        //In cuda header, an int[32] fixes the union size to 128 bytes, we
        //achieve the same in C# if we set at offset 124 an simple int
        [FieldOffset(31 * 4)]
        private int reserved;
    }

    /// <summary>
    /// CUDA Resource descriptor
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaResourceDesc
    {
        #region Constructors
        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaArray1D var)
        {
            resType = CUResourceType.Array;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.hArray = var.CUArray;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaArray2D var)
        {
            resType = CUResourceType.Array;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.hArray = var.CUArray;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaArray3D var)
        {
            resType = CUResourceType.Array;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.hArray = var.CUArray;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaMipmappedArray var)
        {
            resType = CUResourceType.MipmappedArray;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.hMipmappedArray = var.CUMipmappedArray; ;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaDeviceVariable<float> var)
        {
            resType = CUResourceType.Linear;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.linear = new CudaResourceDescLinear();

            res.linear.devPtr = var.DevicePointer;
            res.linear.format = CUArrayFormat.Float;
            res.linear.numChannels = 1;
            res.linear.sizeInBytes = var.SizeInBytes;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaDeviceVariable<VectorTypes.float2> var)
        {
            resType = CUResourceType.Linear;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.linear = new CudaResourceDescLinear();

            res.linear.devPtr = var.DevicePointer;
            res.linear.format = CUArrayFormat.Float;
            res.linear.numChannels = 2;
            res.linear.sizeInBytes = var.SizeInBytes;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaDeviceVariable<VectorTypes.float4> var)
        {
            resType = CUResourceType.Linear;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.linear = new CudaResourceDescLinear();

            res.linear.devPtr = var.DevicePointer;
            res.linear.format = CUArrayFormat.Float;
            res.linear.numChannels = 4;
            res.linear.sizeInBytes = var.SizeInBytes;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaDeviceVariable<int> var)
        {
            resType = CUResourceType.Linear;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.linear = new CudaResourceDescLinear();

            res.linear.devPtr = var.DevicePointer;
            res.linear.format = CUArrayFormat.SignedInt32;
            res.linear.numChannels = 1;
            res.linear.sizeInBytes = var.SizeInBytes;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaDeviceVariable<VectorTypes.int2> var)
        {
            resType = CUResourceType.Linear;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.linear = new CudaResourceDescLinear();

            res.linear.devPtr = var.DevicePointer;
            res.linear.format = CUArrayFormat.SignedInt32;
            res.linear.numChannels = 2;
            res.linear.sizeInBytes = var.SizeInBytes;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaDeviceVariable<VectorTypes.int4> var)
        {
            resType = CUResourceType.Linear;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.linear = new CudaResourceDescLinear();

            res.linear.devPtr = var.DevicePointer;
            res.linear.format = CUArrayFormat.SignedInt16;
            res.linear.numChannels = 4;
            res.linear.sizeInBytes = var.SizeInBytes;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaDeviceVariable<short> var)
        {
            resType = CUResourceType.Linear;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.linear = new CudaResourceDescLinear();

            res.linear.devPtr = var.DevicePointer;
            res.linear.format = CUArrayFormat.SignedInt16;
            res.linear.numChannels = 1;
            res.linear.sizeInBytes = var.SizeInBytes;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaDeviceVariable<VectorTypes.short2> var)
        {
            resType = CUResourceType.Linear;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.linear = new CudaResourceDescLinear();

            res.linear.devPtr = var.DevicePointer;
            res.linear.format = CUArrayFormat.SignedInt16;
            res.linear.numChannels = 2;
            res.linear.sizeInBytes = var.SizeInBytes;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaDeviceVariable<VectorTypes.short4> var)
        {
            resType = CUResourceType.Linear;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.linear = new CudaResourceDescLinear();

            res.linear.devPtr = var.DevicePointer;
            res.linear.format = CUArrayFormat.SignedInt32;
            res.linear.numChannels = 4;
            res.linear.sizeInBytes = var.SizeInBytes;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaDeviceVariable<sbyte> var)
        {
            resType = CUResourceType.Linear;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.linear = new CudaResourceDescLinear();

            res.linear.devPtr = var.DevicePointer;
            res.linear.format = CUArrayFormat.SignedInt8;
            res.linear.numChannels = 1;
            res.linear.sizeInBytes = var.SizeInBytes;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaDeviceVariable<VectorTypes.char2> var)
        {
            resType = CUResourceType.Linear;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.linear = new CudaResourceDescLinear();

            res.linear.devPtr = var.DevicePointer;
            res.linear.format = CUArrayFormat.SignedInt8;
            res.linear.numChannels = 2;
            res.linear.sizeInBytes = var.SizeInBytes;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaDeviceVariable<VectorTypes.char4> var)
        {
            resType = CUResourceType.Linear;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.linear = new CudaResourceDescLinear();

            res.linear.devPtr = var.DevicePointer;
            res.linear.format = CUArrayFormat.SignedInt8;
            res.linear.numChannels = 4;
            res.linear.sizeInBytes = var.SizeInBytes;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaDeviceVariable<byte> var)
        {
            resType = CUResourceType.Linear;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.linear = new CudaResourceDescLinear();

            res.linear.devPtr = var.DevicePointer;
            res.linear.format = CUArrayFormat.UnsignedInt8;
            res.linear.numChannels = 1;
            res.linear.sizeInBytes = var.SizeInBytes;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaDeviceVariable<VectorTypes.uchar2> var)
        {
            resType = CUResourceType.Linear;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.linear = new CudaResourceDescLinear();

            res.linear.devPtr = var.DevicePointer;
            res.linear.format = CUArrayFormat.UnsignedInt8;
            res.linear.numChannels = 2;
            res.linear.sizeInBytes = var.SizeInBytes;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaDeviceVariable<VectorTypes.uchar4> var)
        {
            resType = CUResourceType.Linear;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.linear = new CudaResourceDescLinear();

            res.linear.devPtr = var.DevicePointer;
            res.linear.format = CUArrayFormat.UnsignedInt8;
            res.linear.numChannels = 4;
            res.linear.sizeInBytes = var.SizeInBytes;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaDeviceVariable<ushort> var)
        {
            resType = CUResourceType.Linear;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.linear = new CudaResourceDescLinear();

            res.linear.devPtr = var.DevicePointer;
            res.linear.format = CUArrayFormat.UnsignedInt16;
            res.linear.numChannels = 1;
            res.linear.sizeInBytes = var.SizeInBytes;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaDeviceVariable<VectorTypes.ushort2> var)
        {
            resType = CUResourceType.Linear;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.linear = new CudaResourceDescLinear();

            res.linear.devPtr = var.DevicePointer;
            res.linear.format = CUArrayFormat.UnsignedInt16;
            res.linear.numChannels = 2;
            res.linear.sizeInBytes = var.SizeInBytes;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaDeviceVariable<VectorTypes.ushort4> var)
        {
            resType = CUResourceType.Linear;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.linear = new CudaResourceDescLinear();

            res.linear.devPtr = var.DevicePointer;
            res.linear.format = CUArrayFormat.UnsignedInt16;
            res.linear.numChannels = 4;
            res.linear.sizeInBytes = var.SizeInBytes;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaDeviceVariable<uint> var)
        {
            resType = CUResourceType.Linear;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.linear = new CudaResourceDescLinear();

            res.linear.devPtr = var.DevicePointer;
            res.linear.format = CUArrayFormat.UnsignedInt32;
            res.linear.numChannels = 1;
            res.linear.sizeInBytes = var.SizeInBytes;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaDeviceVariable<VectorTypes.uint2> var)
        {
            resType = CUResourceType.Linear;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.linear = new CudaResourceDescLinear();

            res.linear.devPtr = var.DevicePointer;
            res.linear.format = CUArrayFormat.UnsignedInt32;
            res.linear.numChannels = 2;
            res.linear.sizeInBytes = var.SizeInBytes;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaDeviceVariable<VectorTypes.uint4> var)
        {
            resType = CUResourceType.Linear;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.linear = new CudaResourceDescLinear();

            res.linear.devPtr = var.DevicePointer;
            res.linear.format = CUArrayFormat.UnsignedInt32;
            res.linear.numChannels = 4;
            res.linear.sizeInBytes = var.SizeInBytes;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaResourceDescLinear var)
        {
            resType = CUResourceType.Linear;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.linear = var;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaResourceDescPitch2D var)
        {
            resType = CUResourceType.Pitch2D;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = var;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaPitchedDeviceVariable<float> var)
        {
            resType = CUResourceType.Pitch2D;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.pitch2D.devPtr = var.DevicePointer;
            res.pitch2D.format = CUArrayFormat.Float;
            res.pitch2D.height = var.Height;
            res.pitch2D.numChannels = 1;
            res.pitch2D.pitchInBytes = var.Pitch;
            res.pitch2D.width = var.Width;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaPitchedDeviceVariable<int> var)
        {
            resType = CUResourceType.Pitch2D;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.pitch2D.devPtr = var.DevicePointer;
            res.pitch2D.format = CUArrayFormat.SignedInt32;
            res.pitch2D.height = var.Height;
            res.pitch2D.numChannels = 1;
            res.pitch2D.pitchInBytes = var.Pitch;
            res.pitch2D.width = var.Width;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaPitchedDeviceVariable<short> var)
        {
            resType = CUResourceType.Pitch2D;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.pitch2D.devPtr = var.DevicePointer;
            res.pitch2D.format = CUArrayFormat.SignedInt16;
            res.pitch2D.height = var.Height;
            res.pitch2D.numChannels = 1;
            res.pitch2D.pitchInBytes = var.Pitch;
            res.pitch2D.width = var.Width;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaPitchedDeviceVariable<sbyte> var)
        {
            resType = CUResourceType.Pitch2D;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.pitch2D.devPtr = var.DevicePointer;
            res.pitch2D.format = CUArrayFormat.SignedInt8;
            res.pitch2D.height = var.Height;
            res.pitch2D.numChannels = 1;
            res.pitch2D.pitchInBytes = var.Pitch;
            res.pitch2D.width = var.Width;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaPitchedDeviceVariable<byte> var)
        {
            resType = CUResourceType.Pitch2D;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.pitch2D.devPtr = var.DevicePointer;
            res.pitch2D.format = CUArrayFormat.UnsignedInt8;
            res.pitch2D.height = var.Height;
            res.pitch2D.numChannels = 1;
            res.pitch2D.pitchInBytes = var.Pitch;
            res.pitch2D.width = var.Width;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaPitchedDeviceVariable<ushort> var)
        {
            resType = CUResourceType.Pitch2D;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.pitch2D.devPtr = var.DevicePointer;
            res.pitch2D.format = CUArrayFormat.UnsignedInt16;
            res.pitch2D.height = var.Height;
            res.pitch2D.numChannels = 1;
            res.pitch2D.pitchInBytes = var.Pitch;
            res.pitch2D.width = var.Width;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaPitchedDeviceVariable<uint> var)
        {
            resType = CUResourceType.Pitch2D;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.pitch2D.devPtr = var.DevicePointer;
            res.pitch2D.format = CUArrayFormat.UnsignedInt32;
            res.pitch2D.height = var.Height;
            res.pitch2D.numChannels = 1;
            res.pitch2D.pitchInBytes = var.Pitch;
            res.pitch2D.width = var.Width;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaPitchedDeviceVariable<VectorTypes.float2> var)
        {
            resType = CUResourceType.Pitch2D;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.pitch2D.devPtr = var.DevicePointer;
            res.pitch2D.format = CUArrayFormat.Float;
            res.pitch2D.height = var.Height;
            res.pitch2D.numChannels = 2;
            res.pitch2D.pitchInBytes = var.Pitch;
            res.pitch2D.width = var.Width;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaPitchedDeviceVariable<VectorTypes.int2> var)
        {
            resType = CUResourceType.Pitch2D;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.pitch2D.devPtr = var.DevicePointer;
            res.pitch2D.format = CUArrayFormat.SignedInt32;
            res.pitch2D.height = var.Height;
            res.pitch2D.numChannels = 2;
            res.pitch2D.pitchInBytes = var.Pitch;
            res.pitch2D.width = var.Width;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaPitchedDeviceVariable<VectorTypes.short2> var)
        {
            resType = CUResourceType.Pitch2D;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.pitch2D.devPtr = var.DevicePointer;
            res.pitch2D.format = CUArrayFormat.SignedInt16;
            res.pitch2D.height = var.Height;
            res.pitch2D.numChannels = 2;
            res.pitch2D.pitchInBytes = var.Pitch;
            res.pitch2D.width = var.Width;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaPitchedDeviceVariable<VectorTypes.char2> var)
        {
            resType = CUResourceType.Pitch2D;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.pitch2D.devPtr = var.DevicePointer;
            res.pitch2D.format = CUArrayFormat.SignedInt8;
            res.pitch2D.height = var.Height;
            res.pitch2D.numChannels = 2;
            res.pitch2D.pitchInBytes = var.Pitch;
            res.pitch2D.width = var.Width;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaPitchedDeviceVariable<VectorTypes.uchar2> var)
        {
            resType = CUResourceType.Pitch2D;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.pitch2D.devPtr = var.DevicePointer;
            res.pitch2D.format = CUArrayFormat.UnsignedInt8;
            res.pitch2D.height = var.Height;
            res.pitch2D.numChannels = 2;
            res.pitch2D.pitchInBytes = var.Pitch;
            res.pitch2D.width = var.Width;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaPitchedDeviceVariable<VectorTypes.ushort2> var)
        {
            resType = CUResourceType.Pitch2D;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.pitch2D.devPtr = var.DevicePointer;
            res.pitch2D.format = CUArrayFormat.UnsignedInt16;
            res.pitch2D.height = var.Height;
            res.pitch2D.numChannels = 2;
            res.pitch2D.pitchInBytes = var.Pitch;
            res.pitch2D.width = var.Width;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaPitchedDeviceVariable<VectorTypes.uint2> var)
        {
            resType = CUResourceType.Pitch2D;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.pitch2D.devPtr = var.DevicePointer;
            res.pitch2D.format = CUArrayFormat.UnsignedInt32;
            res.pitch2D.height = var.Height;
            res.pitch2D.numChannels = 2;
            res.pitch2D.pitchInBytes = var.Pitch;
            res.pitch2D.width = var.Width;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaPitchedDeviceVariable<VectorTypes.float4> var)
        {
            resType = CUResourceType.Pitch2D;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.pitch2D.devPtr = var.DevicePointer;
            res.pitch2D.format = CUArrayFormat.Float;
            res.pitch2D.height = var.Height;
            res.pitch2D.numChannels = 4;
            res.pitch2D.pitchInBytes = var.Pitch;
            res.pitch2D.width = var.Width;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaPitchedDeviceVariable<VectorTypes.int4> var)
        {
            resType = CUResourceType.Pitch2D;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.pitch2D.devPtr = var.DevicePointer;
            res.pitch2D.format = CUArrayFormat.SignedInt32;
            res.pitch2D.height = var.Height;
            res.pitch2D.numChannels = 4;
            res.pitch2D.pitchInBytes = var.Pitch;
            res.pitch2D.width = var.Width;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaPitchedDeviceVariable<VectorTypes.short4> var)
        {
            resType = CUResourceType.Pitch2D;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.pitch2D.devPtr = var.DevicePointer;
            res.pitch2D.format = CUArrayFormat.SignedInt16;
            res.pitch2D.height = var.Height;
            res.pitch2D.numChannels = 4;
            res.pitch2D.pitchInBytes = var.Pitch;
            res.pitch2D.width = var.Width;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaPitchedDeviceVariable<VectorTypes.char4> var)
        {
            resType = CUResourceType.Pitch2D;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.pitch2D.devPtr = var.DevicePointer;
            res.pitch2D.format = CUArrayFormat.SignedInt8;
            res.pitch2D.height = var.Height;
            res.pitch2D.numChannels = 4;
            res.pitch2D.pitchInBytes = var.Pitch;
            res.pitch2D.width = var.Width;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaPitchedDeviceVariable<VectorTypes.uchar4> var)
        {
            resType = CUResourceType.Pitch2D;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.pitch2D.devPtr = var.DevicePointer;
            res.pitch2D.format = CUArrayFormat.UnsignedInt8;
            res.pitch2D.height = var.Height;
            res.pitch2D.numChannels = 4;
            res.pitch2D.pitchInBytes = var.Pitch;
            res.pitch2D.width = var.Width;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaPitchedDeviceVariable<VectorTypes.ushort4> var)
        {
            resType = CUResourceType.Pitch2D;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.pitch2D.devPtr = var.DevicePointer;
            res.pitch2D.format = CUArrayFormat.UnsignedInt16;
            res.pitch2D.height = var.Height;
            res.pitch2D.numChannels = 4;
            res.pitch2D.pitchInBytes = var.Pitch;
            res.pitch2D.width = var.Width;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="var"></param>
        public CudaResourceDesc(CudaPitchedDeviceVariable<VectorTypes.uint4> var)
        {
            resType = CUResourceType.Pitch2D;
            flags = 0;
            res = new CudaResourceDescUnion();
            res.hArray = new CUarray();
            res.hMipmappedArray = new CUmipmappedArray();
            res.linear = new CudaResourceDescLinear();
            res.pitch2D = new CudaResourceDescPitch2D();
            res.pitch2D.devPtr = var.DevicePointer;
            res.pitch2D.format = CUArrayFormat.UnsignedInt32;
            res.pitch2D.height = var.Height;
            res.pitch2D.numChannels = 4;
            res.pitch2D.pitchInBytes = var.Pitch;
            res.pitch2D.width = var.Width;
        }
        #endregion


        /// <summary>
        /// Resource type
        /// </summary>
        public CUResourceType resType;

        /// <summary>
        /// Mimics the union in C++
        /// </summary>
        public CudaResourceDescUnion res;

        /// <summary>
        /// Flags (must be zero)
        /// </summary>
        public uint flags;
    }


    /// <summary>
    /// Texture descriptor
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaTextureDescriptor
    {
        /// <summary>
        /// Creates a new CudaTextureDescriptor
        /// </summary>
        /// <param name="aAddressMode">Address modes for all dimensions</param>
        /// <param name="aFilterMode">Filter mode</param>
        /// <param name="aFlags">Flags</param>
        public CudaTextureDescriptor(CUAddressMode aAddressMode, CUFilterMode aFilterMode, CUTexRefSetFlags aFlags)
        {
            addressMode = new CUAddressMode[3];
            addressMode[0] = aAddressMode;
            addressMode[1] = aAddressMode;
            addressMode[2] = aAddressMode;

            filterMode = aFilterMode;

            flags = aFlags;
            maxAnisotropy = 0;
            mipmapFilterMode = CUFilterMode.Point;
            mipmapLevelBias = 0;
            minMipmapLevelClamp = 0;
            maxMipmapLevelClamp = 0;
            borderColor = new float[4];
            _reserved = new int[12];
        }

        /// <summary>
        /// Creates a new CudaTextureDescriptor
        /// </summary>
        /// <param name="aAddressMode">Address modes for all dimensions</param>
        /// <param name="aFilterMode">Filter mode</param>
        /// <param name="aFlags">Flags</param>
        /// <param name="aBorderColor">borderColor (array of size 4)</param>
        public CudaTextureDescriptor(CUAddressMode aAddressMode, CUFilterMode aFilterMode, CUTexRefSetFlags aFlags, float[] aBorderColor)
        {
            addressMode = new CUAddressMode[3];
            addressMode[0] = aAddressMode;
            addressMode[1] = aAddressMode;
            addressMode[2] = aAddressMode;

            filterMode = aFilterMode;

            flags = aFlags;
            maxAnisotropy = 0;
            mipmapFilterMode = CUFilterMode.Point;
            mipmapLevelBias = 0;
            minMipmapLevelClamp = 0;
            maxMipmapLevelClamp = 0;
            borderColor = new float[4];
            borderColor[0] = aBorderColor[0];
            borderColor[1] = aBorderColor[1];
            borderColor[2] = aBorderColor[2];
            borderColor[3] = aBorderColor[3];
            _reserved = new int[12];
        }

        /// <summary>
        /// Creates a new CudaTextureDescriptor
        /// </summary>
        /// <param name="aAddressMode0">Address modes for dimension 0</param>
        /// <param name="aAddressMode1">Address modes for dimension 1</param>
        /// <param name="aAddressMode2">Address modes for dimension 2</param>
        /// <param name="aFilterMode">Filter mode</param>
        /// <param name="aFlags">Flags</param>
        public CudaTextureDescriptor(CUAddressMode aAddressMode0, CUAddressMode aAddressMode1, CUAddressMode aAddressMode2, CUFilterMode aFilterMode, CUTexRefSetFlags aFlags)
        {
            addressMode = new CUAddressMode[3];
            addressMode[0] = aAddressMode0;
            addressMode[1] = aAddressMode1;
            addressMode[2] = aAddressMode2;

            filterMode = aFilterMode;

            flags = aFlags;
            maxAnisotropy = 0;
            mipmapFilterMode = CUFilterMode.Point;
            mipmapLevelBias = 0;
            minMipmapLevelClamp = 0;
            maxMipmapLevelClamp = 0;
            borderColor = new float[4];
            _reserved = new int[12];
        }

        /// <summary>
        /// Creates a new CudaTextureDescriptor
        /// </summary>
        /// <param name="aAddressMode0">Address modes for dimension 0</param>
        /// <param name="aAddressMode1">Address modes for dimension 1</param>
        /// <param name="aAddressMode2">Address modes for dimension 2</param>
        /// <param name="aFilterMode">Filter mode</param>
        /// <param name="aFlags">Flags</param>
        /// <param name="aBorderColor">borderColor (array of size 4)</param>
        public CudaTextureDescriptor(CUAddressMode aAddressMode0, CUAddressMode aAddressMode1, CUAddressMode aAddressMode2, CUFilterMode aFilterMode, CUTexRefSetFlags aFlags, float[] aBorderColor)
        {
            addressMode = new CUAddressMode[3];
            addressMode[0] = aAddressMode0;
            addressMode[1] = aAddressMode1;
            addressMode[2] = aAddressMode2;

            filterMode = aFilterMode;

            flags = aFlags;
            maxAnisotropy = 0;
            mipmapFilterMode = CUFilterMode.Point;
            mipmapLevelBias = 0;
            minMipmapLevelClamp = 0;
            maxMipmapLevelClamp = 0;
            borderColor = new float[4];
            borderColor[0] = aBorderColor[0];
            borderColor[1] = aBorderColor[1];
            borderColor[2] = aBorderColor[2];
            borderColor[3] = aBorderColor[3];
            _reserved = new int[12];
        }

        /// <summary>
        /// Creates a new CudaTextureDescriptor
        /// </summary>
        /// <param name="aAddressMode">Address modes for all dimensions</param>
        /// <param name="aFilterMode">Filter mode</param>
        /// <param name="aFlags">Flags</param>
        /// <param name="aMaxAnisotropy">Maximum anisotropy ratio. Specifies the maximum anistropy ratio to be used when doing anisotropic
        /// filtering. This value will be clamped to the range [1,16].</param>
        /// <param name="aMipmapFilterMode">Mipmap filter mode. Specifies the filter mode when the calculated mipmap level lies between
        /// two defined mipmap levels.</param>
        /// <param name="aMipmapLevelBias">Mipmap level bias. Specifies the offset to be applied to the calculated mipmap level.</param>
        /// <param name="aMinMipmapLevelClamp">Mipmap minimum level clamp. Specifies the lower end of the mipmap level range to clamp access to.</param>
        /// <param name="aMaxMipmapLevelClamp">Mipmap maximum level clamp. Specifies the upper end of the mipmap level range to clamp access to.</param>
        public CudaTextureDescriptor(CUAddressMode aAddressMode, CUFilterMode aFilterMode, CUTexRefSetFlags aFlags, uint aMaxAnisotropy, CUFilterMode aMipmapFilterMode,
            float aMipmapLevelBias, float aMinMipmapLevelClamp, float aMaxMipmapLevelClamp)
        {
            addressMode = new CUAddressMode[3];
            addressMode[0] = aAddressMode;
            addressMode[1] = aAddressMode;
            addressMode[2] = aAddressMode;

            filterMode = aFilterMode;

            flags = aFlags;
            maxAnisotropy = aMaxAnisotropy;
            mipmapFilterMode = aMipmapFilterMode;
            mipmapLevelBias = aMipmapLevelBias;
            minMipmapLevelClamp = aMinMipmapLevelClamp;
            maxMipmapLevelClamp = aMaxMipmapLevelClamp;
            borderColor = new float[4];
            _reserved = new int[12];
        }

        /// <summary>
        /// Creates a new CudaTextureDescriptor
        /// </summary>
        /// <param name="aAddressMode">Address modes for all dimensions</param>
        /// <param name="aFilterMode">Filter mode</param>
        /// <param name="aFlags">Flags</param>
        /// <param name="aMaxAnisotropy">Maximum anisotropy ratio. Specifies the maximum anistropy ratio to be used when doing anisotropic
        /// filtering. This value will be clamped to the range [1,16].</param>
        /// <param name="aMipmapFilterMode">Mipmap filter mode. Specifies the filter mode when the calculated mipmap level lies between
        /// two defined mipmap levels.</param>
        /// <param name="aMipmapLevelBias">Mipmap level bias. Specifies the offset to be applied to the calculated mipmap level.</param>
        /// <param name="aMinMipmapLevelClamp">Mipmap minimum level clamp. Specifies the lower end of the mipmap level range to clamp access to.</param>
        /// <param name="aMaxMipmapLevelClamp">Mipmap maximum level clamp. Specifies the upper end of the mipmap level range to clamp access to.</param>
        /// <param name="aBorderColor">borderColor (array of size 4)</param>
        public CudaTextureDescriptor(CUAddressMode aAddressMode, CUFilterMode aFilterMode, CUTexRefSetFlags aFlags, uint aMaxAnisotropy, CUFilterMode aMipmapFilterMode,
            float aMipmapLevelBias, float aMinMipmapLevelClamp, float aMaxMipmapLevelClamp, float[] aBorderColor)
        {
            addressMode = new CUAddressMode[3];
            addressMode[0] = aAddressMode;
            addressMode[1] = aAddressMode;
            addressMode[2] = aAddressMode;

            filterMode = aFilterMode;

            flags = aFlags;
            maxAnisotropy = aMaxAnisotropy;
            mipmapFilterMode = aMipmapFilterMode;
            mipmapLevelBias = aMipmapLevelBias;
            minMipmapLevelClamp = aMinMipmapLevelClamp;
            maxMipmapLevelClamp = aMaxMipmapLevelClamp;
            borderColor = new float[4];
            borderColor[0] = aBorderColor[0];
            borderColor[1] = aBorderColor[1];
            borderColor[2] = aBorderColor[2];
            borderColor[3] = aBorderColor[3];
            _reserved = new int[12];
        }

        /// <summary>
        /// Creates a new CudaTextureDescriptor
        /// </summary>
        /// <param name="aAddressMode0">Address modes for dimension 0</param>
        /// <param name="aAddressMode1">Address modes for dimension 1</param>
        /// <param name="aAddressMode2">Address modes for dimension 2</param>
        /// <param name="aFilterMode">Filter mode</param>
        /// <param name="aFlags">Flags</param>
        /// <param name="aMaxAnisotropy">Maximum anisotropy ratio. Specifies the maximum anistropy ratio to be used when doing anisotropic
        /// filtering. This value will be clamped to the range [1,16].</param>
        /// <param name="aMipmapFilterMode">Mipmap filter mode. Specifies the filter mode when the calculated mipmap level lies between
        /// two defined mipmap levels.</param>
        /// <param name="aMipmapLevelBias">Mipmap level bias. Specifies the offset to be applied to the calculated mipmap level.</param>
        /// <param name="aMinMipmapLevelClamp">Mipmap minimum level clamp. Specifies the lower end of the mipmap level range to clamp access to.</param>
        /// <param name="aMaxMipmapLevelClamp">Mipmap maximum level clamp. Specifies the upper end of the mipmap level range to clamp access to.</param>
        public CudaTextureDescriptor(CUAddressMode aAddressMode0, CUAddressMode aAddressMode1, CUAddressMode aAddressMode2, CUFilterMode aFilterMode, CUTexRefSetFlags aFlags, uint aMaxAnisotropy, CUFilterMode aMipmapFilterMode,
            float aMipmapLevelBias, float aMinMipmapLevelClamp, float aMaxMipmapLevelClamp)
        {
            addressMode = new CUAddressMode[3];
            addressMode[0] = aAddressMode0;
            addressMode[1] = aAddressMode1;
            addressMode[2] = aAddressMode2;

            filterMode = aFilterMode;

            flags = aFlags;
            maxAnisotropy = aMaxAnisotropy;
            mipmapFilterMode = aMipmapFilterMode;
            mipmapLevelBias = aMipmapLevelBias;
            minMipmapLevelClamp = aMinMipmapLevelClamp;
            maxMipmapLevelClamp = aMaxMipmapLevelClamp;
            borderColor = new float[4];
            _reserved = new int[12];
        }

        /// <summary>
        /// Creates a new CudaTextureDescriptor
        /// </summary>
        /// <param name="aAddressMode0">Address modes for dimension 0</param>
        /// <param name="aAddressMode1">Address modes for dimension 1</param>
        /// <param name="aAddressMode2">Address modes for dimension 2</param>
        /// <param name="aFilterMode">Filter mode</param>
        /// <param name="aFlags">Flags</param>
        /// <param name="aMaxAnisotropy">Maximum anisotropy ratio. Specifies the maximum anistropy ratio to be used when doing anisotropic
        /// filtering. This value will be clamped to the range [1,16].</param>
        /// <param name="aMipmapFilterMode">Mipmap filter mode. Specifies the filter mode when the calculated mipmap level lies between
        /// two defined mipmap levels.</param>
        /// <param name="aMipmapLevelBias">Mipmap level bias. Specifies the offset to be applied to the calculated mipmap level.</param>
        /// <param name="aMinMipmapLevelClamp">Mipmap minimum level clamp. Specifies the lower end of the mipmap level range to clamp access to.</param>
        /// <param name="aMaxMipmapLevelClamp">Mipmap maximum level clamp. Specifies the upper end of the mipmap level range to clamp access to.</param>
        /// <param name="aBorderColor">borderColor (array of size 4)</param>
        public CudaTextureDescriptor(CUAddressMode aAddressMode0, CUAddressMode aAddressMode1, CUAddressMode aAddressMode2, CUFilterMode aFilterMode, CUTexRefSetFlags aFlags, uint aMaxAnisotropy, CUFilterMode aMipmapFilterMode,
            float aMipmapLevelBias, float aMinMipmapLevelClamp, float aMaxMipmapLevelClamp, float[] aBorderColor)
        {
            addressMode = new CUAddressMode[3];
            addressMode[0] = aAddressMode0;
            addressMode[1] = aAddressMode1;
            addressMode[2] = aAddressMode2;

            filterMode = aFilterMode;

            flags = aFlags;
            maxAnisotropy = aMaxAnisotropy;
            mipmapFilterMode = aMipmapFilterMode;
            mipmapLevelBias = aMipmapLevelBias;
            minMipmapLevelClamp = aMinMipmapLevelClamp;
            maxMipmapLevelClamp = aMaxMipmapLevelClamp;
            borderColor = new float[4];
            borderColor[0] = aBorderColor[0];
            borderColor[1] = aBorderColor[1];
            borderColor[2] = aBorderColor[2];
            borderColor[3] = aBorderColor[3];
            _reserved = new int[12];
        }

        /// <summary>
        /// Address modes
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3, ArraySubType = UnmanagedType.I4)]
        public CUAddressMode[] addressMode;
        /// <summary>
        /// Filter mode
        /// </summary>
        public CUFilterMode filterMode;
        /// <summary>
        /// Flags
        /// </summary>
        public CUTexRefSetFlags flags;
        /// <summary>
        /// Maximum anisotropy ratio. Specifies the maximum anistropy ratio to be used when doing anisotropic
        /// filtering. This value will be clamped to the range [1,16].
        /// </summary>
        public uint maxAnisotropy;
        /// <summary>
        /// Mipmap filter mode. Specifies the filter mode when the calculated mipmap level lies between
        /// two defined mipmap levels.
        /// </summary>
        public CUFilterMode mipmapFilterMode;
        /// <summary>
        /// Mipmap level bias. Specifies the offset to be applied to the calculated mipmap level.
        /// </summary>
        public float mipmapLevelBias;
        /// <summary>
        /// Mipmap minimum level clamp. Specifies the lower end of the mipmap level range to clamp access to.
        /// </summary>
        public float minMipmapLevelClamp;
        /// <summary>
        /// Mipmap maximum level clamp. Specifies the upper end of the mipmap level range to clamp access to.
        /// </summary>
        public float maxMipmapLevelClamp;

        /// <summary>
        /// Border Color
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4, ArraySubType = UnmanagedType.R4)]
        public float[] borderColor;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 12, ArraySubType = UnmanagedType.I4)]
        private int[] _reserved;
    }

    /// <summary>
    /// Resource view descriptor
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaResourceViewDesc
    {
        /// <summary>
        /// Resource view format
        /// </summary>
        public CUresourceViewFormat format;
        /// <summary>
        /// Width of the resource view
        /// </summary>
        public SizeT width;
        /// <summary>
        /// Height of the resource view
        /// </summary>
        public SizeT height;
        /// <summary>
        /// Depth of the resource view
        /// </summary>
        public SizeT depth;
        /// <summary>
        /// First defined mipmap level
        /// </summary>
        public uint firstMipmapLevel;
        /// <summary>
        /// Last defined mipmap level
        /// </summary>
        public uint lastMipmapLevel;
        /// <summary>
        /// First layer index
        /// </summary>
        public uint firstLayer;
        /// <summary>
        /// Last layer index
        /// </summary>
        public uint lastLayer;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16, ArraySubType = UnmanagedType.I4)]
        private int[] _reserved;
    }

    /// <summary>
    /// GPU Direct v3 tokens
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaPointerAttributeP2PTokens
    {
        /// <summary>
        /// 
        /// </summary>
        ulong p2pToken;
        /// <summary>
        /// 
        /// </summary>
        uint vaSpaceToken;
    }


    /// <summary>
    /// Per-operation parameters for ::cuStreamBatchMemOp
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct cuuint3264_union
    {
        /// <summary/>
        [FieldOffset(0)]
        public uint value;
        /// <summary/>
        [FieldOffset(0)]
        public ulong value64;
    }

    /// <summary/>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUstreamMemOpWaitValueParams
    {
        /// <summary/>
        public CUstreamBatchMemOpType operation;
        /// <summary/>
        public CUdeviceptr address;
        /// <summary/>
        public cuuint3264_union value;
        /// <summary/>
        public uint flags;
        /// <summary>
        /// For driver internal use. Initial value is unimportant.
        /// </summary>
        public CUdeviceptr alias;
    }

    /// <summary/>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUstreamMemOpWriteValueParams
    {
        /// <summary/>
        public CUstreamBatchMemOpType operation;
        /// <summary/>
        public CUdeviceptr address;
        /// <summary/>
        public cuuint3264_union value;
        /// <summary/>
        public uint flags;
        /// <summary>
        /// For driver internal use. Initial value is unimportant.
        /// </summary>
        public CUdeviceptr alias;
    }

    /// <summary/>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUstreamMemOpFlushRemoteWritesParams
    {
        /// <summary/>
        public CUstreamBatchMemOpType operation;
        /// <summary/>
        public uint flags;
    }

    /// <summary/>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUstreamMemOpMemoryBarrierParams
    {
        /// <summary/>
        public CUstreamBatchMemOpType operation;
        /// <summary/>
        public uint flags;
    }

    /// <summary/>
    [StructLayout(LayoutKind.Explicit)]
    public struct CUstreamBatchMemOpParams
    {
        /// <summary/>
        [FieldOffset(0)]
        public CUstreamBatchMemOpType operation;
        /// <summary/>
        [FieldOffset(0)]
        public CUstreamMemOpWaitValueParams waitValue;
        /// <summary/>
        [FieldOffset(0)]
        public CUstreamMemOpWriteValueParams writeValue;
        /// <summary/>
        [FieldOffset(0)]
        public CUstreamMemOpFlushRemoteWritesParams flushRemoteWrites;
        /// <summary/>
        [FieldOffset(0)]
        public CUstreamMemOpMemoryBarrierParams memoryBarrier;
        //In cuda header, an ulong[6] fixes the union size to 48 bytes, we
        //achieve the same in C# if we set at offset 40 an simple ulong
        [FieldOffset(5 * 8)]
        ulong pad;
    }

    /// <summary>
    /// CudaBatchMemOpNodeParams
    /// </summary>
    public class CudaBatchMemOpNodeParams
    {
        /// <summary/>
        public CUcontext ctx;
        /// <summary/>
        public CUstreamBatchMemOpParams[] paramArray;
        /// <summary/>
        public uint flags;
    }

    /// <summary>
    /// CudaBatchMemOpNodeParams (V1 and V2)
    /// </summary>
    internal struct CudaBatchMemOpNodeParamsInternal
    {
        /// <summary/>
        public CUcontext ctx;
        /// <summary/>
        public uint count;
        /// <summary/>
        public IntPtr paramArray;
        /// <summary/>
        public uint flags;
    }

    /// <summary>
    /// Kernel launch parameters
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaLaunchParams
    {
        /// <summary>
        /// Kernel to launch
        /// </summary>
        public CUfunction function;
        /// <summary>
        /// Width of grid in blocks
        /// </summary>
        public uint gridDimX;
        /// <summary>
        /// Height of grid in blocks
        /// </summary>
        public uint gridDimY;
        /// <summary>
        /// Depth of grid in blocks
        /// </summary>
        public uint gridDimZ;
        /// <summary>
        /// X dimension of each thread block
        /// </summary>
        public uint blockDimX;
        /// <summary>
        /// Y dimension of each thread block
        /// </summary>
        public uint blockDimY;
        /// <summary>
        /// Z dimension of each thread block
        /// </summary>
        public uint blockDimZ;
        /// <summary>
        /// Dynamic shared-memory size per thread block in bytes
        /// </summary>
        public uint sharedMemBytes;
        /// <summary>
        /// Stream identifier
        /// </summary>
        public CUstream hStream;
        /// <summary>
        /// Array of pointers to kernel parameters
        /// </summary>
        public IntPtr kernelParams;
    }



    /// <summary>
    /// GPU kernel node parameters (V2 and V3)
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaKernelNodeParams
    {
        /// <summary>
        /// Kernel to launch
        /// </summary>
        public CUfunction func;
        /// <summary>
        /// Width of grid in blocks
        /// </summary>
        public uint gridDimX;
        /// <summary>
        /// Height of grid in blocks
        /// </summary>
        public uint gridDimY;
        /// <summary>
        /// Depth of grid in blocks
        /// </summary>
        public uint gridDimZ;
        /// <summary>
        /// X dimension of each thread block
        /// </summary>
        public uint blockDimX;
        /// <summary>
        /// Y dimension of each thread block
        /// </summary>
        public uint blockDimY;
        /// <summary>
        /// Z dimension of each thread block
        /// </summary>
        public uint blockDimZ;
        /// <summary>
        /// Dynamic shared-memory size per thread block in bytes
        /// </summary>
        public uint sharedMemBytes;
        /// <summary>
        /// Array of pointers to kernel parameters
        /// </summary>
        public IntPtr kernelParams;
        /// <summary>
        /// Extra options
        /// </summary>
        public IntPtr extra;
        /// <summary>
        /// Kernel to launch, will only be referenced if func is NULL
        /// </summary>
        CUkernel kern;
        /// <summary>
        /// Context for the kernel task to run in. The value NULL will indicate the current context should be used by the api. This field is ignored if func is set.
        /// </summary>
        CUcontext ctx;
    }


    /// <summary>
    /// Memset node parameters (V1)
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaMemsetNodeParams
    {
        /// <summary>
        /// Destination device pointer
        /// </summary>
        public CUdeviceptr dst;
        /// <summary>
        /// Pitch of destination device pointer. Unused if height is 1
        /// </summary>
        public SizeT pitch;
        /// <summary>
        /// Value to be set
        /// </summary>
        public uint value;
        /// <summary>
        /// Size of each element in bytes. Must be 1, 2, or 4.
        /// </summary>
        public uint elementSize;
        /// <summary>
        /// Width of the row in elements
        /// </summary>
        public SizeT width;
        /// <summary>
        /// Number of rows
        /// </summary>
        public SizeT height;

        /// <summary>
        /// Initialises the struct
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="deviceVariable"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static CudaMemsetNodeParams init<T>(CudaDeviceVariable<T> deviceVariable, uint value) where T : struct
        {
            CudaMemsetNodeParams para = new CudaMemsetNodeParams();
            para.dst = deviceVariable.DevicePointer;
            para.pitch = deviceVariable.SizeInBytes;
            para.value = value;
            para.elementSize = deviceVariable.TypeSize;
            para.width = deviceVariable.SizeInBytes;
            para.height = 1;

            return para;
        }
        /// <summary>
        /// Initialises the struct
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="deviceVariable"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static CudaMemsetNodeParams init<T>(CudaPitchedDeviceVariable<T> deviceVariable, uint value) where T : struct
        {
            CudaMemsetNodeParams para = new CudaMemsetNodeParams();
            para.dst = deviceVariable.DevicePointer;
            para.pitch = deviceVariable.Pitch;
            para.value = value;
            para.elementSize = deviceVariable.TypeSize;
            para.width = deviceVariable.WidthInBytes;
            para.height = deviceVariable.Height;

            return para;
        }
    }


    /// <summary>
    /// Memset node parameters (V2)
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaMemsetNodeParamsV2
    {
        /// <summary>
        /// Destination device pointer
        /// </summary>
        public CUdeviceptr dst;
        /// <summary>
        /// Pitch of destination device pointer. Unused if height is 1
        /// </summary>
        public SizeT pitch;
        /// <summary>
        /// Value to be set
        /// </summary>
        public uint value;
        /// <summary>
        /// Size of each element in bytes. Must be 1, 2, or 4.
        /// </summary>
        public uint elementSize;
        /// <summary>
        /// Width of the row in elements
        /// </summary>
        public SizeT width;
        /// <summary>
        /// Number of rows
        /// </summary>
        public SizeT height;
        /// <summary>
        /// Context on which to run the node
        /// </summary>
        public CUcontext ctx;

        /// <summary>
        /// Initialises the struct
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="deviceVariable"></param>
        /// <param name="value"></param>
        /// <param name="ctx"></param>
        /// <returns></returns>
        public static CudaMemsetNodeParamsV2 init<T>(CudaDeviceVariable<T> deviceVariable, uint value, CudaContext ctx) where T : struct
        {
            CudaMemsetNodeParamsV2 para = new CudaMemsetNodeParamsV2();
            para.dst = deviceVariable.DevicePointer;
            para.pitch = deviceVariable.SizeInBytes;
            para.value = value;
            para.elementSize = deviceVariable.TypeSize;
            para.width = deviceVariable.SizeInBytes;
            para.height = 1;
            para.ctx = ctx.Context;

            return para;
        }
        /// <summary>
        /// Initialises the struct
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="deviceVariable"></param>
        /// <param name="value"></param>
        /// <param name="ctx"></param>
        /// <returns></returns>
        public static CudaMemsetNodeParamsV2 init<T>(CudaPitchedDeviceVariable<T> deviceVariable, uint value, CudaContext ctx) where T : struct
        {
            CudaMemsetNodeParamsV2 para = new CudaMemsetNodeParamsV2();
            para.dst = deviceVariable.DevicePointer;
            para.pitch = deviceVariable.Pitch;
            para.value = value;
            para.elementSize = deviceVariable.TypeSize;
            para.width = deviceVariable.WidthInBytes;
            para.height = deviceVariable.Height;
            para.ctx = ctx.Context;

            return para;
        }
    }


    /// <summary>
    /// Memcpy node parameters
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaMemcpyNodeParams
    {
        /// <summary>
        /// Must be zero
        /// </summary>
        int flags;
        /// <summary>
        /// Must be zero
        /// </summary>
        int reserved;
        /// <summary>
        /// Context on which to run the node
        /// </summary>
        CUcontext copyCtx;
        /// <summary>
        /// Parameters for the memory copy
        /// </summary>
        CUDAMemCpy3D copyParams;

        /// <summary>
        /// Initialises the struct for copy device memory to device memory
        /// </summary>
        public static CudaMemcpyNodeParams init<T>(CudaDeviceVariable<T> src, CudaDeviceVariable<T> dst, CudaContext ctx) where T : struct
        {
            CudaMemcpyNodeParams para = new CudaMemcpyNodeParams();
            para.copyParams.srcDevice = src.DevicePointer;
            para.copyParams.srcMemoryType = CUMemoryType.Device;
            para.copyParams.srcPitch = 0;
            para.copyParams.dstDevice = dst.DevicePointer;
            para.copyParams.dstMemoryType = CUMemoryType.Device;
            para.copyParams.dstPitch = 0;
            para.copyParams.Height = 0;
            para.copyParams.WidthInBytes = dst.SizeInBytes;

            para.copyCtx = ctx.Context;

            return para;
        }

        /// <summary>
        /// Initialises the struct for copy array3d to device memory
        /// </summary>
        public static CudaMemcpyNodeParams init<T>(CudaDeviceVariable<T> src, CudaArray3D dst, CudaContext ctx) where T : struct
        {
            CudaMemcpyNodeParams para = new CudaMemcpyNodeParams();
            para.copyParams.srcDevice = src.DevicePointer;
            para.copyParams.srcMemoryType = CUMemoryType.Device;
            para.copyParams.srcPitch = 0;
            para.copyParams.dstArray = dst.CUArray;
            para.copyParams.dstMemoryType = CUMemoryType.Array;
            para.copyParams.Depth = dst.Depth;
            para.copyParams.Height = dst.Height;
            para.copyParams.WidthInBytes = dst.WidthInBytes;

            para.copyCtx = ctx.Context;

            return para;
        }

        /// <summary>
        /// Initialises the struct for copy array3d to device memory
        /// </summary>
        public static CudaMemcpyNodeParams init<T>(CudaArray3D src, CudaDeviceVariable<T> dst, CudaContext ctx) where T : struct
        {
            CudaMemcpyNodeParams para = new CudaMemcpyNodeParams();
            para.copyParams.srcArray = src.CUArray;
            para.copyParams.srcMemoryType = CUMemoryType.Array;
            para.copyParams.dstPitch = 0;
            para.copyParams.dstDevice = dst.DevicePointer;
            para.copyParams.dstMemoryType = CUMemoryType.Device;
            para.copyParams.Depth = src.Depth;
            para.copyParams.Height = src.Height;
            para.copyParams.WidthInBytes = src.WidthInBytes;

            para.copyCtx = ctx.Context;

            return para;
        }

        /// <summary>
        /// Initialises the struct for copy array3d to device memory
        /// </summary>
        public static CudaMemcpyNodeParams init<T>(CudaPitchedDeviceVariable<T> src, CudaArray3D dst, CudaContext ctx) where T : struct
        {
            CudaMemcpyNodeParams para = new CudaMemcpyNodeParams();
            para.copyParams.srcDevice = src.DevicePointer;
            para.copyParams.srcMemoryType = CUMemoryType.Device;
            para.copyParams.srcPitch = src.Pitch;
            para.copyParams.dstArray = dst.CUArray;
            para.copyParams.dstMemoryType = CUMemoryType.Array;
            para.copyParams.Depth = dst.Depth;
            para.copyParams.Height = dst.Height;
            para.copyParams.WidthInBytes = dst.WidthInBytes;

            para.copyCtx = ctx.Context;

            return para;
        }

        /// <summary>
        /// Initialises the struct for copy array3d to device memory
        /// </summary>
        public static CudaMemcpyNodeParams init<T>(CudaArray3D src, CudaPitchedDeviceVariable<T> dst, CudaContext ctx) where T : struct
        {
            CudaMemcpyNodeParams para = new CudaMemcpyNodeParams();
            para.copyParams.srcArray = src.CUArray;
            para.copyParams.srcMemoryType = CUMemoryType.Array;
            para.copyParams.dstPitch = dst.Pitch;
            para.copyParams.dstDevice = dst.DevicePointer;
            para.copyParams.dstMemoryType = CUMemoryType.Device;
            para.copyParams.Depth = src.Depth;
            para.copyParams.Height = src.Height;
            para.copyParams.WidthInBytes = src.WidthInBytes;

            para.copyCtx = ctx.Context;

            return para;
        }

        /// <summary>
        /// Initialises the struct for copy array3d to host memory
        /// </summary>
        public static CudaMemcpyNodeParams init(IntPtr src, CudaArray3D dst, CudaContext ctx)
        {
            CudaMemcpyNodeParams para = new CudaMemcpyNodeParams();
            para.copyParams.srcHost = src;
            para.copyParams.srcMemoryType = CUMemoryType.Host;
            para.copyParams.srcPitch = 0;
            para.copyParams.dstArray = dst.CUArray;
            para.copyParams.dstMemoryType = CUMemoryType.Array;
            para.copyParams.Depth = dst.Depth;
            para.copyParams.Height = dst.Height;
            para.copyParams.WidthInBytes = dst.WidthInBytes;

            para.copyCtx = ctx.Context;

            return para;
        }

        /// <summary>
        /// Initialises the struct for copy array3d to host memory
        /// </summary>
        public static CudaMemcpyNodeParams init(CudaArray3D src, IntPtr dst, CudaContext ctx)
        {
            CudaMemcpyNodeParams para = new CudaMemcpyNodeParams();
            para.copyParams.srcArray = src.CUArray;
            para.copyParams.srcMemoryType = CUMemoryType.Array;
            para.copyParams.dstPitch = 0;
            para.copyParams.dstHost = dst;
            para.copyParams.dstMemoryType = CUMemoryType.Host;
            para.copyParams.Depth = src.Depth;
            para.copyParams.Height = src.Height;
            para.copyParams.WidthInBytes = src.WidthInBytes;

            para.copyCtx = ctx.Context;

            return para;
        }

        /// <summary>
        /// Initialises the struct for copy device memory to device memory
        /// </summary>
        public static CudaMemcpyNodeParams init<T>(CudaPitchedDeviceVariable<T> src, CudaPitchedDeviceVariable<T> dst, CudaContext ctx) where T : struct
        {
            CudaMemcpyNodeParams para = new CudaMemcpyNodeParams();
            para.copyParams.srcDevice = src.DevicePointer;
            para.copyParams.srcMemoryType = CUMemoryType.Device;
            para.copyParams.srcPitch = src.Pitch;
            para.copyParams.dstDevice = dst.DevicePointer;
            para.copyParams.dstMemoryType = CUMemoryType.Device;
            para.copyParams.dstPitch = dst.Pitch;
            para.copyParams.Height = dst.Height;
            para.copyParams.WidthInBytes = dst.Width * dst.TypeSize;

            para.copyCtx = ctx.Context;

            return para;
        }

        /// <summary>
        /// Initialises the struct for copy host memory to device memory
        /// </summary>
        public static CudaMemcpyNodeParams init<T>(IntPtr src, CudaPitchedDeviceVariable<T> dst, CudaContext ctx) where T : struct
        {
            CudaMemcpyNodeParams para = new CudaMemcpyNodeParams();
            para.copyParams.srcHost = src;
            para.copyParams.srcMemoryType = CUMemoryType.Host;
            para.copyParams.srcPitch = 0;
            para.copyParams.dstDevice = dst.DevicePointer;
            para.copyParams.dstMemoryType = CUMemoryType.Device;
            para.copyParams.dstPitch = dst.Pitch;
            para.copyParams.Height = dst.Height;
            para.copyParams.WidthInBytes = dst.Width * dst.TypeSize;

            para.copyCtx = ctx.Context;

            return para;
        }

        /// <summary>
        /// Initialises the struct for copy device memory to host memory
        /// </summary>
        public static CudaMemcpyNodeParams init<T>(CudaPitchedDeviceVariable<T> src, IntPtr dst, CudaContext ctx) where T : struct
        {
            CudaMemcpyNodeParams para = new CudaMemcpyNodeParams();
            para.copyParams.srcDevice = src.DevicePointer;
            para.copyParams.srcMemoryType = CUMemoryType.Device;
            para.copyParams.srcPitch = src.Pitch;
            para.copyParams.dstHost = dst;
            para.copyParams.dstMemoryType = CUMemoryType.Host;
            para.copyParams.dstPitch = 0;
            para.copyParams.Height = src.Height;
            para.copyParams.WidthInBytes = src.Width * src.TypeSize;

            para.copyCtx = ctx.Context;

            return para;
        }

        /// <summary>
        /// Initialises the struct for copy host memory to device memory
        /// </summary>
        public static CudaMemcpyNodeParams init<T>(IntPtr src, CudaDeviceVariable<T> dst, CudaContext ctx) where T : struct
        {
            CudaMemcpyNodeParams para = new CudaMemcpyNodeParams();
            para.copyParams.srcHost = src;
            para.copyParams.srcMemoryType = CUMemoryType.Host;
            para.copyParams.srcPitch = 0;
            para.copyParams.dstDevice = dst.DevicePointer;
            para.copyParams.dstMemoryType = CUMemoryType.Device;
            para.copyParams.dstPitch = 0;
            para.copyParams.Height = 0;
            para.copyParams.WidthInBytes = dst.SizeInBytes;

            para.copyCtx = ctx.Context;

            return para;
        }

        /// <summary>
        /// Initialises the struct for copy device memory to host memory
        /// </summary>
        public static CudaMemcpyNodeParams init<T>(CudaDeviceVariable<T> src, IntPtr dst, CudaContext ctx) where T : struct
        {
            CudaMemcpyNodeParams para = new CudaMemcpyNodeParams();
            para.copyParams.srcDevice = src.DevicePointer;
            para.copyParams.srcMemoryType = CUMemoryType.Device;
            para.copyParams.srcPitch = 0;
            para.copyParams.dstHost = dst;
            para.copyParams.dstMemoryType = CUMemoryType.Host;
            para.copyParams.dstPitch = 0;
            para.copyParams.Height = 0;
            para.copyParams.WidthInBytes = src.SizeInBytes;

            para.copyCtx = ctx.Context;

            return para;
        }
    }


    /// <summary>
    /// Host node parameters (V1 and V2)
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaHostNodeParams
    {
        /// <summary>
        /// The function to call when the node executes
        /// </summary>
        public CUhostFn fn;
        /// <summary>
        /// Argument to pass to the function
        /// </summary>
        public IntPtr userData;
    }

    /// <summary>
    /// Host node parameters (V1 and V2)
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct CudaHostNodeParamsInternal
    {
        /// <summary>
        /// The function to call when the node executes
        /// </summary>
        public IntPtr fn;
        /// <summary>
        /// Argument to pass to the function
        /// </summary>
        public IntPtr userData;
    }

    /// <summary>
    ///  Win32 handle referencing the semaphore object. Valid when
    ///  type is one of the following: <para/>
    ///  - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32<para/>
    ///  - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT<para/>
    ///  - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP<para/>
    ///  - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE<para/>
    ///  Exactly one of 'handle' and 'name' must be non-NULL. If
    ///  type is 
    ///  ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT
    ///   then 'name' must be NULL.
    /// </summary>
    [StructLayout(LayoutKind.Explicit, CharSet = CharSet.Unicode)]
    public struct Win32Handle
    {
        /// <summary>
        /// Valid NT handle. Must be NULL if 'name' is non-NULL
        /// </summary>
        [FieldOffset(0)]
        public IntPtr handle;
        /// <summary>
        /// Name of a valid memory object. Must be NULL if 'handle' is non-NULL.
        /// </summary>
        [FieldOffset(8)]
        [MarshalAs(UnmanagedType.LPStr)]
        public string name;
    }

    /// <summary>
    /// </summary>
    [StructLayout(LayoutKind.Explicit, CharSet = CharSet.Unicode)]
    public struct HandleUnion
    {
        /// <summary>
        /// File descriptor referencing the memory object. Valid when type is CUDA_EXTERNAL_MEMORY_DEDICATED
        /// </summary>
        [FieldOffset(0)]
        public int fd;

        /// <summary>
        /// Win32 handle referencing the semaphore object.
        /// </summary>
        [FieldOffset(0)]
        public Win32Handle win32;

        /// <summary>
        /// A handle representing an NvSciBuf Object.Valid when type
        /// is ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF
        /// </summary>
        [FieldOffset(0)]
        public IntPtr nvSciBufObject;
    }

    /// <summary>
    /// External memory handle descriptor
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CudaExternalMemoryHandleDesc
    {
        /// <summary>
        /// Type of the handle
        /// </summary>
        [FieldOffset(0)]
        public CUexternalMemoryHandleType type;

        /// <summary>
        /// 
        /// </summary>
        [FieldOffset(8)]
        public HandleUnion handle;

        /// <summary>
        /// Size of the memory allocation
        /// </summary>
        [FieldOffset(24)]
        public ulong size;

        /// <summary>
        /// Flags must either be zero or ::CUDA_EXTERNAL_MEMORY_DEDICATED
        /// </summary>
        [FieldOffset(32)]
        public CudaExternalMemory flags;

        //Original struct definition in cuda-header sets a unsigned int[16] array at the end of the struct.
        //To get the same struct size (104 bytes), we simply put an uint at FieldOffset 100.
        [FieldOffset(100)]
        private uint reserved;
    }


    /// <summary>
    /// External semaphore handle descriptor
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CudaExternalSemaphoreHandleDesc
    {
        /// <summary>
        /// Type of the handle
        /// </summary>
        [FieldOffset(0)]
        public CUexternalSemaphoreHandleType type;

        /// <summary>
        /// 
        /// </summary>
        [FieldOffset(8)]
        public HandleUnion handle;
        /// <summary>
        /// Flags reserved for the future. Must be zero.
        /// </summary>
        [FieldOffset(32)]
        public uint flags;

        //Original struct definition in cuda-header sets a unsigned int[16] array at the end of the struct.
        //To get the same struct size (96 bytes), we simply put an uint at FieldOffset 92.
        [FieldOffset(92)]
        private uint reserved;
    }


    /// <summary>
    /// External memory buffer descriptor
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaExternalMemoryBufferDesc
    {
        /// <summary>
        /// Offset into the memory object where the buffer's base is
        /// </summary>
        //[FieldOffset(0)]
        public ulong offset;
        /// <summary>
        /// Size of the buffer
        /// </summary>
        //[FieldOffset(8)]
        public ulong size;
        /// <summary>
        /// Flags reserved for future use. Must be zero.
        /// </summary>
        //[FieldOffset(16)]
        public uint flags;

        //[FieldOffset(84)] //instead of uint[16]
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16, ArraySubType = UnmanagedType.U4)]
        private uint[] reserved;
    }


    /// <summary>
    /// External memory mipmap descriptor
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaExternalMemoryMipmappedArrayDesc
    {
        /// <summary>
        /// Offset into the memory object where the base level of the mipmap chain is.
        /// </summary>
        public ulong offset;
        /// <summary>
        /// Format, dimension and type of base level of the mipmap chain
        /// </summary>
        public CUDAArray3DDescriptor arrayDesc;
        /// <summary>
        /// Total number of levels in the mipmap chain
        /// </summary>
        public uint numLevels;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16, ArraySubType = UnmanagedType.U4)]
        private uint[] reserved;
    }


    /// <summary>
    /// External semaphore signal parameters
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CudaExternalSemaphoreSignalParams
    {
        /// <summary>
        /// Parameters for fence objects
        /// </summary>
        [StructLayout(LayoutKind.Explicit)]
        public struct Parameters
        {
            /// <summary>
            /// Value of fence to be signaled
            /// </summary>
            [StructLayout(LayoutKind.Sequential)]
            public struct Fence
            {
                /// <summary>
                /// Value of fence to be signaled
                /// </summary>
                public ulong value;
            }
            /// <summary/>
            [FieldOffset(0)]
            public Fence fence;
            /// <summary>
            /// Pointer to NvSciSyncFence. Valid if CUexternalSemaphoreHandleType
            /// is of type CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC.
            /// </summary>
            [StructLayout(LayoutKind.Sequential)]
            public struct NvSciSync
            {
                /// <summary>
                /// 
                /// </summary>
                public IntPtr fence;
            }
            /// <summary/>
            [FieldOffset(8)]
            public NvSciSync nvSciSync;

            /// <summary>
            /// Parameters for keyed mutex objects
            /// </summary>
            [StructLayout(LayoutKind.Sequential)]
            public struct KeyedMutex
            {
                /// <summary>
                /// Value of key to acquire the mutex with
                /// </summary>
                public ulong key;
            }
            /// <summary/>
            [FieldOffset(16)]
            public KeyedMutex keyedMutex;

            /// <summary/>
            [FieldOffset(68)] //params.reserved[9];
            private uint reserved;
        }
        /// <summary/>
        [FieldOffset(0)]
        public Parameters parameters;
        /// <summary>
        /// Flags reserved for the future. Must be zero.
        /// </summary>
        [FieldOffset(72)]
        public uint flags;
        [FieldOffset(136)] //offset of reserved[15]
        uint reserved;
    }


    /// <summary>
    /// External semaphore wait parameters
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CudaExternalSemaphoreWaitParams
    {
        /// <summary>
        /// Parameters for fence objects
        /// </summary>
        [StructLayout(LayoutKind.Explicit)]
        public struct Parameters
        {
            /// <summary>
            /// Value of fence to be waited on
            /// </summary>
            [StructLayout(LayoutKind.Sequential)]
            public struct Fence
            {
                /// <summary>
                /// Value of fence to be waited on
                /// </summary>
                public ulong value;
            }
            /// <summary/>
            [FieldOffset(0)]
            public Fence fence;
            /// <summary>
            /// Pointer to NvSciSyncFence. Valid if CUexternalSemaphoreHandleType
            /// is of type CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC.
            /// </summary>
            [StructLayout(LayoutKind.Sequential)]
            public struct NvSciSync
            {
                /// <summary>
                /// 
                /// </summary>
                public IntPtr fence;
            }
            /// <summary/>
            [FieldOffset(8)]
            public NvSciSync nvSciSync;
            /// <summary>
            /// Parameters for keyed mutex objects
            /// </summary>
            [StructLayout(LayoutKind.Sequential)]
            public struct KeyedMutex
            {
                /// <summary>
                /// Value of key to acquire the mutex with
                /// </summary>
                public ulong key;
                /// <summary>
                /// Timeout in milliseconds to wait to acquire the mutex
                /// </summary>
                public uint timeoutMs;
            }
            /// <summary/>
            [FieldOffset(16)]
            public KeyedMutex keyedMutex;
            [FieldOffset(20)]
            private uint reserved;
        }
        /// <summary/>
        [FieldOffset(0)]
        public Parameters parameters;
        /// <summary>
        /// Flags reserved for the future. Must be zero.
        /// </summary>
        [FieldOffset(72)]
        public uint flags;
        [FieldOffset(136)]
        uint reserved;
    }

    /// <summary>
    /// Specifies a location for an allocation.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUmemLocation
    {
        /// <summary>
        /// Specifies the location type, which modifies the meaning of id.
        /// </summary>
        public CUmemLocationType type;
        /// <summary>
        /// identifier for a given this location's ::CUmemLocationType.
        /// </summary>
        public int id;
    }

    /// <summary>
    /// Allocation hint for requesting compressible memory.
    /// On devices that support Compute Data Compression, compressible
    /// memory can be used to accelerate accesses to data with unstructured
    /// sparsity and other compressible data patterns.Applications are
    /// expected to query allocation property of the handle obtained with
    /// ::cuMemCreate using ::cuMemGetAllocationPropertiesFromHandle to
    /// validate if the obtained allocation is compressible or not.Note that
    /// compressed memory may not be mappable on all devices.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct allocFlags
    {
        /// <summary/>
        public CUmemAllocationCompType compressionType;
        /// <summary/>
        public byte gpuDirectRDMACapable;
        /// <summary>
        /// Bitmask indicating intended usage for this allocation
        /// </summary>
        public CUmemCreateUsage usage;
        byte reserved0;
        byte reserved1;
        byte reserved2;
        byte reserved3;
    }

    /// <summary>
    /// Specifies the allocation properties for a allocation.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUmemAllocationProp
    {
        /// <summary>
        /// Allocation type
        /// </summary>
        public CUmemAllocationType type;
        /// <summary>
        /// requested ::CUmemAllocationHandleType
        /// </summary>
        public CUmemAllocationHandleType requestedHandleTypes;
        /// <summary>
        /// Location of allocation
        /// </summary>
        public CUmemLocation location;
        /// <summary>
        /// Windows-specific POBJECT_ATTRIBUTES required when
        /// ::CU_MEM_HANDLE_TYPE_WIN32 is specified.This object attributes structure
        /// includes security attributes that define
        /// the scope of which exported allocations may be transferred to other
        /// processes. In all other cases, this field is required to be zero.
        /// </summary>
        public IntPtr win32HandleMetaData;
        /// <summary>
        /// allocFlags
        /// </summary>
        public allocFlags allocFlags;
    }


    /// <summary>
    ///  Memory access descriptor
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUmemAccessDesc
    {
        /// <summary>
        /// Location on which the request is to change it's accessibility
        /// </summary>
        public CUmemLocation location;
        /// <summary>
        /// ::CUmemProt accessibility flags to set on the request
        /// </summary>
        public CUmemAccess_flags flags;
    }

    /// <summary>
    /// Specifies an access policy for a window, a contiguous extent of memory
    /// beginning at base_ptr and ending at base_ptr + num_bytes.<para/>
    /// num_bytes is limited by CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE.<para/>
    /// Partition into many segments and assign segments such that:<para/>
    /// sum of "hit segments" / window == approx.ratio.<para/>
    /// sum of "miss segments" / window == approx 1-ratio.<para/>
    /// Segments and ratio specifications are fitted to the capabilities of
    /// the architecture.<para/>
    /// Accesses in a hit segment apply the hitProp access policy.<para/>
    /// Accesses in a miss segment apply the missProp access policy.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUaccessPolicyWindow
    {
        /// <summary>
        /// Starting address of the access policy window. CUDA driver may align it.
        /// </summary>
        public IntPtr base_ptr;
        /// <summary>
        /// Size in bytes of the window policy. CUDA driver may restrict the maximum size and alignment.
        /// </summary>
        public SizeT num_bytes;
        /// <summary>
        /// hitRatio specifies percentage of lines assigned hitProp, rest are assigned missProp.
        /// </summary>
        public float hitRatio;
        /// <summary>
        /// ::CUaccessProperty set for hit.
        /// </summary>
        public CUaccessProperty hitProp;
        /// <summary>
        /// ::CUaccessProperty set for miss. Must be either NORMAL or STREAMING
        /// </summary>
        public CUaccessProperty missProp;
    }

    /// <summary>
    /// Graph attributes union, used with ::cuKernelNodeSetAttribute/::cuKernelNodeGetAttribute
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CUkernelNodeAttrValue
    {
        /// <summary>
        /// Attribute ::CUaccessPolicyWindow.
        /// </summary>
        [FieldOffset(0)]
        public CUaccessPolicyWindow accessPolicyWindow;
        /// <summary>
        /// Nonzero indicates a cooperative kernel (see ::cuLaunchCooperativeKernel).
        /// </summary>
        [FieldOffset(0)]
        public int cooperative;
    }

    /// <summary>
    /// Stream attributes union, used with ::cuStreamSetAttribute/::cuStreamGetAttribute
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CUstreamAttrValue
    {
        /// <summary>
        /// Attribute ::CUaccessPolicyWindow.
        /// </summary>
        [FieldOffset(0)]
        public CUaccessPolicyWindow accessPolicyWindow;
        /// <summary>
        /// Value for ::CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY.
        /// </summary>
        [FieldOffset(0)]
        public CUsynchronizationPolicy syncPolicy;
    }


    /// <summary>
    /// CUDA array sparse properties
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaArraySparseProperties
    {

        /// <summary>
        /// TileExtent
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct TileExtent
        {
            /// <summary>
            /// Width of sparse tile in elements
            /// </summary>
            public uint width;
            /// <summary>
            /// Height of sparse tile in elements
            /// </summary>
            public uint height;
            /// <summary>
            /// Depth of sparse tile in elements
            /// </summary>
            public uint depth;
        }
        /// <summary>
        /// TileExtent
        /// </summary>
        public TileExtent tileExtent;

        /// <summary>
        /// First mip level at which the mip tail begins.
        /// </summary>
        public uint miptailFirstLevel;
        /// <summary>
        /// Total size of the mip tail.
        /// </summary>
        public ulong miptailSize;
        /// <summary>
        /// Flags will either be zero or ::CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL 
        /// </summary>
        public CUArraySparsePropertiesFlags flags;
        uint reserved0;
        uint reserved1;
        uint reserved2;
        uint reserved3;
    }




    /// <summary>
    /// Specifies the CUDA array or CUDA mipmapped array memory mapping information
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUarrayMapInfo
    {

        /// <summary>
        /// 
        /// </summary>
        [StructLayout(LayoutKind.Explicit)]
        public struct Resource
        {
            /// <summary>
            /// resource
            /// </summary>
            [FieldOffset(0)]

            public CUmipmappedArray mipmap;
            /// <summary>
            /// resource
            /// </summary>
            [FieldOffset(0)]
            public CUarray array;
        }

        /// <summary>
        /// 
        /// </summary>
        [StructLayout(LayoutKind.Explicit)]
        public struct Subresource
        {
            /// <summary>
            /// 
            /// </summary>
            public struct SparseLevel
            {
                /// <summary>
                /// For CUDA mipmapped arrays must a valid mipmap level. For CUDA arrays must be zero 
                /// </summary>
                public uint level;
                /// <summary>
                /// For CUDA layered arrays must be a valid layer index. Otherwise, must be zero
                /// </summary>
                public uint layer;
                /// <summary>
                /// Starting X offset in elements
                /// </summary>
                public uint offsetX;
                /// <summary>
                /// Starting Y offset in elements
                /// </summary>
                public uint offsetY;
                /// <summary>
                /// Starting Z offset in elements
                /// </summary>
                public uint offsetZ;
                /// <summary>
                /// Width in elements
                /// </summary>
                public uint extentWidth;
                /// <summary>
                /// Height in elements
                /// </summary>
                public uint extentHeight;
                /// <summary>
                /// Depth in elements
                /// </summary>
                public uint extentDepth;
            }
            /// <summary>
            /// 
            /// </summary>
            public struct Miptail
            {
                /// <summary>
                /// For CUDA layered arrays must be a valid layer index. Otherwise, must be zero
                /// </summary>
                public uint layer;
                /// <summary>
                /// Offset within mip tail 
                /// </summary>
                public ulong offset;
                /// <summary>
                /// Extent in bytes
                /// </summary>
                public ulong size;
            }

            /// <summary>
            /// 
            /// </summary>
            [FieldOffset(0)]

            public SparseLevel sparseLevel;
            /// <summary>
            /// 
            /// </summary>
            [FieldOffset(0)]

            public Miptail miptail;
        }

        /// <summary>
        /// Resource type
        /// </summary>
        public CUResourceType resourceType;

        /// <summary>
        /// 
        /// </summary>
        public Resource resource;

        /// <summary>
        /// Sparse subresource type
        /// </summary>
        public CUarraySparseSubresourceType subresourceType;

        /// <summary>
        /// 
        /// </summary>
        public Subresource subresource;

        /// <summary>
        /// Memory operation type
        /// </summary>
        public CUmemOperationType memOperationType;
        /// <summary>
        /// Memory handle type
        /// </summary>
        public CUmemHandleType memHandleType;
        /// <summary>
        /// 
        /// </summary>

        public CUmemGenericAllocationHandle memHandle;

        /// <summary>
        /// Offset within the memory
        /// </summary>
        public ulong offset;
        /// <summary>
        /// Device ordinal bit mask
        /// </summary>
        public uint deviceBitMask;
        /// <summary>
        /// flags for future use, must be zero now.
        /// </summary>
        public uint flags;
        /// <summary>
        /// Reserved for future use, must be zero now.
        /// </summary>
        public uint reserved0;
        /// <summary>
        /// Reserved for future use, must be zero now.
        /// </summary>
        public uint reserved1;
    }


    /// <summary>
    /// Semaphore signal node parameters (V1 and V2)
    /// </summary>
    public class CudaExtSemSignalNodeParams
    {
        /// <summary/>
        public CUexternalSemaphore[] extSemArray;
        /// <summary/>
        public CudaExternalSemaphoreSignalParams[] paramsArray;
    }

    /// <summary>
    /// Semaphore signal node parameters (V1 and V2)
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct CudaExtSemSignalNodeParamsInternal
    {
        /// <summary/>
        public IntPtr extSemArray;
        /// <summary/>
        public IntPtr paramsArray;
        /// <summary/>
        public uint numExtSems;
    }

    /// <summary>
    /// Semaphore wait node parameters (V1 and V2)
    /// </summary>
    public class CudaExtSemWaitNodeParams
    {
        /// <summary/>
        public CUexternalSemaphore[] extSemArray;
        /// <summary/>
        public CudaExternalSemaphoreWaitParams[] paramsArray;
    }

    /// <summary>
    /// Semaphore wait node parameters (V1 and V2)
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct CudaExtSemWaitNodeParamsInternal
    {
        /// <summary/>
        public IntPtr extSemArray;
        /// <summary/>
        public IntPtr paramsArray;
        /// <summary/>
        public uint numExtSems;
    }


    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CUmemPoolProps
    {
        /// <summary>
        /// Allocation type. Currently must be specified as CU_MEM_ALLOCATION_TYPE_PINNED
        /// </summary>
        [FieldOffset(0)]
        public CUmemAllocationType allocType;
        /// <summary>
        /// Handle types that will be supported by allocations from the pool.
        /// </summary>
        [FieldOffset(4)]
        public CUmemAllocationHandleType handleTypes;
        /// <summary>
        /// Location where allocations should reside.
        /// </summary>
        [FieldOffset(8)]
        public CUmemLocation location;
        /// <summary>
        /// Windows-specific LPSECURITYATTRIBUTES required when ::CU_MEM_HANDLE_TYPE_WIN32 is specified.
        /// This security attribute defines the scope of which exported allocations may be transferred 
        /// to other processes. In all other cases, this field is required to be zero.
        /// </summary>
        [FieldOffset(16)]
        public IntPtr win32SecurityAttributes;
        /// <summary>
        /// Maximum pool size. When set to 0, defaults to a system dependent value.
        /// </summary>
        [FieldOffset(24)]
        public SizeT maxSize;
        /// <summary>
        /// Bitmask indicating intended usage for the pool.
        /// </summary>
        [FieldOffset(28)]
        public ushort usage;
        ///// <summary/>
        //[MarshalAs(UnmanagedType.ByValArray, SizeConst = 54, ArraySubType = UnmanagedType.U1)]
        [FieldOffset(84)]
        int reserved;
    }

    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUmemPoolPtrExportData
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 64, ArraySubType = UnmanagedType.U1)]
        byte[] reserved;
    }

    /// <summary>
    /// Value for ::CU_EXEC_AFFINITY_TYPE_SM_COUNT
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUexecAffinitySmCount
    {
        /// <summary>
        /// The number of SMs the context is limited to use.
        /// </summary>
        public uint val;
    }

    /// <summary>
    /// Value for ::CU_EXEC_AFFINITY_TYPE_SM_COUNT
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CUexecAffinityParamUnion
    {
        /// <summary>
        /// The number of SMs the context is limited to use.
        /// </summary>
        [FieldOffset(0)]
        public CUexecAffinitySmCount smCount;
    }

    /// <summary>
    /// Execution Affinity Parameters 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUexecAffinityParam
    {
        CUexecAffinityType type;
        CUexecAffinityParamUnion param;
    }

    /// <summary>
    /// Memory allocation node parameters (V1 and V2)
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaMemAllocNodeParams
    {
        /// <summary>
        /// in: location where the allocation should reside (specified in ::location).
        /// ::handleTypes must be::CU_MEM_HANDLE_TYPE_NONE.IPC is not supported.
        /// </summary>
        public CUmemPoolProps poolProps;
        /// <summary>
        /// in: array of memory access descriptors. Used to describe peer GPU access
        /// </summary>
        public CUmemAccessDesc[] accessDescs;
        /// <summary>
        /// in: size in bytes of the requested allocation
        /// </summary>
        public SizeT bytesize;
        /// <summary>
        /// out: address of the allocation returned by CUDA
        /// </summary>
        public CUdeviceptr dptr;
    }

    /// <summary>
    /// Memory allocation node parameters  (V1 and V2)
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct CudaMemAllocNodeParamsInternal
    {
        /// <summary>
        /// in: location where the allocation should reside (specified in ::location).
        /// ::handleTypes must be::CU_MEM_HANDLE_TYPE_NONE.IPC is not supported.
        /// </summary>
        public CUmemPoolProps poolProps;
        /// <summary>
        /// in: array of memory access descriptors. Used to describe peer GPU access
        /// </summary>
        public IntPtr accessDescs;
        /// <summary>
        /// in: number of memory access descriptors.  Must not exceed the number of GPUs.
        /// </summary>
        public SizeT accessDescCount;
        /// <summary>
        /// in: size in bytes of the requested allocation
        /// </summary>
        public SizeT bytesize;
        /// <summary>
        /// out: address of the allocation returned by CUDA
        /// </summary>
        public CUdeviceptr dptr;
    }

    /// <summary>
    /// Memory free node parameters
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaMemFreeNodeParams
    {
        /// <summary>
        /// in: the pointer to free
        /// </summary>
        public CUdeviceptr dptr;
    }

    /// <summary>
    /// Result information returned by cuGraphExecUpdate
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUgraphExecUpdateResultInfo
    {
        /// <summary>
        /// Gives more specific detail when a cuda graph update fails.
        /// </summary>
        public CUgraphExecUpdateResult result;

        /// <summary>
        /// The "to node" of the error edge when the topologies do not match.
        /// The error node when the error is associated with a specific node.
        /// NULL when the error is generic.
        /// </summary>
        public CUgraphNode errorNode;

        /// <summary>
        /// The from node of error edge when the topologies do not match. Otherwise NULL.
        /// </summary>
        public CUgraphNode errorFromNode;
    }

    /// <summary>
    /// Tensor map descriptor. Requires compiler support for aligning to 64 bytes.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 64)]
    public struct CUtensorMap
    {
        //CU_TENSOR_MAP_NUM_QWORDS = 16; Size of tensor map descriptor
        /// <summary/>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16, ArraySubType = UnmanagedType.U8)]
        public ulong[] opaque;
    }

    /// <summary>
    /// CUDA array memory requirements
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaArrayMemoryRequirements
    {
        /// <summary>
        /// Total required memory size
        /// </summary>
        public SizeT size;
        /// <summary>
        /// alignment requirement
        /// </summary>
        public SizeT alignment;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4, ArraySubType = UnmanagedType.U4)]
        private uint[] reserved;
    }


    /// <summary>
    /// Graph instantiation parameters
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaGraphInstantiateParams
    {
        /// <summary>
        /// Instantiation flags
        /// </summary>
        public ulong flags;
        /// <summary>
        /// Upload stream
        /// </summary>
        public CUstream hUploadStream;
        /// <summary>
        /// The node which caused instantiation to fail, if any
        /// </summary>
        public CUgraphNode hErrNode_out;
        /// <summary>
        /// Whether instantiation was successful.  If it failed, the reason why
        /// </summary>
        public CUgraphInstantiateResult result_out;
    }


    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUlaunchMemSyncDomainMap
    {
        /// <summary>
        /// 
        /// </summary>
        public byte default_;
        /// <summary>
        /// 
        /// </summary>
        public byte remote;
    }


    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CUlaunchAttributeValue
    {
        /// <summary>
        /// Cluster dimensions for the kernel node.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct ClusterDim
        {
            /// <summary/>
            public uint x;
            /// <summary/>
            public uint y;
            /// <summary/>
            public uint z;
        }

        /// <summary>
        /// Value of launch attribute ::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct ProgrammaticEvent
        {
            /// <summary>
            /// Event to fire when all blocks trigger it
            /// </summary>
            public CUevent event_;
            /// <summary>
            /// Does not accept ::CU_EVENT_RECORD_EXTERNAL
            /// </summary>
            public int flags;
            /// <summary>
            /// If this is set to non-0, each block launch will automatically trigger the event
            /// </summary>
            public int triggerAtBlockStart;
        }

        /// <summary>
        /// Value of launch attribute ::CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct LaunchCompletionEvent
        {
            /// <summary>
            /// Event to fire when the last block launches
            /// </summary>
            public CUevent event_;
            /// <summary>
            /// Does not accept ::CU_EVENT_RECORD_EXTERNAL
            /// </summary>
            public int flags;
        }

        /// <summary>
        /// Value of launch attribute ::CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION
        /// that represents the desired preferred cluster dimensions for the kernel.
        /// Opaque type with the following fields:<para/>
        /// - \p x - The X dimension of the preferred cluster, in blocks. Must
        /// be a divisor of the grid X dimension, and must be a
        /// multiple of the \p x field of ::CUlaunchAttributeValue::clusterDim.<para/>
        /// - \p y - The Y dimension of the preferred cluster, in blocks. Must
        /// be a divisor of the grid Y dimension, and must be a
        /// multiple of the \p y field of ::CUlaunchAttributeValue::clusterDim.<para/>
        /// - \p z - The Z dimension of the preferred cluster, in blocks. Must be
        /// equal to the \p z field of ::CUlaunchAttributeValue::clusterDim.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct PreferredClusterDim
        {
            /// <summary>
            /// </summary>
            public uint x;
            /// <summary>
            /// </summary>
            public uint y;
            /// <summary>
            /// </summary>
            public uint z;
        }

        /// <summary>
        /// Value of launch attribute ::CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct DeviceUpdatableKernelNode
        {
            /// <summary>
            /// Whether or not the resulting kernel node should be device-updatable.
            /// </summary>
            int deviceUpdatable;
            /// <summary>
            /// Returns a handle to pass to the various device-side update functions.
            /// </summary>
            CUgraphDeviceNode devNode;
        }

        /// <summary>
        /// Attribute ::CUaccessPolicyWindow.
        /// </summary>
        [FieldOffset(0)]
        CUaccessPolicyWindow accessPolicyWindow;
        /// <summary>
        /// Nonzero indicates a cooperative kernel (see ::cuLaunchCooperativeKernel).
        /// </summary>
        [FieldOffset(0)]
        int cooperative;
        /// <summary>
        /// ::CUsynchronizationPolicy for work queued up in this stream
        /// </summary>
        [FieldOffset(0)]
        CUsynchronizationPolicy syncPolicy;

        /// <summary>
        /// Cluster dimensions for the kernel node.
        /// </summary>
        [FieldOffset(0)]
        ClusterDim clusterDim;
        /// <summary>
        /// Cluster scheduling policy preference for the kernel node.
        /// </summary>
        [FieldOffset(0)]
        CUclusterSchedulingPolicy clusterSchedulingPolicyPreference;
        /// <summary>
        /// 
        /// </summary>
        [FieldOffset(0)]
        int programmaticStreamSerializationAllowed;
        /// <summary>
        /// 
        /// </summary>
        [FieldOffset(0)]
        ProgrammaticEvent programmaticEvent;
        /// <summary>
        /// 
        /// </summary>
        [FieldOffset(0)]
        LaunchCompletionEvent launchCompletionEvent;
        /// <summary>
        /// Execution priority of the kernel.
        /// </summary>
        [FieldOffset(0)]
        int priority;
        /// <summary>
        /// 
        /// </summary>
        [FieldOffset(0)]
        CUlaunchMemSyncDomainMap memSyncDomainMap;
        /// <summary>
        /// 
        /// </summary>
        [FieldOffset(0)]
        CUlaunchMemSyncDomain memSyncDomain;
        /// <summary>
        /// 
        /// </summary>
        [FieldOffset(0)]
        PreferredClusterDim preferredClusterDim;
        /// <summary>
        /// 
        /// </summary>
        [FieldOffset(0)]
        DeviceUpdatableKernelNode deviceUpdatableKernelNode;

        /// <summary>
        /// Value of launch attribute ::CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
        /// </summary>
        [FieldOffset(0)]
        uint sharedMemCarveout;

        /// <summary>
        /// Pad to 64 bytes
        /// </summary>
        [FieldOffset(60)]
        public int pad;
    }



    /// <summary>
    /// 
    /// </summary>
    public struct CUlaunchAttribute
    {
        /// <summary>
        /// 
        /// </summary>
        public CUlaunchAttributeID id;
        /// <summary>
        /// 
        /// </summary>
        public int pad;
        /// <summary>
        /// 
        /// </summary>
        public CUlaunchAttributeValue value;
    }

    internal struct CUlaunchConfigInternal
    {
        /// <summary>
        /// Width of grid in blocks
        /// </summary>
        public uint gridDimX;
        /// <summary>
        /// Height of grid in blocks
        /// </summary>
        public uint gridDimY;
        /// <summary>
        /// Depth of grid in blocks
        /// </summary>
        public uint gridDimZ;
        /// <summary>
        /// X dimension of each thread block
        /// </summary>
        public uint blockDimX;
        /// <summary>
        /// Y dimension of each thread block
        /// </summary>
        public uint blockDimY;
        /// <summary>
        /// Z dimension of each thread block
        /// </summary>
        public uint blockDimZ;
        /// <summary>
        /// Dynamic shared-memory size per thread block in bytes
        /// </summary>
        public uint sharedMemBytes;
        /// <summary>
        /// Stream identifier
        /// </summary>
        public CUstream hStream;
        /// <summary>
        /// nullable if numAttrs == 0
        /// </summary>
        public IntPtr attrs;
        /// <summary>
        /// number of attributes populated in attrs
        /// </summary>
        public uint numAttrs;
    }
    /// <summary>
    /// 
    /// </summary>
    public struct CUlaunchConfig
    {
        /// <summary>
        /// Width of grid in blocks
        /// </summary>
        public uint gridDimX;
        /// <summary>
        /// Height of grid in blocks
        /// </summary>
        public uint gridDimY;
        /// <summary>
        /// Depth of grid in blocks
        /// </summary>
        public uint gridDimZ;
        /// <summary>
        /// X dimension of each thread block
        /// </summary>
        public uint blockDimX;
        /// <summary>
        /// Y dimension of each thread block
        /// </summary>
        public uint blockDimY;
        /// <summary>
        /// Z dimension of each thread block
        /// </summary>
        public uint blockDimZ;
        /// <summary>
        /// Dynamic shared-memory size per thread block in bytes
        /// </summary>
        public uint sharedMemBytes;
        /// <summary>
        /// Stream identifier
        /// </summary>
        public CUstream hStream;
        /// <summary>
        /// nullable if numAttrs == 0
        /// </summary>
        public CUlaunchAttribute[] attrs;
    }

    /// <summary>
    /// Specifies the properties for a multicast object.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUmulticastObjectProp
    {
        /// <summary>
        /// The number of devices in the multicast team that will bind memory to this object
        /// </summary>
        public uint numDevices;
        /// <summary>
        /// The maximum amount of memory that can be bound to this multicast object per device
        /// </summary>
        public SizeT size;
        /// <summary>
        /// Bitmask of exportable handle types (see ::CUmemAllocationHandleType) for this object
        /// </summary>
        public ulong handleTypes;
        /// <summary>
        /// Flags for future use, must be zero now
        /// </summary>
        public ulong flags;
    }

    /// <summary>
    /// Child graph node parameters
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaChildGraphNodeParams
    {
        /// <summary>
        /// The child graph to clone into the node for node creation, or a handle to the graph owned by the node for node query
        /// </summary>
        public CUgraph graph;
    }

    /// <summary>
    /// Event record node parameters 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaEventRecordNodeParams
    {
        /// <summary>
        /// The event to record when the node executes
        /// </summary>
        public CUevent Event;
    }

    /// <summary>
    /// Event wait node parameters
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaEventWaitNodeParams
    {
        /// <summary>
        /// The event to wait on from the node
        /// </summary>
        public CUevent Event;
    }

    /// <summary>
    /// Note that not all fields are public. Private fields must be set using the Set/Get methods that allocate / free additional memory
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CUgraphNodeParams
    {
        /// <summary/>
        [FieldOffset(0)]
        public CUgraphNodeType type;

        /// <summary/>
        [FieldOffset(12)]
        int reserved0;

        /// <summary/>
        [FieldOffset(16)]
        public CudaKernelNodeParams kernel;
        /// <summary/>
        [FieldOffset(16)]
        public CudaMemcpyNodeParams memcpy;
        /// <summary/>
        [FieldOffset(16)]
        public CudaMemsetNodeParamsV2 memset;
        /// <summary/>
        [FieldOffset(16)]
        CudaHostNodeParamsInternal host;
        /// <summary/>
        [FieldOffset(16)]
        public CudaChildGraphNodeParams graph;
        /// <summary/>
        [FieldOffset(16)]
        public CudaEventWaitNodeParams eventWait;
        /// <summary/>
        [FieldOffset(16)]
        public CudaEventRecordNodeParams eventRecord;
        /// <summary/>
        [FieldOffset(16)]
        CudaExtSemSignalNodeParamsInternal extSemSignal;
        /// <summary/>
        [FieldOffset(16)]
        CudaExtSemWaitNodeParamsInternal extSemWait;
        /// <summary/>
        [FieldOffset(16)]
        CudaMemAllocNodeParamsInternal alloc;
        /// <summary/>
        [FieldOffset(16)]
        public CudaMemFreeNodeParams free;
        /// <summary/>
        [FieldOffset(16)]
        CudaBatchMemOpNodeParamsInternal memOp;
        /// <summary/>
        [FieldOffset(16)]
        public CudaConditionalNodeParams conditional;

        [FieldOffset(248)]
        long reserved2;

        /// <summary>
        /// Fills the internal CudaHostNodeParams structure that allocates additional memory. Make sure that the delegate is not garbage collected by pinning it!
        /// </summary>
        /// <param name="hostNodeParams"></param>
        public void Set(ref CudaHostNodeParams hostNodeParams)
        {
            host = new CudaHostNodeParamsInternal();
            host.userData = hostNodeParams.userData;
            host.fn = IntPtr.Zero;
            if (hostNodeParams.fn != null)
            {
                host.fn = Marshal.GetFunctionPointerForDelegate(hostNodeParams.fn);
            }
        }
        /// <summary>
        /// Fills the internal CudaMemAllocNodeParams structure that allocates additional memory. Each Set call must be followed by a call to Get() in order to free the internally allocated memory!
        /// </summary>
        /// <param name="memAllocParams"></param>
        public void Set(CudaMemAllocNodeParams memAllocParams)
        {
            alloc = new CudaMemAllocNodeParamsInternal();
            alloc.poolProps = memAllocParams.poolProps;
            alloc.dptr = memAllocParams.dptr;
            alloc.bytesize = memAllocParams.bytesize;
            alloc.accessDescCount = 0;
            alloc.accessDescs = IntPtr.Zero;

            try
            {
                int arraySize = 0;
                if (memAllocParams.accessDescs != null)
                {
                    arraySize = memAllocParams.accessDescs.Length;
                    alloc.accessDescCount = (uint)arraySize;
                }

                int paramsSize = Marshal.SizeOf(typeof(CUmemAccessDesc));

                if (arraySize > 0)
                {
                    alloc.accessDescs = Marshal.AllocHGlobal(arraySize * paramsSize);
                }

                for (int i = 0; i < arraySize; i++)
                {
                    Marshal.StructureToPtr(memAllocParams.accessDescs[i], alloc.accessDescs + (paramsSize * i), false);
                }
            }
            catch
            {
                //in case of an error, free memory:
                if (alloc.accessDescs != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(alloc.accessDescs);
                    alloc.accessDescs = IntPtr.Zero;
                }
                throw;
            }
        }
        /// <summary>
        /// Fills the internal CudaMemAllocNodeParams structure that allocates additional memory. Each Set call must be followed by a call to Get() in order to free the internally allocated memory!
        /// </summary>
        /// <param name="batchMemOpParams"></param>
        public void Set(CudaBatchMemOpNodeParams batchMemOpParams)
        {
            memOp = new CudaBatchMemOpNodeParamsInternal();
            memOp.ctx = batchMemOpParams.ctx;
            memOp.count = 0;
            memOp.paramArray = IntPtr.Zero;
            memOp.flags = batchMemOpParams.flags;

            try
            {
                int arraySize = 0;
                if (batchMemOpParams.paramArray != null)
                {
                    arraySize = batchMemOpParams.paramArray.Length;
                    memOp.count = (uint)arraySize;
                }

                int paramsSize = Marshal.SizeOf(typeof(CUstreamBatchMemOpParams));

                if (arraySize > 0)
                {
                    memOp.paramArray = Marshal.AllocHGlobal(arraySize * paramsSize);
                }

                for (int i = 0; i < arraySize; i++)
                {
                    Marshal.StructureToPtr(batchMemOpParams.paramArray[i], memOp.paramArray + (paramsSize * i), false);
                }

            }
            catch
            {
                //in case of an error, free memory:
                if (memOp.paramArray != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(memOp.paramArray);
                    memOp.paramArray = IntPtr.Zero;
                }
                throw;
            }
        }
        /// <summary>
        /// Fills the internal CudaExtSemSignalNodeParams structure that allocates additional memory. Each Set call must be followed by a call to Get() in order to free the internally allocated memory!
        /// </summary>
        /// <param name="extSemSignalParams"></param>
        public void Set(CudaExtSemSignalNodeParams extSemSignalParams)
        {
            extSemSignal = new CudaExtSemSignalNodeParamsInternal();
            extSemSignal.extSemArray = IntPtr.Zero;
            extSemSignal.paramsArray = IntPtr.Zero;
            extSemSignal.numExtSems = 0;

            int arraySize = 0;
            if (extSemSignalParams.extSemArray != null && extSemSignalParams.paramsArray != null)
            {
                if (extSemSignalParams.extSemArray.Length != extSemSignalParams.paramsArray.Length)
                {
                    throw new ArgumentException("extSemSignalParams.extSemArray and extSemSignalParams.paramsArray must be of the same length.");
                }
                arraySize = extSemSignalParams.extSemArray.Length;
            }

            try
            {
                int paramsSize = Marshal.SizeOf(typeof(CudaExternalSemaphoreSignalParams));

                if (arraySize > 0)
                {
                    extSemSignal.extSemArray = Marshal.AllocHGlobal(arraySize * IntPtr.Size);
                    extSemSignal.paramsArray = Marshal.AllocHGlobal(arraySize * paramsSize);
                }

                for (int i = 0; i < arraySize; i++)
                {
                    Marshal.StructureToPtr(extSemSignalParams.extSemArray[i], extSemSignal.extSemArray + (IntPtr.Size * i), false);
                    Marshal.StructureToPtr(extSemSignalParams.paramsArray[i], extSemSignal.paramsArray + (paramsSize * i), false);
                }
            }
            catch
            {
                //in case of an error, free memory:
                if (extSemSignal.extSemArray != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(extSemSignal.extSemArray);
                    extSemSignal.extSemArray = IntPtr.Zero;
                }
                if (extSemSignal.paramsArray != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(extSemSignal.paramsArray);
                    extSemSignal.paramsArray = IntPtr.Zero;
                }
                throw;
            }
        }
        /// <summary>
        /// Fills the internal CudaExtSemWaitNodeParams structure that allocates additional memory. Each Set call must be followed by a call to Get() in order to free the internally allocated memory!
        /// </summary>
        /// <param name="extSemWaitParams"></param>
        public void Set(CudaExtSemWaitNodeParams extSemWaitParams)
        {
            extSemWait = new CudaExtSemWaitNodeParamsInternal();
            extSemWait.extSemArray = IntPtr.Zero;
            extSemWait.paramsArray = IntPtr.Zero;
            extSemWait.numExtSems = 0;

            int arraySize = 0;
            if (extSemWaitParams.extSemArray != null && extSemWaitParams.paramsArray != null)
            {
                if (extSemWaitParams.extSemArray.Length != extSemWaitParams.paramsArray.Length)
                {
                    throw new ArgumentException("extSemWaitParams.extSemArray and extSemWaitParams.paramsArray must be of the same length.");
                }
                arraySize = extSemWaitParams.extSemArray.Length;
            }

            try
            {
                int paramsSize = Marshal.SizeOf(typeof(CudaExternalSemaphoreWaitParams));

                if (arraySize > 0)
                {
                    extSemWait.extSemArray = Marshal.AllocHGlobal(arraySize * IntPtr.Size);
                    extSemWait.paramsArray = Marshal.AllocHGlobal(arraySize * paramsSize);
                }

                for (int i = 0; i < arraySize; i++)
                {
                    Marshal.StructureToPtr(extSemWaitParams.extSemArray[i], extSemWait.extSemArray + (IntPtr.Size * i), false);
                    Marshal.StructureToPtr(extSemWaitParams.paramsArray[i], extSemWait.paramsArray + (paramsSize * i), false);
                }
            }
            catch
            {
                //in case of an error, free memory:
                if (extSemWait.extSemArray != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(extSemWait.extSemArray);
                    extSemWait.extSemArray = IntPtr.Zero;
                }
                if (extSemWait.paramsArray != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(extSemWait.paramsArray);
                    extSemWait.paramsArray = IntPtr.Zero;
                }
                throw;
            }
        }


        /// <summary>
        /// Copies the data from the internal structure to CudaMemAllocNodeParams and frees the internally allocated memory. If Set() hasn't been called before on the output structure, the call might fail.
        /// </summary>
        /// <param name="memAllocParams"></param>
        public void Get(ref CudaMemAllocNodeParams memAllocParams)
        {
            try
            {
                // copy return value:
                memAllocParams.dptr = alloc.dptr;
            }
            catch
            {
            }
            finally
            {
                //free memory
                if (alloc.accessDescs != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(alloc.accessDescs);
                    alloc.accessDescs = IntPtr.Zero;
                }
            }
        }
        /// <summary>
        /// Copies the data from the internal structure to CudaBatchMemOpNodeParams and frees the internally allocated memory. If Set() hasn't been called before on the output structure, the call might fail.
        /// </summary>
        public void Get(CudaBatchMemOpNodeParams batchMemOpParams)
        {
            //No return value to set, just clear memory:
            if (memOp.paramArray != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(memOp.paramArray);
                memOp.paramArray = IntPtr.Zero;
            }
        }
        /// <summary>
        /// Copies the data from the internal structure to CudaExtSemSignalNodeParams and frees the internally allocated memory. If Set() hasn't been called before on the output structure, the call might fail.
        /// </summary>
        public void Get(CudaExtSemSignalNodeParams extSemSignalParams)
        {
            //No return value to set, just clear memory:
            if (extSemSignal.extSemArray != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(extSemSignal.extSemArray);
                extSemSignal.extSemArray = IntPtr.Zero;
            }
            if (extSemSignal.paramsArray != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(extSemSignal.paramsArray);
                extSemSignal.paramsArray = IntPtr.Zero;
            }
        }
        /// <summary>
        /// Copies the data from the internal structure to CudaExtSemWaitNodeParams and frees the internally allocated memory. If Set() hasn't been called before on the output structure, the call might fail.
        /// </summary>
        public void Get(CudaExtSemWaitNodeParams extSemWaitParams)
        {
            //No return value to set, just clear memory:
            if (extSemWait.extSemArray != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(extSemWait.extSemArray);
                extSemWait.extSemArray = IntPtr.Zero;
            }
            if (extSemWait.paramsArray != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(extSemWait.paramsArray);
                extSemWait.paramsArray = IntPtr.Zero;
            }
        }

    }

    /// <summary>
    /// Conditional node parameters
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CudaConditionalNodeParams
    {
        /// <summary>
        /// Conditional node handle.<para/>
        /// Handles must be created in advance of creating the node
        /// using ::cuGraphConditionalHandleCreate.
        /// </summary>
        public CUgraphConditionalHandle handle;
        /// <summary>
        /// Type of conditional node.
        /// </summary>
        public CUgraphConditionalNodeType type;
        /// <summary>
        /// Size of graph output array. Allowed values are 1 for CU_GRAPH_COND_TYPE_WHILE, 1 or 2
        /// for CU_GRAPH_COND_TYPE_IF, or any value greater than zero for CU_GRAPH_COND_TYPE_SWITCH.
        /// </summary>
        public uint size;
        /// <summary>
        /// CUDA-owned array populated with conditional node child graphs during creation of the node.<para/>
        /// Valid for the lifetime of the conditional node.<para/>
        /// The contents of the graph(s) are subject to the following constraints:<para/>
        /// - Allowed node types are kernel nodes, empty nodes, child graphs, memsets,
        /// memcopies, and conditionals. This applies recursively to child graphs and conditional bodies.<para/>
        /// - All kernels, including kernels in nested conditionals or child graphs at any level,
        /// must belong to the same CUDA context.<para/>
        /// These graphs may be populated using graph node creation APIs or ::cuStreamBeginCaptureToGraph.
        /// </summary>
        public IntPtr phGraph_out;
        /// <summary>
        /// Context on which to run the node.
        /// Must match context used to create the handle and all body nodes.
        /// </summary>
        public CUcontext ctx;
    }

    /// <summary>
    /// Optional annotation for edges in a CUDA graph. Note, all edges implicitly have annotations and
    /// default to a zero-initialized value if not specified.A zero-initialized struct indicates a
    /// standard full serialization of two nodes with memory visibility.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUgraphEdgeData
    {
        /// <summary>
        /// This indicates when the dependency is triggered from the upstream
        /// node on the edge. The meaning is specfic to the node type. A value
        /// of 0 in all cases means full completion of the upstream node, with
        /// memory visibility to the downstream node or portion thereof
        /// (indicated by \c to_port).<para/>
        /// Only kernel nodes define non-zero ports. A kernel node
        /// can use the following output port types:<para/>
        /// ::CU_GRAPH_KERNEL_NODE_PORT_DEFAULT, ::CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC,
        /// or ::CU_GRAPH_KERNEL_NODE_PORT_LAUNCH_ORDER.
        /// </summary>
        public CUGraphKernelNodePort from_port;
        /// <summary>
        /// This indicates what portion of the downstream node is dependent on
        /// the upstream node or portion thereof (indicated by \c from_port). The
        /// meaning is specific to the node type. A value of 0 in all cases means
        /// the entirety of the downstream node is dependent on the upstream work.<para/>
        /// Currently no node types define non-zero ports. Accordingly, this field
        /// must be set to zero.
        /// </summary>
        public CUGraphKernelNodePort to_port;
        /// <summary>
        /// This should be populated with a value from ::CUgraphDependencyType. (It
        /// is typed as char due to compiler-specific layout of bitfields.) See
        /// ::CUgraphDependencyType.
        /// </summary>
        public CUgraphDependencyType type;
        /// <summary>
        /// These bytes are unused and must be zeroed. This ensures
        /// compatibility if additional fields are added in the future.
        /// </summary>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 5, ArraySubType = UnmanagedType.U1)]
        public byte[] reserved;
    }



    /// <summary>
    /// Information passed to the user via the async notification callback
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct OverBudget
    {
        /// <summary>
        /// 
        /// </summary>
        public ulong bytesOverBudget;
    }

    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CUasyncNotificationInfoUnion
    {
        /// <summary>
        /// 
        /// </summary>
        [FieldOffset(0)]
        public OverBudget overBudget;
    }

    /// <summary>
    /// Information passed to the user via the async notification callback
    /// </summary>
    public struct CUasyncNotificationInfo
    {
        /// <summary>
        /// 
        /// </summary>
        public CUasyncNotificationType type;
        /// <summary>
        /// 
        /// </summary>
        public CUasyncNotificationInfoUnion info;
    }


    /// <summary>
    /// Data for SM-related resources
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUdevSmResource
    {
        /// <summary>
        /// The amount of streaming multiprocessors available in this resource. This is an output parameter only, do not write to this field.
        /// </summary>
        public uint smCount;
    }

    /// <summary>
    /// A tagged union describing different resources identified by the type field. This structure should not be directly modified outside of the API that created it.
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CUdevResource
    {
        /// <summary>
        /// Type of resource, dictates which union field was last set
        /// </summary>
        [FieldOffset(0)]
        public CUdevResourceType type;
        /// <summary>
        /// Resource corresponding to CU_DEV_RESOURCE_TYPE_SM \p. type.
        /// </summary>
        [FieldOffset(96)]
        public CUdevSmResource sm;

        [FieldOffset(144 - 1)]
        byte _oversize;

        /// <summary>
        /// Splits \p CU_DEV_RESOURCE_TYPE_SM resources.<para/>
        /// Splits \p CU_DEV_RESOURCE_TYPE_SM resources into \p nbGroups, adhering to the minimum SM count specified in \p minCount
        /// and the usage flags in \p useFlags.If \p result is NULL, the API simulates a split and provides the amount of groups that
        /// would be created in \p nbGroups. Otherwise, \p nbGroups must point to the amount of elements in \p result and on return,
        /// the API will overwrite \p nbGroups with the amount actually created.The groups are written to the array in \p result.
        /// \p nbGroups can be less than the total amount if a smaller number of groups is needed.
        /// This API is used to spatially partition the input resource.The input resource needs to come from one of
        /// ::cuDeviceGetDevResource, ::cuCtxGetDevResource, or::cuGreenCtxGetDevResource.
        /// A limitation of the API is that the output results cannot be split again without
        /// first creating a descriptor and a green context with that descriptor.
        /// <para/>
        /// When creating the groups, the API will take into account the performance and functional characteristics of the
        /// input resource, and guarantee a split that will create a disjoint set of symmetrical partitions.This may lead to less groups created
        /// than purely dividing the total SM count by the \p minCount due to cluster requirements or
        /// alignment and granularity requirements for the minCount.
        /// <para/>
        /// The \p remainder set, might not have the same functional or performance guarantees as the groups in \p result.
        /// Its use should be carefully planned and future partitions of the \p remainder set are discouraged.
        /// <para/>
        /// A successful API call must either have:
        /// - A valid array of \p result pointers of size passed in \p nbGroups, with \p Input of type \p CU_DEV_RESOURCE_TYPE_SM.
        /// Value of \p minCount must be between 0 and the SM count specified in \p input. \p remaining and \p useFlags are optional.
        /// - NULL passed in for \p result, with a valid integer pointer in \p nbGroups and \p Input of type \p CU_DEV_RESOURCE_TYPE_SM.
        /// Value of \p minCount must be between 0 and the SM count specified in \p input.
        /// This queries the number of groups that would be created by the API.
        /// <para/>
        /// Note: The API is not supported on 32-bit platforms.
        /// </summary>
        public CUdevResource[] SmResourceSplitByCount(uint useFlags, uint minCount)
        {
            CUdevResource[] resources = new CUdevResource[0];
            uint groupCount = 0;
            CUResult res = DriverAPINativeMethods.GreenContextAPI.cuDevSmResourceSplitByCount(IntPtr.Zero, ref groupCount, ref this, IntPtr.Zero, useFlags, minCount);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDevSmResourceSplitByCount", res));
            if (res != CUResult.Success) throw new CudaException(res);

            if (groupCount == 0)
            {
                return resources;
            }

            resources = new CUdevResource[groupCount];
            res = DriverAPINativeMethods.GreenContextAPI.cuDevSmResourceSplitByCount(resources, ref groupCount, ref this, IntPtr.Zero, useFlags, minCount);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDevSmResourceSplitByCount", res));
            if (res != CUResult.Success) throw new CudaException(res);

            return resources;
        }

        /// <summary>
        /// Splits \p CU_DEV_RESOURCE_TYPE_SM resources.<para/>
        /// Splits \p CU_DEV_RESOURCE_TYPE_SM resources into \p nbGroups, adhering to the minimum SM count specified in \p minCount
        /// and the usage flags in \p useFlags.If \p result is NULL, the API simulates a split and provides the amount of groups that
        /// would be created in \p nbGroups. Otherwise, \p nbGroups must point to the amount of elements in \p result and on return,
        /// the API will overwrite \p nbGroups with the amount actually created.The groups are written to the array in \p result.
        /// \p nbGroups can be less than the total amount if a smaller number of groups is needed.
        /// This API is used to spatially partition the input resource.The input resource needs to come from one of
        /// ::cuDeviceGetDevResource, ::cuCtxGetDevResource, or::cuGreenCtxGetDevResource.
        /// A limitation of the API is that the output results cannot be split again without
        /// first creating a descriptor and a green context with that descriptor.
        /// <para/>
        /// When creating the groups, the API will take into account the performance and functional characteristics of the
        /// input resource, and guarantee a split that will create a disjoint set of symmetrical partitions.This may lead to less groups created
        /// than purely dividing the total SM count by the \p minCount due to cluster requirements or
        /// alignment and granularity requirements for the minCount.
        /// <para/>
        /// The \p remainder set, might not have the same functional or performance guarantees as the groups in \p result.
        /// Its use should be carefully planned and future partitions of the \p remainder set are discouraged.
        /// <para/>
        /// A successful API call must either have:
        /// - A valid array of \p result pointers of size passed in \p nbGroups, with \p Input of type \p CU_DEV_RESOURCE_TYPE_SM.
        /// Value of \p minCount must be between 0 and the SM count specified in \p input. \p remaining and \p useFlags are optional.
        /// - NULL passed in for \p result, with a valid integer pointer in \p nbGroups and \p Input of type \p CU_DEV_RESOURCE_TYPE_SM.
        /// Value of \p minCount must be between 0 and the SM count specified in \p input.
        /// This queries the number of groups that would be created by the API.
        /// <para/>
        /// Note: The API is not supported on 32-bit platforms.
        /// </summary>
        public CUdevResource[] SmResourceSplitByCount(uint useFlags, uint minCount, ref CUdevResource remaining)
        {
            remaining = new CUdevResource();
            CUdevResource[] resources = new CUdevResource[0];
            uint groupCount = 0;
            CUResult res = DriverAPINativeMethods.GreenContextAPI.cuDevSmResourceSplitByCount(IntPtr.Zero, ref groupCount, ref this, IntPtr.Zero, useFlags, minCount);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDevSmResourceSplitByCount", res));
            if (res != CUResult.Success) throw new CudaException(res);

            if (groupCount == 0)
            {
                return resources;
            }

            resources = new CUdevResource[groupCount];
            res = DriverAPINativeMethods.GreenContextAPI.cuDevSmResourceSplitByCount(resources, ref groupCount, ref this, ref remaining, useFlags, minCount);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDevSmResourceSplitByCount", res));
            if (res != CUResult.Success) throw new CudaException(res);

            return resources;
        }
    }

    /// <summary>
    /// CIG Context Create Params
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUctxCigParam
    {
        /// <summary>
        /// 
        /// </summary>
        public CUcigDataType sharedDataType;
        /// <summary>
        /// 
        /// </summary>
        public IntPtr sharedData;
    }

    /// <summary>
    /// Params for creating CUDA context. Exactly one of execAffinityParams and cigParams must be non-NULL.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUctxCreateParams
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr execAffinityParams;
        /// <summary>
        /// 
        /// </summary>
        public int numExecAffinityParams;
        /// <summary>
        /// 
        /// </summary>
        public IntPtr cigParams;
    }

    /// <summary>
    /// CUDA checkpoint optional lock arguments
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUcheckpointLockArgs
    {
        /// <summary>
        /// Timeout in milliseconds to attempt to lock the process, 0 indicates no timeout
        /// </summary>
        public uint timeoutMs;
        /// <summary>
        /// Reserved for future use, must be zero
        /// </summary>
        public uint reserved0;
        /// <summary>
        /// Reserved for future use, must be zeroed
        /// </summary>
        ulong reserved1;
        ulong reserved2;
        ulong reserved3;
        ulong reserved4;
        ulong reserved5;
        ulong reserved6;
        ulong reserved7;
    }

    /// <summary>
    /// CUDA checkpoint optional checkpoint arguments
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUcheckpointCheckpointArgs
    {
        /// <summary>
        /// Reserved for future use, must be zeroed
        /// </summary>
        ulong reserved0;
        ulong reserved1;
        ulong reserved2;
        ulong reserved3;
        ulong reserved4;
        ulong reserved5;
        ulong reserved6;
        ulong reserved7;
    }

    /// <summary>
    /// CUDA checkpoint optional restore arguments
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUcheckpointRestoreArgs
    {
        /// <summary>
        /// Reserved for future use, must be zeroed
        /// </summary>
        ulong reserved0;
        ulong reserved1;
        ulong reserved2;
        ulong reserved3;
        ulong reserved4;
        ulong reserved5;
        ulong reserved6;
        ulong reserved7;
    }

    /// <summary>
    /// CUDA checkpoint optional unlock arguments
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUcheckpointUnlockArgs
    {
        /// <summary>
        /// Reserved for future use, must be zeroed
        /// </summary>
        ulong reserved0;
        ulong reserved1;
        ulong reserved2;
        ulong reserved3;
        ulong reserved4;
        ulong reserved5;
        ulong reserved6;
        ulong reserved7;
    }

    /// <summary>
    /// Attributes specific to copies within a batch. For more details on usage see ::cuMemcpyBatchAsync.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUmemcpyAttributes
    {
        /// <summary>
        /// Source access ordering to be observed for copies with this attribute.
        /// </summary>
        public CUmemcpySrcAccessOrder srcAccessOrder;
        /// <summary>
        /// Hint location for the source operand. Ignored when the pointers are not managed memory or memory allocated outside CUDA.
        /// </summary>
        public CUmemLocation srcLocHint;
        /// <summary>
        /// Hint location for the destination operand. Ignored when the pointers are not managed memory or memory allocated outside CUDA.
        /// </summary>
        public CUmemLocation dstLocHint;
        /// <summary>
        /// Additional flags for copies with this attribute. See ::CUmemcpyFlags
        /// </summary>
        public CUmemcpyFlags flags;
    }


    /// <summary>
    /// Struct representing offset into a CUarray in elements
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUoffset3D
    {
        /// <summary/>
        public SizeT x;
        /// <summary/>
        public SizeT y;
        /// <summary/>
        public SizeT z;
    }

    /// <summary>
    /// Struct representing width/height/depth of a CUarray in elements
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUextent3D
    {
        /// <summary/>
        public SizeT width;
        /// <summary/>
        public SizeT height;
        /// <summary/>
        public SizeT depth;
    }


    /// <summary>
    /// Struct representing an operand for copy with ::cuMemcpy3DBatchAsync
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CUmemcpy3DOperand
    {
        /// <summary>
        /// Struct representing an operand when ::CUmemcpy3DOperand::type is ::CU_MEMCPY_OPERAND_TYPE_POINTER
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct Ptr
        {
            /// <summary>
            /// 
            /// </summary>
            public CUdeviceptr ptr;
            /// <summary>
            /// Length of each row in elements.
            /// </summary>
            public SizeT rowLength;
            /// <summary>
            /// Height of each layer in elements.
            /// </summary>
            public SizeT layerHeight;
            /// <summary>
            /// Hint location for the operand. Ignored when the pointers are not managed memory or memory allocated outside CUDA.
            /// </summary>
            public CUmemLocation locHint;
        }

        /// <summary>
        /// Struct representing an operand when ::CUmemcpy3DOperand::type is ::CU_MEMCPY_OPERAND_TYPE_ARRAY
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct Array
        {
            /// <summary/>
            public CUarray array;
            /// <summary/>
            public CUoffset3D offset;
        }

        /// <summary/>
        [FieldOffset(0)]
        public CUmemcpy3DOperandType type;

        /// <summary/>
        [FieldOffset(4)]
        public Ptr ptr;

        /// <summary/>
        [FieldOffset(4)]
        public Array array;
    }

    /// <summary/>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_MEMCPY3D_BATCH_OP
    {
        /// <summary>
        /// Source memcpy operand.
        /// </summary>
        public CUmemcpy3DOperand src;
        /// <summary>
        /// Destination memcpy operand.
        /// </summary>
        public CUmemcpy3DOperand dst;
        /// <summary>
        /// Extents of the memcpy between src and dst. The width, height and depth components must not be 0.
        /// </summary>
        public CUextent3D extent;
        /// <summary>
        /// Source access ordering to be observed for copy from src to dst.
        /// </summary>
        public CUmemcpySrcAccessOrder srcAccessOrder;
        /// <summary>
        /// Additional flags for copies with this attribute. See ::CUmemcpyFlags
        /// </summary>
        public CUmemcpyFlags flags;
    }


    /// <summary>
    /// Structure describing the parameters that compose a single decompression operation.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUmemDecompressParams
    {
        /// <summary>
        /// The number of bytes to be read and decompressed from ::CUmemDecompressParams_st.src.
        /// </summary>
        public SizeT srcNumBytes;
        /// <summary>
        /// The number of bytes that the decompression operation will be expected to
        /// write to::CUmemDecompressParams_st.dst.This value is optional; if
        /// present, it may be used by the CUDA driver as a heuristic for scheduling
        /// the individual decompression operations.
        /// </summary>
        public SizeT dstNumBytes;
        /// <summary>
        /// After the decompression operation has completed, the actual number of
        /// bytes written to::CUmemDecompressParams.dst will be recorded as a 32-bit
        /// unsigned integer in the memory at this address.
        /// </summary>
        public IntPtr dstActBytes;
        /// <summary>
        /// Pointer to a buffer of at least ::CUmemDecompressParams_st.srcNumBytes compressed bytes.
        /// </summary>
        public IntPtr src;
        /// <summary>
        /// Pointer to a buffer where the decompressed data will be written. The number of bytes
        /// written to this location will be recorded in the memory
        /// pointed to by::CUmemDecompressParams_st.dstActBytes
        /// </summary>
        public IntPtr dst;
        /// <summary>
        /// The decompression algorithm to use.
        /// </summary>
        public CUmemDecompressAlgorithm algo;
        /// <summary>
        /// These 20 bytes are unused and must be zeroed. This ensures compatibility if additional fields are added in the future.
        /// </summary>
        ulong padding0;
        ulong padding1;
        uint padding2;
    }

    #endregion
}
