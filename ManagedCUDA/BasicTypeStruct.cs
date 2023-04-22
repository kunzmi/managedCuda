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
    /// CudaBatchMemOpNodeParams
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
    /// GPU kernel node parameters
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
    /// Memset node parameters
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
        /// Initialieses the struct
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
        /// Initialieses the struct
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
    /// Host node parameters
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
    /// Semaphore signal node parameters
    /// </summary>
    public class CudaExtSemSignalNodeParams
    {
        /// <summary/>
        public CUexternalSemaphore[] extSemArray;
        /// <summary/>
        public CudaExternalSemaphoreSignalParams[] paramsArray;
    }

    /// <summary>
    /// Semaphore wait node parameters
    /// </summary>
    public class CudaExtSemWaitNodeParams
    {
        /// <summary/>
        public CUexternalSemaphore[] extSemArray;
        /// <summary/>
        public CudaExternalSemaphoreWaitParams[] paramsArray;
    }


    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUmemPoolProps
    {
        /// <summary/>
        public CUmemAllocationType allocType;
        /// <summary/>
        public CUmemAllocationHandleType handleTypes;
        /// <summary/>
        public CUmemLocation location;
        /// <summary/>
        public IntPtr win32SecurityAttributes;
        /// <summary/>
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 64, ArraySubType = UnmanagedType.U1)]
        byte[] reserved;
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
    /// Memory allocation node parameters
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
    /// Memory allocation node parameters
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
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct ProgrammaticEvent
        {
            /// <summary/>
            public CUevent event_;
            /// <summary>
            /// Does not accept ::CU_EVENT_RECORD_EXTERNAL
            /// </summary>
            public int flags;
            /// <summary/>
            public int triggerAtBlockStart;
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

    //public struct CUlibraryHostUniversalFunctionAndDataTable
    //{
    //    IntPtr functionTable;
    //    SizeT functionWindowSize;
    //    IntPtr dataTable;
    //    SizeT dataWindowSize;
    //}
    #endregion
}
