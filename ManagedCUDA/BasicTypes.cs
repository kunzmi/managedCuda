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
        public void SetParameters(CUDA_HOST_NODE_PARAMS nodeParams)
        {
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphHostNodeSetParams(this, ref nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphHostNodeSetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Sets the parameters of kernel node nodeParams.
        /// </summary>
        /// <param name="nodeParams"></param>
        public void SetParameters(CUDA_KERNEL_NODE_PARAMS nodeParams)
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
        public void SetParameters(CUDA_MEMSET_NODE_PARAMS nodeParams)
        {
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphMemsetNodeSetParams(this, ref nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphMemsetNodeSetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Gets the parameters of host node.
        /// </summary>
        /// <param name="nodeParams"></param>
        public void GetParameters(ref CUDA_HOST_NODE_PARAMS nodeParams)
        {
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphHostNodeGetParams(this, ref nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphHostNodeGetParams", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Gets the parameters of kernel node.
        /// </summary>
        /// <param name="nodeParams"></param>
        public void GetParameters(ref CUDA_KERNEL_NODE_PARAMS nodeParams)
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
        public void GetParameters(ref CUDA_MEMSET_NODE_PARAMS nodeParams)
        {
            CUResult res = DriverAPINativeMethods.GraphManagment.cuGraphMemsetNodeGetParams(this, ref nodeParams);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuGraphMemsetNodeGetParams", res));
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
    #endregion

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
		[FieldOffset(31*4)]
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
        //In cuda header, an ulong[6] fixes the union size to 48 bytes, we
        //achieve the same in C# if we set at offset 40 an simple ulong
        [FieldOffset(5*8)]
        ulong pad;
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
    public struct CUDA_KERNEL_NODE_PARAMS
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
    }


    /// <summary>
    /// Memset node parameters
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CUDA_MEMSET_NODE_PARAMS
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
        /// Width in bytes, of the row
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
        public static CUDA_MEMSET_NODE_PARAMS init<T>(CudaDeviceVariable<T> deviceVariable, uint value) where T : struct
        {
            CUDA_MEMSET_NODE_PARAMS para = new CUDA_MEMSET_NODE_PARAMS();
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
        public static CUDA_MEMSET_NODE_PARAMS init<T>(CudaPitchedDeviceVariable<T> deviceVariable, uint value) where T : struct
        {
            CUDA_MEMSET_NODE_PARAMS para = new CUDA_MEMSET_NODE_PARAMS();
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
    public struct CUDA_HOST_NODE_PARAMS
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
    [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)]
    public struct Win32Handle
    {
        /// <summary>
        /// Valid NT handle. Must be NULL if 'name' is non-NULL
        /// </summary>
        public IntPtr handle;
        /// <summary>
        /// Name of a valid memory object. Must be NULL if 'handle' is non-NULL.
        /// </summary>
        [MarshalAs(UnmanagedType.LPStr)]
        string name;
    }

    /// <summary>
    /// External memory handle descriptor
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC
    {
        /// <summary>
        /// Type of the handle
        /// </summary>
        [FieldOffset(0)]
        public CUexternalMemoryHandleType type;

        /// <summary>
        /// File descriptor referencing the memory object. Valid when type is CUDA_EXTERNAL_MEMORY_DEDICATED
        /// </summary>
        [FieldOffset(4)]
        public int fd;

        /// <summary>
        /// Win32 handle referencing the semaphore object.
        /// </summary>
        [FieldOffset(4)]
        public Win32Handle handle;

		/// <summary>
		/// A handle representing an NvSciBuf Object.Valid when type
		/// is ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF
		/// </summary>
		[FieldOffset(4)]
        public IntPtr nvSciBufObject;

        /// <summary>
        /// Size of the memory allocation
        /// </summary>
        [FieldOffset(20)]
        public ulong size;

        /// <summary>
        /// Flags must either be zero or ::CUDA_EXTERNAL_MEMORY_DEDICATED
        /// </summary>
        [FieldOffset(28)]
        public CudaExternalMemory flags;

        //Original struct definition in cuda-header sets a unsigned int[16] array at the end of the struct.
        //To get the same struct size (96 bytes), we simply put an uint at FieldOffset 92.
        [FieldOffset(92)]
        private uint reserved;
    }


    /// <summary>
    /// External semaphore handle descriptor
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC
    {
        /// <summary>
        /// Type of the handle
        /// </summary>
        [FieldOffset(0)]
        CUexternalSemaphoreHandleType type;

        /// <summary>
        /// File descriptor referencing the semaphore object. Valid when type is CUDA_EXTERNAL_MEMORY_DEDICATED
        /// </summary>
        [FieldOffset(4)]
        public int fd;

        /// <summary>
        /// Win32 handle referencing the semaphore object.
        /// </summary>
        [FieldOffset(4)]
        public Win32Handle handle;

		/// <summary>
		/// Valid NvSciSyncObj. Must be non NULL
		/// </summary>
		[FieldOffset(4)]
		public IntPtr nvSciSyncObj;
		/// <summary>
		/// Flags reserved for the future. Must be zero.
		/// </summary>
		[FieldOffset(20)]
        public uint flags;

        //Original struct definition in cuda-header sets a unsigned int[16] array at the end of the struct.
        //To get the same struct size (88 bytes), we simply put an uint at FieldOffset 84.
        [FieldOffset(84)]
        private uint reserved;
    }


    /// <summary>
    /// External memory buffer descriptor
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC
    {
        /// <summary>
        /// Offset into the memory object where the buffer's base is
        /// </summary>
        [FieldOffset(0)]
        public ulong offset;
        /// <summary>
        /// Size of the buffer
        /// </summary>
        [FieldOffset(8)]
        public ulong size;
        /// <summary>
        /// Flags reserved for future use. Must be zero.
        /// </summary>
        [FieldOffset(16)]
        public uint flags;

        [FieldOffset(32)] //instead of uint[16]
        private uint reserved;
    }


    /// <summary>
    /// External memory mipmap descriptor
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC
    {
        /// <summary>
        /// Offset into the memory object where the base level of the mipmap chain is.
        /// </summary>
        [FieldOffset(0)]
        public ulong offset;
        /// <summary>
        /// Format, dimension and type of base level of the mipmap chain
        /// </summary>
        [FieldOffset(8)]
        public CUDAArray3DDescriptor arrayDesc;
        /// <summary>
        /// Total number of levels in the mipmap chain
        /// </summary>
        [FieldOffset(12)]
        public uint numLevels;

        [FieldOffset(24)] //instead of uint[16]
        private uint reserved;
    }


    /// <summary>
    /// External semaphore signal parameters
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    public struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS
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
            [FieldOffset(0)]
            Fence fence;
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
			[FieldOffset(8)]
			NvSciSync nvSciSync;

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
			[FieldOffset(16)]
			NvSciSync keyedMutex;

			[FieldOffset(68)] //params.reserved[9];
			private uint reserved;
        }
        [FieldOffset(0)]
        Parameters parameters;
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
    public struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS
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
            [FieldOffset(0)]
            Fence fence;
            [FieldOffset(20)]
            private uint reserved;
        }
        [FieldOffset(0)]
        Parameters parameters;
        /// <summary>
        /// Flags reserved for the future. Must be zero.
        /// </summary>
        [FieldOffset(84)]
        public uint flags;
        [FieldOffset(88)]
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
		/// Windows-specific LPSECURITYATTRIBUTES required when
		/// ::CU_MEM_HANDLE_TYPE_WIN32 is specified.This security attribute defines
		/// the scope of which exported allocations may be tranferred to other
		/// processes.In all other cases, this field is required to be zero.
		/// </summary>
		public IntPtr win32HandleMetaData;
		/// <summary>
		/// Reserved for future use, must be zero
		/// </summary>
		public ulong reserved;
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

    #endregion

    #region Enums
    /// <summary>
    /// Texture reference addressing modes
    /// </summary>
    public enum CUAddressMode
	{
		/// <summary>
		/// Wrapping address mode
		/// </summary>
		Wrap = 0,
		
		/// <summary>
		/// Clamp to edge address mode
		/// </summary>
		Clamp = 1,
		
		/// <summary>
		/// Mirror address mode
		/// </summary>
		Mirror = 2,
		
		/// <summary>
		/// Border address mode
		/// </summary>
		Border = 3
	}

	/// <summary>
	/// Array formats
	/// </summary>
	public enum CUArrayFormat
	{
		/// <summary>
		/// Unsigned 8-bit integers
		/// </summary>
		UnsignedInt8 = 0x01,

		/// <summary>
		/// Unsigned 16-bit integers
		/// </summary>
		UnsignedInt16 = 0x02,

		/// <summary>
		/// Unsigned 32-bit integers
		/// </summary>
		UnsignedInt32 = 0x03,

		/// <summary>
		/// Signed 8-bit integers
		/// </summary>
		SignedInt8 = 0x08,

		/// <summary>
		/// Signed 16-bit integers
		/// </summary>
		SignedInt16 = 0x09,

		/// <summary>
		/// Signed 32-bit integers
		/// </summary>
		SignedInt32 = 0x0a,

		/// <summary>
		/// 16-bit floating point
		/// </summary>
		Half = 0x10,

		/// <summary>
		/// 32-bit floating point
		/// </summary>
		Float = 0x20,
	}

	/// <summary>
	/// Compute mode that device is currently in.
	/// </summary>
	public enum CUComputeMode
	{
		/// <summary>
		/// Default mode - Device is not restricted and can have multiple CUDA
		/// contexts present at a single time.
		/// </summary>
		Default = 0,
		
		///// <summary>
		///// Compute-exclusive mode - Device can have only one CUDA context
		///// present on it at a time.
		///// </summary>
		//Exclusive = 1,
		
		/// <summary>
		/// Compute-prohibited mode - Device is prohibited from creating
		/// new CUDA contexts.
		/// </summary>
		Prohibited = 2,
		
		/// <summary>
		/// Compute-exclusive-process mode (Only one context used by a 
		/// single process can be present on this device at a time)
		/// </summary>
		ExclusiveProcess = 2
	}

	/// <summary>
	/// Memory advise values
	/// </summary>
	public enum CUmemAdvise
	{
		/// <summary>
		/// Data will mostly be read and only occassionally be written to
		/// </summary>
		SetReadMostly = 1,
		/// <summary>
		/// Undo the effect of ::CU_MEM_ADVISE_SET_READ_MOSTLY
		/// </summary>
		UnsetReadMostly = 2, 
		/// <summary>
		/// Set the preferred location for the data as the specified device
		/// </summary>
		SetPreferredLocation = 3, 
		/// <summary>
		/// Clear the preferred location for the data
		/// </summary>
		UnsetPreferredLocation = 4, 
		/// <summary>
		/// Data will be accessed by the specified device, so prevent page faults as much as possible
		/// </summary>
		SetAccessedBy = 5, 
		/// <summary>
		/// Let the Unified Memory subsystem decide on the page faulting policy for the specified device
		/// </summary>
		UnsetAccessedBy = 6
	}

	/// <summary>
	/// Context Attach flags
	/// </summary>
	public enum CUCtxAttachFlags
	{
		/// <summary>
		/// None
		/// </summary>
		None = 0
	}

	/// <summary>
	/// Device properties
	/// </summary>
	public enum CUDeviceAttribute
	{
		/// <summary>
		/// Maximum number of threads per block
		/// </summary>
		MaxThreadsPerBlock = 1,
		
		/// <summary>
		/// Maximum block dimension X
		/// </summary>
		MaxBlockDimX = 2,
		 
		/// <summary>
		/// Maximum block dimension Y
		/// </summary>
		MaxBlockDimY = 3,
		 
		/// <summary>
		/// Maximum block dimension Z
		/// </summary>
		MaxBlockDimZ = 4,
		 
		/// <summary>
		/// Maximum grid dimension X
		/// </summary>
		MaxGridDimX = 5,
		 
		/// <summary>
		/// Maximum grid dimension Y
		/// </summary>
		MaxGridDimY = 6,
		 
		/// <summary>
		/// Maximum grid dimension Z
		/// </summary>
		MaxGridDimZ = 7,
		 
		/// <summary>
		/// Maximum amount of shared memory
		/// available to a thread block in bytes; this amount is shared by all thread blocks simultaneously resident on a
		/// multiprocessor
		/// </summary>
		MaxSharedMemoryPerBlock = 8,
		 
		/// <summary>
		/// Deprecated, use MaxSharedMemoryPerBlock
		/// </summary>
		[Obsolete("Use MaxSharedMemoryPerBlock")]
		SharedMemoryPerBlock = 8,
		 
		/// <summary>
		/// Memory available on device for __constant__ variables in a CUDA C kernel in bytes
		/// </summary>
		TotalConstantMemory = 9,
		 
		/// <summary>
		/// Warp size in threads
		/// </summary>
		WarpSize = 10,
		 
		/// <summary>
		/// Maximum pitch in bytes allowed by the memory copy functions
		/// that involve memory regions allocated through <see cref="DriverAPINativeMethods.MemoryManagement.cuMemAllocPitch_v2"/>
		/// </summary>
		MaxPitch = 11,
		 
		/// <summary>
		/// Deprecated, use MaxRegistersPerBlock
		/// </summary>
		[Obsolete("Use MaxRegistersPerBlock")]
		RegistersPerBlock = 12,
		 
		/// <summary>
		/// Maximum number of 32-bit registers available
		/// to a thread block; this number is shared by all thread blocks simultaneously resident on a multiprocessor
		/// </summary>
		MaxRegistersPerBlock = 12,
		 
		/// <summary>
		/// Typical clock frequency in kilohertz
		/// </summary>
		ClockRate = 13,
		 
		/// <summary>
		/// Alignment requirement; texture base addresses
		/// aligned to textureAlign bytes do not need an offset applied to texture fetches
		/// </summary>
		TextureAlignment = 14,
		 
		/// <summary>
		/// 1 if the device can concurrently copy memory between host
		/// and device while executing a kernel, or 0 if not
		/// </summary>
		GPUOverlap = 15,
		 
		/// <summary>
		/// Number of multiprocessors on device
		/// </summary>
		MultiProcessorCount = 0x10,
		 
		/// <summary>
		/// Specifies whether there is a run time limit on kernels. <para/>
		/// 1 if there is a run time limit for kernels executed on the device, or 0 if not
		/// </summary>
		KernelExecTimeout = 0x11,
		 
		/// <summary>
		/// Device is integrated with host memory. 1 if the device is integrated with the memory subsystem, or 0 if not
		/// </summary>
		Integrated = 0x12,
		 
		/// <summary>
		/// Device can map host memory into CUDA address space. 1 if the device can map host memory into the
		/// CUDA address space, or 0 if not
		/// </summary>
		CanMapHostMemory = 0x13,
		 
		/// <summary>
		/// Compute mode (See <see cref="CUComputeMode"/> for details)
		/// </summary>
		ComputeMode = 20,

		 
		/// <summary>
		/// Maximum 1D texture width
		/// </summary>
		MaximumTexture1DWidth = 21, 
		 
		/// <summary>
		/// Maximum 2D texture width
		/// </summary>
		MaximumTexture2DWidth = 22, 
		 
		/// <summary>
		/// Maximum 2D texture height
		/// </summary>
		MaximumTexture2DHeight = 23,
		 
		/// <summary>
		/// Maximum 3D texture width
		/// </summary>
		MaximumTexture3DWidth = 24, 
		 
		/// <summary>
		/// Maximum 3D texture height
		/// </summary>
		MaximumTexture3DHeight = 25,
		 
		/// <summary>
		/// Maximum 3D texture depth
		/// </summary>
		MaximumTexture3DDepth = 26, 
		 
		/// <summary>
		/// Maximum texture array width
		/// </summary>
		MaximumTexture2DArray_Width = 27, 
		 
		/// <summary>
		/// Maximum texture array height
		/// </summary>
		MaximumTexture2DArray_Height = 28,
		 
		/// <summary>
		/// Maximum slices in a texture array
		/// </summary>
		MaximumTexture2DArray_NumSlices = 29, 
		 
		/// <summary>
		/// Alignment requirement for surfaces
		/// </summary>
		SurfaceAllignment = 30, 
		 
		/// <summary>
		/// Device can possibly execute multiple kernels concurrently. <para/>
		/// 1 if the device supports executing multiple kernels
		/// within the same context simultaneously, or 0 if not. It is not guaranteed that multiple kernels will be resident on
		/// the device concurrently so this feature should not be relied upon for correctness.
		/// </summary>
		ConcurrentKernels = 31, 
		 
		/// <summary>
		/// Device has ECC support enabled. 1 if error correction is enabled on the device, 0 if error correction
		/// is disabled or not supported by the device.
		/// </summary>
		ECCEnabled = 32, 
		 
		/// <summary>
		/// PCI bus ID of the device
		/// </summary>
		PCIBusID = 33, 
		 
		/// <summary>
		/// PCI device ID of the device
		/// </summary>
		PCIDeviceID = 34,

		/// <summary>
		/// Device is using TCC driver model
		/// </summary>
		TCCDriver = 35,

		/// <summary>
		/// Peak memory clock frequency in kilohertz
		/// </summary>
		MemoryClockRate = 36,

		/// <summary>
		/// Global memory bus width in bits
		/// </summary>
		GlobalMemoryBusWidth = 37,

		/// <summary>
		/// Size of L2 cache in bytes
		/// </summary>
		L2CacheSize = 38,

		/// <summary>
		/// Maximum resident threads per multiprocessor
		/// </summary>
		MaxThreadsPerMultiProcessor = 39,

		/// <summary>
		/// Number of asynchronous engines
		/// </summary>
		AsyncEngineCount = 40,

		/// <summary>
		/// Device shares a unified address space with the host
		/// </summary>
		UnifiedAddressing = 41,

		/// <summary>
		/// Maximum 1D layered texture width
		/// </summary>
		MaximumTexture1DLayeredWidth = 42,

		/// <summary>
		/// Maximum layers in a 1D layered texture
		/// </summary>
		MaximumTexture1DLayeredLayers = 43,

		/// <summary>
		/// PCI domain ID of the device
		/// </summary>
		PCIDomainID = 50,

		/// <summary>
		/// Pitch alignment requirement for textures
		/// </summary>
		TexturePitchAlignment = 51,
		/// <summary>
		/// Maximum cubemap texture width/height
		/// </summary>
		MaximumTextureCubeMapWidth = 52,      
		/// <summary>
		/// Maximum cubemap layered texture width/height
		/// </summary>
		MaximumTextureCubeMapLayeredWidth = 53,  
		/// <summary>
		/// Maximum layers in a cubemap layered texture
		/// </summary>
		MaximumTextureCubeMapLayeredLayers = 54, 
		/// <summary>
		/// Maximum 1D surface width
		/// </summary>
		MaximumSurface1DWidth = 55,           
		/// <summary>
		/// Maximum 2D surface width
		/// </summary>
		MaximumSurface2DWidth = 56,           
		/// <summary>
		/// Maximum 2D surface height
		/// </summary>
		MaximumSurface2DHeight = 57,          
		/// <summary>
		/// Maximum 3D surface width
		/// </summary>
		MaximumSurface3DWidth = 58,           
		/// <summary>
		/// Maximum 3D surface height
		/// </summary>
		MaximumSurface3DHeight = 59,          
		/// <summary>
		/// Maximum 3D surface depth
		/// </summary>
		MaximumSurface3DDepth = 60,           
		/// <summary>
		/// Maximum 1D layered surface width
		/// </summary>
		MaximumSurface1DLayeredWidth = 61,   
		/// <summary>
		/// Maximum layers in a 1D layered surface
		/// </summary>
		MaximumSurface1DLayeredLayers = 62,  
		/// <summary>
		/// Maximum 2D layered surface width
		/// </summary>
		MaximumSurface2DLayeredWidth = 63,   
		/// <summary>
		/// Maximum 2D layered surface height
		/// </summary>
		MaximumSurface2DLayeredHeight = 64,  
		/// <summary>
		/// Maximum layers in a 2D layered surface
		/// </summary>
		MaximumSurface2DLayeredLayers = 65,  
		/// <summary>
		/// Maximum cubemap surface width
		/// </summary>
		MaximumSurfaceCubemapWidth = 66,      
		/// <summary>
		/// Maximum cubemap layered surface width
		/// </summary>
		MaximumSurfaceCubemapLayeredWidth = 67,  
		/// <summary>
		/// Maximum layers in a cubemap layered surface
		/// </summary>
		MaximumSurfaceCubemapLayeredLayers = 68, 
		/// <summary>
		/// Maximum 1D linear texture width
		/// </summary>
		MaximumTexture1DLinearWidth = 69,    
		/// <summary>
		/// Maximum 2D linear texture width
		/// </summary>
		MaximumTexture2DLinearWidth = 70,    
		/// <summary>
		/// Maximum 2D linear texture height
		/// </summary>
		MaximumTexture2DLinearHeight = 71,   
		/// <summary>
		/// Maximum 2D linear texture pitch in bytes
		/// </summary>
		MaximumTexture2DLinearPitch = 72,
		/// <summary>
		/// Maximum mipmapped 2D texture width
		/// </summary>
		MaximumTexture2DMipmappedWidth = 73,
		/// <summary>
		/// Maximum mipmapped 2D texture height
		/// </summary>
		MaximumTexture2DMipmappedHeight = 74,
		/// <summary>
		/// Major compute capability version number
		/// </summary>
		ComputeCapabilityMajor = 75, 
		/// <summary>
		/// Minor compute capability version number
		/// </summary>
		ComputeCapabilityMinor = 76, 
		/// <summary>
		/// Maximum mipmapped 1D texture width
		/// </summary>
		MaximumTexture1DMipmappedWidth = 77,
		/// <summary>
		/// Device supports stream priorities
		/// </summary>
		StreamPrioritiesSupported = 78,
		/// <summary>
		/// Device supports caching globals in L1
		/// </summary>
		GlobalL1CacheSupported = 79,
		/// <summary>
		/// Device supports caching locals in L1
		/// </summary>
		LocalL1CacheSupported = 80,
		/// <summary>
		/// Maximum shared memory available per multiprocessor in bytes
		/// </summary>
		MaxSharedMemoryPerMultiprocessor = 81,
		/// <summary>
		/// Maximum number of 32-bit registers available per multiprocessor
		/// </summary>
		MaxRegistersPerMultiprocessor = 82,
		/// <summary>
		/// Device can allocate managed memory on this system
		/// </summary>
		ManagedMemory = 83,
		/// <summary>
		/// Device is on a multi-GPU board
		/// </summary>
		MultiGpuBoard = 84,
		/// <summary>
		/// Unique id for a group of devices on the same multi-GPU board
		/// </summary>
		MultiGpuBoardGroupID = 85,
		/// <summary>
		/// Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware)
		/// </summary>
		HostNativeAtomicSupported = 86,
		/// <summary>
		/// Ratio of single precision performance (in floating-point operations per second) to double precision performance
		/// </summary>
		SingleToDoublePrecisionPerfRatio = 87,
		/// <summary>
		/// Device supports coherently accessing pageable memory without calling cudaHostRegister on it
		/// </summary>
		PageableMemoryAccess = 88,
		/// <summary>
		/// Device can coherently access managed memory concurrently with the CPU
		/// </summary>
		ConcurrentManagedAccess = 89,
		/// <summary>
		/// Device supports compute preemption.
		/// </summary>
		ComputePreemptionSupported = 90,
        /// <summary>
        /// Device can access host registered memory at the same virtual address as the CPU.
        /// </summary>
        CanUseHostPointerForRegisteredMem = 91,
        /// <summary>
        /// ::cuStreamBatchMemOp and related APIs are supported.
        /// </summary>
        CanUseStreamMemOps = 92,
        /// <summary>
        /// 64-bit operations are supported in ::cuStreamBatchMemOp and related APIs.
        /// </summary>
        CanUse64BitStreamMemOps = 93,
        /// <summary>
        /// ::CU_STREAM_WAIT_VALUE_NOR is supported.
        /// </summary>
        CanUseStreamWaitValueNOr = 94,
        /// <summary>
        /// Device supports launching cooperative kernels via ::cuLaunchCooperativeKernel
        /// </summary>
        CooperativeLaunch = 95,
        /// <summary>
        /// Device can participate in cooperative kernels launched via ::cuLaunchCooperativeKernelMultiDevice
        /// </summary>
        CooperativeMultiDeviceLaunch = 96,
        /// <summary>
        /// Maximum optin shared memory per block
        /// </summary>
        MaxSharedMemoryPerBlockOptin = 97,
        /// <summary>
        /// Both the ::CU_STREAM_WAIT_VALUE_FLUSH flag and the ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device. See \ref CUDA_MEMOP for additional details.
        /// </summary>
        CanFlushRemoteWrites = 98,
        /// <summary>
        /// Device supports host memory registration via ::cudaHostRegister.
        /// </summary>
        HostRegisterSupported = 99,
        /// <summary>
        /// Device accesses pageable memory via the host's page tables.
        /// </summary>
        PageableMemoryAccessUsesHostPageTables = 100, 
        /// <summary>
        /// The host can directly access managed memory on the device without migration.
        /// </summary>
        DirectManagedMemoryAccessFromHost = 101,
		/// <summary>
		/// Device supports virtual address management APIs like ::cuMemAddressReserve, ::cuMemCreate, ::cuMemMap and related APIs
		/// </summary>
		VirtualAddressManagementSupported = 102,
		/// <summary>
		/// Device supports exporting memory to a posix file descriptor with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
		/// </summary>
		HandleTypePosixFileDescriptorSupported = 103,
		/// <summary>
		/// Device supports exporting memory to a Win32 NT handle with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate
		/// </summary>
		HandleTypeWin32HandleSupported = 104,
		/// <summary>
		/// Device supports exporting memory to a Win32 KMT handle with ::cuMemExportToShareableHandle, if requested ::cuMemCreate
		/// </summary>
		HandleTypeWin32KMTHandleSupported = 105, 


		/// <summary>
		/// Max elems...
		/// </summary>
		MAX
	}

	/// <summary>
	/// Texture reference filtering modes
	/// </summary>
	public enum CUFilterMode
	{
		/// <summary>
		/// Point filter mode
		/// </summary>
		Point = 0,

		/// <summary>
		/// Linear filter mode
		/// </summary>
		Linear = 1
	}

	/// <summary>
	/// Function properties
	/// </summary>
	public enum CUFunctionAttribute
	{
		/// <summary>
		/// <para>The number of threads beyond which a launch of the function would fail.</para>
		/// <para>This number depends on both the function and the device on which the
		/// function is currently loaded.</para>
		/// </summary>
		MaxThreadsPerBlock = 0,
		
		/// <summary>
		/// <para>The size in bytes of statically-allocated shared memory required by
		/// this function. </para><para>This does not include dynamically-allocated shared
		/// memory requested by the user at runtime.</para>
		/// </summary>
		SharedSizeBytes = 1,
		
		/// <summary>
		/// <para>The size in bytes of statically-allocated shared memory required by
		/// this function. </para><para>This does not include dynamically-allocated shared
		/// memory requested by the user at runtime.</para>
		/// </summary>
		ConstSizeBytes = 2,
		
		/// <summary>
		/// The size in bytes of thread local memory used by this function.
		/// </summary>
		LocalSizeBytes = 3,
	   
		/// <summary>
		/// The number of registers used by each thread of this function.
		/// </summary>
		NumRegs = 4,
		
		/// <summary>
		/// The PTX virtual architecture version for which the function was
		/// compiled. This value is the major PTX version * 10 + the minor PTX version, so a PTX version 1.3 function
		/// would return the value 13. Note that this may return the undefined value of 0 for cubins compiled prior to CUDA
		/// 3.0.
		/// </summary>
		PTXVersion = 5,

		/// <summary>
		/// The binary version for which the function was compiled. This
		/// value is the major binary version * 10 + the minor binary version, so a binary version 1.3 function would return
		/// the value 13. Note that this will return a value of 10 for legacy cubins that do not have a properly-encoded binary
		/// architecture version.
		/// </summary>
		BinaryVersion = 6,

		/// <summary>
		/// The attribute to indicate whether the function has been compiled with 
		/// user specified option "-Xptxas --dlcm=ca" set.
		/// </summary>
		CacheModeCA = 7,
        
        /// <summary>
        /// The maximum size in bytes of dynamically-allocated shared memory that can be used by
        /// this function. If the user-specified dynamic shared memory size is larger than this
        /// value, the launch will fail.
        /// </summary>
        MaxDynamicSharedSizeBytes = 8,

        /// <summary>
        /// On devices where the L1 cache and shared memory use the same hardware resources, 
        /// this sets the shared memory carveout preference, in percent of the total resources. 
        /// This is only a hint, and the driver can choose a different ratio if required to execute the function.
        /// </summary>
        PreferredSharedMemoryCarveout = 9,

        /// <summary>
        /// No descritption found...
        /// </summary>
        Max
    }

	/// <summary>
	/// Function cache configurations
	/// </summary>
	public enum CUFuncCache
	{
		/// <summary>
		/// No preference for shared memory or L1 (default)
		/// </summary>
		PreferNone   = 0x00,
		/// <summary>
		/// Function prefers larger shared memory and smaller L1 cache.
		/// </summary>
		PreferShared = 0x01,
		/// <summary>
		/// Function prefers larger L1 cache and smaller shared memory.
		/// </summary>
		PreferL1     = 0x02,
		/// <summary>
		/// Function prefers equal sized L1 cache and shared memory.
		/// </summary>
		PreferEqual     = 0x03
	}

	/// <summary>
	/// Cubin matching fallback strategies
	/// </summary>
	public enum CUJITFallback
	{
		/// <summary>
		/// Prefer to compile ptx if exact binary match not found
		/// </summary>
		PTX = 0,

		/// <summary>
		/// Prefer to fall back to compatible binary code if exact binary match not found
		/// </summary>
		Binary
	}

	/// <summary>
	/// Online compiler options
	/// </summary>
	public enum CUJITOption
	{
		/// <summary>
		/// <para>Max number of registers that a thread may use.</para>
		/// <para>Option type: unsigned int</para>
		/// <para>Applies to: compiler only</para>
		/// </summary>
		MaxRegisters = 0,
		
		/// <summary>
		/// <para>IN: Specifies minimum number of threads per block to target compilation
		/// for</para>
		/// <para>OUT: Returns the number of threads the compiler actually targeted.
		/// This restricts the resource utilization fo the compiler (e.g. max
		/// registers) such that a block with the given number of threads should be
		/// able to launch based on register limitations. Note, this option does not
		/// currently take into account any other resource limitations, such as
		/// shared memory utilization.</para>
		/// <para>Option type: unsigned int</para>
		/// <para>Applies to: compiler only</para>
		/// </summary>
		ThreadsPerBlock,
		
		/// <summary>
		/// Returns a float value in the option of the wall clock time, in
		/// milliseconds, spent creating the cubin<para/>
		/// Option type: float
		/// <para>Applies to: compiler and linker</para>
		/// </summary>
		WallTime,
		
		/// <summary>
		/// <para>Pointer to a buffer in which to print any log messsages from PTXAS
		/// that are informational in nature (the buffer size is specified via
		/// option ::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES)</para>
		/// <para>Option type: char*</para>
		/// <para>Applies to: compiler and linker</para>
		/// </summary>
		InfoLogBuffer,
		
		/// <summary>
		/// <para>IN: Log buffer size in bytes.  Log messages will be capped at this size
		/// (including null terminator)</para>
		/// <para>OUT: Amount of log buffer filled with messages</para>
		/// <para>Option type: unsigned int</para>
		/// <para>Applies to: compiler and linker</para>
		/// </summary>
		InfoLogBufferSizeBytes,
		
		/// <summary>
		/// <para>Pointer to a buffer in which to print any log messages from PTXAS that
		/// reflect errors (the buffer size is specified via option
		/// ::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)</para>
		/// <para>Option type: char*</para>
		/// <para>Applies to: compiler and linker</para>
		/// </summary>
		ErrorLogBuffer,
		
		/// <summary>
		/// <para>IN: Log buffer size in bytes.  Log messages will be capped at this size
		/// (including null terminator)</para>
		/// <para>OUT: Amount of log buffer filled with messages</para>
		/// <para>Option type: unsigned int</para>
		/// <para>Applies to: compiler and linker</para>
		/// </summary>
		ErrorLogBufferSizeBytes,
		
		
		/// <summary>
		/// <para>Level of optimizations to apply to generated code (0 - 4), with 4
		/// being the default and highest level of optimizations.</para>
		/// <para>Option type: unsigned int</para>
		/// <para>Applies to: compiler only</para>
		/// </summary>
		OptimizationLevel,
		
		/// <summary>
		/// <para>No option value required. Determines the target based on the current
		/// attached context (default)</para>
		/// <para>Option type: No option value needed</para>
		/// <para>Applies to: compiler and linker</para>
		/// </summary>
		TargetFromContext,
		
		/// <summary>
		/// <para>Target is chosen based on supplied ::CUjit_target_enum. This option cannot be
        /// used with cuLink* APIs as the linker requires exact matches.</para>
		/// <para>Option type: unsigned int for enumerated type ::CUjit_target_enum</para>
		/// <para>Applies to: compiler and linker</para>
		/// </summary>
		Target,
		
		/// <summary>
		/// <para>Specifies choice of fallback strategy if matching cubin is not found.
		/// Choice is based on supplied ::CUjit_fallback_enum.</para>
		/// <para>Option type: unsigned int for enumerated type ::CUjit_fallback_enum</para>
		/// <para>Applies to: compiler only</para>
		/// </summary>
		FallbackStrategy,

		/// <summary>
		/// Specifies whether to create debug information in output (-g) <para/> (0: false, default)
		/// <para>Option type: int</para>
		/// <para>Applies to: compiler and linker</para>
		/// </summary>
		GenerateDebugInfo,

		/// <summary>
		/// Generate verbose log messages <para/> (0: false, default)
		/// <para>Option type: int</para>
		/// <para>Applies to: compiler and linker</para>
		/// </summary>
		LogVerbose,

		/// <summary>
		/// Generate line number information (-lineinfo) <para/> (0: false, default)
		/// <para>Option type: int</para>
		/// <para>Applies to: compiler only</para>
		/// </summary>
		GenerateLineInfo,

		/// <summary>
		/// Specifies whether to enable caching explicitly (-dlcm)<para/>
		/// Choice is based on supplied ::CUjit_cacheMode_enum.
		/// <para>Option type: unsigned int for enumerated type ::CUjit_cacheMode_enum</para>
		/// <para>Applies to: compiler only</para>
		/// </summary>
		JITCacheMode,

        /// <summary>
        /// The below jit options are used for internal purposes only, in this version of CUDA
        /// </summary>
        NewSM3XOpt,

        /// <summary>
        /// The below jit options are used for internal purposes only, in this version of CUDA
        /// </summary>
        FastCompile,

        /// <summary>
        /// Array of device symbol names that will be relocated to the corresponing
        /// host addresses stored in ::CU_JIT_GLOBAL_SYMBOL_ADDRESSES.<para/>
        /// Must contain ::CU_JIT_GLOBAL_SYMBOL_COUNT entries.<para/>
        /// When loding a device module, driver will relocate all encountered
        /// unresolved symbols to the host addresses.<para/>
        /// It is only allowed to register symbols that correspond to unresolved
        /// global variables.<para/>
        /// It is illegal to register the same device symbol at multiple addresses.<para/>
        /// Option type: const char **<para/>
        /// Applies to: dynamic linker only
        /// </summary>
        GlobalSymbolNames,

        /// <summary>
        /// Array of host addresses that will be used to relocate corresponding
        /// device symbols stored in ::CU_JIT_GLOBAL_SYMBOL_NAMES.<para/>
        /// Must contain ::CU_JIT_GLOBAL_SYMBOL_COUNT entries.<para/>
        /// Option type: void **<para/>
        /// Applies to: dynamic linker only
        /// </summary>
        GlobalSymbolAddresses,

        /// <summary>
        /// Number of entries in ::CU_JIT_GLOBAL_SYMBOL_NAMES and
        /// ::CU_JIT_GLOBAL_SYMBOL_ADDRESSES arrays.<para/>
        /// Option type: unsigned int<para/>
        /// Applies to: dynamic linker only
        /// </summary>
        GlobalSymbolCount
    }

	/// <summary>
	/// Online compilation targets
	/// </summary>
	public enum CUJITTarget
	{	
		/// <summary>
		/// Compute device class 2.0
		/// </summary>
		Compute_20 = 20,
		
		/// <summary>
		/// Compute device class 2.1
		/// </summary>
		Compute_21 = 21,

		/// <summary>
		/// Compute device class 3.0
		/// </summary>
		Compute_30 = 30,

		/// <summary>
		/// Compute device class 3.2
		/// </summary>
		Compute_32 = 32,

		/// <summary>
		/// Compute device class 3.5
		/// </summary>
		Compute_35 = 35,

		/// <summary>
		/// Compute device class 3.7
		/// </summary>
		Compute_37 = 37,

		/// <summary>
		/// Compute device class 5.0
		/// </summary>
		Compute_50 = 50,

		/// <summary>
		/// Compute device class 5.2
		/// </summary>
		Compute_52 = 52,
		
		/// <summary>
		/// Compute device class 5.3
		/// </summary>
		/// 
		Compute_53 = 53, 

		/// <summary>
		/// Compute device class 6.0
		/// </summary>
		Compute_60 = 60,

		/// <summary>
		/// Compute device class 6.1
		/// </summary>
		Compute_61 = 61,

		/// <summary>
		/// Compute device class 6.2.
		/// </summary>
		Compute_62 = 62,

        /// <summary>
        /// Compute device class 7.0.
        /// </summary>
        Compute_70 = 70,

		/// <summary>
		/// Compute device class 7.0.
		/// </summary>
		Compute_72 = 72,

		/// <summary>
		/// Compute device class 7.5.
		/// </summary>
		Compute_75 = 75
    }

	/// <summary>
	/// Online compilation optimization levels
	/// </summary>
	public enum CUJITOptimizationLevel
	{
		/// <summary>
		/// No optimization
		/// </summary>
		ZERO = 0,

		/// <summary>
		/// Optimization level 1
		/// </summary>
		ONE = 1,

		/// <summary>
		/// Optimization level 2
		/// </summary>
		TWO = 2,

		/// <summary>
		/// Optimization level 3
		/// </summary>
		THREE = 3,
		/// <summary>
		/// Best, Default
		/// </summary>
		FOUR = 4,
	}

	
	/// <summary>
	/// Caching modes for dlcm 
	/// </summary>
	public enum CUJITCacheMode
	{
		/// <summary>
		/// Compile with no -dlcm flag specified
		/// </summary>
		None = 0,
		/// <summary>
		/// Compile with L1 cache disabled
		/// </summary>
		Cg,
		/// <summary>
		/// Compile with L1 cache enabled
		/// </summary>
		Ca 
	}

	/// <summary>
	/// Device code formats
	/// </summary>
	public enum CUJITInputType
	{
		/// <summary>
		/// Compiled device-class-specific device code
		/// <para>Applicable options: none</para>
		/// </summary>
		Cubin = 0,

		/// <summary>
		/// PTX source code
		/// <para>Applicable options: PTX compiler options</para>
		/// </summary>
		PTX,

		/// <summary>
		/// Bundle of multiple cubins and/or PTX of some device code
		/// <para>Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY</para>
		/// </summary>
		FatBinary,

		/// <summary>
		/// Host object with embedded device code
		/// <para>Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY</para>
		/// </summary>
		Object,

		/// <summary>
		/// Archive of host objects with embedded device code
		/// <para>Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY</para>
		/// </summary>
		Library
	}
		
	/// <summary>
	/// Array indices for cube faces
	/// </summary>
	public enum CUArrayCubemapFace
	{
		/// <summary>
		/// Positive X face of cubemap
		/// </summary>
		PositiveX  = 0x00, 
		
		/// <summary>
		/// Negative X face of cubemap
		/// </summary>
		NegativeX  = 0x01,

		/// <summary>
		/// Positive Y face of cubemap 
		/// </summary>
		PositiveY  = 0x02,

		/// <summary>
		/// Negative Y face of cubemap
		/// </summary>
		NegativeY  = 0x03,

		/// <summary>
		/// Positive Z face of cubemap
		/// </summary>
		PositiveZ  = 0x04, 

		/// <summary>
		/// Negative Z face of cubemap
		/// </summary>
		NegativeZ  = 0x05 
	}

	/// <summary>
	/// Limits
	/// </summary>
	public enum CULimit
	{
		/// <summary>
		/// GPU thread stack size
		/// </summary>
		StackSize = 0,

		/// <summary>
		/// GPU printf FIFO size
		/// </summary>
		PrintfFIFOSize = 1,

		/// <summary>
		/// GPU malloc heap size
		/// </summary>
		MallocHeapSize = 2,

		/// <summary>
		/// GPU device runtime launch synchronize depth
		/// </summary>
		DevRuntimeSyncDepth = 3,

		/// <summary>
		/// GPU device runtime pending launch count
		/// </summary>
		DevRuntimePendingLaunchCount = 4,

        /// <summary>
        /// A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint
        /// </summary>
        MaxL2FetchGranularity = 0x05

    }

    /// <summary>
    /// Memory types
    /// </summary>
    public enum CUMemoryType : uint
	{
		/// <summary>
		/// Host memory
		/// </summary>
		Host   = 0x01,

		/// <summary>
		/// Device memory
		/// </summary>
		Device = 0x02,

		/// <summary>
		/// Array memory
		/// </summary>
		Array = 0x03,

		/// <summary>
		/// Unified device or host memory
		/// </summary>
		Unified = 4
	}

	
	/// <summary>
		/// Resource types
	/// </summary>
	public enum CUResourceType 
	{
		/// <summary>
		/// Array resoure
		/// </summary>
		Array           = 0x00,
		/// <summary>
		/// Mipmapped array resource
		/// </summary>
		MipmappedArray = 0x01,
		/// <summary>
		/// Linear resource
		/// </summary>
		Linear          = 0x02,
		/// <summary>
		/// Pitch 2D resource
		/// </summary>
		Pitch2D         = 0x03
	}

	/// <summary>
	/// Error codes returned by CUDA driver API calls
	/// </summary>
	public enum CUResult
	{
		/// <summary>
		/// No errors
		/// </summary>
		Success = 0,
		
		/// <summary>
		/// Invalid value
		/// </summary>
		ErrorInvalidValue = 1,
		
		/// <summary>
		/// Out of memory
		/// </summary>
		ErrorOutOfMemory = 2,
		
		/// <summary>
		/// Driver not initialized
		/// </summary>
		ErrorNotInitialized = 3,
		
		/// <summary>
		/// Driver deinitialized
		/// </summary>
		ErrorDeinitialized = 4,
		
		/// <summary>
		/// This indicates profiler is not initialized for this run. This can happen when the application is running with external profiling tools
		/// like visual profiler.
		/// </summary>
		ErrorProfilerDisabled = 5,

		/// <summary>
		/// This error return is deprecated as of CUDA 5.0. It is no longer an error
		/// to attempt to enable/disable the profiling via ::cuProfilerStart or
		/// ::cuProfilerStop without initialization.
		/// </summary>
		[Obsolete("deprecated as of CUDA 5.0")]
		ErrorProfilerNotInitialized = 6,

		/// <summary>
		/// This error return is deprecated as of CUDA 5.0. It is no longer an error
		/// to call cuProfilerStart() when profiling is already enabled.
		/// </summary>
		[Obsolete("deprecated as of CUDA 5.0")]
		ErrorProfilerAlreadyStarted = 7,

		/// <summary>
		/// This error return is deprecated as of CUDA 5.0. It is no longer an error
		/// to call cuProfilerStop() when profiling is already disabled.
		/// </summary>
		[Obsolete("deprecated as of CUDA 5.0")]
		ErrorProfilerAlreadyStopped = 8,

		
		/// <summary>
		/// No CUDA-capable device available
		/// </summary>
		ErrorNoDevice = 100,
		
		/// <summary>
		/// Invalid device
		/// </summary>
		ErrorInvalidDevice = 101,

		
		/// <summary>
		/// Invalid kernel image
		/// </summary>
		ErrorInvalidImage = 200,
		
		/// <summary>
		/// Invalid context
		/// </summary>
		ErrorInvalidContext = 201,
		
		/// <summary>
		/// Context already current
		/// </summary>
		[Obsolete("This error return is deprecated as of CUDA 3.2. It is no longer an error to attempt to push the active context via cuCtxPushCurrent().")]
		ErrorContextAlreadyCurrent = 202,
		
		/// <summary>
		/// Map failed
		/// </summary>
		ErrorMapFailed = 205,
		
		/// <summary>
		/// Unmap failed
		/// </summary>
		ErrorUnmapFailed = 206,
		
		/// <summary>
		/// Array is mapped
		/// </summary>
		ErrorArrayIsMapped = 207,
		
		/// <summary>
		/// Already mapped
		/// </summary>
		ErrorAlreadyMapped = 208,
		
		/// <summary>
		/// No binary for GPU
		/// </summary>
		ErrorNoBinaryForGPU = 209,
		
		/// <summary>
		/// Already acquired
		/// </summary>
		ErrorAlreadyAcquired = 210,
		
		/// <summary>
		/// Not mapped
		/// </summary>
		ErrorNotMapped = 211,
		
		/// <summary>
		/// Mapped resource not available for access as an array
		/// </summary>
		ErrorNotMappedAsArray = 212,
		
		/// <summary>
		/// Mapped resource not available for access as a pointer
		/// </summary>
		ErrorNotMappedAsPointer = 213,
		
		/// <summary>
		/// Uncorrectable ECC error detected
		/// </summary>
		ErrorECCUncorrectable = 214,
		
		/// <summary>
		/// CULimit not supported by device
		/// </summary>
		ErrorUnsupportedLimit = 215,
		
		/// <summary>
		/// This indicates that the <see cref="CUcontext"/> passed to the API call can
		/// only be bound to a single CPU thread at a time but is already 
		/// bound to a CPU thread.
		/// </summary>
		ErrorContextAlreadyInUse = 216,

		/// <summary>
		/// This indicates that peer access is not supported across the given devices.
		/// </summary>
		ErrorPeerAccessUnsupported = 217,

		/// <summary>
		/// This indicates that a PTX JIT compilation failed.
		/// </summary>
		ErrorInvalidPtx = 218,

		/// <summary>
		/// This indicates an error with OpenGL or DirectX context.
		/// </summary>
		ErrorInvalidGraphicsContext = 219,
                
		/// <summary>
		/// This indicates that an uncorrectable NVLink error was detected during the execution.
		/// </summary>
        NVLinkUncorrectable = 220,

        /// <summary>
        /// This indicates that the PTX JIT compiler library was not found.
        /// </summary>
        JITCompilerNotFound = 221,

        /// <summary>
        /// Invalid source
        /// </summary>
        ErrorInvalidSource = 300,
		
		/// <summary>
		/// File not found
		/// </summary>
		ErrorFileNotFound = 301,

		/// <summary>
		/// Link to a shared object failed to resolve
		/// </summary>
		ErrorSharedObjectSymbolNotFound = 302,

		/// <summary>
		/// Shared object initialization failed
		/// </summary>
		ErrorSharedObjectInitFailed = 303,

		/// <summary>
		/// OS call failed
		/// </summary>
		ErrorOperatingSystem = 304,
		
		/// <summary>
		/// Invalid handle
		/// </summary>
		ErrorInvalidHandle = 400,
        
        /// <summary>
		/// This indicates that a resource required by the API call is not in a
        /// valid state to perform the requested operation.
		/// </summary>
        ErrorIllegalState = 401,

        /// <summary>
        /// Not found
        /// </summary>
        ErrorNotFound = 500,

		
		/// <summary>
		/// CUDA not ready
		/// </summary>
		ErrorNotReady = 600,
		
		
		/// <summary>
		/// While executing a kernel, the device encountered a
        /// load or store instruction on an invalid memory address.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        ErrorIllegalAddress = 700,
		
		/// <summary>
		/// Launch exceeded resources
		/// </summary>
		ErrorLaunchOutOfResources = 701,
		
		/// <summary>
		/// This indicates that the device kernel took too long to execute. This can
        /// only occur if timeouts are enabled - see the device attribute
        /// ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        ErrorLaunchTimeout = 702,

		/// <summary>
		/// Launch with incompatible texturing
		/// </summary>
		ErrorLaunchIncompatibleTexturing = 703,

		/// <summary>
		/// This error indicates that a call to <see cref="DriverAPINativeMethods.CudaPeerAccess.cuCtxEnablePeerAccess"/> is
		/// trying to re-enable peer access to a context which has already
		/// had peer access to it enabled.
		/// </summary>
		ErrorPeerAccessAlreadyEnabled = 704,

		/// <summary>
		/// This error indicates that <see cref="DriverAPINativeMethods.CudaPeerAccess.cuCtxDisablePeerAccess"/> is 
		/// trying to disable peer access which has not been enabled yet 
		/// via <see cref="DriverAPINativeMethods.CudaPeerAccess.cuCtxEnablePeerAccess"/>. 
		/// </summary>
		ErrorPeerAccessNotEnabled = 705,

		/// <summary>
		/// This error indicates that the primary context for the specified device
		/// has already been initialized.
		/// </summary>
		ErrorPrimaryContextActice = 708,

		/// <summary>
		/// This error indicates that the context current to the calling thread
		/// has been destroyed using <see cref="DriverAPINativeMethods.ContextManagement.cuCtxDestroy"/>, or is a primary context which
		/// has not yet been initialized. 
		/// </summary>
		ErrorContextIsDestroyed = 709,

		/// <summary>
		/// A device-side assert triggered during kernel execution. The context
		/// cannot be used anymore, and must be destroyed. All existing device 
		/// memory allocations from this context are invalid and must be 
		/// reconstructed if the program is to continue using CUDA.
		/// </summary>
		ErrorAssert = 710,

		/// <summary>
		/// This error indicates that the hardware resources required to enable
		/// peer access have been exhausted for one or more of the devices 
		/// passed to ::cuCtxEnablePeerAccess().
		/// </summary>
		ErrorTooManyPeers = 711,

		/// <summary>
		/// This error indicates that the memory range passed to ::cuMemHostRegister()
		/// has already been registered.
		/// </summary>
		ErrorHostMemoryAlreadyRegistered = 712,

		/// <summary>
		/// This error indicates that the pointer passed to ::cuMemHostUnregister()
		/// does not correspond to any currently registered memory region.
		/// </summary>
		ErrorHostMemoryNotRegistered = 713,

		/// <summary>
		/// While executing a kernel, the device encountered a stack error.
		/// This can be due to stack corruption or exceeding the stack size limit.
		/// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        ErrorHardwareStackError = 714,

		/// <summary>
		/// While executing a kernel, the device encountered an illegal instruction.
		/// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        ErrorIllegalInstruction = 715,

		/// <summary>
		/// While executing a kernel, the device encountered a load or store instruction
		/// on a memory address which is not aligned.
		/// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        ErrorMisalignedAddress = 716,

        /// <summary>
        /// While executing a kernel, the device encountered an instruction
        /// which can only operate on memory locations in certain address spaces
        /// (global, shared, or local), but was supplied a memory address not
        /// belonging to an allowed address space.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        ErrorInvalidAddressSpace = 717,

        /// <summary>
        /// While executing a kernel, the device program counter wrapped its address space.
        /// This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        ErrorInvalidPC = 718,

        /// <summary>
        /// An exception occurred on the device while executing a kernel. Common
        /// causes include dereferencing an invalid device pointer and accessing
        /// out of bounds shared memory. This leaves the process in an inconsistent state and any further CUDA work
        /// will return the same error. To continue using CUDA, the process must be terminated
        /// and relaunched.
        /// </summary>
        ErrorLaunchFailed = 719,
        
        /// <summary>
        /// This error indicates that the number of blocks launched per grid for a kernel that was
        /// launched via either ::cuLaunchCooperativeKernel or ::cuLaunchCooperativeKernelMultiDevice
        /// exceeds the maximum number of blocks as allowed by ::cuOccupancyMaxActiveBlocksPerMultiprocessor
        /// or ::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
        /// as specified by the device attribute ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
        /// </summary>
        ErrorCooperativeLaunchTooLarge = 720,




        //Removed in update CUDA version 3.1 -> 3.2
        ///// <summary>
        ///// Attempted to retrieve 64-bit pointer via 32-bit API function
        ///// </summary>
        //ErrorPointerIs64Bit = 800,

        ///// <summary>
        ///// Attempted to retrieve 64-bit size via 32-bit API function
        ///// </summary>
        //ErrorSizeIs64Bit = 801, 

        /// <summary>
        /// This error indicates that the attempted operation is not permitted.
        /// </summary>
        ErrorNotPermitted = 800,

		/// <summary>
		/// This error indicates that the attempted operation is not supported
		/// on the current system or device.
		/// </summary>
		ErrorNotSupported = 801,

        /// <summary>
        /// This error indicates that the system is not yet ready to start any CUDA
        /// work.  To continue using CUDA, verify the system configuration is in a
        /// valid state and all required driver daemons are actively running.
        /// </summary>
        ErrorSystemNotReady = 802,

		/// <summary>
		/// This error indicates that there is a mismatch between the versions of
		/// the display driver and the CUDA driver. Refer to the compatibility documentation
		/// for supported versions.
		/// </summary>
		ErrorSystemDriverMismatch = 803,

		/// <summary>
		/// This error indicates that the system was upgraded to run with forward compatibility
		/// but the visible hardware detected by CUDA does not support this configuration.
		/// Refer to the compatibility documentation for the supported hardware matrix or ensure
		/// that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES
		/// environment variable.
		/// </summary>
		ErrorCompatNotSupportedOnDevice = 804,

		/// <summary>
		/// This error indicates that the operation is not permitted when the stream is capturing.
		/// </summary>
		ErrorStreamCaptureUnsupported = 900,

        /// <summary>
        /// This error indicates that the current capture sequence on the stream
        /// has been invalidated due to a previous error.
        /// </summary>
        ErrorStreamCaptureInvalidated = 901,

        /// <summary>
        /// This error indicates that the operation would have resulted in a merge of two independent capture sequences.
        /// </summary>
        ErrorStreamCaptureMerge = 902,

        /// <summary>
        /// This error indicates that the capture was not initiated in this stream.
        /// </summary>
        ErrorStreamCaptureUnmatched = 903,

        /// <summary>
        /// This error indicates that the capture sequence contains a fork that was not joined to the primary stream.
        /// </summary>
        ErrorStreamCaptureUnjoined = 904,

        /// <summary>
        /// This error indicates that a dependency would have been created which
        /// crosses the capture sequence boundary. Only implicit in-stream ordering
        /// dependencies are allowed to cross the boundary.
        /// </summary>
        ErrorStreamCaptureIsolation = 905,

        /// <summary>
        /// This error indicates a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy.
        /// </summary>
        ErrorStreamCaptureImplicit = 906,

		/// <summary>
		/// This error indicates that the operation is not permitted on an event which
		/// was last recorded in a capturing stream.
		/// </summary>
		ErrorCapturedEvent = 907,
		
		/// <summary>
		/// A stream capture sequence not initiated with the ::CU_STREAM_CAPTURE_MODE_RELAXED
		/// argument to ::cuStreamBeginCapture was passed to ::cuStreamEndCapture in a
		/// different thread.
		/// </summary>
		ErrorStreamCaptureWrongThread = 908,

		/// <summary>
		/// This error indicates that the timeout specified for the wait operation has lapsed.
		/// </summary>
		ErrorTimeOut = 909,

		/// <summary>
		/// This error indicates that the graph update was not performed because it included 
		/// changes which violated constraints specific to instantiated graph update.
		/// </summary>
		ErrorGraphExecUpdateFailure = 910,

		/// <summary>
		/// Unknown error
		/// </summary>
		ErrorUnknown = 999,
	}
	
	/// <summary>
	/// P2P Attributes
	/// </summary>
	public enum CUdevice_P2PAttribute
	{
		/// <summary>
		/// A relative value indicating the performance of the link between two devices
		/// </summary>
		PerformanceRank = 0x01,
		/// <summary>
		/// P2P Access is enable
		/// </summary>
		AccessSupported = 0x02, 
		/// <summary>
		/// Atomic operation over the link supported
		/// </summary>
		NativeAtomicSupported = 0x03 ,
        /// <summary>
        /// \deprecated use CudaArrayAccessAccessSupported instead
        /// </summary>
        [Obsolete("use CudaArrayAccessAccessSupported instead")]
        AccessAccessSupported = 0x04,
        /// <summary>
        /// Accessing CUDA arrays over the link supported
        /// </summary>
        CudaArrayAccessAccessSupported = 0x04
   
	}
	
	/// <summary>
	/// CUTexRefSetArrayFlags
	/// </summary>
	public enum CUTexRefSetArrayFlags
	{
		/// <summary>
		/// 
		/// </summary>
		None = 0,
		/// <summary>
		/// Override the texref format with a format inferred from the array.
		/// <para/>Flag for <see cref="DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetArray"/>.
		/// </summary>
		OverrideFormat = 1
	}
	
	/// <summary>
	/// CUParameterTexRef
	/// </summary>
	public enum CUParameterTexRef
	{
		/// <summary>
		/// For texture references loaded into the module, use default texunit from texture reference.
		/// </summary>
		Default = -1
	}
	
	/// <summary>
	/// CUSurfRefSetFlags
	/// </summary>
	public enum CUSurfRefSetFlags
	{
		/// <summary>
		/// Currently no CUSurfRefSetFlags flags are defined.
		/// </summary>
		None = 0
	}
	
	/// <summary>
	/// Pointer information
	/// </summary>
	public enum CUPointerAttribute
	{
		/// <summary>
		/// The <see cref="CUcontext"/> on which a pointer was allocated or registered
		/// </summary>
		Context = 1, 

		/// <summary>
		/// The <see cref="CUMemoryType"/> describing the physical location of a pointer 
		/// </summary>
		MemoryType = 2,    

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the device 
		/// </summary>
		DevicePointer = 3, 

		/// <summary>
		/// The address at which a pointer's memory may be accessed on the host 
		/// </summary>
		HostPointer = 4,

		/// <summary>
		/// A pair of tokens for use with the nv-p2p.h Linux kernel interface
		/// </summary>
		P2PTokens = 5,

		/// <summary>
		/// Synchronize every synchronous memory operation initiated on this region
		/// </summary>
		SyncMemops = 6,

		/// <summary>
		/// A process-wide unique ID for an allocated memory region
		/// </summary>
		BufferID = 7,

		/// <summary>
		/// Indicates if the pointer points to managed memory
		/// </summary>
		IsManaged = 8,

        /// <summary>
        /// A device ordinal of a device on which a pointer was allocated or registered
        /// </summary>
        DeviceOrdinal = 9,

		/// <summary>
		/// 1 if this pointer maps to an allocation that is suitable for ::cudaIpcGetMemHandle, 0 otherwise
		/// </summary>
		IsLegacyCudaIPCCapable = 10, 

		/// <summary>
		/// Starting address for this requested pointer
		/// </summary>
		RangeStartAddr = 11,          

		/// <summary>
		/// Size of the address range for this requested pointer
		/// </summary>
		RangeSize = 12, 
		
		/// <summary>
		/// 1 if this pointer is in a valid address range that is mapped to a backing allocation, 0 otherwise
		/// </summary>
		Mapped = 13,    
		
		/// <summary>
		/// Bitmask of allowed ::CUmemAllocationHandleType for this allocation
		/// </summary>
		AllowedHandleTypes = 14       

	}

	/// <summary>
	/// CUDA devices corresponding to a D3D11, D3D10 or D3D9 device
	/// </summary>
	public enum CUd3dXDeviceList
	{
		/// <summary>
		/// The CUDA devices for all GPUs used by a D3D11 device.
		/// </summary>
		All = 0x01,
		/// <summary>
		/// The CUDA devices for the GPUs used by a D3D11 device in its currently rendering frame (in SLI).
		/// </summary>
		CurrentFrame = 0x02,
		/// <summary>
		/// The CUDA devices for the GPUs to be used by a D3D11 device in the next frame (in SLI).
		/// </summary>
		NextFrame = 0x03
	}

	/// <summary>
	/// CUDA devices corresponding to an OpenGL device.
	/// </summary>
	public enum CUGLDeviceList
	{
		/// <summary>
		/// The CUDA devices for all GPUs used by the current OpenGL context
		/// </summary>
		All = 0x01,
		/// <summary>
		/// The CUDA devices for the GPUs used by the current OpenGL context in its currently rendering frame
		/// </summary>
		CurrentFrame = 0x02,
		/// <summary>
		/// The CUDA devices for the GPUs to be used by the current OpenGL context in the next frame
		/// </summary>
		NextFrame = 0x03,
	} 

	/// <summary>
	/// Shared memory configurations
	/// </summary>
	public enum CUsharedconfig 
	{
		/// <summary>
		/// set default shared memory bank size 
		/// </summary>
		DefaultBankSize    = 0x00,
		/// <summary>
		/// set shared memory bank width to four bytes
		/// </summary>
		FourByteBankSize  = 0x01,
		/// <summary>
		/// set shared memory bank width to eight bytes
		/// </summary>
		EightByteBankSize = 0x02
	}

	/// <summary>
	/// CUipcMem_flags
	/// </summary>
	public enum CUipcMem_flags
	{
		/// <summary>
		/// Automatically enable peer access between remote devices as needed
		/// </summary>
		LazyEnablePeerAccess = 0x1
	} 

	/// <summary>
	/// Resource view format
	/// </summary>
	public enum CUresourceViewFormat
	{
		/// <summary>
		/// No resource view format (use underlying resource format)
		/// </summary>
		None = 0x00,
		/// <summary>
		/// 1 channel unsigned 8-bit integers
		/// </summary>
		Uint1X8 = 0x01,
		/// <summary>
		/// 2 channel unsigned 8-bit integers
		/// </summary>
		Uint2X8 = 0x02,
		/// <summary>
		/// 4 channel unsigned 8-bit integers
		/// </summary>
		Uint4X8 = 0x03,
		/// <summary>
		/// 1 channel signed 8-bit integers
		/// </summary>
		Sint1X8 = 0x04,
		/// <summary>
		/// 2 channel signed 8-bit integers
		/// </summary>
		Sint2X8 = 0x05,
		/// <summary>
		/// 4 channel signed 8-bit integers
		/// </summary>
		Sint4X8 = 0x06,
		/// <summary>
		/// 1 channel unsigned 16-bit integers
		/// </summary>
		Uint1X16 = 0x07,
		/// <summary>
		/// 2 channel unsigned 16-bit integers
		/// </summary>
		Uint2X16 = 0x08,
		/// <summary>
		/// 4 channel unsigned 16-bit integers
		/// </summary>
		Uint4X16 = 0x09,
		/// <summary>
		/// 1 channel signed 16-bit integers
		/// </summary>
		Sint1X16 = 0x0a,
		/// <summary>
		/// 2 channel signed 16-bit integers
		/// </summary>
		Sint2X16 = 0x0b,
		/// <summary>
		/// 4 channel signed 16-bit integers
		/// </summary>
		Sint4X16 = 0x0c,
		/// <summary>
		/// 1 channel unsigned 32-bit integers
		/// </summary>
		Uint1X32 = 0x0d,
		/// <summary>
		/// 2 channel unsigned 32-bit integers
		/// </summary>
		Uint2X32 = 0x0e,
		/// <summary>
		/// 4 channel unsigned 32-bit integers 
		/// </summary>
		Uint4X32 = 0x0f,
		/// <summary>
		/// 1 channel signed 32-bit integers
		/// </summary>
		Sint1X32 = 0x10,
		/// <summary>
		/// 2 channel signed 32-bit integers
		/// </summary>
		Sint2X32 = 0x11,
		/// <summary>
		/// 4 channel signed 32-bit integers
		/// </summary>
		Sint4X32 = 0x12,
		/// <summary>
		/// 1 channel 16-bit floating point
		/// </summary>
		Float1X16 = 0x13,
		/// <summary>
		/// 2 channel 16-bit floating point
		/// </summary>
		Float2X16 = 0x14,
		/// <summary>
		/// 4 channel 16-bit floating point
		/// </summary>
		Float4X16 = 0x15,
		/// <summary>
		/// 1 channel 32-bit floating point
		/// </summary>
		Float1X32 = 0x16,
		/// <summary>
		/// 2 channel 32-bit floating point
		/// </summary>
		Float2X32 = 0x17,
		/// <summary>
		/// 4 channel 32-bit floating point
		/// </summary>
		Float4X32 = 0x18,
		/// <summary>
		/// Block compressed 1 
		/// </summary>
		UnsignedBC1 = 0x19,
		/// <summary>
		/// Block compressed 2
		/// </summary>
		UnsignedBC2 = 0x1a,
		/// <summary>
		/// Block compressed 3 
		/// </summary>
		UnsignedBC3 = 0x1b,
		/// <summary>
		/// Block compressed 4 unsigned
		/// </summary>
		UnsignedBC4 = 0x1c,
		/// <summary>
		/// Block compressed 4 signed 
		/// </summary>
		SignedBC4 = 0x1d,
		/// <summary>
		/// Block compressed 5 unsigned
		/// </summary>
		UnsignedBC5 = 0x1e,
		/// <summary>
		/// Block compressed 5 signed
		/// </summary>
		SignedBC5 = 0x1f,
		/// <summary>
		/// Block compressed 6 unsigned half-float
		/// </summary>
		UnsignedBC6H = 0x20,
		/// <summary>
		/// Block compressed 6 signed half-float
		/// </summary>
		SignedBC6H = 0x21,
		/// <summary>
		/// Block compressed 7 
		/// </summary>
		UnsignedBC7 = 0x22  
	}

	/// <summary>
	/// Profiler Output Modes
	/// </summary>
	public enum CUoutputMode
	{
		/// <summary>
		/// Output mode Key-Value pair format.
		/// </summary>
		KeyValuePair  = 0x00, 
		/// <summary>
		/// Output mode Comma separated values format.
		/// </summary>
		CSV           = 0x01
	}

	/// <summary>
	/// CUDA Mem Attach Flags
	/// </summary>
	public enum CUmemAttach_flags
	{
		/// <summary>
		/// Memory can be accessed by any stream on any device
		/// </summary>
		Global = 1,

		/// <summary>
		/// Memory cannot be accessed by any stream on any device
		/// </summary>
		Host = 2,

		/// <summary>
		/// Memory can only be accessed by a single stream on the associated device
		/// </summary>
		Single = 4
	}



	/// <summary>
	/// Occupancy calculator flag
	/// </summary>
	public enum CUoccupancy_flags
	{
		/// <summary>
		/// Default behavior
		/// </summary>
		Default = 0,

		/// <summary>
		/// Assume global caching is enabled and cannot be automatically turned off
		/// </summary>
		DisableCachingOverride = 1
	}

	//These library types do not really fit in here, but all libraries depend on managedCuda...
	/// <summary>
	/// cudaDataType
	/// </summary>
	public enum cudaDataType
	{
		/// <summary>
		/// 16 bit real 
		/// </summary>
		CUDA_R_16F = 2,

		/// <summary>
		/// 16 bit complex
		/// </summary>
		CUDA_C_16F = 6,

		/// <summary>
		/// 32 bit real
		/// </summary>
		CUDA_R_32F = 0,

		/// <summary>
		/// 32 bit complex
		/// </summary>
		CUDA_C_32F = 4,

		/// <summary>
		/// 64 bit real
		/// </summary>
		CUDA_R_64F = 1,

		/// <summary>
		/// 64 bit complex
		/// </summary>
		CUDA_C_64F = 5,

		/// <summary>
		/// 8 bit real as a signed integer 
		/// </summary>
		CUDA_R_8I = 3,

		/// <summary>
		/// 8 bit complex as a pair of signed integers
		/// </summary>
		CUDA_C_8I = 7,

		/// <summary>
		/// 8 bit real as a signed integer 
		/// </summary>
		CUDA_R_8U = 8,

		/// <summary>
		/// 8 bit complex as a pair of signed integers
		/// </summary>
		CUDA_C_8U= 9
	} 


	/// <summary/>
	public enum libraryPropertyType
	{
		/// <summary/>
		MAJOR_VERSION,
		/// <summary/>
		MINOR_VERSION,
		/// <summary/>
		PATCH_LEVEL
	}

    /// <summary>
    /// Operations for ::cuStreamBatchMemOp
    /// </summary>
    public enum CUstreamBatchMemOpType
    {
        /// <summary>
        /// Represents a ::cuStreamWaitValue32 operation
        /// </summary>
        WaitValue32 = 1,
        /// <summary>
        /// Represents a ::cuStreamWriteValue32 operation
        /// </summary>
        WriteValue32 = 2,
        /// <summary>
        /// Represents a ::cuStreamWaitValue64 operation
        /// </summary>
        WaitValue64 = 4,
        /// <summary>
        /// Represents a ::cuStreamWriteValue64 operation
        /// </summary>
        WriteValue64 = 5,
        /// <summary>
        /// This has the same effect as ::CU_STREAM_WAIT_VALUE_FLUSH, but as a standalone operation.
        /// </summary>
        FlushRemoteWrites = 3 
    }

    /// <summary>
    /// 
    /// </summary>
    public enum CUmem_range_attribute
    {
        /// <summary>
        /// Whether the range will mostly be read and only occassionally be written to
        /// </summary>
        ReadMostly = 1,
        /// <summary>
        /// The preferred location of the range
        /// </summary>
        PreferredLocation = 2,
        /// <summary>
        /// Memory range has ::CU_MEM_ADVISE_SET_ACCESSED_BY set for specified device
        /// </summary>
        AccessedBy = 3,
        /// <summary>
        /// The last location to which the range was prefetched
        /// </summary>
        LastPrefetchLocation = 4
    }

    /// <summary>
    /// Shared memory carveout configurations
    /// </summary>
    public enum CUshared_carveout
    {
        /// <summary>
        /// no preference for shared memory or L1 (default)
        /// </summary>
        Default = -1,
        /// <summary>
        /// prefer maximum available shared memory, minimum L1 cache
        /// </summary>
        MaxShared = 100,
        /// <summary>
        /// prefer maximum available L1 cache, minimum shared memory
        /// </summary>
        MaxL1 = 0
    }

    /// <summary>
    /// Graph node types
    /// </summary>
    public enum CUgraphNodeType
    {
        /// <summary>
        /// GPU kernel node
        /// </summary>
        CU_GRAPH_NODE_TYPE_KERNEL = 0, 
        /// <summary>
        /// Memcpy node
        /// </summary>
        CU_GRAPH_NODE_TYPE_MEMCPY = 1, 
        /// <summary>
        /// Memset node
        /// </summary>
        CU_GRAPH_NODE_TYPE_MEMSET = 2, 
        /// <summary>
        /// Host (executable) node
        /// </summary>
        CU_GRAPH_NODE_TYPE_HOST = 3, 
        /// <summary>
        /// Node which executes an embedded graph
        /// </summary>
        CU_GRAPH_NODE_TYPE_GRAPH = 4,
        /// <summary>
        /// Empty (no-op) node
        /// </summary>
        CU_GRAPH_NODE_TYPE_EMPTY = 5, 
        /// <summary>
        /// 
        /// </summary>
        CU_GRAPH_NODE_TYPE_COUNT
    }

    /// <summary>
    /// Possible stream capture statuses returned by ::cuStreamIsCapturing
    /// </summary>
    public enum CUstreamCaptureStatus
    {
        /// <summary>
        /// Stream is not capturing
        /// </summary>
        CU_STREAM_CAPTURE_STATUS_NONE = 0,
        /// <summary>
        /// Stream is actively capturing
        /// </summary>
        CU_STREAM_CAPTURE_STATUS_ACTIVE = 1, 
        /// <summary>
        /// Stream is part of a capture sequence that has been invalidated, but not terminated
        /// </summary>
        CU_STREAM_CAPTURE_STATUS_INVALIDATED = 2
    }
	
	/// <summary>
	/// Possible modes for stream capture thread interactions. For more details see ::cuStreamBeginCapture and ::cuThreadExchangeStreamCaptureMode
	/// </summary>
	public enum CUstreamCaptureMode
	{
		/// <summary>
		/// 
		/// </summary>
		Global = 0,
		/// <summary>
		/// 
		/// </summary>
		Local = 1,
		/// <summary>
		/// 
		/// </summary>
		Relaxed = 2
    }

    /// <summary>
    /// External memory handle types
    /// </summary>
    public enum CUexternalMemoryHandleType
    {
        /// <summary>
        /// Handle is an opaque file descriptor
        /// </summary>
        OpaqueFD = 1,
        /// <summary>
        /// Handle is an opaque shared NT handle
        /// </summary>
        OpaqueWin32 = 2,
        /// <summary>
        /// Handle is an opaque, globally shared handle
        /// </summary>
        OpaqueWin32KMT = 3,
        /// <summary>
        /// Handle is a D3D12 heap object
        /// </summary>
        D3D12Heap = 4,
        /// <summary>
        /// Handle is a D3D12 committed resource
        /// </summary>
        D3D12Resource = 5,
		/// <summary>
		/// Handle is a shared NT handle to a D3D11 resource
		/// </summary>
		D3D11Resource = 6,
		/// <summary>
		/// Handle is a globally shared handle to a D3D11 resource
		/// </summary>
		D3D11ResourceKMT = 7,
		/// <summary>
		/// Handle is an NvSciBuf object
		/// </summary>
		NvSciBuf = 8
	}



    /// <summary>
    /// External semaphore handle types
    /// </summary>
    public enum CUexternalSemaphoreHandleType
    {
        /// <summary>
        /// Handle is an opaque file descriptor
        /// </summary>
        OpaqueFD = 1,
        /// <summary>
        /// Handle is an opaque shared NT handle
        /// </summary>
        OpaqueWin32 = 2,
        /// <summary>
        /// Handle is an opaque, globally shared handle
        /// </summary>
        OpaqueWin32KMT = 3,
        /// <summary>
        /// Handle is a shared NT handle referencing a D3D12 fence object
        /// </summary>
        D3D12DFence = 4,
		/// <summary>
		/// Handle is a shared NT handle referencing a D3D11 fence object
		/// </summary>
		D3D11Fence = 5,
		/// <summary>
		/// Opaque handle to NvSciSync Object
		/// </summary>
		NvSciSync = 6,
		/// <summary>
		/// Handle is a shared NT handle referencing a D3D11 keyed mutex object
		/// </summary>
		D3D11KeyedMutex = 7,
		/// <summary>
		/// Handle is a globally shared handle referencing a D3D11 keyed mutex object
		/// </summary>
		D3D11KeyedMutexKMT = 8
	}


	/// <summary>
	/// Specifies the type of location
	/// </summary>
	public enum CUmemLocationType
	{
		/// <summary>
		/// 
		/// </summary>
		Invalid = 0x0,
		/// <summary>
		/// Location is a device location, thus id is a device ordinal
		/// </summary>
		Device = 0x1
	}

	/// <summary>
	/// Defines the allocation types available
	/// </summary>
	public enum CUmemAllocationType
	{
		/// <summary>
		/// 
		/// </summary>
		Invalid = 0x0,
		/// <summary>
		/// This allocation type is 'pinned', i.e. cannot migrate from its current
		/// location while the application is actively using it
		/// </summary>
		Pinned = 0x1
	}

	/// <summary>
	/// 
	/// </summary>
	public enum CUgraphExecUpdateResult
	{
		/// <summary>
		/// The update succeeded
		/// </summary>
		Success = 0x0, 
		/// <summary>
		/// The update failed for an unexpected reason which is described in the return value of the function
		/// </summary>
		Error = 0x1, 
		/// <summary>
		/// The update failed because the topology changed
		/// </summary>
		ErrorTopologyChanged = 0x2, 
		/// <summary>
		/// The update failed because a node type changed
		/// </summary>
		ErrorNodeTypeChanged = 0x3, 
		/// <summary>
		/// The update failed because the function of a kernel node changed
		/// </summary>
		ErrorFunctionChanged = 0x4, 
		/// <summary>
		/// The update failed because the parameters changed in a way that is not supported
		/// </summary>
		ErrorParametersChanged = 0x5, 
		/// <summary>
		/// The update failed because something about the node is not supported
		/// </summary>
		ErrorNotSupported = 0x6  
	}

	#endregion

	#region Enums (Flags)

	/// <summary>
	/// Flags to register a graphics resource
	/// </summary>
	[Flags]
	public enum CUGraphicsRegisterFlags
	{
		/// <summary>
		/// Specifies no hints about how this resource will be used. 
		/// It is therefore assumed that this resource will be read 
		/// from and written to by CUDA. This is the default value.
		/// </summary>
		None = 0x00,
		/// <summary>
		/// Specifies that CUDA will not write to this resource.
		/// </summary>
		ReadOnly = 0x01,
		/// <summary>
		/// Specifies that CUDA will not read from this resource and 
		/// will write over the entire contents of the resource, so 
		/// none of the data previously stored in the resource will 
		/// be preserved.
		/// </summary>
		WriteDiscard = 0x02,
		/// <summary>
		/// Specifies that CUDA will bind this resource to a surface reference.
		/// </summary>
		SurfaceLDST = 0x04,
		/// <summary>
		/// 
		/// </summary>
		TextureGather = 0x08
	}

	/// <summary>
	/// Flags for mapping and unmapping graphics interop resources
	/// </summary>
	[Flags]
	public enum CUGraphicsMapResourceFlags
	{
		/// <summary>
		/// Specifies no hints about how this resource will be used.
		/// It is therefore assumed that this resource will be read from and written to by CUDA. This is the default value.
		/// </summary>
		None = 0,
		/// <summary>
		/// Specifies that CUDA will not write to this resource.
		/// </summary>
		ReadOnly = 1,
		/// <summary>
		/// Specifies that CUDA will not read from
		/// this resource and will write over the entire contents of the resource, so none of the data previously stored in the
		/// resource will be preserved.
		/// </summary>
		WriteDiscard = 2
	}

	/// <summary>
	/// CUTexRefSetFlags
	/// </summary>
	[Flags]
	public enum CUTexRefSetFlags
	{
		/// <summary>
		/// 
		/// </summary>
		None = 0,
		/// <summary>
		/// Read the texture as integers rather than promoting the values to floats in the range [0,1].
		/// <para/>Flag for <see cref="DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags"/>
		/// </summary>
		ReadAsInteger = 1,

		/// <summary>
		/// Use normalized texture coordinates in the range [0,1) instead of [0,dim).
		/// <para/>Flag for <see cref="DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags"/>
		/// </summary>
		NormalizedCoordinates = 2,

		/// <summary>
		/// Perform sRGB -> linear conversion during texture read.
		/// </summary>
		sRGB = 0x10
	}

	/// <summary>
	/// CUDA driver API initialization flags
	/// </summary>
	[Flags]
	public enum CUInitializationFlags : uint
	{
		/// <summary>
		/// Currently no initialization flags are defined.
		/// </summary>
		None = 0
	}

	/// <summary>
	/// CUDA driver API Context Enable Peer Access flags
	/// </summary>
	[Flags]
	public enum CtxEnablePeerAccessFlags : uint
	{
		/// <summary>
		/// Currently no flags are defined.
		/// </summary>
		None = 0
	}

	/// <summary>
	/// CUDA stream flags
	/// </summary>
	[Flags]
	public enum CUStreamFlags : uint
	{
		/// <summary>
		/// For compatibilty with pre Cuda 5.0, equal to Default
		/// </summary>
		None = 0,
		/// <summary>
		/// Default stream flag
		/// </summary>
		Default = 0x0,
		/// <summary>
		/// Stream does not synchronize with stream 0 (the NULL stream)
		/// </summary>
		NonBlocking = 0x1,
    }

    /// <summary>
    /// CudaCooperativeLaunchMultiDeviceFlags
    /// </summary>
    [Flags]
    public enum CudaCooperativeLaunchMultiDeviceFlags
    {
        /// <summary>
        /// No flags
        /// </summary>
        None = 0,

        /// <summary>
        /// If set, each kernel launched as part of ::cuLaunchCooperativeKernelMultiDevice only
        /// waits for prior work in the stream corresponding to that GPU to complete before the
        /// kernel begins execution.
        /// </summary>
        NoPreLaunchSync = 0x01,

        /// <summary>
        /// If set, any subsequent work pushed in a stream that participated in a call to
        /// ::cuLaunchCooperativeKernelMultiDevice will only wait for the kernel launched on
        /// the GPU corresponding to that stream to complete before it begins execution.
        /// </summary>
        NoPostLaunchSync = 0x02,
    }

    /// <summary>
    /// CUDAArray3DFlags
    /// </summary>
    [Flags]
	public enum CUDAArray3DFlags
	{
		/// <summary>
		/// No flags
		/// </summary>
		None = 0,

		/// <summary>
		/// if set, the CUDA array contains an array of 2D slices and
		/// the Depth member of CUDA_ARRAY3D_DESCRIPTOR specifies the
		/// number of slices, not the depth of a 3D array.
		/// </summary>
		[Obsolete("Since CUDA Version 4.0. Use <Layered> instead")]
		Array2D = 1,

		/// <summary>
		/// if set, the CUDA array contains an array of layers where each layer is either a 1D
		/// or a 2D array and the Depth member of CUDA_ARRAY3D_DESCRIPTOR specifies the number
		/// of layers, not the depth of a 3D array.
		/// </summary>
		Layered = 1,

		/// <summary>
		/// this flag must be set in order to bind a surface reference
		/// to the CUDA array
		/// </summary>
		SurfaceLDST = 2,

		/// <summary>
		/// If set, the CUDA array is a collection of six 2D arrays, representing faces of a cube. The
		/// width of such a CUDA array must be equal to its height, and Depth must be six.
		/// If ::CUDA_ARRAY3D_LAYERED flag is also set, then the CUDA array is a collection of cubemaps
		/// and Depth must be a multiple of six.
		/// </summary>
		Cubemap = 4,

		/// <summary>
		/// This flag must be set in order to perform texture gather operations on a CUDA array.
		/// </summary>
		TextureGather = 8,

		/// <summary>
		/// This flag if set indicates that the CUDA array is a DEPTH_TEXTURE.
		/// </summary>
		DepthTexture = 0x10,

        /// <summary>
        /// This flag indicates that the CUDA array may be bound as a color target in an external graphics API
        /// </summary>
        ColorAttachment = 0x20
    }

	/// <summary>
	/// CUMemHostAllocFlags. All of these flags are orthogonal to one another: a developer may allocate memory that is portable, mapped and/or
	/// write-combined with no restrictions.
	/// </summary>
	[Flags]
	public enum CUMemHostAllocFlags
	{
		/// <summary>
		/// No flags
		/// </summary>
		None = 0,
		/// <summary>
		/// The memory returned by this call will be considered as pinned memory
		/// by all CUDA contexts, not just the one that performed the allocation.
		/// </summary>
		Portable = 1,

		/// <summary>
		/// Maps the allocation into the CUDA address space. The device pointer
		/// to the memory may be obtained by calling <see cref="DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2"/>. This feature is available only on
		/// GPUs with compute capability greater than or equal to 1.1.
		/// </summary>
		DeviceMap = 2,
		
		/// <summary>
		/// Allocates the memory as write-combined (WC). WC memory
		/// can be transferred across the PCI Express bus more quickly on some system configurations, but cannot be read
		/// efficiently by most CPUs. WC memory is a good option for buffers that will be written by the CPU and read by
		/// the GPU via mapped pinned memory or host->device transfers.<para/>
		/// If set, host memory is allocated as write-combined - fast to write,
		/// faster to DMA, slow to read except via SSE4 streaming load instruction
		/// (MOVNTDQA).
		/// </summary>
		WriteCombined = 4
	}

	/// <summary>
	/// Context creation flags. <para></para>
	/// The two LSBs of the flags parameter can be used to control how the OS thread, which owns the CUDA context at
	/// the time of an API call, interacts with the OS scheduler when waiting for results from the GPU.
	/// </summary>
	[Flags]
	public enum CUCtxFlags
	{
		/// <summary>
		/// The default value if the flags parameter is zero, uses a heuristic based on the
		/// number of active CUDA contexts in the process C and the number of logical processors in the system P. If C >
		/// P, then CUDA will yield to other OS threads when waiting for the GPU, otherwise CUDA will not yield while
		/// waiting for results and actively spin on the processor.
		/// </summary>
		SchedAuto = 0,

		/// <summary>
		/// Instruct CUDA to actively spin when waiting for results from the GPU. This can decrease
		/// latency when waiting for the GPU, but may lower the performance of CPU threads if they are performing
		/// work in parallel with the CUDA thread.
		/// </summary>
		SchedSpin = 1,

		/// <summary>
		/// Instruct CUDA to yield its thread when waiting for results from the GPU. This can
		/// increase latency when waiting for the GPU, but can increase the performance of CPU threads performing work
		/// in parallel with the GPU.
		/// </summary>
		SchedYield = 2,

		/// <summary>
		/// Instruct CUDA to block the CPU thread on a synchronization primitive when waiting for the GPU to finish work.
		/// </summary>
		BlockingSync = 4,

		/// <summary>
		/// No description found...
		/// </summary>
		SchedMask = 7,

		/// <summary>
		/// Instruct CUDA to support mapped pinned allocations. This flag must be set in order to allocate pinned host memory that is accessible to the GPU.
		/// </summary>
		MapHost = 8,

		/// <summary>
		/// Instruct CUDA to not reduce local memory after resizing local memory
		/// for a kernel. This can prevent thrashing by local memory allocations when launching many kernels with high
		/// local memory usage at the cost of potentially increased memory usage.
		/// </summary>
		LMemResizeToMax = 16,

		/// <summary>
		/// No description found...
		/// </summary>
		FlagsMask = 0x1f
	}

	/// <summary>
	/// CUMemHostRegisterFlags. All of these flags are orthogonal to one another: a developer may allocate memory that is portable or mapped
	/// with no restrictions.
	/// </summary>
	[Flags]
	public enum CUMemHostRegisterFlags
	{
		/// <summary>
		/// No flags
		/// </summary>
		None = 0,
		/// <summary>
		/// The memory returned by this call will be considered as pinned memory
		/// by all CUDA contexts, not just the one that performed the allocation.
		/// </summary>
		Portable = 1,

		/// <summary>
		/// Maps the allocation into the CUDA address space. The device pointer
		/// to the memory may be obtained by calling <see cref="DriverAPINativeMethods.MemoryManagement.cuMemHostGetDevicePointer_v2"/>. This feature is available only on
		/// GPUs with compute capability greater than or equal to 1.1.
		/// </summary>
		DeviceMap = 2,

		/// <summary>
		/// If set, the passed memory pointer is treated as pointing to some
		/// memory-mapped I/O space, e.g. belonging to a third-party PCIe device.<para/>
		/// On Windows the flag is a no-op.<para/>
		/// On Linux that memory is marked as non cache-coherent for the GPU and
		/// is expected to be physically contiguous.<para/>
		/// On all other platforms, it is not supported and CUDA_ERROR_INVALID_VALUE
		/// is returned.<para/>
		/// </summary>
		IOMemory = 0x04
	}


	/// <summary>
	/// Flag for cuStreamAddCallback()
	/// </summary>
	[Flags]
	public enum CUStreamAddCallbackFlags
	{
		/// <summary>
		/// No flags
		/// </summary>
		None = 0x0,
		///// <summary>
		///// The stream callback blocks the stream until it is done executing.
		///// </summary>
		//Blocking = 0x01,
	}

	/// <summary>
	/// Event creation flags
	/// </summary>
	[Flags]
	public enum CUEventFlags
	{
		/// <summary>
		/// Default event creation flag.
		/// </summary>
		Default = 0,

		/// <summary>
		/// Specifies that event should use blocking synchronization. A CPU thread
		/// that uses <see cref="DriverAPINativeMethods.Events.cuEventSynchronize"/> to wait on an event created with this flag will block until the event has actually
		/// been recorded.
		/// </summary>
		BlockingSync = 1,

		/// <summary>
		/// Event will not record timing data
		/// </summary>
		DisableTiming = 2,

		/// <summary>
		/// Event is suitable for interprocess use. CUEventFlags.DisableTiming must be set
		/// </summary>
		InterProcess = 4
	}

    /// <summary>
    /// Flags for ::cuStreamWaitValue32
    /// </summary>
    [Flags]
    public enum CUstreamWaitValue_flags
    {
        /// <summary>
        /// Wait until (int32_t)(*addr - value) >= 0 (or int64_t for 64 bit values). Note this is a cyclic comparison which ignores wraparound. (Default behavior.) 
        /// </summary>
        GEQ = 0x0,
        /// <summary>
        /// Wait until *addr == value.
        /// </summary>
        EQ = 0x1,
        /// <summary>
        /// Wait until (*addr &amp; value) != 0.
        /// </summary>
        And = 0x2,
        /// <summary>
        /// Wait until ~(*addr | value) != 0. Support for this operation can be
        /// queried with ::cuDeviceGetAttribute() and ::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR. 
        /// Generally, this requires compute capability 7.0 or greater. 
        /// </summary>
        NOr = 0x3,
        /// <summary>
        /// Follow the wait operation with a flush of outstanding remote writes. This
        /// means that, if a remote write operation is guaranteed to have reached the
        /// device before the wait can be satisfied, that write is guaranteed to be
        /// visible to downstream device work. The device is permitted to reorder
        /// remote writes internally. For example, this flag would be required if
        /// two remote writes arrive in a defined order, the wait is satisfied by the
        /// second write, and downstream work needs to observe the first write.
        /// </summary>
        Flush = 1 << 30
    }

    /// <summary>
    /// Flags for ::cuStreamWriteValue32
    /// </summary>
    [Flags]
    public enum CUstreamWriteValue_flags
    {
        /// <summary>
        /// Default behavior
        /// </summary>
        Default = 0x0,
        /// <summary>
        /// Permits the write to be reordered with writes which were issued
        /// before it, as a performance optimization. Normally, ::cuStreamWriteValue32 will provide a memory fence before the
        /// write, which has similar semantics to __threadfence_system() but is scoped to the stream rather than a CUDA thread.
        /// </summary>
        NoMemoryBarrier = 0x1
    }



    /// <summary>
    /// Indicates that the external memory object is a dedicated resource
    /// </summary>
    [Flags]
    public enum CudaExternalMemory
    {
        /// <summary>
        /// No flags
        /// </summary>
        Nothing = 0x0,
        /// <summary>
        /// Indicates that the external memory object is a dedicated resource
        /// </summary>
        Dedicated = 0x01,
    }

	/// <summary>
	/// parameter of ::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS
	/// </summary>
	[Flags]
	public enum CudaExternalSemaphore
	{
		/// <summary>
		/// When the /p flags parameter of ::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS
		/// contains this flag, it indicates that signaling an external semaphore object
		/// should skip performing appropriate memory synchronization operations over all
		/// the external memory objects that are imported as ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF,
		/// which otherwise are performed by default to ensure data coherency with other
		/// importers of the same NvSciBuf memory objects.
		/// </summary>
		SignalSkipNvSciBufMemSync = 0x01,

		/// <summary>
		/// When the /p flags parameter of ::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS
		/// contains this flag, it indicates that waiting on an external semaphore object
		/// should skip performing appropriate memory synchronization operations over all
		/// the external memory objects that are imported as ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF,
		/// which otherwise are performed by default to ensure data coherency with other
		/// importers of the same NvSciBuf memory objects.
		/// </summary>
		WaitSkipNvSciBufMemSync = 0x02,
	}

	/// <summary>
	/// flags of ::cuDeviceGetNvSciSyncAttributes
	/// </summary>
	[Flags]
	public enum NvSciSyncAttr
	{
		/// <summary>
		/// When /p flags of ::cuDeviceGetNvSciSyncAttributes is set to this,
		/// it indicates that application needs signaler specific NvSciSyncAttr
		/// to be filled by ::cuDeviceGetNvSciSyncAttributes.
		/// </summary>
		Signal = 0x01,

		/// <summary>
		/// When /p flags of ::cuDeviceGetNvSciSyncAttributes is set to this,
		/// it indicates that application needs waiter specific NvSciSyncAttr
		/// to be filled by ::cuDeviceGetNvSciSyncAttributes.
		/// </summary>
		Wait = 0x02,
	}

	/// <summary>
	/// Flags for specifying particular handle types
	/// </summary>
	[Flags]
	public enum CUmemAllocationHandleType
	{
		/// <summary>
		/// Allows a file descriptor to be used for exporting. Permitted only on POSIX systems. (int)
		/// </summary>
		PosixFileDescriptor = 0x1,  
		/// <summary>
		/// Allows a Win32 NT handle to be used for exporting. (HANDLE)
		/// </summary>
		Win32 = 0x2, 
		/// <summary>
		/// Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE)
		/// </summary>
		Win32KMT = 0x4
	}

	/// <summary>
	/// Specifies the memory protection flags for mapping.
	/// </summary>
	[Flags]
	public enum CUmemAccess_flags
	{
		/// <summary>
		/// Default, make the address range not accessible
		/// </summary>
		ProtNone = 0x1,
		/// <summary>
		/// Make the address range read accessible
		/// </summary>
		ProtRead = 0x2,
		/// <summary>
		/// Make the address range read-write accessible
		/// </summary>
		ProtReadWrite = 0x3
	}
	

	/// <summary>
	/// Flag for requesting different optimal and required granularities for an allocation.
	/// </summary>
	[Flags]
	public enum CUmemAllocationGranularity_flags
	{
		/// <summary>
		/// Minimum required granularity for allocation
		/// </summary>
		Minimum = 0x0,
		/// <summary>
		/// Recommended granularity for allocation for best performance
		/// </summary>
		Recommended = 0x1
	}

	#endregion


	#region Delegates

	/// <summary>
	/// CUDA stream callback
	/// </summary>
	/// <param name="hStream">The stream the callback was added to, as passed to ::cuStreamAddCallback.  May be NULL.</param>
	/// <param name="status">CUDA_SUCCESS or any persistent error on the stream.</param>
	/// <param name="userData">User parameter provided at registration.</param>
	public delegate void CUstreamCallback(CUstream hStream, CUResult status, IntPtr userData);

	/// <summary>
	/// Block size to per-block dynamic shared memory mapping for a certain
	/// kernel.<para/>
	/// e.g.:
	/// If no dynamic shared memory is used: x => 0<para/>
	/// If 4 bytes shared memory per thread is used: x = 4 * x
	/// </summary>
	/// <param name="aBlockSize">block size</param>
	/// <returns>The dynamic shared memory needed by a block</returns>
	public delegate SizeT del_CUoccupancyB2DSize(int aBlockSize);

    /// <summary>
	/// CUDA host function
	/// </summary>
    /// <param name="userData">Argument value passed to the function</param>
    public delegate void CUhostFn (IntPtr userData);

	#endregion

}
