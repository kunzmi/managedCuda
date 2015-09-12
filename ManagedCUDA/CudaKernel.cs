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
using ManagedCuda.VectorTypes;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace ManagedCuda
{
	/// <summary>
	/// A CUDA function or CUDA kernel
	/// </summary>
	public class CudaKernel
	{
		/// <summary> 
		/// </summary>
		protected CudaContext _cuda;
		/// <summary> 
		/// </summary>
		protected CUmodule _module;
		/// <summary> 
		/// </summary>
		protected CUfunction _function;

		/// <summary> 
		/// </summary>
		protected uint _sharedMemSize;
		/// <summary> 
		/// </summary>
		protected dim3 _blockDim;
		/// <summary> 
		/// </summary>
		protected dim3 _gridDim;
		/// <summary> 
		/// </summary>
		protected string _kernelName;
		/// <summary> 
		/// </summary>
		protected CUResult res;

		/// <summary> 
		/// </summary>
		protected int _maxThreadsPerBlock;
		/// <summary> 
		/// </summary>
		protected int _sharedSizeBytes;
		/// <summary> 
		/// </summary>
		protected int _constSizeBytes;
		/// <summary> 
		/// </summary>
		protected int _localSizeBytes;
		/// <summary> 
		/// </summary>
		protected int _numRegs;
		/// <summary> 
		/// </summary>
		protected Version _ptxVersion;
		/// <summary> 
		/// </summary>
		protected Version _binaryVersion;
		/// <summary> 
		/// </summary>
		protected bool _cacheModeCA;



		#region Contstructors
		/// <summary>
		/// Loads the given CUDA kernel from the CUmodule. Block and Grid dimensions must be set 
		/// before running the kernel. Shared memory size is set to 0.
		/// </summary>
		/// <param name="kernelName">The kernel name as defined in the *.cu file</param>
		/// <param name="module">The CUmodule which contains the kernel</param>
		/// <param name="cuda">CUDA abstraction layer object (= CUDA context) for this Kernel</param>
		public CudaKernel(string kernelName, CUmodule module, CudaContext cuda)
		{
			if (cuda == null) throw new ArgumentNullException("cuda");
			_module = module;
			_cuda = cuda;
			_kernelName = kernelName;


			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetFunction(ref _function, _module, _kernelName);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetFunction", res, _kernelName));
			if (res != CUResult.Success) throw new CudaException(res);

			_blockDim.x = _blockDim.y = 16;
			_blockDim.z = 1;
			_gridDim.x = _gridDim.y = _gridDim.z = 1;

			//Load additional info from kernel image
			
			_maxThreadsPerBlock = 0;
			_sharedSizeBytes = 0;
			_constSizeBytes = 0;
			_localSizeBytes = 0;
			_numRegs = 0;
			int temp = 0;

			res = DriverAPINativeMethods.FunctionManagement.cuFuncGetAttribute(ref _maxThreadsPerBlock, CUFunctionAttribute.MaxThreadsPerBlock, _function);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuFuncGetAttribute", res, _kernelName));
			if (res != CUResult.Success) throw new CudaException(res);

			temp = 0;
			res = DriverAPINativeMethods.FunctionManagement.cuFuncGetAttribute(ref _sharedSizeBytes, CUFunctionAttribute.SharedSizeBytes, _function);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuFuncGetAttribute", res, _kernelName));
			if (res != CUResult.Success) throw new CudaException(res);

			temp = 0;
			res = DriverAPINativeMethods.FunctionManagement.cuFuncGetAttribute(ref _constSizeBytes, CUFunctionAttribute.ConstSizeBytes, _function);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuFuncGetAttribute", res, _kernelName));
			if (res != CUResult.Success) throw new CudaException(res);

			temp = 0;
			res = DriverAPINativeMethods.FunctionManagement.cuFuncGetAttribute(ref _localSizeBytes, CUFunctionAttribute.LocalSizeBytes, _function);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuFuncGetAttribute", res, _kernelName));
			if (res != CUResult.Success) throw new CudaException(res);

			temp = 0;
			res = DriverAPINativeMethods.FunctionManagement.cuFuncGetAttribute(ref _numRegs, CUFunctionAttribute.NumRegs, _function);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuFuncGetAttribute", res, _kernelName));
			if (res != CUResult.Success) throw new CudaException(res);

			temp = 0;
			res = DriverAPINativeMethods.FunctionManagement.cuFuncGetAttribute(ref temp, CUFunctionAttribute.PTXVersion, _function);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuFuncGetAttribute", res, _kernelName));
			if (res != CUResult.Success) throw new CudaException(res);
			_ptxVersion = new Version(temp / 10, temp % 10);

			temp = 0;
			res = DriverAPINativeMethods.FunctionManagement.cuFuncGetAttribute(ref temp, CUFunctionAttribute.BinaryVersion, _function);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuFuncGetAttribute", res, _kernelName));
			if (res != CUResult.Success) throw new CudaException(res);
			_binaryVersion = new Version(temp / 10, temp % 10);

			temp = 0;
			res = DriverAPINativeMethods.FunctionManagement.cuFuncGetAttribute(ref temp, CUFunctionAttribute.CacheModeCA, _function);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuFuncGetAttribute", res, _kernelName));
			if (res != CUResult.Success) throw new CudaException(res);
			_cacheModeCA = temp != 0;
		}

		/// <summary>
		/// Loads the given CUDA kernel from the CUmodule. Block and Grid dimensions are set directly. 
		/// Shared memory size is set to 0.
		/// </summary>
		/// <param name="kernelName">The kernel name as defined in the *.cu file</param>
		/// <param name="module">The CUmodule which contains the kernel</param>
		/// <param name="cuda">CUDA abstraction layer object (= CUDA context) for this Kernel</param>
		/// <param name="blockDim">Dimension of block of threads</param>
		/// <param name="gridDim">Dimension of grid of block of threads</param>
		public CudaKernel(string kernelName, CUmodule module, CudaContext cuda, dim3 blockDim, dim3 gridDim)
			: this(kernelName, module, cuda)
		{
			_blockDim = blockDim;
			_gridDim = gridDim;
		}

		/// <summary>
		/// Loads the given CUDA kernel from the CUmodule. Block dimensions are set directly, 
		/// grid dimensions must be set before running the kernel. Shared memory size is set to 0.
		/// </summary>
		/// <param name="kernelName">The kernel name as defined in the *.cu file</param>
		/// <param name="module">The CUmodule which contains the kernel</param>
		/// <param name="cuda">CUDA abstraction layer object (= CUDA context) for this Kernel</param>
		/// <param name="blockDim">Dimension of block of threads</param>
		public CudaKernel(string kernelName, CUmodule module, CudaContext cuda, dim3 blockDim)
			: this(kernelName, module, cuda)
		{
			_blockDim = blockDim;
		}

		/// <summary>
		/// Loads the given CUDA kernel from the CUmodule. Block and Grid dimensions are set directly. 
		/// Shared memory size is set to 0.
		/// </summary>
		/// <param name="kernelName">The kernel name as defined in the *.cu file</param>
		/// <param name="module">The CUmodule which contains the kernel</param>
		/// <param name="cuda">CUDA abstraction layer object (= CUDA context) for this Kernel</param>
		/// <param name="blockDimX">Dimension of block of threads X</param>
		/// <param name="blockDimY">Dimension of block of threads Y</param>
		/// <param name="blockDimZ">Dimension of block of threads Z</param>
		/// <param name="gridDimX">Dimension of grid of block of threads X</param>
		/// <param name="gridDimY">Dimension of grid of block of threads Y</param>
		public CudaKernel(string kernelName, CUmodule module, CudaContext cuda, uint blockDimX, uint blockDimY, uint blockDimZ, uint gridDimX, uint gridDimY)
			: this(kernelName, module, cuda)
		{
			_blockDim.x = blockDimX;
			_blockDim.y = blockDimY;
			_blockDim.z = blockDimZ;
			_gridDim.x = gridDimX;
			_gridDim.y = gridDimY;
			_gridDim.z = 1;
		}

		/// <summary>
		/// Loads the given CUDA kernel from the CUmodule. Block and Grid dimensions are set directly. 
		/// Shared memory size is set to 0.
		/// </summary>
		/// <param name="kernelName">The kernel name as defined in the *.cu file</param>
		/// <param name="module">The CUmodule which contains the kernel</param>
		/// <param name="cuda">CUDA abstraction layer object (= CUDA context) for this Kernel</param>
		/// <param name="blockDimX">Dimension of block of threads X</param>
		/// <param name="blockDimY">Dimension of block of threads Y</param>
		/// <param name="blockDimZ">Dimension of block of threads Z</param>
		/// <param name="gridDimX">Dimension of grid of block of threads X</param>
		/// <param name="gridDimY">Dimension of grid of block of threads Y</param>
		/// <param name="gridDimZ">Dimension of grid of block of threads Z</param>
		public CudaKernel(string kernelName, CUmodule module, CudaContext cuda, uint blockDimX, uint blockDimY, uint blockDimZ, uint gridDimX, uint gridDimY, uint gridDimZ)
			: this(kernelName, module, cuda)
		{
			_blockDim.x = blockDimX;
			_blockDim.y = blockDimY;
			_blockDim.z = blockDimZ;
			_gridDim.x = gridDimX;
			_gridDim.y = gridDimY;
			_gridDim.z = gridDimZ;
		}

		/// <summary>
		/// Loads the given CUDA kernel from the CUmodule. Block dimensions are set directly, 
		/// grid dimensions must be set before running the kernel. Shared memory size is set to 0.
		/// </summary>
		/// <param name="kernelName">The kernel name as defined in the *.cu file</param>
		/// <param name="module">The CUmodule which contains the kernel</param>
		/// <param name="cuda">CUDA abstraction layer object (= CUDA context) for this Kernel</param>
		/// <param name="blockDimX">Dimension of block of threads X</param>
		/// <param name="blockDimY">Dimension of block of threads Y</param>
		/// <param name="blockDimZ">Dimension of block of threads Z</param>
		public CudaKernel(string kernelName, CUmodule module, CudaContext cuda, uint blockDimX, uint blockDimY, uint blockDimZ)
			: this(kernelName, module, cuda)
		{
			_blockDim.x = blockDimX;
			_blockDim.y = blockDimY;
			_blockDim.z = blockDimZ;
		}

		/// <summary>
		/// Loads the given CUDA kernel from the CUmodule. Block and Grid dimensions must be set 
		/// before running the kernel. Shared memory size is set directly.
		/// </summary>
		/// <param name="kernelName">The kernel name as defined in the *.cu file</param>
		/// <param name="module">The CUmodule which contains the kernel</param>
		/// <param name="cuda">CUDA abstraction layer object (= CUDA context) for this Kernel</param>
		/// <param name="sharedMemory">Dynamic shared memory size in Bytes</param>
		public CudaKernel(string kernelName, CUmodule module, CudaContext cuda, uint sharedMemory)
			: this(kernelName, module, cuda)
		{
			_sharedMemSize = sharedMemory;
		}

		/// <summary>
		/// Loads the given CUDA kernel from the CUmodule. Block and Grid dimensions and shared memory size are set directly.
		/// </summary>
		/// <param name="kernelName">The kernel name as defined in the *.cu file</param>
		/// <param name="module">The CUmodule which contains the kernel</param>
		/// <param name="cuda">CUDA abstraction layer object (= CUDA context) for this Kernel</param>
		/// <param name="blockDim">Dimension of block of threads (2D - z-component is discarded)</param>
		/// <param name="gridDim">Dimension of grid of block of threads (3D)</param>
		/// <param name="sharedMemory">Dynamic shared memory size in Bytes</param>
		public CudaKernel(string kernelName, CUmodule module, CudaContext cuda, dim3 blockDim, dim3 gridDim, uint sharedMemory)
			: this(kernelName, module, cuda, blockDim, gridDim)
		{
			_sharedMemSize = sharedMemory;
		}

		/// <summary>
		/// Loads the given CUDA kernel from the CUmodule. Block dimensions and shared memors size are set directly, 
		/// grid dimensions must be set before running the kernel.
		/// </summary>
		/// <param name="kernelName">The kernel name as defined in the *.cu file</param>
		/// <param name="module">The CUmodule which contains the kernel</param>
		/// <param name="cuda">CUDA abstraction layer object (= CUDA context) for this Kernel</param>
		/// <param name="blockDim">Dimension of block of threads </param>
		/// <param name="sharedMemory">Dynamic shared memory size in Bytes</param>
		public CudaKernel(string kernelName, CUmodule module, CudaContext cuda, dim3 blockDim, uint sharedMemory)
			: this(kernelName, module, cuda, blockDim)
		{
			_sharedMemSize = sharedMemory;
		}

		/// <summary>
		/// Loads the given CUDA kernel from the CUmodule. Block dimensions and shared memors size are set directly, 
		/// grid dimensions must be set before running the kernel.
		/// </summary>
		/// <param name="kernelName">The kernel name as defined in the *.cu file</param>
		/// <param name="module">The CUmodule which contains the kernel</param>
		/// <param name="cuda">CUDA abstraction layer object (= CUDA context) for this Kernel</param>
		/// <param name="blockDimX">Dimension of block of threads X</param>
		/// <param name="blockDimY">Dimension of block of threads Y</param>
		/// <param name="blockDimZ">Dimension of block of threads Z</param>
		/// <param name="gridDimX">Dimension of grid of block of threads X</param>
		/// <param name="gridDimY">Dimension of grid of block of threads Y</param>
		/// <param name="gridDimZ">Dimension of grid of block of threads Z</param>
		/// <param name="sharedMemory">Dynamic shared memory size in Bytes</param>
		public CudaKernel(string kernelName, CUmodule module, CudaContext cuda, uint blockDimX, uint blockDimY, uint blockDimZ, uint gridDimX, uint gridDimY, uint gridDimZ, uint sharedMemory)
			: this(kernelName, module, cuda, blockDimX, blockDimY, blockDimZ, gridDimX, gridDimY, gridDimZ)
		{
			_sharedMemSize = sharedMemory;
		}

		///// <summary>
		///// 
		///// </summary>
		//~CudaKernel()
		//{
		//    Dispose (false);
		//}
		#endregion

		#region Dispose
		///// <summary>
		///// 
		///// </summary>
		//public void Dispose()
		//{
		//    Dispose(true);
		//    GC.SuppressFinalize(this);
		//}

		///// <summary>
		///// 
		///// </summary>
		///// <param name="fDisposing"></param>
		//protected virtual void Dispose (bool fDisposing)
		//{
		//    if (fDisposing && !disposed) 
		//    {
		//        //We don't unload automatically each kernel module, as this is already done during context destruction.
		//        //Unloading the kernel module manually can cause troubles in multi threaded environments, whereas I don't
		//        //really know why.
		//        _cuda.UnloadKernel(this);
		//        disposed = true;
		//    }
		//}
		#endregion

		#region SetConstantVaiable
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable<T>(string name, T value) where T : struct
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));
			if (res != CUResult.Success) throw new CudaException(res);

			GCHandle handle = GCHandle.Alloc(value, GCHandleType.Pinned);
			try
			{
				IntPtr ptr = handle.AddrOfPinnedObject();

				res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ptr, varSize);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));
			}
			finally
			{
				handle.Free();
			}
			if (res != CUResult.Success) throw new CudaException(res);

		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable<T>(string name, T[] value) where T : struct
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));
			if (res != CUResult.Success) throw new CudaException(res);

			GCHandle handle = GCHandle.Alloc(value, GCHandleType.Pinned);
			try
			{
				IntPtr ptr = handle.AddrOfPinnedObject();

				res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ptr, varSize);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));
			}
			finally
			{
				handle.Free();
			}
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, byte value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, sbyte value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, ushort value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, short value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, uint value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, int value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, ulong value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, long value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, float value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, double value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		#region VectorTypes

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, dim3 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, char1 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, char2 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, char3 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, char4 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, uchar1 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, uchar2 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, uchar3 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, uchar4 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, short1 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, short2 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, short3 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, short4 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, ushort1 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, ushort2 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, ushort3 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, ushort4 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, int1 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, int2 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, int3 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, int4 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, uint1 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, uint2 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, uint3 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, uint4 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, long1 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, long2 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, long3 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, long4 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, ulong1 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, ulong2 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, ulong3 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, ulong4 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, float1 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, float2 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, float3 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, float4 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, double1 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, double2 value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, cuDoubleComplex value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, cuDoubleReal value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, cuFloatComplex value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, cuFloatReal value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, ref value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		#endregion
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, byte[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, sbyte[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, ushort[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, short[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, uint[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, int[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, ulong[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, long[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, float[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, double[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		#region VectorTypes
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, dim3[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, char1[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, char2[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, char3[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, char4[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, uchar1[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, uchar2[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, uchar3[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, uchar4[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, short1[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, short2[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, short3[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, short4[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, ushort1[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, ushort2[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, ushort3[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, ushort4[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, int1[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, int2[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, int3[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, int4[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, uint1[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, uint2[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, uint3[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, uint4[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, long1[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, long2[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, long3[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, long4[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, ulong1[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, ulong2[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, ulong3[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, ulong4[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, float1[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, float2[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, float3[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, float4[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, double1[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, double2[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, cuDoubleComplex[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, cuDoubleReal[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, cuFloatComplex[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		/// <summary>
		/// Set the constant variable <c>name</c> to value <c>value</c><para>The constant variable must be defined in the CUDA module.</para>
		/// </summary>
		/// <param name="name">constant variable name</param>
		/// <param name="value">value</param>
		public void SetConstantVariable(string name, cuFloatReal[] value)
		{
			CUdeviceptr dVarPtr = new CUdeviceptr();
			SizeT varSize = 0;
			res = DriverAPINativeMethods.ModuleManagement.cuModuleGetGlobal_v2(ref dVarPtr, ref varSize, _module, name);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuModuleGetGlobal", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(dVarPtr, value, varSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuMemcpyHtoD", res, _kernelName));   
			if (res != CUResult.Success) throw new CudaException(res);
		}
		#endregion
		#endregion

		#region Run methods

		/// <summary>
		/// Executes the kernel on the device
		/// </summary>
		/// <param name="parameters">Parameters as given by the kernel</param>
		/// <returns>Time of execution in milliseconds (using GPU counter)</returns>
		public virtual float Run(params object[] parameters)
		{
			int paramCount = parameters.Length;
			IntPtr[] paramsList = new IntPtr[paramCount];
			GCHandle[] GCHandleList = new GCHandle[paramCount];

			//Get pointers to kernel parameters
			for (int i = 0; i < paramCount; i++)
			{
				GCHandleList[i] = GCHandle.Alloc(parameters[i], GCHandleType.Pinned);
				paramsList[i] = GCHandleList[i].AddrOfPinnedObject();
			}

			//Wait for device to finish previous jobs
			res = DriverAPINativeMethods.ContextManagement.cuCtxSynchronize();
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuCtxSynchronize", res, _kernelName));
			if (res != CUResult.Success) throw new CudaException(res);

			//Create events to measure execution times
			CudaEvent start = new CudaEvent();
			CudaEvent end = new CudaEvent();
			start.Record();

			//Launch the kernel
			res = DriverAPINativeMethods.Launch.cuLaunchKernel(_function, _gridDim.x, _gridDim.y, _gridDim.z, _blockDim.x, _blockDim.y, _blockDim.z, _sharedMemSize, new CUstream(), paramsList, null);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuLaunchKernel", res, _kernelName));
			if (res != CUResult.Success) throw new CudaException(res);

			//Free pinned managed parameters during kernel launch (before sync)
			for (int i = 0; i < paramCount; i++)
			{
				GCHandleList[i].Free();
			}

			end.Record();
			float ms;
			//wait till kernel finished
			end.Synchronize();

			//Get elapsed time
			ms = CudaEvent.ElapsedTime(start, end);

			//Cleanup
			start.Dispose();
			end.Dispose();
			return ms;
		}

		/// <summary>
		/// Executes the kernel on the device asynchronously
		/// </summary>
		/// <param name="stream">Stream</param>
		/// <param name="parameters">Parameters as given by the kernel</param>
		public virtual void RunAsync(CUstream stream, params object[] parameters)
		{
			int paramCount = parameters.Length;
			IntPtr[] paramsList = new IntPtr[paramCount];
			GCHandle[] GCHandleList = new GCHandle[paramCount];

			//Get pointers to kernel parameters
			for (int i = 0; i < paramCount; i++)
			{
				GCHandleList[i] = GCHandle.Alloc(parameters[i], GCHandleType.Pinned);
				paramsList[i] = GCHandleList[i].AddrOfPinnedObject();
			}
			
			//Launch the kernel
			res = DriverAPINativeMethods.Launch.cuLaunchKernel(_function, _gridDim.x, _gridDim.y, _gridDim.z, _blockDim.x, _blockDim.y, _blockDim.z, _sharedMemSize, stream, paramsList, null);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Kernel: {3}", DateTime.Now, "cuLaunchKernel", res, _kernelName));
			if (res != CUResult.Success) throw new CudaException(res);

			//Free pinned managed parameters
			for (int i = 0; i < paramCount; i++)
			{
				GCHandleList[i].Free();
			}
		}
		#endregion

		#region Properties
		/// <summary>
		/// Get or set the thread block dimensions. Block dimenions must be set before the first kernel launch.
		/// </summary>
		public dim3 BlockDimensions
		{
			get { return _blockDim; }
			set { _blockDim = value; }
		}

		/// <summary>
		/// Get or set the thread grid dimensions. Grid dimenions must be set before the first kernel launch.
		/// </summary>
		public dim3 GridDimensions
		{
			get { return _gridDim; }
			set 
			{
				_gridDim = value;
			}
		}

		/// <summary>
		/// Dynamic shared memory size in Bytes. Must be set before the first kernel launch.
		/// </summary>
		public uint DynamicSharedMemory
		{
			get { return _sharedMemSize; }
			set { _sharedMemSize = value; }
		}

		/// <summary>
		/// CUFunction
		/// </summary>
		public CUfunction CUFunction
		{
			get { return _function; }
		}

		/// <summary>
		/// CUModule
		/// </summary>
		public CUmodule CUModule
		{
			get { return _module; }
		}

		/// <summary>
		/// Kernel name as defined in the kernel source code.
		/// </summary>
		public string KernelName
		{
			get { return _kernelName; }
		}

		/// <summary>
		/// <para>The number of threads beyond which a launch of the function would fail.</para>
		/// <para>This number depends on both the function and the device on which the
		/// function is currently loaded.</para>
		/// </summary>
		public int MaxThreadsPerBlock
		{
			get { return _maxThreadsPerBlock; }
		}

		/// <summary>
		/// <para>The size in bytes of statically-allocated shared memory required by
		/// this function. </para><para>This does not include dynamically-allocated shared
		/// memory requested by the user at runtime.</para>
		/// </summary>
		public int SharedMemory
		{
			get { return _sharedSizeBytes; }
		}

		/// <summary>
		/// <para>The size in bytes of statically-allocated shared memory required by
		/// this function. </para><para>This does not include dynamically-allocated shared
		/// memory requested by the user at runtime.</para>
		/// </summary>
		public int ConstMemory
		{
			get { return _constSizeBytes; }
		}

		/// <summary>
		/// The size in bytes of thread local memory used by this function.
		/// </summary>
		public int LocalMemory
		{
			get { return _localSizeBytes; }
		}

		/// <summary>
		/// The number of registers used by each thread of this function.
		/// </summary>
		public int Registers
		{
			get { return _numRegs; }
		}

		/// <summary>
		/// The PTX virtual architecture version for which the function was
		/// compiled. This value is the major PTX version * 10 + the minor PTX version, so a PTX version 1.3 function
		/// would return the value 13. Note that this may return the undefined value of 0 for cubins compiled prior to CUDA
		/// 3.0.
		/// </summary>
		public Version PtxVersion
		{
			get { return _ptxVersion; }
		}

		/// <summary>
		/// The binary version for which the function was compiled. This
		/// value is the major binary version * 10 + the minor binary version, so a binary version 1.3 function would return
		/// the value 13. Note that this will return a value of 10 for legacy cubins that do not have a properly-encoded binary
		/// architecture version.
		/// </summary>
		public Version BinaryVersion
		{
			get { return _binaryVersion; }
		}

		/// <summary>
		/// The attribute to indicate whether the function has been compiled with 
		/// user specified option "-Xptxas --dlcm=ca" set.
		/// </summary>
		public bool CacheModeCA
		{
			get { return _cacheModeCA; }
		}

		#endregion

		#region Settings
		/// <summary>
		/// Sets the shared memory configuration for a device function.<para/>
		/// On devices with configurable shared memory banks, this function will 
		/// force all subsequent launches of the specified device function to have
		/// the given shared memory bank size configuration. On any given launch of the
		/// function, the shared memory configuration of the device will be temporarily
		/// changed if needed to suit the function's preferred configuration. Changes in
		/// shared memory configuration between subsequent launches of functions, 
		/// may introduce a device side synchronization point.<para/>
		/// Any per-function setting of shared memory bank size set via
		/// <see cref="DriverAPINativeMethods.FunctionManagement.cuFuncSetSharedMemConfig"/>  will override the context wide setting set with
		/// <see cref="DriverAPINativeMethods.ContextManagement.cuCtxSetSharedMemConfig"/>.<para/>
		/// Changing the shared memory bank size will not increase shared memory usage
		/// or affect occupancy of kernels, but may have major effects on performance. 
		/// Larger bank sizes will allow for greater potential bandwidth to shared memory,
		/// but will change what kinds of accesses to shared memory will result in bank 
		/// conflicts.<para/>
		/// This function will do nothing on devices with fixed shared memory bank size.<para/>
		/// The supported bank configurations are<para/> 
		/// - <see cref="CUsharedconfig.DefaultBankSize"/>: set bank width to the default initial
		///   setting (currently, four bytes).
		/// - <see cref="CUsharedconfig.FourByteBankSize"/>: set shared memory bank width to
		///   be natively four bytes.
		/// - <see cref="CUsharedconfig.EightByteBankSize"/>: set shared memory bank width to
		///   be natively eight bytes.
		/// </summary>
		/// <param name="config">requested shared memory configuration</param>
		public void SetSharedMemConfig(CUsharedconfig config)
		{
			CUResult res;
			res = DriverAPINativeMethods.FunctionManagement.cuFuncSetSharedMemConfig(_function, config);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuFuncSetSharedMemConfig", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}


		/// <summary>
		/// On devices where the L1 cache and shared memory use the same hardware resources, this sets through <c>config</c>
		/// the preferred cache configuration for the device function <c>hfunc</c>. This is only a preference. The driver will use the
		/// requested configuration if possible, but it is free to choose a different configuration if required to execute <c>hfunc</c>. <para/>
		/// This setting does nothing on devices where the size of the L1 cache and shared memory are fixed.<para/>
		/// Switching between configuration modes may insert a device-side synchronization point for streamed kernel launches.<para/>
		/// The supported cache modes are defined in <see cref="CUFuncCache"/>
		/// </summary>
		/// <param name="config">Requested cache configuration</param>
		public void SetCacheConfig(CUFuncCache config)
		{
			CUResult res;
			res = DriverAPINativeMethods.FunctionManagement.cuFuncSetCacheConfig(_function, config);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuFuncSetCacheConfig", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}
		#endregion

		#region Occupancy


		/// <summary>
		/// Returns the number of the maximum active blocks per
		/// streaming multiprocessor (Ignores current kernel settings).
		/// </summary>
		/// <param name="blockSize">Block size the kernel is intended to be launched with</param>
		/// <param name="dynamicSMemSize">Per-block dynamic shared memory usage intended, in bytes</param>
		/// <returns>number of the maximum active blocks per
		/// streaming multiprocessor.</returns>
		public int GetOccupancyMaxActiveBlocksPerMultiprocessor(int blockSize, SizeT dynamicSMemSize)
		{
			CUResult res;
			int numBlocks = 0;
			res = DriverAPINativeMethods.Occupancy.cuOccupancyMaxActiveBlocksPerMultiprocessor(ref numBlocks, _function, blockSize, dynamicSMemSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuOccupancyMaxActiveBlocksPerMultiprocessor", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
			return numBlocks;
		}

		/// <summary>
		/// Returns the number of the maximum active blocks per
		/// streaming multiprocessor.
		/// </summary>
		/// <returns>number of the maximum active blocks per
		/// streaming multiprocessor.</returns>
		public int GetOccupancyMaxActiveBlocksPerMultiprocessor()
		{
			CUResult res;
			int numBlocks = 0;
			int blockSize = (int)_blockDim.x * (int)_blockDim.y * (int)_blockDim.z;

			res = DriverAPINativeMethods.Occupancy.cuOccupancyMaxActiveBlocksPerMultiprocessor(ref numBlocks, _function, blockSize, _sharedMemSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuOccupancyMaxActiveBlocksPerMultiprocessor", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
			return numBlocks;
		}

		/// <summary>
		/// Small struct to simplify occupancy calculations
		/// </summary>
		public struct Occupancy
		{
			/// <summary>
			/// minimum grid size needed to achieve the maximum occupancy
			/// </summary>
			public int minGridSize;
			/// <summary>
			/// maximum block size that can achieve the maximum occupancy
			/// </summary>
			public int blockSize;
		}

		/// <summary>
		/// Returns in blockSize a reasonable block size that can achieve
		/// the maximum occupancy (or, the maximum number of active warps with
		/// the fewest blocks per multiprocessor), and in minGridSize the
		/// minimum grid size to achieve the maximum occupancy.
		/// Ignoring dynamic shared memory.
		/// </summary>
		public Occupancy GetOccupancyMaxPotentialBlockSize()
		{
			CUResult res;
			Occupancy occupancy = new Occupancy();

			res = DriverAPINativeMethods.Occupancy.cuOccupancyMaxPotentialBlockSize(ref occupancy.minGridSize, ref occupancy.blockSize, _function, null, 0, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuOccupancyMaxPotentialBlockSize", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
			return occupancy;
		}

		/// <summary>
		/// Returns in blockSize a reasonable block size that can achieve
		/// the maximum occupancy (or, the maximum number of active warps with
		/// the fewest blocks per multiprocessor), and in minGridSize the
		/// minimum grid size to achieve the maximum occupancy.
		/// Ignoring dynamic shared memory.
		/// </summary>
		/// <param name="blockSizeLimit">If blockSizeLimit is 0, the configurator will use the maximum
		/// block size permitted by the device / function instead.</param>
		public Occupancy GetOccupancyMaxPotentialBlockSize(int blockSizeLimit)
		{
			CUResult res;
			Occupancy occupancy = new Occupancy();

			res = DriverAPINativeMethods.Occupancy.cuOccupancyMaxPotentialBlockSize(ref occupancy.minGridSize, ref occupancy.blockSize, _function, null, 0, blockSizeLimit);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuOccupancyMaxPotentialBlockSize", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
			return occupancy;
		}

		/// <summary>
		/// Returns in blockSize a reasonable block size that can achieve
		/// the maximum occupancy (or, the maximum number of active warps with
		/// the fewest blocks per multiprocessor), and in minGridSize the
		/// minimum grid size to achieve the maximum occupancy.
		/// Ignoring dynamic shared memory.
		/// </summary>
		/// <param name="blockSizeToDynamicSMemSize">if the per-block dynamic shared memory size varies with
		/// different block sizes, the user needs to provide a unary function
		/// through blockSizeToDynamicSMemSize that computes the dynamic
		/// shared memory needed by func for any given block size.</param>
		public Occupancy GetOccupancyMaxPotentialBlockSize(del_CUoccupancyB2DSize blockSizeToDynamicSMemSize)
		{
			CUResult res;
			Occupancy occupancy = new Occupancy();

			res = DriverAPINativeMethods.Occupancy.cuOccupancyMaxPotentialBlockSize(ref occupancy.minGridSize, ref occupancy.blockSize, _function, blockSizeToDynamicSMemSize, 0, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuOccupancyMaxPotentialBlockSize", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
			return occupancy;
		}

		/// <summary>
		/// Returns in blockSize a reasonable block size that can achieve
		/// the maximum occupancy (or, the maximum number of active warps with
		/// the fewest blocks per multiprocessor), and in minGridSize the
		/// minimum grid size to achieve the maximum occupancy.
		/// Ignoring dynamic shared memory.
		/// </summary>
		/// <param name="blockSizeToDynamicSMemSize">if the per-block dynamic shared memory size varies with
		/// different block sizes, the user needs to provide a unary function
		/// through blockSizeToDynamicSMemSize that computes the dynamic
		/// shared memory needed by func for any given block size.</param>
		/// <param name="blockSizeLimit">If blockSizeLimit is 0, the configurator will use the maximum
		/// block size permitted by the device / function instead.</param>
		public Occupancy GetOccupancyMaxPotentialBlockSize(del_CUoccupancyB2DSize blockSizeToDynamicSMemSize, int blockSizeLimit)
		{
			CUResult res;
			Occupancy occupancy = new Occupancy();

			res = DriverAPINativeMethods.Occupancy.cuOccupancyMaxPotentialBlockSize(ref occupancy.minGridSize, ref occupancy.blockSize, _function, blockSizeToDynamicSMemSize, 0, blockSizeLimit);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuOccupancyMaxPotentialBlockSize", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
			return occupancy;
		}

		/// <summary>
		/// Returns in blockSize a reasonable block size that can achieve
		/// the maximum occupancy (or, the maximum number of active warps with
		/// the fewest blocks per multiprocessor), and in minGridSize the
		/// minimum grid size to achieve the maximum occupancy.
		/// Ignoring dynamic shared memory.
		/// </summary>
		/// <param name="dynamicSMemSize">If per-block dynamic shared memory allocation is needed, then if
		/// the dynamic shared memory size is constant regardless of block
		/// size, the size should be passed through dynamicSMemSize</param>
		public Occupancy GetOccupancyMaxPotentialBlockSize(SizeT dynamicSMemSize)
		{
			CUResult res;
			Occupancy occupancy = new Occupancy();

			res = DriverAPINativeMethods.Occupancy.cuOccupancyMaxPotentialBlockSize(ref occupancy.minGridSize, ref occupancy.blockSize, _function, null, dynamicSMemSize, 0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuOccupancyMaxPotentialBlockSize", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
			return occupancy;
		}

		/// <summary>
		/// Returns in blockSize a reasonable block size that can achieve
		/// the maximum occupancy (or, the maximum number of active warps with
		/// the fewest blocks per multiprocessor), and in minGridSize the
		/// minimum grid size to achieve the maximum occupancy.
		/// Ignoring dynamic shared memory.
		/// </summary>
		/// <param name="dynamicSMemSize">If per-block dynamic shared memory allocation is needed, then if
		/// the dynamic shared memory size is constant regardless of block
		/// size, the size should be passed through dynamicSMemSize</param>
		/// <param name="blockSizeLimit">If blockSizeLimit is 0, the configurator will use the maximum
		/// block size permitted by the device / function instead.</param>
		public Occupancy GetOccupancyMaxPotentialBlockSize(SizeT dynamicSMemSize, int blockSizeLimit)
		{
			CUResult res;
			Occupancy occupancy = new Occupancy();

			res = DriverAPINativeMethods.Occupancy.cuOccupancyMaxPotentialBlockSize(ref occupancy.minGridSize, ref occupancy.blockSize, _function, null, dynamicSMemSize, blockSizeLimit);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuOccupancyMaxPotentialBlockSize", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
			return occupancy;
		}
		#endregion

		#region Occupancy with Flags


		/// <summary>
		/// Returns the number of the maximum active blocks per
		/// streaming multiprocessor (Ignores current kernel settings).
		/// </summary>
		/// <param name="blockSize">Block size the kernel is intended to be launched with</param>
		/// <param name="dynamicSMemSize">Per-block dynamic shared memory usage intended, in bytes</param>
		/// <param name="flags">Flags</param>
		/// <returns>number of the maximum active blocks per
		/// streaming multiprocessor.</returns>
		public int GetOccupancyMaxActiveBlocksPerMultiprocessor(int blockSize, SizeT dynamicSMemSize, CUoccupancy_flags flags)
		{
			CUResult res;
			int numBlocks = 0;
			res = DriverAPINativeMethods.Occupancy.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(ref numBlocks, _function, blockSize, dynamicSMemSize, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
			return numBlocks;
		}

		/// <summary>
		/// Returns the number of the maximum active blocks per
		/// streaming multiprocessor.
		/// </summary>
		/// <param name="flags">Flags</param>
		/// <returns>number of the maximum active blocks per
		/// streaming multiprocessor.</returns>
		public int GetOccupancyMaxActiveBlocksPerMultiprocessor(CUoccupancy_flags flags)
		{
			CUResult res;
			int numBlocks = 0;
			int blockSize = (int)_blockDim.x * (int)_blockDim.y * (int)_blockDim.z;

			res = DriverAPINativeMethods.Occupancy.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(ref numBlocks, _function, blockSize, _sharedMemSize, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
			return numBlocks;
		}

		/// <summary>
		/// Returns in blockSize a reasonable block size that can achieve
		/// the maximum occupancy (or, the maximum number of active warps with
		/// the fewest blocks per multiprocessor), and in minGridSize the
		/// minimum grid size to achieve the maximum occupancy.
		/// Ignoring dynamic shared memory.
		/// </summary>
		/// <param name="flags">Flags</param>
		public Occupancy GetOccupancyMaxPotentialBlockSize(CUoccupancy_flags flags)
		{
			CUResult res;
			Occupancy occupancy = new Occupancy();

			res = DriverAPINativeMethods.Occupancy.cuOccupancyMaxPotentialBlockSizeWithFlags(ref occupancy.minGridSize, ref occupancy.blockSize, _function, null, 0, 0, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuOccupancyMaxPotentialBlockSizeWithFlags", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
			return occupancy;
		}

		/// <summary>
		/// Returns in blockSize a reasonable block size that can achieve
		/// the maximum occupancy (or, the maximum number of active warps with
		/// the fewest blocks per multiprocessor), and in minGridSize the
		/// minimum grid size to achieve the maximum occupancy.
		/// Ignoring dynamic shared memory.
		/// </summary>
		/// <param name="blockSizeLimit">If blockSizeLimit is 0, the configurator will use the maximum
		/// block size permitted by the device / function instead.</param>
		/// <param name="flags">Flags</param>
		public Occupancy GetOccupancyMaxPotentialBlockSize(int blockSizeLimit, CUoccupancy_flags flags)
		{
			CUResult res;
			Occupancy occupancy = new Occupancy();

			res = DriverAPINativeMethods.Occupancy.cuOccupancyMaxPotentialBlockSizeWithFlags(ref occupancy.minGridSize, ref occupancy.blockSize, _function, null, 0, blockSizeLimit, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuOccupancyMaxPotentialBlockSizeWithFlags", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
			return occupancy;
		}

		/// <summary>
		/// Returns in blockSize a reasonable block size that can achieve
		/// the maximum occupancy (or, the maximum number of active warps with
		/// the fewest blocks per multiprocessor), and in minGridSize the
		/// minimum grid size to achieve the maximum occupancy.
		/// Ignoring dynamic shared memory.
		/// </summary>
		/// <param name="blockSizeToDynamicSMemSize">if the per-block dynamic shared memory size varies with
		/// different block sizes, the user needs to provide a unary function
		/// through blockSizeToDynamicSMemSize that computes the dynamic
		/// shared memory needed by func for any given block size.</param>
		/// <param name="flags">Flags</param>
		public Occupancy GetOccupancyMaxPotentialBlockSize(del_CUoccupancyB2DSize blockSizeToDynamicSMemSize, CUoccupancy_flags flags)
		{
			CUResult res;
			Occupancy occupancy = new Occupancy();

			res = DriverAPINativeMethods.Occupancy.cuOccupancyMaxPotentialBlockSizeWithFlags(ref occupancy.minGridSize, ref occupancy.blockSize, _function, blockSizeToDynamicSMemSize, 0, 0, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuOccupancyMaxPotentialBlockSizeWithFlags", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
			return occupancy;
		}

		/// <summary>
		/// Returns in blockSize a reasonable block size that can achieve
		/// the maximum occupancy (or, the maximum number of active warps with
		/// the fewest blocks per multiprocessor), and in minGridSize the
		/// minimum grid size to achieve the maximum occupancy.
		/// Ignoring dynamic shared memory.
		/// </summary>
		/// <param name="blockSizeToDynamicSMemSize">if the per-block dynamic shared memory size varies with
		/// different block sizes, the user needs to provide a unary function
		/// through blockSizeToDynamicSMemSize that computes the dynamic
		/// shared memory needed by func for any given block size.</param>
		/// <param name="blockSizeLimit">If blockSizeLimit is 0, the configurator will use the maximum
		/// block size permitted by the device / function instead.</param>
		/// <param name="flags">Flags</param>
		public Occupancy GetOccupancyMaxPotentialBlockSize(del_CUoccupancyB2DSize blockSizeToDynamicSMemSize, int blockSizeLimit, CUoccupancy_flags flags)
		{
			CUResult res;
			Occupancy occupancy = new Occupancy();

			res = DriverAPINativeMethods.Occupancy.cuOccupancyMaxPotentialBlockSizeWithFlags(ref occupancy.minGridSize, ref occupancy.blockSize, _function, blockSizeToDynamicSMemSize, 0, blockSizeLimit, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuOccupancyMaxPotentialBlockSizeWithFlags", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
			return occupancy;
		}

		/// <summary>
		/// Returns in blockSize a reasonable block size that can achieve
		/// the maximum occupancy (or, the maximum number of active warps with
		/// the fewest blocks per multiprocessor), and in minGridSize the
		/// minimum grid size to achieve the maximum occupancy.
		/// Ignoring dynamic shared memory.
		/// </summary>
		/// <param name="dynamicSMemSize">If per-block dynamic shared memory allocation is needed, then if
		/// the dynamic shared memory size is constant regardless of block
		/// size, the size should be passed through dynamicSMemSize</param>
		/// <param name="flags">Flags</param>
		public Occupancy GetOccupancyMaxPotentialBlockSize(SizeT dynamicSMemSize, CUoccupancy_flags flags)
		{
			CUResult res;
			Occupancy occupancy = new Occupancy();

			res = DriverAPINativeMethods.Occupancy.cuOccupancyMaxPotentialBlockSizeWithFlags(ref occupancy.minGridSize, ref occupancy.blockSize, _function, null, dynamicSMemSize, 0, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuOccupancyMaxPotentialBlockSizeWithFlags", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
			return occupancy;
		}

		/// <summary>
		/// Returns in blockSize a reasonable block size that can achieve
		/// the maximum occupancy (or, the maximum number of active warps with
		/// the fewest blocks per multiprocessor), and in minGridSize the
		/// minimum grid size to achieve the maximum occupancy.
		/// Ignoring dynamic shared memory.
		/// </summary>
		/// <param name="dynamicSMemSize">If per-block dynamic shared memory allocation is needed, then if
		/// the dynamic shared memory size is constant regardless of block
		/// size, the size should be passed through dynamicSMemSize</param>
		/// <param name="blockSizeLimit">If blockSizeLimit is 0, the configurator will use the maximum
		/// block size permitted by the device / function instead.</param>
		/// <param name="flags">Flags</param>
		public Occupancy GetOccupancyMaxPotentialBlockSize(SizeT dynamicSMemSize, int blockSizeLimit, CUoccupancy_flags flags)
		{
			CUResult res;
			Occupancy occupancy = new Occupancy();

			res = DriverAPINativeMethods.Occupancy.cuOccupancyMaxPotentialBlockSizeWithFlags(ref occupancy.minGridSize, ref occupancy.blockSize, _function, null, dynamicSMemSize, blockSizeLimit, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuOccupancyMaxPotentialBlockSizeWithFlags", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
			return occupancy;
		}
		#endregion


		/// <summary>
		/// Sets the grid dimensions according to block dimensions, so that each dimension has at least computeSize threads
		/// </summary>
		/// <param name="computeSizeX">Minimum number of threads in X dimension</param>
		/// <param name="computeSizeY">Minimum number of threads in Y dimension</param>
		/// <param name="computeSizeZ">Minimum number of threads in Z dimension</param>
		public void SetComputeSize(uint computeSizeX, uint computeSizeY, uint computeSizeZ)
		{
			_gridDim.x = (computeSizeX + _blockDim.x - 1) / _blockDim.x;
			_gridDim.y = (computeSizeY + _blockDim.y - 1) / _blockDim.y;
			_gridDim.z = (computeSizeZ + _blockDim.z - 1) / _blockDim.z;
		}

		/// <summary>
		/// Sets the grid dimensions according to block dimensions, so that each dimension has at least computeSize threads
		/// </summary>
		/// <param name="computeSizeX">Minimum number of threads in X dimension</param>
		/// <param name="computeSizeY">Minimum number of threads in Y dimension</param>
		public void SetComputeSize(uint computeSizeX, uint computeSizeY)
		{
			SetComputeSize(computeSizeX, computeSizeY, 1);
		}

		/// <summary>
		/// Sets the grid dimensions according to block dimensions, so that each dimension has at least computeSize threads
		/// </summary>
		/// <param name="computeSizeX">Minimum number of threads in X dimension</param>
		public void SetComputeSize(uint computeSizeX)
		{
			SetComputeSize(computeSizeX, 1, 1);
		}

		/// <summary>
		/// Sets the grid dimensions according to block dimensions, so that each dimension has at least computeSize threads
		/// </summary>
		/// <param name="computeSize">Minimum number of threads in X, Y and Z dimension</param>
		public void SetComputeSize(dim3 computeSize)
		{
			SetComputeSize(computeSize.x, computeSize.y, computeSize.z);
		}

	}
}
