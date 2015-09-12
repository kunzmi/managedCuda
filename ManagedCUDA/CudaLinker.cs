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
	/// A pending JIT linker invocation.
	/// </summary>
	public class CudaLinker : IDisposable
	{
		bool disposed;
		CUlinkState _state;
		CUResult res;

		/// <summary>
		/// Creates a pending JIT linker invocation.
		/// </summary>
		public CudaLinker()
			:this(null)
		{ 
		
		}

		/// <summary>
		/// Creates a pending JIT linker invocation.
		/// </summary>
		/// <param name="options">Collection of linker and compiler options</param>
		public CudaLinker(CudaJitOptionCollection options)
		{
			
			_state = new CUlinkState();

			if (options == null) 
				res = DriverAPINativeMethods.ModuleManagement.cuLinkCreate(0, null, null, ref _state);
			else
				res = DriverAPINativeMethods.ModuleManagement.cuLinkCreate((uint)options.Count, options.Options, options.Values, ref _state);

			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuLinkCreate", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
			

		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaLinker()
		{
			Dispose(false);
		}

		#region Dispose
		/// <summary>
		/// Dispose <para/>
		/// Destroys state for a JIT linker invocation.
		/// </summary>
		public void Dispose()
		{
			Dispose(true);
			GC.SuppressFinalize(this);
		}

		/// <summary>
		/// For IDisposable. <para/>
		/// Destroys state for a JIT linker invocation.
		/// </summary>
		/// <param name="fDisposing"></param>
		protected virtual void Dispose(bool fDisposing)
		{
			if (fDisposing && !disposed)
			{	
				//Ignore if failing
				CUResult res;
				res = DriverAPINativeMethods.ModuleManagement.cuLinkDestroy(_state);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuLinkDestroy", res));
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion


		/// <summary>
		/// Add an input to a pending linker invocation.
		/// </summary>
		/// <param name="data">The input data.  PTX must be NULL-terminated.</param>
		/// <param name="type">The type of the input data.</param>
		/// <param name="name">An optional name for this input in log messages.</param>
		/// <param name="options">Collection of linker and compiler options</param>
		public void AddData(byte[] data, CUJITInputType type, string name, CudaJitOptionCollection options)
		{
			if (options == null)
			{
				res = DriverAPINativeMethods.ModuleManagement.cuLinkAddData(_state, type, data, data.Length, name, 0, null, null);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuLinkAddData", res));
				if (res != CUResult.Success)
					throw new CudaException(res);
			}
			else
			{
				res = DriverAPINativeMethods.ModuleManagement.cuLinkAddData(_state, type, data, data.Length, name, (uint)options.Count, options.Options, options.Values);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuLinkAddData", res));

				if (res != CUResult.Success)
					throw new CudaException(res);
			}
		}

		/// <summary>
		/// Add an input to a pending linker invocation.
		/// </summary>
		/// <param name="data">The input data.  PTX must be NULL-terminated.</param>
		/// <param name="type">The type of the input data.</param>
		/// <param name="name">An optional name for this input in log messages.</param>
		/// <param name="options">Collection of linker and compiler options</param>
		public void AddData(System.IO.Stream data, CUJITInputType type, string name, CudaJitOptionCollection options)
		{
			if (data == null) throw new ArgumentNullException("data");
			byte[] dataArray = new byte[data.Length];

			int bytesToRead = (int)data.Length;
			data.Position = 0;
			while (bytesToRead > 0)
			{
				bytesToRead -= data.Read(dataArray, (int)data.Position, bytesToRead);
			}
			data.Position = 0;

			AddData(dataArray, type, name, options);
		}

		/// <summary>
		/// Add an input to a pending linker invocation.
		/// </summary>
		/// <param name="filename">Path to the input file.</param>
		/// <param name="type">The type of the input data.</param>
		/// <param name="options">Collection of linker and compiler options</param>
		public void AddFile(string filename, CUJITInputType type, CudaJitOptionCollection options)
		{
			if (options == null)
			{
				res = DriverAPINativeMethods.ModuleManagement.cuLinkAddFile(_state, type, filename, 0, null, null);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuLinkAddFile", res));
				if (res != CUResult.Success)
					throw new CudaException(res);
			}
			else
			{
				res = DriverAPINativeMethods.ModuleManagement.cuLinkAddFile(_state, type, filename, (uint)options.Count, options.Options, options.Values);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuLinkAddFile", res));

				if (res != CUResult.Success)
					throw new CudaException(res);
			}
		}


		/// <summary>
		/// Complete a pending linker invocation.<para/>
		/// Completes the pending linker action and returns the cubin image for the linked
		/// device code, which can be used with ::cuModuleLoadData.
		/// </summary>
		public byte[] Complete()
		{
			IntPtr ptr = new IntPtr();
			SizeT size = new SizeT();
			
			res = DriverAPINativeMethods.ModuleManagement.cuLinkComplete(_state, ref ptr, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuLinkComplete", res));
			if (res != CUResult.Success)
				throw new CudaException(res);

			if (size == 0)
				return null;

			byte[] ret = new byte[(int)size];

			Marshal.Copy(ptr, ret, 0, (int)size);

			return ret;
		}
	}
}
