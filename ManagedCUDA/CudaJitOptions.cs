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
	/// A list of JIT compiler / linker option passed to Cuda.<para/>
	/// If buffer options are used (i.e. InfoLogBuffer and ErrorLogBuffer), this 
	/// collection should only be used once as buffer size is overwritten by Cuda.<para/>
	/// To copy data from unmanaged to managed memory, call <see cref="UpdateValues()"/> after
	/// the API call that produced output data.<para/>
	/// Maximum number of options is limited to 30.
	/// </summary>
	public class CudaJitOptionCollection : IDisposable
	{
		/// <summary/>
		protected bool disposed;
		const int MAX_ELEM = 32;
		CUJITOption[] _options = new CUJITOption[MAX_ELEM];
		IntPtr[] _values = new IntPtr[MAX_ELEM];
		List<CudaJitOption> _cudaOptions = new List<CudaJitOption>();
		int _count = 0;

		/// <summary>
		/// Add a single option to the collection.
		/// </summary>
		/// <param name="opt">Option to add</param>
		public void Add(CudaJitOption opt)
		{
			if (_count >= MAX_ELEM - 2)
				throw new Exception("Maximum number of options elements reached!");

			_cudaOptions.Add(opt);

			if (opt is CudaJOErrorLogBuffer) //add two elements
			{
				CUJITOption[] o = opt.Options;
				IntPtr[] v = opt.Values;

				opt.Index = _count;
				_options[_count] = o[0];
				_values[_count] = v[0];
				_count++;
				_options[_count] = o[1];
				_values[_count] = v[1];
			}
			else if (opt is CudaJOInfoLogBuffer) //add two elements
			{
				CUJITOption[] o = opt.Options;
				IntPtr[] v = opt.Values;

				opt.Index = _count;
				_options[_count] = o[0];
				_values[_count] = v[0];
				_count++;
				_options[_count] = o[1];
				_values[_count] = v[1];
			}
			else //add one elements
			{
				CUJITOption[] o = opt.Options;
				IntPtr[] v = opt.Values;

				opt.Index = _count;
				_options[_count] = o[0];
				_values[_count] = v[0];
			}
			_count++;
		}

		/// <summary>
		/// A multiple options to the collection.
		/// </summary>
		/// <param name="options">Options to add</param>
		public void Add(IList<CudaJitOption> options)
		{
			foreach (var item in options)
			{
				Add(item);
			}
		}

		/// <summary>
		/// Copy data from unmanaged to managed memory
		/// </summary>
		public void UpdateValues()
		{
			foreach (var item in _cudaOptions)
			{
				if (item is CudaJOErrorLogBuffer)
				{
					(item as CudaJOErrorLogBuffer).SetValue = _values[item.Index];
				}
				if (item is CudaJOInfoLogBuffer)
				{
					(item as CudaJOInfoLogBuffer).SetValue = _values[item.Index];
				}
				if (item is CudaJOThreadsPerBlock)
				{
					(item as CudaJOThreadsPerBlock).SetValue = _values[item.Index];
				}
				if (item is CudaJOWallTime)
				{
					(item as CudaJOWallTime).SetValue = _values[item.Index];
				}
			}
		}

		/// <summary>
		/// Reset values returned from Cuda API for info and error buffers.
		/// </summary>
		public void ResetValues()
		{
			foreach (var item in _cudaOptions)
			{
				if (item is CudaJOErrorLogBuffer)
				{
					(item as CudaJOErrorLogBuffer).Reset();
				}
				if (item is CudaJOInfoLogBuffer)
				{
					(item as CudaJOInfoLogBuffer).Reset();
				}
			}
		}

		internal CUJITOption[] Options
		{
			get { return _options; }
		}
		
		internal IntPtr[] Values
		{
			get
			{
				if (disposed) throw new ObjectDisposedException(this.ToString());
				return _values; 
			}
		}

		internal int Count
		{
			get { return _count; }
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaJitOptionCollection()
		{
			Dispose(false);
		}

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
				foreach (var item in _cudaOptions)
				{
					item.Dispose();
				}
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCuda not-disposed warning: {0}", this.GetType()));
		}
		#endregion
	}

	/// <summary>
	/// Online compiler options
	/// </summary>
	public abstract class CudaJitOption : IDisposable
	{
		/// <summary>
		/// Option value converted to (void *)
		/// </summary>
		protected IntPtr _ptrValue;
		/// <summary>
		/// Option
		/// </summary>
		protected CUJITOption _option;
		private int _index;
		/// <summary/>
		protected bool disposed;

		internal virtual CUJITOption[] Options
		{
			get { return new CUJITOption[] { _option }; }
		}

		internal virtual IntPtr[] Values
		{
			get { return new IntPtr[] { _ptrValue }; }
		}

		internal int Index
		{
			get { return _index; }
			set { _index = value; }
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaJitOption()
		{
			Dispose(false);
		}

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
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCuda not-disposed warning: {0}", this.GetType()));
		}
		#endregion
	}

	/// <summary>
	/// <para>Max number of registers that a thread may use.</para>
	/// <para>Option type: unsigned int</para>
	/// <para>Applies to: compiler only</para>
	/// </summary>
	public class CudaJOMaxRegisters : CudaJitOption
	{
		/// <summary>
		/// <para>Max number of registers that a thread may use.</para>
		/// <para>Option type: unsigned int</para>
		/// <para>Applies to: compiler only</para>
		/// </summary>
		/// <param name="value"></param>
		public CudaJOMaxRegisters(uint value)
		{
			_option = CUJITOption.MaxRegisters;
			_ptrValue = (IntPtr)(Convert.ToUInt32(value, System.Globalization.CultureInfo.InvariantCulture));
		}
	}
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
	public class CudaJOThreadsPerBlock : CudaJitOption
	{
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
		/// <param name="value"></param>
		public CudaJOThreadsPerBlock(int value)
		{
			_option = CUJITOption.ThreadsPerBlock;
			_ptrValue = (IntPtr)(Convert.ToUInt32(value, System.Globalization.CultureInfo.InvariantCulture));
		}

		/// <summary>
		/// Returns the number of threads the compiler actually targeted.
		/// This restricts the resource utilization fo the compiler (e.g. max
		/// registers) such that a block with the given number of threads should be
		/// able to launch based on register limitations. Note, this option does not
		/// currently take into account any other resource limitations, such as
		/// shared memory utilization.<para/>
		/// The value is only valid after a succesful call to <see cref="CudaJitOptionCollection.UpdateValues()"/>
		/// </summary>
		public int Value
		{
			get
			{
				return (int)_ptrValue;
			}
		}

		internal IntPtr SetValue
		{
			set { _ptrValue = value; }
		}
	}
	/// <summary>
	/// Returns a float value in the option of the wall clock time, in
	/// milliseconds, spent creating the cubin<para/>
	/// Option type: float
	/// <para>Applies to: compiler and linker</para>
	/// </summary>
	public class CudaJOWallTime : CudaJitOption
	{
		/// <summary>
		/// Returns a float value in the option of the wall clock time, in
		/// milliseconds, spent creating the cubin<para/>
		/// Option type: float
		/// <para>Applies to: compiler and linker</para>
		/// </summary>
		public CudaJOWallTime()
		{
			_option = CUJITOption.WallTime;
			_ptrValue = IntPtr.Zero;
		}

		/// <summary>
		/// Returns a float value in the option of the wall clock time, in
		/// milliseconds, spent creating the cubin<para/>
		/// Option type: float
		/// <para>Applies to: compiler and linker</para>
		/// The value is only valid after a succesful call to <see cref="CudaJitOptionCollection.UpdateValues()"/>
		/// </summary>
		public float Value
		{
			get
			{
				uint v = (uint)_ptrValue;
				byte[] bytes = BitConverter.GetBytes(v);
				return BitConverter.ToSingle(bytes, 0);
			}
		}

		internal IntPtr SetValue
		{
			set	{ _ptrValue = value; }
		}
	}
	/// <summary>
	/// <para>Pointer to a buffer in which to print any log messsages from PTXAS
	/// that are informational in nature (the buffer size is specified via
	/// option ::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES)</para>
	/// <para>Option type: char*</para>
	/// <para>Applies to: compiler and linker</para>
	/// <para/>You must free the internal buffer array manually after use by calling <see cref="FreeHandle()"/>!
	/// </summary>
	public class CudaJOInfoLogBuffer : CudaJitOption
	{
		byte[] _buffer;
		int _size;
		IntPtr _returnedSize;
		GCHandle _handle;

		/// <summary>
		/// <para>Pointer to a buffer in which to print any log messsages from PTXAS
		/// that are informational in nature</para>
		/// <para>Option type: char*</para>
		/// <para>Applies to: compiler and linker</para>
		/// <para/>You must free the internal buffer array manually after use by calling <see cref="FreeHandle()"/>!
		/// </summary>
		/// <param name="size">Size of the internal buffer array</param>
		public CudaJOInfoLogBuffer(int size)
		{
			_size = size;
			_buffer = new byte[_size];
			_handle = GCHandle.Alloc(_buffer, GCHandleType.Pinned);
			_ptrValue = _handle.AddrOfPinnedObject();
			_option = CUJITOption.InfoLogBuffer;
			_returnedSize = (IntPtr)_size;
		}

		internal override CUJITOption[] Options
		{
			get
			{
				return new CUJITOption[] { CUJITOption.InfoLogBufferSizeBytes, _option };
			}
		}

		internal override IntPtr[] Values
		{
			get
			{
				if (disposed) throw new ObjectDisposedException(this.ToString());
				return new IntPtr[] { _returnedSize, _ptrValue };
			}
		}

		/// <summary>
		/// ManagedCuda allocates an byte array as buffer and pins it in order to pass it to Cuda.<para/>
		/// You must free the buffer manually if the buffer is not needed anymore.
		/// </summary>
		public void FreeHandle()
		{
			if (_handle != null)
				if (_handle.IsAllocated)
					_handle.Free();
		}

		/// <summary>
		/// Returns the buffer converted to string.<para/>
		/// The value is only valid after a succesful call to <see cref="CudaJitOptionCollection.UpdateValues()"/>
		/// </summary>
		public string Value
		{
			get
			{
				if (disposed) throw new ObjectDisposedException(this.ToString());
				if (!_handle.IsAllocated) return string.Empty;

				string val = Marshal.PtrToStringAnsi(_ptrValue, (int)_returnedSize);
				return val.Replace("\0", "");
			}
		}

		internal IntPtr SetValue
		{
			set
			{
				if (disposed) throw new ObjectDisposedException(this.ToString());
				_returnedSize = value; 
			}
		}

		internal void Reset()
		{
			for (int i = 0; i < _size; i++)
			{
				_buffer[i] = 0;
			}
			_returnedSize = (IntPtr)_size;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="fDisposing"></param>
		protected override void Dispose(bool fDisposing)
		{
			FreeHandle();
			base.Dispose(fDisposing);
		}
	}

	/// <summary>
	/// <para>Pointer to a buffer in which to print any log messages from PTXAS that
	/// reflect errors</para>
	/// <para>Option type: char*</para>
	/// <para>Applies to: compiler and linker</para>
	/// <para/>You must free the internal buffer array manually after use by calling <see cref="FreeHandle()"/>!
	/// </summary>
	public class CudaJOErrorLogBuffer : CudaJitOption
	{
 		byte[] _buffer;
		int _size;
		IntPtr _returnedSize;
		GCHandle _handle;

		/// <summary>
		/// <para>Pointer to a buffer in which to print any log messages from PTXAS that
		/// reflect errors</para>
		/// <para>Option type: char*</para>
		/// <para>Applies to: compiler and linker</para>
		/// <para/>You must free the internal buffer array manually after use by calling <see cref="FreeHandle()"/>!
		/// </summary>
		/// <param name="size"></param>
		public CudaJOErrorLogBuffer(int size)
		{
			_size = size;
			_buffer = new byte[_size];
			_handle = GCHandle.Alloc(_buffer, GCHandleType.Pinned);
			_ptrValue = _handle.AddrOfPinnedObject();
			_option = CUJITOption.ErrorLogBuffer;
			_returnedSize = (IntPtr)_size;
		}

		internal override CUJITOption[] Options
		{
			get
			{
				return new CUJITOption[] { CUJITOption.ErrorLogBufferSizeBytes, _option };
			}
		}

		internal override IntPtr[] Values
		{
			get
			{
				if (disposed) throw new ObjectDisposedException(this.ToString());
				return new IntPtr[] { _returnedSize, _ptrValue };
			}
		}

		/// <summary>
		/// ManagedCuda allocates an byte array as buffer and pins it in order to pass it to Cuda.<para/>
		/// You must free the buffer manually if the buffer is not needed anymore.
		/// </summary>
		public void FreeHandle()
		{
			if (_handle != null)
				if (_handle.IsAllocated)
					_handle.Free();
		}

		/// <summary>
		/// Returns the buffer converted to string.<para/>
		/// The value is only valid after a succesful call to <see cref="CudaJitOptionCollection.UpdateValues()"/>
		/// </summary>
		public string Value
		{
			get
			{
				if (disposed) throw new ObjectDisposedException(this.ToString());
				if (!_handle.IsAllocated) return string.Empty;

				string val = Marshal.PtrToStringAnsi(_ptrValue, (int)_returnedSize);
				return val.Replace("\0", "");
			}
		}

		internal IntPtr SetValue
		{
			set
			{
				if (disposed) throw new ObjectDisposedException(this.ToString());
				_returnedSize = value; 
			}
		}

		internal void Reset()
		{
			for (int i = 0; i < _size; i++)
			{
				_buffer[i] = 0;
			}
			_returnedSize = (IntPtr)_size;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="fDisposing"></param>
		protected override void Dispose(bool fDisposing)
		{
			FreeHandle();
			base.Dispose(fDisposing);
		}
	}

	/// <summary>
	/// <para>Level of optimizations to apply to generated code (0 - 4), with 4
	/// being the default and highest level of optimizations.</para>
	/// <para>Option type: unsigned int</para>
	/// <para>Applies to: compiler only</para>
	/// </summary>
	public class CudaJOOptimizationLevel : CudaJitOption
	{
		/// <summary>
		/// <para>Level of optimizations to apply to generated code (0 - 4), with 4
		/// being the default and highest level of optimizations.</para>
		/// <para>Option type: unsigned int</para>
		/// <para>Applies to: compiler only</para>
		/// </summary>
		/// <param name="value">Level of optimizations to apply to generated code (0 - 4), with 4
		/// being the default and highest level of optimizations.</param>
		public CudaJOOptimizationLevel(uint value)
		{
			_option = CUJITOption.OptimizationLevel;
			_ptrValue = (IntPtr)(Convert.ToUInt32(value, System.Globalization.CultureInfo.InvariantCulture));
		}
	}
	/// <summary>
	/// <para>No option value required. Determines the target based on the current
	/// attached context (default)</para>
	/// <para>Option type: No option value needed</para>
	/// <para>Applies to: compiler and linker</para>
	/// </summary>
	public class CudaJOTargetFromContext : CudaJitOption
	{
		/// <summary>
		/// <para>Determines the target based on the current attached context (default)</para>
		/// <para>Option type: No option value needed</para>
		/// <para>Applies to: compiler and linker</para>
		/// </summary>
		public CudaJOTargetFromContext()
		{
			_option = CUJITOption.TargetFromContext;
			_ptrValue = new IntPtr();
		}
	}
	/// <summary>
	/// <para>Target is chosen based on supplied <see cref="CUJITTarget"/>.</para>
	/// <para>Option type: unsigned int for enumerated type <see cref="CUJITTarget"/></para>
	/// <para>Applies to: compiler and linker</para>
	/// </summary>
	public class CudaJOTarget : CudaJitOption
	{
		/// <summary>
		/// <para>Target is chosen based on supplied ::CUjit_target_enum.</para>
		/// <para>Option type: unsigned int for enumerated type ::CUjit_target_enum</para>
		/// <para>Applies to: compiler and linker</para>
		/// </summary>
		/// <param name="value"></param>
		public CudaJOTarget(CUJITTarget value)
		{
			_option = CUJITOption.Target;
			_ptrValue = (IntPtr)(Convert.ToUInt32(value, System.Globalization.CultureInfo.InvariantCulture));
		}
	}
	/// <summary>
	/// <para>Specifies choice of fallback strategy if matching cubin is not found.
	/// Choice is based on supplied <see cref="CUJITFallback"/>.</para>
	/// <para>Option type: unsigned int for enumerated type <see cref="CUJITFallback"/></para>
	/// <para>Applies to: compiler only</para>
	/// </summary>
	public class CudaJOFallbackStrategy : CudaJitOption
	{
		/// <summary>
		/// <para>Specifies choice of fallback strategy if matching cubin is not found.
		/// Choice is based on supplied <see cref="CUJITFallback"/>.</para>
		/// <para>Option type: unsigned int for enumerated type <see cref="CUJITFallback"/></para>
		/// <para>Applies to: compiler only</para>
		/// </summary>
		/// <param name="value"></param>
		public CudaJOFallbackStrategy(CUJITFallback value)
		{
			_option = CUJITOption.FallbackStrategy;
			_ptrValue = (IntPtr)(Convert.ToUInt32(value, System.Globalization.CultureInfo.InvariantCulture));
		}
	}
	/// <summary>
	/// Specifies whether to create debug information in output (-g) <para/> (0: false, default)
	/// <para>Option type: int</para>
	/// <para>Applies to: compiler and linker</para>
	/// </summary>
	public class CudaJOGenerateDebugInfo : CudaJitOption
	{
		/// <summary>
		/// Specifies whether to create debug information in output (-g) <para/> (0: false, default)
		/// <para>Option type: int</para>
		/// <para>Applies to: compiler and linker</para>
		/// </summary>
		/// <param name="value"></param>
		public CudaJOGenerateDebugInfo(bool value)
		{
			_option = CUJITOption.GenerateDebugInfo;
			_ptrValue = (IntPtr)(value ? 1 : 0);
		}
	}
	
	/// <summary>
	/// Generate verbose log messages <para/> (0: false, default)
	/// <para>Option type: int</para>
	/// <para>Applies to: compiler and linker</para>
	/// </summary>
	public class CudaJOLogVerbose : CudaJitOption
	{
		/// <summary>
		/// Generate verbose log messages <para/> (0: false, default)
		/// <para>Option type: int</para>
		/// <para>Applies to: compiler and linker</para>
		/// </summary>
		/// <param name="value"></param>
		public CudaJOLogVerbose(bool value)
		{
			_option = CUJITOption.LogVerbose;
			_ptrValue = (IntPtr)(value ? 1 : 0);
		}
	}
	/// <summary>
	/// Generate line number information (-lineinfo) <para/> (0: false, default)
	/// <para>Option type: int</para>
	/// <para>Applies to: compiler only</para>
	/// </summary>
	public class CudaJOGenerateLineInfo : CudaJitOption
	{
		/// <summary>
		/// Generate line number information (-lineinfo) <para/> (0: false, default)
		/// <para>Option type: int</para>
		/// <para>Applies to: compiler only</para>
		/// </summary>
		/// <param name="value"></param>
		public CudaJOGenerateLineInfo(bool value)
		{
			_option = CUJITOption.GenerateLineInfo;
			_ptrValue = (IntPtr)(value ? 1 : 0);
		}
	}
	/// <summary>
	/// Specifies whether to enable caching explicitly (-dlcm)<para/>
	/// Choice is based on supplied <see cref="CUJITCacheMode"/>.
	/// <para>Option type: unsigned int for enumerated type <see cref="CUJITCacheMode"/></para>
	/// <para>Applies to: compiler only</para>
	/// </summary>
	public class CudaJOJITCacheMode : CudaJitOption
	{
		/// <summary>
		/// Specifies whether to enable caching explicitly (-dlcm)<para/>
		/// Choice is based on supplied <see cref="CUJITCacheMode"/>.
		/// <para>Option type: unsigned int for enumerated type <see cref="CUJITCacheMode"/></para>
		/// <para>Applies to: compiler only</para>
		/// </summary>
		/// <param name="value"></param>
		public CudaJOJITCacheMode(CUJITCacheMode value)
		{
			_option = CUJITOption.GenerateLineInfo;
			_ptrValue = (IntPtr)(Convert.ToUInt32(value, System.Globalization.CultureInfo.InvariantCulture));
		}
	}

}
