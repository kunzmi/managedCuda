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
using System.Collections.Generic;
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
        private const int MAX_ELEM = 32;
        private CUJITOption[] _options = new CUJITOption[MAX_ELEM];
        private IntPtr[] _values = new IntPtr[MAX_ELEM];
        private List<CudaJitOption> _cudaOptions = new List<CudaJitOption>();
        private int _count = 0;

        /// <summary>
        /// Add a single option to the collection.
        /// </summary>
        /// <param name="opt">Option to add</param>
        public void Add(CudaJitOption opt)
        {
            if (_count >= MAX_ELEM - 2)
                throw new Exception("Maximum number of options elements reached!");

            _cudaOptions.Add(opt);

            CUJITOption[] o = opt.Options;
            IntPtr[] v = opt.Values;

            opt.Index = _count;
            for (int i = 0; i < o.Length; i++)
            {
                _options[_count] = o[i];
                _values[_count] = v[i];
                _count++;
            }

            //if (opt is CudaJOErrorLogBuffer) //add two elements
            //{
            //    CUJITOption[] o = opt.Options;
            //    IntPtr[] v = opt.Values;

            //    _options[_count] = o[0];
            //    _values[_count] = v[0];
            //    _count++;
            //    _options[_count] = o[1];
            //    _values[_count] = v[1];
            //}
            //else if (opt is CudaJOInfoLogBuffer) //add two elements
            //{
            //    CUJITOption[] o = opt.Options;
            //    IntPtr[] v = opt.Values;

            //    opt.Index = _count;
            //    _options[_count] = o[0];
            //    _values[_count] = v[0];
            //    _count++;
            //    _options[_count] = o[1];
            //    _values[_count] = v[1];
            //}
            //else if (opt is CudaJOReferencedKernelNames) //add two elements
            //{
            //    CUJITOption[] o = opt.Options;
            //    IntPtr[] v = opt.Values;

            //    opt.Index = _count;
            //    _options[_count] = o[0];
            //    _values[_count] = v[0];
            //    _count++;
            //    _options[_count] = o[1];
            //    _values[_count] = v[1];
            //}
            //else if (opt is CudaJOReferencedVariableNames) //add two elements
            //{
            //    CUJITOption[] o = opt.Options;
            //    IntPtr[] v = opt.Values;

            //    opt.Index = _count;
            //    _options[_count] = o[0];
            //    _values[_count] = v[0];
            //    _count++;
            //    _options[_count] = o[1];
            //    _values[_count] = v[1];
            //}
            //else if (opt is CudaJOGlobalSymbolNames) //add three elements
            //{
            //    CUJITOption[] o = opt.Options;
            //    IntPtr[] v = opt.Values;

            //    opt.Index = _count;
            //    _options[_count] = o[0];
            //    _values[_count] = v[0];
            //    _count++;
            //    _options[_count] = o[1];
            //    _values[_count] = v[1];
            //    _count++;
            //    _options[_count] = o[2];
            //    _values[_count] = v[2];
            //}
            //else //add one elements
            //{
            //    CUJITOption[] o = opt.Options;
            //    IntPtr[] v = opt.Values;

            //    opt.Index = _count;
            //    _options[_count] = o[0];
            //    _values[_count] = v[0];
            //}
            //_count++;
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
    /// This restricts the resource utilization of the compiler (e.g. max
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
        /// This restricts the resource utilization of the compiler (e.g. max
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
        /// This restricts the resource utilization of the compiler (e.g. max
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
            set { _ptrValue = value; }
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

    /// <summary>
    /// Array of device symbol names that will be relocated to the corresponding
    /// host addresses stored in ::CU_JIT_GLOBAL_SYMBOL_ADDRESSES.<para/>
    /// Must contain ::CU_JIT_GLOBAL_SYMBOL_COUNT entries.<para/>
    /// When loading a device module, driver will relocate all encountered
    /// unresolved symbols to the host addresses.<para/>
    /// It is only allowed to register symbols that correspond to unresolved
    /// global variables.<para/>
    /// It is illegal to register the same device symbol at multiple addresses.<para/>
    /// Option type: const char **<para/>
    /// Applies to: dynamic linker only
    /// </summary>
    public class CudaJOGlobalSymbolNames : CudaJitOption
    {
        IntPtr[] _namesPtr;
        IntPtr[] _addresses;
        GCHandle _handleNames;
        GCHandle _handleAddresses;
        IntPtr _ptrAddresses;
        IntPtr _count;

        /// <summary>
        /// Array of device symbol names that will be relocated to the corresponding
        /// host addresses stored in ::CU_JIT_GLOBAL_SYMBOL_ADDRESSES.<para/>
        /// Must contain ::CU_JIT_GLOBAL_SYMBOL_COUNT entries.<para/>
        /// When loading a device module, driver will relocate all encountered
        /// unresolved symbols to the host addresses.<para/>
        /// It is only allowed to register symbols that correspond to unresolved
        /// global variables.<para/>
        /// It is illegal to register the same device symbol at multiple addresses.<para/>
        /// Option type: const char **<para/>
        /// Applies to: dynamic linker only
        /// </summary>
        /// <param name="globalSymbolNames"></param>
        /// <param name="globalSymbolAddresses"></param>
        public CudaJOGlobalSymbolNames(string[] globalSymbolNames, IntPtr[] globalSymbolAddresses)
        {
            if (globalSymbolNames.Length != globalSymbolAddresses.Length)
            {
                throw new ArgumentException("globalSymbolNames and globalSymbolAddresses must have the same size");
            }

            int count = globalSymbolAddresses.Length;
            _namesPtr = new IntPtr[count];
            _addresses = new IntPtr[count];

            for (int i = 0; i < count; i++)
            {
                _namesPtr[i] = Marshal.StringToHGlobalAnsi(globalSymbolNames[i]);
            }
            _addresses = globalSymbolAddresses;
            _count = (IntPtr)globalSymbolNames.Length;
            _handleNames = GCHandle.Alloc(_namesPtr, GCHandleType.Pinned);
            _ptrValue = _handleNames.AddrOfPinnedObject();
            _handleAddresses = GCHandle.Alloc(_addresses, GCHandleType.Pinned);
            _ptrAddresses = _handleAddresses.AddrOfPinnedObject();
            _option = CUJITOption.GlobalSymbolNames;
        }

        internal override CUJITOption[] Options
        {
            get
            {
                return new CUJITOption[] { CUJITOption.GlobalSymbolCount, _option, CUJITOption.GlobalSymbolAddresses };
            }
        }

        internal override IntPtr[] Values
        {
            get
            {
                if (disposed) throw new ObjectDisposedException(this.ToString());
                return new IntPtr[] { _count, _ptrValue, _ptrAddresses };
            }
        }

        /// <summary>
        /// ManagedCuda allocates an array as buffer and pins it in order to pass it to Cuda.<para/>
        /// You must free the buffer manually if the buffer is not needed anymore.
        /// </summary>
        public void FreeHandle()
        {
            foreach (var item in _namesPtr)
            {
                Marshal.FreeHGlobal(item);
            }
            _namesPtr = null;
            if (_handleNames.IsAllocated)
                _handleNames.Free();
            if (_handleAddresses.IsAllocated)
                _handleAddresses.Free();
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
    /// Enable link-time optimization (-dlto) for device code (0: false, default)<para/>
    /// Option type: int<para/>
    /// Applies to: compiler and linker
    /// </summary>
    [Obsolete("This jit option is deprecated and should not be used.")]
    public class CudaJOJITLto : CudaJitOption
    {
        /// <summary>
        /// Enable link-time optimization (-dlto) for device code (0: false, default)<para/>
        /// Option type: int<para/>
        /// Applies to: compiler and linker
        /// </summary>
        /// <param name="value"></param>
        public CudaJOJITLto(bool value)
        {
            _option = CUJITOption.Lto;
            _ptrValue = (IntPtr)(value ? 1 : 0);
        }
    }

    /// <summary>
    /// Control single-precision denormals (-ftz) support (0: false, default).<para/>
    /// 1 : flushes denormal values to zero<para/>
    /// 0 : preserves denormal values<para/>
    /// Option type: int<para/>
    /// Applies to: link-time optimization specified with CU_JIT_LTO
    /// </summary>
    [Obsolete("This jit option is deprecated and should not be used.")]
    public class CudaJOJITFtz : CudaJitOption
    {
        /// <summary>
        /// Control single-precision denormals (-ftz) support (0: false, default).<para/>
        /// 1 : flushes denormal values to zero<para/>
        /// 0 : preserves denormal values<para/>
        /// Option type: int<para/>
        /// Applies to: link-time optimization specified with CU_JIT_LTO
        /// </summary>
        /// <param name="value"></param>
        public CudaJOJITFtz(bool value)
        {
            _option = CUJITOption.Ftz;
            _ptrValue = (IntPtr)(value ? 1 : 0);
        }
    }


    /// <summary>
    /// Control single-precision floating-point division and reciprocals<para/>
    /// (-prec-div) support (1: true, default).<para/>
    /// 1 : Enables the IEEE round-to-nearest mode<para/>
    /// 0 : Enables the fast approximation mode<para/>
    /// Option type: int<para/>
    /// Applies to: link-time optimization specified with CU_JIT_LTO
    /// </summary>
    [Obsolete("This jit option is deprecated and should not be used.")]
    public class CudaJOJITPrecDiv : CudaJitOption
    {
        /// <summary>
        /// Control single-precision floating-point division and reciprocals<para/>
        /// (-prec-div) support (1: true, default).<para/>
        /// 1 : Enables the IEEE round-to-nearest mode<para/>
        /// 0 : Enables the fast approximation mode<para/>
        /// Option type: int<para/>
        /// Applies to: link-time optimization specified with CU_JIT_LTO
        /// </summary>
        /// <param name="value"></param>
        public CudaJOJITPrecDiv(bool value)
        {
            _option = CUJITOption.PrecDiv;
            _ptrValue = (IntPtr)(value ? 1 : 0);
        }
    }

    /// <summary>
    /// Control single-precision floating-point square root<para/>
    /// (-prec-sqrt) support (1: true, default).<para/>
    /// 1 : Enables the IEEE round-to-nearest mode<para/>
    /// 0 : Enables the fast approximation mode<para/>
    /// Option type: int\n<para/>
    /// Applies to: link-time optimization specified with CU_JIT_LTO
    /// </summary>
    [Obsolete("This jit option is deprecated and should not be used.")]
    public class CudaJOJITPrecSqrt : CudaJitOption
    {
        /// <summary>
        /// Control single-precision floating-point square root<para/>
        /// (-prec-sqrt) support (1: true, default).<para/>
        /// 1 : Enables the IEEE round-to-nearest mode<para/>
        /// 0 : Enables the fast approximation mode<para/>
        /// Option type: int\n<para/>
        /// Applies to: link-time optimization specified with CU_JIT_LTO
        /// </summary>
        /// <param name="value"></param>
        public CudaJOJITPrecSqrt(bool value)
        {
            _option = CUJITOption.PrecSqrt;
            _ptrValue = (IntPtr)(value ? 1 : 0);
        }
    }

    /// <summary>
    /// Enable/Disable the contraction of floating-point multiplies<para/>
    /// and adds/subtracts into floating-point multiply-add (-fma)<para/>
    /// operations (1: Enable, default; 0: Disable).<para/>
    /// Option type: int\n<para/>
    /// Applies to: link-time optimization specified with CU_JIT_LTO
    /// </summary>
    [Obsolete("This jit option is deprecated and should not be used.")]
    public class CudaJOJITFma : CudaJitOption
    {
        /// <summary>
        /// Enable/Disable the contraction of floating-point multiplies<para/>
        /// and adds/subtracts into floating-point multiply-add (-fma)<para/>
        /// operations (1: Enable, default; 0: Disable).<para/>
        /// Option type: int\n<para/>
        /// Applies to: link-time optimization specified with CU_JIT_LTO
        /// </summary>
        /// <param name="value"></param>
        public CudaJOJITFma(bool value)
        {
            _option = CUJITOption.Fma;
            _ptrValue = (IntPtr)(value ? 1 : 0);
        }
    }

    /// <summary>
    /// Array of kernel names that should be preserved at link time while others
    /// can be removed.\n
    /// Must contain ::CU_JIT_REFERENCED_KERNEL_COUNT entries.\n
    /// Note that kernel names can be mangled by the compiler in which case the
    /// mangled name needs to be specified.\n
    /// Wildcard "*" can be used to represent zero or more characters instead of
    /// specifying the full or mangled name.\n
    /// It is important to note that the wildcard "*" is also added implicitly.
    /// For example, specifying "foo" will match "foobaz", "barfoo", "barfoobaz" and
    /// thus preserve all kernels with those names. This can be avoided by providing
    /// a more specific name like "barfoobaz".\n
    /// Option type: const char **\n
    /// Applies to: dynamic linker only
    ///
    /// Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
    /// </summary>
    [Obsolete("This jit option is deprecated and should not be used.")]
    public class CudaJOReferencedKernelNames : CudaJitOption
    {
        IntPtr[] _namesPtr;
        GCHandle _handle;
        IntPtr _count;

        /// <summary>
        /// Array of kernel names that should be preserved at link time while others
        /// can be removed.\n
        /// Must contain ::CU_JIT_REFERENCED_KERNEL_COUNT entries.\n
        /// Note that kernel names can be mangled by the compiler in which case the
        /// mangled name needs to be specified.\n
        /// Wildcard "*" can be used to represent zero or more characters instead of
        /// specifying the full or mangled name.\n
        /// It is important to note that the wildcard "*" is also added implicitly.
        /// For example, specifying "foo" will match "foobaz", "barfoo", "barfoobaz" and
        /// thus preserve all kernels with those names. This can be avoided by providing
        /// a more specific name like "barfoobaz".\n
        /// Option type: const char **\n
        /// Applies to: dynamic linker only
        ///
        /// Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
        /// </summary>
        /// <param name="value"></param>
        public CudaJOReferencedKernelNames(string[] value)
        {
            int count = value.Length;
            _namesPtr = new IntPtr[count];

            for (int i = 0; i < count; i++)
            {
                _namesPtr[i] = Marshal.StringToHGlobalAnsi(value[i]);
            }

            _count = (IntPtr)value.Length;
            _handle = GCHandle.Alloc(_namesPtr, GCHandleType.Pinned);
            _ptrValue = _handle.AddrOfPinnedObject();
            _option = CUJITOption.ReferencedKernelNames;
        }

        internal override CUJITOption[] Options
        {
            get
            {
                return new CUJITOption[] { CUJITOption.ReferencedKernelCount, _option };
            }
        }

        internal override IntPtr[] Values
        {
            get
            {
                if (disposed) throw new ObjectDisposedException(this.ToString());
                return new IntPtr[] { _count, _ptrValue };
            }
        }

        /// <summary>
        /// ManagedCuda allocates an array as buffer and pins it in order to pass it to Cuda.<para/>
        /// You must free the buffer manually if the buffer is not needed anymore.
        /// </summary>
        public void FreeHandle()
        {
            foreach (var item in _namesPtr)
            {
                Marshal.FreeHGlobal(item);
            }
            _namesPtr = null;
            if (_handle.IsAllocated)
                _handle.Free();
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
    /// Array of variable names (__device__ and/or __constant__) that should be
    /// preserved at link time while others can be removed.\n
    /// Must contain ::CU_JIT_REFERENCED_VARIABLE_COUNT entries.\n
    /// Note that variable names can be mangled by the compiler in which case the
    /// mangled name needs to be specified.\n
    /// Wildcard "*" can be used to represent zero or more characters instead of
    /// specifying the full or mangled name.\n
    /// It is important to note that the wildcard "*" is also added implicitly.
    /// For example, specifying "foo" will match "foobaz", "barfoo", "barfoobaz" and
    /// thus preserve all variables with those names. This can be avoided by providing
    /// a more specific name like "barfoobaz".\n
    /// Option type: const char **\n
    /// Applies to: link-time optimization specified with CU_JIT_LTO
    ///
    /// Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
    /// </summary>
    [Obsolete("This jit option is deprecated and should not be used.")]
    public class CudaJOReferencedVariableNames : CudaJitOption
    {
        IntPtr[] _namesPtr;
        GCHandle _handle;
        IntPtr _count;

        /// <summary>
        /// Array of variable names (__device__ and/or __constant__) that should be
        /// preserved at link time while others can be removed.\n
        /// Must contain ::CU_JIT_REFERENCED_VARIABLE_COUNT entries.\n
        /// Note that variable names can be mangled by the compiler in which case the
        /// mangled name needs to be specified.\n
        /// Wildcard "*" can be used to represent zero or more characters instead of
        /// specifying the full or mangled name.\n
        /// It is important to note that the wildcard "*" is also added implicitly.
        /// For example, specifying "foo" will match "foobaz", "barfoo", "barfoobaz" and
        /// thus preserve all variables with those names. This can be avoided by providing
        /// a more specific name like "barfoobaz".\n
        /// Option type: const char **\n
        /// Applies to: link-time optimization specified with CU_JIT_LTO
        ///
        /// Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
        /// </summary>
        /// <param name="value"></param>
        public CudaJOReferencedVariableNames(string[] value)
        {
            int count = value.Length;
            _namesPtr = new IntPtr[count];

            for (int i = 0; i < count; i++)
            {
                _namesPtr[i] = Marshal.StringToHGlobalAnsi(value[i]);
            }

            _count = (IntPtr)value.Length;
            _handle = GCHandle.Alloc(_namesPtr, GCHandleType.Pinned);
            _ptrValue = _handle.AddrOfPinnedObject();
            _option = CUJITOption.ReferencedVariableNames;
        }

        internal override CUJITOption[] Options
        {
            get
            {
                return new CUJITOption[] { CUJITOption.ReferencedVariableCount, _option };
            }
        }

        internal override IntPtr[] Values
        {
            get
            {
                if (disposed) throw new ObjectDisposedException(this.ToString());
                return new IntPtr[] { _count, _ptrValue };
            }
        }

        /// <summary>
        /// ManagedCuda allocates an array as buffer and pins it in order to pass it to Cuda.<para/>
        /// You must free the buffer manually if the buffer is not needed anymore.
        /// </summary>
        public void FreeHandle()
        {
            foreach (var item in _namesPtr)
            {
                Marshal.FreeHGlobal(item);
            }
            _namesPtr = null;
            if (_handle.IsAllocated)
                _handle.Free();
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
    /// This option serves as a hint to enable the JIT compiler/linker
    /// to remove constant (__constant__) and device (__device__) variables
    /// unreferenced in device code (Disabled by default).\n
    /// Note that host references to constant and device variables using APIs like
    /// ::cuModuleGetGlobal() with this option specified may result in undefined behavior unless
    /// the variables are explicitly specified using ::CU_JIT_REFERENCED_VARIABLE_NAMES.\n
    /// Option type: int\n
    /// Applies to: link-time optimization specified with CU_JIT_LTO
    ///
    /// Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
    /// </summary>
    [Obsolete("This jit option is deprecated and should not be used.")]
    public class CudaJOOptimizeUnusedDeviceVariables : CudaJitOption
    {
        /// <summary>
        /// This option serves as a hint to enable the JIT compiler/linker
        /// to remove constant (__constant__) and device (__device__) variables
        /// unreferenced in device code (Disabled by default).\n
        /// Note that host references to constant and device variables using APIs like
        /// ::cuModuleGetGlobal() with this option specified may result in undefined behavior unless
        /// the variables are explicitly specified using ::CU_JIT_REFERENCED_VARIABLE_NAMES.\n
        /// Option type: int\n
        /// Applies to: link-time optimization specified with CU_JIT_LTO
        ///
        /// Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0
        /// </summary>
        /// <param name="value"></param>
        public CudaJOOptimizeUnusedDeviceVariables(bool value)
        {
            _option = CUJITOption.OptimizeUnusedDeviceVariables;
            _ptrValue = (IntPtr)(value ? 1 : 0);
        }
    }


    /// <summary>
    /// Generate position independent code (0: false)\n
    /// Option type: int\n
    /// Applies to: compiler only
    /// </summary>
    public class CudaJOPositionIndependentCode : CudaJitOption
    {
        /// <summary>
        /// Generate position independent code (0: false)\n
        /// Option type: int\n
        /// Applies to: compiler only
        /// </summary>
        /// <param name="value"></param>
        public CudaJOPositionIndependentCode(bool value)
        {
            _option = CUJITOption.PositionIndependentCode;
            _ptrValue = (IntPtr)(value ? 1 : 0);
        }
    }
}
