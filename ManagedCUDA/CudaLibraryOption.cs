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
using System.Diagnostics;

namespace ManagedCuda
{
    /// <summary>
    /// A list of library load option passed to Cuda.<para/>
    /// Maximum number of options is limited to 30.
    /// </summary>
    public class CudaLibraryOptionCollection : IDisposable
    {
        /// <summary/>
        protected bool disposed;
        private const int MAX_ELEM = 32;
        private CUlibraryOption[] _options = new CUlibraryOption[MAX_ELEM];
        private IntPtr[] _values = new IntPtr[MAX_ELEM];
        private List<CudaLibraryOption> _cudaOptions = new List<CudaLibraryOption>();
        private int _count = 0;

        /// <summary>
        /// Add a single option to the collection.
        /// </summary>
        /// <param name="opt">Option to add</param>
        public void Add(CudaLibraryOption opt)
        {
            if (_count >= MAX_ELEM - 2)
                throw new Exception("Maximum number of options elements reached!");

            _cudaOptions.Add(opt);

            CUlibraryOption[] o = opt.Options;
            IntPtr[] v = opt.Values;

            opt.Index = _count;
            for (int i = 0; i < o.Length; i++)
            {
                _options[_count] = o[i];
                _values[_count] = v[i];
                _count++;
            }
        }

        /// <summary>
        /// A multiple options to the collection.
        /// </summary>
        /// <param name="options">Options to add</param>
        public void Add(IList<CudaLibraryOption> options)
        {
            foreach (var item in options)
            {
                Add(item);
            }
        }

        internal CUlibraryOption[] Options
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
        ~CudaLibraryOptionCollection()
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
    public abstract class CudaLibraryOption : IDisposable
    {
        /// <summary>
        /// Option value converted to (void *)
        /// </summary>
        protected IntPtr _ptrValue;
        /// <summary>
        /// Option
        /// </summary>
        protected CUlibraryOption _option;
        private int _index;
        /// <summary/>
        protected bool disposed;

        internal virtual CUlibraryOption[] Options
        {
            get { return new CUlibraryOption[] { _option }; }
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
        ~CudaLibraryOption()
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
    /// <para>Unknown</para>
    /// </summary>
    public class CudaLOHostUniversalFunctionAndDataTable : CudaLibraryOption
    {
        /// <summary>
        /// <para>Unknown</para>
        /// </summary>
        /// <param name="value"></param>
        public CudaLOHostUniversalFunctionAndDataTable(uint value)
        {
            _option = CUlibraryOption.HostUniversalFunctionAndDataTable;
            _ptrValue = (IntPtr)(Convert.ToUInt32(value, System.Globalization.CultureInfo.InvariantCulture));
        }
    }
    /// <summary>
    /// Specifes that the argument \p code passed to ::cuLibraryLoadData() will be preserved.
    /// Specifying this option will let the driver know that \p code can be accessed at any point
    /// until ::cuLibraryUnload(). The default behavior is for the driver to allocate and
    /// maintain its own copy of \p code. Note that this is only a memory usage optimization
    /// hint and the driver can choose to ignore it if required.
    /// Specifying this option with ::cuLibraryLoadFromFile() is invalid and
    /// will return ::CUDA_ERROR_INVALID_VALUE.
    /// </summary>
    public class CudaLOBinaryIsPreserved : CudaLibraryOption
    {
        /// <summary>
        /// Specifes that the argument \p code passed to ::cuLibraryLoadData() will be preserved.
        /// Specifying this option will let the driver know that \p code can be accessed at any point
        /// until ::cuLibraryUnload(). The default behavior is for the driver to allocate and
        /// maintain its own copy of \p code. Note that this is only a memory usage optimization
        /// hint and the driver can choose to ignore it if required.
        /// Specifying this option with ::cuLibraryLoadFromFile() is invalid and
        /// will return ::CUDA_ERROR_INVALID_VALUE.
        /// </summary>
        /// <param name="value"></param>
        public CudaLOBinaryIsPreserved(bool value)
        {
            _option = CUlibraryOption.BinaryIsPreserved;
            _ptrValue = (IntPtr)(value ? 1 : 0);
        }
    }
}
