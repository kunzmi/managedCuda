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
    /// CudaArrayTexture1D
    /// </summary>
    public class CudaTextureArray1D : IDisposable
    {
        CUtexref _texref;
        CUFilterMode _filtermode;
        CUTexRefSetFlags _flags;
        CUAddressMode _addressMode;
        CUArrayFormat _format;
        SizeT _size;
        uint _channelSize;
        SizeT _dataSize;
        int _numChannels;
        string _name;
        CUmodule _module;
        CUfunction _cufunction;
        CudaArray1D _array;
        CUResult res;
        bool disposed;

        #region Constructors
        /// <summary>
        /// Creates a new 1D texture from array memory. Allocates new array.
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressMode"></param>
        /// <param name="filterMode"></param>
        /// <param name="flags"></param>
        /// <param name="format"></param>
        /// <param name="size">In elements</param>
        /// <param name="numChannels"></param>
        public CudaTextureArray1D(CudaKernel kernel, string texName, CUAddressMode addressMode, CUFilterMode filterMode, CUTexRefSetFlags flags, CUArrayFormat format, SizeT size, CudaArray1DNumChannels numChannels)
        {
            _texref = new CUtexref();
            res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref _texref, kernel.CUModule, texName);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
            if (res != CUResult.Success) throw new CudaException(res);

            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 0, addressMode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(_texref, filterMode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(_texref, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(_texref, format, (int)numChannels);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
            if (res != CUResult.Success) throw new CudaException(res);

            _filtermode = filterMode;
            _flags = flags;
            _addressMode = addressMode;
            _format = format;
            _size = size;
            _numChannels = (int)numChannels;
            _name = texName;
            _module = kernel.CUModule;
            _cufunction = kernel.CUFunction;

            _channelSize = CudaHelperMethods.GetChannelSize(format);
            _dataSize = size * _numChannels * _channelSize;
            _array = new CudaArray1D(format, size, numChannels);

            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetArray(_texref, _array.CUArray, CUTexRefSetArrayFlags.OverrideFormat);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetArray", res));
            if (res != CUResult.Success) throw new CudaException(res);
            //res = DriverAPINativeMethods.ParameterManagement.cuParamSetTexRef(_cufunction, CUParameterTexRef.Default, _texref);
            //Debug.WriteLine("{0:G}, {1}: {2}", DateTime.Now, "cuParamSetTexRef", res);
            //if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Creates a new 1D texture from array memory
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressMode"></param>
        /// <param name="filterMode"></param>
        /// <param name="flags"></param>
        /// <param name="array"></param>
        public CudaTextureArray1D(CudaKernel kernel, string texName, CUAddressMode addressMode, CUFilterMode filterMode, CUTexRefSetFlags flags, CudaArray1D array)
        {
            _texref = new CUtexref();
            res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref _texref, kernel.CUModule, texName);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
            if (res != CUResult.Success) throw new CudaException(res);

            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 0, addressMode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(_texref, filterMode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(_texref, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(_texref, array.ArrayDescriptor.Format, (int)array.ArrayDescriptor.NumChannels);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
            if (res != CUResult.Success) throw new CudaException(res);

            _filtermode = filterMode;
            _flags = flags;
            _addressMode = addressMode;
            _format = array.ArrayDescriptor.Format;
            _size = array.Width;
            _numChannels = (int)array.ArrayDescriptor.NumChannels;
            _name = texName;
            _module = kernel.CUModule;
            _cufunction = kernel.CUFunction;

            _channelSize = CudaHelperMethods.GetChannelSize(array.ArrayDescriptor.Format);
            _dataSize = array.Width * array.ArrayDescriptor.NumChannels * _channelSize;
            _array = array;

            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetArray(_texref, _array.CUArray, CUTexRefSetArrayFlags.OverrideFormat);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetArray", res));
            if (res != CUResult.Success) throw new CudaException(res);
            //res = DriverAPINativeMethods.ParameterManagement.cuParamSetTexRef(kernel.CUFunction, CUParameterTexRef.Default, _texref);
            //Debug.WriteLine("{0:G}, {1}: {2}", DateTime.Now, "cuParamSetTexRef", res);
            //if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~CudaTextureArray1D()
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
                _array.Dispose();
                disposed = true;
                // the _texref reference is not destroyed explicitly, as it is done automatically when module is unloaded
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

        #region Properties
        /// <summary>
        /// TextureReference
        /// </summary>
        public CUtexref TextureReference
        {
            get { return _texref; }
        }

        /// <summary>
        /// Flags
        /// </summary>
        public CUTexRefSetFlags Flags
        {
            get { return _flags; }
        }

        /// <summary>
        /// AddressMode
        /// </summary>
        public CUAddressMode AddressMode
        {
            get { return _addressMode; }
        }

        /// <summary>
        /// Format
        /// </summary>
        public CUArrayFormat Format
        {
            get { return _format; }
        }

        /// <summary>
        /// Format
        /// </summary>
        public CUFilterMode Filtermode
        {
            get { return _filtermode; }
        }

        /// <summary>
        /// Size
        /// </summary>
        public SizeT Size
        {
            get { return _size; }
        }

        /// <summary>
        /// ChannelSize
        /// </summary>
        public uint ChannelSize
        {
            get { return _channelSize; }
        }

        /// <summary>
        /// TotalSizeInBytes
        /// </summary>
        public SizeT TotalSizeInBytes
        {
            get { return _dataSize; }
        }

        /// <summary>
        /// NumChannels
        /// </summary>
        public int NumChannels
        {
            get { return _numChannels; }
        }

        /// <summary>
        /// Name
        /// </summary>
        public string Name
        {
            get { return _name; }
        }

        /// <summary>
        /// Module
        /// </summary>
        public CUmodule Module
        {
            get { return _module; }
        }

        /// <summary>
        /// CUFuntion
        /// </summary>
        public CUfunction CUFuntion
        {
            get { return _cufunction; }
        }

        /// <summary>
        /// Array
        /// </summary>
        public CudaArray1D Array
        {
            get { return _array; }
        }
		#endregion
    }
}
