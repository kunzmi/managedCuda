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
    /// CudaArrayTexture3D
    /// </summary>
    public class CudaTextureArray3D : IDisposable
    {
        CUtexref _texref;
        CUFilterMode _filtermode;
        CUTexRefSetFlags _flags;
        CUAddressMode _addressMode0;
        CUAddressMode _addressMode1;
        CUAddressMode _addressMode2;
        CUArrayFormat _format;
        SizeT _height;
        SizeT _width;
        SizeT _depth;
        uint _channelSize;
        SizeT _dataSize;
        int _numChannels;
        string _name;
        CUmodule _module;
        CUfunction _cufunction;
        CudaArray3D _array;
        CUResult res;
		bool disposed;

        #region Constructors
        /// <summary>
        /// Creates a new 3D texture from array memory. Allocates a new 3D array.
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressModeForAllDimensions"></param>
        /// <param name="filterMode"></param>
        /// <param name="flags"></param>
        /// <param name="format"></param>
        /// <param name="width">In elements</param>
        /// <param name="height">In elements</param>
        /// <param name="depth">In elements</param>
        /// <param name="numChannels">1,2 or 4</param>
        public CudaTextureArray3D(CudaKernel kernel, string texName, CUAddressMode addressModeForAllDimensions, CUFilterMode filterMode, CUTexRefSetFlags flags, CUArrayFormat format, SizeT width, SizeT height, SizeT depth, CudaArray3DNumChannels numChannels)
            : this(kernel, texName, addressModeForAllDimensions, addressModeForAllDimensions, addressModeForAllDimensions, filterMode, flags, format, width, height, depth, numChannels)
        {

        }

        /// <summary>
        /// Creates a new 3D texture from array memory. Allocates a new 3D array.
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressMode0"></param>
        /// <param name="addressMode1"></param>
        /// <param name="addressMode2"></param>
        /// <param name="filterMode"></param>
        /// <param name="flags"></param>
        /// <param name="format"></param>
        /// <param name="width">In elements</param>
        /// <param name="height">In elements</param>
        /// <param name="depth">In elements</param>
        /// <param name="numChannels">1,2 or 4</param>
        public CudaTextureArray3D(CudaKernel kernel, string texName, CUAddressMode addressMode0, CUAddressMode addressMode1, CUAddressMode addressMode2, CUFilterMode filterMode, CUTexRefSetFlags flags, CUArrayFormat format, SizeT width, SizeT height, SizeT depth, CudaArray3DNumChannels numChannels)
        {
            _texref = new CUtexref();
            res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref _texref, kernel.CUModule, texName);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
            if (res != CUResult.Success) throw new CudaException(res);

            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 0, addressMode0);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 1, addressMode1);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 2, addressMode2);
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
            _addressMode0 = addressMode0;
            _addressMode1 = addressMode1;
            _addressMode2 = addressMode2;
            _format = format;
            _height = height;
            _width = width;
            _depth = depth;
            _numChannels = (int)numChannels;
            _name = texName;
            _module = kernel.CUModule;
            _cufunction = kernel.CUFunction;

            _channelSize = CudaHelperMethods.GetChannelSize(format);
            _dataSize = height * width * depth * _numChannels * _channelSize;
            _array = new CudaArray3D(format, width, height, depth, numChannels, CUDAArray3DFlags.None);

            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetArray(_texref, _array.CUArray, CUTexRefSetArrayFlags.OverrideFormat);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetArray", res));
            if (res != CUResult.Success) throw new CudaException(res);
            //res = DriverAPINativeMethods.ParameterManagement.cuParamSetTexRef(kernel.CUFunction, CUParameterTexRef.Default, _texref);
            //Debug.WriteLine("{0:G}, {1}: {2}", DateTime.Now, "cuParamSetTexRef", res);
            //if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Creates a new 3D texture from array memory
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressModeForAllDimensions"></param>
        /// <param name="filterMode"></param>
        /// <param name="flags"></param>
        /// <param name="array"></param>
        public CudaTextureArray3D(CudaKernel kernel, string texName, CUAddressMode addressModeForAllDimensions, CUFilterMode filterMode, CUTexRefSetFlags flags, CudaArray3D array)
            : this(kernel, texName, addressModeForAllDimensions, addressModeForAllDimensions, addressModeForAllDimensions, filterMode, flags, array)
        {

        }

        /// <summary>
        /// Creates a new 3D texture from array memory
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressMode0"></param>
        /// <param name="addressMode1"></param>
        /// <param name="addressMode2"></param>
        /// <param name="filterMode"></param>
        /// <param name="flags"></param>
        /// <param name="array"></param>
        public CudaTextureArray3D(CudaKernel kernel, string texName, CUAddressMode addressMode0, CUAddressMode addressMode1, CUAddressMode addressMode2, CUFilterMode filterMode, CUTexRefSetFlags flags, CudaArray3D array)
        {
            _texref = new CUtexref();
            res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref _texref, kernel.CUModule, texName);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
            if (res != CUResult.Success) throw new CudaException(res);

            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 0, addressMode0);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 1, addressMode1);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(_texref, 2, addressMode2);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(_texref, filterMode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(_texref, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(_texref, array.Array3DDescriptor.Format, (int)array.Array3DDescriptor.NumChannels);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
            if (res != CUResult.Success) throw new CudaException(res);

            _filtermode = filterMode;
            _flags = flags;
            _addressMode0 = addressMode0;
            _addressMode1 = addressMode1;
            _addressMode2 = addressMode2;
            _format = array.Array3DDescriptor.Format;
            _height = array.Height;
            _depth = array.Width;
            _width = array.Depth;
            _numChannels = (int)array.Array3DDescriptor.NumChannels;
            _name = texName;
            _module = kernel.CUModule;
            _cufunction = kernel.CUFunction;

            _channelSize = CudaHelperMethods.GetChannelSize(array.Array3DDescriptor.Format);
            _dataSize = _height * _width * _depth * (uint)array.Array3DDescriptor.NumChannels * _channelSize;
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
        ~CudaTextureArray3D()
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
        public CUAddressMode AddressMode0
        {
            get { return _addressMode0; }
        }

        /// <summary>
        /// AddressMode
        /// </summary>
        public CUAddressMode AddressMode1
        {
            get { return _addressMode1; }
        }

        /// <summary>
        /// AddressMode
        /// </summary>
        public CUAddressMode AddressMode2
        {
            get { return _addressMode2; }
        }

        /// <summary>
        /// Format
        /// </summary>
        public CUArrayFormat Format
        {
            get { return _format; }
        }

        /// <summary>
        /// Filtermode
        /// </summary>
        public CUFilterMode Filtermode
        {
            get { return _filtermode; }
        }

        /// <summary>
        /// Depth
        /// </summary>
        public SizeT Depth
        {
            get { return _depth; }
        }

        /// <summary>
        /// Height
        /// </summary>
        public SizeT Height
        {
            get { return _height; }
        }

        /// <summary>
        /// Width
        /// </summary>
        public SizeT Width
        {
            get { return _width; }
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
        public CudaArray3D Array
        {
            get { return _array; }
        }
        #endregion
    }
}
