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
    /// CudaLinearTexture2D
    /// </summary>
    public class CudaTextureLinearPitched2D<T> : IDisposable where T : struct
    {
        CUtexref _texref;
        CUFilterMode _filtermode;
        CUTexRefSetFlags _flags;
        CUAddressMode _addressMode0, _addressMode1;
        CUArrayFormat _format;
        SizeT _height;
        SizeT _width;
        uint _channelSize;
        SizeT _dataSize;
        int _numChannels;
        string _name;
        CUmodule _module;
        CUfunction _cufunction;
        CudaPitchedDeviceVariable<T> _devVar;
        CUResult res;
        bool disposed;

        #region Constructors
        /// <summary>
        /// Creates a new 2D texture from linear memory. Allocates a new device variable
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressMode"></param>
        /// <param name="filterMode"></param>
        /// <param name="flags"></param>
        /// <param name="format"></param>
        /// <param name="width">In elements</param>
        /// <param name="height">In elements</param>
        public CudaTextureLinearPitched2D(CudaKernel kernel, string texName, CUAddressMode addressMode, CUFilterMode filterMode, CUTexRefSetFlags flags, CUArrayFormat format, SizeT width, SizeT height)
            : this(kernel, texName, addressMode, addressMode, filterMode, flags, format, width, height)
        {

        }

        /// <summary>
        /// Creates a new 2D texture from linear memory. Allocates a new device variable
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressMode0"></param>
        /// <param name="addressMode1"></param>
        /// <param name="filterMode"></param>
        /// <param name="flags"></param>
        /// <param name="format"></param>
        /// <param name="width">In elements</param>
        /// <param name="height">In elements</param>
        public CudaTextureLinearPitched2D(CudaKernel kernel, string texName, CUAddressMode addressMode0, CUAddressMode addressMode1, CUFilterMode filterMode, CUTexRefSetFlags flags, CUArrayFormat format, SizeT width, SizeT height)
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
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(_texref, filterMode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(_texref, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
            if (res != CUResult.Success) throw new CudaException(res);
            _numChannels = CudaHelperMethods.GetNumChannels(typeof(T));
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(_texref, format, _numChannels);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
            if (res != CUResult.Success) throw new CudaException(res);

            _filtermode = filterMode;
            _flags = flags;
            _addressMode0 = addressMode0;
            _addressMode1 = addressMode1;
            _format = format;
            _height = height;
            _width = width;
            _name = texName;
            _module = kernel.CUModule;
            _cufunction = kernel.CUFunction;

            _channelSize = CudaHelperMethods.GetChannelSize(format);
            _dataSize = height * width * (uint)_numChannels * _channelSize;
            _devVar = new CudaPitchedDeviceVariable<T>(width, height);

            CUDAArrayDescriptor arrayDescr = new CUDAArrayDescriptor();
            arrayDescr.Format = format;
            arrayDescr.Height = height;
            arrayDescr.NumChannels = (uint)_numChannels;
            arrayDescr.Width = width;
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddress2D_v2(_texref, ref arrayDescr, _devVar.DevicePointer, _devVar.Pitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddress2D", res));
            if (res != CUResult.Success) throw new CudaException(res);
            //res = DriverAPINativeMethods.ParameterManagement.cuParamSetTexRef(kernel.CUFunction, CUParameterTexRef.Default, _texref);
            //Debug.WriteLine("{0:G}, {1}: {2}", DateTime.Now, "cuParamSetTexRef", res);
            //if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// Creates a new 2D texture from linear memory.
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressMode"></param>
        /// <param name="filterMode"></param>
        /// <param name="flags"></param>
        /// <param name="format"></param>
        /// <param name="deviceVar"></param>
        public CudaTextureLinearPitched2D(CudaKernel kernel, string texName, CUAddressMode addressMode, CUFilterMode filterMode, CUTexRefSetFlags flags, CUArrayFormat format, CudaPitchedDeviceVariable<T> deviceVar)
            : this(kernel, texName, addressMode, addressMode, filterMode, flags, format, deviceVar)
        {

        }

        /// <summary>
        /// Creates a new 2D texture from linear memory.
        /// </summary>
        /// <param name="kernel"></param>
        /// <param name="texName"></param>
        /// <param name="addressMode0"></param>
        /// <param name="addressMode1"></param>
        /// <param name="filterMode"></param>
        /// <param name="flags"></param>
        /// <param name="format"></param>
        /// <param name="deviceVar"></param>
        public CudaTextureLinearPitched2D(CudaKernel kernel, string texName, CUAddressMode addressMode0, CUAddressMode addressMode1, CUFilterMode filterMode, CUTexRefSetFlags flags, CUArrayFormat format, CudaPitchedDeviceVariable<T> deviceVar)
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
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(_texref, filterMode);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
            if (res != CUResult.Success) throw new CudaException(res);
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(_texref, flags);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
            if (res != CUResult.Success) throw new CudaException(res);
            _numChannels = CudaHelperMethods.GetNumChannels(typeof(T));
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(_texref, format, _numChannels);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
            if (res != CUResult.Success) throw new CudaException(res);

            _filtermode = filterMode;
            _flags = flags;
            _addressMode0 = addressMode0;
            _addressMode1 = addressMode1;
            _format = format;
            _height = deviceVar.Height;
            _width = deviceVar.Width;
            _name = texName;
            _module = kernel.CUModule;
            _cufunction = kernel.CUFunction;

            _channelSize = CudaHelperMethods.GetChannelSize(format);
            _dataSize = deviceVar.TotalSizeInBytes;
            _devVar = deviceVar;

            CUDAArrayDescriptor arrayDescr = new CUDAArrayDescriptor();
            arrayDescr.Format = format;
            arrayDescr.Height = _height;
            arrayDescr.NumChannels = (uint)_numChannels;
            arrayDescr.Width = _width;
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddress2D_v2(_texref, ref arrayDescr, _devVar.DevicePointer, _devVar.Pitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddress2D", res));
            if (res != CUResult.Success) throw new CudaException(res);
            //res = DriverAPINativeMethods.ParameterManagement.cuParamSetTexRef(kernel.CUFunction, CUParameterTexRef.Default, _texref);
            //Debug.WriteLine("{0:G}, {1}: {2}", DateTime.Now, "cuParamSetTexRef", res);
            //if (res != CUResult.Success) throw new CudaException(res);
        }

        /// <summary>
        /// For dispose
        /// </summary>
        ~CudaTextureLinearPitched2D()
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
                _devVar.Dispose();
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
        /// CUFunction
        /// </summary>
        public CUfunction CUFuntion
        {
            get { return _cufunction; }
        }

        /// <summary>
        /// Device variable in linear Memory
        /// </summary>
        public CudaPitchedDeviceVariable<T> DeviceVar
        {
            get { return _devVar; }
        }
        #endregion

        #region Methods
        /// <summary>
        /// Binds a linear address range to the texture reference. <para/>
        /// Any previous address or CUDA array state associated with the texture reference is superseded by this function. <para/>
        /// Any memory previously bound to the texture reference is unbound.<para/>
        /// Size my differ to the previous bound variable, but type must be the same.
        /// </summary>
        /// <param name="deviceVar">New device variable to bind this texture reference to.</param>
        public void Reset(CudaPitchedDeviceVariable<T> deviceVar)
        {
            if (disposed) throw new ObjectDisposedException(this.ToString());

            _height = deviceVar.Height;
            _width = deviceVar.Width;
            _dataSize = deviceVar.TotalSizeInBytes;
            _devVar = deviceVar;


            CUDAArrayDescriptor arrayDescr = new CUDAArrayDescriptor();
            arrayDescr.Format = _format;
            arrayDescr.Height = _height;
            arrayDescr.NumChannels = (uint)_numChannels;
            arrayDescr.Width = _width;
            res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddress2D_v2(_texref, ref arrayDescr, _devVar.DevicePointer, _devVar.Pitch);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddress2D", res));
            if (res != CUResult.Success) throw new CudaException(res);
        }
        #endregion
    }

}
