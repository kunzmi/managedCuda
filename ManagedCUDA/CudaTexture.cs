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
	/// Provides methods to bind texture references to kernels
	/// </summary>
	public static class CudaTexture
	{
		#region Linear1D
		/// <summary>
		/// Create a new CudaDeviceVariable and bind it to a texture reference.
		/// </summary>
		/// <param name="kernel"></param>
		/// <param name="texName"></param>
		/// <param name="addressMode0"></param>
		/// <param name="flags"></param>
		/// <param name="format"></param>
		/// <param name="size">In elements</param>
		public static CudaDeviceVariable<T> BindTexture<T>(CudaKernel kernel, string texName, CUTexRefSetFlags flags, CUAddressMode addressMode0, CUArrayFormat format, SizeT size) where T : struct
		{
			CUtexref texref = new CUtexref();
			CUResult res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref texref, kernel.CUModule, texName);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 0, addressMode0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(texref, CUFilterMode.Point); //Textures from linear memory can only by point filtered
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(texref, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(texref, format, CudaHelperMethods.GetNumChannels(typeof(T)));
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
			if (res != CUResult.Success) throw new CudaException(res);

			CudaDeviceVariable<T> devVar = new CudaDeviceVariable<T>(size);

			SizeT NULL = 0;
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddress_v2(ref NULL, texref, devVar.DevicePointer, devVar.SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddress", res));
			if (res != CUResult.Success) throw new CudaException(res);

			return devVar;
		}

		/// <summary>
		/// Bind a CudaDeviceVariable to a texture reference.
		/// </summary>
		/// <param name="kernel"></param>
		/// <param name="texName"></param>
		/// <param name="addressMode0"></param>
		/// <param name="flags"></param>
		/// <param name="format"></param>
		/// <param name="deviceVar"></param>
		public static void BindTexture<T>(CudaKernel kernel, string texName, CUAddressMode addressMode0, CUTexRefSetFlags flags, CUArrayFormat format, CudaDeviceVariable<T> deviceVar) where T : struct
		{
			CUtexref texref = new CUtexref();
			CUResult res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref texref, kernel.CUModule, texName);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 0, addressMode0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(texref, CUFilterMode.Point);//Textures from linear memory can only by point filtered
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(texref, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(texref, format, CudaHelperMethods.GetNumChannels(typeof(T)));
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
			if (res != CUResult.Success) throw new CudaException(res);

			SizeT NULL = 0;
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddress_v2(ref NULL, texref, deviceVar.DevicePointer, deviceVar.SizeInBytes);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddress", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}
		#endregion

		#region Linear2D
		/// <summary>
		/// Create a new CudaPitchedDeviceVariable and bind it to a texture reference.
		/// </summary>
		/// <param name="kernel"></param>
		/// <param name="texName"></param>
		/// <param name="addressMode"></param>
		/// <param name="filterMode"></param>
		/// <param name="flags"></param>
		/// <param name="format"></param>
		/// <param name="width">In elements</param>
		/// <param name="height">In elements</param>
		public static CudaPitchedDeviceVariable<T> BindTexture<T>(CudaKernel kernel, string texName, CUAddressMode addressMode, CUFilterMode filterMode, CUTexRefSetFlags flags, CUArrayFormat format, SizeT width, SizeT height) where T : struct
		{
			return BindTexture<T>(kernel, texName, addressMode, addressMode, filterMode, flags, format, width, height);
		}

		/// <summary>
		/// Create a new CudaPitchedDeviceVariable and bind it to a texture reference.
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
		public static CudaPitchedDeviceVariable<T> BindTexture<T>(CudaKernel kernel, string texName, CUAddressMode addressMode0, CUAddressMode addressMode1, CUFilterMode filterMode, CUTexRefSetFlags flags, CUArrayFormat format, SizeT width, SizeT height) where T : struct
		{
			CUtexref texref = new CUtexref();
			CUResult res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref texref, kernel.CUModule, texName);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 0, addressMode0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 1, addressMode1);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(texref, filterMode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(texref, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
			if (res != CUResult.Success) throw new CudaException(res);
			int numChannels = CudaHelperMethods.GetNumChannels(typeof(T));
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(texref, format, numChannels);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
			if (res != CUResult.Success) throw new CudaException(res);

			CudaPitchedDeviceVariable<T> devVar = new CudaPitchedDeviceVariable<T>(width, height);

			CUDAArrayDescriptor arrayDescr = new CUDAArrayDescriptor();
			arrayDescr.Format = format;
			arrayDescr.Height = height;
			arrayDescr.NumChannels = (uint)numChannels;
			arrayDescr.Width = width;
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddress2D_v2(texref, ref arrayDescr, devVar.DevicePointer, devVar.Pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddress2D", res));
			if (res != CUResult.Success) throw new CudaException(res);
			return devVar;
		}

		/// <summary>
		/// Bind a CudaPitchedDeviceVariable to a texture reference.
		/// </summary>
		/// <param name="kernel"></param>
		/// <param name="texName"></param>
		/// <param name="addressMode"></param>
		/// <param name="filterMode"></param>
		/// <param name="flags"></param>
		/// <param name="format"></param>
		/// <param name="deviceVar"></param>
		public static void BindTexture<T>(CudaKernel kernel, string texName, CUAddressMode addressMode, CUFilterMode filterMode, CUTexRefSetFlags flags, CUArrayFormat format, CudaPitchedDeviceVariable<T> deviceVar) where T : struct
		{
			BindTexture(kernel, texName, addressMode, addressMode, filterMode, flags, format, deviceVar);
		}

		/// <summary>
		/// Bind a CudaPitchedDeviceVariable to a texture reference.
		/// </summary>
		/// <param name="kernel"></param>
		/// <param name="texName"></param>
		/// <param name="addressMode0"></param>
		/// <param name="addressMode1"></param>
		/// <param name="filterMode"></param>
		/// <param name="flags"></param>
		/// <param name="format"></param>
		/// <param name="deviceVar"></param>
		public static void BindTexture<T>(CudaKernel kernel, string texName, CUAddressMode addressMode0, CUAddressMode addressMode1, CUFilterMode filterMode, CUTexRefSetFlags flags, CUArrayFormat format, CudaPitchedDeviceVariable<T> deviceVar) where T : struct
		{
			CUtexref texref = new CUtexref();
			CUResult res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref texref, kernel.CUModule, texName);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 0, addressMode0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 1, addressMode1);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(texref, filterMode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(texref, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(texref, format, CudaHelperMethods.GetNumChannels(typeof(T)));
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
			if (res != CUResult.Success) throw new CudaException(res);

			CUDAArrayDescriptor arrayDescr = new CUDAArrayDescriptor();
			arrayDescr.Format = format;
			arrayDescr.Height = deviceVar.Height;
			arrayDescr.NumChannels = (uint)CudaHelperMethods.GetNumChannels(typeof(T));
			arrayDescr.Width = deviceVar.Width;
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddress2D_v2(texref, ref arrayDescr, deviceVar.DevicePointer, deviceVar.Pitch);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddress2D", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}
		#endregion

		#region Array1D
		/// <summary>
		/// Create a new CudaArray1D and bind it to a texture reference.
		/// </summary>
		/// <param name="kernel"></param>
		/// <param name="texName"></param>
		/// <param name="addressMode"></param>
		/// <param name="filterMode"></param>
		/// <param name="flags"></param>
		/// <param name="format"></param>
		/// <param name="size">In elements</param>
		/// <param name="numChannels"></param>
		public static CudaArray1D BindTexture(CudaKernel kernel, string texName, CUAddressMode addressMode, CUFilterMode filterMode, CUTexRefSetFlags flags, CUArrayFormat format, SizeT size, CudaArray1DNumChannels numChannels)
		{
			CUtexref texref = new CUtexref();
			CUResult res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref texref, kernel.CUModule, texName);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 0, addressMode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(texref, filterMode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(texref, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(texref, format, (int)numChannels);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
			if (res != CUResult.Success) throw new CudaException(res);

			CudaArray1D array = new CudaArray1D(format, size, numChannels);

			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetArray(texref, array.CUArray, CUTexRefSetArrayFlags.OverrideFormat);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetArray", res));
			if (res != CUResult.Success) throw new CudaException(res);

			return array;
		}

		/// <summary>
		/// Bind a CudaArray1D to a texture reference.
		/// </summary>
		/// <param name="kernel"></param>
		/// <param name="texName"></param>
		/// <param name="addressMode"></param>
		/// <param name="filterMode"></param>
		/// <param name="flags"></param>
		/// <param name="array"></param>
		public static void BindTexture(CudaKernel kernel, string texName, CUAddressMode addressMode, CUFilterMode filterMode, CUTexRefSetFlags flags, CudaArray1D array)
		{
			CUtexref texref = new CUtexref();
			CUResult res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref texref, kernel.CUModule, texName);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 0, addressMode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(texref, filterMode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(texref, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(texref, array.ArrayDescriptor.Format, (int)array.ArrayDescriptor.NumChannels);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
			if (res != CUResult.Success) throw new CudaException(res);


			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetArray(texref, array.CUArray, CUTexRefSetArrayFlags.OverrideFormat);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetArray", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}
		#endregion

		#region Array2D
		/// <summary>
		/// Create a new CudaArray2D and bind it to a texture reference.
		/// </summary>
		/// <param name="kernel"></param>
		/// <param name="texName"></param>
		/// <param name="addressMode"></param>
		/// <param name="filterMode"></param>
		/// <param name="flags"></param>
		/// <param name="format"></param>
		/// <param name="height">In elements</param>
		/// <param name="width">In elements</param>
		/// <param name="numChannels">1,2 or 4</param>
		public static CudaArray2D BindTexture(CudaKernel kernel, string texName, CUAddressMode addressMode, CUFilterMode filterMode, CUTexRefSetFlags flags, CUArrayFormat format, SizeT width, SizeT height, CudaArray2DNumChannels numChannels)
		{
			return BindTexture(kernel, texName, addressMode, addressMode, filterMode, flags, format, width, height, numChannels);
		}

		/// <summary>
		/// Create a new CudaArray2D and bind it to a texture reference.
		/// </summary>
		/// <param name="kernel"></param>
		/// <param name="texName"></param>
		/// <param name="addressMode0"></param>
		/// <param name="addressMode1"></param>
		/// <param name="filterMode"></param>
		/// <param name="flags"></param>
		/// <param name="format"></param>
		/// <param name="height">In elements</param>
		/// <param name="width">In elements</param>
		/// <param name="numChannels">1,2 or 4</param>
		public static CudaArray2D BindTexture(CudaKernel kernel, string texName, CUAddressMode addressMode0, CUAddressMode addressMode1, CUFilterMode filterMode, CUTexRefSetFlags flags, CUArrayFormat format, SizeT width, SizeT height, CudaArray2DNumChannels numChannels)
		{
			CUtexref texref = new CUtexref();
			CUResult res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref texref, kernel.CUModule, texName);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 0, addressMode0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 1, addressMode1);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(texref, filterMode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(texref, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(texref, format, (int)numChannels);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
			if (res != CUResult.Success) throw new CudaException(res);

			CudaArray2D array = new CudaArray2D(format, width, height, numChannels);

			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetArray(texref, array.CUArray, CUTexRefSetArrayFlags.OverrideFormat);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetArray", res));
			if (res != CUResult.Success) throw new CudaException(res);

			return array;
		}

		/// <summary>
		/// Bind a CudaArray2D to a texture reference.
		/// </summary>
		/// <param name="kernel"></param>
		/// <param name="texName"></param>
		/// <param name="addressMode"></param>
		/// <param name="filterMode"></param>
		/// <param name="flags"></param>
		/// <param name="array"></param>
		public static void BindTexture(CudaKernel kernel, string texName, CUAddressMode addressMode, CUFilterMode filterMode, CUTexRefSetFlags flags, CudaArray2D array)
		{
			BindTexture(kernel, texName, addressMode, addressMode, filterMode, flags, array);
		}

		/// <summary>
		/// Bind a CudaArray2D to a texture reference.
		/// </summary>
		/// <param name="kernel"></param>
		/// <param name="texName"></param>
		/// <param name="addressMode0"></param>
		/// <param name="addressMode1"></param>
		/// <param name="filterMode"></param>
		/// <param name="flags"></param>
		/// <param name="array"></param>
		public static void BindTexture(CudaKernel kernel, string texName, CUAddressMode addressMode0, CUAddressMode addressMode1, CUFilterMode filterMode, CUTexRefSetFlags flags, CudaArray2D array)
		{
			CUtexref texref = new CUtexref();
			CUResult res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref texref, kernel.CUModule, texName);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 0, addressMode0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 1, addressMode1);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(texref, filterMode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(texref, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(texref, array.ArrayDescriptor.Format, (int)array.ArrayDescriptor.NumChannels);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetArray(texref, array.CUArray, CUTexRefSetArrayFlags.OverrideFormat);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetArray", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}
		#endregion

		#region Array3D
		/// <summary>
		/// Create a new CudaArray3D and bind it to a texture reference.
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
		public static CudaArray3D BindTexture(CudaKernel kernel, string texName, CUAddressMode addressModeForAllDimensions, CUFilterMode filterMode, CUTexRefSetFlags flags, CUArrayFormat format, SizeT width, SizeT height, SizeT depth, CudaArray3DNumChannels numChannels)
		{
			return BindTexture(kernel, texName, addressModeForAllDimensions, addressModeForAllDimensions, addressModeForAllDimensions, filterMode, flags, format, width, height, depth, numChannels);
		}

		/// <summary>
		/// Create a new CudaArray3D and bind it to a texture reference.
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
		public static CudaArray3D BindTexture(CudaKernel kernel, string texName, CUAddressMode addressMode0, CUAddressMode addressMode1, CUAddressMode addressMode2, CUFilterMode filterMode, CUTexRefSetFlags flags, CUArrayFormat format, SizeT width, SizeT height, SizeT depth, CudaArray3DNumChannels numChannels)
		{
			CUtexref texref = new CUtexref();
			CUResult res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref texref, kernel.CUModule, texName);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 0, addressMode0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 1, addressMode1);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 2, addressMode2);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(texref, filterMode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(texref, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(texref, format, (int)numChannels);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
			if (res != CUResult.Success) throw new CudaException(res);

			CudaArray3D array = new CudaArray3D(format, width, height, depth, numChannels, CUDAArray3DFlags.None);

			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetArray(texref, array.CUArray, CUTexRefSetArrayFlags.OverrideFormat);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetArray", res));
			if (res != CUResult.Success) throw new CudaException(res);

			return array;
		}

		/// <summary>
		/// Bind a CudaArray3D to a texture reference.
		/// </summary>
		/// <param name="kernel"></param>
		/// <param name="texName"></param>
		/// <param name="addressModeForAllDimensions"></param>
		/// <param name="filterMode"></param>
		/// <param name="flags"></param>
		/// <param name="array"></param>
		public static void BindTexture(CudaKernel kernel, string texName, CUAddressMode addressModeForAllDimensions, CUFilterMode filterMode, CUTexRefSetFlags flags, CudaArray3D array)
		{
			BindTexture(kernel, texName, addressModeForAllDimensions, addressModeForAllDimensions, addressModeForAllDimensions, filterMode, flags, array);
		}

		/// <summary>
		/// Bind a CudaArray3D to a texture reference.
		/// </summary>
		/// <param name="kernel"></param>
		/// <param name="texName"></param>
		/// <param name="addressMode0"></param>
		/// <param name="addressMode1"></param>
		/// <param name="addressMode2"></param>
		/// <param name="filterMode"></param>
		/// <param name="flags"></param>
		/// <param name="array"></param>
		public static void BindTexture(CudaKernel kernel, string texName, CUAddressMode addressMode0, CUAddressMode addressMode1, CUAddressMode addressMode2, CUFilterMode filterMode, CUTexRefSetFlags flags, CudaArray3D array)
		{
			CUtexref texref = new CUtexref();
			CUResult res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref texref, kernel.CUModule, texName);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 0, addressMode0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 1, addressMode1);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 2, addressMode2);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(texref, filterMode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(texref, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(texref, array.Array3DDescriptor.Format, (int)array.Array3DDescriptor.NumChannels);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetArray(texref, array.CUArray, CUTexRefSetArrayFlags.OverrideFormat);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetArray", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}
		#endregion

		#region MipmappedArray
		/// <summary>
		/// Create a new CudaMipmappedArray and bind it to a texture reference.
		/// </summary>
		/// <param name="kernel"></param>
		/// <param name="texName"></param>
		/// <param name="addressModeForAllDimensions"></param>
		/// <param name="filterMode"></param>
		/// <param name="flags"></param>
		/// <param name="descriptor"></param>
		/// <param name="numMipmapLevels"></param>
		/// <param name="maxAniso"></param>
		/// <param name="mipmapFilterMode"></param>
		/// <param name="mipmapLevelBias"></param>
		/// <param name="minMipmapLevelClamp"></param>
		/// <param name="maxMipmapLevelClamp"></param>
		public static CudaMipmappedArray BindTexture(CudaKernel kernel, string texName, CUAddressMode addressModeForAllDimensions,
			CUFilterMode filterMode, CUTexRefSetFlags flags, CUDAArray3DDescriptor descriptor, uint numMipmapLevels,
			uint maxAniso, CUFilterMode mipmapFilterMode, float mipmapLevelBias, float minMipmapLevelClamp, float maxMipmapLevelClamp)
		{
			return BindTexture(kernel, texName, addressModeForAllDimensions, addressModeForAllDimensions, addressModeForAllDimensions, filterMode, flags, descriptor,
			numMipmapLevels, maxAniso, mipmapFilterMode, mipmapLevelBias, minMipmapLevelClamp, maxMipmapLevelClamp);
		}

		/// <summary>
		/// Create a new CudaMipmappedArray and bind it to a texture reference.
		/// </summary>
		/// <param name="kernel"></param>
		/// <param name="texName"></param>
		/// <param name="addressMode0"></param>
		/// <param name="addressMode1"></param>
		/// <param name="addressMode2"></param>
		/// <param name="filterMode"></param>
		/// <param name="flags"></param>
		/// <param name="descriptor"></param>
		/// <param name="numMipmapLevels"></param>
		/// <param name="maxAniso"></param>
		/// <param name="mipmapFilterMode"></param>
		/// <param name="mipmapLevelBias"></param>
		/// <param name="minMipmapLevelClamp"></param>
		/// <param name="maxMipmapLevelClamp"></param>
		public static CudaMipmappedArray BindTexture(CudaKernel kernel, string texName, CUAddressMode addressMode0, CUAddressMode addressMode1, CUAddressMode addressMode2,
			CUFilterMode filterMode, CUTexRefSetFlags flags, CUDAArray3DDescriptor descriptor, uint numMipmapLevels,
			uint maxAniso, CUFilterMode mipmapFilterMode, float mipmapLevelBias, float minMipmapLevelClamp, float maxMipmapLevelClamp)
		{
			CUtexref texref = new CUtexref();
			CUResult res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref texref, kernel.CUModule, texName);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 0, addressMode0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 1, addressMode1);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 2, addressMode2);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(texref, filterMode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(texref, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(texref, descriptor.Format, (int)descriptor.NumChannels);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
			if (res != CUResult.Success) throw new CudaException(res);

			CudaMipmappedArray array = new CudaMipmappedArray(descriptor, numMipmapLevels);

			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetMipmappedArray(texref, array.CUMipmappedArray, CUTexRefSetArrayFlags.OverrideFormat);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetMipmappedArray", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetMaxAnisotropy(texref, maxAniso);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetMaxAnisotropy", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetMipmapFilterMode(texref, mipmapFilterMode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetMipmapFilterMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetMipmapLevelBias(texref, mipmapLevelBias);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetMipmapLevelBias", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetMipmapLevelClamp(texref, minMipmapLevelClamp, maxMipmapLevelClamp);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetMipmapLevelClamp", res));
			if (res != CUResult.Success) throw new CudaException(res);

			return array;
		}

		/// <summary>
		/// Bind a CudaMipmappedArray to a texture reference.
		/// </summary>
		/// <param name="kernel"></param>
		/// <param name="texName"></param>
		/// <param name="addressModeForAllDimensions"></param>
		/// <param name="filterMode"></param>
		/// <param name="flags"></param>
		/// <param name="array"></param>
		/// <param name="maxAniso"></param>
		/// <param name="mipmapFilterMode"></param>
		/// <param name="mipmapLevelBias"></param>
		/// <param name="minMipmapLevelClamp"></param>
		/// <param name="maxMipmapLevelClamp"></param>
		public static void BindTexture(CudaKernel kernel, string texName, CUAddressMode addressModeForAllDimensions, CUFilterMode filterMode, CUTexRefSetFlags flags, CudaMipmappedArray array,
			uint maxAniso, CUFilterMode mipmapFilterMode, float mipmapLevelBias, float minMipmapLevelClamp, float maxMipmapLevelClamp)
		{
			BindTexture(kernel, texName, addressModeForAllDimensions, addressModeForAllDimensions, addressModeForAllDimensions, filterMode, flags, array,
				maxAniso, mipmapFilterMode, mipmapLevelBias, minMipmapLevelClamp, maxMipmapLevelClamp);
		}

		/// <summary>
		/// Bind a CudaMipmappedArray to a texture reference.
		/// </summary>
		/// <param name="kernel"></param>
		/// <param name="texName"></param>
		/// <param name="addressMode0"></param>
		/// <param name="addressMode1"></param>
		/// <param name="addressMode2"></param>
		/// <param name="filterMode"></param>
		/// <param name="flags"></param>
		/// <param name="array"></param>
		/// <param name="maxAniso"></param>
		/// <param name="mipmapFilterMode"></param>
		/// <param name="mipmapLevelBias"></param>
		/// <param name="minMipmapLevelClamp"></param>
		/// <param name="maxMipmapLevelClamp"></param>
		public static void BindTexture(CudaKernel kernel, string texName, CUAddressMode addressMode0, CUAddressMode addressMode1, CUAddressMode addressMode2,
			CUFilterMode filterMode, CUTexRefSetFlags flags, CudaMipmappedArray array,
			uint maxAniso, CUFilterMode mipmapFilterMode, float mipmapLevelBias, float minMipmapLevelClamp, float maxMipmapLevelClamp)
		{
			CUtexref texref = new CUtexref();
			CUResult res = DriverAPINativeMethods.ModuleManagement.cuModuleGetTexRef(ref texref, kernel.CUModule, texName);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}, Texture name: {3}", DateTime.Now, "cuModuleGetTexRef", res, texName));
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 0, addressMode0);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 1, addressMode1);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetAddressMode(texref, 2, addressMode2);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetAddressMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFilterMode(texref, filterMode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFilterMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFlags(texref, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFlags", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetFormat(texref, array.Array3DDescriptor.Format, (int)array.Array3DDescriptor.NumChannels);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetFormat", res));
			if (res != CUResult.Success) throw new CudaException(res);

			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetMipmappedArray(texref, array.CUMipmappedArray, CUTexRefSetArrayFlags.OverrideFormat);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetMipmappedArray", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetMaxAnisotropy(texref, maxAniso);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetMaxAnisotropy", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetMipmapFilterMode(texref, mipmapFilterMode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetMipmapFilterMode", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetMipmapLevelBias(texref, mipmapLevelBias);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetMipmapLevelBias", res));
			if (res != CUResult.Success) throw new CudaException(res);
			res = DriverAPINativeMethods.TextureReferenceManagement.cuTexRefSetMipmapLevelClamp(texref, minMipmapLevelClamp, maxMipmapLevelClamp);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexRefSetMipmapLevelClamp", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}
		#endregion
	}
}
