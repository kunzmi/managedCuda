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
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;

namespace ManagedCuda
{
	/// <summary>
	/// Cuda Texure Object
	/// </summary>
	public class CudaTexObject : IDisposable
	{
		CUtexObject _texObject;
		CudaResourceDesc _resDesc;
		CudaTextureDescriptor _texDesc;
		CudaResourceViewDesc _resViewDesc;
		CUResult res;
		bool disposed;

		#region Constructors
		/// <summary>
		/// Creates a texture object and returns it in pTexObject. pResDesc describes the data to texture from. pTexDesc
		/// describes how the data should be sampled.
		/// </summary>
		/// <param name="resDesc">CudaResourceDesc</param>
		/// <param name="texDesc">CudaTextureDescriptor</param>
		public CudaTexObject(CudaResourceDesc resDesc, CudaTextureDescriptor texDesc)
		{
			_resDesc = resDesc;
			_texDesc = texDesc;

			_texObject = new CUtexObject();
			res = DriverAPINativeMethods.TextureObjects.cuTexObjectCreate(ref _texObject, ref _resDesc, ref _texDesc, IntPtr.Zero);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexObjectCreate", res));
			if (res != CUResult.Success) throw new CudaException(res);


		}
		/// <summary>
		/// Creates a texture object. ResDesc describes the data to texture from. TexDesc
		/// describes how the data should be sampled. resViewDesc is an optional argument that specifies an alternate format
		/// for the data described by pResDesc, and also describes the subresource region to restrict access to when texturing.
		/// pResViewDesc can only be specified if the type of resource is a CUDA array or a CUDA mipmapped array.
		/// </summary>
		/// <param name="resDesc">Describes the data to texture from.</param>
		/// <param name="texDesc">Describes how the data should be sampled.</param>
		/// <param name="resViewDesc">CudaResourceViewDesc. Only valid if type of resource is a CUDA array or a CUDA mipmapped array</param>
		public CudaTexObject(CudaResourceDesc resDesc, CudaTextureDescriptor texDesc, CudaResourceViewDesc resViewDesc)
		{
			_resDesc = resDesc;
			_texDesc = texDesc;
			_resViewDesc = resViewDesc;

			_texObject = new CUtexObject();
			res = DriverAPINativeMethods.TextureObjects.cuTexObjectCreate(ref _texObject, ref _resDesc, ref _texDesc, ref _resViewDesc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexObjectCreate", res));
			if (res != CUResult.Success) throw new CudaException(res);


		}

		///// <summary>
		///// Creates a new 1D surface from array memory. Allocates new array.
		///// </summary>
		///// <param name="kernel"></param>
		///// <param name="surfName"></param>
		///// <param name="flags"></param>
		///// <param name="format"></param>
		///// <param name="size">In elements</param>
		///// <param name="numChannels"></param>
		//public CudaTexObject(CudaArray1D cudaArray, CudaTextureDescriptor texDesc, CudaResourceViewDesc resViewDesc)
		//{
		//    _resDesc = new CudaResourceDesc();
		//    _texDesc = texDesc;
		//    _resViewDesc = resViewDesc;

		//    _resDesc.hArray = cudaArray.CUArray;
		//    _resDesc.resType = CUResourceType.Array;

		//    _texObject = new CUtexObject();
		//    res = DriverAPINativeMethods.TextureObjects.cuTexObjectCreate(ref _texObject, ref _resDesc, ref _texDesc, ref _resViewDesc);
		//    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexObjectCreate", res));
		//    if (res != CUResult.Success) throw new CudaException(res);
		//}

		///// <summary>
		///// Creates a new 1D surface from array memory. Allocates new array.
		///// </summary>
		///// <param name="kernel"></param>
		///// <param name="surfName"></param>
		///// <param name="flags"></param>
		///// <param name="format"></param>
		///// <param name="size">In elements</param>
		///// <param name="numChannels"></param>
		//public CudaTexObject(CudaArray2D cudaArray, CudaTextureDescriptor texDesc, CudaResourceViewDesc resViewDesc)
		//{
		//    _resDesc = new CudaResourceDesc();
		//    _texDesc = texDesc;
		//    _resViewDesc = resViewDesc;

		//    _resDesc.hArray = cudaArray.CUArray;
		//    _resDesc.resType = CUResourceType.Array;

		//    _texObject = new CUtexObject();
		//    res = DriverAPINativeMethods.TextureObjects.cuTexObjectCreate(ref _texObject, ref _resDesc, ref _texDesc, ref _resViewDesc);
		//    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexObjectCreate", res));
		//    if (res != CUResult.Success) throw new CudaException(res);
		//}

		///// <summary>
		///// Creates a new 1D surface from array memory. Allocates new array.
		///// </summary>
		///// <param name="kernel"></param>
		///// <param name="surfName"></param>
		///// <param name="flags"></param>
		///// <param name="format"></param>
		///// <param name="size">In elements</param>
		///// <param name="numChannels"></param>
		//public CudaTexObject(CudaArray3D cudaArray, CudaTextureDescriptor texDesc, CudaResourceViewDesc resViewDesc)
		//{
		//    _resDesc = new CudaResourceDesc();
		//    _texDesc = texDesc;
		//    _resViewDesc = resViewDesc;

		//    _resDesc.hArray = cudaArray.CUArray;
		//    _resDesc.resType = CUResourceType.Array;

		//    _texObject = new CUtexObject();
		//    res = DriverAPINativeMethods.TextureObjects.cuTexObjectCreate(ref _texObject, ref _resDesc, ref _texDesc, ref _resViewDesc);
		//    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexObjectCreate", res));
		//    if (res != CUResult.Success) throw new CudaException(res);
		//}

		///// <summary>
		///// Creates a new 1D surface from array memory. Allocates new array.
		///// </summary>
		///// <param name="kernel"></param>
		///// <param name="surfName"></param>
		///// <param name="flags"></param>
		///// <param name="format"></param>
		///// <param name="size">In elements</param>
		///// <param name="numChannels"></param>
		//public CudaTexObject(CudaDeviceVariable<ICudaVectorTypeForArray> cudaDevVar, CudaTextureDescriptor texDesc, CudaResourceViewDesc resViewDesc)
		//{
		//    _resDesc = new CudaResourceDesc();
		//    _texDesc = texDesc;
		//    _resViewDesc = resViewDesc;

		//    _resDesc.resType = CUResourceType.Linear;
		//    _resDesc.linear = new CudaResourceDescLinear();
		//    _resDesc.linear.devPtr = cudaDevVar.DevicePointer;
		//    _resDesc.linear.sizeInBytes = cudaDevVar.SizeInBytes;
		//    Type type = cudaDevVar.GetType().GetGenericArguments()[0];



		//    //_resDesc.linear.numChannels = cudaDevVar.GetHashCode

		//    _texObject = new CUtexObject();
		//    res = DriverAPINativeMethods.TextureObjects.cuTexObjectCreate(ref _texObject, ref _resDesc, ref _texDesc, ref _resViewDesc);
		//    Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexObjectCreate", res));
		//    if (res != CUResult.Success) throw new CudaException(res);
		//}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaTexObject()
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
				//_array.Dispose();
				disposed = true;
				res = DriverAPINativeMethods.TextureObjects.cuTexObjectDestroy(_texObject);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuTexObjectDestroy", res));
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		/// <summary>
		/// Returns the wrapped CUtexObject
		/// </summary>
		public CUtexObject TexObject
		{
			get { return _texObject; }
		}

		/// <summary>
		/// Returns the CudaResourceDesc used to create the CudaTexObject
		/// </summary>
		public CudaResourceDesc ResourceDesc
		{
			get { return _resDesc; }
		}

		/// <summary>
		/// Returns the CudaTextureDescriptor used to create the CudaTexObject
		/// </summary>
		public CudaTextureDescriptor TextureDescriptor
		{
			get { return _texDesc; }
		}

		/// <summary>
		/// Returns the CudaResourceViewDesc used to create the CudaTexObject
		/// </summary>
		public CudaResourceViewDesc ResourceViewDesc
		{
			get { return _resViewDesc; }
		}
	}
}
