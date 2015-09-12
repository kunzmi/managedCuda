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
	/// Cuda Surface Object
	/// </summary>
	public class CudaSurfObject : IDisposable
	{
		CUsurfObject _surfObject;
		CudaResourceDesc _resDesc;
		CUResult res;
		bool disposed;

		#region Constructors
		/// <summary>
		/// Creates a surface object. <c>ResDesc</c> describes
		/// the data to perform surface load/stores on. <c>ResDesc.resType</c> must be 
		/// <see cref="CUResourceType.Array"/> and  <c>ResDesc.hArray</c>
		/// must be set to a valid CUDA array handle. <c>ResDesc.flags</c> must be set to zero.
		/// </summary>
		/// <param name="resDesc">CudaResourceDesc</param>
		public CudaSurfObject(CudaResourceDesc resDesc)
		{
			_resDesc = resDesc;

			_surfObject = new CUsurfObject();
			res = DriverAPINativeMethods.SurfaceObjects.cuSurfObjectCreate(ref _surfObject, ref _resDesc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuSurfObjectCreate", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Creates a surface object. <c>ResDesc</c> describes
		/// the data to perform surface load/stores on. <c>ResDesc.resType</c> must be 
		/// <see cref="CUResourceType.Array"/> and  <c>ResDesc.hArray</c>
		/// must be set to a valid CUDA array handle.
		/// </summary>
		/// <param name="array">CudaArray1D</param>
		public CudaSurfObject(CudaArray1D array)
		{
			_resDesc = new CudaResourceDesc(array);

			_surfObject = new CUsurfObject();
			res = DriverAPINativeMethods.SurfaceObjects.cuSurfObjectCreate(ref _surfObject, ref _resDesc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuSurfObjectCreate", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Creates a surface object. <c>ResDesc</c> describes
		/// the data to perform surface load/stores on. <c>ResDesc.resType</c> must be 
		/// <see cref="CUResourceType.Array"/> and  <c>ResDesc.hArray</c>
		/// must be set to a valid CUDA array handle.
		/// </summary>
		/// <param name="array">CudaArray2D</param>
		public CudaSurfObject(CudaArray2D array)
		{
			_resDesc = new CudaResourceDesc(array);

			_surfObject = new CUsurfObject();
			res = DriverAPINativeMethods.SurfaceObjects.cuSurfObjectCreate(ref _surfObject, ref _resDesc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuSurfObjectCreate", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// Creates a surface object. <c>ResDesc</c> describes
		/// the data to perform surface load/stores on. <c>ResDesc.resType</c> must be 
		/// <see cref="CUResourceType.Array"/> and  <c>ResDesc.hArray</c>
		/// must be set to a valid CUDA array handle.
		/// </summary>
		/// <param name="array">CudaArray3D</param>
		public CudaSurfObject(CudaArray3D array)
		{
			_resDesc = new CudaResourceDesc(array);

			_surfObject = new CUsurfObject();
			res = DriverAPINativeMethods.SurfaceObjects.cuSurfObjectCreate(ref _surfObject, ref _resDesc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuSurfObjectCreate", res));
			if (res != CUResult.Success) throw new CudaException(res);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaSurfObject()
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
				res = DriverAPINativeMethods.SurfaceObjects.cuSurfObjectDestroy(_surfObject);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuSurfObjectDestroy", res));
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		/// <summary>
		/// Returns the wrapped CUsurfObject
		/// </summary>
		public CUsurfObject SurfObject
		{
			get { return _surfObject; }
		}

		/// <summary>
		/// Returns the CudaResourceDesc used to create the CudaSurfObject
		/// </summary>
		public CudaResourceDesc ResDesc
		{
			get { return _resDesc; }
		}
	}
}
