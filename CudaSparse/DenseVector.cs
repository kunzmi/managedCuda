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
using System.Text;
using System.Diagnostics;
using ManagedCuda.BasicTypes;

namespace ManagedCuda.CudaSparse
{
	/// <summary>
	/// 
	/// </summary>
	public class DenseVector<dataT> : IDisposable where dataT : struct
	{
		private cusparseStatus res;
		private bool disposed;
		private cusparseDnVecDescr descr;
		private cudaDataType typeData;
		private long size;



		#region Contructors
		/// <summary>
		/// </summary>
		public DenseVector(long aSize, CudaDeviceVariable<dataT> values)
		{
			size = aSize;
			descr = new cusparseDnVecDescr();
			typeData = CudaDataTypeTranslator.GetType(typeof(dataT));
			res = CudaSparseNativeMethods.cusparseCreateDnVec(ref descr, size, values.DevicePointer, typeData);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateDnVec", res));
			if (res != cusparseStatus.Success)
				throw new CudaSparseException(res);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~DenseVector()
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
				//Ignore if failing
				res = CudaSparseNativeMethods.cusparseDestroyDnVec(descr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDestroyDnVec", res));
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		/// <summary>
		/// Returns the inner handle.
		/// </summary>
		public cusparseDnVecDescr Descr
		{
			get { return descr; }
		}
		/// <summary>
		/// 
		/// </summary>
		public cudaDataType TypeData
		{
			get { return typeData; }
		}
		/// <summary>
		/// 
		/// </summary>
		public long Size
		{
			get { return size; }
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public CudaDeviceVariable<dataT> GetValues()
		{
			CUdeviceptr devPtr = new CUdeviceptr();
			res = CudaSparseNativeMethods.cusparseDnVecGetValues(descr, ref devPtr);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDnVecGetValues", res));
			if (res != cusparseStatus.Success)
				throw new CudaSparseException(res);
			return new CudaDeviceVariable<dataT>(devPtr);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="data"></param>
		public void SetValues(CudaDeviceVariable<dataT> data)
		{
			res = CudaSparseNativeMethods.cusparseDnVecSetValues(descr, data.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDnVecSetValues", res));
			if (res != cusparseStatus.Success)
				throw new CudaSparseException(res);
		}

		/// <summary>
		/// 
		/// </summary>
		public CudaDeviceVariable<dataT> Get()
		{
			CUdeviceptr ptrValues = new CUdeviceptr();

			res = CudaSparseNativeMethods.cusparseDnVecGet(descr, ref size, ref ptrValues, ref typeData);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDnVecGet", res));
			if (res != cusparseStatus.Success)
				throw new CudaSparseException(res);

			return new CudaDeviceVariable<dataT>(ptrValues);
		}
	}
}
