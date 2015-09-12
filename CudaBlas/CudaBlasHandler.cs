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
using System.Diagnostics;
using System.Runtime.InteropServices;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;

namespace ManagedCuda.CudaBlas
{
	/// <summary>
	/// Wrapper for CUBLAS
	/// </summary>
	public class CudaBlas
	{
		bool disposed;
		CudaBlasHandle _blasHandle;
		CublasStatus _status;
	
		#region Constructors
		/// <summary>
		/// Creates a new cudaBlas handler
		/// </summary>
		public CudaBlas()
		{
			_blasHandle = new CudaBlasHandle();
			_status = CudaBlasNativeMethods.cublasCreate_v2(ref _blasHandle);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCreate_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// Creates a new cudaBlas handler
		/// </summary>
		public CudaBlas(CUstream stream)
			: this()
		{
			Stream = stream;
		}
		/// <summary>
		/// Creates a new cudaBlas handler
		/// </summary>
		public CudaBlas(PointerMode pointermode)
			: this()
		{
			PointerMode = pointermode;
		}
		/// <summary>
		/// Creates a new cudaBlas handler
		/// </summary>
		public CudaBlas(AtomicsMode atomicsmode)
			: this()
		{
			AtomicsMode = atomicsmode;
		}
		/// <summary>
		/// Creates a new cudaBlas handler
		/// </summary>
		public CudaBlas(CUstream stream, PointerMode pointermode)
			: this()
		{
			Stream = stream;
			PointerMode = pointermode;
		}
		/// <summary>
		/// Creates a new cudaBlas handler
		/// </summary>
		public CudaBlas(CUstream stream, AtomicsMode atomicsmode)
			: this()
		{
			Stream = stream;
			AtomicsMode = atomicsmode;
		}
		/// <summary>
		/// Creates a new cudaBlas handler
		/// </summary>
		public CudaBlas(PointerMode pointermode, AtomicsMode atomicsmode)
			: this()
		{
			PointerMode = pointermode;
			AtomicsMode = atomicsmode;
		}
		/// <summary>
		/// Creates a new cudaBlas handler
		/// </summary>
		public CudaBlas(CUstream stream, PointerMode pointermode, AtomicsMode atomicsmode)
			: this()
		{
			Stream = stream;
			PointerMode = pointermode;
			AtomicsMode = atomicsmode;
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaBlas()
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
				_status = CudaBlasNativeMethods.cublasDestroy_v2(_blasHandle);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDestroy_v2", _status));
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("CudaBlas not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties
		/// <summary>
		/// Returns the wrapped cublas handle
		/// </summary>
		public CudaBlasHandle CublasHandle
		{
			get { return _blasHandle; }
		}

		/// <summary>
		/// 
		/// </summary>
		public CUstream Stream
		{
			get
			{
				CUstream stream = new CUstream();
				_status = CudaBlasNativeMethods.cublasGetStream_v2(_blasHandle, ref stream);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasGetStream_v2", _status));
				if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
				return stream;
			}
			set 
			{
				_status = CudaBlasNativeMethods.cublasSetStream_v2(_blasHandle, value);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSetStream_v2", _status));
				if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			}
		}

		/// <summary>
		/// 
		/// </summary>
		public PointerMode PointerMode
		{
			get
			{
				PointerMode pm = new PointerMode();
				_status = CudaBlasNativeMethods.cublasGetPointerMode_v2(_blasHandle, ref pm);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasGetPointerMode_v2", _status));
				if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
				return pm;
			}
			set
			{
				_status = CudaBlasNativeMethods.cublasSetPointerMode_v2(_blasHandle, value);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSetPointerMode_v2", _status));
				if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			}
		}

		/// <summary>
		/// 
		/// </summary>
		public AtomicsMode AtomicsMode
		{
			get
			{
				AtomicsMode am = new AtomicsMode();
				_status = CudaBlasNativeMethods.cublasGetAtomicsMode(_blasHandle, ref am);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasGetAtomicsMode", _status));
				if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
				return am;
			}
			set
			{
				_status = CudaBlasNativeMethods.cublasSetAtomicsMode(_blasHandle, value);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSetAtomicsMode", _status));
				if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			}
		}
		#endregion

		#region Methods

		/// <summary>
		/// 
		/// </summary>
		public Version GetVersion()
		{
			int version = 0;
			_status = CudaBlasNativeMethods.cublasGetVersion_v2(_blasHandle, ref version);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasGetVersion_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return new Version(version / 1000, version % 1000);
		}

		#region BLAS1
		#region Copy
		/// <summary>
		/// This function copies the vector x into the vector y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Copy(CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasScopy_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasScopy_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function copies the vector x into the vector y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Copy(CudaDeviceVariable<float1> x, int incx, CudaDeviceVariable<float1> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasScopy_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasScopy_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function copies the vector x into the vector y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Copy(CudaDeviceVariable<cuFloatReal> x, int incx, CudaDeviceVariable<cuFloatReal> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasScopy_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasScopy_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function copies the vector x into the vector y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Copy(CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasDcopy_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDcopy_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function copies the vector x into the vector y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Copy(CudaDeviceVariable<double1> x, int incx, CudaDeviceVariable<double1> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasDcopy_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDcopy_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function copies the vector x into the vector y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Copy(CudaDeviceVariable<cuDoubleReal> x, int incx, CudaDeviceVariable<cuDoubleReal> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasDcopy_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDcopy_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function copies the vector x into the vector y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Copy(CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasCcopy_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCcopy_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function copies the vector x into the vector y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Copy(CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasZcopy_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZcopy_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region Swap
		/// <summary>
		/// This function interchanges the elements of vector x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Swap(CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasSswap_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSswap_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function interchanges the elements of vector x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Swap(CudaDeviceVariable<float1> x, int incx, CudaDeviceVariable<float1> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasSswap_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSswap_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function interchanges the elements of vector x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Swap(CudaDeviceVariable<cuFloatReal> x, int incx, CudaDeviceVariable<cuFloatReal> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasSswap_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSswap_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function interchanges the elements of vector x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Swap(CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasDswap_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDswap_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function interchanges the elements of vector x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Swap(CudaDeviceVariable<double1> x, int incx, CudaDeviceVariable<double1> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasDswap_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDswap_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function interchanges the elements of vector x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Swap(CudaDeviceVariable<cuDoubleReal> x, int incx, CudaDeviceVariable<cuDoubleReal> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasDswap_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDswap_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function interchanges the elements of vector x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Swap(CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasCswap_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCswap_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function interchanges the elements of vector x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Swap(CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasZswap_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZswap_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region Norm2
		/// <summary>
		/// This function computes the Euclidean norm of the vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Norm2(CudaDeviceVariable<float> x, int incx, ref float result)
		{
			_status = CudaBlasNativeMethods.cublasSnrm2_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSnrm2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function computes the Euclidean norm of the vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public float Norm2(CudaDeviceVariable<float> x, int incx)
		{
			float result = 0;
			_status = CudaBlasNativeMethods.cublasSnrm2_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSnrm2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return result;
		}

		/// <summary>
		/// This function computes the Euclidean norm of the vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Norm2(CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> result)
		{
			_status = CudaBlasNativeMethods.cublasSnrm2_v2(_blasHandle, x.Size, x.DevicePointer, incx, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSnrm2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function computes the Euclidean norm of the vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Norm2(CudaDeviceVariable<double> x, int incx, ref double result)
		{
			_status = CudaBlasNativeMethods.cublasDnrm2_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDnrm2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function computes the Euclidean norm of the vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public double Norm2(CudaDeviceVariable<double> x, int incx)
		{
			double result = 0;
			_status = CudaBlasNativeMethods.cublasDnrm2_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDnrm2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return result;
		}

		/// <summary>
		/// This function computes the Euclidean norm of the vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Norm2(CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> result)
		{
			_status = CudaBlasNativeMethods.cublasDnrm2_v2(_blasHandle, x.Size, x.DevicePointer, incx, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDnrm2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function computes the Euclidean norm of the vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Norm2(CudaDeviceVariable<cuFloatComplex> x, int incx, ref float result)
		{
			_status = CudaBlasNativeMethods.cublasScnrm2_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasScnrm2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function computes the Euclidean norm of the vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public float Norm2(CudaDeviceVariable<cuFloatComplex> x, int incx)
		{
			float result = 0;
			_status = CudaBlasNativeMethods.cublasScnrm2_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasScnrm2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return result;
		}

		/// <summary>
		/// This function computes the Euclidean norm of the vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Norm2(CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<float> result)
		{
			_status = CudaBlasNativeMethods.cublasScnrm2_v2(_blasHandle, x.Size, x.DevicePointer, incx, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasScnrm2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function computes the Euclidean norm of the vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Norm2(CudaDeviceVariable<cuDoubleComplex> x, int incx, ref double result)
		{
			_status = CudaBlasNativeMethods.cublasDznrm2_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDznrm2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function computes the Euclidean norm of the vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public double Norm2(CudaDeviceVariable<cuDoubleComplex> x, int incx)
		{
			double result = 0;
			_status = CudaBlasNativeMethods.cublasDznrm2_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDznrm2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return result;
		}

		/// <summary>
		/// This function computes the Euclidean norm of the vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Norm2(CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<double> result)
		{
			_status = CudaBlasNativeMethods.cublasDznrm2_v2(_blasHandle, x.Size, x.DevicePointer, incx, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDznrm2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region Dot
		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="result"></param>
		public void Dot(CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> y, int incy, ref float result)
		{
			_status = CudaBlasNativeMethods.cublasSdot_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSdot_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public float Dot(CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> y, int incy)
		{
			float result = 0;
			_status = CudaBlasNativeMethods.cublasSdot_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSdot_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return result;
		}

		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="result"></param>
		public void Dot(CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> y, int incy, CudaDeviceVariable<float> result)
		{
			_status = CudaBlasNativeMethods.cublasSdot_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSdot_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="result"></param>
		public void Dot(CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> y, int incy, ref double result)
		{
			_status = CudaBlasNativeMethods.cublasDdot_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDdot_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public double Dot(CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> y, int incy)
		{
			double result = 0;
			_status = CudaBlasNativeMethods.cublasDdot_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDdot_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return result;
		}

		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="result"></param>
		public void Dot(CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> y, int incy, CudaDeviceVariable<double> result)
		{
			_status = CudaBlasNativeMethods.cublasDdot_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDdot_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
	
		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="result"></param>
		public void Dot(CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy, ref cuFloatComplex result)
		{
			_status = CudaBlasNativeMethods.cublasCdotu_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCdotu_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public cuFloatComplex Dot(CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy)
		{
			cuFloatComplex result = new cuFloatComplex();
			_status = CudaBlasNativeMethods.cublasCdotu_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCdotu_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return result;
		}

		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="result"></param>
		public void Dot(CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy, CudaDeviceVariable<cuFloatComplex> result)
		{
			_status = CudaBlasNativeMethods.cublasCdotu_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCdotu_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="result"></param>
		public void Dot(CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy, ref cuDoubleComplex result)
		{
			_status = CudaBlasNativeMethods.cublasZdotu_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZdotu_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public cuDoubleComplex Dot(CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy)
		{
			cuDoubleComplex result = new cuDoubleComplex();
			_status = CudaBlasNativeMethods.cublasZdotu_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZdotu_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return result;
		}

		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="result"></param>
		public void Dot(CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy, CudaDeviceVariable<cuDoubleComplex> result)
		{
			_status = CudaBlasNativeMethods.cublasZdotu_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZdotu_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// <para/>Notice that the conjugate of the element of vector x should be used.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="result"></param>
		public void DotConj(CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy, ref cuFloatComplex result)
		{
			_status = CudaBlasNativeMethods.cublasCdotc_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCdotc_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// <para/>Notice that the conjugate of the element of vector x should be used.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public cuFloatComplex DotConj(CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy)
		{
			cuFloatComplex result = new cuFloatComplex(); ;
			_status = CudaBlasNativeMethods.cublasCdotc_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCdotc_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return result;
		}

		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// <para/>Notice that the conjugate of the element of vector x should be used.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="result"></param>
		public void DotConj(CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy, CudaDeviceVariable<cuFloatComplex> result)
		{
			_status = CudaBlasNativeMethods.cublasCdotc_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCdotc_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// <para/>Notice that the conjugate of the element of vector x should be used.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="result"></param>
		public void DotConj(CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy, ref cuDoubleComplex result)
		{
			_status = CudaBlasNativeMethods.cublasZdotc_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZdotc_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// <para/>Notice that the conjugate of the element of vector x should be used.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public cuDoubleComplex DotConj(CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy)
		{
			cuDoubleComplex result = new cuDoubleComplex();
			_status = CudaBlasNativeMethods.cublasZdotc_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZdotc_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return result;
		}

		/// <summary>
		/// This function computes the dot product of vectors x and y.
		/// <para/>Notice that the conjugate of the element of vector x should be used.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="result"></param>
		public void DotConj(CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy, CudaDeviceVariable<cuDoubleComplex> result)
		{
			_status = CudaBlasNativeMethods.cublasZdotc_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZdotc_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region Scal
		/// <summary>
		/// This function scales the vector x by the scalar and overwrites it with the result.
		/// </summary>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public void Scale(float alpha, CudaDeviceVariable<float> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasSscal_v2(_blasHandle, x.Size, ref alpha, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSscal_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function scales the vector x by the scalar and overwrites it with the result.
		/// </summary>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public void Scale(CudaDeviceVariable<float> alpha, CudaDeviceVariable<float> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasSscal_v2(_blasHandle, x.Size, alpha.DevicePointer, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSscal_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function scales the vector x by the scalar and overwrites it with the result.
		/// </summary>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public void Scale(double alpha, CudaDeviceVariable<double> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasDscal_v2(_blasHandle, x.Size, ref alpha, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDscal_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function scales the vector x by the scalar and overwrites it with the result.
		/// </summary>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public void Scale(CudaDeviceVariable<double> alpha, CudaDeviceVariable<double> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasDscal_v2(_blasHandle, x.Size, alpha.DevicePointer, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDscal_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function scales the vector x by the scalar and overwrites it with the result.
		/// </summary>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public void Scale(cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasCscal_v2(_blasHandle, x.Size, ref alpha, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCscal_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function scales the vector x by the scalar and overwrites it with the result.
		/// </summary>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public void Scale(CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasCscal_v2(_blasHandle, x.Size, alpha.DevicePointer, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCscal_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function scales the vector x by the scalar and overwrites it with the result.
		/// </summary>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public void Scale(float alpha, CudaDeviceVariable<cuFloatComplex> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasCsscal_v2(_blasHandle, x.Size, ref alpha, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCsscal_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function scales the vector x by the scalar and overwrites it with the result.
		/// </summary>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public void Scale(CudaDeviceVariable<float> alpha, CudaDeviceVariable<cuFloatComplex> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasCsscal_v2(_blasHandle, x.Size, alpha.DevicePointer, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCsscal_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function scales the vector x by the scalar and overwrites it with the result.
		/// </summary>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public void Scale(cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasZscal_v2(_blasHandle, x.Size, ref alpha, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZscal_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function scales the vector x by the scalar and overwrites it with the result.
		/// </summary>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public void Scale(CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasZscal_v2(_blasHandle, x.Size, alpha.DevicePointer, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZscal_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function scales the vector x by the scalar and overwrites it with the result.
		/// </summary>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public void Scale(double alpha, CudaDeviceVariable<cuDoubleComplex> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasZdscal_v2(_blasHandle, x.Size, ref alpha, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZdscal_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function scales the vector x by the scalar and overwrites it with the result.
		/// </summary>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public void Scale(CudaDeviceVariable<double> alpha, CudaDeviceVariable<cuDoubleComplex> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasZdscal_v2(_blasHandle, x.Size, alpha.DevicePointer, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZdscal_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region Axpy
		/// <summary>
		/// This function multiplies the vector x by the scalar and adds it to the vector y overwriting
		/// the latest vector with the result.
		/// </summary>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Axpy(float alpha, CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasSaxpy_v2(_blasHandle, x.Size, ref alpha, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSaxpy_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function multiplies the vector x by the scalar and adds it to the vector y overwriting
		/// the latest vector with the result.
		/// </summary>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Axpy(CudaDeviceVariable<float> alpha, CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasSaxpy_v2(_blasHandle, x.Size, alpha.DevicePointer, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSaxpy_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function multiplies the vector x by the scalar and adds it to the vector y overwriting
		/// the latest vector with the result.
		/// </summary>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Axpy(double alpha, CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasDaxpy_v2(_blasHandle, x.Size, ref alpha, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDaxpy_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function multiplies the vector x by the scalar and adds it to the vector y overwriting
		/// the latest vector with the result.
		/// </summary>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Axpy(CudaDeviceVariable<double> alpha, CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasDaxpy_v2(_blasHandle, x.Size, alpha.DevicePointer, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDaxpy_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function multiplies the vector x by the scalar and adds it to the vector y overwriting
		/// the latest vector with the result.
		/// </summary>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Axpy(cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasCaxpy_v2(_blasHandle, x.Size, ref alpha, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCaxpy_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function multiplies the vector x by the scalar and adds it to the vector y overwriting
		/// the latest vector with the result.
		/// </summary>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Axpy(CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasCaxpy_v2(_blasHandle, x.Size, alpha.DevicePointer, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCaxpy_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function multiplies the vector x by the scalar and adds it to the vector y overwriting
		/// the latest vector with the result.
		/// </summary>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Axpy(cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasZaxpy_v2(_blasHandle, x.Size, ref alpha, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZaxpy_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function multiplies the vector x by the scalar and adds it to the vector y overwriting
		/// the latest vector with the result.
		/// </summary>
		/// <param name="alpha"></param>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		public void Axpy(CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasZaxpy_v2(_blasHandle, x.Size, alpha.DevicePointer, x.DevicePointer, incx, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZaxpy_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region Imin
		/// <summary>
		/// This function finds the (smallest) index of the element of the minimum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Min(CudaDeviceVariable<float> x, int incx, ref int result)
		{
			_status = CudaBlasNativeMethods.cublasIsamin_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIsamin_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function finds the (smallest) index of the element of the minimum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public int Min(CudaDeviceVariable<float> x, int incx)
		{
			int result = 0;
			_status = CudaBlasNativeMethods.cublasIsamin_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIsamin_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return result;
		}
		/// <summary>
		/// This function finds the (smallest) index of the element of the minimum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Min(CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<int> result)
		{
			_status = CudaBlasNativeMethods.cublasIsamin_v2(_blasHandle, x.Size, x.DevicePointer, incx, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIsamin_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function finds the (smallest) index of the element of the minimum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Min(CudaDeviceVariable<double> x, int incx, ref int result)
		{
			_status = CudaBlasNativeMethods.cublasIdamin_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIdamin_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function finds the (smallest) index of the element of the minimum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public int Min(CudaDeviceVariable<double> x, int incx)
		{
			int result = 0;
			_status = CudaBlasNativeMethods.cublasIdamin_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIdamin_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return result;
		}
		/// <summary>
		/// This function finds the (smallest) index of the element of the minimum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Min(CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<int> result)
		{
			_status = CudaBlasNativeMethods.cublasIdamin_v2(_blasHandle, x.Size, x.DevicePointer, incx, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIdamin_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function finds the (smallest) index of the element of the minimum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Min(CudaDeviceVariable<cuFloatComplex> x, int incx, ref int result)
		{
			_status = CudaBlasNativeMethods.cublasIcamin_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIcamin_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function finds the (smallest) index of the element of the minimum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public int Min(CudaDeviceVariable<cuFloatComplex> x, int incx)
		{
			int result = 0;
			_status = CudaBlasNativeMethods.cublasIcamin_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIcamin_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return result;
		}
		/// <summary>
		/// This function finds the (smallest) index of the element of the minimum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Min(CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<int> result)
		{
			_status = CudaBlasNativeMethods.cublasIcamin_v2(_blasHandle, x.Size, x.DevicePointer, incx, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIcamin_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function finds the (smallest) index of the element of the minimum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Min(CudaDeviceVariable<cuDoubleComplex> x, int incx, ref int result)
		{
			_status = CudaBlasNativeMethods.cublasIzamin_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIzamin_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function finds the (smallest) index of the element of the minimum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public int Min(CudaDeviceVariable<cuDoubleComplex> x, int incx)
		{
			int result = 0;
			_status = CudaBlasNativeMethods.cublasIzamin_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIzamin_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return result;
		}
		/// <summary>
		/// This function finds the (smallest) index of the element of the minimum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Min(CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<int> result)
		{
			_status = CudaBlasNativeMethods.cublasIzamin_v2(_blasHandle, x.Size, x.DevicePointer, incx, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIzamin_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region Imax
		/// <summary>
		/// This function finds the (smallest) index of the element of the maximum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Max(CudaDeviceVariable<float> x, int incx, ref int result)
		{
			_status = CudaBlasNativeMethods.cublasIsamax_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIsamax_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function finds the (smallest) index of the element of the maximum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public int Max(CudaDeviceVariable<float> x, int incx)
		{
			int result = 0;
			_status = CudaBlasNativeMethods.cublasIsamax_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIsamax_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return result;
		}
		/// <summary>
		/// This function finds the (smallest) index of the element of the maximum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Max(CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<int> result)
		{
			_status = CudaBlasNativeMethods.cublasIsamax_v2(_blasHandle, x.Size, x.DevicePointer, incx, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIsamax_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function finds the (smallest) index of the element of the maximum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Max(CudaDeviceVariable<double> x, int incx, ref int result)
		{
			_status = CudaBlasNativeMethods.cublasIdamax_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIdamax_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function finds the (smallest) index of the element of the maximum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public int Max(CudaDeviceVariable<double> x, int incx)
		{
			int result = 0;
			_status = CudaBlasNativeMethods.cublasIdamax_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIdamax_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return result;
		}
		/// <summary>
		/// This function finds the (smallest) index of the element of the maximum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Max(CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<int> result)
		{
			_status = CudaBlasNativeMethods.cublasIdamax_v2(_blasHandle, x.Size, x.DevicePointer, incx, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIdamax_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function finds the (smallest) index of the element of the maximum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Max(CudaDeviceVariable<cuFloatComplex> x, int incx, ref int result)
		{
			_status = CudaBlasNativeMethods.cublasIcamax_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIcamax_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function finds the (smallest) index of the element of the maximum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public int Max(CudaDeviceVariable<cuFloatComplex> x, int incx)
		{
			int result = 0;
			_status = CudaBlasNativeMethods.cublasIcamax_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIcamax_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return result;
		}
		/// <summary>
		/// This function finds the (smallest) index of the element of the maximum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Max(CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<int> result)
		{
			_status = CudaBlasNativeMethods.cublasIcamax_v2(_blasHandle, x.Size, x.DevicePointer, incx, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIcamax_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function finds the (smallest) index of the element of the maximum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Max(CudaDeviceVariable<cuDoubleComplex> x, int incx, ref int result)
		{
			_status = CudaBlasNativeMethods.cublasIzamax_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIzamax_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function finds the (smallest) index of the element of the maximum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public int Max(CudaDeviceVariable<cuDoubleComplex> x, int incx)
		{
			int result = 0;
			_status = CudaBlasNativeMethods.cublasIzamax_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIzamax_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return result;
		}
		/// <summary>
		/// This function finds the (smallest) index of the element of the maximum magnitude.<para/>
		/// First index starts at 0 (C notation, not Fortran)
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void Max(CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<int> result)
		{
			_status = CudaBlasNativeMethods.cublasIzamax_v2(_blasHandle, x.Size, x.DevicePointer, incx, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIzamax_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region aSum
		/// <summary>
		/// This function computes the sum of the absolute values of the elements of vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void AbsoluteSum(CudaDeviceVariable<float> x, int incx, ref float result)
		{
			_status = CudaBlasNativeMethods.cublasSasum_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSasum_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function computes the sum of the absolute values of the elements of vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public float AbsoluteSum(CudaDeviceVariable<float> x, int incx)
		{
			float result = 0;
			_status = CudaBlasNativeMethods.cublasSasum_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSasum_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return result;
		}
		/// <summary>
		/// This function computes the sum of the absolute values of the elements of vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void AbsoluteSum(CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> result)
		{
			_status = CudaBlasNativeMethods.cublasSasum_v2(_blasHandle, x.Size, x.DevicePointer, incx, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSasum_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function computes the sum of the absolute values of the elements of vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void AbsoluteSum(CudaDeviceVariable<double> x, int incx, ref double result)
		{
			_status = CudaBlasNativeMethods.cublasDasum_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDasum_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function computes the sum of the absolute values of the elements of vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public double AbsoluteSum(CudaDeviceVariable<double> x, int incx)
		{
			double result = 0;
			_status = CudaBlasNativeMethods.cublasDasum_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDasum_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return result;
		}
		/// <summary>
		/// This function computes the sum of the absolute values of the elements of vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void AbsoluteSum(CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> result)
		{
			_status = CudaBlasNativeMethods.cublasIdamax_v2(_blasHandle, x.Size, x.DevicePointer, incx, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasIdamax_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function computes the sum of the absolute values of the elements of vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void AbsoluteSum(CudaDeviceVariable<cuFloatComplex> x, int incx, ref float result)
		{
			_status = CudaBlasNativeMethods.cublasScasum_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasScasum_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function computes the sum of the absolute values of the elements of vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public float AbsoluteSum(CudaDeviceVariable<cuFloatComplex> x, int incx)
		{
			float result = 0;
			_status = CudaBlasNativeMethods.cublasScasum_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasScasum_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return result;
		}
		/// <summary>
		/// This function computes the sum of the absolute values of the elements of vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void AbsoluteSum(CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<float> result)
		{
			_status = CudaBlasNativeMethods.cublasScasum_v2(_blasHandle, x.Size, x.DevicePointer, incx, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasScasum_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function computes the sum of the absolute values of the elements of vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void AbsoluteSum(CudaDeviceVariable<cuDoubleComplex> x, int incx, ref double result)
		{
			_status = CudaBlasNativeMethods.cublasDzasum_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDzasum_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function computes the sum of the absolute values of the elements of vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		public double AbsoluteSum(CudaDeviceVariable<cuDoubleComplex> x, int incx)
		{
			double result = 0;
			_status = CudaBlasNativeMethods.cublasDzasum_v2(_blasHandle, x.Size, x.DevicePointer, incx, ref result);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDzasum_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return result;
		}
		/// <summary>
		/// This function computes the sum of the absolute values of the elements of vector x.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="result"></param>
		public void AbsoluteSum(CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<double> result)
		{
			_status = CudaBlasNativeMethods.cublasDzasum_v2(_blasHandle, x.Size, x.DevicePointer, incx, result.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDzasum_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region Rot
		/// <summary>
		/// This function applies Givens rotation matrix G = |c s; -s c| to vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="c">Cosine component</param>
		/// <param name="s">Sine component</param>
		public void Rot(CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> y, int incy, float c, float s)
		{
			_status = CudaBlasNativeMethods.cublasSrot_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, ref c, ref s);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSrot_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function applies Givens rotation matrix G = |c s; -s c| to vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="c">Cosine component</param>
		/// <param name="s">Sine component</param>
		public void Rot(CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> y, int incy, CudaDeviceVariable<float> c, CudaDeviceVariable<float> s)
		{
			_status = CudaBlasNativeMethods.cublasSrot_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, c.DevicePointer, s.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSrot_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function applies Givens rotation matrix G = |c s; -s c| to vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="c">Cosine component</param>
		/// <param name="s">Sine component</param>
		public void Rot(CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> y, int incy, double c, double s)
		{
			_status = CudaBlasNativeMethods.cublasDrot_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, ref c, ref s);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDrot_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function applies Givens rotation matrix G = |c s; -s c| to vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="c">Cosine component</param>
		/// <param name="s">Sine component</param>
		public void Rot(CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> y, int incy, CudaDeviceVariable<double> c, CudaDeviceVariable<double> s)
		{
			_status = CudaBlasNativeMethods.cublasDrot_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, c.DevicePointer, s.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDrot_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function applies Givens rotation matrix G = |c s; -s c| to vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="c">Cosine component</param>
		/// <param name="s">Sine component</param>
		public void Rot(CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy, float c, cuFloatComplex s)
		{
			_status = CudaBlasNativeMethods.cublasCrot_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, ref c, ref s);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCrot_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function applies Givens rotation matrix G = |c s; -s c| to vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="c">Cosine component</param>
		/// <param name="s">Sine component</param>
		public void Rot(CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy, CudaDeviceVariable<float> c, CudaDeviceVariable<cuFloatComplex> s)
		{
			_status = CudaBlasNativeMethods.cublasCrot_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, c.DevicePointer, s.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCrot_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function applies Givens rotation matrix G = |c s; -s c| to vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="c">Cosine component</param>
		/// <param name="s">Sine component</param>
		public void Rot(CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy, float c, float s)
		{
			_status = CudaBlasNativeMethods.cublasCsrot_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, ref c, ref s);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCsrot_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function applies Givens rotation matrix G = |c s; -s c| to vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="c">Cosine component</param>
		/// <param name="s">Sine component</param>
		public void Rot(CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy, CudaDeviceVariable<float> c, CudaDeviceVariable<float> s)
		{
			_status = CudaBlasNativeMethods.cublasCsrot_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, c.DevicePointer, s.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCsrot_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function applies Givens rotation matrix G = |c s; -s c| to vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="c">Cosine component</param>
		/// <param name="s">Sine component</param>
		public void Rot(CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy, double c, cuDoubleComplex s)
		{
			_status = CudaBlasNativeMethods.cublasZrot_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, ref c, ref s);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZrot_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function applies Givens rotation matrix G = |c s; -s c| to vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="c">Cosine component</param>
		/// <param name="s">Sine component</param>
		public void Rot(CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy, CudaDeviceVariable<double> c, CudaDeviceVariable<cuDoubleComplex> s)
		{
			_status = CudaBlasNativeMethods.cublasZrot_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, c.DevicePointer, s.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZrot_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function applies Givens rotation matrix G = |c s; -s c| to vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="c">Cosine component</param>
		/// <param name="s">Sine component</param>
		public void Rot(CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy, double c, double s)
		{
			_status = CudaBlasNativeMethods.cublasZdrot_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, ref c, ref s);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZdrot_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function applies Givens rotation matrix G = |c s; -s c| to vectors x and y.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="c">Cosine component</param>
		/// <param name="s">Sine component</param>
		public void Rot(CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy, CudaDeviceVariable<double> c, CudaDeviceVariable<double> s)
		{
			_status = CudaBlasNativeMethods.cublasZdrot_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, c.DevicePointer, s.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZdrot_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region Rotg
		/// <summary>
		/// This function constructs the Givens rotation matrix G = |c s; -s c| that zeros out the second entry of a 2x1 vector (a; b)T
		/// </summary>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="c">Cosine component</param>
		/// <param name="s">Sine component</param>
		public void Rotg(CudaDeviceVariable<float> a, CudaDeviceVariable<float> b, CudaDeviceVariable<float> c, CudaDeviceVariable<float> s)
		{
			_status = CudaBlasNativeMethods.cublasSrotg_v2(_blasHandle, a.DevicePointer, b.DevicePointer, c.DevicePointer, s.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSrotg_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function constructs the Givens rotation matrix G = |c s; -s c| that zeros out the second entry of a 2x1 vector (a; b)T
		/// </summary>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="c">Cosine component</param>
		/// <param name="s">Sine component</param>
		public void Rotg(ref float a, ref float b, ref float c, ref float s)
		{
			_status = CudaBlasNativeMethods.cublasSrotg_v2(_blasHandle, ref a, ref b, ref c, ref s);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSrotg_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function constructs the Givens rotation matrix G = |c s; -s c| that zeros out the second entry of a 2x1 vector (a; b)T
		/// </summary>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="c">Cosine component</param>
		/// <param name="s">Sine component</param>
		public void Rotg(CudaDeviceVariable<double> a, CudaDeviceVariable<double> b, CudaDeviceVariable<double> c, CudaDeviceVariable<double> s)
		{
			_status = CudaBlasNativeMethods.cublasDrotg_v2(_blasHandle, a.DevicePointer, b.DevicePointer, c.DevicePointer, s.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDrotg_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function constructs the Givens rotation matrix G = |c s; -s c| that zeros out the second entry of a 2x1 vector (a; b)T
		/// </summary>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="c">Cosine component</param>
		/// <param name="s">Sine component</param>
		public void Rotg(ref double a, ref double b, ref double c, ref double s)
		{
			_status = CudaBlasNativeMethods.cublasDrotg_v2(_blasHandle, ref a, ref b, ref c, ref s);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDrotg_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function constructs the Givens rotation matrix G = |c s; -s c| that zeros out the second entry of a 2x1 vector (a; b)T
		/// </summary>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="c">Cosine component</param>
		/// <param name="s">Sine component</param>
		public void Rotg(CudaDeviceVariable<cuFloatComplex> a, CudaDeviceVariable<cuFloatComplex> b, CudaDeviceVariable<float> c, CudaDeviceVariable<cuFloatComplex> s)
		{
			_status = CudaBlasNativeMethods.cublasCrotg_v2(_blasHandle, a.DevicePointer, b.DevicePointer, c.DevicePointer, s.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCrotg_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function constructs the Givens rotation matrix G = |c s; -s c| that zeros out the second entry of a 2x1 vector (a; b)T
		/// </summary>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="c">Cosine component</param>
		/// <param name="s">Sine component</param>
		public void Rotg(ref cuFloatComplex a, ref cuFloatComplex b, ref float c, ref cuFloatComplex s)
		{
			_status = CudaBlasNativeMethods.cublasCrotg_v2(_blasHandle, ref a, ref b, ref c, ref s);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCrotg_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function constructs the Givens rotation matrix G = |c s; -s c| that zeros out the second entry of a 2x1 vector (a; b)T
		/// </summary>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="c">Cosine component</param>
		/// <param name="s">Sine component</param>
		public void Rotg(CudaDeviceVariable<cuDoubleComplex> a, CudaDeviceVariable<cuDoubleComplex> b, CudaDeviceVariable<cuDoubleComplex> c, CudaDeviceVariable<cuDoubleComplex> s)
		{
			_status = CudaBlasNativeMethods.cublasZrotg_v2(_blasHandle, a.DevicePointer, b.DevicePointer, c.DevicePointer, s.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZrotg_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function constructs the Givens rotation matrix G = |c s; -s c| that zeros out the second entry of a 2x1 vector (a; b)T
		/// </summary>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="c">Cosine component</param>
		/// <param name="s">Sine component</param>
		public void Rotg(ref cuDoubleComplex a, ref cuDoubleComplex b, ref double c, ref cuDoubleComplex s)
		{
			_status = CudaBlasNativeMethods.cublasZrotg_v2(_blasHandle, ref a, ref b, ref c, ref s);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZrotg_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region Rotm
		/// <summary>
		/// This function applies the modified Givens transformation H = |h11 h12; h21 h22| to vectors x and y.<para/>
		/// The elements h11, h21, h12 and h22 of 2x2 matrix H are stored in param[1], param[2], param[3] and param[4], respectively. <para/>
		/// The flag = param[0] defines the following predefined values for the matrix H entries:<para/>
		/// flag=-1.0: H = |h11 h12; h21 h22|<para/>
		/// flag= 0.0: H = |1.0 h12; h21 1.0|<para/> 
		/// flag= 1.0: H = |h11 1.0; -1.0 h22|<para/>
		/// flag=-2.0: H = |1.0 0.0; 0.0 1.0|<para/>
		/// Notice that the values -1.0, 0.0 and 1.0 implied by the flag are not stored in param.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="param"></param>
		public void Rotm(CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> y, int incy, float[] param)
		{
			_status = CudaBlasNativeMethods.cublasSrotm_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, param);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSrotg_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function applies the modified Givens transformation H = |h11 h12; h21 h22| to vectors x and y.<para/>
		/// The elements h11, h21, h12 and h22 of 2x2 matrix H are stored in param[1], param[2], param[3] and param[4], respectively. <para/>
		/// The flag = param[0] defines the following predefined values for the matrix H entries:<para/>
		/// flag=-1.0: H = |h11 h12; h21 h22|<para/>
		/// flag= 0.0: H = |1.0 h12; h21 1.0|<para/> 
		/// flag= 1.0: H = |h11 1.0; -1.0 h22|<para/>
		/// flag=-2.0: H = |1.0 0.0; 0.0 1.0|<para/>
		/// Notice that the values -1.0, 0.0 and 1.0 implied by the flag are not stored in param.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="param"></param>
		public void Rotm(CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> y, int incy, CudaDeviceVariable<float> param)
		{
			_status = CudaBlasNativeMethods.cublasSrotm_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, param.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSrotg_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function applies the modified Givens transformation H = |h11 h12; h21 h22| to vectors x and y.<para/>
		/// The elements h11, h21, h12 and h22 of 2x2 matrix H are stored in param[1], param[2], param[3] and param[4], respectively. <para/>
		/// The flag = param[0] defines the following predefined values for the matrix H entries:<para/>
		/// flag=-1.0: H = |h11 h12; h21 h22|<para/>
		/// flag= 0.0: H = |1.0 h12; h21 1.0|<para/> 
		/// flag= 1.0: H = |h11 1.0; -1.0 h22|<para/>
		/// flag=-2.0: H = |1.0 0.0; 0.0 1.0|<para/>
		/// Notice that the values -1.0, 0.0 and 1.0 implied by the flag are not stored in param.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="param"></param>
		public void Rotm(CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> y, int incy, double[] param)
		{
			_status = CudaBlasNativeMethods.cublasDrotm_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, param);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDrotg_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function applies the modified Givens transformation H = |h11 h12; h21 h22| to vectors x and y.<para/>
		/// The elements h11, h21, h12 and h22 of 2x2 matrix H are stored in param[1], param[2], param[3] and param[4], respectively. <para/>
		/// The flag = param[0] defines the following predefined values for the matrix H entries:<para/>
		/// flag=-1.0: H = |h11 h12; h21 h22|<para/>
		/// flag= 0.0: H = |1.0 h12; h21 1.0|<para/> 
		/// flag= 1.0: H = |h11 1.0; -1.0 h22|<para/>
		/// flag=-2.0: H = |1.0 0.0; 0.0 1.0|<para/>
		/// Notice that the values -1.0, 0.0 and 1.0 implied by the flag are not stored in param.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="incx"></param>
		/// <param name="y"></param>
		/// <param name="incy"></param>
		/// <param name="param"></param>
		public void Rotm(CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> y, int incy, CudaDeviceVariable<double> param)
		{
			_status = CudaBlasNativeMethods.cublasDrotm_v2(_blasHandle, x.Size, x.DevicePointer, incx, y.DevicePointer, incy, param.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDrotg_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region Rotmg
		/// <summary>
		/// This function constructs the modified Givens transformation H = |h11 h12; h21 h22| that zeros out the second entry of a 2x1 vector 
		/// [sqrt(d1)*x1; sqrt(d2)*y1].<para/>
		/// The elements h11, h21, h12 and h22 of 2x2 matrix H are stored in param[1], param[2], param[3] and param[4], respectively. <para/>
		/// The flag = param[0] defines the following predefined values for the matrix H entries:<para/>
		/// flag=-1.0: H = |h11 h12; h21 h22|<para/>
		/// flag= 0.0: H = |1.0 h12; h21 1.0|<para/> 
		/// flag= 1.0: H = |h11 1.0; -1.0 h22|<para/>
		/// flag=-2.0: H = |1.0 0.0; 0.0 1.0|<para/>
		/// Notice that the values -1.0, 0.0 and 1.0 implied by the flag are not stored in param.
		/// </summary>
		/// <param name="d1"></param>
		/// <param name="d2"></param>
		/// <param name="x1"></param>
		/// <param name="y1"></param>
		/// <param name="param"></param>
		public void Rotm(ref float d1, ref float d2, ref float x1, float y1, float[] param)
		{
			_status = CudaBlasNativeMethods.cublasSrotmg_v2(_blasHandle, ref d1, ref d2, ref x1, ref y1, param);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSrotmg_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function constructs the modified Givens transformation H = |h11 h12; h21 h22| that zeros out the second entry of a 2x1 vector 
		/// [sqrt(d1)*x1; sqrt(d2)*y1].<para/>
		/// The elements h11, h21, h12 and h22 of 2x2 matrix H are stored in param[1], param[2], param[3] and param[4], respectively. <para/>
		/// The flag = param[0] defines the following predefined values for the matrix H entries:<para/>
		/// flag=-1.0: H = |h11 h12; h21 h22|<para/>
		/// flag= 0.0: H = |1.0 h12; h21 1.0|<para/> 
		/// flag= 1.0: H = |h11 1.0; -1.0 h22|<para/>
		/// flag=-2.0: H = |1.0 0.0; 0.0 1.0|<para/>
		/// Notice that the values -1.0, 0.0 and 1.0 implied by the flag are not stored in param.
		/// </summary>
		/// <param name="d1"></param>
		/// <param name="d2"></param>
		/// <param name="x1"></param>
		/// <param name="y1"></param>
		/// <param name="param"></param>
		public void Rotm(CudaDeviceVariable<float> d1, CudaDeviceVariable<float> d2, CudaDeviceVariable<float> x1, CudaDeviceVariable<float> y1, CudaDeviceVariable<float> param)
		{
			_status = CudaBlasNativeMethods.cublasSrotmg_v2(_blasHandle, d1.DevicePointer, d2.DevicePointer, x1.DevicePointer, y1.DevicePointer, param.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSrotmg_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function constructs the modified Givens transformation H = |h11 h12; h21 h22| that zeros out the second entry of a 2x1 vector 
		/// [sqrt(d1)*x1; sqrt(d2)*y1].<para/>
		/// The elements h11, h21, h12 and h22 of 2x2 matrix H are stored in param[1], param[2], param[3] and param[4], respectively. <para/>
		/// The flag = param[0] defines the following predefined values for the matrix H entries:<para/>
		/// flag=-1.0: H = |h11 h12; h21 h22|<para/>
		/// flag= 0.0: H = |1.0 h12; h21 1.0|<para/> 
		/// flag= 1.0: H = |h11 1.0; -1.0 h22|<para/>
		/// flag=-2.0: H = |1.0 0.0; 0.0 1.0|<para/>
		/// Notice that the values -1.0, 0.0 and 1.0 implied by the flag are not stored in param.
		/// </summary>
		/// <param name="d1"></param>
		/// <param name="d2"></param>
		/// <param name="x1"></param>
		/// <param name="y1"></param>
		/// <param name="param"></param>
		public void Rotm(ref double d1, ref double d2, ref double x1, double y1, double[] param)
		{
			_status = CudaBlasNativeMethods.cublasDrotmg_v2(_blasHandle, ref d1, ref d2, ref x1, ref y1, param);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDrotmg_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function constructs the modified Givens transformation H = |h11 h12; h21 h22| that zeros out the second entry of a 2x1 vector 
		/// [sqrt(d1)*x1; sqrt(d2)*y1].<para/>
		/// The elements h11, h21, h12 and h22 of 2x2 matrix H are stored in param[1], param[2], param[3] and param[4], respectively. <para/>
		/// The flag = param[0] defines the following predefined values for the matrix H entries:<para/>
		/// flag=-1.0: H = |h11 h12; h21 h22|<para/>
		/// flag= 0.0: H = |1.0 h12; h21 1.0|<para/> 
		/// flag= 1.0: H = |h11 1.0; -1.0 h22|<para/>
		/// flag=-2.0: H = |1.0 0.0; 0.0 1.0|<para/>
		/// Notice that the values -1.0, 0.0 and 1.0 implied by the flag are not stored in param.
		/// </summary>
		/// <param name="d1"></param>
		/// <param name="d2"></param>
		/// <param name="x1"></param>
		/// <param name="y1"></param>
		/// <param name="param"></param>
		public void Rotm(CudaDeviceVariable<double> d1, CudaDeviceVariable<double> d2, CudaDeviceVariable<double> x1, CudaDeviceVariable<double> y1, CudaDeviceVariable<double> param)
		{
			_status = CudaBlasNativeMethods.cublasDrotmg_v2(_blasHandle, d1.DevicePointer, d2.DevicePointer, x1.DevicePointer, y1.DevicePointer, param.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDrotmg_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#endregion

		#region BLAS2
		#region TRMV
		/// <summary>
		/// This function performs the triangular matrix-vector multiplication x= Op(A) x where A is a triangular matrix stored in 
		/// lower or upper mode with or without the main diagonal, and x is a vector. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Trmv(FillMode uplo, Operation trans, DiagType diag, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasStrmv_v2(_blasHandle, uplo, trans, diag, x.Size, A.DevicePointer, lda, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasStrmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the triangular matrix-vector multiplication x= Op(A) x where A is a triangular matrix stored in 
		/// lower or upper mode with or without the main diagonal, and x is a vector. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Trmv(FillMode uplo, Operation trans, DiagType diag, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasDtrmv_v2(_blasHandle, uplo, trans, diag, x.Size, A.DevicePointer, lda, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDtrmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the triangular matrix-vector multiplication x= Op(A) x where A is a triangular matrix stored in 
		/// lower or upper mode with or without the main diagonal, and x is a vector. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Trmv(FillMode uplo, Operation trans, DiagType diag, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasCtrmv_v2(_blasHandle, uplo, trans, diag, x.Size, A.DevicePointer, lda, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCtrmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the triangular matrix-vector multiplication x= Op(A) x where A is a triangular matrix stored in 
		/// lower or upper mode with or without the main diagonal, and x is a vector. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Trmv(FillMode uplo, Operation trans, DiagType diag, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasZtrmv_v2(_blasHandle, uplo, trans, diag, x.Size, A.DevicePointer, lda, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZtrmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region TBMV
		/// <summary>
		/// This function performs the triangular banded matrix-vector multiplication x= Op(A) x where A is a triangular banded matrix, and x is a vector. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="k">number of sub- and super-diagonals of matrix A.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Tbmv(FillMode uplo, Operation trans, DiagType diag, int k, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasStbmv_v2(_blasHandle, uplo, trans, diag, x.Size, k, A.DevicePointer, lda, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasStbmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the triangular banded matrix-vector multiplication x= Op(A) x where A is a triangular banded matrix, and x is a vector. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="k">number of sub- and super-diagonals of matrix A.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Tbmv(FillMode uplo, Operation trans, DiagType diag, int k, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasDtbmv_v2(_blasHandle, uplo, trans, diag, x.Size, k, A.DevicePointer, lda, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDtbmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the triangular banded matrix-vector multiplication x= Op(A) x where A is a triangular banded matrix, and x is a vector. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="k">number of sub- and super-diagonals of matrix A.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Tbmv(FillMode uplo, Operation trans, DiagType diag, int k, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasCtbmv_v2(_blasHandle, uplo, trans, diag, x.Size, k, A.DevicePointer, lda, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCtbmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the triangular banded matrix-vector multiplication x= Op(A) x where A is a triangular banded matrix, and x is a vector. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="k">number of sub- and super-diagonals of matrix A.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Tbmv(FillMode uplo, Operation trans, DiagType diag, int k, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasZtbmv_v2(_blasHandle, uplo, trans, diag, x.Size, k, A.DevicePointer, lda, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZtbmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region TPMV
		/// <summary>
		/// This function performs the triangular packed matrix-vector multiplication x= Op(A) x where A is a triangular matrix stored in packed format, and x is a vector. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="AP">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Tpmv(FillMode uplo, Operation trans, DiagType diag, CudaDeviceVariable<float> AP, CudaDeviceVariable<float> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasStpmv_v2(_blasHandle, uplo, trans, diag, x.Size, AP.DevicePointer, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasStpmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the triangular packed matrix-vector multiplication x= Op(A) x where A is a triangular matrix stored in packed format, and x is a vector. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="AP">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Tpmv(FillMode uplo, Operation trans, DiagType diag, CudaDeviceVariable<double> AP, CudaDeviceVariable<double> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasDtpmv_v2(_blasHandle, uplo, trans, diag, x.Size, AP.DevicePointer, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDtpmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the triangular packed matrix-vector multiplication x= Op(A) x where A is a triangular matrix stored in packed format, and x is a vector. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="AP">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Tpmv(FillMode uplo, Operation trans, DiagType diag, CudaDeviceVariable<cuFloatComplex> AP, CudaDeviceVariable<cuFloatComplex> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasCtpmv_v2(_blasHandle, uplo, trans, diag, x.Size, AP.DevicePointer, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCtpmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the triangular packed matrix-vector multiplication x= Op(A) x where A is a triangular matrix stored in packed format, and x is a vector. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="AP">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Tpmv(FillMode uplo, Operation trans, DiagType diag, CudaDeviceVariable<cuDoubleComplex> AP, CudaDeviceVariable<cuDoubleComplex> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasZtpmv_v2(_blasHandle, uplo, trans, diag, x.Size, AP.DevicePointer, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZtpmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region TRSV
		/// <summary>
		/// This function solves the triangular linear system with a single right-hand-side Op(A)x = b where A is a triangular matrix stored in lower or 
		/// upper mode with or without the main diagonal, and x and b are vectors. The solution x overwrites the right-hand-sides b on exit. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Trsv(FillMode uplo, Operation trans, DiagType diag, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasStrsv_v2(_blasHandle, uplo, trans, diag, x.Size, A.DevicePointer, lda, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasStrsv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function solves the triangular linear system with a single right-hand-side Op(A)x = b where A is a triangular matrix stored in lower or 
		/// upper mode with or without the main diagonal, and x and b are vectors. The solution x overwrites the right-hand-sides b on exit. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Trsv(FillMode uplo, Operation trans, DiagType diag, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasDtrsv_v2(_blasHandle, uplo, trans, diag, x.Size, A.DevicePointer, lda, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDtrsv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function solves the triangular linear system with a single right-hand-side Op(A)x = b where A is a triangular matrix stored in lower or 
		/// upper mode with or without the main diagonal, and x and b are vectors. The solution x overwrites the right-hand-sides b on exit. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Trsv(FillMode uplo, Operation trans, DiagType diag, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasCtrsv_v2(_blasHandle, uplo, trans, diag, x.Size, A.DevicePointer, lda, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCtrsv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function solves the triangular linear system with a single right-hand-side Op(A)x = b where A is a triangular matrix stored in lower or 
		/// upper mode with or without the main diagonal, and x and b are vectors. The solution x overwrites the right-hand-sides b on exit. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Trsv(FillMode uplo, Operation trans, DiagType diag, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasZtrsv_v2(_blasHandle, uplo, trans, diag, x.Size, A.DevicePointer, lda, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZtrsv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region TPSV
		/// <summary>
		/// This function solves the packed triangular linear system with a single right-hand-side Op(A) x = b where A is a triangular matrix stored in packed format, and x and b are vectors. 
		/// The solution x overwrites the right-hand-sides b on exit. n is given by x.Size. No test for singularity or near-singularity is included in this function.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="AP">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Tpsv(FillMode uplo, Operation trans, DiagType diag, CudaDeviceVariable<float> AP, CudaDeviceVariable<float> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasStpsv_v2(_blasHandle, uplo, trans, diag, x.Size, AP.DevicePointer, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasStpsv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function solves the packed triangular linear system with a single right-hand-side Op(A) x = b where A is a triangular matrix stored in packed format, and x and b are vectors. 
		/// The solution x overwrites the right-hand-sides b on exit. n is given by x.Size. No test for singularity or near-singularity is included in this function.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="AP">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Tpsv(FillMode uplo, Operation trans, DiagType diag, CudaDeviceVariable<double> AP, CudaDeviceVariable<double> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasDtpsv_v2(_blasHandle, uplo, trans, diag, x.Size, AP.DevicePointer, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDtpsv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function solves the packed triangular linear system with a single right-hand-side Op(A) x = b where A is a triangular matrix stored in packed format, and x and b are vectors. 
		/// The solution x overwrites the right-hand-sides b on exit. n is given by x.Size. No test for singularity or near-singularity is included in this function.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="AP">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Tpsv(FillMode uplo, Operation trans, DiagType diag, CudaDeviceVariable<cuFloatComplex> AP, CudaDeviceVariable<cuFloatComplex> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasCtpsv_v2(_blasHandle, uplo, trans, diag, x.Size, AP.DevicePointer, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCtpsv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function solves the packed triangular linear system with a single right-hand-side Op(A) x = b where A is a triangular matrix stored in packed format, and x and b are vectors. 
		/// The solution x overwrites the right-hand-sides b on exit. n is given by x.Size. No test for singularity or near-singularity is included in this function.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="AP">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Tpsv(FillMode uplo, Operation trans, DiagType diag, CudaDeviceVariable<cuDoubleComplex> AP, CudaDeviceVariable<cuDoubleComplex> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasZtpsv_v2(_blasHandle, uplo, trans, diag, x.Size, AP.DevicePointer, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZtpsv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region TBSV
		/// <summary>
		/// This function solves the triangular banded linear system with a single right-hand-side Op(A) x = b where A is a triangular banded matrix, and x and b is a vector. 
		/// The solution x overwrites the right-hand-sides b on exit. n is given by x.Size. No test for singularity or near-singularity is included in this function.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="k">number of sub- and super-diagonals of matrix A.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Tbsv(FillMode uplo, Operation trans, DiagType diag, int k, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasStbsv_v2(_blasHandle, uplo, trans, diag, x.Size, k, A.DevicePointer, lda, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasStbsv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function solves the triangular banded linear system with a single right-hand-side Op(A) x = b where A is a triangular banded matrix, and x and b is a vector. 
		/// The solution x overwrites the right-hand-sides b on exit. n is given by x.Size. No test for singularity or near-singularity is included in this function.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="k">number of sub- and super-diagonals of matrix A.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Tbsv(FillMode uplo, Operation trans, DiagType diag, int k, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasDtbsv_v2(_blasHandle, uplo, trans, diag, x.Size, k, A.DevicePointer, lda, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDtbsv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function solves the triangular banded linear system with a single right-hand-side Op(A) x = b where A is a triangular banded matrix, and x and b is a vector. 
		/// The solution x overwrites the right-hand-sides b on exit. n is given by x.Size. No test for singularity or near-singularity is included in this function.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="k">number of sub- and super-diagonals of matrix A.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Tbsv(FillMode uplo, Operation trans, DiagType diag, int k, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasCtbsv_v2(_blasHandle, uplo, trans, diag, x.Size, k, A.DevicePointer, lda, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCtbsv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function solves the triangular banded linear system with a single right-hand-side Op(A) x = b where A is a triangular banded matrix, and x and b is a vector. 
		/// The solution x overwrites the right-hand-sides b on exit. n is given by x.Size. No test for singularity or near-singularity is included in this function.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="k">number of sub- and super-diagonals of matrix A.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		public void Tbsv(FillMode uplo, Operation trans, DiagType diag, int k, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> x, int incx)
		{
			_status = CudaBlasNativeMethods.cublasZtbsv_v2(_blasHandle, uplo, trans, diag, x.Size, k, A.DevicePointer, lda, x.DevicePointer, incx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZtbsv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion

		#region GEMV
		/// <summary>
		/// This function performs the matrix-vector multiplication y = alpha * Op(A) * x + beta * y where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha and beta are scalars.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Gemv(Operation trans, int m, int n, float alpha, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> x, int incx, float beta, CudaDeviceVariable<float> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasSgemv_v2(_blasHandle, trans, m ,n , ref alpha, A.DevicePointer, lda, x.DevicePointer, incx, ref beta, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSgemv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-vector multiplication y = alpha * Op(A) * x + beta * y where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha and beta are scalars.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Gemv(Operation trans, int m, int n, CudaDeviceVariable<float> alpha, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> beta, CudaDeviceVariable<float> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasSgemv_v2(_blasHandle, trans, m, n, alpha.DevicePointer, A.DevicePointer, lda, x.DevicePointer, incx, beta.DevicePointer, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSgemv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the matrix-vector multiplication y = alpha * Op(A) * x + beta * y where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha and beta are scalars.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Gemv(Operation trans, int m, int n, double alpha, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> x, int incx, double beta, CudaDeviceVariable<double> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasDgemv_v2(_blasHandle, trans, m, n, ref alpha, A.DevicePointer, lda, x.DevicePointer, incx, ref beta, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDgemv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-vector multiplication y = alpha * Op(A) * x + beta * y where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha and beta are scalars.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Gemv(Operation trans, int m, int n, CudaDeviceVariable<double> alpha, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> beta, CudaDeviceVariable<double> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasDgemv_v2(_blasHandle, trans, m, n, alpha.DevicePointer, A.DevicePointer, lda, x.DevicePointer, incx, beta.DevicePointer, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDgemv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-vector multiplication y = alpha * Op(A) * x + beta * y where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha and beta are scalars.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Gemv(Operation trans, int m, int n, cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> x, int incx, cuFloatComplex beta, CudaDeviceVariable<cuFloatComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasCgemv_v2(_blasHandle, trans, m, n, ref alpha, A.DevicePointer, lda, x.DevicePointer, incx, ref beta, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCgemv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-vector multiplication y = alpha * Op(A) * x + beta * y where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha and beta are scalars.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Gemv(Operation trans, int m, int n, CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> beta, CudaDeviceVariable<cuFloatComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasCgemv_v2(_blasHandle, trans, m, n, alpha.DevicePointer, A.DevicePointer, lda, x.DevicePointer, incx, beta.DevicePointer, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCgemv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the matrix-vector multiplication y = alpha * Op(A) * x + beta * y where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha and beta are scalars.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Gemv(Operation trans, int m, int n, cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> x, int incx, cuDoubleComplex beta, CudaDeviceVariable<cuDoubleComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasZgemv_v2(_blasHandle, trans, m, n, ref alpha, A.DevicePointer, lda, x.DevicePointer, incx, ref beta, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZgemv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-vector multiplication y = alpha * Op(A) * x + beta * y where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha and beta are scalars.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Gemv(Operation trans, int m, int n, CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> beta, CudaDeviceVariable<cuDoubleComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasZgemv_v2(_blasHandle, trans, m, n, alpha.DevicePointer, A.DevicePointer, lda, x.DevicePointer, incx, beta.DevicePointer, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZgemv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region GBMV
		/// <summary>
		/// This function performs the matrix-vector multiplication y = alpha * Op(A) * x + beta * y where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha and beta are scalars.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="kl">number of subdiagonals of matrix A.</param>
		/// <param name="ku">number of superdiagonals of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Gbmv(Operation trans, int m, int n, int kl, int ku, float alpha, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> x, int incx, float beta, CudaDeviceVariable<float> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasSgbmv_v2(_blasHandle, trans, m, n, kl, ku, ref alpha, A.DevicePointer, lda, x.DevicePointer, incx, ref beta, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSgbmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-vector multiplication y = alpha * Op(A) * x + beta * y where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha and beta are scalars.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="kl">number of subdiagonals of matrix A.</param>
		/// <param name="ku">number of superdiagonals of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Gbmv(Operation trans, int m, int n, int kl, int ku, CudaDeviceVariable<float> alpha, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> beta, CudaDeviceVariable<float> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasSgbmv_v2(_blasHandle, trans, m, n, kl, ku, alpha.DevicePointer, A.DevicePointer, lda, x.DevicePointer, incx, beta.DevicePointer, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSgbmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the matrix-vector multiplication y = alpha * Op(A) * x + beta * y where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha and beta are scalars.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="kl">number of subdiagonals of matrix A.</param>
		/// <param name="ku">number of superdiagonals of matrix A.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Gbmv(Operation trans, int m, int n, int kl, int ku, double alpha, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> x, int incx, double beta, CudaDeviceVariable<double> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasDgbmv_v2(_blasHandle, trans, m, n, kl, ku, ref alpha, A.DevicePointer, lda, x.DevicePointer, incx, ref beta, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDgbmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-vector multiplication y = alpha * Op(A) * x + beta * y where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha and beta are scalars.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="kl">number of subdiagonals of matrix A.</param>
		/// <param name="ku">number of superdiagonals of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Gbmv(Operation trans, int m, int n, int kl, int ku, CudaDeviceVariable<double> alpha, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> beta, CudaDeviceVariable<double> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasDgbmv_v2(_blasHandle, trans, m, n, kl, ku, alpha.DevicePointer, A.DevicePointer, lda, x.DevicePointer, incx, beta.DevicePointer, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDgbmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-vector multiplication y = alpha * Op(A) * x + beta * y where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha and beta are scalars.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="kl">number of subdiagonals of matrix A.</param>
		/// <param name="ku">number of superdiagonals of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Gbmv(Operation trans, int m, int n, int kl, int ku, cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> x, int incx, cuFloatComplex beta, CudaDeviceVariable<cuFloatComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasCgbmv_v2(_blasHandle, trans, m, n, kl, ku, ref alpha, A.DevicePointer, lda, x.DevicePointer, incx, ref beta, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCgbmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-vector multiplication y = alpha * Op(A) * x + beta * y where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha and beta are scalars.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="kl">number of subdiagonals of matrix A.</param>
		/// <param name="ku">number of superdiagonals of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Gbmv(Operation trans, int m, int n, int kl, int ku, CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> beta, CudaDeviceVariable<cuFloatComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasCgbmv_v2(_blasHandle, trans, m, n, kl, ku, alpha.DevicePointer, A.DevicePointer, lda, x.DevicePointer, incx, beta.DevicePointer, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCgbmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the matrix-vector multiplication y = alpha * Op(A) * x + beta * y where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha and beta are scalars.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="kl">number of subdiagonals of matrix A.</param>
		/// <param name="ku">number of superdiagonals of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Gbmv(Operation trans, int m, int n, int kl, int ku, cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> x, int incx, cuDoubleComplex beta, CudaDeviceVariable<cuDoubleComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasZgbmv_v2(_blasHandle, trans, m, n, kl, ku, ref alpha, A.DevicePointer, lda, x.DevicePointer, incx, ref beta, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZgbmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-vector multiplication y = alpha * Op(A) * x + beta * y where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha and beta are scalars.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix A.</param>
		/// <param name="n">number of columns of matrix A.</param>
		/// <param name="kl">number of subdiagonals of matrix A.</param>
		/// <param name="ku">number of superdiagonals of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Gbmv(Operation trans, int m, int n, int kl, int ku, CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> beta, CudaDeviceVariable<cuDoubleComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasZgbmv_v2(_blasHandle, trans, m, n, kl, ku, alpha.DevicePointer, A.DevicePointer, lda, x.DevicePointer, incx, beta.DevicePointer, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZgbmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region SYMV/HEMV
		/// <summary>
		/// This function performs the symmetric matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n symmetric matrix stored in lower or upper mode, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Symv(FillMode uplo, float alpha, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> x, int incx, float beta, CudaDeviceVariable<float> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasSsymv_v2(_blasHandle, uplo, x.Size, ref alpha, A.DevicePointer, lda, x.DevicePointer, incx, ref beta, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSsymv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n symmetric matrix stored in lower or upper mode, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Symv(FillMode uplo, CudaDeviceVariable<float> alpha, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> beta, CudaDeviceVariable<float> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasSsymv_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, A.DevicePointer, lda, x.DevicePointer, incx, beta.DevicePointer, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSsymv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n symmetric matrix stored in lower or upper mode, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Symv(FillMode uplo, double alpha, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> x, int incx, double beta, CudaDeviceVariable<double> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasDsymv_v2(_blasHandle, uplo, x.Size, ref alpha, A.DevicePointer, lda, x.DevicePointer, incx, ref beta, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDsymv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric matrix-vector multiplication y = alpha *A * x + beta * y where A is a n*n symmetric matrix stored in lower or upper mode, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Symv(FillMode uplo, CudaDeviceVariable<double> alpha, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> beta, CudaDeviceVariable<double> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasDsymv_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, A.DevicePointer, lda, x.DevicePointer, incx, beta.DevicePointer, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSsymv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n symmetric matrix stored in lower or upper mode, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Symv(FillMode uplo, cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> x, int incx, cuFloatComplex beta, CudaDeviceVariable<cuFloatComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasCsymv_v2(_blasHandle, uplo, x.Size, ref alpha, A.DevicePointer, lda, x.DevicePointer, incx, ref beta, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCsymv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n symmetric matrix stored in lower or upper mode, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Symv(FillMode uplo, CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> beta, CudaDeviceVariable<cuFloatComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasCsymv_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, A.DevicePointer, lda, x.DevicePointer, incx, beta.DevicePointer, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCsymv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n symmetric matrix stored in lower or upper mode, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Symv(FillMode uplo, cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> x, int incx, cuDoubleComplex beta, CudaDeviceVariable<cuDoubleComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasZsymv_v2(_blasHandle, uplo, x.Size, ref alpha, A.DevicePointer, lda, x.DevicePointer, incx, ref beta, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZsymv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric matrix-vector multiplication y = alpha *A * x + beta * y where A is a n*n symmetric matrix stored in lower or upper mode, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Symv(FillMode uplo, CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> beta, CudaDeviceVariable<cuDoubleComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasZsymv_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, A.DevicePointer, lda, x.DevicePointer, incx, beta.DevicePointer, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZsymv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}


		/// <summary>
		/// This function performs the Hermitian matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n Hermitian matrix stored in lower or upper mode, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Hemv(FillMode uplo, cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> x, int incx, cuFloatComplex beta, CudaDeviceVariable<cuFloatComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasChemv_v2(_blasHandle, uplo, x.Size, ref alpha, A.DevicePointer, lda, x.DevicePointer, incx, ref beta, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasChemv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the Hermitian matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n Hermitian matrix stored in lower or upper mode, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Hemv(FillMode uplo, CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> beta, CudaDeviceVariable<cuFloatComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasChemv_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, A.DevicePointer, lda, x.DevicePointer, incx, beta.DevicePointer, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasChemv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the Hermitian matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n Hermitian matrix stored in lower or upper mode, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Hemv(FillMode uplo, cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> x, int incx, cuDoubleComplex beta, CudaDeviceVariable<cuDoubleComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasZhemv_v2(_blasHandle, uplo, x.Size, ref alpha, A.DevicePointer, lda, x.DevicePointer, incx, ref beta, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZhemv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the Hermitian matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n Hermitian matrix stored in lower or upper mode, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Hemv(FillMode uplo, CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> beta, CudaDeviceVariable<cuDoubleComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasZhemv_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, A.DevicePointer, lda, x.DevicePointer, incx, beta.DevicePointer, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZhemv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region SBMV/HBMV
		/// <summary>
		/// This function performs the symmetric banded matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n symmetric matrix with k subdiagonals and superdiagonals, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="k">number of sub- and super-diagonals of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Sbmv(FillMode uplo, int k, float alpha, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> x, int incx, float beta, CudaDeviceVariable<float> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasSsbmv_v2(_blasHandle, uplo, x.Size, k, ref alpha, A.DevicePointer, lda, x.DevicePointer, incx, ref beta, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSsbmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric banded matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n symmetric matrix with k subdiagonals and superdiagonals, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="k">number of sub- and super-diagonals of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Sbmv(FillMode uplo, int k, CudaDeviceVariable<float> alpha, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> beta, CudaDeviceVariable<float> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasSsbmv_v2(_blasHandle, uplo, x.Size, k, alpha.DevicePointer, A.DevicePointer, lda, x.DevicePointer, incx, beta.DevicePointer, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSsbmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric banded matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n symmetric matrix with k subdiagonals and superdiagonals, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="k">number of sub- and super-diagonals of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Sbmv(FillMode uplo, int k, double alpha, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> x, int incx, double beta, CudaDeviceVariable<double> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasDsbmv_v2(_blasHandle, uplo, x.Size, k, ref alpha, A.DevicePointer, lda, x.DevicePointer, incx, ref beta, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDsbmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric banded matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n symmetric matrix with k subdiagonals and superdiagonals, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="k">number of sub- and super-diagonals of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Sbmv(FillMode uplo, int k, CudaDeviceVariable<double> alpha, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> beta, CudaDeviceVariable<double> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasDsbmv_v2(_blasHandle, uplo, x.Size, k, alpha.DevicePointer, A.DevicePointer, lda, x.DevicePointer, incx, beta.DevicePointer, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDsbmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}


		/// <summary>
		/// This function performs the Hermitian banded matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n Hermitian matrix with k subdiagonals and superdiagonals, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="k">number of sub- and super-diagonals of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Hbmv(FillMode uplo, int k, cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> x, int incx, cuFloatComplex beta, CudaDeviceVariable<cuFloatComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasChbmv_v2(_blasHandle, uplo, x.Size, k, ref alpha, A.DevicePointer, lda, x.DevicePointer, incx, ref beta, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasChbmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric banded matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n symmetric matrix with k subdiagonals and superdiagonals, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="k">number of sub- and super-diagonals of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Hbmv(FillMode uplo, int k, CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> beta, CudaDeviceVariable<cuFloatComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasChbmv_v2(_blasHandle, uplo, x.Size, k, alpha.DevicePointer, A.DevicePointer, lda, x.DevicePointer, incx, beta.DevicePointer, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasChbmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric banded matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n symmetric matrix with k subdiagonals and superdiagonals, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="k">number of sub- and super-diagonals of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Hbmv(FillMode uplo, int k, cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> x, int incx, cuDoubleComplex beta, CudaDeviceVariable<cuDoubleComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasZhbmv_v2(_blasHandle, uplo, x.Size, k, ref alpha, A.DevicePointer, lda, x.DevicePointer, incx, ref beta, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZhbmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric banded matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n symmetric matrix with k subdiagonals and superdiagonals, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="k">number of sub- and super-diagonals of matrix A.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Hbmv(FillMode uplo, int k, CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> beta, CudaDeviceVariable<cuDoubleComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasZhbmv_v2(_blasHandle, uplo, x.Size, k, alpha.DevicePointer, A.DevicePointer, lda, x.DevicePointer, incx, beta.DevicePointer, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZhbmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region SPMV/HPMV
		/// <summary>
		/// This function performs the symmetric packed matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n symmetric matrix stored in packed format, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="AP">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Spmv(FillMode uplo, float alpha, CudaDeviceVariable<float> AP, CudaDeviceVariable<float> x, int incx, float beta, CudaDeviceVariable<float> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasSspmv_v2(_blasHandle, uplo, x.Size, ref alpha, AP.DevicePointer, x.DevicePointer, incx, ref beta, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSspmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric packed matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n symmetric matrix stored in packed format, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="AP">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Spmv(FillMode uplo, CudaDeviceVariable<float> alpha, CudaDeviceVariable<float> AP, CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> beta, CudaDeviceVariable<float> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasSspmv_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, AP.DevicePointer, x.DevicePointer, incx, beta.DevicePointer, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSspmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric packed matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n symmetric matrix stored in packed format, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="AP">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Spmv(FillMode uplo, double alpha, CudaDeviceVariable<double> AP, CudaDeviceVariable<double> x, int incx, double beta, CudaDeviceVariable<double> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasDspmv_v2(_blasHandle, uplo, x.Size, ref alpha, AP.DevicePointer, x.DevicePointer, incx, ref beta, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDspmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric packed matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n symmetric matrix stored in packed format, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="AP">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Spmv(FillMode uplo, CudaDeviceVariable<double> alpha, CudaDeviceVariable<double> AP, CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> beta, CudaDeviceVariable<double> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasDspmv_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, AP.DevicePointer, x.DevicePointer, incx, beta.DevicePointer, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDspmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}


		/// <summary>
		/// This function performs the Hermitian packed matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n Hermitian matrix stored in packed format, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="AP">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Hpmv(FillMode uplo, cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> AP, CudaDeviceVariable<cuFloatComplex> x, int incx, cuFloatComplex beta, CudaDeviceVariable<cuFloatComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasChpmv_v2(_blasHandle, uplo, x.Size, ref alpha, AP.DevicePointer, x.DevicePointer, incx, ref beta, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasChpmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the Hermitian packed matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n Hermitian matrix stored in packed format, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="AP">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Hpmv(FillMode uplo, CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> AP, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> beta, CudaDeviceVariable<cuFloatComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasChpmv_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, AP.DevicePointer, x.DevicePointer, incx, beta.DevicePointer, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasChpmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the Hermitian packed matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n Hermitian matrix stored in packed format, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="AP">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Hpmv(FillMode uplo, cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> AP, CudaDeviceVariable<cuDoubleComplex> x, int incx, cuDoubleComplex beta, CudaDeviceVariable<cuDoubleComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasZhpmv_v2(_blasHandle, uplo, x.Size, ref alpha, AP.DevicePointer, x.DevicePointer, incx, ref beta, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZhpmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the Hermitian packed matrix-vector multiplication y = alpha * A * x + beta * y where A is a n*n Hermitian matrix stored in packed format, 
		/// x and y are vectors, and alpha and beta are scalars. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="AP">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0 then y does not have to be a valid input.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		public void Hpmv(FillMode uplo, CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> AP, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> beta, CudaDeviceVariable<cuDoubleComplex> y, int incy)
		{
			_status = CudaBlasNativeMethods.cublasZhpmv_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, AP.DevicePointer, x.DevicePointer, incx, beta.DevicePointer, y.DevicePointer, incy);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZhpmv_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region GER
		/// <summary>
		/// This function performs the rank-1 update A = alpha * x * y^T + A where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha is a scalar. m = x.Size, n = y.Size.
		/// </summary>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Ger(float alpha, CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> y, int incy, CudaDeviceVariable<float> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasSger_v2(_blasHandle, x.Size, y.Size, ref alpha, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSger_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the rank-1 update A = alpha * x * y^T + A where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha is a scalar. m = x.Size, n = y.Size.
		/// </summary>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Ger(CudaDeviceVariable<float> alpha, CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> y, int incy, CudaDeviceVariable<float> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasSger_v2(_blasHandle, x.Size, y.Size, alpha.DevicePointer, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSger_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the rank-1 update A = alpha * x * y^T + A where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha is a scalar. m = x.Size, n = y.Size.
		/// </summary>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Ger(double alpha, CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> y, int incy, CudaDeviceVariable<double> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasDger_v2(_blasHandle, x.Size, y.Size, ref alpha, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDger_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the rank-1 update A = alpha * x * y^T + A where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha is a scalar. m = x.Size, n = y.Size.
		/// </summary>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Ger(CudaDeviceVariable<double> alpha, CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> y, int incy, CudaDeviceVariable<double> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasDger_v2(_blasHandle, x.Size, y.Size, alpha.DevicePointer, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDger_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the rank-1 update A = alpha * x * y^T + A where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha is a scalar. m = x.Size, n = y.Size.
		/// </summary>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void GerU(cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy, CudaDeviceVariable<cuFloatComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasCgeru_v2(_blasHandle, x.Size, y.Size, ref alpha, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCgeru_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the rank-1 update A = alpha * x * y^T + A where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha is a scalar. m = x.Size, n = y.Size.
		/// </summary>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void GerU(CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy, CudaDeviceVariable<cuFloatComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasCgeru_v2(_blasHandle, x.Size, y.Size, alpha.DevicePointer, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCgeru_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the rank-1 update A = alpha * x * y^H + A where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha is a scalar. m = x.Size, n = y.Size.
		/// </summary>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void GerC(cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy, CudaDeviceVariable<cuFloatComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasCgerc_v2(_blasHandle, x.Size, y.Size, ref alpha, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCgerc_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the rank-1 update A = alpha * x * y^H + A where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha is a scalar. m = x.Size, n = y.Size.
		/// </summary>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void GerC(CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy, CudaDeviceVariable<cuFloatComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasCgerc_v2(_blasHandle, x.Size, y.Size, alpha.DevicePointer, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCgerc_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}


		/// <summary>
		/// This function performs the rank-1 update A = alpha * x * y^T + A where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha is a scalar. m = x.Size, n = y.Size.
		/// </summary>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void GerU(cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy, CudaDeviceVariable<cuDoubleComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasZgeru_v2(_blasHandle, x.Size, y.Size, ref alpha, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZgeru_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the rank-1 update A = alpha * x * y^T + A where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha is a scalar. m = x.Size, n = y.Size.
		/// </summary>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void GerU(CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy, CudaDeviceVariable<cuDoubleComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasZgeru_v2(_blasHandle, x.Size, y.Size, alpha.DevicePointer, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZgeru_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the rank-1 update A = alpha * x * y^H + A where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha is a scalar. m = x.Size, n = y.Size.
		/// </summary>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void GerC(cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy, CudaDeviceVariable<cuDoubleComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasZgerc_v2(_blasHandle, x.Size, y.Size, ref alpha, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZgerc_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the rank-1 update A = alpha * x * y^H + A where A is a m*n matrix stored in column-major format, 
		/// x and y are vectors, and alpha is a scalar. m = x.Size, n = y.Size.
		/// </summary>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void GerC(CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy, CudaDeviceVariable<cuDoubleComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasZgerc_v2(_blasHandle, x.Size, y.Size, alpha.DevicePointer, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZgerc_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region SYR/HER
		/// <summary>
		/// This function performs the symmetric rank-1 update A = alpha * x * x^T + A where A is a n*n symmetric Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Syr(FillMode uplo, float alpha, CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasSsyr_v2(_blasHandle, uplo, x.Size, ref alpha, x.DevicePointer, incx, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSsyr_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric rank-1 update A = alpha * x * x^T + A where A is a n*n symmetric Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Syr(FillMode uplo, CudaDeviceVariable<float> alpha, CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasSsyr_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, x.DevicePointer, incx, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSsyr_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the symmetric rank-1 update A = alpha * x * x^T + A where A is a n*n symmetric Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Syr(FillMode uplo, double alpha, CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasDsyr_v2(_blasHandle, uplo, x.Size, ref alpha, x.DevicePointer, incx, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDsyr_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric rank-1 update A = alpha * x * x^T + A where A is a n*n symmetric Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Syr(FillMode uplo, CudaDeviceVariable<double> alpha, CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasDsyr_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, x.DevicePointer, incx, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDsyr_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the symmetric rank-1 update A = alpha * x * x^T + A where A is a n*n symmetric Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Syr(FillMode uplo, cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasCsyr_v2(_blasHandle, uplo, x.Size, ref alpha, x.DevicePointer, incx, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCsyr_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric rank-1 update A = alpha * x * x^T + A where A is a n*n symmetric Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Syr(FillMode uplo, CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasCsyr_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, x.DevicePointer, incx, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCsyr_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the symmetric rank-1 update A = alpha * x * x^T + A where A is a n*n symmetric Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Syr(FillMode uplo, cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasZsyr_v2(_blasHandle, uplo, x.Size, ref alpha, x.DevicePointer, incx, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZsyr_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric rank-1 update A = alpha * x * x^T + A where A is a n*n symmetric Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Syr(FillMode uplo, CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasZsyr_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, x.DevicePointer, incx, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZsyr_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}


		/// <summary>
		/// This function performs the Hermitian rank-1 update A = alpha * x * x^H + A where A is a n*n Hermitian Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Her(FillMode uplo, float alpha, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasCher_v2(_blasHandle, uplo, x.Size, ref alpha, x.DevicePointer, incx, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCher_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the Hermitian rank-1 update A = alpha * x * x^H + A where A is a n*n Hermitian Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Her(FillMode uplo, CudaDeviceVariable<float> alpha, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasCher_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, x.DevicePointer, incx, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCher_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the Hermitian rank-1 update A = alpha * x * x^H + A where A is a n*n Hermitian Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Her(FillMode uplo, double alpha, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasZher_v2(_blasHandle, uplo, x.Size, ref alpha, x.DevicePointer, incx, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZher_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the Hermitian rank-1 update A = alpha * x * x^H + A where A is a n*n Hermitian Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Her(FillMode uplo, CudaDeviceVariable<double> alpha, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasZher_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, x.DevicePointer, incx, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZher_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		#endregion
		#region SPR/HPR
		/// <summary>
		/// This function performs the symmetric rank-1 update A = alpha * x * x^T + A where A is a n*n symmetric Matrix stored in packed format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		public void Spr(FillMode uplo, float alpha, CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> AP)
		{
			_status = CudaBlasNativeMethods.cublasSspr_v2(_blasHandle, uplo, x.Size, ref alpha, x.DevicePointer, incx, AP.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSspr_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric rank-1 update A = alpha * x * x^T + A where A is a n*n symmetric Matrix stored in packed format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		public void Spr(FillMode uplo, CudaDeviceVariable<float> alpha, CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> AP)
		{
			_status = CudaBlasNativeMethods.cublasSspr_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, x.DevicePointer, incx, AP.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSspr_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the symmetric rank-1 update A = alpha * x * x^T + A where A is a n*n symmetric Matrix stored in packed format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		public void Spr(FillMode uplo, double alpha, CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> AP)
		{
			_status = CudaBlasNativeMethods.cublasDspr_v2(_blasHandle, uplo, x.Size, ref alpha, x.DevicePointer, incx, AP.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDspr_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric rank-1 update A = alpha * x * x^T + A where A is a n*n symmetric Matrix stored in packed format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		public void Spr(FillMode uplo, CudaDeviceVariable<double> alpha, CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> AP)
		{
			_status = CudaBlasNativeMethods.cublasDspr_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, x.DevicePointer, incx, AP.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDspr_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}


		/// <summary>
		/// This function performs the Hermitian rank-1 update A = alpha * x * x^H + A where A is a n*n Hermitian Matrix stored in packed format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		public void Hpr(FillMode uplo, float alpha, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> AP)
		{
			_status = CudaBlasNativeMethods.cublasChpr_v2(_blasHandle, uplo, x.Size, ref alpha, x.DevicePointer, incx, AP.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasChpr_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the Hermitian rank-1 update A = alpha * x * x^H + A where A is a n*n Hermitian Matrix stored in packed format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		public void Hpr(FillMode uplo, CudaDeviceVariable<float> alpha, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> AP)
		{
			_status = CudaBlasNativeMethods.cublasChpr_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, x.DevicePointer, incx, AP.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasChpr_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the Hermitian rank-1 update A = alpha * x * x^H + A where A is a n*n Hermitian Matrix stored in packed format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		public void Hpr(FillMode uplo, double alpha, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> AP)
		{
			_status = CudaBlasNativeMethods.cublasZhpr_v2(_blasHandle, uplo, x.Size, ref alpha, x.DevicePointer, incx, AP.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZhpr_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the Hermitian rank-1 update A = alpha * x * x^H + A where A is a n*n Hermitian Matrix stored in packed format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		public void Hpr(FillMode uplo, CudaDeviceVariable<double> alpha, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> AP)
		{
			_status = CudaBlasNativeMethods.cublasZhpr_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, x.DevicePointer, incx, AP.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZhpr_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region SYR2/HER2
		/// <summary>
		/// This function performs the symmetric rank-2 update A = alpha * (x * y^T + y * y^T) + A where A is a n*n symmetric Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size = y.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Syr2(FillMode uplo, float alpha, CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> y, int incy, CudaDeviceVariable<float> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasSsyr2_v2(_blasHandle, uplo, x.Size, ref alpha, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSsyr2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric rank-2 update A = alpha * (x * y^T + y * y^T) + A where A is a n*n symmetric Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size = y.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Syr2(FillMode uplo, CudaDeviceVariable<float> alpha, CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> y, int incy, CudaDeviceVariable<float> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasSsyr2_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSsyr2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the symmetric rank-2 update A = alpha * (x * y^T + y * y^T) + A where A is a n*n symmetric Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size = y.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Syr2(FillMode uplo, double alpha, CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> y, int incy, CudaDeviceVariable<double> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasDsyr2_v2(_blasHandle, uplo, x.Size, ref alpha, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDsyr2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric rank-2 update A = alpha * (x * y^T + y * y^T) + A where A is a n*n symmetric Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size = y.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Syr2(FillMode uplo, CudaDeviceVariable<double> alpha, CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> y, int incy, CudaDeviceVariable<double> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasDsyr2_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDsyr2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the symmetric rank-2 update A = alpha * (x * y^T + y * y^T) + A where A is a n*n symmetric Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size = y.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Syr2(FillMode uplo, cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy, CudaDeviceVariable<cuFloatComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasCsyr2_v2(_blasHandle, uplo, x.Size, ref alpha, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCsyr2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric rank-2 update A = alpha * (x * y^T + y * y^T) + A where A is a n*n symmetric Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size = y.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Syr2(FillMode uplo, CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy, CudaDeviceVariable<cuFloatComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasCsyr2_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCsyr2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the symmetric rank-2 update A = alpha * (x * y^T + y * y^T) + A where A is a n*n symmetric Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size = y.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Syr2(FillMode uplo, cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy, CudaDeviceVariable<cuDoubleComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasZsyr2_v2(_blasHandle, uplo, x.Size, ref alpha, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZsyr2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric rank-2 update A = alpha * (x * y^T + y * y^T) + A where A is a n*n symmetric Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size = y.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Syr2(FillMode uplo, CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy, CudaDeviceVariable<cuDoubleComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasZsyr2_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZsyr2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}


		/// <summary>
		/// This function performs the symmetric rank-2 update A = alpha * (x * y^T + y * y^T) + A where A is a n*n symmetric Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size = y.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Her2(FillMode uplo, cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy, CudaDeviceVariable<cuFloatComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasCher2_v2(_blasHandle, uplo, x.Size, ref alpha, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCher2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric rank-2 update A = alpha * (x * y^T + y * y^T) + A where A is a n*n symmetric Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size = y.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Her2(FillMode uplo, CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy, CudaDeviceVariable<cuFloatComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasCher2_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCher2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the symmetric rank-2 update A = alpha * (x * y^T + y * y^T) + A where A is a n*n symmetric Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size = y.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Her2(FillMode uplo, cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy, CudaDeviceVariable<cuDoubleComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasZher2_v2(_blasHandle, uplo, x.Size, ref alpha, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZher2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric rank-2 update A = alpha * (x * y^T + y * y^T) + A where A is a n*n symmetric Matrix stored in column-major format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size = y.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of y.</param>
		/// <param name="A">array of dimensions lda * n, with lda >= max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Her2(FillMode uplo, CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy, CudaDeviceVariable<cuDoubleComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasZher2_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, x.DevicePointer, incx, y.DevicePointer, incy, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZher2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region SPR2/HPR2
		/// <summary>
		/// This function performs the packed symmetric rank-2 update A = alpha * (x * y^T + y * x^T) + A where A is a n*n symmetric Matrix stored in packed format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size = y.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of x.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		public void Spr2(FillMode uplo, float alpha, CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> y, int incy, CudaDeviceVariable<float> AP)
		{
			_status = CudaBlasNativeMethods.cublasSspr2_v2(_blasHandle, uplo, x.Size, ref alpha, x.DevicePointer, incx, y.DevicePointer, incy, AP.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSspr2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the packed symmetric rank-2 update A = alpha * (x * y^T + y * x^T) + A where A is a n*n symmetric Matrix stored in packed format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size = y.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of x.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		public void Spr2(FillMode uplo, CudaDeviceVariable<float> alpha, CudaDeviceVariable<float> x, int incx, CudaDeviceVariable<float> y, int incy, CudaDeviceVariable<float> AP)
		{
			_status = CudaBlasNativeMethods.cublasSspr2_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, x.DevicePointer, incx, y.DevicePointer, incy, AP.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSspr2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the packed symmetric rank-2 update A = alpha * (x * y^T + y * x^T) + A where A is a n*n symmetric Matrix stored in packed format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size = y.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of x.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		public void Spr2(FillMode uplo, double alpha, CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> y, int incy, CudaDeviceVariable<double> AP)
		{
			_status = CudaBlasNativeMethods.cublasDspr2_v2(_blasHandle, uplo, x.Size, ref alpha, x.DevicePointer, incx, y.DevicePointer, incy, AP.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDspr2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the packed symmetric rank-2 update A = alpha * (x * y^T + y * x^T) + A where A is a n*n symmetric Matrix stored in packed format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size = y.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of x.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		public void Spr2(FillMode uplo, CudaDeviceVariable<double> alpha, CudaDeviceVariable<double> x, int incx, CudaDeviceVariable<double> y, int incy, CudaDeviceVariable<double> AP)
		{
			_status = CudaBlasNativeMethods.cublasDspr2_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, x.DevicePointer, incx, y.DevicePointer, incy, AP.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDspr2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}


		/// <summary>
		/// This function performs the packed Hermitian rank-2 update A = alpha * (x * y^H + y * x^H) + A where A is a n*n Hermitian Matrix stored in packed format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of x.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		public void Hpr2(FillMode uplo, cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy, CudaDeviceVariable<cuFloatComplex> AP)
		{
			_status = CudaBlasNativeMethods.cublasChpr2_v2(_blasHandle, uplo, x.Size, ref alpha, x.DevicePointer, incx, y.DevicePointer, incy, AP.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasChpr2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the packed Hermitian rank-2 update A = alpha * (x * y^H + y * x^H) + A where A is a n*n Hermitian Matrix stored in packed format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of x.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		public void Hpr2(FillMode uplo, CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> x, int incx, CudaDeviceVariable<cuFloatComplex> y, int incy, CudaDeviceVariable<cuFloatComplex> AP)
		{
			_status = CudaBlasNativeMethods.cublasChpr2_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, x.DevicePointer, incx, y.DevicePointer, incy, AP.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasChpr2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the packed Hermitian rank-2 update A = alpha * (x * y^H + y * x^H) + A where A is a n*n Hermitian Matrix stored in packed format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of x.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		public void Hpr2(FillMode uplo, cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy, CudaDeviceVariable<cuDoubleComplex> AP)
		{
			_status = CudaBlasNativeMethods.cublasZhpr2_v2(_blasHandle, uplo, x.Size, ref alpha, x.DevicePointer, incx, y.DevicePointer, incy, AP.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZhpr2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the packed Hermitian rank-2 update A = alpha * (x * y^H + y * x^H) + A where A is a n*n Hermitian Matrix stored in packed format,
		/// x is a vector, and alpha is a scalar. n is given by x.Size.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="x">vector with n elements.</param>
		/// <param name="incx">stride between consecutive elements of x.</param>
		/// <param name="y">vector with n elements.</param>
		/// <param name="incy">stride between consecutive elements of x.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		public void Hpr2(FillMode uplo, CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> x, int incx, CudaDeviceVariable<cuDoubleComplex> y, int incy, CudaDeviceVariable<cuDoubleComplex> AP)
		{
			_status = CudaBlasNativeMethods.cublasZhpr2_v2(_blasHandle, uplo, x.Size, alpha.DevicePointer, x.DevicePointer, incx, y.DevicePointer, incy, AP.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZhpr2_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#endregion

		#region BLAS3
		#region GEMM
		/// <summary>
		/// This function performs the matrix-matrix multiplication C = alpha * Op(A) * Op(B) + beta * C where 
		/// alpha and beta are scalars, and A, B and C are matrices stored in column-major format with dimensions 
		/// op(A) m*k, op(B) k*n and C m*n, respectively.
		/// </summary>
		/// <param name="transa">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A) and C.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Gemm(Operation transa, Operation transb, int m, int n, int k, float alpha, CudaDeviceVariable<float> A, int lda,
			CudaDeviceVariable<float> B, int ldb, float beta, CudaDeviceVariable<float> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasSgemm_v2(_blasHandle, transa, transb, m, n, k, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSgemm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-matrix multiplication C = alpha * Op(A) * Op(B) + beta * C where 
		/// alpha and beta are scalars, and A, B and C are matrices stored in column-major format with dimensions 
		/// op(A) m*k, op(B) k*n and C m*n, respectively.
		/// </summary>
		/// <param name="transa">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A) and C.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Gemm(Operation transa, Operation transb, int m, int n, int k, CudaDeviceVariable<float> alpha, CudaDeviceVariable<float> A, int lda,
			CudaDeviceVariable<float> B, int ldb, CudaDeviceVariable<float> beta, CudaDeviceVariable<float> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasSgemm_v2(_blasHandle, transa, transb, m, n, k, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSgemm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the matrix-matrix multiplication C = alpha * Op(A) * Op(B) + beta * C where 
		/// alpha and beta are scalars, and A, B and C are matrices stored in column-major format with dimensions 
		/// op(A) m*k, op(B) k*n and C m*n, respectively.
		/// </summary>
		/// <param name="transa">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A) and C.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Gemm(Operation transa, Operation transb, int m, int n, int k, double alpha, CudaDeviceVariable<double> A, int lda,
			CudaDeviceVariable<double> B, int ldb, double beta, CudaDeviceVariable<double> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasDgemm_v2(_blasHandle, transa, transb, m, n, k, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDgemm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-matrix multiplication C = alpha * Op(A) * Op(B) + beta * C where 
		/// alpha and beta are scalars, and A, B and C are matrices stored in column-major format with dimensions 
		/// op(A) m*k, op(B) k*n and C m*n, respectively.
		/// </summary>
		/// <param name="transa">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A) and C.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Gemm(Operation transa, Operation transb, int m, int n, int k, CudaDeviceVariable<double> alpha, CudaDeviceVariable<double> A, int lda,
			CudaDeviceVariable<double> B, int ldb, CudaDeviceVariable<double> beta, CudaDeviceVariable<double> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasDgemm_v2(_blasHandle, transa, transb, m, n, k, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDgemm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-matrix multiplication C = alpha * Op(A) * Op(B) + beta * C where 
		/// alpha and beta are scalars, and A, B and C are matrices stored in column-major format with dimensions 
		/// op(A) m*k, op(B) k*n and C m*n, respectively.
		/// </summary>
		/// <param name="transa">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A) and C.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Gemm(Operation transa, Operation transb, int m, int n, int k, cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> A, int lda,
			CudaDeviceVariable<cuFloatComplex> B, int ldb, cuFloatComplex beta, CudaDeviceVariable<cuFloatComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasCgemm_v2(_blasHandle, transa, transb, m, n, k, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCgemm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-matrix multiplication C = alpha * Op(A) * Op(B) + beta * C where 
		/// alpha and beta are scalars, and A, B and C are matrices stored in column-major format with dimensions 
		/// op(A) m*k, op(B) k*n and C m*n, respectively.
		/// </summary>
		/// <param name="transa">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A) and C.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Gemm(Operation transa, Operation transb, int m, int n, int k, CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> A, int lda,
			CudaDeviceVariable<cuFloatComplex> B, int ldb, CudaDeviceVariable<cuFloatComplex> beta, CudaDeviceVariable<cuFloatComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasCgemm_v2(_blasHandle, transa, transb, m, n, k, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCgemm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the matrix-matrix multiplication C = alpha * Op(A) * Op(B) + beta * C where 
		/// alpha and beta are scalars, and A, B and C are matrices stored in column-major format with dimensions 
		/// op(A) m*k, op(B) k*n and C m*n, respectively.
		/// </summary>
		/// <param name="transa">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A) and C.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Gemm(Operation transa, Operation transb, int m, int n, int k, cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda,
			CudaDeviceVariable<cuDoubleComplex> B, int ldb, cuDoubleComplex beta, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasZgemm_v2(_blasHandle, transa, transb, m, n, k, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZgemm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-matrix multiplication C = alpha * Op(A) * Op(B) + beta * C where 
		/// alpha and beta are scalars, and A, B and C are matrices stored in column-major format with dimensions 
		/// op(A) m*k, op(B) k*n and C m*n, respectively.
		/// </summary>
		/// <param name="transa">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A) and C.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Gemm(Operation transa, Operation transb, int m, int n, int k, CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda,
			CudaDeviceVariable<cuDoubleComplex> B, int ldb, CudaDeviceVariable<cuDoubleComplex> beta, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasZgemm_v2(_blasHandle, transa, transb, m, n, k, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZgemm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the matrix-matrix multiplication C = alpha * Op(A) * Op(B) + beta * C where 
		/// alpha and beta are scalars, and A, B and C are matrices stored in column-major format with dimensions 
		/// op(A) m*k, op(B) k*n and C m*n, respectively.
		/// </summary>
		/// <param name="transa">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A) and C.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Gemm(Operation transa, Operation transb, int m, int n, int k, half alpha, CudaDeviceVariable<half> A, int lda,
			CudaDeviceVariable<half> B, int ldb, half beta, CudaDeviceVariable<half> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasHgemm(_blasHandle, transa, transb, m, n, k, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasHgemm", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-matrix multiplication C = alpha * Op(A) * Op(B) + beta * C where 
		/// alpha and beta are scalars, and A, B and C are matrices stored in column-major format with dimensions 
		/// op(A) m*k, op(B) k*n and C m*n, respectively.
		/// </summary>
		/// <param name="transa">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A) and C.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Gemm(Operation transa, Operation transb, int m, int n, int k, CudaDeviceVariable<half> alpha, CudaDeviceVariable<half> A, int lda,
			CudaDeviceVariable<half> B, int ldb, CudaDeviceVariable<half> beta, CudaDeviceVariable<half> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasHgemm(_blasHandle, transa, transb, m, n, k, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasHgemm", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-matrix multiplication C = alpha * Op(A) * Op(B) + beta * C where 
		/// alpha and beta are scalars, and A, B and C are matrices stored in column-major format with dimensions 
		/// op(A) m*k, op(B) k*n and C m*n, respectively.
		/// </summary>
		/// <param name="transa">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A) and C.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="Atype">enumerant specifying the datatype of matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="Btype">enumerant specifying the datatype of matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		/// <param name="Ctype">enumerant specifying the datatype of matrix C.</param>
		public void GemmEx(Operation transa, Operation transb, int m, int n, int k, float alpha, CUdeviceptr A, DataType Atype, int lda,
			CUdeviceptr B, DataType Btype, int ldb, float beta, CUdeviceptr C, DataType Ctype, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasSgemmEx(_blasHandle, transa, transb, m, n, k, ref alpha, A, Atype, lda, B, Btype, ldb, ref beta, C, Ctype, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSgemmEx", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-matrix multiplication C = alpha * Op(A) * Op(B) + beta * C where 
		/// alpha and beta are scalars, and A, B and C are matrices stored in column-major format with dimensions 
		/// op(A) m*k, op(B) k*n and C m*n, respectively.
		/// </summary>
		/// <param name="transa">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A) and C.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="Atype">enumerant specifying the datatype of matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="Btype">enumerant specifying the datatype of matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		/// <param name="Ctype">enumerant specifying the datatype of matrix C.</param>
		public void GemmEx(Operation transa, Operation transb, int m, int n, int k, CudaDeviceVariable<float> alpha, CUdeviceptr A, DataType Atype, int lda,
			CUdeviceptr B, DataType Btype, int ldb, CudaDeviceVariable<float> beta, CUdeviceptr C, DataType Ctype, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasSgemmEx(_blasHandle, transa, transb, m, n, k, alpha.DevicePointer, A, Atype, lda, B, Btype, ldb, beta.DevicePointer, C, Ctype, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSgemmEx", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region SYRK
		/// <summary>
		/// This function performs the symmetric rank-k update C = alpha * Op(A)*Op(A)^T + beta * C where
		/// alpha and beta are scalars, and A, B and C are matrices stored in lower or upper mode, and A is a matrix with dimensions op(A) n*k.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syrk(FillMode uplo, Operation trans, int n, int k, float alpha, CudaDeviceVariable<float> A, int lda,
			float beta, CudaDeviceVariable<float> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasSsyrk_v2(_blasHandle, uplo, trans, n, k, ref alpha, A.DevicePointer, lda, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSsyrk_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric rank-k update C = alpha * Op(A)*Op(A)^T + beta * C where
		/// alpha and beta are scalars, and A, B and C are matrices stored in lower or upper mode, and A is a matrix with dimensions op(A) n*k.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syrk(FillMode uplo, Operation trans, int n, int k, CudaDeviceVariable<float> alpha, CudaDeviceVariable<float> A, int lda,
			CudaDeviceVariable<float> beta, CudaDeviceVariable<float> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasSsyrk_v2(_blasHandle, uplo, trans, n, k, alpha.DevicePointer, A.DevicePointer, lda, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSsyrk_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the symmetric rank-k update C = alpha * Op(A)*Op(A)^T + beta * C where
		/// alpha and beta are scalars, and A, B and C are matrices stored in lower or upper mode, and A is a matrix with dimensions op(A) n*k.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syrk(FillMode uplo, Operation trans, int n, int k, double alpha, CudaDeviceVariable<double> A, int lda,
			double beta, CudaDeviceVariable<double> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasDsyrk_v2(_blasHandle, uplo, trans, n, k, ref alpha, A.DevicePointer, lda, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDsyrk_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric rank-k update C = alpha * Op(A)*Op(A)^T + beta * C where
		/// alpha and beta are scalars, and A, B and C are matrices stored in lower or upper mode, and A is a matrix with dimensions op(A) n*k.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syrk(FillMode uplo, Operation trans, int n, int k, CudaDeviceVariable<double> alpha, CudaDeviceVariable<double> A, int lda,
			CudaDeviceVariable<double> beta, CudaDeviceVariable<double> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasDsyrk_v2(_blasHandle, uplo, trans, n, k, alpha.DevicePointer, A.DevicePointer, lda, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDsyrk_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}


		/// <summary>
		/// This function performs the symmetric rank-k update C = alpha * Op(A)*Op(A)^T + beta * C where
		/// alpha and beta are scalars, and A, B and C are matrices stored in lower or upper mode, and A is a matrix with dimensions op(A) n*k.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syrk(FillMode uplo, Operation trans, int n, int k, cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> A, int lda,
			cuFloatComplex beta, CudaDeviceVariable<cuFloatComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasCsyrk_v2(_blasHandle, uplo, trans, n, k, ref alpha, A.DevicePointer, lda, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCsyrk_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric rank-k update C = alpha * Op(A)*Op(A)^T + beta * C where
		/// alpha and beta are scalars, and A, B and C are matrices stored in lower or upper mode, and A is a matrix with dimensions op(A) n*k.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syrk(FillMode uplo, Operation trans, int n, int k, CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> A, int lda,
			CudaDeviceVariable<cuFloatComplex> beta, CudaDeviceVariable<cuFloatComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasCsyrk_v2(_blasHandle, uplo, trans, n, k, alpha.DevicePointer, A.DevicePointer, lda, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCsyrk_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the symmetric rank-k update C = alpha * Op(A)*Op(A)^T + beta * C where
		/// alpha and beta are scalars, and A, B and C are matrices stored in lower or upper mode, and A is a matrix with dimensions op(A) n*k.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syrk(FillMode uplo, Operation trans, int n, int k, cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda,
			cuDoubleComplex beta, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasZsyrk_v2(_blasHandle, uplo, trans, n, k, ref alpha, A.DevicePointer, lda, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZsyrk_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric rank-k update C = alpha * Op(A)*Op(A)^T + beta * C where
		/// alpha and beta are scalars, and A, B and C are matrices stored in lower or upper mode, and A is a matrix with dimensions op(A) n*k.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syrk(FillMode uplo, Operation trans, int n, int k, CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda,
			CudaDeviceVariable<cuDoubleComplex> beta, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasZsyrk_v2(_blasHandle, uplo, trans, n, k, alpha.DevicePointer, A.DevicePointer, lda, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZsyrk_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region HERK

		/// <summary>
		/// This function performs the Hermitian rank-k update C = alpha * Op(A)*Op(A)^H + beta * C where
		/// alpha and beta are scalars, and C is a Hermitian matrix stored in lower or upper mode, and A is a matrix with dimensions op(A) n*k.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Herk(FillMode uplo, Operation trans, int n, int k, float alpha, CudaDeviceVariable<cuFloatComplex> A, int lda,
			float beta, CudaDeviceVariable<cuFloatComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasCherk_v2(_blasHandle, uplo, trans, n, k, ref alpha, A.DevicePointer, lda, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCherk_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the Hermitian rank-k update C = alpha * Op(A)*Op(A)^H + beta * C where
		/// alpha and beta are scalars, and C is a Hermitian matrix stored in lower or upper mode, and A is a matrix with dimensions op(A) n*k.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Herk(FillMode uplo, Operation trans, int n, int k, CudaDeviceVariable<float> alpha, CudaDeviceVariable<cuFloatComplex> A, int lda,
			CudaDeviceVariable<float> beta, CudaDeviceVariable<cuFloatComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasCherk_v2(_blasHandle, uplo, trans, n, k, alpha.DevicePointer, A.DevicePointer, lda, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCherk_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the Hermitian rank-k update C = alpha * Op(A)*Op(A)^H + beta * C where
		/// alpha and beta are scalars, and C is a Hermitian matrix stored in lower or upper mode, and A is a matrix with dimensions op(A) n*k.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Herk(FillMode uplo, Operation trans, int n, int k, double alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda,
			double beta, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasZherk_v2(_blasHandle, uplo, trans, n, k, ref alpha, A.DevicePointer, lda, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZherk_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the Hermitian rank-k update C = alpha * Op(A)*Op(A)^H + beta * C where
		/// alpha and beta are scalars, and C is a Hermitian matrix stored in lower or upper mode, and A is a matrix with dimensions op(A) n*k.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Herk(FillMode uplo, Operation trans, int n, int k, CudaDeviceVariable<double> alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda,
			CudaDeviceVariable<double> beta, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasZherk_v2(_blasHandle, uplo, trans, n, k, alpha.DevicePointer, A.DevicePointer, lda, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZherk_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region SYR2K
		/// <summary>
		/// This function performs the symmetric rank-k update C = alpha * (Op(A)*Op(B)^T + Op(B)*Op(A)^T) + beta * C where
		/// alpha and beta are scalars, and C is a symmetrux matrix stored in lower or upper mode, and A and B are matrices with dimensions Op(A) n*k
		/// and Op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * k.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syr2k(FillMode uplo, Operation trans, int n, int k, float alpha, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> B, int ldb,
			float beta, CudaDeviceVariable<float> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasSsyr2k_v2(_blasHandle, uplo, trans, n, k, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSsyr2k_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric rank-k update C = alpha * (Op(A)*Op(B)^T + Op(B)*Op(A)^T) + beta * C where
		/// alpha and beta are scalars, and C is a symmetrux matrix stored in lower or upper mode, and A and B are matrices with dimensions Op(A) n*k
		/// and Op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * k.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syr2k(FillMode uplo, Operation trans, int n, int k, CudaDeviceVariable<float> alpha, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> B, int ldb,
			CudaDeviceVariable<float> beta, CudaDeviceVariable<float> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasSsyr2k_v2(_blasHandle, uplo, trans, n, k, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSsyr2k_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the symmetric rank-k update C = alpha * (Op(A)*Op(B)^T + Op(B)*Op(A)^T) + beta * C where
		/// alpha and beta are scalars, and C is a symmetrux matrix stored in lower or upper mode, and A and B are matrices with dimensions Op(A) n*k
		/// and Op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * k.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syr2k(FillMode uplo, Operation trans, int n, int k, double alpha, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> B, int ldb,
			double beta, CudaDeviceVariable<double> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasDsyr2k_v2(_blasHandle, uplo, trans, n, k, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDsyr2k_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric rank-k update C = alpha * (Op(A)*Op(B)^T + Op(B)*Op(A)^T) + beta * C where
		/// alpha and beta are scalars, and C is a symmetrux matrix stored in lower or upper mode, and A and B are matrices with dimensions Op(A) n*k
		/// and Op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * k.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syr2k(FillMode uplo, Operation trans, int n, int k, CudaDeviceVariable<double> alpha, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> B, int ldb,
			CudaDeviceVariable<double> beta, CudaDeviceVariable<double> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasDsyr2k_v2(_blasHandle, uplo, trans, n, k, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDsyr2k_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}


		/// <summary>
		/// This function performs the symmetric rank-k update C = alpha * (Op(A)*Op(B)^T + Op(B)*Op(A)^T) + beta * C where
		/// alpha and beta are scalars, and C is a symmetrux matrix stored in lower or upper mode, and A and B are matrices with dimensions Op(A) n*k
		/// and Op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * k.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syr2k(FillMode uplo, Operation trans, int n, int k, cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> B, int ldb,
			cuFloatComplex beta, CudaDeviceVariable<cuFloatComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasCsyr2k_v2(_blasHandle, uplo, trans, n, k, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCsyr2k_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric rank-k update C = alpha * (Op(A)*Op(B)^T + Op(B)*Op(A)^T) + beta * C where
		/// alpha and beta are scalars, and C is a symmetrux matrix stored in lower or upper mode, and A and B are matrices with dimensions Op(A) n*k
		/// and Op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * k.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syr2k(FillMode uplo, Operation trans, int n, int k, CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> B, int ldb,
			CudaDeviceVariable<cuFloatComplex> beta, CudaDeviceVariable<cuFloatComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasCsyr2k_v2(_blasHandle, uplo, trans, n, k, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCsyr2k_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the symmetric rank-k update C = alpha * (Op(A)*Op(B)^T + Op(B)*Op(A)^T) + beta * C where
		/// alpha and beta are scalars, and C is a symmetrux matrix stored in lower or upper mode, and A and B are matrices with dimensions Op(A) n*k
		/// and Op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * k.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syr2k(FillMode uplo, Operation trans, int n, int k, cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> B, int ldb,
			cuDoubleComplex beta, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasZsyr2k_v2(_blasHandle, uplo, trans, n, k, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZsyr2k_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric rank-k update C = alpha * (Op(A)*Op(B)^T + Op(B)*Op(A)^T) + beta * C where
		/// alpha and beta are scalars, and C is a symmetrux matrix stored in lower or upper mode, and A and B are matrices with dimensions Op(A) n*k
		/// and Op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * k.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syr2k(FillMode uplo, Operation trans, int n, int k, CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> B, int ldb,
			CudaDeviceVariable<cuDoubleComplex> beta, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasZsyr2k_v2(_blasHandle, uplo, trans, n, k, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZsyr2k_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region HER2K
		/// <summary>
		/// This function performs the Hermitian rank-k update C = alpha * (Op(A)*Op(B)^H + Op(B)*Op(A)^H) + beta * C where
		/// alpha and beta are scalars, and C is a Hermitian matrix stored in lower or upper mode, and A is a matrix with dimensions op(A) n*k and Op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * k.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Her2k(FillMode uplo, Operation trans, int n, int k, cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> B, int ldb,
			float beta, CudaDeviceVariable<cuFloatComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasCher2k_v2(_blasHandle, uplo, trans, n, k, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCher2k_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the Hermitian rank-k update C = alpha * (Op(A)*Op(B)^H + Op(B)*Op(A)^H) + beta * C where
		/// alpha and beta are scalars, and C is a Hermitian matrix stored in lower or upper mode, and A is a matrix with dimensions op(A) n*k and Op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * k.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Her2k(FillMode uplo, Operation trans, int n, int k, CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> B, int ldb,
			CudaDeviceVariable<float> beta, CudaDeviceVariable<cuFloatComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasCher2k_v2(_blasHandle, uplo, trans, n, k, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCher2k_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the Hermitian rank-k update C = alpha * (Op(A)*Op(B)^H + Op(B)*Op(A)^H) + beta * C where
		/// alpha and beta are scalars, and C is a Hermitian matrix stored in lower or upper mode, and A is a matrix with dimensions op(A) n*k and Op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * k.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Her2k(FillMode uplo, Operation trans, int n, int k, cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> B, int ldb,
			double beta, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasZher2k_v2(_blasHandle, uplo, trans, n, k, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZher2k_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the Hermitian rank-k update C = alpha * (Op(A)*Op(B)^H + Op(B)*Op(A)^H) + beta * C where
		/// alpha and beta are scalars, and C is a Hermitian matrix stored in lower or upper mode, and A is a matrix with dimensions op(A) n*k and Op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="k">number of columns of op(A) and rows of op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * k.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Her2k(FillMode uplo, Operation trans, int n, int k, CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> B, int ldb,
			CudaDeviceVariable<double> beta, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasZher2k_v2(_blasHandle, uplo, trans, n, k, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZher2k_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion

		#region SYRKX : eXtended SYRK
		/// <summary>
		/// This function performs a variation of the symmetric rank- update C = alpha * (Op(A)*Op(B))^T + beta * C where alpha 
		/// and beta are scalars, C is a symmetric matrix stored in lower or upper mode, and A
		/// and B are matrices with dimensions op(A) n*k and op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix C lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or transpose.</param>
		/// <param name="n">number of rows of matrix op(A), op(B) and C.</param>
		/// <param name="k">number of columns of matrix op(A) and op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimension lda x k with lda>=max(1,n) if transa == CUBLAS_OP_N and lda x n with lda>=max(1,k) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb x k with ldb>=max(1,n) if transa == CUBLAS_OP_N and ldb x n with ldb>=max(1,k) otherwise.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0, then C does not have to be a valid input.</param>
		/// <param name="C">array of dimensions ldc x n with ldc>=max(1,n).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syrkx(FillMode uplo, Operation trans, int n, int k, ref float alpha, CudaDeviceVariable<float> A, int lda,
                                                    CudaDeviceVariable<float> B, int ldb, ref float beta,  CudaDeviceVariable<float> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasSsyrkx(_blasHandle, uplo, trans, n, k, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSsyrkx", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
                   
		/// <summary>
		/// This function performs a variation of the symmetric rank- update C = alpha * (Op(A)*Op(B))^T + beta * C where alpha 
		/// and beta are scalars, C is a symmetric matrix stored in lower or upper mode, and A
		/// and B are matrices with dimensions op(A) n*k and op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix C lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or transpose.</param>
		/// <param name="n">number of rows of matrix op(A), op(B) and C.</param>
		/// <param name="k">number of columns of matrix op(A) and op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimension lda x k with lda>=max(1,n) if transa == CUBLAS_OP_N and lda x n with lda>=max(1,k) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb x k with ldb>=max(1,n) if transa == CUBLAS_OP_N and ldb x n with ldb>=max(1,k) otherwise.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0, then C does not have to be a valid input.</param>
		/// <param name="C">array of dimensions ldc x n with ldc>=max(1,n).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syrkx(FillMode uplo, Operation trans, int n, int k, ref double alpha, CudaDeviceVariable<double> A, int lda,
                                                    CudaDeviceVariable<double> B, int ldb, ref double beta,  CudaDeviceVariable<double> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasDsyrkx(_blasHandle, uplo, trans, n, k, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDsyrkx", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
                      
		/// <summary>
		/// This function performs a variation of the symmetric rank- update C = alpha * (Op(A)*Op(B))^T + beta * C where alpha 
		/// and beta are scalars, C is a symmetric matrix stored in lower or upper mode, and A
		/// and B are matrices with dimensions op(A) n*k and op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix C lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or transpose.</param>
		/// <param name="n">number of rows of matrix op(A), op(B) and C.</param>
		/// <param name="k">number of columns of matrix op(A) and op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimension lda x k with lda>=max(1,n) if transa == CUBLAS_OP_N and lda x n with lda>=max(1,k) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb x k with ldb>=max(1,n) if transa == CUBLAS_OP_N and ldb x n with ldb>=max(1,k) otherwise.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0, then C does not have to be a valid input.</param>
		/// <param name="C">array of dimensions ldc x n with ldc>=max(1,n).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syrkx(FillMode uplo, Operation trans, int n, int k, ref cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> A, int lda,
                                                    CudaDeviceVariable<cuFloatComplex> B, int ldb, ref cuFloatComplex beta, CudaDeviceVariable<cuFloatComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasCsyrkx(_blasHandle, uplo, trans, n, k, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCsyrkx", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
                     
		/// <summary>
		/// This function performs a variation of the symmetric rank- update C = alpha * (Op(A)*Op(B))^T + beta * C where alpha 
		/// and beta are scalars, C is a symmetric matrix stored in lower or upper mode, and A
		/// and B are matrices with dimensions op(A) n*k and op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix C lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or transpose.</param>
		/// <param name="n">number of rows of matrix op(A), op(B) and C.</param>
		/// <param name="k">number of columns of matrix op(A) and op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimension lda x k with lda>=max(1,n) if transa == CUBLAS_OP_N and lda x n with lda>=max(1,k) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb x k with ldb>=max(1,n) if transa == CUBLAS_OP_N and ldb x n with ldb>=max(1,k) otherwise.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0, then C does not have to be a valid input.</param>
		/// <param name="C">array of dimensions ldc x n with ldc>=max(1,n).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syrkx(FillMode uplo,  Operation trans, int n, int k, ref cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda,
                                                    CudaDeviceVariable<cuDoubleComplex> B, int ldb, ref cuDoubleComplex beta, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasZsyrkx(_blasHandle, uplo, trans, n, k, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZsyrkx", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}


		/// <summary>
		/// This function performs a variation of the symmetric rank- update C = alpha * (Op(A)*Op(B))^T + beta * C where alpha 
		/// and beta are scalars, C is a symmetric matrix stored in lower or upper mode, and A
		/// and B are matrices with dimensions op(A) n*k and op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix C lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or transpose.</param>
		/// <param name="n">number of rows of matrix op(A), op(B) and C.</param>
		/// <param name="k">number of columns of matrix op(A) and op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimension lda x k with lda>=max(1,n) if transa == CUBLAS_OP_N and lda x n with lda>=max(1,k) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb x k with ldb>=max(1,n) if transa == CUBLAS_OP_N and ldb x n with ldb>=max(1,k) otherwise.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0, then C does not have to be a valid input.</param>
		/// <param name="C">array of dimensions ldc x n with ldc>=max(1,n).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syrkx(FillMode uplo, Operation trans, int n, int k, CudaDeviceVariable<float> alpha, CudaDeviceVariable<float> A, int lda,
													CudaDeviceVariable<float> B, int ldb, CudaDeviceVariable<float> beta, CudaDeviceVariable<float> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasSsyrkx(_blasHandle, uplo, trans, n, k, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSsyrkx", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs a variation of the symmetric rank- update C = alpha * (Op(A)*Op(B))^T + beta * C where alpha 
		/// and beta are scalars, C is a symmetric matrix stored in lower or upper mode, and A
		/// and B are matrices with dimensions op(A) n*k and op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix C lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or transpose.</param>
		/// <param name="n">number of rows of matrix op(A), op(B) and C.</param>
		/// <param name="k">number of columns of matrix op(A) and op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimension lda x k with lda>=max(1,n) if transa == CUBLAS_OP_N and lda x n with lda>=max(1,k) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb x k with ldb>=max(1,n) if transa == CUBLAS_OP_N and ldb x n with ldb>=max(1,k) otherwise.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0, then C does not have to be a valid input.</param>
		/// <param name="C">array of dimensions ldc x n with ldc>=max(1,n).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syrkx(FillMode uplo, Operation trans, int n, int k, CudaDeviceVariable<double> alpha, CudaDeviceVariable<double> A, int lda,
													CudaDeviceVariable<double> B, int ldb, CudaDeviceVariable<double> beta, CudaDeviceVariable<double> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasDsyrkx(_blasHandle, uplo, trans, n, k, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDsyrkx", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs a variation of the symmetric rank- update C = alpha * (Op(A)*Op(B))^T + beta * C where alpha 
		/// and beta are scalars, C is a symmetric matrix stored in lower or upper mode, and A
		/// and B are matrices with dimensions op(A) n*k and op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix C lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or transpose.</param>
		/// <param name="n">number of rows of matrix op(A), op(B) and C.</param>
		/// <param name="k">number of columns of matrix op(A) and op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimension lda x k with lda>=max(1,n) if transa == CUBLAS_OP_N and lda x n with lda>=max(1,k) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb x k with ldb>=max(1,n) if transa == CUBLAS_OP_N and ldb x n with ldb>=max(1,k) otherwise.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0, then C does not have to be a valid input.</param>
		/// <param name="C">array of dimensions ldc x n with ldc>=max(1,n).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syrkx(FillMode uplo, Operation trans, int n, int k, CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> A, int lda,
													CudaDeviceVariable<cuFloatComplex> B, int ldb, CudaDeviceVariable<cuFloatComplex> beta, CudaDeviceVariable<cuFloatComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasCsyrkx(_blasHandle, uplo, trans, n, k, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCsyrkx", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs a variation of the symmetric rank- update C = alpha * (Op(A)*Op(B))^T + beta * C where alpha 
		/// and beta are scalars, C is a symmetric matrix stored in lower or upper mode, and A
		/// and B are matrices with dimensions op(A) n*k and op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix C lower or upper part, is stored, the other symmetric part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or transpose.</param>
		/// <param name="n">number of rows of matrix op(A), op(B) and C.</param>
		/// <param name="k">number of columns of matrix op(A) and op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimension lda x k with lda>=max(1,n) if transa == CUBLAS_OP_N and lda x n with lda>=max(1,k) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb x k with ldb>=max(1,n) if transa == CUBLAS_OP_N and ldb x n with ldb>=max(1,k) otherwise.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication, if beta==0, then C does not have to be a valid input.</param>
		/// <param name="C">array of dimensions ldc x n with ldc>=max(1,n).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Syrkx(FillMode uplo, Operation trans, int n, int k, CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda,
													CudaDeviceVariable<cuDoubleComplex> B, int ldb, CudaDeviceVariable<cuDoubleComplex> beta, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasZsyrkx(_blasHandle, uplo, trans, n, k, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZsyrkx", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion

		#region HERKX : eXtended HERK
		                              
		/// <summary>
		/// This function performs a variation of the Hermitian rank-k update C = alpha * Op(A) * Op(B)^H + beta * C where
		/// alpha and beta are scalars, and C is a Hermitian matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A) n*k and Op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other Hermitian part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of rows of matrix op(A), op(B) and C.</param>
		/// <param name="k">number of columns of matrix op(A) and op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimension lda x k with lda>=max(1,n) if transa == CUBLAS_OP_N and lda x n with lda>=max(1,k) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimension ldb x k with ldb>=max(1,n) if transa == CUBLAS_OP_N and ldb x n with ldb>=max(1,k) otherwise.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">real scalar used for multiplication, if beta==0 then C does not have to be a valid input.</param>
		/// <param name="C">array of dimension ldc x n, with ldc>=max(1,n). The imaginary parts of the diagonal elements are assumed and set to zero.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Herkx(FillMode uplo, Operation trans, int n, int k, ref cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> A, int lda,
													CudaDeviceVariable<cuFloatComplex> B, int ldb, ref float beta, CudaDeviceVariable<cuFloatComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasCherkx(_blasHandle, uplo, trans, n, k, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCherkx", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
                                        
		/// <summary>
		/// This function performs a variation of the Hermitian rank-k update C = alpha * Op(A) * Op(B)^H + beta * C where
		/// alpha and beta are scalars, and C is a Hermitian matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A) n*k and Op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other Hermitian part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of rows of matrix op(A), op(B) and C.</param>
		/// <param name="k">number of columns of matrix op(A) and op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimension lda x k with lda>=max(1,n) if transa == CUBLAS_OP_N and lda x n with lda>=max(1,k) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimension ldb x k with ldb>=max(1,n) if transa == CUBLAS_OP_N and ldb x n with ldb>=max(1,k) otherwise.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">real scalar used for multiplication, if beta==0 then C does not have to be a valid input.</param>
		/// <param name="C">array of dimension ldc x n, with ldc>=max(1,n). The imaginary parts of the diagonal elements are assumed and set to zero.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Herkx(FillMode uplo, Operation trans, int n, int k, ref cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda,
													CudaDeviceVariable<cuDoubleComplex> B, int ldb, ref double beta, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasZherkx(_blasHandle, uplo, trans, n, k, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZherkx", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs a variation of the Hermitian rank-k update C = alpha * Op(A) * Op(B)^H + beta * C where
		/// alpha and beta are scalars, and C is a Hermitian matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A) n*k and Op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other Hermitian part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of rows of matrix op(A), op(B) and C.</param>
		/// <param name="k">number of columns of matrix op(A) and op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimension lda x k with lda>=max(1,n) if transa == CUBLAS_OP_N and lda x n with lda>=max(1,k) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimension ldb x k with ldb>=max(1,n) if transa == CUBLAS_OP_N and ldb x n with ldb>=max(1,k) otherwise.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">real scalar used for multiplication, if beta==0 then C does not have to be a valid input.</param>
		/// <param name="C">array of dimension ldc x n, with ldc>=max(1,n). The imaginary parts of the diagonal elements are assumed and set to zero.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Herkx(FillMode uplo, Operation trans, int n, int k, CudaDeviceVariable<cuFloatComplex>alpha, CudaDeviceVariable<cuFloatComplex> A, int lda,
													CudaDeviceVariable<cuFloatComplex> B, int ldb, CudaDeviceVariable<float> beta, CudaDeviceVariable<cuFloatComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasCherkx(_blasHandle, uplo, trans, n, k, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCherkx", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs a variation of the Hermitian rank-k update C = alpha * Op(A) * Op(B)^H + beta * C where
		/// alpha and beta are scalars, and C is a Hermitian matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A) n*k and Op(B) n*k, respectively.
		/// </summary>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other Hermitian part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of rows of matrix op(A), op(B) and C.</param>
		/// <param name="k">number of columns of matrix op(A) and op(B).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimension lda x k with lda>=max(1,n) if transa == CUBLAS_OP_N and lda x n with lda>=max(1,k) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimension ldb x k with ldb>=max(1,n) if transa == CUBLAS_OP_N and ldb x n with ldb>=max(1,k) otherwise.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">real scalar used for multiplication, if beta==0 then C does not have to be a valid input.</param>
		/// <param name="C">array of dimension ldc x n, with ldc>=max(1,n). The imaginary parts of the diagonal elements are assumed and set to zero.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Herkx(FillMode uplo, Operation trans, int n, int k, CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda,
													CudaDeviceVariable<cuDoubleComplex> B, int ldb, CudaDeviceVariable<double> beta, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasZherkx(_blasHandle, uplo, trans, n, k, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZherkx", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion

		#region SYMM
		/// <summary>
		/// This function performs the symmetric matrix-matrix multiplication C = alpha*A*B + beta*C if side==SideMode.Left or C = alpha*B*A + beta*C if side==SideMode.Right 
		/// where A is a symmetric matrix stored in lower or upper mode, B and C are m*n matrices, and alpha and beta are scalars.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of B.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="m">number of rows of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldc * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Symm(SideMode side, FillMode uplo, int m, int n, float alpha, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> B, int ldb,
			float beta, CudaDeviceVariable<float> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasSsymm_v2(_blasHandle, side, uplo, m, n, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSsymm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric matrix-matrix multiplication C = alpha*A*B + beta*C if side==SideMode.Left or C = alpha*B*A + beta*C if side==SideMode.Right 
		/// where A is a symmetric matrix stored in lower or upper mode, B and C are m*n matrices, and alpha and beta are scalars.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of B.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="m">number of rows of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldc * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Symm(SideMode side, FillMode uplo, int m, int n, CudaDeviceVariable<float> alpha, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> B, int ldb,
			CudaDeviceVariable<float> beta, CudaDeviceVariable<float> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasSsymm_v2(_blasHandle, side, uplo, m, n, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSsymm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the symmetric matrix-matrix multiplication C = alpha*A*B + beta*C if side==SideMode.Left or C = alpha*B*A + beta*C if side==SideMode.Right 
		/// where A is a symmetric matrix stored in lower or upper mode, B and C are m*n matrices, and alpha and beta are scalars.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of B.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="m">number of rows of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldc * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Symm(SideMode side, FillMode uplo, int m, int n, double alpha, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> B, int ldb,
			double beta, CudaDeviceVariable<double> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasDsymm_v2(_blasHandle, side, uplo, m, n, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDsymm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric matrix-matrix multiplication C = alpha*A*B + beta*C if side==SideMode.Left or C = alpha*B*A + beta*C if side==SideMode.Right 
		/// where A is a symmetric matrix stored in lower or upper mode, B and C are m*n matrices, and alpha and beta are scalars.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of B.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="m">number of rows of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldc * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Symm(SideMode side, FillMode uplo, int m, int n, CudaDeviceVariable<double> alpha, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> B, int ldb,
			CudaDeviceVariable<double> beta, CudaDeviceVariable<double> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasDsymm_v2(_blasHandle, side, uplo, m, n, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDsymm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}




		/// <summary>
		/// This function performs the symmetric matrix-matrix multiplication C = alpha*A*B + beta*C if side==SideMode.Left or C = alpha*B*A + beta*C if side==SideMode.Right 
		/// where A is a symmetric matrix stored in lower or upper mode, B and C are m*n matrices, and alpha and beta are scalars.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of B.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="m">number of rows of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldc * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Symm(SideMode side, FillMode uplo, int m, int n, cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> B, int ldb,
			cuFloatComplex beta, CudaDeviceVariable<cuFloatComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasCsymm_v2(_blasHandle, side, uplo, m, n, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCsymm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric matrix-matrix multiplication C = alpha*A*B + beta*C if side==SideMode.Left or C = alpha*B*A + beta*C if side==SideMode.Right 
		/// where A is a symmetric matrix stored in lower or upper mode, B and C are m*n matrices, and alpha and beta are scalars.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of B.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="m">number of rows of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldc * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Symm(SideMode side, FillMode uplo, int m, int n, CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> B, int ldb,
			CudaDeviceVariable<cuFloatComplex> beta, CudaDeviceVariable<cuFloatComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasCsymm_v2(_blasHandle, side, uplo, m, n, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCsymm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the symmetric matrix-matrix multiplication C = alpha*A*B + beta*C if side==SideMode.Left or C = alpha*B*A + beta*C if side==SideMode.Right 
		/// where A is a symmetric matrix stored in lower or upper mode, B and C are m*n matrices, and alpha and beta are scalars.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of B.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="m">number of rows of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldc * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Symm(SideMode side, FillMode uplo, int m, int n, cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> B, int ldb,
			cuDoubleComplex beta, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasZsymm_v2(_blasHandle, side, uplo, m, n, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZsymm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the symmetric matrix-matrix multiplication C = alpha*A*B + beta*C if side==SideMode.Left or C = alpha*B*A + beta*C if side==SideMode.Right 
		/// where A is a symmetric matrix stored in lower or upper mode, B and C are m*n matrices, and alpha and beta are scalars.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of B.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="m">number of rows of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldc * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Symm(SideMode side, FillMode uplo, int m, int n, CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> B, int ldb,
			CudaDeviceVariable<cuDoubleComplex> beta, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasZsymm_v2(_blasHandle, side, uplo, m, n, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZsymm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region HEMM

		/// <summary>
		/// This function performs the Hermitian matrix-matrix multiplication C = alpha*A*B + beta*C if side==SideMode.Left or C = alpha*B*A + beta*C if side==SideMode.Right 
		/// where A is a Hermitian matrix stored in lower or upper mode, B and C are m*n matrices, and alpha and beta are scalars.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of B.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="m">number of rows of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldc * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Hemm(SideMode side, FillMode uplo, int m, int n, cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> B, int ldb,
			cuFloatComplex beta, CudaDeviceVariable<cuFloatComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasChemm_v2(_blasHandle, side, uplo, m, n, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasChemm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the Hermitian matrix-matrix multiplication C = alpha*A*B + beta*C if side==SideMode.Left or C = alpha*B*A + beta*C if side==SideMode.Right 
		/// where A is a Hermitian matrix stored in lower or upper mode, B and C are m*n matrices, and alpha and beta are scalars.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of B.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="m">number of rows of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldc * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Hemm(SideMode side, FillMode uplo, int m, int n, CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> B, int ldb,
			CudaDeviceVariable<cuFloatComplex> beta, CudaDeviceVariable<cuFloatComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasChemm_v2(_blasHandle, side, uplo, m, n, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasChemm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the Hermitian matrix-matrix multiplication C = alpha*A*B + beta*C if side==SideMode.Left or C = alpha*B*A + beta*C if side==SideMode.Right 
		/// where A is a Hermitian matrix stored in lower or upper mode, B and C are m*n matrices, and alpha and beta are scalars.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of B.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="m">number of rows of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldc * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Hemm(SideMode side, FillMode uplo, int m, int n, cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> B, int ldb,
			cuDoubleComplex beta, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasZhemm_v2(_blasHandle, side, uplo, m, n, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, ref beta, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZhemm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the Hermitian matrix-matrix multiplication C = alpha*A*B + beta*C if side==SideMode.Left or C = alpha*B*A + beta*C if side==SideMode.Right 
		/// where A is a Hermitian matrix stored in lower or upper mode, B and C are m*n matrices, and alpha and beta are scalars.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of B.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="m">number of rows of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix C and B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldc * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Hemm(SideMode side, FillMode uplo, int m, int n, CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> B, int ldb,
			CudaDeviceVariable<cuDoubleComplex> beta, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasZhemm_v2(_blasHandle, side, uplo, m, n, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, beta.DevicePointer, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZhemm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region TRSM
		/// <summary>
		/// This function solves the triangular linear system with multiple right-hand-sides Op(A)X = alpha*B side==SideMode.Left or XOp(A) = alpha*B if side==SideMode.Right 
		/// where A is a triangular matrix stored in lower or upper mode with or without the maindiagonal, X and B are m*n matrices, and alpha is a scalar.<para/>
		/// The solution X overwrites the right-hand-sides B on exit.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of X.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication. If alpha==0 then A is not referenced and B does not have to be a valid input.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		public void Trsm(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, float alpha, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> B, int ldb)
		{
			_status = CudaBlasNativeMethods.cublasStrsm_v2(_blasHandle, side, uplo, trans, diag, m, n, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasStrsm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function solves the triangular linear system with multiple right-hand-sides Op(A)X = alpha*B side==SideMode.Left or XOp(A) = alpha*B if side==SideMode.Right 
		/// where A is a triangular matrix stored in lower or upper mode with or without the maindiagonal, X and B are m*n matrices, and alpha is a scalar.<para/>
		/// The solution X overwrites the right-hand-sides B on exit.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of X.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication. If alpha==0 then A is not referenced and B does not have to be a valid input.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		public void Trsm(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, CudaDeviceVariable<float> alpha, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> B, int ldb)
		{
			_status = CudaBlasNativeMethods.cublasStrsm_v2(_blasHandle, side, uplo, trans, diag, m, n, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasStrsm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function solves the triangular linear system with multiple right-hand-sides Op(A)X = alpha*B side==SideMode.Left or XOp(A) = alpha*B if side==SideMode.Right 
		/// where A is a triangular matrix stored in lower or upper mode with or without the maindiagonal, X and B are m*n matrices, and alpha is a scalar.<para/>
		/// The solution X overwrites the right-hand-sides B on exit.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of X.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication. If alpha==0 then A is not referenced and B does not have to be a valid input.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		public void Trsm(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, double alpha, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> B, int ldb)
		{
			_status = CudaBlasNativeMethods.cublasDtrsm_v2(_blasHandle, side, uplo, trans, diag, m, n, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDtrsm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function solves the triangular linear system with multiple right-hand-sides Op(A)X = alpha*B side==SideMode.Left or XOp(A) = alpha*B if side==SideMode.Right 
		/// where A is a triangular matrix stored in lower or upper mode with or without the maindiagonal, X and B are m*n matrices, and alpha is a scalar.<para/>
		/// The solution X overwrites the right-hand-sides B on exit.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of X.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication. If alpha==0 then A is not referenced and B does not have to be a valid input.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		public void Trsm(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, CudaDeviceVariable<double> alpha, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<double> B, int ldb)
		{
			_status = CudaBlasNativeMethods.cublasDtrsm_v2(_blasHandle, side, uplo, trans, diag, m, n, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDtrsm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}



		/// <summary>
		/// This function solves the triangular linear system with multiple right-hand-sides Op(A)X = alpha*B side==SideMode.Left or XOp(A) = alpha*B if side==SideMode.Right 
		/// where A is a triangular matrix stored in lower or upper mode with or without the maindiagonal, X and B are m*n matrices, and alpha is a scalar.<para/>
		/// The solution X overwrites the right-hand-sides B on exit.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of X.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication. If alpha==0 then A is not referenced and B does not have to be a valid input.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		public void Trsm(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> B, int ldb)
		{
			_status = CudaBlasNativeMethods.cublasCtrsm_v2(_blasHandle, side, uplo, trans, diag, m, n, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCtrsm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function solves the triangular linear system with multiple right-hand-sides Op(A)X = alpha*B side==SideMode.Left or XOp(A) = alpha*B if side==SideMode.Right 
		/// where A is a triangular matrix stored in lower or upper mode with or without the maindiagonal, X and B are m*n matrices, and alpha is a scalar.<para/>
		/// The solution X overwrites the right-hand-sides B on exit.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of X.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication. If alpha==0 then A is not referenced and B does not have to be a valid input.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		public void Trsm(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> B, int ldb)
		{
			_status = CudaBlasNativeMethods.cublasCtrsm_v2(_blasHandle, side, uplo, trans, diag, m, n, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCtrsm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function solves the triangular linear system with multiple right-hand-sides Op(A)X = alpha*B side==SideMode.Left or XOp(A) = alpha*B if side==SideMode.Right 
		/// where A is a triangular matrix stored in lower or upper mode with or without the maindiagonal, X and B are m*n matrices, and alpha is a scalar.<para/>
		/// The solution X overwrites the right-hand-sides B on exit.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of X.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication. If alpha==0 then A is not referenced and B does not have to be a valid input.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		public void Trsm(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> B, int ldb)
		{
			_status = CudaBlasNativeMethods.cublasZtrsm_v2(_blasHandle, side, uplo, trans, diag, m, n, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZtrsm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function solves the triangular linear system with multiple right-hand-sides Op(A)X = alpha*B side==SideMode.Left or XOp(A) = alpha*B if side==SideMode.Right 
		/// where A is a triangular matrix stored in lower or upper mode with or without the maindiagonal, X and B are m*n matrices, and alpha is a scalar.<para/>
		/// The solution X overwrites the right-hand-sides B on exit.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of X.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication. If alpha==0 then A is not referenced and B does not have to be a valid input.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		public void Trsm(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> B, int ldb)
		{
			_status = CudaBlasNativeMethods.cublasZtrsm_v2(_blasHandle, side, uplo, trans, diag, m, n, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZtrsm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		#endregion
		#region TRMM
		/// <summary>
		/// This function performs the triangular matrix-matrix multiplication C = alpha*Op(A) * B if side==SideMode.Left or C = alpha*B * Op(A) if side==SideMode.Right 
		/// where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, B and C are m*n matrices, and alpha is a scalar.<para/>
		/// Notice that in order to achieve better parallelism CUBLAS differs from the BLAS API only for this routine. The BLAS API assumes an in-place implementation (with results
		/// written back to B), while the CUBLAS API assumes an out-of-place implementation (with results written into C). The application can obtain the in-place functionality of BLAS in
		/// the CUBLAS API by passing the address of the matrix B in place of the matrix C. No other overlapping in the input parameters is supported.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of X.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication. If alpha==0 then A is not referenced and B does not have to be a valid input.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="C">array of dimensions ldc * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Trsm(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, float alpha, CudaDeviceVariable<float> A, int lda,
			CudaDeviceVariable<float> B, int ldb, CudaDeviceVariable<float> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasStrmm_v2(_blasHandle, side, uplo, trans, diag, m, n, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasStrmm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the triangular matrix-matrix multiplication C = alpha*Op(A) * B if side==SideMode.Left or C = alpha*B * Op(A) if side==SideMode.Right 
		/// where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, B and C are m*n matrices, and alpha is a scalar.<para/>
		/// Notice that in order to achieve better parallelism CUBLAS differs from the BLAS API only for this routine. The BLAS API assumes an in-place implementation (with results
		/// written back to B), while the CUBLAS API assumes an out-of-place implementation (with results written into C). The application can obtain the in-place functionality of BLAS in
		/// the CUBLAS API by passing the address of the matrix B in place of the matrix C. No other overlapping in the input parameters is supported.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of X.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication. If alpha==0 then A is not referenced and B does not have to be a valid input.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="C">array of dimensions ldc * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Trsm(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, CudaDeviceVariable<float> alpha, CudaDeviceVariable<float> A, int lda,
			CudaDeviceVariable<float> B, int ldb, CudaDeviceVariable<float> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasStrmm_v2(_blasHandle, side, uplo, trans, diag, m, n, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasStrmm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the triangular matrix-matrix multiplication C = alpha*Op(A) * B if side==SideMode.Left or C = alpha*B * Op(A) if side==SideMode.Right 
		/// where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, B and C are m*n matrices, and alpha is a scalar.<para/>
		/// Notice that in order to achieve better parallelism CUBLAS differs from the BLAS API only for this routine. The BLAS API assumes an in-place implementation (with results
		/// written back to B), while the CUBLAS API assumes an out-of-place implementation (with results written into C). The application can obtain the in-place functionality of BLAS in
		/// the CUBLAS API by passing the address of the matrix B in place of the matrix C. No other overlapping in the input parameters is supported.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of X.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication. If alpha==0 then A is not referenced and B does not have to be a valid input.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="C">array of dimensions ldc * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Trsm(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, double alpha, CudaDeviceVariable<double> A, int lda,
			CudaDeviceVariable<double> B, int ldb, CudaDeviceVariable<double> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasDtrmm_v2(_blasHandle, side, uplo, trans, diag, m, n, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDtrmm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the triangular matrix-matrix multiplication C = alpha*Op(A) * B if side==SideMode.Left or C = alpha*B * Op(A) if side==SideMode.Right 
		/// where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, B and C are m*n matrices, and alpha is a scalar.<para/>
		/// Notice that in order to achieve better parallelism CUBLAS differs from the BLAS API only for this routine. The BLAS API assumes an in-place implementation (with results
		/// written back to B), while the CUBLAS API assumes an out-of-place implementation (with results written into C). The application can obtain the in-place functionality of BLAS in
		/// the CUBLAS API by passing the address of the matrix B in place of the matrix C. No other overlapping in the input parameters is supported.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of X.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication. If alpha==0 then A is not referenced and B does not have to be a valid input.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="C">array of dimensions ldc * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Trsm(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, CudaDeviceVariable<double> alpha, CudaDeviceVariable<double> A, int lda,
			CudaDeviceVariable<double> B, int ldb, CudaDeviceVariable<double> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasDtrmm_v2(_blasHandle, side, uplo, trans, diag, m, n, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDtrmm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}




		/// <summary>
		/// This function performs the triangular matrix-matrix multiplication C = alpha*Op(A) * B if side==SideMode.Left or C = alpha*B * Op(A) if side==SideMode.Right 
		/// where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, B and C are m*n matrices, and alpha is a scalar.<para/>
		/// Notice that in order to achieve better parallelism CUBLAS differs from the BLAS API only for this routine. The BLAS API assumes an in-place implementation (with results
		/// written back to B), while the CUBLAS API assumes an out-of-place implementation (with results written into C). The application can obtain the in-place functionality of BLAS in
		/// the CUBLAS API by passing the address of the matrix B in place of the matrix C. No other overlapping in the input parameters is supported.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of X.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication. If alpha==0 then A is not referenced and B does not have to be a valid input.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="C">array of dimensions ldc * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Trsm(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> A, int lda,
			CudaDeviceVariable<cuFloatComplex> B, int ldb, CudaDeviceVariable<cuFloatComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasCtrmm_v2(_blasHandle, side, uplo, trans, diag, m, n, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCtrmm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the triangular matrix-matrix multiplication C = alpha*Op(A) * B if side==SideMode.Left or C = alpha*B * Op(A) if side==SideMode.Right 
		/// where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, B and C are m*n matrices, and alpha is a scalar.<para/>
		/// Notice that in order to achieve better parallelism CUBLAS differs from the BLAS API only for this routine. The BLAS API assumes an in-place implementation (with results
		/// written back to B), while the CUBLAS API assumes an out-of-place implementation (with results written into C). The application can obtain the in-place functionality of BLAS in
		/// the CUBLAS API by passing the address of the matrix B in place of the matrix C. No other overlapping in the input parameters is supported.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of X.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication. If alpha==0 then A is not referenced and B does not have to be a valid input.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="C">array of dimensions ldc * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Trsm(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> A, int lda,
			CudaDeviceVariable<cuFloatComplex> B, int ldb, CudaDeviceVariable<cuFloatComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasCtrmm_v2(_blasHandle, side, uplo, trans, diag, m, n, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCtrmm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the triangular matrix-matrix multiplication C = alpha*Op(A) * B if side==SideMode.Left or C = alpha*B * Op(A) if side==SideMode.Right 
		/// where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, B and C are m*n matrices, and alpha is a scalar.<para/>
		/// Notice that in order to achieve better parallelism CUBLAS differs from the BLAS API only for this routine. The BLAS API assumes an in-place implementation (with results
		/// written back to B), while the CUBLAS API assumes an out-of-place implementation (with results written into C). The application can obtain the in-place functionality of BLAS in
		/// the CUBLAS API by passing the address of the matrix B in place of the matrix C. No other overlapping in the input parameters is supported.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of X.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication. If alpha==0 then A is not referenced and B does not have to be a valid input.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="C">array of dimensions ldc * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Trsm(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda,
			CudaDeviceVariable<cuDoubleComplex> B, int ldb, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasZtrmm_v2(_blasHandle, side, uplo, trans, diag, m, n, ref alpha, A.DevicePointer, lda, B.DevicePointer, ldb, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZtrmm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the triangular matrix-matrix multiplication C = alpha*Op(A) * B if side==SideMode.Left or C = alpha*B * Op(A) if side==SideMode.Right 
		/// where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, B and C are m*n matrices, and alpha is a scalar.<para/>
		/// Notice that in order to achieve better parallelism CUBLAS differs from the BLAS API only for this routine. The BLAS API assumes an in-place implementation (with results
		/// written back to B), while the CUBLAS API assumes an out-of-place implementation (with results written into C). The application can obtain the in-place functionality of BLAS in
		/// the CUBLAS API by passing the address of the matrix B in place of the matrix C. No other overlapping in the input parameters is supported.
		/// </summary>
		/// <param name="side">indicates if matrix A is on the left or right of X.</param>
		/// <param name="uplo">indicates if matrix A lower or upper part is stored, the other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix A are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B, with matrix A sized accordingly.</param>
		/// <param name="n">number of columns of matrix B, with matrix A sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication. If alpha==0 then A is not referenced and B does not have to be a valid input.</param>
		/// <param name="A">array of dimensions lda * m.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="C">array of dimensions ldc * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Trsm(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda,
			CudaDeviceVariable<cuDoubleComplex> B, int ldb, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasZtrmm_v2(_blasHandle, side, uplo, trans, diag, m, n, alpha.DevicePointer, A.DevicePointer, lda, B.DevicePointer, ldb, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZtrmm_v2", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion

		#endregion

		#region CUBLAS BLAS-like extension
		#region GEAM
		#region Host ptr
		/// <summary>
		/// This function performs the matrix-matrix addition/transposition C = alpha * Op(A) + beta * Op(B) where 
		/// alpha and beta are scalars, and A, B and C are matrices stored in column-major format with dimensions 
		/// op(A) m*n, op(B) m*n and C m*n, respectively.
		/// </summary>
		/// <param name="transa">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A) and C.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Geam(Operation transa, Operation transb, int m, int n, float alpha, CudaDeviceVariable<float> A, int lda,
			CudaDeviceVariable<float> B, int ldb, float beta, CudaDeviceVariable<float> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasSgeam(_blasHandle, transa, transb, m, n, ref alpha, A.DevicePointer, lda, ref beta, 
				B.DevicePointer, ldb, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSgeam", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-matrix addition/transposition C = alpha * Op(A) + beta * Op(B) where 
		/// alpha and beta are scalars, and A, B and C are matrices stored in column-major format with dimensions 
		/// op(A) m*n, op(B) m*n and C m*n, respectively.
		/// </summary>
		/// <param name="transa">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A) and C.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Geam(Operation transa, Operation transb, int m, int n, double alpha, CudaDeviceVariable<double> A, int lda,
			CudaDeviceVariable<double> B, int ldb, double beta, CudaDeviceVariable<double> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasDgeam(_blasHandle, transa, transb, m, n, ref alpha, A.DevicePointer, lda, ref beta,
				B.DevicePointer, ldb, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDgeam", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-matrix addition/transposition C = alpha * Op(A) + beta * Op(B) where 
		/// alpha and beta are scalars, and A, B and C are matrices stored in column-major format with dimensions 
		/// op(A) m*n, op(B) m*n and C m*n, respectively.
		/// </summary>
		/// <param name="transa">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A) and C.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Geam(Operation transa, Operation transb, int m, int n, cuFloatComplex alpha, CudaDeviceVariable<cuFloatComplex> A, int lda,
			CudaDeviceVariable<cuFloatComplex> B, int ldb, cuFloatComplex beta, CudaDeviceVariable<cuFloatComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasCgeam(_blasHandle, transa, transb, m, n, ref alpha, A.DevicePointer, lda, ref beta,
				B.DevicePointer, ldb, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCgeam", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-matrix addition/transposition C = alpha * Op(A) + beta * Op(B) where 
		/// alpha and beta are scalars, and A, B and C are matrices stored in column-major format with dimensions 
		/// op(A) m*n, op(B) m*n and C m*n, respectively.
		/// </summary>
		/// <param name="transa">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A) and C.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Geam(Operation transa, Operation transb, int m, int n, cuDoubleComplex alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda,
			CudaDeviceVariable<cuDoubleComplex> B, int ldb, cuDoubleComplex beta, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasZgeam(_blasHandle, transa, transb, m, n, ref alpha, A.DevicePointer, lda, ref beta,
				B.DevicePointer, ldb, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZgeam", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#region Device ptr
		/// <summary>
		/// This function performs the matrix-matrix addition/transposition C = alpha * Op(A) + beta * Op(B) where 
		/// alpha and beta are scalars, and A, B and C are matrices stored in column-major format with dimensions 
		/// op(A) m*n, op(B) m*n and C m*n, respectively.
		/// </summary>
		/// <param name="transa">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A) and C.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Geam(Operation transa, Operation transb, int m, int n, CudaDeviceVariable<float> alpha, CudaDeviceVariable<float> A, int lda,
			CudaDeviceVariable<float> B, int ldb, CudaDeviceVariable<float> beta, CudaDeviceVariable<float> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasSgeam(_blasHandle, transa, transb, m, n, alpha.DevicePointer, A.DevicePointer, lda, beta.DevicePointer,
				B.DevicePointer, ldb, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSgeam", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-matrix addition/transposition C = alpha * Op(A) + beta * Op(B) where 
		/// alpha and beta are scalars, and A, B and C are matrices stored in column-major format with dimensions 
		/// op(A) m*n, op(B) m*n and C m*n, respectively.
		/// </summary>
		/// <param name="transa">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A) and C.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Geam(Operation transa, Operation transb, int m, int n, CudaDeviceVariable<double> alpha, CudaDeviceVariable<double> A, int lda,
			CudaDeviceVariable<double> B, int ldb, CudaDeviceVariable<double> beta, CudaDeviceVariable<double> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasDgeam(_blasHandle, transa, transb, m, n, alpha.DevicePointer, A.DevicePointer, lda, beta.DevicePointer,
				B.DevicePointer, ldb, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDgeam", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-matrix addition/transposition C = alpha * Op(A) + beta * Op(B) where 
		/// alpha and beta are scalars, and A, B and C are matrices stored in column-major format with dimensions 
		/// op(A) m*n, op(B) m*n and C m*n, respectively.
		/// </summary>
		/// <param name="transa">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A) and C.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Geam(Operation transa, Operation transb, int m, int n, CudaDeviceVariable<cuFloatComplex> alpha, CudaDeviceVariable<cuFloatComplex> A, int lda,
			CudaDeviceVariable<cuFloatComplex> B, int ldb, CudaDeviceVariable<cuFloatComplex> beta, CudaDeviceVariable<cuFloatComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasCgeam(_blasHandle, transa, transb, m, n, alpha.DevicePointer, A.DevicePointer, lda, beta.DevicePointer,
				B.DevicePointer, ldb, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCgeam", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-matrix addition/transposition C = alpha * Op(A) + beta * Op(B) where 
		/// alpha and beta are scalars, and A, B and C are matrices stored in column-major format with dimensions 
		/// op(A) m*n, op(B) m*n and C m*n, respectively.
		/// </summary>
		/// <param name="transa">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A) and C.</param>
		/// <param name="n">number of columns of matrix op(B) and C.</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="A">array of dimensions lda * k.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="B">array of dimensions ldb * n.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B.</param>
		/// <param name="beta">scalar used for multiplication.</param>
		/// <param name="C">array of dimensions ldb * n.</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store matrix C.</param>
		public void Geam(Operation transa, Operation transb, int m, int n, CudaDeviceVariable<cuDoubleComplex> alpha, CudaDeviceVariable<cuDoubleComplex> A, int lda,
			CudaDeviceVariable<cuDoubleComplex> B, int ldb, CudaDeviceVariable<cuDoubleComplex> beta, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasZgeam(_blasHandle, transa, transb, m, n, alpha.DevicePointer, A.DevicePointer, lda, beta.DevicePointer,
				B.DevicePointer, ldb, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZgeam", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#endregion


		#region Batched - MATINV
		/// <summary>
		/// 
		/// </summary>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="Ainv"></param>
		/// <param name="lda_inv"></param>
		/// <param name="INFO"></param>
		/// <param name="batchSize"></param>
		public void MatinvBatchedS(int n, CudaDeviceVariable<CUdeviceptr> A, int lda,
														   CudaDeviceVariable<CUdeviceptr> Ainv, int lda_inv, CudaDeviceVariable<int> INFO, int batchSize)
		{
			_status = CudaBlasNativeMethods.cublasSmatinvBatched(_blasHandle, n, A.DevicePointer, lda, Ainv.DevicePointer, lda_inv, INFO.DevicePointer, batchSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSmatinvBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="Ainv"></param>
		/// <param name="lda_inv"></param>
		/// <param name="INFO"></param>
		/// <param name="batchSize"></param>
		public void MatinvBatchedD(int n, CudaDeviceVariable<CUdeviceptr> A, int lda,
														   CudaDeviceVariable<CUdeviceptr> Ainv, int lda_inv, CudaDeviceVariable<int> INFO, int batchSize)
		{
			_status = CudaBlasNativeMethods.cublasDmatinvBatched(_blasHandle, n, A.DevicePointer, lda, Ainv.DevicePointer, lda_inv, INFO.DevicePointer, batchSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDmatinvBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="Ainv"></param>
		/// <param name="lda_inv"></param>
		/// <param name="INFO"></param>
		/// <param name="batchSize"></param>
		public void MatinvBatchedC(int n, CudaDeviceVariable<CUdeviceptr> A, int lda,
														   CudaDeviceVariable<CUdeviceptr> Ainv, int lda_inv, CudaDeviceVariable<int> INFO, int batchSize)
		{
			_status = CudaBlasNativeMethods.cublasCmatinvBatched(_blasHandle, n, A.DevicePointer, lda, Ainv.DevicePointer, lda_inv, INFO.DevicePointer, batchSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCmatinvBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="n"></param>
		/// <param name="A"></param>
		/// <param name="lda"></param>
		/// <param name="Ainv"></param>
		/// <param name="lda_inv"></param>
		/// <param name="INFO"></param>
		/// <param name="batchSize"></param>
		public void MatinvBatchedZ(int n, CudaDeviceVariable<CUdeviceptr> A, int lda,
														   CudaDeviceVariable<CUdeviceptr> Ainv, int lda_inv, CudaDeviceVariable<int> INFO, int batchSize)
		{
			_status = CudaBlasNativeMethods.cublasZmatinvBatched(_blasHandle, n, A.DevicePointer, lda, Ainv.DevicePointer, lda_inv, INFO.DevicePointer, batchSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZmatinvBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		#endregion

		#region DGMM

		/// <summary>
		/// This function performs the matrix-matrix multiplication C = A x diag(X) if mode == CUBLAS_SIDE_RIGHT, or 
		/// C = diag(X) x A if mode == CUBLAS_SIDE_LEFT.<para/>
		/// where A and C are matrices stored in column-major format with dimensions m*n. X is a
		/// vector of size n if mode == CUBLAS_SIDE_RIGHT and of size m if mode ==
		/// CUBLAS_SIDE_LEFT. X is gathered from one-dimensional array x with stride incx. The
		/// absolute value of incx is the stride and the sign of incx is direction of the stride. If incx
		/// is positive, then we forward x from the first element. Otherwise, we backward x from the
		/// last element.
		/// </summary>
		/// <param name="mode">left multiply if mode == CUBLAS_SIDE_LEFT 
		/// or right multiply if mode == CUBLAS_SIDE_RIGHT</param>
		/// <param name="m">number of rows of matrix A and C.</param>
		/// <param name="n">number of columns of matrix A and C.</param>
		/// <param name="A">array of dimensions lda x n with lda >= max(1,m)</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store the matrix A.</param>
		/// <param name="X">one-dimensional array of size |incx|*m 
		/// if mode == CUBLAS_SIDE_LEFT and |incx|*n
		/// if mode == CUBLAS_SIDE_RIGHT</param>
		/// <param name="incx">stride of one-dimensional array x.</param>
		/// <param name="C">array of dimensions ldc*n with ldc >= max(1,m).</param>
		/// <param name="ldc">leading dimension of a two-dimensional array used to store the matrix C.</param>
		public void Dgmm(SideMode mode, int m, int n, CudaDeviceVariable<float> A, int lda,
			CudaDeviceVariable<float> X, int incx, CudaDeviceVariable<float> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasSdgmm(_blasHandle, mode, m, n, A.DevicePointer, lda, 
				X.DevicePointer, incx, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSdgmm", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-matrix multiplication C = A x diag(X) if mode == CUBLAS_SIDE_RIGHT, or 
		/// C = diag(X) x A if mode == CUBLAS_SIDE_LEFT.<para/>
		/// where A and C are matrices stored in column-major format with dimensions m*n. X is a
		/// vector of size n if mode == CUBLAS_SIDE_RIGHT and of size m if mode ==
		/// CUBLAS_SIDE_LEFT. X is gathered from one-dimensional array x with stride incx. The
		/// absolute value of incx is the stride and the sign of incx is direction of the stride. If incx
		/// is positive, then we forward x from the first element. Otherwise, we backward x from the
		/// last element.
		/// </summary>
		/// <param name="mode">left multiply if mode == CUBLAS_SIDE_LEFT 
		/// or right multiply if mode == CUBLAS_SIDE_RIGHT</param>
		/// <param name="m">number of rows of matrix A and C.</param>
		/// <param name="n">number of columns of matrix A and C.</param>
		/// <param name="A">array of dimensions lda x n with lda >= max(1,m)</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store the matrix A.</param>
		/// <param name="X">one-dimensional array of size |incx|*m 
		/// if mode == CUBLAS_SIDE_LEFT and |incx|*n
		/// if mode == CUBLAS_SIDE_RIGHT</param>
		/// <param name="incx">stride of one-dimensional array x.</param>
		/// <param name="C">array of dimensions ldc*n with ldc >= max(1,m).</param>
		/// <param name="ldc">leading dimension of a two-dimensional array used to store the matrix C.</param>
		public void Dgmm(SideMode mode, int m, int n, CudaDeviceVariable<double> A, int lda,
			CudaDeviceVariable<double> X, int incx, CudaDeviceVariable<double> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasDdgmm(_blasHandle, mode, m, n, A.DevicePointer, lda,
				X.DevicePointer, incx, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDdgmm", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-matrix multiplication C = A x diag(X) if mode == CUBLAS_SIDE_RIGHT, or 
		/// C = diag(X) x A if mode == CUBLAS_SIDE_LEFT.<para/>
		/// where A and C are matrices stored in column-major format with dimensions m*n. X is a
		/// vector of size n if mode == CUBLAS_SIDE_RIGHT and of size m if mode ==
		/// CUBLAS_SIDE_LEFT. X is gathered from one-dimensional array x with stride incx. The
		/// absolute value of incx is the stride and the sign of incx is direction of the stride. If incx
		/// is positive, then we forward x from the first element. Otherwise, we backward x from the
		/// last element.
		/// </summary>
		/// <param name="mode">left multiply if mode == CUBLAS_SIDE_LEFT 
		/// or right multiply if mode == CUBLAS_SIDE_RIGHT</param>
		/// <param name="m">number of rows of matrix A and C.</param>
		/// <param name="n">number of columns of matrix A and C.</param>
		/// <param name="A">array of dimensions lda x n with lda >= max(1,m)</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store the matrix A.</param>
		/// <param name="X">one-dimensional array of size |incx|*m 
		/// if mode == CUBLAS_SIDE_LEFT and |incx|*n
		/// if mode == CUBLAS_SIDE_RIGHT</param>
		/// <param name="incx">stride of one-dimensional array x.</param>
		/// <param name="C">array of dimensions ldc*n with ldc >= max(1,m).</param>
		/// <param name="ldc">leading dimension of a two-dimensional array used to store the matrix C.</param>
		public void Dgmm(SideMode mode, int m, int n, CudaDeviceVariable<cuFloatComplex> A, int lda,
			CudaDeviceVariable<cuFloatComplex> X, int incx, CudaDeviceVariable<cuFloatComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasCdgmm(_blasHandle, mode, m, n, A.DevicePointer, lda,
				X.DevicePointer, incx, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCdgmm", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		/// <summary>
		/// This function performs the matrix-matrix multiplication C = A x diag(X) if mode == CUBLAS_SIDE_RIGHT, or 
		/// C = diag(X) x A if mode == CUBLAS_SIDE_LEFT.<para/>
		/// where A and C are matrices stored in column-major format with dimensions m*n. X is a
		/// vector of size n if mode == CUBLAS_SIDE_RIGHT and of size m if mode ==
		/// CUBLAS_SIDE_LEFT. X is gathered from one-dimensional array x with stride incx. The
		/// absolute value of incx is the stride and the sign of incx is direction of the stride. If incx
		/// is positive, then we forward x from the first element. Otherwise, we backward x from the
		/// last element.
		/// </summary>
		/// <param name="mode">left multiply if mode == CUBLAS_SIDE_LEFT 
		/// or right multiply if mode == CUBLAS_SIDE_RIGHT</param>
		/// <param name="m">number of rows of matrix A and C.</param>
		/// <param name="n">number of columns of matrix A and C.</param>
		/// <param name="A">array of dimensions lda x n with lda >= max(1,m)</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store the matrix A.</param>
		/// <param name="X">one-dimensional array of size |incx|*m 
		/// if mode == CUBLAS_SIDE_LEFT and |incx|*n
		/// if mode == CUBLAS_SIDE_RIGHT</param>
		/// <param name="incx">stride of one-dimensional array x.</param>
		/// <param name="C">array of dimensions ldc*n with ldc >= max(1,m).</param>
		/// <param name="ldc">leading dimension of a two-dimensional array used to store the matrix C.</param>
		public void Dgmm(SideMode mode, int m, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda,
			CudaDeviceVariable<cuDoubleComplex> X, int incx, CudaDeviceVariable<cuDoubleComplex> C, int ldc)
		{
			_status = CudaBlasNativeMethods.cublasZdgmm(_blasHandle, mode, m, n, A.DevicePointer, lda,
				X.DevicePointer, incx, C.DevicePointer, ldc);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZdgmm", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion
		#endregion


		#region BATCH GEMM
		#region device pointer

		/// <summary>
		/// This function performs the matrix-matrix multiplications of an array of matrices.
		/// where and are scalars, and , and are arrays of pointers to matrices stored
		/// in column-major format with dimensions op(A[i])m x k, op(B[i])k x n and op(C[i])m x n, 
		/// respectively.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor. For small sizes, typically smaller than 100x100,
		/// this function improves significantly performance compared to making calls to its
		/// corresponding cublas<![CDATA[<type>]]>gemm routine. However, on GPU architectures that support
		/// concurrent kernels, it might be advantageous to make multiple calls to cublas<![CDATA[<type>]]>gemm
		/// into different streams as the matrix sizes increase.
		/// </summary>
		/// <param name="transa">operation op(A[i]) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B[i]) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A[i]) and C[i].</param>
		/// <param name="n">number of columns of op(B[i]) and C[i].</param>
		/// <param name="k">number of columns of op(A[i]) and rows of op(B[i]).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="Aarray">array of device pointers, with each array/device pointer of dim. lda x k with lda>=max(1,m) if
		/// transa==CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix A[i].</param>
		/// <param name="Barray">array of device pointers, with each array of dim. ldb x n with ldb>=max(1,k) if
		/// transa==CUBLAS_OP_N and ldb x k with ldb>=max(1,n) max(1,) otherwise.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store each matrix B[i].</param>
		/// <param name="beta">scalar used for multiplication. If beta == 0, C does not have to be a valid input.</param>
		/// <param name="Carray">array of device pointers. It has dimensions ldc x n with ldc>=max(1,m).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store each matrix C[i].</param>
		/// <param name="batchCount">number of pointers contained in A, B and C.</param>
		public void GemmBatched(Operation transa, Operation transb, int m, int n, int k, CudaDeviceVariable<float> alpha,
								   CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<CUdeviceptr> Barray, int ldb,
								   CudaDeviceVariable<float> beta, CudaDeviceVariable<CUdeviceptr> Carray, int ldc, int batchCount)
		{
			_status = CudaBlasNativeMethods.cublasSgemmBatched(_blasHandle, transa, transb, m, n, k, alpha.DevicePointer, Aarray.DevicePointer, lda, Barray.DevicePointer, ldb, beta.DevicePointer, Carray.DevicePointer, ldc, batchCount);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSgemmBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}


		/// <summary>
		/// This function performs the matrix-matrix multiplications of an array of matrices.
		/// where and are scalars, and , and are arrays of pointers to matrices stored
		/// in column-major format with dimensions op(A[i])m x k, op(B[i])k x n and op(C[i])m x n, 
		/// respectively.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor. For small sizes, typically smaller than 100x100,
		/// this function improves significantly performance compared to making calls to its
		/// corresponding cublas<![CDATA[<type>]]>gemm routine. However, on GPU architectures that support
		/// concurrent kernels, it might be advantageous to make multiple calls to cublas<![CDATA[<type>]]>gemm
		/// into different streams as the matrix sizes increase.
		/// </summary>
		/// <param name="transa">operation op(A[i]) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B[i]) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A[i]) and C[i].</param>
		/// <param name="n">number of columns of op(B[i]) and C[i].</param>
		/// <param name="k">number of columns of op(A[i]) and rows of op(B[i]).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="Aarray">array of device pointers, with each array/device pointer of dim. lda x k with lda>=max(1,m) if
		/// transa==CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix A[i].</param>
		/// <param name="Barray">array of device pointers, with each array of dim. ldb x n with ldb>=max(1,k) if
		/// transa==CUBLAS_OP_N and ldb x k with ldb>=max(1,n) max(1,) otherwise.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store each matrix B[i].</param>
		/// <param name="beta">scalar used for multiplication. If beta == 0, C does not have to be a valid input.</param>
		/// <param name="Carray">array of device pointers. It has dimensions ldc x n with ldc>=max(1,m).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store each matrix C[i].</param>
		/// <param name="batchCount">number of pointers contained in A, B and C.</param>
		public void GemmBatched(Operation transa, Operation transb, int m, int n, int k, CudaDeviceVariable<double> alpha,
								   CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<CUdeviceptr> Barray, int ldb, CudaDeviceVariable<double> beta,
								   CudaDeviceVariable<CUdeviceptr> Carray, int ldc, int batchCount)
		{
			_status = CudaBlasNativeMethods.cublasDgemmBatched(_blasHandle, transa, transb, m, n, k, alpha.DevicePointer, Aarray.DevicePointer, lda, Barray.DevicePointer, ldb, beta.DevicePointer, Carray.DevicePointer, ldc, batchCount);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDgemmBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}


		/// <summary>
		/// This function performs the matrix-matrix multiplications of an array of matrices.
		/// where and are scalars, and , and are arrays of pointers to matrices stored
		/// in column-major format with dimensions op(A[i])m x k, op(B[i])k x n and op(C[i])m x n, 
		/// respectively.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor. For small sizes, typically smaller than 100x100,
		/// this function improves significantly performance compared to making calls to its
		/// corresponding cublas<![CDATA[<type>]]>gemm routine. However, on GPU architectures that support
		/// concurrent kernels, it might be advantageous to make multiple calls to cublas<![CDATA[<type>]]>gemm
		/// into different streams as the matrix sizes increase.
		/// </summary>
		/// <param name="transa">operation op(A[i]) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B[i]) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A[i]) and C[i].</param>
		/// <param name="n">number of columns of op(B[i]) and C[i].</param>
		/// <param name="k">number of columns of op(A[i]) and rows of op(B[i]).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="Aarray">array of device pointers, with each array/device pointer of dim. lda x k with lda>=max(1,m) if
		/// transa==CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix A[i].</param>
		/// <param name="Barray">array of device pointers, with each array of dim. ldb x n with ldb>=max(1,k) if
		/// transa==CUBLAS_OP_N and ldb x k with ldb>=max(1,n) max(1,) otherwise.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store each matrix B[i].</param>
		/// <param name="beta">scalar used for multiplication. If beta == 0, C does not have to be a valid input.</param>
		/// <param name="Carray">array of device pointers. It has dimensions ldc x n with ldc>=max(1,m).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store each matrix C[i].</param>
		/// <param name="batchCount">number of pointers contained in A, B and C.</param>
		public void GemmBatched(Operation transa, Operation transb, int m, int n, int k, CudaDeviceVariable<cuFloatComplex> alpha, 
			CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<CUdeviceptr> Barray, int ldb,
			CudaDeviceVariable<cuFloatComplex> beta, CudaDeviceVariable<CUdeviceptr> Carray, int ldc, int batchCount)
		{
			_status = CudaBlasNativeMethods.cublasCgemmBatched(_blasHandle, transa, transb, m, n, k, alpha.DevicePointer, Aarray.DevicePointer, lda, Barray.DevicePointer, ldb, beta.DevicePointer, Carray.DevicePointer, ldc, batchCount);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCgemmBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}


		/// <summary>
		/// This function performs the matrix-matrix multiplications of an array of matrices.
		/// where and are scalars, and , and are arrays of pointers to matrices stored
		/// in column-major format with dimensions op(A[i])m x k, op(B[i])k x n and op(C[i])m x n, 
		/// respectively.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor. For small sizes, typically smaller than 100x100,
		/// this function improves significantly performance compared to making calls to its
		/// corresponding cublas<![CDATA[<type>]]>gemm routine. However, on GPU architectures that support
		/// concurrent kernels, it might be advantageous to make multiple calls to cublas<![CDATA[<type>]]>gemm
		/// into different streams as the matrix sizes increase.
		/// </summary>
		/// <param name="transa">operation op(A[i]) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B[i]) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A[i]) and C[i].</param>
		/// <param name="n">number of columns of op(B[i]) and C[i].</param>
		/// <param name="k">number of columns of op(A[i]) and rows of op(B[i]).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="Aarray">array of device pointers, with each array/device pointer of dim. lda x k with lda>=max(1,m) if
		/// transa==CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix A[i].</param>
		/// <param name="Barray">array of device pointers, with each array of dim. ldb x n with ldb>=max(1,k) if
		/// transa==CUBLAS_OP_N and ldb x k with ldb>=max(1,n) max(1,) otherwise.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store each matrix B[i].</param>
		/// <param name="beta">scalar used for multiplication. If beta == 0, C does not have to be a valid input.</param>
		/// <param name="Carray">array of device pointers. It has dimensions ldc x n with ldc>=max(1,m).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store each matrix C[i].</param>
		/// <param name="batchCount">number of pointers contained in A, B and C.</param>
		public void GemmBatched(Operation transa, Operation transb, int m, int n, int k, CudaDeviceVariable<cuDoubleComplex> alpha,
			CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<CUdeviceptr> Barray, int ldb,
			CudaDeviceVariable<cuDoubleComplex> beta, CudaDeviceVariable<CUdeviceptr> Carray, int ldc, int batchCount)
		{
			_status = CudaBlasNativeMethods.cublasZgemmBatched(_blasHandle, transa, transb, m, n, k, alpha.DevicePointer, Aarray.DevicePointer, lda, Barray.DevicePointer, ldb, beta.DevicePointer, Carray.DevicePointer, ldc, batchCount);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZgemmBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		#endregion
		#region host pointer

		/// <summary>
		/// This function performs the matrix-matrix multiplications of an array of matrices.
		/// where and are scalars, and , and are arrays of pointers to matrices stored
		/// in column-major format with dimensions op(A[i])m x k, op(B[i])k x n and op(C[i])m x n, 
		/// respectively.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor. For small sizes, typically smaller than 100x100,
		/// this function improves significantly performance compared to making calls to its
		/// corresponding cublas<![CDATA[<type>]]>gemm routine. However, on GPU architectures that support
		/// concurrent kernels, it might be advantageous to make multiple calls to cublas<![CDATA[<type>]]>gemm
		/// into different streams as the matrix sizes increase.
		/// </summary>
		/// <param name="transa">operation op(A[i]) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B[i]) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A[i]) and C[i].</param>
		/// <param name="n">number of columns of op(B[i]) and C[i].</param>
		/// <param name="k">number of columns of op(A[i]) and rows of op(B[i]).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="Aarray">array of device pointers, with each array/device pointer of dim. lda x k with lda>=max(1,m) if
		/// transa==CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix A[i].</param>
		/// <param name="Barray">array of device pointers, with each array of dim. ldb x n with ldb>=max(1,k) if
		/// transa==CUBLAS_OP_N and ldb x k with ldb>=max(1,n) max(1,) otherwise.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store each matrix B[i].</param>
		/// <param name="beta">scalar used for multiplication. If beta == 0, C does not have to be a valid input.</param>
		/// <param name="Carray">array of device pointers. It has dimensions ldc x n with ldc>=max(1,m).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store each matrix C[i].</param>
		/// <param name="batchCount">number of pointers contained in A, B and C.</param>
		public void GemmBatched(Operation transa, Operation transb, int m, int n, int k, float alpha, CudaDeviceVariable<CUdeviceptr> Aarray,
			int lda, CudaDeviceVariable<CUdeviceptr> Barray, int ldb, float beta, CudaDeviceVariable<CUdeviceptr> Carray, int ldc, int batchCount)
		{
			_status = CudaBlasNativeMethods.cublasSgemmBatched(_blasHandle, transa, transb, m, n, k, ref alpha, Aarray.DevicePointer, lda, Barray.DevicePointer, ldb, ref beta, Carray.DevicePointer, ldc, batchCount);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSgemmBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}


		/// <summary>
		/// This function performs the matrix-matrix multiplications of an array of matrices.
		/// where and are scalars, and , and are arrays of pointers to matrices stored
		/// in column-major format with dimensions op(A[i])m x k, op(B[i])k x n and op(C[i])m x n, 
		/// respectively.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor. For small sizes, typically smaller than 100x100,
		/// this function improves significantly performance compared to making calls to its
		/// corresponding cublas<![CDATA[<type>]]>gemm routine. However, on GPU architectures that support
		/// concurrent kernels, it might be advantageous to make multiple calls to cublas<![CDATA[<type>]]>gemm
		/// into different streams as the matrix sizes increase.
		/// </summary>
		/// <param name="transa">operation op(A[i]) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B[i]) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A[i]) and C[i].</param>
		/// <param name="n">number of columns of op(B[i]) and C[i].</param>
		/// <param name="k">number of columns of op(A[i]) and rows of op(B[i]).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="Aarray">array of device pointers, with each array/device pointer of dim. lda x k with lda>=max(1,m) if
		/// transa==CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix A[i].</param>
		/// <param name="Barray">array of device pointers, with each array of dim. ldb x n with ldb>=max(1,k) if
		/// transa==CUBLAS_OP_N and ldb x k with ldb>=max(1,n) max(1,) otherwise.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store each matrix B[i].</param>
		/// <param name="beta">scalar used for multiplication. If beta == 0, C does not have to be a valid input.</param>
		/// <param name="Carray">array of device pointers. It has dimensions ldc x n with ldc>=max(1,m).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store each matrix C[i].</param>
		/// <param name="batchCount">number of pointers contained in A, B and C.</param>
		public void GemmBatched(Operation transa, Operation transb, int m, int n, int k, double alpha,
			CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<CUdeviceptr> Barray, int ldb, double beta,
			CudaDeviceVariable<CUdeviceptr> Carray, int ldc, int batchCount)
		{
			_status = CudaBlasNativeMethods.cublasDgemmBatched(_blasHandle, transa, transb, m, n, k, ref alpha, Aarray.DevicePointer, lda, Barray.DevicePointer, ldb, ref beta, Carray.DevicePointer, ldc, batchCount);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDgemmBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}


		/// <summary>
		/// This function performs the matrix-matrix multiplications of an array of matrices.
		/// where and are scalars, and , and are arrays of pointers to matrices stored
		/// in column-major format with dimensions op(A[i])m x k, op(B[i])k x n and op(C[i])m x n, 
		/// respectively.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor. For small sizes, typically smaller than 100x100,
		/// this function improves significantly performance compared to making calls to its
		/// corresponding cublas<![CDATA[<type>]]>gemm routine. However, on GPU architectures that support
		/// concurrent kernels, it might be advantageous to make multiple calls to cublas<![CDATA[<type>]]>gemm
		/// into different streams as the matrix sizes increase.
		/// </summary>
		/// <param name="transa">operation op(A[i]) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B[i]) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A[i]) and C[i].</param>
		/// <param name="n">number of columns of op(B[i]) and C[i].</param>
		/// <param name="k">number of columns of op(A[i]) and rows of op(B[i]).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="Aarray">array of device pointers, with each array/device pointer of dim. lda x k with lda>=max(1,m) if
		/// transa==CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix A[i].</param>
		/// <param name="Barray">array of device pointers, with each array of dim. ldb x n with ldb>=max(1,k) if
		/// transa==CUBLAS_OP_N and ldb x k with ldb>=max(1,n) max(1,) otherwise.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store each matrix B[i].</param>
		/// <param name="beta">scalar used for multiplication. If beta == 0, C does not have to be a valid input.</param>
		/// <param name="Carray">array of device pointers. It has dimensions ldc x n with ldc>=max(1,m).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store each matrix C[i].</param>
		/// <param name="batchCount">number of pointers contained in A, B and C.</param>
		public void GemmBatched(Operation transa, Operation transb, int m, int n, int k, cuFloatComplex alpha,
			CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<CUdeviceptr> Barray, int ldb,
			cuFloatComplex beta, CudaDeviceVariable<CUdeviceptr> Carray, int ldc, int batchCount)
		{
			_status = CudaBlasNativeMethods.cublasCgemmBatched(_blasHandle, transa, transb, m, n, k, ref alpha, Aarray.DevicePointer, lda, Barray.DevicePointer, ldb, ref beta, Carray.DevicePointer, ldc, batchCount);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCgemmBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}


		/// <summary>
		/// This function performs the matrix-matrix multiplications of an array of matrices.
		/// where and are scalars, and , and are arrays of pointers to matrices stored
		/// in column-major format with dimensions op(A[i])m x k, op(B[i])k x n and op(C[i])m x n, 
		/// respectively.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor. For small sizes, typically smaller than 100x100,
		/// this function improves significantly performance compared to making calls to its
		/// corresponding cublas<![CDATA[<type>]]>gemm routine. However, on GPU architectures that support
		/// concurrent kernels, it might be advantageous to make multiple calls to cublas<![CDATA[<type>]]>gemm
		/// into different streams as the matrix sizes increase.
		/// </summary>
		/// <param name="transa">operation op(A[i]) that is non- or (conj.) transpose.</param>
		/// <param name="transb">operation op(B[i]) that is non- or (conj.) transpose.</param>
		/// <param name="m">number of rows of matrix op(A[i]) and C[i].</param>
		/// <param name="n">number of columns of op(B[i]) and C[i].</param>
		/// <param name="k">number of columns of op(A[i]) and rows of op(B[i]).</param>
		/// <param name="alpha">scalar used for multiplication.</param>
		/// <param name="Aarray">array of device pointers, with each array/device pointer of dim. lda x k with lda>=max(1,m) if
		/// transa==CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix A[i].</param>
		/// <param name="Barray">array of device pointers, with each array of dim. ldb x n with ldb>=max(1,k) if
		/// transa==CUBLAS_OP_N and ldb x k with ldb>=max(1,n) max(1,) otherwise.</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store each matrix B[i].</param>
		/// <param name="beta">scalar used for multiplication. If beta == 0, C does not have to be a valid input.</param>
		/// <param name="Carray">array of device pointers. It has dimensions ldc x n with ldc>=max(1,m).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store each matrix C[i].</param>
		/// <param name="batchCount">number of pointers contained in A, B and C.</param>
		public void GemmBatched(Operation transa, Operation transb, int m, int n, int k, cuDoubleComplex alpha,
			CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<CUdeviceptr> Barray, int ldb,
			cuDoubleComplex beta, CudaDeviceVariable<CUdeviceptr> Carray, int ldc, int batchCount)
		{
			_status = CudaBlasNativeMethods.cublasZgemmBatched(_blasHandle, transa, transb, m, n, k, ref alpha, Aarray.DevicePointer, lda, Barray.DevicePointer, ldb, ref beta, Carray.DevicePointer, ldc, batchCount);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZgemmBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		#endregion
		#endregion

		#region Batched LU - GETRF
		/// <summary>
		/// This function performs the LU factorization of an array of n x n matrices.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor. The current implementation limits the dimension n to 32.
		/// </summary>
		/// <param name="n">number of rows and columns of A[i].</param>
		/// <param name="A">array of device pointers with each array/device pointer of dim. n x n with lda>=max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix A[i].</param>
		/// <param name="P">array of size n x batchSize that contains the permutation vector 
		/// of each factorization of A[i] stored in a linear fashion.</param>
		/// <param name="INFO">If info=0, the execution is successful.<para/>
		/// If info = -i, the i-th parameter had an illegal value.<para/>
		/// If info = i, aii is 0. The factorization has been completed, but U is exactly singular.</param>
		/// <param name="batchSize">number of pointers contained in A</param>
		public void GetrfBatchedS(int n, CudaDeviceVariable<CUdeviceptr> A, int lda, CudaDeviceVariable<int> P,
			CudaDeviceVariable<int> INFO, int batchSize)
		{
			_status = CudaBlasNativeMethods.cublasSgetrfBatched(_blasHandle, n, A.DevicePointer, lda, P.DevicePointer, INFO.DevicePointer, batchSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSgetrfBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the LU factorization of an array of n x n matrices.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor. The current implementation limits the dimension n to 32.
		/// </summary>
		/// <param name="n">number of rows and columns of A[i].</param>
		/// <param name="A">array of device pointers with each array/device pointer of dim. n x n with lda>=max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix A[i].</param>
		/// <param name="P">array of size n x batchSize that contains the permutation vector 
		/// of each factorization of A[i] stored in a linear fashion.</param>
		/// <param name="INFO">If info=0, the execution is successful.<para/>
		/// If info = -i, the i-th parameter had an illegal value.<para/>
		/// If info = i, aii is 0. The factorization has been completed, but U is exactly singular.</param>
		/// <param name="batchSize">number of pointers contained in A</param>
		public void GetrfBatchedD(int n, CudaDeviceVariable<CUdeviceptr> A, int lda, CudaDeviceVariable<int> P,
			CudaDeviceVariable<int> INFO, int batchSize)
		{
			_status = CudaBlasNativeMethods.cublasDgetrfBatched(_blasHandle, n, A.DevicePointer, lda, P.DevicePointer, INFO.DevicePointer, batchSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDgetrfBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the LU factorization of an array of n x n matrices.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor. The current implementation limits the dimension n to 32.
		/// </summary>
		/// <param name="n">number of rows and columns of A[i].</param>
		/// <param name="A">array of device pointers with each array/device pointer of dim. n x n with lda>=max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix A[i].</param>
		/// <param name="P">array of size n x batchSize that contains the permutation vector 
		/// of each factorization of A[i] stored in a linear fashion.</param>
		/// <param name="INFO">If info=0, the execution is successful.<para/>
		/// If info = -i, the i-th parameter had an illegal value.<para/>
		/// If info = i, aii is 0. The factorization has been completed, but U is exactly singular.</param>
		/// <param name="batchSize">number of pointers contained in A</param>
		public void GetrfBatchedC(int n, CudaDeviceVariable<CUdeviceptr> A, int lda,
			CudaDeviceVariable<int> P, CudaDeviceVariable<int> INFO, int batchSize)
		{
			_status = CudaBlasNativeMethods.cublasCgetrfBatched(_blasHandle, n, A.DevicePointer, lda, P.DevicePointer, INFO.DevicePointer, batchSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCgetrfBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the LU factorization of an array of n x n matrices.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor. The current implementation limits the dimension n to 32.
		/// </summary>
		/// <param name="n">number of rows and columns of A[i].</param>
		/// <param name="A">array of device pointers with each array/device pointer of dim. n x n with lda>=max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix A[i].</param>
		/// <param name="P">array of size n x batchSize that contains the permutation vector 
		/// of each factorization of A[i] stored in a linear fashion.</param>
		/// <param name="INFO">If info=0, the execution is successful.<para/>
		/// If info = -i, the i-th parameter had an illegal value.<para/>
		/// If info = i, aii is 0. The factorization has been completed, but U is exactly singular.</param>
		/// <param name="batchSize">number of pointers contained in A</param>
		public void GetrfBatchedZ(int n, CudaDeviceVariable<CUdeviceptr> A, int lda,
			CudaDeviceVariable<int> P, CudaDeviceVariable<int> INFO, int batchSize)
		{
			_status = CudaBlasNativeMethods.cublasZgetrfBatched(_blasHandle, n, A.DevicePointer, lda, P.DevicePointer, INFO.DevicePointer, batchSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZgetrfBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		#endregion

		#region Batched inversion based on LU factorization from getrf

		/// <summary>
		/// Aarray and Carray are arrays of pointers to matrices stored in column-major format
		/// with dimensions n*n and leading dimension lda and ldc respectively.
		/// This function performs the inversion of matrices A[i] for i = 0, ..., batchSize-1.<para/>
		/// Prior to calling GetriBatched, the matrix A[i] must be factorized first using
		/// the routine GetrfBatched. After the call of GetrfBatched, the matrix
		/// pointing by Aarray[i] will contain the LU factors of the matrix A[i] and the vector
		/// pointing by (PivotArray+i) will contain the pivoting sequence.<para/>
		/// Following the LU factorization, GetriBatched uses forward and backward
		/// triangular solvers to complete inversion of matrices A[i] for i = 0, ..., batchSize-1. The
		/// inversion is out-of-place, so memory space of Carray[i] cannot overlap memory space of
		/// Array[i].
		/// </summary>
		/// <param name="n">number of rows and columns of Aarray[i].</param>
		/// <param name="Aarray">array of pointers to array, with each array of dimension n*n with lda>=max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix Aarray[i].</param>
		/// <param name="P">array of size n*batchSize that contains the pivoting sequence of each factorization of Aarray[i] stored in a linear fashion.</param>
		/// <param name="Carray">array of pointers to array, with each array of dimension n*n with ldc>=max(1,n).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store each matrix Carray[i].</param>
		/// <param name="INFO">array of size batchSize that info(=infoArray[i]) contains the information of inversion of A[i].<para/>
		/// If info=0, the execution is successful.<para/>
		/// If info = k, U(k,k) is 0. The U is exactly singular and the inversion failed.</param>
		/// <param name="batchSize">number of pointers contained in A</param>
		public void GetriBatchedS(int n, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<int> P,
			CudaDeviceVariable<CUdeviceptr> Carray, int ldc, CudaDeviceVariable<int> INFO, int batchSize)
		{
			_status = CudaBlasNativeMethods.cublasSgetriBatched(_blasHandle, n, Aarray.DevicePointer, lda, P.DevicePointer, Carray.DevicePointer, ldc, INFO.DevicePointer, batchSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSgetriBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// Aarray and Carray are arrays of pointers to matrices stored in column-major format
		/// with dimensions n*n and leading dimension lda and ldc respectively.
		/// This function performs the inversion of matrices A[i] for i = 0, ..., batchSize-1.<para/>
		/// Prior to calling GetriBatched, the matrix A[i] must be factorized first using
		/// the routine GetrfBatched. After the call of GetrfBatched, the matrix
		/// pointing by Aarray[i] will contain the LU factors of the matrix A[i] and the vector
		/// pointing by (PivotArray+i) will contain the pivoting sequence.<para/>
		/// Following the LU factorization, GetriBatched uses forward and backward
		/// triangular solvers to complete inversion of matrices A[i] for i = 0, ..., batchSize-1. The
		/// inversion is out-of-place, so memory space of Carray[i] cannot overlap memory space of
		/// Array[i].
		/// </summary>
		/// <param name="n">number of rows and columns of Aarray[i].</param>
		/// <param name="Aarray">array of pointers to array, with each array of dimension n*n with lda>=max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix Aarray[i].</param>
		/// <param name="P">array of size n*batchSize that contains the pivoting sequence of each factorization of Aarray[i] stored in a linear fashion.</param>
		/// <param name="Carray">array of pointers to array, with each array of dimension n*n with ldc>=max(1,n).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store each matrix Carray[i].</param>
		/// <param name="INFO">array of size batchSize that info(=infoArray[i]) contains the information of inversion of A[i].<para/>
		/// If info=0, the execution is successful.<para/>
		/// If info = k, U(k,k) is 0. The U is exactly singular and the inversion failed.</param>
		/// <param name="batchSize">number of pointers contained in A</param>
		public void GetriBatchedD(int n, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<int> P,
			CudaDeviceVariable<CUdeviceptr> Carray, int ldc, CudaDeviceVariable<int> INFO, int batchSize)
		{
			_status = CudaBlasNativeMethods.cublasDgetriBatched(_blasHandle, n, Aarray.DevicePointer, lda, P.DevicePointer, Carray.DevicePointer, ldc, INFO.DevicePointer, batchSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDgetriBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// Aarray and Carray are arrays of pointers to matrices stored in column-major format
		/// with dimensions n*n and leading dimension lda and ldc respectively.
		/// This function performs the inversion of matrices A[i] for i = 0, ..., batchSize-1.<para/>
		/// Prior to calling GetriBatched, the matrix A[i] must be factorized first using
		/// the routine GetrfBatched. After the call of GetrfBatched, the matrix
		/// pointing by Aarray[i] will contain the LU factors of the matrix A[i] and the vector
		/// pointing by (PivotArray+i) will contain the pivoting sequence.<para/>
		/// Following the LU factorization, GetriBatched uses forward and backward
		/// triangular solvers to complete inversion of matrices A[i] for i = 0, ..., batchSize-1. The
		/// inversion is out-of-place, so memory space of Carray[i] cannot overlap memory space of
		/// Array[i].
		/// </summary>
		/// <param name="n">number of rows and columns of Aarray[i].</param>
		/// <param name="Aarray">array of pointers to array, with each array of dimension n*n with lda>=max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix Aarray[i].</param>
		/// <param name="P">array of size n*batchSize that contains the pivoting sequence of each factorization of Aarray[i] stored in a linear fashion.</param>
		/// <param name="Carray">array of pointers to array, with each array of dimension n*n with ldc>=max(1,n).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store each matrix Carray[i].</param>
		/// <param name="INFO">array of size batchSize that info(=infoArray[i]) contains the information of inversion of A[i].<para/>
		/// If info=0, the execution is successful.<para/>
		/// If info = k, U(k,k) is 0. The U is exactly singular and the inversion failed.</param>
		/// <param name="batchSize">number of pointers contained in A</param>
		public void GetriBatchedC(int n, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<int> P,
			CudaDeviceVariable<CUdeviceptr> Carray, int ldc, CudaDeviceVariable<int> INFO, int batchSize)
		{
			_status = CudaBlasNativeMethods.cublasCgetriBatched(_blasHandle, n, Aarray.DevicePointer, lda, P.DevicePointer, Carray.DevicePointer, ldc, INFO.DevicePointer, batchSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCgetriBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// Aarray and Carray are arrays of pointers to matrices stored in column-major format
		/// with dimensions n*n and leading dimension lda and ldc respectively.
		/// This function performs the inversion of matrices A[i] for i = 0, ..., batchSize-1.<para/>
		/// Prior to calling GetriBatched, the matrix A[i] must be factorized first using
		/// the routine GetrfBatched. After the call of GetrfBatched, the matrix
		/// pointing by Aarray[i] will contain the LU factors of the matrix A[i] and the vector
		/// pointing by (PivotArray+i) will contain the pivoting sequence.<para/>
		/// Following the LU factorization, GetriBatched uses forward and backward
		/// triangular solvers to complete inversion of matrices A[i] for i = 0, ..., batchSize-1. The
		/// inversion is out-of-place, so memory space of Carray[i] cannot overlap memory space of
		/// Array[i].
		/// </summary>
		/// <param name="n">number of rows and columns of Aarray[i].</param>
		/// <param name="Aarray">array of pointers to array, with each array of dimension n*n with lda>=max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix Aarray[i].</param>
		/// <param name="P">array of size n*batchSize that contains the pivoting sequence of each factorization of Aarray[i] stored in a linear fashion.</param>
		/// <param name="Carray">array of pointers to array, with each array of dimension n*n with ldc>=max(1,n).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store each matrix Carray[i].</param>
		/// <param name="INFO">array of size batchSize that info(=infoArray[i]) contains the information of inversion of A[i].<para/>
		/// If info=0, the execution is successful.<para/>
		/// If info = k, U(k,k) is 0. The U is exactly singular and the inversion failed.</param>
		/// <param name="batchSize">number of pointers contained in A</param>
		public void GetriBatchedZ(int n, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<int> P,
			CudaDeviceVariable<CUdeviceptr> Carray, int ldc, CudaDeviceVariable<int> INFO, int batchSize)
		{
			_status = CudaBlasNativeMethods.cublasZgetriBatched(_blasHandle, n, Aarray.DevicePointer, lda, P.DevicePointer, Carray.DevicePointer, ldc, INFO.DevicePointer, batchSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZgetriBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		#endregion

		#region TRSM - Batched Triangular Solver
		#region device pointer
		/// <summary>
		/// This function solves an array of triangular linear systems with multiple right-hand-sides.<para/>
		/// The solution overwrites the right-hand-sides on exit.<para/>
		/// No test for singularity or near-singularity is included in this function.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor. The current implementation limits the dimensions m and n to 32.
		/// </summary>
		/// <param name="side">indicates if matrix A[i] is on the left or right of X[i].</param>
		/// <param name="uplo">indicates if matrix A[i] lower or upper part is stored, the
		/// other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A[i]) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix
		/// A[i] are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B[i], with matrix A[i] sized accordingly.</param>
		/// <param name="n">number of columns of matrix B[i], with matrix A[i] is sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication, if alpha==0 then A[i] is not 
		/// referenced and B[i] does not have to be a valid input.</param>
		/// <param name="A">array of device pointers with each array/device pointerarray 
		/// of dim. lda x m with lda>=max(1,m) if transa==CUBLAS_OP_N and lda x n with
		/// lda>=max(1,n) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A[i].</param>
		/// <param name="B">array of device pointers with each array/device pointerarrayof dim.
		/// ldb x n with ldb>=max(1,m)</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B[i].</param>
		/// <param name="batchCount"></param>
		public void TrsmBatched(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, CudaDeviceVariable<float> alpha,
			CudaDeviceVariable<CUdeviceptr> A, int lda, CudaDeviceVariable<CUdeviceptr> B, int ldb, int batchCount)
		{
			_status = CudaBlasNativeMethods.cublasStrsmBatched(_blasHandle, side, uplo, trans, diag, m, n, alpha.DevicePointer,
				A.DevicePointer, lda, B.DevicePointer, ldb, batchCount);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasStrsmBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function solves an array of triangular linear systems with multiple right-hand-sides.<para/>
		/// The solution overwrites the right-hand-sides on exit.<para/>
		/// No test for singularity or near-singularity is included in this function.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor. The current implementation limits the dimensions m and n to 32.
		/// </summary>
		/// <param name="side">indicates if matrix A[i] is on the left or right of X[i].</param>
		/// <param name="uplo">indicates if matrix A[i] lower or upper part is stored, the
		/// other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A[i]) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix
		/// A[i] are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B[i], with matrix A[i] sized accordingly.</param>
		/// <param name="n">number of columns of matrix B[i], with matrix A[i] is sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication, if alpha==0 then A[i] is not 
		/// referenced and B[i] does not have to be a valid input.</param>
		/// <param name="A">array of device pointers with each array/device pointerarray 
		/// of dim. lda x m with lda>=max(1,m) if transa==CUBLAS_OP_N and lda x n with
		/// lda>=max(1,n) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A[i].</param>
		/// <param name="B">array of device pointers with each array/device pointerarrayof dim.
		/// ldb x n with ldb>=max(1,m)</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B[i].</param>
		/// <param name="batchCount"></param>
		public void TrsmBatched(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, CudaDeviceVariable<double> alpha,
			CudaDeviceVariable<CUdeviceptr> A, int lda, CudaDeviceVariable<CUdeviceptr> B, int ldb, int batchCount)
		{
			_status = CudaBlasNativeMethods.cublasDtrsmBatched(_blasHandle, side, uplo, trans, diag, m, n, alpha.DevicePointer,
				A.DevicePointer, lda, B.DevicePointer, ldb, batchCount);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDtrsmBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function solves an array of triangular linear systems with multiple right-hand-sides.<para/>
		/// The solution overwrites the right-hand-sides on exit.<para/>
		/// No test for singularity or near-singularity is included in this function.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor. The current implementation limits the dimensions m and n to 32.
		/// </summary>
		/// <param name="side">indicates if matrix A[i] is on the left or right of X[i].</param>
		/// <param name="uplo">indicates if matrix A[i] lower or upper part is stored, the
		/// other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A[i]) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix
		/// A[i] are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B[i], with matrix A[i] sized accordingly.</param>
		/// <param name="n">number of columns of matrix B[i], with matrix A[i] is sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication, if alpha==0 then A[i] is not 
		/// referenced and B[i] does not have to be a valid input.</param>
		/// <param name="A">array of device pointers with each array/device pointerarray 
		/// of dim. lda x m with lda>=max(1,m) if transa==CUBLAS_OP_N and lda x n with
		/// lda>=max(1,n) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A[i].</param>
		/// <param name="B">array of device pointers with each array/device pointerarrayof dim.
		/// ldb x n with ldb>=max(1,m)</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B[i].</param>
		/// <param name="batchCount"></param>
		public void TrsmBatched(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, CudaDeviceVariable<cuFloatComplex> alpha,
			CudaDeviceVariable<CUdeviceptr> A, int lda, CudaDeviceVariable<CUdeviceptr> B, int ldb, int batchCount)
		{
			_status = CudaBlasNativeMethods.cublasCtrsmBatched(_blasHandle, side, uplo, trans, diag, m, n, alpha.DevicePointer,
				A.DevicePointer, lda, B.DevicePointer, ldb, batchCount);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCtrsmBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function solves an array of triangular linear systems with multiple right-hand-sides.<para/>
		/// The solution overwrites the right-hand-sides on exit.<para/>
		/// No test for singularity or near-singularity is included in this function.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor. The current implementation limits the dimensions m and n to 32.
		/// </summary>
		/// <param name="side">indicates if matrix A[i] is on the left or right of X[i].</param>
		/// <param name="uplo">indicates if matrix A[i] lower or upper part is stored, the
		/// other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A[i]) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix
		/// A[i] are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B[i], with matrix A[i] sized accordingly.</param>
		/// <param name="n">number of columns of matrix B[i], with matrix A[i] is sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication, if alpha==0 then A[i] is not 
		/// referenced and B[i] does not have to be a valid input.</param>
		/// <param name="A">array of device pointers with each array/device pointerarray 
		/// of dim. lda x m with lda>=max(1,m) if transa==CUBLAS_OP_N and lda x n with
		/// lda>=max(1,n) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A[i].</param>
		/// <param name="B">array of device pointers with each array/device pointerarrayof dim.
		/// ldb x n with ldb>=max(1,m)</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B[i].</param>
		/// <param name="batchCount"></param>
		public void TrsmBatched(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, CudaDeviceVariable<cuDoubleComplex> alpha,
			CudaDeviceVariable<CUdeviceptr> A, int lda, CudaDeviceVariable<CUdeviceptr> B, int ldb, int batchCount)
		{
			_status = CudaBlasNativeMethods.cublasZtrsmBatched(_blasHandle, side, uplo, trans, diag, m, n, alpha.DevicePointer,
				A.DevicePointer, lda, B.DevicePointer, ldb, batchCount);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZtrsmBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}


		#endregion
		#region host pointer
		/// <summary>
		/// This function solves an array of triangular linear systems with multiple right-hand-sides.<para/>
		/// The solution overwrites the right-hand-sides on exit.<para/>
		/// No test for singularity or near-singularity is included in this function.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor. The current implementation limits the dimensions m and n to 32.
		/// </summary>
		/// <param name="side">indicates if matrix A[i] is on the left or right of X[i].</param>
		/// <param name="uplo">indicates if matrix A[i] lower or upper part is stored, the
		/// other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A[i]) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix
		/// A[i] are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B[i], with matrix A[i] sized accordingly.</param>
		/// <param name="n">number of columns of matrix B[i], with matrix A[i] is sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication, if alpha==0 then A[i] is not 
		/// referenced and B[i] does not have to be a valid input.</param>
		/// <param name="A">array of device pointers with each array/device pointerarray 
		/// of dim. lda x m with lda>=max(1,m) if transa==CUBLAS_OP_N and lda x n with
		/// lda>=max(1,n) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A[i].</param>
		/// <param name="B">array of device pointers with each array/device pointerarrayof dim.
		/// ldb x n with ldb>=max(1,m)</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B[i].</param>
		/// <param name="batchCount"></param>
		public void TrsmBatched(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, ref float alpha,
			CudaDeviceVariable<CUdeviceptr> A, int lda, CudaDeviceVariable<CUdeviceptr> B, int ldb, int batchCount)
		{
			_status = CudaBlasNativeMethods.cublasStrsmBatched(_blasHandle, side, uplo, trans, diag, m, n, ref alpha,
				A.DevicePointer, lda, B.DevicePointer, ldb, batchCount);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasStrsmBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function solves an array of triangular linear systems with multiple right-hand-sides.<para/>
		/// The solution overwrites the right-hand-sides on exit.<para/>
		/// No test for singularity or near-singularity is included in this function.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor. The current implementation limits the dimensions m and n to 32.
		/// </summary>
		/// <param name="side">indicates if matrix A[i] is on the left or right of X[i].</param>
		/// <param name="uplo">indicates if matrix A[i] lower or upper part is stored, the
		/// other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A[i]) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix
		/// A[i] are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B[i], with matrix A[i] sized accordingly.</param>
		/// <param name="n">number of columns of matrix B[i], with matrix A[i] is sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication, if alpha==0 then A[i] is not 
		/// referenced and B[i] does not have to be a valid input.</param>
		/// <param name="A">array of device pointers with each array/device pointerarray 
		/// of dim. lda x m with lda>=max(1,m) if transa==CUBLAS_OP_N and lda x n with
		/// lda>=max(1,n) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A[i].</param>
		/// <param name="B">array of device pointers with each array/device pointerarrayof dim.
		/// ldb x n with ldb>=max(1,m)</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B[i].</param>
		/// <param name="batchCount"></param>
		public void TrsmBatched(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, ref double alpha,
			CudaDeviceVariable<CUdeviceptr> A, int lda, CudaDeviceVariable<CUdeviceptr> B, int ldb, int batchCount)
		{
			_status = CudaBlasNativeMethods.cublasDtrsmBatched(_blasHandle, side, uplo, trans, diag, m, n, ref alpha,
				A.DevicePointer, lda, B.DevicePointer, ldb, batchCount);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDtrsmBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function solves an array of triangular linear systems with multiple right-hand-sides.<para/>
		/// The solution overwrites the right-hand-sides on exit.<para/>
		/// No test for singularity or near-singularity is included in this function.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor. The current implementation limits the dimensions m and n to 32.
		/// </summary>
		/// <param name="side">indicates if matrix A[i] is on the left or right of X[i].</param>
		/// <param name="uplo">indicates if matrix A[i] lower or upper part is stored, the
		/// other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A[i]) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix
		/// A[i] are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B[i], with matrix A[i] sized accordingly.</param>
		/// <param name="n">number of columns of matrix B[i], with matrix A[i] is sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication, if alpha==0 then A[i] is not 
		/// referenced and B[i] does not have to be a valid input.</param>
		/// <param name="A">array of device pointers with each array/device pointerarray 
		/// of dim. lda x m with lda>=max(1,m) if transa==CUBLAS_OP_N and lda x n with
		/// lda>=max(1,n) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A[i].</param>
		/// <param name="B">array of device pointers with each array/device pointerarrayof dim.
		/// ldb x n with ldb>=max(1,m)</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B[i].</param>
		/// <param name="batchCount"></param>
		public void TrsmBatched(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, ref cuFloatComplex alpha,
			CudaDeviceVariable<CUdeviceptr> A, int lda, CudaDeviceVariable<CUdeviceptr> B, int ldb, int batchCount)
		{
			_status = CudaBlasNativeMethods.cublasCtrsmBatched(_blasHandle, side, uplo, trans, diag, m, n, ref alpha,
				A.DevicePointer, lda, B.DevicePointer, ldb, batchCount);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCtrsmBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function solves an array of triangular linear systems with multiple right-hand-sides.<para/>
		/// The solution overwrites the right-hand-sides on exit.<para/>
		/// No test for singularity or near-singularity is included in this function.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor. The current implementation limits the dimensions m and n to 32.
		/// </summary>
		/// <param name="side">indicates if matrix A[i] is on the left or right of X[i].</param>
		/// <param name="uplo">indicates if matrix A[i] lower or upper part is stored, the
		/// other part is not referenced and is inferred from the stored elements.</param>
		/// <param name="trans">operation op(A[i]) that is non- or (conj.) transpose.</param>
		/// <param name="diag">indicates if the elements on the main diagonal of matrix
		/// A[i] are unity and should not be accessed.</param>
		/// <param name="m">number of rows of matrix B[i], with matrix A[i] sized accordingly.</param>
		/// <param name="n">number of columns of matrix B[i], with matrix A[i] is sized accordingly.</param>
		/// <param name="alpha">scalar used for multiplication, if alpha==0 then A[i] is not 
		/// referenced and B[i] does not have to be a valid input.</param>
		/// <param name="A">array of device pointers with each array/device pointerarray 
		/// of dim. lda x m with lda>=max(1,m) if transa==CUBLAS_OP_N and lda x n with
		/// lda>=max(1,n) otherwise.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A[i].</param>
		/// <param name="B">array of device pointers with each array/device pointerarrayof dim.
		/// ldb x n with ldb>=max(1,m)</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store matrix B[i].</param>
		/// <param name="batchCount"></param>
		public void TrsmBatched(SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, ref cuDoubleComplex alpha,
			CudaDeviceVariable<CUdeviceptr> A, int lda, CudaDeviceVariable<CUdeviceptr> B, int ldb, int batchCount)
		{
			_status = CudaBlasNativeMethods.cublasZtrsmBatched(_blasHandle, side, uplo, trans, diag, m, n, ref alpha,
				A.DevicePointer, lda, B.DevicePointer, ldb, batchCount);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZtrsmBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}


		#endregion
		#endregion

		#region TPTTR : Triangular Pack format to Triangular format
		/// <summary>
		/// This function performs the conversion from the triangular packed format to the
		/// triangular format.<para/>
		/// If uplo == CUBLAS_FILL_MODE_LOWER then the elements of AP are copied into the
		/// lower triangular part of the triangular matrix A and the upper part of A is left untouched.<para/>
		/// If uplo == CUBLAS_FILL_MODE_UPPER then the elements of AP are copied into the
		/// upper triangular part of the triangular matrix A and the lower part of A is left untouched.
		/// </summary>
		/// <param name="uplo">indicates if matrix AP contains lower or upper part of matrix A.</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		/// <param name="A">array of dimensions lda x n , with lda&gt;=max(1,n). The
		/// opposite side of A is left untouched.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Stpttr(FillMode uplo, int n, CudaDeviceVariable<float> AP, CudaDeviceVariable<float> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasStpttr(_blasHandle, uplo, n, AP.DevicePointer, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasStpttr", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the conversion from the triangular packed format to the
		/// triangular format.<para/>
		/// If uplo == CUBLAS_FILL_MODE_LOWER then the elements of AP are copied into the
		/// lower triangular part of the triangular matrix A and the upper part of A is left untouched.<para/>
		/// If uplo == CUBLAS_FILL_MODE_UPPER then the elements of AP are copied into the
		/// upper triangular part of the triangular matrix A and the lower part of A is left untouched.
		/// </summary>
		/// <param name="uplo">indicates if matrix AP contains lower or upper part of matrix A.</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		/// <param name="A">array of dimensions lda x n , with lda&gt;=max(1,n). The
		/// opposite side of A is left untouched.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Dtpttr(FillMode uplo, int n, CudaDeviceVariable<double> AP, CudaDeviceVariable<double> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasDtpttr(_blasHandle, uplo, n, AP.DevicePointer, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDtpttr", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the conversion from the triangular packed format to the
		/// triangular format.<para/>
		/// If uplo == CUBLAS_FILL_MODE_LOWER then the elements of AP are copied into the
		/// lower triangular part of the triangular matrix A and the upper part of A is left untouched.<para/>
		/// If uplo == CUBLAS_FILL_MODE_UPPER then the elements of AP are copied into the
		/// upper triangular part of the triangular matrix A and the lower part of A is left untouched.
		/// </summary>
		/// <param name="uplo">indicates if matrix AP contains lower or upper part of matrix A.</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		/// <param name="A">array of dimensions lda x n , with lda&gt;=max(1,n). The
		/// opposite side of A is left untouched.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Ctpttr(FillMode uplo, int n, CudaDeviceVariable<cuFloatComplex> AP, CudaDeviceVariable<cuFloatComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasCtpttr(_blasHandle, uplo, n, AP.DevicePointer, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCtpttr", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the conversion from the triangular packed format to the
		/// triangular format.<para/>
		/// If uplo == CUBLAS_FILL_MODE_LOWER then the elements of AP are copied into the
		/// lower triangular part of the triangular matrix A and the upper part of A is left untouched.<para/>
		/// If uplo == CUBLAS_FILL_MODE_UPPER then the elements of AP are copied into the
		/// upper triangular part of the triangular matrix A and the lower part of A is left untouched.
		/// </summary>
		/// <param name="uplo">indicates if matrix AP contains lower or upper part of matrix A.</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		/// <param name="A">array of dimensions lda x n , with lda&gt;=max(1,n). The
		/// opposite side of A is left untouched.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		public void Ztpttr(FillMode uplo, int n, CudaDeviceVariable<cuDoubleComplex> AP, CudaDeviceVariable<cuDoubleComplex> A, int lda)
		{
			_status = CudaBlasNativeMethods.cublasZtpttr(_blasHandle, uplo, n, AP.DevicePointer, A.DevicePointer, lda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZtpttr", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion

		#region TRTTP : Triangular format to Triangular Pack format
		/// <summary>
		/// This function performs the conversion from the triangular format to the triangular
		/// packed format.<para/>
		/// If uplo == CUBLAS_FILL_MODE_LOWER then the lower triangular part of the triangular
		/// matrix A is copied into the array AP. <para/>If uplo == CUBLAS_FILL_MODE_UPPER then then
		/// the upper triangular part of the triangular matrix A is copied into the array AP
		/// </summary>
		/// <param name="uplo">indicates which matrix A lower or upper part is referenced</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="A">array of dimensions lda x n , with lda&gt;=max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		public void Strttp(FillMode uplo, int n, CudaDeviceVariable<float> A, int lda, CudaDeviceVariable<float> AP)
		{
			_status = CudaBlasNativeMethods.cublasStrttp(_blasHandle, uplo, n, A.DevicePointer, lda, AP.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasStrttp", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the conversion from the triangular format to the triangular
		/// packed format.<para/>
		/// If uplo == CUBLAS_FILL_MODE_LOWER then the lower triangular part of the triangular
		/// matrix A is copied into the array AP. <para/>If uplo == CUBLAS_FILL_MODE_UPPER then then
		/// the upper triangular part of the triangular matrix A is copied into the array AP
		/// </summary>
		/// <param name="uplo">indicates which matrix A lower or upper part is referenced</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="A">array of dimensions lda x n , with lda&gt;=max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		public void Dtrttp(FillMode uplo, int n, CudaDeviceVariable<double> A, int lda, CudaDeviceVariable<float> AP)
		{
			_status = CudaBlasNativeMethods.cublasDtrttp(_blasHandle, uplo, n, A.DevicePointer, lda, AP.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDtrttp", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the conversion from the triangular format to the triangular
		/// packed format.<para/>
		/// If uplo == CUBLAS_FILL_MODE_LOWER then the lower triangular part of the triangular
		/// matrix A is copied into the array AP. <para/>If uplo == CUBLAS_FILL_MODE_UPPER then then
		/// the upper triangular part of the triangular matrix A is copied into the array AP
		/// </summary>
		/// <param name="uplo">indicates which matrix A lower or upper part is referenced</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="A">array of dimensions lda x n , with lda&gt;=max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		public void Ctrttp(FillMode uplo, int n, CudaDeviceVariable<cuFloatComplex> A, int lda, CudaDeviceVariable<cuFloatComplex> AP)
		{
			_status = CudaBlasNativeMethods.cublasCtrttp(_blasHandle, uplo, n, A.DevicePointer, lda, AP.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCtrttp", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}

		/// <summary>
		/// This function performs the conversion from the triangular format to the triangular
		/// packed format.<para/>
		/// If uplo == CUBLAS_FILL_MODE_LOWER then the lower triangular part of the triangular
		/// matrix A is copied into the array AP. <para/>If uplo == CUBLAS_FILL_MODE_UPPER then then
		/// the upper triangular part of the triangular matrix A is copied into the array AP
		/// </summary>
		/// <param name="uplo">indicates which matrix A lower or upper part is referenced</param>
		/// <param name="n">number of rows and columns of matrix A.</param>
		/// <param name="A">array of dimensions lda x n , with lda&gt;=max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store matrix A.</param>
		/// <param name="AP">array with A stored in packed format.</param>
		public void Ztrttp(FillMode uplo, int n, CudaDeviceVariable<cuDoubleComplex> A, int lda, CudaDeviceVariable<cuDoubleComplex> AP)
		{
			_status = CudaBlasNativeMethods.cublasZtrttp(_blasHandle, uplo, n, A.DevicePointer, lda, AP.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZtrttp", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
		}
		#endregion                                      

		
		#region Batch QR Factorization
		/// <summary>
		/// This function performs the QR factorization of each Aarray[i] for i =
		/// 0, ...,batchSize-1 using Householder reflections. Each matrix Q[i] is represented
		/// as a product of elementary reflectors and is stored in the lower part of each Aarray[i].
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor.<para/>
		/// cublas<![CDATA[<t>]]>geqrfBatched supports arbitrary dimension.<para/>
		/// cublas<![CDATA[<t>]]>geqrfBatched only supports compute capability 2.0 or above.
		/// </summary>
		/// <param name="m">number of rows Aarray[i].</param>
		/// <param name="n">number of columns of Aarray[i].</param>
		/// <param name="Aarray">array of pointers to device array, with each array of dim. m x n with lda&gt;=max(1,m). The array size determines the number of batches.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix Aarray[i].</param>
		/// <param name="TauArray">array of pointers to device vector, with each vector of dim. max(1,min(m,n)).</param>
		/// <returns>0, if the parameters passed to the function are valid, &lt;0, if the parameter in postion -value is invalid</returns>
		public int GeqrfBatchedS(int m, int n, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<CUdeviceptr> TauArray)
		{
			int info = 0;
			_status = CudaBlasNativeMethods.cublasSgeqrfBatched(_blasHandle, m, n, Aarray.DevicePointer, lda, TauArray.DevicePointer, ref info, Aarray.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSgeqrfBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return info;
		}


		/// <summary>
		/// This function performs the QR factorization of each Aarray[i] for i =
		/// 0, ...,batchSize-1 using Householder reflections. Each matrix Q[i] is represented
		/// as a product of elementary reflectors and is stored in the lower part of each Aarray[i].
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor.<para/>
		/// cublas<![CDATA[<t>]]>geqrfBatched supports arbitrary dimension.<para/>
		/// cublas<![CDATA[<t>]]>geqrfBatched only supports compute capability 2.0 or above.
		/// </summary>
		/// <param name="m">number of rows Aarray[i].</param>
		/// <param name="n">number of columns of Aarray[i].</param>
		/// <param name="Aarray">array of pointers to device array, with each array of dim. m x n with lda&gt;=max(1,m). The array size determines the number of batches.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix Aarray[i].</param>
		/// <param name="TauArray">array of pointers to device vector, with each vector of dim. max(1,min(m,n)).</param>
		/// <returns>0, if the parameters passed to the function are valid, &lt;0, if the parameter in postion -value is invalid</returns>
		public int GeqrfBatchedD(int m, int n, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<CUdeviceptr> TauArray)
		{
			int info = 0;
			_status = CudaBlasNativeMethods.cublasDgeqrfBatched(_blasHandle, m, n, Aarray.DevicePointer, lda, TauArray.DevicePointer, ref info, Aarray.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDgeqrfBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return info;
		}


		/// <summary>
		/// This function performs the QR factorization of each Aarray[i] for i =
		/// 0, ...,batchSize-1 using Householder reflections. Each matrix Q[i] is represented
		/// as a product of elementary reflectors and is stored in the lower part of each Aarray[i].
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor.<para/>
		/// cublas<![CDATA[<t>]]>geqrfBatched supports arbitrary dimension.<para/>
		/// cublas<![CDATA[<t>]]>geqrfBatched only supports compute capability 2.0 or above.
		/// </summary>
		/// <param name="m">number of rows Aarray[i].</param>
		/// <param name="n">number of columns of Aarray[i].</param>
		/// <param name="Aarray">array of pointers to device array, with each array of dim. m x n with lda&gt;=max(1,m). The array size determines the number of batches.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix Aarray[i].</param>
		/// <param name="TauArray">array of pointers to device vector, with each vector of dim. max(1,min(m,n)).</param>
		/// <returns>0, if the parameters passed to the function are valid, &lt;0, if the parameter in postion -value is invalid</returns>
		public int GeqrfBatchedC(int m, int n, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<CUdeviceptr> TauArray)
		{
			int info = 0;
			_status = CudaBlasNativeMethods.cublasCgeqrfBatched(_blasHandle, m, n, Aarray.DevicePointer, lda, TauArray.DevicePointer, ref info, Aarray.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCgeqrfBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return info;
		}


		/// <summary>
		/// This function performs the QR factorization of each Aarray[i] for i =
		/// 0, ...,batchSize-1 using Householder reflections. Each matrix Q[i] is represented
		/// as a product of elementary reflectors and is stored in the lower part of each Aarray[i].
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor.<para/>
		/// cublas<![CDATA[<t>]]>geqrfBatched supports arbitrary dimension.<para/>
		/// cublas<![CDATA[<t>]]>geqrfBatched only supports compute capability 2.0 or above.
		/// </summary>
		/// <param name="m">number of rows Aarray[i].</param>
		/// <param name="n">number of columns of Aarray[i].</param>
		/// <param name="Aarray">array of pointers to device array, with each array of dim. m x n with lda&gt;=max(1,m). The array size determines the number of batches.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix Aarray[i].</param>
		/// <param name="TauArray">array of pointers to device vector, with each vector of dim. max(1,min(m,n)).</param>
		/// <returns>0, if the parameters passed to the function are valid, &lt;0, if the parameter in postion -value is invalid</returns>
		public int GeqrfBatchedZ(int m, int n, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<CUdeviceptr> TauArray)
		{
			int info = 0;
			_status = CudaBlasNativeMethods.cublasZgeqrfBatched(_blasHandle, m, n, Aarray.DevicePointer, lda, TauArray.DevicePointer, ref info, Aarray.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZgeqrfBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return info;
		}
		#endregion

		#region Least Square Min only m >= n and Non-transpose supported
		/// <summary>
		/// This function find the least squares solution of a batch of overdetermined systems.
		/// On exit, each Aarray[i] is overwritten with their QR factorization and each Carray[i] is overwritten with the least square solution
		/// GelsBatched supports only the non-transpose operation and only solves overdetermined
		/// systems (m >= n).<para/>
		/// GelsBatched only supports compute capability 2.0 or above.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor.
		/// </summary>
		/// <param name="trans">operation op(Aarray[i]) that is non- or (conj.) transpose. Only non-transpose operation is currently supported.</param>
		/// <param name="m">number of rows Aarray[i].</param>
		/// <param name="n">number of columns of each Aarray[i] and rows of each Carray[i].</param>
		/// <param name="nrhs">number of columns of each Carray[i].</param>
		/// <param name="Aarray">array of pointers to device array, with each array of dim. m x n with lda&gt;=max(1,m). The array size determines the number of batches.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix Aarray[i]</param>
		/// <param name="Carray">array of pointers to device array, with each array of dim. m x n with ldc&gt;=max(1,m).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store each matrix Carray[i].</param>
		/// <param name="devInfoArray">null or optional array of integers of dimension batchsize.</param>
		/// <returns>0, if the parameters passed to the function are valid, &lt;0, if the parameter in postion -value is invalid</returns>
		public int GelsBatchedS(Operation trans, int m, int n, int nrhs, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<CUdeviceptr> Carray, int ldc, CudaDeviceVariable<int> devInfoArray)
		{
			int info = 0;
			CUdeviceptr _devInfoArray = devInfoArray == null ? new CUdeviceptr(0) : devInfoArray.DevicePointer;
			_status = CudaBlasNativeMethods.cublasSgelsBatched(_blasHandle, trans, m, n, nrhs, Aarray.DevicePointer, lda, Carray.DevicePointer, ldc, ref info, _devInfoArray, Aarray.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSgelsBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return info;
		}


		/// <summary>
		/// This function find the least squares solution of a batch of overdetermined systems.
		/// On exit, each Aarray[i] is overwritten with their QR factorization and each Carray[i] is overwritten with the least square solution
		/// GelsBatched supports only the non-transpose operation and only solves overdetermined
		/// systems (m >= n).<para/>
		/// GelsBatched only supports compute capability 2.0 or above.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor.
		/// </summary>
		/// <param name="trans">operation op(Aarray[i]) that is non- or (conj.) transpose. Only non-transpose operation is currently supported.</param>
		/// <param name="m">number of rows Aarray[i].</param>
		/// <param name="n">number of columns of each Aarray[i] and rows of each Carray[i].</param>
		/// <param name="nrhs">number of columns of each Carray[i].</param>
		/// <param name="Aarray">array of pointers to device array, with each array of dim. m x n with lda&gt;=max(1,m). The array size determines the number of batches.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix Aarray[i]</param>
		/// <param name="Carray">array of pointers to device array, with each array of dim. m x n with ldc&gt;=max(1,m).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store each matrix Carray[i].</param>
		/// <param name="devInfoArray">null or optional array of integers of dimension batchsize.</param>
		/// <returns>0, if the parameters passed to the function are valid, &lt;0, if the parameter in postion -value is invalid</returns>
		public int GelsBatchedD(Operation trans, int m, int n, int nrhs, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<CUdeviceptr> Carray, int ldc, CudaDeviceVariable<int> devInfoArray)
		{
			int info = 0;
			CUdeviceptr _devInfoArray = devInfoArray == null ? new CUdeviceptr(0) : devInfoArray.DevicePointer;
			_status = CudaBlasNativeMethods.cublasDgelsBatched(_blasHandle, trans, m, n, nrhs, Aarray.DevicePointer, lda, Carray.DevicePointer, ldc, ref info, _devInfoArray, Aarray.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDgelsBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return info;
		}


		/// <summary>
		/// This function find the least squares solution of a batch of overdetermined systems.
		/// On exit, each Aarray[i] is overwritten with their QR factorization and each Carray[i] is overwritten with the least square solution
		/// GelsBatched supports only the non-transpose operation and only solves overdetermined
		/// systems (m >= n).<para/>
		/// GelsBatched only supports compute capability 2.0 or above.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor.
		/// </summary>
		/// <param name="trans">operation op(Aarray[i]) that is non- or (conj.) transpose. Only non-transpose operation is currently supported.</param>
		/// <param name="m">number of rows Aarray[i].</param>
		/// <param name="n">number of columns of each Aarray[i] and rows of each Carray[i].</param>
		/// <param name="nrhs">number of columns of each Carray[i].</param>
		/// <param name="Aarray">array of pointers to device array, with each array of dim. m x n with lda&gt;=max(1,m). The array size determines the number of batches.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix Aarray[i]</param>
		/// <param name="Carray">array of pointers to device array, with each array of dim. m x n with ldc&gt;=max(1,m).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store each matrix Carray[i].</param>
		/// <param name="devInfoArray">null or optional array of integers of dimension batchsize.</param>
		/// <returns>0, if the parameters passed to the function are valid, &lt;0, if the parameter in postion -value is invalid</returns>
		public int GelsBatchedC(Operation trans, int m, int n, int nrhs, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<CUdeviceptr> Carray, int ldc, CudaDeviceVariable<int> devInfoArray)
		{
			int info = 0;
			CUdeviceptr _devInfoArray = devInfoArray == null ? new CUdeviceptr(0) : devInfoArray.DevicePointer;
			_status = CudaBlasNativeMethods.cublasCgelsBatched(_blasHandle, trans, m, n, nrhs, Aarray.DevicePointer, lda, Carray.DevicePointer, ldc, ref info, _devInfoArray, Aarray.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCgelsBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return info;
		}


		/// <summary>
		/// This function find the least squares solution of a batch of overdetermined systems.
		/// On exit, each Aarray[i] is overwritten with their QR factorization and each Carray[i] is overwritten with the least square solution
		/// GelsBatched supports only the non-transpose operation and only solves overdetermined
		/// systems (m >= n).<para/>
		/// GelsBatched only supports compute capability 2.0 or above.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor.
		/// </summary>
		/// <param name="trans">operation op(Aarray[i]) that is non- or (conj.) transpose. Only non-transpose operation is currently supported.</param>
		/// <param name="m">number of rows Aarray[i].</param>
		/// <param name="n">number of columns of each Aarray[i] and rows of each Carray[i].</param>
		/// <param name="nrhs">number of columns of each Carray[i].</param>
		/// <param name="Aarray">array of pointers to device array, with each array of dim. m x n with lda&gt;=max(1,m). The array size determines the number of batches.</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store each matrix Aarray[i]</param>
		/// <param name="Carray">array of pointers to device array, with each array of dim. m x n with ldc&gt;=max(1,m).</param>
		/// <param name="ldc">leading dimension of two-dimensional array used to store each matrix Carray[i].</param>
		/// <param name="devInfoArray">null or optional array of integers of dimension batchsize.</param>
		/// <returns>0, if the parameters passed to the function are valid, &lt;0, if the parameter in postion -value is invalid</returns>
		public int GelsBatchedZ(Operation trans, int m, int n, int nrhs, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<CUdeviceptr> Carray, int ldc, CudaDeviceVariable<int> devInfoArray)
		{
			int info = 0;
			CUdeviceptr _devInfoArray = devInfoArray == null ? new CUdeviceptr(0) : devInfoArray.DevicePointer;
			_status = CudaBlasNativeMethods.cublasZgelsBatched(_blasHandle, trans, m, n, nrhs, Aarray.DevicePointer, lda, Carray.DevicePointer, ldc, ref info, _devInfoArray, Aarray.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZgelsBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return info;
		}
		#endregion



		//New in Cuda 7.0

		#region Batched solver based on LU factorization from getrf

		/// <summary>
		/// This function solves an array of systems of linear equations of the form:<para/>
		/// op(A[i]) X[i] = a B[i]<para/>
		/// where A[i] is a matrix which has been LU factorized with pivoting, X[i] and B[i] are
		/// n x nrhs matrices.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of rows and columns of Aarray[i].</param>
		/// <param name="nrhs">number of columns of Barray[i].</param>
		/// <param name="Aarray">array of pointers to array, with each array of dim. n 
		/// x n with lda&gt;=max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store
		/// each matrix Aarray[i].</param>
		/// <param name="devIpiv">array of size n x batchSize that contains the pivoting
		/// sequence of each factorization of Aarray[i] stored in a
		/// linear fashion. If devIpiv is nil, pivoting for all Aarray[i]
		/// is ignored.</param>
		/// <param name="Barray">array of pointers to array, with each array of dim. n
		/// x nrhs with ldb&gt;=max(1,n).</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store
		/// each solution matrix Barray[i].</param>
		/// <param name="batchSize">number of pointers contained in A</param>
		/// <returns>If info=0, the execution is successful. If info = -j, the j-th parameter had an illegal value.</returns>
		public int GetrsBatchedS(Operation trans, int n, int nrhs, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<int> devIpiv,
			CudaDeviceVariable<CUdeviceptr> Barray, int ldb, int batchSize)
		{
			int info = 0;
			_status = CudaBlasNativeMethods.cublasSgetrsBatched(_blasHandle, trans, n, nrhs, Aarray.DevicePointer, lda, devIpiv.DevicePointer, Barray.DevicePointer, ldb, ref info, batchSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSgetrsBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return info;
		}

		/// <summary>
		/// This function solves an array of systems of linear equations of the form:<para/>
		/// op(A[i]) X[i] = a B[i]<para/>
		/// where A[i] is a matrix which has been LU factorized with pivoting, X[i] and B[i] are
		/// n x nrhs matrices.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of rows and columns of Aarray[i].</param>
		/// <param name="nrhs">number of columns of Barray[i].</param>
		/// <param name="Aarray">array of pointers to array, with each array of dim. n 
		/// x n with lda&gt;=max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store
		/// each matrix Aarray[i].</param>
		/// <param name="devIpiv">array of size n x batchSize that contains the pivoting
		/// sequence of each factorization of Aarray[i] stored in a
		/// linear fashion. If devIpiv is nil, pivoting for all Aarray[i]
		/// is ignored.</param>
		/// <param name="Barray">array of pointers to array, with each array of dim. n
		/// x nrhs with ldb&gt;=max(1,n).</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store
		/// each solution matrix Barray[i].</param>
		/// <param name="batchSize">number of pointers contained in A</param>
		/// <returns>If info=0, the execution is successful. If info = -j, the j-th parameter had an illegal value.</returns>
		public int GetrsBatchedD(Operation trans, int n, int nrhs, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<int> devIpiv,
			CudaDeviceVariable<CUdeviceptr> Barray, int ldb, int batchSize)
		{
			int info = 0;
			_status = CudaBlasNativeMethods.cublasDgetrsBatched(_blasHandle, trans, n, nrhs, Aarray.DevicePointer, lda, devIpiv.DevicePointer, Barray.DevicePointer, ldb, ref info, batchSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasDgetrsBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return info;
		}

		/// <summary>
		/// This function solves an array of systems of linear equations of the form:<para/>
		/// op(A[i]) X[i] = a B[i]<para/>
		/// where A[i] is a matrix which has been LU factorized with pivoting, X[i] and B[i] are
		/// n x nrhs matrices.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of rows and columns of Aarray[i].</param>
		/// <param name="nrhs">number of columns of Barray[i].</param>
		/// <param name="Aarray">array of pointers to array, with each array of dim. n 
		/// x n with lda&gt;=max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store
		/// each matrix Aarray[i].</param>
		/// <param name="devIpiv">array of size n x batchSize that contains the pivoting
		/// sequence of each factorization of Aarray[i] stored in a
		/// linear fashion. If devIpiv is nil, pivoting for all Aarray[i]
		/// is ignored.</param>
		/// <param name="Barray">array of pointers to array, with each array of dim. n
		/// x nrhs with ldb&gt;=max(1,n).</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store
		/// each solution matrix Barray[i].</param>
		/// <param name="batchSize">number of pointers contained in A</param>
		/// <returns>If info=0, the execution is successful. If info = -j, the j-th parameter had an illegal value.</returns>
		public int GetrsBatchedC(Operation trans, int n, int nrhs, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<int> devIpiv,
			CudaDeviceVariable<CUdeviceptr> Barray, int ldb, int batchSize)
		{
			int info = 0;
			_status = CudaBlasNativeMethods.cublasCgetrsBatched(_blasHandle, trans, n, nrhs, Aarray.DevicePointer, lda, devIpiv.DevicePointer, Barray.DevicePointer, ldb, ref info, batchSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasCgetrsBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return info;
		}

		/// <summary>
		/// This function solves an array of systems of linear equations of the form:<para/>
		/// op(A[i]) X[i] = a B[i]<para/>
		/// where A[i] is a matrix which has been LU factorized with pivoting, X[i] and B[i] are
		/// n x nrhs matrices.<para/>
		/// This function is intended to be used for matrices of small sizes where the launch
		/// overhead is a significant factor.
		/// </summary>
		/// <param name="trans">operation op(A) that is non- or (conj.) transpose.</param>
		/// <param name="n">number of rows and columns of Aarray[i].</param>
		/// <param name="nrhs">number of columns of Barray[i].</param>
		/// <param name="Aarray">array of pointers to array, with each array of dim. n 
		/// x n with lda&gt;=max(1,n).</param>
		/// <param name="lda">leading dimension of two-dimensional array used to store
		/// each matrix Aarray[i].</param>
		/// <param name="devIpiv">array of size n x batchSize that contains the pivoting
		/// sequence of each factorization of Aarray[i] stored in a
		/// linear fashion. If devIpiv is nil, pivoting for all Aarray[i]
		/// is ignored.</param>
		/// <param name="Barray">array of pointers to array, with each array of dim. n
		/// x nrhs with ldb&gt;=max(1,n).</param>
		/// <param name="ldb">leading dimension of two-dimensional array used to store
		/// each solution matrix Barray[i].</param>
		/// <param name="batchSize">number of pointers contained in A</param>
		/// <returns>If info=0, the execution is successful. If info = -j, the j-th parameter had an illegal value.</returns>
		public int GetrsBatchedZ(Operation trans, int n, int nrhs, CudaDeviceVariable<CUdeviceptr> Aarray, int lda, CudaDeviceVariable<int> devIpiv,
			CudaDeviceVariable<CUdeviceptr> Barray, int ldb, int batchSize)
		{
			int info = 0;
			_status = CudaBlasNativeMethods.cublasZgetrsBatched(_blasHandle, trans, n, nrhs, Aarray.DevicePointer, lda, devIpiv.DevicePointer, Barray.DevicePointer, ldb, ref info, batchSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasZgetrsBatched", _status));
			if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
			return info;
		}
		#endregion
		#endregion

		#region Static Methods
		/// <summary>
		/// copies elements from a vector <c>hostSourceVector</c> in CPU memory space to a vector <c>devDestVector</c> 
		/// in GPU memory space. Storage spacing between consecutive elements
		/// is <c>incrHostSource</c> for the source vector <c>hostSourceVector</c> and <c>incrDevDest</c> for the destination vector
		/// <c>devDestVector</c>. Column major format for two-dimensional matrices
		/// is assumed throughout CUBLAS. Therefore, if the increment for a vector 
		/// is equal to 1, this access a column vector while using an increment 
		/// equal to the leading dimension of the respective matrix accesses a 
		/// row vector.
		/// </summary>
		/// <typeparam name="T">Vector datatype </typeparam>
		/// <param name="hostSourceVector">Source vector in host memory</param>
		/// <param name="incrHostSource"></param>
		/// <param name="devDestVector">Destination vector in device memory</param>
		/// <param name="incrDevDest"></param>
		public static void SetVector<T>(T[] hostSourceVector, int incrHostSource, CudaDeviceVariable<T> devDestVector, int incrDevDest) where T : struct
		{
			CublasStatus status;
			GCHandle handle = GCHandle.Alloc(hostSourceVector, GCHandleType.Pinned);
			status = CudaBlasNativeMethods.cublasSetVector(hostSourceVector.Length, devDestVector.TypeSize, handle.AddrOfPinnedObject(), incrHostSource, devDestVector.DevicePointer, incrDevDest);
			handle.Free();
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSetVector", status));
			if (status != CublasStatus.Success) throw new CudaBlasException(status);
		}

		/// <summary>
		/// copies elements from a vector <c>devSourceVector</c> in GPU memory space to a vector <c>hostDestVector</c> 
		/// in CPU memory space. Storage spacing between consecutive elements
		/// is <c>incrHostDest</c> for the source vector <c>devSourceVector</c> and <c>incrDevSource</c> for the destination vector
		/// <c>hostDestVector</c>. Column major format for two-dimensional matrices
		/// is assumed throughout CUBLAS. Therefore, if the increment for a vector 
		/// is equal to 1, this access a column vector while using an increment 
		/// equal to the leading dimension of the respective matrix accesses a 
		/// row vector.
		/// </summary>
		/// <typeparam name="T">Vector datatype</typeparam>
		/// <param name="devSourceVector">Source vector in device memory</param>
		/// <param name="incrDevSource"></param>
		/// <param name="hostDestVector">Destination vector in host memory</param>
		/// <param name="incrHostDest"></param>
		public static void GetVector<T>(CudaDeviceVariable<T> devSourceVector, int incrDevSource, T[] hostDestVector, int incrHostDest) where T : struct
		{
			CublasStatus status;
			GCHandle handle = GCHandle.Alloc(hostDestVector, GCHandleType.Pinned);
			status = CudaBlasNativeMethods.cublasGetVector(hostDestVector.Length, devSourceVector.TypeSize, devSourceVector.DevicePointer, incrDevSource, handle.AddrOfPinnedObject(), incrHostDest);
			handle.Free();
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasGetVector", status));
			if (status != CublasStatus.Success) throw new CudaBlasException(status);
		}

		/// <summary>
		/// copies a tile of <c>rows</c> x <c>cols</c> elements from a matrix <c>hostSource</c> in CPU memory
		/// space to a matrix <c>devDest</c> in GPU memory space. Both matrices are assumed to be stored in column 
		/// major format, with the leading dimension (i.e. number of rows) of 
		/// source matrix <c>hostSource</c> provided in <c>ldHostSource</c>, and the leading dimension of matrix <c>devDest</c>
		/// provided in <c>ldDevDest</c>.
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <param name="rows"></param>
		/// <param name="cols"></param>
		/// <param name="hostSource"></param>
		/// <param name="ldHostSource"></param>
		/// <param name="devDest"></param>
		/// <param name="ldDevDest"></param>
		public static void SetMatrix<T>(int rows, int cols, T[] hostSource, int ldHostSource, CudaDeviceVariable<T> devDest, int ldDevDest) where T : struct
		{
			CublasStatus status;
			GCHandle handle = GCHandle.Alloc(hostSource, GCHandleType.Pinned);
			status = CudaBlasNativeMethods.cublasSetMatrix(rows, cols, devDest.TypeSize, handle.AddrOfPinnedObject(), ldHostSource, devDest.DevicePointer, ldDevDest);
			handle.Free();
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSetMatrix", status));
			if (status != CublasStatus.Success) throw new CudaBlasException(status);
		}

		/// <summary>
		/// copies a tile of <c>rows</c> x <c>cols</c> elements from a matrix <c>devSource</c> in GPU memory
		/// space to a matrix <c>hostDest</c> in CPU memory space. Both matrices are assumed to be stored in column 
		/// major format, with the leading dimension (i.e. number of rows) of 
		/// source matrix <c>devSource</c> provided in <c>devSource</c>, and the leading dimension of matrix <c>hostDest</c>
		/// provided in <c>ldHostDest</c>. 
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <param name="rows"></param>
		/// <param name="cols"></param>
		/// <param name="devSource"></param>
		/// <param name="ldDevSource"></param>
		/// <param name="hostDest"></param>
		/// <param name="ldHostDest"></param>
		public static void GetMatrix<T>(int rows, int cols, CudaDeviceVariable<T> devSource, int ldDevSource, T[] hostDest, int ldHostDest) where T : struct
		{
			CublasStatus status;
			GCHandle handle = GCHandle.Alloc(hostDest, GCHandleType.Pinned);
			status = CudaBlasNativeMethods.cublasGetMatrix(rows, cols, devSource.TypeSize, devSource.DevicePointer, ldDevSource, handle.AddrOfPinnedObject(), ldHostDest);
			handle.Free();
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasGetMatrix", status));
			if (status != CublasStatus.Success) throw new CudaBlasException(status);
		}

		/// <summary>
		/// copies elements from a vector <c>hostSourceVector</c> in CPU memory space to a vector <c>devDestVector</c> 
		/// in GPU memory space. Storage spacing between consecutive elements
		/// is <c>incrHostSource</c> for the source vector <c>hostSourceVector</c> and <c>incrDevDest</c> for the destination vector
		/// <c>devDestVector</c>. Column major format for two-dimensional matrices
		/// is assumed throughout CUBLAS. Therefore, if the increment for a vector 
		/// is equal to 1, this access a column vector while using an increment 
		/// equal to the leading dimension of the respective matrix accesses a 
		/// row vector.
		/// </summary>
		/// <typeparam name="T">Vector datatype </typeparam>
		/// <param name="hostSourceVector">Source vector in host memory</param>
		/// <param name="incrHostSource"></param>
		/// <param name="devDestVector">Destination vector in device memory</param>
		/// <param name="incrDevDest"></param>
		/// <param name="stream"></param>
		public static void SetVectorAsync<T>(T[] hostSourceVector, int incrHostSource, CudaDeviceVariable<T> devDestVector, int incrDevDest, CUstream stream) where T : struct
		{
			CublasStatus status;
			GCHandle handle = GCHandle.Alloc(hostSourceVector, GCHandleType.Pinned);
			status = CudaBlasNativeMethods.cublasSetVectorAsync(hostSourceVector.Length, devDestVector.TypeSize, handle.AddrOfPinnedObject(), incrHostSource, devDestVector.DevicePointer, incrDevDest, stream);
			handle.Free();
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSetVectorAsync", status));
			if (status != CublasStatus.Success) throw new CudaBlasException(status);
		}

		/// <summary>
		/// copies elements from a vector <c>devSourceVector</c> in GPU memory space to a vector <c>hostDestVector</c> 
		/// in CPU memory space. Storage spacing between consecutive elements
		/// is <c>incrHostDest</c> for the source vector <c>devSourceVector</c> and <c>incrDevSource</c> for the destination vector
		/// <c>hostDestVector</c>. Column major format for two-dimensional matrices
		/// is assumed throughout CUBLAS. Therefore, if the increment for a vector 
		/// is equal to 1, this access a column vector while using an increment 
		/// equal to the leading dimension of the respective matrix accesses a 
		/// row vector.
		/// </summary>
		/// <typeparam name="T">Vector datatype</typeparam>
		/// <param name="devSourceVector">Source vector in device memory</param>
		/// <param name="incrDevSource"></param>
		/// <param name="hostDestVector">Destination vector in host memory</param>
		/// <param name="incrHostDest"></param>
		/// <param name="stream"></param>
		public static void GetVectorAsync<T>(CudaDeviceVariable<T> devSourceVector, int incrDevSource, T[] hostDestVector, int incrHostDest, CUstream stream) where T : struct
		{
			CublasStatus status;
			GCHandle handle = GCHandle.Alloc(hostDestVector, GCHandleType.Pinned);
			status = CudaBlasNativeMethods.cublasGetVectorAsync(hostDestVector.Length, devSourceVector.TypeSize, devSourceVector.DevicePointer, incrDevSource, handle.AddrOfPinnedObject(), incrHostDest, stream);
			handle.Free();
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasGetVectorAsync", status));
			if (status != CublasStatus.Success) throw new CudaBlasException(status);
		}

		/// <summary>
		/// copies a tile of <c>rows</c> x <c>cols</c> elements from a matrix <c>hostSource</c> in CPU memory
		/// space to a matrix <c>devDest</c> in GPU memory space. Both matrices are assumed to be stored in column 
		/// major format, with the leading dimension (i.e. number of rows) of 
		/// source matrix <c>hostSource</c> provided in <c>ldHostSource</c>, and the leading dimension of matrix <c>devDest</c>
		/// provided in <c>ldDevDest</c>.
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <param name="rows"></param>
		/// <param name="cols"></param>
		/// <param name="hostSource"></param>
		/// <param name="ldHostSource"></param>
		/// <param name="devDest"></param>
		/// <param name="ldDevDest"></param>
		/// <param name="stream"></param>
		public static void SetMatrixAsync<T>(int rows, int cols, T[] hostSource, int ldHostSource, CudaDeviceVariable<T> devDest, int ldDevDest, CUstream stream) where T : struct
		{
			CublasStatus status;
			GCHandle handle = GCHandle.Alloc(hostSource, GCHandleType.Pinned);
			status = CudaBlasNativeMethods.cublasSetMatrixAsync(rows, cols, devDest.TypeSize, handle.AddrOfPinnedObject(), ldHostSource, devDest.DevicePointer, ldDevDest, stream);
			handle.Free();
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasSetMatrixAsync", status));
			if (status != CublasStatus.Success) throw new CudaBlasException(status);
		}

		/// <summary>
		/// copies a tile of <c>rows</c> x <c>cols</c> elements from a matrix <c>devSource</c> in GPU memory
		/// space to a matrix <c>hostDest</c> in CPU memory space. Both matrices are assumed to be stored in column 
		/// major format, with the leading dimension (i.e. number of rows) of 
		/// source matrix <c>devSource</c> provided in <c>devSource</c>, and the leading dimension of matrix <c>hostDest</c>
		/// provided in <c>ldHostDest</c>. 
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <param name="rows"></param>
		/// <param name="cols"></param>
		/// <param name="devSource"></param>
		/// <param name="ldDevSource"></param>
		/// <param name="hostDest"></param>
		/// <param name="ldHostDest"></param>
		/// <param name="stream"></param>
		public static void GetMatrixAsync<T>(int rows, int cols, CudaDeviceVariable<T> devSource, int ldDevSource, T[] hostDest, int ldHostDest, CUstream stream) where T : struct
		{
			CublasStatus status;
			GCHandle handle = GCHandle.Alloc(hostDest, GCHandleType.Pinned);
			status = CudaBlasNativeMethods.cublasGetMatrixAsync(rows, cols, devSource.TypeSize, devSource.DevicePointer, ldDevSource, handle.AddrOfPinnedObject(), ldHostDest, stream);
			handle.Free();
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cublasGetMatrixAsync", status));
			if (status != CublasStatus.Success) throw new CudaBlasException(status);
		}
		#endregion
	}
}
