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
	/// Wrapper class for csrgemm2Info
	/// </summary>
	public class CudaSparseCsrgemm2Info : IDisposable
	{
		private csrgemm2Info _info;
		private cusparseStatus res;
		private bool disposed;

		#region Contructors
		/// <summary>
		/// </summary>
		public CudaSparseCsrgemm2Info()
		{
			_info = new csrgemm2Info();
			res = CudaSparseNativeMethods.cusparseCreateCsrgemm2Info(ref _info);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateCsrgemm2Info", res));
			if (res != cusparseStatus.Success)
				throw new CudaSparseException(res);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaSparseCsrgemm2Info()
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
				res = CudaSparseNativeMethods.cusparseDestroyCsrgemm2Info(_info);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDestroyCsrgemm2Info", res));
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		/// <summary>
		/// Returns the inner handle.
		/// </summary>
		public csrgemm2Info Csrgemm2Info
		{
			get { return _info; }
		}
	}
}
