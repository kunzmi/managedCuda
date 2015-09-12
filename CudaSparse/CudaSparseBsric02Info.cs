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
	/// Wrapper class for cusparseSolveAnalysisInfo
	/// </summary>
	public class CudaSparseBsric02Info : IDisposable
	{
		private bsric02Info _info;
		private cusparseStatus res;
		private bool disposed;
		
		#region Contructors
		/// <summary>
		/// </summary>
		public CudaSparseBsric02Info()
		{
			_info = new bsric02Info();
			res = CudaSparseNativeMethods.cusparseCreateBsric02Info(ref _info);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseCreateBsric02Info", res));
			if (res != cusparseStatus.Success)
				throw new CudaSparseException(res);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaSparseBsric02Info()
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
				res = CudaSparseNativeMethods.cusparseDestroyBsric02Info(_info);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusparseDestroyBsric02Info", res));
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		/// <summary>
		/// Returns the inner handle.
		/// </summary>
		public bsric02Info Bsric02Info
		{
			get { return _info; }
		}
	}
}
