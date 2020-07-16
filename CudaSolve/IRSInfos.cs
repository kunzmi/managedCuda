//	Copyright (c) 2020, Michael Kunz. All rights reserved.
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

namespace ManagedCuda.CudaSolve
{
	/// <summary>
	/// 
	/// </summary>
	public class IRSInfos : IDisposable
	{
		private cusolverDnIRSInfos _infos;
		private cusolverStatus res;
		private bool disposed;

		#region Contructors
		/// <summary>
		/// </summary>
		public IRSInfos()
		{
			_infos = new cusolverDnIRSInfos();
			res = CudaSolveNativeMethods.Dense.cusolverDnIRSInfosCreate(ref _infos);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnIRSInfosCreate", res));
			if (res != cusolverStatus.Success)
				throw new CudaSolveException(res);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~IRSInfos()
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
				res = CudaSolveNativeMethods.Dense.cusolverDnIRSInfosDestroy(_infos);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnIRSInfosDestroy", res));
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		/// <summary>
		/// Returns the inner handle.
		/// </summary>
		public cusolverDnIRSInfos Infos
		{
			get { return _infos; }
		}

		/// <summary>
		/// </summary>
		public int GetNiters()
		{
			int val = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnIRSInfosGetNiters(_infos, ref val);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnIRSInfosGetNiters", res));
			if (res != cusolverStatus.Success)
				throw new CudaSolveException(res);
			return val;
		}

		/// <summary>
		/// </summary>
		public int GetOuterNiters()
		{
			int val = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnIRSInfosGetOuterNiters(_infos, ref val);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnIRSInfosGetOuterNiters", res));
			if (res != cusolverStatus.Success)
				throw new CudaSolveException(res);
			return val;
		}

		/// <summary>
		/// </summary>
		public int GetMaxIters()
		{
			int val = 0;
			res = CudaSolveNativeMethods.Dense.cusolverDnIRSInfosGetMaxIters(_infos, ref val);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnIRSInfosGetMaxIters", res));
			if (res != cusolverStatus.Success)
				throw new CudaSolveException(res);
			return val;
		}

		/// <summary>
		/// </summary>
		public void RequestResidual()
		{
			res = CudaSolveNativeMethods.Dense.cusolverDnIRSInfosRequestResidual(_infos);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnIRSInfosRequestResidual", res));
			if (res != cusolverStatus.Success)
				throw new CudaSolveException(res);
		}

		/// <summary>
		/// </summary>
		public IntPtr GetResidualHistory()
		{
			IntPtr val = new IntPtr();
			res = CudaSolveNativeMethods.Dense.cusolverDnIRSInfosGetResidualHistory(_infos, ref val);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cusolverDnIRSInfosGetResidualHistory", res));
			if (res != cusolverStatus.Success)
				throw new CudaSolveException(res);
			return val;
		}
	}
}
