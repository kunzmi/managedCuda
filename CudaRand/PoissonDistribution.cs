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

namespace ManagedCuda.CudaRand
{
	/// <summary>
	/// Poisson distribution
	/// </summary>
	public class PoissonDistribution
	{
		bool disposed;
		DiscreteDistribution _distributions;
		double _lambda;
		CurandStatus _status;

		#region Constructors
		/// <summary>
		/// Creates a new poisson distribution.<para/>
		/// Construct histogram array for poisson distribution.<para/>
		/// Construct histogram array for poisson distribution with lambda <c>lambda</c>.
		/// For lambda greater than 2000 optimization with normal distribution is used.
		/// </summary>
		/// <param name="lambda">lambda for poisson distribution</param>
		public PoissonDistribution(double lambda)
		{
			_distributions = new DiscreteDistribution();
			_lambda = lambda;
			_status = CudaRandNativeMethods.curandCreatePoissonDistribution(lambda, ref _distributions);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "curandCreatePoissonDistribution", _status));
			if (_status != CurandStatus.Success) throw new CudaRandException(_status);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~PoissonDistribution()
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
				_status = CudaRandNativeMethods.curandDestroyDistribution(_distributions);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "curandDestroyDistribution", _status));
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("CudaRand not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		/// <summary> 
		/// Generate Poisson-distributed unsigned ints.<para/>
		/// Use <c>generator</c> to generate <c>num</c> unsigned int results into the device memory at
		/// <c>outputPtr</c>.  The device memory must have been previously allocated and be
		/// large enough to hold all the results.  Launches are done with the stream
		/// set using <c>curandSetStream()</c>, or the null stream if no stream has been set.
		/// Results are 32-bit unsigned int point values with poisson distribution based on
		/// an associated poisson distribution with lambda <c>lambda</c>.
		/// </summary>
		/// <param name="generator">Generator to use</param>
		/// <param name="output">Pointer to device memory to store CUDA-generated results</param>
		public void Generate(CudaRandDevice generator, CudaDeviceVariable<uint> output)
		{
			_status = CudaRandNativeMethods.curandGeneratePoisson(generator.Generator, output.DevicePointer, output.Size, _lambda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "curandGeneratePoisson", _status));
			if (_status != CurandStatus.Success) throw new CudaRandException(_status);
		}


		/// <summary> 
		/// Generate Poisson-distributed unsigned ints.<para/>
		/// Use <c>generator</c> to generate <c>num</c> unsigned int results into the device memory at
		/// <c>outputPtr</c>.  The device memory must have been previously allocated and be
		/// large enough to hold all the results.  Launches are done with the stream
		/// set using <c>curandSetStream()</c>, or the null stream if no stream has been set.
		/// Results are 32-bit unsigned int point values with poisson distribution based on
		/// an associated poisson distribution with lambda <c>lambda</c>.
		/// </summary>
		/// <param name="generator">Generator to use</param>
		/// <param name="output">Pointer to host memory to store CPU-generated results</param>
		public void Generate(CudaRandHost generator, uint[] output)
		{
			_status = CudaRandNativeMethods.curandGeneratePoisson(generator.Generator, output, output.Length, _lambda);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "curandGeneratePoisson", _status));
			if (_status != CurandStatus.Success) throw new CudaRandException(_status);
		}
	}
}
