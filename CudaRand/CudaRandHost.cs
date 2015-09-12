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
	/// Wrapper for a CUrand generator handle in host mode
	/// </summary>
	public class CudaRandHost
	{
		const int MaxDimensions = 20000;
		bool disposed;
		CurandGenerator _generator;
		CurandStatus _status;

		#region Constructors
		/// <summary>
		/// Creates a new random number generator of type Type
		/// </summary>
		/// <param name="Type">Generator type</param>
		public CudaRandHost(GeneratorType Type)
		{
			_generator = new CurandGenerator();
			_status = CudaRandNativeMethods.curandCreateGeneratorHost(ref _generator, Type);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "curandCreateGeneratorHost", _status));
			if (_status != CurandStatus.Success) throw new CudaRandException(_status);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~CudaRandHost()
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
				_status = CudaRandNativeMethods.curandDestroyGenerator(_generator);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "curandDestroyGenerator", _status));
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("CudaRandHost not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Properties

		/// <summary>
		/// Returns the wrapped curand generator handle
		/// </summary>
		public CurandGenerator Generator
		{
			get { return _generator; }
		}
		#endregion

		#region Methods

		/// <summary> 
		/// Set the seed value of the pseudorandom number generator.<para/>
		/// All values of seed are valid.  Different seeds will produce different sequences.
		/// Different seeds will often not be statistically correlated with each other,
		/// but some pairs of seed values may generate sequences which are statistically correlated.
		/// </summary>
		/// <param name="seed">All values of seed are valid.</param>
		public void SetPseudoRandomGeneratorSeed(ulong seed)
		{
			_status = CudaRandNativeMethods.curandSetPseudoRandomGeneratorSeed(_generator, seed);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "curandSetPseudoRandomGeneratorSeed", _status));
			if (_status != CurandStatus.Success) throw new CudaRandException(_status);
		}

		/// <summary>
		/// Set the absolute offset of the pseudo or quasirandom number generator.
		/// <para/>
		/// All values of offset are valid.  The offset position is absolute, not 
		/// relative to the current position in the sequence.
		/// </summary>
		/// <param name="offset">All values of offset are valid.</param>
		public void SetOffset(ulong offset)
		{
			_status = CudaRandNativeMethods.curandSetGeneratorOffset(_generator, offset);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "curandSetGeneratorOffset", _status));
			if (_status != CurandStatus.Success) throw new CudaRandException(_status);
		}

		/// <summary>
		/// 
		/// Set the ordering of results of the pseudo or quasirandom number generator.
		/// <para/>
		/// Legal values of order for pseudorandom generators are:<para/>
		/// - <see cref="Ordering.PseudoDefault"/><para/>
		/// - <see cref="Ordering.PseudoBest"/><para/>
		/// - <see cref="Ordering.PseudoSeeded"/><para/>
		/// <para/>
		/// Legal values of order for quasirandom generators are:<para/>
		/// - <see cref="Ordering.QuasiDefault"/>
		/// </summary>
		/// <param name="order"></param>
		public void SetGeneratorOrdering(Ordering order)
		{
			_status = CudaRandNativeMethods.curandSetGeneratorOrdering(_generator, order);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "curandSetGeneratorOrdering", _status));
			if (_status != CurandStatus.Success) throw new CudaRandException(_status);
		}

		/// <summary>
		/// Set the number of dimensions to be generated by the quasirandom number generator.
		/// <para/>
		/// Legal values for dimensions are 1 to 20000.
		/// </summary>
		/// <param name="dimensions">Legal values for dimensions are 1 to 20000.</param>
		public void SetQuasiRandomGeneratorDimensions(uint dimensions)
		{
			_status = CudaRandNativeMethods.curandSetQuasiRandomGeneratorDimensions(_generator, dimensions);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "curandSetQuasiRandomGeneratorDimensions", _status));
			if (_status != CurandStatus.Success) throw new CudaRandException(_status);
		}

		/// <summary>
		/// Use generator to generate num 32-bit results into the device memory at
		/// output. The device memory must have been previously allocated and be
		/// large enough to hold all the results.
		/// <para/>
		/// Results are 32-bit values with every bit random.
		/// </summary>
		/// <param name="output">CudaDeviceVariable</param>
		public void Generate(uint[] output)
		{
			_status = CudaRandNativeMethods.curandGenerate(_generator, output, output.LongLength);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "curandGenerate", _status));
			if (_status != CurandStatus.Success) throw new CudaRandException(_status);
		}

		/// <summary>
		/// Use generator to generate num 64-bit results into the device memory at
		/// output. The device memory must have been previously allocated and be
		/// large enough to hold all the results.
		/// <para/>
		/// Results are 64-bit values with every bit random.
		/// </summary>
		/// <param name="output">CudaDeviceVariable</param>
		public void Generate(ulong[] output)
		{
			_status = CudaRandNativeMethods.curandGenerateLongLong(_generator, output, output.LongLength);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "curandGenerateLongLong", _status));
			if (_status != CurandStatus.Success) throw new CudaRandException(_status);
		}


		/// <summary>
		/// Use generator to generate num float results into the device memory at
		/// outputPtr. The device memory must have been previously allocated and be
		/// large enough to hold all the results. Launches are done with the stream
		/// set using SetStream(), or the null stream if no stream has been set.
		/// <para/>
		/// Results are 32-bit floating point values between 0.0f and 1.0f,
		/// excluding 0.0f and including 1.0f.
		/// </summary>
		/// <param name="output">CudaDeviceVariable</param>
		public void GenerateUniform(float[] output)
		{
			_status = CudaRandNativeMethods.curandGenerateUniform(_generator, output, output.LongLength);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "curandGenerateUniform", _status));
			if (_status != CurandStatus.Success) throw new CudaRandException(_status);
		}

		/// <summary>
		/// Use generator to generate num double results into the device memory at
		/// outputPtr. The device memory must have been previously allocated and be
		/// large enough to hold all the results. Launches are done with the stream
		/// set using curandSetStream(), or the null stream if no stream has been set.
		/// <para/>
		/// Results are 64-bit double precision floating point values between 
		/// 0.0 and 1.0, excluding 0.0 and including 1.0.
		/// </summary>
		/// <param name="output">CudaDeviceVariable</param>
		public void GenerateUniform(double[] output)
		{
			_status = CudaRandNativeMethods.curandGenerateUniformDouble(_generator, output, output.LongLength);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "curandGenerateUniformDouble", _status));
			if (_status != CurandStatus.Success) throw new CudaRandException(_status);
		}


		/// <summary>
		/// Use generator to generate num float results into the device memory at
		/// outputPtr. The device memory must have been previously allocated and be
		/// large enough to hold all the results. Launches are done with the stream
		/// set using curandSetStream(), or the null stream if no stream has been set.
		/// <para/>
		/// Results are 32-bit floating point values with mean  mean and standard
		/// deviation stddev.
		/// <para/>
		/// Normally distributed results are generated from pseudorandom generators
		/// with a Box-Muller transform, and so require num to be even.
		/// Quasirandom generators use an inverse cumulative distribution 
		/// function to preserve dimensionality.
		/// <para/>
		/// There may be slight numerical differences between results generated
		/// on the GPU with generators created with curandCreateGenerator()
		/// and results calculated on the CPU with generators created with
		/// curandCreateGeneratorHost(). These differences arise because of
		/// differences in results for transcendental functions.  In addition,
		/// future versions of CURAND may use newer versions of the CUDA math
		/// library, so different versions of CURAND may give slightly different
		/// numerical values.
		/// </summary>
		/// <param name="output">CudaDeviceVariable</param>
		/// <param name="mean"></param>
		/// <param name="stddev"></param>
		public void GenerateNormal(float[] output, float mean, float stddev)
		{
			_status = CudaRandNativeMethods.curandGenerateNormal(_generator, output, output.LongLength, mean, stddev);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "curandGenerateNormal", _status));
			if (_status != CurandStatus.Success) throw new CudaRandException(_status);
		}

		/// <summary>
		/// Use generator to generate num double results into the device memory at
		/// outputPtr. The device memory must have been previously allocated and be
		/// large enough to hold all the results.  Launches are done with the stream
		/// set using curandSetStream(), or the null stream if no stream has been set.
		/// <para/>
		/// Results are 64-bit floating point values with mean mean and standard
		/// deviation stddev.
		/// <para/>
		/// Normally distributed results are generated from pseudorandom generators
		/// with a Box-Muller transform, and so require num to be even.
		/// Quasirandom generators use an inverse cumulative distribution 
		/// function to preserve dimensionality.
		/// <para/>
		/// There may be slight numerical differences between results generated
		/// on the GPU with generators created with curandCreateGenerator()
		/// and results calculated on the CPU with generators created with
		/// curandCreateGeneratorHost().  These differences arise because of
		/// differences in results for transcendental functions.  In addition,
		/// future versions of CURAND may use newer versions of the CUDA math
		/// library, so different versions of CURAND may give slightly different
		/// numerical values.
		/// </summary>
		/// <param name="output">CudaDeviceVariable</param>
		/// <param name="mean"></param>
		/// <param name="stddev"></param>
		public void GenerateNormal(double[] output, double mean, double stddev)
		{
			_status = CudaRandNativeMethods.curandGenerateNormalDouble(_generator, output, output.LongLength, mean, stddev);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "curandGenerateNormalDouble", _status));
			if (_status != CurandStatus.Success) throw new CudaRandException(_status);
		}
		
		/// <summary>
		/// Use generator to generate num float results into the device memory at
		/// outputPtr.  The device memory must have been previously allocated and be
		/// large enough to hold all the results.  Launches are done with the stream
		/// set using ::curandSetStream(), or the null stream if no stream has been set.
		/// <para/>
		/// Results are 32-bit floating point values with log-normal distribution based on
		/// an associated normal distribution with mean mean and standard deviation stddev.
		/// <para/>
		/// Normally distributed results are generated from pseudorandom generators
		/// with a Box-Muller transform, and so require num to be even.
		/// Quasirandom generators use an inverse cumulative distribution 
		/// function to preserve dimensionality. <para/>
		/// The normally distributed results are transformed into log-normal distribution.
		/// <para/>
		/// There may be slight numerical differences between results generated
		/// on the GPU with generators created with ::curandCreateGenerator()
		/// and results calculated on the CPU with generators created with
		/// ::curandCreateGeneratorHost(). These differences arise because of
		/// differences in results for transcendental functions.  In addition,
		/// future versions of CURAND may use newer versions of the CUDA math
		/// library, so different versions of CURAND may give slightly different
		/// numerical values.
		/// </summary>
		/// <param name="output">CudaDeviceVariable</param>
		/// <param name="mean"></param>
		/// <param name="stddev"></param>
		public void GenerateLogNormal(float[] output, float mean, float stddev)
		{
			_status = CudaRandNativeMethods.curandGenerateLogNormal(_generator, output, output.LongLength, mean, stddev);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "curandGenerateLogNormal", _status));
			if (_status != CurandStatus.Success) throw new CudaRandException(_status);
		}

		/// <summary>
		/// Use generator to generate num double results into the device memory at
		/// outputPtr.  The device memory must have been previously allocated and be
		/// large enough to hold all the results.  Launches are done with the stream
		/// set using curandSetStream(), or the null stream if no stream has been set.
		/// <para/>
		/// Results are 64-bit floating point values with log-normal distribution based on
		/// an associated normal distribution with mean mean and standard deviation stddev.
		/// <para/>
		/// Normally distributed results are generated from pseudorandom generators
		/// with a Box-Muller transform, and so require num to be even.
		/// Quasirandom generators use an inverse cumulative distribution 
		/// function to preserve dimensionality.
		/// The normally distributed results are transformed into log-normal distribution.
		/// <para/>
		/// There may be slight numerical differences between results generated
		/// on the GPU with generators created with ::curandCreateGenerator()
		/// and results calculated on the CPU with generators created with
		/// ::curandCreateGeneratorHost().  These differences arise because of
		/// differences in results for transcendental functions.  In addition,
		/// future versions of CURAND may use newer versions of the CUDA math
		/// library, so different versions of CURAND may give slightly different
		/// numerical values.
		/// </summary>
		/// <param name="output">CudaDeviceVariable</param>
		/// <param name="mean"></param>
		/// <param name="stddev"></param>
		public void GenerateLogNormal(double[] output, double mean, double stddev)
		{
			_status = CudaRandNativeMethods.curandGenerateLogNormalDouble(_generator, output, output.LongLength, mean, stddev);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "curandGenerateLogNormalDouble", _status));
			if (_status != CurandStatus.Success) throw new CudaRandException(_status);
		}

		/// <summary>
		/// Generate the starting state of the generator.  This function is
		/// automatically called by generation functions such as
		/// Generate(CudaDeviceVariable) and GenerateUniform(CudaDeviceVariable).
		/// It can be called manually for performance testing reasons to separate
		/// timings for starting state generation and random number generation.
		/// </summary>
		public void GenerateSeeds()
		{
			_status = CudaRandNativeMethods.curandGenerateSeeds(_generator);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "curandGenerateSeeds", _status));
			if (_status != CurandStatus.Success) throw new CudaRandException(_status);
		}
		#endregion

		#region Static Methods

		/// <summary>
		/// Get scramble constants that can be used for quasirandom number generation.
		/// <para/>
		/// The array contains constants for many dimensions. Each dimension
		/// has a single uint constant. 
		/// </summary>
		/// <returns></returns>
		public static uint[] GetScrambleConstants32()
		{
			return CudaRandDevice.GetScrambleConstants32();
		}

		/// <summary>
		/// Get scramble constants that can be used for quasirandom number generation.
		/// <para/>
		/// The array contains constants for many dimensions. Each dimension
		/// has a single ulong constant. 
		/// </summary>
		/// <returns></returns>
		public static ulong[] GetScrambleConstants64()
		{
			return CudaRandDevice.GetScrambleConstants64();
		}

		/// <summary>
		/// Get an array of direction vectors that can be used for quasirandom number generation.
		/// <para/>
		/// The array contains vectors for many dimensions. Each dimension
		/// has 32 vectors. Each individual vector is an unsigned int.
		/// <para/>
		/// Legal values for set are:
		/// - <see cref="DirectionVectorSet.JoeKuo6_32"/> (20,000 dimensions)
		/// - <see cref="DirectionVectorSet.ScrambledJoeKuo6_32"/> (20,000 dimensions)
		/// </summary>
		/// <param name="set"></param>
		/// <returns></returns>
		public static DirectionVectors32[] GetDirectionVectors32(DirectionVectorSet set)
		{
			return CudaRandDevice.GetDirectionVectors32(set);
		}

		/// <summary>
		/// Get an array of direction vectors that can be used for quasirandom number generation.
		/// <para/>
		/// The array contains vectors for many dimensions. Each dimension
		/// has 64 vectors. Each individual vector is an unsigned long long.
		/// <para/>
		/// Legal values for set are:
		/// - <see cref="DirectionVectorSet.JoeKuo6_64"/> (20,000 dimensions)
		/// - <see cref="DirectionVectorSet.ScrambledJoeKuo6_64"/> (20,000 dimensions)
		/// </summary>
		/// <param name="set"></param>
		/// <returns></returns>
		public static DirectionVectors64[] GetDirectionVectors64(DirectionVectorSet set)
		{
			return CudaRandDevice.GetDirectionVectors64(set);
		}

		/// <summary>
		/// Returns the version number of the dynamically linked CURAND library.   
		/// </summary>
		public static Version GetVersion()
		{
			return CudaRandDevice.GetVersion();
		}
		#endregion
	}
}
