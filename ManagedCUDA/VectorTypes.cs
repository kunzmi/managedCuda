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
using System.Globalization;
using System.Runtime.InteropServices;
using ManagedCuda.BasicTypes;

namespace ManagedCuda.VectorTypes
{
	/// <summary>
	/// Define a common interface for all CUDA vector types
	/// See http://blogs.msdn.com/b/ricom/archive/2006/09/07/745085.aspx why
	/// these vector types look like they are.
	/// </summary>
	public interface ICudaVectorType
	{
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		uint Size
		{
			get;
		}
	}

	/// <summary>
	/// Define a common interface for all CUDA vector types supported by CudaArrays
	/// </summary>
	public interface ICudaVectorTypeForArray
	{
		/// <summary>
		/// Returns the Channel number from vector type, e.g. 3 for float3
		/// </summary>
		/// <returns></returns>
		uint GetChannelNumber();

		/// <summary>
		/// Returns a matching CUArrayFormat. If none is availabe a CudaException is thrown.
		/// </summary>
		/// <returns></returns>
		CUArrayFormat GetCUArrayFormat();
	}

	#region dim
	/// <summary>
	/// CUDA dim3. In difference to the CUDA dim3 type, this dim3 initializes to 0 for each element.
	/// dim3 should be value-types so that we can pack it in an array. But C# value types (structs)
	/// do not garantee to execute an default constructor, why it doesn't exist.
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct dim3
	{
		/// <summary>
		/// X
		/// </summary>
		public uint x;
		/// <summary>
		/// Y
		/// </summary>
		public uint y;
		/// <summary>
		/// Z
		/// </summary>
		public uint z;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 Add(dim3 src, dim3 value)
		{
			dim3 ret = new dim3(src.x + value.x, src.y + value.y, src.z + value.z);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 Add(dim3 src, uint value)
		{
			dim3 ret = new dim3(src.x + value, src.y + value, src.z + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 Add(uint src, dim3 value)
		{
			dim3 ret = new dim3(src + value.x, src + value.y, src + value.z);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 Subtract(dim3 src, dim3 value)
		{
			dim3 ret = new dim3(src.x - value.x, src.y - value.y, src.z - value.z);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 Subtract(dim3 src, uint value)
		{
			dim3 ret = new dim3(src.x - value, src.y - value, src.z - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 Subtract(uint src, dim3 value)
		{
			dim3 ret = new dim3(src - value.x, src - value.y, src - value.z);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 Multiply(dim3 src, dim3 value)
		{
			dim3 ret = new dim3(src.x * value.x, src.y * value.y, src.z * value.z);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 Multiply(dim3 src, uint value)
		{
			dim3 ret = new dim3(src.x * value, src.y * value, src.z * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 Multiply(uint src, dim3 value)
		{
			dim3 ret = new dim3(src * value.x, src * value.y, src * value.z);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 Divide(dim3 src, dim3 value)
		{
			dim3 ret = new dim3(src.x / value.x, src.y / value.y, src.z / value.z);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 Divide(dim3 src, uint value)
		{
			dim3 ret = new dim3(src.x / value, src.y / value, src.z / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 Divide(uint src, dim3 value)
		{
			dim3 ret = new dim3(src / value.x, src / value.y, src / value.z);
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 operator +(dim3 src, dim3 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 operator +(dim3 src, uint value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 operator +(uint src, dim3 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 operator -(dim3 src, dim3 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 operator -(dim3 src, uint value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 operator -(uint src, dim3 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 operator *(dim3 src, dim3 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 operator *(dim3 src, uint value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 operator *(uint src, dim3 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 operator /(dim3 src, dim3 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 operator /(dim3 src, uint value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static dim3 operator /(uint src, dim3 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(dim3 src, dim3 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(dim3 src, dim3 value)
		{
			return !(src == value);
		}
		/// <summary>
		/// implicit cast
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public static implicit operator dim3(int value)
		{
			return new dim3(value);
		}
		/// <summary>
		/// implicit cast
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public static implicit operator dim3(uint value)
		{
			return new dim3(value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is dim3)) return false;

			dim3 value = (dim3)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(dim3 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode();
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1}; {2})", this.x, this.y, this.z);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue">X</param>
		/// <param name="yValue">Y</param>
		/// <param name="zValue">Z</param>
		public dim3(uint xValue, uint yValue, uint zValue)
		{
			this.x = xValue;
			this.y = yValue;
			this.z = zValue;
		}
		/// <summary>
		/// .z = 1
		/// </summary>
		/// <param name="xValue">X</param>
		/// <param name="yValue">Y</param>
		public dim3(uint xValue, uint yValue)
		{
			this.x = xValue;
			this.y = yValue;
			this.z = 1;
		}
		/// <summary>
		/// In contrast to other vector types the .y and .z values are set to 1 and not to val!
		/// </summary>
		/// <param name="val"></param>
		public dim3(uint val)
		{
			this.x = val;
			this.y = 1;
			this.z = 1;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue">X</param>
		/// <param name="yValue">Y</param>
		/// <param name="zValue">Z</param>
		public dim3(int xValue, int yValue, int zValue)
		{
			this.x = (uint)xValue;
			this.y = (uint)yValue;
			this.z = (uint)zValue;
		}
		/// <summary>
		/// .z = 1
		/// </summary>
		/// <param name="xValue">X</param>
		/// <param name="yValue">Y</param>
		public dim3(int xValue, int yValue)
		{
			this.x = (uint)xValue;
			this.y = (uint)yValue;
			this.z = 1;
		}
		/// <summary>
		/// In contrast to other vector types the .y and .z values are set to 1 and not to val!
		/// </summary>
		/// <param name="val"></param>
		public dim3(int val)
		{
			this.x = (uint)val;
			this.y = 1;
			this.z = 1;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static dim3 Min(dim3 aValue, dim3 bValue)
		{
			return new dim3(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y), Math.Min(aValue.z, bValue.z));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static dim3 Max(dim3 aValue, dim3 bValue)
		{
			return new dim3(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y), Math.Max(aValue.z, bValue.z));
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(dim3);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(dim3));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}
	#endregion

	#region complex
	/// <summary>
	/// cuDoubleComplex
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cuDoubleComplex : ICudaVectorType
	{
		/// <summary>
		/// real component
		/// </summary>
		public double real;
		/// <summary>
		/// imaginary component
		/// </summary>
		public double imag;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex Add(cuDoubleComplex src, cuDoubleComplex value)
		{
			cuDoubleComplex ret = new cuDoubleComplex(src.real + value.real, src.imag + value.imag);
			return ret;
		}

		/// <summary>
		/// Add only real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex Add(cuDoubleComplex src, double value)
		{
			cuDoubleComplex ret = new cuDoubleComplex(src.real + value, src.imag);
			return ret;
		}

		/// <summary>
		/// Add only real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex Add(double src, cuDoubleComplex value)
		{
			cuDoubleComplex ret = new cuDoubleComplex(src + value.real, value.imag);
			return ret;
		}

		/// <summary>
		/// Add only real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex Add(cuDoubleComplex src, cuDoubleReal value)
		{
			cuDoubleComplex ret = new cuDoubleComplex(src.real + value.real, src.imag);
			return ret;
		}

		/// <summary>
		/// Add only real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex Add(cuDoubleReal src, cuDoubleComplex value)
		{
			cuDoubleComplex ret = new cuDoubleComplex(src.real + value.real, value.imag);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex Subtract(cuDoubleComplex src, cuDoubleComplex value)
		{
			cuDoubleComplex ret = new cuDoubleComplex(src.real - value.real, src.imag - value.imag);
			return ret;
		}

		/// <summary>
		/// Substract real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex Subtract(cuDoubleComplex src, double value)
		{
			cuDoubleComplex ret = new cuDoubleComplex(src.real - value, src.imag);
			return ret;
		}

		/// <summary>
		/// Substract real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex Subtract(double src, cuDoubleComplex value)
		{
			cuDoubleComplex ret = new cuDoubleComplex(src - value.real, value.imag);
			return ret;
		}

		/// <summary>
		/// Substract real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex Subtract(cuDoubleComplex src, cuDoubleReal value)
		{
			cuDoubleComplex ret = new cuDoubleComplex(src.real - value.real, src.imag);
			return ret;
		}

		/// <summary>
		/// Substract real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex Subtract(cuDoubleReal src, cuDoubleComplex value)
		{
			cuDoubleComplex ret = new cuDoubleComplex(src.real - value.real, value.imag);
			return ret;
		}

		/// <summary>
		/// Complex Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex Multiply(cuDoubleComplex src, cuDoubleComplex value)
		{
			cuDoubleComplex ret = new cuDoubleComplex(src.real * value.real - src.imag * value.imag, src.real * value.imag + src.imag * value.real);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex Multiply(cuDoubleComplex src, double value)
		{
			cuDoubleComplex ret = new cuDoubleComplex(src.real * value, src.imag * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex Multiply(double src, cuDoubleComplex value)
		{
			cuDoubleComplex ret = new cuDoubleComplex(src * value.real, src * value.imag);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex Multiply(cuDoubleComplex src, cuDoubleReal value)
		{
			cuDoubleComplex ret = new cuDoubleComplex(src.real * value.real, src.imag * value.real);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex Multiply(cuDoubleReal src, cuDoubleComplex value)
		{
			cuDoubleComplex ret = new cuDoubleComplex(src.real * value.real, src.real * value.imag);
			return ret;
		}

		/// <summary>
		/// Complex Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex Divide(cuDoubleComplex src, cuDoubleComplex value)
		{
			double a = src.real;
			double b = src.imag;
			double c = value.real;
			double d = value.imag;
			double denominator = (c * c) + (d * d);
			double real = (a * c + b * d) / denominator;
			double imag = (b * c - a * d) / denominator;
			cuDoubleComplex ret = new cuDoubleComplex(real, imag);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex Divide(cuDoubleComplex src, double value)
		{
			cuDoubleComplex ret = new cuDoubleComplex(src.real / value, src.imag / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex Divide(double src, cuDoubleComplex value)
		{
			cuDoubleComplex ret = new cuDoubleComplex(src / value.real, src / value.imag);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex Divide(cuDoubleComplex src, cuDoubleReal value)
		{
			cuDoubleComplex ret = new cuDoubleComplex(src.real / value.real, src.imag / value.real);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex Divide(cuDoubleReal src, cuDoubleComplex value)
		{
			cuDoubleComplex ret = new cuDoubleComplex(src.real / value.real, src.real / value.imag);
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex operator +(cuDoubleComplex src, cuDoubleComplex value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex operator +(cuDoubleComplex src, double value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex operator +(double src, cuDoubleComplex value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex operator +(cuDoubleComplex src, cuDoubleReal value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex operator +(cuDoubleReal src, cuDoubleComplex value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex operator -(cuDoubleComplex src, cuDoubleComplex value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex operator -(cuDoubleComplex src, double value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex operator -(double src, cuDoubleComplex value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex operator -(cuDoubleComplex src, cuDoubleReal value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex operator -(cuDoubleReal src, cuDoubleComplex value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex operator *(cuDoubleComplex src, cuDoubleComplex value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex operator *(cuDoubleComplex src, double value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex operator *(double src, cuDoubleComplex value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex operator *(cuDoubleComplex src, cuDoubleReal value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex operator *(cuDoubleReal src, cuDoubleComplex value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex operator /(cuDoubleComplex src, cuDoubleComplex value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex operator /(cuDoubleComplex src, double value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex operator /(double src, cuDoubleComplex value)
		{
			return Divide(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex operator /(cuDoubleComplex src, cuDoubleReal value)
		{
			return Divide(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleComplex operator /(cuDoubleReal src, cuDoubleComplex value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(cuDoubleComplex src, cuDoubleComplex value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(cuDoubleComplex src, cuDoubleComplex value)
		{
			return !(src == value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(cuDoubleComplex src, cuDoubleReal value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(cuDoubleComplex src, cuDoubleReal value)
		{
			return !(src == value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(cuDoubleReal src, cuDoubleComplex value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(cuDoubleReal src, cuDoubleComplex value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;

			if (obj is cuDoubleComplex)
			{
				cuDoubleComplex value = (cuDoubleComplex)obj;
				bool ret = true;
				ret &= this.real == value.real;
				ret &= this.imag == value.imag;
				return ret;
			}

			if (obj is cuDoubleReal)
			{
				cuDoubleReal value = (cuDoubleReal)obj;
				bool ret = true;
				ret &= this.real == value.real;
				ret &= this.imag == 0;
				return ret;
			}
			return false;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public bool Equals(cuDoubleComplex obj)
		{
			bool ret = true;
			ret &= this.real == obj.real;
			ret &= this.imag == obj.imag;

			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public bool Equals(cuDoubleReal obj)
		{
			bool ret = true;
			ret &= this.real == obj.real;
			ret &= this.imag == 0;

			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return real.GetHashCode() ^ imag.GetHashCode();
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			if (imag < 0)
				return string.Format(CultureInfo.CurrentCulture, "{0} - {1}i", this.real, Math.Abs(this.imag));

			return string.Format(CultureInfo.CurrentCulture, "{0} + {1}i", this.real, this.imag);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="realValue"></param>
		/// <param name="imagValue"></param>
		public cuDoubleComplex(double realValue, double imagValue)
		{
			this.real = realValue;
			this.imag = imagValue;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="realValue"></param>
		public cuDoubleComplex(double realValue)
		{
			this.real = realValue;
			this.imag = 0;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(cuDoubleComplex);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(cuDoubleComplex));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// cuDoubleReal
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cuDoubleReal : ICudaVectorType
	{
		/// <summary>
		/// real component
		/// </summary>
		public double real;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal Add(cuDoubleReal src, cuDoubleReal value)
		{
			cuDoubleReal ret = new cuDoubleReal(src.real + value.real);
			return ret;
		}

		/// <summary>
		/// Add only real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal Add(cuDoubleReal src, double value)
		{
			cuDoubleReal ret = new cuDoubleReal(src.real + value);
			return ret;
		}

		/// <summary>
		/// Add only real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal Add(double src, cuDoubleReal value)
		{
			cuDoubleReal ret = new cuDoubleReal(src + value.real);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal Subtract(cuDoubleReal src, cuDoubleReal value)
		{
			cuDoubleReal ret = new cuDoubleReal(src.real - value.real);
			return ret;
		}

		/// <summary>
		/// Substract real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal Subtract(cuDoubleReal src, double value)
		{
			cuDoubleReal ret = new cuDoubleReal(src.real - value);
			return ret;
		}

		/// <summary>
		/// Substract real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal Subtract(double src, cuDoubleReal value)
		{
			cuDoubleReal ret = new cuDoubleReal(src - value.real);
			return ret;
		}

		/// <summary>
		/// Complex Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal Multiply(cuDoubleReal src, cuDoubleReal value)
		{
			cuDoubleReal ret = new cuDoubleReal(src.real * value.real);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal Multiply(cuDoubleReal src, double value)
		{
			cuDoubleReal ret = new cuDoubleReal(src.real * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal Multiply(double src, cuDoubleReal value)
		{
			cuDoubleReal ret = new cuDoubleReal(src * value.real);
			return ret;
		}

		/// <summary>
		/// Complex Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal Divide(cuDoubleReal src, cuDoubleReal value)
		{
			cuDoubleReal ret = new cuDoubleReal(src.real / value.real);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal Divide(cuDoubleReal src, double value)
		{
			cuDoubleReal ret = new cuDoubleReal(src.real / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal Divide(double src, cuDoubleReal value)
		{
			cuDoubleReal ret = new cuDoubleReal(src / value.real);
			return ret;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static double ToDouble(cuDoubleReal src)
		{
			return src.real;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static cuDoubleReal FromDouble(double src)
		{
			return new cuDoubleReal(src);
		}
		#endregion

		#region operators
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal operator +(cuDoubleReal src, cuDoubleReal value)
		{
			return Add(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal operator +(cuDoubleReal src, double value)
		{
			return Add(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal operator +(double src, cuDoubleReal value)
		{
			return Add(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal operator -(cuDoubleReal src, cuDoubleReal value)
		{
			return Subtract(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal operator -(cuDoubleReal src, double value)
		{
			return Subtract(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal operator -(double src, cuDoubleReal value)
		{
			return Subtract(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal operator *(cuDoubleReal src, cuDoubleReal value)
		{
			return Multiply(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal operator *(cuDoubleReal src, double value)
		{
			return Multiply(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal operator *(double src, cuDoubleReal value)
		{
			return Multiply(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal operator /(cuDoubleReal src, cuDoubleReal value)
		{
			return Divide(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal operator /(cuDoubleReal src, double value)
		{
			return Divide(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuDoubleReal operator /(double src, cuDoubleReal value)
		{
			return Divide(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(cuDoubleReal src, cuDoubleReal value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(cuDoubleReal src, cuDoubleReal value)
		{
			return !(src == value);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator double(cuDoubleReal src)
		{
			return ToDouble(src);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator cuDoubleReal(double src)
		{
			return FromDouble(src);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;

			if (obj is cuDoubleComplex)
			{
				cuDoubleComplex value = (cuDoubleComplex)obj;
				bool ret = true;
				ret &= this.real == value.real;
				ret &= value.imag == 0;
				return ret;
			}
			if (obj is cuDoubleReal)
			{
				cuDoubleReal value = (cuDoubleReal)obj;
				return this.real == value.real;
			}
			return false;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public bool Equals(cuDoubleComplex obj)
		{
			bool ret = true;
			ret &= this.real == obj.real;
			ret &= obj.imag == 0;

			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public bool Equals(cuDoubleReal obj)
		{
			return this.real == obj.real;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return real.GetHashCode();
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "{0} + {1}i)", this.real, 0);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="realValue"></param>
		public cuDoubleReal(double realValue)
		{
			this.real = realValue;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(cuDoubleReal);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(cuDoubleReal));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// cuFloatComplex
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cuFloatComplex : ICudaVectorType, ICudaVectorTypeForArray
	{
		/// <summary>
		/// real component
		/// </summary>
		public float real;
		/// <summary>
		/// imaginary component
		/// </summary>
		public float imag;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex Add(cuFloatComplex src, cuFloatComplex value)
		{
			cuFloatComplex ret = new cuFloatComplex(src.real + value.real, src.imag + value.imag);
			return ret;
		}

		/// <summary>
		/// Add only real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex Add(cuFloatComplex src, float value)
		{
			cuFloatComplex ret = new cuFloatComplex(src.real + value, src.imag);
			return ret;
		}

		/// <summary>
		/// Add only real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex Add(float src, cuFloatComplex value)
		{
			cuFloatComplex ret = new cuFloatComplex(src + value.real, value.imag);
			return ret;
		}

		/// <summary>
		/// Add only real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex Add(cuFloatComplex src, cuFloatReal value)
		{
			cuFloatComplex ret = new cuFloatComplex(src.real + value.real, src.imag);
			return ret;
		}

		/// <summary>
		/// Add only real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex Add(cuFloatReal src, cuFloatComplex value)
		{
			cuFloatComplex ret = new cuFloatComplex(src.real + value.real, value.imag);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex Subtract(cuFloatComplex src, cuFloatComplex value)
		{
			cuFloatComplex ret = new cuFloatComplex(src.real - value.real, src.imag - value.imag);
			return ret;
		}

		/// <summary>
		/// Substract real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex Subtract(cuFloatComplex src, float value)
		{
			cuFloatComplex ret = new cuFloatComplex(src.real - value, src.imag);
			return ret;
		}

		/// <summary>
		/// Substract real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex Subtract(float src, cuFloatComplex value)
		{
			cuFloatComplex ret = new cuFloatComplex(src - value.real, value.imag);
			return ret;
		}

		/// <summary>
		/// Substract real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex Subtract(cuFloatComplex src, cuFloatReal value)
		{
			cuFloatComplex ret = new cuFloatComplex(src.real - value.real, src.imag);
			return ret;
		}

		/// <summary>
		/// Substract real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex Subtract(cuFloatReal src, cuFloatComplex value)
		{
			cuFloatComplex ret = new cuFloatComplex(src.real - value.real, value.imag);
			return ret;
		}

		/// <summary>
		/// Complex Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex Multiply(cuFloatComplex src, cuFloatComplex value)
		{
			cuFloatComplex ret = new cuFloatComplex(src.real * value.real - src.imag * value.imag, src.real * value.imag + src.imag * value.real);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex Multiply(cuFloatComplex src, float value)
		{
			cuFloatComplex ret = new cuFloatComplex(src.real * value, src.imag * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex Multiply(float src, cuFloatComplex value)
		{
			cuFloatComplex ret = new cuFloatComplex(src * value.real, src * value.imag);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex Multiply(cuFloatComplex src, cuFloatReal value)
		{
			cuFloatComplex ret = new cuFloatComplex(src.real * value.real, src.imag * value.real);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex Multiply(cuFloatReal src, cuFloatComplex value)
		{
			cuFloatComplex ret = new cuFloatComplex(src.real * value.real, src.real * value.imag);
			return ret;
		}

		/// <summary>
		/// Complex Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex Divide(cuFloatComplex src, cuFloatComplex value)
		{
			float a = src.real;
			float b = src.imag;
			float c = value.real;
			float d = value.imag;
			float denominator = (c * c) + (d * d);
			float real = (a * c + b * d) / denominator;
			float imag = (b * c - a * d) / denominator;
			cuFloatComplex ret = new cuFloatComplex(real, imag);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex Divide(cuFloatComplex src, float value)
		{
			cuFloatComplex ret = new cuFloatComplex(src.real / value, src.imag / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex Divide(float src, cuFloatComplex value)
		{
			cuFloatComplex ret = new cuFloatComplex(src / value.real, src / value.imag);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex Divide(cuFloatComplex src, cuFloatReal value)
		{
			cuFloatComplex ret = new cuFloatComplex(src.real / value.real, src.imag / value.real);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex Divide(cuFloatReal src, cuFloatComplex value)
		{
			cuFloatComplex ret = new cuFloatComplex(src.real / value.real, src.real / value.imag);
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex operator +(cuFloatComplex src, cuFloatComplex value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex operator +(cuFloatComplex src, float value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex operator +(float src, cuFloatComplex value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex operator +(cuFloatComplex src, cuFloatReal value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex operator +(cuFloatReal src, cuFloatComplex value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex operator -(cuFloatComplex src, cuFloatComplex value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex operator -(cuFloatComplex src, float value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex operator -(float src, cuFloatComplex value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex operator -(cuFloatComplex src, cuFloatReal value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex operator -(cuFloatReal src, cuFloatComplex value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex operator *(cuFloatComplex src, cuFloatComplex value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex operator *(cuFloatComplex src, float value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex operator *(float src, cuFloatComplex value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex operator *(cuFloatComplex src, cuFloatReal value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex operator *(cuFloatReal src, cuFloatComplex value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex operator /(cuFloatComplex src, cuFloatComplex value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex operator /(cuFloatComplex src, float value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex operator /(float src, cuFloatComplex value)
		{
			return Divide(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex operator /(cuFloatComplex src, cuFloatReal value)
		{
			return Divide(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatComplex operator /(cuFloatReal src, cuFloatComplex value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(cuFloatComplex src, cuFloatComplex value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(cuFloatComplex src, cuFloatComplex value)
		{
			return !(src == value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(cuFloatComplex src, cuFloatReal value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(cuFloatComplex src, cuFloatReal value)
		{
			return !(src == value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(cuFloatReal src, cuFloatComplex value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(cuFloatReal src, cuFloatComplex value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;

			if (obj is cuFloatComplex)
			{
				cuFloatComplex value = (cuFloatComplex)obj;
				bool ret = true;
				ret &= this.real == value.real;
				ret &= this.imag == value.imag;
				return ret;
			}

			if (obj is cuFloatReal)
			{
				cuFloatReal value = (cuFloatReal)obj;
				bool ret = true;
				ret &= this.real == value.real;
				ret &= this.imag == 0;
				return ret;
			}
			return false;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public bool Equals(cuFloatComplex obj)
		{
			bool ret = true;
			ret &= this.real == obj.real;
			ret &= this.imag == obj.imag;

			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public bool Equals(cuFloatReal obj)
		{
			bool ret = true;
			ret &= this.real == obj.real;
			ret &= this.imag == 0;

			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return real.GetHashCode() ^ imag.GetHashCode();
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			if (imag < 0)
				return string.Format(CultureInfo.CurrentCulture, "{0} - {1}i", this.real, Math.Abs(this.imag));

			return string.Format(CultureInfo.CurrentCulture, "{0} + {1}i", this.real, this.imag);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="realValue"></param>
		/// <param name="imagValue"></param>
		public cuFloatComplex(float realValue, float imagValue)
		{
			this.real = realValue;
			this.imag = imagValue;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="realValue"></param>
		public cuFloatComplex(float realValue)
		{
			this.real = realValue;
			this.imag = 0;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(cuFloatComplex);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(cuFloatComplex));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Returns the Channel number from vector type.
		/// </summary>
		/// <returns>2</returns>
		public uint GetChannelNumber()
		{
			return 2;
		}

		/// <summary>
		/// Returns a matching CUArrayFormat.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat GetCUArrayFormat()
		{
			return CUArrayFormat.Float;
		}
		#endregion
	}

	/// <summary>
	/// cuFloatReal
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct cuFloatReal : ICudaVectorType, ICudaVectorTypeForArray
	{
		/// <summary>
		/// real component
		/// </summary>
		public float real;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal Add(cuFloatReal src, cuFloatReal value)
		{
			cuFloatReal ret = new cuFloatReal(src.real + value.real);
			return ret;
		}

		/// <summary>
		/// Add only real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal Add(cuFloatReal src, float value)
		{
			cuFloatReal ret = new cuFloatReal(src.real + value);
			return ret;
		}

		/// <summary>
		/// Add only real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal Add(float src, cuFloatReal value)
		{
			cuFloatReal ret = new cuFloatReal(src + value.real);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal Subtract(cuFloatReal src, cuFloatReal value)
		{
			cuFloatReal ret = new cuFloatReal(src.real - value.real);
			return ret;
		}

		/// <summary>
		/// Substract real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal Subtract(cuFloatReal src, float value)
		{
			cuFloatReal ret = new cuFloatReal(src.real - value);
			return ret;
		}

		/// <summary>
		/// Substract real part
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal Subtract(float src, cuFloatReal value)
		{
			cuFloatReal ret = new cuFloatReal(src - value.real);
			return ret;
		}

		/// <summary>
		/// Complex Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal Multiply(cuFloatReal src, cuFloatReal value)
		{
			cuFloatReal ret = new cuFloatReal(src.real * value.real);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal Multiply(cuFloatReal src, float value)
		{
			cuFloatReal ret = new cuFloatReal(src.real * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal Multiply(float src, cuFloatReal value)
		{
			cuFloatReal ret = new cuFloatReal(src * value.real);
			return ret;
		}

		/// <summary>
		/// Complex Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal Divide(cuFloatReal src, cuFloatReal value)
		{
			cuFloatReal ret = new cuFloatReal(src.real / value.real);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal Divide(cuFloatReal src, float value)
		{
			cuFloatReal ret = new cuFloatReal(src.real / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal Divide(float src, cuFloatReal value)
		{
			cuFloatReal ret = new cuFloatReal(src / value.real);
			return ret;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static float ToSingle(cuFloatReal src)
		{
			return src.real;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static cuFloatReal FromSingle(float src)
		{
			return new cuFloatReal(src);
		}
		#endregion

		#region operators
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal operator +(cuFloatReal src, cuFloatReal value)
		{
			return Add(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal operator +(cuFloatReal src, float value)
		{
			return Add(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal operator +(float src, cuFloatReal value)
		{
			return Add(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal operator -(cuFloatReal src, cuFloatReal value)
		{
			return Subtract(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal operator -(cuFloatReal src, float value)
		{
			return Subtract(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal operator -(float src, cuFloatReal value)
		{
			return Subtract(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal operator *(cuFloatReal src, cuFloatReal value)
		{
			return Multiply(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal operator *(cuFloatReal src, float value)
		{
			return Multiply(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal operator *(float src, cuFloatReal value)
		{
			return Multiply(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal operator /(cuFloatReal src, cuFloatReal value)
		{
			return Divide(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal operator /(cuFloatReal src, float value)
		{
			return Divide(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static cuFloatReal operator /(float src, cuFloatReal value)
		{
			return Divide(src, value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(cuFloatReal src, cuFloatReal value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// component wise
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(cuFloatReal src, cuFloatReal value)
		{
			return !(src == value);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator float(cuFloatReal src)
		{
			return ToSingle(src);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator cuFloatReal(float src)
		{
			return FromSingle(src);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (!(obj is cuFloatReal)) return false;

			cuFloatReal value = (cuFloatReal)obj;

			bool ret = true;
			ret &= this.real == value.real;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public bool Equals(cuFloatComplex obj)
		{
			bool ret = false;
			ret &= this.real == obj.real;
			ret &= obj.imag == 0;

			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public bool Equals(cuFloatReal obj)
		{
			return this.real == obj.real;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return real.GetHashCode();
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "{0} + {1}i", this.real, 0);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="realValue"></param>
		public cuFloatReal(float realValue)
		{
			this.real = realValue;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(cuFloatReal);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(cuFloatReal));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion

		#region Methods
		/// <summary>
		/// Returns the Channel number from vector type.
		/// </summary>
		/// <returns>1</returns>
		public uint GetChannelNumber()
		{
			return 1;
		}

		/// <summary>
		/// Returns a matching CUArrayFormat.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat GetCUArrayFormat()
		{
			return CUArrayFormat.Float;
		}
		#endregion
	}
	#endregion

	#region char
	/// <summary>
	/// char1
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct char1 : ICudaVectorType, ICudaVectorTypeForArray
	{
		/// <summary>
		/// X
		/// </summary>
		public sbyte x;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 Add(char1 src, char1 value)
		{
			char1 ret = new char1((sbyte)(src.x + value.x));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 Add(char1 src, sbyte value)
		{
			char1 ret = new char1((sbyte)(src.x + value));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 Add(sbyte src, char1 value)
		{
			char1 ret = new char1((sbyte)(src + value.x));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 Subtract(char1 src, char1 value)
		{
			char1 ret = new char1((sbyte)(src.x - value.x));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 Subtract(char1 src, sbyte value)
		{
			char1 ret = new char1((sbyte)(src.x - value));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 Subtract(sbyte src, char1 value)
		{
			char1 ret = new char1((sbyte)(src - value.x));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 Multiply(char1 src, char1 value)
		{
			char1 ret = new char1((sbyte)(src.x * value.x));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 Multiply(char1 src, sbyte value)
		{
			char1 ret = new char1((sbyte)(src.x * value));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 Multiply(sbyte src, char1 value)
		{
			char1 ret = new char1((sbyte)(src * value.x));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 Divide(char1 src, char1 value)
		{
			char1 ret = new char1((sbyte)(src.x / value.x));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 Divide(char1 src, sbyte value)
		{
			char1 ret = new char1((sbyte)(src.x / value));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 Divide(sbyte src, char1 value)
		{
			char1 ret = new char1((sbyte)(src / value.x));
			return ret;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static sbyte ToSByte(char1 src)
		{
			return src.x;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static char1 FromSByte(sbyte src)
		{
			return new char1(src);
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 operator +(char1 src, char1 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 operator +(char1 src, sbyte value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 operator +(sbyte src, char1 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 operator -(char1 src, char1 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 operator -(char1 src, sbyte value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 operator -(sbyte src, char1 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 operator *(char1 src, char1 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 operator *(char1 src, sbyte value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 operator *(sbyte src, char1 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 operator /(char1 src, char1 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 operator /(char1 src, sbyte value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char1 operator /(sbyte src, char1 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(char1 src, char1 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(char1 src, char1 value)
		{
			return !(src == value);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator sbyte(char1 src)
		{
			return ToSByte(src);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator char1(sbyte src)
		{
			return FromSByte(src);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is char1)) return false;

			char1 value = (char1)obj;

			bool ret = true;
			ret &= this.x == value.x;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public bool Equals(char1 obj)
		{
			bool ret = true;
			ret &= this.x == obj.x;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode();
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0})", this.x);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		public char1(sbyte xValue)
		{
			this.x = xValue;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static char1 Min(char1 aValue, char1 bValue)
		{
			return new char1(Math.Min(aValue.x, bValue.x));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static char1 Max(char1 aValue, char1 bValue)
		{
			return new char1(Math.Max(aValue.x, bValue.x));
		}

		/// <summary>
		/// Returns the Channel number from vector type.
		/// </summary>
		/// <returns>1</returns>
		public uint GetChannelNumber()
		{
			return 1;
		}

		/// <summary>
		/// Returns a matching CUArrayFormat.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat GetCUArrayFormat()
		{
			return CUArrayFormat.SignedInt8;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(char1);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(char1));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// char2
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct char2 : ICudaVectorType, ICudaVectorTypeForArray
	{
		/// <summary>
		/// X
		/// </summary>
		public sbyte x;
		/// <summary>
		/// Y
		/// </summary>
		public sbyte y;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 Add(char2 src, char2 value)
		{
			char2 ret = new char2((sbyte)(src.x + value.x), (sbyte)(src.y + value.y));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 Add(char2 src, sbyte value)
		{
			char2 ret = new char2((sbyte)(src.x + value), (sbyte)(src.y + value));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 Add(sbyte src, char2 value)
		{
			char2 ret = new char2((sbyte)(src + value.x), (sbyte)(src + value.y));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 Subtract(char2 src, char2 value)
		{
			char2 ret = new char2((sbyte)(src.x - value.x), (sbyte)(src.y - value.y));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 Subtract(char2 src, sbyte value)
		{
			char2 ret = new char2((sbyte)(src.x - value), (sbyte)(src.y - value));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 Subtract(sbyte src, char2 value)
		{
			char2 ret = new char2((sbyte)(src - value.x), (sbyte)(src - value.y));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 Multiply(char2 src, char2 value)
		{
			char2 ret = new char2((sbyte)(src.x * value.x), (sbyte)(src.y * value.y));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 Multiply(char2 src, sbyte value)
		{
			char2 ret = new char2((sbyte)(src.x * value), (sbyte)(src.y * value));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 Multiply(sbyte src, char2 value)
		{
			char2 ret = new char2((sbyte)(src * value.x), (sbyte)(src * value.y));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 Divide(char2 src, char2 value)
		{
			char2 ret = new char2((sbyte)(src.x / value.x), (sbyte)(src.y / value.y));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 Divide(char2 src, sbyte value)
		{
			char2 ret = new char2((sbyte)(src.x / value), (sbyte)(src.y / value));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 Divide(sbyte src, char2 value)
		{
			char2 ret = new char2((sbyte)(src / value.x), (sbyte)(src / value.y));
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 operator +(char2 src, char2 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 operator +(char2 src, sbyte value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 operator +(sbyte src, char2 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 operator -(char2 src, char2 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 operator -(char2 src, sbyte value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 operator -(sbyte src, char2 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 operator *(char2 src, char2 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 operator *(char2 src, sbyte value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 operator *(sbyte src, char2 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 operator /(char2 src, char2 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 operator /(char2 src, sbyte value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char2 operator /(sbyte src, char2 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(char2 src, char2 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(char2 src, char2 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is char2)) return false;

			char2 value = (char2)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(char2 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode();
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1})", this.x, this.y);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		public char2(sbyte xValue, sbyte yValue)
		{
			this.x = xValue;
			this.y = yValue;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public char2(sbyte val)
		{
			this.x = val;
			this.y = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static char2 Min(char2 aValue, char2 bValue)
		{
			return new char2(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static char2 Max(char2 aValue, char2 bValue)
		{
			return new char2(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y));
		}

		/// <summary>
		/// Returns the Channel number from vector type.
		/// </summary>
		/// <returns>2</returns>
		public uint GetChannelNumber()
		{
			return 2;
		}

		/// <summary>
		/// Returns a matching CUArrayFormat.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat GetCUArrayFormat()
		{
			return CUArrayFormat.SignedInt8;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(char2);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(char2));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// char3
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct char3 : ICudaVectorType
	{
		/// <summary>
		/// X
		/// </summary>
		public sbyte x;
		/// <summary>
		/// Y
		/// </summary>
		public sbyte y;
		/// <summary>
		/// Z
		/// </summary>
		public sbyte z;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 Add(char3 src, char3 value)
		{
			char3 ret = new char3((sbyte)(src.x + value.x), (sbyte)(src.y + value.y), (sbyte)(src.z + value.z));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 Add(char3 src, sbyte value)
		{
			char3 ret = new char3((sbyte)(src.x + value), (sbyte)(src.y + value), (sbyte)(src.z + value));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 Add(sbyte src, char3 value)
		{
			char3 ret = new char3((sbyte)(src + value.x), (sbyte)(src + value.y), (sbyte)(src + value.z));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 Subtract(char3 src, char3 value)
		{
			char3 ret = new char3((sbyte)(src.x - value.x), (sbyte)(src.y - value.y), (sbyte)(src.z - value.z));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 Subtract(char3 src, sbyte value)
		{
			char3 ret = new char3((sbyte)(src.x - value), (sbyte)(src.y - value), (sbyte)(src.z - value));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 Subtract(sbyte src, char3 value)
		{
			char3 ret = new char3((sbyte)(src - value.x), (sbyte)(src - value.y), (sbyte)(src - value.z));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 Multiply(char3 src, char3 value)
		{
			char3 ret = new char3((sbyte)(src.x * value.x), (sbyte)(src.y * value.y), (sbyte)(src.z * value.z));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 Multiply(char3 src, sbyte value)
		{
			char3 ret = new char3((sbyte)(src.x * value), (sbyte)(src.y * value), (sbyte)(src.z * value));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 Multiply(sbyte src, char3 value)
		{
			char3 ret = new char3((sbyte)(src * value.x), (sbyte)(src * value.y), (sbyte)(src * value.z));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 Divide(char3 src, char3 value)
		{
			char3 ret = new char3((sbyte)(src.x / value.x), (sbyte)(src.y / value.y), (sbyte)(src.z / value.z));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 Divide(char3 src, sbyte value)
		{
			char3 ret = new char3((sbyte)(src.x / value), (sbyte)(src.y / value), (sbyte)(src.z / value));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 Divide(sbyte src, char3 value)
		{
			char3 ret = new char3((sbyte)(src / value.x), (sbyte)(src / value.y), (sbyte)(src / value.z));
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 operator +(char3 src, char3 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 operator +(char3 src, sbyte value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 operator +(sbyte src, char3 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 operator -(char3 src, char3 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 operator -(char3 src, sbyte value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 operator -(sbyte src, char3 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 operator *(char3 src, char3 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 operator *(char3 src, sbyte value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 operator *(sbyte src, char3 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 operator /(char3 src, char3 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 operator /(char3 src, sbyte value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char3 operator /(sbyte src, char3 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(char3 src, char3 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(char3 src, char3 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is char3)) return false;

			char3 value = (char3)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(char3 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode();
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1}; {2})", this.x, this.y, this.z);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		/// <param name="zValue"></param>
		public char3(sbyte xValue, sbyte yValue, sbyte zValue)
		{
			this.x = xValue;
			this.y = yValue;
			this.z = zValue;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public char3(sbyte val)
		{
			this.x = val;
			this.y = val;
			this.z = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static char3 Min(char3 aValue, char3 bValue)
		{
			return new char3(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y), Math.Min(aValue.z, bValue.z));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static char3 Max(char3 aValue, char3 bValue)
		{
			return new char3(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y), Math.Max(aValue.z, bValue.z));
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(char3);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(char3));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// char4
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct char4 : ICudaVectorType, ICudaVectorTypeForArray
	{
		/// <summary>
		/// X
		/// </summary>
		public sbyte x;
		/// <summary>
		/// Y
		/// </summary>
		public sbyte y;
		/// <summary>
		/// Z
		/// </summary>
		public sbyte z;
		/// <summary>
		/// W
		/// </summary>
		public sbyte w;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 Add(char4 src, char4 value)
		{
			char4 ret = new char4((sbyte)(src.x + value.x), (sbyte)(src.y + value.y), (sbyte)(src.z + value.z), (sbyte)(src.w + value.w));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 Add(char4 src, sbyte value)
		{
			char4 ret = new char4((sbyte)(src.x + value), (sbyte)(src.y + value), (sbyte)(src.z + value), (sbyte)(src.w + value));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 Add(sbyte src, char4 value)
		{
			char4 ret = new char4((sbyte)(src + value.x), (sbyte)(src + value.y), (sbyte)(src + value.z), (sbyte)(src + value.w));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 Subtract(char4 src, char4 value)
		{
			char4 ret = new char4((sbyte)(src.x - value.x), (sbyte)(src.y - value.y), (sbyte)(src.z - value.z), (sbyte)(src.w - value.w));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 Subtract(char4 src, sbyte value)
		{
			char4 ret = new char4((sbyte)(src.x - value), (sbyte)(src.y - value), (sbyte)(src.z - value), (sbyte)(src.w - value));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 Subtract(sbyte src, char4 value)
		{
			char4 ret = new char4((sbyte)(src - value.x), (sbyte)(src - value.y), (sbyte)(src - value.z), (sbyte)(src - value.w));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 Multiply(char4 src, char4 value)
		{
			char4 ret = new char4((sbyte)(src.x * value.x), (sbyte)(src.y * value.y), (sbyte)(src.z * value.z), (sbyte)(src.w * value.w));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 Multiply(char4 src, sbyte value)
		{
			char4 ret = new char4((sbyte)(src.x * value), (sbyte)(src.y * value), (sbyte)(src.z * value), (sbyte)(src.w * value));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 Multiply(sbyte src, char4 value)
		{
			char4 ret = new char4((sbyte)(src * value.x), (sbyte)(src * value.y), (sbyte)(src * value.z), (sbyte)(src * value.w));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 Divide(char4 src, char4 value)
		{
			char4 ret = new char4((sbyte)(src.x / value.x), (sbyte)(src.y / value.y), (sbyte)(src.z / value.z), (sbyte)(src.w / value.w));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 Divide(char4 src, sbyte value)
		{
			char4 ret = new char4((sbyte)(src.x / value), (sbyte)(src.y / value), (sbyte)(src.z / value), (sbyte)(src.w / value));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 Divide(sbyte src, char4 value)
		{
			char4 ret = new char4((sbyte)(src / value.x), (sbyte)(src / value.y), (sbyte)(src / value.z), (sbyte)(src / value.w));
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 operator +(char4 src, char4 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 operator +(char4 src, sbyte value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 operator +(sbyte src, char4 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 operator -(char4 src, char4 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 operator -(char4 src, sbyte value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 operator -(sbyte src, char4 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 operator *(char4 src, char4 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 operator *(char4 src, sbyte value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 operator *(sbyte src, char4 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 operator /(char4 src, char4 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 operator /(char4 src, sbyte value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static char4 operator /(sbyte src, char4 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(char4 src, char4 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(char4 src, char4 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is char4)) return false;

			char4 value = (char4)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			ret &= this.w == value.w;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(char4 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			ret &= this.w == value.w;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode() ^ w.GetHashCode();
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1}; {2}; {3})", this.x, this.y, this.z, this.w);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		/// <param name="zValue"></param>
		/// <param name="wValue"></param>
		public char4(sbyte xValue, sbyte yValue, sbyte zValue, sbyte wValue)
		{
			this.x = xValue;
			this.y = yValue;
			this.z = zValue;
			this.w = wValue;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public char4(sbyte val)
		{
			this.x = val;
			this.y = val;
			this.z = val;
			this.w = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static char4 Min(char4 aValue, char4 bValue)
		{
			return new char4(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y), Math.Min(aValue.z, bValue.z), Math.Min(aValue.w, bValue.w));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static char4 Max(char4 aValue, char4 bValue)
		{
			return new char4(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y), Math.Max(aValue.z, bValue.z), Math.Max(aValue.w, bValue.w));
		}

		/// <summary>
		/// Returns the Channel number from vector type.
		/// </summary>
		/// <returns>4</returns>
		public uint GetChannelNumber()
		{
			return 4;
		}

		/// <summary>
		/// Returns a matching CUArrayFormat.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat GetCUArrayFormat()
		{
			return CUArrayFormat.SignedInt8;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(char4);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(char4));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}
	#endregion

	#region uchar
	/// <summary>
	/// uchar1
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct uchar1 : ICudaVectorType, ICudaVectorTypeForArray
	{
		/// <summary>
		/// X
		/// </summary>
		public byte x;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 Add(uchar1 src, uchar1 value)
		{
			uchar1 ret = new uchar1((byte)(src.x + value.x));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 Add(uchar1 src, byte value)
		{
			uchar1 ret = new uchar1((byte)(src.x + value));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 Add(byte src, uchar1 value)
		{
			uchar1 ret = new uchar1((byte)(src + value.x));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 Subtract(uchar1 src, uchar1 value)
		{
			uchar1 ret = new uchar1((byte)(src.x - value.x));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 Subtract(uchar1 src, byte value)
		{
			uchar1 ret = new uchar1((byte)(src.x - value));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 Subtract(byte src, uchar1 value)
		{
			uchar1 ret = new uchar1((byte)(src - value.x));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 Multiply(uchar1 src, uchar1 value)
		{
			uchar1 ret = new uchar1((byte)(src.x * value.x));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 Multiply(uchar1 src, byte value)
		{
			uchar1 ret = new uchar1((byte)(src.x * value));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 Multiply(byte src, uchar1 value)
		{
			uchar1 ret = new uchar1((byte)(src * value.x));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 Divide(uchar1 src, uchar1 value)
		{
			uchar1 ret = new uchar1((byte)(src.x / value.x));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 Divide(uchar1 src, byte value)
		{
			uchar1 ret = new uchar1((byte)(src.x / value));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 Divide(byte src, uchar1 value)
		{
			uchar1 ret = new uchar1((byte)(src / value.x));
			return ret;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static byte ToByte(uchar1 src)
		{
			return src.x;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static uchar1 FromByte(byte src)
		{
			return new uchar1(src);
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 operator +(uchar1 src, uchar1 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 operator +(uchar1 src, byte value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 operator +(byte src, uchar1 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 operator -(uchar1 src, uchar1 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 operator -(uchar1 src, byte value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 operator -(byte src, uchar1 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 operator *(uchar1 src, uchar1 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 operator *(uchar1 src, byte value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 operator *(byte src, uchar1 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 operator /(uchar1 src, uchar1 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 operator /(uchar1 src, byte value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar1 operator /(byte src, uchar1 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(uchar1 src, uchar1 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(uchar1 src, uchar1 value)
		{
			return !(src == value);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator byte(uchar1 src)
		{
			return ToByte(src);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator uchar1(byte src)
		{
			return FromByte(src);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is uchar1)) return false;

			uchar1 value = (uchar1)obj;

			bool ret = true;
			ret &= this.x == value.x;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(uchar1 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode();
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0})", this.x);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		public uchar1(byte xValue)
		{
			this.x = xValue;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static uchar1 Min(uchar1 aValue, uchar1 bValue)
		{
			return new uchar1(Math.Min(aValue.x, bValue.x));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static uchar1 Max(uchar1 aValue, uchar1 bValue)
		{
			return new uchar1(Math.Max(aValue.x, bValue.x));
		}

		/// <summary>
		/// Returns the Channel number from vector type.
		/// </summary>
		/// <returns>1</returns>
		public uint GetChannelNumber()
		{
			return 1;
		}

		/// <summary>
		/// Returns a matching CUArrayFormat.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat GetCUArrayFormat()
		{
			return CUArrayFormat.UnsignedInt8;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(uchar1);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(uchar1));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// uchar2
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct uchar2 : ICudaVectorType, ICudaVectorTypeForArray
	{
		/// <summary>
		/// X
		/// </summary>
		public byte x;
		/// <summary>
		/// Y
		/// </summary>
		public byte y;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 Add(uchar2 src, uchar2 value)
		{
			uchar2 ret = new uchar2((byte)(src.x + value.x), (byte)(src.y + value.y));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 Add(uchar2 src, byte value)
		{
			uchar2 ret = new uchar2((byte)(src.x + value), (byte)(src.y + value));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 Add(byte src, uchar2 value)
		{
			uchar2 ret = new uchar2((byte)(src + value.x), (byte)(src + value.y));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 Subtract(uchar2 src, uchar2 value)
		{
			uchar2 ret = new uchar2((byte)(src.x - value.x), (byte)(src.y - value.y));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 Subtract(uchar2 src, byte value)
		{
			uchar2 ret = new uchar2((byte)(src.x - value), (byte)(src.y - value));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 Subtract(byte src, uchar2 value)
		{
			uchar2 ret = new uchar2((byte)(src - value.x), (byte)(src - value.y));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 Multiply(uchar2 src, uchar2 value)
		{
			uchar2 ret = new uchar2((byte)(src.x * value.x), (byte)(src.y * value.y));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 Multiply(uchar2 src, byte value)
		{
			uchar2 ret = new uchar2((byte)(src.x * value), (byte)(src.y * value));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 Multiply(byte src, uchar2 value)
		{
			uchar2 ret = new uchar2((byte)(src * value.x), (byte)(src * value.y));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 Divide(uchar2 src, uchar2 value)
		{
			uchar2 ret = new uchar2((byte)(src.x / value.x), (byte)(src.y / value.y));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 Divide(uchar2 src, byte value)
		{
			uchar2 ret = new uchar2((byte)(src.x / value), (byte)(src.y / value));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 Divide(byte src, uchar2 value)
		{
			uchar2 ret = new uchar2((byte)(src / value.x), (byte)(src / value.y));
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 operator +(uchar2 src, uchar2 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 operator +(uchar2 src, byte value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 operator +(byte src, uchar2 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 operator -(uchar2 src, uchar2 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 operator -(uchar2 src, byte value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 operator -(byte src, uchar2 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 operator *(uchar2 src, uchar2 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 operator *(uchar2 src, byte value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 operator *(byte src, uchar2 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 operator /(uchar2 src, uchar2 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 operator /(uchar2 src, byte value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar2 operator /(byte src, uchar2 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(uchar2 src, uchar2 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(uchar2 src, uchar2 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is uchar2)) return false;

			uchar2 value = (uchar2)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(uchar2 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode();
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1})", this.x, this.y);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		public uchar2(byte xValue, byte yValue)
		{
			this.x = xValue;
			this.y = yValue;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public uchar2(byte val)
		{
			this.x = val;
			this.y = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static uchar2 Min(uchar2 aValue, uchar2 bValue)
		{
			return new uchar2(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static uchar2 Max(uchar2 aValue, uchar2 bValue)
		{
			return new uchar2(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y));
		}

		/// <summary>
		/// Returns the Channel number from vector type.
		/// </summary>
		/// <returns>2</returns>
		public uint GetChannelNumber()
		{
			return 2;
		}

		/// <summary>
		/// Returns a matching CUArrayFormat.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat GetCUArrayFormat()
		{
			return CUArrayFormat.UnsignedInt8;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(uchar2);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(uchar2));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// uchar3
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct uchar3 : ICudaVectorType
	{
		/// <summary>
		/// X
		/// </summary>
		public byte x;
		/// <summary>
		/// Y
		/// </summary>
		public byte y;
		/// <summary>
		/// Z
		/// </summary>
		public byte z;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 Add(uchar3 src, uchar3 value)
		{
			uchar3 ret = new uchar3((byte)(src.x + value.x), (byte)(src.y + value.y), (byte)(src.z + value.z));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 Add(uchar3 src, byte value)
		{
			uchar3 ret = new uchar3((byte)(src.x + value), (byte)(src.y + value), (byte)(src.z + value));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 Add(byte src, uchar3 value)
		{
			uchar3 ret = new uchar3((byte)(src + value.x), (byte)(src + value.y), (byte)(src + value.z));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 Subtract(uchar3 src, uchar3 value)
		{
			uchar3 ret = new uchar3((byte)(src.x - value.x), (byte)(src.y - value.y), (byte)(src.z - value.z));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 Subtract(uchar3 src, byte value)
		{
			uchar3 ret = new uchar3((byte)(src.x - value), (byte)(src.y - value), (byte)(src.z - value));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 Subtract(byte src, uchar3 value)
		{
			uchar3 ret = new uchar3((byte)(src - value.x), (byte)(src - value.y), (byte)(src - value.z));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 Multiply(uchar3 src, uchar3 value)
		{
			uchar3 ret = new uchar3((byte)(src.x * value.x), (byte)(src.y * value.y), (byte)(src.z * value.z));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 Multiply(uchar3 src, byte value)
		{
			uchar3 ret = new uchar3((byte)(src.x * value), (byte)(src.y * value), (byte)(src.z * value));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 Multiply(byte src, uchar3 value)
		{
			uchar3 ret = new uchar3((byte)(src * value.x), (byte)(src * value.y), (byte)(src * value.z));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 Divide(uchar3 src, uchar3 value)
		{
			uchar3 ret = new uchar3((byte)(src.x / value.x), (byte)(src.y / value.y), (byte)(src.z / value.z));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 Divide(uchar3 src, byte value)
		{
			uchar3 ret = new uchar3((byte)(src.x / value), (byte)(src.y / value), (byte)(src.z / value));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 Divide(byte src, uchar3 value)
		{
			uchar3 ret = new uchar3((byte)(src / value.x), (byte)(src / value.y), (byte)(src / value.z));
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 operator +(uchar3 src, uchar3 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 operator +(uchar3 src, byte value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 operator +(byte src, uchar3 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 operator -(uchar3 src, uchar3 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 operator -(uchar3 src, byte value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 operator -(byte src, uchar3 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 operator *(uchar3 src, uchar3 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 operator *(uchar3 src, byte value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 operator *(byte src, uchar3 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 operator /(uchar3 src, uchar3 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 operator /(uchar3 src, byte value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar3 operator /(byte src, uchar3 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(uchar3 src, uchar3 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(uchar3 src, uchar3 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is uchar3)) return false;

			uchar3 value = (uchar3)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(uchar3 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode();
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1}; {2})", this.x, this.y, this.z);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		/// <param name="zValue"></param>
		public uchar3(byte xValue, byte yValue, byte zValue)
		{
			this.x = xValue;
			this.y = yValue;
			this.z = zValue;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public uchar3(byte val)
		{
			this.x = val;
			this.y = val;
			this.z = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static uchar3 Min(uchar3 aValue, uchar3 bValue)
		{
			return new uchar3(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y), Math.Min(aValue.z, bValue.z));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static uchar3 Max(uchar3 aValue, uchar3 bValue)
		{
			return new uchar3(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y), Math.Max(aValue.z, bValue.z));
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(uchar3);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(uchar3));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// uchar4
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct uchar4 : ICudaVectorType, ICudaVectorTypeForArray
	{
		/// <summary>
		/// X
		/// </summary>
		public byte x;
		/// <summary>
		/// Y
		/// </summary>
		public byte y;
		/// <summary>
		/// Z
		/// </summary>
		public byte z;
		/// <summary>
		/// W
		/// </summary>
		public byte w;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 Add(uchar4 src, uchar4 value)
		{
			uchar4 ret = new uchar4((byte)(src.x + value.x), (byte)(src.y + value.y), (byte)(src.z + value.z), (byte)(src.w + value.w));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 Add(uchar4 src, byte value)
		{
			uchar4 ret = new uchar4((byte)(src.x + value), (byte)(src.y + value), (byte)(src.z + value), (byte)(src.w + value));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 Add(byte src, uchar4 value)
		{
			uchar4 ret = new uchar4((byte)(src + value.x), (byte)(src + value.y), (byte)(src + value.z), (byte)(src + value.w));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 Subtract(uchar4 src, uchar4 value)
		{
			uchar4 ret = new uchar4((byte)(src.x - value.x), (byte)(src.y - value.y), (byte)(src.z - value.z), (byte)(src.w - value.w));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 Subtract(uchar4 src, byte value)
		{
			uchar4 ret = new uchar4((byte)(src.x - value), (byte)(src.y - value), (byte)(src.z - value), (byte)(src.w - value));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 Subtract(byte src, uchar4 value)
		{
			uchar4 ret = new uchar4((byte)(src - value.x), (byte)(src - value.y), (byte)(src - value.z), (byte)(src - value.w));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 Multiply(uchar4 src, uchar4 value)
		{
			uchar4 ret = new uchar4((byte)(src.x * value.x), (byte)(src.y * value.y), (byte)(src.z * value.z), (byte)(src.w * value.w));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 Multiply(uchar4 src, byte value)
		{
			uchar4 ret = new uchar4((byte)(src.x * value), (byte)(src.y * value), (byte)(src.z * value), (byte)(src.w * value));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 Multiply(byte src, uchar4 value)
		{
			uchar4 ret = new uchar4((byte)(src * value.x), (byte)(src * value.y), (byte)(src * value.z), (byte)(src * value.w));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 Divide(uchar4 src, uchar4 value)
		{
			uchar4 ret = new uchar4((byte)(src.x / value.x), (byte)(src.y / value.y), (byte)(src.z / value.z), (byte)(src.w / value.w));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 Divide(uchar4 src, byte value)
		{
			uchar4 ret = new uchar4((byte)(src.x / value), (byte)(src.y / value), (byte)(src.z / value), (byte)(src.w / value));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 Divide(byte src, uchar4 value)
		{
			uchar4 ret = new uchar4((byte)(src / value.x), (byte)(src / value.y), (byte)(src / value.z), (byte)(src / value.w));
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 operator +(uchar4 src, uchar4 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 operator +(uchar4 src, byte value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 operator +(byte src, uchar4 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 operator -(uchar4 src, uchar4 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 operator -(uchar4 src, byte value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 operator -(byte src, uchar4 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 operator *(uchar4 src, uchar4 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 operator *(uchar4 src, byte value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 operator *(byte src, uchar4 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 operator /(uchar4 src, uchar4 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 operator /(uchar4 src, byte value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uchar4 operator /(byte src, uchar4 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(uchar4 src, uchar4 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(uchar4 src, uchar4 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is uchar4)) return false;

			uchar4 value = (uchar4)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			ret &= this.w == value.w;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(uchar4 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			ret &= this.w == value.w;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode() ^ w.GetHashCode();
		}
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1}; {2}; {3})", this.x, this.y, this.z, this.w);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		/// <param name="zValue"></param>
		/// <param name="wValue"></param>
		public uchar4(byte xValue, byte yValue, byte zValue, byte wValue)
		{
			this.x = xValue;
			this.y = yValue;
			this.z = zValue;
			this.w = wValue;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public uchar4(byte val)
		{
			this.x = val;
			this.y = val;
			this.z = val;
			this.w = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static uchar4 Min(uchar4 aValue, uchar4 bValue)
		{
			return new uchar4(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y), Math.Min(aValue.z, bValue.z), Math.Min(aValue.w, bValue.w));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static uchar4 Max(uchar4 aValue, uchar4 bValue)
		{
			return new uchar4(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y), Math.Max(aValue.z, bValue.z), Math.Max(aValue.w, bValue.w));
		}

		/// <summary>
		/// Returns the Channel number from vector type.
		/// </summary>
		/// <returns>4</returns>
		public uint GetChannelNumber()
		{
			return 4;
		}

		/// <summary>
		/// Returns a matching CUArrayFormat.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat GetCUArrayFormat()
		{
			return CUArrayFormat.UnsignedInt8;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(uchar4);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(uchar4));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}
	#endregion

	#region short
	/// <summary>
	/// short1
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct short1 : ICudaVectorType, ICudaVectorTypeForArray
	{
		/// <summary>
		/// X
		/// </summary>
		public short x;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 Add(short1 src, short1 value)
		{
			short1 ret = new short1((short)(src.x + value.x));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 Add(short1 src, short value)
		{
			short1 ret = new short1((short)(src.x + value));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 Add(short src, short1 value)
		{
			short1 ret = new short1((short)(src + value.x));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 Subtract(short1 src, short1 value)
		{
			short1 ret = new short1((short)(src.x - value.x));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 Subtract(short1 src, short value)
		{
			short1 ret = new short1((short)(src.x - value));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 Subtract(short src, short1 value)
		{
			short1 ret = new short1((short)(src - value.x));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 Multiply(short1 src, short1 value)
		{
			short1 ret = new short1((short)(src.x * value.x));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 Multiply(short1 src, short value)
		{
			short1 ret = new short1((short)(src.x * value));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 Multiply(short src, short1 value)
		{
			short1 ret = new short1((short)(src * value.x));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 Divide(short1 src, short1 value)
		{
			short1 ret = new short1((short)(src.x / value.x));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 Divide(short1 src, short value)
		{
			short1 ret = new short1((short)(src.x / value));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 Divide(short src, short1 value)
		{
			short1 ret = new short1((short)(src / value.x));
			return ret;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static short ToInt16(short1 src)
		{
			return src.x;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static short1 FromInt16(short src)
		{
			return new short1(src);
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 operator +(short1 src, short1 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 operator +(short1 src, short value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 operator +(short src, short1 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 operator -(short1 src, short1 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 operator -(short1 src, short value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 operator -(short src, short1 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 operator *(short1 src, short1 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 operator *(short1 src, short value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 operator *(short src, short1 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 operator /(short1 src, short1 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 operator /(short1 src, short value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short1 operator /(short src, short1 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(short1 src, short1 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(short1 src, short1 value)
		{
			return !(src == value);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator short(short1 src)
		{
			return ToInt16(src);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator short1(short src)
		{
			return FromInt16(src);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is short1)) return false;

			short1 value = (short1)obj;

			bool ret = true;
			ret &= this.x == value.x;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(short1 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode();
		}
		
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0})", this.x);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		public short1(short xValue)
		{
			this.x = xValue;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static short1 Min(short1 aValue, short1 bValue)
		{
			return new short1(Math.Min(aValue.x, bValue.x));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static short1 Max(short1 aValue, short1 bValue)
		{
			return new short1(Math.Max(aValue.x, bValue.x));
		}

		/// <summary>
		/// Returns the Channel number from vector type.
		/// </summary>
		/// <returns>1</returns>
		public uint GetChannelNumber()
		{
			return 1;
		}

		/// <summary>
		/// Returns a matching CUArrayFormat.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat GetCUArrayFormat()
		{
			return CUArrayFormat.SignedInt16;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(short1);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(short1));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// short2
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct short2 : ICudaVectorType, ICudaVectorTypeForArray
	{
		/// <summary>
		/// X
		/// </summary>
		public short x;
		/// <summary>
		/// Y
		/// </summary>
		public short y;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 Add(short2 src, short2 value)
		{
			short2 ret = new short2((short)(src.x + value.x), (short)(src.y + value.y));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 Add(short2 src, short value)
		{
			short2 ret = new short2((short)(src.x + value), (short)(src.y + value));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 Add(short src, short2 value)
		{
			short2 ret = new short2((short)(src + value.x), (short)(src + value.y));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 Subtract(short2 src, short2 value)
		{
			short2 ret = new short2((short)(src.x - value.x), (short)(src.y - value.y));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 Subtract(short2 src, short value)
		{
			short2 ret = new short2((short)(src.x - value), (short)(src.y - value));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 Subtract(short src, short2 value)
		{
			short2 ret = new short2((short)(src - value.x), (short)(src - value.y));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 Multiply(short2 src, short2 value)
		{
			short2 ret = new short2((short)(src.x * value.x), (short)(src.y * value.y));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 Multiply(short2 src, short value)
		{
			short2 ret = new short2((short)(src.x * value), (short)(src.y * value));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 Multiply(short src, short2 value)
		{
			short2 ret = new short2((short)(src * value.x), (short)(src * value.y));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 Divide(short2 src, short2 value)
		{
			short2 ret = new short2((short)(src.x / value.x), (short)(src.y / value.y));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 Divide(short2 src, short value)
		{
			short2 ret = new short2((short)(src.x / value), (short)(src.y / value));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 Divide(short src, short2 value)
		{
			short2 ret = new short2((short)(src / value.x), (short)(src / value.y));
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 operator +(short2 src, short2 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 operator +(short2 src, short value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 operator +(short src, short2 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 operator -(short2 src, short2 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 operator -(short2 src, short value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 operator -(short src, short2 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 operator *(short2 src, short2 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 operator *(short2 src, short value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 operator *(short src, short2 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 operator /(short2 src, short2 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 operator /(short2 src, short value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short2 operator /(short src, short2 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(short2 src, short2 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(short2 src, short2 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is short2)) return false;

			short2 value = (short2)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(short2 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1})", this.x, this.y);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		public short2(short xValue, short yValue)
		{
			this.x = xValue;
			this.y = yValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public short2(short val)
		{
			this.x = val;
			this.y = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static short2 Min(short2 aValue, short2 bValue)
		{
			return new short2(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static short2 Max(short2 aValue, short2 bValue)
		{
			return new short2(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y));
		}

		/// <summary>
		/// Returns the Channel number from vector type.
		/// </summary>
		/// <returns>2</returns>
		public uint GetChannelNumber()
		{
			return 2;
		}

		/// <summary>
		/// Returns a matching CUArrayFormat.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat GetCUArrayFormat()
		{
			return CUArrayFormat.SignedInt16;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(short2);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(short2));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// short3
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct short3 : ICudaVectorType
	{
		/// <summary>
		/// X
		/// </summary>
		public short x;
		/// <summary>
		/// Y
		/// </summary>
		public short y;
		/// <summary>
		/// Z
		/// </summary>
		public short z;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 Add(short3 src, short3 value)
		{
			short3 ret = new short3((short)(src.x + value.x), (short)(src.y + value.y), (short)(src.z + value.z));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 Add(short3 src, short value)
		{
			short3 ret = new short3((short)(src.x + value), (short)(src.y + value), (short)(src.z + value));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 Add(short src, short3 value)
		{
			short3 ret = new short3((short)(src + value.x), (short)(src + value.y), (short)(src + value.z));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 Subtract(short3 src, short3 value)
		{
			short3 ret = new short3((short)(src.x - value.x), (short)(src.y - value.y), (short)(src.z - value.z));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 Subtract(short3 src, short value)
		{
			short3 ret = new short3((short)(src.x - value), (short)(src.y - value), (short)(src.z - value));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 Subtract(short src, short3 value)
		{
			short3 ret = new short3((short)(src - value.x), (short)(src - value.y), (short)(src - value.z));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 Multiply(short3 src, short3 value)
		{
			short3 ret = new short3((short)(src.x * value.x), (short)(src.y * value.y), (short)(src.z * value.z));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 Multiply(short3 src, short value)
		{
			short3 ret = new short3((short)(src.x * value), (short)(src.y * value), (short)(src.z * value));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 Multiply(short src, short3 value)
		{
			short3 ret = new short3((short)(src * value.x), (short)(src * value.y), (short)(src * value.z));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 Divide(short3 src, short3 value)
		{
			short3 ret = new short3((short)(src.x / value.x), (short)(src.y / value.y), (short)(src.z / value.z));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 Divide(short3 src, short value)
		{
			short3 ret = new short3((short)(src.x / value), (short)(src.y / value), (short)(src.z / value));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 Divide(short src, short3 value)
		{
			short3 ret = new short3((short)(src / value.x), (short)(src / value.y), (short)(src / value.z));
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 operator +(short3 src, short3 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 operator +(short3 src, short value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 operator +(short src, short3 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 operator -(short3 src, short3 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 operator -(short3 src, short value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 operator -(short src, short3 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 operator *(short3 src, short3 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 operator *(short3 src, short value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 operator *(short src, short3 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 operator /(short3 src, short3 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 operator /(short3 src, short value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short3 operator /(short src, short3 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(short3 src, short3 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(short3 src, short3 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is short3)) return false;

			short3 value = (short3)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(short3 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1}; {2})", this.x, this.y, this.z);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		/// <param name="zValue"></param>
		public short3(short xValue, short yValue, short zValue)
		{
			this.x = xValue;
			this.y = yValue;
			this.z = zValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public short3(short val)
		{
			this.x = val;
			this.y = val;
			this.z = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static short3 Min(short3 aValue, short3 bValue)
		{
			return new short3(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y), Math.Min(aValue.z, bValue.z));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static short3 Max(short3 aValue, short3 bValue)
		{
			return new short3(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y), Math.Max(aValue.z, bValue.z));
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(short3);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(short3));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// short4
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct short4 : ICudaVectorType, ICudaVectorTypeForArray
	{
		/// <summary>
		/// X
		/// </summary>
		public short x;
		/// <summary>
		/// Y
		/// </summary>
		public short y;
		/// <summary>
		/// Z
		/// </summary>
		public short z;
		/// <summary>
		/// W
		/// </summary>
		public short w;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 Add(short4 src, short4 value)
		{
			short4 ret = new short4((short)(src.x + value.x), (short)(src.y + value.y), (short)(src.z + value.z), (short)(src.w + value.w));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 Add(short4 src, short value)
		{
			short4 ret = new short4((short)(src.x + value), (short)(src.y + value), (short)(src.z + value), (short)(src.w + value));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 Add(short src, short4 value)
		{
			short4 ret = new short4((short)(src + value.x), (short)(src + value.y), (short)(src + value.z), (short)(src + value.w));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 Subtract(short4 src, short4 value)
		{
			short4 ret = new short4((short)(src.x - value.x), (short)(src.y - value.y), (short)(src.z - value.z), (short)(src.w - value.w));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 Subtract(short4 src, short value)
		{
			short4 ret = new short4((short)(src.x - value), (short)(src.y - value), (short)(src.z - value), (short)(src.w - value));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 Subtract(short src, short4 value)
		{
			short4 ret = new short4((short)(src - value.x), (short)(src - value.y), (short)(src - value.z), (short)(src - value.w));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 Multiply(short4 src, short4 value)
		{
			short4 ret = new short4((short)(src.x * value.x), (short)(src.y * value.y), (short)(src.z * value.z), (short)(src.w * value.w));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 Multiply(short4 src, short value)
		{
			short4 ret = new short4((short)(src.x * value), (short)(src.y * value), (short)(src.z * value), (short)(src.w * value));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 Multiply(short src, short4 value)
		{
			short4 ret = new short4((short)(src * value.x), (short)(src * value.y), (short)(src * value.z), (short)(src * value.w));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 Divide(short4 src, short4 value)
		{
			short4 ret = new short4((short)(src.x / value.x), (short)(src.y / value.y), (short)(src.z / value.z), (short)(src.w / value.w));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 Divide(short4 src, short value)
		{
			short4 ret = new short4((short)(src.x / value), (short)(src.y / value), (short)(src.z / value), (short)(src.w / value));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 Divide(short src, short4 value)
		{
			short4 ret = new short4((short)(src / value.x), (short)(src / value.y), (short)(src / value.z), (short)(src / value.w));
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 operator +(short4 src, short4 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 operator +(short4 src, short value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 operator +(short src, short4 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 operator -(short4 src, short4 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 operator -(short4 src, short value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 operator -(short src, short4 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 operator *(short4 src, short4 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 operator *(short4 src, short value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 operator *(short src, short4 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 operator /(short4 src, short4 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 operator /(short4 src, short value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static short4 operator /(short src, short4 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(short4 src, short4 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(short4 src, short4 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is short4)) return false;

			short4 value = (short4)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			ret &= this.w == value.w;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(short4 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			ret &= this.w == value.w;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode() ^ w.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1}; {2}; {3})", this.x, this.y, this.z, this.w);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		/// <param name="zValue"></param>
		/// <param name="wValue"></param>
		public short4(short xValue, short yValue, short zValue, short wValue)
		{
			this.x = xValue;
			this.y = yValue;
			this.z = zValue;
			this.w = wValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public short4(short val)
		{
			this.x = val;
			this.y = val;
			this.z = val;
			this.w = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static short4 Min(short4 aValue, short4 bValue)
		{
			return new short4(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y), Math.Min(aValue.z, bValue.z), Math.Min(aValue.w, bValue.w));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static short4 Max(short4 aValue, short4 bValue)
		{
			return new short4(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y), Math.Max(aValue.z, bValue.z), Math.Max(aValue.w, bValue.w));
		}

		/// <summary>
		/// Returns the Channel number from vector type.
		/// </summary>
		/// <returns>4</returns>
		public uint GetChannelNumber()
		{
			return 4;
		}

		/// <summary>
		/// Returns a matching CUArrayFormat.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat GetCUArrayFormat()
		{
			return CUArrayFormat.SignedInt16;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(short4);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(short4));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}
	#endregion

	#region ushort
	/// <summary>
	/// ushort1
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct ushort1 : ICudaVectorType, ICudaVectorTypeForArray
	{
		/// <summary>
		/// X
		/// </summary>
		public ushort x;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 Add(ushort1 src, ushort1 value)
		{
			ushort1 ret = new ushort1((ushort)(src.x + value.x));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 Add(ushort1 src, ushort value)
		{
			ushort1 ret = new ushort1((ushort)(src.x + value));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 Add(ushort src, ushort1 value)
		{
			ushort1 ret = new ushort1((ushort)(src + value.x));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 Subtract(ushort1 src, ushort1 value)
		{
			ushort1 ret = new ushort1((ushort)(src.x - value.x));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 Subtract(ushort1 src, ushort value)
		{
			ushort1 ret = new ushort1((ushort)(src.x - value));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 Subtract(ushort src, ushort1 value)
		{
			ushort1 ret = new ushort1((ushort)(src - value.x));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 Multiply(ushort1 src, ushort1 value)
		{
			ushort1 ret = new ushort1((ushort)(src.x * value.x));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 Multiply(ushort1 src, ushort value)
		{
			ushort1 ret = new ushort1((ushort)(src.x * value));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 Multiply(ushort src, ushort1 value)
		{
			ushort1 ret = new ushort1((ushort)(src * value.x));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 Divide(ushort1 src, ushort1 value)
		{
			ushort1 ret = new ushort1((ushort)(src.x / value.x));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 Divide(ushort1 src, ushort value)
		{
			ushort1 ret = new ushort1((ushort)(src.x / value));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 Divide(ushort src, ushort1 value)
		{
			ushort1 ret = new ushort1((ushort)(src / value.x));
			return ret;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static ushort ToUInt16(ushort1 src)
		{
			return src.x;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static ushort1 FromUInt16(ushort src)
		{
			return new ushort1(src);
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 operator +(ushort1 src, ushort1 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 operator +(ushort1 src, ushort value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 operator +(ushort src, ushort1 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 operator -(ushort1 src, ushort1 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 operator -(ushort1 src, ushort value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 operator -(ushort src, ushort1 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 operator *(ushort1 src, ushort1 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 operator *(ushort1 src, ushort value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 operator *(ushort src, ushort1 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 operator /(ushort1 src, ushort1 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 operator /(ushort1 src, ushort value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort1 operator /(ushort src, ushort1 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(ushort1 src, ushort1 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(ushort1 src, ushort1 value)
		{
			return !(src == value);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator ushort(ushort1 src)
		{
			return ToUInt16(src);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator ushort1(ushort src)
		{
			return FromUInt16(src);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is ushort1)) return false;

			ushort1 value = (ushort1)obj;

			bool ret = true;
			ret &= this.x == value.x;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(ushort1 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0})", this.x);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		public ushort1(ushort xValue)
		{
			this.x = xValue;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static ushort1 Min(ushort1 aValue, ushort1 bValue)
		{
			return new ushort1(Math.Min(aValue.x, bValue.x));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static ushort1 Max(ushort1 aValue, ushort1 bValue)
		{
			return new ushort1(Math.Max(aValue.x, bValue.x));
		}

		/// <summary>
		/// Returns the Channel number from vector type.
		/// </summary>
		/// <returns>1</returns>
		public uint GetChannelNumber()
		{
			return 1;
		}

		/// <summary>
		/// Returns a matching CUArrayFormat.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat GetCUArrayFormat()
		{
			return CUArrayFormat.UnsignedInt16;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(ushort1);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(ushort1));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// ushort2
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct ushort2 : ICudaVectorType, ICudaVectorTypeForArray
	{
		/// <summary>
		/// X
		/// </summary>
		public ushort x;
		/// <summary>
		/// Y
		/// </summary>
		public ushort y;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 Add(ushort2 src, ushort2 value)
		{
			ushort2 ret = new ushort2((ushort)(src.x + value.x), (ushort)(src.y + value.y));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 Add(ushort2 src, ushort value)
		{
			ushort2 ret = new ushort2((ushort)(src.x + value), (ushort)(src.y + value));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 Add(ushort src, ushort2 value)
		{
			ushort2 ret = new ushort2((ushort)(src + value.x), (ushort)(src + value.y));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 Subtract(ushort2 src, ushort2 value)
		{
			ushort2 ret = new ushort2((ushort)(src.x - value.x), (ushort)(src.y - value.y));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 Subtract(ushort2 src, ushort value)
		{
			ushort2 ret = new ushort2((ushort)(src.x - value), (ushort)(src.y - value));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 Subtract(ushort src, ushort2 value)
		{
			ushort2 ret = new ushort2((ushort)(src - value.x), (ushort)(src - value.y));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 Multiply(ushort2 src, ushort2 value)
		{
			ushort2 ret = new ushort2((ushort)(src.x * value.x), (ushort)(src.y * value.y));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 Multiply(ushort2 src, ushort value)
		{
			ushort2 ret = new ushort2((ushort)(src.x * value), (ushort)(src.y * value));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 Multiply(ushort src, ushort2 value)
		{
			ushort2 ret = new ushort2((ushort)(src * value.x), (ushort)(src * value.y));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 Divide(ushort2 src, ushort2 value)
		{
			ushort2 ret = new ushort2((ushort)(src.x / value.x), (ushort)(src.y / value.y));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 Divide(ushort2 src, ushort value)
		{
			ushort2 ret = new ushort2((ushort)(src.x / value), (ushort)(src.y / value));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 Divide(ushort src, ushort2 value)
		{
			ushort2 ret = new ushort2((ushort)(src / value.x), (ushort)(src / value.y));
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 operator +(ushort2 src, ushort2 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 operator +(ushort2 src, ushort value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 operator +(ushort src, ushort2 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 operator -(ushort2 src, ushort2 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 operator -(ushort2 src, ushort value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 operator -(ushort src, ushort2 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 operator *(ushort2 src, ushort2 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 operator *(ushort2 src, ushort value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 operator *(ushort src, ushort2 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 operator /(ushort2 src, ushort2 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 operator /(ushort2 src, ushort value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort2 operator /(ushort src, ushort2 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(ushort2 src, ushort2 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(ushort2 src, ushort2 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is ushort2)) return false;

			ushort2 value = (ushort2)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(ushort2 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1})", this.x, this.y);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		public ushort2(ushort xValue, ushort yValue)
		{
			this.x = xValue;
			this.y = yValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public ushort2(ushort val)
		{
			this.x = val;
			this.y = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static ushort2 Min(ushort2 aValue, ushort2 bValue)
		{
			return new ushort2(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static ushort2 Max(ushort2 aValue, ushort2 bValue)
		{
			return new ushort2(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y));
		}

		/// <summary>
		/// Returns the Channel number from vector type.
		/// </summary>
		/// <returns>2</returns>
		public uint GetChannelNumber()
		{
			return 2;
		}

		/// <summary>
		/// Returns a matching CUArrayFormat.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat GetCUArrayFormat()
		{
			return CUArrayFormat.UnsignedInt16;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(ushort2);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(ushort2));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// ushort3
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct ushort3 : ICudaVectorType
	{
		/// <summary>
		/// X
		/// </summary>
		public ushort x;
		/// <summary>
		/// Y
		/// </summary>
		public ushort y;
		/// <summary>
		/// Z
		/// </summary>
		public ushort z;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 Add(ushort3 src, ushort3 value)
		{
			ushort3 ret = new ushort3((ushort)(src.x + value.x), (ushort)(src.y + value.y), (ushort)(src.z + value.z));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 Add(ushort3 src, ushort value)
		{
			ushort3 ret = new ushort3((ushort)(src.x + value), (ushort)(src.y + value), (ushort)(src.z + value));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 Add(ushort src, ushort3 value)
		{
			ushort3 ret = new ushort3((ushort)(src + value.x), (ushort)(src + value.y), (ushort)(src + value.z));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 Subtract(ushort3 src, ushort3 value)
		{
			ushort3 ret = new ushort3((ushort)(src.x - value.x), (ushort)(src.y - value.y), (ushort)(src.z - value.z));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 Subtract(ushort3 src, ushort value)
		{
			ushort3 ret = new ushort3((ushort)(src.x - value), (ushort)(src.y - value), (ushort)(src.z - value));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 Subtract(ushort src, ushort3 value)
		{
			ushort3 ret = new ushort3((ushort)(src - value.x), (ushort)(src - value.y), (ushort)(src - value.z));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 Multiply(ushort3 src, ushort3 value)
		{
			ushort3 ret = new ushort3((ushort)(src.x * value.x), (ushort)(src.y * value.y), (ushort)(src.z * value.z));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 Multiply(ushort3 src, ushort value)
		{
			ushort3 ret = new ushort3((ushort)(src.x * value), (ushort)(src.y * value), (ushort)(src.z * value));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 Multiply(ushort src, ushort3 value)
		{
			ushort3 ret = new ushort3((ushort)(src * value.x), (ushort)(src * value.y), (ushort)(src * value.z));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 Divide(ushort3 src, ushort3 value)
		{
			ushort3 ret = new ushort3((ushort)(src.x / value.x), (ushort)(src.y / value.y), (ushort)(src.z / value.z));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 Divide(ushort3 src, ushort value)
		{
			ushort3 ret = new ushort3((ushort)(src.x / value), (ushort)(src.y / value), (ushort)(src.z / value));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 Divide(ushort src, ushort3 value)
		{
			ushort3 ret = new ushort3((ushort)(src / value.x), (ushort)(src / value.y), (ushort)(src / value.z));
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 operator +(ushort3 src, ushort3 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 operator +(ushort3 src, ushort value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 operator +(ushort src, ushort3 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 operator -(ushort3 src, ushort3 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 operator -(ushort3 src, ushort value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 operator -(ushort src, ushort3 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 operator *(ushort3 src, ushort3 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 operator *(ushort3 src, ushort value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 operator *(ushort src, ushort3 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 operator /(ushort3 src, ushort3 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 operator /(ushort3 src, ushort value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort3 operator /(ushort src, ushort3 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(ushort3 src, ushort3 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(ushort3 src, ushort3 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is ushort3)) return false;

			ushort3 value = (ushort3)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(ushort3 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1}; {2})", this.x, this.y, this.z);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		/// <param name="zValue"></param>
		public ushort3(ushort xValue, ushort yValue, ushort zValue)
		{
			this.x = xValue;
			this.y = yValue;
			this.z = zValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public ushort3(ushort val)
		{
			this.x = val;
			this.y = val;
			this.z = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static ushort3 Min(ushort3 aValue, ushort3 bValue)
		{
			return new ushort3(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y), Math.Min(aValue.z, bValue.z));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static ushort3 Max(ushort3 aValue, ushort3 bValue)
		{
			return new ushort3(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y), Math.Max(aValue.z, bValue.z));
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(ushort3);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(ushort3));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// 
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct ushort4 : ICudaVectorType, ICudaVectorTypeForArray
	{
		/// <summary>
		/// X
		/// </summary>
		public ushort x;
		/// <summary>
		/// Y
		/// </summary>
		public ushort y;
		/// <summary>
		/// Z
		/// </summary>
		public ushort z;
		/// <summary>
		/// W
		/// </summary>
		public ushort w;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 Add(ushort4 src, ushort4 value)
		{
			ushort4 ret = new ushort4((ushort)(src.x + value.x), (ushort)(src.y + value.y), (ushort)(src.z + value.z), (ushort)(src.w + value.w));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 Add(ushort4 src, ushort value)
		{
			ushort4 ret = new ushort4((ushort)(src.x + value), (ushort)(src.y + value), (ushort)(src.z + value), (ushort)(src.w + value));
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 Add(ushort src, ushort4 value)
		{
			ushort4 ret = new ushort4((ushort)(src + value.x), (ushort)(src + value.y), (ushort)(src + value.z), (ushort)(src + value.w));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 Subtract(ushort4 src, ushort4 value)
		{
			ushort4 ret = new ushort4((ushort)(src.x - value.x), (ushort)(src.y - value.y), (ushort)(src.z - value.z), (ushort)(src.w - value.w));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 Subtract(ushort4 src, ushort value)
		{
			ushort4 ret = new ushort4((ushort)(src.x - value), (ushort)(src.y - value), (ushort)(src.z - value), (ushort)(src.w - value));
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 Subtract(ushort src, ushort4 value)
		{
			ushort4 ret = new ushort4((ushort)(src - value.x), (ushort)(src - value.y), (ushort)(src - value.z), (ushort)(src - value.w));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 Multiply(ushort4 src, ushort4 value)
		{
			ushort4 ret = new ushort4((ushort)(src.x * value.x), (ushort)(src.y * value.y), (ushort)(src.z * value.z), (ushort)(src.w * value.w));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 Multiply(ushort4 src, ushort value)
		{
			ushort4 ret = new ushort4((ushort)(src.x * value), (ushort)(src.y * value), (ushort)(src.z * value), (ushort)(src.w * value));
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 Multiply(ushort src, ushort4 value)
		{
			ushort4 ret = new ushort4((ushort)(src * value.x), (ushort)(src * value.y), (ushort)(src * value.z), (ushort)(src * value.w));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 Divide(ushort4 src, ushort4 value)
		{
			ushort4 ret = new ushort4((ushort)(src.x / value.x), (ushort)(src.y / value.y), (ushort)(src.z / value.z), (ushort)(src.w / value.w));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 Divide(ushort4 src, ushort value)
		{
			ushort4 ret = new ushort4((ushort)(src.x / value), (ushort)(src.y / value), (ushort)(src.z / value), (ushort)(src.w / value));
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 Divide(ushort src, ushort4 value)
		{
			ushort4 ret = new ushort4((ushort)(src / value.x), (ushort)(src / value.y), (ushort)(src / value.z), (ushort)(src / value.w));
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 operator +(ushort4 src, ushort4 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 operator +(ushort4 src, ushort value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 operator +(ushort src, ushort4 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 operator -(ushort4 src, ushort4 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 operator -(ushort4 src, ushort value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 operator -(ushort src, ushort4 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 operator *(ushort4 src, ushort4 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 operator *(ushort4 src, ushort value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 operator *(ushort src, ushort4 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 operator /(ushort4 src, ushort4 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 operator /(ushort4 src, ushort value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ushort4 operator /(ushort src, ushort4 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(ushort4 src, ushort4 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(ushort4 src, ushort4 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is ushort4)) return false;

			ushort4 value = (ushort4)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			ret &= this.w == value.w;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(ushort4 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			ret &= this.w == value.w;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode() ^ w.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1}; {2}; {3})", this.x, this.y, this.z, this.w);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		/// <param name="zValue"></param>
		/// <param name="wValue"></param>
		public ushort4(ushort xValue, ushort yValue, ushort zValue, ushort wValue)
		{
			this.x = xValue;
			this.y = yValue;
			this.z = zValue;
			this.w = wValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public ushort4(ushort val)
		{
			this.x = val;
			this.y = val;
			this.z = val;
			this.w = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static ushort4 Min(ushort4 aValue, ushort4 bValue)
		{
			return new ushort4(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y), Math.Min(aValue.z, bValue.z), Math.Min(aValue.w, bValue.w));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static ushort4 Max(ushort4 aValue, ushort4 bValue)
		{
			return new ushort4(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y), Math.Max(aValue.z, bValue.z), Math.Max(aValue.w, bValue.w));
		}

		/// <summary>
		/// Returns the Channel number from vector type.
		/// </summary>
		/// <returns>4</returns>
		public uint GetChannelNumber()
		{
			return 4;
		}

		/// <summary>
		/// Returns a matching CUArrayFormat.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat GetCUArrayFormat()
		{
			return CUArrayFormat.UnsignedInt16;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(ushort4);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(ushort4));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}
	#endregion

	#region int
	/// <summary>
	/// int1
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct int1 : ICudaVectorType, ICudaVectorTypeForArray
	{
		/// <summary>
		/// X
		/// </summary>
		public int x;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 Add(int1 src, int1 value)
		{
			int1 ret = new int1(src.x + value.x);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 Add(int1 src, int value)
		{
			int1 ret = new int1(src.x + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 Add(int src, int1 value)
		{
			int1 ret = new int1(src + value.x);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 Subtract(int1 src, int1 value)
		{
			int1 ret = new int1(src.x - value.x);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 Subtract(int1 src, int value)
		{
			int1 ret = new int1(src.x - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 Subtract(int src, int1 value)
		{
			int1 ret = new int1(src - value.x);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 Multiply(int1 src, int1 value)
		{
			int1 ret = new int1(src.x * value.x);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 Multiply(int1 src, int value)
		{
			int1 ret = new int1(src.x * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 Multiply(int src, int1 value)
		{
			int1 ret = new int1(src * value.x);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 Divide(int1 src, int1 value)
		{
			int1 ret = new int1(src.x / value.x);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 Divide(int1 src, int value)
		{
			int1 ret = new int1(src.x / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 Divide(int src, int1 value)
		{
			int1 ret = new int1(src / value.x);
			return ret;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static int ToInt32(int1 src)
		{
			return src.x;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static int1 FromInt32(int src)
		{
			return new int1(src);
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 operator +(int1 src, int1 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 operator +(int1 src, int value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 operator +(int src, int1 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 operator -(int1 src, int1 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 operator -(int1 src, int value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 operator -(int src, int1 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 operator *(int1 src, int1 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 operator *(int1 src, int value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 operator *(int src, int1 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 operator /(int1 src, int1 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 operator /(int1 src, int value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int1 operator /(int src, int1 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(int1 src, int1 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(int1 src, int1 value)
		{
			return !(src == value);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator int(int1 src)
		{
			return ToInt32(src);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator int1(int src)
		{
			return FromInt32(src);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is int1)) return false;

			int1 value = (int1)obj;

			bool ret = true;
			ret &= this.x == value.x;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(int1 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0})", this.x);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		public int1(int xValue)
		{
			this.x = xValue;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static int1 Min(int1 aValue, int1 bValue)
		{
			return new int1(Math.Min(aValue.x, bValue.x));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static int1 Max(int1 aValue, int1 bValue)
		{
			return new int1(Math.Max(aValue.x, bValue.x));
		}

		/// <summary>
		/// Returns the Channel number from vector type.
		/// </summary>
		/// <returns>1</returns>
		public uint GetChannelNumber()
		{
			return 1;
		}

		/// <summary>
		/// Returns a matching CUArrayFormat.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat GetCUArrayFormat()
		{
			return CUArrayFormat.SignedInt32;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(int1);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(int1));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// int2
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct int2 : ICudaVectorType, ICudaVectorTypeForArray
	{
		/// <summary>
		/// X
		/// </summary>
		public int x;
		/// <summary>
		/// Y
		/// </summary>
		public int y;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 Add(int2 src, int2 value)
		{
			int2 ret = new int2(src.x + value.x, src.y + value.y);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 Add(int2 src, int value)
		{
			int2 ret = new int2(src.x + value, src.y + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 Add(int src, int2 value)
		{
			int2 ret = new int2(src + value.x, src + value.y);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 Subtract(int2 src, int2 value)
		{
			int2 ret = new int2(src.x - value.x, src.y - value.y);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 Subtract(int2 src, int value)
		{
			int2 ret = new int2(src.x - value, src.y - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 Subtract(int src, int2 value)
		{
			int2 ret = new int2(src - value.x, src - value.y);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 Multiply(int2 src, int2 value)
		{
			int2 ret = new int2(src.x * value.x, src.y * value.y);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 Multiply(int2 src, int value)
		{
			int2 ret = new int2(src.x * value, src.y * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 Multiply(int src, int2 value)
		{
			int2 ret = new int2(src * value.x, src * value.y);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 Divide(int2 src, int2 value)
		{
			int2 ret = new int2(src.x / value.x, src.y / value.y);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 Divide(int2 src, int value)
		{
			int2 ret = new int2(src.x / value, src.y / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 Divide(int src, int2 value)
		{
			int2 ret = new int2(src / value.x, src / value.y);
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 operator +(int2 src, int2 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 operator +(int2 src, int value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 operator +(int src, int2 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 operator -(int2 src, int2 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 operator -(int2 src, int value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 operator -(int src, int2 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 operator *(int2 src, int2 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 operator *(int2 src, int value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 operator *(int src, int2 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 operator /(int2 src, int2 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 operator /(int2 src, int value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int2 operator /(int src, int2 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(int2 src, int2 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(int2 src, int2 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is int2)) return false;

			int2 value = (int2)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(int2 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1})", this.x, this.y);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		public int2(int xValue, int yValue)
		{
			this.x = xValue;
			this.y = yValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public int2(int val)
		{
			this.x = val;
			this.y = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static int2 Min(int2 aValue, int2 bValue)
		{
			return new int2(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static int2 Max(int2 aValue, int2 bValue)
		{
			return new int2(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y));
		}

		/// <summary>
		/// Returns the Channel number from vector type.
		/// </summary>
		/// <returns>2</returns>
		public uint GetChannelNumber()
		{
			return 2;
		}

		/// <summary>
		/// Returns a matching CUArrayFormat.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat GetCUArrayFormat()
		{
			return CUArrayFormat.SignedInt32;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(int2);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(int2));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// 
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct int3 : ICudaVectorType
	{
		/// <summary>
		/// X
		/// </summary>
		public int x;
		/// <summary>
		/// Y
		/// </summary>
		public int y;
		/// <summary>
		/// Z
		/// </summary>
		public int z;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 Add(int3 src, int3 value)
		{
			int3 ret = new int3(src.x + value.x, src.y + value.y, src.z + value.z);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 Add(int3 src, int value)
		{
			int3 ret = new int3(src.x + value, src.y + value, src.z + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 Add(int src, int3 value)
		{
			int3 ret = new int3(src + value.x, src + value.y, src + value.z);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 Subtract(int3 src, int3 value)
		{
			int3 ret = new int3(src.x - value.x, src.y - value.y, src.z - value.z);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 Subtract(int3 src, int value)
		{
			int3 ret = new int3(src.x - value, src.y - value, src.z - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 Subtract(int src, int3 value)
		{
			int3 ret = new int3(src - value.x, src - value.y, src - value.z);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 Multiply(int3 src, int3 value)
		{
			int3 ret = new int3(src.x * value.x, src.y * value.y, src.z * value.z);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 Multiply(int3 src, int value)
		{
			int3 ret = new int3(src.x * value, src.y * value, src.z * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 Multiply(int src, int3 value)
		{
			int3 ret = new int3(src * value.x, src * value.y, src * value.z);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 Divide(int3 src, int3 value)
		{
			int3 ret = new int3(src.x / value.x, src.y / value.y, src.z / value.z);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 Divide(int3 src, int value)
		{
			int3 ret = new int3(src.x / value, src.y / value, src.z / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 Divide(int src, int3 value)
		{
			int3 ret = new int3(src / value.x, src / value.y, src / value.z);
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 operator +(int3 src, int3 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 operator +(int3 src, int value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 operator +(int src, int3 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 operator -(int3 src, int3 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 operator -(int3 src, int value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 operator -(int src, int3 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 operator *(int3 src, int3 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 operator *(int3 src, int value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 operator *(int src, int3 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 operator /(int3 src, int3 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 operator /(int3 src, int value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int3 operator /(int src, int3 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(int3 src, int3 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(int3 src, int3 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is int3)) return false;

			int3 value = (int3)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(int3 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1}; {2})", this.x, this.y, this.z);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		/// <param name="zValue"></param>
		public int3(int xValue, int yValue, int zValue)
		{
			this.x = xValue;
			this.y = yValue;
			this.z = zValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public int3(int val)
		{
			this.x = val;
			this.y = val;
			this.z = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static int3 Min(int3 aValue, int3 bValue)
		{
			return new int3(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y), Math.Min(aValue.z, bValue.z));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static int3 Max(int3 aValue, int3 bValue)
		{
			return new int3(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y), Math.Max(aValue.z, bValue.z));
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(int3);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(int3));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// int4
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct int4 : ICudaVectorType, ICudaVectorTypeForArray
	{
		/// <summary>
		/// X
		/// </summary>
		public int x;
		/// <summary>
		/// Y
		/// </summary>
		public int y;
		/// <summary>
		/// Z
		/// </summary>
		public int z;
		/// <summary>
		/// W
		/// </summary>
		public int w;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 Add(int4 src, int4 value)
		{
			int4 ret = new int4(src.x + value.x, src.y + value.y, src.z + value.z, src.w + value.w);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 Add(int4 src, int value)
		{
			int4 ret = new int4(src.x + value, src.y + value, src.z + value, src.w + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 Add(int src, int4 value)
		{
			int4 ret = new int4(src + value.x, src + value.y, src + value.z, src + value.w);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 Subtract(int4 src, int4 value)
		{
			int4 ret = new int4(src.x - value.x, src.y - value.y, src.z - value.z, src.w - value.w);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 Subtract(int4 src, int value)
		{
			int4 ret = new int4(src.x - value, src.y - value, src.z - value, src.w - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 Subtract(int src, int4 value)
		{
			int4 ret = new int4(src - value.x, src - value.y, src - value.z, src - value.w);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 Multiply(int4 src, int4 value)
		{
			int4 ret = new int4(src.x * value.x, src.y * value.y, src.z * value.z, src.w * value.w);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 Multiply(int4 src, int value)
		{
			int4 ret = new int4(src.x * value, src.y * value, src.z * value, src.w * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 Multiply(int src, int4 value)
		{
			int4 ret = new int4(src * value.x, src * value.y, src * value.z, src * value.w);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 Divide(int4 src, int4 value)
		{
			int4 ret = new int4(src.x / value.x, src.y / value.y, src.z / value.z, src.w / value.w);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 Divide(int4 src, int value)
		{
			int4 ret = new int4(src.x / value, src.y / value, src.z / value, src.w / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 Divide(int src, int4 value)
		{
			int4 ret = new int4(src / value.x, src / value.y, src / value.z, src / value.w);
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 operator +(int4 src, int4 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 operator +(int4 src, int value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 operator +(int src, int4 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 operator -(int4 src, int4 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 operator -(int4 src, int value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 operator -(int src, int4 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 operator *(int4 src, int4 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 operator *(int4 src, int value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 operator *(int src, int4 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 operator /(int4 src, int4 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 operator /(int4 src, int value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static int4 operator /(int src, int4 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(int4 src, int4 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(int4 src, int4 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is int4)) return false;

			int4 value = (int4)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			ret &= this.w == value.w;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(int4 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			ret &= this.w == value.w;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode() ^ w.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1}; {2}; {3})", this.x, this.y, this.z, this.w);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		/// <param name="zValue"></param>
		/// <param name="wValue"></param>
		public int4(int xValue, int yValue, int zValue, int wValue)
		{
			this.x = xValue;
			this.y = yValue;
			this.z = zValue;
			this.w = wValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public int4(int val)
		{
			this.x = val;
			this.y = val;
			this.z = val;
			this.w = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static int4 Min(int4 aValue, int4 bValue)
		{
			return new int4(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y), Math.Min(aValue.z, bValue.z), Math.Min(aValue.w, bValue.w));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static int4 Max(int4 aValue, int4 bValue)
		{
			return new int4(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y), Math.Max(aValue.z, bValue.z), Math.Max(aValue.w, bValue.w));
		}

		/// <summary>
		/// Returns the Channel number from vector type.
		/// </summary>
		/// <returns>4</returns>
		public uint GetChannelNumber()
		{
			return 4;
		}

		/// <summary>
		/// Returns a matching CUArrayFormat.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat GetCUArrayFormat()
		{
			return CUArrayFormat.SignedInt32;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(int4);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(int4));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}
	#endregion

	#region uint
	/// <summary>
	/// uint1
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct uint1 : ICudaVectorType, ICudaVectorTypeForArray
	{
		/// <summary>
		/// X
		/// </summary>
		public uint x;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 Add(uint1 src, uint1 value)
		{
			uint1 ret = new uint1(src.x + value.x);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 Add(uint1 src, uint value)
		{
			uint1 ret = new uint1(src.x + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 Add(uint src, uint1 value)
		{
			uint1 ret = new uint1(src + value.x);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 Subtract(uint1 src, uint1 value)
		{
			uint1 ret = new uint1(src.x - value.x);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 Subtract(uint1 src, uint value)
		{
			uint1 ret = new uint1(src.x - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 Subtract(uint src, uint1 value)
		{
			uint1 ret = new uint1(src - value.x);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 Multiply(uint1 src, uint1 value)
		{
			uint1 ret = new uint1(src.x * value.x);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 Multiply(uint1 src, uint value)
		{
			uint1 ret = new uint1(src.x * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 Multiply(uint src, uint1 value)
		{
			uint1 ret = new uint1(src * value.x);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 Divide(uint1 src, uint1 value)
		{
			uint1 ret = new uint1(src.x / value.x);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 Divide(uint1 src, uint value)
		{
			uint1 ret = new uint1(src.x / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 Divide(uint src, uint1 value)
		{
			uint1 ret = new uint1(src / value.x);
			return ret;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static uint ToUInt32(uint1 src)
		{
			return src.x;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static uint1 FromUInt32(uint src)
		{
			return new uint1(src);
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 operator +(uint1 src, uint1 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 operator +(uint1 src, uint value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 operator +(uint src, uint1 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 operator -(uint1 src, uint1 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 operator -(uint1 src, uint value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 operator -(uint src, uint1 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 operator *(uint1 src, uint1 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 operator *(uint1 src, uint value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 operator *(uint src, uint1 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 operator /(uint1 src, uint1 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 operator /(uint1 src, uint value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint1 operator /(uint src, uint1 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(uint1 src, uint1 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(uint1 src, uint1 value)
		{
			return !(src == value);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator uint(uint1 src)
		{
			return ToUInt32(src);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator uint1(uint src)
		{
			return FromUInt32(src);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is uint1)) return false;

			uint1 value = (uint1)obj;

			bool ret = true;
			ret &= this.x == value.x;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(uint1 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0})", this.x);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		public uint1(uint xValue)
		{
			this.x = xValue;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static uint1 Min(uint1 aValue, uint1 bValue)
		{
			return new uint1(Math.Min(aValue.x, bValue.x));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static uint1 Max(uint1 aValue, uint1 bValue)
		{
			return new uint1(Math.Max(aValue.x, bValue.x));
		}

		/// <summary>
		/// Returns the Channel number from vector type.
		/// </summary>
		/// <returns>1</returns>
		public uint GetChannelNumber()
		{
			return 1;
		}

		/// <summary>
		/// Returns a matching CUArrayFormat.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat GetCUArrayFormat()
		{
			return CUArrayFormat.UnsignedInt32;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(uint1);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(uint1));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// uint2
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct uint2 : ICudaVectorType, ICudaVectorTypeForArray
	{
		/// <summary>
		/// X
		/// </summary>
		public uint x;
		/// <summary>
		/// Y
		/// </summary>
		public uint y;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 Add(uint2 src, uint2 value)
		{
			uint2 ret = new uint2(src.x + value.x, src.y + value.y);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 Add(uint2 src, uint value)
		{
			uint2 ret = new uint2(src.x + value, src.y + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 Add(uint src, uint2 value)
		{
			uint2 ret = new uint2(src + value.x, src + value.y);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 Subtract(uint2 src, uint2 value)
		{
			uint2 ret = new uint2(src.x - value.x, src.y - value.y);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 Subtract(uint2 src, uint value)
		{
			uint2 ret = new uint2(src.x - value, src.y - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 Subtract(uint src, uint2 value)
		{
			uint2 ret = new uint2(src - value.x, src - value.y);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 Multiply(uint2 src, uint2 value)
		{
			uint2 ret = new uint2(src.x * value.x, src.y * value.y);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 Multiply(uint2 src, uint value)
		{
			uint2 ret = new uint2(src.x * value, src.y * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 Multiply(uint src, uint2 value)
		{
			uint2 ret = new uint2(src * value.x, src * value.y);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 Divide(uint2 src, uint2 value)
		{
			uint2 ret = new uint2(src.x / value.x, src.y / value.y);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 Divide(uint2 src, uint value)
		{
			uint2 ret = new uint2(src.x / value, src.y / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 Divide(uint src, uint2 value)
		{
			uint2 ret = new uint2(src / value.x, src / value.y);
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 operator +(uint2 src, uint2 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 operator +(uint2 src, uint value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 operator +(uint src, uint2 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 operator -(uint2 src, uint2 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 operator -(uint2 src, uint value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 operator -(uint src, uint2 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 operator *(uint2 src, uint2 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 operator *(uint2 src, uint value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 operator *(uint src, uint2 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 operator /(uint2 src, uint2 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 operator /(uint2 src, uint value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint2 operator /(uint src, uint2 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(uint2 src, uint2 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(uint2 src, uint2 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is uint2)) return false;

			uint2 value = (uint2)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(uint2 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1})", this.x, this.y);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		public uint2(uint xValue, uint yValue)
		{
			this.x = xValue;
			this.y = yValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public uint2(uint val)
		{
			this.x = val;
			this.y = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static uint2 Min(uint2 aValue, uint2 bValue)
		{
			return new uint2(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static uint2 Max(uint2 aValue, uint2 bValue)
		{
			return new uint2(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y));
		}

		/// <summary>
		/// Returns the Channel number from vector type.
		/// </summary>
		/// <returns>2</returns>
		public uint GetChannelNumber()
		{
			return 2;
		}

		/// <summary>
		/// Returns a matching CUArrayFormat.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat GetCUArrayFormat()
		{
			return CUArrayFormat.UnsignedInt32;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(uint2);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(uint2));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// uint3
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct uint3 : ICudaVectorType
	{
		/// <summary>
		/// X
		/// </summary>
		public uint x;
		/// <summary>
		/// Y
		/// </summary>
		public uint y;
		/// <summary>
		/// Z
		/// </summary>
		public uint z;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 Add(uint3 src, uint3 value)
		{
			uint3 ret = new uint3(src.x + value.x, src.y + value.y, src.z + value.z);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 Add(uint3 src, uint value)
		{
			uint3 ret = new uint3(src.x + value, src.y + value, src.z + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 Add(uint src, uint3 value)
		{
			uint3 ret = new uint3(src + value.x, src + value.y, src + value.z);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 Subtract(uint3 src, uint3 value)
		{
			uint3 ret = new uint3(src.x - value.x, src.y - value.y, src.z - value.z);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 Subtract(uint3 src, uint value)
		{
			uint3 ret = new uint3(src.x - value, src.y - value, src.z - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 Subtract(uint src, uint3 value)
		{
			uint3 ret = new uint3(src - value.x, src - value.y, src - value.z);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 Multiply(uint3 src, uint3 value)
		{
			uint3 ret = new uint3(src.x * value.x, src.y * value.y, src.z * value.z);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 Multiply(uint3 src, uint value)
		{
			uint3 ret = new uint3(src.x * value, src.y * value, src.z * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 Multiply(uint src, uint3 value)
		{
			uint3 ret = new uint3(src * value.x, src * value.y, src * value.z);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 Divide(uint3 src, uint3 value)
		{
			uint3 ret = new uint3(src.x / value.x, src.y / value.y, src.z / value.z);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 Divide(uint3 src, uint value)
		{
			uint3 ret = new uint3(src.x / value, src.y / value, src.z / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 Divide(uint src, uint3 value)
		{
			uint3 ret = new uint3(src / value.x, src / value.y, src / value.z);
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 operator +(uint3 src, uint3 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 operator +(uint3 src, uint value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 operator +(uint src, uint3 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 operator -(uint3 src, uint3 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 operator -(uint3 src, uint value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 operator -(uint src, uint3 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 operator *(uint3 src, uint3 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 operator *(uint3 src, uint value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 operator *(uint src, uint3 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 operator /(uint3 src, uint3 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 operator /(uint3 src, uint value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint3 operator /(uint src, uint3 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(uint3 src, uint3 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(uint3 src, uint3 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is uint3)) return false;

			uint3 value = (uint3)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(uint3 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1}; {2})", this.x, this.y, this.z);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		/// <param name="zValue"></param>
		public uint3(uint xValue, uint yValue, uint zValue)
		{
			this.x = xValue;
			this.y = yValue;
			this.z = zValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public uint3(uint val)
		{
			this.x = val;
			this.y = val;
			this.z = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static uint3 Min(uint3 aValue, uint3 bValue)
		{
			return new uint3(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y), Math.Min(aValue.z, bValue.z));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static uint3 Max(uint3 aValue, uint3 bValue)
		{
			return new uint3(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y), Math.Max(aValue.z, bValue.z));
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(uint3);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(uint3));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// uint4
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct uint4 : ICudaVectorType, ICudaVectorTypeForArray
	{
		/// <summary>
		/// X
		/// </summary>
		public uint x;
		/// <summary>
		/// Y
		/// </summary>
		public uint y;
		/// <summary>
		/// Z
		/// </summary>
		public uint z;
		/// <summary>
		/// W
		/// </summary>
		public uint w;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 Add(uint4 src, uint4 value)
		{
			uint4 ret = new uint4(src.x + value.x, src.y + value.y, src.z + value.z, src.w + value.w);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 Add(uint4 src, uint value)
		{
			uint4 ret = new uint4(src.x + value, src.y + value, src.z + value, src.w + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 Add(uint src, uint4 value)
		{
			uint4 ret = new uint4(src + value.x, src + value.y, src + value.z, src + value.w);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 Subtract(uint4 src, uint4 value)
		{
			uint4 ret = new uint4(src.x - value.x, src.y - value.y, src.z - value.z, src.w - value.w);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 Subtract(uint4 src, uint value)
		{
			uint4 ret = new uint4(src.x - value, src.y - value, src.z - value, src.w - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 Subtract(uint src, uint4 value)
		{
			uint4 ret = new uint4(src - value.x, src - value.y, src - value.z, src - value.w);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 Multiply(uint4 src, uint4 value)
		{
			uint4 ret = new uint4(src.x * value.x, src.y * value.y, src.z * value.z, src.w * value.w);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 Multiply(uint4 src, uint value)
		{
			uint4 ret = new uint4(src.x * value, src.y * value, src.z * value, src.w * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 Multiply(uint src, uint4 value)
		{
			uint4 ret = new uint4(src * value.x, src * value.y, src * value.z, src * value.w);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 Divide(uint4 src, uint4 value)
		{
			uint4 ret = new uint4(src.x / value.x, src.y / value.y, src.z / value.z, src.w / value.w);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 Divide(uint4 src, uint value)
		{
			uint4 ret = new uint4(src.x / value, src.y / value, src.z / value, src.w / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 Divide(uint src, uint4 value)
		{
			uint4 ret = new uint4(src / value.x, src / value.y, src / value.z, src / value.w);
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 operator +(uint4 src, uint4 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 operator +(uint4 src, uint value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 operator +(uint src, uint4 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 operator -(uint4 src, uint4 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 operator -(uint4 src, uint value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 operator -(uint src, uint4 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 operator *(uint4 src, uint4 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 operator *(uint4 src, uint value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 operator *(uint src, uint4 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 operator /(uint4 src, uint4 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 operator /(uint4 src, uint value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static uint4 operator /(uint src, uint4 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(uint4 src, uint4 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(uint4 src, uint4 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is uint4)) return false;

			uint4 value = (uint4)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			ret &= this.w == value.w;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(uint4 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			ret &= this.w == value.w;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode() ^ w.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1}; {2}; {3})", this.x, this.y, this.z, this.w);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		/// <param name="zValue"></param>
		/// <param name="wValue"></param>
		public uint4(uint xValue, uint yValue, uint zValue, uint wValue)
		{
			this.x = xValue;
			this.y = yValue;
			this.z = zValue;
			this.w = wValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public uint4(uint val)
		{
			this.x = val;
			this.y = val;
			this.z = val;
			this.w = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static uint4 Min(uint4 aValue, uint4 bValue)
		{
			return new uint4(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y), Math.Min(aValue.z, bValue.z), Math.Min(aValue.w, bValue.w));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static uint4 Max(uint4 aValue, uint4 bValue)
		{
			return new uint4(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y), Math.Max(aValue.z, bValue.z), Math.Max(aValue.w, bValue.w));
		}

		/// <summary>
		/// Returns the Channel number from vector type.
		/// </summary>
		/// <returns>4</returns>
		public uint GetChannelNumber()
		{
			return 4;
		}

		/// <summary>
		/// Returns a matching CUArrayFormat.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat GetCUArrayFormat()
		{
			return CUArrayFormat.UnsignedInt32;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(uint4);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(uint4));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}
	#endregion

	#region long
	/// <summary>
	/// long1. long stands here for the long .NET type, i.e. long long or a 64bit long in C++/CUDA
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct long1 : ICudaVectorType
	{
		/// <summary>
		/// X
		/// </summary>
		public long x;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 Add(long1 src, long1 value)
		{
			long1 ret = new long1(src.x + value.x);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 Add(long1 src, long value)
		{
			long1 ret = new long1(src.x + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 Add(long src, long1 value)
		{
			long1 ret = new long1(src + value.x);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 Subtract(long1 src, long1 value)
		{
			long1 ret = new long1(src.x - value.x);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 Subtract(long1 src, long value)
		{
			long1 ret = new long1(src.x - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 Subtract(long src, long1 value)
		{
			long1 ret = new long1(src - value.x);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 Multiply(long1 src, long1 value)
		{
			long1 ret = new long1(src.x * value.x);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 Multiply(long1 src, long value)
		{
			long1 ret = new long1(src.x * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 Multiply(long src, long1 value)
		{
			long1 ret = new long1(src * value.x);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 Divide(long1 src, long1 value)
		{
			long1 ret = new long1(src.x / value.x);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 Divide(long1 src, long value)
		{
			long1 ret = new long1(src.x / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 Divide(long src, long1 value)
		{
			long1 ret = new long1(src / value.x);
			return ret;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static long ToInt64(long1 src)
		{
			return src.x;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static long1 FromInt64(long src)
		{
			return new long1(src);
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 operator +(long1 src, long1 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 operator +(long1 src, long value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 operator +(long src, long1 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 operator -(long1 src, long1 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 operator -(long1 src, long value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 operator -(long src, long1 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 operator *(long1 src, long1 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 operator *(long1 src, long value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 operator *(long src, long1 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 operator /(long1 src, long1 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 operator /(long1 src, long value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long1 operator /(long src, long1 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(long1 src, long1 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(long1 src, long1 value)
		{
			return !(src == value);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator long(long1 src)
		{
			return ToInt64(src);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator long1(long src)
		{
			return FromInt64(src);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is long1)) return false;

			long1 value = (long1)obj;

			bool ret = true;
			ret &= this.x == value.x;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(long1 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0})", this.x);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		public long1(long xValue)
		{
			this.x = xValue;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static long1 Min(long1 aValue, long1 bValue)
		{
			return new long1(Math.Min(aValue.x, bValue.x));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static long1 Max(long1 aValue, long1 bValue)
		{
			return new long1(Math.Max(aValue.x, bValue.x));
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(long1);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(long1));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// long2. long stands here for the long .NET type, i.e. long long or a 64bit long in C++/CUDA
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct long2 : ICudaVectorType
	{
		/// <summary>
		/// X
		/// </summary>
		public long x;
		/// <summary>
		/// Y
		/// </summary>
		public long y;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 Add(long2 src, long2 value)
		{
			long2 ret = new long2(src.x + value.x, src.y + value.y);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 Add(long2 src, long value)
		{
			long2 ret = new long2(src.x + value, src.y + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 Add(long src, long2 value)
		{
			long2 ret = new long2(src + value.x, src + value.y);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 Subtract(long2 src, long2 value)
		{
			long2 ret = new long2(src.x - value.x, src.y - value.y);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 Subtract(long2 src, long value)
		{
			long2 ret = new long2(src.x - value, src.y - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 Subtract(long src, long2 value)
		{
			long2 ret = new long2(src - value.x, src - value.y);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 Multiply(long2 src, long2 value)
		{
			long2 ret = new long2(src.x * value.x, src.y * value.y);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 Multiply(long2 src, long value)
		{
			long2 ret = new long2(src.x * value, src.y * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 Multiply(long src, long2 value)
		{
			long2 ret = new long2(src * value.x, src * value.y);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 Divide(long2 src, long2 value)
		{
			long2 ret = new long2(src.x / value.x, src.y / value.y);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 Divide(long2 src, long value)
		{
			long2 ret = new long2(src.x / value, src.y / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 Divide(long src, long2 value)
		{
			long2 ret = new long2(src / value.x, src / value.y);
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 operator +(long2 src, long2 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 operator +(long2 src, long value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 operator +(long src, long2 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 operator -(long2 src, long2 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 operator -(long2 src, long value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 operator -(long src, long2 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 operator *(long2 src, long2 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 operator *(long2 src, long value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 operator *(long src, long2 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 operator /(long2 src, long2 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 operator /(long2 src, long value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long2 operator /(long src, long2 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(long2 src, long2 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(long2 src, long2 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is long2)) return false;

			long2 value = (long2)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(long2 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1})", this.x, this.y);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		public long2(long xValue, long yValue)
		{
			this.x = xValue;
			this.y = yValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public long2(long val)
		{
			this.x = val;
			this.y = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static long2 Min(long2 aValue, long2 bValue)
		{
			return new long2(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static long2 Max(long2 aValue, long2 bValue)
		{
			return new long2(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y));
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(long2);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(long2));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// long3. long stands here for the long .NET type, i.e. long long or a 64bit long in C++/CUDA
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct long3 : ICudaVectorType
	{
		/// <summary>
		/// X
		/// </summary>
		public long x;
		/// <summary>
		/// Y
		/// </summary>
		public long y;
		/// <summary>
		/// Z
		/// </summary>
		public long z;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 Add(long3 src, long3 value)
		{
			long3 ret = new long3(src.x + value.x, src.y + value.y, src.z + value.z);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 Add(long3 src, long value)
		{
			long3 ret = new long3(src.x + value, src.y + value, src.z + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 Add(long src, long3 value)
		{
			long3 ret = new long3(src + value.x, src + value.y, src + value.z);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 Subtract(long3 src, long3 value)
		{
			long3 ret = new long3(src.x - value.x, src.y - value.y, src.z - value.z);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 Subtract(long3 src, long value)
		{
			long3 ret = new long3(src.x - value, src.y - value, src.z - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 Subtract(long src, long3 value)
		{
			long3 ret = new long3(src - value.x, src - value.y, src - value.z);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 Multiply(long3 src, long3 value)
		{
			long3 ret = new long3(src.x * value.x, src.y * value.y, src.z * value.z);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 Multiply(long3 src, long value)
		{
			long3 ret = new long3(src.x * value, src.y * value, src.z * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 Multiply(long src, long3 value)
		{
			long3 ret = new long3(src * value.x, src * value.y, src * value.z);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 Divide(long3 src, long3 value)
		{
			long3 ret = new long3(src.x / value.x, src.y / value.y, src.z / value.z);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 Divide(long3 src, long value)
		{
			long3 ret = new long3(src.x / value, src.y / value, src.z / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 Divide(long src, long3 value)
		{
			long3 ret = new long3(src / value.x, src / value.y, src / value.z);
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 operator +(long3 src, long3 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 operator +(long3 src, long value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 operator +(long src, long3 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 operator -(long3 src, long3 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 operator -(long3 src, long value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 operator -(long src, long3 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 operator *(long3 src, long3 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 operator *(long3 src, long value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 operator *(long src, long3 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 operator /(long3 src, long3 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 operator /(long3 src, long value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long3 operator /(long src, long3 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(long3 src, long3 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(long3 src, long3 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is long3)) return false;

			long3 value = (long3)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(long3 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1}; {2})", this.x, this.y, this.z);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		/// <param name="zValue"></param>
		public long3(long xValue, long yValue, long zValue)
		{
			this.x = xValue;
			this.y = yValue;
			this.z = zValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public long3(long val)
		{
			this.x = val;
			this.y = val;
			this.z = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static long3 Min(long3 aValue, long3 bValue)
		{
			return new long3(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y), Math.Min(aValue.z, bValue.z));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static long3 Max(long3 aValue, long3 bValue)
		{
			return new long3(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y), Math.Max(aValue.z, bValue.z));
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(long3);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(long3));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// long4. long stands here for the long .NET type, i.e. long long or a 64bit long in C++/CUDA
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct long4 : ICudaVectorType
	{
		/// <summary>
		/// X
		/// </summary>
		public long x;
		/// <summary>
		/// Y
		/// </summary>
		public long y;
		/// <summary>
		/// Z
		/// </summary>
		public long z;
		/// <summary>
		/// W
		/// </summary>
		public long w;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 Add(long4 src, long4 value)
		{
			long4 ret = new long4(src.x + value.x, src.y + value.y, src.z + value.z, src.w + value.w);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 Add(long4 src, long value)
		{
			long4 ret = new long4(src.x + value, src.y + value, src.z + value, src.w + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 Add(long src, long4 value)
		{
			long4 ret = new long4(src + value.x, src + value.y, src + value.z, src + value.w);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 Subtract(long4 src, long4 value)
		{
			long4 ret = new long4(src.x - value.x, src.y - value.y, src.z - value.z, src.w - value.w);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 Subtract(long4 src, long value)
		{
			long4 ret = new long4(src.x - value, src.y - value, src.z - value, src.w - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 Subtract(long src, long4 value)
		{
			long4 ret = new long4(src - value.x, src - value.y, src - value.z, src - value.w);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 Multiply(long4 src, long4 value)
		{
			long4 ret = new long4(src.x * value.x, src.y * value.y, src.z * value.z, src.w * value.w);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 Multiply(long4 src, long value)
		{
			long4 ret = new long4(src.x * value, src.y * value, src.z * value, src.w * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 Multiply(long src, long4 value)
		{
			long4 ret = new long4(src * value.x, src * value.y, src * value.z, src * value.w);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 Divide(long4 src, long4 value)
		{
			long4 ret = new long4(src.x / value.x, src.y / value.y, src.z / value.z, src.w / value.w);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 Divide(long4 src, long value)
		{
			long4 ret = new long4(src.x / value, src.y / value, src.z / value, src.w / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 Divide(long src, long4 value)
		{
			long4 ret = new long4(src / value.x, src / value.y, src / value.z, src / value.w);
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 operator +(long4 src, long4 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 operator +(long4 src, long value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 operator +(long src, long4 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 operator -(long4 src, long4 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 operator -(long4 src, long value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 operator -(long src, long4 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 operator *(long4 src, long4 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 operator *(long4 src, long value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 operator *(long src, long4 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 operator /(long4 src, long4 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 operator /(long4 src, long value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static long4 operator /(long src, long4 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(long4 src, long4 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(long4 src, long4 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is long4)) return false;

			long4 value = (long4)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			ret &= this.w == value.w;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(long4 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			ret &= this.w == value.w;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode() ^ w.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1}; {2}; {3})", this.x, this.y, this.z, this.w);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		/// <param name="zValue"></param>
		/// <param name="wValue"></param>
		public long4(long xValue, long yValue, long zValue, long wValue)
		{
			this.x = xValue;
			this.y = yValue;
			this.z = zValue;
			this.w = wValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public long4(long val)
		{
			this.x = val;
			this.y = val;
			this.z = val;
			this.w = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static long4 Min(long4 aValue, long4 bValue)
		{
			return new long4(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y), Math.Min(aValue.z, bValue.z), Math.Min(aValue.w, bValue.w));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static long4 Max(long4 aValue, long4 bValue)
		{
			return new long4(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y), Math.Max(aValue.z, bValue.z), Math.Max(aValue.w, bValue.w));
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(long4);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(long4));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}
	#endregion

	#region ulong
	/// <summary>
	/// ulong1. ulong stands here for the ulong .NET type, i.e. unsigned long long or a 64bit unsigned long in C++/CUDA
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct ulong1 : ICudaVectorType
	{
		/// <summary>
		/// X
		/// </summary>
		public ulong x;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 Add(ulong1 src, ulong1 value)
		{
			ulong1 ret = new ulong1(src.x + value.x);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 Add(ulong1 src, ulong value)
		{
			ulong1 ret = new ulong1(src.x + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 Add(ulong src, ulong1 value)
		{
			ulong1 ret = new ulong1(src + value.x);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 Subtract(ulong1 src, ulong1 value)
		{
			ulong1 ret = new ulong1(src.x - value.x);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 Subtract(ulong1 src, ulong value)
		{
			ulong1 ret = new ulong1(src.x - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 Subtract(ulong src, ulong1 value)
		{
			ulong1 ret = new ulong1(src - value.x);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 Multiply(ulong1 src, ulong1 value)
		{
			ulong1 ret = new ulong1(src.x * value.x);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 Multiply(ulong1 src, ulong value)
		{
			ulong1 ret = new ulong1(src.x * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 Multiply(ulong src, ulong1 value)
		{
			ulong1 ret = new ulong1(src * value.x);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 Divide(ulong1 src, ulong1 value)
		{
			ulong1 ret = new ulong1(src.x / value.x);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 Divide(ulong1 src, ulong value)
		{
			ulong1 ret = new ulong1(src.x / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 Divide(ulong src, ulong1 value)
		{
			ulong1 ret = new ulong1(src / value.x);
			return ret;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static ulong ToUInt64(ulong1 src)
		{
			return src.x;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static ulong1 FromUInt64(ulong src)
		{
			return new ulong1(src);
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 operator +(ulong1 src, ulong1 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 operator +(ulong1 src, ulong value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 operator +(ulong src, ulong1 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 operator -(ulong1 src, ulong1 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 operator -(ulong1 src, ulong value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 operator -(ulong src, ulong1 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 operator *(ulong1 src, ulong1 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 operator *(ulong1 src, ulong value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 operator *(ulong src, ulong1 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 operator /(ulong1 src, ulong1 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 operator /(ulong1 src, ulong value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong1 operator /(ulong src, ulong1 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(ulong1 src, ulong1 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(ulong1 src, ulong1 value)
		{
			return !(src == value);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator ulong(ulong1 src)
		{
			return ToUInt64(src);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator ulong1(ulong src)
		{
			return FromUInt64(src);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is ulong1)) return false;

			ulong1 value = (ulong1)obj;

			bool ret = true;
			ret &= this.x == value.x;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(ulong1 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0})", this.x);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		public ulong1(ulong xValue)
		{
			this.x = xValue;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static ulong1 Min(ulong1 aValue, ulong1 bValue)
		{
			return new ulong1(Math.Min(aValue.x, bValue.x));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static ulong1 Max(ulong1 aValue, ulong1 bValue)
		{
			return new ulong1(Math.Max(aValue.x, bValue.x));
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(ulong1);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(ulong1));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// ulong2. ulong stands here for the ulong .NET type, i.e. unsigned long long or a 64bit unsigned long in C++/CUDA
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct ulong2 : ICudaVectorType
	{
		/// <summary>
		/// X
		/// </summary>
		public ulong x;
		/// <summary>
		/// Y
		/// </summary>
		public ulong y;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 Add(ulong2 src, ulong2 value)
		{
			ulong2 ret = new ulong2(src.x + value.x, src.y + value.y);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 Add(ulong2 src, ulong value)
		{
			ulong2 ret = new ulong2(src.x + value, src.y + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 Add(ulong src, ulong2 value)
		{
			ulong2 ret = new ulong2(src + value.x, src + value.y);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 Subtract(ulong2 src, ulong2 value)
		{
			ulong2 ret = new ulong2(src.x - value.x, src.y - value.y);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 Subtract(ulong2 src, ulong value)
		{
			ulong2 ret = new ulong2(src.x - value, src.y - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 Subtract(ulong src, ulong2 value)
		{
			ulong2 ret = new ulong2(src - value.x, src - value.y);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 Multiply(ulong2 src, ulong2 value)
		{
			ulong2 ret = new ulong2(src.x * value.x, src.y * value.y);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 Multiply(ulong2 src, ulong value)
		{
			ulong2 ret = new ulong2(src.x * value, src.y * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 Multiply(ulong src, ulong2 value)
		{
			ulong2 ret = new ulong2(src * value.x, src * value.y);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 Divide(ulong2 src, ulong2 value)
		{
			ulong2 ret = new ulong2(src.x / value.x, src.y / value.y);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 Divide(ulong2 src, ulong value)
		{
			ulong2 ret = new ulong2(src.x / value, src.y / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 Divide(ulong src, ulong2 value)
		{
			ulong2 ret = new ulong2(src / value.x, src / value.y);
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 operator +(ulong2 src, ulong2 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 operator +(ulong2 src, ulong value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 operator +(ulong src, ulong2 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 operator -(ulong2 src, ulong2 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 operator -(ulong2 src, ulong value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 operator -(ulong src, ulong2 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 operator *(ulong2 src, ulong2 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 operator *(ulong2 src, ulong value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 operator *(ulong src, ulong2 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 operator /(ulong2 src, ulong2 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 operator /(ulong2 src, ulong value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong2 operator /(ulong src, ulong2 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(ulong2 src, ulong2 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(ulong2 src, ulong2 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is ulong2)) return false;

			ulong2 value = (ulong2)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(ulong2 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1})", this.x, this.y);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		public ulong2(ulong xValue, ulong yValue)
		{
			this.x = xValue;
			this.y = yValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public ulong2(ulong val)
		{
			this.x = val;
			this.y = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static ulong2 Min(ulong2 aValue, ulong2 bValue)
		{
			return new ulong2(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static ulong2 Max(ulong2 aValue, ulong2 bValue)
		{
			return new ulong2(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y));
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(ulong2);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(ulong2));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// ulong3. ulong stands here for the ulong .NET type, i.e. unsigned long long or a 64bit unsigned long in C++/CUDA
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct ulong3 : ICudaVectorType
	{
		/// <summary>
		/// X
		/// </summary>
		public ulong x;
		/// <summary>
		/// Y
		/// </summary>
		public ulong y;
		/// <summary>
		/// Z
		/// </summary>
		public ulong z;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 Add(ulong3 src, ulong3 value)
		{
			ulong3 ret = new ulong3(src.x + value.x, src.y + value.y, src.z + value.z);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 Add(ulong3 src, ulong value)
		{
			ulong3 ret = new ulong3(src.x + value, src.y + value, src.z + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 Add(ulong src, ulong3 value)
		{
			ulong3 ret = new ulong3(src + value.x, src + value.y, src + value.z);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 Subtract(ulong3 src, ulong3 value)
		{
			ulong3 ret = new ulong3(src.x - value.x, src.y - value.y, src.z - value.z);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 Subtract(ulong3 src, ulong value)
		{
			ulong3 ret = new ulong3(src.x - value, src.y - value, src.z - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 Subtract(ulong src, ulong3 value)
		{
			ulong3 ret = new ulong3(src - value.x, src - value.y, src - value.z);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 Multiply(ulong3 src, ulong3 value)
		{
			ulong3 ret = new ulong3(src.x * value.x, src.y * value.y, src.z * value.z);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 Multiply(ulong3 src, ulong value)
		{
			ulong3 ret = new ulong3(src.x * value, src.y * value, src.z * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 Multiply(ulong src, ulong3 value)
		{
			ulong3 ret = new ulong3(src * value.x, src * value.y, src * value.z);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 Divide(ulong3 src, ulong3 value)
		{
			ulong3 ret = new ulong3(src.x / value.x, src.y / value.y, src.z / value.z);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 Divide(ulong3 src, ulong value)
		{
			ulong3 ret = new ulong3(src.x / value, src.y / value, src.z / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 Divide(ulong src, ulong3 value)
		{
			ulong3 ret = new ulong3(src / value.x, src / value.y, src / value.z);
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 operator +(ulong3 src, ulong3 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 operator +(ulong3 src, ulong value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 operator +(ulong src, ulong3 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 operator -(ulong3 src, ulong3 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 operator -(ulong3 src, ulong value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 operator -(ulong src, ulong3 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 operator *(ulong3 src, ulong3 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 operator *(ulong3 src, ulong value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 operator *(ulong src, ulong3 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 operator /(ulong3 src, ulong3 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 operator /(ulong3 src, ulong value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong3 operator /(ulong src, ulong3 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(ulong3 src, ulong3 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(ulong3 src, ulong3 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is ulong3)) return false;

			ulong3 value = (ulong3)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(ulong3 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1}; {2})", this.x, this.y, this.z);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		/// <param name="zValue"></param>
		public ulong3(ulong xValue, ulong yValue, ulong zValue)
		{
			this.x = xValue;
			this.y = yValue;
			this.z = zValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public ulong3(ulong val)
		{
			this.x = val;
			this.y = val;
			this.z = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static ulong3 Min(ulong3 aValue, ulong3 bValue)
		{
			return new ulong3(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y), Math.Min(aValue.z, bValue.z));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static ulong3 Max(ulong3 aValue, ulong3 bValue)
		{
			return new ulong3(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y), Math.Max(aValue.z, bValue.z));
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(ulong3);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(ulong3));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// ulong4. ulong stands here for the ulong .NET type, i.e. unsigned long long or a 64bit unsigned long in C++/CUDA
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct ulong4 : ICudaVectorType
	{
		/// <summary>
		/// X
		/// </summary>
		public ulong x;
		/// <summary>
		/// Y
		/// </summary>
		public ulong y;
		/// <summary>
		/// Z
		/// </summary>
		public ulong z;
		/// <summary>
		/// W
		/// </summary>
		public ulong w;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 Add(ulong4 src, ulong4 value)
		{
			ulong4 ret = new ulong4(src.x + value.x, src.y + value.y, src.z + value.z, src.w + value.w);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 Add(ulong4 src, ulong value)
		{
			ulong4 ret = new ulong4(src.x + value, src.y + value, src.z + value, src.w + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 Add(ulong src, ulong4 value)
		{
			ulong4 ret = new ulong4(src + value.x, src + value.y, src + value.z, src + value.w);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 Subtract(ulong4 src, ulong4 value)
		{
			ulong4 ret = new ulong4(src.x - value.x, src.y - value.y, src.z - value.z, src.w - value.w);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 Subtract(ulong4 src, ulong value)
		{
			ulong4 ret = new ulong4(src.x - value, src.y - value, src.z - value, src.w - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 Subtract(ulong src, ulong4 value)
		{
			ulong4 ret = new ulong4(src - value.x, src - value.y, src - value.z, src - value.w);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 Multiply(ulong4 src, ulong4 value)
		{
			ulong4 ret = new ulong4(src.x * value.x, src.y * value.y, src.z * value.z, src.w * value.w);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 Multiply(ulong4 src, ulong value)
		{
			ulong4 ret = new ulong4(src.x * value, src.y * value, src.z * value, src.w * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 Multiply(ulong src, ulong4 value)
		{
			ulong4 ret = new ulong4(src * value.x, src * value.y, src * value.z, src * value.w);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 Divide(ulong4 src, ulong4 value)
		{
			ulong4 ret = new ulong4(src.x / value.x, src.y / value.y, src.z / value.z, src.w / value.w);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 Divide(ulong4 src, ulong value)
		{
			ulong4 ret = new ulong4(src.x / value, src.y / value, src.z / value, src.w / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 Divide(ulong src, ulong4 value)
		{
			ulong4 ret = new ulong4(src / value.x, src / value.y, src / value.z, src / value.w);
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 operator +(ulong4 src, ulong4 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 operator +(ulong4 src, ulong value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 operator +(ulong src, ulong4 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 operator -(ulong4 src, ulong4 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 operator -(ulong4 src, ulong value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 operator -(ulong src, ulong4 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 operator *(ulong4 src, ulong4 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 operator *(ulong4 src, ulong value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 operator *(ulong src, ulong4 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 operator /(ulong4 src, ulong4 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 operator /(ulong4 src, ulong value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static ulong4 operator /(ulong src, ulong4 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(ulong4 src, ulong4 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(ulong4 src, ulong4 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is ulong4)) return false;

			ulong4 value = (ulong4)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			ret &= this.w == value.w;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(ulong4 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			ret &= this.w == value.w;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode() ^ w.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1}; {2}; {3})", this.x, this.y, this.z, this.w);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		/// <param name="zValue"></param>
		/// <param name="wValue"></param>
		public ulong4(ulong xValue, ulong yValue, ulong zValue, ulong wValue)
		{
			this.x = xValue;
			this.y = yValue;
			this.z = zValue;
			this.w = wValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public ulong4(ulong val)
		{
			this.x = val;
			this.y = val;
			this.z = val;
			this.w = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function imini
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static ulong4 Min(ulong4 aValue, ulong4 bValue)
		{
			return new ulong4(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y), Math.Min(aValue.z, bValue.z), Math.Min(aValue.w, bValue.w));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function imaxi
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static ulong4 Max(ulong4 aValue, ulong4 bValue)
		{
			return new ulong4(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y), Math.Max(aValue.z, bValue.z), Math.Max(aValue.w, bValue.w));
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(ulong4);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(ulong4));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}
	#endregion

	#region float
	/// <summary>
	/// float1
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct float1 : ICudaVectorType, ICudaVectorTypeForArray
	{
		/// <summary>
		/// X
		/// </summary>
		public float x;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 Add(float1 src, float1 value)
		{
			float1 ret = new float1(src.x + value.x);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 Add(float1 src, float value)
		{
			float1 ret = new float1(src.x + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 Add(float src, float1 value)
		{
			float1 ret = new float1(src + value.x);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 Subtract(float1 src, float1 value)
		{
			float1 ret = new float1(src.x - value.x);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 Subtract(float1 src, float value)
		{
			float1 ret = new float1(src.x - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 Subtract(float src, float1 value)
		{
			float1 ret = new float1(src - value.x);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 Multiply(float1 src, float1 value)
		{
			float1 ret = new float1(src.x * value.x);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 Multiply(float1 src, float value)
		{
			float1 ret = new float1(src.x * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 Multiply(float src, float1 value)
		{
			float1 ret = new float1(src * value.x);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 Divide(float1 src, float1 value)
		{
			float1 ret = new float1(src.x / value.x);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 Divide(float1 src, float value)
		{
			float1 ret = new float1(src.x / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 Divide(float src, float1 value)
		{
			float1 ret = new float1(src / value.x);
			return ret;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static float ToSingle(float1 src)
		{
			return src.x;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static float1 FromSingle(float src)
		{
			return new float1(src);
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 operator +(float1 src, float1 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 operator +(float1 src, float value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 operator +(float src, float1 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 operator -(float1 src, float1 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 operator -(float1 src, float value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 operator -(float src, float1 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 operator *(float1 src, float1 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 operator *(float1 src, float value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 operator *(float src, float1 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 operator /(float1 src, float1 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 operator /(float1 src, float value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float1 operator /(float src, float1 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(float1 src, float1 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(float1 src, float1 value)
		{
			return !(src == value);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator float(float1 src)
		{
			return ToSingle(src);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator float1(float src)
		{
			return FromSingle(src);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is float1)) return false;

			float1 value = (float1)obj;

			bool ret = true;
			ret &= this.x == value.x;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(float1 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0})", this.x);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		public float1(float xValue)
		{
			this.x = xValue;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function fminf
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static float1 Min(float1 aValue, float1 bValue)
		{
			return new float1(Math.Min(aValue.x, bValue.x));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function fmaxf
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static float1 Max(float1 aValue, float1 bValue)
		{
			return new float1(Math.Max(aValue.x, bValue.x));
		}

		/// <summary>
		/// Returns the Channel number from vector type.
		/// </summary>
		/// <returns>1</returns>
		public uint GetChannelNumber()
		{
			return 1;
		}

		/// <summary>
		/// Returns a matching CUArrayFormat.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat GetCUArrayFormat()
		{
			return CUArrayFormat.Float;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(float1);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(float1));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// float2
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct float2 : ICudaVectorType, ICudaVectorTypeForArray
	{
		/// <summary>
		/// X
		/// </summary>
		public float x;
		/// <summary>
		/// Y
		/// </summary>
		public float y;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 Add(float2 src, float2 value)
		{
			float2 ret = new float2(src.x + value.x, src.y + value.y);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 Add(float2 src, float value)
		{
			float2 ret = new float2(src.x + value, src.y + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 Add(float src, float2 value)
		{
			float2 ret = new float2(src + value.x, src + value.y);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 Subtract(float2 src, float2 value)
		{
			float2 ret = new float2(src.x - value.x, src.y - value.y);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 Subtract(float2 src, float value)
		{
			float2 ret = new float2(src.x - value, src.y - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 Subtract(float src, float2 value)
		{
			float2 ret = new float2(src - value.x, src - value.y);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 Multiply(float2 src, float2 value)
		{
			float2 ret = new float2(src.x * value.x, src.y * value.y);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 Multiply(float2 src, float value)
		{
			float2 ret = new float2(src.x * value, src.y * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 Multiply(float src, float2 value)
		{
			float2 ret = new float2(src * value.x, src * value.y);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 Divide(float2 src, float2 value)
		{
			float2 ret = new float2(src.x / value.x, src.y / value.y);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 Divide(float2 src, float value)
		{
			float2 ret = new float2(src.x / value, src.y / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 Divide(float src, float2 value)
		{
			float2 ret = new float2(src / value.x, src / value.y);
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 operator +(float2 src, float2 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 operator +(float2 src, float value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 operator +(float src, float2 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 operator -(float2 src, float2 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 operator -(float2 src, float value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 operator -(float src, float2 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 operator *(float2 src, float2 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 operator *(float2 src, float value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 operator *(float src, float2 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 operator /(float2 src, float2 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 operator /(float2 src, float value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float2 operator /(float src, float2 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(float2 src, float2 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(float2 src, float2 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is float2)) return false;

			float2 value = (float2)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(float2 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1})", this.x, this.y);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		public float2(float xValue, float yValue)
		{
			this.x = xValue;
			this.y = yValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public float2(float val)
		{
			this.x = val;
			this.y = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function fminf
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static float2 Min(float2 aValue, float2 bValue)
		{
			return new float2(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function fmaxf
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static float2 Max(float2 aValue, float2 bValue)
		{
			return new float2(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y));
		}

		/// <summary>
		/// Returns the Channel number from vector type.
		/// </summary>
		/// <returns>2</returns>
		public uint GetChannelNumber()
		{
			return 2;
		}

		/// <summary>
		/// Returns a matching CUArrayFormat.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat GetCUArrayFormat()
		{
			return CUArrayFormat.Float;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(float2);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(float2));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// float3
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct float3 : ICudaVectorType
	{
		/// <summary>
		/// X
		/// </summary>
		public float x;
		/// <summary>
		/// Y
		/// </summary>
		public float y;
		/// <summary>
		/// Z
		/// </summary>
		public float z;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 Add(float3 src, float3 value)
		{
			float3 ret = new float3(src.x + value.x, src.y + value.y, src.z + value.z);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 Add(float3 src, float value)
		{
			float3 ret = new float3(src.x + value, src.y + value, src.z + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 Add(float src, float3 value)
		{
			float3 ret = new float3(src + value.x, src + value.y, src + value.z);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 Subtract(float3 src, float3 value)
		{
			float3 ret = new float3(src.x - value.x, src.y - value.y, src.z - value.z);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 Subtract(float3 src, float value)
		{
			float3 ret = new float3(src.x - value, src.y - value, src.z - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 Subtract(float src, float3 value)
		{
			float3 ret = new float3(src - value.x, src - value.y, src - value.z);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 Multiply(float3 src, float3 value)
		{
			float3 ret = new float3(src.x * value.x, src.y * value.y, src.z * value.z);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 Multiply(float3 src, float value)
		{
			float3 ret = new float3(src.x * value, src.y * value, src.z * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 Multiply(float src, float3 value)
		{
			float3 ret = new float3(src * value.x, src * value.y, src * value.z);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 Divide(float3 src, float3 value)
		{
			float3 ret = new float3(src.x / value.x, src.y / value.y, src.z / value.z);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 Divide(float3 src, float value)
		{
			float3 ret = new float3(src.x / value, src.y / value, src.z / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 Divide(float src, float3 value)
		{
			float3 ret = new float3(src / value.x, src / value.y, src / value.z);
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 operator +(float3 src, float3 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 operator +(float3 src, float value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 operator +(float src, float3 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 operator -(float3 src, float3 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 operator -(float3 src, float value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 operator -(float src, float3 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 operator *(float3 src, float3 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 operator *(float3 src, float value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 operator *(float src, float3 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 operator /(float3 src, float3 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 operator /(float3 src, float value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 operator /(float src, float3 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(float3 src, float3 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(float3 src, float3 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is float3)) return false;

			float3 value = (float3)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(float3 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1}; {2})", this.x, this.y, this.z);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		/// <param name="zValue"></param>
		public float3(float xValue, float yValue, float zValue)
		{
			this.x = xValue;
			this.y = yValue;
			this.z = zValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public float3(float val)
		{
			this.x = val;
			this.y = val;
			this.z = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Dot product
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static float Dot(float3 aValue, float3 bValue)
		{
			return aValue.x * bValue.x + aValue.y * bValue.y + aValue.z * bValue.z;
		}

		/// <summary>
		/// Dot product
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public float Dot(float3 value)
		{
			return this.x * value.x + this.y * value.y + this.z * value.z;
		}

		/// <summary>
		/// Cross product
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static float3 Cross(float3 aValue, float3 bValue)
		{
			return new float3(aValue.y * bValue.z - aValue.z * bValue.y,
							  aValue.z * bValue.x - aValue.x * bValue.z,
							  aValue.x * bValue.y - aValue.y * bValue.x);
		}

		/// <summary>
		/// Cross product (this x b)
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public float3 Cross(float3 value)
		{
			return new float3(this.y * value.z - this.z * value.y,
							  this.z * value.x - this.x * value.z,
							  this.x * value.y - this.y * value.x);
		}

		/// <summary>
		/// Vector length
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float GetLength(float3 value)
		{
			return (float)Math.Sqrt(Dot(value,value));
		}

		/// <summary>
		/// Vector length
		/// </summary>
		public float Length
		{ get { return (float)Math.Sqrt(Dot(this,this)); } }

		/// <summary>
		/// Normalize vector
		/// </summary>
		public void Normalize()
		{
			this = this / Length;
		}

		/// <summary>
		/// Normalize vector
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float3 Normalize(float3 value)
		{
			return value / GetLength(value);
		}

		/// <summary>
		/// Component wise minimum as the CUDA function fminf
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static float3 Min(float3 aValue, float3 bValue)
		{
			return new float3(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y), Math.Min(aValue.z, bValue.z));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function fmaxf
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static float3 Max(float3 aValue, float3 bValue)
		{
			return new float3(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y), Math.Max(aValue.z, bValue.z));
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="angle"></param>
		public void RotateX(double angle)
		{
			double zValue, yValue;
			yValue = y * Math.Cos(angle) - z * Math.Sin(angle);
			zValue = y * Math.Sin(angle) + z * Math.Cos(angle);
			z = (float)zValue;
			y = (float)yValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="angle"></param>
		public void RotateY(double angle)
		{
			double xValue, zValue;
			zValue = z * Math.Cos(angle) - x * Math.Sin(angle);
			xValue = z * Math.Sin(angle) + x * Math.Cos(angle);
			x = (float)xValue;
			z = (float)zValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="angle"></param>
		public void RotateZ(double angle)
		{
			double xValue, yValue;
			xValue = x * Math.Cos(angle) - y * Math.Sin(angle);
			yValue = x * Math.Sin(angle) + y * Math.Cos(angle);
			x = (float)xValue;
			y = (float)yValue;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(float3);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(float3));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// float4
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct float4 : ICudaVectorType, ICudaVectorTypeForArray
	{
		/// <summary>
		/// X
		/// </summary>
		public float x;
		/// <summary>
		/// Y
		/// </summary>
		public float y;
		/// <summary>
		/// Z
		/// </summary>
		public float z;
		/// <summary>
		/// W
		/// </summary>
		public float w;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 Add(float4 src, float4 value)
		{
			float4 ret = new float4(src.x + value.x, src.y + value.y, src.z + value.z, src.w + value.w);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 Add(float4 src, float value)
		{
			float4 ret = new float4(src.x + value, src.y + value, src.z + value, src.w + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 Add(float src, float4 value)
		{
			float4 ret = new float4(src + value.x, src + value.y, src + value.z, src + value.w);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 Subtract(float4 src, float4 value)
		{
			float4 ret = new float4(src.x - value.x, src.y - value.y, src.z - value.z, src.w - value.w);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 Subtract(float4 src, float value)
		{
			float4 ret = new float4(src.x - value, src.y - value, src.z - value, src.w - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 Subtract(float src, float4 value)
		{
			float4 ret = new float4(src - value.x, src - value.y, src - value.z, src - value.w);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 Multiply(float4 src, float4 value)
		{
			float4 ret = new float4(src.x * value.x, src.y * value.y, src.z * value.z, src.w * value.w);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 Multiply(float4 src, float value)
		{
			float4 ret = new float4(src.x * value, src.y * value, src.z * value, src.w * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 Multiply(float src, float4 value)
		{
			float4 ret = new float4(src * value.x, src * value.y, src * value.z, src * value.w);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 Divide(float4 src, float4 value)
		{
			float4 ret = new float4(src.x / value.x, src.y / value.y, src.z / value.z, src.w / value.w);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 Divide(float4 src, float value)
		{
			float4 ret = new float4(src.x / value, src.y / value, src.z / value, src.w / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 Divide(float src, float4 value)
		{
			float4 ret = new float4(src / value.x, src / value.y, src / value.z, src / value.w);
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 operator +(float4 src, float4 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 operator +(float4 src, float value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 operator +(float src, float4 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 operator -(float4 src, float4 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 operator -(float4 src, float value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 operator -(float src, float4 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 operator *(float4 src, float4 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 operator *(float4 src, float value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 operator *(float src, float4 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 operator /(float4 src, float4 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 operator /(float4 src, float value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static float4 operator /(float src, float4 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(float4 src, float4 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(float4 src, float4 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is float4)) return false;

			float4 value = (float4)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			ret &= this.w == value.w;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(float4 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			ret &= this.w == value.w;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode() ^ w.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1}; {2}; {3})", this.x, this.y, this.z, this.w);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		/// <param name="zValue"></param>
		/// <param name="wValue"></param>
		public float4(float xValue, float yValue, float zValue, float wValue)
		{
			this.x = xValue;
			this.y = yValue;
			this.z = zValue;
			this.w = wValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public float4(float val)
		{
			this.x = val;
			this.y = val;
			this.z = val;
			this.w = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function fminf
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static float4 Min(float4 aValue, float4 bValue)
		{
			return new float4(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y), Math.Min(aValue.z, bValue.z), Math.Min(aValue.w, bValue.w));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function fmaxf
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static float4 Max(float4 aValue, float4 bValue)
		{
			return new float4(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y), Math.Max(aValue.z, bValue.z), Math.Max(aValue.w, bValue.w));
		}

		/// <summary>
		/// Returns the Channel number from vector type.
		/// </summary>
		/// <returns>4</returns>
		public uint GetChannelNumber()
		{
			return 4;
		}

		/// <summary>
		/// Returns a matching CUArrayFormat.
		/// </summary>
		/// <returns></returns>
		public CUArrayFormat GetCUArrayFormat()
		{
			return CUArrayFormat.Float;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(float4);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(float4));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}
	#endregion

	#region double
	/// <summary>
	/// double1
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct double1 : ICudaVectorType
	{
		/// <summary>
		/// X
		/// </summary>
		public double x;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 Add(double1 src, double1 value)
		{
			double1 ret = new double1(src.x + value.x);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 Add(double1 src, double value)
		{
			double1 ret = new double1(src.x + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 Add(double src, double1 value)
		{
			double1 ret = new double1(src + value.x);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 Subtract(double1 src, double1 value)
		{
			double1 ret = new double1(src.x - value.x);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 Subtract(double1 src, double value)
		{
			double1 ret = new double1(src.x - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 Subtract(double src, double1 value)
		{
			double1 ret = new double1(src - value.x);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 Multiply(double1 src, double1 value)
		{
			double1 ret = new double1(src.x * value.x);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 Multiply(double1 src, double value)
		{
			double1 ret = new double1(src.x * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 Multiply(double src, double1 value)
		{
			double1 ret = new double1(src * value.x);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 Divide(double1 src, double1 value)
		{
			double1 ret = new double1(src.x / value.x);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 Divide(double1 src, double value)
		{
			double1 ret = new double1(src.x / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 Divide(double src, double1 value)
		{
			double1 ret = new double1(src / value.x);
			return ret;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static double ToDouble(double1 src)
		{
			return src.x;
		}

		/// <summary>
		/// Cast Method
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static double1 FromDouble(double src)
		{
			return new double1(src);
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 operator +(double1 src, double1 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 operator +(double1 src, double value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 operator +(double src, double1 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 operator -(double1 src, double1 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 operator -(double1 src, double value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 operator -(double src, double1 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 operator *(double1 src, double1 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 operator *(double1 src, double value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 operator *(double src, double1 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 operator /(double1 src, double1 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 operator /(double1 src, double value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double1 operator /(double src, double1 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(double1 src, double1 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(double1 src, double1 value)
		{
			return !(src == value);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator double(double1 src)
		{
			return ToDouble(src);
		}

		/// <summary>
		/// Implicit cast
		/// </summary>
		/// <param name="src"></param>
		/// <returns></returns>
		public static implicit operator double1(double src)
		{
			return FromDouble(src);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is double1)) return false;

			double1 value = (double1)obj;

			bool ret = true;
			ret &= this.x == value.x;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(double1 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0})", this.x);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		public double1(double xValue)
		{
			this.x = xValue;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function fminf
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static double1 Min(double1 aValue, double1 bValue)
		{
			return new double1(Math.Min(aValue.x, bValue.x));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function fmaxf
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static double1 Max(double1 aValue, double1 bValue)
		{
			return new double1(Math.Max(aValue.x, bValue.x));
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(double1);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(double1));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// double2
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct double2 : ICudaVectorType
	{
		/// <summary>
		/// X
		/// </summary>
		public double x;
		/// <summary>
		/// Y
		/// </summary>
		public double y;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 Add(double2 src, double2 value)
		{
			double2 ret = new double2(src.x + value.x, src.y + value.y);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 Add(double2 src, double value)
		{
			double2 ret = new double2(src.x + value, src.y + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 Add(double src, double2 value)
		{
			double2 ret = new double2(src + value.x, src + value.y);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 Subtract(double2 src, double2 value)
		{
			double2 ret = new double2(src.x - value.x, src.y - value.y);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 Subtract(double2 src, double value)
		{
			double2 ret = new double2(src.x - value, src.y - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 Subtract(double src, double2 value)
		{
			double2 ret = new double2(src - value.x, src - value.y);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 Multiply(double2 src, double2 value)
		{
			double2 ret = new double2(src.x * value.x, src.y * value.y);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 Multiply(double2 src, double value)
		{
			double2 ret = new double2(src.x * value, src.y * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 Multiply(double src, double2 value)
		{
			double2 ret = new double2(src * value.x, src * value.y);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 Divide(double2 src, double2 value)
		{
			double2 ret = new double2(src.x / value.x, src.y / value.y);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 Divide(double2 src, double value)
		{
			double2 ret = new double2(src.x / value, src.y / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 Divide(double src, double2 value)
		{
			double2 ret = new double2(src / value.x, src / value.y);
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 operator +(double2 src, double2 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 operator +(double2 src, double value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 operator +(double src, double2 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 operator -(double2 src, double2 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 operator -(double2 src, double value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 operator -(double src, double2 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 operator *(double2 src, double2 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 operator *(double2 src, double value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 operator *(double src, double2 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 operator /(double2 src, double2 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 operator /(double2 src, double value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double2 operator /(double src, double2 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(double2 src, double2 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(double2 src, double2 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is double2)) return false;

			double2 value = (double2)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(double2 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1})", this.x, this.y);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		public double2(double xValue, double yValue)
		{
			this.x = xValue;
			this.y = yValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public double2(double val)
		{
			this.x = val;
			this.y = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function fminf
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static double2 Min(double2 aValue, double2 bValue)
		{
			return new double2(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function fmaxf
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static double2 Max(double2 aValue, double2 bValue)
		{
			return new double2(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y));
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(double2);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(double2));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}


	/// <summary>
	/// double3
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct double3 : ICudaVectorType
	{
		/// <summary>
		/// X
		/// </summary>
		public double x;
		/// <summary>
		/// Y
		/// </summary>
		public double y;
		/// <summary>
		/// Z
		/// </summary>
		public double z;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 Add(double3 src, double3 value)
		{
			double3 ret = new double3(src.x + value.x, src.y + value.y, src.z + value.z);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 Add(double3 src, double value)
		{
			double3 ret = new double3(src.x + value, src.y + value, src.z + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 Add(double src, double3 value)
		{
			double3 ret = new double3(src + value.x, src + value.y, src + value.z);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 Subtract(double3 src, double3 value)
		{
			double3 ret = new double3(src.x - value.x, src.y - value.y, src.z - value.z);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 Subtract(double3 src, double value)
		{
			double3 ret = new double3(src.x - value, src.y - value, src.z - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 Subtract(double src, double3 value)
		{
			double3 ret = new double3(src - value.x, src - value.y, src - value.z);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 Multiply(double3 src, double3 value)
		{
			double3 ret = new double3(src.x * value.x, src.y * value.y, src.z * value.z);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 Multiply(double3 src, double value)
		{
			double3 ret = new double3(src.x * value, src.y * value, src.z * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 Multiply(double src, double3 value)
		{
			double3 ret = new double3(src * value.x, src * value.y, src * value.z);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 Divide(double3 src, double3 value)
		{
			double3 ret = new double3(src.x / value.x, src.y / value.y, src.z / value.z);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 Divide(double3 src, double value)
		{
			double3 ret = new double3(src.x / value, src.y / value, src.z / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 Divide(double src, double3 value)
		{
			double3 ret = new double3(src / value.x, src / value.y, src / value.z);
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 operator +(double3 src, double3 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 operator +(double3 src, double value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 operator +(double src, double3 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 operator -(double3 src, double3 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 operator -(double3 src, double value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 operator -(double src, double3 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 operator *(double3 src, double3 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 operator *(double3 src, double value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 operator *(double src, double3 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 operator /(double3 src, double3 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 operator /(double3 src, double value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 operator /(double src, double3 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(double3 src, double3 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(double3 src, double3 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is double3)) return false;

			double3 value = (double3)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(double3 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1}; {2})", this.x, this.y, this.z);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		/// <param name="zValue"></param>
		public double3(double xValue, double yValue, double zValue)
		{
			this.x = xValue;
			this.y = yValue;
			this.z = zValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public double3(double val)
		{
			this.x = val;
			this.y = val;
			this.z = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Dot product
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static double Dot(double3 aValue, double3 bValue)
		{
			return aValue.x * bValue.x + aValue.y * bValue.y + aValue.z * bValue.z;
		}

		/// <summary>
		/// Dot product
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public double Dot(double3 value)
		{
			return this.x * value.x + this.y * value.y + this.z * value.z;
		}

		/// <summary>
		/// Cross product
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static double3 Cross(double3 aValue, double3 bValue)
		{
			return new double3(aValue.y * bValue.z - aValue.z * bValue.y,
							  aValue.z * bValue.x - aValue.x * bValue.z,
							  aValue.x * bValue.y - aValue.y * bValue.x);
		}

		/// <summary>
		/// Cross product (this x b)
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public double3 Cross(double3 value)
		{
			return new double3(this.y * value.z - this.z * value.y,
							  this.z * value.x - this.x * value.z,
							  this.x * value.y - this.y * value.x);
		}

		/// <summary>
		/// Vector length
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double GetLength(double3 value)
		{
			return Math.Sqrt(Dot(value, value));
		}

		/// <summary>
		/// Vector length
		/// </summary>
		public double Length
		{ get { return Math.Sqrt(Dot(this, this)); } }

		/// <summary>
		/// Normalize vector
		/// </summary>
		public void Normalize()
		{
			this = this / Length;
		}

		/// <summary>
		/// Normalize vector
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double3 Normalize(double3 value)
		{
			return value / GetLength(value);
		}

		/// <summary>
		/// Component wise minimum as the CUDA function fminf
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static double3 Min(double3 aValue, double3 bValue)
		{
			return new double3(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y), Math.Min(aValue.z, bValue.z));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function fmaxf
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static double3 Max(double3 aValue, double3 bValue)
		{
			return new double3(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y), Math.Max(aValue.z, bValue.z));
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="angle"></param>
		public void RotateX(double angle)
		{
			double zValue, yValue;
			yValue = y * Math.Cos(angle) - z * Math.Sin(angle);
			zValue = y * Math.Sin(angle) + z * Math.Cos(angle);
			z = zValue;
			y = yValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="angle"></param>
		public void RotateY(double angle)
		{
			double xValue, zValue;
			zValue = z * Math.Cos(angle) - x * Math.Sin(angle);
			xValue = z * Math.Sin(angle) + x * Math.Cos(angle);
			x = xValue;
			z = zValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="angle"></param>
		public void RotateZ(double angle)
		{
			double xValue, yValue;
			xValue = x * Math.Cos(angle) - y * Math.Sin(angle);
			yValue = x * Math.Sin(angle) + y * Math.Cos(angle);
			x = xValue;
			y = yValue;
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(double3);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(double3));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}

	/// <summary>
	/// double4
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct double4 : ICudaVectorType
	{
		/// <summary>
		/// X
		/// </summary>
		public double x;
		/// <summary>
		/// Y
		/// </summary>
		public double y;
		/// <summary>
		/// Z
		/// </summary>
		public double z;
		/// <summary>
		/// W
		/// </summary>
		public double w;

		#region Operator Methods
		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 Add(double4 src, double4 value)
		{
			double4 ret = new double4(src.x + value.x, src.y + value.y, src.z + value.z, src.w + value.w);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 Add(double4 src, double value)
		{
			double4 ret = new double4(src.x + value, src.y + value, src.z + value, src.w + value);
			return ret;
		}

		/// <summary>
		/// per element Add
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 Add(double src, double4 value)
		{
			double4 ret = new double4(src + value.x, src + value.y, src + value.z, src + value.w);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 Subtract(double4 src, double4 value)
		{
			double4 ret = new double4(src.x - value.x, src.y - value.y, src.z - value.z, src.w - value.w);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 Subtract(double4 src, double value)
		{
			double4 ret = new double4(src.x - value, src.y - value, src.z - value, src.w - value);
			return ret;
		}

		/// <summary>
		/// per element Substract
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 Subtract(double src, double4 value)
		{
			double4 ret = new double4(src - value.x, src - value.y, src - value.z, src - value.w);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 Multiply(double4 src, double4 value)
		{
			double4 ret = new double4(src.x * value.x, src.y * value.y, src.z * value.z, src.w * value.w);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 Multiply(double4 src, double value)
		{
			double4 ret = new double4(src.x * value, src.y * value, src.z * value, src.w * value);
			return ret;
		}

		/// <summary>
		/// per element Multiply
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 Multiply(double src, double4 value)
		{
			double4 ret = new double4(src * value.x, src * value.y, src * value.z, src * value.w);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 Divide(double4 src, double4 value)
		{
			double4 ret = new double4(src.x / value.x, src.y / value.y, src.z / value.z, src.w / value.w);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 Divide(double4 src, double value)
		{
			double4 ret = new double4(src.x / value, src.y / value, src.z / value, src.w / value);
			return ret;
		}

		/// <summary>
		/// per element Divide
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 Divide(double src, double4 value)
		{
			double4 ret = new double4(src / value.x, src / value.y, src / value.z, src / value.w);
			return ret;
		}
		#endregion

		#region operators
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 operator +(double4 src, double4 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 operator +(double4 src, double value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 operator +(double src, double4 value)
		{
			return Add(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 operator -(double4 src, double4 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 operator -(double4 src, double value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 operator -(double src, double4 value)
		{
			return Subtract(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 operator *(double4 src, double4 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 operator *(double4 src, double value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 operator *(double src, double4 value)
		{
			return Multiply(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 operator /(double4 src, double4 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 operator /(double4 src, double value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static double4 operator /(double src, double4 value)
		{
			return Divide(src, value);
		}

		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator ==(double4 src, double4 value)
		{
			if (object.ReferenceEquals(src, value)) return true;
			return src.Equals(value);
		}
		/// <summary>
		/// per element
		/// </summary>
		/// <param name="src"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		public static bool operator !=(double4 src, double4 value)
		{
			return !(src == value);
		}
		#endregion

		#region Override Methods
		/// <summary>
		/// 
		/// </summary>
		/// <param name="obj"></param>
		/// <returns></returns>
		public override bool Equals(object obj)
		{
			if (obj == null) return false;
			if (!(obj is double4)) return false;

			double4 value = (double4)obj;

			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			ret &= this.w == value.w;
			return ret;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="value"></param>
		/// <returns></returns>
		public bool Equals(double4 value)
		{
			bool ret = true;
			ret &= this.x == value.x;
			ret &= this.y == value.y;
			ret &= this.z == value.z;
			ret &= this.w == value.w;
			return ret;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override int GetHashCode()
		{
			return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode() ^ w.GetHashCode();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return string.Format(CultureInfo.CurrentCulture, "({0}; {1}; {2}; {3})", this.x, this.y, this.z, this.w);
		}
		#endregion

		#region constructors
		/// <summary>
		/// 
		/// </summary>
		/// <param name="xValue"></param>
		/// <param name="yValue"></param>
		/// <param name="zValue"></param>
		/// <param name="wValue"></param>
		public double4(double xValue, double yValue, double zValue, double wValue)
		{
			this.x = xValue;
			this.y = yValue;
			this.z = zValue;
			this.w = wValue;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="val"></param>
		public double4(double val)
		{
			this.x = val;
			this.y = val;
			this.z = val;
			this.w = val;
		}
		#endregion

		#region Methods
		/// <summary>
		/// Component wise minimum as the CUDA function fminf
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static double4 Min(double4 aValue, double4 bValue)
		{
			return new double4(Math.Min(aValue.x, bValue.x), Math.Min(aValue.y, bValue.y), Math.Min(aValue.z, bValue.z), Math.Min(aValue.w, bValue.w));
		}

		/// <summary>
		/// Component wise maximum as the CUDA function fmaxf
		/// </summary>
		/// <param name="aValue"></param>
		/// <param name="bValue"></param>
		/// <returns></returns>
		public static double4 Max(double4 aValue, double4 bValue)
		{
			return new double4(Math.Max(aValue.x, bValue.x), Math.Max(aValue.y, bValue.y), Math.Max(aValue.z, bValue.z), Math.Max(aValue.w, bValue.w));
		}
		#endregion

		#region SizeOf
		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(double4);</c>
		/// </summary>
		public static uint SizeOf
		{
			get
			{
				return (uint)Marshal.SizeOf(typeof(double4));
			}
		}

		/// <summary>
		/// Gives the size of this type in bytes. <para/>
		/// Is equal to <c>Marshal.SizeOf(this);</c>
		/// </summary>
		public uint Size
		{
			get
			{
				return (uint)Marshal.SizeOf(this);
			}
		}
		#endregion
	}
	#endregion

}
