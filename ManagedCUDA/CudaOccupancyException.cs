//	Copyright (c) 2014, Michael Kunz. All rights reserved.
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
using System.Linq;
using System.Text;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.InteropServices;

namespace ManagedCuda
{
	/// <summary>
	/// A CudaOccupancy exception is thrown if a CudaOccupancy API method call does not return 0
	/// </summary>
	[Serializable] 
	public class CudaOccupancyException : Exception, System.Runtime.Serialization.ISerializable
	{
		private CudaOccupancy.cudaOccError _cudaOccError;

		#region Constructors
		/// <summary>
		/// 
		/// </summary>
		public CudaOccupancyException()
		{ 
		
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="serInfo"></param>
		/// <param name="streamingContext"></param>
		protected CudaOccupancyException(SerializationInfo serInfo, StreamingContext streamingContext)
			: base(serInfo, streamingContext)
		{
		}


		/// <summary>
		/// 
		/// </summary>
		/// <param name="error"></param>
		public CudaOccupancyException(CudaOccupancy.cudaOccError error)
			: base(GetErrorMessageFromCudaOccError(error))
		{
			this._cudaOccError = error;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="message"></param>
		public CudaOccupancyException(string message)
			: base(message)
		{

		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="message"></param>
		/// <param name="exception"></param>
		public CudaOccupancyException(string message, Exception exception)
			: base(message, exception)
		{

		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="error"></param>
		/// <param name="message"></param>
		/// <param name="exception"></param>
		public CudaOccupancyException(CudaOccupancy.cudaOccError error, string message, Exception exception)
			: base(message, exception)
		{
			this._cudaOccError = error;
		}
		#endregion

		#region Methods
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return this.CudaOccError.ToString();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="info"></param>
		/// <param name="context"></param>
		public override void GetObjectData(SerializationInfo info, StreamingContext context)
		{
			base.GetObjectData(info, context);
			info.AddValue("Cuda Occupancy Error", this._cudaOccError);
		}
		#endregion

		#region Static methods
		private static string GetErrorMessageFromCudaOccError(CudaOccupancy.cudaOccError error)
		{
			string message = string.Empty;

			switch (error)
			{
				case CudaOccupancy.cudaOccError.None:
					message = "No error.";
					break;
				case CudaOccupancy.cudaOccError.ErrorInvalidInput:
					message = "Input parameter is invalid.";
					break;
				case CudaOccupancy.cudaOccError.ErrorUnknownDevice:
					message = "Requested device is not supported in current implementation or device is invalid.";
					break;
				
				default:
					break;
			}
			return error.ToString() + ": " + message;
		}
		/// <summary>
		/// Checks if value is zero. If value is zero, CudaOccupancyException is thrown.
		/// </summary>
		/// <param name="value"></param>
		public static void CheckZero(int value)
		{
			if (value == 0)
				throw new CudaOccupancyException(CudaOccupancy.cudaOccError.ErrorInvalidInput);
		}
		#endregion

		#region Properties
		/// <summary>
		/// 
		/// </summary>
		public CudaOccupancy.cudaOccError CudaOccError
		{
			get
			{
				return this._cudaOccError;
			}
			set
			{
				this._cudaOccError = value;
			}
		}
		#endregion
	}
}
