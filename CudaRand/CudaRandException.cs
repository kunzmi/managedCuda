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
using System.IO;
using System.Runtime.Serialization;

namespace ManagedCuda.CudaRand
{
	/// <summary>
	/// An CudaRandException is thrown, if any wrapped call to the CURAND-library does not return <see cref="CurandStatus.Success"/>.
	/// </summary>
	public class CudaRandException : Exception, System.Runtime.Serialization.ISerializable
	{
		private CurandStatus _cudaRandError;

		#region Constructors
		/// <summary>
		/// 
		/// </summary>
		public CudaRandException()
		{

		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="serInfo"></param>
		/// <param name="streamingContext"></param>
		protected CudaRandException(SerializationInfo serInfo, StreamingContext streamingContext)
			: base(serInfo, streamingContext)
		{
		}


		/// <summary>
		/// 
		/// </summary>
		/// <param name="error"></param>
		public CudaRandException(CurandStatus error)
			: base(GetErrorMessageFromCUResult(error))
		{
			this._cudaRandError = error;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="message"></param>
		public CudaRandException(string message)
			: base(message)
		{

		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="message"></param>
		/// <param name="exception"></param>
		public CudaRandException(string message, Exception exception)
			: base(message, exception)
		{

		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="error"></param>
		/// <param name="message"></param>
		/// <param name="exception"></param>
		public CudaRandException(CurandStatus error, string message, Exception exception)
			: base(message, exception)
		{
			this._cudaRandError = error;
		}
		#endregion

		#region Methods
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return this._cudaRandError.ToString();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="info"></param>
		/// <param name="context"></param>
		public override void GetObjectData(SerializationInfo info, StreamingContext context)
		{
			base.GetObjectData(info, context);
			info.AddValue("CudaRandError", this._cudaRandError);
		}
		#endregion

		#region Static methods
		private static string GetErrorMessageFromCUResult(CurandStatus error)
		{
			string message = string.Empty;

			switch (error)
			{
				case CurandStatus.Success:
					message = "Any CURAND operation is successful.";
					break;
				case CurandStatus.VersionMismatch:
					message = "Header file and linked library version do not match.";
					break;
				case CurandStatus.NotInitialized:
					message = "Generator not initialized.";
					break;
				case CurandStatus.AllocationFailed:
					message = "Memory allocation failed.";
					break;
				case CurandStatus.TypeError:
					message = "Generator is wrong type.";
					break;
				case CurandStatus.OutOfRange:
					message = "Argument out of range.";
					break;
				case CurandStatus.LengthNotMultiple:
					message = "Length requested is not a multple of dimension.";
					break;
				case CurandStatus.DoublePrecisionRequired:
					message = "GPU does not have double precision required by MRG32k3a.";
					break;
				case CurandStatus.LaunchFailure:
					message = "Kernel launch failure.";
					break;
				case CurandStatus.PreexistingFailure:
					message = "Preexisting failure on library entry.";
					break;
				case CurandStatus.InitializationFailed:
					message = "Initialization of CUDA failed.";
					break;
				case CurandStatus.ArchMismatch:
					message = "Architecture mismatch, GPU does not support requested feature.";
					break;
				case CurandStatus.InternalError:
					message = "Internal library error.";
					break;
				default:
					break;
			}


			return error.ToString() + ": " + message;
		}
		#endregion

		#region Properties
		/// <summary>
		/// 
		/// </summary>
		public CurandStatus CudaRandError
		{
			get
			{
				return this._cudaRandError;
			}
			set
			{
				this._cudaRandError = value;
			}
		}
		#endregion
	}
}
