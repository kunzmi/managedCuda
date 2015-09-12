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

namespace ManagedCuda.CudaBlas
{
	/// <summary>
	/// An CudaBlasException is thrown, if any wrapped call to the CUBLAS-library does not return <see cref="CublasStatus.Success"/>.
	/// </summary>
	public class CudaBlasException : Exception, System.Runtime.Serialization.ISerializable
	{

		private CublasStatus _cudaBlasError;

		#region Constructors
		/// <summary>
		/// 
		/// </summary>
		public CudaBlasException()
		{

		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="serInfo"></param>
		/// <param name="streamingContext"></param>
		protected CudaBlasException(SerializationInfo serInfo, StreamingContext streamingContext)
			: base(serInfo, streamingContext)
		{
		}


		/// <summary>
		/// 
		/// </summary>
		/// <param name="error"></param>
		public CudaBlasException(CublasStatus error)
			: base(GetErrorMessageFromCUResult(error))
		{
			this._cudaBlasError = error;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="message"></param>
		public CudaBlasException(string message)
			: base(message)
		{

		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="message"></param>
		/// <param name="exception"></param>
		public CudaBlasException(string message, Exception exception)
			: base(message, exception)
		{

		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="error"></param>
		/// <param name="message"></param>
		/// <param name="exception"></param>
		public CudaBlasException(CublasStatus error, string message, Exception exception)
			: base(message, exception)
		{
			this._cudaBlasError = error;
		}
		#endregion

		#region Methods
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return this._cudaBlasError.ToString();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="info"></param>
		/// <param name="context"></param>
		public override void GetObjectData(SerializationInfo info, StreamingContext context)
		{
			base.GetObjectData(info, context);
			info.AddValue("CudaBlasError", this._cudaBlasError);
		}
		#endregion

		#region Static methods
		private static string GetErrorMessageFromCUResult(CublasStatus error)
		{
			string message = string.Empty;

			switch (error)
			{
				case CublasStatus.Success:
					message = "Any CUBLAS operation is successful.";
					break;
				case CublasStatus.NotInitialized:
					message = "The CUBLAS library was not initialized.";
					break;
				case CublasStatus.AllocFailed:
					message = "Resource allocation failed.";
					break;
				case CublasStatus.InvalidValue:
					message = "An invalid numerical value was used as an argument.";
					break;
				case CublasStatus.ArchMismatch:
					message = "An absent device architectural feature is required.";
					break;
				case CublasStatus.MappingError:
					message = "An access to GPU memory space failed.";
					break;
				case CublasStatus.ExecutionFailed:
					message = "An access to GPU memory space failed.";
					break;
				case CublasStatus.InternalError:
					message = "An internal operation failed.";
					break;
				case CublasStatus.NotSupported:
					message = "Error: Not supported.";
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
		public CublasStatus CudaBlasError
		{
			get
			{
				return this._cudaBlasError;
			}
			set
			{
				this._cudaBlasError = value;
			}
		}
		#endregion
	}
}
