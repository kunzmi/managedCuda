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
using System.Linq;
using System.Text;
using System.IO;
using System.Runtime.Serialization;

namespace ManagedCuda.CudaSparse
{
	/// <summary>
	/// An CudaSparseException is thrown, if any wrapped call to the CUSPARSE-library does not return <see cref="cusparseStatus.Success"/>.
	/// </summary>
	public class CudaSparseException : Exception, System.Runtime.Serialization.ISerializable
	{
		private cusparseStatus _cudaSparseError;

		#region Constructors
		/// <summary>
		/// 
		/// </summary>
		public CudaSparseException()
		{

		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="serInfo"></param>
		/// <param name="streamingContext"></param>
		protected CudaSparseException(SerializationInfo serInfo, StreamingContext streamingContext)
			: base(serInfo, streamingContext)
		{
		}


		/// <summary>
		/// 
		/// </summary>
		/// <param name="error"></param>
		public CudaSparseException(cusparseStatus error)
			: base(GetErrorMessageFromCUResult(error))
		{
			this._cudaSparseError = error;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="message"></param>
		public CudaSparseException(string message)
			: base(message)
		{

		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="message"></param>
		/// <param name="exception"></param>
		public CudaSparseException(string message, Exception exception)
			: base(message, exception)
		{

		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="error"></param>
		/// <param name="message"></param>
		/// <param name="exception"></param>
		public CudaSparseException(cusparseStatus error, string message, Exception exception)
			: base(message, exception)
		{
			this._cudaSparseError = error;
		}
		#endregion

		#region Methods
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return this.CudaSparseError.ToString();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="info"></param>
		/// <param name="context"></param>
		public override void GetObjectData(SerializationInfo info, StreamingContext context)
		{
			base.GetObjectData(info, context);
			info.AddValue("CudaSparseError", this._cudaSparseError);
		}
		#endregion

		#region Static methods
		private static string GetErrorMessageFromCUResult(cusparseStatus error)
		{
			string message = string.Empty;

			switch (error)
			{
				case cusparseStatus.Success:
					message = "Any CUSPARSE operation is successful.";
					break;
				case cusparseStatus.NotInitialized:
					message = "The CUSPARSE library was not initialized. This is usually caused by the lack of a prior "+
						"cusparseCreate() call, an error in the CUDA Runtime API called by the CUSPARSE routine, or an "+
						"error in the hardware setup. To correct: call cusparseCreate() prior to the function call; and"+
						" check that the hardware, an appropriate version of the driver, and the CUSPARSE library are "+
						"correctly installed.";
					break;
				case cusparseStatus.AllocFailed:
					message = "Resource allocation failed inside the CUSPARSE library. This is usually caused by a "+
						"cudaMalloc() failure. To correct: prior to the function call, deallocate previously allocated " +
						"memory as much as possible.";
					break;
				case cusparseStatus.InvalidValue:
					message = "An unsupported value or parameter was passed to the function (a negative vector size, "+
						"for example). To correct: ensure that all the parameters being passed have valid values.";
					break;
				case cusparseStatus.ArchMismatch:
					message = "The function requires a feature absent from the device architecture; usually caused by "+
						"the lack of support for atomic operations or double precision. To correct: compile and run the"+
						" application on a device with appropriate compute capability, which is 1.1 for 32-bit atomic "+
						"operations and 1.3 for double precision.";
					break;
				case cusparseStatus.MappingError:
					message = "An access to GPU memory space failed, which is usually caused by a failure to bind a texture. "+
						"To correct: prior to the function call, unbind any previously bound textures.";
					break;
				case cusparseStatus.ExecutionFailed:
					message = "The GPU program failed to execute. This is often caused by a launch failure of the kernel on "+
						"the GPU, which can be caused by multiple reasons. To correct: check that the hardware, an appropriate"+
						" version of the driver, and the CUSPARSE library are correctly installed.";
					break;
				case cusparseStatus.InternalError:
					message = "An internal CUSPARSE operation failed. This error is usually caused by a cudaMemcpyAsync() "+
						"failure. To correct: check that the hardware, an appropriate version of the driver, and the CUSPARSE"+
						" library are correctly installed. Also, check that the memory passed as a parameter to the routine "+
						"is not being deallocated prior to the routine’s completion.";
					break;
				case cusparseStatus.MatrixTypeNotSupported:
					message = "The matrix type is not supported by this function. This is usually caused by passing an invalid "+
						"matrix descriptor to the function. To correct: check that the fields in cusparseMatDescr_t descrA were "+
						"set correctly.";
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
		public cusparseStatus CudaSparseError
		{
			get
			{
				return this._cudaSparseError;
			}
			set
			{
				this._cudaSparseError = value;
			}
		}
		#endregion
	}
}
