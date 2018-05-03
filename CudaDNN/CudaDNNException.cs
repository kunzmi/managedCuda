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

namespace ManagedCuda.CudaDNN
{
	/// <summary>
	/// An CudaDNNException is thrown, if any wrapped call to the cudnn-library does not return <see cref="cudnnStatus.Success"/>.
	/// </summary>
	public class CudaDNNException : Exception, System.Runtime.Serialization.ISerializable
	{
		private cudnnStatus _cudnnStatus;

		#region Constructors
		/// <summary>
		/// 
		/// </summary>
		public CudaDNNException()
		{

		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="serInfo"></param>
		/// <param name="streamingContext"></param>
		protected CudaDNNException(SerializationInfo serInfo, StreamingContext streamingContext)
			: base(serInfo, streamingContext)
		{
		}


		/// <summary>
		/// 
		/// </summary>
		/// <param name="error"></param>
		public CudaDNNException(cudnnStatus error)
			: base(GetErrorMessageFromCUResult(error))
		{
			this._cudnnStatus = error;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="message"></param>
		public CudaDNNException(string message)
			: base(message)
		{

		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="message"></param>
		/// <param name="exception"></param>
		public CudaDNNException(string message, Exception exception)
			: base(message, exception)
		{

		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="error"></param>
		/// <param name="message"></param>
		/// <param name="exception"></param>
		public CudaDNNException(cudnnStatus error, string message, Exception exception)
			: base(message, exception)
		{
			this._cudnnStatus = error;
		}
		#endregion

		#region Methods
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return this.DNNStatus.ToString();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="info"></param>
		/// <param name="context"></param>
		public override void GetObjectData(SerializationInfo info, StreamingContext context)
		{
			base.GetObjectData(info, context);
			info.AddValue("DNNStatus", this._cudnnStatus);
		}
		#endregion

		#region Static methods
		private static string GetErrorMessageFromCUResult(cudnnStatus error)
		{
			string message = string.Empty;

			switch (error)
			{
				case cudnnStatus.Success:
					message = "The operation completed successfully.";
					break;
				case cudnnStatus.NotInitialized:
					message = "The cuDNN library was not initialized properly.\nThis error is usually returned when a call to cudnnCreate() fails or when cudnnCreate() has not been called prior to calling another cuDNN routine. In the former case, it is usually due to an error in the CUDA Runtime API called by cudnnCreate() or by an error in the hardware setup.";
					break;
				case cudnnStatus.AllocFailed:
					message = "Resource allocation failed inside the cuDNN library. This is usually caused by an internal cudaMalloc() failure.\nTo correct: prior to the function call, deallocate previously allocated memory as much as possible.";
					break;
				case cudnnStatus.BadParam:
					message = "An incorrect value or parameter was passed to the function.\nTo correct: ensure that all the parameters being passed have valid values.";
					break;
				case cudnnStatus.InternalError:
					message = "An internal cuDNN operation failed.";
					break;
				case cudnnStatus.InvalidValue:
					message = "";
					break;
				case cudnnStatus.ArchMismatch:
					message = "The function requires a feature absent from the current GPU device. Note that cuDNN only supports devices with compute capabilities greater than or equal to 3.0.\nTo correct: compile and run the application on a device with appropriate compute capability.";
					break;
				case cudnnStatus.MappingError:
					message = "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.\nTo correct: prior to the function call, unbind any previously bound textures.\nOtherwise, this may indicate an internal error/bug in the library.";
					break;
				case cudnnStatus.ExecutionFailed:
					message = "The GPU program failed to execute. This is usually caused by a failure to launch some cuDNN kernel on the GPU, which can occur for multiple reasons.\nTo correct: check that the hardware, an appropriate version of the driver, and the cuDNN library are correctly installed.\nOtherwise, this may indicate a internal error/bug in the library.";
					break;
				case cudnnStatus.NotSupported:
					message = "The functionality requested is not presently supported by cuDNN.";
					break;
				case cudnnStatus.LicenseError:
					message = "The functionality requested requires some license and an error was detected when trying to check the current licensing. This error can happen if the license is not present or is expired or if the environment variable NVIDIA_LICENSE_FILE is not set properly.";
					break;
                case cudnnStatus.RuntimePrerequisiteMissing:
                    message = "Runtime library required by RNN calls (libcuda.so or nvcuda.dll) cannot be found in predefined search paths.";
                    break;
                case cudnnStatus.RuntimInProgress:
                    message = "Some tasks in the user stream are not completed.";
                    break;
                case cudnnStatus.RuntimeFPOverflow:
                    message = "Numerical overflow occurred during the GPU kernel execution.";
                    break;
				default:
					break;
			}

			return error.ToString() + ": " + message ;
		}
		#endregion

		#region Properties
		/// <summary>
		/// 
		/// </summary>
		public cudnnStatus DNNStatus
		{
			get
			{
				return this._cudnnStatus;
			}
			set
			{
				this._cudnnStatus = value;
			}
		}
		#endregion
	}
}
