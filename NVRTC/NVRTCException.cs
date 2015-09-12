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

namespace ManagedCuda.NVRTC
{
	/// <summary>
	/// An NVRTCException is thrown, if any wrapped call to the NVRTC-library does not return <see cref="nvrtcResult.Success"/>.
	/// </summary>
	public class NVRTCException : Exception, System.Runtime.Serialization.ISerializable
	{
		private nvrtcResult _NVRTCError;

		#region Constructors
		/// <summary>
		/// 
		/// </summary>
		public NVRTCException()
		{

		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="serInfo"></param>
		/// <param name="streamingContext"></param>
		protected NVRTCException(SerializationInfo serInfo, StreamingContext streamingContext)
			: base(serInfo, streamingContext)
		{
		}


		/// <summary>
		/// 
		/// </summary>
		/// <param name="error"></param>
		public NVRTCException(nvrtcResult error)
			: base(GetErrorMessageFromCUResult(error))
		{
			this._NVRTCError = error;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="message"></param>
		public NVRTCException(string message)
			: base(message)
		{

		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="message"></param>
		/// <param name="exception"></param>
		public NVRTCException(string message, Exception exception)
			: base(message, exception)
		{

		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="error"></param>
		/// <param name="message"></param>
		/// <param name="exception"></param>
		public NVRTCException(nvrtcResult error, string message, Exception exception)
			: base(message, exception)
		{
			this._NVRTCError = error;
		}
		#endregion

		#region Methods
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return this.NVRTCError.ToString();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="info"></param>
		/// <param name="context"></param>
		public override void GetObjectData(SerializationInfo info, StreamingContext context)
		{
			base.GetObjectData(info, context);
			info.AddValue("NVRTCError", this._NVRTCError);
		}
		#endregion

		#region Static methods
		private static string GetErrorMessageFromCUResult(nvrtcResult error)
		{
			string message = string.Empty;

			switch (error)
			{
				case nvrtcResult.Success:
					message = "No Error.";
					break;
				case nvrtcResult.ErrorOutOfMemory:
					message = "Error out of memory.";
					break;
				case nvrtcResult.ErrorProgramCreationFailure:
					message = "Program creation failure.";
					break;
				case nvrtcResult.ErrorInvalidInput:
					message = "Invalid Input.";
					break;
				case nvrtcResult.ErrorInvalidProgram:
					message = "Invalid program.";
					break;
				case nvrtcResult.ErrorInvalidOption:
					message = "Invalid option.";
					break;
				case nvrtcResult.ErrorCompilation:
					message = "Compilation error.";
					break;
				case nvrtcResult.ErrorBuiltinOperationFailure:
					message = "Builtin operation failure.";
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
		public nvrtcResult NVRTCError
		{
			get
			{
				return this._NVRTCError;
			}
			set
			{
				this._NVRTCError = value;
			}
		}
		#endregion
	}
}
