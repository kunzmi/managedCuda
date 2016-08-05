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
using System.Runtime.InteropServices;

namespace ManagedCuda.NVGraph
{
	/// <summary>
	/// A NVGraph exception is thrown if a NVGraph API method call does not return <see cref="nvgraphContext.Success"/>
	/// </summary>
	[Serializable]
	public class NVGraphException : Exception, System.Runtime.Serialization.ISerializable
	{
		private nvgraphStatus _nvgraphError;
		private string _internalName;

		#region Constructors
		/// <summary>
		/// 
		/// </summary>
		public NVGraphException()
		{

		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="serInfo"></param>
		/// <param name="streamingContext"></param>
		protected NVGraphException(SerializationInfo serInfo, StreamingContext streamingContext)
			: base(serInfo, streamingContext)
		{
		}


		/// <summary>
		/// 
		/// </summary>
		/// <param name="error"></param>
		public NVGraphException(nvgraphStatus error)
			: base(GetErrorMessageFromNVgraphStatus(error))
		{
			this._nvgraphError = error;
			this._internalName = GetInternalNameFromNVgraphStatus(error);
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="message"></param>
		public NVGraphException(string message)
			: base(message)
		{

		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="message"></param>
		/// <param name="exception"></param>
		public NVGraphException(string message, Exception exception)
			: base(message, exception)
		{

		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="error"></param>
		/// <param name="message"></param>
		/// <param name="exception"></param>
		public NVGraphException(nvgraphStatus error, string message, Exception exception)
			: base(message, exception)
		{
			this._nvgraphError = error;
			this._internalName = GetInternalNameFromNVgraphStatus(error);
		}
		#endregion

		#region Methods
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return this.NVgraphStatus.ToString();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="info"></param>
		/// <param name="context"></param>
		public override void GetObjectData(SerializationInfo info, StreamingContext context)
		{
			base.GetObjectData(info, context);
			info.AddValue("NVGraphError", this._nvgraphError);
		}
		#endregion

		#region Static methods
		private static string GetErrorMessageFromNVgraphStatus(nvgraphStatus error)
		{
			string message = string.Empty;

			switch (error)
			{
				case nvgraphStatus.Success:
					message = "No error.";
					break;
				case nvgraphStatus.NotInitialized:
					message = "NotInitialized";
					break;
				case nvgraphStatus.AllocFailed:
					message = "AllocFailed";
					break;
				case nvgraphStatus.InvalidValue:
					message = "InvalidValue";
					break;
				case nvgraphStatus.ArchMismatch:
					message = "ArchMismatch";
					break;
				case nvgraphStatus.MappingError:
					message = "MappingError";
					break;
				case nvgraphStatus.ExecutionFailed:
					message = "ExecutionFailed";
					break;
				case nvgraphStatus.InternalError:
					message = "InternalError";
					break;
				case nvgraphStatus.TypeNotSupported:
					message = "TypeNotSupported";
					break;
				case nvgraphStatus.NotConverged:
					message = "NotConverged";
					break;
				default:
					break;
			}
			return error.ToString() + ": " + message;
		}

		private static string GetInternalNameFromNVgraphStatus(nvgraphStatus error)
		{
			string val = NVGraphNativeMathods.nvgraphStatusGetString(error);
			return val;
		}
		#endregion

		#region Properties
		/// <summary>
		/// 
		/// </summary>
		public nvgraphStatus NVgraphStatus
		{
			get
			{
				return this._nvgraphError;
			}
			set
			{
				this._nvgraphError = value;
			}
		}

		/// <summary>
		/// Error name as returned by NVGraph API
		/// </summary>
		public string InternalErrorName
		{
			get
			{
				return this._internalName;
			}
		}
		#endregion
	}
}

