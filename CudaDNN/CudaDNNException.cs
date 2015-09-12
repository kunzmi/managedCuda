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
					break;
				case cudnnStatus.NotInitialized:
					break;
				case cudnnStatus.AllocFailed:
					break;
				case cudnnStatus.BadParam:
					break;
				case cudnnStatus.InternalError:
					break;
				case cudnnStatus.InvalidValue:
					break;
				case cudnnStatus.ArchMismatch:
					break;
				case cudnnStatus.MappingError:
					break;
				case cudnnStatus.ExecutionFailed:
					break;
				case cudnnStatus.NotSupported:
					break;
				case cudnnStatus.LicenseError:
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
