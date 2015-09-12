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
using System.ComponentModel;
using System.Text;
using ManagedCuda.BasicTypes;
using System.IO;
using System.Runtime.Serialization;

namespace ManagedCuda.NPP
{
	/// <summary>
	/// Exception thrown in NPP library if a native NPP function returns a negative error code
	/// </summary>
	public class NPPException : Exception, System.Runtime.Serialization.ISerializable
	{

		private NppStatus _nppError;

		#region Constructors
		/// <summary>
		/// 
		/// </summary>
		public NPPException()
		{

		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="serInfo"></param>
		/// <param name="streamingContext"></param>
		protected NPPException(SerializationInfo serInfo, StreamingContext streamingContext)
			: base(serInfo, streamingContext)
		{
		}


		/// <summary>
		/// 
		/// </summary>
		/// <param name="error"></param>
		public NPPException(NppStatus error)
			: base(GetErrorMessageFromNppStatus(error))
		{
			this._nppError = error;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="message"></param>
		public NPPException(string message)
			: base(message)
		{

		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="message"></param>
		/// <param name="exception"></param>
		public NPPException(string message, Exception exception)
			: base(message, exception)
		{

		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="error"></param>
		/// <param name="message"></param>
		/// <param name="exception"></param>
		public NPPException(NppStatus error, string message, Exception exception)
			: base(message, exception)
		{
			this._nppError = error;
		}
		#endregion

		#region Methods
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return this._nppError.ToString();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="info"></param>
		/// <param name="context"></param>
		public override void GetObjectData(SerializationInfo info, StreamingContext context)
		{
			base.GetObjectData(info, context);
			info.AddValue("NppError", this._nppError);
		}
		#endregion

		#region Static methods
		internal static string GetErrorMessageFromNppStatus(NppStatus error)
		{
			string message = string.Empty;

			switch (error)
			{
				case NppStatus.NotSupportedModeError:
					break;
				case NppStatus.InvalidHostPointerError:
					break;
				case NppStatus.InvalidDevicePointerError:
					break;
				case NppStatus.LUTPaletteBitsizeError:
					break;
				case NppStatus.ZCModeNotSupportedError:
					message = "ZeroCrossing mode not supported.";
					break;
				case NppStatus.NotSufficientComputeCapability:
					break;
				case NppStatus.TextureBindError:
					break;
				case NppStatus.WrongIntersectionRoiError:
					break;
				case NppStatus.HaarClassifierPixelMatchError:
					break;
				case NppStatus.MemfreeError:
					break;
				case NppStatus.MemsetError:
					break;
				case NppStatus.MemcpyError:
					break;
				case NppStatus.AlignmentError:
					break;
				case NppStatus.CudaKernelExecutionError:
					break;
				case NppStatus.RoundModeNotSupportedError:
					message = "Unsupported round mode.";
					break;
				case NppStatus.QualityIndexError:
					message = "Image pixels are constant for quality index.";
					break;
				case NppStatus.ResizeNoOperationError:
					message = "One of the output image dimensions is less than 1 pixel.";
					break;
				case NppStatus.OverflowError:
					message = "Number overflows the upper or lower limit of the data type.";
					break;
				case NppStatus.NotEvenStepError:
					message = "Step value is not pixel multiple.";
					break;
				case NppStatus.HistogramNumberOfLevelsError:
					message = "Number of levels for histogram is less than 2.";
					break;
				case NppStatus.LutMumberOfLevelsError:
					message = "Number of levels for LUT is less than 2.";
					break;
				case NppStatus.ChannelOrderError:
					message = "Wrong order of the destination channels.";
					break;
				case NppStatus.ZeroMaskValueError:
					message = "All values of the mask are zero.";
					break;
				case NppStatus.QuadrangleError:
					message = "The quadrangle is nonconvex or degenerates into triangle, line or point.";
					break;
				case NppStatus.RectangleError:
					message = "Size of the rectangle region is less than or equal to 1.";
					break;
				case NppStatus.CoefficientError:
					message = "Unallowable values of the transformation coefficients.";
					break;
				case NppStatus.NumberOfChannelsError:
					message = "Bad or unsupported number of channels.";
					break;
				case NppStatus.ChannelOfInterestError:
					message = "Channel of interest is not 1, 2, or 3.";
					break;
				case NppStatus.DivisorError:
					message = "Divisor is equal to zero.";
					break;
				case NppStatus.CorruptedDataError:
					message = "Processed data is corrupted.";
					break;
				case NppStatus.ChannelError:
					message = "Illegal channel index.";
					break;
				case NppStatus.StrideError:
					message = "Stride is less than the row length.";
					break;
				case NppStatus.AnchorError:
					message = "Anchor point is outside mask.";
					break;
				case NppStatus.MaskSizeError:
					message = "Lower bound is larger than upper bound.";
					break;
				case NppStatus.ResizeFactorError:
					break;
				case NppStatus.InterpolationError:
					break;
				case NppStatus.MirrorFlipError:
					break;
				case NppStatus.Moment00ZeroErro:
					break;
				case NppStatus.ThresholdNegativeLevelError:
					break;
				case NppStatus.ThresholdError:
					break;
				case NppStatus.ContextMatchError:
					break;
				case NppStatus.FFTFlagError:
					break;
				case NppStatus.FFTOrderError:
					break;
				case NppStatus.StepError:
					message = "Step is less or equal zero.";
					break;
				case NppStatus.ScaleRangeError:
					break;
				case NppStatus.DataTypeError:
					break;
				case NppStatus.OutOfRangeError:
					break;
				case NppStatus.DivideByZeroError:
					break;
				case NppStatus.MemoryAllocationError:
					break;
				case NppStatus.NullPointerError:
					break;
				case NppStatus.RangeError:
					break;
				case NppStatus.SizeError:
					break;
				case NppStatus.BadArgumentError:
					break;
				case NppStatus.NoMemoryError:
					break;
				case NppStatus.NotImplementedError:
					break;
				case NppStatus.Error:
					break;
				case NppStatus.ErrorReserved:
					break;
				case NppStatus.NoError:
					message = "Successful operation.";
					break;
				//case NppStatus.Success:
				//    break;
				case NppStatus.NoOperationWarning:
					message = "Indicates that no operation was performed.";
					break;
				case NppStatus.DivideByZeroWarning:
					message = "Divisor is zero however does not terminate the execution.";
					break;
				case NppStatus.AffineQuadIncorrectWarning:
					message = "Indicates that the quadrangle passed to one of affine warping functions doesn't have necessary properties. First 3 vertices are used, the fourth vertex discarded.";
					break;
				case NppStatus.WrongIntersectionRoiWarning:
					message = "The given ROI has no interestion with either the source or destination ROI. Thus no operation was performed.";
					break;
				case NppStatus.WrongIntersectionQuadWarning:
					message = "The given quadrangle has no intersection with either the source or destination ROI. Thus no operation was performed.";
					break;
				case NppStatus.DoubleSizeWarning:
					message = "Image size isn't multiple of two. Indicates that in case of 422/411/420 sampling the ROI width/height was modified for proper processing.";
					break;
				case NppStatus.MisalignedDstRoiWarning:
					message = "Speed reduction due to uncoalesced memory accesses warning.";
					break;
				default:
					break;
			}

			return error.ToString() + ": " + message;
		}

		internal static void CheckNppStatus(NppStatus status, bool throwWarnings, object sender)
		{
			if (status == NppStatus.NoError) return;
			if ((int)status < 0)
				throw new NPPException(status);
			if (throwWarnings)
				throw new NPPWarning(status);
			else
				NPPWarningHandler.GetInstance().NotifyNPPWarning(sender, status, GetErrorMessageFromNppStatus(status));

		}

		internal static void CheckNppStatus(NppStatus status, object sender)
		{
			CheckNppStatus(status, false, sender);
		}
		#endregion

		#region Properties
		/// <summary>
		/// 
		/// </summary>
		public NppStatus NppError
		{
			get
			{
				return this._nppError;
			}
			set
			{
				this._nppError = value;
			}
		}
		#endregion
	}

	/// <summary>
	/// WarningException thrown if configured and a native NPP function returns a positive error code
	/// </summary>
	public class NPPWarning : WarningException, System.Runtime.Serialization.ISerializable
	{

		private NppStatus _nppError;

		#region Constructors
		/// <summary>
		/// 
		/// </summary>
		public NPPWarning()
		{

		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="serInfo"></param>
		/// <param name="streamingContext"></param>
		protected NPPWarning(SerializationInfo serInfo, StreamingContext streamingContext)
			: base(serInfo, streamingContext)
		{
		}


		/// <summary>
		/// 
		/// </summary>
		/// <param name="error"></param>
		public NPPWarning(NppStatus error)
			: base(NPPException.GetErrorMessageFromNppStatus(error))
		{
			this._nppError = error;
		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="message"></param>
		public NPPWarning(string message)
			: base(message)
		{

		}
		/// <summary>
		/// 
		/// </summary>
		/// <param name="message"></param>
		/// <param name="exception"></param>
		public NPPWarning(string message, Exception exception)
			: base(message, exception)
		{

		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="error"></param>
		/// <param name="message"></param>
		/// <param name="exception"></param>
		public NPPWarning(NppStatus error, string message, Exception exception)
			: base(message, exception)
		{
			this._nppError = error;
		}
		#endregion

		#region Methods
		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return this._nppError.ToString();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="info"></param>
		/// <param name="context"></param>
		public override void GetObjectData(SerializationInfo info, StreamingContext context)
		{
			base.GetObjectData(info, context);
			info.AddValue("NppError", this._nppError);
		}
		#endregion
		
		#region Properties
		/// <summary>
		/// 
		/// </summary>
		public NppStatus NppError
		{
			get
			{
				return this._nppError;
			}
			set
			{
				this._nppError = value;
			}
		}
		#endregion
	}
}
