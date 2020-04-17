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
using System.Runtime.InteropServices;
using System.Diagnostics;
using ManagedCuda.BasicTypes;

namespace ManagedCuda.NPP
{
	/// <summary>
	/// 
	/// </summary>
	public partial class NPPImage_32scC3 : NPPImageBase
	{
		#region Copy
		/// <summary>
		/// Image copy.
		/// </summary>
		/// <param name="dst">Destination image</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Copy(NPPImage_32scC3 dst, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MemCopy.nppiCopy_32sc_C3R_Ctx(_devPtrRoi, _pitch, dst.DevicePointerRoi, dst.Pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiCopy_32sc_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Set
		/// <summary>
		/// Set pixel values to nValue.
		/// </summary>
		/// <param name="nValue">Value to be set</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Set(Npp32sc[] nValue, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MemSet.nppiSet_32sc_C3R_Ctx(nValue, _devPtrRoi, _pitch, _sizeRoi, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSet_32sc_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Add
		/// <summary>
		/// Image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Add(NPPImage_32scC3 src2, NPPImage_32scC3 dest, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Add.nppiAdd_32sc_C3RSfs_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_32sc_C3RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Add(NPPImage_32scC3 src2, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Add.nppiAdd_32sc_C3IRSfs_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAdd_32sc_C3IRSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nConstant">Value to add</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Add(Npp32sc[] nConstant, NPPImage_32scC3 dest, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.AddConst.nppiAddC_32sc_C3RSfs_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_32sc_C3RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Add constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
		/// </summary>
		/// <param name="nConstant">Value to add</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Add(Npp32sc[] nConstant, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.AddConst.nppiAddC_32sc_C3IRSfs_Ctx(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAddC_32sc_C3IRSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Sub
		/// <summary>
		/// Image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sub(NPPImage_32scC3 src2, NPPImage_32scC3 dest, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Sub.nppiSub_32sc_C3RSfs_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_32sc_C3RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sub(NPPImage_32scC3 src2, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Sub.nppiSub_32sc_C3IRSfs_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSub_32sc_C3IRSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nConstant">Value to subtract</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sub(Npp32sc[] nConstant, NPPImage_32scC3 dest, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.SubConst.nppiSubC_32sc_C3RSfs_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_32sc_C3RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Subtract constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
		/// </summary>
		/// <param name="nConstant">Value to subtract</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Sub(Npp32sc[] nConstant, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.SubConst.nppiSubC_32sc_C3IRSfs_Ctx(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiSubC_32sc_C3IRSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Mul
		/// <summary>
		/// Image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mul(NPPImage_32scC3 src2, NPPImage_32scC3 dest, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Mul.nppiMul_32sc_C3RSfs_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_32sc_C3RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mul(NPPImage_32scC3 src2, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Mul.nppiMul_32sc_C3IRSfs_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMul_32sc_C3IRSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mul(Npp32sc[] nConstant, NPPImage_32scC3 dest, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MulConst.nppiMulC_32sc_C3RSfs_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_32sc_C3RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Multiply constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Mul(Npp32sc[] nConstant, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.MulConst.nppiMulC_32sc_C3IRSfs_Ctx(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMulC_32sc_C3IRSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region Div
		/// <summary>
		/// Image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Div(NPPImage_32scC3 src2, NPPImage_32scC3 dest, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Div.nppiDiv_32sc_C3RSfs_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_32sc_C3RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// In place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Div(NPPImage_32scC3 src2, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.Div.nppiDiv_32sc_C3IRSfs_Ctx(src2.DevicePointerRoi, src2.Pitch, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDiv_32sc_C3IRSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="dest">Destination image</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Div(Npp32sc[] nConstant, NPPImage_32scC3 dest, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.DivConst.nppiDivC_32sc_C3RSfs_Ctx(_devPtrRoi, _pitch, nConstant, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_32sc_C3RSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Divide constant to image, scale by 2^(-nScaleFactor), then clamp to saturated value. Inplace.
		/// </summary>
		/// <param name="nConstant">Value</param>
		/// <param name="nScaleFactor">scaling factor</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void Div(Npp32sc[] nConstant, int nScaleFactor, NppStreamContext nppStreamCtx)
		{
			status = NPPNativeMethods_Ctx.NPPi.DivConst.nppiDivC_32sc_C3IRSfs_Ctx(nConstant, _devPtrRoi, _pitch, _sizeRoi, nScaleFactor, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiDivC_32sc_C3IRSfs_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		#endregion

		#region MaxError
		/// <summary>
		/// image maximum error. User buffer is internally allocated and freed.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MaxError(NPPImage_32scC3 src2, CudaDeviceVariable<double> pError, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaxErrorGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods_Ctx.NPPi.MaximumError.nppiMaximumError_32sc_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_32sc_C3R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image maximum error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the MaxError operation.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MaxError(NPPImage_32scC3 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaxErrorGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MaximumError.nppiMaximumError_32sc_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumError_32sc_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for MaxError.
		/// </summary>
		/// <returns></returns>
		public int MaxErrorGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.MaximumError.nppiMaximumErrorGetBufferHostSize_32sc_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumErrorGetBufferHostSize_32sc_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}
		#endregion

		#region AverageError
		/// <summary>
		/// image average error. User buffer is internally allocated and freed.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void AverageError(NPPImage_32scC3 src2, CudaDeviceVariable<double> pError, NppStreamContext nppStreamCtx)
		{
			int bufferSize = AverageErrorGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods_Ctx.NPPi.AverageError.nppiAverageError_32sc_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_32sc_C3R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image average error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the AverageError operation.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void AverageError(NPPImage_32scC3 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = AverageErrorGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.AverageError.nppiAverageError_32sc_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageError_32sc_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for AverageError.
		/// </summary>
		/// <returns></returns>
		public int AverageErrorGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.AverageError.nppiAverageErrorGetBufferHostSize_32sc_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageErrorGetBufferHostSize_32sc_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}
		#endregion

		#region MaximumRelative_Error
		/// <summary>
		/// image maximum relative error. User buffer is internally allocated and freed.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MaximumRelativeError(NPPImage_32scC3 src2, CudaDeviceVariable<double> pError, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaximumRelativeErrorGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods_Ctx.NPPi.MaximumRelativeError.nppiMaximumRelativeError_32sc_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_32sc_C3R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image maximum relative error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the MaximumRelativeError operation.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void MaximumRelativeError(NPPImage_32scC3 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = MaximumRelativeErrorGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.MaximumRelativeError.nppiMaximumRelativeError_32sc_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeError_32sc_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for MaximumRelativeError.
		/// </summary>
		/// <returns></returns>
		public int MaximumRelativeErrorGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.MaximumRelativeError.nppiMaximumRelativeErrorGetBufferHostSize_32sc_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiMaximumRelativeErrorGetBufferHostSize_32sc_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}
		#endregion

		#region AverageRelative_Error
		/// <summary>
		/// image average relative error. User buffer is internally allocated and freed.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void AverageRelativeError(NPPImage_32scC3 src2, CudaDeviceVariable<double> pError, NppStreamContext nppStreamCtx)
		{
			int bufferSize = AverageRelativeErrorGetBufferHostSize(nppStreamCtx);
			CudaDeviceVariable<byte> buffer = new CudaDeviceVariable<byte>(bufferSize);
			status = NPPNativeMethods_Ctx.NPPi.AverageRelativeError.nppiAverageRelativeError_32sc_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_32sc_C3R_Ctx", status));
			buffer.Dispose();
			NPPException.CheckNppStatus(status, this);
		}

		/// <summary>
		/// image average relative error.
		/// </summary>
		/// <param name="src2">2nd source image</param>
		/// <param name="pError">Pointer to the computed error.</param>
		/// <param name="buffer">Pointer to the user-allocated scratch buffer required for the AverageRelativeError operation.</param>
		/// <param name="nppStreamCtx">NPP stream context.</param>
		public void AverageRelativeError(NPPImage_32scC3 src2, CudaDeviceVariable<double> pError, CudaDeviceVariable<byte> buffer, NppStreamContext nppStreamCtx)
		{
			int bufferSize = AverageRelativeErrorGetBufferHostSize(nppStreamCtx);
			if (bufferSize > buffer.Size) throw new NPPException("Provided buffer is too small.");

			status = NPPNativeMethods_Ctx.NPPi.AverageRelativeError.nppiAverageRelativeError_32sc_C3R_Ctx(_devPtrRoi, _pitch, src2.DevicePointerRoi, src2.Pitch, _sizeRoi, pError.DevicePointer, buffer.DevicePointer, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeError_32sc_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
		}
		/// <summary>
		/// Device scratch buffer size (in bytes) for AverageRelativeError.
		/// </summary>
		/// <returns></returns>
		public int AverageRelativeErrorGetBufferHostSize(NppStreamContext nppStreamCtx)
		{
			int bufferSize = 0;
			status = NPPNativeMethods_Ctx.NPPi.AverageRelativeError.nppiAverageRelativeErrorGetBufferHostSize_32sc_C3R_Ctx(_sizeRoi, ref bufferSize, nppStreamCtx);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiAverageRelativeErrorGetBufferHostSize_32sc_C3R_Ctx", status));
			NPPException.CheckNppStatus(status, this);
			return bufferSize;
		}
		#endregion
	}
}
