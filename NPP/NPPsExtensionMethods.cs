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
using System.Diagnostics;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.NPP;

namespace ManagedCuda.NPP.NPPsExtensions
{
	/// <summary>
	/// Extensions methods extending CudaDeviceVariable with NPPs features.
	/// </summary>
	public static class NPPsExtensionMethods
	{
		#region Arithmetic
		/// <summary>
		/// 8-bit unsigned char in place signal add constant,
		/// scale, then clamp to saturated value
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be added to each vector element</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void AddC(this CudaDeviceVariable<byte> pSrcDst, byte nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddC.nppsAddC_8u_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddC_8u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 8-bit unsigned charvector add constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be added to each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void AddC(this CudaDeviceVariable<byte> pSrc, CudaDeviceVariable<byte> pDst, byte nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddC.nppsAddC_8u_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddC_8u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit unsigned short in place signal add constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be added to each vector element</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void AddC(this CudaDeviceVariable<ushort> pSrcDst, ushort nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddC.nppsAddC_16u_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddC_16u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short vector add constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be added to each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void AddC(this CudaDeviceVariable<ushort> pSrc, CudaDeviceVariable<ushort> pDst, ushort nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddC.nppsAddC_16u_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddC_16u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit signed short in place  signal add constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be added to each vector element</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void AddC(this CudaDeviceVariable<short> pSrcDst, short nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddC.nppsAddC_16s_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddC_16s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short signal add constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be added to each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void AddC(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pDst, short nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddC.nppsAddC_16s_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddC_16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit integer complex number (16 bit real, 16 bit imaginary)signal add constant, 
		/// scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be added to each vector element</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void AddC(this CudaDeviceVariable<Npp16sc> pSrcDst, Npp16sc nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddC.nppsAddC_16sc_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddC_16sc_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit integer complex number (16 bit real, 16 bit imaginary) signal add constant,
		/// scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be added to each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void AddC(this CudaDeviceVariable<Npp16sc> pSrc, CudaDeviceVariable<Npp16sc> pDst, Npp16sc nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddC.nppsAddC_16sc_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddC_16sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit signed integer in place signal add constant and scale.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be added to each vector element</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void AddC(this CudaDeviceVariable<int> pSrcDst, int nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddC.nppsAddC_32s_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddC_32s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit signed integersignal add constant and scale.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be added to each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void AddC(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pDst, int nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddC.nppsAddC_32s_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddC_32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal
		/// add constant and scale.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be added to each vector element</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void AddC(this CudaDeviceVariable<Npp32sc> pSrcDst, Npp32sc nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddC.nppsAddC_32sc_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddC_32sc_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit integer complex number (32 bit real, 32 bit imaginary) signal add constant
		/// and scale.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be added to each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void AddC(this CudaDeviceVariable<Npp32sc> pSrc, CudaDeviceVariable<Npp32sc> pDst, Npp32sc nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddC.nppsAddC_32sc_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddC_32sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point in place signal add constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be added to each vector element</param>

		public static void AddC(this CudaDeviceVariable<float> pSrcDst, float nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddC.nppsAddC_32f_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddC_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point signal add constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be added to each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void AddC(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pDst, float nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddC.nppsAddC_32f(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddC_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
		/// place signal add constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be added to each vector element</param>

		public static void AddC(this CudaDeviceVariable<Npp32fc> pSrcDst, Npp32fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddC.nppsAddC_32fc_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddC_32fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
		/// add constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be added to each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void AddC(this CudaDeviceVariable<Npp32fc> pSrc, CudaDeviceVariable<Npp32fc> pDst, Npp32fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddC.nppsAddC_32fc(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddC_32fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit floating point, in place signal add constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be added to each vector element</param>
		public static void AddC(this CudaDeviceVariable<double> pSrcDst, double nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddC.nppsAddC_64f_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddC_64f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating pointsignal add constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be added to each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void AddC(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pDst, double nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddC.nppsAddC_64f(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddC_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
		/// place signal add constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be added to each vector element</param>

		public static void AddC(this CudaDeviceVariable<Npp64fc> pSrcDst, Npp64fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddC.nppsAddC_64fc_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddC_64fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
		/// add constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be added to each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void AddC(this CudaDeviceVariable<Npp64fc> pSrc, CudaDeviceVariable<Npp64fc> pDst, Npp64fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddC.nppsAddC_64fc(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddC_64fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point signal add product of signal times constant to destination signal.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void AddProductC(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pDst, float nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddProductC.nppsAddProductC_32f(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddProductC_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 8-bit unsigned char in place signal times constant,
		/// scale, then clamp to saturated value
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void MulC(this CudaDeviceVariable<byte> pSrcDst, byte nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulC.nppsMulC_8u_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMulC_8u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 8-bit unsigned char signal times constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void MulC(this CudaDeviceVariable<byte> pSrc, CudaDeviceVariable<byte> pDst, byte nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulC.nppsMulC_8u_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMulC_8u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit unsigned short in place signal times constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void MulC(this CudaDeviceVariable<ushort> pSrcDst, ushort nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulC.nppsMulC_16u_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMulC_16u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short signal times constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void MulC(this CudaDeviceVariable<ushort> pSrc, CudaDeviceVariable<ushort> pDst, ushort nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulC.nppsMulC_16u_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMulC_16u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit signed short in place signal times constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void MulC(this CudaDeviceVariable<short> pSrcDst, short nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulC.nppsMulC_16s_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMulC_16s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short signal times constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void MulC(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pDst, short nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulC.nppsMulC_16s_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMulC_16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit integer complex number (16 bit real, 16 bit imaginary)signal times constant, 
		/// scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void MulC(this CudaDeviceVariable<Npp16sc> pSrcDst, Npp16sc nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulC.nppsMulC_16sc_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMulC_16sc_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit integer complex number (16 bit real, 16 bit imaginary)signal times constant,
		/// scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void MulC(this CudaDeviceVariable<Npp16sc> pSrc, CudaDeviceVariable<Npp16sc> pDst, Npp16sc nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulC.nppsMulC_16sc_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMulC_16sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit signed integer in place signal times constant and scale.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void MulC(this CudaDeviceVariable<int> pSrcDst, int nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulC.nppsMulC_32s_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMulC_32s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit signed integer signal times constant and scale.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void MulC(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pDst, int nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulC.nppsMulC_32s_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMulC_32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal
		/// times constant and scale.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void MulC(this CudaDeviceVariable<Npp32sc> pSrcDst, Npp32sc nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulC.nppsMulC_32sc_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMulC_32sc_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit integer complex number (32 bit real, 32 bit imaginary) signal times constant
		/// and scale.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void MulC(this CudaDeviceVariable<Npp32sc> pSrc, CudaDeviceVariable<Npp32sc> pDst, Npp32sc nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulC.nppsMulC_32sc_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMulC_32sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point in place signal times constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>

		public static void MulC(this CudaDeviceVariable<float> pSrcDst, float nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulC.nppsMulC_32f_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMulC_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point signal times constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void MulC(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pDst, float nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulC.nppsMulC_32f(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMulC_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point signal times constant with output converted to 16-bit signed integer.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void MulC_Low(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<short> pDst, float nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulC.nppsMulC_Low_32f16s(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMulC_Low_32f16s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point signal times constant with output converted to 16-bit signed integer
		/// with scaling and saturation of output result.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>

		public static void MulC(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<short> pDst, float nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulC.nppsMulC_32f16s_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMulC_32f16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
		/// place signal times constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>

		public static void MulC(this CudaDeviceVariable<Npp32fc> pSrcDst, Npp32fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulC.nppsMulC_32fc_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMulC_32fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
		/// times constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void MulC(this CudaDeviceVariable<Npp32fc> pSrc, CudaDeviceVariable<Npp32fc> pDst, Npp32fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulC.nppsMulC_32fc(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMulC_32fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit floating point, in place signal times constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>
		public static void MulC(this CudaDeviceVariable<double> pSrcDst, double nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulC.nppsMulC_64f_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMulC_64f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point signal times constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void MulC(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pDst, double nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulC.nppsMulC_64f(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMulC_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit floating point signal times constant with in place conversion to 64-bit signed integer
		/// and with scaling and saturation of output result.
		/// </summary>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void MulC(this CudaDeviceVariable<long> pDst, double nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulC.nppsMulC_64f64s_ISfs(nValue, pDst.DevicePointer, pDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMulC_64f64s_ISfs", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
		/// place signal times constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>

		public static void MulC(this CudaDeviceVariable<Npp64fc> pSrcDst, Npp64fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulC.nppsMulC_64fc_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMulC_64fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
		/// times constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be multiplied by each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void MulC(this CudaDeviceVariable<Npp64fc> pSrc, CudaDeviceVariable<Npp64fc> pDst, Npp64fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulC.nppsMulC_64fc(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMulC_64fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 8-bit unsigned char in place signal subtract constant,
		/// scale, then clamp to saturated value
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be subtracted from each vector element</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubC(this CudaDeviceVariable<byte> pSrcDst, byte nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubC.nppsSubC_8u_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubC_8u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 8-bit unsigned char signal subtract constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be subtracted from each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubC(this CudaDeviceVariable<byte> pSrc, CudaDeviceVariable<byte> pDst, byte nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubC.nppsSubC_8u_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubC_8u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit unsigned short in place signal subtract constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be subtracted from each vector element</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubC(this CudaDeviceVariable<ushort> pSrcDst, ushort nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubC.nppsSubC_16u_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubC_16u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short signal subtract constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be subtracted from each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubC(this CudaDeviceVariable<ushort> pSrc, CudaDeviceVariable<ushort> pDst, ushort nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubC.nppsSubC_16u_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubC_16u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit signed short in place signal subtract constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be subtracted from each vector element</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubC(this CudaDeviceVariable<short> pSrcDst, short nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubC.nppsSubC_16s_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubC_16s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short signal subtract constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be subtracted from each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubC(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pDst, short nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubC.nppsSubC_16s_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubC_16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract constant, 
		/// scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be subtracted from each vector element</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubC(this CudaDeviceVariable<Npp16sc> pSrcDst, Npp16sc nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubC.nppsSubC_16sc_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubC_16sc_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract constant,
		/// scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be subtracted from each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubC(this CudaDeviceVariable<Npp16sc> pSrc, CudaDeviceVariable<Npp16sc> pDst, Npp16sc nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubC.nppsSubC_16sc_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubC_16sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit signed integer in place signal subtract constant and scale.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be subtracted from each vector element</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubC(this CudaDeviceVariable<int> pSrcDst, int nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubC.nppsSubC_32s_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubC_32s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit signed integer signal subtract constant and scale.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be subtracted from each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubC(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pDst, int nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubC.nppsSubC_32s_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubC_32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal
		/// subtract constant and scale.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be subtracted from each vector element</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubC(this CudaDeviceVariable<Npp32sc> pSrcDst, Npp32sc nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubC.nppsSubC_32sc_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubC_32sc_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit integer complex number (32 bit real, 32 bit imaginary)signal subtract constant
		/// and scale.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be subtracted from each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubC(this CudaDeviceVariable<Npp32sc> pSrc, CudaDeviceVariable<Npp32sc> pDst, Npp32sc nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubC.nppsSubC_32sc_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubC_32sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point in place signal subtract constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be subtracted from each vector element</param>

		public static void SubC(this CudaDeviceVariable<float> pSrcDst, float nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubC.nppsSubC_32f_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubC_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point signal subtract constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be subtracted from each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void SubC(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pDst, float nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubC.nppsSubC_32f(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubC_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
		/// place signal subtract constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be subtracted from each vector element</param>

		public static void SubC(this CudaDeviceVariable<Npp32fc> pSrcDst, Npp32fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubC.nppsSubC_32fc_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubC_32fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
		/// subtract constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be subtracted from each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void SubC(this CudaDeviceVariable<Npp32fc> pSrc, CudaDeviceVariable<Npp32fc> pDst, Npp32fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubC.nppsSubC_32fc(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubC_32fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit floating point, in place signal subtract constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be subtracted from each vector element</param>
		public static void SubC(this CudaDeviceVariable<double> pSrcDst, double nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubC.nppsSubC_64f_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubC_64f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point signal subtract constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be subtracted from each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void SubC(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pDst, double nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubC.nppsSubC_64f(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubC_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
		/// place signal subtract constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be subtracted from each vector element</param>

		public static void SubC(this CudaDeviceVariable<Npp64fc> pSrcDst, Npp64fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubC.nppsSubC_64fc_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubC_64fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
		/// subtract constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be subtracted from each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void SubC(this CudaDeviceVariable<Npp64fc> pSrc, CudaDeviceVariable<Npp64fc> pDst, Npp64fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubC.nppsSubC_64fc(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubC_64fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 8-bit unsigned char in place signal subtract from constant,
		/// scale, then clamp to saturated value
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value each vector element is to be subtracted from</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubCRev(this CudaDeviceVariable<byte> pSrcDst, byte nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubCRev.nppsSubCRev_8u_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubCRev_8u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 8-bit unsigned char signal subtract from constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value each vector element is to be subtracted from</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubCRev(this CudaDeviceVariable<byte> pSrc, CudaDeviceVariable<byte> pDst, byte nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubCRev.nppsSubCRev_8u_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubCRev_8u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit unsigned short in place signal subtract from constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value each vector element is to be subtracted from</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubCRev(this CudaDeviceVariable<ushort> pSrcDst, ushort nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubCRev.nppsSubCRev_16u_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubCRev_16u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short signal subtract from constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value each vector element is to be subtracted from</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubCRev(this CudaDeviceVariable<ushort> pSrc, CudaDeviceVariable<ushort> pDst, ushort nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubCRev.nppsSubCRev_16u_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubCRev_16u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit signed short in place signal subtract from constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value each vector element is to be subtracted from</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubCRev(this CudaDeviceVariable<short> pSrcDst, short nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubCRev.nppsSubCRev_16s_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubCRev_16s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short signal subtract from constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value each vector element is to be subtracted from</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubCRev(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pDst, short nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubCRev.nppsSubCRev_16s_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubCRev_16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract from constant, 
		/// scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value each vector element is to be subtracted from</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubCRev(this CudaDeviceVariable<Npp16sc> pSrcDst, Npp16sc nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubCRev.nppsSubCRev_16sc_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubCRev_16sc_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract from constant,
		/// scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value each vector element is to be subtracted from</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubCRev(this CudaDeviceVariable<Npp16sc> pSrc, CudaDeviceVariable<Npp16sc> pDst, Npp16sc nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubCRev.nppsSubCRev_16sc_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubCRev_16sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit signed integer in place signal subtract from constant and scale.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value each vector element is to be subtracted from</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubCRev(this CudaDeviceVariable<int> pSrcDst, int nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubCRev.nppsSubCRev_32s_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubCRev_32s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit signed integersignal subtract from constant and scale.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value each vector element is to be subtracted from</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubCRev(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pDst, int nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubCRev.nppsSubCRev_32s_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubCRev_32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal
		/// subtract from constant and scale.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value each vector element is to be subtracted from</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubCRev(this CudaDeviceVariable<Npp32sc> pSrcDst, Npp32sc nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubCRev.nppsSubCRev_32sc_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubCRev_32sc_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit integer complex number (32 bit real, 32 bit imaginary) signal subtract from constant
		/// and scale.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value each vector element is to be subtracted from</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void SubCRev(this CudaDeviceVariable<Npp32sc> pSrc, CudaDeviceVariable<Npp32sc> pDst, Npp32sc nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubCRev.nppsSubCRev_32sc_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubCRev_32sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point in place signal subtract from constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value each vector element is to be subtracted from</param>

		public static void SubCRev(this CudaDeviceVariable<float> pSrcDst, float nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubCRev.nppsSubCRev_32f_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubCRev_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point signal subtract from constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value each vector element is to be subtracted from</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void SubCRev(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pDst, float nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubCRev.nppsSubCRev_32f(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubCRev_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
		/// place signal subtract from constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value each vector element is to be subtracted from</param>

		public static void SubCRev(this CudaDeviceVariable<Npp32fc> pSrcDst, Npp32fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubCRev.nppsSubCRev_32fc_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubCRev_32fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
		/// subtract from constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value each vector element is to be subtracted from</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void SubCRev(this CudaDeviceVariable<Npp32fc> pSrc, CudaDeviceVariable<Npp32fc> pDst, Npp32fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubCRev.nppsSubCRev_32fc(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubCRev_32fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit floating point, in place signal subtract from constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value each vector element is to be subtracted from</param>
		public static void SubCRev(this CudaDeviceVariable<double> pSrcDst, double nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubCRev.nppsSubCRev_64f_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubCRev_64f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point signal subtract from constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value each vector element is to be subtracted from</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void SubCRev(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pDst, double nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubCRev.nppsSubCRev_64f(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubCRev_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
		/// place signal subtract from constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value each vector element is to be subtracted from</param>

		public static void SubCRev(this CudaDeviceVariable<Npp64fc> pSrcDst, Npp64fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubCRev.nppsSubCRev_64fc_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubCRev_64fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
		/// subtract from constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value each vector element is to be subtracted from</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void SubCRev(this CudaDeviceVariable<Npp64fc> pSrc, CudaDeviceVariable<Npp64fc> pDst, Npp64fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubCRev.nppsSubCRev_64fc(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSubCRev_64fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 8-bit unsigned char in place signal divided by constant,
		/// scale, then clamp to saturated value
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be divided into each vector element</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void DivC(this CudaDeviceVariable<byte> pSrcDst, byte nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivC.nppsDivC_8u_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDivC_8u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 8-bit unsigned char signal divided by constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be divided into each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void DivC(this CudaDeviceVariable<byte> pSrc, CudaDeviceVariable<byte> pDst, byte nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivC.nppsDivC_8u_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDivC_8u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit unsigned short in place signal divided by constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be divided into each vector element</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void DivC(this CudaDeviceVariable<ushort> pSrcDst, ushort nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivC.nppsDivC_16u_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDivC_16u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short signal divided by constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be divided into each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void DivC(this CudaDeviceVariable<ushort> pSrc, CudaDeviceVariable<ushort> pDst, ushort nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivC.nppsDivC_16u_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDivC_16u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit signed short in place signal divided by constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be divided into each vector element</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void DivC(this CudaDeviceVariable<short> pSrcDst, short nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivC.nppsDivC_16s_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDivC_16s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short signal divided by constant, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be divided into each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void DivC(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pDst, short nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivC.nppsDivC_16s_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDivC_16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit integer complex number (16 bit real, 16 bit imaginary)signal divided by constant, 
		/// scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be divided into each vector element</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void DivC(this CudaDeviceVariable<Npp16sc> pSrcDst, Npp16sc nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivC.nppsDivC_16sc_ISfs(nValue, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDivC_16sc_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit integer complex number (16 bit real, 16 bit imaginary) signal divided by constant,
		/// scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be divided into each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void DivC(this CudaDeviceVariable<Npp16sc> pSrc, CudaDeviceVariable<Npp16sc> pDst, Npp16sc nValue, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivC.nppsDivC_16sc_Sfs(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDivC_16sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point in place signal divided by constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be divided into each vector element</param>

		public static void DivC(this CudaDeviceVariable<float> pSrcDst, float nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivC.nppsDivC_32f_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDivC_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point signal divided by constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be divided into each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void DivC(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pDst, float nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivC.nppsDivC_32f(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDivC_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point complex number (32 bit real, 32 bit imaginary) in
		/// place signal divided by constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be divided into each vector element</param>

		public static void DivC(this CudaDeviceVariable<Npp32fc> pSrcDst, Npp32fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivC.nppsDivC_32fc_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDivC_32fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal
		/// divided by constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be divided into each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void DivC(this CudaDeviceVariable<Npp32fc> pSrc, CudaDeviceVariable<Npp32fc> pDst, Npp32fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivC.nppsDivC_32fc(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDivC_32fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit floating point in place signal divided by constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be divided into each vector element</param>
		public static void DivC(this CudaDeviceVariable<double> pSrcDst, double nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivC.nppsDivC_64f_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDivC_64f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point signal divided by constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be divided into each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void DivC(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pDst, double nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivC.nppsDivC_64f(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDivC_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit floating point complex number (64 bit real, 64 bit imaginary) in
		/// place signal divided by constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be divided into each vector element</param>

		public static void DivC(this CudaDeviceVariable<Npp64fc> pSrcDst, Npp64fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivC.nppsDivC_64fc_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDivC_64fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal
		/// divided by constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be divided into each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void DivC(this CudaDeviceVariable<Npp64fc> pSrc, CudaDeviceVariable<Npp64fc> pDst, Npp64fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivC.nppsDivC_64fc(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDivC_64fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit unsigned short in place constant divided by signal, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be divided by each vector element</param>

		public static void DivCRev(this CudaDeviceVariable<ushort> pSrcDst, ushort nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivCRev.nppsDivCRev_16u_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDivCRev_16u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short signal divided by constant, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be divided by each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void DivCRev(this CudaDeviceVariable<ushort> pSrc, CudaDeviceVariable<ushort> pDst, ushort nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivCRev.nppsDivCRev_16u(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDivCRev_16u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point in place constant divided by signal.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be divided by each vector element</param>

		public static void DivCRev(this CudaDeviceVariable<float> pSrcDst, float nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivCRev.nppsDivCRev_32f_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDivCRev_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point constant divided by signal.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be divided by each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void DivCRev(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pDst, float nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivCRev.nppsDivCRev_32f(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDivCRev_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit signed short signal add signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be added to signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Add(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<short> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_16s(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_16s", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit unsigned short signal add signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be added to signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Add(this CudaDeviceVariable<ushort> pSrc1, CudaDeviceVariable<ushort> pSrc2, CudaDeviceVariable<ushort> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_16u(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_16u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit unsigned int signal add signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be added to signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Add(this CudaDeviceVariable<uint> pSrc1, CudaDeviceVariable<uint> pSrc2, CudaDeviceVariable<uint> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_32u(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_32u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit floating point signal add signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be added to signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Add(this CudaDeviceVariable<float> pSrc1, CudaDeviceVariable<float> pSrc2, CudaDeviceVariable<float> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_32f(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_32f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit floating point signal add signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be added to signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Add(this CudaDeviceVariable<double> pSrc1, CudaDeviceVariable<double> pSrc2, CudaDeviceVariable<double> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_64f(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_64f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit complex floating point signal add signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be added to signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Add(this CudaDeviceVariable<Npp32fc> pSrc1, CudaDeviceVariable<Npp32fc> pSrc2, CudaDeviceVariable<Npp32fc> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_32fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_32fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit complex floating point signal add signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be added to signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Add(this CudaDeviceVariable<Npp64fc> pSrc1, CudaDeviceVariable<Npp64fc> pSrc2, CudaDeviceVariable<Npp64fc> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_64fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_64fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 8-bit unsigned char signal add signal with 16-bit unsigned result,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be added to signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Add(this CudaDeviceVariable<byte> pSrc1, CudaDeviceVariable<byte> pSrc2, CudaDeviceVariable<ushort> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_8u16u(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_8u16u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit signed short signal add signal with 32-bit floating point result,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be added to signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Add(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<float> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_16s32f(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_16s32f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 8-bit unsigned char add signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal2 elements to be added to signal1 elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Add(this CudaDeviceVariable<byte> pSrc1, CudaDeviceVariable<byte> pSrc2, CudaDeviceVariable<byte> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_8u_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_8u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit unsigned short add signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal2 elements to be added to signal1 elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Add(this CudaDeviceVariable<ushort> pSrc1, CudaDeviceVariable<ushort> pSrc2, CudaDeviceVariable<ushort> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_16u_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_16u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit signed short add signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal2 elements to be added to signal1 elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Add(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<short> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_16s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit signed integer add signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal2 elements to be added to signal1 elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Add(this CudaDeviceVariable<int> pSrc1, CudaDeviceVariable<int> pSrc2, CudaDeviceVariable<int> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_32s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit signed integer add signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal2 elements to be added to signal1 elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Add(this CudaDeviceVariable<long> pSrc1, CudaDeviceVariable<long> pSrc2, CudaDeviceVariable<long> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_64s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_64s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit signed complex short add signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal2 elements to be added to signal1 elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Add(this CudaDeviceVariable<Npp16sc> pSrc1, CudaDeviceVariable<Npp16sc> pSrc2, CudaDeviceVariable<Npp16sc> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_16sc_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_16sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit signed complex integer add signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal2 elements to be added to signal1 elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Add(this CudaDeviceVariable<Npp32sc> pSrc1, CudaDeviceVariable<Npp32sc> pSrc2, CudaDeviceVariable<Npp32sc> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_32sc_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_32sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit signed short in place signal add signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be added to signal1 elements</param>

		public static void Add(this CudaDeviceVariable<short> pSrcDst, CudaDeviceVariable<short> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_16s_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_16s_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point in place signal add signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be added to signal1 elements</param>

		public static void Add(this CudaDeviceVariable<float> pSrcDst, CudaDeviceVariable<float> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_32f_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point in place signal add signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be added to signal1 elements</param>

		public static void Add(this CudaDeviceVariable<double> pSrcDst, CudaDeviceVariable<double> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_64f_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_64f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit complex floating point in place signal add signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be added to signal1 elements</param>

		public static void Add(this CudaDeviceVariable<Npp32fc> pSrcDst, CudaDeviceVariable<Npp32fc> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_32fc_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_32fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit complex floating point in place signal add signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be added to signal1 elements</param>

		public static void Add(this CudaDeviceVariable<Npp64fc> pSrcDst, CudaDeviceVariable<Npp64fc> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_64fc_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_64fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16/32-bit signed short in place signal add signal with 32-bit signed integer results,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be added to signal1 elements</param>

		public static void Add(this CudaDeviceVariable<int> pSrcDst, CudaDeviceVariable<short> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_16s32s_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_16s32s_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 8-bit unsigned char in place signal add signal, with scaling,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be added to signal1 elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Add(this CudaDeviceVariable<byte> pSrcDst, CudaDeviceVariable<byte> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_8u_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_8u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short in place signal add signal, with scaling,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be added to signal1 elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Add(this CudaDeviceVariable<ushort> pSrcDst, CudaDeviceVariable<ushort> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_16u_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_16u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short in place signal add signal, with scaling, 
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be added to signal1 elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Add(this CudaDeviceVariable<short> pSrcDst, CudaDeviceVariable<short> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_16s_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_16s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit signed integer in place signal add signal, with scaling, 
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be added to signal1 elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Add(this CudaDeviceVariable<int> pSrcDst, CudaDeviceVariable<int> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_32s_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_32s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit complex signed short in place signal add signal, with scaling, 
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be added to signal1 elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Add(this CudaDeviceVariable<Npp16sc> pSrcDst, CudaDeviceVariable<Npp16sc> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_16sc_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_16sc_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit complex signed integer in place signal add signal, with scaling, 
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be added to signal1 elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Add(this CudaDeviceVariable<Npp32sc> pSrcDst, CudaDeviceVariable<Npp32sc> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddSignal.nppsAdd_32sc_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAdd_32sc_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point signal add product of source signal times destination signal to destination signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer. product of source1 and source2 signal elements to be added to destination elements</param>

		public static void AddProduct(this CudaDeviceVariable<float> pSrc1, CudaDeviceVariable<float> pSrc2, CudaDeviceVariable<float> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddProductSignal.nppsAddProduct_32f(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddProduct_32f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit floating point signal add product of source signal times destination signal to destination signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer. product of source1 and source2 signal elements to be added to destination elements</param>

		public static void AddProduct(this CudaDeviceVariable<double> pSrc1, CudaDeviceVariable<double> pSrc2, CudaDeviceVariable<double> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddProductSignal.nppsAddProduct_64f(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddProduct_64f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit complex floating point signal add product of source signal times destination signal to destination signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer. product of source1 and source2 signal elements to be added to destination elements</param>

		public static void AddProduct(this CudaDeviceVariable<Npp32fc> pSrc1, CudaDeviceVariable<Npp32fc> pSrc2, CudaDeviceVariable<Npp32fc> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddProductSignal.nppsAddProduct_32fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddProduct_32fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit complex floating point signal add product of source signal times destination signal to destination signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer. product of source1 and source2 signal elements to be added to destination elements</param>

		public static void AddProduct(this CudaDeviceVariable<Npp64fc> pSrc1, CudaDeviceVariable<Npp64fc> pSrc2, CudaDeviceVariable<Npp64fc> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddProductSignal.nppsAddProduct_64fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddProduct_64fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit signed short signal add product of source signal1 times source signal2 to destination signal,
		/// with scaling, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer. product of source1 and source2 signal elements to be added to destination elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void AddProduct(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<short> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddProductSignal.nppsAddProduct_16s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddProduct_16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit signed short signal add product of source signal1 times source signal2 to destination signal,
		/// with scaling, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer. product of source1 and source2 signal elements to be added to destination elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void AddProduct(this CudaDeviceVariable<int> pSrc1, CudaDeviceVariable<int> pSrc2, CudaDeviceVariable<int> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddProductSignal.nppsAddProduct_32s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddProduct_32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit signed short signal add product of source signal1 times source signal2 to 32-bit signed integer destination signal,
		/// with scaling, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer. product of source1 and source2 signal elements to be added to destination elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void AddProduct(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<int> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.AddProductSignal.nppsAddProduct_16s32s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAddProduct_16s32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit signed short signal times signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be multiplied by signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Mul(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<short> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_16s(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_16s", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit floating point signal times signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be multiplied by signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Mul(this CudaDeviceVariable<float> pSrc1, CudaDeviceVariable<float> pSrc2, CudaDeviceVariable<float> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_32f(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_32f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit floating point signal times signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be multiplied by signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Mul(this CudaDeviceVariable<double> pSrc1, CudaDeviceVariable<double> pSrc2, CudaDeviceVariable<double> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_64f(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_64f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit complex floating point signal times signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be multiplied by signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Mul(this CudaDeviceVariable<Npp32fc> pSrc1, CudaDeviceVariable<Npp32fc> pSrc2, CudaDeviceVariable<Npp32fc> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_32fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_32fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit complex floating point signal times signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be multiplied by signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Mul(this CudaDeviceVariable<Npp64fc> pSrc1, CudaDeviceVariable<Npp64fc> pSrc2, CudaDeviceVariable<Npp64fc> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_64fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_64fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 8-bit unsigned char signal times signal with 16-bit unsigned result,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be multiplied by signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Mul(this CudaDeviceVariable<byte> pSrc1, CudaDeviceVariable<byte> pSrc2, CudaDeviceVariable<ushort> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_8u16u(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_8u16u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit signed short signal times signal with 32-bit floating point result,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be multiplied by signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Mul(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<float> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_16s32f(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_16s32f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit floating point signal times 32-bit complex floating point signal with complex 32-bit floating point result,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be multiplied by signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Mul(this CudaDeviceVariable<float> pSrc1, CudaDeviceVariable<Npp32fc> pSrc2, CudaDeviceVariable<Npp32fc> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_32f32fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_32f32fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 8-bit unsigned char signal times signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal2 elements to be multiplied by signal1 elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Mul(this CudaDeviceVariable<byte> pSrc1, CudaDeviceVariable<byte> pSrc2, CudaDeviceVariable<byte> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_8u_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_8u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit unsigned short signal time signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal2 elements to be multiplied by signal1 elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Mul(this CudaDeviceVariable<ushort> pSrc1, CudaDeviceVariable<ushort> pSrc2, CudaDeviceVariable<ushort> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_16u_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_16u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit signed short signal times signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal2 elements to be multiplied by signal1 elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Mul(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<short> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_16s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit signed integer signal times signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal2 elements to be multiplied by signal1 elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Mul(this CudaDeviceVariable<int> pSrc1, CudaDeviceVariable<int> pSrc2, CudaDeviceVariable<int> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_32s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit signed complex short signal times signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal2 elements to be multiplied by signal1 elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Mul(this CudaDeviceVariable<Npp16sc> pSrc1, CudaDeviceVariable<Npp16sc> pSrc2, CudaDeviceVariable<Npp16sc> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_16sc_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_16sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit signed complex integer signal times signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal2 elements to be multiplied by signal1 elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Mul(this CudaDeviceVariable<Npp32sc> pSrc1, CudaDeviceVariable<Npp32sc> pSrc2, CudaDeviceVariable<Npp32sc> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_32sc_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_32sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit unsigned short signal times 16-bit signed short signal, scale, then clamp to 16-bit signed saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal2 elements to be multiplied by signal1 elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Mul(this CudaDeviceVariable<ushort> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<short> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_16u16s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_16u16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit signed short signal times signal, scale, then clamp to 32-bit signed saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal2 elements to be multiplied by signal1 elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Mul(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<int> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_16s32s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_16s32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit signed integer signal times 32-bit complex signed integer signal, scale, then clamp to 32-bit complex integer saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal2 elements to be multiplied by signal1 elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Mul(this CudaDeviceVariable<int> pSrc1, CudaDeviceVariable<Npp32sc> pSrc2, CudaDeviceVariable<Npp32sc> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_32s32sc_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_32s32sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit signed integer signal times signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal2 elements to be multiplied by signal1 elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Mul_Low(this CudaDeviceVariable<int> pSrc1, CudaDeviceVariable<int> pSrc2, CudaDeviceVariable<int> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_Low_32s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_Low_32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit signed short in place signal times signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be multiplied by signal1 elements</param>

		public static void Mul(this CudaDeviceVariable<short> pSrcDst, CudaDeviceVariable<short> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_16s_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_16s_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point in place signal times signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be multiplied by signal1 elements</param>

		public static void Mul(this CudaDeviceVariable<float> pSrcDst, CudaDeviceVariable<float> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_32f_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point in place signal times signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be multiplied by signal1 elements</param>

		public static void Mul(this CudaDeviceVariable<double> pSrcDst, CudaDeviceVariable<double> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_64f_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_64f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit complex floating point in place signal times signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be multiplied by signal1 elements</param>

		public static void Mul(this CudaDeviceVariable<Npp32fc> pSrcDst, CudaDeviceVariable<Npp32fc> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_32fc_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_32fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit complex floating point in place signal times signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be multiplied by signal1 elements</param>

		public static void Mul(this CudaDeviceVariable<Npp64fc> pSrcDst, CudaDeviceVariable<Npp64fc> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_64fc_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_64fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit complex floating point in place signal times 32-bit floating point signal,
		/// then clamp to 32-bit complex floating point saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be multiplied by signal1 elements</param>

		public static void Mul(this CudaDeviceVariable<Npp32fc> pSrcDst, CudaDeviceVariable<float> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_32f32fc_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_32f32fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 8-bit unsigned char in place signal times signal, with scaling,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be multiplied by signal1 elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Mul(this CudaDeviceVariable<byte> pSrcDst, CudaDeviceVariable<byte> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_8u_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_8u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short in place signal times signal, with scaling,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be multiplied by signal1 elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Mul(this CudaDeviceVariable<ushort> pSrcDst, CudaDeviceVariable<ushort> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_16u_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_16u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short in place signal times signal, with scaling, 
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be multiplied by signal1 elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Mul(this CudaDeviceVariable<short> pSrcDst, CudaDeviceVariable<short> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_16s_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_16s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit signed integer in place signal times signal, with scaling, 
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be multiplied by signal1 elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Mul(this CudaDeviceVariable<int> pSrcDst, CudaDeviceVariable<int> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_32s_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_32s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit complex signed short in place signal times signal, with scaling, 
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be multiplied by signal1 elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Mul(this CudaDeviceVariable<Npp16sc> pSrcDst, CudaDeviceVariable<Npp16sc> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_16sc_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_16sc_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit complex signed integer in place signal times signal, with scaling, 
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be multiplied by signal1 elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Mul(this CudaDeviceVariable<Npp32sc> pSrcDst, CudaDeviceVariable<Npp32sc> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_32sc_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_32sc_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit complex signed integer in place signal times 32-bit signed integer signal, with scaling, 
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be multiplied by signal1 elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Mul(this CudaDeviceVariable<Npp32sc> pSrcDst, CudaDeviceVariable<int> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MulSignal.nppsMul_32s32sc_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMul_32s32sc_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short signal subtract signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal1 elements to be subtracted from signal2 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Sub(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<short> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubSignal.nppsSub_16s(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSub_16s", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit floating point signal subtract signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal1 elements to be subtracted from signal2 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Sub(this CudaDeviceVariable<float> pSrc1, CudaDeviceVariable<float> pSrc2, CudaDeviceVariable<float> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubSignal.nppsSub_32f(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSub_32f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit floating point signal subtract signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal1 elements to be subtracted from signal2 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Sub(this CudaDeviceVariable<double> pSrc1, CudaDeviceVariable<double> pSrc2, CudaDeviceVariable<double> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubSignal.nppsSub_64f(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSub_64f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit complex floating point signal subtract signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal1 elements to be subtracted from signal2 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Sub(this CudaDeviceVariable<Npp32fc> pSrc1, CudaDeviceVariable<Npp32fc> pSrc2, CudaDeviceVariable<Npp32fc> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubSignal.nppsSub_32fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSub_32fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit complex floating point signal subtract signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal1 elements to be subtracted from signal2 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Sub(this CudaDeviceVariable<Npp64fc> pSrc1, CudaDeviceVariable<Npp64fc> pSrc2, CudaDeviceVariable<Npp64fc> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubSignal.nppsSub_64fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSub_64fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit signed short signal subtract 16-bit signed short signal,
		/// then clamp and convert to 32-bit floating point saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal1 elements to be subtracted from signal2 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Sub(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<float> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubSignal.nppsSub_16s32f(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSub_16s32f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 8-bit unsigned char signal subtract signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal1 elements to be subtracted from signal2 elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sub(this CudaDeviceVariable<byte> pSrc1, CudaDeviceVariable<byte> pSrc2, CudaDeviceVariable<byte> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubSignal.nppsSub_8u_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSub_8u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit unsigned short signal subtract signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal1 elements to be subtracted from signal2 elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sub(this CudaDeviceVariable<ushort> pSrc1, CudaDeviceVariable<ushort> pSrc2, CudaDeviceVariable<ushort> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubSignal.nppsSub_16u_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSub_16u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit signed short signal subtract signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal1 elements to be subtracted from signal2 elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sub(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<short> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubSignal.nppsSub_16s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSub_16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit signed integer signal subtract signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal1 elements to be subtracted from signal2 elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sub(this CudaDeviceVariable<int> pSrc1, CudaDeviceVariable<int> pSrc2, CudaDeviceVariable<int> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubSignal.nppsSub_32s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSub_32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit signed complex short signal subtract signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal1 elements to be subtracted from signal2 elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sub(this CudaDeviceVariable<Npp16sc> pSrc1, CudaDeviceVariable<Npp16sc> pSrc2, CudaDeviceVariable<Npp16sc> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubSignal.nppsSub_16sc_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSub_16sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit signed complex integer signal subtract signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal1 elements to be subtracted from signal2 elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sub(this CudaDeviceVariable<Npp32sc> pSrc1, CudaDeviceVariable<Npp32sc> pSrc2, CudaDeviceVariable<Npp32sc> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubSignal.nppsSub_32sc_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSub_32sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit signed short in place signal subtract signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal1 elements to be subtracted from signal2 elements</param>

		public static void Sub(this CudaDeviceVariable<short> pSrcDst, CudaDeviceVariable<short> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubSignal.nppsSub_16s_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSub_16s_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point in place signal subtract signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal1 elements to be subtracted from signal2 elements</param>

		public static void Sub(this CudaDeviceVariable<float> pSrcDst, CudaDeviceVariable<float> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubSignal.nppsSub_32f_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSub_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point in place signal subtract signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal1 elements to be subtracted from signal2 elements</param>

		public static void Sub(this CudaDeviceVariable<double> pSrcDst, CudaDeviceVariable<double> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubSignal.nppsSub_64f_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSub_64f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit complex floating point in place signal subtract signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal1 elements to be subtracted from signal2 elements</param>

		public static void Sub(this CudaDeviceVariable<Npp32fc> pSrcDst, CudaDeviceVariable<Npp32fc> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubSignal.nppsSub_32fc_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSub_32fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit complex floating point in place signal subtract signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal1 elements to be subtracted from signal2 elements</param>

		public static void Sub(this CudaDeviceVariable<Npp64fc> pSrcDst, CudaDeviceVariable<Npp64fc> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubSignal.nppsSub_64fc_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSub_64fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 8-bit unsigned char in place signal subtract signal, with scaling,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal1 elements to be subtracted from signal2 elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sub(this CudaDeviceVariable<byte> pSrcDst, CudaDeviceVariable<byte> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubSignal.nppsSub_8u_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSub_8u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short in place signal subtract signal, with scaling,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal1 elements to be subtracted from signal2 elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sub(this CudaDeviceVariable<ushort> pSrcDst, CudaDeviceVariable<ushort> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubSignal.nppsSub_16u_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSub_16u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short in place signal subtract signal, with scaling, 
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal1 elements to be subtracted from signal2 elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sub(this CudaDeviceVariable<short> pSrcDst, CudaDeviceVariable<short> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubSignal.nppsSub_16s_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSub_16s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit signed integer in place signal subtract signal, with scaling, 
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal1 elements to be subtracted from signal2 elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sub(this CudaDeviceVariable<int> pSrcDst, CudaDeviceVariable<int> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubSignal.nppsSub_32s_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSub_32s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit complex signed short in place signal subtract signal, with scaling, 
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal1 elements to be subtracted from signal2 elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sub(this CudaDeviceVariable<Npp16sc> pSrcDst, CudaDeviceVariable<Npp16sc> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubSignal.nppsSub_16sc_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSub_16sc_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit complex signed integer in place signal subtract signal, with scaling, 
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal1 elements to be subtracted from signal2 elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sub(this CudaDeviceVariable<Npp32sc> pSrcDst, CudaDeviceVariable<Npp32sc> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SubSignal.nppsSub_32sc_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSub_32sc_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 8-bit unsigned char signal divide signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Div(this CudaDeviceVariable<byte> pSrc1, CudaDeviceVariable<byte> pSrc2, CudaDeviceVariable<byte> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivSignal.nppsDiv_8u_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_8u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit unsigned short signal divide signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Div(this CudaDeviceVariable<ushort> pSrc1, CudaDeviceVariable<ushort> pSrc2, CudaDeviceVariable<ushort> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivSignal.nppsDiv_16u_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_16u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit signed short signal divide signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Div(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<short> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivSignal.nppsDiv_16s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit signed integer signal divide signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Div(this CudaDeviceVariable<int> pSrc1, CudaDeviceVariable<int> pSrc2, CudaDeviceVariable<int> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivSignal.nppsDiv_32s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit signed complex short signal divide signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Div(this CudaDeviceVariable<Npp16sc> pSrc1, CudaDeviceVariable<Npp16sc> pSrc2, CudaDeviceVariable<Npp16sc> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivSignal.nppsDiv_16sc_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_16sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit signed integer signal divided by 16-bit signed short signal, scale, then clamp to 16-bit signed short saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Div(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<int> pSrc2, CudaDeviceVariable<short> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivSignal.nppsDiv_32s16s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_32s16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit floating point signal divide signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Div(this CudaDeviceVariable<float> pSrc1, CudaDeviceVariable<float> pSrc2, CudaDeviceVariable<float> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivSignal.nppsDiv_32f(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_32f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit floating point signal divide signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Div(this CudaDeviceVariable<double> pSrc1, CudaDeviceVariable<double> pSrc2, CudaDeviceVariable<double> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivSignal.nppsDiv_64f(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_64f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit complex floating point signal divide signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Div(this CudaDeviceVariable<Npp32fc> pSrc1, CudaDeviceVariable<Npp32fc> pSrc2, CudaDeviceVariable<Npp32fc> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivSignal.nppsDiv_32fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_32fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit complex floating point signal divide signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Div(this CudaDeviceVariable<Npp64fc> pSrc1, CudaDeviceVariable<Npp64fc> pSrc2, CudaDeviceVariable<Npp64fc> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivSignal.nppsDiv_64fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_64fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 8-bit unsigned char in place signal divide signal, with scaling,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Div(this CudaDeviceVariable<byte> pSrcDst, CudaDeviceVariable<byte> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivSignal.nppsDiv_8u_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_8u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short in place signal divide signal, with scaling,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Div(this CudaDeviceVariable<ushort> pSrcDst, CudaDeviceVariable<ushort> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivSignal.nppsDiv_16u_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_16u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short in place signal divide signal, with scaling, 
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Div(this CudaDeviceVariable<short> pSrcDst, CudaDeviceVariable<short> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivSignal.nppsDiv_16s_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_16s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit complex signed short in place signal divide signal, with scaling, 
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Div(this CudaDeviceVariable<Npp16sc> pSrcDst, CudaDeviceVariable<Npp16sc> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivSignal.nppsDiv_16sc_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_16sc_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit signed integer in place signal divide signal, with scaling, 
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Div(this CudaDeviceVariable<int> pSrcDst, CudaDeviceVariable<int> pSrc, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivSignal.nppsDiv_32s_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_32s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point in place signal divide signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>

		public static void Div(this CudaDeviceVariable<float> pSrcDst, CudaDeviceVariable<float> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivSignal.nppsDiv_32f_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point in place signal divide signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>

		public static void Div(this CudaDeviceVariable<double> pSrcDst, CudaDeviceVariable<double> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivSignal.nppsDiv_64f_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_64f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit complex floating point in place signal divide signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>

		public static void Div(this CudaDeviceVariable<Npp32fc> pSrcDst, CudaDeviceVariable<Npp32fc> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivSignal.nppsDiv_32fc_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_32fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit complex floating point in place signal divide signal,
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>

		public static void Div(this CudaDeviceVariable<Npp64fc> pSrcDst, CudaDeviceVariable<Npp64fc> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivSignal.nppsDiv_64fc_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_64fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 8-bit unsigned char signal divide signal, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nRndMode">various rounding modes.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Div_Round(this CudaDeviceVariable<byte> pSrc1, CudaDeviceVariable<byte> pSrc2, CudaDeviceVariable<byte> pDst, NppRoundMode nRndMode, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivRoundSignal.nppsDiv_Round_8u_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nRndMode, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_Round_8u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit unsigned short signal divide signal, scale, round, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nRndMode">various rounding modes.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Div_Round(this CudaDeviceVariable<ushort> pSrc1, CudaDeviceVariable<ushort> pSrc2, CudaDeviceVariable<ushort> pDst, NppRoundMode nRndMode, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivRoundSignal.nppsDiv_Round_16u_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nRndMode, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_Round_16u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit signed short signal divide signal, scale, round, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nRndMode">various rounding modes.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Div_Round(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<short> pDst, NppRoundMode nRndMode, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivRoundSignal.nppsDiv_Round_16s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size, nRndMode, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_Round_16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 8-bit unsigned char in place signal divide signal, with scaling, rounding
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>
		/// <param name="nRndMode">various rounding modes.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Div_Round(this CudaDeviceVariable<byte> pSrcDst, CudaDeviceVariable<byte> pSrc, NppRoundMode nRndMode, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivRoundSignal.nppsDiv_Round_8u_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nRndMode, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_Round_8u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short in place signal divide signal, with scaling, rounding
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>
		/// <param name="nRndMode">various rounding modes.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Div_Round(this CudaDeviceVariable<ushort> pSrcDst, CudaDeviceVariable<ushort> pSrc, NppRoundMode nRndMode, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivRoundSignal.nppsDiv_Round_16u_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nRndMode, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_Round_16u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short in place signal divide signal, with scaling, rounding
		/// then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>
		/// <param name="nRndMode">various rounding modes.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Div_Round(this CudaDeviceVariable<short> pSrcDst, CudaDeviceVariable<short> pSrc, NppRoundMode nRndMode, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DivRoundSignal.nppsDiv_Round_16s_ISfs(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size, nRndMode, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDiv_Round_16s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short signal absolute value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Abs(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.AbsoluteValueSignal.nppsAbs_16s(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAbs_16s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit signed integer signal absolute value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Abs(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.AbsoluteValueSignal.nppsAbs_32s(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAbs_32s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point signal absolute value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Abs(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.AbsoluteValueSignal.nppsAbs_32f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAbs_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit floating point signal absolute value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Abs(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.AbsoluteValueSignal.nppsAbs_64f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAbs_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit signed short signal absolute value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>

		public static void Abs(this CudaDeviceVariable<short> pSrcDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.AbsoluteValueSignal.nppsAbs_16s_I(pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAbs_16s_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit signed integer signal absolute value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>

		public static void Abs(this CudaDeviceVariable<int> pSrcDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.AbsoluteValueSignal.nppsAbs_32s_I(pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAbs_32s_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point signal absolute value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>

		public static void Abs(this CudaDeviceVariable<float> pSrcDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.AbsoluteValueSignal.nppsAbs_32f_I(pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAbs_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point signal absolute value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>

		public static void Abs(this CudaDeviceVariable<double> pSrcDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.AbsoluteValueSignal.nppsAbs_64f_I(pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAbs_64f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point signal squared.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Sqr(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareSignal.nppsSqr_32f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqr_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit floating point signal squared.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Sqr(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareSignal.nppsSqr_64f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqr_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit complex floating point signal squared.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Sqr(this CudaDeviceVariable<Npp32fc> pSrc, CudaDeviceVariable<Npp32fc> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareSignal.nppsSqr_32fc(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqr_32fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit complex floating point signal squared.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Sqr(this CudaDeviceVariable<Npp64fc> pSrc, CudaDeviceVariable<Npp64fc> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareSignal.nppsSqr_64fc(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqr_64fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point signal squared.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>

		public static void Sqr(this CudaDeviceVariable<float> pSrcDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareSignal.nppsSqr_32f_I(pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqr_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point signal squared.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>

		public static void Sqr(this CudaDeviceVariable<double> pSrcDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareSignal.nppsSqr_64f_I(pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqr_64f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit complex floating point signal squared.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>

		public static void Sqr(this CudaDeviceVariable<Npp32fc> pSrcDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareSignal.nppsSqr_32fc_I(pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqr_32fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit complex floating point signal squared.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>

		public static void Sqr(this CudaDeviceVariable<Npp64fc> pSrcDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareSignal.nppsSqr_64fc_I(pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqr_64fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 8-bit unsigned char signal squared, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sqr(this CudaDeviceVariable<byte> pSrc, CudaDeviceVariable<byte> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareSignal.nppsSqr_8u_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqr_8u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit unsigned short signal squared, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sqr(this CudaDeviceVariable<ushort> pSrc, CudaDeviceVariable<ushort> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareSignal.nppsSqr_16u_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqr_16u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit signed short signal squared, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sqr(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareSignal.nppsSqr_16s_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqr_16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit complex signed short signal squared, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sqr(this CudaDeviceVariable<Npp16sc> pSrc, CudaDeviceVariable<Npp16sc> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareSignal.nppsSqr_16sc_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqr_16sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 8-bit unsigned char signal squared, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sqr(this CudaDeviceVariable<byte> pSrcDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareSignal.nppsSqr_8u_ISfs(pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqr_8u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short signal squared, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sqr(this CudaDeviceVariable<ushort> pSrcDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareSignal.nppsSqr_16u_ISfs(pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqr_16u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short signal squared, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sqr(this CudaDeviceVariable<short> pSrcDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareSignal.nppsSqr_16s_ISfs(pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqr_16s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit complex signed short signal squared, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sqr(this CudaDeviceVariable<Npp16sc> pSrcDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareSignal.nppsSqr_16sc_ISfs(pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqr_16sc_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point signal square root.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Sqrt(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareRootSignal.nppsSqrt_32f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqrt_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit floating point signal square root.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Sqrt(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareRootSignal.nppsSqrt_64f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqrt_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit complex floating point signal square root.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Sqrt(this CudaDeviceVariable<Npp32fc> pSrc, CudaDeviceVariable<Npp32fc> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareRootSignal.nppsSqrt_32fc(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqrt_32fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit complex floating point signal square root.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Sqrt(this CudaDeviceVariable<Npp64fc> pSrc, CudaDeviceVariable<Npp64fc> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareRootSignal.nppsSqrt_64fc(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqrt_64fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point signal square root.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>

		public static void Sqrt(this CudaDeviceVariable<float> pSrcDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareRootSignal.nppsSqrt_32f_I(pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqrt_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point signal square root.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>

		public static void Sqrt(this CudaDeviceVariable<double> pSrcDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareRootSignal.nppsSqrt_64f_I(pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqrt_64f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit complex floating point signal square root.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>

		public static void Sqrt(this CudaDeviceVariable<Npp32fc> pSrcDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareRootSignal.nppsSqrt_32fc_I(pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqrt_32fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit complex floating point signal square root.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>

		public static void Sqrt(this CudaDeviceVariable<Npp64fc> pSrcDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareRootSignal.nppsSqrt_64fc_I(pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqrt_64fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 8-bit unsigned char signal square root, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sqrt(this CudaDeviceVariable<byte> pSrc, CudaDeviceVariable<byte> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareRootSignal.nppsSqrt_8u_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqrt_8u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit unsigned short signal square root, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sqrt(this CudaDeviceVariable<ushort> pSrc, CudaDeviceVariable<ushort> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareRootSignal.nppsSqrt_16u_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqrt_16u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit signed short signal square root, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sqrt(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareRootSignal.nppsSqrt_16s_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqrt_16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit complex signed short signal square root, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sqrt(this CudaDeviceVariable<Npp16sc> pSrc, CudaDeviceVariable<Npp16sc> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareRootSignal.nppsSqrt_16sc_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqrt_16sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit signed integer signal square root, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sqrt(this CudaDeviceVariable<long> pSrc, CudaDeviceVariable<long> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareRootSignal.nppsSqrt_64s_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqrt_64s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit signed integer signal square root, scale, then clamp to 16-bit signed integer saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sqrt(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<short> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareRootSignal.nppsSqrt_32s16s_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqrt_32s16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit signed integer signal square root, scale, then clamp to 16-bit signed integer saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sqrt(this CudaDeviceVariable<long> pSrc, CudaDeviceVariable<short> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareRootSignal.nppsSqrt_64s16s_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqrt_64s16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 8-bit unsigned char signal square root, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sqrt(this CudaDeviceVariable<byte> pSrcDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareRootSignal.nppsSqrt_8u_ISfs(pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqrt_8u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short signal square root, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sqrt(this CudaDeviceVariable<ushort> pSrcDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareRootSignal.nppsSqrt_16u_ISfs(pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqrt_16u_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short signal square root, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sqrt(this CudaDeviceVariable<short> pSrcDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareRootSignal.nppsSqrt_16s_ISfs(pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqrt_16s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit complex signed short signal square root, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sqrt(this CudaDeviceVariable<Npp16sc> pSrcDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareRootSignal.nppsSqrt_16sc_ISfs(pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqrt_16sc_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit signed integer signal square root, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sqrt(this CudaDeviceVariable<long> pSrcDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.SquareRootSignal.nppsSqrt_64s_ISfs(pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSqrt_64s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point signal cube root.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Cubrt(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.CubeRootSignal.nppsCubrt_32f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsCubrt_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit signed integer signal cube root, scale, then clamp to 16-bit signed integer saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Cubrt(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<short> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.CubeRootSignal.nppsCubrt_32s16s_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsCubrt_32s16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point signal exponent.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Exp(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.ExponentSignal.nppsExp_32f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsExp_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit floating point signal exponent.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Exp(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.ExponentSignal.nppsExp_64f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsExp_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point signal exponent with 64-bit floating point result.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Exp(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<double> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.ExponentSignal.nppsExp_32f64f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsExp_32f64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point signal exponent.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>

		public static void Exp(this CudaDeviceVariable<float> pSrcDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.ExponentSignal.nppsExp_32f_I(pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsExp_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point signal exponent.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>

		public static void Exp(this CudaDeviceVariable<double> pSrcDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.ExponentSignal.nppsExp_64f_I(pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsExp_64f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short signal exponent, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Exp(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.ExponentSignal.nppsExp_16s_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsExp_16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit signed integer signal exponent, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Exp(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.ExponentSignal.nppsExp_32s_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsExp_32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit signed integer signal exponent, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Exp(this CudaDeviceVariable<long> pSrc, CudaDeviceVariable<long> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.ExponentSignal.nppsExp_64s_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsExp_64s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit signed short signal exponent, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Exp(this CudaDeviceVariable<short> pSrcDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.ExponentSignal.nppsExp_16s_ISfs(pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsExp_16s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit signed integer signal exponent, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Exp(this CudaDeviceVariable<int> pSrcDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.ExponentSignal.nppsExp_32s_ISfs(pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsExp_32s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit signed integer signal exponent, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Exp(this CudaDeviceVariable<long> pSrcDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.ExponentSignal.nppsExp_64s_ISfs(pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsExp_64s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point signal natural logarithm.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Ln(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.NaturalLogarithmSignal.nppsLn_32f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsLn_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit floating point signal natural logarithm.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Ln(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.NaturalLogarithmSignal.nppsLn_64f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsLn_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit floating point signal natural logarithm with 32-bit floating point result.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Ln(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<float> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.NaturalLogarithmSignal.nppsLn_64f32f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsLn_64f32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point signal natural logarithm.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>

		public static void Ln(this CudaDeviceVariable<float> pSrcDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.NaturalLogarithmSignal.nppsLn_32f_I(pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsLn_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point signal natural logarithm.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>

		public static void Ln(this CudaDeviceVariable<double> pSrcDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.NaturalLogarithmSignal.nppsLn_64f_I(pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsLn_64f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short signal natural logarithm, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Ln(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.NaturalLogarithmSignal.nppsLn_16s_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsLn_16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit signed integer signal natural logarithm, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Ln(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.NaturalLogarithmSignal.nppsLn_32s_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsLn_32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit signed integer signal natural logarithm, scale, then clamp to 16-bit signed short saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Ln(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<short> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.NaturalLogarithmSignal.nppsLn_32s16s_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsLn_32s16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit signed short signal natural logarithm, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Ln(this CudaDeviceVariable<short> pSrcDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.NaturalLogarithmSignal.nppsLn_16s_ISfs(pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsLn_16s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit signed integer signal natural logarithm, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Ln(this CudaDeviceVariable<int> pSrcDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.NaturalLogarithmSignal.nppsLn_32s_ISfs(pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsLn_32s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit signed integer signal 10 times base 10 logarithm, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void TenTimesLog10(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.TenTimesBaseTenLogarithmSignal.npps10Log10_32s_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "npps10Log10_32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit signed integer signal 10 times base 10 logarithm, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void TenTimesLog10(this CudaDeviceVariable<int> pSrcDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.TenTimesBaseTenLogarithmSignal.npps10Log10_32s_ISfs(pSrcDst.DevicePointer, pSrcDst.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "npps10Log10_32s_ISfs", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for 32f SumLn.
		/// This primitive provides the correct buffer size for nppsSumLn_32f.
		/// </summary>
		public static int SumLnGetBufferSize(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.SumLn.nppsSumLnGetBufferSize_32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSumLnGetBufferSize_32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit floating point signal sum natural logarithm.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation.</param>
		public static void SumLn(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.SumLn.nppsSumLn_32f(pSrc.DevicePointer, pSrc.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSumLn_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for 64f SumLn.
		/// This primitive provides the correct buffer size for nppsSumLn_64f.
		/// </summary>
		public static int SumLnGetBufferSize(this CudaDeviceVariable<double> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.SumLn.nppsSumLnGetBufferSize_64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSumLnGetBufferSize_64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 64-bit floating point signal sum natural logarithm.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation.</param>
		public static void SumLn(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.SumLn.nppsSumLn_64f(pSrc.DevicePointer, pSrc.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSumLn_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for 32f64f SumLn.
		/// This primitive provides the correct buffer size for nppsSumLn_32f64f.
		/// </summary>
		public static int SumLnGetBufferSize64f(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.SumLn.nppsSumLnGetBufferSize_32f64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSumLnGetBufferSize_32f64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit flaoting point input, 64-bit floating point output signal sum natural logarithm.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation.</param>
		public static void SumLn(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.SumLn.nppsSumLn_32f64f(pSrc.DevicePointer, pSrc.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSumLn_32f64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for 16s32f SumLn.
		/// This primitive provides the correct buffer size for nppsSumLn_16s32f.
		/// </summary>
		public static int SumLnGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.SumLn.nppsSumLnGetBufferSize_16s32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSumLnGetBufferSize_16s32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer input, 32-bit floating point output signal sum natural logarithm.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation.</param>
		public static void SumLn(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<float> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.SumLn.nppsSumLn_16s32f(pSrc.DevicePointer, pSrc.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSumLn_16s32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point signal inverse tangent.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Arctan(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.InverseTangentSignal.nppsArctan_32f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsArctan_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit floating point signal inverse tangent.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Arctan(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.InverseTangentSignal.nppsArctan_64f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsArctan_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point signal inverse tangent.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>

		public static void Arctan(this CudaDeviceVariable<float> pSrcDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.InverseTangentSignal.nppsArctan_32f_I(pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsArctan_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point signal inverse tangent.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>

		public static void Arctan(this CudaDeviceVariable<double> pSrcDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.InverseTangentSignal.nppsArctan_64f_I(pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsArctan_64f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point signal normalize.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="vSub">value subtracted from each signal element before division</param>
		/// <param name="vDiv">divisor of post-subtracted signal element dividend</param>
		public static void Normalize(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pDst, float vSub, float vDiv)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormalizeSignal.nppsNormalize_32f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, vSub, vDiv);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormalize_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit complex floating point signal normalize.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="vSub">value subtracted from each signal element before division</param>
		/// <param name="vDiv">divisor of post-subtracted signal element dividend</param>
		public static void Normalize(this CudaDeviceVariable<Npp32fc> pSrc, CudaDeviceVariable<Npp32fc> pDst, Npp32fc vSub, float vDiv)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormalizeSignal.nppsNormalize_32fc(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, vSub, vDiv);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormalize_32fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit floating point signal normalize.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="vSub">value subtracted from each signal element before division</param>
		/// <param name="vDiv">divisor of post-subtracted signal element dividend</param>
		public static void Normalize(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pDst, double vSub, double vDiv)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormalizeSignal.nppsNormalize_64f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, vSub, vDiv);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormalize_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit complex floating point signal normalize.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="vSub">value subtracted from each signal element before division</param>
		/// <param name="vDiv">divisor of post-subtracted signal element dividend</param>
		public static void Normalize(this CudaDeviceVariable<Npp64fc> pSrc, CudaDeviceVariable<Npp64fc> pDst, Npp64fc vSub, double vDiv)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormalizeSignal.nppsNormalize_64fc(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, vSub, vDiv);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormalize_64fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit signed short signal normalize, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="vSub">value subtracted from each signal element before division</param>
		/// <param name="vDiv">divisor of post-subtracted signal element dividend</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Normalize(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pDst, short vSub, int vDiv, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormalizeSignal.nppsNormalize_16s_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, vSub, vDiv, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormalize_16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit complex signed short signal normalize, scale, then clamp to saturated value.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="vSub">value subtracted from each signal element before division</param>
		/// <param name="vDiv">divisor of post-subtracted signal element dividend</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Normalize(this CudaDeviceVariable<Npp16sc> pSrc, CudaDeviceVariable<Npp16sc> pDst, Npp16sc vSub, int vDiv, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormalizeSignal.nppsNormalize_16sc_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, vSub, vDiv, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormalize_16sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit floating point signal Cauchy error calculation.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nParam">constant used in Cauchy formula</param>
		public static void Cauchy(this CudaDeviceVariable<float> pSrcDst, float nParam)
		{
			NppStatus status = NPPNativeMethods.NPPs.Cauchy.nppsCauchy_32f_I(pSrcDst.DevicePointer, pSrcDst.Size, nParam);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsCauchy_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point signal Cauchy first derivative.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nParam">constant used in Cauchy formula</param>
		public static void CauchyD(this CudaDeviceVariable<float> pSrcDst, float nParam)
		{
			NppStatus status = NPPNativeMethods.NPPs.Cauchy.nppsCauchyD_32f_I(pSrcDst.DevicePointer, pSrcDst.Size, nParam);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsCauchyD_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point signal Cauchy first and second derivatives.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="pD2FVal">Source signal pointer. This signal contains the second derivative</param>
		/// <param name="nParam">constant used in Cauchy formula</param>
		public static void CauchyDD2(this CudaDeviceVariable<float> pSrcDst, CudaDeviceVariable<float> pD2FVal, float nParam)
		{
			NppStatus status = NPPNativeMethods.NPPs.Cauchy.nppsCauchyDD2_32f_I(pSrcDst.DevicePointer, pD2FVal.DevicePointer, pSrcDst.Size, nParam);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsCauchyDD2_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 8-bit unsigned char signal and with constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be anded with each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void AndC(this CudaDeviceVariable<byte> pSrc, CudaDeviceVariable<byte> pDst, byte nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.AndC.nppsAndC_8u(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAndC_8u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit unsigned short signal and with constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be anded with each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void AndC(this CudaDeviceVariable<ushort> pSrc, CudaDeviceVariable<ushort> pDst, ushort nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.AndC.nppsAndC_16u(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAndC_16u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit unsigned integer signal and with constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be anded with each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void AndC(this CudaDeviceVariable<uint> pSrc, CudaDeviceVariable<uint> pDst, uint nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.AndC.nppsAndC_32u(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAndC_32u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 8-bit unsigned char in place signal and with constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be anded with each vector element</param>

		public static void AndC(this CudaDeviceVariable<byte> pSrcDst, byte nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.AndC.nppsAndC_8u_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAndC_8u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short in place signal and with constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be anded with each vector element</param>

		public static void AndC(this CudaDeviceVariable<ushort> pSrcDst, ushort nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.AndC.nppsAndC_16u_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAndC_16u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit unsigned signed integer in place signal and with constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be anded with each vector element</param>

		public static void AndC(this CudaDeviceVariable<uint> pSrcDst, uint nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.AndC.nppsAndC_32u_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAndC_32u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 8-bit unsigned char signal and with signal.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be anded with signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void And(this CudaDeviceVariable<byte> pSrc1, CudaDeviceVariable<byte> pSrc2, CudaDeviceVariable<byte> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.And.nppsAnd_8u(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAnd_8u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit unsigned short signal and with signal.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be anded with signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void And(this CudaDeviceVariable<ushort> pSrc1, CudaDeviceVariable<ushort> pSrc2, CudaDeviceVariable<ushort> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.And.nppsAnd_16u(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAnd_16u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit unsigned integer signal and with signal.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be anded with signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void And(this CudaDeviceVariable<uint> pSrc1, CudaDeviceVariable<uint> pSrc2, CudaDeviceVariable<uint> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.And.nppsAnd_32u(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAnd_32u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 8-bit unsigned char in place signal and with signal.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be anded with signal1 elements</param>

		public static void And(this CudaDeviceVariable<byte> pSrcDst, CudaDeviceVariable<byte> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.And.nppsAnd_8u_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAnd_8u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short in place signal and with signal.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be anded with signal1 elements</param>

		public static void And(this CudaDeviceVariable<ushort> pSrcDst, CudaDeviceVariable<ushort> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.And.nppsAnd_16u_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAnd_16u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit unsigned integer in place signal and with signal.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be anded with signal1 elements</param>

		public static void And(this CudaDeviceVariable<uint> pSrcDst, CudaDeviceVariable<uint> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.And.nppsAnd_32u_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAnd_32u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 8-bit unsigned char signal or with constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be ored with each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void OrC(this CudaDeviceVariable<byte> pSrc, CudaDeviceVariable<byte> pDst, byte nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.OrC.nppsOrC_8u(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsOrC_8u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit unsigned short signal or with constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be ored with each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void OrC(this CudaDeviceVariable<ushort> pSrc, CudaDeviceVariable<ushort> pDst, ushort nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.OrC.nppsOrC_16u(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsOrC_16u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit unsigned integer signal or with constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be ored with each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void OrC(this CudaDeviceVariable<uint> pSrc, CudaDeviceVariable<uint> pDst, uint nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.OrC.nppsOrC_32u(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsOrC_32u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 8-bit unsigned char in place signal or with constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be ored with each vector element</param>

		public static void OrC(this CudaDeviceVariable<byte> pSrcDst, byte nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.OrC.nppsOrC_8u_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsOrC_8u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short in place signal or with constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be ored with each vector element</param>

		public static void OrC(this CudaDeviceVariable<ushort> pSrcDst, ushort nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.OrC.nppsOrC_16u_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsOrC_16u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit unsigned signed integer in place signal or with constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be ored with each vector element</param>

		public static void OrC(this CudaDeviceVariable<uint> pSrcDst, uint nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.OrC.nppsOrC_32u_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsOrC_32u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 8-bit unsigned char signal or with signal.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be ored with signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Or(this CudaDeviceVariable<byte> pSrc1, CudaDeviceVariable<byte> pSrc2, CudaDeviceVariable<byte> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Or.nppsOr_8u(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsOr_8u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit unsigned short signal or with signal.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be ored with signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Or(this CudaDeviceVariable<ushort> pSrc1, CudaDeviceVariable<ushort> pSrc2, CudaDeviceVariable<ushort> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Or.nppsOr_16u(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsOr_16u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit unsigned integer signal or with signal.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be ored with signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Or(this CudaDeviceVariable<uint> pSrc1, CudaDeviceVariable<uint> pSrc2, CudaDeviceVariable<uint> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Or.nppsOr_32u(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsOr_32u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 8-bit unsigned char in place signal or with signal.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be ored with signal1 elements</param>

		public static void Or(this CudaDeviceVariable<byte> pSrcDst, CudaDeviceVariable<byte> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.Or.nppsOr_8u_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsOr_8u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short in place signal or with signal.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be ored with signal1 elements</param>

		public static void Or(this CudaDeviceVariable<ushort> pSrcDst, CudaDeviceVariable<ushort> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.Or.nppsOr_16u_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsOr_16u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit unsigned integer in place signal or with signal.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be ored with signal1 elements</param>

		public static void Or(this CudaDeviceVariable<uint> pSrcDst, CudaDeviceVariable<uint> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.Or.nppsOr_32u_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsOr_32u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 8-bit unsigned char signal exclusive or with constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be exclusive ored with each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void XorC(this CudaDeviceVariable<byte> pSrc, CudaDeviceVariable<byte> pDst, byte nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.XorC.nppsXorC_8u(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsXorC_8u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit unsigned short signal exclusive or with constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be exclusive ored with each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void XorC(this CudaDeviceVariable<ushort> pSrc, CudaDeviceVariable<ushort> pDst, ushort nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.XorC.nppsXorC_16u(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsXorC_16u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit unsigned integer signal exclusive or with constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be exclusive ored with each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void XorC(this CudaDeviceVariable<uint> pSrc, CudaDeviceVariable<uint> pDst, uint nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.XorC.nppsXorC_32u(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsXorC_32u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 8-bit unsigned char in place signal exclusive or with constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be exclusive ored with each vector element</param>

		public static void XorC(this CudaDeviceVariable<byte> pSrcDst, byte nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.XorC.nppsXorC_8u_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsXorC_8u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short in place signal exclusive or with constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be exclusive ored with each vector element</param>

		public static void XorC(this CudaDeviceVariable<ushort> pSrcDst, ushort nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.XorC.nppsXorC_16u_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsXorC_16u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit unsigned signed integer in place signal exclusive or with constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be exclusive ored with each vector element</param>

		public static void XorC(this CudaDeviceVariable<uint> pSrcDst, uint nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.XorC.nppsXorC_32u_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsXorC_32u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 8-bit unsigned char signal exclusive or with signal.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be exclusive ored with signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Xor(this CudaDeviceVariable<byte> pSrc1, CudaDeviceVariable<byte> pSrc2, CudaDeviceVariable<byte> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Xor.nppsXor_8u(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsXor_8u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit unsigned short signal exclusive or with signal.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be exclusive ored with signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Xor(this CudaDeviceVariable<ushort> pSrc1, CudaDeviceVariable<ushort> pSrc2, CudaDeviceVariable<ushort> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Xor.nppsXor_16u(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsXor_16u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit unsigned integer signal exclusive or with signal.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer. signal2 elements to be exclusive ored with signal1 elements</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Xor(this CudaDeviceVariable<uint> pSrc1, CudaDeviceVariable<uint> pSrc2, CudaDeviceVariable<uint> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Xor.nppsXor_32u(pSrc1.DevicePointer, pSrc2.DevicePointer, pDst.DevicePointer, pSrc1.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsXor_32u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 8-bit unsigned char in place signal exclusive or with signal.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be exclusive ored with signal1 elements</param>

		public static void Xor(this CudaDeviceVariable<byte> pSrcDst, CudaDeviceVariable<byte> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.Xor.nppsXor_8u_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsXor_8u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short in place signal exclusive or with signal.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be exclusive ored with signal1 elements</param>

		public static void Xor(this CudaDeviceVariable<ushort> pSrcDst, CudaDeviceVariable<ushort> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.Xor.nppsXor_16u_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsXor_16u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit unsigned integer in place signal exclusive or with signal.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer. signal2 elements to be exclusive ored with signal1 elements</param>

		public static void Xor(this CudaDeviceVariable<uint> pSrcDst, CudaDeviceVariable<uint> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.Xor.nppsXor_32u_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsXor_32u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 8-bit unsigned char not signal.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Not(this CudaDeviceVariable<byte> pSrc, CudaDeviceVariable<byte> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Not.nppsNot_8u(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNot_8u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit unsigned short not signal.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Not(this CudaDeviceVariable<ushort> pSrc, CudaDeviceVariable<ushort> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Not.nppsNot_16u(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNot_16u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit unsigned integer not signal.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void Not(this CudaDeviceVariable<uint> pSrc, CudaDeviceVariable<uint> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Not.nppsNot_32u(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNot_32u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 8-bit unsigned char in place not signal.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>

		public static void Not(this CudaDeviceVariable<byte> pSrcDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Not.nppsNot_8u_I(pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNot_8u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short in place not signal.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>

		public static void Not(this CudaDeviceVariable<ushort> pSrcDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Not.nppsNot_16u_I(pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNot_16u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit unsigned signed integer in place not signal.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>

		public static void Not(this CudaDeviceVariable<uint> pSrcDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Not.nppsNot_32u_I(pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNot_32u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 8-bit unsigned char signal left shift with constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be used to left shift each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void LShiftC(this CudaDeviceVariable<byte> pSrc, CudaDeviceVariable<byte> pDst, int nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.LShiftC.nppsLShiftC_8u(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsLShiftC_8u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit unsigned short signal left shift with constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be used to left shift each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void LShiftC(this CudaDeviceVariable<ushort> pSrc, CudaDeviceVariable<ushort> pDst, int nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.LShiftC.nppsLShiftC_16u(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsLShiftC_16u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit signed short signal left shift with constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be used to left shift each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void LShiftC(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pDst, int nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.LShiftC.nppsLShiftC_16s(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsLShiftC_16s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit unsigned integer signal left shift with constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be used to left shift each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void LShiftC(this CudaDeviceVariable<uint> pSrc, CudaDeviceVariable<uint> pDst, int nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.LShiftC.nppsLShiftC_32u(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsLShiftC_32u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit signed integer signal left shift with constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be used to left shift each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void LShiftC(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pDst, int nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.LShiftC.nppsLShiftC_32s(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsLShiftC_32s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 8-bit unsigned char in place signal left shift with constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be used to left shift each vector element</param>

		public static void LShiftC(this CudaDeviceVariable<byte> pSrcDst, int nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.LShiftC.nppsLShiftC_8u_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsLShiftC_8u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short in place signal left shift with constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be used to left shift each vector element</param>

		public static void LShiftC(this CudaDeviceVariable<ushort> pSrcDst, int nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.LShiftC.nppsLShiftC_16u_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsLShiftC_16u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short in place signal left shift with constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be used to left shift each vector element</param>

		public static void LShiftC(this CudaDeviceVariable<short> pSrcDst, int nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.LShiftC.nppsLShiftC_16s_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsLShiftC_16s_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit unsigned signed integer in place signal left shift with constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be used to left shift each vector element</param>

		public static void LShiftC(this CudaDeviceVariable<uint> pSrcDst, int nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.LShiftC.nppsLShiftC_32u_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsLShiftC_32u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit signed signed integer in place signal left shift with constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be used to left shift each vector element</param>

		public static void LShiftC(this CudaDeviceVariable<int> pSrcDst, int nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.LShiftC.nppsLShiftC_32s_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsLShiftC_32s_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 8-bit unsigned char signal right shift with constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be used to right shift each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void RShiftC(this CudaDeviceVariable<byte> pSrc, CudaDeviceVariable<byte> pDst, int nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.RShiftC.nppsRShiftC_8u(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsRShiftC_8u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit unsigned short signal right shift with constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be used to right shift each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void RShiftC(this CudaDeviceVariable<ushort> pSrc, CudaDeviceVariable<ushort> pDst, int nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.RShiftC.nppsRShiftC_16u(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsRShiftC_16u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit signed short signal right shift with constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be used to right shift each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void RShiftC(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pDst, int nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.RShiftC.nppsRShiftC_16s(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsRShiftC_16s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit unsigned integer signal right shift with constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be used to right shift each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void RShiftC(this CudaDeviceVariable<uint> pSrc, CudaDeviceVariable<uint> pDst, int nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.RShiftC.nppsRShiftC_32u(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsRShiftC_32u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit signed integer signal right shift with constant.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="nValue">Constant value to be used to right shift each vector element</param>
		/// <param name="pDst">Destination signal pointer.</param>

		public static void RShiftC(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pDst, int nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.RShiftC.nppsRShiftC_32s(pSrc.DevicePointer, nValue, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsRShiftC_32s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 8-bit unsigned char in place signal right shift with constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be used to right shift each vector element</param>

		public static void RShiftC(this CudaDeviceVariable<byte> pSrcDst, int nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.RShiftC.nppsRShiftC_8u_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsRShiftC_8u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short in place signal right shift with constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be used to right shift each vector element</param>

		public static void RShiftC(this CudaDeviceVariable<ushort> pSrcDst, int nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.RShiftC.nppsRShiftC_16u_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsRShiftC_16u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short in place signal right shift with constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be used to right shift each vector element</param>

		public static void RShiftC(this CudaDeviceVariable<short> pSrcDst, int nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.RShiftC.nppsRShiftC_16s_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsRShiftC_16s_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit unsigned signed integer in place signal right shift with constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be used to right shift each vector element</param>

		public static void RShiftC(this CudaDeviceVariable<uint> pSrcDst, int nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.RShiftC.nppsRShiftC_32u_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsRShiftC_32u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit signed signed integer in place signal right shift with constant.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nValue">Constant value to be used to right shift each vector element</param>

		public static void RShiftC(this CudaDeviceVariable<int> pSrcDst, int nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.RShiftC.nppsRShiftC_32s_I(nValue, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsRShiftC_32s_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		#endregion

		#region Filtering
		/// <summary>
		/// IntegralGetBufferSize
		/// </summary>
		public static int IntegralGetBufferSize(this CudaDeviceVariable<int> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.FilteringFunctions.nppsIntegralGetBufferSize_32s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsIntegralGetBufferSize_32s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Integral
		/// </summary>
		public static void Integral(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.FilteringFunctions.nppsIntegral_32s(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsIntegral_32s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 8-bit unsigned char, vector zero method.
		/// </summary>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Zero(this CudaDeviceVariable<byte> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Zero.nppsZero_8u(pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsZero_8u", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 16-bit integer, vector zero method.
		/// </summary>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Zero(this CudaDeviceVariable<short> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Zero.nppsZero_16s(pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsZero_16s", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 16-bit integer complex, vector zero method.
		/// </summary>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Zero(this CudaDeviceVariable<Npp16sc> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Zero.nppsZero_16sc(pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsZero_16sc", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 32-bit integer, vector zero method.
		/// </summary>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Zero(this CudaDeviceVariable<int> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Zero.nppsZero_32s(pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsZero_32s", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 32-bit integer complex, vector zero method.
		/// </summary>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Zero(this CudaDeviceVariable<Npp32sc> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Zero.nppsZero_32sc(pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsZero_32sc", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 32-bit float, vector zero method.
		/// </summary>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Zero(this CudaDeviceVariable<float> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Zero.nppsZero_32f(pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsZero_32f", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 32-bit float complex, vector zero method.
		/// </summary>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Zero(this CudaDeviceVariable<Npp32fc> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Zero.nppsZero_32fc(pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsZero_32fc", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 64-bit long long integer, vector zero method.
		/// </summary>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Zero(this CudaDeviceVariable<long> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Zero.nppsZero_64s(pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsZero_64s", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 64-bit long long integer complex, vector zero method.
		/// </summary>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Zero(this CudaDeviceVariable<Npp64sc> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Zero.nppsZero_64sc(pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsZero_64sc", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 64-bit double, vector zero method.
		/// </summary>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Zero(this CudaDeviceVariable<double> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Zero.nppsZero_64f(pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsZero_64f", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 64-bit double complex, vector zero method.
		/// </summary>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Zero(this CudaDeviceVariable<Npp64fc> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Zero.nppsZero_64fc(pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsZero_64fc", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 8-bit unsigned char, vector set method.
		/// </summary>
		/// <param name="nValue">Value used to initialize the vector pDst.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Set(this CudaDeviceVariable<byte> pDst, byte nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Set.nppsSet_8u(nValue, pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSet_8u", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 8-bit signed char, vector set method.
		/// </summary>
		/// <param name="nValue">Value used to initialize the vector pDst.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Set(this CudaDeviceVariable<sbyte> pDst, sbyte nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Set.nppsSet_8s(nValue, pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSet_8s", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 16-bit integer, vector set method.
		/// </summary>
		/// <param name="nValue">Value used to initialize the vector pDst.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Set(this CudaDeviceVariable<short> pDst, short nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Set.nppsSet_16s(nValue, pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSet_16s", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 16-bit unsigned integer, vector set method.
		/// </summary>
		/// <param name="nValue">Value used to initialize the vector pDst.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Set(this CudaDeviceVariable<ushort> pDst, ushort nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Set.nppsSet_16u(nValue, pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSet_16u", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 16-bit integer complex, vector set method.
		/// </summary>
		/// <param name="nValue">Value used to initialize the vector pDst.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Set(this CudaDeviceVariable<Npp16sc> pDst, Npp16sc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Set.nppsSet_16sc(nValue, pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSet_16sc", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 32-bit integer, vector set method.
		/// </summary>
		/// <param name="nValue">Value used to initialize the vector pDst.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Set(this CudaDeviceVariable<int> pDst, int nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Set.nppsSet_32s(nValue, pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSet_32s", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 32-bit unsigned integer, vector set method.
		/// </summary>
		/// <param name="nValue">Value used to initialize the vector pDst.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Set(this CudaDeviceVariable<uint> pDst, uint nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Set.nppsSet_32u(nValue, pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSet_32u", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 32-bit integer complex, vector set method.
		/// </summary>
		/// <param name="nValue">Value used to initialize the vector pDst.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Set(this CudaDeviceVariable<Npp32sc> pDst, Npp32sc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Set.nppsSet_32sc(nValue, pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSet_32sc", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 32-bit float, vector set method.
		/// </summary>
		/// <param name="nValue">Value used to initialize the vector pDst.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Set(this CudaDeviceVariable<float> pDst, float nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Set.nppsSet_32f(nValue, pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSet_32f", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 32-bit float complex, vector set method.
		/// </summary>
		/// <param name="nValue">Value used to initialize the vector pDst.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Set(this CudaDeviceVariable<Npp32fc> pDst, Npp32fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Set.nppsSet_32fc(nValue, pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSet_32fc", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 64-bit long long integer, vector set method.
		/// </summary>
		/// <param name="nValue">Value used to initialize the vector pDst.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Set(this CudaDeviceVariable<long> pDst, long nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Set.nppsSet_64s(nValue, pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSet_64s", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 64-bit long long integer complex, vector set method.
		/// </summary>
		/// <param name="nValue">Value used to initialize the vector pDst.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Set(this CudaDeviceVariable<Npp64sc> pDst, Npp64sc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Set.nppsSet_64sc(nValue, pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSet_64sc", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 64-bit double, vector set method.
		/// </summary>
		/// <param name="nValue">Value used to initialize the vector pDst.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Set(this CudaDeviceVariable<double> pDst, double nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Set.nppsSet_64f(nValue, pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSet_64f", status));
			NPPException.CheckNppStatus(status, pDst);
		}

		/// <summary>
		/// 64-bit double complex, vector set method.
		/// </summary>
		/// <param name="nValue">Value used to initialize the vector pDst.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		public static void Set(this CudaDeviceVariable<Npp64fc> pDst, Npp64fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Set.nppsSet_64fc(nValue, pDst.DevicePointer, pDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSet_64fc", status));
			NPPException.CheckNppStatus(status, pDst);
		}



		#endregion

		#region Convert
		/// <summary>
		/// 8-bit signed byte signal to 16-bit signed short.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<sbyte> pSrc, CudaDeviceVariable<short> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_8s16s(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_8s16s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 8-bit signed byte signal to 32-bit float.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<sbyte> pSrc, CudaDeviceVariable<float> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_8s32f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_8s32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 8-bit unsigned byte signal to 32-bit float.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<byte> pSrc, CudaDeviceVariable<float> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_8u32f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_8u32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit signed short signal to 8-bit signed byte.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<sbyte> pDst, NppRoundMode eRoundMode, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_16s8s_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, eRoundMode, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_16s8s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit signed short signal to 32-bit signed int.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<int> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_16s32s(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_16s32s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit signed short signal to 32-bit float.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<float> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_16s32f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_16s32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit unsigned short signal to 32-bit float.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<ushort> pSrc, CudaDeviceVariable<float> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_16u32f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_16u32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit signed int signal to 16-bit signed short.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<short> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_32s16s(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_32s16s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit signed int signal to 32-bit float.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<float> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_32s32f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_32s32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit signed int signal to 64-bit double.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<double> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_32s64f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_32s64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit float signal to 64-bit double.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<double> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_32f64f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_32f64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit signed long signal to 64-bit double.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<long> pSrc, CudaDeviceVariable<double> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_64s64f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_64s64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit double signal to 32-bit float.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<float> pDst)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_64f32f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_64f32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit signed short signal to 32-bit float.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<float> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_16s32f_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_16s32f_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit signed short signal to 64-bit double.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<double> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_16s64f_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_16s64f_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit signed int signal to 16-bit signed short.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<short> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_32s16s_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_32s16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit signed int signal to 32-bit float.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<float> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_32s32f_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_32s32f_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit signed int signal to 64-bit double.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<double> pDst, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_32s64f_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_32s64f_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit float signal to 8-bit signed byte.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<sbyte> pDst, NppRoundMode eRoundMode, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_32f8s_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, eRoundMode, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_32f8s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit float signal to 8-bit unsigned byte.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<byte> pDst, NppRoundMode eRoundMode, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_32f8u_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, eRoundMode, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_32f8u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit float signal to 16-bit signed short.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<short> pDst, NppRoundMode eRoundMode, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_32f16s_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, eRoundMode, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_32f16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit float signal to 16-bit unsigned short.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<ushort> pDst, NppRoundMode eRoundMode, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_32f16u_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, eRoundMode, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_32f16u_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit float signal to 32-bit signed int.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<int> pDst, NppRoundMode eRoundMode, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_32f32s_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, eRoundMode, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_32f32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit signed long signal to 32-bit signed int.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<long> pSrc, CudaDeviceVariable<int> pDst, NppRoundMode eRoundMode, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_64s32s_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, eRoundMode, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_64s32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit double signal to 16-bit signed short.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<short> pDst, NppRoundMode eRoundMode, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_64f16s_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, eRoundMode, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_64f16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit double signal to 32-bit signed int.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<int> pDst, NppRoundMode eRoundMode, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_64f32s_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, eRoundMode, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_64f32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit double signal to 64-bit signed long.
		/// </summary>
		public static void Convert(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<long> pDst, NppRoundMode eRoundMode, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Convert.nppsConvert_64f64s_Sfs(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, eRoundMode, nScaleFactor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsConvert_64f64s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit signed short signal threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		/// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
		public static void Threshold(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pDst, short nLevel, NppCmpOp nRelOp)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_16s(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel, nRelOp);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_16s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit in place signed short signal threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		/// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
		public static void Threshold(this CudaDeviceVariable<short> pSrcDst, short nLevel, NppCmpOp nRelOp)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_16s_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel, nRelOp);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_16s_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short complex number signal threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		/// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
		public static void Threshold(this CudaDeviceVariable<Npp16sc> pSrc, CudaDeviceVariable<Npp16sc> pDst, short nLevel, NppCmpOp nRelOp)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_16sc(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel, nRelOp);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_16sc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit in place signed short complex number signal threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		/// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
		public static void Threshold(this CudaDeviceVariable<Npp16sc> pSrcDst, short nLevel, NppCmpOp nRelOp)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_16sc_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel, nRelOp);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_16sc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point signal threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		/// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
		public static void Threshold(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pDst, float nLevel, NppCmpOp nRelOp)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_32f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel, nRelOp);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit in place floating point signal threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		/// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
		public static void Threshold(this CudaDeviceVariable<float> pSrcDst, float nLevel, NppCmpOp nRelOp)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_32f_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel, nRelOp);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point complex number signal threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		/// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
		public static void Threshold(this CudaDeviceVariable<Npp32fc> pSrc, CudaDeviceVariable<Npp32fc> pDst, float nLevel, NppCmpOp nRelOp)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_32fc(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel, nRelOp);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_32fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit in place floating point complex number signal threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		/// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
		public static void Threshold(this CudaDeviceVariable<Npp32fc> pSrcDst, float nLevel, NppCmpOp nRelOp)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_32fc_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel, nRelOp);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_32fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point signal threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		/// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
		public static void Threshold(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pDst, double nLevel, NppCmpOp nRelOp)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_64f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel, nRelOp);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit in place floating point signal threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		/// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
		public static void Threshold(this CudaDeviceVariable<double> pSrcDst, double nLevel, NppCmpOp nRelOp)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_64f_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel, nRelOp);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_64f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point complex number signal threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		/// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
		public static void Threshold(this CudaDeviceVariable<Npp64fc> pSrc, CudaDeviceVariable<Npp64fc> pDst, double nLevel, NppCmpOp nRelOp)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_64fc(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel, nRelOp);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_64fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit in place floating point complex number signal threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		/// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
		public static void Threshold(this CudaDeviceVariable<Npp64fc> pSrcDst, double nLevel, NppCmpOp nRelOp)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_64fc_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel, nRelOp);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_64fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		public static void Threshold_LT(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pDst, short nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LT_16s(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LT_16s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit in place signed short signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		public static void Threshold_LT(this CudaDeviceVariable<short> pSrcDst, short nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LT_16s_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LT_16s_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short complex number signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		public static void Threshold_LT(this CudaDeviceVariable<Npp16sc> pSrc, CudaDeviceVariable<Npp16sc> pDst, short nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LT_16sc(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LT_16sc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit in place signed short complex number signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		public static void Threshold_LT(this CudaDeviceVariable<Npp16sc> pSrcDst, short nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LT_16sc_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LT_16sc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		public static void Threshold_LT(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pDst, float nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LT_32f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LT_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit in place floating point signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		public static void Threshold_LT(this CudaDeviceVariable<float> pSrcDst, float nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LT_32f_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LT_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point complex number signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		public static void Threshold_LT(this CudaDeviceVariable<Npp32fc> pSrc, CudaDeviceVariable<Npp32fc> pDst, float nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LT_32fc(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LT_32fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit in place floating point complex number signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		public static void Threshold_LT(this CudaDeviceVariable<Npp32fc> pSrcDst, float nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LT_32fc_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LT_32fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		public static void Threshold_LT(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pDst, double nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LT_64f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LT_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit in place floating point signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		public static void Threshold_LT(this CudaDeviceVariable<double> pSrcDst, double nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LT_64f_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LT_64f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point complex number signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		public static void Threshold_LT(this CudaDeviceVariable<Npp64fc> pSrc, CudaDeviceVariable<Npp64fc> pDst, double nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LT_64fc(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LT_64fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit in place floating point complex number signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		public static void Threshold_LT(this CudaDeviceVariable<Npp64fc> pSrcDst, double nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LT_64fc_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LT_64fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		public static void Threshold_GT(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pDst, short nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GT_16s(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GT_16s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit in place signed short signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		public static void Threshold_GT(this CudaDeviceVariable<short> pSrcDst, short nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GT_16s_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GT_16s_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short complex number signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		public static void Threshold_GT(this CudaDeviceVariable<Npp16sc> pSrc, CudaDeviceVariable<Npp16sc> pDst, short nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GT_16sc(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GT_16sc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit in place signed short complex number signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		public static void Threshold_GT(this CudaDeviceVariable<Npp16sc> pSrcDst, short nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GT_16sc_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GT_16sc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		public static void Threshold_GT(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pDst, float nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GT_32f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GT_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit in place floating point signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		public static void Threshold_GT(this CudaDeviceVariable<float> pSrcDst, float nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GT_32f_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GT_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point complex number signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		public static void Threshold_GT(this CudaDeviceVariable<Npp32fc> pSrc, CudaDeviceVariable<Npp32fc> pDst, float nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GT_32fc(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GT_32fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit in place floating point complex number signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		public static void Threshold_GT(this CudaDeviceVariable<Npp32fc> pSrcDst, float nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GT_32fc_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GT_32fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		public static void Threshold_GT(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pDst, double nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GT_64f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GT_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit in place floating point signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		public static void Threshold_GT(this CudaDeviceVariable<double> pSrcDst, double nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GT_64f_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GT_64f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point complex number signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		public static void Threshold_GT(this CudaDeviceVariable<Npp64fc> pSrc, CudaDeviceVariable<Npp64fc> pDst, double nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GT_64fc(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GT_64fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit in place floating point complex number signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		public static void Threshold_GT(this CudaDeviceVariable<Npp64fc> pSrcDst, double nLevel)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GT_64fc_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GT_64fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_LTVal(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pDst, short nLevel, short nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LTVal_16s(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LTVal_16s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit in place signed short signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_LTVal(this CudaDeviceVariable<short> pSrcDst, short nLevel, short nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LTVal_16s_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LTVal_16s_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short complex number signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_LTVal(this CudaDeviceVariable<Npp16sc> pSrc, CudaDeviceVariable<Npp16sc> pDst, short nLevel, Npp16sc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LTVal_16sc(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LTVal_16sc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit in place signed short complex number signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_LTVal(this CudaDeviceVariable<Npp16sc> pSrcDst, short nLevel, Npp16sc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LTVal_16sc_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LTVal_16sc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_LTVal(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pDst, float nLevel, float nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LTVal_32f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LTVal_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit in place floating point signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_LTVal(this CudaDeviceVariable<float> pSrcDst, float nLevel, float nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LTVal_32f_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LTVal_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point complex number signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_LTVal(this CudaDeviceVariable<Npp32fc> pSrc, CudaDeviceVariable<Npp32fc> pDst, float nLevel, Npp32fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LTVal_32fc(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LTVal_32fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit in place floating point complex number signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_LTVal(this CudaDeviceVariable<Npp32fc> pSrcDst, float nLevel, Npp32fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LTVal_32fc_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LTVal_32fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_LTVal(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pDst, double nLevel, double nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LTVal_64f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LTVal_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit in place floating point signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_LTVal(this CudaDeviceVariable<double> pSrcDst, double nLevel, double nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LTVal_64f_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LTVal_64f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point complex number signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_LTVal(this CudaDeviceVariable<Npp64fc> pSrc, CudaDeviceVariable<Npp64fc> pDst, double nLevel, Npp64fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LTVal_64fc(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LTVal_64fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit in place floating point complex number signal NPP_CMP_LESS threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_LTVal(this CudaDeviceVariable<Npp64fc> pSrcDst, double nLevel, Npp64fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_LTVal_64fc_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_LTVal_64fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_GTVal(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pDst, short nLevel, short nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GTVal_16s(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GTVal_16s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit in place signed short signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_GTVal(this CudaDeviceVariable<short> pSrcDst, short nLevel, short nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GTVal_16s_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GTVal_16s_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short complex number signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_GTVal(this CudaDeviceVariable<Npp16sc> pSrc, CudaDeviceVariable<Npp16sc> pDst, short nLevel, Npp16sc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GTVal_16sc(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GTVal_16sc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit in place signed short complex number signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_GTVal(this CudaDeviceVariable<Npp16sc> pSrcDst, short nLevel, Npp16sc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GTVal_16sc_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GTVal_16sc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_GTVal(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pDst, float nLevel, float nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GTVal_32f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GTVal_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit in place floating point signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_GTVal(this CudaDeviceVariable<float> pSrcDst, float nLevel, float nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GTVal_32f_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GTVal_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point complex number signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_GTVal(this CudaDeviceVariable<Npp32fc> pSrc, CudaDeviceVariable<Npp32fc> pDst, float nLevel, Npp32fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GTVal_32fc(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GTVal_32fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit in place floating point complex number signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_GTVal(this CudaDeviceVariable<Npp32fc> pSrcDst, float nLevel, Npp32fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GTVal_32fc_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GTVal_32fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_GTVal(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pDst, double nLevel, double nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GTVal_64f(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GTVal_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit in place floating point signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_GTVal(this CudaDeviceVariable<double> pSrcDst, double nLevel, double nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GTVal_64f_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GTVal_64f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point complex number signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pDst">Destination signal pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_GTVal(this CudaDeviceVariable<Npp64fc> pSrc, CudaDeviceVariable<Npp64fc> pDst, double nLevel, Npp64fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GTVal_64fc(pSrc.DevicePointer, pDst.DevicePointer, pSrc.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GTVal_64fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit in place floating point complex number signal NPP_CMP_GREATER threshold with constant level.
		/// </summary>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		/// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
		/// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
		public static void Threshold_GTVal(this CudaDeviceVariable<Npp64fc> pSrcDst, double nLevel, Npp64fc nValue)
		{
			NppStatus status = NPPNativeMethods.NPPs.Threshold.nppsThreshold_GTVal_64fc_I(pSrcDst.DevicePointer, pSrcDst.Size, nLevel, nValue);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsThreshold_GTVal_64fc_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}


		#endregion

		#region Statistik
		/// <summary>
		/// 8-bit in place min value for each pair of elements.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		public static void MinEvery(this CudaDeviceVariable<byte> pSrcDst, CudaDeviceVariable<byte> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxEvery.nppsMinEvery_8u_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinEvery_8u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short integer in place min value for each pair of elements.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		public static void MinEvery(this CudaDeviceVariable<ushort> pSrcDst, CudaDeviceVariable<ushort> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxEvery.nppsMinEvery_16u_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinEvery_16u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short integer in place min value for each pair of elements.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		public static void MinEvery(this CudaDeviceVariable<short> pSrcDst, CudaDeviceVariable<short> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxEvery.nppsMinEvery_16s_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinEvery_16s_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit signed integer in place min value for each pair of elements.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		public static void MinEvery(this CudaDeviceVariable<int> pSrcDst, CudaDeviceVariable<int> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxEvery.nppsMinEvery_32s_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinEvery_32s_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point in place min value for each pair of elements.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		public static void MinEvery(this CudaDeviceVariable<float> pSrcDst, CudaDeviceVariable<float> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxEvery.nppsMinEvery_32f_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinEvery_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 64-bit floating point in place min value for each pair of elements.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		public static void MinEvery(this CudaDeviceVariable<double> pSrcDst, CudaDeviceVariable<double> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxEvery.nppsMinEvery_64f_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinEvery_64f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 8-bit in place max value for each pair of elements.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		public static void MaxEvery(this CudaDeviceVariable<byte> pSrcDst, CudaDeviceVariable<byte> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxEvery.nppsMaxEvery_8u_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxEvery_8u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit unsigned short integer in place max value for each pair of elements.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		public static void MaxEvery(this CudaDeviceVariable<ushort> pSrcDst, CudaDeviceVariable<ushort> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxEvery.nppsMaxEvery_16u_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxEvery_16u_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 16-bit signed short integer in place max value for each pair of elements.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		public static void MaxEvery(this CudaDeviceVariable<short> pSrcDst, CudaDeviceVariable<short> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxEvery.nppsMaxEvery_16s_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxEvery_16s_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit signed integer in place max value for each pair of elements.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		public static void MaxEvery(this CudaDeviceVariable<int> pSrcDst, CudaDeviceVariable<int> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxEvery.nppsMaxEvery_32s_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxEvery_32s_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// 32-bit floating point in place max value for each pair of elements.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSrcDst">In-Place Signal Pointer.</param>
		public static void MaxEvery(this CudaDeviceVariable<float> pSrcDst, CudaDeviceVariable<float> pSrc)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxEvery.nppsMaxEvery_32f_I(pSrc.DevicePointer, pSrcDst.DevicePointer, pSrcDst.Size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxEvery_32f_I", status));
			NPPException.CheckNppStatus(status, pSrcDst);
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsSum_32f.
		/// </summary>
		public static int SumGetBufferSize(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Sum.nppsSumGetBufferSize_32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSumGetBufferSize_32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsSum_32fc.
		/// </summary>
		public static int SumGetBufferSize(this CudaDeviceVariable<Npp32fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Sum.nppsSumGetBufferSize_32fc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSumGetBufferSize_32fc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsSum_64f.
		/// </summary>
		public static int SumGetBufferSize(this CudaDeviceVariable<double> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Sum.nppsSumGetBufferSize_64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSumGetBufferSize_64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsSum_64fc.
		/// </summary>
		public static int SumGetBufferSize(this CudaDeviceVariable<Npp64fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Sum.nppsSumGetBufferSize_64fc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSumGetBufferSize_64fc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsSum_16s_Sfs.
		/// </summary>
		public static int SumGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Sum.nppsSumGetBufferSize_16s_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSumGetBufferSize_16s_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsSum_16sc_Sfs.
		/// </summary>
		public static int SumGetBufferSize(this CudaDeviceVariable<Npp16sc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Sum.nppsSumGetBufferSize_16sc_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSumGetBufferSize_16sc_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsSum_16sc32sc_Sfs.
		/// </summary>
		public static int SumGetBufferSize32sc(this CudaDeviceVariable<Npp16sc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Sum.nppsSumGetBufferSize_16sc32sc_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSumGetBufferSize_16sc32sc_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsSum_32s_Sfs.
		/// </summary>
		public static int SumGetBufferSize(this CudaDeviceVariable<int> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Sum.nppsSumGetBufferSize_32s_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSumGetBufferSize_32s_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsSum_16s32s_Sfs.
		/// </summary>
		public static int SumGetBufferSize32s(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Sum.nppsSumGetBufferSize_16s32s_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSumGetBufferSize_16s32s_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit float vector sum method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSum">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Sum(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pSum, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Sum.nppsSum_32f(pSrc.DevicePointer, pSrc.Size, pSum.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSum_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit float complex vector sum method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSum">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Sum(this CudaDeviceVariable<Npp32fc> pSrc, CudaDeviceVariable<Npp32fc> pSum, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Sum.nppsSum_32fc(pSrc.DevicePointer, pSrc.Size, pSum.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSum_32fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit double vector sum method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSum">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Sum(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pSum, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Sum.nppsSum_64f(pSrc.DevicePointer, pSrc.Size, pSum.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSum_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit double complex vector sum method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSum">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Sum(this CudaDeviceVariable<Npp64fc> pSrc, CudaDeviceVariable<Npp64fc> pSum, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Sum.nppsSum_64fc(pSrc.DevicePointer, pSrc.Size, pSum.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSum_64fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit short vector sum with integer scaling method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSum">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sum(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pSum, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Sum.nppsSum_16s_Sfs(pSrc.DevicePointer, pSrc.Size, pSum.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSum_16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit integer vector sum with integer scaling method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSum">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sum(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pSum, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Sum.nppsSum_32s_Sfs(pSrc.DevicePointer, pSrc.Size, pSum.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSum_32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit short complex vector sum with integer scaling method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSum">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sum(this CudaDeviceVariable<Npp16sc> pSrc, CudaDeviceVariable<Npp16sc> pSum, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Sum.nppsSum_16sc_Sfs(pSrc.DevicePointer, pSrc.Size, pSum.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSum_16sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit short complex vector sum (32bit int complex) with integer scaling
		/// method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSum">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sum(this CudaDeviceVariable<Npp16sc> pSrc, CudaDeviceVariable<Npp32sc> pSum, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Sum.nppsSum_16sc32sc_Sfs(pSrc.DevicePointer, pSrc.Size, pSum.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSum_16sc32sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit integer vector sum (32bit) with integer scaling method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pSum">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Sum(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<int> pSum, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Sum.nppsSum_16s32s_Sfs(pSrc.DevicePointer, pSrc.Size, pSum.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsSum_16s32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMax_16s.
		/// </summary>
		public static int MaxGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMaxGetBufferSize_16s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxGetBufferSize_16s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMax_32s.
		/// </summary>
		public static int MaxGetBufferSize(this CudaDeviceVariable<int> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMaxGetBufferSize_32s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxGetBufferSize_32s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMax_32f.
		/// </summary>
		public static int MaxGetBufferSize(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMaxGetBufferSize_32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxGetBufferSize_32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMax_64f.
		/// </summary>
		public static int MaxGetBufferSize(this CudaDeviceVariable<double> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMaxGetBufferSize_64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxGetBufferSize_64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit integer vector max method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMax">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Max(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pMax, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMax_16s(pSrc.DevicePointer, pSrc.Size, pMax.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMax_16s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit integer vector max method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMax">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Max(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pMax, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMax_32s(pSrc.DevicePointer, pSrc.Size, pMax.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMax_32s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit float vector max method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMax">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Max(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pMax, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMax_32f(pSrc.DevicePointer, pSrc.Size, pMax.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMax_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit float vector max method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMax">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Max(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pMax, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMax_64f(pSrc.DevicePointer, pSrc.Size, pMax.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMax_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMaxIndx_16s.
		/// </summary>
		public static int MaxIndxGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMaxIndxGetBufferSize_16s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxIndxGetBufferSize_16s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMaxIndx_32s.
		/// </summary>
		public static int MaxIndxGetBufferSize(this CudaDeviceVariable<int> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMaxIndxGetBufferSize_32s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxIndxGetBufferSize_32s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMaxIndx_32f.
		/// </summary>
		public static int MaxIndxGetBufferSize(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMaxIndxGetBufferSize_32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxIndxGetBufferSize_32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMaxIndx_64f.
		/// </summary>
		public static int MaxIndxGetBufferSize(this CudaDeviceVariable<double> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMaxIndxGetBufferSize_64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxIndxGetBufferSize_64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit integer vector max index method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMax">Pointer to the output result.</param>
		/// <param name="pIndx">Pointer to the index value of the first maximum element.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaxIndx(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pMax, CudaDeviceVariable<int> pIndx, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMaxIndx_16s(pSrc.DevicePointer, pSrc.Size, pMax.DevicePointer, pIndx.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxIndx_16s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit integer vector max index method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMax">Pointer to the output result.</param>
		/// <param name="pIndx">Pointer to the index value of the first maximum element.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaxIndx(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pMax, CudaDeviceVariable<int> pIndx, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMaxIndx_32s(pSrc.DevicePointer, pSrc.Size, pMax.DevicePointer, pIndx.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxIndx_32s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit float vector max index method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMax">Pointer to the output result.</param>
		/// <param name="pIndx">Pointer to the index value of the first maximum element.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaxIndx(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pMax, CudaDeviceVariable<int> pIndx, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMaxIndx_32f(pSrc.DevicePointer, pSrc.Size, pMax.DevicePointer, pIndx.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxIndx_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit float vector max index method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMax">Pointer to the output result.</param>
		/// <param name="pIndx">Pointer to the index value of the first maximum element.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaxIndx(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pMax, CudaDeviceVariable<int> pIndx, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMaxIndx_64f(pSrc.DevicePointer, pSrc.Size, pMax.DevicePointer, pIndx.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxIndx_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMaxAbs_16s.
		/// </summary>
		public static int MaxAbsGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMaxAbsGetBufferSize_16s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxAbsGetBufferSize_16s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMaxAbs_32s.
		/// </summary>
		public static int MaxAbsGetBufferSize(this CudaDeviceVariable<int> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMaxAbsGetBufferSize_32s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxAbsGetBufferSize_32s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit integer vector max absolute method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMaxAbs">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaxAbs(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pMaxAbs, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMaxAbs_16s(pSrc.DevicePointer, pSrc.Size, pMaxAbs.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxAbs_16s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit integer vector max absolute method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMaxAbs">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaxAbs(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pMaxAbs, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMaxAbs_32s(pSrc.DevicePointer, pSrc.Size, pMaxAbs.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxAbs_32s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMaxAbsIndx_16s.
		/// </summary>
		public static int MaxAbsIndxGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMaxAbsIndxGetBufferSize_16s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxAbsIndxGetBufferSize_16s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMaxAbsIndx_32s.
		/// </summary>
		public static int MaxAbsIndxGetBufferSize(this CudaDeviceVariable<int> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMaxAbsIndxGetBufferSize_32s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxAbsIndxGetBufferSize_32s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit integer vector max absolute index method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMaxAbs">Pointer to the output result.</param>
		/// <param name="pIndx">Pointer to the index value of the first maximum element.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaxAbsIndx(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pMaxAbs, CudaDeviceVariable<int> pIndx, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMaxAbsIndx_16s(pSrc.DevicePointer, pSrc.Size, pMaxAbs.DevicePointer, pIndx.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxAbsIndx_16s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit integer vector max absolute index method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMaxAbs">Pointer to the output result.</param>
		/// <param name="pIndx">Pointer to the index value of the first maximum element.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaxAbsIndx(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pMaxAbs, CudaDeviceVariable<int> pIndx, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Max.nppsMaxAbsIndx_32s(pSrc.DevicePointer, pSrc.Size, pMaxAbs.DevicePointer, pIndx.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaxAbsIndx_32s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMin_16s.
		/// </summary>
		public static int MinGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMinGetBufferSize_16s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinGetBufferSize_16s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMin_32s.
		/// </summary>
		public static int MinGetBufferSize(this CudaDeviceVariable<int> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMinGetBufferSize_32s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinGetBufferSize_32s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMin_32f.
		/// </summary>
		public static int MinGetBufferSize(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMinGetBufferSize_32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinGetBufferSize_32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMin_64f.
		/// </summary>
		public static int MinGetBufferSize(this CudaDeviceVariable<double> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMinGetBufferSize_64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinGetBufferSize_64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit integer vector min method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMin">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Min(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pMin, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMin_16s(pSrc.DevicePointer, pSrc.Size, pMin.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMin_16s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit integer vector min method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMin">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Min(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pMin, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMin_32s(pSrc.DevicePointer, pSrc.Size, pMin.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMin_32s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit integer vector min method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMin">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Min(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pMin, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMin_32f(pSrc.DevicePointer, pSrc.Size, pMin.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMin_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit integer vector min method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMin">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Min(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pMin, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMin_64f(pSrc.DevicePointer, pSrc.Size, pMin.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMin_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMinIndx_16s.
		/// </summary>
		public static int MinIndxGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMinIndxGetBufferSize_16s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinIndxGetBufferSize_16s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMinIndx_32s.
		/// </summary>
		public static int MinIndxGetBufferSize(this CudaDeviceVariable<int> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMinIndxGetBufferSize_32s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinIndxGetBufferSize_32s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMinIndx_32f.
		/// </summary>
		public static int MinIndxGetBufferSize(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMinIndxGetBufferSize_32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinIndxGetBufferSize_32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMinIndx_64f.
		/// </summary>
		public static int MinIndxGetBufferSize(this CudaDeviceVariable<double> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMinIndxGetBufferSize_64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinIndxGetBufferSize_64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit integer vector min index method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMin">Pointer to the output result.</param>
		/// <param name="pIndx">Pointer to the index value of the first minimum element.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MinIndx(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pMin, CudaDeviceVariable<int> pIndx, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMinIndx_16s(pSrc.DevicePointer, pSrc.Size, pMin.DevicePointer, pIndx.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinIndx_16s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit integer vector min index method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMin">Pointer to the output result.</param>
		/// <param name="pIndx">Pointer to the index value of the first minimum element.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MinIndx(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pMin, CudaDeviceVariable<int> pIndx, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMinIndx_32s(pSrc.DevicePointer, pSrc.Size, pMin.DevicePointer, pIndx.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinIndx_32s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit float vector min index method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMin">Pointer to the output result.</param>
		/// <param name="pIndx">Pointer to the index value of the first minimum element.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MinIndx(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pMin, CudaDeviceVariable<int> pIndx, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMinIndx_32f(pSrc.DevicePointer, pSrc.Size, pMin.DevicePointer, pIndx.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinIndx_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit float vector min index method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMin">Pointer to the output result.</param>
		/// <param name="pIndx">Pointer to the index value of the first minimum element.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MinIndx(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pMin, CudaDeviceVariable<int> pIndx, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMinIndx_64f(pSrc.DevicePointer, pSrc.Size, pMin.DevicePointer, pIndx.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinIndx_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMinAbs_16s.
		/// </summary>
		public static int MinAbsGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMinAbsGetBufferSize_16s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinAbsGetBufferSize_16s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMinAbs_32s.
		/// </summary>
		public static int MinAbsGetBufferSize(this CudaDeviceVariable<int> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMinAbsGetBufferSize_32s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinAbsGetBufferSize_32s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit integer vector min absolute method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMinAbs">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MinAbs(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pMinAbs, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMinAbs_16s(pSrc.DevicePointer, pSrc.Size, pMinAbs.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinAbs_16s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit integer vector min absolute method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMinAbs">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MinAbs(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pMinAbs, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMinAbs_32s(pSrc.DevicePointer, pSrc.Size, pMinAbs.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinAbs_32s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMinAbsIndx_16s.
		/// </summary>
		public static int MinAbsIndxGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMinAbsIndxGetBufferSize_16s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinAbsIndxGetBufferSize_16s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMinAbsIndx_32s.
		/// </summary>
		public static int MinAbsIndxGetBufferSize(this CudaDeviceVariable<int> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMinAbsIndxGetBufferSize_32s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinAbsIndxGetBufferSize_32s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit integer vector min absolute index method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMinAbs">Pointer to the output result.</param>
		/// <param name="pIndx">Pointer to the index value of the first minimum element.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MinAbsIndx(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pMinAbs, CudaDeviceVariable<int> pIndx, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMinAbsIndx_16s(pSrc.DevicePointer, pSrc.Size, pMinAbs.DevicePointer, pIndx.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinAbsIndx_16s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit integer vector min absolute index method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMinAbs">Pointer to the output result.</param>
		/// <param name="pIndx">Pointer to the index value of the first minimum element.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MinAbsIndx(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pMinAbs, CudaDeviceVariable<int> pIndx, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Min.nppsMinAbsIndx_32s(pSrc.DevicePointer, pSrc.Size, pMinAbs.DevicePointer, pIndx.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinAbsIndx_32s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMean_32f.
		/// </summary>
		public static int MeanGetBufferSize(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsMeanGetBufferSize_32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMeanGetBufferSize_32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMean_32fc.
		/// </summary>
		public static int MeanGetBufferSize(this CudaDeviceVariable<Npp32fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsMeanGetBufferSize_32fc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMeanGetBufferSize_32fc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMean_64f.
		/// </summary>
		public static int MeanGetBufferSize(this CudaDeviceVariable<double> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsMeanGetBufferSize_64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMeanGetBufferSize_64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMean_64fc.
		/// </summary>
		public static int MeanGetBufferSize(this CudaDeviceVariable<Npp64fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsMeanGetBufferSize_64fc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMeanGetBufferSize_64fc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMean_16s_Sfs.
		/// </summary>
		public static int MeanGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsMeanGetBufferSize_16s_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMeanGetBufferSize_16s_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMean_32s_Sfs.
		/// </summary>
		public static int MeanGetBufferSize(this CudaDeviceVariable<int> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsMeanGetBufferSize_32s_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMeanGetBufferSize_32s_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMean_16sc_Sfs.
		/// </summary>
		public static int MeanGetBufferSize(this CudaDeviceVariable<Npp16sc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsMeanGetBufferSize_16sc_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMeanGetBufferSize_16sc_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit float vector mean method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMean">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Mean(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pMean, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsMean_32f(pSrc.DevicePointer, pSrc.Size, pMean.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMean_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit float complex vector mean method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMean">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Mean(this CudaDeviceVariable<Npp32fc> pSrc, CudaDeviceVariable<Npp32fc> pMean, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsMean_32fc(pSrc.DevicePointer, pSrc.Size, pMean.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMean_32fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit double vector mean method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMean">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Mean(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pMean, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsMean_64f(pSrc.DevicePointer, pSrc.Size, pMean.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMean_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit double complex vector mean method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMean">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Mean(this CudaDeviceVariable<Npp64fc> pSrc, CudaDeviceVariable<Npp64fc> pMean, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsMean_64fc(pSrc.DevicePointer, pSrc.Size, pMean.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMean_64fc", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit short vector mean with integer scaling method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMean">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Mean(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pMean, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsMean_16s_Sfs(pSrc.DevicePointer, pSrc.Size, pMean.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMean_16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit integer vector mean with integer scaling method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMean">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Mean(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pMean, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsMean_32s_Sfs(pSrc.DevicePointer, pSrc.Size, pMean.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMean_32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit short complex vector mean with integer scaling method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMean">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		public static void Mean(this CudaDeviceVariable<Npp16sc> pSrc, CudaDeviceVariable<Npp16sc> pMean, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsMean_16sc_Sfs(pSrc.DevicePointer, pSrc.Size, pMean.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMean_16sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsStdDev_32f.
		/// </summary>
		public static int StdDevGetBufferSize(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsStdDevGetBufferSize_32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsStdDevGetBufferSize_32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsStdDev_64f.
		/// </summary>
		public static int StdDevGetBufferSize(this CudaDeviceVariable<double> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsStdDevGetBufferSize_64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsStdDevGetBufferSize_64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsStdDev_16s32s_Sfs.
		/// </summary>
		public static int StdDevGetBufferSize32s(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsStdDevGetBufferSize_16s32s_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsStdDevGetBufferSize_16s32s_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsStdDev_16s_Sfs.
		/// </summary>
		public static int StdDevGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsStdDevGetBufferSize_16s_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsStdDevGetBufferSize_16s_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit float vector standard deviation method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pStdDev">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void StdDev(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pStdDev, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsStdDev_32f(pSrc.DevicePointer, pSrc.Size, pStdDev.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsStdDev_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit float vector standard deviation method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pStdDev">Pointer to the output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void StdDev(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pStdDev, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsStdDev_64f(pSrc.DevicePointer, pSrc.Size, pStdDev.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsStdDev_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit float vector standard deviation method (return value is 32-bit)
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pStdDev">Pointer to the output result.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void StdDev(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<int> pStdDev, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsStdDev_16s32s_Sfs(pSrc.DevicePointer, pSrc.Size, pStdDev.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsStdDev_16s32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit float vector standard deviation method (return value is also 16-bit)
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pStdDev">Pointer to the output result.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void StdDev(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pStdDev, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsStdDev_16s_Sfs(pSrc.DevicePointer, pSrc.Size, pStdDev.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsStdDev_16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMeanStdDev_32f.
		/// </summary>
		public static int MeanStdDevGetBufferSize(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsMeanStdDevGetBufferSize_32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMeanStdDevGetBufferSize_32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMeanStdDev_64f.
		/// </summary>
		public static int MeanStdDevGetBufferSize(this CudaDeviceVariable<double> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsMeanStdDevGetBufferSize_64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMeanStdDevGetBufferSize_64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMeanStdDev_16s32s_Sfs.
		/// </summary>
		public static int MeanStdDevGetBufferSize32s(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsMeanStdDevGetBufferSize_16s32s_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMeanStdDevGetBufferSize_16s32s_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device scratch buffer size (in bytes) for nppsMeanStdDev_16s_Sfs.
		/// </summary>
		public static int MeanStdDevGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsMeanStdDevGetBufferSize_16s_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMeanStdDevGetBufferSize_16s_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit float vector mean and standard deviation method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMean">Pointer to the output mean value.</param>
		/// <param name="pStdDev">Pointer to the output standard deviation value.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MeanStdDev(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pMean, CudaDeviceVariable<float> pStdDev, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsMeanStdDev_32f(pSrc.DevicePointer, pSrc.Size, pMean.DevicePointer, pStdDev.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMeanStdDev_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit float vector mean and standard deviation method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMean">Pointer to the output mean value.</param>
		/// <param name="pStdDev">Pointer to the output standard deviation value.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MeanStdDev(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pMean, CudaDeviceVariable<double> pStdDev, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsMeanStdDev_64f(pSrc.DevicePointer, pSrc.Size, pMean.DevicePointer, pStdDev.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMeanStdDev_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit float vector mean and standard deviation method (return values are 32-bit)
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMean">Pointer to the output mean value.</param>
		/// <param name="pStdDev">Pointer to the output standard deviation value.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MeanStdDev(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<int> pMean, CudaDeviceVariable<int> pStdDev, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsMeanStdDev_16s32s_Sfs(pSrc.DevicePointer, pSrc.Size, pMean.DevicePointer, pStdDev.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMeanStdDev_16s32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit float vector mean and standard deviation method (return values are also 16-bit)
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMean">Pointer to the output mean value.</param>
		/// <param name="pStdDev">Pointer to the output standard deviation value.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MeanStdDev(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pMean, CudaDeviceVariable<short> pStdDev, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.MeanStdDev.nppsMeanStdDev_16s_Sfs(pSrc.DevicePointer, pSrc.Size, pMean.DevicePointer, pStdDev.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMeanStdDev_16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMinMax_8u.
		/// </summary>

		public static int MinMaxGetBufferSize(this CudaDeviceVariable<byte> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMaxGetBufferSize_8u(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMaxGetBufferSize_8u", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMinMax_16s.
		/// </summary>

		public static int MinMaxGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMaxGetBufferSize_16s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMaxGetBufferSize_16s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMinMax_16u.
		/// </summary>

		public static int MinMaxGetBufferSize(this CudaDeviceVariable<ushort> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMaxGetBufferSize_16u(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMaxGetBufferSize_16u", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMinMax_32s.
		/// </summary>

		public static int MinMaxGetBufferSize(this CudaDeviceVariable<int> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMaxGetBufferSize_32s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMaxGetBufferSize_32s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMinMax_32u.
		/// </summary>

		public static int MinMaxGetBufferSize(this CudaDeviceVariable<uint> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMaxGetBufferSize_32u(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMaxGetBufferSize_32u", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMinMax_32f.
		/// </summary>

		public static int MinMaxGetBufferSize(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMaxGetBufferSize_32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMaxGetBufferSize_32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMinMax_64f.
		/// </summary>

		public static int MinMaxGetBufferSize(this CudaDeviceVariable<double> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMaxGetBufferSize_64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMaxGetBufferSize_64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 8-bit char vector min and max method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMin">Pointer to the min output result.</param>
		/// <param name="pMax">Pointer to the max output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MinMax(this CudaDeviceVariable<byte> pSrc, CudaDeviceVariable<byte> pMin, CudaDeviceVariable<byte> pMax, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMax_8u(pSrc.DevicePointer, pSrc.Size, pMin.DevicePointer, pMax.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMax_8u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit signed short vector min and max method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMin">Pointer to the min output result.</param>
		/// <param name="pMax">Pointer to the max output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MinMax(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pMin, CudaDeviceVariable<short> pMax, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMax_16s(pSrc.DevicePointer, pSrc.Size, pMin.DevicePointer, pMax.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMax_16s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit unsigned short vector min and max method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMin">Pointer to the min output result.</param>
		/// <param name="pMax">Pointer to the max output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MinMax(this CudaDeviceVariable<ushort> pSrc, CudaDeviceVariable<ushort> pMin, CudaDeviceVariable<ushort> pMax, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMax_16u(pSrc.DevicePointer, pSrc.Size, pMin.DevicePointer, pMax.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMax_16u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit unsigned int vector min and max method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMin">Pointer to the min output result.</param>
		/// <param name="pMax">Pointer to the max output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MinMax(this CudaDeviceVariable<uint> pSrc, CudaDeviceVariable<uint> pMin, CudaDeviceVariable<uint> pMax, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMax_32u(pSrc.DevicePointer, pSrc.Size, pMin.DevicePointer, pMax.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMax_32u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit signed int vector min and max method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMin">Pointer to the min output result.</param>
		/// <param name="pMax">Pointer to the max output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MinMax(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pMin, CudaDeviceVariable<int> pMax, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMax_32s(pSrc.DevicePointer, pSrc.Size, pMin.DevicePointer, pMax.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMax_32s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit float vector min and max method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMin">Pointer to the min output result.</param>
		/// <param name="pMax">Pointer to the max output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MinMax(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pMin, CudaDeviceVariable<float> pMax, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMax_32f(pSrc.DevicePointer, pSrc.Size, pMin.DevicePointer, pMax.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMax_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit double vector min and max method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMin">Pointer to the min output result.</param>
		/// <param name="pMax">Pointer to the max output result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MinMax(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pMin, CudaDeviceVariable<double> pMax, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMax_64f(pSrc.DevicePointer, pSrc.Size, pMin.DevicePointer, pMax.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMax_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMinMaxIndx_8u.
		/// </summary>

		public static int MinMaxIndxGetBufferSize(this CudaDeviceVariable<byte> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMaxIndxGetBufferSize_8u(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMaxIndxGetBufferSize_8u", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMinMaxIndx_16s.
		/// </summary>

		public static int MinMaxIndxGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMaxIndxGetBufferSize_16s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMaxIndxGetBufferSize_16s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMinMaxIndx_16u.
		/// </summary>

		public static int MinMaxIndxGetBufferSize(this CudaDeviceVariable<ushort> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMaxIndxGetBufferSize_16u(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMaxIndxGetBufferSize_16u", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMinMaxIndx_32s.
		/// </summary>

		public static int MinMaxIndxGetBufferSize(this CudaDeviceVariable<int> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMaxIndxGetBufferSize_32s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMaxIndxGetBufferSize_32s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMinMaxIndx_32u.
		/// </summary>

		public static int MinMaxIndxGetBufferSize(this CudaDeviceVariable<uint> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMaxIndxGetBufferSize_32u(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMaxIndxGetBufferSize_32u", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMinMaxIndx_32f.
		/// </summary>

		public static int MinMaxIndxGetBufferSize(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMaxIndxGetBufferSize_32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMaxIndxGetBufferSize_32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMinMaxIndx_64f.
		/// </summary>

		public static int MinMaxIndxGetBufferSize(this CudaDeviceVariable<double> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMaxIndxGetBufferSize_64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMaxIndxGetBufferSize_64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 8-bit char vector min and max with indices method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMin">Pointer to the min output result.</param>
		/// <param name="pMinIndx">Pointer to the index of the first min value.</param>
		/// <param name="pMax">Pointer to the max output result.</param>
		/// <param name="pMaxIndx">Pointer to the index of the first max value.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MinMaxIndx(this CudaDeviceVariable<byte> pSrc, CudaDeviceVariable<byte> pMin, CudaDeviceVariable<int> pMinIndx, CudaDeviceVariable<byte> pMax, CudaDeviceVariable<int> pMaxIndx, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMaxIndx_8u(pSrc.DevicePointer, pSrc.Size, pMin.DevicePointer, pMinIndx.DevicePointer, pMax.DevicePointer, pMaxIndx.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMaxIndx_8u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit signed short vector min and max with indices method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMin">Pointer to the min output result.</param>
		/// <param name="pMinIndx">Pointer to the index of the first min value.</param>
		/// <param name="pMax">Pointer to the max output result.</param>
		/// <param name="pMaxIndx">Pointer to the index of the first max value.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MinMaxIndx(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<short> pMin, CudaDeviceVariable<int> pMinIndx, CudaDeviceVariable<short> pMax, CudaDeviceVariable<int> pMaxIndx, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMaxIndx_16s(pSrc.DevicePointer, pSrc.Size, pMin.DevicePointer, pMinIndx.DevicePointer, pMax.DevicePointer, pMaxIndx.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMaxIndx_16s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 16-bit unsigned short vector min and max with indices method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMin">Pointer to the min output result.</param>
		/// <param name="pMinIndx">Pointer to the index of the first min value.</param>
		/// <param name="pMax">Pointer to the max output result.</param>
		/// <param name="pMaxIndx">Pointer to the index of the first max value.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MinMaxIndx(this CudaDeviceVariable<ushort> pSrc, CudaDeviceVariable<ushort> pMin, CudaDeviceVariable<int> pMinIndx, CudaDeviceVariable<ushort> pMax, CudaDeviceVariable<int> pMaxIndx, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMaxIndx_16u(pSrc.DevicePointer, pSrc.Size, pMin.DevicePointer, pMinIndx.DevicePointer, pMax.DevicePointer, pMaxIndx.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMaxIndx_16u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit signed short vector min and max with indices method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMin">Pointer to the min output result.</param>
		/// <param name="pMinIndx">Pointer to the index of the first min value.</param>
		/// <param name="pMax">Pointer to the max output result.</param>
		/// <param name="pMaxIndx">Pointer to the index of the first max value.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MinMaxIndx(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pMin, CudaDeviceVariable<int> pMinIndx, CudaDeviceVariable<int> pMax, CudaDeviceVariable<int> pMaxIndx, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMaxIndx_32s(pSrc.DevicePointer, pSrc.Size, pMin.DevicePointer, pMinIndx.DevicePointer, pMax.DevicePointer, pMaxIndx.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMaxIndx_32s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit unsigned short vector min and max with indices method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMin">Pointer to the min output result.</param>
		/// <param name="pMinIndx">Pointer to the index of the first min value.</param>
		/// <param name="pMax">Pointer to the max output result.</param>
		/// <param name="pMaxIndx">Pointer to the index of the first max value.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MinMaxIndx(this CudaDeviceVariable<uint> pSrc, CudaDeviceVariable<uint> pMin, CudaDeviceVariable<int> pMinIndx, CudaDeviceVariable<uint> pMax, CudaDeviceVariable<int> pMaxIndx, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMaxIndx_32u(pSrc.DevicePointer, pSrc.Size, pMin.DevicePointer, pMinIndx.DevicePointer, pMax.DevicePointer, pMaxIndx.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMaxIndx_32u", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 32-bit float vector min and max with indices method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMin">Pointer to the min output result.</param>
		/// <param name="pMinIndx">Pointer to the index of the first min value.</param>
		/// <param name="pMax">Pointer to the max output result.</param>
		/// <param name="pMaxIndx">Pointer to the index of the first max value.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MinMaxIndx(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pMin, CudaDeviceVariable<int> pMinIndx, CudaDeviceVariable<float> pMax, CudaDeviceVariable<int> pMaxIndx, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMaxIndx_32f(pSrc.DevicePointer, pSrc.Size, pMin.DevicePointer, pMinIndx.DevicePointer, pMax.DevicePointer, pMaxIndx.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMaxIndx_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// 64-bit float vector min and max with indices method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pMin">Pointer to the min output result.</param>
		/// <param name="pMinIndx">Pointer to the index of the first min value.</param>
		/// <param name="pMax">Pointer to the max output result.</param>
		/// <param name="pMaxIndx">Pointer to the index of the first max value.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MinMaxIndx(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pMin, CudaDeviceVariable<int> pMinIndx, CudaDeviceVariable<double> pMax, CudaDeviceVariable<int> pMaxIndx, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MinMaxIndex.nppsMinMaxIndx_64f(pSrc.DevicePointer, pSrc.Size, pMin.DevicePointer, pMinIndx.DevicePointer, pMax.DevicePointer, pMaxIndx.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMinMaxIndx_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNorm_Inf_32f.
		/// </summary>

		public static int NormInfGetBufferSize(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNormInfGetBufferSize_32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormInfGetBufferSize_32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit float vector C norm method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Norm_Inf(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNorm_Inf_32f(pSrc.DevicePointer, pSrc.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNorm_Inf_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNorm_Inf_64f.
		/// </summary>

		public static int NormInfGetBufferSize(this CudaDeviceVariable<double> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNormInfGetBufferSize_64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormInfGetBufferSize_64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 64-bit float vector C norm method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Norm_Inf(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNorm_Inf_64f(pSrc.DevicePointer, pSrc.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNorm_Inf_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNorm_Inf_16s32f.
		/// </summary>

		public static int NormInfGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNormInfGetBufferSize_16s32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormInfGetBufferSize_16s32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer vector C norm method, return value is 32-bit float.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Norm_Inf(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<float> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNorm_Inf_16s32f(pSrc.DevicePointer, pSrc.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNorm_Inf_16s32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNorm_Inf_32fc32f.
		/// </summary>

		public static int NormInfGetBufferSize(this CudaDeviceVariable<Npp32fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNormInfGetBufferSize_32fc32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormInfGetBufferSize_32fc32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit float complex vector C norm method, return value is 32-bit float.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Norm_Inf(this CudaDeviceVariable<Npp32fc> pSrc, CudaDeviceVariable<float> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNorm_Inf_32fc32f(pSrc.DevicePointer, pSrc.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNorm_Inf_32fc32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNorm_Inf_64fc64f.
		/// </summary>

		public static int NormInfGetBufferSize(this CudaDeviceVariable<Npp64fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNormInfGetBufferSize_64fc64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormInfGetBufferSize_64fc64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 64-bit float complex vector C norm method, return value is 64-bit float.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Norm_Inf(this CudaDeviceVariable<Npp64fc> pSrc, CudaDeviceVariable<double> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNorm_Inf_64fc64f(pSrc.DevicePointer, pSrc.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNorm_Inf_64fc64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNorm_Inf_16s32s_Sfs.
		/// </summary>

		public static int NormInfGetBufferSize32s(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNormInfGetBufferSize_16s32s_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormInfGetBufferSize_16s32s_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer vector C norm method, return value is 32-bit signed integer.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Norm_Inf(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<int> pNorm, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNorm_Inf_16s32s_Sfs(pSrc.DevicePointer, pSrc.Size, pNorm.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNorm_Inf_16s32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNorm_L1_32f.
		/// </summary>

		public static int NormL1GetBufferSize(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNormL1GetBufferSize_32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormL1GetBufferSize_32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit float vector L1 norm method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Norm_L1(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNorm_L1_32f(pSrc.DevicePointer, pSrc.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNorm_L1_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNorm_L1_64f.
		/// </summary>

		public static int NormL1GetBufferSize(this CudaDeviceVariable<double> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNormL1GetBufferSize_64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormL1GetBufferSize_64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 64-bit float vector L1 norm method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Norm_L1(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNorm_L1_64f(pSrc.DevicePointer, pSrc.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNorm_L1_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNorm_L1_16s32f.
		/// </summary>

		public static int NormL1GetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNormL1GetBufferSize_16s32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormL1GetBufferSize_16s32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer vector L1 norm method, return value is 32-bit float.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the L1 norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Norm_L1(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<float> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNorm_L1_16s32f(pSrc.DevicePointer, pSrc.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNorm_L1_16s32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNorm_L1_32fc64f.
		/// </summary>

		public static int NormL1GetBufferSize(this CudaDeviceVariable<Npp32fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNormL1GetBufferSize_32fc64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormL1GetBufferSize_32fc64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit float complex vector L1 norm method, return value is 64-bit float.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Norm_L1(this CudaDeviceVariable<Npp32fc> pSrc, CudaDeviceVariable<double> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNorm_L1_32fc64f(pSrc.DevicePointer, pSrc.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNorm_L1_32fc64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNorm_L1_64fc64f.
		/// </summary>

		public static int NormL1GetBufferSize(this CudaDeviceVariable<Npp64fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNormL1GetBufferSize_64fc64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormL1GetBufferSize_64fc64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 64-bit float complex vector L1 norm method, return value is 64-bit float.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Norm_L1(this CudaDeviceVariable<Npp64fc> pSrc, CudaDeviceVariable<double> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNorm_L1_64fc64f(pSrc.DevicePointer, pSrc.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNorm_L1_64fc64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNorm_L1_16s32s_Sfs.
		/// </summary>

		public static int NormL1GetBufferSize32s(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNormL1GetBufferSize_16s32s_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormL1GetBufferSize_16s32s_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer vector L1 norm method, return value is 32-bit signed integer.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Norm_L1(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<int> pNorm, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNorm_L1_16s32s_Sfs(pSrc.DevicePointer, pSrc.Size, pNorm.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNorm_L1_16s32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNorm_L1_16s64s_Sfs.
		/// </summary>

		public static int NormL1GetBufferSize64s(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNormL1GetBufferSize_16s64s_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormL1GetBufferSize_16s64s_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer vector L1 norm method, return value is 64-bit signed integer.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Norm_L1(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<long> pNorm, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNorm_L1_16s64s_Sfs(pSrc.DevicePointer, pSrc.Size, pNorm.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNorm_L1_16s64s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNorm_L2_32f.
		/// </summary>

		public static int NormL2GetBufferSize(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNormL2GetBufferSize_32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormL2GetBufferSize_32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit float vector L2 norm method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Norm_L2(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNorm_L2_32f(pSrc.DevicePointer, pSrc.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNorm_L2_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNorm_L2_64f.
		/// </summary>

		public static int NormL2GetBufferSize(this CudaDeviceVariable<double> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNormL2GetBufferSize_64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormL2GetBufferSize_64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 64-bit float vector L2 norm method
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Norm_L2(this CudaDeviceVariable<double> pSrc, CudaDeviceVariable<double> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNorm_L2_64f(pSrc.DevicePointer, pSrc.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNorm_L2_64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNorm_L2_16s32f.
		/// </summary>

		public static int NormL2GetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNormL2GetBufferSize_16s32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormL2GetBufferSize_16s32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer vector L2 norm method, return value is 32-bit float.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Norm_L2(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<float> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNorm_L2_16s32f(pSrc.DevicePointer, pSrc.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNorm_L2_16s32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNorm_L2_32fc64f.
		/// </summary>

		public static int NormL2GetBufferSize(this CudaDeviceVariable<Npp32fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNormL2GetBufferSize_32fc64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormL2GetBufferSize_32fc64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit float complex vector L2 norm method, return value is 64-bit float.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Norm_L2(this CudaDeviceVariable<Npp32fc> pSrc, CudaDeviceVariable<double> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNorm_L2_32fc64f(pSrc.DevicePointer, pSrc.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNorm_L2_32fc64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNorm_L2_64fc64f.
		/// </summary>

		public static int NormL2GetBufferSize(this CudaDeviceVariable<Npp64fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNormL2GetBufferSize_64fc64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormL2GetBufferSize_64fc64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 64-bit float complex vector L2 norm method, return value is 64-bit float.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Norm_L2(this CudaDeviceVariable<Npp64fc> pSrc, CudaDeviceVariable<double> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNorm_L2_64fc64f(pSrc.DevicePointer, pSrc.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNorm_L2_64fc64f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNorm_L2_16s32s_Sfs.
		/// </summary>

		public static int NormL2GetBufferSize32s(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNormL2GetBufferSize_16s32s_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormL2GetBufferSize_16s32s_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer vector L2 norm method, return value is 32-bit signed integer.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Norm_L2(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<int> pNorm, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNorm_L2_16s32s_Sfs(pSrc.DevicePointer, pSrc.Size, pNorm.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNorm_L2_16s32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNorm_L2Sqr_16s64s_Sfs.
		/// </summary>

		public static int NormL2SqrGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNormL2SqrGetBufferSize_16s64s_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormL2SqrGetBufferSize_16s64s_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer vector L2 Square norm method, return value is 64-bit signed integer.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void Norm_L2Sqr(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<long> pNorm, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.Norm.nppsNorm_L2Sqr_16s64s_Sfs(pSrc.DevicePointer, pSrc.Size, pNorm.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNorm_L2Sqr_16s64s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNormDiff_Inf_32f.
		/// </summary>

		public static int NormDiffInfGetBufferSize(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiffInfGetBufferSize_32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiffInfGetBufferSize_32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit float C norm method on two vectors' difference
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void NormDiff_Inf(this CudaDeviceVariable<float> pSrc1, CudaDeviceVariable<float> pSrc2, CudaDeviceVariable<float> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiff_Inf_32f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiff_Inf_32f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNormDiff_Inf_64f.
		/// </summary>

		public static int NormDiffInfGetBufferSize(this CudaDeviceVariable<double> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiffInfGetBufferSize_64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiffInfGetBufferSize_64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 64-bit float C norm method on two vectors' difference
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void NormDiff_Inf(this CudaDeviceVariable<double> pSrc1, CudaDeviceVariable<double> pSrc2, CudaDeviceVariable<double> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiff_Inf_64f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiff_Inf_64f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNormDiff_Inf_16s32f.
		/// </summary>

		public static int NormDiffInfGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiffInfGetBufferSize_16s32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiffInfGetBufferSize_16s32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer C norm method on two vectors' difference, return value is 32-bit float.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void NormDiff_Inf(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<float> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiff_Inf_16s32f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiff_Inf_16s32f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNormDiff_Inf_32fc32f.
		/// </summary>

		public static int NormDiffInfGetBufferSize(this CudaDeviceVariable<Npp32fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiffInfGetBufferSize_32fc32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiffInfGetBufferSize_32fc32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit float complex C norm method on two vectors' difference, return value is 32-bit float.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void NormDiff_Inf(this CudaDeviceVariable<Npp32fc> pSrc1, CudaDeviceVariable<Npp32fc> pSrc2, CudaDeviceVariable<float> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiff_Inf_32fc32f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiff_Inf_32fc32f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNormDiff_Inf_64fc64f.
		/// </summary>

		public static int NormDiffInfGetBufferSize(this CudaDeviceVariable<Npp64fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiffInfGetBufferSize_64fc64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiffInfGetBufferSize_64fc64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 64-bit float complex C norm method on two vectors' difference, return value is 64-bit float.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void NormDiff_Inf(this CudaDeviceVariable<Npp64fc> pSrc1, CudaDeviceVariable<Npp64fc> pSrc2, CudaDeviceVariable<double> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiff_Inf_64fc64f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiff_Inf_64fc64f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNormDiff_Inf_16s32s_Sfs.
		/// </summary>

		public static int NormDiffInfGetBufferSize32s(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiffInfGetBufferSize_16s32s_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiffInfGetBufferSize_16s32s_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer C norm method on two vectors' difference, return value is 32-bit signed integer.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void NormDiff_Inf(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<int> pNorm, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiff_Inf_16s32s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pNorm.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiff_Inf_16s32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNormDiff_L1_32f.
		/// </summary>

		public static int NormDiffL1GetBufferSize(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiffL1GetBufferSize_32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiffL1GetBufferSize_32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit float L1 norm method on two vectors' difference
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void NormDiff_L1(this CudaDeviceVariable<float> pSrc1, CudaDeviceVariable<float> pSrc2, CudaDeviceVariable<float> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiff_L1_32f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiff_L1_32f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNormDiff_L1_64f.
		/// </summary>

		public static int NormDiffL1GetBufferSize(this CudaDeviceVariable<double> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiffL1GetBufferSize_64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiffL1GetBufferSize_64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 64-bit float L1 norm method on two vectors' difference
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void NormDiff_L1(this CudaDeviceVariable<double> pSrc1, CudaDeviceVariable<double> pSrc2, CudaDeviceVariable<double> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiff_L1_64f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiff_L1_64f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNormDiff_L1_16s32f.
		/// </summary>

		public static int NormDiffL1GetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiffL1GetBufferSize_16s32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiffL1GetBufferSize_16s32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer L1 norm method on two vectors' difference, return value is 32-bit float.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the L1 norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void NormDiff_L1(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<float> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiff_L1_16s32f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiff_L1_16s32f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNormDiff_L1_32fc64f.
		/// </summary>

		public static int NormDiffL1GetBufferSize(this CudaDeviceVariable<Npp32fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiffL1GetBufferSize_32fc64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiffL1GetBufferSize_32fc64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit float complex L1 norm method on two vectors' difference, return value is 64-bit float.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void NormDiff_L1(this CudaDeviceVariable<Npp32fc> pSrc1, CudaDeviceVariable<Npp32fc> pSrc2, CudaDeviceVariable<double> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiff_L1_32fc64f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiff_L1_32fc64f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNormDiff_L1_64fc64f.
		/// </summary>

		public static int NormDiffL1GetBufferSize(this CudaDeviceVariable<Npp64fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiffL1GetBufferSize_64fc64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiffL1GetBufferSize_64fc64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 64-bit float complex L1 norm method on two vectors' difference, return value is 64-bit float.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void NormDiff_L1(this CudaDeviceVariable<Npp64fc> pSrc1, CudaDeviceVariable<Npp64fc> pSrc2, CudaDeviceVariable<double> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiff_L1_64fc64f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiff_L1_64fc64f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNormDiff_L1_16s32s_Sfs.
		/// </summary>

		public static int NormDiffL1GetBufferSize32s(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiffL1GetBufferSize_16s32s_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiffL1GetBufferSize_16s32s_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer L1 norm method on two vectors' difference, return value is 32-bit signed integer.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer..</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void NormDiff_L1(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<int> pNorm, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiff_L1_16s32s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pNorm.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiff_L1_16s32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNormDiff_L1_16s64s_Sfs.
		/// </summary>

		public static int NormDiffL1GetBufferSize64s(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiffL1GetBufferSize_16s64s_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiffL1GetBufferSize_16s64s_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer L1 norm method on two vectors' difference, return value is 64-bit signed integer.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void NormDiff_L1(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<long> pNorm, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiff_L1_16s64s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pNorm.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiff_L1_16s64s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNormDiff_L2_32f.
		/// </summary>

		public static int NormDiffL2GetBufferSize(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiffL2GetBufferSize_32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiffL2GetBufferSize_32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit float L2 norm method on two vectors' difference
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void NormDiff_L2(this CudaDeviceVariable<float> pSrc1, CudaDeviceVariable<float> pSrc2, CudaDeviceVariable<float> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiff_L2_32f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiff_L2_32f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNormDiff_L2_64f.
		/// </summary>

		public static int NormDiffL2GetBufferSize(this CudaDeviceVariable<double> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiffL2GetBufferSize_64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiffL2GetBufferSize_64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 64-bit float L2 norm method on two vectors' difference
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void NormDiff_L2(this CudaDeviceVariable<double> pSrc1, CudaDeviceVariable<double> pSrc2, CudaDeviceVariable<double> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiff_L2_64f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiff_L2_64f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNormDiff_L2_16s32f.
		/// </summary>

		public static int NormDiffL2GetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiffL2GetBufferSize_16s32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiffL2GetBufferSize_16s32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer L2 norm method on two vectors' difference, return value is 32-bit float.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void NormDiff_L2(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<float> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiff_L2_16s32f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiff_L2_16s32f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNormDiff_L2_32fc64f.
		/// </summary>

		public static int NormDiffL2GetBufferSize(this CudaDeviceVariable<Npp32fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiffL2GetBufferSize_32fc64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiffL2GetBufferSize_32fc64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit float complex L2 norm method on two vectors' difference, return value is 64-bit float.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void NormDiff_L2(this CudaDeviceVariable<Npp32fc> pSrc1, CudaDeviceVariable<Npp32fc> pSrc2, CudaDeviceVariable<double> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiff_L2_32fc64f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiff_L2_32fc64f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNormDiff_L2_64fc64f.
		/// </summary>

		public static int NormDiffL2GetBufferSize(this CudaDeviceVariable<Npp64fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiffL2GetBufferSize_64fc64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiffL2GetBufferSize_64fc64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 64-bit float complex L2 norm method on two vectors' difference, return value is 64-bit float.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void NormDiff_L2(this CudaDeviceVariable<Npp64fc> pSrc1, CudaDeviceVariable<Npp64fc> pSrc2, CudaDeviceVariable<double> pNorm, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiff_L2_64fc64f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pNorm.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiff_L2_64fc64f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNormDiff_L2_16s32s_Sfs.
		/// </summary>

		public static int NormDiffL2GetBufferSize32s(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiffL2GetBufferSize_16s32s_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiffL2GetBufferSize_16s32s_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer L2 norm method on two vectors' difference, return value is 32-bit signed integer.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void NormDiff_L2(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<int> pNorm, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiff_L2_16s32s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pNorm.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiff_L2_16s32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsNormDiff_L2Sqr_16s64s_Sfs.
		/// </summary>

		public static int NormDiffL2SqrGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiffL2SqrGetBufferSize_16s64s_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiffL2SqrGetBufferSize_16s64s_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer L2 Square norm method on two vectors' difference, return value is 64-bit signed integer.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pNorm">Pointer to the norm result.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void NormDiff_L2Sqr(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<long> pNorm, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.NormDiff.nppsNormDiff_L2Sqr_16s64s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pNorm.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsNormDiff_L2Sqr_16s64s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_32f.
		/// </summary>

		public static int DotProdGetBufferSize(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit float dot product method, return value is 32-bit float.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<float> pSrc1, CudaDeviceVariable<float> pSrc2, CudaDeviceVariable<float> pDp, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_32f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_32f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_32fc.
		/// </summary>

		public static int DotProdGetBufferSize(this CudaDeviceVariable<Npp32fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_32fc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_32fc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit float complex dot product method, return value is 32-bit float complex.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<Npp32fc> pSrc1, CudaDeviceVariable<Npp32fc> pSrc2, CudaDeviceVariable<Npp32fc> pDp, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_32fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_32fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_32f32fc.
		/// </summary>

		public static int DotProdGetBufferSize32fc(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_32f32fc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_32f32fc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit float and 32-bit float complex dot product method, return value is 32-bit float complex.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<float> pSrc1, CudaDeviceVariable<Npp32fc> pSrc2, CudaDeviceVariable<Npp32fc> pDp, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_32f32fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_32f32fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_32f64f.
		/// </summary>

		public static int DotProdGetBufferSize64f(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_32f64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_32f64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit float dot product method, return value is 64-bit float.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<float> pSrc1, CudaDeviceVariable<float> pSrc2, CudaDeviceVariable<double> pDp, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_32f64f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_32f64f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_32fc64fc.
		/// </summary>

		public static int DotProdGetBufferSize64fc(this CudaDeviceVariable<Npp32fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_32fc64fc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_32fc64fc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit float complex dot product method, return value is 64-bit float complex.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<Npp32fc> pSrc1, CudaDeviceVariable<Npp32fc> pSrc2, CudaDeviceVariable<Npp64fc> pDp, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_32fc64fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_32fc64fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_32f32fc64fc.
		/// </summary>

		public static int DotProdGetBufferSize64fc(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_32f32fc64fc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_32f32fc64fc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit float and 32-bit float complex dot product method, return value is 64-bit float complex.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<float> pSrc1, CudaDeviceVariable<Npp32fc> pSrc2, CudaDeviceVariable<Npp64fc> pDp, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_32f32fc64fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_32f32fc64fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_64f.
		/// </summary>

		public static int DotProdGetBufferSize(this CudaDeviceVariable<double> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 64-bit float dot product method, return value is 64-bit float.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<double> pSrc1, CudaDeviceVariable<double> pSrc2, CudaDeviceVariable<double> pDp, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_64f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_64f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_64fc.
		/// </summary>

		public static int DotProdGetBufferSize(this CudaDeviceVariable<Npp64fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_64fc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_64fc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 64-bit float complex dot product method, return value is 64-bit float complex.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<Npp64fc> pSrc1, CudaDeviceVariable<Npp64fc> pSrc2, CudaDeviceVariable<Npp64fc> pDp, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_64fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_64fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_64f64fc.
		/// </summary>

		public static int DotProdGetBufferSize64fc(this CudaDeviceVariable<double> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_64f64fc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_64f64fc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 64-bit float and 64-bit float complex dot product method, return value is 64-bit float complex.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<double> pSrc1, CudaDeviceVariable<Npp64fc> pSrc2, CudaDeviceVariable<Npp64fc> pDp, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_64f64fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_64f64fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_16s64s.
		/// </summary>

		public static int DotProdGetBufferSize64s(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_16s64s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_16s64s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer dot product method, return value is 64-bit signed integer.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<long> pDp, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_16s64s(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_16s64s", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_16sc64sc.
		/// </summary>

		public static int DotProdGetBufferSize64sc(this CudaDeviceVariable<Npp16sc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_16sc64sc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_16sc64sc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer complex dot product method, return value is 64-bit signed integer complex.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<Npp16sc> pSrc1, CudaDeviceVariable<Npp16sc> pSrc2, CudaDeviceVariable<Npp64sc> pDp, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_16sc64sc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_16sc64sc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_16s16sc64sc.
		/// </summary>

		public static int DotProdGetBufferSize16sc64sc(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_16s16sc64sc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_16s16sc64sc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer and 16-bit signed short integer short dot product method, return value is 64-bit signed integer complex.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<Npp16sc> pSrc2, CudaDeviceVariable<Npp64sc> pDp, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_16s16sc64sc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_16s16sc64sc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_16s32f.
		/// </summary>

		public static int DotProdGetBufferSize32f(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_16s32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_16s32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer dot product method, return value is 32-bit float.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<float> pDp, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_16s32f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_16s32f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_16sc32fc.
		/// </summary>

		public static int DotProdGetBufferSize32fc(this CudaDeviceVariable<Npp16sc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_16sc32fc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_16sc32fc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer complex dot product method, return value is 32-bit float complex.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<Npp16sc> pSrc1, CudaDeviceVariable<Npp16sc> pSrc2, CudaDeviceVariable<Npp32fc> pDp, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_16sc32fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_16sc32fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_16s16sc32fc.
		/// </summary>

		public static int DotProdGetBufferSize32fc(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_16s16sc32fc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_16s16sc32fc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer and 16-bit signed short integer complex dot product method, return value is 32-bit float complex.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<Npp16sc> pSrc2, CudaDeviceVariable<Npp32fc> pDp, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_16s16sc32fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_16s16sc32fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_16s_Sfs.
		/// </summary>

		public static int DotProdGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_16s_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_16s_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer dot product method, return value is 16-bit signed short integer.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<short> pDp, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_16s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_16s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_16sc_Sfs.
		/// </summary>

		public static int DotProdGetBufferSize(this CudaDeviceVariable<Npp16sc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_16sc_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_16sc_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer complex dot product method, return value is 16-bit signed short integer complex.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<Npp16sc> pSrc1, CudaDeviceVariable<Npp16sc> pSrc2, CudaDeviceVariable<Npp16sc> pDp, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_16sc_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_16sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_32s_Sfs.
		/// </summary>

		public static int DotProdGetBufferSize(this CudaDeviceVariable<int> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_32s_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_32s_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit signed integer dot product method, return value is 32-bit signed integer.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<int> pSrc1, CudaDeviceVariable<int> pSrc2, CudaDeviceVariable<int> pDp, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_32s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_32sc_Sfs.
		/// </summary>

		public static int DotProdGetBufferSize(this CudaDeviceVariable<Npp32sc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_32sc_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_32sc_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit signed integer complex dot product method, return value is 32-bit signed integer complex.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<Npp32sc> pSrc1, CudaDeviceVariable<Npp32sc> pSrc2, CudaDeviceVariable<Npp32sc> pDp, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_32sc_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_32sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_16s32s_Sfs.
		/// </summary>

		public static int DotProdGetBufferSize32s(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_16s32s_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_16s32s_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer dot product method, return value is 32-bit signed integer.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result. </param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<int> pDp, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_16s32s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_16s32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_16s16sc32sc_Sfs.
		/// </summary>

		public static int DotProdGetBufferSize32sc(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_16s16sc32sc_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_16s16sc32sc_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer and 16-bit signed short integer complex dot product method, return value is 32-bit signed integer complex.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<Npp16sc> pSrc2, CudaDeviceVariable<Npp32sc> pDp, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_16s16sc32sc_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_16s16sc32sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_16s32s32s_Sfs.
		/// </summary>

		public static int DotProdGetBufferSize32s32s(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_16s32s32s_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_16s32s32s_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer and 32-bit signed integer dot product method, return value is 32-bit signed integer.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<int> pSrc2, CudaDeviceVariable<int> pDp, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_16s32s32s_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_16s32s32s_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_16s16sc_Sfs.
		/// </summary>

		public static int DotProdGetBufferSize16sc(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_16s16sc_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_16s16sc_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer and 16-bit signed short integer complex dot product method, return value is 16-bit signed short integer complex.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<Npp16sc> pSrc2, CudaDeviceVariable<Npp16sc> pDp, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_16s16sc_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_16s16sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_16sc32sc_Sfs.
		/// </summary>

		public static int DotProdGetBufferSize32sc(this CudaDeviceVariable<Npp16sc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_16sc32sc_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_16sc32sc_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer complex dot product method, return value is 32-bit signed integer complex.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<Npp16sc> pSrc1, CudaDeviceVariable<Npp16sc> pSrc2, CudaDeviceVariable<Npp32sc> pDp, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_16sc32sc_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_16sc32sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsDotProd_32s32sc_Sfs.
		/// </summary>

		public static int DotProdGetBufferSize32sc(this CudaDeviceVariable<int> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProdGetBufferSize_32s32sc_Sfs(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProdGetBufferSize_32s32sc_Sfs", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit signed short integer and 32-bit signed short integer complex dot product method, return value is 32-bit signed integer complex.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDp">Pointer to the dot product result.</param>
		/// <param name="nScaleFactor">Integer Result Scaling.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void DotProd(this CudaDeviceVariable<int> pSrc1, CudaDeviceVariable<Npp32sc> pSrc2, CudaDeviceVariable<Npp32sc> pDp, CudaDeviceVariable<byte> pDeviceBuffer, int nScaleFactor)
		{
			NppStatus status = NPPNativeMethods.NPPs.DotProduct.nppsDotProd_32s32sc_Sfs(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDp.DevicePointer, nScaleFactor, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsDotProd_32s32sc_Sfs", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsCountInRange_32s.
		/// </summary>

		public static int CountInRangeGetBufferSize(this CudaDeviceVariable<int> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.CountInRange.nppsCountInRangeGetBufferSize_32s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsCountInRangeGetBufferSize_32s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Computes the number of elements whose values fall into the specified range on a 32-bit signed integer array.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pCounts">Pointer to the number of elements.</param>
		/// <param name="nLowerBound">Lower bound of the specified range.</param>
		/// <param name="nUpperBound">Upper bound of the specified range.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void CountInRange(this CudaDeviceVariable<int> pSrc, CudaDeviceVariable<int> pCounts, CudaDeviceVariable<byte> pDeviceBuffer, int nLowerBound, int nUpperBound)
		{
			NppStatus status = NPPNativeMethods.NPPs.CountInRange.nppsCountInRange_32s(pSrc.DevicePointer, pSrc.Size, pCounts.DevicePointer, nLowerBound, nUpperBound, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsCountInRange_32s", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsZeroCrossing_16s32f.
		/// </summary>

		public static int ZeroCrossingGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.ZeroCrossing.nppsZeroCrossingGetBufferSize_16s32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsZeroCrossingGetBufferSize_16s32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 16-bit signed short integer zero crossing method, return value is 32-bit floating point.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pValZC">Pointer to the output result.</param>
		/// <param name="tZCType">Type of the zero crossing measure: nppZCR, nppZCXor or nppZCC.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void ZeroCrossing(this CudaDeviceVariable<short> pSrc, CudaDeviceVariable<float> pValZC, CudaDeviceVariable<byte> pDeviceBuffer, NppsZCType tZCType)
		{
			NppStatus status = NPPNativeMethods.NPPs.ZeroCrossing.nppsZeroCrossing_16s32f(pSrc.DevicePointer, pSrc.Size, pValZC.DevicePointer, tZCType, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsZeroCrossing_16s32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsZeroCrossing_32f.
		/// </summary>

		public static int ZeroCrossingGetBufferSize(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.ZeroCrossing.nppsZeroCrossingGetBufferSize_32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsZeroCrossingGetBufferSize_32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 32-bit floating-point zero crossing method, return value is 32-bit floating point.
		/// </summary>
		/// <param name="pSrc">Source signal pointer.</param>
		/// <param name="pValZC">Pointer to the output result.</param>
		/// <param name="tZCType">Type of the zero crossing measure: nppZCR, nppZCXor or nppZCC.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void ZeroCrossing(this CudaDeviceVariable<float> pSrc, CudaDeviceVariable<float> pValZC, CudaDeviceVariable<byte> pDeviceBuffer, NppsZCType tZCType)
		{
			NppStatus status = NPPNativeMethods.NPPs.ZeroCrossing.nppsZeroCrossing_32f(pSrc.DevicePointer, pSrc.Size, pValZC.DevicePointer, tZCType, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsZeroCrossing_32f", status));
			NPPException.CheckNppStatus(status, pSrc);
		}

		//new in Cuda 6.0

				/// <summary>
		/// 8-bit unsigned char maximum method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumError(this CudaDeviceVariable<byte> pSrc1, CudaDeviceVariable<byte> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumError_8u(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumError_8u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 8-bit signed char maximum method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumError(this CudaDeviceVariable<sbyte> pSrc1, CudaDeviceVariable<sbyte> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumError_8s(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumError_8s", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit unsigned short integer maximum method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumError(this CudaDeviceVariable<ushort> pSrc1, CudaDeviceVariable<ushort> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumError_16u(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumError_16u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit signed short integer maximum method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumError(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumError_16s(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumError_16s", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit unsigned short complex integer maximum method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumError(this CudaDeviceVariable<Npp16sc> pSrc1, CudaDeviceVariable<Npp16sc> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumError_16sc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumError_16sc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit unsigned short integer maximum method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumError(this CudaDeviceVariable<uint> pSrc1, CudaDeviceVariable<uint> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumError_32u(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumError_32u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit signed short integer maximum method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumError(this CudaDeviceVariable<int> pSrc1, CudaDeviceVariable<int> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumError_32s(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumError_32s", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit unsigned short complex integer maximum method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumError(this CudaDeviceVariable<Npp32sc> pSrc1, CudaDeviceVariable<Npp32sc> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumError_32sc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumError_32sc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit signed short integer maximum method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumError(this CudaDeviceVariable<long> pSrc1, CudaDeviceVariable<long> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumError_64s(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumError_64s", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit unsigned short complex integer maximum method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumError(this CudaDeviceVariable<Npp64sc> pSrc1, CudaDeviceVariable<Npp64sc> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumError_64sc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumError_64sc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit floating point maximum method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumError(this CudaDeviceVariable<float> pSrc1, CudaDeviceVariable<float> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumError_32f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumError_32f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit floating point complex maximum method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumError(this CudaDeviceVariable<Npp32fc> pSrc1, CudaDeviceVariable<Npp32fc> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumError_32fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumError_32fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit floating point maximum method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumError(this CudaDeviceVariable<double> pSrc1, CudaDeviceVariable<double> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumError_64f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumError_64f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit floating point complex maximum method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumError(this CudaDeviceVariable<Npp64fc> pSrc1, CudaDeviceVariable<Npp64fc> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumError_64fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumError_64fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumError_8u.
		/// </summary>
		public static int MaximumErrorGetBufferSize(this CudaDeviceVariable<byte> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumErrorGetBufferSize_8u(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumErrorGetBufferSize_8u", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumError_8s.
		/// </summary>
		public static int MaximumErrorGetBufferSize(this CudaDeviceVariable<sbyte> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumErrorGetBufferSize_8s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumErrorGetBufferSize_8s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumError_16u.
		/// </summary>
		public static int MaximumErrorGetBufferSize(this CudaDeviceVariable<ushort> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumErrorGetBufferSize_16u(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumErrorGetBufferSize_16u", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumError_16s.
		/// </summary>
		public static int MaximumErrorGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumErrorGetBufferSize_16s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumErrorGetBufferSize_16s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumError_16sc.
		/// </summary>
		public static int MaximumErrorGetBufferSize(this CudaDeviceVariable<Npp16sc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumErrorGetBufferSize_16sc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumErrorGetBufferSize_16sc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumError_32u.
		/// </summary>
		public static int MaximumErrorGetBufferSize(this CudaDeviceVariable<uint> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumErrorGetBufferSize_32u(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumErrorGetBufferSize_32u", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumError_32s.
		/// </summary>
		public static int MaximumErrorGetBufferSize(this CudaDeviceVariable<int> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumErrorGetBufferSize_32s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumErrorGetBufferSize_32s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumError_32sc.
		/// </summary>
		public static int MaximumErrorGetBufferSize(this CudaDeviceVariable<Npp32sc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumErrorGetBufferSize_32sc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumErrorGetBufferSize_32sc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumError_64s.
		/// </summary>
		public static int MaximumErrorGetBufferSize(this CudaDeviceVariable<long> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumErrorGetBufferSize_64s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumErrorGetBufferSize_64s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumError_64sc.
		/// </summary>
		public static int MaximumErrorGetBufferSize(this CudaDeviceVariable<Npp64sc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumErrorGetBufferSize_64sc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumErrorGetBufferSize_64sc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumError_32f.
		/// </summary>
		public static int MaximumErrorGetBufferSize(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumErrorGetBufferSize_32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumErrorGetBufferSize_32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumError_32fc.
		/// </summary>
		public static int MaximumErrorGetBufferSize(this CudaDeviceVariable<Npp32fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumErrorGetBufferSize_32fc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumErrorGetBufferSize_32fc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumError_64f.
		/// </summary>
		public static int MaximumErrorGetBufferSize(this CudaDeviceVariable<double> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumErrorGetBufferSize_64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumErrorGetBufferSize_64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumError_64fc.
		/// </summary>
		public static int MaximumErrorGetBufferSize(this CudaDeviceVariable<Npp64fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumError.nppsMaximumErrorGetBufferSize_64fc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumErrorGetBufferSize_64fc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 8-bit unsigned char Average method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageError(this CudaDeviceVariable<byte> pSrc1, CudaDeviceVariable<byte> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageError_8u(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageError_8u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 8-bit signed char Average method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageError(this CudaDeviceVariable<sbyte> pSrc1, CudaDeviceVariable<sbyte> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageError_8s(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageError_8s", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit unsigned short integer Average method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageError(this CudaDeviceVariable<ushort> pSrc1, CudaDeviceVariable<ushort> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageError_16u(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageError_16u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit signed short integer Average method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageError(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageError_16s(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageError_16s", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit unsigned short complex integer Average method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageError(this CudaDeviceVariable<Npp16sc> pSrc1, CudaDeviceVariable<Npp16sc> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageError_16sc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageError_16sc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit unsigned short integer Average method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageError(this CudaDeviceVariable<uint> pSrc1, CudaDeviceVariable<uint> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageError_32u(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageError_32u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit signed short integer Average method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageError(this CudaDeviceVariable<int> pSrc1, CudaDeviceVariable<int> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageError_32s(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageError_32s", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit unsigned short complex integer Average method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageError(this CudaDeviceVariable<Npp32sc> pSrc1, CudaDeviceVariable<Npp32sc> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageError_32sc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageError_32sc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit signed short integer Average method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageError(this CudaDeviceVariable<long> pSrc1, CudaDeviceVariable<long> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageError_64s(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageError_64s", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit unsigned short complex integer Average method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageError(this CudaDeviceVariable<Npp64sc> pSrc1, CudaDeviceVariable<Npp64sc> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageError_64sc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageError_64sc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit floating point Average method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageError(this CudaDeviceVariable<float> pSrc1, CudaDeviceVariable<float> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageError_32f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageError_32f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit floating point complex Average method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageError(this CudaDeviceVariable<Npp32fc> pSrc1, CudaDeviceVariable<Npp32fc> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageError_32fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageError_32fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit floating point Average method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageError(this CudaDeviceVariable<double> pSrc1, CudaDeviceVariable<double> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageError_64f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageError_64f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit floating point complex Average method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageError(this CudaDeviceVariable<Npp64fc> pSrc1, CudaDeviceVariable<Npp64fc> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageError_64fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageError_64fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageError_8u.
		/// </summary>
		public static int AverageErrorGetBufferSize(this CudaDeviceVariable<byte> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageErrorGetBufferSize_8u(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageErrorGetBufferSize_8u", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageError_8s.
		/// </summary>
		public static int AverageErrorGetBufferSize(this CudaDeviceVariable<sbyte> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageErrorGetBufferSize_8s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageErrorGetBufferSize_8s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageError_16u.
		/// </summary>
		public static int AverageErrorGetBufferSize(this CudaDeviceVariable<ushort> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageErrorGetBufferSize_16u(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageErrorGetBufferSize_16u", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageError_16s.
		/// </summary>
		public static int AverageErrorGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageErrorGetBufferSize_16s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageErrorGetBufferSize_16s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageError_16sc.
		/// </summary>
		public static int AverageErrorGetBufferSize(this CudaDeviceVariable<Npp16sc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageErrorGetBufferSize_16sc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageErrorGetBufferSize_16sc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageError_32u.
		/// </summary>
		public static int AverageErrorGetBufferSize(this CudaDeviceVariable<uint> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageErrorGetBufferSize_32u(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageErrorGetBufferSize_32u", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageError_32s.
		/// </summary>
		public static int AverageErrorGetBufferSize(this CudaDeviceVariable<int> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageErrorGetBufferSize_32s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageErrorGetBufferSize_32s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageError_32sc.
		/// </summary>
		public static int AverageErrorGetBufferSize(this CudaDeviceVariable<Npp32sc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageErrorGetBufferSize_32sc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageErrorGetBufferSize_32sc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageError_64s.
		/// </summary>
		public static int AverageErrorGetBufferSize(this CudaDeviceVariable<long> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageErrorGetBufferSize_64s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageErrorGetBufferSize_64s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageError_64sc.
		/// </summary>
		public static int AverageErrorGetBufferSize(this CudaDeviceVariable<Npp64sc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageErrorGetBufferSize_64sc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageErrorGetBufferSize_64sc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageError_32f.
		/// </summary>
		public static int AverageErrorGetBufferSize(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageErrorGetBufferSize_32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageErrorGetBufferSize_32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageError_32fc.
		/// </summary>
		public static int AverageErrorGetBufferSize(this CudaDeviceVariable<Npp32fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageErrorGetBufferSize_32fc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageErrorGetBufferSize_32fc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageError_64f.
		/// </summary>
		public static int AverageErrorGetBufferSize(this CudaDeviceVariable<double> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageErrorGetBufferSize_64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageErrorGetBufferSize_64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageError_64fc.
		/// </summary>
		public static int AverageErrorGetBufferSize(this CudaDeviceVariable<Npp64fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageError.nppsAverageErrorGetBufferSize_64fc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageErrorGetBufferSize_64fc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 8-bit unsigned char MaximumRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumRelativeError(this CudaDeviceVariable<byte> pSrc1, CudaDeviceVariable<byte> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeError_8u(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeError_8u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 8-bit signed char MaximumRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumRelativeError(this CudaDeviceVariable<sbyte> pSrc1, CudaDeviceVariable<sbyte> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeError_8s(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeError_8s", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit unsigned short integer MaximumRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumRelativeError(this CudaDeviceVariable<ushort> pSrc1, CudaDeviceVariable<ushort> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeError_16u(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeError_16u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit signed short integer MaximumRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumRelativeError(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeError_16s(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeError_16s", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit unsigned short complex integer MaximumRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumRelativeError(this CudaDeviceVariable<Npp16sc> pSrc1, CudaDeviceVariable<Npp16sc> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeError_16sc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeError_16sc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit unsigned short integer MaximumRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumRelativeError(this CudaDeviceVariable<uint> pSrc1, CudaDeviceVariable<uint> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeError_32u(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeError_32u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit signed short integer MaximumRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumRelativeError(this CudaDeviceVariable<int> pSrc1, CudaDeviceVariable<int> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeError_32s(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeError_32s", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit unsigned short complex integer MaximumRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumRelativeError(this CudaDeviceVariable<Npp32sc> pSrc1, CudaDeviceVariable<Npp32sc> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeError_32sc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeError_32sc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit signed short integer MaximumRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumRelativeError(this CudaDeviceVariable<long> pSrc1, CudaDeviceVariable<long> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeError_64s(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeError_64s", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit unsigned short complex integer MaximumRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumRelativeError(this CudaDeviceVariable<Npp64sc> pSrc1, CudaDeviceVariable<Npp64sc> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeError_64sc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeError_64sc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit floating point MaximumRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumRelativeError(this CudaDeviceVariable<float> pSrc1, CudaDeviceVariable<float> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeError_32f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeError_32f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit floating point complex MaximumRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumRelativeError(this CudaDeviceVariable<Npp32fc> pSrc1, CudaDeviceVariable<Npp32fc> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeError_32fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeError_32fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit floating point MaximumRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumRelativeError(this CudaDeviceVariable<double> pSrc1, CudaDeviceVariable<double> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeError_64f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeError_64f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit floating point complex MaximumRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void MaximumRelativeError(this CudaDeviceVariable<Npp64fc> pSrc1, CudaDeviceVariable<Npp64fc> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeError_64fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeError_64fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumRelativeError_8u.
		/// </summary>
		public static int MaximumRelativeErrorGetBufferSize(this CudaDeviceVariable<byte> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeErrorGetBufferSize_8u(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeErrorGetBufferSize_8u", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumRelativeError_8s.
		/// </summary>
		public static int MaximumRelativeErrorGetBufferSize(this CudaDeviceVariable<sbyte> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeErrorGetBufferSize_8s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeErrorGetBufferSize_8s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumRelativeError_16u.
		/// </summary>
		public static int MaximumRelativeErrorGetBufferSize(this CudaDeviceVariable<ushort> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeErrorGetBufferSize_16u(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeErrorGetBufferSize_16u", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumRelativeError_16s.
		/// </summary>
		public static int MaximumRelativeErrorGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeErrorGetBufferSize_16s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeErrorGetBufferSize_16s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumRelativeError_16sc.
		/// </summary>
		public static int MaximumRelativeErrorGetBufferSize(this CudaDeviceVariable<Npp16sc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeErrorGetBufferSize_16sc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeErrorGetBufferSize_16sc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumRelativeError_32u.
		/// </summary>
		public static int MaximumRelativeErrorGetBufferSize(this CudaDeviceVariable<uint> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeErrorGetBufferSize_32u(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeErrorGetBufferSize_32u", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumRelativeError_32s.
		/// </summary>
		public static int MaximumRelativeErrorGetBufferSize(this CudaDeviceVariable<int> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeErrorGetBufferSize_32s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeErrorGetBufferSize_32s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumRelativeError_32sc.
		/// </summary>
		public static int MaximumRelativeErrorGetBufferSize(this CudaDeviceVariable<Npp32sc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeErrorGetBufferSize_32sc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeErrorGetBufferSize_32sc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumRelativeError_64s.
		/// </summary>
		public static int MaximumRelativeErrorGetBufferSize(this CudaDeviceVariable<long> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeErrorGetBufferSize_64s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeErrorGetBufferSize_64s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumRelativeError_64sc.
		/// </summary>
		public static int MaximumRelativeErrorGetBufferSize(this CudaDeviceVariable<Npp64sc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeErrorGetBufferSize_64sc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeErrorGetBufferSize_64sc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumRelativeError_32f.
		/// </summary>
		public static int MaximumRelativeErrorGetBufferSize(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeErrorGetBufferSize_32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeErrorGetBufferSize_32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumRelativeError_32fc.
		/// </summary>
		public static int MaximumRelativeErrorGetBufferSize(this CudaDeviceVariable<Npp32fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeErrorGetBufferSize_32fc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeErrorGetBufferSize_32fc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumRelativeError_64f.
		/// </summary>
		public static int MaximumRelativeErrorGetBufferSize(this CudaDeviceVariable<double> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeErrorGetBufferSize_64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeErrorGetBufferSize_64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsMaximumRelativeError_64fc.
		/// </summary>
		public static int MaximumRelativeErrorGetBufferSize(this CudaDeviceVariable<Npp64fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.MaximumRelativeError.nppsMaximumRelativeErrorGetBufferSize_64fc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsMaximumRelativeErrorGetBufferSize_64fc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// 8-bit unsigned char AverageRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageRelativeError(this CudaDeviceVariable<byte> pSrc1, CudaDeviceVariable<byte> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeError_8u(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeError_8u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 8-bit signed char AverageRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageRelativeError(this CudaDeviceVariable<sbyte> pSrc1, CudaDeviceVariable<sbyte> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeError_8s(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeError_8s", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit unsigned short integer AverageRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageRelativeError(this CudaDeviceVariable<ushort> pSrc1, CudaDeviceVariable<ushort> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeError_16u(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeError_16u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit signed short integer AverageRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageRelativeError(this CudaDeviceVariable<short> pSrc1, CudaDeviceVariable<short> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeError_16s(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeError_16s", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 16-bit unsigned short complex integer AverageRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageRelativeError(this CudaDeviceVariable<Npp16sc> pSrc1, CudaDeviceVariable<Npp16sc> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeError_16sc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeError_16sc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit unsigned short integer AverageRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageRelativeError(this CudaDeviceVariable<uint> pSrc1, CudaDeviceVariable<uint> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeError_32u(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeError_32u", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit signed short integer AverageRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageRelativeError(this CudaDeviceVariable<int> pSrc1, CudaDeviceVariable<int> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeError_32s(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeError_32s", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit unsigned short complex integer AverageRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageRelativeError(this CudaDeviceVariable<Npp32sc> pSrc1, CudaDeviceVariable<Npp32sc> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeError_32sc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeError_32sc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit signed short integer AverageRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageRelativeError(this CudaDeviceVariable<long> pSrc1, CudaDeviceVariable<long> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeError_64s(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeError_64s", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit unsigned short complex integer AverageRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageRelativeError(this CudaDeviceVariable<Npp64sc> pSrc1, CudaDeviceVariable<Npp64sc> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeError_64sc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeError_64sc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit floating point AverageRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageRelativeError(this CudaDeviceVariable<float> pSrc1, CudaDeviceVariable<float> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeError_32f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeError_32f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 32-bit floating point complex AverageRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageRelativeError(this CudaDeviceVariable<Npp32fc> pSrc1, CudaDeviceVariable<Npp32fc> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeError_32fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeError_32fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit floating point AverageRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageRelativeError(this CudaDeviceVariable<double> pSrc1, CudaDeviceVariable<double> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeError_64f(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeError_64f", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// 64-bit floating point complex AverageRelative method.
		/// </summary>
		/// <param name="pSrc1">Source signal pointer.</param>
		/// <param name="pSrc2">Source signal pointer.</param>
		/// <param name="pDst">Pointer to the error result.</param>
		/// <param name="pDeviceBuffer">Pointer to the required device memory allocation. </param>
		public static void AverageRelativeError(this CudaDeviceVariable<Npp64fc> pSrc1, CudaDeviceVariable<Npp64fc> pSrc2, CudaDeviceVariable<double> pDst, CudaDeviceVariable<byte> pDeviceBuffer)
		{
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeError_64fc(pSrc1.DevicePointer, pSrc2.DevicePointer, pSrc1.Size, pDst.DevicePointer, pDeviceBuffer.DevicePointer);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeError_64fc", status));
			NPPException.CheckNppStatus(status, pSrc1);
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageRelativeError_8u.
		/// </summary>
		public static int AverageRelativeErrorGetBufferSize(this CudaDeviceVariable<byte> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeErrorGetBufferSize_8u(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeErrorGetBufferSize_8u", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageRelativeError_8s.
		/// </summary>
		public static int AverageRelativeErrorGetBufferSize(this CudaDeviceVariable<sbyte> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeErrorGetBufferSize_8s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeErrorGetBufferSize_8s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageRelativeError_16u.
		/// </summary>
		public static int AverageRelativeErrorGetBufferSize(this CudaDeviceVariable<ushort> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeErrorGetBufferSize_16u(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeErrorGetBufferSize_16u", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageRelativeError_16s.
		/// </summary>
		public static int AverageRelativeErrorGetBufferSize(this CudaDeviceVariable<short> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeErrorGetBufferSize_16s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeErrorGetBufferSize_16s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageRelativeError_16sc.
		/// </summary>
		public static int AverageRelativeErrorGetBufferSize(this CudaDeviceVariable<Npp16sc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeErrorGetBufferSize_16sc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeErrorGetBufferSize_16sc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageRelativeError_32u.
		/// </summary>
		public static int AverageRelativeErrorGetBufferSize(this CudaDeviceVariable<uint> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeErrorGetBufferSize_32u(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeErrorGetBufferSize_32u", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageRelativeError_32s.
		/// </summary>
		public static int AverageRelativeErrorGetBufferSize(this CudaDeviceVariable<int> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeErrorGetBufferSize_32s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeErrorGetBufferSize_32s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageRelativeError_32sc.
		/// </summary>
		public static int AverageRelativeErrorGetBufferSize(this CudaDeviceVariable<Npp32sc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeErrorGetBufferSize_32sc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeErrorGetBufferSize_32sc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageRelativeError_64s.
		/// </summary>
		public static int AverageRelativeErrorGetBufferSize(this CudaDeviceVariable<long> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeErrorGetBufferSize_64s(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeErrorGetBufferSize_64s", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageRelativeError_64sc.
		/// </summary>
		public static int AverageRelativeErrorGetBufferSize(this CudaDeviceVariable<Npp64sc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeErrorGetBufferSize_64sc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeErrorGetBufferSize_64sc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageRelativeError_32f.
		/// </summary>
		public static int AverageRelativeErrorGetBufferSize(this CudaDeviceVariable<float> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeErrorGetBufferSize_32f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeErrorGetBufferSize_32f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageRelativeError_32fc.
		/// </summary>
		public static int AverageRelativeErrorGetBufferSize(this CudaDeviceVariable<Npp32fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeErrorGetBufferSize_32fc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeErrorGetBufferSize_32fc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageRelativeError_64f.
		/// </summary>
		public static int AverageRelativeErrorGetBufferSize(this CudaDeviceVariable<double> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeErrorGetBufferSize_64f(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeErrorGetBufferSize_64f", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}

		/// <summary>
		/// Device-buffer size (in bytes) for nppsAverageRelativeError_64fc.
		/// </summary>
		public static int AverageRelativeErrorGetBufferSize(this CudaDeviceVariable<Npp64fc> devVar)
		{
			int size = 0;
			NppStatus status = NPPNativeMethods.NPPs.AverageRelativeError.nppsAverageRelativeErrorGetBufferSize_64fc(devVar.Size, ref size);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppsAverageRelativeErrorGetBufferSize_64fc", status));
			NPPException.CheckNppStatus(status, devVar);
			return size;
		}


		#endregion
	}
}
