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
using ManagedCuda.VectorTypes;

namespace ManagedCuda.NPP
{
	static class NPPExtensionMethods
	{
		#region 8u
		public static NPPImage_8uC1 ToNPPImage(this CudaPitchedDeviceVariable<byte> deviceVar)
		{
			return new NPPImage_8uC1(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}

		public static NPPImage_8uC2 ToNPPImage(this CudaPitchedDeviceVariable<uchar2> deviceVar)
		{
			return new NPPImage_8uC2(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}

		public static NPPImage_8uC3 ToNPPImage(this CudaPitchedDeviceVariable<uchar3> deviceVar)
		{
			return new NPPImage_8uC3(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}

		public static NPPImage_8uC4 ToNPPImage(this CudaPitchedDeviceVariable<uchar4> deviceVar)
		{
			return new NPPImage_8uC4(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}
		#endregion

		#region 8s
		public static NPPImage_8sC1 ToNPPImage(this CudaPitchedDeviceVariable<sbyte> deviceVar)
		{
			return new NPPImage_8sC1(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}

		public static NPPImage_8sC2 ToNPPImage(this CudaPitchedDeviceVariable<char2> deviceVar)
		{
			return new NPPImage_8sC2(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}

		public static NPPImage_8sC3 ToNPPImage(this CudaPitchedDeviceVariable<char3> deviceVar)
		{
			return new NPPImage_8sC3(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}

		public static NPPImage_8sC4 ToNPPImage(this CudaPitchedDeviceVariable<char4> deviceVar)
		{
			return new NPPImage_8sC4(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}
		#endregion

		#region 16u
		public static NPPImage_16uC1 ToNPPImage(this CudaPitchedDeviceVariable<ushort> deviceVar)
		{
			return new NPPImage_16uC1(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}

		public static NPPImage_16uC2 ToNPPImage(this CudaPitchedDeviceVariable<ushort2> deviceVar)
		{
			return new NPPImage_16uC2(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}

		public static NPPImage_16uC3 ToNPPImage(this CudaPitchedDeviceVariable<ushort3> deviceVar)
		{
			return new NPPImage_16uC3(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}

		public static NPPImage_16uC4 ToNPPImage(this CudaPitchedDeviceVariable<ushort4> deviceVar)
		{
			return new NPPImage_16uC4(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}
		#endregion

		#region 16s
		public static NPPImage_16sC1 ToNPPImage(this CudaPitchedDeviceVariable<short> deviceVar)
		{
			return new NPPImage_16sC1(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}

		public static NPPImage_16sC2 ToNPPImage(this CudaPitchedDeviceVariable<short2> deviceVar)
		{
			return new NPPImage_16sC2(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}

		public static NPPImage_16sC3 ToNPPImage(this CudaPitchedDeviceVariable<short3> deviceVar)
		{
			return new NPPImage_16sC3(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}

		public static NPPImage_16sC4 ToNPPImage(this CudaPitchedDeviceVariable<short4> deviceVar)
		{
			return new NPPImage_16sC4(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}
		#endregion

		#region 32f
		public static NPPImage_32fC1 ToNPPImage(this CudaPitchedDeviceVariable<float> deviceVar)
		{
			return new NPPImage_32fC1(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}

		public static NPPImage_32fC2 ToNPPImage(this CudaPitchedDeviceVariable<float2> deviceVar)
		{
			return new NPPImage_32fC2(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}

		public static NPPImage_32fC3 ToNPPImage(this CudaPitchedDeviceVariable<float3> deviceVar)
		{
			return new NPPImage_32fC3(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}

		public static NPPImage_32fC4 ToNPPImage(this CudaPitchedDeviceVariable<float4> deviceVar)
		{
			return new NPPImage_32fC4(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}
		#endregion

		#region 32s
		public static NPPImage_32sC1 ToNPPImage(this CudaPitchedDeviceVariable<int> deviceVar)
		{
			return new NPPImage_32sC1(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}

		public static NPPImage_32sC3 ToNPPImage(this CudaPitchedDeviceVariable<int3> deviceVar)
		{
			return new NPPImage_32sC3(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}

		public static NPPImage_32sC4 ToNPPImage(this CudaPitchedDeviceVariable<int4> deviceVar)
		{
			return new NPPImage_32sC4(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}
		#endregion

		#region 32u
		public static NPPImage_32uC1 ToNPPImage(this CudaPitchedDeviceVariable<uint> deviceVar)
		{
			return new NPPImage_32uC1(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}

		public static NPPImage_32uC4 ToNPPImage(this CudaPitchedDeviceVariable<uint4> deviceVar)
		{
			return new NPPImage_32uC4(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}
		#endregion

		#region complex
		public static NPPImage_32fcC1 ToNPPImage(this CudaPitchedDeviceVariable<cuFloatComplex> deviceVar)
		{
			return new NPPImage_32fcC1(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}

		public static NPPImage_32scC1 ToNPPImage(this CudaPitchedDeviceVariable<int2> deviceVar)
		{
			return new NPPImage_32scC1(deviceVar.DevicePointer, deviceVar.Width, deviceVar.Height, deviceVar.Pitch);
		}
		#endregion
	}
}
