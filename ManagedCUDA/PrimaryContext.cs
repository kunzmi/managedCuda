//	Copyright (c) 2014, Michael Kunz. All rights reserved.
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
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace ManagedCuda
{
	/// <summary>
	/// The primary context unique per device and it's shared with CUDA runtime API.
	/// Those functions allows seemless integration with other libraries using CUDA.
	/// </summary>
	public class PrimaryContext : CudaContext
	{
		
		#region Constructors
		/// <summary>
		/// Create a new instace of managed Cuda. Retains the primary cuda context of device with ID 0.
		/// </summary>
		public PrimaryContext()
			: this(0)
		{

		}

		/// <summary>
		/// Create a new instace of managed Cuda. Retains the primary cuda context of device with ID deviceId.
		/// Using <see cref="CUCtxFlags.SchedAuto"/>
		/// </summary>
		/// <param name="deviceId">DeviceID</param>
		public PrimaryContext(int deviceId)
		{
			CUResult res;
			int deviceCount = 0;
			res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetCount(ref deviceCount);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDeviceGetCount", res));

			if (res == CUResult.ErrorNotInitialized)
			{
				res = DriverAPINativeMethods.cuInit(CUInitializationFlags.None);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuInit", res));
				if (res != CUResult.Success)
					throw new CudaException(res);

				res = DriverAPINativeMethods.DeviceManagement.cuDeviceGetCount(ref deviceCount);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDeviceGetCount", res));
				if (res != CUResult.Success)
					throw new CudaException(res);
			}
			else if (res != CUResult.Success)
				throw new CudaException(res);

			if (deviceCount == 0)
			{
				throw new CudaException(CUResult.ErrorNoDevice, "Cuda initialization error: There is no device supporting CUDA", null);
			}

			_deviceID = deviceId;

			res = DriverAPINativeMethods.DeviceManagement.cuDeviceGet(ref _device, deviceId);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDeviceGet", res));
			if (res != CUResult.Success)
				throw new CudaException(res);


			
			res = DriverAPINativeMethods.ContextManagement.cuDevicePrimaryCtxRetain(ref _context, _device);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDevicePrimaryCtxRetain", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// For dispose
		/// </summary>
		~PrimaryContext()
		{
			Dispose(false);
		}
		#endregion

		#region Dispose

		/// <summary>
		/// For IDisposable. Releases the wrapped primary context
		/// </summary>
		/// <param name="fDisposing"></param>
		protected override void Dispose(bool fDisposing)
		{
			if (fDisposing && !disposed)
			{
				//Ignore if failing
				CUResult res;
				res = DriverAPINativeMethods.ContextManagement.cuDevicePrimaryCtxRelease(_device);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDevicePrimaryCtxRelease", res));
				
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Static Methods

		/// <summary>
		/// Set flags for the primary context<para/>
		/// Sets the flags for the primary context on the device overwriting perviously
		/// set ones. If the primary context is already created
		/// ::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE is returned.
		/// <para/>
		///	The three LSBs of the \p flags parameter can be used to control how the OS
		///	thread, which owns the CUDA context at the time of an API call, interacts
		///	with the OS scheduler when waiting for results from the GPU. Only one of
		///	the scheduling flags can be set when creating a context.
		/// </summary>
		/// <param name="deviceID">Device for which the primary context flags are set</param>
		/// <param name="flags">New flags for the device</param>
		public static void SetFlags(int deviceID, CUCtxFlags flags)
		{
			CUdevice device = GetCUdevice(deviceID);
			SetFlags(device, flags);
		}

		/// <summary>
		/// Set flags for the primary context<para/>
		/// Sets the flags for the primary context on the device overwriting perviously
		/// set ones. If the primary context is already created
		/// ::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE is returned.
		/// <para/>
		///	The three LSBs of the \p flags parameter can be used to control how the OS
		///	thread, which owns the CUDA context at the time of an API call, interacts
		///	with the OS scheduler when waiting for results from the GPU. Only one of
		///	the scheduling flags can be set when creating a context.
		/// </summary>
		/// <param name="device">Device for which the primary context flags are set</param>
		/// <param name="flags">New flags for the device</param>
		public static void SetFlags(CUdevice device, CUCtxFlags flags)
		{
			CUResult res;
			res = DriverAPINativeMethods.ContextManagement.cuDevicePrimaryCtxSetFlags(device, flags);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDevicePrimaryCtxSetFlags", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}

		/// <summary>
		/// Get the state of the primary context<para/>
		/// Returns in flags the flags for the primary context of device, and in
		/// active whether it is active.  See ::cuDevicePrimaryCtxSetFlags for flag
		/// values.
		/// </summary>
		/// <param name="deviceID">Device to get primary context flags for</param>
		/// <param name="flags">Pointer to store flags</param>
		/// <param name="active">Pointer to store context state</param>
		public static void GetState(int deviceID, out CUCtxFlags flags, out bool active)
		{
			CUdevice device = GetCUdevice(deviceID);
			GetState(device, out flags, out active);
		}

		/// <summary>
		/// Get the state of the primary context<para/>
		/// Returns in flags the flags for the primary context of device, and in
		/// active whether it is active.  See ::cuDevicePrimaryCtxSetFlags for flag
		/// values.
		/// </summary>
		/// <param name="device">Device to get primary context flags for</param>
		/// <param name="flags">Pointer to store flags</param>
		/// <param name="active">Pointer to store context state</param>
		public static void GetState(CUdevice device, out CUCtxFlags flags, out bool active)
		{
			CUResult res;
			flags = new CUCtxFlags();
			int temp = 0;
			res = DriverAPINativeMethods.ContextManagement.cuDevicePrimaryCtxGetState(device, ref flags, ref temp);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDevicePrimaryCtxGetState", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
			active = temp == 1;
		}

		/// <summary>
		/// Destroy all allocations and reset all state on the primary context
		/// <para/>
		/// Explicitly destroys and cleans up all resources associated with the current
		/// device in the current process.
		/// <para/>
		/// Note that it is responsibility of the calling function to ensure that no
		/// other module in the process is using the device any more. For that reason
		/// it is recommended to use ::cuDevicePrimaryCtxRelease() in most cases.
		/// However it is safe for other modules to call ::cuDevicePrimaryCtxRelease()
		/// even after resetting the device.
		/// </summary>
		/// <param name="deviceID">Device for which primary context is destroyed</param>
		public static void Reset(int deviceID)
		{
			CUdevice device = GetCUdevice(deviceID);
			Reset(device);
		}

		/// <summary>
		/// Destroy all allocations and reset all state on the primary context
		/// <para/>
		/// Explicitly destroys and cleans up all resources associated with the current
		/// device in the current process.
		/// <para/>
		/// Note that it is responsibility of the calling function to ensure that no
		/// other module in the process is using the device any more. For that reason
		/// it is recommended to use ::cuDevicePrimaryCtxRelease() in most cases.
		/// However it is safe for other modules to call ::cuDevicePrimaryCtxRelease()
		/// even after resetting the device.
		/// </summary>
		/// <param name="device">Device for which primary context is destroyed</param>
		public static void Reset(CUdevice device)
		{
			CUResult res;
			res = DriverAPINativeMethods.ContextManagement.cuDevicePrimaryCtxReset(device);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuDevicePrimaryCtxReset", res));
			if (res != CUResult.Success)
				throw new CudaException(res);
		}
		#endregion
	}
}
