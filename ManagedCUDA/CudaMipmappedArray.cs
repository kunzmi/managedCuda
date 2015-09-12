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
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace ManagedCuda
{
	/// <summary>
	/// Number of channels in array
	/// </summary>
	public enum CudaMipmappedArrayNumChannels
	{
		/// <summary>
		/// One channel, e.g. float1, int1, float, int
		/// </summary>
		One = 1,
		/// <summary>
		/// Two channels, e.g. float2, int2
		/// </summary>
		Two = 2,
		/// <summary>
		/// Four channels, e.g. float4, int4
		/// </summary>
		Four = 4
	}

	/// <summary>
	/// A mipmapped Cuda array
	/// </summary>
	public class CudaMipmappedArray : IDisposable
	{
		CUmipmappedArray _mipmappedArray;
		CUDAArray3DDescriptor _arrayDescriptor;
		CUResult res;
		bool disposed;
		bool _isOwner;

		/// <summary>
		/// Creates a CUDA mipmapped array according to <c>descriptor</c>. <para/>
		/// Width, Height, and Depth are the width, height, and depth of the CUDA array (in elements); the following
		/// types of CUDA arrays can be allocated:<para/>
		/// – A 1D mipmapped array is allocated if Height and Depth extents are both zero.<para/>
		/// – A 2D mipmapped array is allocated if only Depth extent is zero.<para/>
		/// – A 3D mipmapped array is allocated if all three extents are non-zero.<para/>
		/// – A 1D layered CUDA mipmapped array is allocated if only Height is zero and the <see cref="CUDAArray3DFlags.Layered"/> 
		///   flag is set. Each layer is a 1D array. The number of layers is determined by the depth extent.<para/>
		/// – A 2D layered CUDA mipmapped array is allocated if all three extents are non-zero and the <see cref="CUDAArray3DFlags.Layered"/> 
		///   flag is set. Each layer is a 2D array. The number of layers is determined by the depth extent.<para/>
		/// – A cubemap CUDA mipmapped array is allocated if all three extents are non-zero and the <see cref="CUDAArray3DFlags.Cubemap"/>
		///   flag is set. Width must be equal to Height, and Depth must be six. A
		///   cubemap is a special type of 2D layered CUDA array, where the six layers represent the six faces of a
		///   cube. The order of the six layers in memory is the same as that listed in CUarray_cubemap_face.<para/>
		/// – A cubemap layered CUDA mipmapped array is allocated if all three extents are non-zero, and both,
		///   <see cref="CUDAArray3DFlags.Cubemap"/> and <see cref="CUDAArray3DFlags.Layered"/> flags are set. Width must be equal
		///   to Height, and Depth must be a multiple of six. A cubemap layered CUDA array is a special type of
		///   2D layered CUDA array that consists of a collection of cubemaps. The first six layers represent the first
		///   cubemap, the next six layers form the second cubemap, and so on.<para/>
		/// Flags may be set to:<para/>
		/// – <see cref="CUDAArray3DFlags.Layered"/> to enable creation of layered CUDA mipmapped arrays. If this flag is set,
		///   Depth specifies the number of layers, not the depth of a 3D array.<para/>
		/// – <see cref="CUDAArray3DFlags.Cubemap"/> to enable creation of mipmapped cubemaps. If this flag is set, Width
		///   must be equal to Height, and Depth must be six. If the CUDA_ARRAY3D_LAYERED flag is also set,
		///   then Depth must be a multiple of six.<para/>
		/// – <see cref="CUDAArray3DFlags.TextureGather"/> to indicate that the CUDA mipmapped array will be used for
		///   texture gather. Texture gather can only be performed on 2D CUDA mipmapped arrays.
		/// </summary>
		/// <param name="descriptor">mipmapped array descriptor</param>
		/// <param name="numMipmapLevels">Number of mipmap levels. This value is clamped to the range [1, 1 + floor(log2(max(width, height, depth)))]</param>
		public CudaMipmappedArray(CUDAArray3DDescriptor descriptor, uint numMipmapLevels)
		{
			_mipmappedArray = new CUmipmappedArray();
			_arrayDescriptor = descriptor;

			res = DriverAPINativeMethods.ArrayManagement.cuMipmappedArrayCreate(ref _mipmappedArray, ref _arrayDescriptor, numMipmapLevels);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMipmappedArrayCreate", res));
            if (res != CUResult.Success) throw new CudaException(res);
            _isOwner = true;        
		}

		/// <summary>
		/// Creates a CUDA mipmapped array according to <c>descriptor</c>. <para/>
		/// Width, Height, and Depth are the width, height, and depth of the CUDA array (in elements); the following
		/// types of CUDA arrays can be allocated:<para/>
		/// – A 1D mipmapped array is allocated if Height and Depth extents are both zero.<para/>
		/// – A 2D mipmapped array is allocated if only Depth extent is zero.<para/>
		/// – A 3D mipmapped array is allocated if all three extents are non-zero.<para/>
		/// – A 1D layered CUDA mipmapped array is allocated if only Height is zero and the <see cref="CUDAArray3DFlags.Layered"/> 
		///   flag is set. Each layer is a 1D array. The number of layers is determined by the depth extent.
		/// – A 2D layered CUDA mipmapped array is allocated if all three extents are non-zero and the <see cref="CUDAArray3DFlags.Layered"/> 
		///   flag is set. Each layer is a 2D array. The number of layers is determined by the depth extent.
		/// – A cubemap CUDA mipmapped array is allocated if all three extents are non-zero and the <see cref="CUDAArray3DFlags.Cubemap"/>
		///   flag is set. Width must be equal to Height, and Depth must be six. A
		///   cubemap is a special type of 2D layered CUDA array, where the six layers represent the six faces of a
		///   cube. The order of the six layers in memory is the same as that listed in CUarray_cubemap_face.
		/// – A cubemap layered CUDA mipmapped array is allocated if all three extents are non-zero, and both,
		///   <see cref="CUDAArray3DFlags.Cubemap"/> and <see cref="CUDAArray3DFlags.Layered"/> flags are set. Width must be equal
		///   to Height, and Depth must be a multiple of six. A cubemap layered CUDA array is a special type of
		///   2D layered CUDA array that consists of a collection of cubemaps. The first six layers represent the first
		///   cubemap, the next six layers form the second cubemap, and so on.
		/// </summary>
		/// <param name="format">Array format</param>
		/// <param name="width">Array width. See general description.</param>
		/// <param name="height">Array height. See general description.</param>
		/// <param name="depth">Array depth or layer count. See general description.</param>
		/// <param name="numChannels">number of channels</param>
		/// <param name="flags">Flags may be set to:<para/>
		/// – <see cref="CUDAArray3DFlags.Layered"/> to enable creation of layered CUDA mipmapped arrays. If this flag is set,
		///   Depth specifies the number of layers, not the depth of a 3D array.<para/>
		/// – <see cref="CUDAArray3DFlags.Cubemap"/> to enable creation of mipmapped cubemaps. If this flag is set, Width
		///   must be equal to Height, and Depth must be six. If the CUDA_ARRAY3D_LAYERED flag is also set,
		///   then Depth must be a multiple of six.<para/>
		/// – <see cref="CUDAArray3DFlags.TextureGather"/> to indicate that the CUDA mipmapped array will be used for
		///   texture gather. Texture gather can only be performed on 2D CUDA mipmapped arrays.</param>
		/// <param name="numMipmapLevels">Number of mipmap levels. This value is clamped to the range [1, 1 + floor(log2(max(width, height, depth)))]</param>
		public CudaMipmappedArray(CUArrayFormat format, SizeT width, SizeT height, SizeT depth, CudaMipmappedArrayNumChannels numChannels, CUDAArray3DFlags flags, uint numMipmapLevels)
		{
			_mipmappedArray = new CUmipmappedArray();
			_arrayDescriptor = new CUDAArray3DDescriptor();
			_arrayDescriptor.Width = width;
			_arrayDescriptor.Height = height;
			_arrayDescriptor.Depth = depth;
			_arrayDescriptor.NumChannels = (uint)numChannels;
			_arrayDescriptor.Flags = flags;
			_arrayDescriptor.Format = format;


			res = DriverAPINativeMethods.ArrayManagement.cuMipmappedArrayCreate(ref _mipmappedArray, ref _arrayDescriptor, numMipmapLevels);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMipmappedArrayCreate", res));
			if (res != CUResult.Success) throw new CudaException(res);
			_isOwner = true;
		}

		/// <summary>
		/// Creates a CUDA mipmapped array from an existing mipmap array handle.
		/// </summary>
		/// <param name="handle">handle to wrap</param>
		/// <param name="format">Array format of the wrapped array. Cannot be gathered through CUDA API.</param>
		/// <param name="numChannels">Number of channels of wrapped array.</param>
		public CudaMipmappedArray(CUmipmappedArray handle, CUArrayFormat format, CudaMipmappedArrayNumChannels numChannels)
		{
			_mipmappedArray = handle;
			_arrayDescriptor = new CUDAArray3DDescriptor();
			_arrayDescriptor.Format = format;
			_arrayDescriptor.NumChannels = (uint)numChannels;
			_isOwner = false;
		}

		#region Dispose
		/// <summary>
		/// Dispose
		/// </summary>
		public void Dispose()
		{
			Dispose(true);
			GC.SuppressFinalize(this);
		}

		/// <summary>
		/// For IDisposable
		/// </summary>
		/// <param name="fDisposing"></param>
		protected virtual void Dispose(bool fDisposing)
		{
			if (fDisposing && !disposed)
			{
				if (_isOwner)
				{
					res = DriverAPINativeMethods.ArrayManagement.cuMipmappedArrayDestroy(_mipmappedArray);
					Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMipmappedArrayDestroy", res));
				}
				disposed = true;
			}
			if (!fDisposing && !disposed)
				Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
		}
		#endregion

		#region Methods
		/// <summary>
		/// Returns a CUDA array that represents a single mipmap level
		/// of the CUDA mipmapped array.
		/// </summary>
		/// <param name="level">Mipmap level</param>
		public CudaArray1D GetLevelAsArray1D(uint level)
		{
			CUarray array = new CUarray();

			res = DriverAPINativeMethods.ArrayManagement.cuMipmappedArrayGetLevel(ref array, _mipmappedArray, level);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMipmappedArrayGetLevel", res));
			if (res != CUResult.Success)
				throw new CudaException(res);

			return new CudaArray1D(array, false);
		}

		/// <summary>
		/// Returns a CUDA array that represents a single mipmap level
		/// of the CUDA mipmapped array.
		/// </summary>
		/// <param name="level">Mipmap level</param>
		public CudaArray2D GetLevelAsArray2D(uint level)
		{
			CUarray array = new CUarray();

			res = DriverAPINativeMethods.ArrayManagement.cuMipmappedArrayGetLevel(ref array, _mipmappedArray, level);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMipmappedArrayGetLevel", res));
			if (res != CUResult.Success)
				throw new CudaException(res);

			return new CudaArray2D(array, false);
		}

		/// <summary>
		/// Returns a CUDA array that represents a single mipmap level
		/// of the CUDA mipmapped array.
		/// </summary>
		/// <param name="level">Mipmap level</param>
		public CudaArray3D GetLevelAsArray3D(uint level)
		{
			CUarray array = new CUarray();

			res = DriverAPINativeMethods.ArrayManagement.cuMipmappedArrayGetLevel(ref array, _mipmappedArray, level);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMipmappedArrayGetLevel", res));
			if (res != CUResult.Success)
				throw new CudaException(res);

			return new CudaArray3D(array, false);
		}

		/// <summary>
		/// Returns a CUDA array that represents a single mipmap level
		/// of the CUDA mipmapped array.
		/// </summary>
		/// <param name="level">Mipmap level</param>
		public CUarray GetLevelAsCUArray(uint level)
		{
			CUarray array = new CUarray();

			res = DriverAPINativeMethods.ArrayManagement.cuMipmappedArrayGetLevel(ref array, _mipmappedArray, level);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "cuMipmappedArrayGetLevel", res));
			if (res != CUResult.Success)
				throw new CudaException(res);

			return array;
		}
		#endregion

		#region Properties
		/// <summary>
		/// Returns the wrapped CUmipmappedArray
		/// </summary>
		public CUmipmappedArray CUMipmappedArray
		{
			get { return _mipmappedArray; }
		}

		/// <summary>
		/// Returns the wrapped CUDAArray3DDescriptor
		/// </summary>
		public CUDAArray3DDescriptor Array3DDescriptor
		{
			get { return _arrayDescriptor; }
		}

		/// <summary>
		/// Returns the Depth of the array
		/// </summary>
		public SizeT Depth
		{
			get { return _arrayDescriptor.Depth; }
		}

		/// <summary>
		/// Returns the Height of the array
		/// </summary>
		public SizeT Height
		{
			get { return _arrayDescriptor.Height; }
		}

		/// <summary>
		/// Returns the array width in elements
		/// </summary>
		public SizeT Width
		{
			get { return _arrayDescriptor.Width; }
		}

		/// <summary>
		/// Returns the array creation flags
		/// </summary>
		public CUDAArray3DFlags Flags
		{
			get { return _arrayDescriptor.Flags; }
		}

		/// <summary>
		/// Returns the array format
		/// </summary>
		public CUArrayFormat Format
		{
			get { return _arrayDescriptor.Format; }
		}

		/// <summary>
		/// Returns number of channels
		/// </summary>
		public uint NumChannels
		{
			get { return _arrayDescriptor.NumChannels; }
		}

		/// <summary>
		/// If the wrapper class instance is the owner of a CUDA handle, it will be destroyed while disposing.
		/// </summary>
		public bool IsOwner
		{
			get { return _isOwner; }
		}
		#endregion
	}
}
