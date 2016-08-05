//	Copyright (c) 2016, Michael Kunz. All rights reserved.
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
using System.Runtime.InteropServices;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;

namespace ManagedCuda.NVGraph
{
	#region Enums
	/// <summary>
	/// nvGRAPH status type returns
	/// </summary>
	public enum nvgraphStatus
	{
		Success = 0,
		NotInitialized = 1,
		AllocFailed = 2,
		InvalidValue = 3,
		ArchMismatch = 4,
		MappingError = 5,
		ExecutionFailed = 6,
		InternalError = 7,
		TypeNotSupported = 8,
		NotConverged = 9
    }

	/// <summary>
	/// Semi-ring types
	/// </summary>
	public enum nvgraphSemiring
	{
		PlusTimesSR = 0,
		MinPlusSR = 1,
		MaxMinSR = 2,
		OrAndSR = 3,
	}

	/// <summary>
	/// Topology types
	/// </summary>
	public enum nvgraphTopologyType
	{
		CSR32 = 0,
		CSC32 = 1,
	}

	#endregion

	#region structs (opaque handles)
	/// <summary>
	/// Opaque structure holding nvGRAPH library context
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct nvgraphContext
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Handle;
	}

	/// <summary>
	/// Opaque structure holding the graph descriptor
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct nvgraphGraphDescr
	{
		/// <summary>
		/// 
		/// </summary>
		public IntPtr Handle;
	}
	#endregion

	#region structs
	/// <summary>
	/// 
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct nvgraphCSRTopology32I
	{
		public int nvertices;
		public int nedges;
		public IntPtr source_offsets;
		public IntPtr destination_indices;
	}

	/// <summary>
	/// 
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct nvgraphCSCTopology32I
	{
		public int nvertices;
		public int nedges;
		public IntPtr destination_offsets;
		public IntPtr source_indices;
	}
	#endregion

}
