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
        COO32 = 2
    }

    public enum nvgraphTag
    {
        /// <summary>
        /// Default is unsorted.
        /// </summary>
        Default = 0,
        /// <summary>
        /// 
        /// </summary>
        Unsorted = 1,
        /// <summary>
        /// CSR
        /// </summary>
        SortedBySource = 2,
        /// <summary>
        /// CSC 
        /// </summary>
        SortedByDestination = 3
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

    #region classes
    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public abstract class nvgraphTopologyBase
    {

    }

    /// <summary>
    /// 
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
	public class nvgraphCSRTopology32I : nvgraphTopologyBase
	{
        /// <summary>
        /// n+1
        /// </summary>
		public int nvertices;
        /// <summary>
        /// nnz
        /// </summary>
		public int nedges;
        /// <summary>
        /// rowPtr
        /// </summary>
		public IntPtr source_offsets;
        /// <summary>
        /// colInd
        /// </summary>
		public IntPtr destination_indices;
	}

	/// <summary>
	/// 
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public class nvgraphCSCTopology32I : nvgraphTopologyBase
    {
        /// <summary>
        /// n+1
        /// </summary>
		public int nvertices;
        /// <summary>
        /// nnz
        /// </summary>
		public int nedges;
        /// <summary>
        /// colPtr
        /// </summary>
		public IntPtr destination_offsets;
        /// <summary>
        /// rowInd
        /// </summary>
		public IntPtr source_indices;
	}

	/// <summary>
	/// 
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public class nvgraphCOOTopology32I : nvgraphTopologyBase
    {
        /// <summary>
        /// n+1
        /// </summary>
		public int nvertices;
        /// <summary>
        /// nnz
        /// </summary>
		public int nedges;
        /// <summary>
        /// rowInd
        /// </summary>
		public IntPtr destination_offsets;
        /// <summary>
        /// colInd
        /// </summary>
		public IntPtr source_indices;
        /// <summary>
        /// 
        /// </summary>
        public nvgraphTag tag;
	}
	#endregion

}
