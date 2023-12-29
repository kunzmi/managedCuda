// Copyright (c) 2023, Michael Kunz and Artic Imaging SARL. All rights reserved.
// http://kunzmi.github.io/managedCuda
//
// This file is part of ManagedCuda.
//
// Commercial License Usage
//  Licensees holding valid commercial ManagedCuda licenses may use this
//  file in accordance with the commercial license agreement provided with
//  the Software or, alternatively, in accordance with the terms contained
//  in a written agreement between you and Artic Imaging SARL. For further
//  information contact us at managedcuda@articimaging.eu.
//  
// GNU General Public License Usage
//  Alternatively, this file may be used under the terms of the GNU General
//  Public License as published by the Free Software Foundation, either 
//  version 3 of the License, or (at your option) any later version.
//  
//  ManagedCuda is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with this program. If not, see <http://www.gnu.org/licenses/>.


using System;
using System.Runtime.InteropServices;

namespace ManagedCuda.NvJitLink
{
    /// <summary>
    /// The enumerated type nvJitLinkResult defines API call result codes.
	/// nvJitLink APIs return nvJitLinkResult codes to indicate the result.
	/// </summary>
	public enum nvJitLinkResult
    {
        /// <summary/>
        Success = 0,
        /// <summary/>
        ErrorUnrecognizedOption,
        /// <summary>
        /// -arch=sm_NN option not specified
        /// </summary>
        ErrorMissingArch,
        /// <summary/>
        ErrorInvalidInput,
        /// <summary/>
        ErrorPtxCompile,
        /// <summary/>
        ErrorNVVMCompile,
        /// <summary/>
        ErrorInternal,
        /// <summary/>
        ErrorThreadPool
    }

    /// <summary>
    /// The enumerated type nvJitLinkInputType defines the kind of inputs
    /// that can be passed to nvJitLinkAdd* APIs.
	/// </summary>
	public enum nvJitLinkInputType
    {
        /// <summary>
        /// Error
        /// </summary>
        None = 0, // error
        /// <summary/>
        Cubin = 1,
        /// <summary/>
        Ptx,
        /// <summary/>
        LTOIR,
        /// <summary/>
        FatBin,
        /// <summary/>
        Object,
        /// <summary/>
        Library
    }

    /// <summary>
    /// nvJitLinkHandle is the unit of linking, and an opaque handle for a program.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct nvJitLinkHandle
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }
}
