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

namespace ManagedCuda.NVRTC
{
    /// <summary>
    /// CUDA Online Compiler API call result code.
    /// </summary>
    public enum nvrtcResult
    {
        /// <summary/>
        Success = 0,
        /// <summary/>
        ErrorOutOfMemory = 1,
        /// <summary/>
        ErrorProgramCreationFailure = 2,
        /// <summary/>
        ErrorInvalidInput = 3,
        /// <summary/>
        ErrorInvalidProgram = 4,
        /// <summary/>
        ErrorInvalidOption = 5,
        /// <summary/>
        ErrorCompilation = 6,
        /// <summary/>
        ErrorBuiltinOperationFailure = 7,
        /// <summary/>
        NoNameExpressionsAfterCompilation = 8,
        /// <summary/>
        NoLoweredNamesBeforeCompilation = 9,
        /// <summary/>
        ExpressionNotValid = 10,
        /// <summary/>
        InternalError = 11
    }

    /// <summary>
    /// the unit of compilation, and an opaque handle for a program.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct nvrtcProgram
    {
        /// <summary>
        /// 
        /// </summary>
        public IntPtr Pointer;
    }
}
