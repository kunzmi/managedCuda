using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Text;

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
