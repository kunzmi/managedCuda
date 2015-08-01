using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace ManagedCuda.NVRTC
{
	/// <summary>
	/// Cuda runtime compiler
	/// </summary>
	public class CudaRuntimeCompiler : IDisposable
	{
		nvrtcProgram _program;
		bool disposed = false;
		nvrtcResult res;
        #region Contructors
        /// <summary>
        /// Creates a runtime compiler instance.
        /// </summary>
		/// <param name="src">CUDA program source.</param>
		/// <param name="name">CUDA program name.<para/>
		/// name can be NULL; "default_program" is used when name is NULL.</param>
		/// <param name="includeNames">Sources of the headers.</param>
		/// <param name="headers">Name of each header by which they can be included in the CUDA program source.</param>
        public CudaRuntimeCompiler(string src, string name, string[] headers, string[] includeNames)
        {
			int headerCount = 0;
			IntPtr[] headersPtr = null;
			IntPtr[] includeNamesPtr = null;

			try
			{
				if (headers != null && includeNames != null)
				{
					if (headers.Length != includeNames.Length)
						throw new ArgumentException("headers and includeNames must have same length.");

					if (headers == null)
						throw new ArgumentNullException("headers can't be NULL if includeNames is not NULL");
				
					if (includeNames == null)
						throw new ArgumentNullException("includeNames can't be NULL if headers is not NULL");

					headerCount = headers.Length;

					headersPtr = new IntPtr[headerCount];
					includeNamesPtr = new IntPtr[headerCount];

					for (int i = 0; i < headerCount; i++)
					{
						headersPtr[i] = Marshal.StringToHGlobalAnsi(headers[i]);
						includeNamesPtr[i] = Marshal.StringToHGlobalAnsi(includeNames[i]);
					}
				}

				_program = new nvrtcProgram();
				res = NVRTCNativeMethods.nvrtcCreateProgram(ref _program, src, name, headerCount, headersPtr, includeNamesPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcCreateProgram", res));
			}
			finally
			{
				if (headersPtr != null)
					for (int i = 0; i < headersPtr.Length; i++)
					{
						Marshal.FreeHGlobal(headersPtr[i]);
					}

				if (includeNamesPtr != null)
					for (int i = 0; i < includeNamesPtr.Length; i++)
					{
						Marshal.FreeHGlobal(includeNamesPtr[i]);
					}
			}

			if (res != nvrtcResult.Success)
				throw new NVRTCException(res);
        }
        /// <summary>
        /// Creates a runtime compiler instance.
        /// </summary>
		/// <param name="src">CUDA program source.</param>
		/// <param name="name">CUDA program name.<para/>
		/// name can be NULL; "default_program" is used when name is NULL.</param>
		public CudaRuntimeCompiler(string src, string name)
			: this(src, name, null, null)
		{

		}

        /// <summary>
        /// For dispose
        /// </summary>
		~CudaRuntimeCompiler()
        {
            Dispose(false);
        }
        #endregion

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
                //Ignore if failing
				res = NVRTCNativeMethods.nvrtcDestroyProgram(ref _program);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcDestroyProgram", res));
                disposed = true;
            }
            if (!fDisposing && !disposed)
                Debug.WriteLine(String.Format("ManagedCUDA not-disposed warning: {0}", this.GetType()));
        }
        #endregion

		#region Static Methods
		/// <summary/>
		public static Version GetVersion()
		{
			int major = 0;
			int minor = 0;
			nvrtcResult res = NVRTCNativeMethods.nvrtcVersion(ref major, ref minor);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcVersion", res));
			if (res != nvrtcResult.Success)
				throw new NVRTCException(res);
			return new Version(major, minor);
		}
		#endregion

		#region Methods
		/// <summary/>
		public void Compile(string[] options)
		{ 
			int optionCount = 0;
			IntPtr[] optionsPtr = null;

			try
			{
				if (options != null)
				{
					optionCount = options.Length;
					optionsPtr = new IntPtr[optionCount];

					for (int i = 0; i < optionCount; i++)
					{
						optionsPtr[i] = Marshal.StringToHGlobalAnsi(options[i]);
					}
				}

				res = NVRTCNativeMethods.nvrtcCompileProgram(_program, optionCount, optionsPtr);
				Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcCompileProgram", res));		
			}
			finally
			{
				if (optionsPtr != null)
					for (int i = 0; i < optionsPtr.Length; i++)
					{
						Marshal.FreeHGlobal(optionsPtr[i]);
					}
			}
			if (res != nvrtcResult.Success)
				throw new NVRTCException(res);
		}

		/// <summary/>
		public byte[] GetPTX()
		{
			SizeT ptxSize = new SizeT();

			res = NVRTCNativeMethods.nvrtcGetPTXSize(_program, ref ptxSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcGetPTXSize", res));	
			if (res != nvrtcResult.Success)
				throw new NVRTCException(res);	
			
			byte[] ptxCode = new byte[ptxSize];

			res = NVRTCNativeMethods.nvrtcGetPTX(_program, ptxCode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcGetPTX", res));
			if (res != nvrtcResult.Success)
				throw new NVRTCException(res);

			return ptxCode;
		}

		/// <summary/>
		public string GetPTXAsString()
		{
			byte[] ptxCode = GetPTX();
			ASCIIEncoding enc = new ASCIIEncoding();

			string ptxString = enc.GetString(ptxCode);
			return ptxString.Replace("\0", "");
		}

		/// <summary/>
		public byte[] GetLog()
		{
			SizeT logSize = new SizeT();

			res = NVRTCNativeMethods.nvrtcGetProgramLogSize(_program, ref logSize);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcGetProgramLogSize", res));
			if (res != nvrtcResult.Success)
				throw new NVRTCException(res);

			byte[] logCode = new byte[logSize];

			res = NVRTCNativeMethods.nvrtcGetProgramLog(_program, logCode);
			Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nvrtcGetProgramLog", res));
			if (res != nvrtcResult.Success)
				throw new NVRTCException(res);

			return logCode;
		}

		/// <summary/>
		public string GetLogAsString()
		{
			byte[] logCode = GetLog();
			ASCIIEncoding enc = new ASCIIEncoding();

			string logString = enc.GetString(logCode);
			return logString.Replace("\0", "");
		}
		#endregion
	}
}
