﻿// Copyright (c) 2023, Michael Kunz and Artic Imaging SARL. All rights reserved.
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
using System.Runtime.Serialization;

namespace ManagedCuda.CudaFFT
{
    /// <summary>
    /// An CudaFFTException is thrown, if any wrapped call to the CUFFT-library does not return <see cref="cufftResult.Success"/>.
    /// </summary>
    public class CudaFFTException : Exception, ISerializable
    {
        private cufftResult _cudaFFTError;

        #region Constructors
        /// <summary>
        /// 
        /// </summary>
        public CudaFFTException()
        {

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="serInfo"></param>
        /// <param name="streamingContext"></param>
        protected CudaFFTException(SerializationInfo serInfo, StreamingContext streamingContext)
            : base(serInfo, streamingContext)
        {
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="error"></param>
        public CudaFFTException(cufftResult error)
            : base(GetErrorMessageFromCUResult(error))
        {
            this._cudaFFTError = error;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="message"></param>
        public CudaFFTException(string message)
            : base(message)
        {

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="message"></param>
        /// <param name="exception"></param>
        public CudaFFTException(string message, Exception exception)
            : base(message, exception)
        {

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="error"></param>
        /// <param name="message"></param>
        /// <param name="exception"></param>
        public CudaFFTException(cufftResult error, string message, Exception exception)
            : base(message, exception)
        {
            this._cudaFFTError = error;
        }
        #endregion

        #region Methods
        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return this.CudaFFTError.ToString();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="info"></param>
        /// <param name="context"></param>
        public override void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            base.GetObjectData(info, context);
            info.AddValue("CudaFFTError", this._cudaFFTError);
        }
        #endregion

        #region Static methods
        private static string GetErrorMessageFromCUResult(cufftResult error)
        {
            string message = string.Empty;

            switch (error)
            {
                case cufftResult.Success:
                    message = "Any CUFFT operation is successful.";
                    break;
                case cufftResult.InvalidPlan:
                    message = "CUFFT is passed an invalid plan handle.";
                    break;
                case cufftResult.AllocFailed:
                    message = "CUFFT failed to allocate GPU memory.";
                    break;
                case cufftResult.InvalidType:
                    message = "The user requests an unsupported type.";
                    break;
                case cufftResult.InvalidValue:
                    message = "The user specifies a bad memory pointer.";
                    break;
                case cufftResult.InternalError:
                    message = "Used for all internal driver errors.";
                    break;
                case cufftResult.ExecFailed:
                    message = "CUFFT failed to execute an FFT on the GPU.";
                    break;
                case cufftResult.SetupFailed:
                    message = "The CUFFT library failed to initialize.";
                    break;
                case cufftResult.InvalidSize:
                    message = "The user specifies an unsupported FFT size.";
                    break;
                case cufftResult.UnalignedData:
                    message = "Input or output does not satisfy texture alignment requirements.";
                    break;
                case cufftResult.IncompleteParameterList:
                    message = "Missing parameter in call.";
                    break;
                case cufftResult.InvalidDevice:
                    message = "Plan creation and execution are on different device.";
                    break;
                case cufftResult.ParseError:
                    message = "Internal plan database error.";
                    break;
                case cufftResult.NoWorkspace:
                    message = "Workspace not initialized.";
                    break;
                case cufftResult.NotImplemented:
                    message = "Not implemented.";
                    break;
                case cufftResult.LicenseError:
                    message = "License error.";
                    break;
                case cufftResult.NotSupported:
                    message = "Not supported error.";
                    break;
                default:
                    break;
            }


            return error.ToString() + ": " + message;
        }
        #endregion

        #region Properties
        /// <summary>
        /// 
        /// </summary>
        public cufftResult CudaFFTError
        {
            get
            {
                return this._cudaFFTError;
            }
            set
            {
                this._cudaFFTError = value;
            }
        }
        #endregion
    }
}
