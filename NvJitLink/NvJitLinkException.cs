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
using System.Runtime.Serialization;

namespace ManagedCuda.NvJitLink
{
    /// <summary>
    /// An NvJitLinkException is thrown, if any wrapped call to the NvJitLink-library does not return <see cref="nvJitLinkResult.Success"/>.
    /// </summary>
    public class NvJitLinkException : Exception, System.Runtime.Serialization.ISerializable
    {
        private nvJitLinkResult _NvJitLinkError;

        #region Constructors
        /// <summary>
        /// 
        /// </summary>
        public NvJitLinkException()
        {

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="serInfo"></param>
        /// <param name="streamingContext"></param>
        protected NvJitLinkException(SerializationInfo serInfo, StreamingContext streamingContext)
            : base(serInfo, streamingContext)
        {
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="error"></param>
        public NvJitLinkException(nvJitLinkResult error)
            : base(GetErrorMessageFromCUResult(error))
        {
            this._NvJitLinkError = error;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="message"></param>
        public NvJitLinkException(string message)
            : base(message)
        {

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="message"></param>
        /// <param name="exception"></param>
        public NvJitLinkException(string message, Exception exception)
            : base(message, exception)
        {

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="error"></param>
        /// <param name="message"></param>
        /// <param name="exception"></param>
        public NvJitLinkException(nvJitLinkResult error, string message, Exception exception)
            : base(message, exception)
        {
            this._NvJitLinkError = error;
        }
        #endregion

        #region Methods
        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return this.nvJitLinkError.ToString();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="info"></param>
        /// <param name="context"></param>
        public override void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            base.GetObjectData(info, context);
            info.AddValue("NvJitLinkError", this._NvJitLinkError);
        }
        #endregion

        #region Static methods
        private static string GetErrorMessageFromCUResult(nvJitLinkResult error)
        {
            string message = string.Empty;

            switch (error)
            {
                case nvJitLinkResult.Success:
                    message = "No Error.";
                    break;
                case nvJitLinkResult.ErrorUnrecognizedOption:
                    message = "ErrorUnrecognizedOption";
                    break;
                case nvJitLinkResult.ErrorMissingArch:
                    message = "ErrorMissingArch: -arch=sm_NN option not specified";
                    break;
                case nvJitLinkResult.ErrorInvalidInput:
                    message = "ErrorInvalidInput";
                    break;
                case nvJitLinkResult.ErrorPtxCompile:
                    message = "ErrorPtxCompile";
                    break;
                case nvJitLinkResult.ErrorNVVMCompile:
                    message = "ErrorNVVMCompile";
                    break;
                case nvJitLinkResult.ErrorInternal:
                    message = "ErrorInternal";
                    break;
                case nvJitLinkResult.ErrorThreadPool:
                    message = "ErrorThreadPool";
                    break;
                case nvJitLinkResult.UnrecognizedInput:
                    message = "UnrecognizedInput";
                    break;
                case nvJitLinkResult.ErrorFinalize:
                    message = "ErrorFinalize";
                    break;
                case nvJitLinkResult.NullInput:
                    message = "NullInput";
                    break;
                case nvJitLinkResult.IncompatibleOptions:
                    message = "IncompatibleOptions";
                    break;
                case nvJitLinkResult.IncorrectInputType:
                    message = "IncorrectInputType";
                    break;
                case nvJitLinkResult.ArchMismatch:
                    message = "ArchMismatch";
                    break;
                case nvJitLinkResult.OutdatedLibrary:
                    message = "OutdatedLibrary";
                    break;
                case nvJitLinkResult.MissingFatBin:
                    message = "MissingFatBin";
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
        public nvJitLinkResult nvJitLinkError
        {
            get
            {
                return this._NvJitLinkError;
            }
            set
            {
                this._NvJitLinkError = value;
            }
        }
        #endregion
    }
}
