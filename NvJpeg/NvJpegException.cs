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

namespace ManagedCuda.NvJpeg
{
    /// <summary>
    /// An NvJpegException is thrown, if any wrapped call to the NvJpeg-library does not return <see cref="nvjpegStatus.Success"/>.
    /// </summary>
    public class NvJpegException : Exception, ISerializable
    {

        private nvjpegStatus _nvjpegError;

        #region Constructors
        /// <summary>
        /// 
        /// </summary>
        public NvJpegException()
        {

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="serInfo"></param>
        /// <param name="streamingContext"></param>
        protected NvJpegException(SerializationInfo serInfo, StreamingContext streamingContext)
            : base(serInfo, streamingContext)
        {
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="error"></param>
        public NvJpegException(nvjpegStatus error)
            : base(GetErrorMessageFromCUResult(error))
        {
            this._nvjpegError = error;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="message"></param>
        public NvJpegException(string message)
            : base(message)
        {

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="message"></param>
        /// <param name="exception"></param>
        public NvJpegException(string message, Exception exception)
            : base(message, exception)
        {

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="error"></param>
        /// <param name="message"></param>
        /// <param name="exception"></param>
        public NvJpegException(nvjpegStatus error, string message, Exception exception)
            : base(message, exception)
        {
            this._nvjpegError = error;
        }
        #endregion

        #region Methods
        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return this._nvjpegError.ToString();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="info"></param>
        /// <param name="context"></param>
        public override void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            base.GetObjectData(info, context);
            info.AddValue("NvJpegError", this._nvjpegError);
        }
        #endregion

        #region Static methods
        private static string GetErrorMessageFromCUResult(nvjpegStatus error)
        {
            string message = string.Empty;

            switch (error)
            {
                case nvjpegStatus.Success:
                    message = "The API call has finished successfully. Note that many of the calls are asynchronous and some of the errors may be seen only after synchronization. ";
                    break;
                case nvjpegStatus.NotInitialized:
                    message = "The library handle was not initialized. A call to nvjpegCreate() is required to initialize the handle.";
                    break;
                case nvjpegStatus.InvalidParameter:
                    message = "Wrong parameter was passed. For example, a null pointer as input data, or an image index not in the allowed range.";
                    break;
                case nvjpegStatus.BadJPEG:
                    message = "Cannot parse the JPEG stream. Check that the encoded JPEG stream and its size parameters are correct.";
                    break;
                case nvjpegStatus.JPEGNotSupported:
                    message = "Attempting to decode a JPEG stream that is not supported by the nvJPEG library.";
                    break;
                case nvjpegStatus.AllocatorFailure:
                    message = "The user-provided allocator functions, for either memory allocation or for releasing the memory, returned a non-zero code.";
                    break;
                case nvjpegStatus.ExecutionFailed:
                    message = "Error during the execution of the device tasks.";
                    break;
                case nvjpegStatus.ArchMismatch:
                    message = "The device capabilities are not enough for the set of input parameters provided (input parameters such as backend, encoded stream parameters, output format).";
                    break;
                case nvjpegStatus.InternalError:
                    message = "Error during the execution of the device tasks.";
                    break;
                case nvjpegStatus.ImplementationNotSupported:
                    message = "Not supported.";
                    break;
                case nvjpegStatus.IncompleteBitstream:
                    message = "Incomplete bit stream.";
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
        public nvjpegStatus NvJpegError
        {
            get
            {
                return this._nvjpegError;
            }
            set
            {
                this._nvjpegError = value;
            }
        }
        #endregion
    }
}

