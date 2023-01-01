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


#define ADD_MISSING_CTX
using System;
using System.Diagnostics;

namespace ManagedCuda.NPP
{
    /// <summary>
    /// 
    /// </summary>
    public partial class NPPImage_16fC2 : NPPImageBase
    {
        #region ColorTwist
        /// <summary>
        /// An input color twist matrix with floating-point pixel values is applied
        /// within ROI.
        /// </summary>
        /// <param name="dest">Destination image</param>
        /// <param name="twistMatrix">The color twist matrix with floating-point pixel values [3,4].</param>
        /// <param name="nppStreamCtx">NPP stream context.</param>
        public void ColorTwist(NPPImage_16fC2 dest, float[,] twistMatrix, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.ColorTwist.nppiColorTwist32f_16f_C2R_Ctx(_devPtrRoi, _pitch, dest.DevicePointerRoi, dest.Pitch, _sizeRoi, twistMatrix, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist32f_16f_C2R_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        /// <summary>
        /// in place color twist.
        /// 
        /// An input color twist matrix with floating-point coefficient values is applied
        /// within ROI.
        /// </summary>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values. [3,4]</param>
        /// <param name="nppStreamCtx">NPP stream context.</param>
        public void ColorTwist(float[,] aTwist, NppStreamContext nppStreamCtx)
        {
            status = NPPNativeMethods_Ctx.NPPi.ColorTwist.nppiColorTwist32f_16f_C2IR_Ctx(_devPtrRoi, _pitch, _sizeRoi, aTwist, nppStreamCtx);
            Debug.WriteLine(String.Format("{0:G}, {1}: {2}", DateTime.Now, "nppiColorTwist32f_16f_C2IR_Ctx", status));
            NPPException.CheckNppStatus(status, this);
        }

        #endregion
    }
}
