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

namespace ManagedCuda.NPP
{
    /// <summary>
    /// Singleton NPPWarning handler. Use the <see cref="OnNPPWarning"/> event 
    /// to get notified when a NPP functions returns a NPP warning status code.
    /// </summary>
    public class NPPWarningHandler
    {
        private static volatile NPPWarningHandler _instance;
        private static object _lock = new object();

        private NPPWarningHandler()
        { }

        /// <summary>
        /// Get the singleton instance
        /// </summary>
        /// <returns></returns>
        public static NPPWarningHandler GetInstance()
        {
            if (_instance == null)
            {
                lock (_lock)
                {
                    if (_instance == null)
                    {
                        _instance = new NPPWarningHandler();
                    }
                }
            }
            return _instance;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        public delegate void NPPWarningEventHandler(object sender, NPPWarningEventArgs e);

        /// <summary>
        /// This event is raised by the NPP library if a NPP function returns a warning NPP status code.
        /// </summary>
        public event NPPWarningEventHandler OnNPPWarning;

        internal void RaiseOnNPPWarning(object sender, NPPWarningEventArgs e)
        {
            if (OnNPPWarning != null)
                OnNPPWarning(sender, e);
        }

        internal void NotifyNPPWarning(object sender, NppStatus status, string message)
        {
            NPPWarningEventArgs e = new NPPWarningEventArgs(status, message);
            RaiseOnNPPWarning(sender, e);
        }

        /// <summary>
        /// NPP warning event args
        /// </summary>
        public class NPPWarningEventArgs : EventArgs
        {
            private NppStatus _status;
            private string _message;

            /// <summary>
            /// 
            /// </summary>
            /// <param name="status"></param>
            /// <param name="message"></param>
            public NPPWarningEventArgs(NppStatus status, string message)
            {
                _status = status;
                _message = message;
            }

            /// <summary>
            /// 
            /// </summary>
            public NppStatus Status
            {
                get { return (_status); }
            }

            /// <summary>
            /// 
            /// </summary>
            public string Message
            {
                get { return (_message); }
            }
        }

    }
}
