using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;

namespace FluidsGLCSharp
{
    static class Program
    {
        /// <summary>
        /// Der Haupteinstiegspunkt für die Anwendung.
        /// </summary>
        [STAThread]
        static void Main()
        {
            Console.WriteLine("[" + System.AppDomain.CurrentDomain.FriendlyName + "] starting...");
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());
        }
    }
}
