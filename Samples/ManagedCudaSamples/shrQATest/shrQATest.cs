/*
 * This code is based on code from the NVIDIA CUDA SDK. (Ported from C++ to C# using managedCUDA)
 * This software contains source code provided by NVIDIA Corporation.
 *
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;

namespace shrQATest
{
	public static class ShrQATest
	{
		const int EXIT_SUCCESS = 0;
		const int EXIT_FAILURE = 1;

		public static void shrQAStart(string[] args)
		{
			bool bQATest = false;
			for (int i = 1; i < args.Length; i++)
			{
				if (args[i].ToLower().Contains("-qatest"))
					bQATest = true;
			}

			if (bQATest)
			{
				Console.Write("&&&& RUNNING {0}", Assembly.GetEntryAssembly().GetName().Name + ".exe");
				foreach (var item in args)
				{
					Console.Write(" ");
					Console.Write(item);
				}
				Console.WriteLine();
			}
			else
			{
				Console.WriteLine("[{0}] starting...", Assembly.GetEntryAssembly().GetName().Name + ".exe");
			}
		}

		public enum eQAstatus
		{
			QA_FAILED = 0,
			QA_PASSED = 1,
			QA_WAIVED = 2
		};

		public static void ExitInTime(int seconds)
		{
			Console.Write("> exiting in {0} seconds: ", seconds);

			DateTime now = DateTime.Now;
			DateTime then = now.AddSeconds(seconds);

			while (now < then)
			{
				Console.Write(Math.Round(((double)((then - now).TotalMilliseconds))/1000.0));
				Console.Write("...");
				Console.Out.Flush();
				System.Threading.Thread.Sleep(1000);
				now = DateTime.Now;
			}
			Console.WriteLine("done!");
		}


		public static void shrQAFinish(string[] args, eQAstatus iStatus)
		{
			// By default QATest is disabled and NoPrompt is Enabled (times out at seconds passed into ExitInTime() )
			bool bQATest = false, bNoPrompt = true, bQuitInTime = true;
			string[] sStatus = new string[] { "FAILED", "PASSED", "WAIVED" };

			for (int i = 1; i < args.Length; i++)
			{
				if (args[i].ToLower().Contains("-qatest"))
					bQATest = true;

				// For SDK individual samples that don't specify -noprompt or -prompt, 
				// a 3 second delay will happen before exiting, giving a user time to view results
				if (args[i].ToLower().Contains("-noprompt"))
				{
					bNoPrompt = true;
					bQuitInTime = false;
				}

				if (args[i].ToLower().Contains("-prompt"))
				{
					bNoPrompt = false;
					bQuitInTime = false;
				}
			}

			if (bQATest)
			{
				Console.Write("&&&& {0} {1}", sStatus[(int)iStatus], Assembly.GetEntryAssembly().GetName().Name + ".exe");
				foreach (var item in args)
				{
					Console.Write(" ");
					Console.Write(item);
				}
				Console.WriteLine();
			}
			else
			{
				Console.Write("[{0}] test results...\n{1}\n", Assembly.GetEntryAssembly().GetName().Name + ".exe", sStatus[(int)iStatus]);
			}

			if (bQuitInTime)
			{
				ExitInTime(3);
			}
			else
			{
				if (!bNoPrompt)
				{
					Console.WriteLine("\nPress <Enter> to exit...");
					Console.ReadLine();
				}
			}
		}

		public static void shrQAFinish2(bool bQATest, string[] args, eQAstatus iStatus)
		{
			bool bQuitInTime = true;
			string[] sStatus = new string[] { "FAILED", "PASSED", "WAIVED" };

			for (int i = 1; i < args.Length; i++)
			{
				if (args[i].ToLower().Contains("-qatest"))
					bQATest = true;

				// For SDK individual samples that don't specify -noprompt or -prompt, 
				// a 3 second delay will happen before exiting, giving a user time to view results
				if (args[i].ToLower().Contains("-noprompt"))
				{
					bQuitInTime = false;
				}

				if (args[i].ToLower().Contains("-prompt"))
				{
					bQuitInTime = false;
				}
			}

			if (bQATest)
			{
				Console.Write("&&&& {0} {1}", sStatus[(int)iStatus], Assembly.GetEntryAssembly().GetName().Name + ".exe");
				foreach (var item in args)
				{
					Console.Write(" ");
					Console.Write(item);
				}
				Console.WriteLine();
			}
			else
			{
				Console.Write("[{0}] test results...\n{1}\n", Assembly.GetEntryAssembly().GetName().Name + ".exe", sStatus[(int)iStatus]);
			}

			if (bQuitInTime)
			{
				ExitInTime(3);
			}
		}

		public static void shrQAFinishExit(string[] args, eQAstatus iStatus)
		{
			shrQAFinish(args, iStatus);
			Environment.Exit(iStatus == eQAstatus.QA_PASSED ? EXIT_SUCCESS : EXIT_FAILURE);
		}

		public static void shrQAFinishExit2(bool bQAtest, string[] args, eQAstatus iStatus)
		{
			shrQAFinish2(bQAtest, args, iStatus);
			Environment.Exit(iStatus == eQAstatus.QA_PASSED ? EXIT_SUCCESS : EXIT_FAILURE);
		}
	}
}
