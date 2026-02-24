using System;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;

namespace VisionOS.Receiver
{
    class Program
    {
        // Import native Windows API for literal ZERO latency mouse control
        [DllImport("user32.dll")]
        static extern bool SetCursorPos(int X, int Y);

        [DllImport("user32.dll")]
        static extern void mouse_event(uint dwFlags, int dx, int dy, int dwData, int dwExtraInfo);

        private const uint MOUSEEVENTF_LEFTDOWN = 0x0002;
        private const uint MOUSEEVENTF_LEFTUP = 0x0004;
        private const uint MOUSEEVENTF_RIGHTDOWN = 0x0008;
        private const uint MOUSEEVENTF_RIGHTUP = 0x0010;
        private const uint MOUSEEVENTF_WHEEL = 0x0800;

        static void Main(string[] args)
        {
            // Open UDP Port 5005 to listen to Python
            UdpClient listener = new UdpClient(5005);
            IPEndPoint groupEP = new IPEndPoint(IPAddress.Any, 5005);

            Console.WriteLine("🟢 Vision OS C# Core Online.");
            Console.WriteLine("Listening for zero-latency Python telemetry on port 5005...");

            try
            {
                while (true)
                {
                    // Catch the data packet from Python
                    byte[] bytes = listener.Receive(ref groupEP);
                    string[] cmd = Encoding.UTF8.GetString(bytes).Split('|');

                    // Execute the command instantly via Windows OS
                    switch (cmd[0])
                    {
                        case "MOVE":
                            SetCursorPos(int.Parse(cmd[1]), int.Parse(cmd[2]));
                            break;
                        case "LDOWN":
                            mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
                            break;
                        case "LUP":
                            mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
                            break;
                        case "RCLICK":
                            mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0);
                            mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0);
                            break;
                        case "SCROLL":
                            mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int.Parse(cmd[1]), 0);
                            break;
                    }
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("Error: " + e.Message);
            }
            finally
            {
                listener.Close();
            }
        }
    }
}