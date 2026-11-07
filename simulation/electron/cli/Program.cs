using System.Diagnostics;

internal static class Program
{
    private static readonly string Help = """
Rotomolding CLI
Usage: RotomoldingSimulation-cli.exe batch --input <STEP/STP or directory> [--input <...>] --output <result directory> [options]
Usage: RotomoldingSimulation-cli.exe sweep --training-input <directory> --validation-input <directory> --output <result directory> [--gpu-index <N>]
Sweep runs all 25 combinations of 0.2, 0.4, 0.6, 0.8, and 1.0 rad/s on the primary and secondary axes for each model.
Run --list-gpus to enumerate selectable Windows GPUs. Use --gpu-index <N> or --gpu-device <vendorId:deviceId> to require one device.
Options: --mesh-size 0.020 --rotation-speed 0.8 --rotation-speed2 0.3 --fill 0.20 --repose-angle 35 --deposition-rate 0.0005 --termination-condition particles|cycles --termination-cycles 1 --config <JSON file> --gpu-index <N> --gpu-device <vendorId:deviceId> --parallelism <1-16>
Automatic parallelism selects 4 or 6 workers from CPU/GPU headroom; --parallelism overrides it.
Legacy --gpu <default|low-power|high-performance> remains available but only expresses a preference, not a device selection.
Sweep resumes from <output>\sweep-progress.json. Use --retry-failed to retry failed jobs or --prepare-only to create the output workspace without running simulations.
Progress is emitted as JSON Lines. Exit code 0 means success, 1 means one or more simulations failed, and 2 means invalid input.
""";

    private static async Task<int> Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.Title = "Rotomolding Simulation CLI";
            Console.Out.WriteLine(Help);
            WaitForInteractiveClose();
            return 0;
        }

        if (args.Any(arg => arg is "--help" or "-h"))
        {
            Console.Out.WriteLine(Help);
            return 0;
        }

        var baseDirectory = AppContext.BaseDirectory;
        var unpackedAppPath = Path.Combine(baseDirectory, "win-unpacked", "RotomoldingSimulation.exe");
        var appPath = File.Exists(unpackedAppPath) ? unpackedAppPath : Directory.EnumerateFiles(baseDirectory, "RotomoldingSimulation-*-x64.exe")
            .FirstOrDefault(path => !Path.GetFileName(path).Contains("-cli", StringComparison.OrdinalIgnoreCase));
        if (appPath is null)
        {
            Console.Error.WriteLine("{\"type\":\"error\",\"message\":\"RotomoldingSimulation portable executable was not found beside the CLI.\"}");
            return 2;
        }

        var statusFile = Path.Combine(Path.GetTempPath(), $"rotomolding-{Guid.NewGuid():N}.jsonl");
        var startInfo = new ProcessStartInfo(appPath)
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = Environment.CurrentDirectory,
        };
        startInfo.Environment.Remove("ELECTRON_RUN_AS_NODE");
        startInfo.ArgumentList.Add("--cli");
        var expectsPath = false;
        foreach (var arg in args)
        {
            if (expectsPath)
            {
                startInfo.ArgumentList.Add(Path.GetFullPath(arg));
                expectsPath = false;
                continue;
            }

            startInfo.ArgumentList.Add(arg);
            expectsPath = arg is "--input" or "--output" or "--config" or "--status-file" or "--training-input" or "--validation-input";
        }
        if (expectsPath)
        {
            Console.Error.WriteLine("{\"type\":\"error\",\"message\":\"A path option is missing its value.\"}");
            return 2;
        }
        startInfo.ArgumentList.Add("--status-file");
        startInfo.ArgumentList.Add(statusFile);

        using var process = Process.Start(startInfo);
        if (process is null)
        {
            Console.Error.WriteLine("{\"type\":\"error\",\"message\":\"Could not start RotomoldingSimulation.\"}");
            return 2;
        }

        var cancelled = false;
        ConsoleCancelEventHandler cancelHandler = (_, eventArgs) =>
        {
            cancelled = true;
            eventArgs.Cancel = true;
            try
            {
                if (!process.HasExited) process.Kill(entireProcessTree: true);
            }
            catch (InvalidOperationException)
            {
                // The child process exited between the state check and Kill.
            }
        };
        Console.CancelKeyPress += cancelHandler;

        long position = 0;
        var emptyReadsAfterExit = 0;
        var forwardedAnyStatus = false;
        var reportedInputError = false;
        try
        {
            while (!process.HasExited || emptyReadsAfterExit < 3)
            {
                var wroteLine = ForwardNewStatusLines(statusFile, ref position, ref reportedInputError);
                forwardedAnyStatus |= wroteLine;
                if (process.HasExited)
                    emptyReadsAfterExit = wroteLine ? 0 : emptyReadsAfterExit + 1;
                await Task.Delay(100);
            }
            await process.WaitForExitAsync();
            forwardedAnyStatus |= ForwardNewStatusLines(statusFile, ref position, ref reportedInputError);
            if (!forwardedAnyStatus)
                reportedInputError |= await ForwardWholeStatusFile(statusFile);
            return cancelled ? 130 : reportedInputError ? 2 : process.ExitCode;
        }
        finally
        {
            Console.CancelKeyPress -= cancelHandler;
            try { File.Delete(statusFile); } catch { }
        }
    }

    private static void WaitForInteractiveClose()
    {
        if (Console.IsInputRedirected) return;
        Console.Out.WriteLine("Press Enter to close this window.");
        Console.ReadLine();
    }

    private static bool ForwardNewStatusLines(string statusFile, ref long position, ref bool reportedInputError)
    {
        try
        {
            if (!File.Exists(statusFile)) return false;
            var wroteLine = false;
            using var stream = new FileStream(statusFile, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            if (position > stream.Length) position = 0;
            stream.Seek(position, SeekOrigin.Begin);
            using var reader = new StreamReader(stream, leaveOpen: true);
            string line;
            while ((line = reader.ReadLine()) is not null)
            {
                Console.Out.WriteLine(line);
                reportedInputError |= line.Contains("\"type\":\"error\"", StringComparison.Ordinal);
                wroteLine = true;
            }
            position = stream.Position;
            return wroteLine;
        }
        catch (IOException)
        {
            return false;
        }
    }

    private static async Task<bool> ForwardWholeStatusFile(string statusFile)
    {
        for (var attempt = 0; attempt < 10; attempt++)
        {
            try
            {
                if (File.Exists(statusFile))
                {
                    var content = File.ReadAllText(statusFile);
                    if (!string.IsNullOrWhiteSpace(content))
                    {
                        Console.Out.Write(content);
                        return content.Contains("\"type\":\"error\"", StringComparison.Ordinal);
                    }
                }
            }
            catch (IOException)
            {
                // The Electron process can still be releasing the file handle.
            }
            await Task.Delay(100);
        }
        return false;
    }
}
