$dir = "C:\Users\satas\Downloads\AI-GP Simulator v1.0.3385-VQ1\PyAIPilotExample-v1"
$py  = "C:\Users\satas\projects\ai-grand-prix\.venv\Scripts\python.exe"
$log = "$dir\racer_mujoco\runs_mj\dash.log"
$cmd = "cmd.exe /c `"cd /d `"`"$dir`"`" && `"`"$py`"`" -u -m racer_mujoco.dashboard --port 8060 > `"`"$log`"`" 2>&1`""
$p = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd }
Write-Output "dashboard pid=$($p.ProcessId) rc=$($p.ReturnValue)"
