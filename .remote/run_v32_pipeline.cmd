@echo off
cd /d C:\Users\henry\aigp
>worldmodel\v32_allgate_registry_flywheel\pipeline_status.txt echo training
call .remote\run_v32_allgate_registry_flywheel.cmd
if errorlevel 1 (
  >worldmodel\v32_allgate_registry_flywheel\pipeline_status.txt echo training_failed
  exit /b 1
)
>worldmodel\v32_allgate_registry_flywheel\pipeline_status.txt echo auditing
call .remote\audit_v32_allgate_registry_flywheel.cmd
if errorlevel 1 (
  >worldmodel\v32_allgate_registry_flywheel\pipeline_status.txt echo audit_failed
  exit /b 1
)
>worldmodel\v32_allgate_registry_flywheel\pipeline_status.txt echo complete
exit /b 0
