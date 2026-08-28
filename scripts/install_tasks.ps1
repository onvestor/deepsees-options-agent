<#
.SYNOPSIS
    Register the trading session's Windows scheduled tasks.

.DESCRIPTION
    Two tasks:

      DeepSees-Session    Mon-Fri at 05:45 local (= 08:45 ET). Runs the live
                          session through to just after the close.
      DeepSees-Preflight  One-off, for proving the scheduler actually fires
                          before a session depends on it. Pass -TestAt.

    Both survive a reboot: Windows persists task definitions, and
    -StartWhenAvailable makes a trigger missed while the machine was off run as
    soon as it is back rather than being skipped silently.

.PARAMETER ExpectAccount
    The account number the preflight must see. Stored in the Windows task
    definition, which is operator state outside this repository -- never write
    it into a file under version control.

.PARAMETER TestAt
    Local datetime for a one-off preflight, e.g. '2026-08-29 09:00'.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_tasks.ps1 `
        -ExpectAccount PA0000000000 -TestAt '2026-08-29 09:00'

.NOTES
    08:45 ET is 05:45 local on a Pacific machine. Both zones observe DST on the
    same dates, so the offset is stable; if a firing ever looks an hour out,
    the task's local time is the thing to check.
#>
[CmdletBinding()]
param(
    [string]$ExpectAccount = '',
    [string]$TestAt = '',
    [string]$SessionTimeLocal = '05:45',
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$wrapper = Join-Path $repo 'scripts\scheduled_run.ps1'
if (-not (Test-Path $wrapper)) { throw "wrapper not found: $wrapper" }

$SESSION_TASK = 'DeepSees-Session'
$PREFLIGHT_TASK = 'DeepSees-Preflight'

function Remove-IfPresent([string]$Name) {
    $existing = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $Name -Confirm:$false
        Write-Output "removed existing task $Name"
    }
}

if ($Remove) {
    Remove-IfPresent $SESSION_TASK
    Remove-IfPresent $PREFLIGHT_TASK
    Write-Output 'done'
    return
}

# Shared settings. StartWhenAvailable is the reboot-survival half: the task
# definition persists on its own, but a trigger that fired while the machine
# was off is dropped unless this is set.
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 10) `
    -MultipleInstances IgnoreNew `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 5)

# S4U runs the task whether or not the user is logged on, without storing a
# password -- which is what an unattended 08:45 session needs. Registering it
# requires elevation, so this falls back to Interactive when not elevated and
# says so loudly: an Interactive task fires ONLY while this user is logged on,
# and a machine sitting at the lock screen on Monday morning would simply not
# trade. That is a real downgrade, not a detail.
$elevated = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

$logon = if ($elevated) { 'S4U' } else { 'Interactive' }
$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType $logon `
    -RunLevel Limited

if (-not $elevated) {
    Write-Warning ('Not elevated: registering with LogonType=Interactive. The task ' +
                   'will fire ONLY while ' + $env:USERNAME + ' is logged on. To make ' +
                   'it run regardless, re-run this script from an elevated PowerShell.')
}

function New-WrapperAction([string]$Mode) {
    $argline = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$wrapper`" -Mode $Mode"
    if ($ExpectAccount) { $argline += " -ExpectAccount $ExpectAccount" }
    New-ScheduledTaskAction `
        -Execute 'powershell.exe' `
        -Argument $argline `
        -WorkingDirectory $repo
}

# --- the weekday session ----------------------------------------------------

Remove-IfPresent $SESSION_TASK
$sessionTrigger = New-ScheduledTaskTrigger `
    -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At $SessionTimeLocal

Register-ScheduledTask `
    -TaskName $SESSION_TASK `
    -Description ("DeepSees live trading session. Fires $SessionTimeLocal local " +
                  '(08:45 ET) Mon-Fri and runs to just after the close.') `
    -Action (New-WrapperAction 'session') `
    -Trigger $sessionTrigger `
    -Settings $settings `
    -Principal $principal | Out-Null

$et = [System.TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time')
$next = (Get-ScheduledTaskInfo -TaskName $SESSION_TASK).NextRunTime
Write-Output "registered $SESSION_TASK -- $SessionTimeLocal local, Mon-Fri"
if ($next) {
    $nextEt = [System.TimeZoneInfo]::ConvertTime($next, $et)
    Write-Output "  next run: $next local = $nextEt ET"
}

# --- the one-off proving run ------------------------------------------------

if ($TestAt) {
    Remove-IfPresent $PREFLIGHT_TASK
    $when = [datetime]::Parse($TestAt)
    Register-ScheduledTask `
        -TaskName $PREFLIGHT_TASK `
        -Description ('One-off read-only account check. Proves the scheduler ' +
                      'fires before a session depends on it.') `
        -Action (New-WrapperAction 'preflight') `
        -Trigger (New-ScheduledTaskTrigger -Once -At $when) `
        -Settings $settings `
        -Principal $principal | Out-Null

    $nextP = (Get-ScheduledTaskInfo -TaskName $PREFLIGHT_TASK).NextRunTime
    Write-Output "registered $PREFLIGHT_TASK -- one-off at $when local"
    if ($nextP) {
        Write-Output ("  next run: $nextP local = " +
                      "$([System.TimeZoneInfo]::ConvertTime($nextP, $et)) ET")
    }
}

Write-Output ''
Write-Output "logon type: $logon$(if (-not $elevated) {' (requires an interactive login -- re-run elevated for S4U)'})"
Write-Output ''
Write-Output 'verify with:'
Write-Output "  Get-ScheduledTask -TaskName 'DeepSees-*' | Get-ScheduledTaskInfo"
Write-Output '  Get-Content logs\scheduler.log -Tail 20'
