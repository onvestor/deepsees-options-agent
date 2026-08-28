<#
.SYNOPSIS
    Wrapper the Windows scheduled task invokes. Logs its own start and exit.

.DESCRIPTION
    A scheduled task that fails silently is worse than no scheduled task: the
    session simply does not happen and nothing says so. So this writes a line
    to logs/scheduler.log before it does anything, and another when it exits,
    with the exit code. If the log has a start line and no exit line, the
    process died hard -- which is itself the diagnosis.

    Two modes:
      session   - the real trading session (cli.run_session --live)
      preflight - a read-only account check that exits immediately, used to
                  prove the task actually fires before trusting it with a
                  session

    The account number is passed in at registration and lives in the Windows
    task definition, never in this repository. See CLAUDE.md: account numbers
    are operator state.

.NOTES
    Times here are LOCAL. The machine is Pacific and the market is Eastern, so
    08:45 ET is 05:45 local. Both zones share DST transition dates, so the
    three-hour offset holds year-round -- but the task is registered in local
    time and that is the number to check if a firing ever looks an hour off.
#>
[CmdletBinding()]
param(
    [ValidateSet('session', 'preflight')]
    [string]$Mode = 'session',

    [string]$ExpectAccount = '',

    [string]$RepoRoot = ''
)

$ErrorActionPreference = 'Stop'

# Resolved here, not in a param default: $PSScriptRoot is not bound during
# parameter binding under every invocation path, and an empty default there
# fails before the log exists to record why.
if (-not $RepoRoot) {
    $here = $PSScriptRoot
    if (-not $here) { $here = Split-Path -Parent $MyInvocation.MyCommand.Path }
    $RepoRoot = Split-Path -Parent $here
}

$logDir = Join-Path $RepoRoot 'logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$schedulerLog = Join-Path $logDir 'scheduler.log'

function Write-Line([string]$Message) {
    $utc = (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd HH:mm:ss')
    $et = [System.TimeZoneInfo]::ConvertTime(
        (Get-Date), [System.TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time')
    ).ToString('yyyy-MM-dd HH:mm:ss')
    $line = "$utc UTC | $et ET | $Message"
    Add-Content -Path $schedulerLog -Value $line -Encoding utf8
    Write-Output $line
}

Write-Line "START mode=$Mode host=$env:COMPUTERNAME user=$env:USERNAME repo=$RepoRoot"

try {
    Set-Location $RepoRoot

    $python = (Get-Command python -ErrorAction Stop).Source
    Write-Line "python=$python"

    $stamp = (Get-Date).ToString('yyyyMMdd')

    if ($Mode -eq 'preflight') {
        $args = @('-u', '-m', 'cli.preflight', '--require-level', '3',
                  '--out', (Join-Path $logDir 'preflight.jsonl'))
        if ($ExpectAccount) { $args += @('--expect-account', $ExpectAccount) }
        $out = Join-Path $logDir "preflight-$stamp.log"
    }
    else {
        # Gate the session on the account check. Which account a key pair
        # addresses is a property of the keys, so nothing in the code can tell
        # you a swap went wrong -- only the broker can. Trading the wrong
        # account is not a recoverable mistake, so this refuses to start rather
        # than finding out from the fills.
        #
        # Deliberately NOT --require-flat: this is a swing system that holds
        # overnight by design, so open positions at 08:45 are the normal case.
        if ($ExpectAccount) {
            $gate = @('-u', '-m', 'cli.preflight', '--require-level', '3',
                      '--expect-account', $ExpectAccount,
                      '--out', (Join-Path $logDir 'preflight.jsonl'))
            Write-Line "GATE $python $($gate -join ' ')"
            $prev = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            try {
                & $python @gate 2>&1 | Tee-Object -FilePath (Join-Path $logDir "preflight-$stamp.log") -Append
                $gateCode = $LASTEXITCODE
            }
            finally { $ErrorActionPreference = $prev }
            if ($gateCode -ne 0) {
                Write-Line "ABORT preflight failed (code=$gateCode) -- NOT starting the session"
                exit $gateCode
            }
            Write-Line 'GATE ok'
        }
        else {
            Write-Line 'GATE skipped -- no -ExpectAccount given'
        }

        # --live with no --catch-up: an 08:45 ET start is genuinely pre-market,
        # so Agent 2 fires at 09:05 and Agent 1 at 09:15 from the phase they
        # belong to. --until defaults to just after the close.
        $args = @('-u', '-m', 'cli.run_session', '--live',
                  '--out', (Join-Path $logDir "session_report-$stamp.json"))
        $out = Join-Path $logDir "session-$stamp.log"
    }

    Write-Line "EXEC $python $($args -join ' ')"

    # Windows PowerShell 5.1 wraps every stderr line from a native executable
    # in a NativeCommandError, which under ErrorActionPreference='Stop' throws
    # even when the process exits 0. Python's logging writes to stderr, so the
    # first log line would abort the session. Relaxed for the call itself and
    # restored after; $LASTEXITCODE is the process's real verdict.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $python @args 2>&1 | Tee-Object -FilePath $out -Append
        $code = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previous
    }
    Write-Line "EXIT mode=$Mode code=$code log=$out"
    exit $code
}
catch {
    Write-Line "ERROR mode=$Mode $($_.Exception.Message)"
    exit 1
}
