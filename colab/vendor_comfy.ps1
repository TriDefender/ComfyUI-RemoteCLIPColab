param(
    [Parameter(Mandatory = $true)]
    [string]$ComfyRepo
)
# Syncs the text-encoder stack from a ComfyUI checkout into ./comfy.
# Only the `comfy/` package is copied (minus caches); ComfyUI server files are not needed.
$ErrorActionPreference = "Stop"
$repo = Resolve-Path $ComfyRepo
$dst = Join-Path $PSScriptRoot "comfy"
$src = Join-Path $repo "comfy"
if (-not (Test-Path $src)) { throw "no comfy/ package under $repo" }
if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
Copy-Item -Recurse $src $dst
# comfy/ depends on one repo-root module (node_helpers, used by comfy/hooks.py)
Copy-Item (Join-Path $repo "node_helpers.py") (Join-Path $PSScriptRoot "node_helpers.py") -Force
Get-ChildItem -Recurse $dst -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Recurse $dst -Include "*.pyc" -File | Remove-Item -Force
@'
Vendored from ComfyUI (https://github.com/comfyanonymous/ComfyUI), which is
licensed under the GNU General Public License v3.0. Copies here are modified
only by the sync script (vendor_comfy.ps1); see the upstream repository for
the source of each file and its full license text.
'@ | Set-Content (Join-Path $dst "NOTICE") -Encoding utf8
$files = (Get-ChildItem -Recurse $dst -File).Count
$mb = [math]::Round(((Get-ChildItem -Recurse $dst -File | Measure-Object Length -Sum).Sum / 1MB), 1)
Write-Host "vendored comfy/ -> $files files, $mb MB"
