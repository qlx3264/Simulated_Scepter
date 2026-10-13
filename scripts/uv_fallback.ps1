<#
.SYNOPSIS
    在多个 PyPI 镜像源上依次执行同一条 uv 命令，某个源不可用时自动切换下一个。

.DESCRIPTION
    uv 不会在某个索引源不可达时改用其它索引源（同一次解析只在当前源上重试），
    所以多源回退只能由调用方逐个尝试。脚本每次尝试用 UV_DEFAULT_INDEX 把默认索引
    换成当前镜像源：该变量优先于 pyproject.toml 里的 [[tool.uv.index]]，因此每次尝试
    只使用一个源，不会与项目声明的清华源叠加。

    换源会让 uv 重新解析依赖，并把 uv.lock 里记录的源地址改成当前镜像源；脚本在结束时
    把 uv.lock 恢复成运行前的内容，避免仓库里出现与本次安装无关的锁文件改动。

    全部镜像源都失败时，脚本以最后一次 uv 的退出码结束（非 0），调用方据此报错停止；
    调用点应当检查 $LASTEXITCODE。

.PARAMETER Executable
    要执行的 uv 可执行文件，默认 uv；需要回退 uvx（uv tool run）时传 uvx。

.PARAMETER Arguments
    传给 Executable 的参数，默认 @('sync')；多个参数可以直接传数组，也可以写成空格分隔的
    一个字符串（用 `powershell -File` 调用时参数只会作为一个字符串传进来）。

.PARAMETER Mirrors
    覆盖镜像源顺序，用于自定义源或在验证中检查回退逻辑；与 Arguments 一样支持数组或
    空格分隔的字符串。

.NOTES
    本文件保存为带 BOM 的 UTF-8：Windows PowerShell 5.1 没有 BOM 时会按 ANSI 解析，
    中文注释会变成乱码并导致语法错误。修改后请保留 BOM，或改用纯 ASCII 文案。
#>
[CmdletBinding()]
param(
    [string]$Executable = 'uv',

    [string[]]$Arguments = @('sync'),

    [string[]]$Mirrors
)

$ErrorActionPreference = 'Stop'

# 直接调用脚本时参数是数组；用 `powershell -File script.ps1 -Arguments 'a b'` 调用时
# 整个字符串只会作为一个元素传进来，所以这里再按空格展开一次。只按空格拆分是为了保留
# 参数内部的逗号（例如 ruff 的 --select=F821,E722,F403）。
function Expand-ListArgument {
    param([string[]]$Values)

    $expanded = @()
    foreach ($value in @($Values)) {
        if ([string]::IsNullOrWhiteSpace($value)) {
            continue
        }
        $expanded += @($value -split '\s+' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    }
    return $expanded
}

$uvArguments = @(Expand-ListArgument -Values $Arguments)

# 镜像源顺序：清华源与原行为一致排在首位，其后是国内备用源，最后是官方 PyPI。
$DefaultMirrors = @(
    'https://pypi.tuna.tsinghua.edu.cn/simple',  # 清华大学 TUNA
    'https://mirrors.aliyun.com/pypi/simple/',   # 阿里云
    'https://mirrors.ustc.edu.cn/pypi/simple/',  # 中国科学技术大学
    'https://pypi.org/simple/'                   # Python 官方索引
)

$mirrorList = @($DefaultMirrors)
if ($PSBoundParameters.ContainsKey('Mirrors')) {
    $mirrorList = @(Expand-ListArgument -Values $Mirrors)
}

if ($mirrorList.Count -eq 0) {
    Write-Host '[uv-fallback] 错误：没有可用的镜像源，无法继续。'
    exit 1
}

try {
    $uv = (Get-Command -Name $Executable -ErrorAction Stop).Source
} catch {
    Write-Host "[uv-fallback] 错误：找不到可执行文件 '$Executable'，请先按项目文档准备 uv 环境。"
    exit 1
}

# uv 与脚本都从当前目录向上查找项目；这里用同样的方式定位锁文件，保证能恢复它。
$lockPath = $null
$lockDir = Get-Item -LiteralPath (Get-Location).Path
while ($null -ne $lockDir) {
    $candidate = Join-Path $lockDir.FullName 'uv.lock'
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $lockPath = $candidate
        break
    }
    $lockDir = $lockDir.Parent
}

$lockBefore = $null
$lockHashBefore = $null
if ($null -ne $lockPath) {
    $lockBefore = [System.IO.File]::ReadAllBytes($lockPath)
    $lockHashBefore = (Get-FileHash -LiteralPath $lockPath -Algorithm SHA256).Hash
}

function Restore-UvLock {
    if (($null -eq $lockPath) -or ($null -eq $lockBefore)) {
        return
    }
    if (-not (Test-Path -LiteralPath $lockPath -PathType Leaf)) {
        return
    }
    $hashAfter = (Get-FileHash -LiteralPath $lockPath -Algorithm SHA256).Hash
    if ($hashAfter -eq $lockHashBefore) {
        return
    }
    [System.IO.File]::WriteAllBytes($lockPath, $lockBefore)
    Write-Host '[uv-fallback] uv.lock 已恢复为运行前内容：本次安装的镜像源不写回仓库。'
}

$previousIndex = $env:UV_DEFAULT_INDEX
$attempt = 0
$usedMirror = $null
# 注意：不要用 $LASTEXITCODE 之外的大小写变体命名（如 $lastExitCode），
# PowerShell 变量名不区分大小写，那会遮蔽自动变量、读不到子进程真实退出码。
$failureCode = 1

try {
    foreach ($mirror in $mirrorList) {
        $attempt++
        Write-Host "[uv-fallback] 镜像源 $attempt/$($mirrorList.Count)：$mirror"
        $env:UV_DEFAULT_INDEX = $mirror

        $exitCode = 1
        try {
            & $uv @uvArguments
            $exitCode = $LASTEXITCODE
        } catch {
            Write-Host "[uv-fallback] 执行 '$Executable $($uvArguments -join ' ')' 失败：$($_.Exception.Message)"
            $exitCode = 1
        }

        if ($exitCode -eq 0) {
            $usedMirror = $mirror
            $failureCode = 0
            break
        }

        $failureCode = $exitCode
        Write-Host "[uv-fallback] 该源失败（退出码 $exitCode），切换下一个源。"
    }
} finally {
    if ($null -eq $previousIndex) {
        Remove-Item Env:UV_DEFAULT_INDEX -ErrorAction SilentlyContinue
    } else {
        $env:UV_DEFAULT_INDEX = $previousIndex
    }
    Restore-UvLock
}

if ($null -ne $usedMirror) {
    Write-Host "[uv-fallback] 完成，使用镜像源：$usedMirror"
} else {
    Write-Host "[uv-fallback] 错误：$($mirrorList.Count) 个镜像源全部失败（最后退出码 $failureCode），已按顺序尝试：$($mirrorList -join '、')"
}

exit $failureCode
