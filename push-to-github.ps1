<#
    push-to-github.ps1 —— 一键把本仓库发布到 GitHub

    它会自动完成：
      1. 设置 git 身份（如果本机还没设）
      2. 用 GitHub API 创建仓库 HAZAKIKO/lunar-shadow-algorithms（已存在则跳过）
      3. 添加/校正 remote
      4. 推送并建立上游分支

    运行方式：右键本文件 ->「使用 PowerShell 运行」
    或者在本目录打开终端执行：  powershell -ExecutionPolicy Bypass -File .\push-to-github.ps1

    需要：本机已安装 Git（https://git-scm.com/download/win）
    推送时会弹出浏览器让你登录 GitHub 授权 —— 这是 GitHub 的账号验证，必须你本人点一下。
#>

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$User   = 'HAZAKIKO'
$Email  = '13335195392@163.com'
$Repo   = 'lunar-shadow-algorithms'
$Desc   = '月面光照与阴影算法：基于 DEM 的阴影正演（射线行进 + 地平线图）、图像阴影反演与太阳几何重建，含 29 项自检'

function Say($msg, $color = 'Gray') { Write-Host $msg -ForegroundColor $color }

Say "`n=== 1/4 检查 Git ===" Cyan
$git = (Get-Command git -ErrorAction SilentlyContinue)
if (-not $git) {
    Say "  ✗ 没找到 git 命令。" Red
    Say "    请先安装 Git for Windows：https://git-scm.com/download/win"
    Say "    装完关闭并重新打开终端，再运行本脚本。"
    Read-Host "`n按回车退出"
    exit 1
}
Say "  ✓ $($git.Source)"
& git --version

Say "`n=== 2/4 设置身份 ===" Cyan
if (-not (git config --global user.name))  { & git config --global user.name  $User }
if (-not (git config --global user.email)) { & git config --global user.email $Email }
& git config --global init.defaultBranch main
& git config --global core.quotepath false
Say "  ✓ user.name  = $(& git config --global user.name)"
Say "  ✓ user.email = $(& git config --global user.email)"

Say "`n=== 3/4 检查 GitHub 上的仓库 ===" Cyan
$exists = $false
try {
    $r = Invoke-WebRequest -Uri "https://api.github.com/repos/$User/$Repo" `
                           -Headers @{ 'User-Agent' = 'push-script' } -TimeoutSec 30 -UseBasicParsing
    if ($r.StatusCode -eq 200) { $exists = $true }
} catch {
    if ($_.Exception.Response.StatusCode.value__ -eq 404) { $exists = $false }
    else { Say "  ! 无法确认仓库状态（$($_.Exception.Message)），继续尝试推送。" Yellow }
}

if ($exists) {
    Say "  ✓ 仓库已存在，跳过创建（若为空仓库可直接推送）"
    Say "    注意：如果该仓库已有提交（例如建仓时勾了 README），推送会被拒绝。" Yellow
    Say "    那种情况下请先在网页上删除该仓库，或告诉我，我来处理合并。" Yellow

    # 已有提交时先 fetch，避免非快进被拒
    & git remote remove origin 2>$null | Out-Null
    & git remote add origin "https://github.com/$User/$Repo.git"
    Say "  拉取远端信息 ..."
    & git fetch origin 2>&1 | Out-Null
    $remoteHasCommits = $false
    try {
        $n = (& git rev-list --count origin/HEAD 2>$null)
        if ($n -and [int]$n -gt 0) { $remoteHasCommits = $true }
    } catch { }
    if ($remoteHasCommits) {
        Say "  ! 远端已有提交，改用 pull --rebase 后再推送" Yellow
        & git pull --rebase origin main
    }
} else {
    Say "  仓库不存在，尝试用 GitHub API 创建 ..."
    Say "  （需要一次浏览器授权：下面会弹出 GitHub 登录窗口）"
    $gh = (Get-Command gh -ErrorAction SilentlyContinue)
    if ($gh) {
        & gh auth status 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) { & gh auth login --hostname github.com --web --git-protocol https }
        & gh repo create "$User/$Repo" --public --description $Desc --source=. --remote=origin
    } else {
        Say "  ! 本机没有 gh CLI，无法自动建仓。" Yellow
        Say "    请打开下面这个链接，填好仓库名后点最下方的「Create repository」：" Cyan
        Say "      https://github.com/new?name=$Repo&visibility=public" Cyan
        Say "    建仓时【不要】勾选 Add a README / .gitignore / license。"
        Say "    建好后回到这里按回车继续。"
        Read-Host "`n建好仓库后按回车继续"
        & git remote remove origin 2>$null | Out-Null
        & git remote add origin "https://github.com/$User/$Repo.git"
    }
}

Say "`n=== 4/4 推送 ===" Cyan
Say "  接下来会弹出浏览器，请登录 GitHub 并点授权（只需一次）。"
& git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Say "`n✓ 推送成功！" Green
    Say "  仓库地址：https://github.com/$User/$Repo" Green
} else {
    Say "`n✗ 推送失败，请把上面的报错发给我。" Red
}
Read-Host "`n按回车退出"
