# 第一次 GitHub 提交操作指南

## 1. 在 GitHub 创建空仓库

登录 GitHub，点击 **New repository**：

- Repository name：建议 `pi05-libero-safety-supervisor`
- Description：`Layered execution supervision, rollback and replanning for pi0.5 on LIBERO`
- Public / Private：套磁展示可选 Public；若尚未确认上游代码许可，先选 Private 更稳妥。
- 不要勾选自动创建 README、`.gitignore` 或 License，因为本地已经准备好这些内容。

创建后复制仓库 HTTPS 地址，例如：

```text
https://github.com/601-hi/pi05-libero-supervisor.git
```

## 2. 在本项目 PowerShell 中检查待提交内容

```powershell
Set-Location -LiteralPath 'C:\Users\60170\Documents\Codex\2026-08-30\pi05-libero-2026-08-30-m'
git status --short
git diff --cached --stat
```

确认输出中不存在 `.ssh_research`、checkpoint、原始 `.jsonl`、`work/` 或密码文件。

## 3. 设置提交身份

只需首次设置：

```powershell
git config user.name "你的GitHub显示名"
git config user.email "你的GitHub邮箱或GitHub noreply邮箱"
```

若希望全电脑通用，在 `config` 后加 `--global`。

## 4. 创建第一次提交

本目录已经初始化并暂存过大部分候选文件。由于 Codex 沙箱不能更新 `.git/index`，最后一次 `.gitignore` 更新及三个收尾文档需要由你的普通 PowerShell 完成。先从旧索引移除一个已排除的压缩包，再重新暂存：

```powershell
git rm --cached --ignore-unmatch -- "outputs/goal_progress_review_seed36_20260920.tgz"
git add -A
git status --short
git commit -m "feat: freeze first successful supervised rollback and replanning pipeline"
```

上述 `git rm --cached` 只从提交索引移除文件，不删除本地文件。

## 5. 连接并推送 GitHub

将下面地址换成你新建仓库的真实地址：

```powershell
git remote add origin https://github.com/601-hi/pi05-libero-supervisor.git
git branch -M main
git push -u origin main
```

GitHub 若要求认证，请在浏览器登录授权，或使用 Personal Access Token；不要把密码或 Token 写进仓库文件。

## 6. 推送后检查

在 GitHub 网页确认：

- 首页正确显示 `README.md`；
- `release/pi05-supervisor-v0.1-20261004/` 中有源码、三个小快照和成功视频；
- `.ssh_research/` 不存在；
- 没有模型 checkpoint、大型轨迹或服务器密码；
- 中文文档显示正常，无乱码。

随后可创建冻结标签：

```powershell
git tag -a v0.1-main-chain -m "First video-confirmed end-to-end recovery success"
git push origin v0.1-main-chain
```

