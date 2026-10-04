# 首次提交收尾检查

## 已完成

- 初始化本地 Git 仓库，分支为 `main`；
- 冻结服务器实际成功版本，约 2.82 MB；
- 保存原始故障、确定失败第二轮起点、成功第二轮起点三个完整 MuJoCo 快照；
- 保存人工确认成功视频；
- 生成九个关键文件 SHA-256，全部复核一致；
- 添加 README、第三方说明、对比快照说明、结果 JSON 和手动提交指南；
- 添加研究范围、两阶段回退分工、作者署名要求及 MuJoCo oracle 真机替代说明；
- `.ssh_research/`、模型权重、原始 JSONL、大型工作目录和缓存已加入 `.gitignore`；
- 已暂存路径中没有 SSH 私钥、`authorized_keys`、常见 Token 或明文密码模式；
- 服务器测试基线为 `540 passed`，最终新增的快照接线已通过 Python 语法检查和真实 GPU 运行；
- 成功轨迹由视频人工确认，而非只依赖 LIBERO 布尔值。

## 用户提交前必须执行

Codex 沙箱对 `.git/index` 的后续写入被 Windows ACL 拒绝，因此最后三份新文档尚未暂存，一个13.19 MB的旧 `.tgz`仍留在旧索引中。请在普通 PowerShell 执行：

```powershell
Set-Location -LiteralPath 'C:\Users\60170\Documents\Codex\2026-08-30\pi05-libero-2026-08-30-m'
git rm --cached --ignore-unmatch -- "outputs/goal_progress_review_seed36_20260920.tgz"
git add -A
git status --short
```

确认 `.ssh_research/` 和 `.tgz` 不在输出中，再按 `GITHUB_SUBMISSION_GUIDE.md` 提交和推送。

公开前还必须在 `PROJECT_SCOPE_AND_AUTHORSHIP.md` 填写作者真实姓名、学校、邮箱和 GitHub 地址；不要带着占位符公开。

## 冻结版本

- 目录：`release/pi05-supervisor-v0.1-20261004/`
- 推荐提交信息：`feat: freeze first successful supervised rollback and replanning pipeline`
- 推荐标签：`v0.1-main-chain`

