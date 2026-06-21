---
name: always-push-to-remote
description: After committing, always push to the remote hub (origin/master on GitHub)
metadata:
  type: feedback
---

每次 `git commit` 之后，紧接着执行 `git push` 推送到 remote origin。

**Why:** 用户明确要求"以后上库都提交到 remote hub"，确保所有本地提交同步到 GitHub 远程仓库。

**How to apply:** 每次 commit 完成后自动执行 `git push`，无需额外询问。
