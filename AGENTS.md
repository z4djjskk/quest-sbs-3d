# AGENTS.md — Single-file Continuity + Plan (compaction-safe)

## Working Contract (do not edit this section)
- This file is the canonical workspace contract AND the only place to maintain continuity + plan.
- Do not rely on earlier chat text unless it is reflected in the Ledger section of this file.
- Never guess. Mark uncertainty as `UNCONFIRMED`.

### Mandatory workflow (every assistant turn)
1) Read this file top-to-bottom.
2) Update the **Ledger** section first to reflect latest: goal / constraints-assumptions / key decisions / state (Done/Now/Next) / important tool outcomes.
3) If the task is multi-hour or multi-stage, update the **Execution Plan** section (3–7 steps).
4) Then proceed with the work.
5) When material changes happen, update Ledger again.

### Reply format
- Begin each reply with a brief “Ledger Snapshot”:
  - Goal
  - Now
  - Next
  - Open Questions
- Print the full Ledger only when it materially changes or when the user asks.

### Ledger vs Plan boundary
- Ledger = long-running continuity across compaction (“what/why/current state”), not a step-by-step transcript.
- Plan = short-term execution scaffold (3–7 steps max), with pending/in-progress/completed.
- Keep them consistent: when plan/state changes materially, update Ledger at the intent/progress level.

---

## CONTINUITY LEDGER (edit this section only; keep headings)

- Goal (incl. success criteria):
  - 上传到 GitHub，并写一份十分详细的使用说明。
  - Success criteria:
    - [x] 生成详细使用说明 Markdown（中文）。
    - [x] 文档已推送到 GitHub（USAGE.zh-CN.md + README.zh-CN.md + agent.md）。
    - [ ] 完整代码同步到 GitHub。

- Constraints/Assumptions:
  - 本地 Git 无法写入 `.git`；代码将通过 GitHub API 推送。

- Key decisions:
  - 代码同步走 GitHub API（覆盖式更新 main）。

- State:
  - Done:
    - 已完成提交与推送（main 已与 origin/main 同步）。
    - 已生成并推送文档更新。
    - 已设置用户级 `GITHUB_TOKEN`。
  - Now:
    - 推送完整代码到 GitHub。
  - Next:
    - 核验远端内容与提交。

- Open questions (UNCONFIRMED if needed):
  - <none>

- Working set (files/ids/commands):
  - Files:
    - push_payload.json
  - Commands:
    - <none>
  - Artifacts/Refs:
    - origin: https://github.com/z4djjskk/quest-sbs-3d

---

## EXECUTION PLAN (PLANS) 〞 multi-hour execution scaffold (edit this section)

### Goal
- 提交并推送当前代码到 GitHub。

### Acceptance Criteria
- [ ] 清理阻塞的 git lock。
- [ ] 完成 git add/commit。
- [ ] 推送到远端仓库成功。

### Plan (3-7 steps max)
- [x] 清理 .git/index.lock 并重试 git add。 — completed
- [x] 提交 AGENTS.md。 — completed
- [x] 推送到远端仓库。 — completed

### Verification commands (copy-paste)
    <none>

### Progress
- Pending:
  - <none>
- In Progress:
  - <none>
- Completed:
  - 清理 .git/index.lock 并重试 git add。
  - 提交 AGENTS.md。
  - 推送到远端仓库。

### Risks / Tradeoffs
- Risk: 锁文件删除不当。  Mitigation: 确认无 git 进程后再删除。
