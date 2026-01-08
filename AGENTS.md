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
  - 将本地项目更新到 https://github.com/z4djjskk/quest-sbs-3d，并同步到 https://github.com/apple/ml-sharp 的指定分支。
  - Success criteria:
    - [ ] 本地改动已提交并推送到 quest-sbs-3d。
    - [ ] 相关改动已推送到 ml-sharp 的目标分支（或已建立 PR）。
    - [ ] 用户确认推送结果与分支目标。

- Constraints/Assumptions:
  - 具备对两个仓库的 Git 推送权限与认证。
  - 不使用破坏性命令（如 reset/checkout --）。
  - 默认将当前工作区改动全部纳入提交（除非用户指定排除文件）。

- Key decisions:
  - 先检查现有分支与 remotes，再决定推送策略（直接推送或新建分支）。 — rationale: 避免覆盖错误分支。

- State:
  - Done:
    - 已检查 git 状态/分支/远程（main 跟踪 origin/main）。
  - Now:
    - 确认提交范围（是否包含未跟踪文件）与 ml-sharp 目标分支。
  - Next:
    - 整理提交并推送到 quest-sbs-3d。
    - 配置 ml-sharp 远程并推送到指定分支或创建 PR。

- Open questions (UNCONFIRMED if needed):
  - [UNCONFIRMED] ml-sharp 需要同步到哪个分支？是否需要 PR 方式？
  - [UNCONFIRMED] 未跟踪文件（AGENT_COORDINATION.md、codex_last_message.txt、push_payload.json）是否需要提交？

- Working set (files/ids/commands):
  - Files:
    - AGENTS.md
  - Commands:
    - git status -sb
    - git remote -v
    - git branch -vv
    - git log -1 --oneline
  - Artifacts/Refs:
    - origin: https://github.com/z4djjskk/quest-sbs-3d
    - target: https://github.com/apple/ml-sharp
---

## EXECUTION PLAN (PLANS) 〞 multi-hour execution scaffold (edit this section)

### Goal
- 将本地项目更新到 quest-sbs-3d 并同步到 ml-sharp 目标分支。

### Acceptance Criteria
- [ ] quest-sbs-3d 已包含本地最新提交。
- [ ] ml-sharp 目标分支已更新或 PR 已创建。
- [ ] 用户确认分支与推送结果。

### Plan (3-7 steps max)
- [ ] 核对 git 状态/分支/远程配置。 — pending
- [ ] 整理提交并推送到 quest-sbs-3d。 — pending
- [ ] 同步到 ml-sharp 目标分支或创建 PR。 — pending

### Verification commands (copy-paste)
    <none>

### Progress
- Pending:
  - 核对 git 状态/分支/远程配置。
  - 整理提交并推送到 quest-sbs-3d。
  - 同步到 ml-sharp 目标分支或创建 PR。
- In Progress:
  - <none>
- Completed:
  - <none>

### Risks / Tradeoffs
- Risk: 审查范围大且近期改动多。  Mitigation: 先覆盖核心路径与异常分支。
