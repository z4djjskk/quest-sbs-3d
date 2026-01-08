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
  - 提交并推送 README 中的错误反馈模板（中英版）。
  - Success criteria:
    - [x] README 改动已提交。
    - [x] README 改动已推送到 origin/main。

- Constraints/Assumptions:
  - 不提交未跟踪文件（用户要求）。

- Key decisions:
  - 仅提交 README.md 与 README.zh-CN.md。 — rationale: 用户仅要求文档模板。

- State:
  - Done:
    - 已提交并推送反馈模板（commit ffef9d8）。
  - Now:
    - 回复用户并给出提交信息。
  - Next:
    - 等待用户反馈。

- Open questions (UNCONFIRMED if needed):
  - [UNCONFIRMED] 是否需要将模板扩展到 issue 模板文件？

- Working set (files/ids/commands):
  - Files:
    - README.md
    - README.zh-CN.md
  - Commands:
    - git push origin main
  - Artifacts/Refs:
    - origin: https://github.com/z4djjskk/quest-sbs-3d
    - commit: ffef9d8
    - PR: https://github.com/apple/ml-sharp/pull/64
## EXECUTION PLAN (PLANS) 〞 multi-hour execution scaffold (edit this section)

### Goal
- 提交并推送 README 中的错误反馈模板（中英版）。

### Acceptance Criteria
- [x] README 改动已提交。
- [x] README 改动已推送到 origin/main。

### Plan (3-7 steps max)
- [x] Stage README 改动。 — completed
- [x] 提交改动。 — completed
- [x] 推送到 origin/main。 — completed

### Verification commands (copy-paste)
    git status -sb

### Progress
- Pending:
  - <none>
- In Progress:
  - <none>
- Completed:
  - Stage README 改动。
  - 提交改动。
  - 推送到 origin/main。

### Risks / Tradeoffs
- Risk: 工作区存在未跟踪文件。  Mitigation: 仅 add README 文件。
