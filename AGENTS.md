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
  - 提交并推送 cl.exe 自动定位修复与 README 常见报错补充。
  - Success criteria:
    - [ ] 相关改动已提交。
    - [ ] 相关改动已推送到 origin/main。

- Constraints/Assumptions:
  - 不提交未跟踪文件（用户要求）。

- Key decisions:
  - 使用 `vswhere` 自动定位 VS 安装并注入 cl.exe 到 PATH。 — rationale: 兼容 Community/BuildTools。
  - README 增加 `opencv.cuda: MISSING` 说明。 — rationale: 明确 pip OpenCV CPU-only。

- State:
  - Done:
    - 已实现 cl.exe 自动定位（tools/web_server.py, tools/video_to_sbs.py, tools/precompile.py）。
    - 已更新 README.md 与 README.zh-CN.md（opencv.cuda 说明）。
  - Now:
    - 准备提交并推送改动。
  - Next:
    - 提交改动。
    - 推送到 origin/main。

- Open questions (UNCONFIRMED if needed):
  - [UNCONFIRMED] 提交信息是否有偏好？

- Working set (files/ids/commands):
  - Files:
    - tools/web_server.py
    - tools/video_to_sbs.py
    - tools/precompile.py
    - README.md
    - README.zh-CN.md
  - Commands:
    - git status -sb
    - git add -u
    - git commit -m "fix: auto-detect cl.exe with vswhere"
    - git push origin main
  - Artifacts/Refs:
    - origin: https://github.com/z4djjskk/quest-sbs-3d
    - PR: https://github.com/apple/ml-sharp/pull/64
## EXECUTION PLAN (PLANS) 〞 multi-hour execution scaffold (edit this section)

### Goal
- 提交并推送 cl.exe 自动定位修复与 README 常见报错补充。

### Acceptance Criteria
- [ ] 改动已提交。
- [ ] 改动已推送到 origin/main。

### Plan (3-7 steps max)
- [ ] Stage 相关文件。 — pending
- [ ] 提交改动。 — pending
- [ ] 推送到 origin/main。 — pending

### Verification commands (copy-paste)
    git status -sb

### Progress
- Pending:
  - Stage 相关文件。
  - 提交改动。
  - 推送到 origin/main。
- In Progress:
  - <none>
- Completed:
  - <none>

### Risks / Tradeoffs
- Risk: 工作区有未跟踪文件。  Mitigation: 使用 `git add -u` 仅提交已跟踪文件。
