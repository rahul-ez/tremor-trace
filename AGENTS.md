## Context Files

Before writing or modifying any code, read these files in this exact order:

1. `context/project-overview.md`
2. `context/architecture.md`
3. `context/code-standards.md`
4. `context/library-docs.md`
5. `context/build-plan.md`
6. `context/progress-tracker.md`

Do not begin implementation until all applicable context files have been read.

## Non-Negotiable Rules

## Non-Negotiable Rules

* Read all context files in the specified order before writing code.
* Follow `architecture.md` and `code-standards.md` exactly.
* Check installed skills before using any third-party library, then read `library-docs.md`.
* Do not introduce unapproved dependencies, technologies, or architectural patterns.
* Follow `build-plan.md` and implement one feature at a time.
* Verify every feature before marking it complete.
* Update `progress-tracker.md` after every completed feature.
* Never use hardcoded colors or raw framework-specific color classes.
* If the same problem persists after one corrective attempt, stop and run `/recover`.


## Skills


* `/architect` — before any complex feature.
* `/imprint` — after any new UI component.
* `/review` — before a demo or when something feels wrong.
* `/recover` — when the same problem persists after one corrective attempt.
* `/remember save` — when a feature spans multiple sessions.
* `/remember restore` — when returning to a multi-session feature.

## Implementation Order

* Follow `context/build-plan.md` strictly.
* Implement one feature at a time.
* Verify a feature before marking it complete.
* Keep `context/progress-tracker.md` accurate at all times.
* Preserve the architecture boundaries defined in `context/architecture.md`.
