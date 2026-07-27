# Domain Docs

Layout: **single-context**.

- `CONTEXT.md` at repo root — domain model, terminology, ubiquitous language for this project.
- `docs/adr/` at repo root — Architecture Decision Records.

Consumer rules: skills that need domain context (`to-spec`, `domain-modeling`, `codebase-design`) read `CONTEXT.md` and `docs/adr/*.md` directly. ADRs are edited only via the `domain-modeling` skill, not written ad hoc.
