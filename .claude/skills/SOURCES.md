# Vendored agent skills

Third-party skills checked in so every Claude Code session in this repo (cloud or local) loads them.
Each skill folder keeps its upstream files and license unchanged. Installed 2026-10-08.

| Skill folder(s) | Upstream | Commit | License | Installed with |
| --- | --- | --- | --- | --- |
| `impeccable/` + `../agents/impeccable-*.md` | [pbakaus/impeccable](https://github.com/pbakaus/impeccable) ([impeccable.style](https://impeccable.style)), release 4.5.1 | `9dad388a41944a0d2b8d1fb547c8556d2ecb49e7` | Apache-2.0 (`impeccable/LICENSE`, `impeccable/NOTICE.md`) | Copied from upstream's Claude Code project build (`.claude/skills/impeccable`, `.claude/agents`). `npx impeccable install` failed here: its signed-bundle download was blocked by the network proxy. |
| `gsap-core`, `gsap-frameworks`, `gsap-performance`, `gsap-plugins`, `gsap-react`, `gsap-scrolltrigger`, `gsap-timeline`, `gsap-utils` | [greensock/gsap-skills](https://github.com/greensock/gsap-skills) (official GreenSock) | `aed9cfd3277740755f6bfc1155c7aa645403b760` | MIT (`LICENSE` in each folder) | `npx skills add greensock/gsap-skills --agent claude-code` |
| `transitions-dev`, `transitions-polish` | [jakubantalik/transitions.dev](https://github.com/jakubantalik/transitions.dev) ([transitions.dev](https://transitions.dev)) | `859ef7a820aae7559fb74a034d8a517287eb6c96` | Transitions.dev License (`LICENSE.txt` in each folder): free to use in our product, not to republish as a competing library | `npx skills add jakubantalik/transitions.dev --agent claude-code` |
| `make-interfaces-feel-better` | [jakubkrehel/make-interfaces-feel-better](https://github.com/jakubkrehel/make-interfaces-feel-better) | `35545ea1512ad59fa463e6b1f95ca9c052981fe6` | MIT (`LICENSE`) | `npx skills add jakubkrehel/make-interfaces-feel-better --agent claude-code` |

`/skills-lock.json` is written by the `skills` CLI; `npx skills update` refreshes the three packs it tracks.
To refresh Impeccable, run `npx impeccable update` (or re-copy from upstream).

## Impeccable notes

- The skill's launcher (`impeccable/scripts/impeccable`) downloads its engine binary to `~/.impeccable/bin/` on first run; no binary is committed.
- Upstream's optional design-detector hooks (SessionStart, PostToolUse on Edit/Write, and a 30s Stop pass) are **not** enabled, because they would run on every Claude session in this repo, backend work included. To turn them on, run `npx impeccable install --providers=claude --scope=project`, which writes them to `.claude/settings.json`.
- Run `/impeccable init` once to create the `PRODUCT.md` / `DESIGN.md` context files it reads.
