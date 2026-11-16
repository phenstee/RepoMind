# Design

RepoMind's interface is a **lamplit workbench**: warm paper (light) and graphite (dark)
neutrals, one ember accent, and type that reads like a well-set engineering document. It is an
Operate surface: the tool should disappear into the task, so expression lives in precise
details, not decoration. Tokens live at the top of `frontend/app/globals.css`.

## Color

- **Neutrals** are warm, never blue-grey: `--bg` paper/graphite, `--surface` for the working
  card, `--sidebar` for the quieter second layer, `--surface-2/3` for insets and chips.
- **Accent (`--accent`, ember orange)** marks only action, current selection, and live state:
  primary buttons, the active tab underline, the selected repository avatar, the live
  timeline dot, focus rings, caret, and text selection. Never decoration.
- **Semantic colors** are separated from the accent on purpose: success green, danger
  crimson-rose (not red-orange, so it never reads as the accent), warning amber.
- **Run-type tones** (`--tone-ask/investigate/code/index`) appear only on run-history icons.
- Contrast targets WCAG AA in both themes; `--muted` text clears 4.5:1 on every surface.

## Type

- Interface: **Hanken Grotesk** (variable, self-hosted via `@fontsource-variable`).
- Code, paths, line ranges, timestamps, and counts: **JetBrains Mono** (variable). Mono is for
  data only, never as a "technical" costume for prose or labels.
- Base 15px; fixed rem scale with tight steps (h1 1.85rem, h2 1.06rem, h3 1rem). Answer prose
  is 0.97rem / 1.7 at a 70ch measure. Dynamic numbers use tabular figures.

## Layout

- Desktop (≥1280px): sidebar (repositories, files) | workspace | sticky live-progress rail.
  1025–1279px: the rail drops under the workspace. ≤1024px: one column, sidebar panels side by
  side; ≤760px: everything stacks.
- The workspace header is not a card: the repository title, metadata, job strip, and mode tabs
  sit directly on the page. The mode panel is the single raised card.
- Answers are document sections separated by a rule, not cards nested inside a card.
- Elevation comes from layered shadows plus a 1px ring; borders are kept for structure.

## Motion

- One authored moment: a new answer resolves out of a 4px blur into place (420ms, exponential
  ease-out), and its sources settle in after it with a short stagger.
- Everything else is state feedback at 140–220ms: tab underline slide, hover/press
  (`scale(0.96)`), icon swap on copy, the thinking-text shimmer while a job runs.
- `prefers-reduced-motion` removes movement and keeps state feedback (spinners, color changes).
- No animation library: CSS covers every case here, and a timeline library would only invite
  choreography an Operate surface should not have.
