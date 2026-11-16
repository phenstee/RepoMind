# Product

<!-- impeccable:product-schema 1 -->

> Written by `/impeccable init` on 2026-11-16 from the README, `docs/`, and the frontend code.
> The owner was not interviewed for this first version: every line below is inferred from the
> repository and should be corrected where it is wrong.

## Platform

web

## Users

Software engineers who need to understand, and sometimes safely change, a repository they did
not write. They run RepoMind locally against a checkout in their workspace and sit with it for
long, focused sessions at a desktop: asking questions, reading cited source, kicking off
investigations, and reviewing gated code changes. (Inferred.)

## Product Purpose

RepoMind answers grounded questions about a codebase, investigates it with a read-only
tool-using agent, and makes small, verified edits under deterministic test/lint/Git gates.
Success is an answer the engineer can trust because it shows exactly where in the source it
came from, and a change they can accept because real checks, not model prose, approved it.

## Positioning

Trust is the product: hybrid retrieval with line-level citations, indexed hits that must be
re-read from current source before they count, and a completion gate the model cannot override.
The interface should make that evidence and those gates visible rather than decorative.

## Operating Context

- Local web app (Next.js) talking to a local FastAPI backend at `127.0.0.1:8000`.
- Long operations run as durable jobs with live Server-Sent Events progress, which the UI
  follows, can stop viewing, resume, and cancel.
- Persisted run traces are inspected after the fact in Run history.

## Capabilities and Constraints

- Modes: Ask (RAG with selectable retrieval strategy), Investigate (read-only agent,
  filesystem or indexed navigation), Code (planned, verified, reviewed edits).
- Repositories are registered by workspace-relative path and indexed before Ask works.
- Light and dark themes; the explicit choice persists, otherwise the OS preference wins.
- No arbitrary shell access is ever exposed; copy must not imply otherwise.

## Brand Commitments

- Name: RepoMind. Existing graph-node brand mark (three connected nodes).
- Voice: plain, precise, engineering-literate; states what happened and what to do next.

## Evidence on Hand

- `repo-eval-v5` retrieval benchmark (`docs/evaluation.md`). No customers, testimonials, or
  usage metrics exist; none may be invented.

## Product Principles

1. Evidence before prose: citations, observed locations, and check results are first-class.
2. State is never ambiguous: running, disconnected, cancelled, and failed always look different.
3. The tool disappears into the task: familiar controls, dense where it helps, calm by default.

## Accessibility & Inclusion

WCAG 2.2 AA contrast in both themes, full keyboard operation, visible focus, and
`prefers-reduced-motion` honored (movement removed, state feedback kept).
