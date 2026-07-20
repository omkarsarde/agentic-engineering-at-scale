# Editorial contract

This file is binding for every chapter in this book.

## Reader and outcome

Assume neural-network fundamentals, backpropagation, basic Python, and beginner PyTorch. Define every LLM, retrieval, agent, infrastructure, evaluation, and security term in plain language when it first becomes necessary. The goal is senior engineering judgment earned through understandable mechanisms, not jargon density.

## Two routes only

- Route A reads Chapters 1–32.
- Route B reads Chapter 1 and then Chapters 12–32.

Route-B dependencies on Chapters 2–11 must appear in the audited backfill boxes. Chapters 23 and 25 carry self-contained primers. No additional reading paths or hidden prerequisites are allowed.

## D2L-style teaching contract

Teach concepts just in time to accomplish a concrete end. A chapter should contain six to ten teaching sections and normally 4,000–7,000 words of explanatory prose. A section should introduce one primary idea. Dense surveys belong in a collapsible note, a dated Landscape box, or Appendix C.

For a new conceptual mechanism, use this rhythm:

1. Concrete motivation or failure.
2. One-sentence capability promise.
3. Plain-language intuition.
4. Load-bearing mathematics with every symbol defined.
5. Small runnable implementation or inspection.
6. Verification, measurement, or deliberate break.
7. Production implication and honest residual limitation.
8. Two-to-four-sentence synthesis.

For a systems mechanism, use:

1. User-visible invariant and concrete incident.
2. Smallest enforceable mechanism.
3. Deliberate fault, race, replay, overload, or hostile input.
4. Assertion, SLI, trace, or counter that exposes it.
5. Mapping to the production stack.
6. Residual application responsibility.

Prose is the teaching spine. Tables summarize after explanation; they do not replace explanation. Bullets are for genuine enumerations, not paragraph fragments. Avoid front-loading a glossary whose terms will not be used for many pages.

## Chapter apparatus

Every numbered chapter uses this order:

1. Exact H1 title and durability tag.
2. `What you need going in` block.
3. Route-B backfill or mandatory primer when audited.
4. `Contents` linking every H2.
5. Opening problem, chapter promise, and `What you will build` callout.
6. Six to ten teaching sections.
7. Exactly one integrated `Build`.
8. `What endures, what changes`.
9. Five to eight escalating `Exercises`.

Quarto supplies previous/next navigation. The closing synthesis may point to the next chapter in one sentence; never make the reader choose among a maze of paths.

## Progressive reference artifacts

Each chapter names one integrated build and uses one reference artifact per genuinely distinct mechanism. Do not create v0/v1/v2 copies of the same program. Introduce the smallest useful artifact once, then show deltas of no more than 40 lines. Each delta names its file and location and tags changed regions.

After a code-bearing section, use an artifact checkpoint:

| Artifact state | New code | Invariant now verified |
|---|---:|---|
| `agent_loop.py` after the gate | 24 lines | denied actions cannot reach a handler |

The complete artifact should normally remain below 200 lines. Print it at most once. Public entry points receive useful docstrings; helpers receive one-line documentation only when their purpose is not evident. Code must be runnable, free of pseudocode gaps, deterministic by default, and explicit about optional network/GPU adapters.

A comparison build may share one core with thin strategies or adapters. It may not independently reimplement the core behavior.

## Visual contract

Use two to five purposeful figures per chapter. Systems chapters require at least three diagrams that answer different questions. Follow `VISUAL-SYSTEM.md`. Every figure has a caption, adjacent explanation, a text conclusion, and semantics that survive grayscale. Numeric figures are generated from code. Diagram and SVG source is version-controlled.

## Durable spine and dated landscape

The spine teaches mechanisms, equations, interfaces, invariants, and decision criteria. Names, versions, prices, benchmark leaderboards, hardware SKUs, exact legal dates, and provider-specific parameters live in a block or section titled `Landscape 2026 (dated)`.

The book uses two deliberately different controls for moving facts:

1. A chapter-local `.landscape-2026` callout owns a teaching-local snapshot. It must state an ISO `YYYY-MM-DD` verification date, link the primary sources that support its claims, and end with an explicit **Verify live** instruction. Deleting it must leave a coherent chapter. These local snapshots do not each require duplicate rows in Appendix C.
2. Appendix C's machine-readable registry is a curated control plane for a smaller set of cross-chapter, deployment-sensitive, high-stakes, or operational pins. A registry row is justified by a maintenance or migration decision, not merely because a dated fact appears in prose.

For a registry row, `owner_chapter` is the repository-relative `.qmd` file responsible for maintaining or consuming that row. The linter requires the owner file to contain the exact primary-source URL or the row's declared, row-specific visible/HTML anchor. Registry-only records are owned by Appendix C and anchored where the appendix interprets them. This makes ownership auditable without forcing every chapter callout to mirror its contents into the registry.

An annual audit may still revise the spine when the mechanism itself changes.

## Voice

Write directly, concretely, and without ceremony. Intuition precedes rigor. Do not use “obviously,” “clearly,” “trivially,” “of course,” gatekeeping language, or interview stakes as motivation. Design-defense prompts belong only in exercises and Appendix B.

## Ownership boundaries

- KV memory arithmetic: Chapter 3.
- Behavior specifications: Chapter 7.
- Hallucination, calibration, and abstention: Chapter 9.
- Model customization and merging: Chapter 11.
- Compaction and caller-side context engineering: Chapter 13.
- Core RAG and grounding metrics: Chapter 14.
- Agentic retrieval control: Chapter 15.
- Raw agent specification and loop: Chapter 16.
- Skills and approval mechanics: Chapter 17.
- Memory content policy and deletion propagation: Chapter 18.
- MCP security and framework checkpointers: Chapter 19.
- Statistical evaluation and judge calibration: Chapter 22.
- Protocol-neutral action security: Chapter 24.
- Cost, durable effects, and delivery: Chapter 26.
- Observability, SLOs, incidents, HITL queues, and privacy operations: Chapter 27.
- Product, people, ownership, and AIMS: Chapter 28.
- Media provenance mechanics: Chapter 30.

Elsewhere, recap in one paragraph or point to the canonical section.
