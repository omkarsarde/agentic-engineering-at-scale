# Visual system

The book uses the smallest visual grammar that makes a relationship easier to understand than prose.

## Selection guide

| Question | Default medium | Why |
|---|---|---|
| What calls what, and in which order? | Mermaid sequence diagram | Sequence, retries, and cancellation are explicit. |
| Which states and transitions are legal? | Mermaid state diagram | Invariants and forbidden transitions stay reviewable as text. |
| Where does trust or authority change? | Mermaid flowchart with labeled subgraphs | Boundaries and effect paths remain adjacent to the security prose. |
| How do many dependencies or lineage edges connect? | Graphviz/DOT | Automatic layout handles dense directed graphs better than Mermaid. |
| How does a quantity change? | Executable matplotlib/Altair plot | The claim and its generating data cannot silently diverge. |
| Does the reader benefit from exploration? | Observable/Plotly HTML with static SVG fallback | Sliders help only when changing a parameter teaches the mechanism. |
| Is the idea spatial, tensor-geometric, or visually bespoke? | Editable SVG from draw.io/Excalidraw/Figma | Deliberate composition beats automatic graph layout. |
| Are the values exact mappings or choices? | Table | Readers can compare precisely without decoding arrows. |

Quarto natively renders Mermaid and Graphviz. Custom sources live under `assets/diagrams/source/`; exported SVG files live under `assets/diagrams/rendered/`. Screenshots are never the sole source.

## Semantic language

- Model or probabilistic component: rounded rectangle.
- Code-owned policy or deterministic control: rectangle.
- Decision: diamond.
- Durable state: cylinder or double-bordered node.
- External/untrusted input: dashed amber boundary.
- Irreversible or externally visible effect: red-outlined node.
- Measurement or verifier: green-outlined node.

Every arrow receives a label when its payload, authority, or timing matters. Dashed arrows mean asynchronous or advisory flow; solid arrows mean the normal synchronous path. A legend is included when a diagram introduces more than two of these conventions.

## Figure contract

Each figure must answer one question stated by its caption. The paragraph before it creates the need; the paragraph after it states the invariant or decision to retain. Do not place several pages between a figure and its interpretation.

Every visual supplies:

- a stable cross-reference label;
- a useful caption;
- alt text or an adjacent textual equivalent;
- source under version control;
- a print-safe fallback;
- no information encoded by color alone.

Use two to five figures per chapter. A systems chapter normally includes one structural view, one sequence or state view, and one failure/measurement view. Three cosmetic variants of the same architecture do not satisfy the requirement.

## Interactive figures

Interactive figures are optional and rare. Use one only when the reader learns by changing a parameter—context length, cache size, retry probability, concurrency, or confidence threshold. The page must contain a static plot or table conveying the default case for PDF, EPUB, reduced-motion settings, and JavaScript-disabled readers.

## Quantitative honesty

Plot labels include units. Captions identify whether data is synthetic, measured locally, reported by a primary source, or illustrative. Hardware/model measurements record configuration and seed. Axes do not silently truncate, and uncertainty is shown when the claim depends on stochastic trials.

