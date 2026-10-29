# preferences for learning, code, and visuals

this guide records the user's stated preferences from the conversation. the final section records the appearance of the supplied Excalidraw revision. those visual observations are a reference for similar diagrams, not universal requirements for every interface.

## what is the learning goal?

- understand the entire file, in code order, rather than a few selected functions.
- learn how physical phenomena become mathematical models and how those models become simulations.
- assess whether the implementation and its assumptions represent the real device accurately.
- understand missing effects, possible inaccuracies, and the reasons for choosing a simpler model.

## what knowledge can be assumed?

- use chapters 1–5 of `~/interactive-textbooks/lightmatter-capstone` as the photonics baseline.
- the user knows ODEs well. do not assume familiarity with photonics concepts beyond chapter 5.
- introduce unfamiliar terminology before relying on it. distinguish overloaded terms, such as transverse waveguide mode and ring resonance order.
- build on concepts already understood. do not repeat port directions, field conventions, or earlier derivations after the user confirms them.
- distinguish a topic that was explained from a topic the user confirmed understanding.
- when the user gets lost, identify the specific missing connection before introducing more concepts.

## how should an explanation progress?

1. start with the physical phenomenon and the question the calculation must answer.
2. give a simple physical picture or diagram that establishes the geometry, directions, and interactions.
3. explain what the mathematical model needs to represent and what assumptions it makes.
4. develop the equations from that picture. explain why someone would choose those equations.
5. connect the equations to the actual code, with links and useful snippets.
6. finish with limitations and a high-level account of verification.

- do not begin with implementation details and make the user infer the physical purpose.
- do not present an unexplained formula and then merely define its symbols.
- explain the equation's overall job before explaining its individual factors.
- give physical intuition without removing mathematical depth or hiding complexity.
- move efficiently by grouping repeated operations under a rule already established, such as the product rule.
- keep each chunk coherent. do not jump among unrelated functions or introduce several new frameworks at once.
- connect explanations to the original transmission curve, ring geometry, and previously understood models when relevant.

## how should equations and notation be presented?

- render mathematical expressions, physics symbols, and units with LaTeX in explanations and mathematical visuals.
- include numbers in math rendering when that improves consistency or clarity.
- define symbols, units, inputs, and outputs close to their first use.
- explain the intuition and origin of each new formula, including why it should work physically.
- distinguish physical laws, modelling choices, phase conventions, and algebraic rearrangements.
- make clear what varies and what stays fixed when taking a derivative.
- distinguish phase derivatives from time derivatives and steady optical solutions from time evolution.
- identify whether an amplitude is normalized to power or energy. do not silently interchange field, power, and stored energy.
- distinguish light-travel direction from the coordinate direction used to trace the geometry.
- use small numerical examples when they make magnitudes, phases, or limiting cases easier to picture.
- identify tunable settings as hyperparameters. in particular, the controller's 70% monitor target is a hyperparameter, not a physical constant.

## how should code be explained and commented?

- consult and reference the actual code. do not give a generic explanation that leaves the implementation disconnected.
- explain every input, output, implemented formula, and non-obvious operation in the portion being covered.
- define mysterious names and map them to the mathematical quantities they represent.
- explain signs, conjugates, inverse factors, matrix order, indexing, and temporary variables when they affect understanding.
- keep code comments short, minimal, and useful. explain purpose and reasoning rather than narrating obvious syntax.
- preserve the user's inline notes and diagrams. clarify misleading annotations without discarding their intent.
- prefer lowercase prose and dashes instead of colons or semicolons in explanatory text.
- follow ASD-STE100 principles through short sentences, active voice, one idea per sentence, and consistent terms. do not claim formal controlled-vocabulary compliance.
- preserve required capitalization in physical symbols, units, APIs, and code syntax.
- keep long derivations in documentation or the explanation, rather than crowding the source.
- preserve meaning and numerical behaviour during readability edits.

`/Users/kenny/mrm-model/docs/CODE_STYLE.md` contains the more detailed repository conventions, including comment placement, parameter and unit annotations, and naming rules.

## how should accuracy and verification be discussed?

- explain the accuracy limitations after the physical model and implementation are understood.
- state what the model includes, what it omits, when the omissions matter, and what a more complete model would require.
- explain the relevant tradeoff, such as computation cost or missing physical parameters. distinguish documented reasons from inference.
- keep verification high-level. describe comparisons with fuller mathematical calculations, conservation checks, or measurements without walking through test internals.
- distinguish correct implementation of a model from evidence that the model matches a real device.
- do not present agreement between two mathematical descriptions as experimental validation.

## what should a physical diagram look like?

- keep it minimalistic, spacious, and directly labelled.
- show the essential geometry, components, ports, and propagation directions.
- keep physical layout diagrams separate from equation diagrams and calculation-flow boards.
- do not combine the ring, all equations, code notes, energy calculations, and accuracy discussion into one crowded board.
- use a simple ring with the two couplers and two equivalent scatterers for this model.
- show clockwise and counterclockwise directions clearly in the same ring.
- label incoming and outgoing port fields consistently with the code. retain the `s` and `o` naming convention.
- place labels near the relevant components without touching paths, arrows, or other text.
- avoid unnecessary legends, repeated labels, decorative boxes, and implementation-heavy titles.
- provide a usable Excalidraw file when requested.

## what should equation diagrams look like?

- use properly rendered LaTeX and mathematical symbols, rather than plaintext approximations of equations or matrices.
- keep the equation artwork purely mathematical, without explanatory prose inside it.
- provide separate, focused diagrams rather than one dense collection.
- use generous spacing and consistent, readable mathematical sizing.
- be candid about rendering limitations. do not substitute raw LaTeX or plaintext while claiming it is rendered.
- when Excalidraw contains rendered equation images, say so. preserve the LaTeX source so the equations can be changed.

## what styling applies to applications and plots?

- use the requested Vercel design reference at `https://vercel.com/design.md` for the relevant simulation websites.
- use SF Mono for application text where requested. use a compatible mathematical style where feasible, while preserving legibility.
- size rendered math to match surrounding text. it must not appear noticeably smaller.
- keep spacing, cards, boxes, and visualizations spacious and appropriately sized.
- keep y-axis labels horizontal rather than rotated 90 degrees.
- give every graph a short, clear blurb stating the question it answers and what it shows.
- show all eight rings together for relevant per-ring quantities, with enough separation to avoid clutter and overlap.
- make variable maps visually clear, consistent with the surrounding styling, and easy to follow.
- organize explanatory documentation around concrete questions, such as where the operating point sits.

## what working approach was requested?

- for the readability pass, work sequentially, one file at a time, without subagents. this superseded the earlier request for an agent per file because of rate limiting.
- keep a small, clear Markdown design or progress document for long work so the requirements and completed steps are not lost.
- preserve the user's concurrent edits.
- keep readability changes distinct from changes to the physical model or numerical behaviour.

## what does the latest edited Excalidraw establish?

these are observations from the supplied revision, not additional verbal instructions.

- use the plain title “Ring model” for the physical schematic.
- the diagram is monochrome. all 39 elements use stroke color `#1e1e1e` and transparent fills.
- the drawing uses sketch-style strokes. the supplied settings are stroke width `4` and roughness `1`.
- all text uses the same Excalidraw font setting, `fontFamily = 5`, at canvas font size `36`.
- the larger labels and hand-drawn appearance differ from the earlier monospaced, color-coded draft. follow this revision for similar ring schematics instead of forcing the application font onto them.
- retain circles for couplers, diamonds for scatterers, a circular ring, and vertical bus waveguides.
- retain solid optical paths and dashed coupling-location marks.
- distinguish directions through arrows and labels rather than depending on color.
- keep the simple incoming/outgoing legend and the four named ports.
- treat font size and stroke settings as the reference at this canvas scale, not mandatory pixel values for every export.

reference revision — `/Users/kenny/.codex/attachments/0b2e9bba-27e1-4f0b-ae98-f125ea644ca6/Pasted text.txt`.
