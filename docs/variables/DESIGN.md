# what must the derivation map explain?

start from any symbol, equation, code name, constant, or function. show where it is defined and follow its inputs to the foundations.

- retain every record and dependency from the generated variable reference.
- index source functions, their comments, signatures, locations, and direct calls across `mrm` without importing or executing the simulation.
- add verified mathematical definitions for intermediate quantities that do not have storage records, including the TCMT backscatter rate.
- keep mathematical definitions, preset derivation paths, storage mappings, and function calls distinct. an identity can have several equivalent forms; it must not create a false calculation cycle.
- start with one selected quantity and all of its direct relationships. place mathematical inputs, the selected equation, and mathematical dependents in separate columns. stack them on narrow screens.
- separate model conditions and implementation references from algebraic operands. use document flow instead of crossing connector lines.
- keep the entire index searchable and browsable without result or neighbour caps. a direct-relationship view is a stated semantic scope, not an arbitrary item limit.
- retain the complete ancestry as a foundation-first reading order. keep the large global graph as an optional overview, loaded only on request.
- separate node boxes and route orthogonal connections through reserved gutters. validate every node and edge, including source context and feedback references.
- distinguish model assumptions, approximations, numerical methods, equation operands, and static code relationships. show the configured fidelity switches and provenance.
- offer a foundation-first reading order as well as backward tracing. show all reachable inputs, with no fixed depth cutoff.
- search mathematical notation and code aliases together. permit a physics-only filter, but do not remove programming records from the index.
- show file and line references with source excerpts. state when an input is a preset anchor, a constant, a caller argument, or a boundary of the recorded metadata.
- validate the backscatter chain numerically and check graph traversal, source locations, search aliases, and LaTeX rendering.

the source index is generated with `python3 tools/gen_derivation_sources.py`. `derivations.js` holds reviewed equation definitions and code bindings. the existing generated variable data and simulation calculations remain unchanged.

`model-context.js` records reviewed physical assumptions and numerical choices. it distinguishes exact ring scattering, its zero-backscatter Airy case, TCMT, settled optics, thermal lags, rigid resonance shifts, junction models, data averaging, the forward bus approximation, and implicit self-heating. the indexed reference uses exact optics and disables backscatter. nonzero TCMT examples state their changed switch conditions.

`relationships.js` groups every incident edge by its actual meaning and renders the default focused view. the exploration path lets readers return to earlier quantities. search opens matching results; the full index can also be opened without a query.

`graph-layout.js` and `graph-view.js` provide the optional global graph. the initial page does not render thousands of canvas nodes. no external layout service is required. source excerpts include complete preset blocks and function definitions.
