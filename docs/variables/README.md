# what does the variable map explain?

the default view answers **where does this quantity come from?** search a symbol, equation, code name, or function, then trace its inputs back to constants, preset anchors, or caller arguments.

## how do i trace a derivation?

open **derivation map** and search everything. try `g`, `r_b`, `t_rt`, `tcmt_backscatter_rate`, or paste `g=\frac{r_b}{t_\mathrm{rt}}`.

- **relationships** starts with one quantity. its direct mathematical inputs, selected equation, and mathematical dependents occupy separate columns. click an input to learn how it is built; click a dependent to follow where it leads. the exploration path lets you return.
- **under which model conditions?** separates assumptions and numerical methods from algebraic operands. **where does this connect to the code?** contains all direct source, setting, storage, and call relationships.
- **complete derivation** places all recorded ancestors in a foundation-first reading order, without a depth cutoff. each step links to its inputs.
- **whole-index overview** opens the large pan-and-zoom graph only when requested. it retains the full-index and complete-selected-ancestry scopes.
- **where is this established?** opens source comments, file and line references, and the implementation. selecting another quantity keeps a return button.
- **math & physics** searches physical quantities and documented mathematical function arguments. programming records remain available under **everything** or **programming & simulation**.
- **include code mappings in the reading order** controls the optional physics-only reading view. programming relationships remain available in the focused view and complete index.

the focused view includes every direct relationship, without a neighbour cap. search covers the entire index and displays every match. open **browse all definitions and records** to browse without a query. no search, notation, stored-value, or ancestry list has an item cutoff. changing quantity changes the stated scope rather than dropping records based on a count.

green card borders identify mathematical quantities, amber identifies model conditions, and neutral identifies implementation references. the focused view uses headings and ordinary document flow, with no crossing lines or canvas navigation. cards stack on narrow screens. the optional global graph retains its dashed context links and routed connectors.

the index retains all 980 generated model records and adds source functions and module constants across `mrm`. documented input and output comments provide more mathematical quantities. supported scalar expressions render as equations; other implementations remain readable source code. the search also covers source bodies, so a local name can lead to the function that defines it.

the backscatter example connects the configured splitting and FSR to the half-splitting, round-trip coupling, ring geometry, group delay, and TCMT rate. it also shows the TCMT approximation and fidelity switches. the indexed g26 reference selects exact optics and disables backscatter, so its round-trip coefficient is zero. the nonzero numerical example explicitly enables backscatter and disables device variation. it does not describe the current playback state. stored reference values retain their stated units; convert them to SI before using SI equations.

mathematical inputs, preset provenance, function arguments, and function calls are different relationships. a function argument can remain a caller-supplied boundary; a static call link is not a proof of a physical derivation or an execution trace. uncertain and missing registry dependencies remain identified instead of being invented.

regenerate source locations, comments, and function relationships after source edits —

```sh
python3 tools/gen_derivation_sources.py
python3 tools/gen_derivation_sources.py --check
node --test tests/verification/test_derivation_explorer.mjs
node --test tests/verification/test_visual_variable_graph.mjs
```

`source-index.json` is generated without importing or executing the model. `derivations.js` contains the reviewed mathematical definitions and traversal rules. `relationships.js` renders the focused relationships; `explorer.js` connects search, navigation, reading order, and source inspection. [DESIGN.md](DESIGN.md) records the implementation requirements.

## how do i open it?

from the running MRM viewer, choose **Math & physics ↗**, or open [the integrated map](http://localhost:8000/variables/). the site route points to this folder, so there is one editable copy.

for a standalone preview, from the repository root —

```sh
python3 -m http.server 8010 --bind 127.0.0.1 --directory docs/variables
```

open [the variable map](http://127.0.0.1:8010/). the page needs a local server to read `data.json`. KaTeX 0.16.11 loads from the existing CDN and renders mathematical notation. a text fallback remains available if it cannot load.

## what is in physical overview?

- a physical feedback diagram with eight quantities and arrows routed through reserved gaps. the absorbed-power return path has its own lane.
- selecting a quantity opens its question, equation, symbol definitions, units, and assumptions. the symbols remain visible on every diagram card.
- a searchable notation reference covers the model's defined physical symbols. filter optics, thermal physics, electrical drive, or photodiodes. every matching row is shown.
- storage names and the original recorded context sit under **code mapping & recorded context**. identical symbols retain their separate reference-state and data-level records.

arrows in this diagram explain physical influence. they do not indicate time samples, signs of sensitivity, or measured values. each equation states its scope. the drop-port Airy expression, for example, describes one laser line and data level without backscatter. the thermal equation uses multiple physical lag states.

## what belongs in stored values?

all generated records remain searchable. this layer includes physical values, cached arrays, observer indices, numerical steps, ADC codes, controller interfaces, and statistics.

badges distinguish defined physical quantities, physical values stored under code names, and programming or simulation terms. a code name does not acquire a physics symbol merely to make the graph look mathematical.

open a record to see all of its immediate **computed from → selected record → used by** relationships. there is no neighbour cutoff. use **trace all inputs from the foundations** to open the derivation explorer, then **complete derivation** for the full ancestry.

original links such as `#ring_float.t_ring` open that record. `#physics/temperature` opens the physical temperature explanation. the two main view buttons preserve the current physical selection.

## where do the definitions come from?

`data.json`, `symbols.json`, and `docs/VARIABLES.md` are generated reference files. this interface leaves those files and the simulation unchanged.

`physics.js` contains the teaching diagram, explicit physical definitions, display-unit corrections, and classification rules. definitions refer to the current optics, phase, thermal, electrical, and sensing code. three physical quantities with missing registry notation receive explicitly defined presentation symbols — thermal lag rise, thermal observer temperature, and local tuning curvature.

display units use the stored units or explicit compound suffixes. current-noise density is A/√Hz, and the analytic depletion coefficient is m/V raised to the depletion exponent. these correct sparse display metadata without changing a simulation value.

`map.js`, `index.html`, and `style.css` implement the interface. math and text use the existing SF Mono style. mathematical text keeps the same visible scale as nearby prose. the diagram reflows into a readable quantity list on a narrow screen.

regenerate the underlying reference only after changing its source registry, preset, or relations —

```sh
.venv/bin/python tools/gen_variable_map.py
```
