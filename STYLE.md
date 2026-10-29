# scientific visuals

use this style for explanatory plots and physics diagrams. the goal is to connect a physical mechanism, its mathematical model, and the simulated device.

## output

- deliver plots and diagrams as PNG images that appear directly in the conversation.
- use standard plotting and drawing tools for scientific figures. calculate curves from the stated equations or actual model outputs.
- prefer PNG over JPEG for crisp text, curves, and line diagrams.
- keep the generating script and data with the output when practical, without changing simulation code to make a figure.

## teaching structure

- start with the physical question the figure answers.
- show the familiar physical picture before the quantitative plot when it helps understanding.
- connect views of the same phenomenon. for example, show carriers, fixed charge, electric field, potential, and energy bands with consistent spatial directions.
- connect the local mechanism to the complete device. for example, show a junction cross-section, optical-mode overlap, and the resulting ring transmission.
- compare a reference state with one changed state. hold other inputs fixed and state what changed.
- use a short caption to explain what the plot shows and the main physical connection.
- define symbols and units before using them in the explanation. explain the physical reasoning before pointing to code.
- show enough numerical context to connect an input change to an output change.

## typography and language

- use SF Mono throughout. on macOS, the font used for these examples is `/System/Library/Fonts/SFNSMono.ttf`.
- use a readable monospace fallback when SF Mono is unavailable.
- prefer lowercase prose, short sentences, and consistent physical terms.
- use a dash instead of a colon or semicolon in explanatory prose. preserve code syntax and scientific notation.
- preserve scientific capitalization in symbols and units, such as K, V, W, Hz, and eV.
- render mathematical notation with LaTeX or Matplotlib mathtext, including subscripts, superscripts, fractions, and Greek letters.
- keep math visually the same size as adjacent text. use matching letterforms where supported and readable mathematical-symbol fallbacks where needed.
- use regular font weight. establish hierarchy through size, position, and whitespace.

starting sizes for a figure about 12 inches wide —

| element | size |
| --- | --- |
| figure title | 20 pt |
| panel title | 14–15 pt |
| axis labels and main text | 14 pt |
| ticks, units, legends, and annotations | 12 pt |
| secondary captions | 11–12 pt |

check readability at the image's actual display width. enlarge text or split the figure when necessary.

## composition and spacing

- use a white background with dark text and generous whitespace.
- align the main title and subtitle to the left.
- give the figure a concrete title about the phenomenon. avoid generic titles such as “simulation results”.
- use a short subtitle for fixed conditions, changed inputs, and model identity.
- arrange direct comparisons side by side with aligned axes. use shared limits when comparing magnitudes.
- use two columns for compact related panels. let an important device-level plot span the full width.
- reserve separate vertical space for the title, subtitle, panel titles, units, plots, axis labels, legend, and caption.
- leave visible gaps between all of those elements. do not rely on automatic layout to prevent collisions.
- keep the outer margins comfortable. a starting range is 7.5–10% on the left and 4% on the right.
- use open layouts rather than boxes around every panel. remove top and right plot spines.
- split a crowded explanation into several images instead of shrinking labels or adding overlapping callouts.

## axes and annotations

- keep all axis labels horizontal. place the y quantity and units above the upper-left corner of the plot.
- put the x quantity and units below the plot.
- use few, readable ticks and restrained numerical precision.
- use thin, pale horizontal grid lines only when they help read values.
- distinguish a reference curve with a dashed gray line and the changed state with a solid blue line.
- keep curve identities consistent across all related panels.
- place legends in reserved whitespace, away from axis labels and curves.
- prefer short direct labels for important features. use thin arrows only when their direction or attachment adds meaning.
- annotate the important numerical result, such as a resonance shift or transmitted-power change.
- label the physical meaning of shaded regions, boundaries, and markers.
- avoid placing text on data curves. reposition the label or increase the available space.

## palette

use a small, muted palette. color must identify a physical region, quantity, or comparison state.

| role | color |
| --- | --- |
| main text | `#20252b` |
| secondary text | `#505861` |
| reference state | `#71777e` |
| changed state | `#246a9d` |
| p-type region | `#b35a38` |
| n-type region | `#286c9b` |
| optical mode or a second distinct quantity | `#7e55a5` |
| depletion-region shading | `#e5c15b` |
| grid | `#e7eaed` |
| axes | `#cbd1d6` |

- keep large region fills pale, usually with opacity around 0.07–0.35.
- keep quantitative curves clear, usually around 2 pt wide.
- pair color with line style, labels, or marker shape so color alone never carries the explanation.
- reuse these roles where they apply. define another consistent mapping when the physics requires it.

## diagrams and scientific accuracy

- distinguish illustrative geometry from extracted geometry and actual simulation output.
- label an idealized junction, assumed carrier profile, or illustrative optical mode as a teaching model.
- state the assumptions that determine plotted numbers, including doping, temperature, bias, and reference state where relevant.
- distinguish actual device inputs from teaching choices, such as a conveniently placed probe wavelength.
- preserve physical directions, charge signs, energy conventions, units, and reference offsets across panels.
- state what a curve represents. an energy-band edge is not an electron trajectory, and a mode contour is not a fabricated boundary.
- do not imply that a compact runtime model solves spatial charge or band diagrams at every step.
- keep accuracy limits concise and specific. separate model consistency from agreement with hardware measurements.

## rendering and review

- start around 12 inches wide at 180 dpi, giving an image about 2160 pixels wide. adjust height to the content.
- leave enough room for captions and legends when exporting. do not crop them to recover space.
- inspect every exported PNG before delivery.
- check titles against subtitles, panel headings against unit labels, legends against axes, and annotations against curves.
- check that every symbol renders, every image opens, and meaningful labels remain readable at conversation width.
- verify units, representative numerical values, and the connection between the stated input change and plotted result.
- make layout corrections and inspect the revised image again.

## reference examples

the approved examples are the pn-junction figures generated on 29 october 2026 —

- [charge and electric field](/Users/kenny/.codex/visualizations/2026/10/29/01a188b0-ed14-7dd0-9dd4-bcfac0719cd7/pn-charge-and-field.png)
- [potential and energy bands](/Users/kenny/.codex/visualizations/2026/10/29/01a188b0-ed14-7dd0-9dd4-bcfac0719cd7/pn-potential-and-bands.png)
- [junction to ring](/Users/kenny/.codex/visualizations/2026/10/29/01a188b0-ed14-7dd0-9dd4-bcfac0719cd7/junction-to-ring.png)
- [junction lookup curves](/Users/kenny/.codex/visualizations/2026/10/29/01a188b0-ed14-7dd0-9dd4-bcfac0719cd7/junction-lookup-curves.png)
