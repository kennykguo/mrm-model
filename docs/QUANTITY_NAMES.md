# physical quantity names

name the quantity given by the expression, then state its symbol and units in the comment.
use the same name for the same quantity in the scalar, staged, cached, and reference calculations.
prefer verbosity over abbreviations that require the reader to decode a name from a comment.
row constants name both the row and the stored quantity, such as `RING_RESPONSE_ROW_REFLECTED_POWER_FRACTION`.
workspace row constants distinguish `COMPLEX_WORKSPACE` from `CLOCKWISE`.
derivative row constants identify round-trip phase as the independent variable.
packed-header names include `HEADER_INDEX`, cache indices include `CACHE_FIELD`, and view indices include `VIEW_INDEX`.
group the storage layouts and their counts at the top of the module, after imports.
the index cast is `_as_unsigned_array_index` - it keeps the original unsigned 64-bit representation.

| quantity | code name | meaning |
|---|---|---|
| $p=e^{-(\xi+i\phi)/4}$ | `quarter_propagation` | complex field propagation factor |
| $\partial p/\partial\phi$ | `d_quarter_propagation_dphi` | propagation derivative per radian |
| $\xi=-\ln(a_{\mathrm{rt}})$ | `round_trip_field_loss_exponent` | dimensionless field attenuation exponent |
| $a_{\mathrm{rt}}=e^{-\xi}$ | `round_trip_field_transmission` | retained field amplitude per trip |
| $\rho$ | `scatterer_reflection` | complex field reflection coefficient |
| $|\rho|^2$ | `scatterer_reflectance` | reflected power fraction |
| $b_{\mathrm{cw}}$ | `cw_derivative_rhs` | clockwise right-hand-side entry in the field phase derivative equation |
| $b_{\mathrm{ccw}}$ | `ccw_derivative_rhs` | counter-clockwise entry in the same equation |
| $\kappa/2+i\Delta$ | `complex_mode_decay_rate_per_s` | field decay and phase evolution |
| $P_{\mathrm{circ}}/P_{\mathrm{in}}$ | `circulating_power_ratio` | circulating power relative to input power |
| $e^{-h/\tau}$ | `state_retention` | thermal state fraction retained over a step |
| $S_{\mathrm{shot}}$ | `shot_noise_psd_a2_per_hz` | one-sided shot noise power density |

for derivatives, identify the numerator and the differentiation variable.
for example, `d_absorbed_power_d_ring_temp` differs from `d_absorbed_power_d_detuning`.
use a bound name when the value bounds a derivative rather than evaluating it.
keep equation symbols in the input and output comments, even when code names are longer.

keep public keyword arguments, recorded field names, configuration keys, and packed array layouts compatible.
retain conventional matrix names when their state-space role is clear.
reflow long code expressions without changing operation order or wrapping comment statements.

verification and progress are recorded in [the readability record](READABILITY_PASS.md).
