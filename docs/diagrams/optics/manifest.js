window.OPTICS_DIAGRAMS = [
  {
    "slug": "00-ring-model-reference",
    "title": "ring model - your reference",
    "subtitle": "the supplied drawing, preserved without redrawing",
    "code": "_exact_line",
    "reference": true
  },
  {
    "slug": "01-operating-side",
    "title": "where does the operating point sit?",
    "subtitle": "wavelength scan - hold the ring temperature fixed for this snapshot",
    "code": "ring_exact; _set_trial_temperature_phase"
  },
  {
    "slug": "02-resonance-orders",
    "title": "why can one ring have many resonances?",
    "subtitle": "one transverse mode can support many round-trip resonance orders",
    "code": "_phase_parts; _interpolated_scatterer_amplitude"
  },
  {
    "slug": "03-coupling-and-interference",
    "title": "why do fields need direction and phase?",
    "subtitle": "a coupler mixes fields - matching outgoing waves then interfere",
    "code": "_exact_line; _exact_line_d"
  },
  {
    "slug": "04-backscatter-and-splitting",
    "title": "what does backscatter change?",
    "subtitle": "part of the circulating light turns around and couples to the opposite direction",
    "code": "_interpolated_scatterer_amplitude; _exact_line"
  },
  {
    "slug": "05-travel-and-coordinate",
    "title": "which way does each field travel?",
    "subtitle": "physical travel and the order used to walk the ring are different ideas",
    "code": "_exact_line; _exact_stages"
  },
  {
    "slug": "06-field-closure",
    "title": "how are the circulating fields found?",
    "subtitle": "at a fixed temperature, the field must agree with itself after one complete trip",
    "code": "_exact_line; _exact_line_d"
  },
  {
    "slug": "07-energy-and-heating",
    "title": "where does incoming optical power go?",
    "subtitle": "distinguish travelling fields, circulating energy, optical loss, and deposited heat",
    "code": "_quarter_ring_energy_and_loss; evaluate_ring_optical_outputs; RingResponse"
  },
  {
    "slug": "08-two-data-levels",
    "title": "how does fast binary data enter the model?",
    "subtitle": "calculate both voltage states at the same temperature before averaging their powers",
    "code": "nrz_levels; evaluate_ring_optical_outputs; average_binary_data_levels"
  },
  {
    "slug": "09-eight-ring-bus",
    "title": "how do eight rings share the light?",
    "subtitle": "eight laser lines share the bus - upstream rings and waveguides remove power",
    "code": "bus_propagate_nb; _bus_pout_nb; BusResult"
  },
  {
    "slug": "10-reflections",
    "title": "which returned light is included?",
    "subtitle": "follow the actual optical path before deciding how its power is combined",
    "code": "_bus_refl_nb; _bus_pout_nb"
  },
  {
    "slug": "11-heat-paths",
    "title": "what determines a ring's temperature?",
    "subtitle": "heat arrives through several paths - each path retains its own thermal memory",
    "code": "ThermalSystem; mrm/plant/thermal.py"
  },
  {
    "slug": "12-self-heating-loop",
    "title": "how can temperature affect its own heating?",
    "subtitle": "follow the loop with the laser wavelength held fixed",
    "code": "_set_trial_temperature_phase; absorbed_power_at_trial_temperature"
  },
  {
    "slug": "13-heating-cooling-vector-fields",
    "title": "which temperatures attract or repel?",
    "subtitle": "compare absorbed heating with heat loss after accounting for other heating",
    "code": "selfheat.py - equilibrium interpretation"
  },
  {
    "slug": "14-folds-and-history",
    "title": "why can the ring suddenly jump?",
    "subtitle": "follow steady temperatures as the heater input changes slowly",
    "code": "selfheat.py - branch history; lorentzian_folds"
  },
  {
    "slug": "15-thermal-memory",
    "title": "what does a thermal step preserve?",
    "subtitle": "temperature approaches a new balance gradually - earlier warmth is retained",
    "code": "thermal.py - lag_coeffs; selfheat.py - implicit_step_b"
  },
  {
    "slug": "16-trial-temperature",
    "title": "what happens inside the temperature solve?",
    "subtitle": "candidate temperatures refer to one endpoint - solver iterations do not advance physical time",
    "code": "solve_ring_temperature; absorbed_power_at_trial_temperature; commit_optics_at_solved_temperature"
  },
  {
    "slug": "17-local-slope-and-bounds",
    "title": "what do derivatives and bounds tell the solver?",
    "subtitle": "a local slope describes the current sensitivity - a bound covers a whole specified interval",
    "code": "dpabs_dlambda_res_nb; pabs_bounds_nb; pabs_slope_bound_nb; selfheat_bounds_general"
  },
  {
    "slug": "18-controller-target",
    "title": "why hold a fraction of the monitor peak?",
    "subtitle": "the target is a tunable hyperparameter - it is not a fixed transmission or maximum-slope rule",
    "code": "mrm/control - PI controller; controller.params.fraction"
  },
  {
    "slug": "19-monitor-measurement",
    "title": "what does the controller actually measure?",
    "subtitle": "a larger ADC reading can come from more than a resonance shift",
    "code": "mrm/plant - monitor model; mrm/sim/kernel.py - sensor pipeline"
  },
  {
    "slug": "20-link-metrics",
    "title": "how can link quality be estimated without tracing every bit?",
    "subtitle": "retain the zero and one output powers - their separation predicts receiver margin",
    "code": "_bus_pout_nb; mrm/plant/datapath.py - ber_from_oma; errors_tick_row"
  },
  {
    "slug": "21-model-and-evidence",
    "title": "what establishes accuracy?",
    "subtitle": "calculation checks establish consistency - measurements test agreement with the device",
    "code": "tests/verification/test_optics.py; docs/KNOWN_ISSUES.md; docs/OPTICS_CONSOLIDATION.md"
  },
  {
    "slug": "22-one-control-interval",
    "title": "how do all the time scales fit together?",
    "subtitle": "fast data changes the optical state - slow thermal and control updates use its statistics",
    "code": "mrm/plant/plant.py; mrm/sim/kernel.py; mrm/plant/datapath.py"
  },
  {
    "slug": "23-stored-heat-and-cooling",
    "title": "why does temperature take time to change?",
    "subtitle": "incoming heat changes stored thermal energy while heat also escapes",
    "code": "mrm/plant/thermal.py - thermal lag states"
  },
  {
    "slug": "24-ring-model-choice",
    "title": "what does the chosen ring model resolve?",
    "subtitle": "one production path - solve the fields travelling in both directions",
    "code": "_exact_line; archive/optics/tcmt.py; mrm/waveform/tcmt_time.py"
  },
  {
    "slug": "25-peak-seeking-and-lock-loss",
    "title": "why can seeking the largest monitor reading lose lock?",
    "subtitle": "with strong self-heating, the high-reading state can sit close to a fold",
    "code": "selfheat.py - branch following; PI peak-fraction target"
  }
];
