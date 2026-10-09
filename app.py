"""Launch: .venv/bin/streamlit run app.py --server.address 127.0.0.1"""

import io
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mrm_model.cli import json_safe, prbs9
from mrm_model.config import ExperimentConfig, default_config
from mrm_model.optics import NonlinearParams, optical_response
from mrm_model.simulation import _steady_state, compare_controllers
from mrm_model.waveform import simulate_nrz

st.set_page_config(page_title="Microring control laboratory", page_icon="◉", layout="wide")
st.title("Microring control laboratory")
st.caption("A configurable physical plant for locking experiments · SI units · public-source constraints")
st.info(
    "Device calibration is pending. Defaults are illustrative hypotheses; simulated tracking and OMA "
    "are not measured Lightmatter performance."
)

if "pending_config" in st.session_state:
    st.session_state.config_text = st.session_state.pop("pending_config")
if "config_text" not in st.session_state:
    st.session_state.config_text = default_config().to_json()

with st.sidebar:
    st.subheader("Build an experiment")
    ring_count = st.selectbox("Rings on the bus", [1, 4, 8], index=0)
    duration_ms = st.number_input("Duration (ms)", min_value=0.1, max_value=1000.0, value=10.0, step=1.0)
    sample_us = st.number_input(
        "Controller period (µs)", min_value=1.0, max_value=1000.0, value=10.0, step=1.0
    )
    disturbance = st.selectbox(
        "Disturbance", ["ambient_ramp", "ambient_step", "ambient_triangle", "local_heat", "constant"]
    )
    amplitude = st.number_input("Boundary amplitude (K)", min_value=0.0, max_value=120.0, value=2.0)
    ramp = st.number_input("Boundary ramp (K/s)", min_value=0.0, max_value=10000.0, value=808.0)
    thermal_scale = st.slider("Thermal conductance multiplier", 0.25, 4.0, 1.0, 0.05)
    capacity_scale = st.slider("Thermal capacity multiplier", 0.25, 4.0, 1.0, 0.05)
    if st.button("Load these settings", width="stretch"):
        c = default_config(ring_count)
        thermal = replace(
            c.thermal,
            conductance_w_per_k=c.thermal.conductance_w_per_k * thermal_scale,
            boundary_map_w_per_k=c.thermal.boundary_map_w_per_k * thermal_scale,
            capacitance_j_per_k=c.thermal.capacitance_j_per_k * capacity_scale,
        )
        heat = None
        if disturbance == "local_heat":
            heat = tuple((c.thermal.heater_map[:, 0] * 0.01).tolist())
        c = replace(
            c,
            thermal=thermal,
            run=replace(
                c.run,
                duration_s=duration_ms / 1000,
                sample_period_s=sample_us / 1e6,
                record_every=max(1, int(duration_ms / 1000 / (sample_us / 1e6) / 10000)),
            ),
            disturbance=replace(
                c.disturbance,
                kind=disturbance,
                ambient_amplitude_k=amplitude,
                ramp_rate_k_per_s=ramp,
                start_s=duration_ms / 5000,
                period_s=duration_ms / 2000,
                external_heat_peak_w=heat,
            ),
        )
        st.session_state.config_text = c.to_json()
        st.session_state.pop("results", None)
    st.caption(
        "These controls create a profile. The JSON below is the shared source of every model parameter."
    )
    st.caption("The Model guide tab explains the physics, assumptions and textbook differences.")

with st.expander("Full configuration · device, hardware, disturbance and controller", expanded=False):
    st.text_area("Experiment JSON", key="config_text", height=380)
    st.download_button(
        "Download configuration", st.session_state.config_text, "experiment.json", "application/json"
    )
    upload = st.file_uploader("Load configuration JSON", type=["json"])
    if upload is not None and st.button("Use uploaded configuration"):
        st.session_state.pending_config = upload.getvalue().decode()
        st.rerun()
try:
    config = ExperimentConfig.from_json(st.session_state.config_text)
except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
    st.error(f"Configuration needs attention: {error}")
    st.stop()

dynamics_tab, spectrum_tab, waveform_tab, calibration_tab, extraction_tab, guide_tab = st.tabs(
    [
        "Controller comparison",
        "Optical spectrum",
        "NRZ waveform",
        "Calibrate",
        "Physics extraction",
        "Model guide",
    ]
)

with dynamics_tab:
    choices = st.multiselect("Controllers", ["open_loop", "pi", "dither", "mpc"], default=["open_loop", "pi"])
    st.caption(
        "Each controller sees ADC codes and timestamps. All runs begin with the same plant and noise seed. "
        "PI, adaptive dither and local MPC hold the same calibrated flank. Peak seeking is optional."
    )
    if st.button("Run comparison", type="primary", disabled=not choices):
        with st.spinner("Integrating coupled physics and sampled hardware…"):
            try:
                st.session_state.results = compare_controllers(config, choices)
                st.session_state.results_config = config.to_json()
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                st.error(f"Experiment failed: {error}")
    results = st.session_state.get("results")
    if results:
        if st.session_state.get("results_config") != config.to_json():
            st.warning(
                "These results belong to the previous configuration. Run again to evaluate your edits."
            )
        rows = []
        for kind, result in results.items():
            m = result.metrics
            rows.append(
                {
                    "Controller": kind,
                    "Worst RMS (GHz)": max(m["tracking_rms_hz_per_lane"]) / 1e9,
                    "Worst excursion (GHz)": max(m["tracking_max_abs_hz_per_lane"]) / 1e9,
                    "Minimum lock (%)": min(m["lock_fraction_per_lane"]) * 100,
                    "Heater energy (µJ)": sum(m["heater_energy_j_per_lane"]) * 1e6,
                    "Lock excursions": sum(m["lock_excursions_per_lane"]),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        lane = st.selectbox("Inspect lane", list(range(len(next(iter(results.values())).config.optical))))
        figs = [go.Figure() for _ in range(4)]
        for kind, result in results.items():
            tr = result.traces
            for fig, key, scale in zip(
                figs,
                ["tracking_error_hz", "command_codes", "adc_codes", "oma_w"],
                [1e-9, 1, 1, 1e6],
                strict=True,
            ):
                fig.add_trace(
                    go.Scatter(x=tr["time_s"] * 1e3, y=tr[key][:, lane] * scale, name=kind, mode="lines")
                )
        for fig, title, unit in zip(
            figs,
            ["Assigned resonance tracking", "Heater commands", "Measured monitor", "Quasi-static OMA"],
            ["Error (GHz)", "DAC code", "ADC code", "OMA (µW)"],
            strict=True,
        ):
            fig.update_layout(
                title=title,
                xaxis_title="Time (ms)",
                yaxis_title=unit,
                height=320,
                margin={"l": 20, "r": 20, "t": 50, "b": 30},
            )
        left, right = st.columns(2)
        with left:
            st.plotly_chart(figs[0], width="stretch")
            st.plotly_chart(figs[2], width="stretch")
        with right:
            st.plotly_chart(figs[1], width="stretch")
            st.plotly_chart(figs[3], width="stretch")
        st.download_button(
            "Download comparison metrics",
            json.dumps(json_safe({k: v.metrics for k, v in results.items()}), indent=2),
            "comparison.json",
            "application/json",
        )
        st.download_button(
            "Download traces for selected lane",
            pd.DataFrame(
                {
                    f"{kind}_{key}": (value[:, lane] if value.ndim == 2 else value)
                    for kind, result in results.items()
                    for key, value in result.traces.items()
                }
            ).to_csv(index=False),
            "traces.csv",
            "text/csv",
        )
        with st.expander("Validity and diagnostics"):
            for kind, result in results.items():
                st.write(kind)
                for warning in result.warnings:
                    st.caption(warning)
                st.json(result.metrics)
    else:
        st.write(
            "Run the baseline and PI together, then add dither or model predictive control. "
            "Try changing the heat capacity, DAC delay or laser drift in the shared configuration."
        )

with spectrum_tab:
    index = st.selectbox("Ring", list(range(len(config.optical))), key="spectrum_ring")
    p = config.optical[index]
    temp = st.number_input(
        "Ring temperature (K)", min_value=1.0, max_value=700.0, value=float(p.reference_temperature_k)
    )
    voltage = st.number_input("Junction voltage (V)", value=0.0, step=0.1)
    x = np.linspace(p.reference_frequency_hz - p.fsr_hz, p.reference_frequency_hz + p.fsr_hz, 4001)
    try:
        optical = optical_response(p, x, temp, voltage, 1)
        fig = go.Figure()
        for label, values in (
            ("Through", optical.through_power_w),
            ("Drop monitor", optical.drop_power_w),
            ("Dissipated", optical.dissipated_power_w),
        ):
            fig.add_trace(go.Scatter(x=(x - p.reference_frequency_hz) / 1e9, y=values, name=label))
        fig.update_layout(
            xaxis_title="Offset from reference resonance (GHz)", yaxis_title="Power fraction", height=430
        )
        st.plotly_chart(fig, width="stretch")
        cols = st.columns(3)
        cols[0].metric("Derived FSR", f"{p.fsr_hz / 1e12:.3f} THz")
        cols[1].metric("Exact periodic linewidth", f"{p.exact_fwhm_hz / 1e9:.2f} GHz")
        cols[2].metric("Energy lifetime", f"{p.photon_lifetime_s * 1e12:.2f} ps")
        st.caption(
            "Exact periodic transfer for uniform loss, ideal couplers and one traveling mode. "
            "Adjacent resonances are retained. A low FSR alone does not determine WDM failure."
        )
    except ValueError as error:
        st.error(str(error))

with waveform_tab:
    st.write(
        "A short, independent experiment resolves electrical settling, photon lifetime and the NRZ eye. "
        "It uses the first ring at the configured initial thermal operating point."
    )
    count = st.number_input("PRBS9 bits", min_value=32, max_value=20000, value=511, step=511)
    rate = st.number_input("Data rate (Gb/s)", min_value=1.0, max_value=200.0, value=50.0)
    resolution = st.select_slider("Samples per bit", [8, 16, 32, 64], value=32)
    nonlinear_json = st.text_area(
        "Optional nonlinear coefficients (SI; {} disables nonlinear effects)", value="{}", height=90
    )
    backscatter = st.number_input(
        "Counterpropagating coupling (rad/s)", min_value=0.0, value=0.0, format="%.3e"
    )
    waveform_inputs = {
        "config": config.to_dict(),
        "bits": int(count),
        "bit_rate_bps": rate * 1e9,
        "samples_per_bit": resolution,
        "nonlinear_json": nonlinear_json,
        "backscatter_rad_per_s": backscatter,
    }
    if st.button("Simulate waveform"):
        try:
            nonlinear = NonlinearParams(**json.loads(nonlinear_json))
            network, _ = _steady_state(config)
            with st.spinner("Integrating the optical cavity…"):
                st.session_state.waveform = simulate_nrz(
                    config.optical[0],
                    prbs9(int(count)),
                    bit_rate_bps=rate * 1e9,
                    samples_per_bit=resolution,
                    voltage_low_v=config.run.junction_low_v,
                    voltage_high_v=config.run.junction_high_v,
                    laser_frequency_hz=config.run.laser_frequency_hz[0],
                    temperature_k=network.ring_temperatures_k[0],
                    input_power_w=config.run.input_power_w,
                    nonlinearity=nonlinear,
                    backscatter_rad_per_s=backscatter,
                )
                st.session_state.waveform_resolution = resolution
                st.session_state.waveform_rate = rate * 1e9
                st.session_state.waveform_inputs = waveform_inputs
        except (ValueError, TypeError, RuntimeError) as error:
            st.error(str(error))
    if "waveform" in st.session_state:
        w = st.session_state.waveform
        if st.session_state.get("waveform_inputs") != waveform_inputs:
            st.warning(
                "This waveform belongs to previous configuration or waveform settings. "
                "Simulate again to evaluate your edits."
            )
        st.json(json_safe(w["metrics"]))
        fig = go.Figure()
        samples = min(len(w["time_s"]), 24 * st.session_state.waveform_resolution)
        fig.add_trace(
            go.Scatter(
                x=w["time_s"][:samples] * 1e12, y=w["through_power_w"][:samples] * 1e6, name="Through power"
            )
        )
        fig.update_layout(xaxis_title="Time (ps)", yaxis_title="Optical power (µW)")
        st.plotly_chart(fig, width="stretch")
        eye = go.Figure()
        width = 2 * st.session_state.waveform_resolution
        for i in range(
            16 * st.session_state.waveform_resolution, min(len(w["time_s"]) - width, width * 120), width
        ):
            eye.add_trace(
                go.Scatter(
                    x=np.arange(width) / st.session_state.waveform_resolution,
                    y=w["receiver_power_w"][i : i + width] * 1e6,
                    mode="lines",
                    line={"color": "rgba(48,114,212,.2)"},
                    showlegend=False,
                )
            )
        eye.update_layout(title="NRZ eye overlays", xaxis_title="Unit intervals", yaxis_title="Power (µW)")
        st.plotly_chart(eye, width="stretch")
        st.caption(
            "Gaussian Q/BER fields are analytical proxies for this assumed receiver. "
            "They do not reproduce the paper's measured BER or certify link reliability."
        )
        st.json(json_safe(w["validity"]))
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **{k: v for k, v in w.items() if isinstance(v, np.ndarray)})
        st.download_button("Download waveform arrays", buffer.getvalue(), "waveform.npz")
        st.download_button(
            "Download waveform inputs",
            json.dumps(st.session_state.get("waveform_inputs"), indent=2, allow_nan=False),
            "waveform_inputs.json",
            "application/json",
        )

with calibration_tab:
    st.write(
        "Fit measured spectra or heater steps and inspect residuals on held-out samples. "
        "A good fit to one trace does not establish accuracy across temperatures or devices."
    )
    fit_kind = st.radio("Dataset", ["Optical spectrum", "Thermal step"], horizontal=True)
    csv_upload = st.file_uploader("Calibration CSV", type=["csv"], key="calibration_csv")
    st.caption(
        "Spectrum: frequency_hz, through_power_w, optional drop_power_w. "
        "Thermal: time_s, frequency_shift_hz. Use a known constant optical input or heater step."
    )
    power = st.number_input(
        "Known optical input / heater step (W)", min_value=1e-12, value=0.001, format="%.6g"
    )
    modes = st.selectbox("Thermal fit modes", [1, 2, 3])
    if st.button("Fit uploaded data", disabled=csv_upload is None):
        try:
            from mrm_model.calibration import fit_optical_spectrum, fit_thermal_step

            data = pd.read_csv(csv_upload)
            if fit_kind == "Optical spectrum":
                fit = fit_optical_spectrum(
                    config.optical[0],
                    data.frequency_hz,
                    data.through_power_w,
                    drop_power_w=data.drop_power_w if "drop_power_w" in data else None,
                    input_power_w=power,
                )
            else:
                fit = fit_thermal_step(
                    data.time_s, data.frequency_shift_hz, n_modes=modes, step_power_w=power
                )
            st.json(json_safe(asdict(fit)))
            st.download_button(
                "Download fit and residual diagnostics",
                json.dumps(json_safe(asdict(fit)), indent=2),
                "fit.json",
                "application/json",
            )
        except (ValueError, KeyError, AttributeError) as error:
            st.error(f"Could not fit dataset: {error}")

with extraction_tab:
    st.write(
        "Extract a passive thermal reduced-order model from a generic SOI cross-section using sparse FEM. "
        "The geometry, material constants and boundary conditions are assumptions."
    )
    if st.button("Run thermal mesh and time convergence"):
        from mrm_model.extraction import thermal_fem_convergence

        with st.spinner("Solving coarse/fine sparse heat models…"):
            try:
                st.session_state.fem = thermal_fem_convergence(n_modes=32)
            except (ValueError, RuntimeError, ImportError) as error:
                st.error(str(error))
    if "fem" in st.session_state:
        report = st.session_state.fem
        st.json({key: json_safe(value) for key, value in report.items() if key not in {"coarse", "fine"}})
        st.download_button(
            "Download passive FEM model and convergence report",
            json.dumps(json_safe(report), indent=2),
            "thermal_fem.json",
            "application/json",
        )
    st.caption(
        "See docs/extraction.md for full-vector optical mode extraction. No commercial license is needed."
    )

with guide_tab:
    guide = Path(__file__).parent / "docs/model-guide.md"
    if guide.exists():
        st.markdown(guide.read_text())
    else:
        st.write(
            "Heater commands deposit watts into a coupled heat network. Temperature moves resonance. "
            "The bus/drop optical response becomes photocurrent, filtered and quantized for a controller. "
            "Reference physics and parameter provenance are documented in the repository."
        )
