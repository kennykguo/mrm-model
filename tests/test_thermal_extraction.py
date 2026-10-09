import numpy as np

from ringlab.extraction.thermal import (
    Layer, Region, SlabGeometry, assemble_slab, assumed_slab_geometry,
)


def test_uniform_slab_converges_to_analytic_volume_average_at_second_order():
    # 1-D uniform generation, bottom Dirichlet, top insulated:
    # mean temperature rise = P L / (3 k A). No empirical target is involved.
    length, area, conductivity = 2e-6, 3e-12, 1.4
    region = Region("uniform", -0.5e-6, 0.5e-6, 0, length)
    geometry = SlabGeometry(1e-6, 3e-6, (Layer("synthetic uniform", length, conductivity, 1e6),),
                            (region,), (region,), "analytic verification geometry")
    exact_gain = length / (3 * conductivity * area)
    errors = []
    for n in (8, 16, 32):
        model = assemble_slab(geometry, nx=1, cells_per_layer=(n,))
        gain = model.dc_gain_k_per_w().item()
        errors.append(abs(gain - exact_gain))
        temperature = model.steady_state([1e-4], 300)
        outflow = model.boundary_conductance_w_per_k[:, 0] @ (temperature - 300)
        np.testing.assert_allclose(outflow, 1e-4, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(errors[:-1]) / errors[1:], [4, 4], rtol=1e-8)


def test_spatial_source_power_and_modal_stability_are_preserved():
    model = assemble_slab(assumed_slab_geometry(2), nx=48, cells_per_layer=(4, 3, 1, 3))
    np.testing.assert_allclose(model.source_map.sum(axis=0), 1)
    np.testing.assert_allclose(model.observer_matrix.sum(axis=1), 1)
    np.testing.assert_allclose(np.asarray(model.conductance_w_per_k.sum(axis=1)).ravel(),
                               model.boundary_conductance_w_per_k[:, 0], atol=1e-16)
    reduced = model.modal_reduce(16)
    assert np.all(reduced.decay_rates_per_s > 0)
    assert reduced.n_modes == 16
    # Modal coordinates are not temperatures. Stable eigenvalues alone do not
    # imply a useful reduction: the extractor must expose its actual DC error.
    assert 0 <= reduced.dc_gain_relative_error < 1
    powers = np.array([1e-3, 0, 0, 0, 0, 0])
    observed = model.observer_matrix @ model.steady_state(powers)
    assert observed[0] > observed[1] > 300
    sparse_bytes = (model.conductance_w_per_k.data.nbytes + model.conductance_w_per_k.indices.nbytes
                    + model.conductance_w_per_k.indptr.nbytes)
    assert sparse_bytes < model.n_nodes * 100


def test_more_modes_reduce_error_without_instantaneous_temperature_jump():
    model = assemble_slab(assumed_slab_geometry(1), nx=32, cells_per_layer=(3, 2, 1, 2))
    network = model.to_network()
    assert network.is_asymptotically_stable
    np.testing.assert_allclose(network.observe(network.steady_state([1e-3, 0, 0], 300)),
                               model.observer_matrix @ model.steady_state([1e-3, 0, 0]), atol=1e-8)
    small = model.modal_reduce(4)
    larger = model.modal_reduce(32)
    assert larger.dc_gain_relative_error < small.dc_gain_relative_error
    np.testing.assert_allclose(larger.step_response([0], [1e-3, 0, 0]), 300)


def test_krylov_projection_preserves_dc_and_resolves_step_dynamics():
    model = assemble_slab(assumed_slab_geometry(1), nx=32, cells_per_layer=(3, 2, 1, 2))
    reduced = model.krylov_reduce(4)
    assert reduced.n_modes <= 12
    assert np.all(reduced.decay_rates_per_s > 0)
    assert reduced.dc_gain_relative_error < 1e-10
    times = np.r_[0, np.geomspace(1e-8, 1e-3, 60)]
    reference = model.step_response(times, [1e-3, 0, 0])
    prediction = reduced.step_response(times, [1e-3, 0, 0])
    assert np.max(np.abs(reference - prediction)) < 0.01


def test_sparse_transient_reaches_the_same_steady_temperature():
    layer = Layer("analytic", 1e-6, 10, 1e6)
    region = Region("uniform", -0.5e-6, 0.5e-6, 0, 1e-6)
    model = assemble_slab(SlabGeometry(1e-6, 1e-6, (layer,), (region,), (region,)),
                           nx=2, cells_per_layer=(8,))
    values = model.step_response(np.r_[0, np.geomspace(1e-11, 1e-5, 30)], [1e-4])
    assert np.all(np.diff(values[:, 0]) >= -1e-7)
    expected = model.observer_matrix @ model.steady_state([1e-4])
    np.testing.assert_allclose(values[-1], expected, atol=1e-7)

