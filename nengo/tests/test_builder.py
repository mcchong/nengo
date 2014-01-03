"""Tests for nengo.builder"""

import numpy as np
import pytest

import nengo
import nengo.builder

node_attrs = ('output',)
ens_attrs = ('label', 'dimensions', 'radius')
connection_attrs = ('filter', 'transform')


def compare(orig, copy):
    """Compare some nengo object before and after build."""
    if isinstance(orig, nengo.Node):
        attrs = node_attrs
    elif isinstance(orig, nengo.Ensemble):
        attrs = ens_attrs
    elif isinstance(orig, nengo.Connection):
        attrs = connection_attrs

    for attr in attrs:
        assert getattr(orig, attr) == getattr(copy, attr)
    for p_o, p_c in zip(orig.probes.values(), copy.probes.values()):
        assert len(p_o) == len(p_c)


def mybuilder(model, dt):
    """Mostly a stub builder so that we can predetermine operators."""
    model.dt = dt
    model.seed = 0
    if not hasattr(model, 'probes'):
        model.probes = []
    return model


def test_pyfunc():
    """Test PythonFunction nonlinearity"""
    dims = 3
    n_steps = 3
    n_trials = 3

    rng = np.random.RandomState(seed=987)

    for _ in range(n_trials):
        activities = rng.normal(size=(dims, dims))
        func = lambda t, x: np.cos(np.dot(activities, x))
        x = np.random.normal(size=dims)

        m = nengo.Model("")
        ins = nengo.builder.Signal(x, name='ins')
        pop = nengo.builder.PythonFunction(fn=func, n_in=dims, n_out=dims)
        m.operators = []
        builder = nengo.builder.Builder()
        builder.model = m
        builder.build_pyfunc(pop)
        m.operators += [
            nengo.builder.DotInc(
                nengo.builder.Signal(np.eye(dims)), ins, pop.input_signal),
            nengo.builder.ProdUpdate(nengo.builder.Signal(np.eye(dims)),
                                     pop.output_signal,
                                     nengo.builder.Signal(0),
                                     ins)
        ]

        sim = nengo.Simulator(m, builder=mybuilder)

        pop0 = np.zeros(dims)
        x = np.array(x)
        for _ in range(n_steps):
            tmp = pop0
            pop0 = func(0, x)
            x = tmp
            sim.step()
            assert np.allclose(x, sim.signals[ins])
            assert np.allclose(pop0, sim.signals[pop.output_signal])


def test_build():
    """Ensure build process doesn't modify objects."""
    m = nengo.Model('test_build', seed=123)
    inp = nengo.Node(output=1)
    pre = nengo.Ensemble(nengo.LIF(40), 1)
    post = nengo.Ensemble(nengo.LIF(20), 1)
    nengo.Connection(inp, pre)
    nengo.Connection(pre, post, function=lambda x: x ** 2)
    mcopy = nengo.Simulator(m).model
    assert [o.label for o in m.objs] == [o.label for o in mcopy.objs]
    for obj, copy_o in zip(m.objs, mcopy.objs):
        compare(obj, copy_o)
    for conn, copy_c in zip(m.connections, mcopy.connections):
        compare(conn, copy_c)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
