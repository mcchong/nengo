"""Tests for nengo.model.Model"""

import numpy as np
import pytest

import nengo


def test_seeding():
    """Test that setting the model seed fixes everything"""

    # TODO: this really just checks random parameters in ensembles.
    #   Are there other objects with random parameters that should be
    #   tested? (Perhaps initial weights of learned connections)

    m = nengo.Model('test_seeding')
    inp = nengo.Node(output=1, label='input')
    pre = nengo.Ensemble(nengo.LIF(40), 1, label='A')
    post = nengo.Ensemble(nengo.LIF(20), 1, label='B')
    nengo.Connection(inp, pre)
    nengo.Connection(pre, post, function=lambda x: x ** 2)

    m.seed = 872
    md1 = nengo.Simulator(m).model
    md2 = nengo.Simulator(m).model
    m.seed = 873
    md3 = nengo.Simulator(m).model

    def compare_objs(obj1, obj2, attrs, equal=True):
        """Compare the passed attributes of the two passed objects"""
        for attr in attrs:
            check = (np.all(getattr(obj1, attr) == getattr(obj2, attr))
                     if equal else
                     np.any(getattr(obj1, attr) != getattr(obj2, attr)))
            if not check:
                print(getattr(obj1, attr))  # pylint:disable=superfluous-parens
                print(getattr(obj2, attr))  # pylint:disable=superfluous-parens
            assert check

    ens_attrs = ('encoders', 'max_rates', 'intercepts')
    pre = [next(o for o in mi.objs if o.label == 'A')
           for mi in [md1, md2, md3]]
    post = [next(o for o in mi.objs if o.label == 'B')
            for mi in [md1, md2, md3]]
    compare_objs(pre[0], pre[1], ens_attrs)
    compare_objs(post[0], post[1], ens_attrs)
    compare_objs(pre[0], pre[2], ens_attrs, equal=False)
    compare_objs(post[0], post[2], ens_attrs, equal=False)

    neur_attrs = ('gain', 'bias')
    compare_objs(pre[0].neurons, pre[1].neurons, neur_attrs)
    compare_objs(post[0].neurons, post[1].neurons, neur_attrs)
    compare_objs(pre[0].neurons, pre[2].neurons, neur_attrs, equal=False)
    compare_objs(post[0].neurons, post[2].neurons, neur_attrs, equal=False)


def test_time(Simulator):
    """Ensure time flows one dt at a time"""
    m = nengo.Model('test_time', seed=123)
    sim = Simulator(m)
    sim.run(0.003)
    assert np.allclose(sim.trange(), [0.00, .001, .002])


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
