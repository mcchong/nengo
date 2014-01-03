"""Tests for nengo.helpers.whitenoise"""

import numpy as np
import pytest

import nengo
from nengo.helpers import whitenoise


def test_rms():
    """When setting rms, it should actually have that RMS"""
    t = np.linspace(0, 1, 1000)

    for rms_desired in [0, 0.5, 1, 100]:
        func = whitenoise(1, 100, rms=rms_desired)
        rms = np.sqrt(np.mean([func(tt) ** 2 for tt in t]))
        assert np.allclose(rms, rms_desired, atol=.1, rtol=.01)


def test_array():
    """Passing dimensions gives the right shape and right RMS"""
    rms_desired = 0.5
    func = whitenoise(1, 5, rms=rms_desired, dimensions=5)

    t = np.linspace(0, 1, 1000)
    data = np.array([func(tt) for tt in t])
    assert data.shape[1] == 5
    rms = np.sqrt(np.mean(data**2, axis=0))
    assert np.allclose(rms, rms_desired, atol=.1, rtol=.01)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
