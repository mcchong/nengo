"""Symbolic neurons classes.

These represent all of the neuron types implemented
in the reference simulator. We recommend that other simulators
implement these neuron types if possible, as it will
help for testing and comparing to the reference simulator.
However, if that isn't possible, create a Neurons subclass
that represents information about the neurons implemented
in your simulator so that Ensembles can be created
with those neurons.

"""

import copy
import logging

import numpy as np

import nengo
from . import decoders

logger = logging.getLogger(__name__)


class Neurons(object):
    """Neurons superclass.

    This defines the interface that Ensembles and the reference simulator
    expect from a Neuron class. However, see nengo.neurons.Direct
    for an example where these functions are stubbed instead.

    """

    def __init__(self, n_neurons, bias=None, gain=None, label=None):
        self.n_neurons = n_neurons
        self.bias = bias
        self.gain = gain
        if label is None:
            label = "<%s%d>" % (self.__class__.__name__, id(self))
        self.label = label

        self.probes = {'output': []}

    def __str__(self):
        return  "{0} ({1}{2})".format(
            self.__class__.__name__,
            self.label if hasattr(self, 'label') else "id " + str(id(self)),
            ", %dN" if hasattr(self, 'n_neurons') else "")

    def __repr__(self):
        return str(self)

    def default_encoders(self, dimensions, rng):
        """Returns a set of encoders.

        This is called if the owning ensemble has no encoders.
        """
        raise NotImplementedError("Neurons must provide default_encoders")

    def rates(self, x):
        """Returns the firing rates of these neurons given input vector x."""
        raise NotImplementedError("Neurons must provide rates")

    def set_gain_bias(self, max_rates, intercepts):
        """Sets the gain and bias of these neurons automatically.

        And by automatically we mean given the maximum firing rates
        and x-intercepts of all of the neurons. Essentially,
        this function should generate tuning curves under the constraints
        of having the appropriate maximum firing rates and intercepts.
        """
        raise NotImplementedError("Neurons must provide set_gain_bias")

    def probe(self, probe):
        """Probes the output of these neurons."""
        self.probes[probe.attr].append(probe)

        if probe.attr == 'output':
            nengo.Connection(self, probe, filter=probe.filter)
        else:
            raise NotImplementedError(
                "Probe target '%s' is not probable" % probe.attr)
        return probe

    def add_to_model(self, model):
        """Adds these neurons to the passed model."""
        model.objs.append(self)


class Direct(Neurons):
    """Does the computation directly, without a neuron approximation.
    Representations are exact, and transformations are directly computed.

    Most of this is stubbed out; see nengo.builder.Builder to see
    how the direct computation is implemented.
    """

    def __init__(self, n_neurons=None, label=None):
        # n_neurons is ignored, but accepted to maintain compatibility
        # with other neuron types
        Neurons.__init__(self, 0, label=label)

    def default_encoders(self, dimensions, rng):
        return np.identity(dimensions)

    def rates(self, x):
        return x

    def set_gain_bias(self, max_rates, intercepts):
        pass


class LIFRate(Neurons):
    """Leaky integrate-and-fire neuron that emits firing rates."""

    def __init__(self, n_neurons, tau_rc=0.02, tau_ref=0.002, label=None):
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        Neurons.__init__(self, n_neurons, label=label)

    def default_encoders(self, dimensions, rng):
        return decoders.sample_hypersphere(
            dimensions, self.n_neurons, rng, surface=True)

    def rates_from_current(self, J):
        """LIF firing rates in Hz for input current (incl. bias)."""
        old = np.seterr(divide='ignore')
        try:
            j = np.maximum(J - 1, 0.)
            retval = 1. / (self.tau_ref + self.tau_rc * np.log1p(1. / j))
        finally:
            np.seterr(**old)  # pylint:disable=star-args
        return retval

    def rates(self, x):
        """LIF firing rates in Hz for vector space.

        Parameters
        ---------
        x: ndarray of any shape
            vector-space inputs
        """
        J = self.gain * x + self.bias
        return self.rates_from_current(J)

    def set_gain_bias(self, max_rates, intercepts):
        """Compute the alpha and bias needed to get the given max_rate
        and intercept values.

        Returns gain (alpha) and offset (j_bias) values of neurons.

        Parameters
        ---------
        max_rates : list of floats
            Maximum firing rates of neurons.
        intercepts : list of floats
            X-intercepts of neurons.

        """
        logging.debug("Setting gain and bias on %s", self.label)
        max_rates = np.asarray(max_rates)
        intercepts = np.asarray(intercepts)
        x = 1.0 / (1 - np.exp(
            (self.tau_ref - (1.0 / max_rates)) / self.tau_rc))
        self.gain = (1 - x) / (intercepts - 1.0)
        self.bias = 1 - self.gain * intercepts

    def step(self, dt, J):
        """Compute rates for input current (incl. bias)"""
        return dt * self.rates_from_current(J)


class LIF(LIFRate):
    """Leaky integrate-and-fire neuron that emits spikes."""

    def __init__(self, n_neurons, upsample=1, **kwargs):
        LIFRate.__init__(self, n_neurons, **kwargs)
        self.upsample = upsample

    # pylint:disable=arguments-differ
    def step(self, dt, J, voltage, refractory_time, spiked):
        if self.upsample != 1:
            raise NotImplementedError()

        # update voltage using Euler's method
        dV = (dt / self.tau_rc) * (J - voltage)
        voltage += dV
        voltage[voltage < 0] = 0  # clip values below zero

        # update refractory period assuming no spikes for now
        refractory_time -= dt

        # set voltages of neurons still in their refractory period to 0
        # and reduce voltage of neurons partway out of their ref. period
        voltage *= (1 - refractory_time / dt).clip(0, 1)

        # determine which neurons spike (if v > 1 set spiked = 1, else 0)
        spiked[:] = (voltage > 1)

        # linearly approximate time since neuron crossed spike threshold
        overshoot = (voltage[spiked > 0] - 1) / dV[spiked > 0]
        spiketime = dt * (1 - overshoot)

        # set spiking neurons' voltages to zero, and ref. time to tau_ref
        voltage[spiked > 0] = 0
        refractory_time[spiked > 0] = self.tau_ref + spiketime
