import numpy as np
import matplotlib.pyplot as plt

import nengo
from nengo.nonlinearities import PES
from nengo_ocl.sim_ocl import Simulator as SimOCL
from nengo_ocl.sim_npy import Simulator as SimNumpy
import pyopencl as cl

sim_class = SimNumpy

N = 30
D = 2

model = nengo.Model('Learn Communication', seed=123)

# Create ensembles
pre = nengo.Ensemble(label='Pre', neurons=nengo.LIF(N * D), dimensions=D)
post = nengo.Ensemble(label='Post', neurons=nengo.LIF(N * D), dimensions=D)
error = nengo.Ensemble(label='Error', neurons=nengo.LIF(N * D), dimensions=D)

# Create an input signal
inn = nengo.Node(label='Input', output=lambda t: [np.sin(t), np.cos(t)])

nengo.Connection(inn, pre)

# Set the modulatory signal.
nengo.Connection(pre, error)
nengo.Connection(post, error, transform=np.eye(D) * -1)

# Create a modulated connection between the 'pre' and 'post' ensembles
nengo.Connection(pre, post, function=lambda x: -1 * np.ones(x.shape),
              learning_rule=PES(error))

# For testing purposes
error = nengo.Ensemble(label='Actual error', neurons=nengo.LIF(N * D), dimensions=D)
nengo.Connection(pre, error)
nengo.Connection(post, error, transform=np.eye(D) * -1)

pre_p = nengo.Probe(pre, 'decoded_output', filter=0.02)
post_p = nengo.Probe(post, 'decoded_output', filter=0.02)
error_p = nengo.Probe(error, 'decoded_output', filter=0.02)

if sim_class == SimOCL:
    ctx = cl.create_some_context()
    sim = sim_class(model, dt=0.001, context=ctx)
else:
    sim = sim_class(model, dt=0.001)

sim.run(5)

# Plot results
t = sim.trange()
plt.figure(figsize=(6,5))
plt.subplot(211)
plt.plot(t, sim.data(pre_p), label='Pre')
plt.plot(t, sim.data(post_p), label='Post')
plt.legend()
plt.subplot(212)
plt.plot(t, sim.data(error_p), label='Error')
plt.legend()
plt.tight_layout()
plt.show()
#plt.savefig('learning.pdf')
