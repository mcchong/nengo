import nengo

import javaviz
import numpy as np


model = nengo.Model('Visualization Test')
with model:
    input = nengo.Node(lambda t: [np.sin(t), np.cos(t)], label='input')
    
    a = nengo.Ensemble(neurons=100, dimensions=2, label='A')
    b = nengo.Ensemble(neurons=100, dimensions=1, label='B')
    nengo.Connection(input, a, filter=0.01)
    nengo.Connection(a, b, filter=0.01, function=lambda x: [x[0]*x[1]])
    
    
    
javaviz.View(model)

import nengo_neurogrid_hardware    
params = nengo_neurogrid_hardware.Parameters()
params.set(model, training_time=5.0, use_software=True, software_global_pstc=0.01)
params.set(a, fmax=1000, offset=100)
params.set(b, fmax=1000, offset=100)
sim = nengo_neurogrid_hardware.Simulator(model, dt=0.001, seed=1, parameters=params)
sim.run(100000)
    
    
#sim = nengo.Simulator(model)
#sim.run(100000)
    
    