import nengo

import javaviz
import numpy as np


model = nengo.Model('Visualization Test')
with model:
    input = nengo.Node(lambda t: [np.sin(t), np.cos(t)], label='input')
    
    a = nengo.Ensemble(neurons=100, dimensions=2, label='A')
    b = nengo.Ensemble(neurons=100, dimensions=1, label='B')
    nengo.Connection(input, a, filter=None)
    nengo.Connection(a, b, filter=0.01, function=lambda x: x[0]*x[1])
    
    
    
javaviz.View(model)
    
sim = nengo.Simulator(model)
sim.run(100000)
    
    