from .. import objects
from ..objects import Uniform
from . import Network
from ..templates import EnsembleArray

import nengo
import numpy as np

class BasalGanglia(Network):
    # connection weights from (Gurney, Prescott, & Redgrave, 2001)
    mm = 1
    mp = 1
    me = 1
    mg = 1
    ws = 1
    wt = 1
    wm = 1
    wg = 1
    wp = 0.9
    we = 0.3
    e = 0.2
    ep = -0.25
    ee = -0.2
    eg = -0.2
    le = 0.2
    lg = 0.2
    
    def make(self, dimensions, n_neurons_per_ensemble=100, radius=1.5,
             tau_ampa=0.002, tau_gaba=0.008, output_weight=-3):
        
        

        encoders = np.ones((n_neurons_per_ensemble, 1))
        ea_params = {
            'neurons': nengo.LIF(n_neurons_per_ensemble * dimensions),
            'n_ensembles': dimensions,
            'radius': radius,
            'encoders': encoders,
        }

        strD1 = self.ensemble_array('Striatal D1 neurons', intercepts=objects.Uniform(self.e, 1), **ea_params)

        strD2 = self.ensemble_array('Striatal D2 neurons', intercepts=objects.Uniform(self.e, 1), **ea_params)

        stn = self.ensemble_array('Subthalamic nucleus', intercepts=objects.Uniform(self.ep, 1), **ea_params)

        gpi = self.ensemble_array('Globus pallidus internus', intercepts=objects.Uniform(self.eg, 1), **ea_params)

        gpe = self.ensemble_array('Globus pallidus externus', intercepts=objects.Uniform(self.ee, 1), **ea_params)

        self.input = self.passthrough("input", dimensions=dimensions)
        self.output = self.passthrough("output", dimensions=dimensions)

        # spread the input to StrD1, StrD2, and STN
        self.connect(input, strD1, filter=None,
            transform=np.eye(dimensions) * self.ws * (1 + self.lg))
        self.connect(input, strD2, filter=None,
            transform=np.eye(dimensions) * self.ws * (1 - self.le))
        self.connect(input, stn, filter=None,
            transform=np.eye(dimensions) * self.wt)

        # connect the striatum to the GPi and GPe (inhibitory)
        def func_str(x):
            if x[0] < self.e:
                return 0
            return self.mm * (x[0] - self.e)
        self.connect(strD1, gpi, function=func_str, filter=tau_gaba,
                         transform=-np.eye(dimensions) * self.wm)
        self.connect(strD2, gpe, function=func_str, filter=tau_gaba,
                        transform=-np.eye(dimensions) * self.wm)

        # connect the STN to GPi and GPe (broad and excitatory)
        def func_stn(x):
            if x[0] < self.ep:
                return 0
            return self.mp * (x[0] - self.ep)
        tr = np.ones((dimensions, dimensions)) * self.wp
        self.connect(stn, gpi, function=func_stn, transform=tr, filter=tau_ampa)
        self.connect(stn, gpe, function=func_stn, transform=tr, filter=tau_ampa)

        # connect the GPe to GPi and STN (inhibitory)
        def func_gpe(x):
            if x[0] < self.ee:
                return 0
            return self.me * (x[0] - self.ee)
        self.connect(gpe, gpi, function=func_gpe, filter=tau_gaba,
                       transform=-np.eye(dimensions) * self.we)
        self.connect(gpe, stn, function=func_gpe, filter=tau_gaba,
                       transform=-np.eye(dimensions) * self.wg)

        #connect GPi to output (inhibitory)
        def func_gpi(x):
            if x[0] < self.eg:
                return 0
            return self.mg * (x[0] - self.eg)
        self.connect(gpi, self.output, function=func_gpi, filter=None,
                       transform=np.eye(dimensions) * output_weight)