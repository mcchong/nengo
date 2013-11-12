from .. import objects
from .. import templates

class Network(object):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.objects = []
        self.make(*args, **kwargs)

    def add(self, obj):
        self.objects.append(obj)
        return obj

    def make(self, *args, **kwargs):
        raise NotImplementedError("Networks should implement this function.")

    def add_to_model(self, model):
        for obj in self.objects:
            obj.name = self.name + '.' +  obj.name
            model.add(obj)

    def connect(self, o1, o2, **kwargs):
        o1.connect_to(o2, **kwargs)
    
    def ensemble_array(self, *args, **kwargs):
        return self.add(templates.EnsembleArray(*args, **kwargs))
    
    def passthrough(self, *args, **kwargs):
        return self.add(objects.PassthroughNode(*args, **kwargs))