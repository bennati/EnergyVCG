from MeasurementGen import BaseMeasurementGen
from utils import renormalize
import pandas as pd
import numpy as np
import itertools

class MeasurementGenEV(BaseMeasurementGen):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.n1=int(kwargs["n1"])
        self.n2=int(kwargs["n2"])
        assert(self.n1>=0)
        assert(self.n2>self.n1)
        self.n=int(kwargs["N"])
        self.t=int(kwargs["T"])
        self.thresh=float(kwargs["thresh"])
        baseload=pd.read_csv("../datasets/energy/origBaseload.csv",names=["time","load"])
        production=pd.read_csv("../datasets/energy/WindGenRelative.csv",names=["time","prod"])
        production["prod"]/=100.0
        baseload["time"]=pd.to_datetime(baseload["time"])
        production["time"]=pd.date_range(production["time"][0], periods=production.shape[0], freq='15min')
        data=pd.merge(baseload,production,on="time")
        # TODO apply by row, if load is larger than value, set it to default
        thresh=data.apply(lambda x: (x["prod"]-x["load"] if x["load"]<=x["prod"] else -1),axis=1)
        thresh=[renormalize(r,[thresh.min(),thresh.max()],[0,1]) for r in thresh if r>=0]
        assert(all([t>=0 and t<=1 for t in thresh]))
        del data,baseload,production
        if self.t is None:
            print("setting t to default")
            self.t=1000
        if len(thresh)<self.t:
            thresh+=thresh*(self.t//len(thresh))
        self.thresh=(1.0-np.asarray(thresh[:self.t+1]))*float(self.n) # subtract excess production from fixed threshold
        assert(all([t>=0 and t<=self.n for t in self.thresh]))

    def get_measurements(self,population,timestep):
        """
        Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
        """
        if timestep>self.t:
            return None
        else:
            vals=[np.random.uniform(max(1,self.n1),self.n2) for _ in population]
            costs=[max(0,np.random.normal(v,abs((self.n2-self.n1)/5.0))) for v in vals]
            # thresh=max(1,int(sum(vals)*np.random.uniform(0,1)))
            # thresh=len(population) #np.random.randint(1,3)
            thresh=self.thresh[timestep]
            assert(thresh<=sum(vals))
            ret=[{"value":v,"cost":c,"timestep":timestep,"agentID":i,"threshold":thresh} for i,(v,c) in enumerate(zip(vals,costs))]
            return ret
