import numpy as np
class BaseMeasurementGen():
    def __init__(self):
        np.random.seed()
        pass

    def get_measurements(self,population,timestep):
        """
        Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
        """
        return [{"value":0,"timestep":timestep,"agentID":i,"threshold":0} for i in range(len(population))]

class MeasurementGenUniform(BaseMeasurementGen):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.n1=int(kwargs["n1"])
        self.n2=int(kwargs["n2"])
        self.n=int(kwargs["N"])
        self.t=int(kwargs["T"])
        # self.thresh=int(kwargs["thresh"]*self.n*self.n2/2.0)
        self.thresh=int(kwargs["thresh"]*self.n)
        assert(self.n1>=0)
        assert(self.n2>self.n1)
        if self.t is None:
            print("setting t to default")
            self.t=1000


    def get_measurements(self,population,timestep):
        """
        Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
        """
        if timestep>self.t:
            return None
        else:
            vals=[np.random.randint(max(1,self.n1),self.n2) for _ in population]
            costs=[np.random.randint(max(1,self.n1),self.n2) for _ in population]
            # thresh=max(1,int(sum(vals)*np.random.uniform(0,1)))
            # thresh=len(population) #np.random.randint(1,3)
            if self.thresh>sum(vals):
                print("Thresh too high")
                self.thresh=np.floor(sum(vals))
            ret=[{"value":v,"cost":c,"timestep":timestep,"agentID":i,"threshold":self.thresh} for i,(v,c) in enumerate(zip(vals,costs))]
            return ret

class MeasurementGenNormal(BaseMeasurementGen):
    def __init__(self,*args, **kwargs):
        super().__init__()
        self.mu=kwargs["mu"]
        self.s=3

    def get_measurements(self,population,timestep):
        """
        Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
        """
        vals=[max(0.01,np.random.normal(loc=self.mu,scale=self.s)) for _ in population]
        thresh=len(population) #np.random.randint(1,10)
        assert(thresh<=sum(vals))
        ret=[{"value":v,"cost":0,"timestep":timestep,"threshold":thresh} for v in vals]
        return ret

class MeasurementGenBinomial(BaseMeasurementGen):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mu1=kwargs["mu1"]
        self.s1=1
        self.mu2=kwargs["mu2"]
        self.s2=1
        self.sep=kwargs["rich"]

    def get_measurements(self,population,timestep):
        """
        Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
        """
        vals=[(max(0.01,np.random.normal(loc=self.mu1,scale=self.s1))
               if i>len(population)*self.sep else
               max(0.01,np.random.normal(loc=self.mu2,scale=self.s2))) for i in range(len(population))]
        thresh=len(population) #np.random.randint(1,10)
        assert(thresh<=sum(vals))
        ret=[{"value":v,"cost":0,"timestep":timestep,"threshold":thresh} for v in vals]
        return ret
