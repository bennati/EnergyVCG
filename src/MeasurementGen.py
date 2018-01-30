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
