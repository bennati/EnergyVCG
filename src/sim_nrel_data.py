from MeasurementGen import BaseMeasurementGen
import pandas as pd
import numpy as np
import itertools

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(itertools.islice(iterable, n))

#partition list a into k partitions
def partition_list(a, k):
    """https://stackoverflow.com/questions/35517051/split-a-list-of-numbers-into-n-chunks-such-that-the-chunks-have-close-to-equal
    a is a list of tuples, where the first element is the index of the data and the second its length. The algorithm divides in k groups whose sum of lenghts is balanced
    """
    #check degenerate conditions
    if k <= 1: return [a]
    if k >= len(a): return [[x] for x in a]
    #create a list of indexes to partition between, using the index on the
    #left of the partition to indicate where to partition
    #to start, roughly partition the array into equal groups of len(a)/k (note
    #that the last group may be a different size)
    partition_between = []
    for i in range(k-1):
        partition_between.append(int((i+1)*len(a)/k))
    #the ideal size for all partitions is the total height of the list divided
    #by the number of paritions
    average_height = float(sum([v for k,v in a]))/k
    best_score = None
    best_partitions = None
    count = 0
    no_improvements_count = 0
    #loop over possible partitionings
    while True:
        #partition the list
        partitions = []
        index = 0
        for div in partition_between:
            #create partitions based on partition_between
            partitions.append(a[index:div])
            index = div
        #append the last partition, which runs from the last partition divider
        #to the end of the list
        partitions.append(a[index:])
        #evaluate the partitioning
        worst_height_diff = 0
        worst_partition_index = -1
        for p in partitions:
            #compare the partition height to the ideal partition height
            height_diff = average_height - sum([v for k,v in p])
            #if it's the worst partition we've seen, update the variables that
            #track that
            if abs(height_diff) > abs(worst_height_diff):
                worst_height_diff = height_diff
                worst_partition_index = partitions.index(p)
        #if the worst partition from this run is still better than anything
        #we saw in previous iterations, update our best-ever variables
        if best_score is None or abs(worst_height_diff) < best_score:
            best_score = abs(worst_height_diff)
            best_partitions = partitions
            no_improvements_count = 0
        else:
            no_improvements_count += 1
        #decide if we're done: if all our partition heights are ideal, or if
        #we haven't seen improvement in >5 iterations, or we've tried 100
        #different partitionings
        #the criteria to exit are important for getting a good result with
        #complex data, and changing them is a good way to experiment with getting
        #improved results
        if worst_height_diff == 0 or no_improvements_count > 5 or count > 100:
            return best_partitions
        count += 1
        #adjust the partitioning of the worst partition to move it closer to the
        #ideal size. the overall goal is to take the worst partition and adjust
        #its size to try and make its height closer to the ideal. generally, if
        #the worst partition is too big, we want to shrink the worst partition
        #by moving one of its ends into the smaller of the two neighboring
        #partitions. if the worst partition is too small, we want to grow the
        #partition by expanding the partition towards the larger of the two
        #neighboring partitions
        if worst_partition_index == 0:   #the worst partition is the first one
            if worst_height_diff < 0: partition_between[0] -= 1   #partition too big, so make it smaller
            else: partition_between[0] += 1   #partition too small, so make it bigger
        elif worst_partition_index == len(partitions)-1: #the worst partition is the last one
            if worst_height_diff < 0: partition_between[-1] += 1   #partition too small, so make it bigger
            else: partition_between[-1] -= 1   #partition too big, so make it smaller
        else:   #the worst partition is in the middle somewhere
            left_bound = worst_partition_index - 1   #the divider before the partition
            right_bound = worst_partition_index   #the divider after the partition
            if worst_height_diff < 0:   #partition too big, so make it smaller
                if sum([v for k,v in partitions[worst_partition_index-1]]) > sum([v for k,v in partitions[worst_partition_index+1]]):   #the partition on the left is bigger than the one on the right, so make the one on the right bigger
                    partition_between[right_bound] -= 1
                else:   #the partition on the left is smaller than the one on the right, so make the one on the left bigger
                    partition_between[left_bound] += 1
            else:   #partition too small, make it bigger
                if sum([v for k,v in partitions[worst_partition_index-1]]) > sum([v for k,v in partitions[worst_partition_index+1]]): #the partition on the left is bigger than the one on the right, so make the one on the left smaller
                    partition_between[left_bound] -= 1
                else:   #the partition on the left is smaller than the one on the right, so make the one on the right smaller
                    partition_between[right_bound] += 1

class MeasurementGenNREL(BaseMeasurementGen):
    def __init__(self, *args, **kwargs):
        super().__init__()
        filename="../datasets/nrel_atlanta_rc/arc_sorted_by_person/final_data.csv.bz2"
        self.n1=1
        self.n2=7
        ## collect the data
        self.data=pd.read_csv(filename)
        self.data=self.data.groupby("ids").apply(lambda x: pd.DataFrame(data={
            "value":[np.array(pd.DataFrame(x)["value_delta"])],
            # "value_int":[np.array(pd.DataFrame(x)["value_delta_int"])],
            "cost":[np.array(pd.DataFrame(x)["cost_lin_distance"])],
            # "cost_int":[np.array(pd.DataFrame(x)["cost_int"])],
            "l":pd.DataFrame(x).shape[0]}))
        ## reshape the data
        self.n=kwargs["N"]
        if self.data.shape[0]<self.n:
            print("warning not enough data for "+str(self.n)+" agents, setting it to "+str(self.data.shape[0]))
            self.n=self.data.shape[0]
        # if T is not None:
        #     self.t=T
        #     tmp=self.data[self.data["l"]>=self.t] # subset of records that have enough data points
        #     if tmp.shape[0]>self.N:
        #         self.data=tmp
        #         del tmp
        #     else:
        #         print("warning not enough data satisfies the criterion T="+str(self.t)+" ignoring it")
        ## partition data into N sets, one for each user
        lens=[(i,v[1]["l"]) for i,v in enumerate(self.data.iterrows())]
        np.random.shuffle(lens)
        data_lists=partition_list(lens,self.n)
        self.t=kwargs["T"]
        if self.t is None:
            self.t=min([sum([v for k,v in j]) for j in data_lists]) # the shortest lenght of all data streams
            print("simulation length: "+str(self.t)+" for "+str(self.n)+" agents")
        self.data_streams=[self.data.ix[[k for k,v in a]].drop("l",axis=1) for a in data_lists]
        # concat all data
        self.data_streams=[d.apply(lambda x: list(itertools.chain.from_iterable(x)),axis=0) for d in self.data_streams]
        # set up initial repetition
        # if longs.shape[0]<n:               # not enough user records, combine short records into long ones
        #     shorts=self.data[self.data["l"]<self.t] # subset of records that have enough data points
        #     lens=[(i,v[1]["l"]) for i,v in enumerate(shorts.iterrows())]
        #     pairs=addsubsets(lens,t)
        #     ret=[]
        #     for p,_ in pairs:
        #         df=shorts.ix[p]
        #         ret.append(pd.DataFrame(data={"value":[list(itertools.chain.from_iterable(df["value"]))],"raw":[list(itertools.chain.from_iterable(df["raw"]))],"cost":[list(itertools.chain.from_iterable(df["cost"]))],"l":sum(df["l"])}))
        # self.data=pd.concat([longs]+ret).reset_index().drop("index",axis=1)
        # print(self.data)
        self.values=[iter(d["value"]) for d in self.data_streams]
        # self.values_int=[iter(d["value_int"]) for d in self.data_streams]
        self.costs=[iter(d["cost"]) for d in self.data_streams]
        # self.costs_int=[iter(d["cost_int"]) for d in self.data_streams]

    def get_measurements(self,population,timestep):
        """
        Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
        """
        if timestep<=self.t:
            try:
                vals=[next(i) for i in self.values]
                costs=[next(i) for i in self.costs]
                # vals_int=[next(i) for i in self.values_int]
                # costs_int=[next(i) for i in self.costs_int]
                # thresh=max(1,int(sum(vals)*np.random.uniform(0,1)))
                thresh=len(population)*0.8 #np.random.randint(1,5)
                assert(thresh<=sum(vals))
                if thresh>sum(vals):
                    print("Warning, threshold is too high")
                ret=[{"value":v,"cost":c,"timestep":timestep,"agentID":i,"threshold":thresh} for i,(v,c) in enumerate(zip(vals,costs))]
                return ret
            except:
                return None
        else:
            return None
