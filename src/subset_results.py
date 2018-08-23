import argparse
import pandas as pd
import numpy as np
import os

def gini(array):
    """Calculate the Gini coefficient of a numpy array.
    https://github.com/oliviaguest/gini/blob/master/gini.py
    based on bottom eq:
    http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    from:
    http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    """
    # All values are treated equally, arrays must be 1d:
    array=np.nan_to_num(array)
    array=np.array(array,dtype=np.float64)
    array = array.flatten()
    if len(array)==0:
        return -1
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

parser = argparse.ArgumentParser(description='reads in parameters')

# Add the arguments for the parser to be passed on the cmd-line
# Defaults could be added default=
parser.add_argument('--indir', metavar='indir', nargs='?',default="data",help='the directory where to read the files')
parser.add_argument('--outdir', metavar='outdir', nargs='?',default="plots",help='the directory where to plot')
args = parser.parse_args()

datadir=args.indir
print("reading from "+str(datadir))
savedir=args.outdir
print("plotting in "+str(savedir))
varnames=["N","n2"]

files=os.listdir("./"+datadir)
if not os.path.exists("./"+savedir):
    os.makedirs("./"+savedir)
for prefix in ["decisions","evaluation","perception","reward"]:
    print("starting with files: "+str(prefix))
    flist=[f for f in files if f.startswith(prefix)]
    data=[]
    ghist=[]
    g=[]
    for f in flist:
        tmp=pd.read_csv(os.path.join("./"+datadir,f)).drop_duplicates()
        if prefix=="decisions":
            ghist.append(tmp.groupby(["agentID"],as_index=False).agg({"contributed":np.sum,"contribution":np.sum,"cost":np.sum,"N":np.mean,"n2":np.mean,"repetition":np.mean})) # sum up all contributions in each simulation (over all timesteps)
        if "timestep" in tmp.columns:
            tmp=tmp[tmp["timestep"]==tmp["timestep"].max()]
            if prefix=="decisions":
                g.append(tmp.groupby(["timestep"],as_index=False).agg({"contributed":gini,"contribution":gini,"cost":gini,"N":np.mean,"n2":np.mean,"repetition":np.mean}))
        data.append(tmp)
    if len(data)>0:
        data=pd.concat(data).drop_duplicates()
        data.to_csv(os.path.join(savedir,prefix+".csv.gz"),index=False,compression='gzip')
    tmp=None; data=None
    if len(ghist)>0:
        ghist=pd.concat(ghist).drop_duplicates()
        stats_gini_hist=ghist.groupby(varnames+["repetition"],as_index=False).agg({"contributed":gini,"contribution":gini,"cost":gini}) # compute gini coefficient across agents
        stats_gini_hist=stats_gini_hist.rename(columns={"contributed":"contributed_hist","contribution":"contribution_hist","cost":"cost_hist"})
        if len(g)>0:
            stats_gini=pd.concat(g).drop_duplicates()
            stats_gini.drop("timestep",axis=1,inplace=True)
            stats_gini_hist=pd.merge(stats_gini_hist,stats_gini,on=varnames+["repetition"])
        stats_gini_hist.to_csv(os.path.join(savedir,"stats_gini_contribs.csv.gz"),index=False,compression='gzip')
    del tmp,data
