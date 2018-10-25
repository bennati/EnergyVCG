import pandas as pd
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})
import numpy as np
import math

def positive_sampling(mu,std):
    'returns a positive value while keeping constant the sampling probability of all positive values'
    ret=-1
    while ret<=0:
        ret=np.random.normal(loc=mu,scale=std)
    return ret

def renormalize(n, range1, range2):
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    ret=(delta2 * (n - range1[0]) / delta1) + range2[0]
    if not hasattr(ret,"__len__"):
        ret=[ret]
    for i,r in enumerate(ret):
        if np.isnan(r):
            ret[i]=range2[0]
        if np.isinf(r):
            ret[i]=range2[1]
        assert(ret[i]>=range2[0] and ret[i] <=range2[1])
    return ret

def boltzmann(qtable,temp):
    probs=[math.exp(q/temp) for q in qtable] # boltzmann equation
    if sum(probs)==0:
        probs=[0.5,0.5]
    else:
        probs=[p/sum(probs) for p in probs] # normalize
    try:
        assert(sum(probs)==1)
        return probs
    except:
        s=sum(probs)
        # print("Probs "+str(probs)+" don't sum to one but to "+str(s))
        if s<1:
            probs[np.argmin(probs)]+=1-s
        else:
            probs[np.argmax(probs)]-=s-1
        return probs

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    if len(iterable)==1:
        return [[iterable[0],iterable[0]]]
    else:
        a, b = itertools.tee(iterable)
        next(b, None)
        return list(zip(a, b))

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

def success(thresh,tot_contrib):
    """
    Returns the value of success for one round
    Args:
    thresh: the needs
    tot_contrib: the sum of contributions
    Returns: either 1 if successful or a fraction corresponding to the needs covered
    """
    assert(thresh>=0)
    return (tot_contrib/thresh) if thresh>tot_contrib else 1

def success_freq(successes):
    """
    Returns the frequency of successes over time
    Args:
    successes: a list of success values
    Returns:
    A float representing the frequence of success
    """
    assert(all([i<=1 for i in successes]))
    return np.mean(list(map(int,successes))) # convert values to integers

def efficiency(thresh,tot_contrib):
    """
    Returns the value of efficiency for one round.
    Similar values of needs and total contributions correspond to high efficiency
    Args:
    thresh: the needs
    tot_contrib: the sum of contributions
    Returns: either the ratio between needs and contributions if successful or 0
    """
    return (np.nan if tot_contrib==0 else ((thresh/tot_contrib) if tot_contrib>=thresh else 0))

def efficiency_mean(efficiencies):
    """
    Returns the mean efficiency over time
    Args:
    efficiencies: a list of efficiency values
    Returns:
    A float representing the mean efficiency (among all successful rounds) or 0 if there were no successful rounds
    """
    assert(all(np.array(efficiencies)<=1))
    vals=[i for i in efficiencies if i>0]
    return 0 if len(vals)==0 else np.mean(vals)

def cost(costs):
    """
    Computes the average cost for the current round
    Args:
    costs: a list of costs, one for each agent
    Returns: the average cost
    """
    return np.nanmean(costs)

def social_welfare(costs,rewards):
    """
    Computes the social welfare for the current round
    Args:
    costs: a list of costs, one for each agent
    rewards: a list of rewards, one for each agent
    Returns: the social welfare
    """
    assert(len(costs)==len(rewards))
    return np.mean(np.array(rewards)-np.array(costs))

def contributions(decisions):
    """
    Computes the ratio of volunteering and free riding
    Args:
    decisions: the actions of agents, 1 for volunteers, 0 for free riders
    Returns:
    The proportion of volunteers
    """
    assert(all(np.logical_or(np.array(decisions)==1,np.array(decisions)==0))) # either 0 or 1
    return np.mean(decisions)

def tot_contributions(decisions):
    assert(all(np.logical_or(np.array(decisions)==1,np.array(decisions)==0))) # either 0 or 1
    return np.sum(decisions)

def compute_stats(data,idx=False,columns=False,drop_count=True):
    """
    Computes statistics (mean,std,confidence interval) for the given columns

    Args:
    data_: a data frame

    Kwargs:
    idx: a list of indexes on which to group, must be a list of valid column names. By default the index of the dataframe is used.
    columns: the columns to aggregate, must be a list of valide column names. By default all columns are considered

    Returns:
A data frame with columns 'X_mean', 'X_std' and 'X_ci' containing the statistics for each column name X in 'columns'
    """
    data_=data.copy()
    assert(not idx or isinstance(idx,list))
    assert(not columns or isinstance(columns,list))
    if isinstance(data_,list):
        data_=pd.concat(data_,copy=False) # join all files
    if not idx:
        idx=data_.index
        idx_c=[]
    else:
        idx_c=idx
    if not columns:
        columns=list(data_.columns[np.invert(data_.columns.isin(idx_c))])
    data_["count"]=1
    aggregations={c:[np.mean,np.std] for c in columns if c in data_._get_numeric_data().columns} # compute mean and std for each column
    aggregations.update({"count":np.sum})                # count samples in every bin
    data_=data_[columns+["count"]+idx_c].groupby(idx,as_index=False).agg(aggregations)
    # flatten hierarchy of col names
    data_.columns=["_".join(col).strip().strip("_") for col in data_.columns.values] # rename
    # compute confidence interval
    for c in columns:
        data_[c+"_ci"]=data_[c+"_std"]*1.96/np.sqrt(data_["count_sum"])
    if drop_count:
        data_.drop("count_sum",1,inplace=True)
    return data_

def plot_trend(df,xname,filename,trends=None):
    if trends is None:
        trends=[d[:-5] for d in df.columns if ("_mean" in d)]
    fig,ax=plt.subplots()
    x=df[xname]
    ax.set_xlabel(xname)
    #fig.suptitle(title)
    #ax.set_ylabel(ylab or str(y))
    # if ylim:
    #     ax.set_ylim(ylim)
    for y in trends:
        ax.plot(x,df[y+"_mean"],label=y)
        ax.fill_between(x,np.asarray(df[y+"_mean"])-np.asarray(df[y+"_ci"]),np.asarray(df[y+"_mean"])+np.asarray(df[y+"_ci"]),alpha=0.2)
    fig.legend()
    fig.savefig(filename,format='pdf')
    plt.close(fig)

def plot_measures(df,xname,filename,trends=None):
    fig=plt.figure()
    for measures,ylim,i in [[["efficiency","success","gini"],[0,1],0]
                            ,[["cost","social_welfare","tot_contrib"],None,1]]:
        ax = fig.add_subplot(121+i)
        x=df[xname]
        ax.set_xlabel(xname)
    #fig.suptitle(title)
    #ax.set_ylabel(ylab or str(y))
    # if ylim:
    #     ax.set_ylim(ylim)
        for y in measures:
            ax.plot(x,df[y+"_mean"],label=y)
            ax.fill_between(x,np.asarray(df[y+"_mean"])-np.asarray(df[y+"_ci"]),np.asarray(df[y+"_mean"])+np.asarray(df[y+"_ci"]),alpha=0.2)
        ax.legend()
    fig.savefig(filename,format='pdf')
    plt.close(fig)

def expandgrid(dct):
    """
    dct: A dictionary where the keys are variables and the values are lists of values for these variables
    Returns: A dataframe where the columns are the variables and the rows contain combinations of values
    """
    return pd.DataFrame(list(itertools.product(*dct.values())),columns=list(dct.keys()))
