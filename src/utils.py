import pandas as pd
import itertools
import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})
import numpy as np
import math

def boltzmann(qtable,temp):
    probs=[math.exp(q/temp) for q in qtable] # boltzmann equation
    if sum(probs)==0:
        probs=[0.5,0.5]
    else:
        probs=[np.round(p/sum(probs),2) for p in probs] # normalize
    assert(sum(probs)==1)
    return np.cumsum(probs)

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
    return np.mean(costs)

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

def get_stats(log,varname,idx=["timestep"],cols=None):
    """
    Log: a list of dictionaries or data frames
    """
    df=[pd.DataFrame(i[varname]) for i in log]
    return compute_stats(df,idx=idx,columns=cols)

def plot_trend(df,xname,filename,trends=None,yname=None):
    if trends is None:
        trends=[d[:-5] for d in df.columns if ("_mean" in d)]
    fig,ax=plt.subplots()
    ax.set_xlabel(xname)
    if yname is None:
        data=[(df,None)]
    else:
        data=[(df[df[yname]==i],i) for i in df[yname].unique()]
    #fig.suptitle(title)
    #ax.set_ylabel(ylab or str(y))
    # if ylim:
    #     ax.set_ylim(ylim)
    for d,l in data:
        x=d[xname]
    for y in trends:
            lab=(y if l is None else yname+"="+str(l))
            ax.plot(x,d[y+"_mean"],label=lab)
            ax.fill_between(x,np.asarray(d[y+"_mean"])-np.asarray(d[y+"_ci"]),np.asarray(d[y+"_mean"])+np.asarray(d[y+"_ci"]),alpha=0.2)
    fig.legend()
    fig.savefig(filename,format='pdf')
    plt.close(fig)

def plot_measures(df,xname,filename,trends=None):
    fig=plt.figure()
    for measures,ylim,i in [[["efficiency","success","gini"],[0,1],0]
                            ,[["cost","social_welfare","num_contrib"],None,1]]:
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

def plot_hmap(heatmap,title,filename,plot_dir,xlab="compr2",xlab2="",ylab="compr",xlim=None,ylim=None,ticks=None,ticklabs=None,scale_percent=False,ticklabs_2x=None,ticklabs_2y=None,show_contour=False,font_size=16,cmap=None,num_decimals_legend=2,display_text=False,inverty=True):
    fig,ax=plt.subplots()
    fig.suptitle(title,fontsize=font_size)
    masked_array = np.ma.array (heatmap, mask=np.isnan(heatmap))
    if cmap==None:
        cmap = matplotlib.cm.jet
    cmap.set_bad('white',1.)
    if scale_percent:
        plt.imshow(masked_array, interpolation='nearest', cmap=cmap,vmin=0,vmax=1)
        cbar=plt.colorbar()
        t=np.arange(0,1.01,0.2)
        cbar.set_ticks(t)
        cbar.set_ticklabels([str(int(i*100))+"%" for i in t])
        cbar.ax.tick_params(labelsize=font_size)
    else:
        plt.imshow(masked_array, interpolation='nearest', cmap=cmap, aspect='auto')
        vmax=round(np.max(masked_array),num_decimals_legend)
        vmin=round(np.min(masked_array),num_decimals_legend)
        term=False
        step=10**(-num_decimals_legend)
        while not term:
            cbar_ticks=np.arange(vmin,vmax+step,step)
            if len(cbar_ticks)<10:
                term=True
            else:
                step*=2
        # cbar=plt.colorbar(format="%."+str(num_decimals_legend)+"f")
        cbar=plt.colorbar()
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticks)
        cbar.ax.tick_params(labelsize=font_size)
    if show_contour:
        CS = plt.contour(masked_array,colors='k')
        plt.clabel(CS, inline=1, fontsize=font_size)
    ax.set_ylabel(ylab,fontsize=font_size)
    ax.set_xlabel(xlab,fontsize=font_size)
    if inverty:
    plt.gca().invert_yaxis()
    if not ticks==None:
        if len(ticks)==2:
            xticks=ticks[0]
            yticks=ticks[1]
        else:
            xticks=ticks
            yticks=ticks
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
    if not ticklabs==None:
        if len(ticklabs)==2:
            xticks_l=ticklabs[0]
            yticks_l=ticklabs[1]
        else:
            xticks_l=ticklabs
            yticks_l=ticklabs
        ax.set_xticklabels(xticks_l)
        ax.set_yticklabels(yticks_l)
    ax.tick_params(labelsize=font_size)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if not ticklabs_2x==None:
        ax2=ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(xticks)
        ax2.set_xlabel(xlab2,fontsize=font_size)
        ax2.tick_params(labelsize=font_size)
        ax2.set_xticklabels(ticklabs_2x)
    if not ticklabs_2y==None:
        ax2y=ax.twinx()
        ax2y.set_ylim(ax.get_ylim())
        ax2y.set_yticks(yticks)
        plt.setp(ax2y.yaxis.get_majorticklabels(),rotation=-90)
        ax2y.set_yticklabels(ticklabs_2y)
        #ax2y.tick_params(labelsize=16)
    if display_text:
        for j,i in apply(itertools.product,[range(x) for x in heatmap.shape]): # every cell in the matrix
            if (not xlim or (i>=xlim[0] and i<=xlim[1])) and (not ylim or (j>=ylim[0] and j<=ylim[1])):
                ax.text(i,j,round(heatmap[j,i],3), va='center', ha='center',color='y')
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.savefig(os.path.join(plot_dir,filename),format='pdf')
    plt.close(fig)

def plot_qtable_hist(qtab,filename,xcol,ycol,valcol,title):
    xs=qtab[xcol].unique()
    ys=qtab[ycol].unique()
    fig,ax=plt.subplots(len(ys),len(xs),sharex="col",sharey="col")
    if not isinstance(ax,list) and not isinstance(ax,np.ndarray):
        print(ax)
        ax=[[ax]]
    fig.suptitle(title)
    fig.text(0.5, 0.01, xcol, ha='center')
    fig.text(0.01, 0.5, ycol, va='center', rotation='vertical')
    # find common ticks
    ticks=qtab[valcol].unique()
    ticks.sort()
    plt.setp(ax, xticks=ticks,xticklabels=["No"]+[""]*(len(ticks)-2)+["Yes"])
    for i,(c,v) in expandgrid({'c':ys,'v':xs}).iterrows():
        q_state=qtab[(qtab[xcol]==v) & (qtab[ycol]==c)] # subset with current state
        hist=q_state[valcol].value_counts().sort_index()
        assert(sum(hist)==q_state["N"].max()*(q_state["repetition"].max()+1)) # the number of rows corresponds to pop size * repetitions
        hist/=sum(hist)         # normalize
        #a=hist.plot.bar(ax=ax[i//len(xs)][i%len(xs)])                ## todo check that order of plots is correct
        ax[i//len(xs)][i%len(xs)].bar(x=hist.index,height=hist,width=0.1)
        #a.set_title(str(i)+" v:"+str(v)+" c:"+str(c))
        # set labels
        if i%len(xs)==0:
            ax[i//len(xs)][i%len(xs)].set_ylabel(ys[i//len(xs)])
        if i//len(xs)==0:
            ax[i//len(xs)][i%len(xs)].set_title(xs[i%len(xs)])
    fig.tight_layout()
    fig.savefig(filename,format='pdf')

