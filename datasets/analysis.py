import numpy as np
import pandas as pd
import os
import math
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import scipy.stats as st

def compute_binning(a,m,t=int,slices=None):
    """
    Args:
    a: an array of values
    m: the number of bins

    Kwargs:
    t: the type of bins. By default is int, which means slices are rounded up
    slices: the extremes of the binning intervals, useful to force the same binning on multiple datasets.

    Returns
    (slices: the extremes of the binning intervals
    ,labs: an array of the same shape as 'a' containing the bin number corresponding to each value of 'a'
    ,counts: the number of times each value in 'labs' is repeated
    )
    """
    if slices is None:
        # slices=np.arange(min(a),max(a)+step,step)
        slices=np.linspace(min(a),max(a),m)
    labs=np.digitize(a,slices)
    return slices,labs,pd.DataFrame(labs)[0].value_counts()

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

def compute_conf_interval(a):
    return st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))

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

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def boltzmann(qtable,temp):
    probs=[math.exp(p/temp) for p in qtable] # boltzmann equation
    if sum(probs)==0:
        probs=[0.5]*len(probs)
    else:
        probs=[round(p/sum(probs),3) for p in probs] # normalize
    return probs

def renormalize(n, range1, range2):
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]

def group_list(l,n):
    """
    l -> [[X0],[X0,X1],...[X0,X1,...Xn],[X1,X2,...Xn+1]...[Xlast-n,Xlast-n+1,...Xlast]]
    """
    ret=[]
    iters=itertools.tee(l,n)
    for i in range(1,n):
        ret.append(l[:i])
        for j in range(i):
            next(iters[i],None) # progress iterators
    ret=ret+list(zip(*iters))
    return list(map(list,ret))

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

n2s=[3,4,6,8,9,10]
datadir="./"
dataset=os.path.join(datadir,"nrel_atlanta_rc","arc_sorted_by_person")
# datadir=os.path.join(basedir,d,str(l),"vis")
if not os.path.exists(datadir):
    print("Warning: data cannot be found in dir "+str(datadir))
else:
    # datasets = [item for l in
    #             [[os.path.join(datadir,d,d1) for d1 in os.listdir(d) if os.path.isdir(os.path.join(datadir, d,d1))] # folders containing datafiles
    #             for d in [d for d in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, d))] # folders containing datasets
    #             ]
    #             for item in l]  # individual entries
    # for dataset in datasets:
    filename=os.path.join(dataset,"filtered_data.csv.bz2")
    if not os.path.exists(filename):
        print(filename+" not found")
        tab=[]
        tab2=[]
        for participant in [d for d in os.listdir(dataset) if d.startswith("person")]:
            points=pd.read_csv(os.path.join(dataset,participant,"gps_points.csv"))
            trips=pd.read_csv(os.path.join(dataset,participant,"gps_trips.csv"))
            tab.append(pd.merge(points,trips,on=["sampno","perno","gpstripid"]))
            tab2.append(pd.read_csv(os.path.join(dataset,participant,"household.csv")))
        tab=pd.concat(tab)
        tab2=pd.concat(tab2)
        tab.to_csv(os.path.join(dataset,"complete_data.csv.bz2"),compression="bz2",index=False)
        tab2.to_csv(os.path.join(dataset,"complete_data_households.csv.bz2"),compression="bz2",index=False)
        tab["ids"]=list(map("-".join, zip(tab["sampno"].astype(str),tab["perno"].astype(str),tab["gpstripid"].astype(str))))
        tab=tab[tab["no_transport"]==0]
        tab=tab[tab["onsite"]==0]
        tab.drop(["localid","trip_stages","travel_mode","travmodelist","uni_trav_mode_count","uni_trav_mode","no_transport","workrelated","onsite","looptrip"],axis=1,inplace=True)
        # keep only if enough data
        keepers=tab["ids"].value_counts()
        keepers=keepers[keepers > 100]
        tab_filter=tab[tab["ids"].isin(keepers.index)]
        tab_filter.to_csv(filename,compression="bz2",index=False)
    else:
        print("reading "+filename)
        # tab=pd.read_csv(os.path.join(dataset,"complete_data.csv.bz2"))
        # filter
        tab_filter=pd.read_csv(filename)
    ###  compute values
    ## compute value dependent on max speed
    def value_fct(x,n):
        asd=np.array(x)
        asd=[asd.max()/a for a in asd] # ratio to max value
        asd=[a/sum(asd) for a in asd] # normalize
        asd=[renormalize(a,[min(asd),max(asd)],[1,n+1]) for a in asd]
        # asd=np.round(asd,0) # convert to int
        return asd
    def delta_fct(x,n,w):
        asd=group_list(x,w)
        asd=[asd[0][0] # initial measurement has no previous history to compute a difference with
        ]+[abs(a[-1]-np.mean(a[:-1])) for a in asd[1:]] # compute the difference between current value and average of last w-1
        asd=[a/sum(asd) for a in asd] # normalize
        asd=[renormalize(a,[min(asd),max(asd)],[1,n+1]) for a in asd]
        # asd=np.round(asd,0) # convert to int
        return asd
    def cost_fct(x,n):
        ret=abs(np.array(range(x))-x/2.0) # compute the linear distance from the mean value
        ret=[renormalize(i,[0,max(ret)],[1,n+1]) for i in ret] # renormalize between 0 and n
        # ret=np.round(ret,0) # convert to int
        return ret
    def measure_data(df,cols2keep=["gpsspeed"],w=5,n=5,nd=5,nc=5):
        """
        Args:
        df: a slice of dataframe from groupby

        Kwargs:
        w: the window length for value_delta
        n: the number of possible values for value
        nd: the number of possible values for value_delta
        nc: the number of possible values for cost_lin_distance
        """
        x=pd.DataFrame(df)["gpsspeed"]
        values=value_fct(x,n) # convert speeds to int between 0 and n, funct is applied to data of an individual trip as the service provider cares about each individual trip
        values_delta=delta_fct(x,nd,w) # compute acceleration/deceleration and assign a value proportional to that
        costs=cost_fct(len(x),nc) # compute linear distance from the start and destination, the closer to either points the higher the cost
        data={
            "value":values
            ,"value_int":np.round(values,0)
            ,"value_delta":values_delta
            ,"value_delta_int":np.round(values_delta,0)
            ,"cost_lin_distance":costs
            ,"cost_int":np.round(costs,0)
        }
        for c in cols2keep: # bring over columns from df
            data.update({c:pd.DataFrame(df)[c]})
        return pd.DataFrame(data=data)
    for n2 in n2s:
        filename=os.path.join(dataset,"final_data_n"+str(n2)+".csv.bz2")
        if not os.path.exists(filename):
            print(filename+" not found")
            measures=tab_filter.groupby("ids").apply(lambda x: measure_data(x,cols2keep=["ids","sampno","perno","gpstripid","gpsspeed","max_speed_mph",'distance_miles'],n=n2,nd=n2,nc=n2)) # avg_speed_mph contains wrong values
            measures.to_csv(filename,compression="bz2",index=False)
        else:
            print("reading "+filename)
            measures=pd.read_csv(filename)

        ### do some plotting
        ## plot average trip speed

        avg_speeds=compute_stats(measures,idx=["sampno"],columns=["gpsspeed"])
        trip_lengths=tab_filter["ids"].value_counts().reset_index()
        trip_lengths["sampno"]=trip_lengths["index"].transform(lambda x: x.split("-")[0])
        trip_lengths["perno"]=trip_lengths["index"].transform(lambda x: x.split("-")[1])
        trip_lengths["gpstripid"]=trip_lengths["index"].transform(lambda x: x.split("-")[2])
        avg_trip_lengths=compute_stats(trip_lengths,idx=["sampno"],columns=["ids"])

        speed_distr=compute_stats(measures,idx=["ids"],columns=["gpsspeed"])
        speeds,freq,counts=compute_binning(speed_distr["gpsspeed_mean"],int(max(speed_distr["gpsspeed_mean"]))+1)
        tmp=counts.reset_index().sort_values(by=["index"],axis=0)
        fig,ax=plt.subplots()
        ax.set_xlabel("Speed")
        ax.plot(tmp["index"],tmp[0])
        fig.savefig(os.path.join(dataset,"figures","speed_freqs_n"+str(n2)+".pdf"),format='pdf')
        plt.close(fig)

        miles_distr=measures.groupby("ids").apply(lambda x: pd.DataFrame(x)["distance_miles"].unique()[0])
        miles,freq,miles_counts=compute_binning(miles_distr,int(max(miles_distr)+1))#,0,slices=list(range(20))+list(range(20,100,10))+list(range(100,2000,100)))
        tmp=miles_counts.reset_index().sort_values(by=["index"],axis=0)
        fig,ax=plt.subplots()
        ax.set_xlabel("Miles")
        ax.semilogx(tmp["index"],tmp[0])
        fig.savefig(os.path.join(dataset,"figures","miles_freqs_n"+str(n2)+".pdf"),format='pdf')
        plt.close(fig)

        ## compute heatmap with frequencies of value-cost pairs
        tmp=measures.copy()
        tmp["ones"]=1
        tmp["cost_lin_dist_int"]=tmp["cost_lin_distance"].astype(np.int)
        for v in ["value","value_delta"]:
            tmp[v+"_int"]=tmp[v].astype(np.int)
            tmp1=tmp.groupby([v+"_int","cost_lin_dist_int"]).agg({"ones":np.sum}).reset_index()
            x=tmp1[v+"_int"]
            xbins=list(x.unique())#+[max(x)+1]
            y=tmp1["cost_lin_dist_int"]
            ybins=list(y.unique())#+[max(y)+1]
            heatmap,xlabs,ylabs=np.histogram2d(x,y,bins=[xbins,ybins],weights=tmp1["ones"])
            plot_hmap(heatmap,"Freq of "+v+"/cost pairs",os.path.join(dataset,"figures",v+"_heat_n"+str(n2)+".pdf"),"./",xlab="Cost",ylab=v,ticks=[np.array(xbins[:-1])-1,np.array(ybins[:-1])-1],ticklabs=[xbins[:-1],ybins[:-1]],inverty=False)
