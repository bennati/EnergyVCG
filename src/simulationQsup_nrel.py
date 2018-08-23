from simulation import *
import argparse

def run_experiment_par(test,conf,datadir):
    print("starting "+str(test))
    if not os.path.exists(os.path.join(datadir,test)):
        os.makedirs(os.path.join(datadir,test))
    part_fun=functools.partial(body,conf=conf,test=test,datadir=datadir,alpha=conf["A"],gamma=conf["G"])
    if __name__ == '__main__':
        print("starting processes")
        pool=Pool()
        ans=pool.map(part_fun,range(conf["reps"]))
    else:
        ans=map(part_fun,range(conf["reps"]))

parser = argparse.ArgumentParser(description='reads in parameters')

# Add the arguments for the parser to be passed on the cmd-line
# Defaults could be added default=
parser.add_argument('--outdir', metavar='outdir', nargs='?',default="data",help='the directory where to save the results')
parser.add_argument('--Ns', metavar='Ns',nargs='+',required=True,type=int,help='The pop sizes')
parser.add_argument('--n2s', metavar='n2s',nargs='+',required=True,type=int,help='The space sizes')
parser.add_argument('--T', metavar='T',nargs='?',default=5000,type=int,help='The sim len')
parser.add_argument('--reps', metavar='reps',nargs='?',default=20,type=int,help='The num of reps')
parser.add_argument('--thresh', metavar='thresh',nargs='?',default=0.8,type=float,help='The thresh')
args = parser.parse_args()

if __name__ == '__main__':
    tests={"nrel_Qsup":{"T":args.T,"A":0.001,"G":0.0,"reps":args.reps,"params":{"N":args.Ns,"n1":[0],"n2":args.n2s,"thresh":[args.thresh]}
                        ,"meas_fct":MeasurementGenNREL,"dec_fct_sup":DecisionLogicSupervisorDQ,"dec_fct":DecisionLogicEmpty,"rew_fct":RewardLogicUniform}}
    for test,conf in tests.items():
        run_experiment_par(test,conf,args.outdir)
