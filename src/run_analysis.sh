#!/bin/bash

outdir="data_newthresh"

for f in "unif_asp" "nrel_asp" ; do #"ev_asp"
    bsub -n 1 -W 24:00 -R "rusage[mem=50000]" python subset_results.py --indir ../wtest/$outdir/$f/ --outdir ./$outdir/$f
done

for f in "unif_Q" "nrel_Q" ; do #"ev_Q"
    bsub -n 1 -W 24:00 -R "rusage[mem=50000]" python subset_results.py --indir ../wtest/$outdir/$f/ --outdir ./$outdir/$f
done

for f in "unif_knap" "nrel_knap" ; do #"ev_knap"
    bsub -n 1 -W 24:00 -R "rusage[mem=50000]" python subset_results.py --indir ../wtest/$outdir/$f/ --outdir ./$outdir/$f
done

for f in "unif_mand" "nrel_mand" ; do #"ev_mand"
    bsub -n 1 -W 24:00 -R "rusage[mem=50000]" python subset_results.py --indir ../wtest/$outdir/$f/ --outdir ./$outdir/$f
done

for f in "unif_rand" "nrel_rand" ; do #"ev_rand"
    bsub -n 1 -W 24:00 -R "rusage[mem=50000]" python subset_results.py --indir ../wtest/$outdir/$f/ --outdir ./$outdir/$f
done

# for f in "unif_asp" "unif_Q" "unif_knap" "unif_mand" "unif_rand" ; do
#     bsub -n 1 -W 24:00 -R "rusage[mem=50000]" python subset_results.py --indir ../wtest/$outdir/$f/ --outdir ./$f
# done
