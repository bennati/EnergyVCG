#!/bin/bash

Ns=(5 7 15 20 30 50)
n2s=(2 3 4 6 8 10)
outdir="data"
thresh=0.8
T=5000
reps=20

for f in "Asp" "Asp_nrel" ; do #"Asp_ev"
  bsub -R "rusage[mem=3000]" -W 24:00 -n $reps python simulation"$f".py --outdir $outdir --T $T --reps $reps --Ns "${Ns[@]}" --n2s "${n2s[@]}" --thresh $thresh
done

for f in "Q" "Q_nrel" ; do #"Q_ev"
  bsub -R "rusage[mem=3000]" -W 24:00 -n $reps python simulation"$f".py --outdir $outdir --T $T --reps $reps --Ns "${Ns[@]}" --n2s "${n2s[@]}" --thresh $thresh
done

for f in "Qsup" "Qsup_nrel" ; do #"Qsup_ev"
  bsub -R "rusage[mem=3000]" -W 24:00 -n $reps python simulation"$f".py --outdir $outdir --T $T --reps $reps --Ns "${Ns[@]}" --n2s "${n2s[@]}" --thresh $thresh
done

T=100
for f in "Cen" "Cen_nrel" ; do #"Cen_ev"
  bsub -R "rusage[mem=3000]" -W 24:00 -n $reps python simulation"$f".py --outdir $outdir --T $T --reps $reps --Ns "${Ns[@]}" --n2s "${n2s[@]}" --thresh $thresh
done

for f in "Rand" "Rand_nrel" ; do #"Rand_ev"
  bsub -R "rusage[mem=3000]" -W 24:00 -n $reps python simulation"$f".py --outdir $outdir --T $T --reps $reps --Ns "${Ns[@]}" --n2s "${n2s[@]}" --thresh $thresh
done
