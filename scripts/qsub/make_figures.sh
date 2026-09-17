#!/bin/bash
#$ -M bpiguave@nd.edu
#$ -m a
#$ -pe smp 4
#$ -N tr_figures
#$ -o logs/make_figures.$JOB_ID.out
#$ -e logs/make_figures.$JOB_ID.err
#$ -cwd

# Regenerate every figure and generated/numbers.tex for the technical report.
#   cd technical_report && qsub scripts/qsub/make_figures.sh
#   qsub scripts/qsub/make_figures.sh problem ladder   # a subset by stem

module load python
cd /groups/bsavoie2/bpiguave/SiliconePolymers/technical_report || exit 1
source ../rtv_foam_spr/.venv/bin/activate
mkdir -p logs figures generated
export OMP_NUM_THREADS=$NSLOTS OPENBLAS_NUM_THREADS=$NSLOTS MKL_NUM_THREADS=$NSLOTS
python make_figures.py "$@"
