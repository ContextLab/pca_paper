#!/bin/bash -l

# DO NOT MODIFY THIS FILE!
# MODIFY config.py AND create_and_submit_jobs.py AS NEEDED

# SLURM lines begin with "#SBATCH".  to-be-replaced text is sandwiched between angled brackets

# declare a name for this job
#SBATCH --job-name=<config['jobname']>

# specify the number of cores and nodes (estimate 4GB of RAM per core)
#SBATCH -N <config['nnodes']>

#SBATCH -n <config['ppn']>

# set the working directory *of this script* to the directory from which the job was submitted

# set the working directory *of the job* to the specified start directory
cd <config['startdir']>

echo ACTIVATING timecorr VIRTUAL ENVIRONMENT

module load python

source activate pca_env

# run the job
<config['cmd_wrapper']> <job_command> #note: job_command is reserved for the job command; it should not be specified in config.py

source deactivate pca_env

