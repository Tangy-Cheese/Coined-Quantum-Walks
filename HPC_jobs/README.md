# HPC Jobs

These codes were originally used for HPC clusters to utilise parallel processing. This directory contains some of the scripts used to run the source code. The codes can be run locally but make sure some of the paramaters are adjusted sensibly.

## Usage
1. Navigate to the right directory eg 
   ```bash 
    .../HPC_jobs/1D_jobs/
    ```
2. Edit the job scripts to match the desired parameters and cluster settings (nodes, tasks, walltime, etc.). 
3. Make the script executable:
   ```bash
   chmod -rwx job.sh
   ```
4. Submit the job:
   ```bash
   sbatch job.sh
   ```