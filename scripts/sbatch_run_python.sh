#!/bin/bash

# Function to display help
show_help() {
    cat << EOF
Usage: $0 <config_filepath>

Submit evolutionary algorithm job to SLURM batch system.

Arguments:
    config_filepath    Path to the YAML configuration file for the evolutionary algorithm

Options:
    --help, -h        Show this help message and exit

Example:
    $0 examples/config.yaml
    $0 /path/to/my/config.yaml

This script will:
1. Validate the config file exists
2. Submit a SLURM job that runs the evolutionary algorithm
EOF
}

# Check for help flag
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    show_help
    exit 0
fi

# Check if config file argument is provided
if [[ $# -eq 0 ]]; then
    echo "Error: No config file specified."
    echo "Use --help for usage information."
    exit 1
fi

CONFIG_FILEPATH="$1"

# Check if config file exists
if [[ ! -f "$CONFIG_FILEPATH" ]]; then
    echo "Error: Config file '$CONFIG_FILEPATH' does not exist."
    exit 1
fi

# Convert to absolute path
CONFIG_FILEPATH=$(realpath "$CONFIG_FILEPATH")

# Create slurm_logs directory in the parent folder of the config file
CONFIG_DIR=$(dirname "$CONFIG_FILEPATH")
SLURM_LOGS_DIR="$CONFIG_DIR/slurm_logs"
mkdir -p "$SLURM_LOGS_DIR"

# Extract job name from config filepath
JOB_NAME=$(basename "$CONFIG_FILEPATH" .yaml)

echo "Submitting job with config: $CONFIG_FILEPATH"
echo "SLURM logs will be written to: $SLURM_LOGS_DIR"

# Submit the job to SLURM
sbatch <<EOF
#!/bin/bash
#SBATCH -A kt-lab
#SBATCH --partition=preempt
#SBATCH -c 24
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --nodelist=cn-v-[1-9]
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$SLURM_LOGS_DIR/slurm-%j.out
#SBATCH --error=$SLURM_LOGS_DIR/slurm-%j.err
#SBATCH --requeue

echo "===== SLURM Directives ===="
echo "  SLURM_JOB_ID: \$SLURM_JOB_ID"
echo "  SLURM_JOB_NAME: \$SLURM_JOB_NAME"
echo "  SLURM_CPUS_ON_NODE: \$SLURM_CPUS_ON_NODE"
echo "  SLURM_MEM_PER_NODE: \$SLURM_MEM_PER_NODE"
echo "  SLURM_JOB_NUM_NODES: \$SLURM_JOB_NUM_NODES"
echo "  SLURM_NODELIST: \$SLURM_NODELIST"
echo "  SLURM_SUBMIT_DIR: \$SLURM_SUBMIT_DIR"
echo "  SLURM_PARTITION: \$SLURM_JOB_PARTITION"
echo "  SLURM_OUTPUT: \$SBATCH_OUTPUT"
echo "  SLURM_ERROR: \$SBATCH_ERROR"

cd ~/evo-mariners/
echo "===== Running git commands for local ~/evo-mariners ====="
echo "---- GIT LOG ----"
git log --oneline -n 5 --no-color
echo "---- GIT STATUS ----"
git status
echo "---- GIT DIFF ----"
git --no-pager diff --no-color
echo ""

echo "===== Running git commands for moos-ivp code in apptainer ====="
APPTAINER_IMG=~/evo-mariners/apptainer/ubuntu_20.04_ivp_2680_learn.sif

echo "---- cd /home/moos/moos-ivp/ && git log ----"
apptainer exec "\$APPTAINER_IMG" /bin/bash -c "cd /home/moos/moos-ivp/ && git log --oneline -n 5 --no-color"

echo "---- cd /home/moos/moos-ivp-2680/ && git log ----"
apptainer exec "\$APPTAINER_IMG" /bin/bash -c "cd /home/moos/moos-ivp-2680/ && git log --oneline -n 5 --no-color"

echo "---- cd /home/moos/moos-ivp-learn/ && git log ----"
apptainer exec "\$APPTAINER_IMG" /bin/bash -c "cd /home/moos/moos-ivp-learn/ && git log --oneline -n 5 --no-color"

source ~/hpc-share/miniforge/bin/activate
conda activate mariners
echo "===== Conda Environment Info ====="
echo "---- python --version ----"
python --version
echo "---- conda list ----"
conda list
echo "---- conda info ----"
conda info

python simple_evo.py "$CONFIG_FILEPATH"
EOF
