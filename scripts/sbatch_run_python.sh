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

source ~/hpc-share/miniforge/bin/activate
conda activate mariners

cd ~/evo-mariners/
python simple_evo.py "$CONFIG_FILEPATH"
EOF
