#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/projects/%u/SpokenEnglish/logs/%j.log
#SBATCH --job-name=spoken_english
#SBATCH --partition=blanca-clearlab2
#SBATCH --account=blanca-clearlab2
#SBATCH --qos=blanca-clearlab2
#SBATCH --mail-type=END,FAIL

export HF_HOME="/projects/$USER/.cache/huggingface"
mkdir -p $HF_HOME

set -uo pipefail

REPO_ROOT="/projects/$USER"
cd "$REPO_ROOT"

module purge
module load anaconda

set +u && conda activate SpokenEnglish && set -u

cd "$REPO_ROOT/SpokenEnglish/python_file"

export LD_PRELOAD=$CONDA_PREFIX/lib/libintel_threading_attr.so:$CONDA_PREFIX/lib/libmkl_rt.so
python3 main.py