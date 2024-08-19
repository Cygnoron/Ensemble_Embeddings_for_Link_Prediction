#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=01:30:00
#SBATCH --mem=488000
#SBATCH --job-name=Baseline_DistMult_WN18RR
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:1
#SBATCH --chdir /home/hk-project-test-p0021631/st_st162185/Ensemble_Embedding_for_Link_Prediction/experiments/DistMult
#SBATCH --mail-user="st162185@stud.uni-stuttgart.de"
#SBATCH --mail-type=ALL

cd ..
cd ..

source set_env.sh

# Default values
batch_size="450"
rank="32"
reg="0.05"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --batch_size) batch_size="$2"; shift ;;
        --rank) rank="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


if [[ $rank == 500 ]]; then
    reg="0.1"
fi

python run.py \
          --dataset WN18RR \
          --model DistMult \
          --rank "$rank" \
          --regularizer N3 \
          --reg "$reg" \
          --optimizer Adagrad \
          --max_epochs 500 \
          --patience 15 \
          --valid 5 \
          --batch_size "$batch_size" \
          --neg_sample_size -1 \
          --init_size 0.001 \
          --learning_rate 0.1 \
          --gamma 0.0 \
          --bias learn \
          --dtype single \
          --no_progress_bar

cd experiments/DistMult
