#!/bin/bash
#SBATCH -A hk-project-pai00011
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --mem=50000
#SBATCH --job-name=Base FB15K-237
#SBATCH --partition=accelerated-h100
#SBATCH --gres=gpu:4
#SBATCH --chdir /home/hk-project-test-p0021631/st_st162185/Ensemble_Embedding_for_Link_Prediction/experiments/FB15K-237
#SBATCH --mail-user="st162185@stud.uni-stuttgart.de"
#SBATCH --mail-type=ALL

cd ..
cd ..

source set_env.sh

# Model parameters
params_model=("TransE" "DistMult" "ComplEx" "RotatE")
params_reg=(0.0 0.0 0.0 0.0)
params_optimizer=("Adagrad" "Adagrad" "Adagrad" "Adagrad")
params_neg_sample_size=(250 250 250 250)
params_learning_rate=(0.05 0.05 0.05 0.05)
params_bias=("none" "none" "none" "none")

# Parse arguments
rank="500"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --rank) rank="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Determine the number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Loop over the number of GPUs and launch a job on each
for (( i=0; i<$NUM_GPUS; i++ ))
 do

  model=${params_model[$i]}
  reg=${params_reg[$i]}
  optimizer=${params_optimizer[$i]}
  neg_sample_size=${params_neg_sample_size[$i]}
  learning_rate=${params_learning_rate[$i]}
  bias=${params_bias[$i]}

  MODEL_PARAMS=(--dataset FB15K-237 \
          --model "$model" \
          --rank "$rank" \
          --regularizer N3 \
          --reg "$reg" \
          --optimizer "$optimizer" \
          --max_epochs 500 \
          --patience 15 \
          --valid 5 \
          --batch_size 500 \
          --neg_sample_size "$neg_sample_size" \
          --init_size 0.001 \
          --learning_rate "$learning_rate" \
          --gamma 0.0 \
          --bias "$bias" \
          --dtype single \
          --wandb_project "Experiments" \
          --no_progress_bar \
          --only_valid \
          --baseline \
          )

  echo "-----------------------------------"
  echo "CUDA $i -> $model"
  echo "${MODEL_PARAMS[@]}"
  echo "-----------------------------------"
  CUDA_VISIBLE_DEVICES=$i python run_ensemble_embedding.py "${MODEL_PARAMS[@]}" &
done

# Wait for all background processes to finish
wait

cd experiments/FB15K-237 || exit 1

exit 0