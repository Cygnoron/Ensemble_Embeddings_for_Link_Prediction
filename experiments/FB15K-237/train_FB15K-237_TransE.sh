#!/bin/bash
#SBATCH -A hk-project-pai00011
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --mem=488000
#SBATCH --job-name=TransE_FB15K
#SBATCH --partition=accelerated-h100
#SBATCH --gres=gpu:4
#SBATCH --chdir /home/hk-project-test-p0021631/st_st162185/Ensemble_Embedding_for_Link_Prediction/experiments/FB15K-237
#SBATCH --mail-user="st162185@stud.uni-stuttgart.de"
#SBATCH --mail-type=ALL

cd ..
cd ..

source set_env.sh

# Default values
sampling_method="Entity"
rho="-1"
rank="32"
aggregation_method="average"
subgraph_size_range="(0.2, 0.3)"
subgraph_amount=5

params_rho=("-1" "0.5" "1" "2")

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --rank) rank="$2"; shift ;;
        --aggregation_method) aggregation_method="$2"; shift ;;
        --subgraph_size_range) subgraph_size_range="$2"; shift ;;
        --subgraph_amount) subgraph_amount="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Determine the number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Loop over the number of GPUs and launch a job on each
for (( i=0; i<$NUM_GPUS; i++ ))
 do

  if [[ ${params_rho[$i]} == "-1" ]]; then
    sampling_method="Entity"
  elif [[ ${params_rho[$i]} != "-1" ]]; then
    sampling_method="Feature"
  fi

  rho=${params_rho[$i]}

  if [[ $rank == 32 ]]; then
    MODEL_PARAMS=(--dataset FB15K-237 \
                  --model TransE \
                  --rank "$rank" \
                  --regularizer N3 \
                  --reg 0.0 \
                  --optimizer Adagrad \
                  --max_epochs 500 \
                  --patience 15 \
                  --valid 5 \
                  --batch_size 500 \
                  --neg_sample_size "{\"Unified\": -1, \"rest\": 250}" \
                  --init_size 0.001 \
                  --learning_rate 0.05 \
                  --gamma 0.0 \
                  --bias none \
                  --dtype single \
                  --subgraph_amount "$subgraph_amount" \
                  --subgraph_size_range "$subgraph_size_range" \
                  --sampling_method "$sampling_method" \
                  --rho "$rho" \
                  --aggregation_method "$aggregation_method" \
                  --model_dropout_factor 10 \
                  --wandb_project "Experiments" \
                  --only_valid \
                  --no_progress_bar \
                  --no_sampling \
                  )
  elif [[ $rank == 500 ]]; then
    MODEL_PARAMS=(--dataset FB15K-237 \
                  --model TransE \
                  --rank "$rank" \
                  --regularizer N3 \
                  --reg 0.0 \
                  --optimizer Adagrad \
                  --max_epochs 500 \
                  --patience 15 \
                  --valid 5 \
                  --batch_size 500 \
                  --neg_sample_size "{\"Unified\": -1, \"rest\": 250}" \
                  --init_size 0.001 \
                  --learning_rate 0.05 \
                  --gamma 0.0 \
                  --bias none \
                  --dtype single \
                  --subgraph_amount "$subgraph_amount" \
                  --subgraph_size_range "$subgraph_size_range" \
                  --sampling_method "$sampling_method" \
                  --rho "$rho" \
                  --aggregation_method "$aggregation_method" \
                  --model_dropout_factor 10 \
                  --wandb_project "Experiments" \
                  --only_valid \
                  --no_progress_bar \
                  --no_sampling \
                  )
  fi


  CUDA_VISIBLE_DEVICES=$i python run_ensemble_embedding.py "${MODEL_PARAMS[@]}" &
done

# Wait for all background processes to finish
wait

cd experiments/FB15K-237 || exit 1

exit 0