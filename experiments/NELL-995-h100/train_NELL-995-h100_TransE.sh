#!/bin/bash
#SBATCH -A hk-project-test-p0022606
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --mem=488000
#SBATCH --job-name=Ensemble_experiment_NELL-995-h100_TransE
#SBATCH --partition=accelerated-h100
#SBATCH --gres=gpu:1
#SBATCH --chdir /home/hk-project-test-p0021631/st_st162185/Ensemble_Embedding_for_Link_Prediction/experiments/NELL-995-h100
#SBATCH --mail-user="st162185@stud.uni-stuttgart.de"
#SBATCH --mail-type=ALL

cd ..
cd ..

source set_env.sh

# Default values
sampling_method="Entity"
rho="-1"
rank="32"
reg="0.0"
aggregation_method="average"
subgraph_size_range="(0.6, 0.7)"
subgraph_amount=10

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --sampling_method) sampling_method="$2"; shift ;;
        --rho) rho="$2"; shift ;;
        --rank) rank="$2"; shift ;;
        --aggregation_method) aggregation_method="$2"; shift ;;
        --subgraph_size_range) subgraph_size_range="$2"; shift ;;
        --subgraph_amount) subgraph_amount="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


python run_ensemble_embedding.py --dataset NELL-995-h100 \
                                 --model TransE \
                                 --rank "$rank" \
                                 --regularizer N3 \
                                 --reg "$reg" \
                                 --optimizer Adam \
                                 --max_epochs 500 \
                                 --patience 15 \
                                 --valid 5 \
                                 --batch_size 1000 \
                                 --neg_sample_size "{\"Unified\": -1, \"rest\": -1}" \
                                 --init_size 0.001 \
                                 --learning_rate 0.001 \
                                 --gamma 0.0 \
                                 --bias learn \
                                 --dtype double \
                                 --subgraph_amount "$subgraph_amount" \
                                 --subgraph_size_range "$subgraph_size_range" \
                                 --sampling_method "$sampling_method" \
                                 --rho "$rho" \
                                 --aggregation_method "$aggregation_method" \
                                 --model_dropout_factor 10 \
                                 --wandb_project "Experiments" \
                                 --only_valid \
                                 --no_progress_bar \
                                 --no_sampling

cd experiments/NELL-995-h100 || exit 1

exit 0