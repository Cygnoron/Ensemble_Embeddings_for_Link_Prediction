#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=488000
#SBATCH --job-name=Create_subgraphs_WN18RR
#SBATCH --partition=cpuonly
#SBATCH --chdir /home/hk-project-test-p0021631/st_st162185/Ensemble_Embedding_for_Link_Prediction/experiments/subgraphs
#SBATCH --mail-user="st162185@stud.uni-stuttgart.de"
#SBATCH --mail-type=ALL

cd ..
cd ..
source set_env.sh

params_subgraph_amount=("10" "30" "60")
params_sampling_method=("Entity" "Feature")
params_rho=("0.5" "1" "2")

for subgraph_amount in "${params_subgraph_amount[@]}"; do
  echo
  echo "$subgraph_amount"
    for sampling_method in "${params_sampling_method[@]}"; do
        if [[ $sampling_method == "Feature" ]]; then
            for rho in "${params_rho[@]}"; do
                echo "- Feature $rho -"
                python run_ensemble_embedding.py  --dataset WN18RR \
                                                  --subgraph_amount "$subgraph_amount" \
                                                  --sampling_method "$sampling_method" \
                                                  --rho "$rho" \
                                                  --no_time_dependent_file_path \
                                                  --no_training \
                                                  --no_progress_bar
            done

        elif [[ $sampling_method != "Feature" ]]; then
            echo "- Entity -"
            python run_ensemble_embedding.py  --dataset WN18RR \
                                              --subgraph_amount "$subgraph_amount" \
                                              --sampling_method "$sampling_method" \
                                              --no_time_dependent_file_path \
                                              --no_training \
                                              --no_progress_bar
        fi

    done
done

cd experiments/subgraphs
