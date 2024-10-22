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

params_subgraph_amount=("25")
params_sampling_method=("Entity")
params_rho=("0.5" "1" "2")
params_entities_per_step=("1" "2" "4" "6" "8" "10" "15" "20" "25" "30" "40" "50")
#params_entities_per_step=("1")

for entities_per_step in "${params_entities_per_step[@]}"; do
  for subgraph_amount in "${params_subgraph_amount[@]}"; do
    echo
#    echo "$subgraph_amount"
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
            echo "- Entity - $subgraph_amount - $entities_per_step"
            python run_ensemble_embedding.py  --dataset WN18RR \
                                              --subgraph_amount "$subgraph_amount" \
                                              --sampling_method "$sampling_method" \
                                              --entities_per_step "$entities_per_step"\
                                              --no_training \
                                              --sampling_seed "random" \
                                              --subgraph_size_range "(0.6, 0.7)"

#                                                --no_progress_bar
#                                                --no_time_dependent_file_path \
        fi
      done
  done
done

cd experiments/subgraphs || exit 1
