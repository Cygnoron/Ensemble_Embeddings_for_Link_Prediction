#!/bin/bash
cd ..
source set_env.sh

params_subgraph_amount=("10" "30" "60")

# iterate over subgraph amount
for subgraph_amount in "${params_subgraph_amount[@]}"; do
    # sample subgraphs for the given subgraph amount
    python run_ensemble_embedding.py  --dataset WN18RR \
                                      --subgraph_amount "$subgraph_amount" \
                                      --sampling_method Entity \
                                      --no_time_dependent_file_path \
                                      --no_training
done

cd experiments/
