#!/bin/bash

params_rank=("32")
aggregation_method="average"
params_subgraph_size_range=("(0.6, 0.7)" "(0.2, 0.3)")
#params_subgraph_amount=("5" "10" "30")
# params_subgraph_amount=("5")
 params_subgraph_amount=("10")
# params_subgraph_amount=("30")

for rank in "${params_rank[@]}"; do
  for subgraph_size_range in "${params_subgraph_size_range[@]}"; do
    for subgraph_amount in "${params_subgraph_amount[@]}"; do

      echo "--rank $rank --aggregation_method $aggregation_method --subgraph_amount $subgraph_amount --subgraph_size_range $subgraph_size_range"
      sbatch train_WN18RR_TransE.sh --rank "$rank" --aggregation_method "$aggregation_method" --subgraph_amount "$subgraph_amount" --subgraph_size_range "$subgraph_size_range"
      sbatch train_WN18RR_DistMult.sh --rank "$rank" --aggregation_method "$aggregation_method" --subgraph_amount "$subgraph_amount" --subgraph_size_range "$subgraph_size_range"
      sbatch train_WN18RR_RotatE.sh --rank "$rank" --aggregation_method "$aggregation_method" --subgraph_amount "$subgraph_amount" --subgraph_size_range "$subgraph_size_range"
      sbatch train_WN18RR_ComplEx.sh --rank "$rank" --aggregation_method "$aggregation_method" --subgraph_amount "$subgraph_amount" --subgraph_size_range "$subgraph_size_range"
#        sbatch train_WN18RR_AttE.sh --rank "$rank" --sampling_method "$sampling_method" --rho "$rho" --aggregation_method "$aggregation_method" --subgraph_amount "$subgraph_amount" --subgraph_size_range "$subgraph_size_range"
#        sbatch train_WN18RR_AttH.sh --rank "$rank" --sampling_method "$sampling_method" --rho "$rho" --aggregation_method "$aggregation_method" --subgraph_amount "$subgraph_amount" --subgraph_size_range "$subgraph_size_range"
    done
  done
done
