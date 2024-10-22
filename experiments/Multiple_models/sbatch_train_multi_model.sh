#!/bin/bash

params_rank=("32")
aggregation_method="average"
params_subgraph_size_range=("(0.6, 0.7)" "(0.2, 0.3)")
params_subgraph_amount=("5" "10" "30")
# params_subgraph_amount=("5")
# params_subgraph_amount=("10")
# params_subgraph_amount=("30")

for rank in "${params_rank[@]}"; do
  for subgraph_size_range in "${params_subgraph_size_range[@]}"; do
    for subgraph_amount in "${params_subgraph_amount[@]}"; do

      echo "--rank $rank --aggregation_method $aggregation_method --subgraph_amount $subgraph_amount --subgraph_size_range $subgraph_size_range"
      sbatch train_SEA.sh --rank "$rank" --aggregation_method "$aggregation_method" --subgraph_amount "$subgraph_amount" --subgraph_size_range "$subgraph_size_range"
      sbatch train_TransE_DistMult_ComplEx.sh --rank "$rank" --aggregation_method "$aggregation_method" --subgraph_amount "$subgraph_amount" --subgraph_size_range "$subgraph_size_range"
      sbatch train_TransE_DistMult_ComplEx_AttE.sh --rank "$rank" --aggregation_method "$aggregation_method" --subgraph_amount "$subgraph_amount" --subgraph_size_range "$subgraph_size_range"
#      sbatch train_NELL-995-h100_ComplEx.sh --rank "$rank" --aggregation_method "$aggregation_method" --subgraph_amount "$subgraph_amount" --subgraph_size_range "$subgraph_size_range"
#      sbatch train_NELL-995-h100_AttE.sh --rank "$rank" --aggregation_method "$aggregation_method" --subgraph_amount "$subgraph_amount" --subgraph_size_range "$subgraph_size_range"
#      sbatch train_NELL-995-h100_AttH.sh --rank "$rank" --aggregation_method "$aggregation_method" --subgraph_amount "$subgraph_amount" --subgraph_size_range "$subgraph_size_range"
    done
  done
done
