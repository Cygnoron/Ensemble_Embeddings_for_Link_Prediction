#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --mem=488000
#SBATCH --job-name=Ensemble_experiment
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:4
#SBATCH --chdir /home/hk-project-test-p0021631/st_st162185/Ensemble_Embedding_for_Link_Prediction/experiments
#SBATCH --mail-user="st162185@stud.uni-stuttgart.de"
#SBATCH --mail-type=ALL

cd ..
source set_env.sh

# lists of parameters
# params_dataset=("WN18RR" "YAGO3-10" "NELL-995")
params_dataset=("WN18RR")
# params_model=('{"TransE":[]}' '{"DistMult":[]}' '{"RotatE":[]}' '{"ComplEx":[]}' '{"AttE":[]}' '{"AttH":[]}')
params_model=('{"ComplEx":[]}')
# params_subgraph_amount=("10" "30" "60")
# params_subgraph_amount=("10")
params_subgraph_amount=("30")
# params_subgraph_amount=("60")
# params_theta=("regular" "reversed" "relation")
params_theta=("regular")

# iterate over datasets
for dataset in "${params_dataset[@]}"; do
    # iterate over models
    for model in "${params_model[@]}"; do
        # iterate over subgraph amount
        for subgraph_amount in "${params_subgraph_amount[@]}"; do
            # iterate over theta
            for theta in "${params_theta[@]}"; do
                # run ensemble embedding
                python run_ensemble_embedding.py --dataset "$dataset" \
                                                 --model "$model" \
                                                 --rank 32 \
                                                 --regularizer N3 \
                                                 --reg 0.05 \
                                                 --optimizer Adagrad \
                                                 --max_epochs 500 \
                                                 --patience 15 \
                                                 --valid 5 \
                                                 --batch_size 750 \
                                                 --neg_sample_size -1 \
                                                 --init_size 0.001 \
                                                 --learning_rate 0.1 \
                                                 --gamma 0.0 \
                                                 --bias none \
                                                 --dtype double \
                                                 --multi_c \
                                                 --subgraph_amount "$subgraph_amount" \
                                                 --sampling_method Entity \
                                                 --aggregation_method average \
                                                 --no_time_dependent_file_path \
                                                 --no_sampling \
                                                 #--no_progress_bar \
                                                 --theta "$theta" \
                                                 --model_dropout_factor 10
            done
        done
    done
done

cd experiments/
