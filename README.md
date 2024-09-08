# Ensemble Approaches for Link Prediction

---

This code is the official PyTorch implementation of Ensemble Approaches for Link Prediction [1].
This implementation lies on the KGEmb framework developed by [2].

## Datasets

---

Download and pre-process the datasets:

<pre>
source datasets/download.sh
python datasets/process.py
</pre>

## Installation

---

<!-- TODO get correct python version and update dependencies -->
First, create a python 3.# environment and install dependencies:
<pre>
virtualenv -p python3.7 hyp_kg_env
source hyp_kg_env/bin/activate
pip install -r requirements.txt
</pre>
Then, set environment variables and activate your environment:
<pre>
source set_env.sh
</pre>

## Usage

---

To train and evaluate a KG embedding model for the link prediction task, use the run_ensemble_embedding.py script:
<pre>
usage: run_ensemble_embedding.py [-h] [--dataset DATASET] [--max_epochs MAX_EPOCHS] [--rank RANK] [--patience PATIENCE] 
                                 [--valid VALID] [--batch_size BATCH_SIZE] [--dtype {single,double}] [--debug] 
                                 [--model MODEL] [--regularizer {N3,F2}] [--reg REG] [--optimizer OPTIMIZER] 
                                 [--neg_sample_size NEG_SAMPLE_SIZE] [--dropout DROPOUT] [--init_size INIT_SIZE] 
                                 [--learning_rate LEARNING_RATE] [--gamma GAMMA] [--bias BIAS] [--double_neg] 
                                 [--multi_c] [--subgraph_amount SUBGRAPH_AMOUNT] 
                                 [--subgraph_size_range SUBGRAPH_SIZE_RANGE] [--sampling_method {Entity,Feature}] 
                                 [--rho RHO] [--random_seed RANDOM_SEED] [--entities_per_step ENTITIES_PER_STEP] 
                                 [--enforcement ENFORCEMENT] [--aggregation_method {max,average}] 
                                 [--model_dropout_factor MODEL_DROPOUT_FACTOR] 
                                 [--logging {critical,error,warning,info,debug,data}]  [--wandb_project WANDB_PROJECT] 
                                 [--only_valid] [--no_sampling] [--no_training] [--no_progress_bar] 
                                 [--no_time_dependent_file_path] [--baseline]

Ensemble Approaches for Link Prediction

options:
  -h, --help            show this help message and exit
  --dataset DATASET     Knowledge Graph dataset
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs to train for
  --rank RANK           Embedding dimension
  --patience PATIENCE   Number of epochs before early stopping
  --valid VALID         Number of epochs before validation
  --batch_size BATCH_SIZE
                        Batch size
  --dtype {single,double}
                        Machine precision
  --debug               Only use 1000 examples for debugging
  --model MODEL         JSON string of the mapping from embedding methods to subgraphs. 
                        - <"subgraph number"> in a mapping sets the specified subgraphs to this method 
                        - 'all' in a mapping sets all subgraphs to this method. This has the same effect as --model <"MODEL_NAME"> 
                        - 'rest' in a mapping allows all unmapped subgraphs to be embedded by this method. 
                           If nothing was specified in the mapping, all subgraphs can be embedded by the given embedding method.
  --regularizer {N3,F2} 
                        Regularizer
  --reg REG             Regularization weight
  --optimizer OPTIMIZER
                        Optimizer
  --neg_sample_size NEG_SAMPLE_SIZE
                        Negative sample size, -1 to not use negative sampling
  --dropout DROPOUT     Dropout rate
  --init_size INIT_SIZE
                        Initial embeddings' scale
  --learning_rate LEARNING_RATE
                        Learning rate
  --gamma GAMMA         Margin for distance-based losses
  --bias BIAS           Bias type (none for no bias)
  --double_neg          Whether to negative sample both head and tail entities
  --multi_c             Multiple curvatures per relation
  --subgraph_amount SUBGRAPH_AMOUNT
                        The amount of subgraphs, that will be present in the ensemble.
  --subgraph_size_range SUBGRAPH_SIZE_RANGE
                        A tuple (min, normal) with the relative subgraph sizes, where 
                        - 'min' is the minimal relative subgraph size that will be used with Feature sampling. 
                        - 'normal' is the normal relative subgraph size, that needs to be reached under normal conditions.
  --sampling_method {Entity,Feature}
                        The sampling method, that will be used.
  --rho RHO             Factor for Feature sampling, which specifies, how many relation names should be present in the subgraph, 
                        which is calculated by the formula 'rho * √|Relation Names|'.
  --random_seed RANDOM_SEED
                        The seed for random processes. Set as 'random' for a random seed.
  --entities_per_step ENTITIES_PER_STEP
                        The amount of entities, that will be selected per sampling step.
  --enforcement ENFORCEMENT
                        Enforcement level for ensuring the inclusion of all entities and relation names.
  --aggregation_method {max,average}
                        The method by which all scores from the ensemble are aggregated.
  --model_dropout_factor MODEL_DROPOUT_FACTOR
                        The multiplier for the first validation loss in order to disregard a model as diverged.
  --logging {critical,error,warning,info,debug,data}
                        Determines the level of logging. 
                        - 'info': Contains information about the progress. 
                        - 'debug': Also contains information about variables, e.g. tensor sizes. 
                        - 'data': Also contains embedding weights and other data from variables, which is printed directly to the log.'
  --wandb_project WANDB_PROJECT
                        Turn on logging of metrics via Weights&Biases and synchronize with the given project name.
  --only_valid          Only calculate metrics on the validation set.
  --no_sampling         Turn off sampling.
  --no_training         Turn off training.
  --no_progress_bar     Turn off all progress bars.
  --no_time_dependent_file_path
                        Turn off specifying the current time in file path, when creating logs and other files.
  --baseline            Start a run using a baseline method and all given parameters.

</pre>


The implementation supports tracking of metrics via Weights&Biases. To use it, use wandb.login() and set LOG_WANDB in
ensemble/Constants.py to True and set your PROJECT_NAME

### Syntax for multiple models and individual hyperparameters

#### Embedding methods

- --model <embedding_method>:  
  The whole ensemble should be embedded by <embedding_method>
- --model "{\"<embedding_method_m>\": [\<options_m>], ... }":  
  The subgraphs specified in \<options_m> will be embedded by <embedding_method_m>
    - \<options>:
        - 0, 1, ... : List of integers, which subgraphs should be embedded by this method. All integers, which
          are larger than there are subgraphs available, are ignored.
        - 'rest': All subgraphs, which are not directly mapped, may be embedded by one of the methods, with the
          'rest' keyword. If no 'rest' is included, non-mapped subgraphs may be embedded by any of the given
          methods.
        - 'all': All subgraphs will be embedded by this method, ignoring all other mappings. Only the first
          occurrence of 'all' is used.

#### Hyperparameters

The hyperparameters "batch_size", "bias", "double_neg", "dropout", "gamma", "init_size", "learning_rate",
"multi_c", "neg_sample_size", "optimizer", "regularizer" and "reg" may be given for each method individually. All
other hyperparameters can only be given for the whole ensemble.
Similar to --model, instead of \<embedding_method_m>, 'rest' will set the hyperparameter to \<value> for all embedding
methods, and 'all' overrides all other specifications given for the hyperparameter. Setting 'all' is equivalent to
directly giving the value. If no 'rest' or 'all' was given
and a method wasn't specified, the first entry will be chosen as value.
- Example "batch_size":
- --batch_size \<integer>:  
  \<integer> is the batch_size for the whole ensemble.
- --batch_size "{\"<embedding_method_m>\": \<integer>, ... }":  
  \<integer> is the batch_size for the embedding method \<embedding_method_m>.

## Examples

---

We provide example scripts with hyperparameters for WN18RR in the examples/ folder. For dimensions 32 and 500, these
models should achieve the following test MRRs:
<!-- TODO create example files and put results here -->

## Process

---

The rough outline of the training and score calculation are as described below.

### Training
<pre>
# initialize unified model with random embeddings
unified_model = random_init(models, aggregation_method)

for epoch:
    for batch:
        
        # calculate attention based on single model theta and embedding 
        attention = calculate_attention(models, unified_model)
        
        # do forward pass on single models and combine predictions and factors according to attention
        combined_predictions, combined_factors = unified_model.forward(batch, models, attention)
        
        # calculate loss based on combined predictions
        unified_loss = calculate_loss(combined_predictions, combined_factors, true_values)
        
        # do backward pass on unified embedding 
        unified_model.backward(unified_loss)
        
        # update single model embeddings based on attention and unified embedding
        for model in models:
            model[embeddings] = model.update_embeddings(unified_model, attention)

</pre>

### Score calculation

<pre>
# get which embedding methods are in the ensemble
methods = get_used_methods(models)

# if only one method is present, skip aggregation 
if len(methods) is 1:
    # calculate scores and targets for single method
    scores, targets = unified_model.calculate_score(method)

    # directly calculate metrics from scores and targets
    metrics = calculate_metrics(scores, targets)

# do aggregation if multiple different methods are used
else:
    # calculate scores based on each methods score function
    for method in methods:
        method_score, method_target = unified_model.calculate_score(method)
    
    # combine scores from different methods according to the aggregation method
    aggregated_scores, aggregated_targets = combine_method_scores(method_scores, method_targets, aggregation_method)
    
    # calculate metrics from combined scores
    metrics = calculate_metrics(aggregated_scores, aggregated_targets)
</pre>

## New models

---

To add a new (complex/hyperbolic/Euclidean) Knowledge Graph embedding model, implement the corresponding query embedding
under models/, e.g.:
<pre>
def get_queries(self, queries):
    head_e = self.entity(queries[:, 0])
    rel_e = self.rel(queries[:, 1])
    lhs_e = ### Do something here ###
    lhs_biases = self.bh(queries[:, 0])

    self.update_theta(queries)

    return lhs_e, lhs_biases
</pre>

## Citation

---

If you use this implementation, please cite the following paper [1]:
<!-- TODO Get reference for citation -->
<pre>
@inproceedings{}
</pre>

## References

---

<!-- TODO Get reference for citation -->
[1] REFERENCE

[2] Chami, Ines, et al. "Low-Dimensional Hyperbolic Knowledge Graph Embeddings." Annual Meeting of the Association for
Computational Linguistics. 2020.









