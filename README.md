# Ensemble_Embedding_for_Link_Prediction

## Usage:

- --model
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
- The hyperparameters "batch_size", "bias", "double_neg", "dropout", "gamma", "init_size", "learning_rate",
  "multi_c", "neg_sample_size", "optimizer", "regularizer" and "reg" may be given for each method individually. All
  other hyperparameters can only be given for the whole ensemble.
  Similar to --model, instead of \<embedding_method_m>, 'rest' will set the hyperparameter to \<value> for all embedding
  methods, and 'all' overrides all other specifications given for the hyperparameter. If no 'rest' or 'all' was given
  and a method wasn't specified, the first entry will be chosen as value.
    - Example "batch_size":
    - --batch_size \<integer>:  
      \<integer> is the batch_size for the whole ensemble.
    - --batch_size "{\"<embedding_method_m>\": \<integer>, ... }":  
      \<integer> is the batch_size for the embedding method \<embedding_method_m>.
      
