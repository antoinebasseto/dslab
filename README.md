# Droplet Tracking

Repository for the 2022 Data Science Lab Project. Section `Prerequisites` describes
how to set up environment. Section `Training` describes how to train a neural network
using the framework. Section `Layout` describes the content and organization of this
repository.


## Prerequisites

### Software

1. Install all packages in `environment.yml` by running
    ```
    conda env create -f environment.yml
   ```

### Datasets

#### TODO

## Training

1. Create a checkpoint.
   * Train it from scratch by running `python train.py <EXPERIMENT_ID>`. This will create
   a checkpoint binary of the model with the highest score during training.
   The checkpoint gets stored to `experiments/<EXPERIMENT_ID>/<RUN>`.

## Layout

(Change if deemed necessary, just a first suggestion)

```
.
├── experiments # stores experiment configuration files 
│   ├── 0097.toml
│   ├── ...
├── models
│   ├── deepRanking.py  # DeepRanking model
│   └── svd.py  # svd base approach
├── notebooks
│   └── #Add notebooks if deemed necessary
├── README.md
├── environment.yml
├── train.py  # runs experiments
└── utils
    ├── dataset.py  # Create datasets 
    ├── models_common.py  # shared training code for models (if any)
    └── processing.py  # various utility functions
```