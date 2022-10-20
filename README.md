# Droplet Tracking

Repository for the 2022 Data Science Lab Project. Section `Prerequisites` describes
how to set up environment. Section `Training` describes how to train a neural network
using the framework. Section `Layout` describes the content and organization of this
repository.


## Prerequisites

### Software

1. At the moment, a basic environment is given in `utils/droplets_and_cells/droplets_and_cells_environment.yml` and can be installed by running
    ```
    conda env create -f environment.yml
   ```

### Images

1. The nd2reader library does not work on all images. Instead, we use the nd2 library.  To that end, `get_image_as_ndarray` in `utils/raw_image_reader.py` can be used to extract frames and channels as numpy arrays from the nd2 image.

### Datasets

1. Currently, there is only one image (small movement 1) for which we have detected droplets. The dataset is in `utils/droplets_and_cells/finished_outputs/smallMovement1_droplets_idtest1.csv`. From this csv file which only contains droplet locations etc, a dataset can be created by using the according nd2 image and the function `create_dataset` in `utils/droplet_retreiver.py` which should be documented. Not all free parameters / options of `create_dataset` have been enabled (like slack around the droplet and whether to suppress pixels outside of the droplet).

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
    ├── visualizer.py  # utility function to visualize results
    └── processing.py  # various utility functions
```
