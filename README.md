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

1. Currently, we have 3 images (small movement 1, 2, and 3) for which we have detected droplets and cells. The datasets are in `utils/droplets_and_cells/finished_outputs`. In there you will find the csv tables for both droplets and cells for all three images mentioned above. Additionally, you can also find the legacy dataset for the image "small movement 1" which has a `legacy` at the end of its name.
To create a "drolet dataset" from these csv files, one needs to use the according nd2 image (that matches the droplet csv file) and the function `create_dataset` in `utils/droplet_retreiver.py` which should be documented.
This droplet dataset will give you a sort of list of all droplets together with cut out image patches where the droplet is located.

2. The cells dataset can then be used to augment the droplet dataset (has to be done manually right now) as it allows to read off the intensity and persistency scores (and also the locations) of all the signal-spikes that have been detected in the various droplets, which can then be used to improve the accuracy of similarity scores.
The "intensity score" of the cells is related to the "height" of the spike in the DAPI channel, relative to the background noise.
In detail, the score measures how high the intensity of the peak is, and the units are 10 * standrad deviation of the estimated noise.
On the other hand, the "persistency score" is related to how "wide" the detected peaks are.
The persistence score computes the average distance in pixels from the peak location, to the 10 closest pixels which are categorized as "noise". 
I.e., it is an estimate of the margin between the peak center and the closest point "where the noise begins".
In the legacy dataset, the scores were squashed between 0 and 1 to make it a bit more scale invariant but I found that the loss of precision was significant when we actually want to work with
the raw / unsquashed measurements. For this reason, in the new datasets, the scores are not between 0 and 1 anymore, but rather 0 and infinity.
The intensity and persistence scores will in general be reasonably positively correlated but there are cases where one can be big and the other one small and vice versa.
IMPORTANT: The "persistency score" is not available in the legacy dataset. 
IMPORTANT: The scores in the new datasets are not squashed between 0 and 1 and are between 0 and infinity instead. 


### Droplet detection

1. Droplet detection can be performed by executing `utils/droplets_and_cells/droplets_and_cells.py`. 
This function will also detect cells. It will measure peaks in the DAPI channel to find possible cells. Not all significant peaks will however be counted towards the "nr_cells" column in the droplet dataset. 
Only signals that surpass a certain threshold will be counted towards the "nr_cells" columns in the `droplets.csv` dataset. 
However, all detected signifcant peaks will be outputted to the `cells.csv` dataset. 
So combining the "cells" and "droplets" datasets is recommended as they complement each other.

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
