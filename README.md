# Droplet Tracking

Repository for the 2022 Data Science Lab Project. Section `Prerequisites` describes
how to set up environment. Section `Training` describes how to train a neural network
using the framework. Section `Layout` describes the content and organization of this
repository.


## Prerequisites

### Software

1. At the moment, a basic environment is given in `utils/env_prefixless.yml` and can be installed by running
    ```
    conda env create -f env_prefixless.yml
   ```
   As far as I understand, this will install the environment in the default conda/miniconda envs directory.
   To install the env in a specific location, use 
   ```
    conda env create --prefix <path to desired directory> -f env_prefixless.yml
   ```

   If you are on MacOS and using homebrew, and have installed miniconda over homebrew, you probably want / need to change the prefix to point to the miniconda directory inside the homebrew filespace. To that end, you may want to try and use `utils/env_brewprefix.yml` which already contains a prefix that should point to the correct location within the homebrew system so you do not have to pass a `--prefix` option when creating the conda env. If you use conda inside homebrew on a mac, you will also probably need to execute python via the `/opt/homebrew/Caskroom/miniconda/base/envs/<env name here (should be "dsl" if everything goes right)>/bin/python3` command if you do not change the default python path (which is not recommeneded as python is already installed on the mac and the system depends on that installation).

### Images

1. The nd2reader library does not work on all images. Instead, we use the nd2 library.  To that end, `get_image_as_ndarray` in `utils/raw_image_reader.py` can be used to extract frames and channels as numpy arrays from the nd2 image.

### Datasets

1. Currently, we have 3 images (small movement 1, 2, and 3) for which we have detected droplets and cells. The datasets are in `utils/droplets_and_cells/finished_outputs`. In there you will find the csv tables for both droplets and cells for all three images mentioned above. Additionally, you can also find the legacy dataset for the image "small movement 1" which has a `legacy` at the end of its name.
To create a "droplet dataset" from these csv files, one needs to use the according nd2 image (that matches the droplet csv file) and the function `create_dataset` in `utils/droplet_retreiver.py` which should be documented.
This droplet dataset will give you a sort of list of all droplets together with cut out image patches where the droplet is located.

2. The cells dataset can be used to augment the droplet dataset (see point 3.) as it allows to read off the intensity and persistency scores (and also the locations) of all the signal-spikes that have been detected in the various droplets, which can then be used to improve the accuracy of similarity scores.
The "intensity score" of the cells is related to the "height" of the spike in the DAPI channel, relative to the background noise.
In detail, the score measures how high the intensity of the peak is, and the units are 10 * standrad deviation of the estimated noise.
On the other hand, the "persistency score" is related to how "wide" the detected peaks are.
The persistence score computes the average distance in pixels from the peak location, to the 10 closest pixels which are categorized as "noise". 
I.e., it is an estimate of the margin between the peak center and the closest point "where the noise begins".
In the legacy dataset, the scores were squashed between 0 and 1 to make it a bit more scale invariant but I found that the loss of precision was significant when we actually want to work with the raw / unsquashed measurements. For this reason, in the new datasets, the scores are not between 0 and 1 anymore, but rather 0 and infinity.
The intensity and persistence scores will in general be reasonably positively correlated but there are cases where one can be big and the other one small and vice versa.
IMPORTANT: The "persistence score" is not available in the legacy dataset. 
IMPORTANT: The scores in the new datasets are not squashed between 0 and 1 and are between 0 and infinity instead. 

3. To get the combined information of the cells and droplets dataset (ie, get additional information about all significant peaks in each droplet), you can use `create_dataset_cell_enhanced` in `utils/droplet_retreiver.py`. This function will give the same output as the `create_dataset` function, except that additonally, in each entry corresponding to a droplet and frame in the returned dataset, there will be an additional column "cell_signals" which is a pandas dataframe (I think) which contains all information about all detected peaks in the corresponding droplet and frame. So basically each entry in the returned dataset contains a column which is again a dataframe which contains significant spikes in the DAPI channel. This nested dataframe has multiple columns telling you various scores about the detected peaks, their locations and their IDs.




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
