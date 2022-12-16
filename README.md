# Droplet Tracking

Repository for the 2022 Data Science Lab Project. Section `Prerequisites` describes
how to set up environment. Section `Training` describes how to train a neural network
using the framework. Section `Layout` describes the content and organization of this
repository.


## Prerequisites

### Software

1. PLEASE READ THE WHOLE SECTION BEFORE EXECUTING ANYTHING. At the moment, a basic environment is given in `utils/env_prefixless.yml` and can be installed by running. 
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


### Visualization

1. Use the `visualizer.py` script in the `utils` directory. The function inputs are documented.

2. How to use it: The visualizer requires the path to the nd2 image, the path to the droplets table (which is associated to the nd2 image) and the path to the pairing table (which is associated to the droplet table). With just these inputs, it will compute trajectories and display the brightfield images of the different frames with an overlay showing boundingboxes, number of cells, trajectory IDs and lines displaying the movement to the next frame. lastly, it will also show an image with the stitched-together trajectories over all frames. Without further options, the whole image is processed (can be slow)

3. How to use it 2: Per se, the script will only display the images + overlays, and not actually store them, and it requires user input to cycle through the various images. CHECK THE TERMINAL when executing the script, it tells you what commands to use to cycle through the images. Especially, the key 'p' can be used to take a photo of the currently displayed image, if you whish to do so. The visualizer has functionality for manually adjusting total brightness and the brightness of the different channels.

4. OPTIONS: The script has options to: Specify a region of interest which you want to focus on by specifying upper left and bottom right corner of the window of interest. This can speed up the script. It has the option to change the output path. By default, outputs are written into the results directory. It has the option to pass an id (string), the default id is "tmp". It has the option to additionally store the droplet table, augmented with the trajectory IDs computed from the pairing table. If the droplet table given to the script already has a trajectory_id column, the pairing table is ignored and the trajectory IDs from the droplet table are used. However, typically you will need the pairing table.

5. Example: `visualizer.py "<path to raw images>/smallMovement1.nd2" "droplets_and_cells/finished_outputs/smallMovement1_droplets.csv" --pairingpath "droplets_and_cells/finished_outputs/smallMovement1_pairings_simpleNN.csv" --width 1000 --height 1000 --row 1000 --col 1000 --experimentid "idtest" --returntrajectories True`






### Droplet detection

1. Droplet detection can be performed by executing `utils/droplets_and_cells/droplets_and_cells.py`. 
This function will also detect cells. It will measure peaks in the DAPI channel to find possible cells. Not all significant peaks will however be counted towards the "nr_cells" column in the droplet dataset. 
Only signals that surpass a certain threshold will be counted towards the "nr_cells" columns in the `droplets.csv` dataset. 
However, all detected signifcant peaks will be outputted to the `cells.csv` dataset. 
So combining the "cells" and "droplets" datasets is recommended as they complement each other.

## Training and using Deep Learning Features

1. Analysing regular images
   * In this scenario, no training is needed. Simply running the code via `python main.py` is enough. If embeddings have already been created (if a file called `embeddings_{image_name}.csv` is already present
   in `data/03_features`), then one can pass the flag `-g` in order not to re-generate the embeddings, and speed up the overall process.
   Not using embeddings is also possible, by passing the flag `-e`.

2. Using on another dataset
   * Here two distinct options are possible. If a training dataset is already present (composed of several droplet images), alter the file `experiments/model.toml` and place the training dataset and validation
   datasets paths in, respectively, `train_dataset` and `val_dataset`. If it is not, the model will be trained on the generated droplet images. While training, under `experiments/{image_name}` a new checkpoint
   will be created. After the code finishes running, in order to not retrain the model when analysing newer images, replace the path present on `experiments/0003.toml`, in the line
   checkpoint with the latest present checkpoint. Suppose, for instance that the image name is `smallmovement1`. Then there should be a folder `experiments/smallmovement1/000/_____.pt`. In this scenario,
   one should then replace `checkpoint={previous value}` with `checkpoint=experiments/smallmovement1/000/_____.pt`.
   * In order to train a new model, pass the flag `-t` to `python main.py`.

   
