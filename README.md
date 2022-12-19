# Droplet Tracking

Repository for the 2022 Data Science Lab Project. Section `Prerequisites` describes
how to set up environment. Section `Training` describes how to train a neural network
using the framework. Section `Layout` describes the content and organization of this
repository. Section `Utils` describes briefly what the main utlity functions are.


## Prerequisites

### Software

TODO: We need a proper env setup and tutorial on how to excute the code

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

### Datasets

1. The droplet tracking algorithm works only with images that have similar statistics as the images supplied to the group. In particular, the images must be relatively focused and the resolution must be significantly high enough for details within the droplets to be visible. Additionally, the images must be provided in an `.nd2` image format and the images must only contain data about the images of the different channeles across the different frames (just like the images provided to the group).

2. To account for slight changes in the experiments which cause the droplets to have different diameters, one can adjust via the options `--radius_min` and `radius_max` the minimum and maximum radius of the droplets (in pixels) to be detected. The defaults are 12 and 25 respectively. Of course these bounds should not be very tight as due to noise there may be fluctuations in the measured radii. We suggest picking bounds with at least 2 to 3 pixels of slack. The defuault settings work fine for the images given to the group.

3. There is an issue of image size and number of droplets in the image. We do not suggest to use too big images or images with 100s of thousands of droplets. The reason is that this will consume an ungodly amount of memory and simply crash the program or eat up a whole bunch of disk space.
Images of the size that have been provided to the group (ca 4k x 4k pixels and 8k-10k droplets) work (takes about 10-20 minutes or so to do everything and takes up a bit more than 8 GB of RAM) but if it gets bigger than that, it becomes problematic (mainly due to memory issues).
However, it is also not reasonable to use the very big images for another reason which would be, that nobody is going to be able to analyze the large amount of data produced by the algorithm in those cases (100k trajectories etc). Because in the end the output of the algorithm needs to be checked by a human anyways. In the case of small movement a human is still necessary in order to filter out the 1% of trajectories that are wrong while in the case of large movement, a human is necessary to detect which regions in the image contain useful and robust trackings. When we talk about images with 100k droplets, this is just not doable.
The group suggest to use images with approximately 2k droplets and of dimensions 2k x 2k pixels (about 2 times 2 patches taken by the microscope camera).

### Visualization

1. The visualization tool can be executed by calling `visualizer.py` (example given below).

2. The visualizer has some neat features. After executing the vsiualizer, a small window should pop up, like this:

<img align="center" src="./tutorial_images/visualizer_first_image.png" width="300">

If you click on the magnification glass on the bottom left, you can select a region on the image via left click-and-drag and focus on the selected part of the image.
If you click on the home buttoin, you will return to the top-most view of the image.
If you click on the left- and right-arrow buttons on the bottom left, you can go back and forth between "focus-levels" you had selected.
The floppy disc icon allows you to store a screenshot of the currently displayed stuff.
The symbol with the 4 arrows in each direction allows you to pan over the image.
By clicking `f` you can switch between fulscreen mode and normal mode.
The visualizer can be exited by pressing `q`.
By pressing the buttons 0 to 9 one can overlay the brightfield images of the corresponding images of 0 up to 9. (Note that the first frame is frame 0, not frame 1).

3. The visualizer also has some tools with which one can repair faulty trajectories and store a specific selection of droplets. The command line from where the visualizer was executed should also show information on how to use the tools. There are 3 main tools: 

4. Selection tool: If one does a left-click and draws a path/region with the mouse, the visualizer will compute which droplets are inside the region drawn (the first occurrence of the droplet is used, i.e. frame 1) by you and mark those droplets as droplets to "keep" (more on that later) and they will be marked orange. You can keep adding droplets with this region-selection tool anytime. When you are done selecting droplets, you can press 'c' which will take all those droplets marked as "keep" (the orange ones), and create a new csv file with those exclusively. All other droplets will not be present in this new csv file. The command line should give information on where this file is stored and on what its name is. Typically theis file will be stored under `/data/05_results/results_<imagename>_<date>.csv`. Here is an example of how selected droplets appear as orange:

<img align="center" src="./tutorial_images/tmp1.png" width="300">

5. Swap tool: Sometimes the automatic tracking algorithm will get confused between two nearby droplets and mess up the tracking in one specific frame, while the tracking between all other frames is perfectly fine. Here is an example: 

<img align="center" src="./tutorial_images/tmp2.png" width="300">

It is easy for a person to see that here the tracking algorithm just messed up in one single frame and that the movement of the droplets is simply a very slow horizontal movement and that the jump of the two trajectories is clearly wrong and that the jump should not happen. To allow a manual correction of these easy-to-see errors, the visualizer has the "swap tool". The swap tool works like this: First you press `a` to activate the tool (`a` will also again de-activate the tool. Check the command line for teh current status of the tool). One then selects two edges (which must represent droplet movements between the same two frames, e.g. frame 1 and 2) by clicking on them with the mouse. The two selected edges will be marked in green. If you are happy with the selection, you can then press 'enter' to confirm the swap between the two selected edges. The swap will bascially exchange the selected edges. After the swap is confirmed, the tool is deactivated automatically and must be re-activated by pressing `a` again. Here is an example:

Select the edges:

<img align="center" src="./tutorial_images/tmp3.png" width="300">

Press enter to confirm swap and repair the trajectories:

<img align="center" src="./tutorial_images/tmp4.png" width="300">

Of course, swapping is only allowed if the selected edges "happen at the same time", i.e. represnet movement between the same two frames.
If one clicks on more than 2 edges, teh program will simply consider the last two clicked edges as selected edges. If one clicks on two edges that are not "at the same time", the program will assume the last selected edge has the correct time and will adjust the other edge to be at the same time.

6. Cut tool: Sometimes trackings are just simply wrong but perhaps the tracking is wrong only between frame 0 and 1 while for all other frames the tracking is perfect. In such cases it makes sense to just cut the link between frame 0 and 1 for one single droplet, while keeping the links between all other frames intact. This way one can still make use of the correct tracking for the remaining frames without having to discard everything. An example of such a case is:

<img align="center" src="./tutorial_images/tmp5.png" width="300">

where the long line in the center is not possible due to ther droplets being in the way.
This is where the cut tool comes into play. The cut tool is activated by pressing `w`. Then one can select an edge by left click, which will highlight the edge in red.

<img align="center" src="./tutorial_images/tmp6.png" width="300">

By pressing enter one can then confirm the edge to be cut, which will basically cut the trajectory at the selected edge and split the trajectory into two new, disjoint trajectories.

<img align="center" src="./tutorial_images/tmp7.png" width="300">

Afterwards one can use the selection tool described before to select the good trajectory and store it in a table.

7. Example on how to execute the visualizer: `python3 visualizer.py "smallMovement1.nd2" "results_smallmovement1.csv"`. 
Explaination: `python3 visualizer.py` tells the computer to exeucte the program `visualizer.py` by means of the programming language `python3` (or simply Python). 
`"smallMovement1.nd2"` tells the program, which image to display in the overlay.
`"results_smallmovement1.csv"` tells the program in which file the trackings computed by the algorithm are located. 
In particular this means, the visualizer can only be executed after one has executed `main.py` and gotten the results from the algorithm.
Sometimes, the visualizer may bug and mouse clicks do not register or something like that. In this case, simply click outside of the window of the visualizer (such that the program goes "out-of-focus") and then click on it again. It should work after that.


## Training and using Deep Learning Features

1. Analysing regular images
   * In this scenario, no training is needed. Simply running the code via `python main.py` is enough. If embeddings have already been created (if a file called `embeddings_{image_name}.csv` is already present
   in `data/03_features`), then one can pass the flag `-g` in order not to re-generate the embeddings, and speed up the overall process.
   Not using embeddings is also possible, by passing the flag `-ne`.

2. Using on another dataset
   * Here two distinct options are possible. If a training dataset is already present (composed of several droplet images), alter the file `experiments/model.toml` and place the training dataset and validation
   datasets paths in, respectively, `train_dataset` and `val_dataset`. If it is not, the model will be trained on the generated droplet images. While training, under `experiments/{image_name}` a new checkpoint
   will be created. After the code finishes running, in order to not retrain the model when analysing newer images, replace the path present on `experiments/0003.toml`, in the line
   checkpoint with the latest present checkpoint. Suppose, for instance that the image name is `smallmovement1`. Then there should be a folder `experiments/smallmovement1/000/_____.pt`. In this scenario,
   one should then replace `checkpoint={previous value}` with `checkpoint=experiments/smallmovement1/000/_____.pt`.
   * In order to train a new model, pass the flag `-t` to `python main.py`.

   
