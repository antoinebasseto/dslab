import cv2 as cv
import pandas as pd
import numpy as np
import math
from tqdm.auto import tqdm
from preprocessing.raw_image_reader import get_image_as_ndarray
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.path import Path as matplotlibPath
from matplotlib.widgets import LassoSelector
from matplotlib.lines import Line2D
import matplotlib.cm as cm

PROJECT_PATH = Path(os.getcwd())
DATA_PATH = Path(PROJECT_PATH / "data")
RESULT_PATH = Path(DATA_PATH/ "05_results")

SAVE_KEY = "c"
SWAPPING_KEY = "a"
CUTTING_KEY = "w"
CONFIRMATION_KEY = "enter"

# Auxilliary stuff for being able to swap edges
swap_mode = False
selected_for_swapping = []
selected_for_swapping_frames = []
auxilliary_swap_lines_drawn = []
auxilliary_swap_lines= []

# Auxilliary stuff for being able to cut edges
cut_mode = False
selected_for_cutting_trajecotry = None
selected_for_cutting_frame = None
auxilliary_cut_lines_drawn = None

# Global variables needed in order to update the dataset from inside the callback functions
results_df_x = None
results_df_y = None 
results_df = None
all_lines = None

# Global variables to keep track of where the trajectories begin (not trivial for when we have partial trajectories)
trajectory_beginning = None


def select_trajectories(image_path, results_path, image_name):
    # Muste declare to use the global variables
    global results_df_x
    global results_df_y
    global results_df
    global all_lines
    global trajectory_beginning


    # Int32 is a special pandas datatype, which is a integer 32 bit datatype plus it can have value NAN
    results_df = pd.read_csv(results_path, dtype="Int32")

    # There is no real need to drop the NA rows
    # results_df = results_df.replace(0, np.nan)
    # results_df = results_df.dropna()
    results_df['discard'] = True

    results_df_x = results_df[[i for i in results_df.columns if str(i).startswith("x")]].astype(np.float32)
    results_df_y = results_df[[i for i in results_df.columns if str(i).startswith("y")]].astype(np.float32)

    trajectory_beginning = np.zeros((results_df_x.shape[0], 2))

    for index, row in results_df_x.iterrows():
        trajectory_begin_idx = np.argmax(np.logical_not(np.isnan(row.to_numpy())) * 1)
        trajectory_beginning[index, 0] = row.iloc[trajectory_begin_idx]
        trajectory_beginning[index, 1] = results_df_y.iloc[index, trajectory_begin_idx]


    # Store lines for later manipulation
    all_lines = plt.plot(results_df_x.T, results_df_y.T, color="C0", marker=".", linewidth=1, picker=True, pickradius=5)

    # Give all lines information abotu which trajecotry they are
    for idx, line in enumerate(all_lines):
        plt.setp(line, gid = str(idx))

    # Load the image and plot each frame
    image = get_image_as_ndarray(None, ["BF"], image_path, allFrames = True, allChannels = False)
    frames = []
    for i, frame in enumerate(image):
        f = plt.imshow(frame[0], cmap="gray", alpha=0.3)
        # By default, only have the first and last frames be visible
        if i != 0 or i != (len(image) - 1):
            f.set_visible(False)

        frames.append(f)

    # Define callback for clicking on a line (or other thing)
    def pickline(event):
        # Muste declare to use the global variables
        global cut_mode
        global swap_mode
        global selected_for_swapping
        global selected_for_swapping_frames
        global auxilliary_swap_lines_drawn
        global auxilliary_swap_lines 
        global selected_for_cutting_trajecotry
        global selected_for_cutting_frame 
        global auxilliary_cut_lines_drawn     
        # If we clicked a line
        if isinstance(event.artist, Line2D):
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()

            # Get trajectory id and frame 
            prev_frame = int(min(event.ind))
            traj_id = int(plt.getp(thisline, "gid"))
            
            # Check that the line is well defined
            if prev_frame + 1 >= xdata.size or math.isnan(xdata[prev_frame + 1]):
                return

            if swap_mode:
                # If we clicked on teh same line twice, do nothing because we cant swap tracking within one trajectory
                if (len(selected_for_swapping) > 0 and selected_for_swapping[-1] == traj_id):
                    print("Invalid selecton. Latest selected line is part of the same trajectory as the currently selected one.")
                    return

                # Datastrcutures to memorize the last two valid lines we clicked
                auxilliary_swap_lines.append(thisline)
                selected_for_swapping.append(traj_id)
                selected_for_swapping_frames.append(prev_frame)
                auxilliary_swap_lines_drawn.append(plt.plot(xdata[[prev_frame, prev_frame + 1]], ydata[[prev_frame, prev_frame + 1]], color="lime", marker=".", linewidth=1))

                # If we have more than two line selected, discard the oldest selected line
                if (len(selected_for_swapping) > 2):
                    auxilliary_swap_lines_drawn[0][0].remove()
                    auxilliary_swap_lines_drawn = auxilliary_swap_lines_drawn[-2:]
                    auxilliary_swap_lines = auxilliary_swap_lines[-2:]
                    selected_for_swapping = selected_for_swapping[-2:]
                    selected_for_swapping_frames = selected_for_swapping_frames[-2:]
                print("Trajectories registered for swapping: ", selected_for_swapping, " in frames ", selected_for_swapping_frames)

                # We only allow swapping tracking between the same frames (since we are sure about in which frame what happened).
                # Hence, if the operator selects two lines that are in different frames, adjust such that we select two lines from the trajectories selected, which are both
                #  in the last frame selected by the operator
                if (len(selected_for_swapping) == 2 and selected_for_swapping_frames[0] != selected_for_swapping_frames[1]):
                    print("Line segments selected do not belong to same frame. Frame from last click gets enforced.")
                    auxilliary_swap_lines_drawn[0][0].remove()

                    # Check that the other trajectory even exists at the same time
                    if np.isnan(auxilliary_swap_lines[0].get_xdata()[[prev_frame, prev_frame + 1]].to_numpy()).any():
                        # We have a problem because the other selected trajectory does not exist at the same time
                        # We solve this problem by removing this incompatible trajectory
                        print("Frame enforcing not possible. Selections do not overlap in time.")
                        auxilliary_swap_lines_drawn = auxilliary_swap_lines_drawn[-1:]
                        auxilliary_swap_lines = auxilliary_swap_lines[-1:]
                        selected_for_swapping = selected_for_swapping[-1:]
                        selected_for_swapping_frames = selected_for_swapping_frames[-1:]
                    else:
                        auxilliary_swap_lines_drawn[0] = plt.plot(auxilliary_swap_lines[0].get_xdata()[[prev_frame, prev_frame + 1]], auxilliary_swap_lines[0].get_ydata()[[prev_frame, prev_frame + 1]], color="lime", marker=".", linewidth=1)
                        selected_for_swapping_frames[0] = selected_for_swapping_frames[1]
                        print("Trajectories registered for swapping: ", selected_for_swapping, " in frames ", selected_for_swapping_frames)
                # If we have two lines selected, we are ready to swap
                if (len(selected_for_swapping) == 2):
                    print("Ready to swap. Press enter to confirm swap.")
                plt.gcf().canvas.draw_idle()
            if cut_mode:
                # If we already have a line selected, de-select that one
                if selected_for_cutting_trajecotry is not None:
                    auxilliary_cut_lines_drawn[0].remove()
                
                # Select new line
                selected_for_cutting_trajecotry = traj_id
                selected_for_cutting_frame = prev_frame
                auxilliary_cut_lines_drawn = plt.plot(xdata[[prev_frame, prev_frame + 1]], ydata[[prev_frame, prev_frame + 1]], color="r", marker=".", linewidth=1)
                print("Trajectory registered for cutting: ", selected_for_cutting_trajecotry, " in frame ", selected_for_cutting_frame)
                print("Ready to cut. Press enter to confirm cut.")


    # Define a callback function that will be called when the user
    # finishes drawing the lasso on the image
    def onselect(verts):
        global all_lines
        global results_df_x
        global results_df
        global trajectory_beginning
        # The "verts" argument contains the coordinates of the
        # points that the user selected with the lasso
        path = matplotlibPath(verts)

        
        # Check if trajectory beginnings are within selected region
        inside = path.contains_points(trajectory_beginning)


        # Set the color of the selected points to orange
        for idx, value in enumerate(inside):
            if value:
                all_lines[idx].set_color('C1')
        results_df['discard'] = results_df['discard'] & (~pd.Series(inside, index=results_df.index))

        # Redraw the figure
        plt.gcf().canvas.draw_idle()

    # Define a callback function that will be called when the user
    # presses a key
    def onpress(event):
        global swap_mode
        global results_df_x
        global results_df_y
        global results_df
        global all_lines
        global selected_for_swapping
        global selected_for_swapping_frames
        global auxilliary_swap_lines_drawn
        global auxilliary_swap_lines
        global cut_mode
        global selected_for_cutting_trajecotry
        global selected_for_cutting_frame 
        global auxilliary_cut_lines_drawn  
        global trajectory_beginning
        print(f" > Pressed {event.key} < ")
        if event.key == SAVE_KEY:
            # Get the current date and time
            now = datetime.now()
            # Format the date and time to create a unique filename
            save_path = Path(RESULT_PATH / f"results_{image_name}_{now.strftime('%Y-%m-%d-%H-%M-%S')}.csv")
            print(f"Saving new csv at {save_path}...")
            tmp_results_df = results_df.loc[:, results_df.columns != 'discard']
            tmp_results_df[results_df['discard'] == False].to_csv(save_path, index = False)
        if event.key == SWAPPING_KEY:
            # Toggle swap mode
            swap_mode = not swap_mode
            if swap_mode:
                cut_mode = False
                print("Swap mode enabled")
            else:
                print("Swap mode disabled")
        if event.key == CUTTING_KEY:
            # Toggle cutting mode
            cut_mode = not cut_mode
            if cut_mode:
                swap_mode = False
                print("Cut mode enabled")
            else:
                print("Cut mode disabled")

        if event.key in [str(i) for i in range(len(frames))]:
            frames[int(event.key)].set_visible(not frames[int(event.key)].get_visible())
            
            # Redraw the figure
            plt.gcf().canvas.draw_idle()

        # Confirm swapping (or other future tool)
        if event.key == CONFIRMATION_KEY:
            # If swap tool is currently selected
            if swap_mode:
                # If we are able to swap
                if len(selected_for_swapping) == 2:
                    print("Swapping...")
                    # Swap lines in results_df and then recompute results_df_x and results_df_y
                    tmp = results_df.iloc[selected_for_swapping[0], 1 + 2 * (selected_for_swapping_frames[0] + 1): -1].copy()
                    results_df.iloc[selected_for_swapping[0], 1 + 2 * (selected_for_swapping_frames[0] + 1): -1] = results_df.iloc[selected_for_swapping[1], 1 + 2 * (selected_for_swapping_frames[0] + 1): -1].copy()
                    results_df.iloc[selected_for_swapping[1], 1 + 2 * (selected_for_swapping_frames[0] + 1): -1] = tmp
                    results_df_x = results_df[[i for i in results_df.columns if str(i).startswith("x")]].astype(np.float32)
                    results_df_y = results_df[[i for i in results_df.columns if str(i).startswith("y")]].astype(np.float32)

                    # Adjust the drawings by udating teh relevant data
                    for swapped_idx in selected_for_swapping:
                        all_lines[swapped_idx].set_xdata(results_df_x.iloc[swapped_idx, :].to_numpy().T)
                        all_lines[swapped_idx].set_ydata(results_df_y.iloc[swapped_idx, :].to_numpy().T)
                    # Toggle off Swap mode for safety and clean up swapping datastructures
                    swap_mode = False
                    print("Swap mode toggled off for safety")
                    selected_for_swapping = []
                    selected_for_swapping_frames = []
                    for l in auxilliary_swap_lines_drawn:
                        l[0].remove()
                    auxilliary_swap_lines_drawn = []
                    auxilliary_swap_lines= []
                    plt.gcf().canvas.draw()
                    plt.gcf().canvas.flush_events()
                else:
                    print("Unable to swap. Missing selections.")
            if cut_mode:
                if selected_for_cutting_trajecotry is None:
                    print("Unable to cut. Missing selection.")
                    return
                print("Cutting...")
                # Cut lines in results_df and then recompute results_df_x and results_df_y
                tmp = results_df.iloc[selected_for_cutting_trajecotry, :].copy()
                tmp[1: 1 + 2 * (selected_for_cutting_frame + 1)] = pd.NA
                results_df.iloc[selected_for_cutting_trajecotry, 1 + 2 * (selected_for_cutting_frame + 1): -1] = pd.NA
                results_df.loc[len(results_df.index)] = tmp
                results_df.iloc[-1, 0] = results_df.iloc[-2, 0] + 1
                results_df_x = results_df[[i for i in results_df.columns if str(i).startswith("x")]].astype(np.float32)
                results_df_y = results_df[[i for i in results_df.columns if str(i).startswith("y")]].astype(np.float32)

                # Update trajectory beginnings
                begin = np.asarray([results_df_x.iloc[-1, selected_for_cutting_frame + 1], results_df_y.iloc[-1, selected_for_cutting_frame + 1]])
                trajectory_beginning = np.vstack([trajectory_beginning, begin])

                # Adjust the drawings by udating teh relevant data and drawing a new line
                all_lines[selected_for_cutting_trajecotry].set_xdata(results_df_x.iloc[selected_for_cutting_trajecotry, :].to_numpy().T)
                all_lines[selected_for_cutting_trajecotry].set_ydata(results_df_y.iloc[selected_for_cutting_trajecotry, :].to_numpy().T)
                # print(results_df_y.iloc[selected_for_cutting_trajecotry, :].to_numpy().T)
                new_line = plt.plot(results_df_x.iloc[-1, :], results_df_y.iloc[-1, :], color="C0", marker=".", linewidth=1)
                plt.setp(new_line[0], gid = str(results_df.iloc[-1, 0]))
                # print(new_line)
                # print(all_lines)
                # assert(False)
                all_lines = all_lines + new_line
                cut_mode = False
                print("Cut mode toggled off for safety")
                auxilliary_cut_lines_drawn[0].remove()
                selected_for_cutting_trajecotry = None
                selected_for_cutting_frame = None
                auxilliary_cut_lines_drawn = None
                plt.gcf().canvas.draw()
                plt.gcf().canvas.flush_events()

    # Create the lasso selector and connect it to the image
    selector = LassoSelector(plt.gca(), onselect)

    # Listen for keypress events
    plt.gcf().canvas.mpl_connect('key_press_event', onpress)
    plt.gcf().canvas.mpl_connect('pick_event', pickline)

    # Show the plot
    print("Showing the plot...")
    print("---------- VISUALIZER INSTRUCTIONS ----------")
    print("Click >left mouse< and drag to select regions of droplets to keep. Selection is based on the initial location of the droplet.")
    print(f"Press >{SAVE_KEY}< to save an updated results csv which only contains droplets selected (marked with orange)")
    print(f"Press >{SWAPPING_KEY}< to toggle swap mode which can swap two edges")
    print(f"In swap mode, >left mouse< click on two edges and then press >{CONFIRMATION_KEY}< to confirm the edge swap")
    print(f"Press >{CUTTING_KEY}< to toggle cut mode which can swap two edges")
    print(f"In cut mode, >left mouse< click on two edges and then press >{CONFIRMATION_KEY}< to confirm the edge cut")
    plt.gcf().canvas.manager.set_window_title('Visualizer ' + image_name)
    plt.show()


# # def look_at_image_singlechannel(img, window_name, path, id):
# #     instance_id = 0
# #     tmp_img = combine_into_rgb(None, None, None, img)
# #     while (True):
# #         cv.imshow(window_name, tmp_img)
# #         k = cv.waitKey(0)
# #         if k == ord('d'):
# #             print("Destroying Window ...")
# #             cv.destroyWindow(window_name)
# #             return
# #         elif k == ord('p'):
# #             photoname = window_name + "_" + str(id) + "_" + str(instance_id)
# #             instance_id = instance_id + 1
# #             print("Taking Photo with name: " + photoname + " Stored at location: " + path)
# #             cv.imwrite(path + photoname + ".tiff", tmp_img)
# #         elif k == ord('n'):
# #             print("Jump to next image ...")
# #             return
# #         cv.destroyWindow(window_name)


# This is meant to be a general function for displaying multiple greyscale images as one rgb image with functionality such as color correction and taking photos of the image
# red, green, blue, white are the 4 grescale images you want to display together with the respective color. 
# All except for one of these 4 channels can be "None" in case you dont have 4 grescale images to display at once.
def look_at_image_multichannel(red, green, blue, white, window_name, path, id, settings = None):
    print("\n\nClick 'n' to go to next image")
    print("Click 'd' to destroy this image and go to next image")
    print("Click 'p' to save the current image")
    print("Click '1' to brighten the current image")
    print("Click '2' to darken the current image")
    print("Click 'r' to adjust red channel brightness")
    print("Click 'g' to adjust green channel brightness")
    print("Click 'b' to adjust blue channel brightness")
    instance_id = 0
    tmpred = None
    tmpgreen= None
    tmpblue = None
    tmpwhite = None
    multipliers = np.ones((4,), dtype = np.float32)
    brightness = 1.0
    if settings is not None:
        multipliers = settings[0]
        brightness = settings[1]
    maxes = np.ones((4,))
    if red is not None:
        maxes[0] = red.max()
        tmpred = red * multipliers[0] / maxes[0]
    if green is not None:
        maxes[1] = green.max()
        tmpgreen= green * multipliers[1] / maxes[1]
    if blue is not None:
        maxes[2] = blue.max()
        tmpblue = blue * multipliers[2] / maxes[2]
    if white is not None:
        maxes[3] = white.max()
        tmpwhite = white * multipliers[3] / maxes[3]
    tmp_img = combine_into_rgb(tmpred, tmpgreen, tmpblue, tmpwhite, normalize = False, brightness = brightness)
    while (True):
        cv.imshow(window_name, tmp_img)
        k = cv.waitKey(0)
        if k == ord('d'):
            print("Destroying Window ...")
            cv.destroyWindow(window_name)
            return
        elif k == ord('p'):
            photoname = window_name + " id:" + str(id) + "_" + str(instance_id)
            instance_id = instance_id + 1
            print("Taking Photo with name: " + photoname + " Stored at location: " + path)
            cv.imwrite(path + "/" + photoname + ".tiff", np.uint16(np.clip(tmp_img * (2 ** 16 - 1), 0, 2**16 - 1)))
        elif k == ord('n'):
            print("Jump to next image ...")
            break
        elif k == ord('1'):
            print("Brightening image ...")
            brightness = min(brightness + 0.1, 3.0)
            tmp_img = combine_into_rgb(tmpred, tmpgreen, tmpblue, tmpwhite, normalize = False, brightness = brightness)
        elif k == ord('2'):
            print("Darkening image ...")
            brightness = max(brightness - 0.1, 0.1)
            tmp_img = combine_into_rgb(tmpred, tmpgreen, tmpblue, tmpwhite, normalize = False, brightness = brightness)
        elif k == ord('r'):
            print("Cycling Red Brightness ...")
            multipliers[0] = multipliers[0] - 0.1
            if (multipliers[0] < 0.0):
                multipliers[0] = 1.0
            if red is not None:
                tmpred = red * multipliers[0] / maxes[0]
            tmp_img = combine_into_rgb(tmpred, tmpgreen, tmpblue, tmpwhite, normalize = False, brightness = brightness)
        elif k == ord('g'):
            print("Cycling Green Brightness ...")
            multipliers[1] = multipliers[1] - 0.1
            if (multipliers[1] < 0.0):
                multipliers[1] = 1.0
            if green is not None:
                tmpgreen = green * multipliers[1] / maxes[1]
            tmp_img = combine_into_rgb(tmpred, tmpgreen, tmpblue, tmpwhite, normalize = False, brightness = brightness)
        elif k == ord('b'):
            print("Cycling Blue Brightness ...")
            multipliers[2] = multipliers[2] - 0.1
            if (multipliers[2] < 0.0):
                multipliers[2] = 1.0
            if blue is not None:
                tmpblue = blue * multipliers[2] / maxes[2]
            tmp_img = combine_into_rgb(tmpred, tmpgreen, tmpblue, tmpwhite, normalize = False, brightness = brightness)
        elif k == ord('w'):
            print("Cycling White Brightness ...")
            multipliers[3] = multipliers[3] - 0.1
            if (multipliers[3] < 0.0):
                multipliers[3] = 1.0
            if white is not None:
                tmpwhite = white * multipliers[3] / maxes[3]
            tmp_img = combine_into_rgb(tmpred, tmpgreen, tmpblue, tmpwhite, normalize = False, brightness = brightness)
        cv.destroyWindow(window_name)
    return (multipliers, brightness)


# image_path is the path to the nd2 image in question
# droplets_path is the path to the droplet detection csv table.
# tracking_path is the path to the trackign results. It should contain 4 columns "framePrev", "frameNext", "dropletIdPrev", and "dropletIdNext"
# output_path is th epath where we want to save images of the visualizer
# focus_upper_arg is a 2-tuple of integers which indicates the upper left corner (if the image is seen as  a matrix) of the window we want to focus at
# focus_lower_arg is a 2-tuple of integers which indicates the lower right corner (if the image is seen as  a matrix) of the window we want to focus at
def read_in_results(image_path, droplets_path, tracking_path, output_path, focus_upper_arg = None, focus_lower_arg = None, id = "tmp", store_traj_table = False):


    # If you need to hijack this function in order to display not the default  BF and DAPI images, but rather postprocessed versions, you may need to change this next line.
    # which is where I read in the channels from the nd2 image.
    window_of_interest = get_image_as_ndarray(None, ["BF", "DAPI"], image_path, allFrames = True, allChannels = False)

    focus_upper = focus_upper_arg
    focus_lower = focus_lower_arg

    if focus_upper is not None and focus_lower is not None:
        window_of_interest = window_of_interest[:, :, focus_upper[0]: focus_lower[0], focus_upper[1]: focus_lower[1]]
    elif focus_upper is not None and focus_lower is None:
        focus_lower = (window_of_interest.shape[2], window_of_interest.shape[3])
        window_of_interest = window_of_interest[:, :, focus_upper[0]:, focus_upper[1]:]
    else:
        focus_lower = (window_of_interest.shape[2], window_of_interest.shape[3])
        focus_upper = (0, 0)
    window_of_interest = np.float64(window_of_interest)
    print("\nNormalizing the images a bit ...\n")
    for i in tqdm(range(window_of_interest.shape[0])):
        window_of_interest[i, 0] = -window_of_interest[i, 0]
        window_of_interest[i, 0] = (window_of_interest[i, 0] - np.quantile(window_of_interest[i, 0], 0.1)) / (window_of_interest[i, 0].max() - np.quantile(window_of_interest[i, 0], 0.1))
        window_of_interest[i, 1] = (window_of_interest[i, 1] - np.quantile(window_of_interest[i, 1], 0.1)) / (window_of_interest[i, 1].max() - np.quantile(window_of_interest[i, 1], 0.1))
    window_of_interest = np.clip(window_of_interest, 0.0, 1.0)
    numframes = window_of_interest.shape[0]

    droplet_table = pd.read_csv(droplets_path, index_col = False)
    droplet_table = droplet_table[droplet_table["center_row"] < focus_lower[0]]
    droplet_table = droplet_table[droplet_table["center_col"] < focus_lower[1]]
    droplet_table["center_row"] = droplet_table["center_row"] - focus_upper[0]
    droplet_table["center_col"] = droplet_table["center_col"] - focus_upper[1]
    droplet_table = droplet_table[droplet_table["center_row"] >= 0]
    droplet_table = droplet_table[droplet_table["center_col"] >= 0]

    # print(tracking_table)
    # print(droplet_table)

    if 'trajectory_id' not in droplet_table.columns:
        tracking_table = pd.read_csv(tracking_path, index_col = False)
        droplet_table = trajectory_expand_droplets(droplet_table, tracking_table, numframes)
        droplet_table.to_csv(output_path + "/trajectory_augmented_droplets_id:" + id + ".csv", index = False)

    droplet_overlay, droplet_overlay_additional = create_droplet_overlays(droplet_table, numframes, (window_of_interest.shape[2], window_of_interest.shape[3]), overlay_type = "boundingbox")
    droplet_overlay = np.clip(droplet_overlay + droplet_overlay_additional, 0.0, 1.0)

    trajectory_overlay, trajectory_overlay_additional = create_trajectory_overlay(droplet_table, (window_of_interest.shape[2], window_of_interest.shape[3]))
    trajectory_overlay = np.clip(trajectory_overlay + trajectory_overlay_additional, 0.0, 1.0)

    print("\nDisplaying BF + Overlays ...")
    numframes = 0
    settings = None
    for i, fr in enumerate(droplet_overlay):
        numframes = i + 1
        winname = "Overlay Frame " + str(i)
        settings = look_at_image_multichannel(fr, window_of_interest[i, 1], droplet_overlay_additional[i], window_of_interest[i, 0], winname, output_path, id, settings)
        # bf_plus_overlay = combine_into_rgb(fr, window_of_interest[i, 1], None, window_of_interest[i, 0])
        # cv.imshow(winname, bf_plus_overlay)
        # k = cv.waitKey(0)
        # if k == ord('d'):
        #     cv.destroyWindow(winname)
    print("\nDisplaying Trajectories ...")
    look_at_image_multichannel(trajectory_overlay, None, None, window_of_interest[0, 0], "Trajectory Overlay", output_path, id, settings)

def create_trajectory_overlay(droplet_table, dimensions):
    assert('trajectory_id' in droplet_table.columns)
    unique_trajectories = np.unique(droplet_table['trajectory_id'].to_numpy(dtype = np.int32).flatten())
    relevant_data = droplet_table[['frame', 'trajectory_id', 'center_row', 'center_col']].to_numpy(dtype = np.int32)
    ans = np.zeros(dimensions, dtype = np.float32)
    ans_optional = np.zeros(dimensions, dtype = np.float32)
    font = cv.FONT_HERSHEY_PLAIN
    print("\nGenerating trajectory overlay ...\n")
    for traj_id in tqdm(unique_trajectories):
        traj_points = relevant_data[relevant_data[:, 1] == traj_id, :]
        traj_points = np.flip((traj_points[np.argsort(traj_points[:, 0]), :])[:, 2: 4], axis = 1)
        # print(traj_points)
        cv.polylines(ans, np.int32([traj_points]), False, 1.0, 2, cv.LINE_AA)
        text_loc2 = traj_points[0, :] + np.asarray([-20, -10])
        cv.putText(ans_optional, str(traj_id), text_loc2, font, 1, 1.0, 2, cv.LINE_AA)
    return ans, ans_optional

# This function takes in a droplet table and a trackign table and basically combines the two by returning a droplet table with an additional column 'trajectory_id'
# which assigns each droplet in each frame to a trajectory. So droplets in the returned table with the same trajectory_id are effectively thought of being the same droplet.
def trajectory_expand_droplets(droplet_table, tracking_table, numframes):
    droplet_ids = droplet_table['droplet_id'].to_numpy(dtype = np.int32).flatten()
    frames = droplet_table['frame'].to_numpy(dtype = np.int32).flatten()
    trajectories = -np.ones((frames.size,), dtype = np.int32)
    tmp1 = np.argwhere(frames == (numframes - 1)).flatten()
    trajectories[tmp1] = np.arange(tmp1.size)
    trajectory_counter = tmp1.size
    print("\nGenerating trajectory IDs for droplets ...\n")
    for i in tqdm(range(numframes - 2, -1, -1)):
        tmp1 = np.argwhere(frames == i)
        ids = droplet_ids[tmp1].flatten()
        for j, id in tqdm(enumerate(ids)):
            step = tracking_table[(tracking_table["framePrev"] == i) & (tracking_table["frameNext"] == (i + 1)) & (tracking_table["dropletIdPrev"] == id)]["dropletIdNext"].to_numpy(dtype = np.int32).flatten()
            # print(step)
            if step.size == 0:
                trajectories[tmp1[j]] = trajectory_counter
                trajectory_counter = trajectory_counter + 1
            elif step.size == 1:
                match = trajectories[np.logical_and(droplet_ids == step[0], frames == (i + 1))]
                if match.size == 1:
                    trajectories[tmp1[j]] = trajectories[np.logical_and(droplet_ids == step[0], frames == (i + 1))]
                else:
                    trajectories[tmp1[j]] = trajectory_counter
                    trajectory_counter = trajectory_counter + 1
            else:
                assert(False)
    ans = droplet_table.copy()
    ans['trajectory_id'] = trajectories
    return ans
    


# droplet_table is the table of droplets detected
# numframes is the number of frames of the image
# dimensions is a tuple (rows, cols) that specifies how big the overlay should be
def create_droplet_overlays(droplet_table, numframes, dimensions, overlay_type = "boundingbox"):
    assert(overlay_type == "boundingbox")
    ans = np.zeros((numframes, dimensions[0], dimensions[1]), dtype = np.float32)
    ans_cellcounter = np.zeros((numframes, dimensions[0], dimensions[1]), dtype = np.float32)
    font = cv.FONT_HERSHEY_PLAIN
    print("\nGenerating droplet overlay ...\n")
    for fr_num in tqdm(range(numframes)):
        droplets_in_frame = droplet_table[droplet_table["frame"] == fr_num]
        # print(droplets_in_frame)
        for idx, droplet in tqdm(droplets_in_frame.iterrows()):
            center = np.asarray([droplet["center_row"], droplet["center_col"]])
            text_loc = center - droplet["radius"] - 1
            text_loc1 = text_loc + np.asarray([2 * droplet["radius"], 1])
            if 'trajectory_id' in droplets_in_frame.columns:
                traj_id = droplet["trajectory_id"]
                next_droplet = (droplet_table[(droplet_table["frame"] == (fr_num + 1)) & (droplet_table["trajectory_id"] == traj_id)])[["center_row", "center_col"]].to_numpy(dtype = np.int32).flatten()
                if next_droplet.size == 2:
                    cv.line(ans[fr_num], np.flip(center), np.flip(next_droplet), 1.0, 2, cv.LINE_AA)
                textSize = cv.getTextSize(text = str(traj_id), fontFace = font, fontScale = 1, thickness = 2)
                text_loc2 = text_loc + np.asarray([textSize[0][1] + 2, 1])
                cv.putText(ans[fr_num], str(traj_id), np.flip(text_loc2), font, 1, 1.0, 2, cv.LINE_AA)
            cv.putText(ans_cellcounter[fr_num], str(droplet["nr_cells"]), np.flip(text_loc1), font, 1, 1.0, 2, cv.LINE_AA)
            cv.rectangle(ans[fr_num], np.flip(center) - droplet["radius"] - 1, np.flip(center) + droplet["radius"] + 1, 1.0, 1)
    return ans, ans_cellcounter

# Combines red, blue, green and white channels into a single image. Channels can be none if not avaialable.
def combine_into_rgb(red, green, blue, white, normalize = True, brightness = 1.0):
    # assert(normalize == True)
    red_flt32 = 0
    green_flt32 = 0
    blue_flt32 = 0
    white_flt32 = 0
    if normalize:
        if red is not None:
            red_flt32 = np.float32(red / red.max()) * brightness
        if green is not None:
            green_flt32 = np.float32(green / green.max()) * brightness
        if blue is not None:
            blue_flt32 = np.float32(blue / blue.max()) * brightness 
        if white is not None:
            white_flt32 = np.float32(white / white.max()) * brightness
    else:
        if red is not None:
            red_flt32 = np.float32(red) * brightness
        if green is not None:
            green_flt32 = np.float32(green) * brightness
        if blue is not None:
            blue_flt32 = np.float32(blue) * brightness 
        if white is not None:
            white_flt32 = np.float32(white) * brightness
    white_flt32 = np.clip(white_flt32 - red_flt32 - green_flt32 - blue_flt32, 0.0, 1.0)
    combined = np.clip(np.float32(np.transpose(np.asarray([blue_flt32 + white_flt32, green_flt32 + white_flt32, red_flt32 + white_flt32]), [1, 2, 0])), 0.0, 1.0)
    return combined