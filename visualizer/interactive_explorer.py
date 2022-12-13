import pandas as pd
import numpy as np
import os
from preprocessing.raw_image_reader import get_image_as_ndarray
from pathlib import Path

from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.path import Path as matplotlibPath
from matplotlib.widgets import LassoSelector

PROJECT_PATH = Path(os.getcwd())
DATA_PATH = Path(PROJECT_PATH / "data")
RESULT_PATH = Path(DATA_PATH/ "05_results")

SAVE_KEY = "c"

def select_trajectories(image_path, results_path, image_name):
    results_df = pd.read_csv(results_path)
    results_df = results_df.replace(0, np.nan)
    results_df = results_df.dropna()
    results_df['discard'] = False

    results_df_x = results_df[[i for i in results_df.columns if str(i).startswith("x")]]
    results_df_y = results_df[[i for i in results_df.columns if str(i).startswith("y")]]

    plt.plot(results_df_y.T, results_df_x.T, color="C0", marker=".", linewidth=1)

    # Load the image and plot each frame
    image = get_image_as_ndarray(None, ["BF"], image_path, allFrames = True, allChannels = False)
    frames = []
    for i, frame in enumerate(image):
        f = plt.imshow(frame[0], cmap="gray", alpha=0.3)
        # By default, only have the first and last frames be visible
        if i != 0 or i != (len(image) - 1):
            f.set_visible(False)

        frames.append(f)


    # Define a callback function that will be called when the user
    # finishes drawing the lasso on the image
    def onselect(verts):
        # The "verts" argument contains the coordinates of the
        # points that the user selected with the lasso
        path = matplotlibPath(verts)
        inside = path.contains_points(results_df[['y1', 'x1']].values)
        results_df['discard'] = results_df['discard'] | pd.Series(inside, index=results_df.index)

        # Set the color of the selected points to red
        plt.plot(results_df_y[results_df["discard"]].T, results_df_x[results_df["discard"]].T, marker='.', color='C1')

        # Redraw the figure
        plt.gcf().canvas.draw_idle()

    # Define a callback function that will be called when the user
    # presses a key
    def onpress(event):
        print(f"Pressed {event.key}")
        if event.key == SAVE_KEY:
            # Get the current date and time
            now = datetime.now()
            # Format the date and time to create a unique filename
            save_path = Path(RESULT_PATH / f"results_{image_name}_{now.strftime('%Y-%m-%d-%H-%M-%S')}.csv")
            print(f"Saving new csv at {save_path}...")
            results_df[results_df['discard'] == False].to_csv(save_path)

        if event.key in [str(i) for i in range(len(frames))]:
            frames[int(event.key)].set_visible(not frames[int(event.key)].get_visible())
            
            # Redraw the figure
            plt.gcf().canvas.draw_idle()

    # Create the lasso selector and connect it to the image
    selector = LassoSelector(plt.gca(), onselect)

    # Listen for keypress events
    plt.gcf().canvas.mpl_connect('key_press_event', onpress)

    # Show the plot
    print("Showing the plot...")
    print("---------- VISUALIZER INSTRUCTIONS ----------")
    print("Select trajectories with the mouse to discard them")
    print(f"Press {SAVE_KEY} to save an updated results csv")
    plt.show()