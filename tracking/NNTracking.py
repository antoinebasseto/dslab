import pandas as pd
import numpy as np

NUM_FRAMES = 8
table_path = 'droplets_and_cells/finished_outputs/largeMovement1_droplets.csv'
table = pd.read_csv(table_path)

data = []
for i in range(NUM_FRAMES):
    data.append(table[table['frame'] == i][['center_row','center_col','radius','nr_cells']].values)


def neareast_neighbour(data, prev_frame, current_frame, prev_frame_list):
    current_frame_list = []
    for img1 in prev_frame_list:
        best_sim_score = np.inf
        closest_img = None
        for img2 in range(len(data[current_frame])):
            sim_score = np.linalg.norm(data[prev_frame][img1] - data[current_frame][img2])
            if sim_score < best_sim_score:
                best_sim_score = sim_score
                closest_img = img2

        current_frame_list.append(closest_img)
    return current_frame_list


prev_frame_list = np.arange(len(data[0]))
final_output = data[0][:]
for i in range(NUM_FRAMES-1):
    current_frame_list = neareast_neighbour(data,i,i+1,prev_frame_list)
    final_output = np.hstack((final_output,data[i+1][current_frame_list]))
    prev_frame_list = current_frame_list


fo = pd.DataFrame(final_output, columns = ['x1','y1','r1','nc1',
                                     'x2','y2','r2','nc2',
                                     'x3','y3','r3','nc3',
                                     'x4','y4','r4','nc4',
                                     'x5','y5','r5','nc5',
                                     'x6','y6','r6','nc6',
                                    'x7','y7','r7','nc7',
                                    'x8','y8','r8','nc8',])

fo.to_csv('largeMovement1_NNtracking.csv')