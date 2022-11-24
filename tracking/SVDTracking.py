from droplet_retreiver import create_dataset
from droplet_retreiver import resize_patch
import numpy as np
import cv2 as cv
import pandas as pd
from raw_image_reader import get_image_as_ndarray

image_path = 'droplets_and_cells/raw_images/smallMovement1.nd2'
table_path = 'droplets_and_cells/finished_outputs/smallMovement1_droplets.csv'

dataset = create_dataset([0], ['BF'], image_path, table_path, allFrames = True, allChannels = True)

all_droplets = []

for fr in dataset:
    frame_droplets = []
    for droplet in fr:
        frame_droplets.append(resize_patch(droplet['patch'], 100))
    all_droplets.append(frame_droplets)


def create_svd_data(all_droplets):
    X = []
    for i in range(len(all_droplets)):
        all_droplets_fr = np.array(all_droplets[i])
        X.append(all_droplets_fr[:,4,:,:].reshape(all_droplets_fr.shape[0],all_droplets_fr.shape[2]*all_droplets_fr.shape[3]))
    return X

X = create_svd_data(all_droplets)

X_01 = np.vstack((X[0],X[1]))


def pca(X, n_pc):
    M = np.mean(X, axis=0)
    Y = X - M
    U, S, V = np.linalg.svd(Y)
    C = V[:n_pc]
    P = U[:, :n_pc] * S[:n_pc]

    weights = np.dot(Y, C.T)

    return weights


fr_01 = pca(X_01,20)
fr1 = fr_01[:X[0].shape[0]]
fr2 = fr_01[X[0].shape[0]:]

classifier_table = []
for img1 in range(len(all_droplets[0])):
    best_sim_score = np.inf
    closest_img = None
    for img2 in range(len(all_droplets[1])):
        sim_score = np.linalg.norm(fr1[img1] - fr2[img2])
        if sim_score < best_sim_score:
            best_sim_score = sim_score
            closest_img = img2
    classifier_table.append(
        [img1, dataset[0][img1]['center_row'], dataset[0][img1]['center_col'], dataset[0][img1]['radius'],
         dataset[0][img1]['nr_cells'],
         dataset[1][closest_img]['center_row'], dataset[1][closest_img]['center_col'],
         dataset[1][closest_img]['radius'], dataset[1][closest_img]['nr_cells']])

fo = pd.DataFrame(classifier_table, columns = ['drop_id','x1','y1','r1','nc1','x2','y2','r2','nc2'])

fo.to_csv('smallMovement1_SVDtracking.csv')


