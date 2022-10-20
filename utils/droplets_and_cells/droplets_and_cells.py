import nd2
import cv2 as cv
import numpy as np
import pandas as pd
import sys, getopt
from tqdm import tqdm

from cell_detector import cell_detector
from manual_circle_hough import manual_circle_hough

def get_droplet_output(bf_image, refine):
    droplet_mask, droplet_circles = manual_circle_hough(bf_image, refine)

def generate_output(input_string, output_string_droplets, output_string_cells, refine, optional_output_directory, optional_output):
    f = nd2.ND2File(input_string)

    nr_frames = f.sizes['T']
    nr_channels = f.sizes['C']

    bf_idx = 0
    dapi_idx = 0

    for c in f.metadata.channels:
        channelname = c.channel.name
        channelidx = c.channel.index
        if (channelname == 'DAPI'):
            dapi_idx = channelidx
        if (channelname == 'BF'):
            bf_idx = channelidx

    droplets = []
    cells_dict = []
    for frame_nr in tqdm(range(nr_frames)):
        dapi_channel = f.asarray()[frame_nr, dapi_idx, :, :]
        bf_channel = f.asarray()[frame_nr, bf_idx, :, :]
        visualization_channel = np.zeros(bf_channel.shape, dtype = np.float32)

        # cv.imshow("test", bf_channel)
        # cv.waitKey(0)

        # print(bf_channel.shape)
        # print(dapi_channel.shape)

        circles_in_frame = manual_circle_hough(bf_channel, refine)

        cells = cell_detector(dapi_channel, circles_in_frame)

        visualization_channel = cv.morphologyEx(cells, cv.MORPH_DILATE, np.ones((3,3)))

        cell_id_counter = 0
        for id, circ in tqdm(enumerate(circles_in_frame)):
            center = np.asarray([circ[0], circ[1]])
            radius = circ[2]
            patch_x = (max(int(center[0]) - radius - 2, 0), min(int(center[0]) + radius + 2, cells.shape[0] - 1))
            patch_y = (max(int(center[1]) - radius - 2, 0), min(int(center[1]) + radius + 2, cells.shape[1] - 1))
            local_cells = cells[patch_x[0]: patch_x[1], patch_y[0]: patch_y[1]]
            local_mask = np.zeros(local_cells.shape)
            center_in_patch = center - np.asarray([max(int(center[0]) - radius - 2, 0), max(int(center[1]) - radius - 2, 0)])
            cv.circle(local_mask, np.flip(center_in_patch), radius, 1.0 , -1)
            local_cells = local_cells * local_mask
            # local_bf = bf_channel[patch_x[0]: patch_x[1], patch_y[0]: patch_y[1]]
            # local_bf = local_bf / local_bf.max()
            # cv.circle(local_bf, np.flip(center_in_patch), radius, 1.0 , 1)
            # cv.imshow("test", local_bf)
            # cv.waitKey(0)
        

            nr_cells_estimated = np.sum(local_cells >= 0.5)
            cv.circle(visualization_channel, np.flip(center), radius, 1.0, 1)
            droplets.append({"droplet_id": id, "frame": frame_nr, "center_row": circ[0], "center_col": circ[1], "radius": circ[2], "nr_cells": nr_cells_estimated})
            cell_coords = np.transpose(np.asarray(np.where(local_cells != 0.0)))
            for coord in cell_coords:
                global_center = coord + np.asarray([max(int(center[0]) - radius - 2, 0), max(int(center[1]) - radius - 2, 0)])
                cells_dict.append({"cell_id": cell_id_counter, "droplet_id": id, "frame": frame_nr, "center_row": global_center[0], "center_col": global_center[1], "intensity_score": local_cells[coord[0], coord[1]]})
                cell_id_counter = cell_id_counter + 1
        if optional_output:
            to_display = np.float32(np.transpose(np.asarray([visualization_channel * 1, (bf_channel - bf_channel.min()) / (bf_channel.max() - bf_channel.min()), 1.0 * (dapi_channel - dapi_channel.min()) / (dapi_channel.max() - dapi_channel.min())]), [1, 2, 0]))
            cv.imwrite(optional_output_directory + 'detection_visualization_frame_' + str(frame_nr) + '.tiff', to_display)
    droplet_df = pd.DataFrame(droplets)
    droplet_df.to_csv(output_string_droplets, index = False)

    cell_df = pd.DataFrame(cells_dict)
    cell_df.to_csv(output_string_cells, index = False)
    
        


def main(argv):
    imgname = ''
    path = ''
    suffix = '.nd2'

    output_id = ''
    output_path = ''

    refine = False
    opt_out = False

    try:
        opts, args = getopt.getopt(argv, "ro", ["imgname=", "imgdir=", "outdir=", "outid="])
    except getopt.GetoptError:
        print('droplets_and_cells.py --imgname <image name> --imgdir <image dir> --outid <output id> --outdir <output dir> -r -o')
        print('--outdir is optional. If not specified, the output is generated at the inputs directory.')
        print('--outid can be used to distinguish different runs of the program over the same images. Will append an id to the name of the output tables.')
        print('-r stands for \'refined\' and will make the droplet detection slower but more accurate via postprocessing. I recommend it.')
        print('-o stands for \'optional\' and will cause the generation of output images that show detected droplets and cells.')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "--imgname" :
            imgname = arg
        elif (opt == "--imgdir"):
            path = arg
        elif (opt == "--outdir"):
            output_path = arg
        elif (opt == "--outid"):
            output_id = '_id' + arg
        elif (opt == "-r"):
            refine = True
        elif (opt == "-o"):
            opt_out = True
    if len(imgname) == 0:
        print('Image Name Missing')
        sys.exit(2)
    if len(path) == 0:
        print('Image Directory Missing')
        sys.exit(2)
    if len(output_path) == 0:
        print('Output Directory Missing')
        output_path = path
        print('Output Directory set to: ' + output_path)
    if len(output_id) == 0:
        print('Output ID Missing')
        print('Output ID Omitted')
    
    complete_image_path = path + imgname + suffix
    complete_output_path_droplets = output_path + imgname + '_droplets' + output_id + '.csv'
    complete_output_path_cells = output_path + imgname + '_cells' + output_id + '.csv'

    print('Input file is ' + complete_image_path)
    print('Output file for droplets is ' + complete_output_path_droplets)
    print('Output file for cells is ' + complete_output_path_cells)

    generate_output(complete_image_path, complete_output_path_droplets, complete_output_path_cells, refine, output_path, opt_out)

if __name__ == "__main__":
   main(sys.argv[1:])