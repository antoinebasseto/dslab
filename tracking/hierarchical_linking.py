import numpy as np
from pathlib import Path
import pandas as pd
import cv2 as cv
from tqdm.auto import tqdm
import os
from data_creation.droplet_retriever import create_dataset_cell_enhanced, resize_patch
from ortools.graph.python import min_cost_flow
from visualizer.interactive_explorer import trajectory_expand_droplets



def compute_droplet_statistics (droplet_entry):
    # print(droplet_entry)
    ans = {}
    ans['frame'] = droplet_entry['frame']
    ans['droplet_id'] = droplet_entry['droplet_id']
    ans['center_row'] = droplet_entry['center_row']
    ans['center_col'] = droplet_entry['center_col']
    ans['radius'] = droplet_entry['radius']
    ans['nr_cells'] = droplet_entry['nr_cells']
    ans['nr_signals'] = droplet_entry['cell_signals'].shape[0]
    if ans['nr_signals'] > 0:
        tmp_cell_matrix = droplet_entry['cell_signals'].to_numpy()
        cell_loc = tmp_cell_matrix[:, 1: 3] - np.asarray([ans['center_row'], ans['center_col']])[None, :]
        cell_intens = tmp_cell_matrix[:, 3]
        cell_persis = tmp_cell_matrix[:, 4]
        ans['integrated_brightness'] = np.sum(cell_intens * cell_persis)
    else:
        ans['integrated_brightness'] = 0
        # ans['signals_weighted_CoM'] = np.zeros((2,))
        # ans['signals_deviation_from_CoM'] = 0
        # ans['max_sig_strength'] = 0
        # ans['mean_sig_strength'] = 0
        # ans['std_sig_strength'] = 0
        # ans['max_sig_persis'] = 0
        # ans['mean_sig_persis'] = 0
        # ans['std_sig_persis'] = 0
        # ans['max_sig_intens'] = 0
        # ans['mean_sig_intens'] = 0
        # ans['std_sig_intens'] = 0
    x = droplet_entry['patch'].shape[1]
    y = droplet_entry['patch'].shape[2]
    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    patch = cv.resize(droplet_entry['patch'].reshape(x, y, 2), (64, 64), interpolation=cv.INTER_CUBIC).astype(np.uint8)
    hog = cv.HOGDescriptor(_winSize=winSize, _blockSize=blockSize, _blockStride=blockStride, _cellSize=cellSize, _nbins=9)
    vec1 = hog.compute(patch[:, :, :1])
    #vec2 = hog.compute(patch[:, :, 1:2])
    #ans['hog'] = np.concatenate((vec1, vec2), axis=0)
    ans['hog'] = vec1

    return ans

# edge_exists_mask is an n x m matrix where entry edge_exists_mask[i, j] is nonzero (or "True") iff droplet i in previous frame can be matched to droplet i in the next frame.
# cost_matrix is an integer n x m matrix where entry cost_matrix[i, j] is the cost of assigning droplet i in previous frame to droplet j in the next frame. Typically this could be the euclidian distance between the two droplets.
# sink_cost_vector is an integer vector of size n where the entry sink_cost_vector[i] is the cost associated to assigning droplet i in the previous frame to a "sink state" in the next frame, i.e. the cost of a "no-match-in-next-frame".
# This function retunrns a tuple (assigned, sinked), where assigned is a numpy vector of shape (x, 3) where each row contains three values, a, b, c, where a is the droplet in the first frame, b the droplet in the second frame and c is the cost of assignment of the two.
# A row in "assigned" means that the droplets a and b were assignet to each other. In contrast, "sinked" is a (n - x) dimensional vector which contains all droplets of the previous frame, which have not been assigned to any droplet in the next frame, i.e. droplets that have been mapped to the sink state.
def solve_assignment_with_sink(edge_exists_mask, cost_matrix, sink_cost_vector):
    fist_layer_offset = 1
    second_layer_offset = fist_layer_offset + edge_exists_mask.shape[0]
    second_layer_sink = second_layer_offset + edge_exists_mask.shape[1]
    source = 0
    sink = second_layer_sink + 1
    smcf = min_cost_flow.SimpleMinCostFlow()
    smcf.set_node_supply(source, edge_exists_mask.shape[0])
    smcf.set_node_supply(second_layer_sink, 0)
    for row_idx, row in enumerate(cost_matrix):
        smcf.add_arc_with_capacity_and_unit_cost(source, row_idx + fist_layer_offset, 1, 0)
        smcf.set_node_supply(row_idx + fist_layer_offset, 0)
        smcf.add_arc_with_capacity_and_unit_cost(row_idx + fist_layer_offset, second_layer_sink, 1, sink_cost_vector[row_idx])
        for col_idx, value in enumerate(row):
            if bool(edge_exists_mask[row_idx, col_idx]):
                smcf.set_node_supply(col_idx + second_layer_offset, 0)
                smcf.add_arc_with_capacity_and_unit_cost(row_idx + fist_layer_offset, col_idx + second_layer_offset, 1, value)
    for col_idx in range(cost_matrix.shape[1]):
        smcf.set_node_supply(col_idx + second_layer_offset, 0)
        smcf.add_arc_with_capacity_and_unit_cost(col_idx + second_layer_offset, sink, 1, 0)
    smcf.add_arc_with_capacity_and_unit_cost(second_layer_sink, sink, cost_matrix.shape[0], 0)
    smcf.set_node_supply(sink, -cost_matrix.shape[0])
    status = smcf.solve()
    sinked = []
    assigned = []
    if status == smcf.OPTIMAL:
        print('Total cost = ', smcf.optimal_cost())
        print()
        for arc in range(smcf.num_arcs()):
            if smcf.tail(arc) != source and smcf.head(arc) != sink:
                if smcf.flow(arc) > 0 and smcf.head(arc) != second_layer_sink:
                    assigned.append([smcf.tail(arc) - fist_layer_offset, smcf.head(arc) - second_layer_offset, smcf.unit_cost(arc)])
                    # print('Droplet %d assigned to droplet %d in next frame. Cost = %d' % (smcf.tail(arc) - 1, smcf.head(arc) - second_layer_offset, smcf.unit_cost(arc)))
                elif smcf.flow(arc) > 0:
                    sinked.append([smcf.tail(arc) - fist_layer_offset])
    else:
        print('There was an issue with the min cost flow input.')
        print(f'Status: {status}')
        assert(False)
    sinked = np.asarray(sinked, dtype = np.int64)
    assigned = np.asarray(assigned, dtype = np.int64)
    return (assigned, sinked)
    


def iterative_refinement_method (droplet_table_path, cell_table_path, image_path, tracking_table_path):
    dataset = create_dataset_cell_enhanced(None, ["BF", "DAPI"], image_path, droplet_table_path, cell_table_path, allFrames = True, buffer = -2, suppress_rest = True, suppression_slack = -3, median_filter_preprocess = True) 
    better_dataset = []
    for ds in dataset:
        concatenated_df = pd.concat(ds, axis = 1)
        concatenated_df = concatenated_df.T
        better_dataset.append(concatenated_df)
    nr_frames = len(dataset)
    feature_dataset = []
    image_dataset = []

    for ds in better_dataset:
        tmp1 = []
        tmp2 = np.zeros((ds.shape[0], 2, 40, 40), dtype = np.float64)
        iterator = 0
        for idx, row in ds.iterrows():
            tmp = compute_droplet_statistics(row)
            tmp3 = np.float64(row['patch'][1, :, :]) * (2.0 ** (-16))
            tmp3 -= np.median(tmp3)
            tmp['integrated_brightness'] = np.sum(tmp3)
            tmp['max_brightness'] = np.max(tmp3)
            tmp2[iterator, :, :, :] = np.float64(resize_patch(row['patch'], 40) * (2.0 ** (-16)))
            tmp2[iterator, 0, :, :] = cv.GaussianBlur(tmp2[iterator, 0, :, :], (3, 3), 0)
            tmp2[iterator, 0, :, :] = tmp2[iterator, 0, :, :] - np.mean(tmp2[iterator, 0, :, :])
            tmp2[iterator, 1, :, :] = tmp2[iterator, 1, :, :] - np.mean(tmp2[iterator, 1, :, :])
            # cv.imshow("test", tmp2[iterator, 0, :, :] / np.max(tmp2[iterator, 0, :, :]))
            # cv.waitKey(0)
            # cv.imshow("test", tmp2[iterator, 1, :, :])
            # cv.waitKey(0)
            iterator += 1
            tmp1.append(tmp)
        tmp1 = pd.DataFrame(tmp1)
        feature_dataset.append(tmp1)
        image_dataset.append(tmp2)

    out = []
    for frame_nr in range(nr_frames - 1):

        # IDEA: For each droplet in this frame, filter out other droplets that are 90% furthest away in terms of each feature. Do this for every feature.
        # In the end, the only remaining droplets will all be in top 10% closest across all features to the current droplet. 
        this_fr = frame_nr
        next_fr = this_fr + 1

        # higher bound. Droplets must have difference smaller than percentile
        quantile_level_matching = 0.9

        # lower bound. Droplets must be above percentile
        quantile_level_significant = 0.0
        quantile_level_significant_corr = 0.0
        maximal_distance = 250

        validity_mask = np.ones((feature_dataset[this_fr].shape[0], feature_dataset[next_fr].shape[0]), dtype = np.int32)

        distance_matrix = np.asarray([feature_dataset[this_fr]['center_row'].to_numpy(dtype = np.float64), feature_dataset[this_fr]['center_col'].to_numpy(dtype = np.float64)]).transpose()
        distance_matrix = np.linalg.norm(distance_matrix[:, None, :] - np.asarray([feature_dataset[next_fr]['center_row'].to_numpy(dtype = np.float64), feature_dataset[next_fr]['center_col'].to_numpy(dtype = np.float64)]).transpose()[None, :, :], axis = 2)
        sqdistance_matrix = np.int64(distance_matrix ** 2)
        within_dist_mask = (distance_matrix <= maximal_distance)

        similarity_matrix_bf_dapi = np.zeros((image_dataset[this_fr].shape[0], image_dataset[next_fr].shape[0], 2), dtype = np.float64)
        for row_idx, row in tqdm(enumerate(within_dist_mask)):
            similarity_matrix_bf_dapi[row_idx, row, :] = np.sum(image_dataset[this_fr][row_idx, None, :, :, :] * image_dataset[next_fr][None, row, :, :, :], axis = (3, 4))
            similarity_matrix_bf_dapi[row_idx, row, :] /= np.sqrt(np.sum((image_dataset[this_fr][row_idx, :, :, :])**2, axis = (1, 2)))[None, :] * np.sqrt(np.sum((image_dataset[next_fr][row, :, :, :])**2, axis = (2, 3)))[:, :]
        
        #  np.sum(image_dataset[this_fr][:, None, :, :, :] * image_dataset[next_fr][None, :, :, :, :], axis = (3, 4))


        # significant_droplets_mask = (similarity_matrix_bf_dapi[:, :, 0] > np.quantile(similarity_matrix_bf_dapi[within_dist_mask, 0], quantile_level_significant_corr)) * 1
        # validity_mask *= significant_droplets_mask
        # significant_droplets_mask = (similarity_matrix_bf_dapi[:, :, 1] > np.quantile(similarity_matrix_bf_dapi[within_dist_mask, 1], quantile_level_significant_corr)) * 1
        # validity_mask *= significant_droplets_mask

        # tmp1 = feature_dataset[this_fr]['integrated_brightness'].to_numpy(dtype = np.float32)
        # significant_droplets_mask = (tmp1 >= np.quantile(tmp1, quantile_level_significant)) * 1
        # validity_mask *= significant_droplets_mask[:, None]

        # tmp1 = feature_dataset[this_fr]['nr_signals'].to_numpy(dtype = np.int32)
        # significant_droplets_mask = (tmp1 > np.quantile(tmp1, quantile_level_significant)) * 1
        # validity_mask *= significant_droplets_mask[:, None]

        # print(np.sum(validity_mask))

        tmp1 = feature_dataset[this_fr]['nr_cells'].to_numpy(dtype = np.float32)
        tmp2 = feature_dataset[next_fr]['nr_cells'].to_numpy(dtype = np.float32)
        celldiff_mat = np.abs(tmp1[:, None] - tmp2[None, :])
        # celldiff_mat /= np.sqrt(1.0 + np.abs(tmp1)[:, None] * np.abs(tmp2)[None, :])
        # celldiff_mat[np.logical_not(within_dist_mask)] = np.nan
        celldiff_thresh = 1
        validity_mask *= (np.logical_or(celldiff_mat <= celldiff_thresh, np.logical_and(celldiff_mat <= 2, np.logical_or(tmp1[:, None] >= 4, tmp2[None, :] >= 4)))) * 1
        # celldiff_thresh = np.nanquantile(celldiff_mat, quantile_level_matching, axis = 1)
        # validity_mask *= (celldiff_mat <= celldiff_thresh[:, None]) * 1
        # validity_mask[np.logical_not(within_dist_mask)] = 0

        # cellmatching_mat = (((1 - 2 * (feature_dataset[this_fr]['nr_cells'].to_numpy(dtype = np.int32) > 0))[:, None] * (1 - 2 * (feature_dataset[next_fr]['nr_cells'].to_numpy(dtype = np.int32) > 0))[None, :]) == 1) * 1
        # validity_mask *= cellmatching_mat

        # cellexists_mat = (((feature_dataset[this_fr]['nr_cells'].to_numpy(dtype = np.int32) > 0)[:, None] * (feature_dataset[next_fr]['nr_cells'].to_numpy(dtype = np.int32) > 0)[None, :])) * 1
        # validity_mask *= cellexists_mat

        # signaldiff_mat = np.abs(feature_dataset[this_fr]['nr_signals'].to_numpy(dtype = np.float32)[:, None] - feature_dataset[next_fr]['nr_signals'].to_numpy(dtype = np.float32)[None, :])
        # signaldiff_mat /= np.sqrt(1.0 + np.abs(feature_dataset[this_fr]['nr_signals'].to_numpy(dtype = np.float32))[:, None] * np.abs(feature_dataset[next_fr]['nr_signals'].to_numpy(dtype = np.float32))[None, :])
        # signaldiff_mat[np.logical_not(within_dist_mask)] = np.nan
        # signaldiff_thresh = np.nanquantile(signaldiff_mat, quantile_level_matching, axis = 1)
        # validity_mask *= (signaldiff_mat <= signaldiff_thresh[:, None]) * 1
        # validity_mask[np.logical_not(within_dist_mask)] = 0
        # print(np.sum(validity_mask))

        # signalmatching_mat = (((1 - 2 * (feature_dataset[this_fr]['nr_signals'].to_numpy(dtype = np.int32) > 0))[:, None] * (1 - 2 * (feature_dataset[next_fr]['nr_signals'].to_numpy(dtype = np.int32) > 0))[None, :]) == 1) * 1
        # validity_mask *= signalmatching_mat

        # signalexists_mat = (((feature_dataset[this_fr]['nr_signals'].to_numpy(dtype = np.int32) > 0)[:, None] * (feature_dataset[next_fr]['nr_signals'].to_numpy(dtype = np.int32) > 0)[None, :])) * 1
        # validity_mask *= signalexists_mat
        # signalexists_now_mat = (feature_dataset[this_fr]['nr_signals'].to_numpy(dtype = np.int32) > 0) * 1
        # validity_mask *= signalexists_now_mat[:, None]

        radiusdiff_mat = np.abs(feature_dataset[this_fr]['radius'].to_numpy(dtype = np.int32)[:, None] - feature_dataset[next_fr]['radius'].to_numpy(dtype = np.int32)[None, :])
        validity_mask *= (radiusdiff_mat <= 3) * 1
        # radiusdiff_thresh = np.quantile(radiusdiff_mat, quantile_level, axis = 1)
        # validity_mask *= (radiusdiff_mat <= radiusdiff_thresh[:, None]) * 1

        bf_corr_mat = similarity_matrix_bf_dapi[:, :, 0]
        bf_corr_mat[np.logical_not(within_dist_mask)] = np.nan
        bf_corr_thresh = np.nanquantile(bf_corr_mat, 1.0 - quantile_level_matching, axis = 1)
        validity_mask *= (bf_corr_mat >= bf_corr_thresh[:, None]) * 1
        # validity_mask *= (bf_corr_mat >= 0.9) * 1

        dapi_corr_mat = similarity_matrix_bf_dapi[:, :, 1]
        dapi_corr_mat[np.logical_not(within_dist_mask)] = np.nan
        dapi_corr_thresh = np.nanquantile(dapi_corr_mat, 1.0 - quantile_level_matching, axis = 1)
        validity_mask *= (dapi_corr_mat >= dapi_corr_thresh[:, None]) * 1
        # validity_mask *= (dapi_corr_mat >= 0.9) * 1



        intbrightdiff_mat = np.abs(feature_dataset[this_fr]['integrated_brightness'].to_numpy(dtype = np.float32)[:, None] - feature_dataset[next_fr]['integrated_brightness'].to_numpy(dtype = np.float32)[None, :])
        intbrightdiff_mat /= np.sqrt(np.abs(feature_dataset[this_fr]['integrated_brightness'].to_numpy(dtype = np.float32)[:, None]) * np.abs(feature_dataset[next_fr]['integrated_brightness'].to_numpy(dtype = np.float32)[None, :]))
        intbrightdiff_mat[np.logical_not(within_dist_mask)] = np.nan
        intbrightdiff_thresh = np.nanquantile(intbrightdiff_mat, quantile_level_matching, axis = 1)
        validity_mask *= (intbrightdiff_mat <= intbrightdiff_thresh[:, None]) * 1

        maxbrightdiff_mat = np.abs(feature_dataset[this_fr]['max_brightness'].to_numpy(dtype = np.float32)[:, None] - feature_dataset[next_fr]['max_brightness'].to_numpy(dtype = np.float32)[None, :])
        maxbrightdiff_mat /= np.sqrt(np.abs(feature_dataset[this_fr]['max_brightness'].to_numpy(dtype = np.float32)[:, None]) * np.abs(feature_dataset[next_fr]['max_brightness'].to_numpy(dtype = np.float32)[None, :]))
        maxbrightdiff_mat[np.logical_not(within_dist_mask)] = np.nan
        maxbrightdiff_thresh = np.nanquantile(maxbrightdiff_mat, quantile_level_matching, axis = 1)
        validity_mask *= (maxbrightdiff_mat <= maxbrightdiff_thresh[:, None]) * 1



        # print(np.sum(brightdiff_mat <= brightdiff_thresh[:, None]))
        # print(np.max(brightdiff_mat[within_dist_mask]))
        # print(np.median(brightdiff_mat[within_dist_mask]))
        # print(np.mean(brightdiff_mat[within_dist_mask]))
        # print(np.min(brightdiff_mat[within_dist_mask]))
        # assert(False)

        # brightdiff_mat = np.abs(feature_dataset[this_fr]['integrated_brightness'].to_numpy(dtype = np.float32)[:, None] - feature_dataset[next_fr]['integrated_brightness'].to_numpy(dtype = np.float32)[None, :])
        # brightdiff_mat /= np.sqrt(np.abs(feature_dataset[this_fr]['integrated_brightness'].to_numpy(dtype = np.float32)[:, None]) * np.abs(feature_dataset[next_fr]['integrated_brightness'].to_numpy(dtype = np.float32)[None, :]))
        # brightdiff_mat[np.logical_not(within_dist_mask)] = np.nan
        # brightdiff_thresh = np.nanquantile(brightdiff_mat, quantile_level_matching, axis = 1)
        # validity_mask *= (brightdiff_mat <= brightdiff_thresh[:, None]) * 1



        hog_mat = np.zeros((feature_dataset[this_fr].shape[0], feature_dataset[next_fr].shape[0]))
        hog_prev = feature_dataset[this_fr]['hog'].values
        hog_next = feature_dataset[next_fr]['hog'].values
        for i in range(feature_dataset[this_fr].shape[0]):
            for j in range(feature_dataset[next_fr].shape[0]):
                hog_mat[i,j] = np.linalg.norm(hog_prev[i] - hog_next[j])


        hog_mat[np.logical_not(within_dist_mask)] = np.nan
        hog_thresh = np.nanquantile(hog_mat, quantile_level_matching, axis=1)
        validity_mask *= (hog_mat <= hog_thresh[:, None]) * 1



        validity_mask[np.logical_not(within_dist_mask)] = 0

        # print(np.sum(validity_mask))

        # sqdistance_matrix[within_dist_mask] = np.int64(sqdistance_matrix[within_dist_mask] * (maxbrightdiff_mat[within_dist_mask] + 1.0) * (intbrightdiff_mat[within_dist_mask] + 1.0))

        distance_matrix[within_dist_mask] = np.int64(distance_matrix[within_dist_mask] * (maxbrightdiff_mat[within_dist_mask] + 1.0) * (intbrightdiff_mat[within_dist_mask] + 1.0) * (1.0 - bf_corr_mat[within_dist_mask] + 1.0) * (1.0 - dapi_corr_mat[within_dist_mask] + 1.0))

        print("")
        # assigned, sinked = solve_assignment_with_sink(validity_mask, sqdistance_matrix, np.ones((validity_mask.shape[0],), dtype = np.int32) * maximal_distance ** 2)
        # Here we do some weird scaling shenanigans because the flow optimization algorithm used only accepts integer cost.
        assigned, sinked = solve_assignment_with_sink(validity_mask, np.int64(distance_matrix * 10), np.ones((validity_mask.shape[0],), dtype = np.int32) * maximal_distance * 10)
        print("Number of assignments found: " + str(assigned.shape[0]))
        print("Number of unassigned droplets: " + str(sinked.size))

        print("Assignments: ")
        print(assigned)

        print("Average distance:" + str(np.mean(np.sqrt(assigned[:, 2]))))
        print("Median distance:" + str(np.median(np.sqrt(assigned[:, 2]))))

        print("Mean possibilities per droplet: " + str(np.mean(np.sum(validity_mask, axis = 1))))
        print("Percentage of droplets with possibilities: " + str(np.mean(np.sum(validity_mask, axis = 1) > 0)))
        print("Mean possibilities per droplet for droplets with possibilities: " + str(np.mean(np.sum(validity_mask, axis = 1)) / np.mean(np.sum(validity_mask, axis = 1) > 0)))

        for ass in assigned:
            out.append({"framePrev": this_fr, "frameNext": next_fr, "dropletIdPrev": ass[0], "dropletIdNext": ass[1]})

        # violation_counter = np.zeros((feature_dataset[this_fr].shape[0], feature_dataset[next_fr].shape[0]), dtype = np.int32)
        # hascells_mat = np.int32(feature_dataset[this_fr]['nr_cells'].to_numpy(dtype = np.int32) > 0)[:, None] * np.int32(feature_dataset[next_fr]['nr_cells'].to_numpy(dtype = np.int32) > 0)[None, :]
        # violation_counter += (1 - hascells_mat)
        # radius_absdiff_mat = np.abs(feature_dataset[this_fr]['radius'].to_numpy(dtype = np.int32)[:, None] - feature_dataset[next_fr]['radius'].to_numpy(dtype = np.int32)[None, :])
        # violation_counter +=  np.int32(radius_absdiff_mat > 2)
        # nrcells_absdiff_mat = np.abs(feature_dataset[this_fr]['nr_cells'].to_numpy(dtype = np.int32)[:, None] - feature_dataset[next_fr]['nr_cells'].to_numpy(dtype = np.int32)[None, :])
        # violation_counter +=  np.int32(nrcells_absdiff_mat > 1)
        # nrsignals_absdiff_mat = np.abs(feature_dataset[this_fr]['nr_signals'].to_numpy(dtype = np.int32)[:, None] - feature_dataset[next_fr]['nr_signals'].to_numpy(dtype = np.int32)[None, :])
        # violation_counter +=  np.int32(nrsignals_absdiff_mat > 2)
        # bright_this_fr = feature_dataset[this_fr]['integrated_brightness'].to_numpy(dtype = np.float32)
        # bright_next_fr = feature_dataset[next_fr]['integrated_brightness'].to_numpy(dtype = np.float32)
        # bright_this_fr = bright_this_fr / np.sqrt(np.mean(bright_this_fr**2))
        # bright_next_fr = bright_next_fr / np.sqrt(np.mean(bright_next_fr**2))
        # bright_absdiff_mat = np.abs(bright_this_fr[:, None] - bright_next_fr[None, :])
        # violation_counter += np.int32(bright_absdiff_mat > 0.2)
        # robust_viable_mask = (violation_counter == 0)
        # robust_viable_rows_mask = (np.sum(robust_viable_mask, axis = 1) > 0)
        # # print(np.sum(robust_viable_rows_mask))
        # # robust_viable_relevant_mask = robust_viable_mask[robust_viable_rows_mask == 1, :]
        # distance_matrix = np.asarray([feature_dataset[this_fr]['center_row'].to_numpy(dtype = np.float64), feature_dataset[this_fr]['center_col'].to_numpy(dtype = np.float64)]).transpose()
        # distance_matrix = np.linalg.norm(distance_matrix[:, None, :] - np.asarray([feature_dataset[next_fr]['center_row'].to_numpy(dtype = np.float64), feature_dataset[next_fr]['center_col'].to_numpy(dtype = np.float64)]).transpose(), axis = 2)
        # distance_matrix = np.int64(distance_matrix ** 2)
        

        # assert(False)
    tracking_df = pd.DataFrame(out)
    tracking_df.to_csv(tracking_table_path, index = False)
        



def droplet_linking_feature_based_voting (droplet_table_path, cell_table_path, bf_image_path, dapi_image_path,tracking_table_path):
    bf_image = np.load(bf_image_path, allow_pickle = True)
    dapi_image = np.load(dapi_image_path, allow_pickle = True)
    image = np.transpose(np.asarray([bf_image, dapi_image]), axes = [1, 0, 2, 3])
    dataset = create_dataset_cell_enhanced_from_ndarray(None, image, droplet_table_path, cell_table_path, allFrames = True, buffer = -2, suppress_rest = True, suppression_slack = -3) 
    better_dataset = []
    for ds in dataset:
        concatenated_df = pd.concat(ds, axis = 1)
        concatenated_df = concatenated_df.T
        better_dataset.append(concatenated_df)
    nr_frames = len(dataset)
    feature_dataset = []
    image_dataset = []
    for ds in better_dataset:
        tmp1 = []
        tmp2 = np.zeros((ds.shape[0], 2, 40, 40), dtype = np.float64)
        iterator = 0
        for idx, row in ds.iterrows():
            # cv.imshow("test", row['patch'][0, :, :])
            # cv.waitKey(0)
            # cv.imshow("test", row['patch'][1, :, :])
            # cv.waitKey(0)

            tmp = compute_droplet_statistics(row)
            tmp3 = np.float64(row['patch'][1, :, :]) * (2.0 ** (-16))
            tmp['integrated_brightness'] = np.sum(tmp3)
            tmp['max_brightness'] = np.max(tmp3)
            tmp2[iterator, :, :, :] = np.float64(resize_patch(row['patch'], 40) * (2.0 ** (-16)))
            tmp2[iterator, 0, :, :] = tmp2[iterator, 0, :, :] - np.mean(tmp2[iterator, 0, :, :])
            tmp2[iterator, 1, :, :] = tmp2[iterator, 1, :, :] - np.mean(tmp2[iterator, 1, :, :])
            # cv.imshow("test", tmp2[iterator, 0, :, :] / np.max(tmp2[iterator, 0, :, :]))
            # cv.waitKey(0)
            # cv.imshow("test", tmp2[iterator, 1, :, :])
            # cv.waitKey(0)
            iterator += 1
            tmp1.append(tmp)
        tmp1 = pd.DataFrame(tmp1)
        feature_dataset.append(tmp1)
        image_dataset.append(tmp2)
    out = []
    for frame_nr in range(nr_frames - 1):

        # IDEA: Voting system

        # For readability, get indices of this and next frame
        this_fr = frame_nr
        next_fr = this_fr + 1

        # Maximal distance allowed for two droplets to be matched
        maximal_distance = 250

        # An integer matrix which at the end, will contain in entry i,j the number of votes that favor linking droplet i from the previous frame to droplet j in the next frame.
        # The votes will come from features
        voting_bins = np.zeros((feature_dataset[this_fr].shape[0], feature_dataset[next_fr].shape[0]), dtype = np.int32)

        # Compute the distance between all droplets
        distance_matrix = np.asarray([feature_dataset[this_fr]['center_row'].to_numpy(dtype = np.float64), feature_dataset[this_fr]['center_col'].to_numpy(dtype = np.float64)]).transpose()
        distance_matrix = np.linalg.norm(distance_matrix[:, None, :] - np.asarray([feature_dataset[next_fr]['center_row'].to_numpy(dtype = np.float64), feature_dataset[next_fr]['center_col'].to_numpy(dtype = np.float64)]).transpose()[None, :, :], axis = 2)
        # Square distance in case we need it
        sqdistance_matrix = np.int64(distance_matrix ** 2)
        # A boolean mask that tells us if two droplets in the two subsequent frames are within range of each other
        within_dist_mask = (distance_matrix <= maximal_distance)

        # This matrix will contain all correlations / similarities between droplets that are close enough to each other for both the dapi and bf channel.
        # Correlation here literally means linear correlation / convolution
        similarity_matrix_bf_dapi = np.zeros((image_dataset[this_fr].shape[0], image_dataset[next_fr].shape[0], 2), dtype = np.float64)
        # Computing these similarities must be done batch-wise because otherwise we use too much memory
        for row_idx, row in tqdm(enumerate(within_dist_mask)):
            similarity_matrix_bf_dapi[row_idx, row, :] = np.sum(image_dataset[this_fr][row_idx, None, :, :, :] * image_dataset[next_fr][None, row, :, :, :], axis = (3, 4))
            similarity_matrix_bf_dapi[row_idx, row, :] /= np.sqrt(np.sum((image_dataset[this_fr][row_idx, :, :, :])**2, axis = (1, 2)))[None, :] * np.sqrt(np.sum((image_dataset[next_fr][row, :, :, :])**2, axis = (2, 3)))[:, :]
        
        #  np.sum(image_dataset[this_fr][:, None, :, :, :] * image_dataset[next_fr][None, :, :, :, :], axis = (3, 4))


        # significant_droplets_mask = (similarity_matrix_bf_dapi[:, :, 0] > np.quantile(similarity_matrix_bf_dapi[within_dist_mask, 0], quantile_level_significant_corr)) * 1
        # validity_mask *= significant_droplets_mask
        # significant_droplets_mask = (similarity_matrix_bf_dapi[:, :, 1] > np.quantile(similarity_matrix_bf_dapi[within_dist_mask, 1], quantile_level_significant_corr)) * 1
        # validity_mask *= significant_droplets_mask

        # tmp1 = feature_dataset[this_fr]['integrated_brightness'].to_numpy(dtype = np.float32)
        # significant_droplets_mask = (tmp1 >= np.quantile(tmp1, quantile_level_significant)) * 1
        # validity_mask *= significant_droplets_mask[:, None]

        # tmp1 = feature_dataset[this_fr]['nr_signals'].to_numpy(dtype = np.int32)
        # significant_droplets_mask = (tmp1 > np.quantile(tmp1, quantile_level_significant)) * 1
        # validity_mask *= significant_droplets_mask[:, None]

        # print(np.sum(validity_mask))

        # Get the vector with nr of cells in the previous frame
        tmp1 = feature_dataset[this_fr]['nr_cells'].to_numpy(dtype = np.int32)
        # Get the vector with nr of cells in the next frame
        tmp2 = feature_dataset[next_fr]['nr_cells'].to_numpy(dtype = np.int32)
        # Compute the absolute differences in terms of nr of cells detected between all droplets in prevous and next frame
        celldiff_mat = np.abs(tmp1[:, None] - tmp2[None, :])
        # celldiff_mat /= np.sqrt(1.0 + np.abs(tmp1)[:, None] * np.abs(tmp2)[None, :])
        # celldiff_mat[np.logical_not(within_dist_mask)] = np.nan
        # celldiff_thresh = 1

        # Now its time to vote
        # We vote in favor of matching two droplets between two frames if the difference in nr of cells is leq 1 or if it is leq 2 and one of the droplets has at least 4 cells detected
        # voting_bins += (np.logical_or(celldiff_mat <= 1, np.logical_and(celldiff_mat <= 2, np.logical_or(tmp1[:, None] >= 4, tmp2[None, :] >= 4)))) * 1
        # validity_mask *= (np.logical_or(celldiff_mat <= 1, np.logical_and(celldiff_mat <= 2, np.logical_or(tmp1[:, None] >= 4, tmp2[None, :] >= 4)))) * 1
        # We give negative votes proportional to teh absolute difference in nr of detected cells
        voting_bins -= celldiff_mat
        
        
        # celldiff_thresh = np.nanquantile(celldiff_mat, quantile_level_matching, axis = 1)
        # validity_mask *= (celldiff_mat <= celldiff_thresh[:, None]) * 1
        # validity_mask[np.logical_not(within_dist_mask)] = 0

        # cellmatching_mat = (((1 - 2 * (feature_dataset[this_fr]['nr_cells'].to_numpy(dtype = np.int32) > 0))[:, None] * (1 - 2 * (feature_dataset[next_fr]['nr_cells'].to_numpy(dtype = np.int32) > 0))[None, :]) == 1) * 1
        # validity_mask *= cellmatching_mat

        # cellexists_mat = (((feature_dataset[this_fr]['nr_cells'].to_numpy(dtype = np.int32) > 0)[:, None] * (feature_dataset[next_fr]['nr_cells'].to_numpy(dtype = np.int32) > 0)[None, :])) * 1
        # validity_mask *= cellexists_mat

        # signaldiff_mat = np.abs(feature_dataset[this_fr]['nr_signals'].to_numpy(dtype = np.float32)[:, None] - feature_dataset[next_fr]['nr_signals'].to_numpy(dtype = np.float32)[None, :])
        # signaldiff_mat /= np.sqrt(1.0 + np.abs(feature_dataset[this_fr]['nr_signals'].to_numpy(dtype = np.float32))[:, None] * np.abs(feature_dataset[next_fr]['nr_signals'].to_numpy(dtype = np.float32))[None, :])
        # signaldiff_mat[np.logical_not(within_dist_mask)] = np.nan
        # signaldiff_thresh = np.nanquantile(signaldiff_mat, quantile_level_matching, axis = 1)
        # validity_mask *= (signaldiff_mat <= signaldiff_thresh[:, None]) * 1
        # validity_mask[np.logical_not(within_dist_mask)] = 0
        # print(np.sum(validity_mask))

        # signalmatching_mat = (((1 - 2 * (feature_dataset[this_fr]['nr_signals'].to_numpy(dtype = np.int32) > 0))[:, None] * (1 - 2 * (feature_dataset[next_fr]['nr_signals'].to_numpy(dtype = np.int32) > 0))[None, :]) == 1) * 1
        # validity_mask *= signalmatching_mat

        # signalexists_mat = (((feature_dataset[this_fr]['nr_signals'].to_numpy(dtype = np.int32) > 0)[:, None] * (feature_dataset[next_fr]['nr_signals'].to_numpy(dtype = np.int32) > 0)[None, :])) * 1
        # validity_mask *= signalexists_mat
        # signalexists_now_mat = (feature_dataset[this_fr]['nr_signals'].to_numpy(dtype = np.int32) > 0) * 1
        # validity_mask *= signalexists_now_mat[:, None]

        # Compute the difference in droplet radius between the two frames for every droplet
        radiusdiff_mat = np.abs(feature_dataset[this_fr]['radius'].to_numpy(dtype = np.int32)[:, None] - feature_dataset[next_fr]['radius'].to_numpy(dtype = np.int32)[None, :])
        # Give negative votes based on how big the discrepancy between the radii is
        voting_bins -= radiusdiff_mat
        # validity_mask *= (radiusdiff_mat <= 3) * 1
        # radiusdiff_thresh = np.quantile(radiusdiff_mat, quantile_level, axis = 1)
        # validity_mask *= (radiusdiff_mat <= radiusdiff_thresh[:, None]) * 1
        
        # Look at the correlations between droplets in the brightfield channel
        bf_corr_mat = similarity_matrix_bf_dapi[:, :, 0]
        # Set droplets out of reach to nan
        bf_corr_mat[np.logical_not(within_dist_mask)] = np.nan
        for perc in [0.125]:
            # Give a vote if droplet in next frame is amongst top k% best matches for dropet in this frame
            bf_corr_thresh = np.nanquantile(bf_corr_mat, perc, axis = 1)
            voting_bins += (bf_corr_mat >= bf_corr_thresh[:, None]) * 1

        
        # Look at the correlations between droplets in the dapi channel
        dapi_corr_mat = similarity_matrix_bf_dapi[:, :, 1]
        # Set droplets out of reach to nan
        dapi_corr_mat[np.logical_not(within_dist_mask)] = np.nan
        for perc in [0.125]:
            # Give a vote if droplet in next frame is amongst top k% best matches for dropet in this frame
            dapi_corr_thresh = np.nanquantile(dapi_corr_mat, perc, axis = 1)
            voting_bins += (dapi_corr_mat >= dapi_corr_thresh[:, None]) * 1

        # Look at the difference of integrated brightness for every pair of droplets
        intbrightdiff_mat = np.abs(feature_dataset[this_fr]['integrated_brightness'].to_numpy(dtype = np.float32)[:, None] - feature_dataset[next_fr]['integrated_brightness'].to_numpy(dtype = np.float32)[None, :])
        intbrightdiff_mat /= np.sqrt(np.abs(feature_dataset[this_fr]['integrated_brightness'].to_numpy(dtype = np.float32)[:, None]) * np.abs(feature_dataset[next_fr]['integrated_brightness'].to_numpy(dtype = np.float32)[None, :]))
        # Set droplets out of reach to nan
        intbrightdiff_mat[np.logical_not(within_dist_mask)] = np.nan
        for perc in [0.875]:
            # Give a vote if droplet in next frame is amongst top k% best matches for dropet in this frame
            intbrightdiff_thresh = np.nanquantile(intbrightdiff_mat, perc, axis = 1)
            voting_bins += (intbrightdiff_mat <= intbrightdiff_thresh[:, None]) * 1

        # Look at the difference of maximum brightness for every pair of droplets
        maxbrightdiff_mat = np.abs(feature_dataset[this_fr]['max_brightness'].to_numpy(dtype = np.float32)[:, None] - feature_dataset[next_fr]['max_brightness'].to_numpy(dtype = np.float32)[None, :])
        maxbrightdiff_mat /= np.sqrt(np.abs(feature_dataset[this_fr]['max_brightness'].to_numpy(dtype = np.float32)[:, None]) * np.abs(feature_dataset[next_fr]['max_brightness'].to_numpy(dtype = np.float32)[None, :]))
        # Set droplets out of reach to nan
        maxbrightdiff_mat[np.logical_not(within_dist_mask)] = np.nan
        for perc in [0.875]:
            # Give a vote if droplet in next frame is amongst top k% best matches for dropet in this frame
            maxbrightdiff_thresh = np.nanquantile(maxbrightdiff_mat, perc, axis = 1)
            voting_bins += (maxbrightdiff_mat <= maxbrightdiff_thresh[:, None]) * 1


        # Set voting bins for matchings that  are out of reach to nan
        voting_bins = np.float32(voting_bins)
        voting_bins[np.logical_not(within_dist_mask)] = np.nan
        # For every droplet, look at the 50% of droplets in the next frame that are within range and that have the most votes.
        #  Those are the droplets wit which we allow a matching. All other possible matchings get discarded
        vote_threshold_per_droplet = np.nanquantile(voting_bins, 0.5, axis = 1)
        # Create a mask of which matchings are allowed based on voting
        validity_mask = (voting_bins >= vote_threshold_per_droplet[:, None]) * 1
        # Set validity_mask for matchings that are out of reach to 0.
        validity_mask[np.logical_not(within_dist_mask)] = 0

        # print(np.sum(brightdiff_mat <= brightdiff_thresh[:, None]))
        # print(np.max(brightdiff_mat[within_dist_mask]))
        # print(np.median(brightdiff_mat[within_dist_mask]))
        # print(np.mean(brightdiff_mat[within_dist_mask]))
        # print(np.min(brightdiff_mat[within_dist_mask]))
        # assert(False)

        # brightdiff_mat = np.abs(feature_dataset[this_fr]['integrated_brightness'].to_numpy(dtype = np.float32)[:, None] - feature_dataset[next_fr]['integrated_brightness'].to_numpy(dtype = np.float32)[None, :])
        # brightdiff_mat /= np.sqrt(np.abs(feature_dataset[this_fr]['integrated_brightness'].to_numpy(dtype = np.float32)[:, None]) * np.abs(feature_dataset[next_fr]['integrated_brightness'].to_numpy(dtype = np.float32)[None, :]))
        # brightdiff_mat[np.logical_not(within_dist_mask)] = np.nan
        # brightdiff_thresh = np.nanquantile(brightdiff_mat, quantile_level_matching, axis = 1)
        # validity_mask *= (brightdiff_mat <= brightdiff_thresh[:, None]) * 1

        # print(np.sum(validity_mask))

        # sqdistance_matrix[within_dist_mask] = np.int64(sqdistance_matrix[within_dist_mask] * (maxbrightdiff_mat[within_dist_mask] + 1.0) * (intbrightdiff_mat[within_dist_mask] + 1.0))

        # distance_matrix[within_dist_mask] = np.int64(distance_matrix[within_dist_mask] * (maxbrightdiff_mat[within_dist_mask] + 1.0) * (intbrightdiff_mat[within_dist_mask] + 1.0) * (1.0 - bf_corr_mat[within_dist_mask] + 1.0) * (1.0 - dapi_corr_mat[within_dist_mask] + 1.0))

        print("")
        # assigned, sinked = solve_assignment_with_sink(validity_mask, sqdistance_matrix, np.ones((validity_mask.shape[0],), dtype = np.int32) * maximal_distance ** 2)
        
        # Here we do some weird scaling shenanigans because the flow optimization algorithm used only accepts integer cost.
        assigned, sinked = solve_assignment_with_sink(validity_mask, np.int64(distance_matrix * 10), np.ones((validity_mask.shape[0],), dtype = np.int32) * maximal_distance * 10)
        print("Number of assignments found: " + str(assigned.shape[0]))
        print("Number of unassigned droplets: " + str(sinked.size))

        print("Assignments: ")
        print(assigned)

        print("Average distance:" + str(np.mean(np.sqrt(assigned[:, 2]))))
        print("Median distance:" + str(np.median(np.sqrt(assigned[:, 2]))))

        print("Mean possibilities per droplet: " + str(np.mean(np.sum(validity_mask, axis = 1))))
        print("Percentage of droplets with possibilities: " + str(np.mean(np.sum(validity_mask, axis = 1) > 0)))
        print("Mean possibilities per droplet for droplets with possibilities: " + str(np.mean(np.sum(validity_mask, axis = 1)) / np.mean(np.sum(validity_mask, axis = 1) > 0)))

        for ass in assigned:
            out.append({"framePrev": this_fr, "frameNext": next_fr, "dropletIdPrev": ass[0], "dropletIdNext": ass[1]})

        # violation_counter = np.zeros((feature_dataset[this_fr].shape[0], feature_dataset[next_fr].shape[0]), dtype = np.int32)
        # hascells_mat = np.int32(feature_dataset[this_fr]['nr_cells'].to_numpy(dtype = np.int32) > 0)[:, None] * np.int32(feature_dataset[next_fr]['nr_cells'].to_numpy(dtype = np.int32) > 0)[None, :]
        # violation_counter += (1 - hascells_mat)
        # radius_absdiff_mat = np.abs(feature_dataset[this_fr]['radius'].to_numpy(dtype = np.int32)[:, None] - feature_dataset[next_fr]['radius'].to_numpy(dtype = np.int32)[None, :])
        # violation_counter +=  np.int32(radius_absdiff_mat > 2)
        # nrcells_absdiff_mat = np.abs(feature_dataset[this_fr]['nr_cells'].to_numpy(dtype = np.int32)[:, None] - feature_dataset[next_fr]['nr_cells'].to_numpy(dtype = np.int32)[None, :])
        # violation_counter +=  np.int32(nrcells_absdiff_mat > 1)
        # nrsignals_absdiff_mat = np.abs(feature_dataset[this_fr]['nr_signals'].to_numpy(dtype = np.int32)[:, None] - feature_dataset[next_fr]['nr_signals'].to_numpy(dtype = np.int32)[None, :])
        # violation_counter +=  np.int32(nrsignals_absdiff_mat > 2)
        # bright_this_fr = feature_dataset[this_fr]['integrated_brightness'].to_numpy(dtype = np.float32)
        # bright_next_fr = feature_dataset[next_fr]['integrated_brightness'].to_numpy(dtype = np.float32)
        # bright_this_fr = bright_this_fr / np.sqrt(np.mean(bright_this_fr**2))
        # bright_next_fr = bright_next_fr / np.sqrt(np.mean(bright_next_fr**2))
        # bright_absdiff_mat = np.abs(bright_this_fr[:, None] - bright_next_fr[None, :])
        # violation_counter += np.int32(bright_absdiff_mat > 0.2)
        # robust_viable_mask = (violation_counter == 0)
        # robust_viable_rows_mask = (np.sum(robust_viable_mask, axis = 1) > 0)
        # # print(np.sum(robust_viable_rows_mask))
        # # robust_viable_relevant_mask = robust_viable_mask[robust_viable_rows_mask == 1, :]
        # distance_matrix = np.asarray([feature_dataset[this_fr]['center_row'].to_numpy(dtype = np.float64), feature_dataset[this_fr]['center_col'].to_numpy(dtype = np.float64)]).transpose()
        # distance_matrix = np.linalg.norm(distance_matrix[:, None, :] - np.asarray([feature_dataset[next_fr]['center_row'].to_numpy(dtype = np.float64), feature_dataset[next_fr]['center_col'].to_numpy(dtype = np.float64)]).transpose(), axis = 2)
        # distance_matrix = np.int64(distance_matrix ** 2)
        

        # assert(False)
    tracking_df = pd.DataFrame(out)
    tracking_df.to_csv(tracking_table_path, index = False)


def linking(image_name,FEATURE_PATH,RESULT_PATH):
    '''dataset = create_dataset_cell_enhanced(None, ["BF", "DAPI"], image_path, droplet_table_path, cell_table_path,
                                           allFrames=True, buffer=-2, suppress_rest=True, suppression_slack=-3,
                                           median_filter_preprocess=True)'''
    path = Path(FEATURE_PATH / f"fulldataset_{image_name}.npy")
    dataset = np.load(path, allow_pickle=True)
    better_dataset = []
    for ds in dataset:
        concatenated_df = pd.concat(ds, axis=1)
        concatenated_df = concatenated_df.T
        better_dataset.append(concatenated_df)
    nr_frames = len(dataset)
    feature_dataset = []
    image_dataset = []

    for ds in better_dataset:
        tmp1 = []
        tmp2 = np.zeros((ds.shape[0], 2, 40, 40), dtype=np.float64)
        iterator = 0
        for idx, row in ds.iterrows():
            tmp = compute_droplet_statistics(row)
            tmp3 = np.float64(row['patch'][1, :, :]) * (2.0 ** (-16))
            tmp3 -= np.median(tmp3)
            tmp['integrated_brightness'] = np.sum(tmp3)
            tmp['max_brightness'] = np.max(tmp3)
            tmp2[iterator, :, :, :] = np.float64(resize_patch(row['patch'], 40) * (2.0 ** (-16)))
            tmp2[iterator, 0, :, :] = cv.GaussianBlur(tmp2[iterator, 0, :, :], (3, 3), 0)
            tmp2[iterator, 0, :, :] = tmp2[iterator, 0, :, :] - np.mean(tmp2[iterator, 0, :, :])
            tmp2[iterator, 1, :, :] = tmp2[iterator, 1, :, :] - np.mean(tmp2[iterator, 1, :, :])
            # cv.imshow("test", tmp2[iterator, 0, :, :] / np.max(tmp2[iterator, 0, :, :]))
            # cv.waitKey(0)
            # cv.imshow("test", tmp2[iterator, 1, :, :])
            # cv.waitKey(0)
            iterator += 1
            tmp1.append(tmp)
        tmp1 = pd.DataFrame(tmp1)
        feature_dataset.append(tmp1)
        image_dataset.append(tmp2)

    out = []
    for frame_nr in range(nr_frames - 1):

        # IDEA: For each droplet in this frame, filter out other droplets that are 90% furthest away in terms of each feature. Do this for every feature.
        # In the end, the only remaining droplets will all be in top 10% closest across all features to the current droplet.
        this_fr = frame_nr
        next_fr = this_fr + 1

        # higher bound. Droplets must have difference smaller than percentile
        quantile_level_matching = 0.9

        # lower bound. Droplets must be above percentile
        quantile_level_significant = 0.0
        quantile_level_significant_corr = 0.0
        maximal_distance = 250

        validity_mask = np.ones((feature_dataset[this_fr].shape[0], feature_dataset[next_fr].shape[0]), dtype=np.int32)

        distance_matrix = np.asarray([feature_dataset[this_fr]['center_row'].to_numpy(dtype=np.float64),
                                      feature_dataset[this_fr]['center_col'].to_numpy(dtype=np.float64)]).transpose()
        distance_matrix = np.linalg.norm(distance_matrix[:, None, :] - np.asarray(
            [feature_dataset[next_fr]['center_row'].to_numpy(dtype=np.float64),
             feature_dataset[next_fr]['center_col'].to_numpy(dtype=np.float64)]).transpose()[None, :, :], axis=2)
        sqdistance_matrix = np.int64(distance_matrix ** 2)
        within_dist_mask = (distance_matrix <= maximal_distance)

        similarity_matrix_bf_dapi = np.zeros((image_dataset[this_fr].shape[0], image_dataset[next_fr].shape[0], 2),
                                             dtype=np.float64)
        for row_idx, row in tqdm(enumerate(within_dist_mask)):
            similarity_matrix_bf_dapi[row_idx, row, :] = np.sum(
                image_dataset[this_fr][row_idx, None, :, :, :] * image_dataset[next_fr][None, row, :, :, :],
                axis=(3, 4))
            similarity_matrix_bf_dapi[row_idx, row, :] /= np.sqrt(
                np.sum((image_dataset[this_fr][row_idx, :, :, :]) ** 2, axis=(1, 2)))[None, :] * np.sqrt(
                np.sum((image_dataset[next_fr][row, :, :, :]) ** 2, axis=(2, 3)))[:, :]

        #  np.sum(image_dataset[this_fr][:, None, :, :, :] * image_dataset[next_fr][None, :, :, :, :], axis = (3, 4))

        # significant_droplets_mask = (similarity_matrix_bf_dapi[:, :, 0] > np.quantile(similarity_matrix_bf_dapi[within_dist_mask, 0], quantile_level_significant_corr)) * 1
        # validity_mask *= significant_droplets_mask
        # significant_droplets_mask = (similarity_matrix_bf_dapi[:, :, 1] > np.quantile(similarity_matrix_bf_dapi[within_dist_mask, 1], quantile_level_significant_corr)) * 1
        # validity_mask *= significant_droplets_mask

        # tmp1 = feature_dataset[this_fr]['integrated_brightness'].to_numpy(dtype = np.float32)
        # significant_droplets_mask = (tmp1 >= np.quantile(tmp1, quantile_level_significant)) * 1
        # validity_mask *= significant_droplets_mask[:, None]

        # tmp1 = feature_dataset[this_fr]['nr_signals'].to_numpy(dtype = np.int32)
        # significant_droplets_mask = (tmp1 > np.quantile(tmp1, quantile_level_significant)) * 1
        # validity_mask *= significant_droplets_mask[:, None]

        # print(np.sum(validity_mask))

        tmp1 = feature_dataset[this_fr]['nr_cells'].to_numpy(dtype=np.float32)
        tmp2 = feature_dataset[next_fr]['nr_cells'].to_numpy(dtype=np.float32)
        celldiff_mat = np.abs(tmp1[:, None] - tmp2[None, :])
        # celldiff_mat /= np.sqrt(1.0 + np.abs(tmp1)[:, None] * np.abs(tmp2)[None, :])
        # celldiff_mat[np.logical_not(within_dist_mask)] = np.nan
        celldiff_thresh = 1
        validity_mask *= (np.logical_or(celldiff_mat <= celldiff_thresh, np.logical_and(celldiff_mat <= 2,
                                                                                        np.logical_or(
                                                                                            tmp1[:, None] >= 4,
                                                                                            tmp2[None, :] >= 4)))) * 1
        # celldiff_thresh = np.nanquantile(celldiff_mat, quantile_level_matching, axis = 1)
        # validity_mask *= (celldiff_mat <= celldiff_thresh[:, None]) * 1
        # validity_mask[np.logical_not(within_dist_mask)] = 0

        # cellmatching_mat = (((1 - 2 * (feature_dataset[this_fr]['nr_cells'].to_numpy(dtype = np.int32) > 0))[:, None] * (1 - 2 * (feature_dataset[next_fr]['nr_cells'].to_numpy(dtype = np.int32) > 0))[None, :]) == 1) * 1
        # validity_mask *= cellmatching_mat

        # cellexists_mat = (((feature_dataset[this_fr]['nr_cells'].to_numpy(dtype = np.int32) > 0)[:, None] * (feature_dataset[next_fr]['nr_cells'].to_numpy(dtype = np.int32) > 0)[None, :])) * 1
        # validity_mask *= cellexists_mat

        # signaldiff_mat = np.abs(feature_dataset[this_fr]['nr_signals'].to_numpy(dtype = np.float32)[:, None] - feature_dataset[next_fr]['nr_signals'].to_numpy(dtype = np.float32)[None, :])
        # signaldiff_mat /= np.sqrt(1.0 + np.abs(feature_dataset[this_fr]['nr_signals'].to_numpy(dtype = np.float32))[:, None] * np.abs(feature_dataset[next_fr]['nr_signals'].to_numpy(dtype = np.float32))[None, :])
        # signaldiff_mat[np.logical_not(within_dist_mask)] = np.nan
        # signaldiff_thresh = np.nanquantile(signaldiff_mat, quantile_level_matching, axis = 1)
        # validity_mask *= (signaldiff_mat <= signaldiff_thresh[:, None]) * 1
        # validity_mask[np.logical_not(within_dist_mask)] = 0
        # print(np.sum(validity_mask))

        # signalmatching_mat = (((1 - 2 * (feature_dataset[this_fr]['nr_signals'].to_numpy(dtype = np.int32) > 0))[:, None] * (1 - 2 * (feature_dataset[next_fr]['nr_signals'].to_numpy(dtype = np.int32) > 0))[None, :]) == 1) * 1
        # validity_mask *= signalmatching_mat

        # signalexists_mat = (((feature_dataset[this_fr]['nr_signals'].to_numpy(dtype = np.int32) > 0)[:, None] * (feature_dataset[next_fr]['nr_signals'].to_numpy(dtype = np.int32) > 0)[None, :])) * 1
        # validity_mask *= signalexists_mat
        # signalexists_now_mat = (feature_dataset[this_fr]['nr_signals'].to_numpy(dtype = np.int32) > 0) * 1
        # validity_mask *= signalexists_now_mat[:, None]

        radiusdiff_mat = np.abs(
            feature_dataset[this_fr]['radius'].to_numpy(dtype=np.int32)[:, None] - feature_dataset[next_fr][
                                                                                       'radius'].to_numpy(
                dtype=np.int32)[None, :])
        validity_mask *= (radiusdiff_mat <= 3) * 1
        # radiusdiff_thresh = np.quantile(radiusdiff_mat, quantile_level, axis = 1)
        # validity_mask *= (radiusdiff_mat <= radiusdiff_thresh[:, None]) * 1

        bf_corr_mat = similarity_matrix_bf_dapi[:, :, 0]
        bf_corr_mat[np.logical_not(within_dist_mask)] = np.nan
        bf_corr_thresh = np.nanquantile(bf_corr_mat, 1.0 - quantile_level_matching, axis=1)
        validity_mask *= (bf_corr_mat >= bf_corr_thresh[:, None]) * 1
        # validity_mask *= (bf_corr_mat >= 0.9) * 1

        dapi_corr_mat = similarity_matrix_bf_dapi[:, :, 1]
        dapi_corr_mat[np.logical_not(within_dist_mask)] = np.nan
        dapi_corr_thresh = np.nanquantile(dapi_corr_mat, 1.0 - quantile_level_matching, axis=1)
        validity_mask *= (dapi_corr_mat >= dapi_corr_thresh[:, None]) * 1
        # validity_mask *= (dapi_corr_mat >= 0.9) * 1

        intbrightdiff_mat = np.abs(
            feature_dataset[this_fr]['integrated_brightness'].to_numpy(dtype=np.float32)[:, None] -
            feature_dataset[next_fr]['integrated_brightness'].to_numpy(dtype=np.float32)[None, :])
        intbrightdiff_mat /= np.sqrt(
            np.abs(feature_dataset[this_fr]['integrated_brightness'].to_numpy(dtype=np.float32)[:, None]) * np.abs(
                feature_dataset[next_fr]['integrated_brightness'].to_numpy(dtype=np.float32)[None, :]))
        intbrightdiff_mat[np.logical_not(within_dist_mask)] = np.nan
        intbrightdiff_thresh = np.nanquantile(intbrightdiff_mat, quantile_level_matching, axis=1)
        validity_mask *= (intbrightdiff_mat <= intbrightdiff_thresh[:, None]) * 1

        maxbrightdiff_mat = np.abs(
            feature_dataset[this_fr]['max_brightness'].to_numpy(dtype=np.float32)[:, None] - feature_dataset[next_fr][
                                                                                                 'max_brightness'].to_numpy(
                dtype=np.float32)[None, :])
        maxbrightdiff_mat /= np.sqrt(
            np.abs(feature_dataset[this_fr]['max_brightness'].to_numpy(dtype=np.float32)[:, None]) * np.abs(
                feature_dataset[next_fr]['max_brightness'].to_numpy(dtype=np.float32)[None, :]))
        maxbrightdiff_mat[np.logical_not(within_dist_mask)] = np.nan
        maxbrightdiff_thresh = np.nanquantile(maxbrightdiff_mat, quantile_level_matching, axis=1)
        validity_mask *= (maxbrightdiff_mat <= maxbrightdiff_thresh[:, None]) * 1

        # print(np.sum(brightdiff_mat <= brightdiff_thresh[:, None]))
        # print(np.max(brightdiff_mat[within_dist_mask]))
        # print(np.median(brightdiff_mat[within_dist_mask]))
        # print(np.mean(brightdiff_mat[within_dist_mask]))
        # print(np.min(brightdiff_mat[within_dist_mask]))
        # assert(False)

        # brightdiff_mat = np.abs(feature_dataset[this_fr]['integrated_brightness'].to_numpy(dtype = np.float32)[:, None] - feature_dataset[next_fr]['integrated_brightness'].to_numpy(dtype = np.float32)[None, :])
        # brightdiff_mat /= np.sqrt(np.abs(feature_dataset[this_fr]['integrated_brightness'].to_numpy(dtype = np.float32)[:, None]) * np.abs(feature_dataset[next_fr]['integrated_brightness'].to_numpy(dtype = np.float32)[None, :]))
        # brightdiff_mat[np.logical_not(within_dist_mask)] = np.nan
        # brightdiff_thresh = np.nanquantile(brightdiff_mat, quantile_level_matching, axis = 1)
        # validity_mask *= (brightdiff_mat <= brightdiff_thresh[:, None]) * 1




        #hog_mat = np.zeros((feature_dataset[this_fr].shape[0], feature_dataset[next_fr].shape[0]))

        hog_prev = np.sum(np.stack(feature_dataset[this_fr]['hog'].values)**2, axis=1)
        hog_next = np.sum(np.stack(feature_dataset[next_fr]['hog'].values)**2, axis=1)
        hog_mat = np.stack(feature_dataset[this_fr]['hog'].values) @ np.stack(feature_dataset[next_fr]['hog'].values).T
        hog_prev = hog_prev.reshape(-1,1)
        dist = hog_prev - 2 * hog_mat + hog_next

        dist[np.logical_not(within_dist_mask)] = np.nan
        hog_thresh = np.nanquantile(dist, quantile_level_matching, axis=1)
        validity_mask *= (dist <= hog_thresh[:, None]) * 1

        validity_mask[np.logical_not(within_dist_mask)] = 0

        # print(np.sum(validity_mask))

        # sqdistance_matrix[within_dist_mask] = np.int64(sqdistance_matrix[within_dist_mask] * (maxbrightdiff_mat[within_dist_mask] + 1.0) * (intbrightdiff_mat[within_dist_mask] + 1.0))

        distance_matrix[within_dist_mask] = np.int64(
            distance_matrix[within_dist_mask] * (maxbrightdiff_mat[within_dist_mask] + 1.0) * (
                        intbrightdiff_mat[within_dist_mask] + 1.0) * (1.0 - bf_corr_mat[within_dist_mask] + 1.0) * (
                        1.0 - dapi_corr_mat[within_dist_mask] + 1.0))

        print("")
        # assigned, sinked = solve_assignment_with_sink(validity_mask, sqdistance_matrix, np.ones((validity_mask.shape[0],), dtype = np.int32) * maximal_distance ** 2)
        # Here we do some weird scaling shenanigans because the flow optimization algorithm used only accepts integer cost.
        assigned, sinked = solve_assignment_with_sink(validity_mask, np.int64(distance_matrix * 10),
                                                      np.ones((validity_mask.shape[0],),
                                                              dtype=np.int32) * maximal_distance * 10)
        print("Number of assignments found: " + str(assigned.shape[0]))
        print("Number of unassigned droplets: " + str(sinked.size))

        print("Assignments: ")
        print(assigned)

        print("Average distance:" + str(np.mean(np.sqrt(assigned[:, 2]))))
        print("Median distance:" + str(np.median(np.sqrt(assigned[:, 2]))))

        print("Mean possibilities per droplet: " + str(np.mean(np.sum(validity_mask, axis=1))))
        print("Percentage of droplets with possibilities: " + str(np.mean(np.sum(validity_mask, axis=1) > 0)))
        print("Mean possibilities per droplet for droplets with possibilities: " + str(
            np.mean(np.sum(validity_mask, axis=1)) / np.mean(np.sum(validity_mask, axis=1) > 0)))

        for ass in assigned:
            out.append({"framePrev": this_fr, "frameNext": next_fr, "dropletIdPrev": ass[0], "dropletIdNext": ass[1]})

        # violation_counter = np.zeros((feature_dataset[this_fr].shape[0], feature_dataset[next_fr].shape[0]), dtype = np.int32)
        # hascells_mat = np.int32(feature_dataset[this_fr]['nr_cells'].to_numpy(dtype = np.int32) > 0)[:, None] * np.int32(feature_dataset[next_fr]['nr_cells'].to_numpy(dtype = np.int32) > 0)[None, :]
        # violation_counter += (1 - hascells_mat)
        # radius_absdiff_mat = np.abs(feature_dataset[this_fr]['radius'].to_numpy(dtype = np.int32)[:, None] - feature_dataset[next_fr]['radius'].to_numpy(dtype = np.int32)[None, :])
        # violation_counter +=  np.int32(radius_absdiff_mat > 2)
        # nrcells_absdiff_mat = np.abs(feature_dataset[this_fr]['nr_cells'].to_numpy(dtype = np.int32)[:, None] - feature_dataset[next_fr]['nr_cells'].to_numpy(dtype = np.int32)[None, :])
        # violation_counter +=  np.int32(nrcells_absdiff_mat > 1)
        # nrsignals_absdiff_mat = np.abs(feature_dataset[this_fr]['nr_signals'].to_numpy(dtype = np.int32)[:, None] - feature_dataset[next_fr]['nr_signals'].to_numpy(dtype = np.int32)[None, :])
        # violation_counter +=  np.int32(nrsignals_absdiff_mat > 2)
        # bright_this_fr = feature_dataset[this_fr]['integrated_brightness'].to_numpy(dtype = np.float32)
        # bright_next_fr = feature_dataset[next_fr]['integrated_brightness'].to_numpy(dtype = np.float32)
        # bright_this_fr = bright_this_fr / np.sqrt(np.mean(bright_this_fr**2))
        # bright_next_fr = bright_next_fr / np.sqrt(np.mean(bright_next_fr**2))
        # bright_absdiff_mat = np.abs(bright_this_fr[:, None] - bright_next_fr[None, :])
        # violation_counter += np.int32(bright_absdiff_mat > 0.2)
        # robust_viable_mask = (violation_counter == 0)
        # robust_viable_rows_mask = (np.sum(robust_viable_mask, axis = 1) > 0)
        # # print(np.sum(robust_viable_rows_mask))
        # # robust_viable_relevant_mask = robust_viable_mask[robust_viable_rows_mask == 1, :]
        # distance_matrix = np.asarray([feature_dataset[this_fr]['center_row'].to_numpy(dtype = np.float64), feature_dataset[this_fr]['center_col'].to_numpy(dtype = np.float64)]).transpose()
        # distance_matrix = np.linalg.norm(distance_matrix[:, None, :] - np.asarray([feature_dataset[next_fr]['center_row'].to_numpy(dtype = np.float64), feature_dataset[next_fr]['center_col'].to_numpy(dtype = np.float64)]).transpose(), axis = 2)
        # distance_matrix = np.int64(distance_matrix ** 2)

        # assert(False)
    tracking_df = pd.DataFrame(out)
    result_feature_path = Path(RESULT_PATH / f"tracking_{image_name}.csv")
    tracking_df.to_csv(result_feature_path, index=False)
    path = Path(FEATURE_PATH / f"droplets_{image_name}.csv")
    droplet_table = pd.read_csv(path)
    create_final_output(droplet_table,tracking_df,nr_frames,RESULT_PATH,image_name)



def create_final_output(droplet_table,tracking_table,nr_frames,RESULT_PATH,image_name):
    traj = trajectory_expand_droplets(droplet_table, tracking_table, nr_frames)
    grp = list(traj.groupby(['trajectory_id']))
    final = np.zeros((len(grp), 2*nr_frames+1))
    for i in range(len(grp)):
        final[i][0] = i
        tmp = grp[i][1]
        for index, row in tmp.iterrows():
            final[i][2 * row['frame'] + 1] = row['center_row']
            final[i][2 * row['frame'] + 2] = row['center_col']

    cols = ['drop_id']
    for i in range(nr_frames):
        cols.append("x" + str(i + 1))
        cols.append("y" + str(i + 1))

    result = pd.DataFrame(final, columns=cols)

    result_feature_path = Path(RESULT_PATH / f"results_{image_name}.csv")
    result.to_csv(result_feature_path, index=False)




