import cv2 as cv
import numpy as np

# Takes in a blank image with points on circle
def circle_RANSAC(points_img, edge_img, rmin, rmax):
  np.random.seed(11000)

  s = points_img.shape
  diagonal = np.linalg.norm(np.asarray([s[0], s[1]]))
  tmp1 = np.uint8(points_img * 255)
  components, labels = cv.connectedComponents(tmp1, connectivity = 8, ltype = cv.CV_32S)
  cc_mask = np.ones(s)
  for i in range(components):
    if np.sum(labels == i) < 20:
      cc_mask[labels == i] = 0.0
  filtered_points_img = points_img * cc_mask

  nnz = np.sum(filtered_points_img != 0)
  canvas = np.zeros(s, dtype = np.float32)
  coords = np.transpose(np.asarray(np.where(filtered_points_img != 0.0)))
  # cv.imshow("test", filtered_points_img)
  # cv.waitKey(0)
  nr_of_samples = 800

  random_idxs = np.random.randint(0, nnz, size = (2, nr_of_samples), dtype = int)
  random_idxs[1, :] =  np.clip(random_idxs[1, :] + 1.0 * (random_idxs[1, :] >= random_idxs[0, :]), 0, nnz - 1)

  coords1 = coords[random_idxs[0, :]]
  coords2 = coords[random_idxs[1, :]]
  sample_distances = np.linalg.norm(coords1 - coords2, axis = 1)
  sample_viability = (sample_distances >= 20) * 1.0
  # print(sample_distances.shape)
  # print(sample_viability)
  midpoints = (coords1 + coords2) / 2.0
  # print(midpoints.shape)
  line_directions = coords1 - coords2
  line_directions = np.transpose(np.asarray([line_directions[:, 1], -line_directions[:, 0]]))
  line_directions = line_directions /  np.linalg.norm(line_directions, axis = 1)[:, None]
  # print(line_directions.shape)
  endpoints1 = np.int32(midpoints + line_directions * diagonal)
  endpoints2 = np.int32(midpoints - line_directions * diagonal)
  # print(endpoints2.shape)
  intensities = np.min(np.asarray((filtered_points_img[(coords1[:, 0], coords1[:, 1])], filtered_points_img[(coords2[:, 0], coords2[:, 1])])), axis = 0) * sample_viability
  # print(intensities.shape)

  drawn_lines = np.zeros((nr_of_samples, s[0], s[1]), dtype = np.float32)
  
  for i in range(nr_of_samples):
    cv.line(drawn_lines[i, :, :], np.flip(endpoints1[i, :]), np.flip(endpoints2[i, :]), color = int(intensities[i] * 255), thickness = 1, lineType = cv.LINE_AA)
  # print(drawn_lines.shape)
  canvas = np.sum(drawn_lines, axis = 0)
  canvas = canvas / canvas.max()
  # print(canvas.shape)
  # cv.imshow("test", canvas)
  # cv.waitKey(0)
  # assert(False)

  # for i in range(800):
  #   smpl1 = np.random.randint(0, nnz, dtype=int)
  #   smpl2 = np.random.randint(0, nnz - 1, dtype=int)
  #   if (smpl2 >= smpl1):
  #     smpl2 = smpl2 + 1
  # # for smpl1 in range(nnz):
  # #   for smpl2 in range(smpl1 + 1, nnz):
  #   smpl_dist = np.linalg.norm(coords[smpl1, :] - coords[smpl2, :])
  #   if (smpl_dist >= 20):
  #     midpoint = (coords[smpl1, :] + coords[smpl2, :]) / 2.0
  #     line_direction = coords[smpl1, :] - coords[smpl2, :]
  #     line_direction = np.asarray([line_direction[1], -line_direction[0]])
  #     line_direction = line_direction / np.linalg.norm(line_direction)
  #     endpoint1 = np.int32(midpoint + line_direction * diagonal)
  #     endpoint2 = np.int32(midpoint - line_direction * diagonal)
  #     drawn_line = np.zeros(s, dtype = np.float32)
  #     intensity = min(filtered_points_img[coords[smpl1, 0], coords[smpl1, 1]], filtered_points_img[coords[smpl2, 0], coords[smpl2, 1]])
  #     cv.line(drawn_line, np.flip(endpoint1), np.flip(endpoint2), color = int(intensity * 255), thickness = 1, lineType = cv.LINE_AA)
  #     canvas = canvas + drawn_line
  # canvas = cv.GaussianBlur(canvas, (3, 3), 0)
  # canvas = canvas / canvas.max()
  # cv.imshow("test", canvas)
  # cv.waitKey(0)
  # print(np.argmax(canvas))
  center = np.asarray(np.unravel_index(np.argmax(canvas, axis = None), s))

  best_rad = rmin
  best_corr = 0
  for i, rad in enumerate(range(rmin, rmax + 1)):
    mask = np.zeros(s, dtype = np.float32)
    cv.circle(mask, np.flip(center), rad, 2.0, 1)
    # mask = cv.GaussianBlur(mask, (3, 3), 0.5)
    # mask = mask / np.sum(mask)
    corr = np.sum(mask * filtered_points_img)
    if (corr > best_corr and rad >= rmin and rad <= rmax):
      best_corr = corr
      best_rad = rad
  return (center[0], center[1], best_rad)

def circle_RANSAC3(points_img, edge_img, rmin, rmax):
  np.random.seed(11000)
  s = points_img.shape

  tmp1 = np.uint8(points_img * 255)
  components, labels = cv.connectedComponents(tmp1, connectivity = 8, ltype = cv.CV_32S)
  cc_mask = np.ones(s)
  for i in range(components):
    if np.sum(labels == i) < 20:
      cc_mask[labels == i] = 0.0
  filtered_points_img = points_img * cc_mask

  diagonal = np.linalg.norm(np.asarray([s[0], s[1]]))
  nnz = np.sum(filtered_points_img == 1.0)
  canvas = np.zeros(s, dtype = np.float32)
  coords = np.transpose(np.asarray(np.where(filtered_points_img == 1.0)))
  # print(coords.shape)
  # print(coords)

  nr_of_samples = 500
  random_idxs = np.random.randint(0, nnz, size = (3, nr_of_samples), dtype = int)
  random_idxs[1, :] = np.random.randint(0, nnz - 1, size = nr_of_samples, dtype = int)
  random_idxs[2, :] = np.random.randint(0, nnz - 2, size = nr_of_samples, dtype = int)

  random_idxs[1, :] =  random_idxs[1, :] + 1 * (random_idxs[1, :] >= random_idxs[0, :])
  random_idxs[2, :] =  random_idxs[2, :] + 1 * (random_idxs[2, :] >= np.min(random_idxs[0: 2, :], axis = 0)) 
  random_idxs[2, :] =  random_idxs[2, :] + 1 * (random_idxs[2, :] >= np.max(random_idxs[0: 2, :], axis = 0))
  # random_idxs[2, :] =  np.clip(random_idxs[2, :] + (random_idxs[2, :] >= np.max(random_idxs[0: 2, :], axis = 0)), 0, nnz - 1)
  # random_idxs[1, :] = np.mod(random_idxs[1, :], nnz)
  # random_idxs[2, :] = np.mod(random_idxs[2, :], nnz)
  
  # assert(np.all(random_idxs[0, :] != random_idxs[1, :]))
  # assert(np.all(random_idxs[0, :] != random_idxs[2, :]))
  # assert(np.all(random_idxs[1, :] != random_idxs[2, :]))


  coords1 = coords[random_idxs[0, :]]
  coords2 = coords[random_idxs[1, :]]
  coords3 = coords[random_idxs[2, :]]
  # sample_distances1 = np.linalg.norm(coords1 - coords2, axis = 1)
  # sample_distances2 = np.linalg.norm(coords2 - coords3, axis = 1)
  # sample_viability = (sample_distances >= 20) * 1.0
  # print(sample_distances.shape)
  # print(sample_viability)
  midpoints1 = (coords1 + coords2) / 2.0
  midpoints2 = (coords2 + coords3) / 2.0

  line_directions1 = coords1 - coords2
  line_directions1 = np.transpose(np.asarray([line_directions1[:, 1], -line_directions1[:, 0]]))
  # assert(np.all(np.linalg.norm(line_directions1, axis = 1)[:, None] != 0.0))
  line_directions1 = line_directions1 /  np.linalg.norm(line_directions1, axis = 1)[:, None]
  line_directions2 = coords2 - coords3
  line_directions2 = np.transpose(np.asarray([line_directions2[:, 1], -line_directions2[:, 0]]))
  # assert(np.all(np.linalg.norm(line_directions2, axis = 1)[:, None] != 0.0))
  line_directions2 = line_directions2 /  np.linalg.norm(line_directions2, axis = 1)[:, None]

  sample_viability = (np.abs(np.sum(line_directions1 * line_directions2, axis = 1)) <= 0.8)
  # print(sample_viability)

  coords1 = coords1[sample_viability, :]
  coords2 = coords2[sample_viability, :]
  coords3 = coords3[sample_viability, :]
  midpoints1 = midpoints1[sample_viability, :]
  midpoints2 = midpoints2[sample_viability, :]
  line_directions1 = line_directions1[sample_viability, :]
  line_directions2 = line_directions2[sample_viability, :]

  cs = midpoints1 - midpoints2
  # print(cs.shape)

  As = np.transpose(np.asarray([-np.transpose(line_directions1), np.transpose(line_directions2)]), (2, 1, 0))

  # print(As.shape)

  results = np.linalg.solve(As, cs)
  # print(results.shape)
  centers = (midpoints1 + line_directions1 * results[:, 0, None] + midpoints2 + line_directions2 * results[:, 1, None]) / 2.0
  # print(centers.shape)
  radii = (np.linalg.norm(centers - coords1, axis = 1) + np.linalg.norm(centers - coords2, axis = 1) + np.linalg.norm(centers - coords3, axis = 1)) / 3.0
  # print(radii.shape)
  distances_to_centers = np.transpose(np.linalg.norm(centers - coords[:, None, :], axis = 2))
  # print(distances_to_centers.shape)
  distance_within_rad = np.logical_and(distances_to_centers <= radii[:, None] + 1.0, distances_to_centers >= radii[:, None] - 1.0)
  # print(distance_within_rad.shape)
  score = np.sum(distance_within_rad, axis = 1)
  # print(score)
  winner = np.argmax(score)
  center_to_return = centers[winner, :]
  radius_to_return = round(radii[winner])
  # print(center_to_return)
  # print(radius_to_return)
  # assert(False)

  # print(0.5)
  # print(int(0.5))
  # print(0.9)
  # print(int(0.9))
  # print(0.2)
  # print(int(0.2))
  # print(np.asarray((0.2, 0.5, 0.9)))
  # print(np.int32(np.asarray((0.2, 0.5, 0.9))))
  # assert(False)

  return(round(center_to_return[0]), round(center_to_return[1]), radius_to_return)


  # best_corr = 0
  # best_rad = rmin
  # best_center = np.asarray([int((s[0] - 1) / 2), int((s[1] - 1) / 2)], dtype = np.int32)
  # for i in range(500):
  #   smpl1 = np.random.randint(0, nnz, dtype=int)
  #   smpl2 = np.random.randint(0, nnz - 1, dtype=int)
  #   if (smpl2 >= smpl1):
  #     smpl2 = smpl2 + 1
  #   smpl3 = np.random.randint(0, nnz - 2, dtype=int)
  #   if (smpl3 >= max(smpl2, smpl1)):
  #     smpl3 = smpl3 + 2
  #   elif (smpl3 >= smpl1 or smpl3 >= smpl2):
  #     smpl3 = smpl3 + 1
  #   midpoint1 = (coords[smpl1, :] + coords[smpl2, :]) / 2.0
  #   midpoint2 = (coords[smpl2, :] + coords[smpl3, :]) / 2.0

  #   line_direction1 = coords[smpl1, :] - coords[smpl2, :]
  #   line_direction1 = np.asarray([line_direction1[1], -line_direction1[0]])
  #   line_direction1 = line_direction1 / np.linalg.norm(line_direction1)

  #   line_direction2 = coords[smpl2, :] - coords[smpl3, :]
  #   line_direction2 = np.asarray([line_direction2[1], -line_direction2[0]])
  #   line_direction2 = line_direction2 / np.linalg.norm(line_direction2)

  #   if (abs(np.sum(line_direction1 * line_direction2)) < 0.7):
  #     c = midpoint1 - midpoint2
  #     A = np.transpose(np.asarray([-line_direction1, line_direction2]))
  #     res = np.linalg.solve(A, c)
  #     center = np.int32((midpoint1 + res[0] * line_direction1 + midpoint2 + res[1] * line_direction2) / 2)
  #     radius = np.int32((np.linalg.norm(center - coords[smpl1, :]) + np.linalg.norm(center - coords[smpl2, :]) + np.linalg.norm(center - coords[smpl3, :])) / 3)

  #     hypothesis = np.zeros(s, dtype = np.float32)
  #     cv.circle(hypothesis, center, radius, 1.0, 1)
  #     # hypothesis = cv.GaussianBlur(hypothesis, (5, 5), 0)
  #     hypothesis = hypothesis / np.sum(hypothesis)
  #     corr = np.sum(hypothesis * points_img)
  #     # cv.imshow("test", hypothesis * 10 + points_img)
  #     # cv.waitKey(0)

  #     if (corr > best_corr and radius >= rmin and radius <= rmax):
  #       best_corr = corr
  #       best_rad = radius
  #       best_center = center

  # return (best_center[0], best_center[1], best_rad)
    

