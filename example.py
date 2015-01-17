# -*- coding: UTF-8 -*-
import maxflow
import csv
import numpy as np
import os
import math


def get_foot_point(startx, starty, endx, endy, x, y):
    u = ((endx - startx) * (endx - startx) + (float(endy)-starty) * (endy-starty))
    u = ((endx - startx) * (endx - x) + (endy - starty) * (endy-y)) / u
    footx = u*startx + (1 - u) * endx
    footy = u*starty + (1 - u) * endy
    return footx, footy


def min_cut(voxel_code_array, pole_position_array, intensity_array, point_counts_array):
    """
    segment the voxels in to n segments based on the position array using min cut algorithm

    the original min cut algorithm comes from the paper:
    "An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision."Yuri Boykov and
    Vladimir Kolmogorov. In IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), September 2004

    Args:
        voxel_code_array: the voxels to be segmented
        position_array: the position label of the voxels, the label could be more than 2
    Return:

    """
    # K in the article
    k = 3.40282e+038
    voxel_length = len(voxel_code_array)
    reached_flag = [False] * voxel_length
    voxel_count = 0

    intensity_var = np.var(np.vectorize(int)(intensity_array))
    point_count_var = np.var(np.vectorize(int)(point_counts_array))
    g = maxflow.GraphFloat()
    nodes = g.add_nodes(voxel_length)

    positions = pole_position_array[pole_position_array != '0']
    unique_positions = list(set(positions))

    first_positions = voxel_code_array[pole_position_array == unique_positions[0]]
    second_positions = voxel_code_array[pole_position_array == unique_positions[1]]

    center_x1 = int(first_positions[0][0:4])
    center_y1 = int(first_positions[0][4:8])

    center_x2 = int(second_positions[0][0:4])
    center_y2 = int(second_positions[0][4:8])

    distance = ((center_x1 - center_x2)**2 + (center_y1 - center_y2)**2)**0.5

    while voxel_count < voxel_length:
        center_x = int(voxel_code_array[voxel_count][0:4])
        center_y = int(voxel_code_array[voxel_count][4:8])
        foot_x, foot_y = get_foot_point(center_x1, center_y1, center_x2, center_y2, center_x, center_y)
        # d1 = ((center_x - center_x1)**2 + (center_y - center_y1)**2)**0.5
        # d2 = ((center_x - center_x2)**2 + (center_y - center_y2)**2)**0.5
        d1 = ((foot_x - center_x1)**2 + (foot_y - center_y1)**2)**0.5
        d2 = ((foot_x - center_x2)**2 + (foot_y - center_y2)**2)**0.5

        v = (foot_x - center_x1) * (foot_x - center_x2) + (foot_y - center_y1) * (foot_y - center_y2)

        if voxel_code_array[voxel_count] in first_positions:
            g.add_tedge(nodes[voxel_count], k, 0)
        elif voxel_code_array[voxel_count] in second_positions:
            g.add_tedge(nodes[voxel_count], 0, k)
        elif v >= 0:
            if d1 > d2:
                g.add_tedge(nodes[voxel_count], 0, k)
            else:
                g.add_tedge(nodes[voxel_count], k, 0)
        else:
            g.add_tedge(nodes[voxel_count], -math.log(d1 / distance)*0.02, -math.log(d2 / distance)*0.02)

        reached_flag[voxel_count] = True
        right = "{:0>4d}".format(int(voxel_code_array[voxel_count][0:4]) + 1) + voxel_code_array[voxel_count][4:12]
        left = "{:0>4d}".format(int(voxel_code_array[voxel_count][0:4]) - 1) + voxel_code_array[voxel_count][4:12]
        front = voxel_code_array[voxel_count][0:4] + "{:0>4d}".format(int(voxel_code_array[voxel_count][4:8]) + 1) +\
                voxel_code_array[voxel_count][8:12]
        back = voxel_code_array[voxel_count][0:4] + "{:0>4d}".format(int(voxel_code_array[voxel_count][4:8]) - 1) +\
               voxel_code_array[voxel_count][8:12]
        up = voxel_code_array[voxel_count][0:8] + "{:0>4d}".format(int(voxel_code_array[voxel_count][8:12]) + 1)
        down = voxel_code_array[voxel_count][0:8] + "{:0>4d}".format(int(voxel_code_array[voxel_count][8:12]) - 1)
        neighbor_list = [right, left, front, back, up, down]
        for neighbor in neighbor_list:
            indice = np.where(np.array(voxel_code_array) == neighbor)
            if len(indice[0]) == 0:
                continue
            indice = indice[0][0]
            if reached_flag[indice] is False:
                intensity_dif = math.fabs(int(intensity_array[indice]) - int(intensity_array[voxel_count]))
                point_count_dif = math.fabs(int(point_counts_array[indice]) - int(point_counts_array[voxel_count]))
                smoothcost = math.exp(-point_count_dif**2 / (2 * point_count_var))
                g.add_edge(nodes[voxel_count], nodes[indice], smoothcost, smoothcost)
        voxel_count += 1
    flow = g.maxflow()
    return map(g.get_segment, nodes)


with open('cut_0.2_5_4.csv') as incsv:
    reader = csv.reader(incsv)
    lines = [[row[0], row[1], row[2], row[3], row[4], row[5], row[6]] for row in reader]
original_code_array = np.array(lines)[:, 0]
point_counts_array = np.array(lines)[:, 1]
intensity_array = np.array(lines)[:, 2]
olocation_array = np.array(lines)[:, 3]
mlocation_array = np.array(lines)[:, 4]
pole_id_array = np.array(lines)[:, 5]
back_fore_array = np.array(lines)[:, 6]

foreground_indices = np.where(back_fore_array != '0')
segments = min_cut(original_code_array[foreground_indices], pole_id_array[foreground_indices],
                   intensity_array[foreground_indices], point_counts_array[foreground_indices])

segments_array = np.array([0] * len(original_code_array))
segments_array[foreground_indices[0][np.array(segments) == 1]] = 2
segments_array[foreground_indices[0][np.array(segments) == 0]] = 1
import laspy
file = laspy.file.File('cut_0.2_5.las', mode='rw')
file.pt_src_id[:] = 0
count = 0
for voxel_index in file.voxel_index:
    file.pt_src_id[count] = segments_array[voxel_index] + 1
    count += 1
file.close()
print('Done')
os.system("pause")