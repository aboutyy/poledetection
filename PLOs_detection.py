# -*- coding: UTF-8 -*-
__author__ = 'Administrator'
# lidar_classification.py
import laspy
import time
import os
import csv
import numpy as np
import scipy
from scipy import linalg as la
import scipy.spatial
OPTIMAL_DIMENSION_NAME = 'optimal_dimensionalities'
OPTIMAL_NX_NAME = 'optimal_nx'
OPTIMAL_NY_NAME = 'optimal_ny'
OPTIMAL_NZ_name = 'optimal_nz'
OPTIMAL_PX_NAME = 'optimal_px'
OPTIMAL_PY_NAME = 'optimal_py'
OPTIMAL_PZ_NAME = 'optimal_pz'
OPTIMAL_RADIUS_NAME = 'optimal_radius'
VOXEL_INDEX = 'voxel_index'
LOCATION = 'olocation'


def add_dimension(infile_path, outfile_path, names, types, descriptions):
    """
    add new dimensions to the las file

    Args:
        names: names array of the dimensions
        types: types array of the dimensions
                0	Raw Extra Bytes	Value of “options”
                1	unsigned char	1 byte
                2	Char	1 byte
                3	unsigned short	2 bytes
                4	Short	2 bytes
                5	unsigned long	4 bytes
                6	Long	4 bytes
                7	unsigned long long	8 bytes
                8	long long	8 bytes
                9	Float	4 bytes
                10	Double	8 bytes
                11	unsigned char[2]	2 byte
                12	char[2]	2 byte
                13	unsigned short[2]	4 bytes
                14	short[2]	4 bytes
                15	unsigned long[2]	8 bytes
                16	long[2]	8 bytes
                17	unsigned long long[2]	16 bytes
                18	long long[2]	16 bytes
                19	float[2]	8 bytes
                20	double[2]	16 bytes
                21	unsigned char[3]	3 byte
                22	char[3]	3 byte
                23	unsigned short[3]	6 bytes
                24	short[3]	6 bytes
                25	unsigned long[3]	12 bytes
                26	long[3]	12 bytes
                27	unsigned long long[3]	24 bytes
                28	long long[3]	24 bytes
                29	float[3]	12 bytes
                30	double[3]	24 bytes
        description: discription of the dimension
    Returns:
        None
    """
    infile = laspy.file.File(infile_path, mode="r")
    outfile = laspy.file.File(outfile_path, mode="w", header=infile.header)
    exist_names = []
    for dimension in infile.point_format:
        exist_names.append(dimension.name)
    for name, datatype, description in zip(names, types, descriptions):
        if exist_names.count(name) != 0:
            print "dimension %s already exist!!", name
            continue
        outfile.define_new_dimension(name, datatype, description)
    for dimension in infile.point_format:
        data = infile.reader.get_dimension(dimension.name)
        outfile.writer.set_dimension(dimension.name, data)
        exist_names.append(dimension.name)
    infile.close()
    outfile.close()


def voxelization(infile_path, outfile_path, voxel_size):
    """
    voxelization of point cloud, save the voxel-first point as a file and the point-index to las file

    Args:
        infile_path: the point cloud file *.las
        outfile_path: the ASCII file that save the voxel-first point pair values, line number is voxel index and value
            is point index
        voxel_size: voxel size for voxelization

    Returns:
        None
    """

    infile = laspy.file.File(infile_path, mode="rw")

    # 计算每个点的voxel码
    scaled_x = np.vectorize(int)((1 / voxel_size) * (infile.x - infile.header.min[0]))
    scaled_y = np.vectorize(int)((1 / voxel_size) * (infile.y - infile.header.min[1]))
    scaled_z = np.vectorize(int)((1 / voxel_size) * (infile.z - infile.header.min[2]))
    indices = np.lexsort((scaled_z, scaled_y, scaled_x))
    voxel_count = 0
    point_count = 0
    point_lengh = len(infile.x)

    # the array to store the code of the voxel, this is actually the row, columm and height number of the voxel
    code_array = []

    # the array to store the point number in each voxel
    points_in_one_voxel_array = []
    intensity_in_one_voxel_array = []
    while point_count < point_lengh - 1:

        # the counter of points number in one voxel
        points_in_one_voxel_count = 1
        intensity_in_one_voxel_count = 0
        # loop of finding points with same code
        while point_count < point_lengh - 1 and \
                        scaled_x[indices[point_count + 1]] == scaled_x[indices[point_count]] and \
                        scaled_y[indices[point_count + 1]] == scaled_y[indices[point_count]] and \
                        scaled_z[indices[point_count + 1]] == scaled_z[indices[point_count]]:
            # add a voxel index label to the point
            infile.voxel_index[indices[point_count]] = voxel_count
            intensity_in_one_voxel_count += infile.intensity[indices[point_count]]
            point_count += 1
            points_in_one_voxel_count += 1

        infile.voxel_index[indices[point_count]] = voxel_count
        intensity_in_one_voxel_count += infile.intensity[indices[point_count]]
        intensity_in_one_voxel_array.append(intensity_in_one_voxel_count / points_in_one_voxel_count)
        points_in_one_voxel_array.append(points_in_one_voxel_count)
        # save the code to an array which later will be stored in the csv file
        code = "{:0>4d}".format(scaled_x[indices[point_count]]) + \
               "{:0>4d}".format(scaled_y[indices[point_count]]) + \
               "{:0>4d}".format(scaled_z[indices[point_count]])
        code_array.append(code)
        point_count += 1
        voxel_count += 1
    # save the code to the csv file sequentially
    code_array_length = len(code_array)
    with open(outfile_path, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        count = 0
        while count < code_array_length:
            writer.writerow([code_array[count], points_in_one_voxel_array[count], intensity_in_one_voxel_array[count]])
            count += 1


def original_localization(csv_path, csv_out_path, voxel_size, min_position_height):
    """
    the original step of finding a potential location of an object in the voxel

    recognize potential object location in all the voxels by judging if one horizontal location has enough
    vertically connected voxels
    Args:
        infile_path: the filepath of the original voxel file, the *.voxel file
        outfile_path: the filepath of the processed voxel file, add a column to the original file which label
                      the potential location ID of the voxel, if the voxel is not a location then the value
                      will be 0
        min_position_height: the minimun height of continuous connected voxel which specify an object
    Returns:
        None
    """

    with open(csv_path, 'rb') as in_csv_file:
        reader = csv.reader(in_csv_file)
        line = [[row[0], row[1], row[2]] for row in reader]
    voxel_code_array = np.array(line)[:, 0]
    points_count_array = np.array(line)[:, 1]
    intensity_array = np.array(line)[:, 2]

    # parse the string array to integer array for later calculation
    voxel_code_int_array = np.vectorize(long)(voxel_code_array)

    # counter of the voxel
    voxel_count = 0
    location_count = 1
    length = len(voxel_code_array)

    # the array that stores the location label of the voxel, if the voxel does not belong to a voxel the value is zero
    location_id_array = np.array([0] * length)

    vertical_count_threshold = int(min_position_height / voxel_size)
    temp = [1] * vertical_count_threshold
    temp1 = [1] * 2
    # traverse all the voxel to assign location label
    while voxel_count < length - vertical_count_threshold:
        v1 = voxel_code_int_array[voxel_count:voxel_count + vertical_count_threshold]
        v2 = voxel_code_int_array[voxel_count + 1:voxel_count + 1 + vertical_count_threshold]
        v3 = voxel_code_int_array[voxel_count:voxel_count + 1]
        v4 = voxel_code_int_array[voxel_count + 1:voxel_count + 2]
        v = list(map(lambda x: x[0] - x[1], zip(v2, v1)))
        # judge if the vertical_count_threshold number of voxel value are continuous
        if v == temp:
            location_id_array[voxel_count:voxel_count + 1 + vertical_count_threshold] = location_count
            voxel_count += vertical_count_threshold + 1
            while voxel_code_int_array[voxel_count] - voxel_code_int_array[voxel_count - 1] == 1 \
                    and voxel_count < length:
                location_id_array[voxel_count] = location_count
                voxel_count += 1
            location_count += 1
            # when the voxels after more than threshold number of voxels are not continuous
            # then these voxels are considered as no-location voxel

            change_flag = int(voxel_code_int_array[voxel_count] / 10000) - \
                                        int(voxel_code_int_array[voxel_count - 1] / 10000)
            while change_flag == 0 and voxel_count < length - 1:
                voxel_count += 1
                change_flag = int(voxel_code_int_array[voxel_count] / 10000) - \
                              int(voxel_code_int_array[voxel_count - 1] / 10000)
        # 1. when the voxels in one location has less than threshold number of continuous voxel
        # 2. when the voxels after more than threshold number of voxels are not continuous
        # then these voxels are considered as no-location voxel
        # 此处可以测试是否第二种情况可以间隔几个voxel也算location，这样可以解决稀疏情况下的问题
        # 如果存在连续三个voxel在同一x，y处的话，则不标记，继续，不是背景点也不是物体点
        elif list(map(lambda x: x[0] - x[1], zip(v4, v3))) == [1, 1]:
            voxel_count += 3
        # 不是物体点则被标记为背景点
        else:
            voxel_count += 1
            change_flag = int(voxel_code_int_array[voxel_count] / 10000) - \
                                        int(voxel_code_int_array[voxel_count - 1] / 10000)
            while change_flag == 0 and voxel_count < length - 1:
                voxel_count += 1
                change_flag = int(voxel_code_int_array[voxel_count] / 10000) - \
                              int(voxel_code_int_array[voxel_count - 1] / 10000)

    with open(csv_out_path, 'wb') as out_csv_file:
        csvwriter = csv.writer(out_csv_file)
        row = 0
        while row < length:
            csvwriter.writerow([voxel_code_array[row], points_count_array[row], intensity_array[row],
                                location_id_array[row]])
            row += 1


def merging_neighbor_location(csv_path, csv_out_path):
    """
    merge neighbor location to form the location of an object, each location is then corresponding to each object

    merge the neighboring location of the original location, the merged location is then labeled to each point in the
    las file
    Args:
        csvfile: the csv file which stores the code and the original location label
    Returns:
        None
    """
    with open(csv_path, 'rb+') as in_csv_file:
        reader = csv.reader(in_csv_file)
        line = [[row[0], row[1], row[2], row[3]] for row in reader]
    original_code_array = np.array(line)[:, 0]
    point_counts_array = np.array(line)[:, 1]
    intensity_array = np.array(line)[:, 2]
    olocation_array = np.array(line)[:, 3]

    # get the first 8 char of the code list
    original_code_array_eight = np.array(map(lambda x: x[:8], original_code_array))

    # only save the code that has be selected as an olocation
    code_with_location_array = original_code_array_eight[olocation_array != '0']

    # merged location counter
    mlocation_count = 1

    # remove the repetitive code in code_with_location_array
    unique_location_code_array = list(set(code_with_location_array))
    unique_location_code_array.sort(key=list(code_with_location_array).index)

    unique_code_length = len(unique_location_code_array)
    # each unique 8bit code correspond to a mloacation
    mlocation_array = [0] * len(unique_location_code_array)
    mlocation_array = np.array(mlocation_array)

    flag_array = [False] * len(unique_location_code_array)
    flag_array = np.array(flag_array)

    x_array = map(lambda x: x[:4], unique_location_code_array)
    y_array = map(lambda x: x[-4:], unique_location_code_array)
    x_int_array = np.vectorize(int)(x_array)
    y_int_array = np.vectorize(int)(y_array)

    # the loop for finding all the merged locations
    while False in flag_array:
        idx = list(flag_array).index(False)
        # the list to store
        location_list = [idx]
        new_location_list = location_list[:]
        current_length = 0
        # the loop for finding a merged location
        while len(location_list) != current_length:
            current_length = len(location_list)
            # find the neighbors of the new_location member
            neighbors_list = []
            for item in new_location_list:
                # find the neighbors of one code
                x_valid = np.logical_and((x_int_array - x_int_array[item] < 2),(x_int_array - x_int_array[item] > -2))
                y_valid = np.logical_and((y_int_array - y_int_array[item] < 2),(y_int_array - y_int_array[item] > -2))
                neighbors = np.where(np.logical_and(x_valid, y_valid))
                neighbors_list += list(neighbors[0])
                neighbors_list = list(set(neighbors_list))
            #     count = item + 1
            #     neighbors = []
            #     if count >= unique_code_length:
            #         break
            #     while count < unique_code_length - 1 and x_int_array[count] - x_int_array[item] < 2:
            #         if y_int_array[count] - y_int_array[item] < 2:
            #             neighbors.append(count)
            #             count += 1
            #         else:
            #             count += 1
            #     neighbors_list += neighbors
            # # update new location list
            new_location_list = []
            for item in neighbors_list:
                if item not in location_list:
                    new_location_list.append(item)
                    location_list.append(item)
            # sort the new location to abandon the repetetive codes and to find neighbor just in one direction(right and
            # downword)
            # new_location_list.sort()
        # when we find out all the olocations, we merge them to a mlocation
        flag_array[location_list] = True
        mlocation_array[location_list] = mlocation_count
        mlocation_count += 1

    mlocation_all_array =np.array([0] * len(original_code_array))
    for code, mlocation in zip(unique_location_code_array, mlocation_array):
        indices = np.logical_and((original_code_array_eight == code), (olocation_array != '0'))
        mlocation_all_array[indices] = mlocation
    row = 0
    length = len(original_code_array)
    with open(csv_out_path, 'wb') as out_csv_file:
        writer = csv.writer(out_csv_file)
        while row< length:
            writer.writerow([original_code_array[row], point_counts_array[row], intensity_array[row],
                             olocation_array[row], mlocation_all_array[row]])
            row += 1


def connected_component_labeling(voxelset, radius):
    label_array = np.array([0] * len(voxelset))
    parent_array = np.array([0] * len(voxelset))
    label_count = 0
    voxel_count = 0
    tree = scipy.spatial.cKDTree(voxelset)
    # first pass
    for voxel in voxelset:
        # finding the neighbors of each voxel of foreground
        neighbors = tree.query_ball_point(voxel, radius)
        # the situation that voxel is isolated or surrounded by non-labeled voxels
        if len(neighbors) == 0 or list(label_array[neighbors]) == [0] * len(neighbors):
            label_count += 1
            label_array[voxel_count] = label_count
            parent_array[voxel_count] = label_count
        # the situation that the voxel is surrounded by same voxels
        elif len(set(label_array[neighbors])) == 1:
            label_array[voxel_count] = label_array[neighbors[0]]
        elif len(set(label_array[neighbors])) == 2 and list(label_array[neighbors]).count(0) != 0:
            neighbors_labels = label_array[neighbors]
            label_array[voxel_count] = neighbors_labels[neighbors_labels != 0][0]
            parent_array[voxel_count] = neighbors_labels[neighbors_labels != 0][0]
        # the situation that the voxel is surrounded by different voxels, we assign min neighbor's parent neighbor to
        # current voxel and parent
        else:
            if list(label_array[neighbors]).count(0) == 0:
                label_array[voxel_count], parent_array[voxel_count] = \
                    min(label_array[neighbors]), min(label_array[neighbors])
            else:
                new_set = set(label_array[neighbors])
                new_set.remove(0)
                label_array[voxel_count], parent_array[voxel_count] = min(new_set), min(new_set)
            for item in new_set:
                if item != parent_array[voxel_count]:
                    parent_array[label_array == item] = parent_array[voxel_count]
        voxel_count += 1
    # second pass
    pattern_count = 1
    parent_set = set(parent_array)
    label_array[:] = 0
    for unique_parent in parent_set:
        unique_parent_indices = np.where(parent_array == unique_parent)[0]
        if len(label_array[unique_parent_indices]) > 200:
            label_array[parent_array == unique_parent] = pattern_count
            pattern_count += 1
    return label_array


def remove_background_region_growing(in_file, outfile_path, fixed_radius, normal_threshold):
    """
    存有区域增长的代码，以后备用
    romove the back ground based on the vertical continuous height of each x,y on xy plane
    if the continious height
    """

    with open(in_file, 'rb') as incsv:
        reader = csv.reader(incsv)
        lines = [[row[0], row[1], row[2]] for row in reader]
    original_code_array = np.array(lines)[:, 0]
    original_x_int_array = np.vectorize(int)(map(lambda x: x[:4], original_code_array))
    original_y_int_array = np.vectorize(int)(map(lambda x: x[4:8], original_code_array))
    original_z_int_array = np.vectorize(int)(map(lambda x: x[8:12], original_code_array))

    points_count_array = np.array(lines)[:, 1]
    intensity_array = np.array(lines)[:, 2]

    normal_list = []
    dataset = np.vstack([original_x_int_array, original_y_int_array, original_z_int_array]).transpose()
    tree = scipy.spatial.cKDTree(dataset)
    for x, y, z in zip(original_x_int_array, original_y_int_array, original_z_int_array):
        indices = tree.query_ball_point([x, y, z], fixed_radius)
        if len(indices) <= 3:
            normal_list.append(0)
            continue
        idx = tuple(indices)
        data = np.vstack([dataset[idx, 0], dataset[idx, 1], dataset[idx, 2]])
        cov = np.cov(data)
        evals, evects = la.eigh(cov)
        evals = np.abs(evals)
        index = evals.argsort()[::-1]
        evects = evects[:, index]
        normal = evects[2][2]
        normal_list.append(normal)

    # codes below were region growing algorithm implemented based pseudocode in
    # http://pointclouds.org/documentation/tutorials/region_growing_segmentation.php#region-growing-segmentation
    # awailable voxel list index
    seeds = np.where(np.logical_or(np.array(normal_list) > 0.7, np.array(normal_list) < -0.7))[0]
    # seeds = original_z_int_array[seeds].argsort()
    seeds = list(seeds)
    dataset1 = np.vstack([original_x_int_array[seeds], original_y_int_array[seeds],
                          original_z_int_array[seeds]]).transpose()
    tree1 = scipy.spatial.cKDTree(dataset1)
    # region list
    regions = []
    while len(seeds) > 0:
        current_region_voxels = []
        current_seeds = []
        # voxel with lowest z value
        lowest_voxel = seeds[0]
        current_seeds.append(lowest_voxel)
        current_region_voxels.append(lowest_voxel)
        del seeds[0]
        count = 0
        while count < len(current_seeds):
            current_seed = current_seeds[count]
            count += 1
            current_seed_neighbors = \
                tree1.query_ball_point([original_x_int_array[current_seed], original_y_int_array[current_seed],
                                      original_z_int_array[current_seed]], fixed_radius)
            for neighbor in current_seed_neighbors:
                if seeds.count(neighbor) != 0:
                        if current_region_voxels.count(neighbor) == 0:
                            current_region_voxels.append(neighbor)
                        seeds.remove(neighbor)
                        if current_seeds.count(neighbor) == 0:
                            current_seeds.append(neighbor)
        regions.append(current_region_voxels)
    length = len(original_code_array)
    ground_list = [0] * length
    region_count = 1
    for region in regions:
        for voxel in region:
            ground_list[voxel] = region_count
        region_count += 1
    with open(outfile_path, 'wb') as out_csv_file:
        csvwriter = csv.writer(out_csv_file)
        row = 0
        while row < length:
            csvwriter.writerow([original_code_array[row], points_count_array[row], intensity_array[row],
                                normal_list[row], ground_list[row]])
            row += 1


def region_growing(voxelset, radius):
    # codes below were region growing algorithm implemented based pseudocode in
    # http://pointclouds.org/documentation/tutorials/region_growing_segmentation.php#region-growing-segmentation
    # awailable voxel list index

    tree1 = scipy.spatial.cKDTree(voxelset)
    length = len(voxelset)
    voxel_indices = range(length)
    seed_length = len(voxel_indices)
    # region list
    regions = []
    while seed_length > 0:
        current_region_voxels = []
        current_seeds = []
        # voxel with lowest z value
        lowest_voxel_indice = voxel_indices[0]
        current_seeds.append(lowest_voxel_indice)
        current_region_voxels.append(lowest_voxel_indice)
        del voxel_indices[0]
        count = 0
        while count < len(current_seeds):
            current_seed = current_seeds[count]
            count += 1
            current_seed_neighbors = \
                tree1.query_ball_point([voxelset[:, 0][current_seed], voxelset[:, 1][current_seed],
                                        voxelset[:, 2][current_seed]], radius)
            for neighbor in current_seed_neighbors:
                if voxel_indices.count(neighbor) != 0:
                        if current_region_voxels.count(neighbor) == 0:
                            current_region_voxels.append(neighbor)
                        voxel_indices.remove(neighbor)
                        if current_seeds.count(neighbor) == 0:
                            current_seeds.append(neighbor)
        regions.append(current_region_voxels)
        seed_length = len(voxel_indices)
    return regions


def remove_background(in_file, outfile_path, fixed_radius, normal_threshold):
    """
    romove the back ground based on the vertical continuous height of each x,y on xy plane

    if the continious height
    """
    with open(in_file, 'rb') as incsv:
        reader = csv.reader(incsv)
        lines = [[row[0], row[1], row[2], row[3], row[4]] for row in reader]
    original_code_array = np.array(lines)[:, 0]
    length = len(original_code_array)
    original_x_int_array = np.vectorize(int)(map(lambda x: x[:4], original_code_array))
    original_y_int_array = np.vectorize(int)(map(lambda x: x[4:8], original_code_array))
    original_z_int_array = np.vectorize(int)(map(lambda x: x[8:12], original_code_array))

    points_count_array = np.array(lines)[:, 1]
    intensity_array = np.array(lines)[:, 2]
    olocation_array = np.array(lines)[:, 3]
    mlocation_array = np.array(lines)[:, 4]
    normal_list = []
    dataset = np.vstack([original_x_int_array, original_y_int_array, original_z_int_array]).transpose()
    tree = scipy.spatial.cKDTree(dataset)
    for x, y, z in zip(original_x_int_array, original_y_int_array, original_z_int_array):
        indices = tree.query_ball_point([x, y, z], fixed_radius)
        if len(indices) <= 3:
            normal_list.append(0)
            continue
        idx = tuple(indices)
        data = np.vstack([dataset[idx, 0], dataset[idx, 1], dataset[idx, 2]])
        cov = np.cov(data)
        evals, evects = la.eigh(cov)
        evals = np.abs(evals)
        index = evals.argsort()[::-1]
        evects = evects[:, index]
        normal = evects[2][2]
        normal_list.append(normal)

    # codes below were region growing algorithm implemented based pseudocode in
    # http://pointclouds.org/documentation/tutorials/region_growing_segmentation.php#region-growing-segmentation
    # awailable voxel list index
    seeds = np.where(np.logical_or(np.array(normal_list) > normal_threshold, np.array(normal_list) <
                                   -normal_threshold))[0]
    # seeds = original_z_int_array[seeds].argsort()
    seeds = list(seeds)
    voxel_set = np.vstack([original_x_int_array[seeds], original_y_int_array[seeds],
                          original_z_int_array[seeds]]).transpose()
    regions = region_growing(voxel_set, 1.5)
    # regions = connected_component_labeling(voxel_set, 1.5)
    ground_list = [0] * length
    region_count = 1
    for region in regions:
        if float(len(region))/length < 0.1:
            continue
        else:
            for indice in region:
                ground_list[seeds[indice]] = region_count

    # label_list = connected_component_labeling(voxel_set, 1.6)
    # length = len(original_code_array)
    # ground_list = [0] * length
    # count = 0
    # for label in label_list:
    #     ground_list[seeds[count]] = label
    #     count += 1

    with open(outfile_path, 'wb') as out_csv_file:
        csvwriter = csv.writer(out_csv_file)
        row = 0
        while row < length:
            csvwriter.writerow([original_code_array[row], points_count_array[row], intensity_array[row],
                                olocation_array[row], mlocation_array[row], normal_list[row], ground_list[row]])
            row += 1


def pole_position_detection(infile_path, outfile_path, voxelsize, fixed_radius, normal_threshold):
    with open(infile_path, 'rb') as incsv:
        reader = csv.reader(incsv)
        lines = [[row[0], row[1], row[2]] for row in reader]
    original_code_array = np.array(lines)[:, 0]
    original_x_int_array = np.vectorize(int)(map(lambda x: x[:4], original_code_array))
    original_y_int_array = np.vectorize(int)(map(lambda x: x[4:8], original_code_array))
    original_z_int_array = np.vectorize(int)(map(lambda x: x[8:12], original_code_array))
    point_counts_array = np.array(lines)[:, 1]
    intensity_array = np.array(lines)[:, 2]
    olocation_array = np.array(lines)[:, 3]
    mlocation_array = np.array(lines)[:, 4]
    normal_array = np.array(lines)[:, 5]
    ground_list = np.array(lines)[:, 6]


def write_normal(in_file, out_file, fixed_radius):
    import scipy
    from scipy import linalg as la

    with open(in_file, 'rb') as incsv:
        reader = csv.reader(incsv)
        lines = [[row[0], row[1], row[2], row[3], row[4]] for row in reader]
    original_code_array = np.array(lines)[:, 0]
    original_x_int_array = np.vectorize(int)(map(lambda x: x[:4], original_code_array))
    original_y_int_array = np.vectorize(int)(map(lambda x: x[4:8], original_code_array))
    original_z_int_array = np.vectorize(int)(map(lambda x: x[8:12], original_code_array))

    point_counts_array = np.array(lines)[:, 1]
    intensity_array = np.array(lines)[:, 2]
    olocation_array = np.array(lines)[:, 3]
    mlocation_array = np.array(lines)[:, 4]

    normal_list = []
    dataset = np.vstack([original_x_int_array, original_y_int_array, original_z_int_array]).transpose()
    tree = scipy.spatial.cKDTree(dataset)
    for x, y, z in zip(original_x_int_array, original_y_int_array, original_z_int_array):
        indices = tree.query_ball_point([x, y, z], fixed_radius)
        if len(indices) <= 3:
            continue
        idx = tuple(indices)
        data = np.vstack([dataset[idx, 0], dataset[idx, 1], dataset[idx, 2]])
        cov = np.cov(data)
        evals, evects = la.eigh(cov)
        evals = np.abs(evals)
        index = evals.argsort()[::-1]
        evects = evects[:, index]
        normal_list.append(evects[2])
    length = len(original_code_array)
    count = 0
    with open(out_file, 'wb') as out_csv:
        writer = csv.writer(out_csv)
        while count < length:
            writer.writerow([original_code_array[count], point_counts_array[count], intensity_array[count],
                             olocation_array[count], mlocation_array[count], normal_list[count]])
            count += 1


def connected_component_labeling_file(in_file, out_file):
    """
    perform connected component labeling on voxels

    we implement the algorithm in :
        http://www.codeproject.com/Articles/336915/Connected-Component-Labeling-Algorithm#Unify
    """
    with open(in_file) as incsv:
        reader = csv.reader(incsv)
        lines = [[row[0], row[1], row[2], row[3], row[4], row[5]] for row in reader]

    original_code_array = np.array(lines)[:, 0]
    point_counts_array = np.array(lines)[:, 1]
    intensity_array = np.array(lines)[:, 2]
    olocation_array = np.array(lines)[:, 3]
    mlocation_array = np.array(lines)[:, 4]
    foreground_flag = np.array(lines)[:, 5]

    foreground_indices = np.where(foreground_flag != '0')[0]
    x_int_array = np.vectorize(int)(map(lambda x: x[:4], original_code_array[foreground_indices]))
    y_int_array = np.vectorize(int)(map(lambda x: x[4:8], original_code_array[foreground_indices]))
    z_int_array = np.vectorize(int)(map(lambda x: x[8:12], original_code_array[foreground_indices]))

    fore_voxel_length = len(foreground_indices)

    label_array = np.array([0] * fore_voxel_length)
    parent_array = np.array([0] * fore_voxel_length)
    label_count = 0
    voxel_count = 0

    # first pass
    while voxel_count < fore_voxel_length:
        # finding the neighbors of each voxel of foreground
        x_valid = np.logical_and(x_int_array - x_int_array[voxel_count] >= -1,
                                 x_int_array - x_int_array[voxel_count] <= 1)
        y_valid = np.logical_and(y_int_array - y_int_array[voxel_count] >= -1,
                                 y_int_array - y_int_array[voxel_count] <= 1)
        z_valid = np.logical_and(z_int_array - z_int_array[voxel_count] >= -1,
                                 z_int_array - z_int_array[voxel_count] <= 1)
        neighbors = np.where(x_valid & y_valid & z_valid)[0]

        # the situation that voxel is isolated or surrounded by non-labeled voxels
        if len(neighbors) == 0 or list(label_array[neighbors]) == [0] * len(neighbors):
            label_count += 1
            label_array[voxel_count] = label_count
            parent_array[voxel_count] = label_count
        # the situation that the voxel is surrounded by same voxels
        elif len(set(label_array[neighbors])) == 1:
            label_array[voxel_count] = label_array[neighbors[0]]
        elif len(set(label_array[neighbors])) == 2 and list(label_array[neighbors]).count(0) != 0:
            neighbors_labels = label_array[neighbors]
            label_array[voxel_count] = neighbors_labels[neighbors_labels != 0][0]
            parent_array[voxel_count] = neighbors_labels[neighbors_labels != 0][0]
        # the situation that the voxel is surrounded by different voxels, we assign min neighbor's parent neighbor to
        # current voxel and parent
        else:
            if list(label_array[neighbors]).count(0) == 0:
                label_array[voxel_count], parent_array[voxel_count] = \
                    min(label_array[neighbors]), min(label_array[neighbors])
            else:
                new_set = set(label_array[neighbors])
                new_set.remove(0)
                label_array[voxel_count], parent_array[voxel_count] = min(new_set), min(new_set)
            for item in new_set:
                if item != parent_array[voxel_count]:
                    parent_array[label_array == item] = parent_array[voxel_count]
        voxel_count += 1
    # second pass
    pattern_count = 1
    parent_set = set(parent_array)
    label_array[:] = 0
    for unique_parent in parent_set:
        unique_parent_indices = np.where(parent_array == unique_parent)[0]
        foregound_mlocation_array = mlocation_array[foreground_indices]
        if len(set(foregound_mlocation_array[unique_parent_indices])) > 1:
            if len(label_array[unique_parent_indices]) > 30:
                label_array[parent_array == unique_parent] = pattern_count
                pattern_count += 1

    # label all the voxels, back ground is labeled as 0

    all_label_array = np.array([0] * len(original_code_array))
    count = 0
    for indice in foreground_indices:
        all_label_array[indice] = label_array[count]
        count += 1

    count = 0
    voxel_length = len(original_code_array)
    with open(out_file, 'wb') as out_csv:
        writer = csv.writer(out_csv)
        while count < voxel_length:
            writer.writerow([original_code_array[count], point_counts_array[count], intensity_array[count],
                             olocation_array[count], mlocation_array[count], foreground_flag[count],
                             all_label_array[count]])
            count += 1


def min_cut(in_file, out_file):
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
    import maxflow
    with open(in_file, 'rb') as incsv:
        reader = csv.reader(incsv)
        lines = [[row[0], row[1], row[2], row[3], row[4]] for row in reader]
    original_code_array = np.array(lines)[:, 0]
    original_x_int_array = np.vectorize(int)(map(lambda x: x[:4], original_code_array))
    original_y_int_array = np.vectorize(int)(map(lambda x: x[4:8], original_code_array))
    original_z_int_array = np.vectorize(int)(map(lambda x: x[8:12], original_code_array))

    point_counts_array = np.array(lines)[:, 1]
    olocation_array = np.array(lines)[:, 2]
    mlocation_array = np.array(lines)[:, 3]
    pole_id_array = np.array(lines)[:, 4]

    pole_location_set = list(set(pole_id_array[pole_id_array != '0']))

    voxel_length = len(voxel_code_array)
    voxel_code_set = set(voxel_code_array)
    reached_flag = [False] * voxel_length
    voxel_count = 0

    g = maxflow.GraphFloat()
    nodes = g.add_nodes(voxel_length)

    positions = position_array[position_array != '0']
    unique_positions = list(set(positions))

    first_positions = voxel_code_array[position_array == unique_positions[0]]
    second_positions = voxel_code_array[position_array == unique_positions[1]]

    center_x1 = int(first_positions[0][0:4])
    center_y1 = int(first_positions[0][4:8])

    center_x2 = int(second_positions[0][0:4])
    center_y2 = int(second_positions[0][4:8])

    distance = ((center_x1 - center_x2)**2 + (center_y1 - center_y2)**2)**0.5

    for voxel_code in voxel_code_set:
        center_x = int(voxel_code_array[voxel_count][0:4])
        center_y = int(voxel_code_array[voxel_count][4:8])
        d1 = ((center_x - center_x1)**2 + (center_y - center_y1)**2)**0.5
        d2 = ((center_x - center_x2)**2 + (center_y - center_y2)**2)**0.5
        g.add_tedge(nodes[voxel_count], d1 / 5, d2 / 5)
        reached_flag[voxel_count] = True
        right = "{:0>4d}".format(int(voxel_code[0:4]) + 1) + voxel_code[4:12]
        left = "{:0>4d}".format(int(voxel_code[0:4]) - 1) + voxel_code[4:12]
        front = "{:0>4d}".format(int(voxel_code[4:8]) - 1) + voxel_code[0:4] + voxel_code[8:12]
        back = "{:0>4d}".format(int(voxel_code[4:8]) - 1) + voxel_code[0:4] + voxel_code[8:12]
        up = "{:0>4d}".format(int(voxel_code[8:12]) + 1) + voxel_code[0:8]
        down = "{:0>4d}".format(int(voxel_code[8:12]) - 1) + voxel_code[0:8]
        neighbor_list = [right, left, front, back, up, down]
        for neighbor in neighbor_list:
            indice = np.where(np.array(voxel_code_array) == neighbor)
            if len(indice) == 0:
                continue
            indice = indice[0][0]
            if neighbor in voxel_code_set and reached_flag[indice] is False:
                g.add_edge(nodes[voxel_count], nodes[indice], 1, 0)

    flow = g.maxflow()
    g.get_segment(nodes[3])


def label_points_location(las_path, voxel_path):
    """
    label the location label in the voxel path to corresponding points in las file

    """
    import csv
    lasfile = laspy.file.File(las_path, mode="rw")
    with open(voxel_path) as voxel_file:
        reader = csv.reader(voxel_file)
        location_label = [[row[3], row[4], row[5], row[6]] for row in reader]
    point_count = 0
    lasfile.user_data[:] = 0
    lasfile.gps_time[:] = 0
    for voxel_index in lasfile.voxel_index:
        lasfile.olocation[point_count] = location_label[voxel_index][0]
        lasfile.mlocation[point_count] = location_label[voxel_index][1]

        # classification is corresponding to fore or back ground
        lasfile.gps_time[point_count] = location_label[voxel_index][2]
        lasfile.user_data[point_count] = location_label[voxel_index][3]
        #
        point_count += 1
    lasfile.close()


print '''Welcome to the PLOs detection System!!

Please Follow the classification step:
1. add a dimension to the las file
2. voxelize the point cloud
3. detect original localization
4. merge the original locations
5. detect pole part
6. remove background
7. labeling points in las file
8. default value is to do all the jobs
'''
loop = True
while loop:
    infilepath = raw_input('\n Now please input the original las file name: \n')
    if infilepath[-4:] == '.las':
        loop = False
    elif infilepath == 'q':
        loop = False
    else:
        print("Please input a *.las file!!!")

size_of_voxel = 0.2

# minimun height of voxel to constitute a postion
minimun_position_height = 1.0

# poles are suposed to be lower than the max_height_of_pole
max_height_of_pole = 15

# the height of the cylinder to calculate the ratio of the pole parts
cylinder_height = 1.4
cylinder_ratio = 0.95
inner_radius = 0.6
outer_radius = 1.4

startheight = 0.1
# the cylinder height radius to find foreground voxels
cylinder_radius_for_foreground = 5
cylinder_height_for_foreground = 12

outlas = infilepath[:-4] + '_' + str(size_of_voxel) + '_' + str(int(minimun_position_height / size_of_voxel)) + '.las'
outcsv = outlas[:-4] + '.csv'
outcsv1 = outlas[:-4] + '_1.csv'
outcsv2 = outlas[:-4] + '_2.csv'
outcsv3 = outlas[:-4] + '_3.csv'
outcsv4 = outlas[:-4] + '_4.csv'
outcsv5 = outlas[:-4] + '_5.csv'
loop = True
while loop:
    user_input = raw_input("\n Step:")
    if user_input == 'q':
        loop = False
    elif user_input == '1':
        start = time.clock()
        add_dimension(infilepath, outlas, ["voxel_index", "olocation", "mlocation"], [5, 5, 3],
                      ["voxel num the point in", "original location label", "merged location label"])
        print "Step 1 costs seconds", time.clock() - start
    elif user_input == '2':
        start = time.clock()
        voxelization(outlas, outcsv, 0.3)
        print "Step 2 costs seconds", time.clock() - start

    elif user_input == '3':
        start = time.clock()
        original_localization(outcsv, outcsv1, 9)
        print "Step 3 costs seconds", time.clock() - start
    elif user_input == '4':
        start = time.clock()
        merging_neighbor_location(outcsv1, outcsv2)
        print "Step 4 costs seconds", time.clock() - start

    elif user_input == '6':
        start = time.clock()
        remove_background(outcsv2, outcsv3, fixed_radius=2, normal_threshold=0.86)
        print 'Step 6 costs', time.clock() - start
    elif user_input == '7':
        start = time.clock()
        connected_component_labeling(outcsv3, outcsv4)
        print 'Step 7 costs', time.clock() - start
    elif user_input == '8':
        start = time.clock()
        label_points_location(outlas, outcsv5)
        print "Step 8 costs seconds", time.clock() - start
        loop = False
    else:
        start = time.clock()
        print ' 1.Adding dimensions...'
        add_dimension(infilepath, outlas, ["voxel_index", "olocation", "mlocation"], [5, 5, 3],
                      ["voxel num the point in", "original location label", "merged location label"])
        print "Step 1 costs seconds ", time.clock() - start
        start = time.clock()

        print '\n 2.Voxelization...voxel size is %f m ' % size_of_voxel
        voxelization(outlas, outcsv, size_of_voxel)
        print "Step 2 costs seconds ", time.clock() - start
        start = time.clock()

        print '''\n 3.Detecting original position...
        at least %d continuous vertical voxels constitute a location''' % int(minimun_position_height / size_of_voxel)
        original_localization(outcsv, outcsv1, size_of_voxel, minimun_position_height)
        print "Step 3 costs seconds", time.clock() - start

        print '''\n 4.Calculating merged position...'''
        tart = time.clock()
        merging_neighbor_location(outcsv1, outcsv2)
        print "Step 4 costs seconds", time.clock() - start

        print '''\n 5.Removing background...'''
        start = time.clock()
        remove_background(outcsv2, outcsv3, fixed_radius=1, normal_threshold=0.6)
        print 'Step 5 costs', time.clock() - start

        print '''\n 6.Labeling points...'''
        start = time.clock()
        label_points_location(outlas, outcsv3)
        print "Step 6 costs seconds", time.clock() - start
        loop = False
os.system('pause')

