# -*- coding: UTF-8 -*-
__author__ = 'Administrator'
# PLOs_detection.py
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


def region_growing(voxelset, radius):
    # codes below were region growing algorithm implemented based pseudocode in
    # http://pointclouds.org/documentation/tutorials/region_growing_segmentation.php#region-growing-segmentation

    tree = scipy.spatial.cKDTree(voxelset)
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
                tree.query_ball_point([voxelset[:, 0][current_seed], voxelset[:, 1][current_seed],
                                        voxelset[:, 2][current_seed]], radius)
            for neighbor in current_seed_neighbors:
                if voxel_indices.count(neighbor) != 0:
                        if current_region_voxels.count(neighbor) == 0:
                            current_region_voxels.append(neighbor)
                        voxel_indices.remove(neighbor)
                        if current_seeds.count(neighbor) == 0:
                            current_seeds.append(neighbor)
        regions.append(np.array(current_region_voxels))
        seed_length = len(voxel_indices)
    return regions


def two_dimensional_region_growing(voxelset, radius):
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
                tree1.query_ball_point([voxelset[:, 0][current_seed], voxelset[:, 1][current_seed]], radius)
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


def remove_background(original_code_array, mlocation_array, radius, normal_threshold):
    """
    romove the back ground based on the vertical continuous height of each x,y on xy plane

    if the continious height
    """
    length = len(original_code_array)
    original_x_int_array = np.vectorize(int)(map(lambda x: x[:4], original_code_array))
    original_y_int_array = np.vectorize(int)(map(lambda x: x[4:8], original_code_array))
    original_z_int_array = np.vectorize(int)(map(lambda x: x[8:12], original_code_array))

    normal_list = []
    dataset = np.vstack([original_x_int_array, original_y_int_array, original_z_int_array]).transpose()
    tree = scipy.spatial.cKDTree(dataset)
    for x, y, z in zip(original_x_int_array, original_y_int_array, original_z_int_array):
        indices = tree.query_ball_point([x, y, z], radius)
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
    seeds = np.logical_or(np.array(normal_list) > normal_threshold, np.array(normal_list) < -normal_threshold)

    seeds = list(np.where(seeds)[0])
    # 背景点中不能包含mlocation点
    for mlocations in mlocation_array:
        for voxel in mlocations:
            if voxel in seeds:
                seeds.remove(voxel)
    voxel_set = np.vstack([original_x_int_array[seeds], original_y_int_array[seeds],
                          original_z_int_array[seeds]]).transpose()
    regions = region_growing(voxel_set, 1.5)

    seeds = np.array(seeds)
    ground = []
    for region in regions:
        if float(len(region)) / length < 0.1:
            continue
        else:
            ground.append(seeds[region])
    return normal_list, ground


def original_localization_by_rigid_continuity(voxel_code_array, voxel_size, min_position_height):
    """
    通过严格的连续条件来查找位置点

    所谓严格条件指的是一个xy位置点必须连续存在若干个z方向上的连续点，而且只考虑该x，y，而不考虑其周围点的影响，这样检测不出一些倾斜杆

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

    # parse the string array to integer array for later calculation
    voxel_code_int_array = np.vectorize(long)(voxel_code_array)

    # counter of the voxel
    voxel_count = 0
    location_count = 1
    length = len(voxel_code_array)
    indices = range(length)

    # the array that stores the location label of the voxel, if the voxel does not belong to a voxel the value is zero
    location_id_array = np.array([0] * length)
    olocation_indices_array = []
    vertical_count_threshold = int(min_position_height / voxel_size)
    temp = [1] * vertical_count_threshold
    # traverse all the voxel to assign location label
    while voxel_count < length - vertical_count_threshold:
        v1 = voxel_code_int_array[voxel_count:voxel_count + vertical_count_threshold]
        v2 = voxel_code_int_array[voxel_count + 1:voxel_count + 1 + vertical_count_threshold]
        v = list(map(lambda x: x[0] - x[1], zip(v2, v1)))
        # judge if the vertical_count_threshold number of voxel value are continuous
        if v == temp:
            temp_locations = []
            location_id_array[voxel_count:voxel_count + 1 + vertical_count_threshold] = location_count
            temp_locations += indices[voxel_count:voxel_count + 1 + vertical_count_threshold]
            voxel_count += vertical_count_threshold + 1
            while voxel_code_int_array[voxel_count] - voxel_code_int_array[voxel_count - 1] == 1 \
                    and voxel_count < length:
                location_id_array[voxel_count] = location_count
                temp_locations.append(voxel_count)
                voxel_count += 1
            location_count += 1
            olocation_indices_array.append(temp_locations)
            # 连续的某个位置出现位置点后，其上面的不连续的点全部被当作非位置点
            change_flag = int(voxel_code_int_array[voxel_count] / 10000) - \
                                        int(voxel_code_int_array[voxel_count - 1] / 10000)
            while change_flag == 0 and voxel_count < length - 1:
                voxel_count += 1
                change_flag = int(voxel_code_int_array[voxel_count] / 10000) - \
                              int(voxel_code_int_array[voxel_count - 1] / 10000)
        else:
            # 若最低位不是位置点，则该x，y位置所有点都不是位置点，直接过滤掉
            voxel_count += 1
            change_flag = int(voxel_code_int_array[voxel_count] / 10000) - \
                                        int(voxel_code_int_array[voxel_count - 1] / 10000)
            while change_flag == 0 and voxel_count < length - 1:
                voxel_count += 1
                change_flag = int(voxel_code_int_array[voxel_count] / 10000) - \
                              int(voxel_code_int_array[voxel_count - 1] / 10000)
            # # when the voxels after more than threshold number of voxels are not continuous
            # # then these voxels are considered as non-location voxel
            # change_flag = int(voxel_code_int_array[voxel_count] / 10000) - \
            #                             int(voxel_code_int_array[voxel_count - 1] / 10000)
            # while change_flag == 0 and voxel_count < length - 1:
            #     voxel_count += 1
            #     change_flag = int(voxel_code_int_array[voxel_count] / 10000) - \
            #                   int(voxel_code_int_array[voxel_count - 1] / 10000)
        # 1. when the voxels in one location has less than threshold number of continuous voxel
        # 2. when the voxels after more than threshold number of voxels are not continuous
        # then these voxels are considered as no-location voxel
        # 此处可以测试是否第二种情况可以间隔几个voxel也算location，这样可以解决稀疏情况下的问题
        # else:
        #     voxel_count += 1
        #     change_flag = int(voxel_code_int_array[voxel_count] / 10000) - \
        #                                 int(voxel_code_int_array[voxel_count - 1] / 10000)
        #     while change_flag == 0 and voxel_count < length - 1:
        #         voxel_count += 1
        #         change_flag = int(voxel_code_int_array[voxel_count] / 10000) - \
        #                       int(voxel_code_int_array[voxel_count - 1] / 10000)
    return np.array(olocation_indices_array)


def original_localization_by_no_rigid_continuity(voxel_code_array, voxel_size, min_position_height):
    """
    通过不严格的连续条件来查找位置点

    所谓不严格条件指的是一个xy位置点不一定要严格连续存在若干个z方向上的连续点，比如中间隔了若干个格子也可以当作是连续的，这里需要设置
    一个阈值。而且考虑向上连续性是不只是考虑x，y点，而是考虑x，y周围的一个圆柱范围内的点。这样可以检测有一定倾斜度的杆状物。

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

    # parse the string array to integer array for later calculation
    voxel_code_int_array = np.vectorize(long)(voxel_code_array)

    # counter of the voxel
    voxel_count = 0
    location_count = 1
    length = len(voxel_code_array)
    indices = range(length)

    # the array that stores the location label of the voxel, if the voxel does not belong to a voxel the value is zero
    location_id_array = np.array([0] * length)
    olocation_indices_array = []
    vertical_count_threshold = int(min_position_height / voxel_size)
    temp = [1] * vertical_count_threshold
    # traverse all the voxel to assign location label
    while voxel_count < length - vertical_count_threshold:
        v1 = voxel_code_int_array[voxel_count:voxel_count + vertical_count_threshold]
        v2 = voxel_code_int_array[voxel_count + 1:voxel_count + 1 + vertical_count_threshold]
        v = list(map(lambda x: x[0] - x[1], zip(v2, v1)))
        # judge if the vertical_count_threshold number of voxel value are continuous
        if v == temp:
            temp_locations = []
            location_id_array[voxel_count:voxel_count + 1 + vertical_count_threshold] = location_count
            temp_locations += indices[voxel_count:voxel_count + 1 + vertical_count_threshold]
            voxel_count += vertical_count_threshold + 1
            while voxel_code_int_array[voxel_count] - voxel_code_int_array[voxel_count - 1] == 1 and voxel_count < length:
                location_id_array[voxel_count] = location_count
                temp_locations.append(voxel_count)
                voxel_count += 1
            location_count += 1
            olocation_indices_array.append(temp_locations)
            # 连续的某个位置出现位置点后，其上面的不连续的点全部被当作非位置点
            change_flag = int(voxel_code_int_array[voxel_count] / 10000) - int(voxel_code_int_array[voxel_count - 1] / 10000)
            while change_flag == 0 and voxel_count < length - 1:
                voxel_count += 1
                change_flag = int(voxel_code_int_array[voxel_count] / 10000) - int(voxel_code_int_array[voxel_count - 1] / 10000)
        else:
            # 若最低位不是位置点，则该x，y位置所有点都不是位置点，直接过滤掉
            voxel_count += 1
            change_flag = int(voxel_code_int_array[voxel_count] / 10000) - int(voxel_code_int_array[voxel_count - 1] / 10000)
            while change_flag == 0 and voxel_count < length - 1:
                voxel_count += 1
                change_flag = int(voxel_code_int_array[voxel_count] / 10000) - int(voxel_code_int_array[voxel_count - 1] / 10000)
    return np.array(olocation_indices_array)


def vertical_continiuity_analysis(voxel_code_array, voxel_size, min_position_height, point_count_array):
    """

    向上连续性分析，分析出具有连续性的位置点

    通过从最低位置点开始，向上面方向的邻居做增长，选取包含最多点的体素作为增长的方向，依次类推，直到没有了向上的体素为止。
    """
    # 存储所有位置的最低点作为增长的种子点

    x_array = np.vectorize(int)(map(lambda x: x[:4], voxel_code_array))
    y_array = np.vectorize(int)(map(lambda x: x[4:8], voxel_code_array))
    z_array = np.vectorize(int)(map(lambda x: x[8:12], voxel_code_array))
    seeds_list = []
    voxel_length = len(x_array)
    count = 1
    previous_x = x_array[0]
    previous_y = y_array[0]
    minz = z_array[0]
    seed_index = 0
    # 计算出所有种子点
    while count < voxel_length:
        if x_array[count] == previous_x and y_array[count] == previous_y:
            if z_array[count] < minz:
                minz = z_array[count]
                seed_index = count
        else:
            seeds_list.append(seed_index)
            seed_index = count
            minz = z_array[count]
        previous_x = x_array[count]
        previous_y = y_array[count]
        count += 1
    dataset = np.vstack([x_array, y_array, z_array]).transpose()
    tree = scipy.spatial.cKDTree(dataset)
    # 存储3维位置点信息
    location_list_list = []
    # 存储水平位置点集合
    horizontal_location_list = []
    for seed in seeds_list:
        location_list = []
        vertical_count = 0
        current_seed = seed
        up_neighbor_lenght = 1
        location_list.append(current_seed)
        # 选择26邻居体素
        while up_neighbor_lenght > 0:
            neighbors = tree.query_ball_point([x_array[current_seed], y_array[current_seed], z_array[current_seed]], 1.8)
            neighbors = np.array(neighbors)
            up_neighbor_lenght = 0
            if len(neighbors) > 0:
                up_index = np.where(z_array[neighbors] - z_array[current_seed] == 1)[0]
                up_neighbor_lenght = len(up_index)
                if up_neighbor_lenght > 0:
                    vertical_count += 1
                    if up_neighbor_lenght == 1:
                        current_seed = neighbors[up_index][0]
                    else:
                        temp_index = np.where(point_count_array[neighbors[up_index]] ==
                                              max(point_count_array[neighbors[up_index]]))[0][0]
                        # temp_index = point_count_array[neighbors[up_index]].index(max(point_count_array[neighbors[up_index]]))
                        current_seed = neighbors[up_index[temp_index]]
                    location_list.append(current_seed)
        # 若向上增长能达到一定高度，则被认为是一个潜在的位置点
        if len(location_list) * voxel_size >= minimun_position_height:
            location_list_list.append(location_list)
            horizontal_location_list.append(seed)
    return horizontal_location_list, location_list_list


def merging_neighbor_location(original_code_array, olocation_indices_array):
    """
    merge neighbor location to form the location of an object, each location is then corresponding to each object

    merge the neighboring location of the original location, the merged location is then labeled to each point in the
    las file
    Args:
        csvfile: the csv file which stores the code and the original location label
    Returns:
        None
    """

    # get the first 8 char of the code list
    x_list = []
    y_list = []
    for olocation_indices in olocation_indices_array:
        x_list.append(int(original_code_array[olocation_indices[0]][0:4]))
        y_list.append(int(original_code_array[olocation_indices[0]][4:8]))
    dataset = np.vstack([x_list, y_list]).transpose()
    merged_locations = two_dimensional_region_growing(dataset, 1.5)
    mlocation_indices_array = []
    for mlocations in merged_locations:
        temp_mlocations = []
        for mlocation in mlocations:
            temp_mlocations += list(olocation_indices_array[mlocation])
        mlocation_indices_array.append(temp_mlocations)
    return mlocation_indices_array


def pole_position_detection_by_fixed_inner_radius(points_count_in_one_voxel_array, original_code_array,
                                                  mlocation_indices_array, ground_indices, inner_radius, outer_radius,
                                                  cyliner_height, ratio_threshold, voxel_size,max_height):
    """
    利用双圆柱模型，从mlocation中筛选出属于pole的position

    这里用的双圆柱的内圆柱大小为固定大小--inner_radius
    We define three rules to detecing poles:
    1. The radius of the merged location should be less than a threshold, this is calculated by r/voxelsize
    2. The ratio between the merged location and sum of the merged location and their neigborsshould be higher than a
    threshold
    3. The merged lcoation should be higher than some threshold(this has been refined in the olocation detection step)

    Args:
        in_file: the csv file which has code and merged location column
        out_file: the csv file which added the pole label information
        inner_radius: the inner radius when applying isolation analysis using the double cylinder model
        outer_radius: the outer radius when applying isolation analysis using the double cylinder model
        ratio_threshold: the ratio of the points in the out ring and in the inner ring of the double cylinder model
        voxel_size:
        cylinder_height: only points within this from bottom of mlocation will be used to perform the double cylinder
        analysis
        max_height: height threshold of pole like objects
    Return:
        None
    """

    fore_ground_indices = range(len(points_count_in_one_voxel_array))
    ground_list = []
    for ground_region in ground_indices:
        ground_list += list(ground_region)
    ground_list.sort(cmp=None, key=None, reverse=True)
    for indice in ground_list:
        del fore_ground_indices[indice]

    fore_ground_indices = np.array(fore_ground_indices)

    original_x_int_array = np.vectorize(int)(map(lambda x: x[:4], original_code_array))
    original_y_int_array = np.vectorize(int)(map(lambda x: x[4:8], original_code_array))
    original_z_int_array = np.vectorize(int)(map(lambda x: x[8:12], original_code_array))

    # 建立KD树
    dataset = np.vstack([original_x_int_array[fore_ground_indices], original_y_int_array[fore_ground_indices],
                         original_z_int_array[fore_ground_indices]]).transpose()
    tree = scipy.spatial.cKDTree(dataset)
    # 存储杆位置的索引号，每个元素对应一个杆的位置
    pole_indices_array = []
    # 圆柱高度对应的voxe个数
    voxel_cylinder_height = int(cyliner_height / voxel_size)

    # 计算内圈，外圈的voxel的相邻个数
    voxel_inner_radius = int(inner_radius / voxel_size + 0.5) * 2**0.5
    voxel_outer_radius = int(outer_radius / voxel_size + 0.5) * 2**0.5

    # 外切球半径: 圆柱中心点与圆柱底边圆上点的距离
    query_radius = int((voxel_outer_radius**2 + (voxel_cylinder_height/2)**2)**0.5 + 0.5)

    # loop of traversing all the mlocations
    for one_mlocation_indices in mlocation_indices_array:
        x_int_array = original_x_int_array[one_mlocation_indices]
        y_int_array = original_y_int_array[one_mlocation_indices]
        z_int_array = original_z_int_array[one_mlocation_indices]


        # selecte codes under a height threshold in one mlocation
        minz = min(z_int_array)
        minx = min(x_int_array)
        maxx = max(x_int_array)
        miny = min(y_int_array)
        maxy = max(y_int_array)
        location_radius = 0.5 * ((maxx - minx)**2 + (maxy - miny)**2)**0.5
        # 如果位置点太大则不属于杆状物
        if location_radius > 4:
            continue
        # 高于一定阈值，则不能判定为杆子
        if z_int_array.max() - z_int_array.min() > max_height / voxel_size:
            continue

        # finding the center of mlocation
        center_x = int(sum(x_int_array) / float(len(x_int_array)) + 0.5)
        center_y = int(sum(y_int_array) / float(len(y_int_array)) + 0.5)

        # 为了提高速度，先计算圆柱所在大球内所有的voxel
        temp_indices = tree.query_ball_point([center_x, center_y, minz + 0.5 * cylinder_height / voxel_size],
                                             query_radius)

        # 计算球内位于小圆柱内的voxel
        temp_indices = np.array(temp_indices)
        in_voxels_flag1 = np.logical_and(original_z_int_array[fore_ground_indices[temp_indices]] >= minz,
                                         original_z_int_array[fore_ground_indices[temp_indices]] <= minz +
                                         voxel_cylinder_height)
        temp_distances = ((original_x_int_array[fore_ground_indices[temp_indices]] - center_x)**2 +
                               (original_y_int_array[fore_ground_indices[temp_indices]] - center_y)**2)**0.5
        in_voxels = np.where((temp_distances <= voxel_inner_radius) &
                             (original_z_int_array[fore_ground_indices[temp_indices]] >= minz) &
                             (original_z_int_array[fore_ground_indices[temp_indices]] <= minz + voxel_cylinder_height))[0]
        # 计算位于大圆柱内的voxel
        in_and_out_voxels = np.where((temp_distances <= voxel_outer_radius) &
                                     (original_z_int_array[fore_ground_indices[temp_indices]] >= minz) &
                                     (original_z_int_array[fore_ground_indices[temp_indices]] <=
                                      minz + voxel_cylinder_height))[0]
        in_point_count = sum(points_count_in_one_voxel_array[fore_ground_indices[temp_indices[in_voxels]]])
        in_and_out_point_count = sum(points_count_in_one_voxel_array[fore_ground_indices[temp_indices[in_and_out_voxels]]])
        # 计算内圆柱内点个数占大圆柱内点个数的比例
        if in_point_count / float(in_and_out_point_count) >= ratio_threshold:
            pole_indices_array.append(one_mlocation_indices)
        else:
            continue
        # else:
        #     outer_point_count = sum(np.vectorize(int)(points_count_in_one_voxel_array[outer_voxel_list]))
        # inner_point_count = sum(np.vectorize(int)(points_count_in_one_voxel_array[inner_voxel_list]))
        # if float(inner_point_count) / (inner_point_count + outer_point_count) > ratio_threshold:
        #     pole_indices_array.append(one_mlocation_indices)
    return np.array(pole_indices_array)


def pole_position_detection_by_mlocation_radius(points_count_in_one_voxel_array, original_code_array,
                                                  mlocation_indices_array, ground_indices, cyliner_height,
                                                  ratio_threshold, voxel_size,max_height):
    """
    利用双圆柱模型，从mlocation中筛选出属于pole的position

    这里用的双圆柱的内圆柱大小为mlocation外包框的大小
    We define three rules to detecing poles:
    1. The radius of the merged location should be less than a threshold, this is calculated by r/voxelsize
    2. The ratio between the merged location and sum of the merged location and their neigborsshould be higher than a
    threshold
    3. The merged lcoation should be higher than some threshold(this has been refined in the olocation detection step)

    Args:
        in_file: the csv file which has code and merged location column
        out_file: the csv file which added the pole label information
        inner_radius: the inner radius when applying isolation analysis using the double cylinder model
        outer_radius: the outer radius when applying isolation analysis using the double cylinder model
        ratio_threshold: the ratio of the points in the out ring and in the inner ring of the double cylinder model
        voxel_size:
        cylinder_height: only points within this from bottom of mlocation will be used to perform the double cylinder
        analysis
        max_height: height threshold of pole like objects
    Return:
        None
    """
    fore_ground_indices = range(len(points_count_in_one_voxel_array))
    back_ground_indices_sorted = []
    for ground_region in ground_indices:
        back_ground_indices_sorted += list(ground_region)
    back_ground_indices_sorted.sort(cmp=None, key=None, reverse=True)
    for indice in back_ground_indices_sorted:
        del fore_ground_indices[indice]

    fore_ground_indices = np.array(fore_ground_indices)
    original_x_int_array = np.vectorize(int)(map(lambda x: x[:4], original_code_array))
    original_y_int_array = np.vectorize(int)(map(lambda x: x[4:8], original_code_array))
    original_z_int_array = np.vectorize(int)(map(lambda x: x[8:12], original_code_array))

    # 建立KD树
    fore_dataset = np.vstack([original_x_int_array[fore_ground_indices], original_y_int_array[fore_ground_indices],
                         original_z_int_array[fore_ground_indices]]).transpose()
    fore_tree = scipy.spatial.cKDTree(fore_dataset)
    back_dataset = np.vstack([original_x_int_array[back_ground_indices_sorted],
                              original_y_int_array[back_ground_indices_sorted],
                              original_z_int_array[back_ground_indices_sorted]]).transpose()
    back_tree = scipy.spatial.cKDTree(back_dataset)
    # 存储杆位置的索引号，每个元素对应一个杆的位置
    pole_indices_array = []
    # 圆柱高度对应的voxe个数
    voxel_cylinder_height = int(cyliner_height / voxel_size)

    # loop of traversing all the mlocations
    for one_mlocation_indices in mlocation_indices_array:
        x_int_array = original_x_int_array[one_mlocation_indices]
        y_int_array = original_y_int_array[one_mlocation_indices]
        z_int_array = original_z_int_array[one_mlocation_indices]
        # 存储当前位置的最低点，用来计算杆子位置与地面位置的距离，过滤掉悬空杆
        mlocation_bottom_indice = np.argmin(z_int_array)
        distance, neighbor = back_tree.query([x_int_array[mlocation_bottom_indice],
                                              y_int_array[mlocation_bottom_indice],
                                              z_int_array[mlocation_bottom_indice]])
        # 如果最近的地面点与位置底部距离大于2，则表示是悬空杆
        z_neighbor = original_z_int_array[back_ground_indices_sorted[neighbor]]
        if z_int_array[mlocation_bottom_indice] - z_neighbor > 1:
            continue
        # selecte codes under a height threshold in one mlocation
        minz = min(z_int_array)
        minx = min(x_int_array)
        maxx = max(x_int_array)
        miny = min(y_int_array)
        maxy = max(y_int_array)
        location_radius = 0.5 * ((maxx - minx)**2 + (maxy - miny)**2)**0.5
        # 如果位置点太大则不属于杆状物
        if location_radius > 4:
            continue
        # 高于一定阈值，则不能判定为杆子
        if z_int_array.max() - z_int_array.min() > max_height / voxel_size:
            continue

        # 定义进行双圆柱分析的内外圆柱半径
        voxel_inner_radius = location_radius
        voxel_outer_radius = location_radius + 1

        # 如果垂直方向长度与半径比小于等于2，则该位置不能判断为杆位置
        if voxel_inner_radius > 0:
            if (z_int_array.max() - z_int_array.min()) / voxel_inner_radius < 4:
                continue

        # 外切球半径: 圆柱中心点与圆柱底边圆上点的距离
        query_radius = int((voxel_outer_radius**2 + (voxel_cylinder_height/2)**2)**0.5 + 0.5)
        # 找到mlocation的中心点，实际为外包框的中心点
        center_x = 0.5 * (maxx + minx)
        center_y = 0.5 * (maxy + miny)

        # 为了提高速度，先计算圆柱所在大球内所有的voxel
        temp_indices = fore_tree.query_ball_point([center_x, center_y, minz + 0.5 * cylinder_height / voxel_size],
                                             query_radius)

        # 计算球内位于小圆柱内的voxel
        temp_indices = np.array(temp_indices)
        temp_distances = ((original_x_int_array[fore_ground_indices[temp_indices]] - center_x)**2 +
                               (original_y_int_array[fore_ground_indices[temp_indices]] - center_y)**2)**0.5
        in_voxels = np.where((temp_distances <= voxel_inner_radius) &
                             (original_z_int_array[fore_ground_indices[temp_indices]] >= minz) &
                             (original_z_int_array[fore_ground_indices[temp_indices]] <= minz + voxel_cylinder_height))[0]
        # 计算位于大圆柱内的voxel
        in_and_out_voxels = np.where((temp_distances <= voxel_outer_radius) &
                                     (original_z_int_array[fore_ground_indices[temp_indices]] >= minz) &
                                     (original_z_int_array[fore_ground_indices[temp_indices]] <=
                                      minz + voxel_cylinder_height))[0]
        in_point_count = sum(points_count_in_one_voxel_array[fore_ground_indices[temp_indices[in_voxels]]])
        in_and_out_point_count = sum(points_count_in_one_voxel_array[fore_ground_indices[temp_indices[in_and_out_voxels]]])
        # 计算内圆柱内点个数占大圆柱内点个数的比例
        if in_point_count / float(in_and_out_point_count) >= ratio_threshold:
            pole_indices_array.append(one_mlocation_indices)
        else:
            continue
        # else:
        #     outer_point_count = sum(np.vectorize(int)(points_count_in_one_voxel_array[outer_voxel_list]))
        # inner_point_count = sum(np.vectorize(int)(points_count_in_one_voxel_array[inner_voxel_list]))
        # if float(inner_point_count) / (inner_point_count + outer_point_count) > ratio_threshold:
        #     pole_indices_array.append(one_mlocation_indices)
    return np.array(pole_indices_array)


def get_foot_point(startx, starty, endx, endy, x, y):
    u = ((endx - startx) * (endx - startx) + (float(endy)-starty) * (endy-starty))
    u = ((endx - startx) * (endx - x) + (endy - starty) * (endy-y)) / u
    footx = u*startx + (1 - u) * endx
    footy = u*starty + (1 - u) * endy
    return footx, footy


def min_cut(center_x, center_y, pole_position_dataset, voxel_set, intensity_array, points_count_array, neighbor_count,
            distance_threshold, radius):
    """
    segment the voxels in to n segments based on the position array using min cut algorithm

    the original min cut algorithm comes from the paper:
    "An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision."Yuri Boykov and
    Vladimir Kolmogorov. In IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), September 2004

    Args:
        center_x,center_y: 指定的前景点的中心
        pole_indices: 前景点的index
        voxel_set： 待分割的voxel集合
        intensity_array： 每个voxel对应的强度值
        points_count_array： 每个voxel 对应的voxel包含的点数
        neighbor_count：建立graph时邻居点的个数
        distance_threshold：只计算规定距离范围内的voxel来构造graph
        radius：默认的pole的最大半径，半径外的点被认为是背景点
    Return:

    """
    # K in the article
    import maxflow
    import math
    k = 3.40282e+038
    voxel_length = len(voxel_set)
    tree = scipy.spatial.cKDTree(voxel_set)
    intensity_var = np.var(np.vectorize(int)(intensity_array))
    point_count_var = np.var(np.vectorize(int)(points_count_array))
    g = maxflow.GraphFloat()
    nodes = g.add_nodes(voxel_length)

    # 求所有距离的方差
    distances, neighbors = tree.query(voxel_set, k=neighbor_count, eps=0, p=2, distance_upper_bound=distance_threshold)
    distance_all_array = []
    for distance_array in distances:
        for distance in distance_array:
            if 0 < distance <= distance_threshold:
                distance_all_array.append(distance)
    distance_var = np.var(distance_all_array)
    voxel_count = 0
    # 遍历每一个voxel 建立对应的graph
    for distance_list, neighbor_list in zip(distances, neighbors):
        distance_to_center = ((voxel_set[voxel_count][0] - center_x) ** 2
                              + (voxel_set[voxel_count][1] - center_y) ** 2) ** 0.5
        # 添加区域权重，根据与杆中心点的位置确定 Rp(1) = -ln Pr(Ip|’obj’)； Rp(0) = -ln Pr(Ip|’bkg’)
        if any(np.equal(pole_position_dataset, voxel_set[voxel_count]).all(1)) or distance_to_center == 0:
            g.add_tedge(voxel_count, k, 0)
        elif distance_to_center >= radius:
            g.add_tedge(voxel_count, 0, k)
        else:
            g.add_tedge(voxel_count, -math.log(distance_to_center / radius), -math.log(1 - distance_to_center / radius))

        for neighbor, distance in zip(neighbor_list, distance_list):
            # 因为返回的邻居点包含原本点，需要去除掉.如果距离过大也不建立graph
            if neighbor == voxel_count or distance > 3:
                continue
            point_count_dif = (points_count_array[voxel_count] - points_count_array[neighbor]) ** 2
            # intensity_dif = (intensity_array[voxel_count] - intensity_array[neighbor]) ** 2
            # 计算两个voxel的差异
            # voxel_dif = 0.5 * distance / distance_var + 0.25 * point_count_dif / point_count_var \
            #             + 0.25 * intensity_dif / intensity_var
            voxel_dif = distance ** 2 / 2
            # smoothcost = math.exp(-voxel_dif)
            smoothcost = math.exp(-point_count_dif**2 / (2 * point_count_var))
            g.add_edge(voxel_count, neighbor, smoothcost, 0)
        voxel_count += 1
    g.maxflow()
    results = np.array(map(g.get_segment, nodes))
    poles = np.where(results == 1)[0]
    return poles


def region_growing_by_seeds(voxelset, seeds_array, distance_threshold):
    """
    返回seeds_array中数组大小对应个数的块，index与整个voxel一一对应
    """
    tree = scipy.spatial.cKDTree(voxelset)
    reached_flag = np.array([False] * len(voxelset))
    # 存储对应每一组seeds的区域的集合
    regions = []
    # 遍历所有种子点组，求出对应的区域
    for seeds in seeds_array:
        # new_seeds存储新加入的需要计算邻居点的点
        if reached_flag[seeds[0]] == True:
            continue
        new_seeds = []
        new_seeds += seeds
        # 记录seeds对应的增长区域
        region = []
        region += new_seeds
        # 通过计算心加入种子点的邻居点来增长区域，直到没有新种子点加入为止
        while len(new_seeds) > 0:
            current_seed_neighbors = tree.query_ball_point(voxelset[new_seeds], distance_threshold)
            new_seeds = []
            for neighbors in current_seed_neighbors:
                for neighbor in neighbors:
                    if region.count(neighbor) == 0:
                        region.append(neighbor)
                        new_seeds.append(neighbor)
            # 默认太大的物体不是杆状物
            if len(region) > 500:
                break
        reached_flag[region] = True
        regions.append(np.array(region))
    # 返回的是对应所有voxel的索引
    return regions


def count_mlocations(region_voxels, mlocation_indices_array):
    """
    计算块中包含几个mlocations
    """
    count = 0
    for mlocations in mlocation_indices_array:
        for voxel in mlocations:
            if voxel in region_voxels:
                count += 1
                break
    return count


def pole_segmentation(points_count_array, intensity_array,  voxel_code_array, mlocation_indices_array, ground_array,
                      pole_position_array):
    """
    implement the min cut algorithm in http://pmneila.github.io/PyMaxflow/index.html#
    """
    # K in the article
    # all_indices = range(len(points_count_array))

    poles_array = []
    fore_ground_indices = range(len(voxel_code_array))
    ground_indices = []
    for ground_region in ground_array:
        ground_indices += list(ground_region)
    ground_indices.sort(cmp=None, key=None, reverse=True)
    for indice in ground_indices:
        del fore_ground_indices[indice]

    fore_ground_indices = np.array(fore_ground_indices)
    x_int_array = np.vectorize(int)(map(lambda x: x[:4], voxel_code_array))
    y_int_array = np.vectorize(int)(map(lambda x: x[4:8], voxel_code_array))
    z_int_array = np.vectorize(int)(map(lambda x: x[8:12], voxel_code_array))

    voxel_set = np.vstack([x_int_array, y_int_array, z_int_array]).transpose()
    # 求出每一个pole location对应的块(一一对应)，如果包含>1个的mlocation则进行graph cut操作

    # 存储前景点对应的pole位置点的index
    pole_seeds_list = []
    for pole_position in pole_position_array:
        temp_seed_list = []
        for voxel in pole_position:
            temp_seed_list.append(np.where(fore_ground_indices == voxel)[0][0])
        pole_seeds_list.append(temp_seed_list)
    regions = region_growing_by_seeds(voxel_set[fore_ground_indices], pole_seeds_list, 1.8)

    # for region, pole_position in zip(regions, pole_seeds_list):
    #     # 存储一个pole的位置
    #     center_x = sum(x_int_array[fore_ground_indices[pole_position]]) / len(pole_position)
    #     center_y = sum(y_int_array[fore_ground_indices[pole_position]]) / len(pole_position)
    #     region_dataset = voxel_set[fore_ground_indices[region]]
    #     mlocation_count = count_mlocations(fore_ground_indices[region], mlocation_indices_array)
    #     pole_position_dataset = voxel_set[fore_ground_indices[pole_position]]
    #     # 如果包含两个物体进行min cut 如果只包含一个则直接加入杆物体列表
    #     if mlocation_count > 1:
    #         pole = min_cut(center_x, center_y, pole_position_dataset, region_dataset, intensity_array[region],
    #                        points_count_array[region], 26, 2, 25)
    #         poles_array.append(region[pole])
    #     else:
    #         poles_array.append(fore_ground_indices[region])
    regions = np.array(regions)
    fore_ground_indices = np.array(fore_ground_indices)
    result_regions = []
    for region in regions:
        result_regions.append(fore_ground_indices[region])
    return np.array(result_regions)


def search_two_dimensional_array(array, element):
    a = np.where(array == element)
    if len(a[0]) == 0:
        return 0
    elif len(a[0]) > 1:
        return a[0][0] + 1
    else:
        return a[0] + 1


def pole_detection(lasfile, csvfile, voxel_size, position_height, normal_radius, normal_threshold, inner_radius,
                   outer_radius, cyliner_height, ratio_threshold, max_height):
    import time
    start = time.clock()
    # 如果已经添加过字段了就不用再添加
    if not os.path.exists(lasfile):
        print "Adding dimensions..."
        add_dimension(infilepath, lasfile, ["voxel_index", "olocation", "mlocation"], [5, 5, 3],
                      ["voxel num the point in", "original location label", "merged location label"])
    print "Adding dimensions costs seconds ", time.clock() - start

    # 如果体素化了下一次就不用体素化了
    if not os.path.exists(lasfile) or not os.path.exists(csvfile):
        print "\n     voxelizing..."
        voxelization(lasfile, csvfile, voxel_size)

    with open(csvfile, 'rb') as in_csv_file:
        reader = csv.reader(in_csv_file)
        line = [[row[0], row[1], row[2]] for row in reader]
    voxel_code_array = np.array(line)[:, 0]
    points_count_array = np.vectorize(int)(np.array(line)[:, 1])
    intensity_array = np.vectorize(int)(np.array(line)[:, 2])

    start = time.clock()
    print "\n    Step1...original location detecting"
    olocation_indices_array = original_localization_by_rigid_continuity(voxel_code_array, voxel_size, position_height)
    print "\n    time: ", time.clock()-start

    start = time.clock()
    print "\n    Step2...merging location"
    mlocation_indices_array = merging_neighbor_location(voxel_code_array, olocation_indices_array)
    print "\n    time: ", time.clock()-start

    start = time.clock()
    print "\n    Step3...removing background"
    normal_array, ground_indices = remove_background(voxel_code_array, mlocation_indices_array, normal_radius,
                                                    normal_threshold)
    print "\n    time: ", time.clock()-start

    start = time.clock()
    print "\n    Step4...detecting pole position"
    # pole_position_indices_array = pole_position_detection_by_fixed_inner_radius(points_count_array, voxel_code_array, mlocation_indices_array,
    #                                                       ground_indices, inner_radius, outer_radius, cyliner_height,
    #                                                       ratio_threshold, voxel_size, max_height)
    pole_position_indices_array = pole_position_detection_by_mlocation_radius(points_count_array, voxel_code_array,
                                                                              mlocation_indices_array, ground_indices,
                                                                              cyliner_height, ratio_threshold,
                                                                              voxel_size, max_height)
    print "\n    time: ", time.clock()-start

    start = time.clock()
    print '\n    Step5...pole segmentation'
    poles_array = pole_segmentation(points_count_array, intensity_array, voxel_code_array, mlocation_indices_array,
                                    ground_indices, pole_position_indices_array)
    print "\n    time: ", time.clock()-start
    if len(poles_array) == 0:
        return

    print "\n    Step6....labeling points in lasfile"
    mlocation_array = np.array([0] * len(voxel_code_array))
    ground_array = np.array([0] * len(voxel_code_array))
    pole_position_array = np.array([0] * len(voxel_code_array))
    pole_array = np.array([0] * len(voxel_code_array))
    count = 1

    for mlocation in mlocation_indices_array:
        mlocation_array[mlocation] = count
        count += 1

    count = 1
    for ground in ground_indices:
        ground_array[ground] = count
        count += 1
    count = 1
    for pole_position in pole_position_indices_array:
        pole_position_array[pole_position] = count
        count += 1
    count = 1
    for pole in poles_array:
        pole_array[pole] = count
        count += 1

    lasfile = laspy.file.File(lasfile, mode="rw")
    point_count = 0
    lasfile.user_data[:] = 0
    lasfile.gps_time[:] = 0
    lasfile.raw_classification[:] = 0
    lasfile.pt_src_id[:] = 0
    for voxel_index in lasfile.voxel_index:
        lasfile.mlocation[point_count] = mlocation_array[voxel_index]
        # classification is corresponding to fore or back ground
        lasfile.gps_time[point_count] = normal_array[voxel_index]
        lasfile.user_data[point_count] = pole_position_array[voxel_index]
        lasfile.raw_classification[point_count] = ground_array[voxel_index]
        lasfile.pt_src_id[point_count] = pole_array[voxel_index]
        point_count += 1
    lasfile.close()
    print "Done!"


def label_points_location(las_path, voxel_path):
    """
    label the location label in the voxel path to corresponding points in las file

    """
    import csv
    lasfile = laspy.file.File(las_path, mode="rw")
    with open(voxel_path) as voxel_file:
        reader = csv.reader(voxel_file)
        location_label = [[row[3], row[4], row[5], row[6], row[7]] for row in reader]
    point_count = 0
    lasfile.user_data[:] = 0
    lasfile.gps_time[:] = 0
    lasfile.pt_src_id[:] = 0
    for voxel_index in lasfile.voxel_index:
        lasfile.olocation[point_count] = location_label[voxel_index][0]
        lasfile.mlocation[point_count] = location_label[voxel_index][1]

        # classification is corresponding to fore or back ground
        lasfile.gps_time[point_count] = location_label[voxel_index][2]
        lasfile.user_data[point_count] = location_label[voxel_index][3]
        lasfile.pt_src_id[point_count] = location_label[voxel_index][4]
        #
        point_count += 1
    lasfile.close()


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

if __name__ == '__main__':
    print '''Welcome to the PLOs detection System!!

    '''
    loop = True
    while loop:
        infilepath = raw_input('\n Now please input the original las file name: \n')
        if infilepath[-4:] == '.las':
            loop = False
        else:
            print("Please input a *.las file!!!")

    size_of_voxel = 0.1

    # 定义一个位置点的最小高度
    minimun_position_height = 0.8

    # 超过这个高度的不被认为是杆
    max_height_of_pole = 15

    # 用来进行isolation分析的圆柱的高
    cylinder_height = 0.8
    cylinder_ratio = 0.91
    in_radius = 0.4
    out_radius = 0.6

    # 计算法向量的搜索半径
    fixed_radius = 1
    # 法向量大于这个值的才被认为是地面点
    normal_threshold = 0.5
    outlas = infilepath[:-4] + '_' + str(size_of_voxel) + '_' + str(int(minimun_position_height / size_of_voxel)) + '.las'
    outcsv = outlas[:-4] + '.csv'

    start = time.clock()
    # 如果已经添加过字段了就不用再添加
    if not os.path.exists(outlas):
        print "Adding dimensions..."
        add_dimension(infilepath, outlas, ["voxel_index", "olocation", "mlocation"], [5, 5, 3],
                      ["voxel num the point in", "original location label", "merged location label"])
    print "Adding dimensions costs seconds ", time.clock() - start

    ############### 1.体素化 ################
    # 如果体素化了下一次就不用体素化了
    if not os.path.exists(outlas) or not os.path.exists(outcsv):
        print "\n     voxelizing..."
        voxelization(outlas, outcsv, size_of_voxel)

    with open(outcsv, 'rb') as in_csv_file:
        reader = csv.reader(in_csv_file)
        line = [[row[0], row[1], row[2]] for row in reader]
    voxel_code_array = np.array(line)[:, 0]
    points_count_array = np.vectorize(int)(np.array(line)[:, 1])
    intensity_array = np.vectorize(int)(np.array(line)[:, 2])

    horizontal_location_list, location_list_list = vertical_continiuity_analysis(voxel_code_array, size_of_voxel, minimun_position_height, points_count_array)

    # 标记点
    location_array = np.array([0] * len(voxel_code_array))
    horizontal_location_array = np.array([0]*len(voxel_code_array))
    count = 1
    for location_list in location_list_list:
        location_array[location_list] = count
        count += 1

    count = 1
    for location in horizontal_location_list:
        horizontal_location_array[location] = count
        count += 1

    lasfile = laspy.file.File(outlas, mode="rw")
    point_count = 0
    lasfile.user_data[:] = 0
    lasfile.gps_time[:] = 0
    lasfile.raw_classification[:] = 0
    lasfile.pt_src_id[:] = 0
    for voxel_index in lasfile.voxel_index:
        lasfile.olocation[point_count] = location_array[voxel_index]
        lasfile.gps_time[point_count] = horizontal_location_array[voxel_index]
        point_count += 1
    lasfile.close()

    # start = time.clock()
    # print '\nPLOs detection...'
    # pole_detection(outlas, outcsv, size_of_voxel, minimun_position_height, fixed_radius, normal_threshold, in_radius,
    #                out_radius, cylinder_height, cylinder_ratio, max_height_of_pole)
    # print "PLOs detection costs seconds ", time.clock() - start

    os.system('pause')

