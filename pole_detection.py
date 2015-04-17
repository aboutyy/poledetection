#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by You Li on 2015-03-09 0009

import laspy
import timeit
import os
import csv
import numpy as np
import scipy
from scipy import linalg as la
import scipy.spatial

###### 定义常量#######
# 体素大小
VOXEL_SIZE = 0.15

# 最小杆高度
MIN_HEIGHT = 1.0

# 杆离地面最大距离
DISTANCE_TO_GROUND = 1.05

# 杆在地面的最大面积
MAX_AREA = 0.5

# 邻居最远体素距离
MAX_NEIGHBOR_DISTANCE = 0.34

# 判断地面的法向量
GROUND_NORMAL_THRESHOLD = 0.7

# 作圆柱判断的最小圆柱高度
MIN_CYLINDER_HEIGHT = 0.6

# 内圆柱半径
INNER_RADIUS = 0.45

# 双圆柱内外圆柱之间的距离
DISTANCE_OF_IN2OUT = 0.3

# 双圆柱用来定义杆的内外点比例
RATIO_OF_POINTS_COUNT = 0.95

# 是否进行地面距离判定
USE_GROUND = True

# 地面点判断时的邻居点最远距离
GROUND_NEIGHBOR = 0.3

# 合并相邻垂直voxel组的距离阈值
MERGING_DISTANCE = 0.6

# 过滤长宽
FILTERING_LENGTH = 1.05

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

    # the array to store the average intensity of points in a voxel
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


def vertical_continiuity_analysis(dataset, point_count_array):
    """

    向上连续性分析，分析出具有连续性的位置点

    通过从最低位置点开始，向上面方向的邻居做增长，选取包含最多点的体素作为增长的方向，依次类推，直到没有了向上的体素为止。
    """

    # 存储所有位置的最低点作为增长的种子点
    seeds_list = []
    voxel_length = len(dataset)
    count = 1
    previous_x = dataset[:, 0][0]
    previous_y = dataset[:, 1][0]
    flag = 0
    # 计算出所有种子
    while count < voxel_length:
        if dataset[:, 0][count] == previous_x and dataset[:, 1][count] == previous_y:
            if dataset[:, 2][count] - dataset[:,2][count-1] < 3 and flag == count - 1:
                # 过滤边缘点
                if points_count_array[count] > 1:
                    seeds_list.append(count)
        else:
            flag = count
            previous_x = dataset[:, 0][count]
            previous_y = dataset[:, 1][count]
        count += 1
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
            neighbors = tree.query_ball_point(dataset[current_seed], 1.8)
            neighbors = np.array(neighbors)
            up_neighbor_lenght = 0
            if len(neighbors) > 0:
                # 找出上邻居点
                up_indexs = np.where(dataset[:, 2][neighbors] - dataset[:, 2][current_seed] == 1)[0]
                # 找出正上点
                up_index = np.where((dataset[:, 2][neighbors] - dataset[:, 2][current_seed] == 1) &
                                    (dataset[:, 0][neighbors] == dataset[:, 0][current_seed]) &
                                    (dataset[:, 1][neighbors] == dataset[:, 1][current_seed]))[0]
                up_neighbor_lenght = len(up_indexs)
                if up_neighbor_lenght > 0:
                    vertical_count += 1
                    if up_neighbor_lenght == 1:
                        current_seed = neighbors[up_indexs][0]
                    elif len(up_index) != 0:
                        current_seed = neighbors[up_index[0]]
                    else:
                        temp_index = np.where(point_count_array[neighbors[up_indexs]] ==
                                              max(point_count_array[neighbors[up_indexs]]))[0][0]
                        current_seed = neighbors[up_indexs[temp_index]]
                    # 加入所有邻居点到潜在杆位置点中
                    for index in neighbors[up_indexs]:
                        location_list.append(index)
        # 若向上增长能达到一定高度，则被认为是一个潜在的位置点
        height = max(dataset[:, 2][location_list]) - min(dataset[:, 2][location_list])
        if height * VOXEL_SIZE >= MIN_HEIGHT:
            location_list_list.append(location_list)
            horizontal_location_list.append(seed)
    return horizontal_location_list, location_list_list


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


def region_growing_two_dimensional(dataset, radius):
    # codes below were region growing algorithm implemented based pseudocode in
    # http://pointclouds.org/documentation/tutorials/region_growing_segmentation.php#region-growing-segmentation

    tree = scipy.spatial.cKDTree(dataset)
    length = len(dataset)
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
            current_seed_neighbors = tree.query_ball_point([dataset[:, 0][current_seed], dataset[:, 1][current_seed]], radius)
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


def background_detection(dataset, radius, normal_threshold):
    """
    romove the back ground based on the vertical continuous height of each x,y on xy plane

    if the continious height
    """
    length = len(dataset)
    normal_list = []
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

    voxel_set = np.vstack([original_x_int_array[seeds], original_y_int_array[seeds],
                          original_z_int_array[seeds]]).transpose()
    regions = region_growing(voxel_set, GROUND_NEIGHBOR / VOXEL_SIZE)

    seeds = np.array(seeds)
    ground = []
    max_len = 0
    if len(regions) == 1:
        ground += seeds[regions[0]]
    else:
        for region in regions:
            if len(region) > max_len:
                max_len = len(region)
                ground = []
                ground = seeds[region]
            # if float(len(region)) / length < 0.2:
            #     continue
            # else:
            #     ground += list(seeds[region])
    # if len(ground) == 0:
    #     ground.append(temp_ground)
    return normal_list, ground


def filtering_by_distance_to_ground(dataset, ground_indices, horizontal_location_list):
    """
    通过计算位置点与地面点的距离来过滤一些悬空位置点

    """
    filtered_indices = []
    back_ground_indices = []
    for indices in ground_indices:
        back_ground_indices.append(indices)
    back_dataset = dataset[back_ground_indices]
    back_tree = scipy.spatial.cKDTree(back_dataset)
    for location in horizontal_location_list:
        distance, neighbor = back_tree.query(dataset[location])
        z_neighbor = dataset[:, 2][back_ground_indices[neighbor]]
        if dataset[:, 2][location] - z_neighbor >= DISTANCE_TO_GROUND / VOXEL_SIZE:
            item = horizontal_location_list.index(location)
            filtered_indices.append(item)
    return filtered_indices


def selection_by_area_deltax_deltay(dataset1, dataset2, distance, length):
    selected_region_list = []
    neighbor_distance = distance / VOXEL_SIZE
    regions = region_growing_two_dimensional(dataset1, neighbor_distance)
    for region in regions:
        temp_x = []
        temp_y = []
        for item in region:
            temp_x.append(dataset2[item][0])
            temp_y.append(dataset2[item][1])
        if len(temp_x) * VOXEL_SIZE ** 2 <= MAX_AREA:
            deltax = max(temp_x) - min(temp_x)
            deltay = max(temp_y) - min(temp_y)
            if deltax < length/VOXEL_SIZE and deltay < length/VOXEL_SIZE:
                selected_region_list.append(region)
    return selected_region_list


def svd_line_fit(data):
    """
    利用SVD来拟合直线
    """
    # Calculate the mean of the points, i.e. the 'center' of the cloud
    datamean = data.mean(axis=0)

    # Do an SVD on the mean-centered data.
    uu, dd, vv = np.linalg.svd(data - datamean)
    # Now vv[0] contains the first principal component, i.e. the direction
    # vector of the 'best fit' line in the least squares sense.

    # Now generate some points along this best fit line, for plotting.

    # I use -7, 7 since the spread of the data is roughly 14
    # and we want it to have mean 0 (like the points we did
    # the svd on). Also, it's a straight line, so we only need 2 points.
    linepts = vv[0] * np.mgrid[-7:7:2j][:, np.newaxis]

    # shift by the mean to get the line in the right place
    linepts += datamean
    # import matplotlib.pyplot as plt
    # import mpl_toolkits.mplot3d as m3d
    #
    # ax = m3d.Axes3D(plt.figure())
    # ax.scatter(data[:, 0], data[:, 1], data[:, 2], zdir='z', s=100, c='r', marker='.')
    # ax.plot3D(*linepts.T)
    # plt.show()
    return linepts[0][0], linepts[0][1], linepts[0][2], vv[0][0], vv[0][1], vv[0][2]


def get_line_model(p1, p2):
    """
    根据两个点p1 p2计算出三维线参数方程的参数

    返回一个list 分别是x0，y0，z0， m，n，p 分别是经过的一个点和线的方向
    """
    return [p1[0], p1[1], p1[2], p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]


def get_point_line_distance(point1, point2, point0):
    """
    计算点到直线的距离

    point1,point2是直线上的点，point0是要计算的点；
    计算方法参考 # http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    返回距离
    """
    return np.linalg.norm(np.cross(point0 - point1, point0 - point2))/np.linalg.norm(point2 - point1)


def error_function_distance(dataset, model, t):
    """
    计算dataset数据的残差,这里计算距离和
    """
    angle = model[5] / (model[3]**2 + model[4]**2 + model[5]**2)**0.5
    if abs(angle) < 0.8:
        return float('inf')
    vector1 = np.array([model[0], model[1], model[2]])
    vector2 = np.array([model[0] + model[3], model[1] + model[4], model[2] + model[5]])
    dissum = 0
    for data in dataset:
        vector0 = np.array([data[0], data[1], data[2]])
        dis = get_point_line_distance(vector1, vector2, vector0)
        if dis <= t:
            dissum += dis
    return dissum


def error_function_count(dataset, model, t):
    """
    计算dataset数据的残差,这里计算距离和
    """
    count = 0
    angle = model[5] / (model[3]**2 + model[4]**2 + model[5]**2)**0.5
    if abs(angle) < 0.8:
        return 0
    vector1 = np.array([model[0], model[1], model[2]])
    vector2 = np.array([model[0] + model[3], model[1] + model[4], model[2] + model[5]])
    for data in dataset:
        vector0 = np.array([data[0], data[1], data[2]])
        dis = get_point_line_distance(vector1, vector2, vector0)
        if dis <= t:
            count += 1
    return count


def ransac_fit_line_3d(dataset, n, k, t, d):
    """
    RANSACf方法拟合三维直线

    Args:
        data —— 一组观测数据
        model —— 适应于数据的模型
        n —— 适用于模型的最少数据个数
        k —— 算法的迭代次数
        t —— 用于决定数据是否适应于模型的阀值
        d —— 判定模型是否适用于数据集的数据数目
    Returns:
        best_model —— 跟数据最匹配的模型参数（如果没有找到好的模型，返回null）
        best_consensus_set —— 估计出模型的数据点
        best_error —— 跟数据相关的估计出的模型错误
    """
    import random
    iterations = 0
    best_model = []
    best_consensus_set = []

    index_indices = range(len(dataset))
    # score根据局内点来确定
    # best_score = float('inf')
    best_score = 0
    while iterations < k:
        # 从数据集中随机选择n个点
        maybe_inliers = random.sample(set(index_indices), n)

        # 适合于maybe_inliers的模型参数
        maybe_model = get_line_model(dataset[maybe_inliers[0]], dataset[maybe_inliers[1]])
        consensus_set = maybe_inliers[:]

        # 对每个数据集中不属于maybe_inliers的点
        for indice in index_indices:
            if indice not in maybe_inliers:
                # 如果点适合于maybe_model，且错误小于t,将点添加到consensus_set
                if get_point_line_distance(dataset[maybe_inliers[0]], dataset[maybe_inliers[1]], dataset[indice]) <= t:
                    consensus_set.append(indice)
        # consensus_set中的元素数目大于d, 表示已经找到了好的模型，则现在测试该模型到底有多好
        if len(consensus_set) > d:
            # 获取适合于consensus_set中所有点的模型参数
            # better_model = svd_line_fit(dataset[consensus_set])
            # score计算的方法是计算局内点的个数
            # this_score = error_function_count(dataset, maybe_model, t)
            this_score = len(consensus_set)
            if this_score > best_score:
                # 我们发现了比以前好的模型，保存该模型直到更好的模型出现
                best_model = maybe_model
                best_score = this_score

        # 增加迭代次数
        iterations += 1
    if best_score ==0:
        return None
    # import matplotlib.pyplot as plt
    # import mpl_toolkits.mplot3d as m3d
    # linepts = [[best_model[0], best_model[1], best_model[2]], [best_model[0] + 10 * best_model[3], best_model[1] + 10 * best_model[4], best_model[2] + 10 * best_model[5]]]
    # linepts = np.array(linepts)
    # ax = m3d.Axes3D(plt.figure())
    # ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], zdir='z', s=100, c='r', marker='.')
    # ax.plot3D(linepts[:,0], linepts[:,1], linepts[:,2], 'r')
    # plt.show()
    return best_model[0], best_model[1], best_model[2], best_model[3], best_model[4], best_model[5]


def estimate_radius(dataset):
    pass


def oneandtwo(a, b, c):
    """

    求解ax**2 + bx + c = 0方程的解

    """
    import math
    #根的判别式
    delta = b**2-4*a*c
    if delta < 0:
        return None
    else:
        x1 = (-b+math.sqrt(delta))/(2*a)
        x2 = (-b-math.sqrt(delta))/(2*a)
    return x1, x2


def isolation_analysis(dataset, horizontal_location_list_list, location_list_list, points_in_one_voxel_array):
    """
    传统圆柱分析方法
    """
    tree = scipy.spatial.cKDTree(dataset)
    location_count = 0
    selected_indices = []
    for location_list, horizontal_location_list in zip(location_list_list, horizontal_location_list_list):
        # inner_radius = estimate_radius(dataset[horizontal_location_list])
        inner_radius = INNER_RADIUS / VOXEL_SIZE
        outer_radius = inner_radius + DISTANCE_OF_IN2OUT / VOXEL_SIZE
        # 为了减少计算，先求出一个球内的点，再从中筛选
        query_radius = ((MIN_CYLINDER_HEIGHT*0.5/VOXEL_SIZE)**2 + (outer_radius) ** 2)**0.5
        bottom_z = min(dataset[location_list][:, 2])
        top_z = max(dataset[location_list][:, 2])
        current_bottom = bottom_z
        current_top = current_bottom + MIN_CYLINDER_HEIGHT / VOXEL_SIZE
        while current_bottom + MIN_CYLINDER_HEIGHT / VOXEL_SIZE <= top_z:
            # 当前范围内的位置点
            current_indices = np.where((dataset[location_list][:,2] >= current_bottom)&(dataset[location_list][:,2] <= current_bottom+MIN_CYLINDER_HEIGHT / VOXEL_SIZE))
            center_x = sum(dataset[np.array(location_list)][current_indices][:, 0]) / len(current_indices[0])
            center_y = sum(dataset[np.array(location_list)][current_indices][:, 1]) / len(current_indices[0])
            center_z = (current_bottom + current_top) * 0.5
            subset_indices = tree.query_ball_point([center_x, center_y, center_z], query_radius)
            subset_indices = np.array(subset_indices)
            sub_dataset = dataset[subset_indices]
            distance_list = []
            for data in sub_dataset:
                dis = ((data[0] - center_x)**2 + (data[1] - center_y)**2)**0.5
                distance_list.append(dis)
            distance_array = np.array(distance_list)

            in_voxel_indices = np.where(np.logical_and((abs(distance_array - inner_radius) <= 1e-10) | (distance_array <
                                                                                                        inner_radius),
                                                       ((sub_dataset[:, 2] > current_bottom) |
                                                        (abs(sub_dataset[:, 2] - current_bottom) <= 1e-10)) &
                                                       ((sub_dataset[:, 2] < current_top) |
                                                        (abs(sub_dataset[:,2] - current_top) <= 1e-10))))
            out_voxel_indices = np.where(np.logical_and((distance_array > inner_radius) & ((abs(distance_array - outer_radius) <= 1e-10)|(distance_array < outer_radius)),
                                                        ((sub_dataset[:, 2] > current_bottom) |
                                                        (abs(sub_dataset[:, 2] - current_bottom) <= 1e-10)) &
                                                       ((sub_dataset[:, 2] < current_top) |
                                                        (abs(sub_dataset[:,2] - current_top) <= 1e-10))))
            in_points_count = sum(points_in_one_voxel_array[subset_indices[in_voxel_indices]])
            out_points_count = sum(points_in_one_voxel_array[subset_indices[out_voxel_indices]])

            # 向上移动一格
            current_bottom += 1
            current_top += 1
            if in_points_count > 0:
                if in_points_count / (in_points_count + out_points_count) >= RATIO_OF_POINTS_COUNT:
                    selected_indices.append(location_count)
                    break
        location_count += 1
    return selected_indices


def imporved_isolation_analysis(dataset, horizontal_location_list_list, location_list_list, points_in_one_voxel_array):
    """
    通过计算轴心线来判断杆是否部分符合双圆柱模型

    因为计算轴心了，所以杆是可以倾斜的；因为计算的是部分杆，所以杆是可以包含其他部分的；因为估算圆柱大小了，所以是可以测算与其他物体靠近
    的杆状物的；

    流程是：1.根据位置点拟合出线方程 2.以线为轴心构造圆柱 3.判断在圆柱内外的体素
    判断体素是否在圆柱内的方法是：1.判断体素是否在圆柱两地面所在的平面之间 2.计算体素与轴心的距离 3.若小于圆柱半径则位于其内部，反之则
    不在
    判断体素是否在圆柱内加速方法：先通过kdtree计算出圆柱所在最小球的所有体素来缩小计算范围，再一个个判断球内体素是否在圆柱内

    """
    tree = scipy.spatial.cKDTree(dataset)
    location_count = 0
    selected_indices = []
    for location_list, horizontal_location_list in zip(location_list_list, horizontal_location_list_list):
        # inner_radius = estimate_radius(dataset[horizontal_location_list])
        inner_radius = INNER_RADIUS / VOXEL_SIZE
        outer_radius = inner_radius + DISTANCE_OF_IN2OUT / VOXEL_SIZE
        # 为了减少计算，先求出一个球内的点，再从中筛选
        query_radius = ((MIN_CYLINDER_HEIGHT*0.5/VOXEL_SIZE)**2 + (outer_radius) ** 2)**0.5
        bottom_z = min(dataset[location_list][:, 2])
        top_z = max(dataset[location_list][:, 2])
        current_bottom = bottom_z

        while current_bottom + MIN_CYLINDER_HEIGHT / VOXEL_SIZE <= top_z:
            # 空间直线的参数方程来表示拟合的轴心线
            line_indices = np.where((dataset[location_list][:,2] >= current_bottom)&(dataset[location_list][:,2] <= current_bottom+MIN_CYLINDER_HEIGHT / VOXEL_SIZE))
            fit_dataset = dataset[np.array(location_list)[line_indices]]
            min_length = min(MIN_CYLINDER_HEIGHT / VOXEL_SIZE, 0.5*len(fit_dataset) + 1)
            model = ransac_fit_line_3d(fit_dataset, 2, 25, 0, min_length)
            if model is None:
                current_bottom += 1
                continue
            # 圆柱的第一个圆心点
            a, b, c, m, n, p = model[0], model[1], model[2], model[3], model[4], model[5],
            x0 = a + m * (current_bottom - c) / p
            y0 = b + n * (current_bottom - c) / p
            z0 = current_bottom

            a1 = m**2 + n**2 + p**2
            b1 = 2 * (m * (a - x0) + n * (b - y0) + p * (c - z0))
            c1 = (a - x0)**2 + (b - y0)**2 + (c - z0)**2 - (MIN_CYLINDER_HEIGHT / VOXEL_SIZE)**2
            t1, t2 = oneandtwo(a1, b1, c1)
            # 圆柱的第二个圆心点
            if c + p*t1 > c + p*t2:
                x1 = a + m * t1
                y1 = b + n * t1
                z1 = c + p * t1
            else:
                x1 = a + m * t2
                y1 = b + n * t2
                z1 = c + p * t2

            centerx = (x0 + x1) / 2
            centery = (y0 + y1) / 2
            centerz = (z0 + z1) / 2

            subset_indices = tree.query_ball_point([centerx, centery, centerz], query_radius)
            subset_indices = np.array(subset_indices)

            sub_dataset = dataset[subset_indices]
            # 计算所有球内点离圆柱中心点的距离
            # distances = ((dataset[subset_indices][:, 0] - centerx)**2 + (dataset[subset_indices][:, 1] - centery)**2 +
            #              (dataset[subset_indices][:, 0] - centerx)**2)**0.5

            # 计算点到轴线的距离
            # http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
            vector1 = np.array([a, b, c])
            vector2 = np.array([a+m, b+n, c+p])
            distance_list = []
            for data in sub_dataset:
                vector0 = data
                dis = np.linalg.norm(np.cross(vector0-vector1, vector0-vector2))/np.linalg.norm(vector2-vector1)
                distance_list.append(dis)
            distance_array = np.array(distance_list)
            # # 计算位于两个圆柱面所在平面间的点
            # between_planes_flag = (m*(sub_dataset[:, 0] - x0) + n*(sub_dataset[:, 1] - y0) +
            #                        p*(sub_dataset[:, 2] - z0)) * (m*(sub_dataset[:, 0] - x1) +
            #                                                       n*(sub_dataset[:, 1] - y1) +
            #                                                       p*(sub_dataset[:, 2] - z1)) < 0
            # 计算点到两圆柱平面的距离之和，和小于等于圆柱高的为平面之间的点
            # 参考http://mathworld.wolfram.com/Point-PlaneDistance.html
            distances_sum = (np.vectorize(abs)(m*sub_dataset[:, 0] + n*sub_dataset[:, 1] + p*sub_dataset[:, 2] -
                                               m*x0 - n*y0 -p*z0) + np.vectorize(abs)(m*sub_dataset[:, 0] +
                                                                                      n*sub_dataset[:, 1] +
                                                                                      p*sub_dataset[:, 2] -
                                                                                      m*x1 - n*y1 -p*z1)) /\
                            (m**2 + n**2 + p**2)**0.5

            in_voxel_indices = np.where(np.logical_and((abs(distance_array - inner_radius) <= 1e-10) | (distance_array < inner_radius),
                                                       (abs(distances_sum - MIN_CYLINDER_HEIGHT / VOXEL_SIZE <= 0.1e-10))|(distances_sum < MIN_CYLINDER_HEIGHT/VOXEL_SIZE)))
            out_voxel_indices = np.where(np.logical_and((distance_array > inner_radius) & ((abs(distance_array - outer_radius) <= 1e-10)|(distance_array < outer_radius)),
                                                        distances_sum - MIN_CYLINDER_HEIGHT / VOXEL_SIZE <= 0.1e-10))
            in_points_count = sum(points_in_one_voxel_array[subset_indices[in_voxel_indices]])
            out_points_count = sum(points_in_one_voxel_array[subset_indices[out_voxel_indices]])

            # 向上移动一格
            current_bottom += 1
            if in_points_count > 0:
                if in_points_count / (in_points_count + out_points_count) >= RATIO_OF_POINTS_COUNT:
                    selected_indices.append(location_count)
                    break
        location_count += 1
    return selected_indices


if __name__ == '__main__':
    import yylog

    loop = True
    while loop:
        inputpath = raw_input('\n Input las file name: \n')
        infilepath = inputpath + '.las'
        if os.path.exists(infilepath):
            loop = False
        else:
            print 'File not exist!!'
            loop = True
    outlas = inputpath + '_' + str(VOXEL_SIZE) + '_' + str(int(MIN_HEIGHT / VOXEL_SIZE)) + '.las'
    outcsv = outlas[:-4] + '.csv'
    outcsv1 = outcsv[:-4] + '_1' + '.csv'

    ####################新建加入新字段的las文件###################
    # 如果已经添加过字段了就不用再添加
    if not os.path.exists(outlas):
        print "Adding dimensions..."
        add_dimension(infilepath, outlas, ["voxel_index", "tlocation", "olocation", "flocation"], [5, 5, 5, 5],
                      ["voxel num the point in", "original location label", "merged location label", "temp"])

    ############### 1.体素化 ################
    # 如果体素化了下一次就不用体素化了
    if not os.path.exists(outlas) or not os.path.exists(outcsv):
        print "\n1. voxelizing..."
        voxelization(outlas, outcsv, VOXEL_SIZE)

    with open(outcsv, 'rb') as in_csv_file:
        reader = csv.reader(in_csv_file)
        line = [[row[0], row[1], row[2]] for row in reader]
    voxel_code_array = np.array(line)[:, 0]
    points_count_array = np.vectorize(int)(np.array(line)[:, 1])
    intensity_array = np.vectorize(int)(np.array(line)[:, 2])
    original_x_int_array = np.vectorize(int)(map(lambda x: x[:4], voxel_code_array))
    original_y_int_array = np.vectorize(int)(map(lambda x: x[4:8], voxel_code_array))
    original_z_int_array = np.vectorize(int)(map(lambda x: x[8:12], voxel_code_array))
    original_dataset = np.vstack([original_x_int_array, original_y_int_array, original_z_int_array]).transpose()
    ############## 2.垂直连续性分析 ################
    log = yylog.LOG('pole')
    try:
        start = timeit.default_timer()
        print '\n2. vertical_continiuity_analysis...'
        horizontal_location_list, location_list_list = vertical_continiuity_analysis(original_dataset, points_count_array)

    ################### 3.过滤 ####################
        if USE_GROUND:
            print '\n3. filtering...'
            print '\n   3.1 background_detection...'
            if not os.path.exists(outcsv1):
                normals, ground = background_detection(original_dataset, MAX_NEIGHBOR_DISTANCE / VOXEL_SIZE, GROUND_NORMAL_THRESHOLD)
                with open(outcsv1, 'wb') as out_csv:
                    # count = 0
                    writer = csv.writer(out_csv)
                    writer.writerow(ground)
            else:
                print '\n      3.1.1 Reading from files...'
                ground = []
                with open(outcsv1, 'rb') as in_csv:
                    count = 0
                    reader = csv.reader(in_csv)
                    for row1 in reader:
                        for item in row1:
                            ground.append(int(item))

            print '\n   3.2 filtering_by_distance_to_ground...'
            filtered_indices = filtering_by_distance_to_ground(original_dataset, ground, horizontal_location_list)
            filtered_indices.sort(cmp=None, key=None, reverse=True)
            for item in filtered_indices:
                horizontal_location_list.remove(horizontal_location_list[item])
                location_list_list.remove(location_list_list[item])

        original_location_list_list = location_list_list[:]
        original_horizontal_list_list = horizontal_location_list[:]

        print '\n   3.3 selection_by_area...'
        x_list = []
        y_list = []
        # 计算垂直块的中心
        for location_list in location_list_list:
            data = original_dataset[location_list]
            x_list.append(sum(data[:, 0]) / float(len(data)))
            y_list.append(sum(data[:, 1]) / float(len(data)))
        dataset1 = np.vstack([x_list, y_list]).transpose()
        dataset2 = original_dataset[horizontal_location_list]
        selected_regions = selection_by_area_deltax_deltay(dataset1, dataset2, MERGING_DISTANCE, FILTERING_LENGTH)
        filtered_horizontal_location_list_list = []
        filtered_location_list_list = []
        horizontal_location_list = np.array(horizontal_location_list)
        for region in selected_regions:
            filtered_horizontal_location_list_list.append(horizontal_location_list[region])
            temp_list = []
            for item in region:
                temp_list += location_list_list[item]
            temp_list = list(set(temp_list))
            filtered_location_list_list.append(temp_list)

     ################# 4.双圆柱分析 #################
        print '\n4. double cylinder analysis...'
        selected_indices1 = isolation_analysis(original_dataset, filtered_horizontal_location_list_list, filtered_location_list_list, points_count_array)
        selected_indices2 = imporved_isolation_analysis(original_dataset, filtered_horizontal_location_list_list, filtered_location_list_list, points_count_array)
        final_horizontal_list_list1 =[]
        final_list_list1 = []
        final_horizontal_list_list2 =[]
        final_list_list2 = []
        for indice1, indice2 in zip(selected_indices1, selected_indices2):
            final_horizontal_list_list1.append(filtered_horizontal_location_list_list[indice1])
            final_list_list1.append(filtered_location_list_list[indice1])
            final_horizontal_list_list2.append(filtered_horizontal_location_list_list[indice2])
            final_list_list2.append(filtered_location_list_list[indice2])
        stop = timeit.default_timer()
        print stop - start
    ################ 5. 增加邻居点 #################
        # print '\n5. adding neighbors...'
        # tree = scipy.spatial.cKDTree(original_dataset)
        # count = 0
        # for items in final_list_list:
        #     temp_list = items[:]
        #     for item in temp_list:
        #         neighbors = tree.query_ball_point(original_dataset[item], 1)
        #         for neighbor in neighbors:
        #             if neighbor not in items:
        #                 final_list_list[count].append(neighbor)
        #     count += 1

    ################## 6.标记点云 ##################
        print '\n6. lableling...'
        original_location_array = np.array([0] * len(voxel_code_array))
        location_array = np.array([0] * len(voxel_code_array))
        horizontal_location_array = np.array([0]*len(voxel_code_array))
        slected_location_array1 = np.array([0] * len(voxel_code_array))
        slected_location_array2 = np.array([0] * len(voxel_code_array))
        selected_horizontal_location_array1 = np.array([0]*len(voxel_code_array))
        selected_horizontal_location_array2 = np.array([0]*len(voxel_code_array))
        if USE_GROUND:
            ground_array = np.array([0] * len(voxel_code_array))

        count = 1
        for location in original_location_list_list:
            original_location_array[location] = count
            count += 1

        if USE_GROUND:
            ground_array[ground] = 1

        count = 1
        for location_list in filtered_location_list_list:
            location_array[location_list] = count
            count += 1

        count = 1
        for location in original_horizontal_list_list:
            horizontal_location_array[location] = count
            count += 1

        count = 1
        for location_list in final_list_list1:
            slected_location_array1[location_list] = count
            count += 1

        count = 1
        for location_list in final_list_list2:
            slected_location_array2[location_list] = count
            count += 1

        count = 1
        for location_list in final_horizontal_list_list1:
            selected_horizontal_location_array1[location_list] = count
            count += 1
        count = 1
        for location_list in final_horizontal_list_list2:
            selected_horizontal_location_array2[location_list] = count
            count += 1

        lasfile = laspy.file.File(outlas, mode="rw")
        point_count = 0
        # lasfile.olocation[:] = 0
        # lasfile.tlocation[:] = 0
        # lasfile.flocation[:] = 0
        lasfile.user_data[:] = 0
        lasfile.gps_time[:] = 0
        if USE_GROUND:
            lasfile.raw_classification[:] = 0
        lasfile.pt_src_id[:] = 0
        for voxel_index in lasfile.voxel_index:
            lasfile.olocation[point_count] = original_location_array[voxel_index]
            lasfile.tlocation[point_count] = location_array[voxel_index]
            lasfile.flocation[point_count] = slected_location_array1[voxel_index]
            lasfile.gps_time[point_count] = horizontal_location_array[voxel_index]
            lasfile.pt_src_id[point_count] = slected_location_array2[voxel_index]
            if USE_GROUND:
                lasfile.raw_classification[point_count] = ground_array[voxel_index]
            lasfile.user_data[point_count] = selected_horizontal_location_array1[voxel_index]

            point_count += 1
        lasfile.close()

    except:
        log.error()  # 使用系统自己的错误描述
        os.system('pause')
        exit()
    os.system('pause')