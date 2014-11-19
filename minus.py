# -*- coding: UTF-8 -*-
import  laspy
import numpy as np
import scipy.spatial
import os
def point_cloud_minnus(infile_path1, infile_path2, distance):
    """
    points in infile_path1 minus points in infile_path2

    如果第一个点云中的点与第二个点云中最近的点的距离小于distance，则把该点标记为1
    """
    infile1 = laspy.file.File(infile_path1, mode='rw')
    infile2 = laspy.file.File(infile_path2)
    dataset1 = np.vstack([infile1.x, infile1.y, infile1.z]).transpose()
    dataset2 = np.vstack([infile2.x, infile2.y, infile2.z]).transpose()
    tree1 = scipy.spatial.cKDTree(dataset1)
    tree2 = scipy.spatial.cKDTree(dataset2)
    results = tree2.query_ball_tree(tree1, distance)
    print 'detected %d points' %len(results)
    while '' in results:
        results.remove('')
    print 'detected %d not empty points' %len(results)
    # indices = []
    count = 0
    # while count < len(results):
    #     # indices = list(set(results[count] + indices))
    #     if len(results[count]) > 1:
    #         for item in results[count]:
    #             if not item in indices:
    #                 indices.append(item)
    #     else:
    #         if not results[count][0] in indices:
    #             indices.append(results[0])
    #     count += 1
    #     if count % 100 == 0:
    #         print count
    infile1.user_data[:] = 0
    while count < len(results):
        if count % 200 == 0:
            print count
        infile1.user_data[results[count]] = 1
        count += 1
    infile1.close()
    infile2.close()

infile_path1 = raw_input("Please input the first las file(big file):")
infile_path2 = raw_input("Please input the second las file to be minused(small file):")
distance = raw_input("Please input the distance threshold(Suggest one is 0.00001):")
if distance is '':
    distance =0.00001
point_cloud_minnus(infile_path1, infile_path2, float(distance))
os.system("pause")