# -*- coding: UTF-8 -*-
__author__ = 'Administrator'
# lidar_classification.py
import laspy
import numpy as np
import scipy.spatial
import struct
import liblas
import time
import os
import csv
# from matplotlib.mlab import PCA
# import matplotlib.pyplot as plt

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


def segmentation1(infilepath, distancethreshold, clustersizethreshold):
    """

    重构KDTREE的分割方法
    详细描述。

    """
    infile = laspy.file.File(infilepath)
    dataset = np.vstack([infile.x, infile.y, infile.z]).transpose()
    currentpoints = infile.points[:]
    print 'There are %d points' % len(dataset)
    pointused = 0
    while len(dataset) > 0:  # go over all points
        tree = scipy.spatial.KDTree(dataset)
        startidx = 0
        start = dataset[startidx]
        lenindice = 0  # 存储前一次循环的聚类块点的个数
        indice = tree.query_ball_point(start, distancethreshold)
        newindice = indice[:]  # newly added point index
        while len(indice) != lenindice:  # 当不再有新点加入时，找到一个聚类块
            lenindice = len(indice)  # 前一轮循环的点的个数
            neighbors = tree.query_ball_point(dataset[newindice], distancethreshold)  # 新加入点求临近点
            newindice = []  # 清除原始的新点
            for item in neighbors:  # 因为是多个点求出来的邻居点是个多维数组
                for subitem in item:
                    if indice.count(subitem) == 0:
                        newindice.append(subitem)
                        indice.append(subitem)
            # print 'new indice count is %d' % len(newindice)
            if len(indice) > 10000:
                break
            print 'indice count is %d' % len(indice)
        print 'No %d indice point count is %d' % (pointused, len(indice))
        if len(indice) > clustersizethreshold:
            outfile = laspy.file.File(str(pointused) + '.las', mode='w', header=infile.header)
            outfile.points = currentpoints[indice]
            outfile.close()
        pointused += 1
        dataset = np.delete(dataset, indice, 0)  # 更新数据
        currentpoints = np.delete(currentpoints, indice, 0)  # Update Points
    else:
        print 'Done!'
        infile.close()


def segmentation(infile):
    infile = laspy.file.File(infile)
    dataset = np.vstack([infile.x, infile.y, infile.z]).transpose()
    tree = scipy.spatial.KDTree(dataset)
    allindicesidx = range(len(dataset))  # store all points' index
    pointused = 0
    startidx = 0
    print 'There are %d points' % len(dataset)
    isover = False
    while not isover:  # go over all points
        start = dataset[startidx]
        lenindice = 0  # 存储前一次循环的聚类块点的个数
        indice = tree.query_ball_point(start, 0.5)
        newindice = indice[:]  # newly added point index
        while len(indice) != lenindice:  # 当不再有新点加入时，找到一个聚类块
            lenindice = len(indice)  # 前一轮循环的点的个数
            neighbors = tree.query_ball_point(dataset[newindice], 0.5)  # 新加入点求临近点
            newindice = []  # 清除原始的新点
            for item in neighbors:  # 因为是多个点求出来的邻居点是个多维数组
                for subitem in item:
                    if indice.count(subitem) == 0:
                        newindice.append(subitem)
                        indice.append(subitem)
            print 'new indice count is %d' % len(newindice)
            print 'indice count is %d' % len(indice)
        print 'No %d indice point count is %d' % (pointused, len(indice))
        if len(indice) > 90:
            outfile = laspy.file.File(str(pointused) + '.las', mode='w', header=infile.header)
            outfile.points = infile.points[indice]
            outfile.close()
        pointused += 1
        for item in indice:  # 删除已经存储好的点的标号
            allindicesidx[item] = -1
        for item in allindicesidx:
            if item != -1:
                startidx = item
                break
            else:
                continue
        isover = True
    else:
        print 'Done!'
        infile.close()


def remove_ground(infilepath, outfilepath, heighthreshhold):
    """
    移出地面点
    """
    infile = laspy.file.File(infilepath, mode='r')
    outfile = laspy.file.File(outfilepath, mode='w', header=infile.header)
    upgroundindex = np.where(infile.z > heighthreshhold)
    outfile.points = infile.points[upgroundindex]
    print '%d points has been extracted' % len(outfile.points)
    infile.close()
    outfile.close()


def testclassification(infilepath):
    infile = laspy.file.File(infilepath, mode='r')
    outfile = laspy.file.File(r'E:\class1.las', mode='w', header=infile.header)
    outfile.points = infile.points
    # outFile.close()
    infile.close()
    # outFile=laspy.file.File(r'E:\class1.las',mode='rw')
    indice2 = range(10000, 20000, 1)
    indice3 = range(20000, 40000, 1)
    outfile.classification[0] = 12
    outfile.Classification[2] = 1
    outfile.classification[indice2] = 2
    outfile.classification[indice3] = 3
    outfile.close()


def readpolygonfromshp(shppath):
    """
    从shp文件中读取polygon中的各个点返回

    http://blog.csdn.net/liminlu0314/article/details/8828983
    http://lab.osgeo.cn/1457.html
    """
    from osgeo import ogr
    from osgeo import gdal
    # 为了支持中文路径，请添加下面这句代码
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
    # 为了使属性表字段支持中文，请添加下面这句
    gdal.SetConfigOption("SHAPE_ENCODING", "")
    strvectorfile = shppath
    # 注册所有的驱动
    ogr.RegisterAll()
    # 打开数据
    ds = ogr.Open(strvectorfile, 0)
    if ds is None:
        print "打开文件【%s】失败！" % strvectorfile
        return

    print"打开文件【%s】成功！" % strvectorfile

    # 获取该数据源中的图层个数，一般shp数据图层只有一个，如果是mdb、dxf等图层就会有多个
    ilayercount = ds.GetLayerCount()

    # 获取第一个图层
    olayer = ds.GetLayerByIndex(0)
    if olayer is None:
        print "获取第%d个图层失败！\n" % 0
        return

    # 对图层进行初始化，如果对图层进行了过滤操作，执行这句后，之前的过滤全部清空
    olayer.ResetReading()

    # 输出图层中的要素个数
    print"要素个数 = %d" % olayer.GetFeatureCount(0)
    ofeature = olayer.GetNextFeature()
    ogeometry = ofeature.GetGeometryRef()
    geotype = ogeometry.GetGeometryType()
    if geotype != 3:
        return
    polygon = ogeometry.GetGeometryRef(0)  # 读取多边形
    return polygon.GetPoints()
    # 下面开始遍历图层中的要素
    # while ofeature is not None:
    # print "当前处理第%d个:" %ofeature.GetFID()
    # # 获取要素中的几何体
    # oGeometry =ofeature.GetGeometryRef()
    # geoType=oGeometry.GetGeometryType()
    # print geoType
    # #暂时只处理一个要素
    # break


def getptinpolygon(infile, ptpolygon):
    '''
    功能：判断点是否在多边形内
    方法：求解通过该点的水平线与多边形各边的交点
    结论：单边交点为奇数，成立!
    参数：
    POINT p 指定的某个点
    LPPOINT ptPolygon 多边形的各个顶点坐标（首末点可以不一致）
    int nCount 多边形定点的个数
    返回：
    多边形内的点
    源码改造自：http://hi.baidu.com/zle_yuanfang/item/1cac58d48f5a2a49dcf9be05
    '''
    indexarray = []
    index = 0
    for itemx, itemy in zip(infile.x, infile.y):  # 循环两个变量的方法
        ncross = 0
        i = 0
        for polygon in ptpolygon:
            p1 = ptpolygon[i]
            p2 = ptpolygon[(i + 1) % len(polygon)]
            i += 1
            # 求解 y=p.y 与 p1p2 的交点
            if p1[1] == p2[1]:  # p1p2 与 y=p0.y平行
                continue
            if itemy < min(p1[1], p2[1]):  # 交点在p1p2延长线上
                continue
            if itemy >= max(p1[1], p2[1]):  # 交点在p1p2延长线上
                continue
            # 求交点的 X 坐标 --------------------------------------------------------------
            x = float(itemy - p1[1]) * float(p2[0] - p1[0]) / float(p2[1] - p1[1]) + p1[0]
            if x > itemx:
                ncross += 1  # 只统计单边交点
        # 单边交点为偶数，点在多边形之外 ---
        if ncross % 2 == 1:
            indexarray.append(index)
        index += 1
    return indexarray


def ply2xyz(infile, outfilepath):
    file = open(infile, 'rb')
    # header=laspy.header.Header()
    # outfile=laspy.file.File(filepath,mode='w',header=header)
    # outfile.header.offset[0]=649000
    # outfile.header.offset[1]=6.84e+06
    # outfile.header.scale[0]=1
    # outfile.header.scale[1]=1
    # outfile.header.scale[2]=1
    # outfile.header.count=12000000
    while 1:
        line = file.readline()
        if line == 'end_header\n':
            break
        print line
    count = 0
    f = open(outfilepath, 'a')
    while count < 12000000:
        data = struct.unpack("dfffffffffIIBB", file.read(54))
        print count
        # 顺序为 x y z GPSTIME X Y Z Intensity range theta id class num_echo nb_of_echo
        line = str(data[1] + 649000) + ' ' + str(data[2] + 6.84e+06) + ' ' + str(data[3]) + ' ' + str(
            data[0]) + ' ' + str(data[4]) + ' ' + str(data[5]) + ' ' + str(data[6]) + ' ' + str(
            (int)(-1.0 * data[7] + 0.5)) + ' ' + str(data[8]) + ' ' + str(data[9]) + ' ' + str(data[10]) + ' ' + str(
            data[11]) + ' ' + str(data[12]) + ' ' + str(data[13]) + '\n'
        # x y z Intensity id class
        line1 = str(data[1] + 649000) + ' ' + str(data[2] + 6.84e+06) + ' ' + str(data[3]) + ' ' + str(
            (int)(-10.0 * data[7] + 0.5)) + ' ' + str(data[10]) + ' ' + str(data[11]) + '\n'
        f.write(line1)
        count += 1
        # outfile.X[count]=data[1]
        # outfile.Y[count]=data[2]
        # outfile.Z[count]=data[3]
        # outfile.Intensity[count]=(int)(-1.0*data[7]+0.5)  #强度四舍五入转化为整数值
        # outfile.pt_src_id[count]=data[10]
        # outfile.return_num = data[12]
        # outfile.num_returns = data[13]
    f.close()
    file.close()


def addinfo2las(laspath, plypath):
    classlist = [303040200, 202040000, 202020000, 202030000, 203000000, 302020300, 0, 302020900, 202010000, 303030302,
                 302020400, 302021000, 303020000, 304020000, 303020200, 303020600, 303020300, 302020600, 301000000,
                 302030400, 304040000, 303030202]
    classlist.sort()
    f = laspy.file.File(laspath, mode='rw')
    file = open(plypath, 'rb')
    while 1:
        line = file.readline()
        if line == 'end_header\n':
            break
        print line
    count = 0
    while count < 12000000:
        data = struct.unpack("dfffffffffIIBB", file.read(54))  # 根据属性表作的分解
        # print count
        # f.intensity[count]=str((int)(-10.0*data[7]+0.5))
        # f.pt_src_id[count]=str(data[10])
        # f.raw_classification[count]
        f.return_num[count] = int(data[12])
        f.num_returns[count] = int(data[13])
        count += 1
        print count
    file.close()
    f.close()


def countclass(plypath):
    clslist = []
    f = laspy.file.File(laspath, mode='rw')
    file = open(plypath, 'rb')
    while 1:
        line = file.readline()
        if line == 'end_header\n':
            break
        print line
    count = 0
    while count < 12000000:
        data = struct.unpack("dfffffffffIIBB", file.read(54))
        print count
        if clslist.count(data[11]) == 0:
            clslist.append(data[11])
        count += 1
    print clslist


def ply2las(plypath, laspath):
    file = open(plypath, 'rb')
    while 1:
        line = file.readline()
        if line == 'end_header\n':
            break
        print line
    count = 0
    header = liblas.header.Header()
    header.major_version = 1
    header.minor_version = 2
    header.data_format_id = 3
    header.scale = [0.01, 0.01, 0.01]
    header.offset = [650903.38, 6861122.49, 38.99]
    f = liblas.file.File(laspath, header, 'w')
    classlist = [303040200, 202040000, 202020000, 202030000, 203000000, 302020300, 0, 302020900, 202010000, 303030302,
                 302020400, 302021000, 303020000, 304020000, 303020200, 303020600, 303020300, 302020600, 301000000,
                 302030400, 304040000, 303030202]
    classlist.sort()
    minx, maxx = 0, 0
    miny, maxy = 0, 0
    minz, maxz = 0, 0
    while count < 12000000:
        data = struct.unpack("dfffffffffIIBB", file.read(54))
        # print count
        # 顺序为 x y z GPSTIME X Y Z Intensity range theta id class num_echo nb_of_echo
        line = str(data[1] + 649000) + ' ' + str(data[2] + 6.84e+06) + ' ' + str(data[3]) + ' ' + str(
            data[0]) + ' ' + str(data[4]) + ' ' + str(data[5]) + ' ' + str(data[6]) + ' ' + str(
            (int)(-1.0 * data[7] + 0.5)) + ' ' + str(data[8]) + ' ' + str(data[9]) + ' ' + str(data[10]) + ' ' + str(
            data[11]) + ' ' + str(data[12]) + ' ' + str(data[13]) + '\n'
        # x y z Intensity id class
        # line1=str(data[1]+649000)+' ' +str(data[2]+6.84e+06)+' '+str(data[3])+' '+str((int)(-10.0*data[7]+0.5))+' '+str(data[10])+' '+str(data[11])+'\n'
        # f.write(line1)
        point = liblas.point.Point()
        x = (data[1] - 1903.38) / header.scale[0]
        y = (data[2] - 21122.49) / header.scale[1]
        z = (data[3] - 38.99) / header.scale[2]
        minx, maxx = min(minx, x), max(maxx, x)
        miny, maxy = min(miny, y), max(maxy, y)
        minz, maxz = min(minz, z), max(maxz, z)
        intensity = (int)(-10.0 * data[7] + 0.5)  # 强度四舍五入转化为整数值
        point.raw_x = int(x)
        point.raw_y = int(y)
        point.raw_z = int(z)
        point.intensity = intensity
        point.user_data = data[10]
        point.set_point_source_id(data[10])
        point.classification = classlist.index(data[11])
        point.return_number = data[12]
        point.number_of_returns = data[13]
        f.write(point)
        count += 1
        print count
    f.header.min = [minx, miny, minz]
    f.header.max = [maxx, maxy, maxz]
    f.close()
    file.close()
    print 'Done!'


def getplyinfo(plypath):
    file = open(plypath, 'rb')
    while 1:
        line = file.readline()
        if line == 'end_header\n':
            break
        print line
    count = 0
    header = liblas.header.Header()
    header.major_version = 1
    header.minor_version = 2
    header.data_format_id = 3
    header.offset = [649000, 6.84e+06, 0]
    f = liblas.file.File(laspath, header, 'w')
    while count < 12000000:
        data = struct.unpack("dfffffffffIIBB", file.read(54))
        print data
        count += 1


def get_descent_eignvalues(dataset, tree, x, y, z, radius):
    """
    计算数据的对应的维度值1d,2d,3d,并按由大到小顺序返回
    """
    from scipy import linalg as la

    indices = tree.query_ball_point([x, y, z], radius)
    if len(indices) <= 3:  # 邻居点少于三个的情况，计算不了协方差矩阵和特征值。让它的熵值最大，然后就可以继续选点；
        return
    idx = tuple(indices)
    data = np.vstack([dataset[idx, 0], dataset[idx, 1], dataset[idx, 2]])
    cov = np.cov(data)
    evals = la.eigh(cov, eigvals_only=True)  # #如果用通常用的la.eig 函数的话特征值会出现复数
    evals = np.abs(evals)  # 因为超出精度范围，所以值有可能出现负数，这里折中取个绝对值，因为反正都很小，可以忽略不计
    index = evals.argsort()[::-1]
    # evects=evects[:,index]
    evals = evals[index]
    return evals


def get_normals(dataset, tree, x, y, z, radius):
    from scipy import linalg as la

    indices = tree.query_ball_point([x, y, z], radius)
    if len(indices) <= 3:
        return
    idx = tuple(indices)
    data = np.vstack([dataset[idx, 0], dataset[idx, 1], dataset[idx, 2]])
    cov = np.cov(data)
    evals, evects = la.eigh(cov)
    evals = np.abs(evals)
    index = evals.argsort()[::-1]
    evects = evects[:, index]
    return evects[2]


def get_eigenvectors(dataset, tree, x, y, z, radius):
    from scipy import linalg as la

    indices = tree.query_ball_point([x, y, z], radius)
    if len(indices) <= 3:
        return
    idx = tuple(indices)
    data = np.vstack([dataset[idx, 0], dataset[idx, 1], dataset[idx, 2]])
    cov = np.cov(data)
    evals, evects = la.eigh(cov)
    evals = np.abs(evals)
    index = evals.argsort()[::-1]
    evects = evects[:, index]
    return evects


def get_dimensions(dataset, tree, x, y, z, radius):
    """
    return the dimensionality values of a point within a radius

    the returned values are stored in a list in sequence,
    they 1d value, 2d value, 3d value respectivesly

    Args:
        dataset: the input x,y,z data of all points
        tree: kdtree build upon dataset
        x,y,z: the coordinate of the point to get dimensionality
        radius: the radius to define the dimensionality

    Returns:
        a list which stores the 1d value, 2d value, 3d value in order

    Raises:

    """
    import math

    evals = get_descent_eignvalues(dataset, tree, x, y, z, radius)
    if evals is None:
        return
    mu1 = math.sqrt(evals[0])
    mu2 = math.sqrt(evals[1])
    mu3 = math.sqrt(evals[2])
    # 出现重复点也可能导致mu1为0，让它的熵值最大，然后就可以继续选点；
    if mu1 == 0:
        return
    a1d, a2d, a3d = 1.0 - mu2 / mu1, mu2 / mu1 - mu3 / mu1, mu3 / mu1
    return [a1d, a2d, a3d]


def entropy_function(dimensions):
    import math

    if dimensions[0] <= 0 or dimensions[1] <= 0 or dimensions[2] <= 0:  #
        return 3.40e+38
    else:
        return -dimensions[0] * math.log(dimensions[0]) - dimensions[1] * math.log(dimensions[1]) - dimensions[
                                                                                                        2] * math.log(
            dimensions[2])


def get_optimal_radius(dataset, kdtree, x, y, z, rmin, rmax, deltar):
    """
    通过计算最小熵值，来求最优临近半径
    """
    rtemp = rmin
    dimensions = get_dimensions(dataset, kdtree, x, y, z, rtemp)
    if dimensions is None:
        ef = 3.40282e+038
    else:
        ef = entropy_function(dimensions)
    efmin = ef
    rotpimal = rmin
    rtemp += deltar
    count = 1
    while rtemp < rmax:
        dimensions = get_dimensions(dataset, kdtree, x, y, z, rtemp)
        # 按e**（0.12*count**2)递增
        # rtemp += 2.71828 ** (0.12 * count * count) * deltar
        rtemp += 0.08 * count
        count += 1
        if dimensions is None:
            continue
        ef = entropy_function(dimensions)
        if ef < efmin:
            efmin = ef
            rotpimal = rtemp
    return rotpimal


def write_normals(infilepath, outfilepath, radius):
    """
    write the normals(nx, ny, nz) value of points in a file to a new field

    if the field does not exist, add a new field named normal
    """
    from scipy import linalg as la

    infile = laspy.file.File(infilepath, mode='rw')
    outfile = laspy.file.File(outfilepath, mode='w', header=infile.header)
    dataset = np.vstack((infile.x, infile.y, infile.z)).transpose()
    kd_tree = scipy.spatial.cKDTree(dataset)
    count = 0
    for x, y, z in zip(infile.x, infile.y, infile.z):
        indices = kd_tree.query_ball_point([x, y, z], radius)
        # 邻居点少于三个的情况，计算不了协方差矩阵和特征值。让它的熵值最大，然后就可以继续选点；
        if len(indices) <= 3:
            continue
        idx = tuple(indices)
        data = np.vstack([dataset[idx, 0], dataset[idx, 1], dataset[idx, 2]])
        cov = np.cov(data)
        eign_values, eign_vectors = la.eig(cov)
        index = eign_values.argsort()[::-1]
        eign_vectors = eign_vectors[:, index]
        infile.gps_time[count] = eign_vectors[2][2]
        count += 1
        print count
    infile.close()
    print 'Write %d Normal values successfully!' % count


def filter_ground_by_region_growing(infilepath, radius1, radius2, heightdif1, heightdif2):
    """
    提取前景点：通过region growing方法，先找出当前最高点附近radius1范围内的最低点。
    限制最高点与最低高差大于heightdif1，其他点与种子点距离大于radius2；
    并且与地面高差大于heightdif2
    """
    infile = laspy.file.File(infilepath, mode="rw")
    infile.pt_src_id[:] = 0
    infile.flag_byte[:] = 0
    dataset = np.vstack([infile.x, infile.y, infile.z]).transpose()
    tree = scipy.spatial.cKDTree(dataset)
    print 'There are %d points' % len(dataset)
    segmentcount = 0
    while True:  # go over all points
        restidx = list(np.where(infile.flag_byte == 0))
        tempxarray = infile.x[restidx]
        tempyarray = infile.y[restidx]
        tempzarray = infile.z[restidx]
        maxzidx = np.argmax(tempzarray)
        coords = np.vstack((infile.x[restidx], infile.y[restidx])).transpose()
        # Pull off the first point
        first_point = np.array([tempxarray[maxzidx], tempyarray[maxzidx]])

        # Calculate the euclidean distance from all points to the first point
        distances = np.sum((coords - first_point) ** 2, axis=1)

        # Create an array of indicators for whether or not a point is less than
        # 1 units away from the first point
        keep_points = list(np.where(distances < 1))

        minz = min(infile.z[keep_points])

        # 如果最高点与周围最低点高差小于heightdif1则表示，剩下的全部属于背景点了
        if tempzarray[maxzidx] - minz < heightdif1:
            return

        start = np.array([tempxarray[maxzidx], tempyarray[maxzidx], tempzarray[maxzidx]])
        lenindice = 0  # 存储前一次循环的聚类块点的个数
        indicetemp = tree.query_ball_point(start, radius2)
        #与最低点高差满足要求的留下
        indice = []
        for item in indicetemp:
            if infile.z[item] - minz > heightdif2:
                indice.append(item)
        # newly added point index
        newindice = indice[:]

        # 当不再有新点加入时，表示找到一个聚类块
        while len(indice) != lenindice:
            lenindice = len(indice)  # 前一轮循环的点的个数
            # 新加入点求临近点
            neighbors = tree.query_ball_point(dataset[newindice], radius2)
            # 清除原始的新点
            newindice = []
            # 循环多个点的邻居点集合
            for item in neighbors:  # 因为是多个点求出来的邻居点是个多维数组
                for subitem in item:
                    if indice.count(subitem) == 0 and infile.z[subitem] - minz > heightdif2:
                        newindice.append(subitem)
                        indice.append(subitem)
                        # print 'new indice count is %d' % len(newindice)
                        # if len(indice)>10000:
                        #     break
                        # print 'indice count is %d' % len(indice)
        print 'No %d indice point count is %d' % (segmentcount, len(indice))
        # #判断分割块大小
        # if len(indice) > 10:
        infile.pt_src_id[indice] = segmentcount
        # 用这个字段来记录该点是否已经提取过
        infile.flag_byte[indice] = 1
        segmentcount += 1
        # outfile = laspy.file.File(str(pointused) + '.las', mode='w', header=infile.header)
        # outfile.points = currentpoints[indice]
        # outfile.close()


        # dataset = np.delete(dataset, indice, 0)  # 更新数据
        # currentpoints = np.delete(currentpoints, indice, 0)  # Update Points
    else:
        print 'Done!'
        infile.close()


def filter_ground(infile_path, angle_threshold):
    """
    filter ground points out

    默认与水平面角度小于angle_threshold的点为地面点
    """

    infile = laspy.file.File(infile_path, mode='r')
    outfile = laspy.file.File(r'out.las', mode='w', header=infile.header)
    dataset = np.vstack([infile.x, infile.y, infile.z]).transpose()
    tree = scipy.spatial.cKDTree(dataset)
    infile.flag_byte[:] = 0
    segmentcount = 0
    while True:
        restidx = list(np.where(infile.flag_byte == 0))
        if len(restidx) == 0:
            return
        tempxarr = infile.x[restidx]
        tempyarr = infile.y[restidx]
        tempzarr = infile.z[restidx]
        minzidx = np.argmin(infile.z[restidx])
        start = np.array((tempxarr[minzidx], tempyarr[minzidx], tempzarr[minzidx]))
        indices = []
        indices = tree.query_ball_point(start, 0.8)
        newindice[:] = indices[:]
        lenindice = 0
        while len(indices) != lenindice:
            lenindice = len(indices)
            # 新加入点求临近点
            neighbors = tree.query_ball_point(dataset[newindice], 0.8)
            # 清除原始的新点
            newindice = []
            # 循环多个点的邻居点集合
            for item in neighbors:  # 因为是多个点求出来的邻居点是个多维数组
                for subitem in item:
                    if indices.count(subitem) == 0:
                        newindice.append(subitem)
                        indices.append(subitem)
        infile.flag_byte = 1
        if len(indices) > 50000:
            infile.raw_classification[indices] = 1
        if len(indices) > 50:
            infile.pt_src_id[indices] = segmentcount
            segmentcount += 1


def classify_ground(infile_path):
    infile = laspy.file.File(infile_path, mode='r')
    plane_indices = np.where(infile.optimal_dimensionalities == 2)
    dataset = np.vstack((infile.x[plane_indices], infile.y[plane_indices, infile.z[plane_indices]])).transpose()
    tree = scipy.spatial.cKDTree(dataset)


def filter_scatter_points(infile_path, outfile_path, radius, point_count):
    """
    去除点云文件中的离散点

    radius范围内如果少于point_count个点，这个点就属于离散点
    """
    infile = laspy.file.File(infile_path, mode='r')
    out_indices = []
    outfile = laspy.file.File(outfile_path, mode='w', header=infile.header)
    dataset = np.vstack([infile.x, infile.y, infile.z]).transpose()
    tree = scipy.spatial.cKDTree(dataset)
    count = 0
    for x, y, z in zip(infile.x, infile.y, infile.z):
        indices = tree.query_ball_point([x, y, z], radius)
        if len(indices) >= point_count:
            out_indices.append(count)
        count += 1
        print count
    outfile.points = infile.points[out_indices]
    outfile.close()
    infile.close()
    print 'Filter done!'


def write_optimal_local_information(infilepath):
    """
    write the dimensionality and normals and principle directions

    write the dimensionalities, normals, principle directions of all points in a file
    to new fields when the radius is optimal
    Args:
        infilepath: the path of the file to be written
    """
    import os

    start = time.clock()
    # the name of output file
    outfile_path = infilepath.replace('.las', '_optimal.las')
    infile = laspy.file.File(infilepath, mode='r')
    outfile = laspy.file.File(outfile_path, mode='w', header=infile.header)
    outfile.define_new_dimension(OPTIMAL_RADIUS_NAME, 9, 'optimal radius')
    outfile.define_new_dimension(OPTIMAL_DIMENSION_NAME, 1, 'dimensionality with optimal radius')

    outfile.define_new_dimension(OPTIMAL_NX_NAME, 9, 'normals nx with optimal radius')
    outfile.define_new_dimension(OPTIMAL_NY_NAME, 9, 'normals ny with optimal radius')
    outfile.define_new_dimension(OPTIMAL_NZ_name, 9, 'normals nz with optimal radius')

    outfile.define_new_dimension(OPTIMAL_PX_NAME, 9, 'principle directions px with optimal radius')
    outfile.define_new_dimension(OPTIMAL_PY_NAME, 9, 'principle directions py with optimal radius')
    outfile.define_new_dimension(OPTIMAL_PZ_NAME, 9, 'principle directions pz with optimal radius')
    for dimension in infile.point_format:
        data = infile.reader.get_dimension(dimension.name)
        outfile.writer.set_dimension(dimension.name, data)
    dataset = np.vstack([outfile.x, outfile.y, outfile.z]).transpose()
    kdtree = scipy.spatial.cKDTree(dataset)
    print len(outfile.points)
    length = len(outfile.points)
    count = 0
    try:
        while count < length:
            x, y, z = outfile.x[count], outfile.y[count], outfile.z[count]
            optimal_radius = get_optimal_radius(dataset, kdtree, x, y, z, 0.08, 1.0, 0.08)
            outfile.optimal_radius[count] = optimal_radius
            eigenvectors = get_eigenvectors(dataset, kdtree, x, y, z, optimal_radius)
            if eigenvectors is None:
                count += 1
                continue
            outfile.optimal_nx[count] = eigenvectors[2][0]
            outfile.optimal_ny[count] = eigenvectors[2][1]
            outfile.optimal_nz[count] = eigenvectors[2][2]
            outfile.optimal_px[count] = eigenvectors[0][0]
            outfile.optimal_py[count] = eigenvectors[0][1]
            outfile.optimal_pz[count] = eigenvectors[0][2]
            dimensions = get_dimensions(dataset, kdtree, x, y, z, optimal_radius)
            # if the point has no dimension values it means it doesn't have enough neighbouring points
            if dimensions is None:
                outfile.optimal_dimensionalities[count] = 3
            else:
                dimension = max(dimensions[0], dimensions[1], dimensions[2])
                if dimensions[0] == dimension:
                    outfile.optimal_dimensionalities[count] = 1
                elif dimensions[1] == dimension:
                    outfile.optimal_dimensionalities[count] = 2
                elif dimensions[2] == dimension:
                    outfile.optimal_dimensionalities[count] = 3
            count += 1
            if count % 100 == 0:
                print count
    except:
        print time.clock() - start
        print "Wrong"
        time.sleep(1000)
    else:
        infile.close()
        outfile.close()
        print time.clock() - start
        print 'Done!'
        os.system("pause")


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
    print 'detected %d points' % len(results)
    while '' in results:
        results.remove('')
    print 'detected %d not empty points' % len(results)
    # indices = []
    count = 0
    # while count < len(results):
    # # indices = list(set(results[count] + indices))
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


def add_dimention(infile_path, outfile_path, names, types, descriptions):
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
    import csv

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

    import csv
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

    # traverse all the voxel to assign location label
    while voxel_count < length - vertical_count_threshold:
        v1 = voxel_code_int_array[voxel_count:voxel_count + vertical_count_threshold]
        v2 = voxel_code_int_array[voxel_count + 1:voxel_count + 1 + vertical_count_threshold]
        v = list(map(lambda x: x[0] - x[1], zip(v2, v1)))
        # judge if the vertical_count_threshold number of voxel value are continuous
        if v == temp:
            location_id_array[voxel_count:voxel_count + 1 + vertical_count_threshold] = location_count
            voxel_count += vertical_count_threshold +1
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
    import csv
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


def pole_part_detection(in_file, out_file, inner_radius, outer_radius, cyliner_height, ratio_threshold, voxel_size, max_height):
    """
    detecting pole part of objects

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
    import csv
    with open(in_file, 'rb') as incsv:
        reader = csv.reader(incsv)
        lines = [[row[0], row[1], row[2], row[3], row[4]] for row in reader]
    original_code_array = np.array(lines)[:, 0]
    original_x_int_array = np.vectorize(int)(map(lambda x: x[:4], original_code_array))
    original_y_int_array = np.vectorize(int)(map(lambda x: x[4:8], original_code_array))
    original_z_int_array = np.vectorize(int)(map(lambda x: x[8:12], original_code_array))
    points_count_in_one_voxel = np.array(lines)[:, 1]
    intensity_array = np.array(lines)[:, 2]
    olocation_array = np.array(lines)[:, 3]
    mlocation_array = np.array(lines)[:, 4]
    code_valid_array = original_code_array[mlocation_array != '0']
    mlocation_valid_array = mlocation_array[mlocation_array != '0']

    # none-repetetive mlocation values
    mlocation_set = list(set(mlocation_valid_array))

    # pole number array to store all the voxel pole number, the value is 0 if it is not a pole
    pole_number_array = np.array([0] * len(mlocation_array))

    pole_count = 1

    # loop of traversing all the mlocations
    for mlocation in mlocation_set:
        # all the indices in one mlocation
        one_mlocation_code_indices = np.where(mlocation_array == mlocation)
        one_mlocation_code_indices = list(one_mlocation_code_indices[0])

        # all the codes of one mlocation
        one_mlocation_code_array = original_code_array[one_mlocation_code_indices]

        x_int_array = np.vectorize(int)(map(lambda x: x[:4], one_mlocation_code_array))
        y_int_array = np.vectorize(int)(map(lambda x: x[4:8], one_mlocation_code_array))
        z_int_array = np.vectorize(int)(map(lambda x: x[8:12], one_mlocation_code_array))

        # counter for calculating the number of voxel the perform the cylinder analysis, we declude the lowest voxel
        vertical_count = int(cyliner_height / voxel_size) + 1

        # selecte codes under a height threshold in one mlocation
        minz = min(z_int_array)

        # judge if the mlocation satisfy rule 1
        # if x_int_array.max() - x_int_array.min() > inner_radius / voxel_size \
        #         or y_int_array.max() - y_int_array.min() >= inner_radius /voxel_size \
        #         or z_int_array.max() - z_int_array.min() > max_height / voxel_size:
        #     continue
        if z_int_array.max() - z_int_array.min() > max_height / voxel_size:
            continue

        # finding neighbors, only six-connected neighbors are taken into consideration

        # finding the center of mlocation
        olocation_in_one_mlocation_array = olocation_array[one_mlocation_code_indices]
        from collections import Counter
        counts = Counter(olocation_in_one_mlocation_array)
        top_olocation = counts.most_common(1)[0][0]
        center_x = x_int_array[olocation_in_one_mlocation_array == top_olocation][0]
        center_y = y_int_array[olocation_in_one_mlocation_array == top_olocation][0]

        # 计算内圈，外圈的voxel的相邻个数,比如（0.6，1）对应（1，2）；（0.8，1.4）对应（2，4)
        inner_radius_count = (int(inner_radius / voxel_size + 0.5)) / 2
        outer_radius_count = (int((outer_radius - inner_radius) / voxel_size) + 1) / 2 + inner_radius_count
        in_x_valid = np.logical_and(original_x_int_array - center_x >= -inner_radius_count,
                                    original_x_int_array - center_x <= inner_radius_count)
        out_x_valid1 = np.logical_and(outer_radius_count >= original_x_int_array - center_x,
                                      original_x_int_array - center_x > inner_radius_count)
        out_x_valid2 = np.logical_and(-outer_radius_count <= original_x_int_array - center_x,
                                      original_x_int_array - center_x < -inner_radius_count)
        out_x_valid = out_x_valid1 | out_x_valid2
        in_y_valid = np.logical_and(original_y_int_array - center_y >= -inner_radius_count,
                                    original_y_int_array - center_y <= inner_radius_count)
        out_y_valid1 = np.logical_and(outer_radius_count >= original_y_int_array - center_y,
                                      original_y_int_array - center_y> inner_radius_count)
        out_y_valid2 = np.logical_and(-outer_radius_count <= original_y_int_array - center_y,
                                      original_y_int_array - center_y < -inner_radius_count)
        z_valid = np.logical_and(original_z_int_array > minz + 1, original_z_int_array < minz + vertical_count + 1)
        out_y_valid = out_y_valid1 | out_y_valid2
        in_voxels = np.where(in_x_valid & in_y_valid & z_valid)
        out_voxels = np.where(out_x_valid & out_y_valid & z_valid)
        outer_voxel_list = list(out_voxels[0])
        inner_voxel_list = list(in_voxels[0])

        # mlocation_points_count = sum(np.vectorize(int)(points_count_in_one_voxel[cylinder_indices]))
        if len(outer_voxel_list) == 0:
            outer_point_count = 0
        else:
            outer_point_count = sum(np.vectorize(int)(points_count_in_one_voxel[outer_voxel_list]))

        inner_point_count = sum(np.vectorize(int)(points_count_in_one_voxel[inner_voxel_list]))
        if float(inner_point_count) / (inner_point_count + outer_point_count) > ratio_threshold:
            # pole_number_array[mlocation_array == mlocation] = pole_count
            pole_number_array[inner_voxel_list] = pole_count
            # pole_number_array[outer_voxel_list] = 255
            pole_count += 1
        # if float(len(one_mlocation_code_array)) / len(neighbors_list) > ratio_threshold:
        #     pole_number_array[mlocation_array == mlocation] = pole_count
        #     pole_count += 1
    length = len(original_code_array)
    with open(out_file, 'wb') as outcsv:
        writer = csv.writer(outcsv)
        row = 0
        while row < length:
            writer.writerow([original_code_array[row], points_count_in_one_voxel[row], intensity_array[row],
                             olocation_array[row], mlocation_array[row], pole_number_array[row]])
            row += 1


def remove_background(in_file, out_file, start_height, radius, max_height, voxel_size):
    """
    find out both the background voxel and foreground voxel and label them 0 and 1 respectively

    Args:
        in_file: the path of input csv file comtaining code, point number, olocation, mlocation, pole part column
        out_file: add a new column to store the background foreground label
        start_height: the cylinder will start from start_height to build the cylinder to filter foreground
    """
    import csv
    with open(in_file, 'rb') as incsv:
        reader = csv.reader(incsv)
        lines = [[row[0], row[1], row[2], row[3], row[4], row[5]] for row in reader]
    original_code_array = np.array(lines)[:, 0]
    original_x_int_array = np.vectorize(int)(map(lambda x: x[:4], original_code_array))
    original_y_int_array = np.vectorize(int)(map(lambda x: x[4:8], original_code_array))
    original_z_int_array = np.vectorize(int)(map(lambda x: x[8:12], original_code_array))

    point_counts_array = np.array(lines)[:, 1]
    intensity_array = np.array(lines)[:, 2]
    olocation_array = np.array(lines)[:, 3]
    mlocation_array = np.array(lines)[:, 4]
    pole_id_array = np.array(lines)[:, 5]
    pole_location_set = list(set(pole_id_array[pole_id_array != '0']))
    foreground_indices = []

    for pole_location in pole_location_set:
        voxel_indices = np.where(pole_id_array == pole_location)
        voxel_indices = list(voxel_indices[0])
        center_x = int(sum(original_x_int_array[voxel_indices]) / len(original_x_int_array[voxel_indices]) + 0.5)
        center_y = int(sum(original_y_int_array[voxel_indices]) / len(original_y_int_array[voxel_indices]) + 0.5)
        center_z = min(original_z_int_array[voxel_indices]) + start_height / voxel_size

        x_valid = np.logical_and(original_x_int_array - center_x <= radius /voxel_size,
                                 original_x_int_array - center_x >= -radius /voxel_size)
        y_valid = np.logical_and(original_y_int_array - center_y <= radius / voxel_size,
                                 original_y_int_array - center_y >= -radius / voxel_size)
        z_valid = np.logical_and(original_z_int_array >= center_z,
                                 original_z_int_array <= max_height / voxel_size + center_z)
        valid_voxel_indices = np.where(x_valid & y_valid & z_valid)
        foreground_indices = np.append(foreground_indices, valid_voxel_indices)
        foreground_indices = np.array(list(set(foreground_indices)))

    length = len(original_code_array)
    foreground_flag = np.array([0] * length)
    foreground_flag[list(foreground_indices)] = 1
    foreground_flag[pole_id_array != '0'] = 1
    count = 0
    with open(out_file, 'wb') as outcsv:
        writer = csv.writer(outcsv)
        while count < length:
            writer.writerow([original_code_array[count], point_counts_array[count], intensity_array[count],
                             olocation_array[count], mlocation_array[count], pole_id_array[count],
                             foreground_flag[count]])
            count += 1


def connected_component_labeling(in_file, out_file):
    """
    perform connected component labeling on voxels

    we implement the algorithm in :
        http://www.codeproject.com/Articles/336915/Connected-Component-Labeling-Algorithm#Unify
    """
    import csv
    with open(in_file) as incsv:
        reader = csv.reader(incsv)
        lines = [[row[0], row[1], row[2], row[3], row[4], row[5], row[6]] for row in reader]

    original_code_array = np.array(lines)[:, 0]

    point_counts_array = np.array(lines)[:, 1]
    intensity_array = np.array(lines)[:, 2]
    olocation_array = np.array(lines)[:, 3]
    mlocation_array = np.array(lines)[:, 4]
    pole_id_array = np.array(lines)[:, 5]
    foreground_flag = np.array(lines)[:, 6]

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
        if len(label_array[parent_array == unique_parent]) > 50:
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
    with open(out_file, 'wb') as outcsv:
        writer = csv.writer(outcsv)
        while count < voxel_length:
            writer.writerow([original_code_array[count], point_counts_array[count], intensity_array[count],
                             olocation_array[count], mlocation_array[count], pole_id_array[count],
                             foreground_flag[count], all_label_array[count]])
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


def label_points_location(las_path, voxel_path):
    """
    label the location label in the voxel path to corresponding points in las file

    """
    import csv
    lasfile = laspy.file.File(las_path, mode="rw")
    location_label = []
    with open(voxel_path) as voxel_file:
        reader = csv.reader(voxel_file)
        location_label = [[row[2], row[3], row[4], row[5], row[6]] for row in reader]
    point_count = 0
    lasfile.user_data[:] = 0
    for voxel_index in lasfile.voxel_index:
        lasfile.olocation[point_count] = location_label[voxel_index][0]
        lasfile.mlocation[point_count] = location_label[voxel_index][1]
        # user_data is corresponding to pole location
        lasfile.user_data[point_count] = location_label[voxel_index][2]
        # classification is corresponding to fore or back ground
        lasfile.raw_classification[point_count] = location_label[voxel_index][3]
        #
        lasfile.pt_src_id[point_count] = location_label[voxel_index][4]
        point_count += 1
    lasfile.close()


    # indices = np.where(olocation_label_array != '0')
    # indices = list(indices[0])
    # for indice in indices:
    #     olabel = olocation_label_array[indice]
    #     indices_point = np.where(lasfile.voxel_index == indice)
    #     indices_point = list(indices_point[0])
    #     lasfile.olocation[indices_point] = olabel
    #     lasfile.olocation[indices_point] =
    # lasfile.close()
    # voxel_file.close()
    # print 'Done'



# start = time.clock()
# point_cloud_minnus(infile_path1, infile_path2, 0.00001)
# print time.clock() - start


# infile_path = r'france_optimal_dimensionality.las'
# write_optimal_local_information(infile_path)

print '''Welcome to the  YYClassification System!!

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

voxel_size = 0.25
# minimun height of voxel to constitute a postion
min_position_height = 1.0

# poles are suposed to be lower than the max_height_of_pole
max_height_of_pole = 15

# the height of the cylinder to calculate the ratio of the pole parts
cylinder_height = 1.4
cylinder_ratio = 0.95
inner_radius = 0.6
outer_radius = 1.4

# the cylinder height radius to find foreground voxels
cylinder_radius_for_foreground = 5
cylinder_height_for_foreground = 12

outlas = infilepath[:-4] + '_' + str(voxel_size) + '_' + str(int(min_position_height / voxel_size)) + '.las'
outcsv = outlas[:-4] + '.csv'
outcsv1 = outlas[:-4] + '_1.csv'
outcsv2 = outlas[:-4] + '_2.csv'
outcsv3 = outlas[:-4] + '_3.csv'
outcsv4 = outlas[:-4] + '_4.csv'
outcsv5 = outlas[:-4] + '_5.csv'
loop = True
while loop:
    x = raw_input("\n Step:")
    if x == 'q':
        loop = False
    elif x == '1':
        start = time.clock()
        add_dimention(infilepath, outlas, ["voxel_index", "olocation", "mlocation"], [5, 5, 3],
                      ["voxel num the point in", "original location label", "merged location label"])
        print "Step 1 costs seconds", time.clock() - start
    elif x == '2':
        start = time.clock()
        voxelization(outlas, outcsv, 0.3)
        print "Step 2 costs seconds", time.clock() - start

    elif x == '3':
        start = time.clock()
        original_localization(outcsv, outcsv1, 9)
        print "Step 3 costs seconds", time.clock() - start
    elif x == '4':
        start = time.clock()
        merging_neighbor_location(outcsv1, outcsv2)
        print "Step 4 costs seconds", time.clock() - start
    elif x == '5':
        start = time.clock()
        pole_part_detection(outcsv2, outcsv3, inner_radius, outer_radius, cylinder_height, cylinder_ratio, voxel_size,
                       max_height_of_pole)
        print "Step 5 costs seconds", time.clock() - start
    elif x == '6':
        start = time.clock()
        remove_background(outcsv3, outcsv4, min_position_height, cylinder_radius_for_foreground,
                          cylinder_height_for_foreground, voxel_size)
        print 'Step 6 costs', time.clock() - start
    elif x == '7':
        start = time.clock()
        connected_component_labeling(outcsv4, outcsv5)
        print 'Step 7 costs', time.clock() - start
    elif x == '8':
        start = time.clock()
        label_points_location(outlas, outcsv5)
        print "Step 8 costs seconds", time.clock() - start
        loop = False
    else:
        start = time.clock()
        print ' 1.Adding dimensions...'
        add_dimention(infilepath, outlas, ["voxel_index", "olocation", "mlocation"], [5, 5, 3],
                      ["voxel num the point in", "original location label", "merged location label"])
        print "Step 1 costs seconds ", time.clock() - start
        start = time.clock()

        print '\n 2.Voxelization...voxel size is %f m ' % voxel_size
        voxelization(outlas, outcsv, voxel_size)
        print "Step 2 costs seconds ", time.clock() - start
        start = time.clock()

        print '''\n 3.Detecting original position...
        at least %d continuous vertical voxels constitute a location''' % int(min_position_height / voxel_size)
        original_localization(outcsv, outcsv1, voxel_size, min_position_height)
        print "Step 3 costs seconds", time.clock() - start

        print '''\n 4.Calculating merged position...'''
        tart = time.clock()
        merging_neighbor_location(outcsv1, outcsv2)
        print "Step 4 costs seconds", time.clock() - start

        print "\n 5. Detecting poles..."
        start =time.clock()
        pole_part_detection(outcsv2, outcsv3, inner_radius, outer_radius, cylinder_height, cylinder_ratio, voxel_size,
                       max_height_of_pole)
        print "Step 5 costs seconds", time.clock() - start

        print '''\n 6.Removing background...'''
        start = time.clock()
        remove_background(outcsv3, outcsv4, min_position_height, cylinder_radius_for_foreground,
                          cylinder_height_for_foreground, voxel_size)
        print 'Step 6 costs', time.clock() - start

        print '''\n 7.Connected component labeling...'''
        start = time.clock()
        connected_component_labeling(outcsv4, outcsv5)
        print 'Step 7 costs', time.clock() - start

        print '''\n 8.Labeling points...'''
        start = time.clock()
        label_points_location(outlas, outcsv5)
        print "Step 8 costs seconds", time.clock() - start
        loop = False
os.system('pause')

"""
过滤地面点

infilepath = r"france.las"
infile = laspy.file.File(infilepath,mode='r')
outfile= laspy.file.File(r'francefore.las', mode='w',header=infile.header)
outfile1= laspy.file.File(r'franceback.las', mode='w',header=infile.header)
#找出二维平面点
planeidx = list(np.where(np.logical_and((infile.user_data==2),(np.abs(infile.gps_time)>0.866))))
backidx =  list(np.where(np.logical_or((infile.user_data!=2),(np.abs(infile.gps_time)<=0.866))))
outfile1.points = infile.points[planeidx]
outfile.points = infile.points[backidx]

outfile.close()
outfile1.close()
infile.close()

"""

"""

# 计算dimentionality

import time
import os
start=time.clock()
infilepath = r"ply2las3-clip.las"
infile=laspy.file.File(infilepath,mode='rw')
dataset = np.vstack([infile.x, infile.y, infile.z]).transpose()
kdtree=scipy.spatial.cKDTree(dataset)
print len(infile.points)
count=0
try:
    for x,y,z in zip(infile.x,infile.y,infile.z) :
        optimalradius=getoptimalradius(dataset,kdtree,x,y,z,0.1,0.58,0.08)
        print optimalradius
        infile.flag_byte[count]=int(100*optimalradius)
        a1d,a2d,a3d=getdimention(dataset,kdtree,x,y,z,optimalradius)
        dimention=max(a1d,a2d,a3d)
        if a1d==dimention:
            infile.user_data[count]=1
        elif a2d==dimention:
            infile.user_data[count]=2
        elif a3d==dimention:
            infile.user_data[count]=3
        count+=1
        print count

    # x=infile.x[3596]
    # y=infile.y[3596]
    # z=infile.z[3596]
    # optimalradius=getoptimalradius(dataset,kdtree,x,y,z,0.1,0.58,0.08)
    # a1d,a2d,a3d=getdimention(dataset,kdtree,x,y,z,optimalradius)
except:
    print time.clock()-start
    print "Wrong"
    os.system("pause")
else:
    infile.close()
    print time.clock()-start
    print 'Done!'
    os.system("pause")
"""

'''
#转换PLY为LAS

# infile= r"E:\DATA\LIDAR\France MLS Data\classified Cassette_idclass\Cassette_GT.ply"
# outfile="ply2las1.xyz"
# ply2las(infile,outfile)
laspath="ply2las3.las"
plypath=r"E:\DATA\LIDAR\France MLS Data\classified Cassette_idclass\Cassette_GT.ply"
ply2las(plypath,laspath)
# getplyinfo(plypath)
# addinfo2las(laspath,plypath)
# countclass(plypath)
'''

'''

分割

infilepath = r'E:\Thesis Experiment\Data\Test\lyx.las'
outfilepath = r'E:\Thesis Experiment\Data\Test\lyxupground.las'
#removegroud(infilepath,outfilepath,16.9)
starttime = datetime.datetime.now()
segmentation1(outfilepath,0.1,100)
endtime = datetime.datetime.now()
interval=(endtime.minute - starttime.minute)
print 'Segmentation used %d minutes' %interval
'''

'''

# 裁减las文件

infilepath = r"ply2las3 - Cloud1.las"
# shpPath = r"E:\Thesis Experiment\Data\Test\Clip1.shp"
shpPath = r"E:\MyProgram\Python\HelloWorld\shp\ply2las3 - Cloud1_aoi.shp"
inFile = laspy.file.File(infilepath, mode='r')
count = 4
polygon = readpolygonfromshp(shpPath)
inPointsIndex = getptinpolygon(inFile, polygon)
outFile = laspy.file.File(r"ply2las3-clip.las", mode='w', header=inFile.header)
outFile.points = inFile.points[inPointsIndex]
print '共裁减了%d个点' %len(outFile.points)
inFile.close()
outFile.close()
print "Done"
'''

'''

测试分类

infilepath = r"E:\class1.las"
infile=laspy.file.File(infilepath, mode='rw')
infile.red[:]=255
infile.blue[:]=0
infile.green[:]=0
a=range(10000)
b=range(10000,30001,1)
# infile.classification[:]=3
infile.raw_classification[a]=4 # 设置分类是要设置raw_classification
infile.raw_classification[b]=8
infile.classification_flags[:]=5
infile.close()
'''

'''

提取地上点

inFile = laspy.file.File(r"E:\test.las", mode='r')
upground_index=np.where(np.logical_and(inFile.z>17.1,inFile.z<50))
upground_point=inFile.points[upground_index]
outFile=laspy.file.File(r"E:\test1.las",mode='w',header=inFile.header)
inFile.close()
# plt.hist(inFile.z, 200)
# plt.show()
outFile.points=upground_point
outFile.close()
print 'Done!'
'''

'''

官方代码

#inFile = laspy.file.File("./laspytest/data/simple.las", mode = "r")
inFile = laspy.file.File(r"E:\DATA\LIDAR\Lyx\3Merged.las", mode='r')
# Grab all of the points from the file.
point_records = inFile.points
print 'Length of record is %d' % len(point_records)
# Grab just the X dimension from the file, and scale it.
def scaled_x_dimension( las_file ):
    x_dimension = las_file.X
    scale = las_file.header.scale[0]
    offset = las_file.header.offset[0]
    return ( x_dimension * scale + offset )
def scaled_z_dimension( las_file ):
    z_dimension = las_file.Z
    scale = las_file.header.scale[2]
    offset = las_file.header.offset[2]
    return ( z_dimension * scale + offset )
scaled_x = scaled_x_dimension(inFile)
scaled_z = scaled_z_dimension(inFile)
print 'Original X is %ld' % inFile.X[0]
print 'ScaleX is %e' % inFile.header.scale[0]
print 'ScaleZ is %e' % inFile.header.scale[2]
print 'Offset is %f' % inFile.header.offset[0]
print 'Scaled X is %f' % scaled_x[0]
print inFile.X[1], inFile.X[2], inFile.X[3]
print scaled_x[1], scaled_x[2], scaled_x[3]
print inFile.Z[1],inFile.Z[2],inFile.Z[3]
print scaled_z[1],scaled_z[2],scaled_z[3]'''
