#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by You Li on 2014-12-10 0010


class VoxelFile:
    """
    File to store voxel information. Provides access to most voxel functionality,
    """
    def __init__(self, filename,voxelsize=0.2):
        """
        initiate a voxel file instance, file format is *.csv
        Args:
            filename: file name of the csv file
            voxelsize: size for the voxel
        """

        self.filename = filename
        self.voxelsize = voxelsize

    def get_voxelsize(self):
        return self.voxelsize

    def set_voxelsize(self, value):
        self.voxelsize = value

    doc = 'voxel size of the voxel file'
    voxelsize = property(get_voxelsize, set_voxelsize, None, doc)

    def get_code(self):
        pass

    def set_code(self, code):
        pass

    doc = 'the code of voxel'
    code = property(get_code, set_code, None, doc)

    def get_intensity(self):
        pass

    def set_intensity(self, code):
        pass
    doc = 'intensity of voxel'
    code = property(get_intensity, set_intensity, None, doc)

    def get_olocation(self):
        pass

    def set_olocation(self, code):
        pass
    doc = 'the original location index of each voxel'
    code = property(get_olocation, set_olocation, None, doc)

    def get_mlocation(self):
        pass

    def set_mlocation(self, code):
        pass
    doc = 'the merged location index of each voxel'
    code = property(get_mlocation, set_mlocation, None, doc)

    def get_background(self):
        pass

    def set_background(self, code):
        pass

    doc = 'flags that indicate if the voxel is background or not'
    code = property(get_background, set_background, None, doc)

    def get_code(self):
        pass

    def set_code(self, code):
        pass

    doc = 'the code of voxel'
    code = property(get_code, set_code, None, doc)
    def get_code(self):
        pass

    def set_code(self, code):
        pass

    doc = 'the code of voxel'
    code = property(get_code, set_code, None, doc)
    def get_code(self):
        pass

    def set_code(self, code):
        pass

    doc = 'the code of voxel'
    code = property(get_code, set_code, None, doc)

    def get_code(self):
        pass

    def set_code(self, code):
        pass

    doc = 'the code of voxel'
    code = property(get_code, set_code, None, doc)

    def get_code(self):
        pass

    def set_code(self, code):
        pass

    doc = 'the code of voxel'
    code = property(get_code, set_code, None, doc)
if __name__ == '__main__':
    print 'default voxelsize is 0.2 meters'