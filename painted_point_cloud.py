import painted_pointcloud.calibration_kitti as calibration_kitti
from utils import  Plot
import numpy as np
import cv2
import os

def load_velo_scan(file):
    '''Load and parse a velodyne binary file'''
    file_type = file.split('.')
    if file_type[-1] == 'npy':
        voxel_add = np.load(file)
    else:
        voxel_add = np.fromfile(file,dtype=np.float32)

    voxel_add = voxel_add.reshape(-1, 4)

    return voxel_add[:,0:3]


def painted_point_cloud(calib_path,img_path,lidar_path,out_path,num):

    print('Now tracking num:',num)
    calib_path = calib_path + str(num).zfill(6)+ '.txt'
    img_path = img_path+ str(num).zfill(6) + '.png'
    lidar_path = lidar_path + str(num).zfill(6) +'.bin'
    out_path = out_path + str(num).zfill(6) +'.npy'

    raw_pointcloud =load_velo_scan(lidar_path)
    calib = calibration_kitti.Calibration(calib_path)
    img = cv2.imread(img_path)
    img_shape = img.shape
    pts_img,pts_rect_depth = calib.lidar_to_img(raw_pointcloud) # [N,3] in lidar to [N,2] in img

    pts_img = np.round(pts_img).astype(int)#四舍五入
    #after lidar to img,filtering points in img's range ;
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    ## in img's range ;all lidar points
    pts_img = pts_img[pts_valid_flag]
    pts_img[:, [0, 1]] = pts_img[:, [1, 0]]# height,width to width height[1242,375] to [375,1242]


    row = pts_img[:,0]
    col = pts_img[:,1]
    '''
    for i in range(1,img_shape[0]-1):
        for j in range(1,img_shape[1]-1):
            #print('before:',img[i,j])
            img[i,j] = (img[i-1,j-1]+img[i,j-1]+img[i+1,j-1]+img[i-1,j]+img[i+1,j]+img[i-1,j+1]+img[i,j+1]+img[i+1,j+1] +img[i,j])/9.
            #img[i,j] = (img[i, j - 1] + img[i - 1, j]+img[i,j]) / 3.
            #print('after,',img[i, j])
    '''
    img = img.astype(int)
    raw_pointcloud_color = img[row,col,:] # [N,3] b g r

    raw_pointcloud_color[:,[0,1,2]] = raw_pointcloud_color[:,[2,1,0]]
    painted_point_cloud_res = np.hstack((raw_pointcloud[pts_valid_flag],raw_pointcloud_color)) # [N,6] x,y,z,r,g,b
    Plot.draw_pc(painted_point_cloud_res)
    np.save(out_path,painted_point_cloud_res)
    return  painted_point_cloud_res

calib_path = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet/data/kitti/training/calib/'
img_path = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet/data/kitti/training/image_2/'
lidar_path = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet/data/kitti/training/velodyne/'
out_path = '/media/ddd/data2/kitti_detection/velodyne/training/velodyne_painted/'
lidar_list = os.listdir(lidar_path)
for l in lidar_list:
    if '.bin' in l:
        num = l.split('.')[0]
        painted_point_cloud(calib_path, img_path, lidar_path,out_path, num)
