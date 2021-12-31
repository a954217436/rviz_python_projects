#!/usr/bin/python2
import cv2
import numpy as np
import os
import pandas as pd 


TRACKING_COLUMN_NAMES = ['frame', 'track_id', 'type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top','bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']

IMU_COLUMN_NAMES = ['lat', 'lon', 'alt', 'roll', 'pitch', 'yaw', 'vn', 've', 'vf', 'vl', 'vu', 'ax', 'ay', 'az', 'af','al', 'au', 'wx', 'wy', 'wz', 'wf', 'wl', 'wu', 'posacc', 'velacc', 'navstat', 'numsats', 'posmode','velmode', 'orimode']

def read_camera(path):
    return cv2.imread(path)

def read_point_cloud(path):
    return np.fromfile(path,dtype=np.float32).reshape(-1,4)

def read_tracking(path):
    df = pd.read_csv(path, header=None, sep=' ')
    df.columns = TRACKING_COLUMN_NAMES
    df = df[df['track_id']>=0] # remove DontCare objects
    df.loc[df.type.isin(['Bus', 'Truck', 'Van', 'Tram']), 'type'] = 'Car' # Set all vehicle type to Car
    df = df[df.type.isin(['Car', 'Pedestrian', 'Cyclist'])]
    return df

def read_imu(path):
    df = pd.read_csv(path, header=None, sep=' ')
    df.columns = IMU_COLUMN_NAMES
    return df


def read_pred(path):
    preds = np.loadtxt(path)
    # print(path, " has %d results..."%len(preds))
    return preds


# def compute_3d_box_pred(h, w, l, x, y, z, yaw):
def compute_3d_box_pred(x, y, z, dx, dy, dz, yaw):
    # print(x, y, z, dx, dy, dz, yaw)
    
    R = np.array([[np.cos(yaw),  0, np.sin(yaw)], 
                  [0,            1,           0], 
                  [-np.sin(yaw), 0, np.cos(yaw)]])

    l, w, h = dx, dy, dz
    x_corners = [l/2, l/2, -l/2, -l/2,
                 l/2, l/2, -l/2, -l/2]

    y_corners = [0,  0, 0, 0,
                 -h,-h,-h,-h]

    z_corners = [w/2,-w/2,-w/2,w/2,
                 w/2,-w/2,-w/2,w/2]

    corners_3d_cam2 = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d_cam2[0,:] += x
    corners_3d_cam2[1,:] += y
    corners_3d_cam2[2,:] += z
    return corners_3d_cam2


# def compute_3d_box_pred(h, w, l, x, y, z, yaw):
def compute_3d_box_pred2(x, y, z, dx, dy, dz, yaw):
    # R = np.array([[np.cos(yaw),  0, np.sin(yaw)], 
    #               [0,            1,           0], 
    #               [-np.sin(yaw), 0, np.cos(yaw)]])
    R = np.array([[ np.cos(yaw), np.sin(yaw), 0], 
                  [-np.sin(yaw), np.cos(yaw), 0], 
                  [           0,           0, 1]])

    x_corners = [dx/2, dx/2, -dx/2, -dx/2,
                 dx/2, dx/2, -dx/2, -dx/2]

    y_corners = [dy/2, -dy/2, -dy/2, dy/2,
                 dy/2, -dy/2, -dy/2, dy/2]

    z_corners = [-dz/2,  -dz/2,  -dz/2,  -dz/2,
                 dz/2, dz/2, dz/2, dz/2]
    
    xyz = np.vstack([x_corners, y_corners, z_corners])
    # print(xyz)
    corners_3d_cam2 = np.dot(R, xyz)
    corners_3d_cam2[0,:] += x
    corners_3d_cam2[1,:] += y
    corners_3d_cam2[2,:] += z
    # print(corners_3d_cam2.shape)
    return corners_3d_cam2



    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()