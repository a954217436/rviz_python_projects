import cv2
import pickle
import open3d
import numpy as np

import matplotlib
import matplotlib.cm
matplotlib.use("cairo")
cmap = matplotlib.cm.get_cmap("viridis")


# connect vertic
LINES = [[0, 1], [1, 2], [2, 3], [3, 0]] # lower face
LINES+= [[4, 5], [5, 6], [6, 7], [7, 4]] # upper face
LINES+= [[4, 0], [5, 1], [6, 2], [7, 3]] # connect lower face and upper face
LINES+= [[1, 6], [2, 5]] # front face and draw x


def draw_bbox(img, bboxes):
    for box in bboxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
    return img


def read_detections(det_pkl):
    all_dets = pickle.load(open(det_pkl, "rb"))
    return all_dets


def get_dets_fname(all_data_dict, fname, thres=0.5, keys=['boxes', 'scores', 'classes']):
    # fname ： 'xxx.pcd'/'xxx.pkl'
    box3d_lidar = all_data_dict[fname][keys[0]]    
    scores      = all_data_dict[fname][keys[1]]    
    label_preds = all_data_dict[fname][keys[2]]
    mask = scores > thres

    return np.array(box3d_lidar[mask]), np.array(label_preds[mask]), np.array(scores[mask])

    # mask2 = box3d_lidar[:, 1] < 15  # -5
    # mask3 = box3d_lidar[:, 1] > 0  # -15
    # mask4 = box3d_lidar[:, 0] > 0  # 25
    # mask5 = box3d_lidar[:, 0] < 10  # 26
    # maskAll = mask * mask2 * mask3 * mask4 * mask5
    # return np.array(box3d_lidar[maskAll]), np.array(label_preds[maskAll]), np.array(scores[maskAll])


def read_pcd(file_path, padding=0):
    pcd = open3d.io.read_point_cloud(file_path)
    pcd_npy = np.asarray(pcd.points)
    if padding > 0:
        pcd_npy = np.append(pcd_npy, np.zeros((pcd_npy.shape[0], padding)), axis=1)
    return pcd_npy

def get_cv_image(img_path):
    img = cv2.imread(img_path)
    return np.empty(0) if img is None else img


def display_3dbox_on_image(img, corners8, lidar_to_image):
    # print(corners8)
    corners8_1 = np.concatenate((corners8, np.ones((8,1))), axis=1)
    corners8_proj = np.einsum('ij,bj->bi', lidar_to_image, corners8_1)
    # print(corners8_proj)
    mask = corners8_proj[:,2] > 0
    if not mask.all() > 0:
        return
    
    # Project the points onto the image.
    corners8_proj = corners8_proj[:,:2] / corners8_proj[:,2:3]
    
    # Filter points which are outside the image.
    mask = np.logical_and(
        np.logical_and(corners8_proj[:,0] > 0, corners8_proj[:,0] < img.shape[1]),
        np.logical_and(corners8_proj[:,1] > 0, corners8_proj[:,1] < img.shape[1]))
    if not mask.all() > 0:
        return

    # Draw a circle for each point.
    for i in range(corners8_proj.shape[0]):
        cv2.circle(img, (int(corners8_proj[i,0]),int(corners8_proj[i,1])), 5, (0,0,255), -1)
        # print("--"*20)
    
    for l in LINES:
        p1 = (int(corners8_proj[l[0], 0]), int(corners8_proj[l[0], 1]))
        p2 = (int(corners8_proj[l[1], 0]), int(corners8_proj[l[1], 1]))
        cv2.line(img, p1, p2, (255,0,255), 2, cv2.LINE_8)


def display_laser_on_image(img, pcl, lidar_to_image, pcl_attr=None):
    # Convert the pointcloud to homogeneous coordinates.
    pcl1 = np.concatenate((pcl, np.ones_like(pcl[:,0:1])), axis=1)
    # print(pcl1)
    # Transform the point cloud to image space.
    proj_pcl = np.einsum('ij,bj->bi', lidar_to_image, pcl1) 
    # print(proj_pcl)
    # Filter LIDAR points which are behind the camera.
    mask = proj_pcl[:,2] > 0
    proj_pcl = proj_pcl[mask]
    if pcl_attr is None:
        pcl_attr = np.ones_like(pcl[:,0:1])
    proj_pcl_attr = pcl_attr[mask]

    # Project the point cloud onto the image.
    proj_pcl = proj_pcl[:,:2]/proj_pcl[:,2:3]

    # Filter points which are outside the image.
    mask = np.logical_and(
        np.logical_and(proj_pcl[:,0] > 0, proj_pcl[:,0] < img.shape[1]),
        np.logical_and(proj_pcl[:,1] > 0, proj_pcl[:,1] < img.shape[1]))

    proj_pcl = proj_pcl[mask]
    proj_pcl_attr = proj_pcl_attr[mask]

    # Colour code the points based on attributes (distance/intensity...)
    coloured_intensity = 255*cmap(proj_pcl_attr[:,0]/30)
    # print(coloured_intensity)

    # Draw a circle for each point.
    for i in range(proj_pcl.shape[0]):
        cv2.circle(img, (int(proj_pcl[i,0]),int(proj_pcl[i,1])), 1, coloured_intensity[i], -1)


def compute_3d_cornors(bbox, pose=None):
    x, y, z, dx, dy, dz, yaw = bbox[:7]
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
    corners_3d_cam2 = np.zeros((4,8),dtype=np.float32)
    corners_3d_cam2[-1] = 1
    # print(xyz)
    corners_3d_cam2[:3] = np.dot(R, xyz)
    corners_3d_cam2[0,:] += x
    corners_3d_cam2[1,:] += y
    corners_3d_cam2[2,:] += z
    # print(corners_3d_cam2.shape)

    if pose is not None:
        pose = np.matrix(pose)
        corners_3d_cam2 = np.matmul(pose.I, corners_3d_cam2)

    return corners_3d_cam2[:3]



#########################################################################################
### pc 3D-bbox -> image 2D-bbox
#########################################################################################


def get_box_transformation_matrix(box):
    """
    Create a transformation matrix for a given box pose.
    """

    # tx,ty,tz = box.center_x,box.center_y,box.center_z
    tx,ty,tz = box[0], box[1], box[2]
    c = np.cos(box[6])
    s = np.sin(box[6])

    sl, sw, sh = box[3], box[4], box[5]    # 这里如果读取的是 det3d 的 det box，注意顺序

    return np.array([
        [ sl*c, -sw*s,  0, tx],
        [ sl*s,  sw*c,  0, ty],
        [    0,     0, sh, tz],
        [    0,     0,  0,  1]])


def get_3d_box_projected_corners(vehicle_to_image, boxes7d):
    """
    Get the 2D coordinates of the 8 corners of a label's 3D bounding box.
    args:
        vehicle_to_image: Transformation matrix from the vehicle frame to the image frame.
        boxes7d: The object label, numpy (7)
    """

    box = boxes7d

    # Get the vehicle pose
    box_to_vehicle = get_box_transformation_matrix(box)

    # Calculate the projection from the box space to the image space.
    box_to_image = np.matmul(vehicle_to_image, box_to_vehicle)

    # Loop through the 8 corners constituting the 3D box
    # and project them onto the image
    vertices = np.empty([2,2,2,2])
    for k in [0, 1]:
        for l in [0, 1]:
            for m in [0, 1]:
                # 3D point in the box space
                v = np.array([(k-0.5), (l-0.5), (m-0.5), 1.])

                # Project the point onto the image
                v = np.matmul(box_to_image, v)

                # If any of the corner is behind the camera, ignore this object.
                if v[2] < 0:
                    return None

                vertices[k,l,m,:] = [v[0]/v[2], v[1]/v[2]]

    vertices = vertices.astype(np.int32)
    return vertices


def compute_2d_bounding_box(img_or_shape, points):
    """Compute the 2D bounding box for a set of 2D points.
    
    img_or_shape: Either an image or the shape of an image.
                  img_or_shape is used to clamp the bounding box coordinates.
    
    points: The set of 2D points to use
    """

    if isinstance(img_or_shape, tuple):
        shape = img_or_shape
    else:
        shape = img_or_shape.shape

    # Compute the 2D bounding box and draw a rectangle
    x1 = np.amin(points[...,0])
    x2 = np.amax(points[...,0])
    y1 = np.amin(points[...,1])
    y2 = np.amax(points[...,1])

    x1 = min(max(0,x1),shape[1])
    x2 = min(max(0,x2),shape[1])
    y1 = min(max(0,y1),shape[0])
    y2 = min(max(0,y2),shape[0])

    return (x1,y1,x2,y2)