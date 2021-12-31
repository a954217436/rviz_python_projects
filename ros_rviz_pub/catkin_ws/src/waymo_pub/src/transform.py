import math
import numpy as np


def get_box_transformation_matrix(box):
    """
    Create a transformation matrix for a given box pose.
    """

    # tx,ty,tz = box.center_x,box.center_y,box.center_z
    tx,ty,tz = box[0], box[1], box[2]
    c = math.cos(box[6])
    s = math.sin(box[6])

    sl, sw, sh = box[3], box[4], box[5]    # 这里如果读取的是 det3d 的 det box，注意顺序

    return np.array([
        [ sl*c, -sw*s,  0, tx],
        [ sl*s,  sw*c,  0, ty],
        [    0,     0, sh, tz],
        [    0,     0,  0,  1]])


# def get_image_transform(camera_calibration):
def get_image_transform(extrinsic, intrinsic):
    """ 
        For a given camera calibration(waymo), compute the transformation matrix
        from the vehicle reference frame to the image space.
        args:
            extrinsic: numpy (4 * 4)
            intrinsic: numpy (9)  [fx, fy, cx, cy, k1, k2, p1, p2, k3]
    """

    # # TODO: Handle the camera distortions
    # extrinsic = np.array(camera_calibration.extrinsic.transform).reshape(4,4)
    # intrinsic = camera_calibration.intrinsic

    # Camera model:
    # | fx  0 cx 0 |
    # |  0 fy cy 0 |
    # |  0  0  1 0 |
    camera_model = np.array([
        [intrinsic[0], 0, intrinsic[2], 0],
        [0, intrinsic[1], intrinsic[3], 0],
        [0, 0,                       1, 0]])

    # Swap the axes around
    axes_transformation = np.array([
        [0,-1,0,0],
        [0,0,-1,0],
        [1,0,0,0],
        [0,0,0,1]])

    # Compute the projection matrix from the vehicle space to image space.
    vehicle_to_image = np.matmul(camera_model, np.matmul(axes_transformation, np.linalg.inv(extrinsic)))
    return vehicle_to_image


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