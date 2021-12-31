import numpy as np

def correction_angle(theta):
    if theta >= np.pi: 
        theta -= np.pi * 2    # make the theta still in the range
    if theta < -np.pi: 
        theta += np.pi * 2
    return theta


def acute_angle(new_theta, predicted_theta):
    correction_theta = predicted_theta

    # if the angle of two theta is not acute angle
    if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:
        correction_theta += np.pi       
        correction_theta = correction_angle(correction_theta)

    # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
    if abs(new_theta - correction_theta) >= np.pi * 3 / 2.0:
        if new_theta > 0: 
            correction_theta += np.pi * 2
        else: 
            correction_theta -= np.pi * 2
    return correction_theta