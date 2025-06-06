import numpy as np


def generate_rotational_matrix(roll, pitch, yaw):
    c_roll = np.cos(roll)
    c_pitch = np.cos(pitch)
    c_yaw = np.cos(yaw)

    s_roll = np.sin(roll)
    s_pitch = np.sin(pitch)
    s_yaw = np.sin(yaw)

    row_mat_x = np.array([[1, 0, 0], [0, c_roll, -s_roll], [0, s_roll, c_roll]])
    row_mat_y = np.array([[c_pitch, 0, s_pitch], [0, 1, 0], [-s_pitch, 0, c_pitch]])
    row_mat_z = np.array([[c_yaw, -s_yaw, 0], [s_yaw, c_yaw, 0], [0, 0, 1]])

    return np.dot(np.dot(row_mat_x, row_mat_y), row_mat_z)
