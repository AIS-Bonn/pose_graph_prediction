from math import acos, atan2, cos, pi, sin

import numpy as np

from typing import List, Union


def get_angle_2d(vector_a_2d: Union[np.ndarray, List[float]],
                 vector_b_2d: Union[np.ndarray, List[float]]) -> float:
    angle = atan2(vector_b_2d[1], vector_b_2d[0]) - atan2(vector_a_2d[1], vector_a_2d[0])
    while angle < 0.0:
        angle += 2 * pi
    return angle


def get_angle_3d(vector_a_3d: np.ndarray,
                 vector_b_3d: np.ndarray) -> float:
    vector_a_length = np.linalg.norm(vector_a_3d)
    vector_b_length = np.linalg.norm(vector_b_3d)
    dot_product = np.dot(vector_a_3d, vector_b_3d)
    return acos(dot_product/(vector_a_length * vector_b_length))


def get_rotation_matrix_around_z_axis(angle):
    return np.array([[cos(angle), -sin(angle), 0],
                     [sin(angle),  cos(angle), 0],
                     [         0,           0, 1]])


def get_rotation_matrix_around_x_axis(angle):
    return np.array([[1,          0,           0],
                     [0, cos(angle), -sin(angle)],
                     [0, sin(angle),  cos(angle)]])


def get_rotation_matrix_around_y_axis(angle):
    return np.array([[ cos(angle), 0, sin(angle)],
                     [          0, 1,          0],
                     [-sin(angle), 0, cos(angle)]])
