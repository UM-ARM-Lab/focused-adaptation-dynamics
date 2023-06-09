from math import pi

import numpy as np
import tensorflow as tf
from tensorflow_graphics.geometry.transformation.rotation_matrix_3d import from_euler


def homogeneous(points):
    return tf.concat([points, tf.ones_like(points[..., 0:1])], axis=-1)


def transform_points_3d(transform_matrix, points):
    """

    Args:
        transform_matrix: [b1, b2, ..., 4, 4]
        points: [b1, b2, ..., 3]

    Returns:

    """
    points_homo = homogeneous(points)
    points_homo = tf.expand_dims(points_homo, axis=-1)
    transformed_points = tf.matmul(transform_matrix, points_homo)
    return tf.squeeze(transformed_points, axis=-1)[..., :3]


def rotate_points_3d(rotation_matrix, points):
    """

    Args:
        rotation_matrix: [b1, b2, ..., b2, 3, 3]
        points: [b1, b2, ..., b2, 3]

    Returns:

    """
    rotated_points = tf.matmul(rotation_matrix, tf.expand_dims(points, axis=-1))
    return tf.squeeze(rotated_points, axis=-1)


def gather_transform(batch_indices, points, rotation, translation):
    rotation_gather = tf.gather(rotation, batch_indices)
    translation_gather = tf.gather(translation, batch_indices)
    return rotate_points_3d(rotation_gather, points) + translation_gather


def gather_translate(batch_indices, points, translation):
    translation_gather = tf.gather(translation, batch_indices)
    return points + translation_gather


def pairwise_squared_distances(a, b):
    """
    Adapted from https://github.com/ClayFlannigan/icp
    Computes pairwise distances between to sets of points

    Args:
        a: [b, ..., n, k]
        b:  [b, ..., m, k]

    Returns: [b, ..., n, m]

    """
    a_s = tf.reduce_sum(tf.square(a), axis=-1, keepdims=True)  # [b, ..., n, 1]
    b_s = tf.reduce_sum(tf.square(b), axis=-1, keepdims=True)  # [b, ..., m, 1]
    dist = a_s - 2 * tf.matmul(a, b, transpose_b=True) + tf.linalg.matrix_transpose(b_s)  # [b, ..., n, m]
    return dist


def best_fit_transform(a, b):
    """
    Adapted from https://github.com/ClayFlannigan/icp
    Calculates the least-squares best-fit transform that maps corresponding points a to b in m spatial dimensions
    Input:
      a: Nxm numpy array of corresponding points
      b: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps a on to b
      R: mxm rotation matrix
      t: mx1 translation vector
    """
    # get number of dimensions
    m = a.shape[1]

    # translate points to their centroids
    centroid_a = tf.reduce_mean(a, axis=0)
    centroid_b = tf.reduce_mean(b, axis=0)
    aa = a - centroid_a
    bb = b - centroid_b

    # rotation matrix
    h = tf.matmul(tf.transpose(aa, [1, 0]), bb)
    s, u, v = tf.linalg.svd(h)
    rotation = tf.matmul(tf.transpose(v, [1, 0]), tf.transpose(u, [1, 0]))

    # special reflection case
    if tf.linalg.det(rotation) < 0:
        v[m - 1, :] *= -1
        rotation = tf.matmul(tf.transpose(v, [1, 0]), tf.transpose(u, [1, 0]))

    # translation
    translation = tf.expand_dims(centroid_b, 1) - tf.matmul(rotation, tf.expand_dims(centroid_a, 1))

    return rotation, translation


def best_fit_translation(a, b):
    """
    Best fit translation that moves a to b
    Args:
        a: [b, ..., n, k], where k is usually 2 or 3
        b: [b, ..., n, k]

    Returns: [b, ..., k]

    """
    translation = tf.reduce_mean(b - a, axis=-2)
    return translation


def transform_dict_of_points_vectors(m: np.ndarray, d, keys):
    d_out = {}
    for k in keys:
        points = np.reshape(d[k], [-1, 3, 1])
        points_homo = np.concatenate([points, np.ones([points.shape[0], 1, 1])], axis=1)
        points_aug = np.matmul(m[None], points_homo)[:, :3, 0]
        d_out[k] = np.reshape(points_aug, -1).astype(np.float32)
    return d_out


def xyzrpy_to_matrices(params):
    """

    Args:
        params:  [b1,b2,...,6] in the form [x,y,z,roll,pitch,yaw]

    Returns: [b1,b2,...,4,4] with the assumption of roll, pitch, yaw, then translation (aka the normal thing)

    """
    translation = params[..., :3][..., None]
    angles = params[..., 3:]
    r33 = from_euler(angles)
    r34 = tf.concat([r33, translation], axis=-1)
    bottom_row = tf.constant([0, 0, 0, 1], dtype=tf.float32)
    bottom_row = tf.ones(params.shape[:-1] + [1, 4], tf.float32) * bottom_row
    matrices = tf.concat([r34, bottom_row], axis=-2)
    return matrices


# GENERATED BY SYMPY
def transformation_jacobian(params):
    """

    Args:
        params:  [b1,b2,...,6]

    Returns:

    """
    x, y, z, roll, pitch, yaw = tf.unstack(params, axis=-1)
    ones = tf.ones_like(x)
    zeros = tf.zeros_like(x)
    jacobian = tf.stack([
        [
            [zeros, zeros, zeros, ones],
            [zeros, zeros, zeros, zeros],
            [zeros, zeros, zeros, zeros],
            [zeros, zeros, zeros, zeros]
        ],
        [
            [zeros, zeros, zeros, zeros],
            [zeros, zeros, zeros, ones],
            [zeros, zeros, zeros, zeros],
            [zeros, zeros, zeros, zeros]
        ],
        [
            [zeros, zeros, zeros, zeros],
            [zeros, zeros, zeros, zeros],
            [zeros, zeros, zeros, ones],
            [zeros, zeros, zeros, zeros]
        ],
        [
            [zeros, tf.sin(pitch) * tf.cos(roll) * tf.cos(yaw) + tf.sin(roll) * tf.sin(yaw),
             -tf.sin(pitch) * tf.sin(roll) * tf.cos(yaw) + tf.sin(yaw) * tf.cos(roll), zeros],
            [zeros, tf.sin(pitch) * tf.sin(yaw) * tf.cos(roll) - tf.sin(roll) * tf.cos(yaw),
             -tf.sin(pitch) * tf.sin(roll) * tf.sin(yaw) - tf.cos(roll) * tf.cos(yaw), zeros],
            [zeros, tf.cos(pitch) * tf.cos(roll), -tf.sin(roll) * tf.cos(pitch), zeros],
            [zeros, zeros, zeros, zeros]
        ],
        [
            [-tf.sin(pitch) * tf.cos(yaw), tf.sin(roll) * tf.cos(pitch) * tf.cos(yaw),
             tf.cos(pitch) * tf.cos(roll) * tf.cos(yaw), zeros],
            [-tf.sin(pitch) * tf.sin(yaw), tf.sin(roll) * tf.sin(yaw) * tf.cos(pitch),
             tf.sin(yaw) * tf.cos(pitch) * tf.cos(roll), zeros],
            [-tf.cos(pitch), -tf.sin(pitch) * tf.sin(roll), -tf.sin(pitch) * tf.cos(roll), zeros],
            [zeros, zeros, zeros, zeros]
        ],
        [
            [-tf.sin(yaw) * tf.cos(pitch), -tf.sin(pitch) * tf.sin(roll) * tf.sin(yaw) - tf.cos(roll) * tf.cos(yaw),
             -tf.sin(pitch) * tf.sin(yaw) * tf.cos(roll) + tf.sin(roll) * tf.cos(yaw), zeros],
            [tf.cos(pitch) * tf.cos(yaw), tf.sin(pitch) * tf.sin(roll) * tf.cos(yaw) - tf.sin(yaw) * tf.cos(roll),
             tf.sin(pitch) * tf.cos(roll) * tf.cos(yaw) + tf.sin(roll) * tf.sin(yaw), zeros],
            [zeros, zeros, zeros, zeros],
            [zeros, zeros, zeros, zeros]
        ]
    ])  # [6,4,4,b1,b2,...]
    batch_axes = list(np.arange(len(params.shape) - 1) + 3)
    jacobian = tf.transpose(jacobian, batch_axes + [0, 1, 2])
    return jacobian


def quat_dist(quat1, quat2):
    return 1 - tf.square(tf.reduce_sum(quat1 * quat2))


def euler_angle_diff(euler1, euler2):
    abs_diff = tf.abs(euler1 - euler2)
    return tf.minimum(abs_diff, 2 * pi - abs_diff)


def densify_points(batch_size, points, num_densify=5):
    """
    Args:
        points: [b, n, 3]
    Returns: [b, n * num_density, 3]
    """
    if points.shape[1] <= 1:
        return points

    starts = points[:, :-1]
    ends = points[:, 1:]
    linspaced = tf.linspace(starts, ends, num_densify, axis=2)  # [b, n, num_density, 3]
    densitifed_points = tf.reshape(linspaced, [batch_size, -1, 3])
    return densitifed_points