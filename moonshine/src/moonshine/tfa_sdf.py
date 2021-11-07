import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_addons.image import euclidean_dist_transform
from tensorflow_felzenszwalb_edt import edt1d

import ros_numpy
import rospy
from arc_utilities import ros_init
from sensor_msgs.msg import PointCloud2


def get_grid_points(origin_point, res, shape):
    indices = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.stack(indices, axis=-1)
    points = (indices * res) - origin_point
    return points


def visualize_sdf(pub, sdf: np.ndarray, shape, res, origin_point):
    points = get_grid_points(origin_point, res, shape)
    list_of_tuples = [(p[0], p[1], p[2], d) for p, d in zip(points.reshape([-1, 3]), sdf.flatten())]
    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('distance', np.float32)]
    np_record_array = np.array(list_of_tuples, dtype=dtype)
    msg = ros_numpy.msgify(PointCloud2, np_record_array, frame_id='world', stamp=rospy.Time.now())
    pub.publish(msg)


def build_sdf_2d(vg, res, origin_point):
    binary_vg_batch = tf.cast([vg], tf.uint8)
    filled_distance_field = euclidean_dist_transform(binary_vg_batch)
    empty_distance_field = tf.maximum(euclidean_dist_transform(1 - binary_vg_batch) - 1, 0)
    distance_field = empty_distance_field + -filled_distance_field

    plt.figure()
    plt.imshow(distance_field[0, :, :, 0])
    plt.yticks(range(vg.shape[0]))
    plt.xticks(range(vg.shape[1]))
    plt.show()


def build_sdf_3d(vg, res, origin_point):
    """

    Args:
        vg: [b, h, w ,c] of type float32
        res:
        origin_point:

    Returns:

    """
    # NOTE: this is how the morphological EDT works, you first scale everything by a big number
    s = np.sum(np.array(vg.shape) ** 2)
    s = 10
    filled_vg = (1 - vg) * s
    filled_distance_field = edt1d(filled_vg, 1)[0]
    filled_distance_field = edt1d(filled_distance_field, 2)[0]
    filled_distance_field = edt1d(filled_distance_field, 3)[0]

    empty_vg = vg * s
    empty_distance_field = edt1d(empty_vg, 1)[0]
    empty_distance_field = edt1d(empty_distance_field, 2)[0]
    empty_distance_field = edt1d(empty_distance_field, 3)[0]
    empty_distance_field

    distance_field = empty_distance_field + -filled_distance_field

    plt.figure()
    plt.imshow(distance_field[0, :, :, 0])
    plt.yticks(range(vg.shape[1]))
    plt.xticks(range(vg.shape[2]))
    plt.show()


@ros_init.with_ros("tfa_sdf")
def main():
    sdf_pub = rospy.Publisher("sdf", PointCloud2, queue_size=10)

    res = [0.04]
    shape = [1, 25, 20, 1]
    origin_point = np.array([[0, 0, 0]], dtype=np.float32)

    vg = np.zeros(shape)
    vg[0, :5, :5, :5] = 1.0

    sdf = build_sdf_3d(vg, res, origin_point)

    for b in range(1):
        visualize_sdf(sdf_pub, sdf[0].numpy(), shape, res[0], origin_point[0])


if __name__ == '__main__':
    main()
