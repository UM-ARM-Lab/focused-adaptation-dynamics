from typing import Optional, Dict, List

import numpy as np
import tensorflow as tf

from link_bot_planning.experiment_scenario import ExperimentScenario
from link_bot_pycommon.link_bot_sdf_utils import get_local_env_and_origin_differentiable
from moonshine.action_smear_layer import smear_action


def differentiable_get_local_env():
    pass


def old_raster(state, res, origin, h, w):
    """
    state: [batch, n]
    res: [batch] scalar float
    origin: [batch, 2] index (so int, or technically float is fine too)
    h: scalar int
    w: scalar int
    return: [batch, h, w, n_points]
    """
    b = int(state.shape[0])
    points = np.reshape(state, [b, -1, 2])
    n_points = int(points.shape[1])

    res = res[0]  # NOTE: assume constant resolution

    # points[:,1] is y, origin[0] is row index, so yes this is correct
    row_y_indices = (points[:, :, 1] / res + origin[:, 0:1]).astype(np.int64).flatten()
    col_x_indices = (points[:, :, 0] / res + origin[:, 1:2]).astype(np.int64).flatten()
    channel_indices = np.tile(np.arange(n_points), b)
    batch_indices = np.repeat(np.arange(b), n_points)

    # filter out invalid indices, which can happen during training
    state_images = np.zeros([b, h, w, n_points], dtype=np.float32)
    valid_indices = np.where(np.all([row_y_indices >= 0,
                                     row_y_indices < h,
                                     col_x_indices >= 0,
                                     col_x_indices < w], axis=0))

    state_images[batch_indices[valid_indices],
                 row_y_indices[valid_indices],
                 col_x_indices[valid_indices],
                 channel_indices[valid_indices]] = 1.0
    return state_images


@tf.function
def raster_differentiable(state, res, origin, h, w):
    """
    Even though this data is batched, we use singular and reserve plural for sequences in time
    state: [batch, n]
    res: [batch] scalar float
    origins: [batch, 2] index (so int, or technically float is fine too)
    h: scalar int
    w: scalar int
    return: [batch, h, w, n_points]
    """
    b = int(state.shape[0])
    points = tf.reshape(state, [b, -1, 2])
    n_points = int(points.shape[1])

    res = res[0]

    k = 10000.0

    ## Below is a un-vectorized implementation, which is much easier to read and understand
    # rope_images = np.zeros([b, h, w, n_points], dtype=np.float32)
    # for batch_index in range(b):
    #     for point_idx in range(n_points):
    #         for row, col in np.ndindex(h, w):
    #             point_in_meters = points[batch_index, point_idx]
    #             pixel_center_in_meters = idx_to_point(row, col, res, origins[batch_index])
    #             squared_distance = np.sum(np.square(point_in_meters - pixel_center_in_meters))
    #             pixel_value = np.exp(-k*squared_distance)
    #             rope_images[batch_index, row, col, point_idx] += pixel_value
    # rope_images = rope_images

    ## vectorized implementation

    # add h & w dimensions
    tiled_points = tf.expand_dims(tf.expand_dims(points, axis=1), axis=1)
    tiled_points = tf.tile(tiled_points, [1, h, w, 1, 1])
    pixel_row_indices = tf.range(0, h, dtype=tf.float32)
    pixel_col_indices = tf.range(0, w, dtype=tf.float32)
    # pixel_indices is b, n_points, 2
    pixel_indices = tf.stack(tf.meshgrid(pixel_row_indices, pixel_col_indices), axis=2)
    # add batch dim
    pixel_indices = tf.expand_dims(pixel_indices, axis=0)
    pixel_indices = tf.tile(pixel_indices, [b, 1, 1, 1])

    # shape [b, h, w, 2]
    pixel_centers = (pixel_indices - origin) * res

    # add n_points dim
    pixel_centers = tf.expand_dims(pixel_centers, axis=3)
    pixel_centers = tf.tile(pixel_centers, [1, 1, 1, n_points, 1])

    squared_distances = tf.reduce_sum(tf.square(pixel_centers - tiled_points), axis=4)
    pixel_values = tf.exp(-k * squared_distances)
    rope_images = tf.reshape(pixel_values, [b, h, w, n_points])
    ##############################################################################
    # FIXME: figure out whether to do clipping or normalization
    ##############################################################################
    return rope_images


def raster(state, res, origin, h, w):
    rope_image = raster_differentiable(state=state,
                                       origin=origin,
                                       res=res,
                                       h=h,
                                       w=w)
    return rope_image.numpy()


@tf.function
def make_transition_image(full_env,
                          full_env_origin,
                          res,
                          planned_states,
                          action,
                          planned_next_states,
                          scenario: ExperimentScenario,
                          local_env_h: int,
                          local_env_w: int,
                          action_in_image: Optional[bool] = False):
    """
    :param planned_states: each element should be [batch,n_state]
    :param action: [batch,n_action]
    :param planned_next_states: each element should be [batch,n_state]
    :param res: [batch]
    :param action_in_image: include new channels for actions
    :return: [batch,n_points*2+n_action+1], aka  [batch,n_state+n_action+1]
    """
    local_env_center_point = scenario.local_environment_center_differentiable(planned_states)
    # TODO: these functions are scattered all over the place, organize better
    local_env, local_env_origin = get_local_env_and_origin_differentiable(center_point=local_env_center_point,
                                                                          full_env=full_env,
                                                                          full_env_origin=full_env_origin,
                                                                          res=res,
                                                                          local_h_rows=local_env_h,
                                                                          local_w_cols=local_env_w)
    local_env = local_env
    local_env_origin = local_env_origin
    tf.print(local_env[0,0,0], local_env_origin[0,0])

    concat_args = [tf.zeros([1, local_env_h, local_env_w, 1])]
    # for planned_state in planned_states.values():
    #     planned_rope_image = raster_differentiable(state=planned_state,
    #                                                res=res,
    #                                                origin=local_env_origin,
    #                                                h=local_env_h,
    #                                                w=local_env_w)
    #     concat_args.append(planned_rope_image)
    # for planned_next_state in planned_next_states.values():
    #     planned_next_rope_image = raster_differentiable(state=planned_next_state,
    #                                                     origin=local_env_origin,
    #                                                     res=res,
    #                                                     h=local_env_h,
    #                                                     w=local_env_w)
    #     concat_args.append(planned_next_rope_image)
    #
    # if action_in_image:
    #     # FIXME: use tf to make sure its differentiable
    #     action_image = smear_action(action, local_env_h, local_env_w)
    #     concat_args.append(action_image)
    image = tf.concat(concat_args, axis=3)
    return image


def raster_rope_images(planned_states: List[Dict],
                       res,
                       origin,
                       h: float,
                       w: float):
    """
    Raster all the state into one fixed-channel image representation using color gradient in the green channel
    :param planned_states: each element is [batch, n_state]
    :param res: [batch]
    :param origin: [batch, 2]
    :param h: scalar
    :param w: scalar
    :return: [batch, h, w, 2 * n_points]
    """
    n_time_steps = len(planned_states)
    binary_rope_images = []
    time_colored_rope_images = []
    for t in range(n_time_steps):
        planned_state_t = planned_states[t]
        # iterate over the dict, each element of which is a component of our state
        for s_t_k in planned_state_t.values():
            rope_img_t = old_raster(state=s_t_k, origin=origin, res=res, h=h, w=w)
            time_color = float(t) / n_time_steps
            time_color_image_t = rope_img_t * time_color
            binary_rope_images.append(rope_img_t)
            time_colored_rope_images.append(time_color_image_t)
    binary_rope_images = tf.reduce_sum(binary_rope_images, axis=0)
    time_colored_rope_images = tf.reduce_sum(time_colored_rope_images, axis=0)
    rope_images = tf.concat((binary_rope_images, time_colored_rope_images), axis=3)
    return rope_images


def make_traj_images(full_env,
                     full_env_origin,
                     res,
                     states: List[Dict]):
    """
    :param full_env: [batch, h, w]
    :param full_env_origin:  [batch, 2]
    :param res: [batch]
    :param states: each element is [batch, time, n]
    :return: [batch, h, w, 3]
    """
    h = int(full_env.shape[1])
    w = int(full_env.shape[2])

    # add channel index
    full_env = tf.expand_dims(full_env, axis=3)

    rope_imgs = raster_rope_images(states, res, full_env_origin, h, w)

    image = tf.concat((full_env, rope_imgs), axis=3)
    return image
