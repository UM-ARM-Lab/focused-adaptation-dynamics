from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from link_bot_data.visualization import plot_rope_configuration, my_arrow, plot_arrow
from link_bot_pycommon import link_bot_sdf_utils


def plot_classifier_data(
        planned_env: Optional = None,
        planned_env_extent: Optional = None,
        planned_state: Optional = None,
        planned_next_state: Optional = None,
        planned_env_origin: Optional = None,
        res: Optional = None,
        state: Optional = None,
        next_state: Optional = None,
        title='',
        action: Optional = None,
        actual_env: Optional = None,
        actual_env_extent: Optional = None,
        label: Optional = None,
        ax: Optional = None):
    # TODO: use scenario plotting here
    if ax is None:
        plt.figure()
        ax = plt.gca()

    if actual_env is not None:
        ax.imshow(np.flipud(actual_env), extent=actual_env_extent, zorder=1,cmap='Greys')
    if state is not None:
        plot_rope_configuration(ax, state, c='red', label='state', zorder=2, linewidth=5)
    if next_state is not None:
        plot_rope_configuration(ax, next_state, c='orange', label='next state', zorder=4, linestyle='--', linewidth=5)
    if state is not None and action is not None:
        plot_arrow(ax, state[-2], state[-1], action[0]/2, action[1]/2, color='cyan', linewidth=3)

    if planned_env_origin is not None and res is not None:
        origin_x, origin_y = link_bot_sdf_utils.idx_to_point(0, 0, res, planned_env_origin)
        ax.scatter(origin_x, origin_y, label='origin', marker='*')

    if planned_state is not None:
        plot_rope_configuration(ax, planned_state, c='green', label='planned state', zorder=3)
    if planned_next_state is not None:
        plot_rope_configuration(ax, planned_next_state, c='blue', label='planned next state', zorder=5, linestyle='-.', linewidth=5)
    if state is not None:
        ax.scatter(state[-2], state[-1], c='k')
    if planned_state is not None:
        ax.scatter(planned_state[-2], planned_state[-1], c='k')

    if label is not None and planned_env_extent is not None:
        label_color = 'g' if label else 'r'
        ax.plot(
            [planned_env_extent[0], planned_env_extent[0], planned_env_extent[1], planned_env_extent[1], planned_env_extent[0]],
            [planned_env_extent[2], planned_env_extent[3], planned_env_extent[3], planned_env_extent[2], planned_env_extent[2]],
            c=label_color, linewidth=4)

    ax.axis("equal")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    # ax.legend()


def make_interpretable_image(image: np.ndarray, n_points: int):
    image_23d = image.squeeze()
    assert(image.shape[2] == 23)
    pre_rope = np.sum(image_23d[:, :, 0:n_points], axis=2)
    post_rope = np.sum(image_23d[:, :, n_points:2 * n_points], axis=2)
    local_env = image_23d[:, :, 2 * n_points]
    interpretable_image = np.stack([pre_rope, post_rope, local_env], axis=2)
    return interpretable_image
