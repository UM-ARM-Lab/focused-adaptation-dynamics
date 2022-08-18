import numpy as np

from arm_gazebo_msgs.srv import ComputeOccupancyResponse


class SoftGymServices():

    def __init__(self):
        super().__init__()
        self._scene = None

    def setup_env(self, *args, **kwargs):
        pass

    def set_scene(self, scene):
        self._scene = scene

    def __call__(self, args, **kwargs):
        self.compute_occupancy(*args, **kwargs)

    def get_world_initial_sdf(self):
        #Needs to be implemented but not used. Ideal to roswarn once
        return None

    def is_occupied(self, x, y, z, env_indices):
        coords = env_indices[x, y, z]
        # only checks if occupied by the box!!
        fixed_params = self._scene._wrapped_env.glass_params
        poured_height = fixed_params["poured_height"]
        poured_x = fixed_params["poured_glass_dis_x"]
        poured_y = fixed_params["poured_height"]
        poured_z = fixed_params["poured_glass_dis_z"]
        poured_x_dims = [fixed_params["poured_glass_x_center"] - poured_x / 2,
                         fixed_params["poured_glass_x_center"] + poured_x / 2]
        poured_z_dims = [0, poured_z / 2]
        poured_y_dims = [-poured_y / 2, poured_y / 2]
        dim_ranges = [poured_x_dims, poured_y_dims, poured_z_dims]
        in_ranges = np.array([False, False, False])
        for dim_idx, dim_range in enumerate(dim_ranges):
            if coords[dim_idx] >= dim_range[0] and coords[dim_idx] <= dim_range[1]:
                in_ranges[dim_idx] = True
        if np.all(in_ranges):
            return True

        return False

    def pause(self):
        pass

    def play(self):
        pass

    def create_env_coords(self, x_dims, y_dims, z_dims, center, res):
        # assuming cube for now
        x_extent = x_dims * res
        y_extent = y_dims * res
        z_extent = z_dims * res
        env_coords = np.zeros((x_dims, y_dims, z_dims, 3))
        xs = np.linspace(center[0] - x_extent / 2, center[0] + x_extent / 2, x_dims)
        ys = np.linspace(center[1] - y_extent / 2, center[1] + y_extent / 2, y_dims)
        zs = np.linspace(center[2] - z_extent / 2, center[2] + z_extent / 2, z_dims)
        for x in range(x_dims):
            for y in range(y_dims):
                for z in range(z_dims):
                    env_coords[x, y, z, :] = np.array([xs[x], ys[y], zs[z]])
        # idx_to_point_3d_in_env or idx_to_point_3d
        return env_coords

    def compute_occupancy(self, req):
        assert self._scene is not None
        center = np.array([req.center.x, req.center.y, req.center.z])
        res = req.resolution
        # make grid
        x_dims = req.w_cols
        y_dims = req.h_rows
        z_dims = req.c_channels
        grid = np.zeros((x_dims, y_dims, z_dims))
        env_coords = self.create_env_coords(x_dims, y_dims, z_dims, center, res)
        response = ComputeOccupancyResponse()
        for x in range(x_dims):
            for y in range(y_dims):
                for z in range(z_dims):
                    if self.is_occupied(x, y, z, env_coords):
                        grid[x, y, z] = 1
        response.grid = grid
        return response
