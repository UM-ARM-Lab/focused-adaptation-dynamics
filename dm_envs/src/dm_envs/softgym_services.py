import numpy as np
try:
    import fcl
except ImportError:
    print("FCL not there. softgym stack will not work")

from arm_gazebo_msgs.srv import ComputeOccupancyResponse


class SoftGymServices():

    def __init__(self):
        super().__init__()
        self._scene = None
        self.cached_grid = None

    def setup_env(self, *args, **kwargs):
        pass

    def set_scene(self, scene):
        self._scene = scene
        self.cached_grid = np.load("/home/lagrassa/catkin_ws/src/link_bot/link_bot_data/cached_grid.npy")


    def __call__(self, args, **kwargs):
        self.compute_occupancy(*args, **kwargs)

    def get_world_initial_sdf(self):
        #Needs to be implemented but not used. Ideal to roswarn once
        return None

    def is_occupied(self, x, y, z, env_indices, res, sphere):
        coords = env_indices[x, y, z]
        obj_idxs = ["plant", "poured"]
        radius = res/2.
        return self._scene.in_collision_sphere_premade(coords, sphere, obj_idxs=obj_idxs)

    def pause(self):
        pass

    def play(self):
        pass

    def create_env_coords(self, x_dims, y_dims, z_dims, center, res):
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
        response = ComputeOccupancyResponse()
        if self.cached_grid is not None:
            assert grid.shape == self.cached_grid.shape
            response.grid = self.cached_grid
        else:
            env_coords = self.create_env_coords(x_dims, y_dims, z_dims, center, res)
            sphere_shape = fcl.Sphere(res/2.)
            tf = fcl.Transform(np.array([0,0,0]))
            sphere = fcl.CollisionObject(sphere_shape, tf)
            for x in range(x_dims):
                print(f"{x} out of {x_dims}")
                for y in range(y_dims):
                    for z in range(z_dims):
                        if self.is_occupied(x, y, z, env_coords, res, sphere):
                            grid[x, y, z] = 1
        response.grid = grid
        return response
