# FOCUS

This repository contains a snapshot of the code used for the paper "Focused Adaptation of Dynamcs Models for Deformable Object Manipulation".

The loss function shown in Equation (1) in the paper is implemented [here](https://github.com/UM-ARM-Lab/focused-adaptation-dynamics/blob/master/state_space_dynamics/src/state_space_dynamics/torch_udnn.py#L18). The hyperparameters used for experiments are in the folder `state_space_dynamics/hparams`. For example, the one used for simulated rope manipulation is [here](https://github.com/UM-ARM-Lab/focused-adaptation-dynamics/blob/master/state_space_dynamics/hparams/iterative_lowest_error_soft_online.hjson). The main scripts to run the full online learning method are in `link_bot_planning/scripts`:

- [online_rope_real.py](https://github.com/UM-ARM-Lab/focused-adaptation-dynamics/blob/master/link_bot_planning/scripts/online_rope_real.py)
- [online_rope_sim.py](https://github.com/UM-ARM-Lab/focused-adaptation-dynamics/blob/master/link_bot_planning/scripts/online_rope_sim.py)
- [online_water_sim.py](https://github.com/UM-ARM-Lab/focused-adaptation-dynamics/blob/master/link_bot_planning/scripts/online_water_sim.py)

For rope manipulation, we use Gazebo. For water simulation, we use PyFlex.

## Setup

The repository should be used as a reference impementation, as running the code requires a lot of setup, and extensive documentation is not provided. To start, clone this repo, as well as all the other UM-ARM-Lab repos you will need, such as `sdf_tools`, `arm_robots`, `arm_gazebo`, and possibly others.

```
# in the catkin workspace, inside the `src` folder
virtualenv --system-site-packages venv
source venv/bin/activate  # you will need to source the virtual environment whenever you want to run anything.
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
rosdep update
rosdep install -y -r --from-paths . --ignore-src
cd ..
catkin config -DOMPL_BUILD_PYBINDINGS=ON -DCATKIN_ENABLE_TESTING=OFF -DCMAKE_BUILD_TYPE=Release --merge-devel
catkin build
```
