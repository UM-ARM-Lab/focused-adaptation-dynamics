# Setup

clone this repo as, well as all the other repos you will need, such as `sdf_tools`, `arm_robots`, `arm_gazebo`, etc...

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
