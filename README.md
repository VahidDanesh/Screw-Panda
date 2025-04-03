# Screw-Panda
[![Powered by the Robotics Toolbox](https://raw.githubusercontent.com/petercorke/robotics-toolbox-python/master/.github/svg/rtb_powered.min.svg)](https://github.com/petercorke/robotics-toolbox-python)
[![Powered by Python Robotics](https://raw.githubusercontent.com/petercorke/robotics-toolbox-python/master/.github/svg/pr_powered.min.svg)](https://github.com/petercorke/robotics-toolbox-python)

A framework for robotic manipulation planning using dual quaternions and screw theory for primitive motions.

## Overview

Screw-Panda provides tools for efficient motion planning of robot manipulator arms using:

- Dual quaternion representation for object poses
- Screw theory for primitive motions
- Efficient trajectory generation and interpolation
- Integration with popular robotics packages

## Features

- **Dual Quaternion Representation**: Efficient pose representation using dual quaternions
- **Screw Motion Planning**: Generate trajectories using screw linear interpolation
- **Primitive Motion Primitives**: Sliding, pivoting, rolling, and grasping
- **Contact Planning**: Plan manipulation sequences with contact constraints
- **Robot Control**: Resolved rate control with null-space optimization

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Screw-Panda.git
cd Screw-Panda

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- numpy
- spatialmath
- roboticstoolbox-python
- pytransform3d
- matplotlib
- swift

## Getting Started

```python
from spatialmath import SE3
from Screw-Panda import Screw, DualQuaternionUtils

# Define start and end poses
start_pose = SE3(0.5, 0.0, 0.1) * SE3.RPY(0, 0, 0)
end_pose = SE3(0.5, 0.4, 0.3) * SE3.RPY(np.pi/2, 0, np.pi/4)

# Create a screw motion between poses
screw = Screw.from_se3(start_pose, end_pose)

# Generate trajectory
poses, twists = Screw.generate_trajectory(start_pose, end_pose, num_steps=20)
```

## Demonstrations

Run the demonstration script to see the framework in action:

```bash
python screw_motion_demo.py
```

## Documentation

For more detailed documentation, see the docstrings in each module.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
