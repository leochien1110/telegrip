# URDF Models

This directory contains URDF models for SO-ARM robot manipulators.

## SO100

This is the 6DOF SO100 arm. The original assets are from [here](https://github.com/TheRobotStudio/SO-ARM100) and were provided under a [Apache 2.0 License](LICENSE)

Changes made:
- Fixed joint limits to reflect real world behavior
- Fixed joint tags from continuous to revolute which permit joint limits
- Fixed joint directions and orientations to match the real robot's joints
- Removed spaces in link names
- Manual decomposition of gripper link collision meshes into simpler meshes

## SO101

This is the 6DOF SO101 arm, an improved version with updated mechanical design.

The URDF and mesh files are derived from the calibrated SO-101 description package.

Features:
- Calibrated joint parameters and inertial properties
- Updated mesh files for all links
- Compatible joint structure with SO100 for easy migration