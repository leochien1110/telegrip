# UR10e with Robotiq 2F-85 Gripper

This folder contains a standalone description of the UR10e robot equipped with a Robotiq 2F-85 gripper.

## Structure

- **urdf/**: Contains the combined URDF file `ur10e_2f85.urdf`.
- **meshes/**: Contains all visual and collision meshes for both the robot and the gripper.
    - `ur10e/`: Meshes for the UR10e arm.
    - `robotiq_2f85/`: Meshes for the Robotiq gripper.
- **assemble_urdf.py**: Python script used to generate the combined URDF from source files.

## Usage

To use this robot description, load `urdf/ur10e_2f85.urdf`. The mesh paths are relative to the URDF file, so it should load correctly in most visualization tools (RViz, PyBullet, etc.) without additional package path configuration.
