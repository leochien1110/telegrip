"""
Kinematics utilities for the SO100 robot.
Contains forward and inverse kinematics solvers using PyBullet.
"""

import math
import numpy as np
import pybullet as p
from typing import Optional, Tuple, List
import logging
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class ForwardKinematics:
    """Forward kinematics solver using PyBullet."""
    
    def __init__(self, physics_client, robot_id: int, joint_indices: list, end_effector_link_index: int, num_joints: int):
        self.physics_client = physics_client
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        self.end_effector_link_index = end_effector_link_index
        self.num_joints = num_joints
    
    def compute(self, joint_angles_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics for given joint angles.
        
        Args:
            joint_angles_deg: Joint angles in degrees
            
        Returns:
            Tuple of (position, quaternion) of end effector
        """
        if self.physics_client is None or self.robot_id is None:
            return np.array([0.2, 0.0, 0.15]), np.array([0, 0, 0, 1])
        
        # Use joint angles
        fk_state_angles = joint_angles_deg.copy()
        
        # NOTE: logic about resetting gripper index 5 might be SO-100 specific.
        # But if num_joints <= 6, index 5 is gripper or last joint.
        # For general purpose, we might not want to hardcode index 5 reset.
        # But for stability, we assume gripper is at the end.
        if len(fk_state_angles) > 5:
             # Assuming last joint is gripper, or specialized handling needed.
             # For now, let's just use the angles as is, assuming FK handles all joints.
             pass

        # Set joint positions
        joint_angles_rad = np.deg2rad(fk_state_angles)
        for i in range(self.num_joints):
            if i < len(self.joint_indices) and self.joint_indices[i] is not None:
                # Ensure we don't go out of bounds of angles array
                if i < len(joint_angles_rad):
                    p.resetJointState(self.robot_id, self.joint_indices[i], joint_angles_rad[i])
        
        # Get end effector position and orientation
        link_state = p.getLinkState(self.robot_id, self.end_effector_link_index)
        position = np.array(link_state[0])
        quaternion = np.array(link_state[1])
        
        return position, quaternion


class IKSolver:
    """Inverse kinematics solver using PyBullet with multiple reference poses."""
    
    def __init__(self, physics_client, robot_id: int, joint_indices: list, 
                 end_effector_link_index: int, joint_limits_min_deg: np.ndarray, 
                 joint_limits_max_deg: np.ndarray, num_joints: int, num_ik_joints: int,
                 use_reference_poses: bool = False, reference_poses_file: str = "",
                 ik_position_error_threshold: float = 0.001,
                 ik_hysteresis_threshold: float = 0.05,
                 ik_movement_penalty_weight: float = 0.01,
                 arm_name: str = ""):
        self.physics_client = physics_client
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        self.end_effector_link_index = end_effector_link_index
        self.joint_limits_min_deg = joint_limits_min_deg
        self.joint_limits_max_deg = joint_limits_max_deg
        self.num_joints = num_joints
        self.num_ik_joints = num_ik_joints
        self.use_reference_poses = use_reference_poses
        self.reference_poses_file = reference_poses_file
        self.ik_position_error_threshold = ik_position_error_threshold
        self.ik_hysteresis_threshold = ik_hysteresis_threshold
        self.ik_movement_penalty_weight = ik_movement_penalty_weight
        self.arm_name = arm_name
        
        # Precompute IK limits for first num_ik_joints
        self.ik_lower_limits = np.deg2rad(joint_limits_min_deg[:self.num_ik_joints])
        self.ik_upper_limits = np.deg2rad(joint_limits_max_deg[:self.num_ik_joints])
        self.ik_ranges = self.ik_upper_limits - self.ik_lower_limits
        
        # Load reference poses
        self.reference_poses = self._load_reference_poses()
        
        # Create FK solver for evaluating solutions
        self.fk_solver = ForwardKinematics(physics_client, robot_id, joint_indices, end_effector_link_index, num_joints)
    
    def _load_reference_poses(self) -> List[np.ndarray]:
        """Load reference poses from file for this arm."""
        reference_poses = []
        
        # Check if reference poses are enabled
        if not self.use_reference_poses:
            # Only log if explicitly disabled via config, to avoid spam if just not set
            # logger.info("Reference poses disabled in configuration")
            return reference_poses
        
        try:
            from ..utils import get_absolute_path
            # Handle empty file path
            if not self.reference_poses_file:
                return reference_poses
                
            cache_file = get_absolute_path(self.reference_poses_file)
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                arm_poses = data.get(self.arm_name, [])
                if arm_poses:
                    # Convert to numpy arrays and extract only the first num_ik_joints joints for IK
                    for pose in arm_poses:
                        pose_array = np.array(pose[:self.num_ik_joints])
                        pose_rad = np.deg2rad(pose_array)
                        reference_poses.append(pose_rad)
                    
                    logger.info(f"Loaded {len(reference_poses)} reference poses for {self.arm_name} arm")
                else:
                    logger.info(f"No reference poses found for {self.arm_name} arm")
            else:
                logger.info("No reference poses file found. Use read_pose.py to record reference poses.")
                
        except Exception as e:
            logger.warning(f"Failed to load reference poses: {e}")
        
        return reference_poses
    
    def _evaluate_ik_solution(self, solution: np.ndarray, target_position: np.ndarray, 
                             current_joints_rad: Optional[np.ndarray] = None, 
                             hysteresis_threshold: float = 0.05) -> float:
        """
        Evaluate the quality of an IK solution based on position error and joint movement.
        """
        try:
            # Convert solution to full joint array (keep other joints at 0)
            full_angles = np.zeros(self.num_joints)
            full_angles[:self.num_ik_joints] = np.rad2deg(solution)
            
            # Compute forward kinematics
            achieved_position, _ = self.fk_solver.compute(full_angles)
            
            # Calculate position error
            position_error = np.linalg.norm(achieved_position - target_position)
            
            # Add joint movement penalty if current joints provided
            movement_penalty = 0.0
            if current_joints_rad is not None:
                # Calculate joint space distance (only for IK joints)
                joint_diff = solution - current_joints_rad[:self.num_ik_joints]
                joint_movement = np.linalg.norm(joint_diff)
                
                # Convert joint movement to a position-equivalent penalty
                movement_penalty = joint_movement * self.ik_movement_penalty_weight
                
            # Total cost combines position error and movement penalty
            total_cost = position_error + movement_penalty
            return total_cost
            
        except Exception as e:
            logger.warning(f"Error evaluating IK solution: {e}")
            return float('inf')
    
    def solve(self, target_position: np.ndarray, target_orientation_quat: Optional[np.ndarray], 
              current_angles_deg: np.ndarray) -> np.ndarray:
        """
        Solve inverse kinematics for position control.
        """
        if self.physics_client is None or self.robot_id is None:
            return current_angles_deg[:self.num_ik_joints]
        
        # Get current actual robot position and error
        current_actual_position, _ = self.fk_solver.compute(current_angles_deg)
        current_actual_error = np.linalg.norm(current_actual_position - target_position)
        
        # Convert current angles to radians and prepare for IK state
        ik_state_angles = current_angles_deg.copy()
        # For SO-100, we might reset gripper. For general, we might not.
        # But if we rely on stored rest poses, they might assume neutral gripper.
        # Let's keep it simple and just use current state.
        
        current_angles_rad = np.deg2rad(ik_state_angles)
        
        # Helper functions for state management
        def set_robot_to_current_state():
            """Helper to set robot to exact current state"""
            for i in range(self.num_joints):
                if i < len(self.joint_indices) and self.joint_indices[i] is not None:
                    # Ensure we have enough angles
                    if i < len(current_angles_rad):
                        p.resetJointState(self.robot_id, self.joint_indices[i], current_angles_rad[i])
        
        def set_robot_to_reference_state(ref_pose_rad: np.ndarray):
            """Helper to set robot to reference pose state"""
            full_ref_state = current_angles_rad.copy()
            # Only update IK joints
            full_ref_state[:self.num_ik_joints] = ref_pose_rad
            for i in range(self.num_joints):
                if i < len(self.joint_indices) and self.joint_indices[i] is not None:
                    if i < len(full_ref_state):
                       p.resetJointState(self.robot_id, self.joint_indices[i], full_ref_state[i])
        
        # Prepare list of rest poses to try
        rest_poses_to_try = []
        
        # 1. Current configuration (most likely to be close to solution)
        current_rest_pose = np.deg2rad(current_angles_deg[:self.num_ik_joints])
        rest_poses_to_try.append(('current', current_rest_pose))
        
        # 2. Reference poses from recorded configurations
        for i, ref_pose in enumerate(self.reference_poses):
            rest_poses_to_try.append((f'reference_{i+1}', ref_pose))
        
        best_solution = None
        best_error = float('inf')
        best_source = None
        current_solution_error = None
        current_solution_joints = None
        
        # Track best reference pose separately from overall best
        best_reference_solution = None
        best_reference_error = float('inf')
        best_reference_source = None
        best_reference_position_error = float('inf')  # Pure position error without movement penalty
        
        # Try each rest pose configuration
        for source_name, rest_pose in rest_poses_to_try:
            try:
                # Seed the IK from the current simulation state (last solution)
                # This makes movement much more stable and continuous
                # if source_name == 'current':
                #     set_robot_to_current_state()
                # else:
                #     set_robot_to_reference_state(rest_pose)
                
                # Perform IK
                ik_params = {
                    "bodyUniqueId": self.robot_id,
                    "endEffectorLinkIndex": self.end_effector_link_index,
                    "targetPosition": target_position.tolist(),
                    "lowerLimits": self.ik_lower_limits.tolist(),
                    "upperLimits": self.ik_upper_limits.tolist(),
                    "jointRanges": self.ik_ranges.tolist(),
                    "solver": 0,                                # 0 = DLS
                    "maxNumIterations": 1000,
                    "residualThreshold": 1e-5
                }
                
                if target_orientation_quat is not None:
                    ik_params["targetOrientation"] = target_orientation_quat.tolist()
                
                ik_solution = p.calculateInverseKinematics(**ik_params)
                
                # Restore state
                set_robot_to_current_state()
                
                # Evaluate this solution
                solution_array = np.array(ik_solution[:self.num_ik_joints])
                
                # Handle joint limits wrapping and clamping
                joint_limits_min_deg = np.rad2deg(self.ik_lower_limits)
                joint_limits_max_deg = np.rad2deg(self.ik_upper_limits)
                solution_degrees = np.rad2deg(solution_array)
                
                # Check and wrap shoulder_pan (first joint) if outside limits
                if solution_degrees[0] < joint_limits_min_deg[0] or solution_degrees[0] > joint_limits_max_deg[0]:
                    for offset in [-360.0, 360.0]:
                        wrapped_angle = solution_degrees[0] + offset
                        if joint_limits_min_deg[0] <= wrapped_angle <= joint_limits_max_deg[0]:
                            solution_degrees[0] = wrapped_angle
                            break
                    else:
                        clamped_angle = np.clip(solution_degrees[0], joint_limits_min_deg[0], joint_limits_max_deg[0])
                        solution_degrees[0] = clamped_angle
                
                # Clamp other joints normally
                if len(solution_degrees) > 1:
                    solution_degrees[1:] = np.clip(solution_degrees[1:], joint_limits_min_deg[1:], joint_limits_max_deg[1:])
                
                solution_array = np.deg2rad(solution_degrees)
                
                if source_name == 'current':
                    error = self._evaluate_ik_solution(solution_array, target_position, None)
                    current_solution_error = error
                    current_solution_joints = solution_array.copy()
                else:
                    position_only_error = self._evaluate_ik_solution(solution_array, target_position, None)
                    error = self._evaluate_ik_solution(solution_array, target_position, current_angles_rad)
                    
                    if error < best_reference_error:
                        best_reference_error = error
                        best_reference_solution = solution_array
                        best_reference_source = source_name
                        best_reference_position_error = position_only_error
                
                if error < best_error:
                    best_error = error
                    best_solution = solution_array
                    best_source = source_name
                
                if source_name == 'current' and error < 0.0001:
                    break
                    
            except Exception as e:
                logger.debug(f"IK failed with {source_name} rest pose: {e}")
                set_robot_to_current_state()
                continue
        
        set_robot_to_current_state()
        
        final_solution = current_solution_joints
        final_error = current_solution_error
        final_source = 'current'
        
        if (best_reference_source is not None and current_actual_error is not None):
            position_improvement = current_actual_error - best_reference_position_error
            if position_improvement > self.ik_hysteresis_threshold:
                logger.info(f"IK: Using {best_reference_source} (significant position improvement: {position_improvement:.4f}m > {self.ik_hysteresis_threshold}m)")
                final_solution = best_reference_solution
                final_error = best_reference_error
                final_source = best_reference_source
        
        if final_solution is not None:
            final_angles = np.rad2deg(final_solution)
            return final_angles
        else:
            logger.warning("All IK attempts failed, returning current angles")
            return current_angles_deg[:self.num_ik_joints]


def vr_to_robot_coordinates(vr_pos: dict, scale: float = 1.0) -> np.ndarray:
    """
    Convert VR controller position to robot coordinate system.
    
    VR coordinate system: X=right, Y=up, Z=back (towards user)
    Robot coordinate system: X=forward, Y=left, Z=up
    """
    return np.array([
        -vr_pos['x'] * scale,   # VR +Z (back) -> Robot +X (forward)
        vr_pos['z'] * scale,    # VR +X (right) -> Robot -Y (right) 
        vr_pos['y'] * scale     # VR +Y (up) -> Robot +Z (up)
    ])


def compute_relative_position(current_vr_pos: dict, origin_vr_pos: dict, scale: float = 1.0) -> np.ndarray:
    """Compute relative position from VR origin to current position."""
    delta_vr = {
        'x': current_vr_pos['x'] - origin_vr_pos['x'],
        'y': current_vr_pos['y'] - origin_vr_pos['y'], 
        'z': current_vr_pos['z'] - origin_vr_pos['z']
    }
    return vr_to_robot_coordinates(delta_vr, scale)