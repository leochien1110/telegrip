"""
Robot interface module for the SO100 teleoperation system.
Provides a clean wrapper around robot devices with safety checks and convenience methods.
"""

import numpy as np
import torch
import time
import logging
import os
import sys
import contextlib
from typing import Optional, Dict, Tuple

# New lerobot structure imports
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower

from ..config import (
    TelegripConfig, ROBOT_CONFIGS
)
from .kinematics import ForwardKinematics, IKSolver

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr output at the file descriptor level."""
    # Save original file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    
    # Save original file descriptors
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    
    try:
        # Open devnull
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        
        # Redirect stdout and stderr to devnull
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        
        yield
        
    finally:
        # Restore original file descriptors
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        
        # Close saved file descriptors
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


class RobotInterface:
    """High-level interface for SO100 robot control with safety features."""
    
    def __init__(self, config: TelegripConfig):
        self.config = config
        self.left_robot = None
        self.right_robot = None
        self.is_connected = False
        self.is_engaged = False  # New state for motor engagement
        
        # Load robot-specific configuration
        self.robot_type = self.config.robot_type
        if self.robot_type not in ROBOT_CONFIGS:
            logger.warning(f"Unknown robot type {self.robot_type}, falling back to so100")
            self.robot_config = ROBOT_CONFIGS["so100"]
        else:
            self.robot_config = ROBOT_CONFIGS[self.robot_type]
            
        self.joint_names = self.robot_config["joint_names"]
        self.num_joints = len(self.joint_names)
        self.num_ik_joints = self.robot_config["num_ik_joints"]
        
        # Individual arm connection status
        self.left_arm_connected = False
        self.right_arm_connected = False
        
        # Joint state
        self.left_arm_angles = np.zeros(self.num_joints)
        self.right_arm_angles = np.zeros(self.num_joints)
        
        # Joint limits (will be set by visualizer)
        self.joint_limits_min_deg = np.full(self.num_joints, -180.0)
        self.joint_limits_max_deg = np.full(self.num_joints, 180.0)
        
        # Kinematics solvers (will be set after PyBullet setup)
        self.fk_solvers = {'left': None, 'right': None}
        self.ik_solvers = {'left': None, 'right': None}
        
        # Control timing
        self.last_send_time = 0
        
        # Error tracking - separate for each arm
        self.left_arm_errors = 0
        self.right_arm_errors = 0
        self.general_errors = 0
        self.max_arm_errors = 3  # Allow fewer errors per arm before marking as disconnected
        self.max_general_errors = 8  # Allow more general errors before full disconnection
        
        # Initial positions for safe shutdown
        # TODO: Make this dynamic based on robot type/URDF
        # For now, using logic similar to old implementation but resized
        self.initial_left_arm = np.zeros(self.num_joints)
        self.initial_right_arm = np.zeros(self.num_joints)
        
        # Set default SO-100 initial pose if applicable
        if self.robot_type == "so100" and self.num_joints >= 6:
            # np.array([0, -100, 100, 60, 0, 0])
            self.initial_left_arm[:6] = [0, -100, 100, 60, 0, 0]
            self.initial_right_arm[:6] = [0, -100, 100, 60, 0, 0]
    
    def setup_robot_configs(self) -> Tuple[SO100FollowerConfig, SO100FollowerConfig]:
        """Create robot configurations for both arms."""
        logger.info(f"Setting up robot configs with ports: {self.config.follower_ports}")
        
        left_config = SO100FollowerConfig(
            port=self.config.follower_ports["left"],
            use_degrees=True,  # Use degrees for easier debugging
            disable_torque_on_disconnect=True
        )
        # Set the robot name for calibration file lookup
        left_config.id = "left_follower"
        
        right_config = SO100FollowerConfig(
            port=self.config.follower_ports["right"],
            use_degrees=True,  # Use degrees for easier debugging
            disable_torque_on_disconnect=True
        )
        # Set the robot name for calibration file lookup
        right_config.id = "right_follower"
        
        return left_config, right_config
    
    def connect(self) -> bool:
        """Connect to robot hardware."""
        if self.is_connected:
            logger.info("Robot interface already connected")
            return True
        
        if not self.config.enable_robot:
            logger.info("Robot interface disabled in config")
            self.is_connected = True  # Mark as "connected" for testing
            return True
        
        # Setup suppression if requested
        should_suppress = (self.config.log_level == "warning" or 
                          self.config.log_level == "critical" or 
                          self.config.log_level == "error")
        
        try:
            left_config, right_config = self.setup_robot_configs()
            if not should_suppress:
                logger.info("Connecting to robot...")
            
            # Connect left arm
            if self.config.is_arm_enabled("left"):
                try:
                    if should_suppress:
                        with suppress_stdout_stderr():
                            self.left_robot = SO100Follower(left_config)
                            self.left_robot.connect()
                    else:
                        self.left_robot = SO100Follower(left_config)
                        self.left_robot.connect()
                    self.left_arm_connected = True
                    logger.info("✅ Left arm connected successfully")
                except Exception as e:
                    logger.error(f"❌ Left arm connection failed: {e}")
                    self.left_arm_connected = False
            else:
                logger.info("⏭️  Left arm disabled (not connecting)")
                self.left_arm_connected = False
            
            # Connect right arm  
            if self.config.is_arm_enabled("right"):
                try:
                    if should_suppress:
                        with suppress_stdout_stderr():
                            self.right_robot = SO100Follower(right_config)
                            self.right_robot.connect()
                    else:
                        self.right_robot = SO100Follower(right_config)
                        self.right_robot.connect()
                    self.right_arm_connected = True
                    logger.info("✅ Right arm connected successfully")
                except Exception as e:
                    logger.error(f"❌ Right arm connection failed: {e}")
                    self.right_arm_connected = False
            else:
                logger.info("⏭️  Right arm disabled (not connecting)")
                self.right_arm_connected = False
                
            # Mark as connected if at least one arm is connected
            self.is_connected = self.left_arm_connected or self.right_arm_connected
            
            if self.is_connected:
                # Initialize joint states
                self._read_initial_state()
                logger.info(f"🤖 Robot interface connected: Left={self.left_arm_connected}, Right={self.right_arm_connected}")
            else:
                logger.error("❌ Failed to connect any robot arms")
                
            return self.is_connected
            
        except Exception as e:
            logger.error(f"❌ Robot connection failed with exception: {e}")
            self.is_connected = False
            return False
    
    def _read_initial_state(self):
        """Read initial joint state from robot."""
        try:
            if self.left_robot and self.left_arm_connected:
                observation = self.left_robot.get_observation()
                if observation:
                    # Extract joint positions from observation using dynamic names
                    angles = []
                    for name in self.joint_names:
                        key = f"{name}.pos"
                        if key in observation:
                            angles.append(observation[key])
                        else:
                            angles.append(0.0) # Fallback
                    self.left_arm_angles = np.array(angles)
                    logger.info(f"Left arm initial state: {self.left_arm_angles.round(1)}")
                    
            if self.right_robot and self.right_arm_connected:
                observation = self.right_robot.get_observation()
                if observation:
                    angles = []
                    for name in self.joint_names:
                        key = f"{name}.pos"
                        if key in observation:
                            angles.append(observation[key])
                        else:
                            angles.append(0.0)
                    self.right_arm_angles = np.array(angles)
                    logger.info(f"Right arm initial state: {self.right_arm_angles.round(1)}")
                    
        except Exception as e:
            logger.error(f"Error reading initial state: {e}")
    
    def setup_kinematics(self, physics_client, robot_ids: Dict, joint_indices: Dict, 
                        end_effector_link_indices: Dict, joint_limits_min_deg: np.ndarray, 
                        joint_limits_max_deg: np.ndarray):
        """Setup kinematics solvers using PyBullet components for both arms."""
        
        # Ensure limits match current num_joints
        if len(joint_limits_min_deg) == self.num_joints:
             self.joint_limits_min_deg = joint_limits_min_deg.copy()
             self.joint_limits_max_deg = joint_limits_max_deg.copy()
        
        # Setup solvers for both arms
        for arm in ['left', 'right']:
            self.fk_solvers[arm] = ForwardKinematics(
                physics_client, robot_ids[arm], joint_indices[arm], end_effector_link_indices[arm],
                self.num_joints
            )
            
            self.ik_solvers[arm] = IKSolver(
                physics_client, robot_ids[arm], joint_indices[arm], end_effector_link_indices[arm],
                self.joint_limits_min_deg, self.joint_limits_max_deg, 
                self.num_joints, self.num_ik_joints,
                use_reference_poses=self.config.use_reference_poses,
                reference_poses_file=self.config.reference_poses_file,
                ik_position_error_threshold=self.config.ik_position_error_threshold,
                ik_hysteresis_threshold=self.config.ik_hysteresis_threshold,
                ik_movement_penalty_weight=self.config.ik_movement_penalty_weight,
                arm_name=arm
            )
        
        logger.info("Kinematics solvers initialized for both arms")
    
    def get_current_end_effector_position(self, arm: str) -> np.ndarray:
        """Get current end effector position for specified arm."""
        if arm == "left":
            angles = self.left_arm_angles
        elif arm == "right":
            angles = self.right_arm_angles
        else:
            raise ValueError(f"Invalid arm: {arm}")
        
        if self.fk_solvers[arm]:
            position, _ = self.fk_solvers[arm].compute(angles)
            return position
        else:
            default_position = np.array([0.2, 0.0, 0.15])
            return default_position
    
    def solve_ik(self, arm: str, target_position: np.ndarray, 
                 target_orientation: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve inverse kinematics for specified arm."""
        if arm == "left":
            current_angles = self.left_arm_angles
        elif arm == "right":
            current_angles = self.right_arm_angles
        else:
            raise ValueError(f"Invalid arm: {arm}")
        
        if self.ik_solvers[arm]:
            return self.ik_solvers[arm].solve(target_position, target_orientation, current_angles)
        else:
            return current_angles[:self.num_ik_joints]  # Return current angles if no IK solver
    
    def clamp_joint_angles(self, joint_angles: np.ndarray) -> np.ndarray:
        """Clamp joint angles to safe limits with margins for problem joints."""
        # Create a copy to avoid modifying the original
        processed_angles = joint_angles.copy()
        
        # First, normalize angles that can wrap around (like shoulder_pan)
        # Check if first joint (shoulder_pan) is outside limits but can be wrapped
        shoulder_pan_idx = 0
        shoulder_pan_angle = processed_angles[shoulder_pan_idx]
        min_limit = self.joint_limits_min_deg[shoulder_pan_idx]  # -120.3° (example)
        max_limit = self.joint_limits_max_deg[shoulder_pan_idx]  # +120.3°
        
        # Try to wrap the angle to an equivalent angle within limits
        if shoulder_pan_angle < min_limit or shoulder_pan_angle > max_limit:
            # Try wrapping by ±360°
            for offset in [-360.0, 360.0]:
                wrapped_angle = shoulder_pan_angle + offset
                if min_limit <= wrapped_angle <= max_limit:
                    logger.debug(f"Wrapped shoulder_pan from {shoulder_pan_angle:.1f}° to {wrapped_angle:.1f}°")
                    processed_angles[shoulder_pan_idx] = wrapped_angle
                    break
        
        # Apply standard joint limits to all joints
        return np.clip(processed_angles, self.joint_limits_min_deg, self.joint_limits_max_deg)
    
    def update_arm_angles(self, arm: str, ik_angles: np.ndarray, wrist_flex: float, wrist_roll: float, gripper: float):
        """Update joint angles for specified arm with IK solution and direct wrist/gripper control.
           If IK covers wrists, wrist params are ignored.
        """
        if arm == "left":
            target_angles = self.left_arm_angles
        elif arm == "right":
            target_angles = self.right_arm_angles
        else:
            raise ValueError(f"Invalid arm: {arm}")
        
        # Update joints handled by IK
        target_angles[:self.num_ik_joints] = ik_angles
        
        # Helper to get index
        def get_idx(key):
            return self.robot_config.get(key)

        # Handle Wrist Flex if NOT in IK
        wf_idx = get_idx("wrist_flex_index")
        if wf_idx is not None and wf_idx >= self.num_ik_joints:
             target_angles[wf_idx] = wrist_flex

        # Handle Wrist Roll if NOT in IK
        wr_idx = get_idx("wrist_roll_index")
        if wr_idx is not None and wr_idx >= self.num_ik_joints:
             target_angles[wr_idx] = wrist_roll

        # Handle Gripper if NOT in IK (usually not)
        g_idx = get_idx("gripper_index")
        if g_idx is not None:
             target_angles[g_idx] = np.clip(gripper, self.config.gripper_open_angle, self.config.gripper_closed_angle)
        
        # Apply joint limits to all joints
        clamped_angles = self.clamp_joint_angles(target_angles)
        
        # Preserve gripper control (don't clamp gripper if it was set explicitly here and might slightly exceed limit if noise)
        # Actually clamping above is good.
        
        if arm == "left":
            self.left_arm_angles = clamped_angles
        else:
            self.right_arm_angles = clamped_angles
    
    def engage(self) -> bool:
        """Engage robot motors (start sending commands)."""
        if not self.is_connected:
            logger.warning("Cannot engage robot: not connected")
            return False
        
        self.is_engaged = True
        logger.info("🔌 Robot motors ENGAGED - commands will be sent")
        return True
    
    def disengage(self) -> bool:
        """Disengage robot motors (stop sending commands)."""
        if not self.is_connected:
            logger.info("Robot already disconnected")
            return True
        
        try:
            # Return to safe position before disengaging
            self.return_to_initial_position()
            
            # Disable torque
            self.disable_torque()
            
            self.is_engaged = False
            logger.info("🔌 Robot motors DISENGAGED - commands stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error disengaging robot: {e}")
            return False
    
    def send_command(self) -> bool:
        """Send current joint angles to robot using dictionary format."""
        if not self.is_connected or not self.is_engaged:
            return False
        
        current_time = time.time()
        if current_time - self.last_send_time < self.config.send_interval:
            return True  # Don't send too frequently
        
        try:
            # Send commands with dictionary format
            success = True
            
            def construct_action_dict(angles):
                action = {}
                for i, name in enumerate(self.joint_names):
                    if i < len(angles):
                        action[f"{name}.pos"] = float(angles[i])
                return action

            # Send left arm command
            if self.left_robot and self.left_arm_connected:
                try:
                    action_dict = construct_action_dict(self.left_arm_angles)
                    self.left_robot.send_action(action_dict)
                except Exception as e:
                    logger.error(f"Error sending left arm command: {e}")
                    self.left_arm_errors += 1
                    if self.left_arm_errors > self.max_arm_errors:
                        self.left_arm_connected = False
                        logger.error("❌ Left arm disconnected due to repeated errors")
                    success = False
            
            # Send right arm command
            if self.right_robot and self.right_arm_connected:
                try:
                    action_dict = construct_action_dict(self.right_arm_angles)
                    self.right_robot.send_action(action_dict)
                except Exception as e:
                    logger.error(f"Error sending right arm command: {e}")
                    self.right_arm_errors += 1
                    if self.right_arm_errors > self.max_arm_errors:
                        self.right_arm_connected = False
                        logger.error("❌ Right arm disconnected due to repeated errors")
                    success = False
            
            self.last_send_time = current_time
            return success
            
        except Exception as e:
            logger.error(f"Error sending robot command: {e}")
            self.general_errors += 1
            if self.general_errors > self.max_general_errors:
                self.is_connected = False
                logger.error("❌ Robot interface disconnected due to repeated errors")
            return False
    
    def set_gripper(self, arm: str, closed: bool):
        """Set gripper state for specified arm."""
        angle = self.config.gripper_closed_angle if closed else self.config.gripper_open_angle
        
        # Get gripper index
        g_idx = self.robot_config.get("gripper_index")
        
        if g_idx is None:
             return # No gripper

        if arm == "left":
            self.left_arm_angles[g_idx] = angle
        elif arm == "right":
            self.right_arm_angles[g_idx] = angle
        else:
            raise ValueError(f"Invalid arm: {arm}")
    
    def get_arm_angles(self, arm: str) -> np.ndarray:
        """Get current joint angles for specified arm."""
        if arm == "left":
            angles = self.left_arm_angles.copy()
        elif arm == "right":
            angles = self.right_arm_angles.copy()
        else:
            raise ValueError(f"Invalid arm: {arm}")
        
        return angles
    
    def get_arm_angles_for_visualization(self, arm: str) -> np.ndarray:
        """Get current joint angles for specified arm, for PyBullet visualization."""
        # Return raw angles without any correction for proper diagnosis
        return self.get_arm_angles(arm)
    
    def get_actual_arm_angles(self, arm: str) -> np.ndarray:
        """Get actual joint angles from robot hardware (not commanded angles)."""
        try:
            if arm == "left" and self.left_robot and self.left_arm_connected:
                observation = self.left_robot.get_observation()
                if observation:
                    # Construct array from observation
                    vals = []
                    for name in self.joint_names:
                        vals.append(observation.get(f"{name}.pos", 0.0))
                    return np.array(vals)
                    
            elif arm == "right" and self.right_robot and self.right_arm_connected:
                observation = self.right_robot.get_observation()
                if observation:
                    vals = []
                    for name in self.joint_names:
                         vals.append(observation.get(f"{name}.pos", 0.0))
                    return np.array(vals)
                    
        except Exception as e:
            logger.debug(f"Error reading actual arm angles for {arm}: {e}")
        
        # Fallback to commanded angles if we can't read actual angles
        return self.get_arm_angles(arm)
    
    def return_to_initial_position(self):
        """Return both arms to initial position."""
        logger.info("⏪ Returning robot to initial position...")
        
        try:
            # Set initial positions
            self.left_arm_angles = self.initial_left_arm.copy()
            self.right_arm_angles = self.initial_right_arm.copy()
            
            # Send commands for a few iterations to ensure movement
            for i in range(10):
                self.send_command()
                time.sleep(0.1)
                
            logger.info("✅ Robot returned to initial position")
        except Exception as e:
            logger.error(f"Error returning to initial position: {e}")
    
    def disable_torque(self, arm: str = None):
        """Disable torque on robot joints.

        Args:
            arm: 'left', 'right', or None for both arms
        """
        if not self.is_connected:
            return

        try:
            if arm is None or arm == "left":
                if self.left_robot and self.left_arm_connected:
                    logger.info("Disabling torque on LEFT arm...")
                    self.left_robot.bus.disable_torque()

            if arm is None or arm == "right":
                if self.right_robot and self.right_arm_connected:
                    logger.info("Disabling torque on RIGHT arm...")
                    self.right_robot.bus.disable_torque()

        except Exception as e:
            logger.error(f"Error disabling torque: {e}")
    
    def disconnect(self):
        """Disconnect from robot hardware."""
        if not self.is_connected:
            return
        
        logger.info("Disconnecting from robot...")
        
        # Return to initial positions if engaged
        if self.is_engaged:
            try:
                self.return_to_initial_position()
            except Exception as e:
                logger.error(f"Error returning to initial position: {e}")
        
        # Disconnect both arms
        if self.left_robot:
            try:
                self.left_robot.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting left arm: {e}")
            self.left_robot = None
            
        if self.right_robot:
            try:
                self.right_robot.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting right arm: {e}")
            self.right_robot = None
        
        self.is_connected = False
        self.is_engaged = False
        self.left_arm_connected = False
        self.right_arm_connected = False
        logger.info("🔌 Robot disconnected")
    
    def get_arm_connection_status(self, arm: str) -> bool:
        """Get connection status for specific arm.
        
        In pure simulation mode (robot disabled), check if arm is enabled.
        When robot is enabled, check device file existence.
        """
        # First check if arm is enabled in configuration
        if not self.config.is_arm_enabled(arm):
            return False
        
        # In pure simulation mode, treat enabled arms as connected
        if not self.config.enable_robot:
            return True
        
        # When robot is enabled, check device file existence
        if arm == "left":
            device_path = self.config.follower_ports["left"]
            return os.path.exists(device_path)
        elif arm == "right":
            device_path = self.config.follower_ports["right"] 
            return os.path.exists(device_path)
        else:
            return False

    def update_arm_connection_status(self):
        """Update individual arm connection status based on device file existence."""
        if self.is_connected:
            self.left_arm_connected = os.path.exists(self.config.follower_ports["left"])
            self.right_arm_connected = os.path.exists(self.config.follower_ports["right"])
    
    @property
    def status(self) -> Dict:
        """Get robot status information."""
        return {
            "connected": self.is_connected,
            "left_arm_connected": self.left_arm_connected,
            "right_arm_connected": self.right_arm_connected,
            "left_arm_angles": self.left_arm_angles.tolist(),
            "right_arm_angles": self.right_arm_angles.tolist(),
            "joint_limits_min": self.joint_limits_min_deg.tolist(),
            "joint_limits_max": self.joint_limits_max_deg.tolist(),
        }