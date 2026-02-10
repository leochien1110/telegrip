"""
Configuration module for the unified teleoperation system.
Loads configuration from config.yaml file with fallback to default values.
"""

import os
import yaml
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import logging
from .utils import get_absolute_path, get_project_root

logger = logging.getLogger(__name__)

# --- Robot Type Definitions ---
ROBOT_CONFIGS = {
    "so100": {
        "joint_names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        "num_ik_joints": 3,
        "urdf_path": "URDF/SO100/so100.urdf",
        "wrist_flex_index": 3,
        "wrist_roll_index": 4,
        "gripper_index": 5,
        "end_effector_link_name": "Fixed_Jaw_tip",
        # Mapping from URDF joint names to internal names
        "joint_mapping": {
            "1": "shoulder_pan",
            "2": "shoulder_lift",
            "3": "elbow_flex",
            "4": "wrist_flex",
            "5": "wrist_roll",
            "6": "gripper"
        }
    },
    "ur10e": {
        "joint_names": ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        "num_ik_joints": 6,
        "urdf_path": "URDF/UR10e/ur10e.urdf",
        "wrist_flex_index": None, # Handled by IK
        "wrist_roll_index": None, # Handled by IK
        "gripper_index": None,    # No gripper
        "end_effector_link_name": "tool0",
        "joint_mapping": None     # Use joint names directly
    },
    "ur10e_2f85": {
        "joint_names": ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint", "finger_joint"],
        "num_ik_joints": 6,
        "urdf_path": "URDF/UR10e_2f85/urdf/ur10e_2f85.urdf",
        "wrist_flex_index": None, # Handled by IK
        "wrist_roll_index": None, # Handled by IK
        "gripper_index": 6,
        "end_effector_link_name": "tool0", # Or maybe a frame on the gripper? sticking to tool0 for IK usually
        "joint_mapping": None     # Use joint names directly
    }
}

# Default configuration values (fallback if YAML file doesn't exist)
DEFAULT_CONFIG = {
    "robot_type": "so100",  # Default to SO-100
    "network": {
        "https_port": 8443,
        "websocket_port": 8442,
        "host_ip": "0.0.0.0"
    },
    "ssl": {
        "certfile": "cert.pem",
        "keyfile": "key.pem"
    },
    "robot": {
        "left_arm": {
            "name": "Left Arm",
            "port": "/dev/ttyACM0",
            "enabled": True
        },
        "right_arm": {
            "name": "Right Arm",
            "port": "/dev/ttyACM1",
            "enabled": True
        },
        "vr_to_robot_scale": 1.0,
        "send_interval": 0.05,
    },
    "control": {
        "keyboard": {
            "enabled": True,
            "pos_step": 0.01,
            "angle_step": 5.0,
            "gripper_step": 10.0
        },
        "vr": {
            "enabled": True
        },
        "pybullet": {
            "enabled": True
        }
    },
    "paths": {
        "urdf_path": None # Will terminate to default based on robot_type if None
    },
    "gripper": {
        "open_angle": 0.0,
        "closed_angle": 45.0
    },
    "ik": {
        "use_reference_poses": True,
        "reference_poses_file": "reference_poses.json",
        "position_error_threshold": 0.001,
        "hysteresis_threshold": 0.01,
        "movement_penalty_weight": 0.01
    }
}

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file with fallback to defaults."""
    config = DEFAULT_CONFIG.copy()
    
    # Try to load from project root first (package installation directory)
    package_config_path = get_absolute_path(config_path)
    
    # Check if config exists in package directory
    if package_config_path.exists():
        config_file_to_use = package_config_path
    # Fallback to current working directory (for user-provided configs)
    elif os.path.exists(config_path):
        config_file_to_use = Path(config_path).absolute()
    else:
        logger.info(f"Config file {config_path} not found")
        return config
    
    print(f"DEBUG: load_config loading from {config_file_to_use}")
    try:
        with open(config_file_to_use, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                # Deep merge yaml config into default config
                _deep_merge(config, yaml_config)
    except Exception as e:
        logger.warning(f"Could not load config from {config_file_to_use}: {e}")
    
    return config

def save_config(config: dict, config_path: str = "config.yaml"):
    """Save configuration to YAML file in project root."""
    # Always save to project root directory
    abs_config_path = get_absolute_path(config_path)
    try:
        with open(abs_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving config to {abs_config_path}: {e}")
        return False

def _deep_merge(base: dict, update: dict):
    """Deep merge update dict into base dict."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value

def reload_config(config_path: str = "config.yaml"):
    """Reload configuration from a different file and update all global constants."""
    global _config_data, ROBOT_TYPE, HTTPS_PORT, WEBSOCKET_PORT, HOST_IP
    global CERTFILE, KEYFILE, VR_TO_ROBOT_SCALE, SEND_INTERVAL
    global POS_STEP, ANGLE_STEP, GRIPPER_STEP, URDF_PATH
    global GRIPPER_OPEN_ANGLE, GRIPPER_CLOSED_ANGLE
    global USE_REFERENCE_POSES, REFERENCE_POSES_FILE
    global IK_POSITION_ERROR_THRESHOLD, IK_HYSTERESIS_THRESHOLD, IK_MOVEMENT_PENALTY_WEIGHT
    global ENABLE_ROBOT, ENABLE_PYBULLET, ENABLE_PYBULLET_GUI, ENABLE_VR, ENABLE_KEYBOARD
    global JOINT_NAMES, NUM_JOINTS, NUM_IK_JOINTS, WRIST_FLEX_INDEX, WRIST_ROLL_INDEX, GRIPPER_INDEX
    global END_EFFECTOR_LINK_NAME, URDF_TO_INTERNAL_NAME_MAP, DEFAULT_FOLLOWER_PORTS
    
    # Reload config data
    print(f"DEBUG: reload_config called with {config_path}")
    _config_data = load_config(config_path)
    print(f"DEBUG: _config_data updated. robot_type={_config_data.get('robot_type')}, enable_robot={_config_data.get('enable_robot')}")
    
    # Update robot type
    ROBOT_TYPE = _config_data.get("robot_type", "so100")
    if ROBOT_TYPE not in ROBOT_CONFIGS:
        logger.warning(f"Unknown robot type: {ROBOT_TYPE}, falling back to so100")
        ROBOT_TYPE = "so100"
    
    _robot_config = ROBOT_CONFIGS[ROBOT_TYPE]
    
    # Update all global constants
    HTTPS_PORT = _config_data["network"]["https_port"]
    WEBSOCKET_PORT = _config_data["network"]["websocket_port"]
    HOST_IP = _config_data["network"]["host_ip"]
    
    CERTFILE = _config_data["ssl"]["certfile"]
    KEYFILE = _config_data["ssl"]["keyfile"]
    
    VR_TO_ROBOT_SCALE = _config_data["robot"]["vr_to_robot_scale"]
    SEND_INTERVAL = _config_data["robot"]["send_interval"]
    
    POS_STEP = _config_data["control"]["keyboard"]["pos_step"]
    ANGLE_STEP = _config_data["control"]["keyboard"]["angle_step"]
    GRIPPER_STEP = _config_data["control"]["keyboard"]["gripper_step"]
    
    URDF_PATH = _config_data["paths"].get("urdf_path") or _robot_config["urdf_path"]
    
    GRIPPER_OPEN_ANGLE = _config_data["gripper"]["open_angle"]
    GRIPPER_CLOSED_ANGLE = _config_data["gripper"]["closed_angle"]
    
    USE_REFERENCE_POSES = _config_data["ik"]["use_reference_poses"]
    REFERENCE_POSES_FILE = _config_data["ik"]["reference_poses_file"]
    IK_POSITION_ERROR_THRESHOLD = _config_data["ik"]["position_error_threshold"]
    IK_HYSTERESIS_THRESHOLD = _config_data["ik"]["hysteresis_threshold"]
    IK_MOVEMENT_PENALTY_WEIGHT = _config_data["ik"]["movement_penalty_weight"]
    
    ENABLE_ROBOT = _config_data.get("enable_robot", True)
    ENABLE_PYBULLET = _config_data.get("enable_pybullet", True)
    ENABLE_PYBULLET_GUI = _config_data.get("enable_pybullet_gui", True)
    ENABLE_VR = _config_data.get("enable_vr", True)
    ENABLE_KEYBOARD = _config_data.get("enable_keyboard", True)
    
    JOINT_NAMES = _robot_config["joint_names"]
    NUM_JOINTS = len(JOINT_NAMES)
    NUM_IK_JOINTS = _robot_config["num_ik_joints"]
    WRIST_FLEX_INDEX = _robot_config["wrist_flex_index"]
    WRIST_ROLL_INDEX = _robot_config["wrist_roll_index"]
    GRIPPER_INDEX = _robot_config["gripper_index"]
    
    END_EFFECTOR_LINK_NAME = _robot_config["end_effector_link_name"]
    
    URDF_TO_INTERNAL_NAME_MAP = _robot_config.get("joint_mapping")
    if URDF_TO_INTERNAL_NAME_MAP is None:
        URDF_TO_INTERNAL_NAME_MAP = {}
    
    print(f"DEBUG: reload_config finished. ROBOT_TYPE={ROBOT_TYPE}, ENABLE_ROBOT={ENABLE_ROBOT}, URDF_PATH={URDF_PATH}")
    
    DEFAULT_FOLLOWER_PORTS = {
        "left": _config_data["robot"]["left_arm"]["port"],
        "right": _config_data["robot"]["right_arm"]["port"]
    }


# Load configuration
_config_data = load_config()

# determine Robot Type and apply specific configurations
ROBOT_TYPE = _config_data.get("robot_type", "so100")
if ROBOT_TYPE not in ROBOT_CONFIGS:
    logger.warning(f"Unknown robot type: {ROBOT_TYPE}, falling back to so100")
    ROBOT_TYPE = "so100"

_robot_config = ROBOT_CONFIGS[ROBOT_TYPE]

# Extract values for backward compatibility
HTTPS_PORT = _config_data["network"]["https_port"]
WEBSOCKET_PORT = _config_data["network"]["websocket_port"]
HOST_IP = _config_data["network"]["host_ip"]

CERTFILE = _config_data["ssl"]["certfile"]
KEYFILE = _config_data["ssl"]["keyfile"]

VR_TO_ROBOT_SCALE = _config_data["robot"]["vr_to_robot_scale"]
SEND_INTERVAL = _config_data["robot"]["send_interval"]

POS_STEP = _config_data["control"]["keyboard"]["pos_step"]
ANGLE_STEP = _config_data["control"]["keyboard"]["angle_step"]
GRIPPER_STEP = _config_data["control"]["keyboard"]["gripper_step"]

# Use configured path or default for robot type
URDF_PATH = _config_data["paths"].get("urdf_path") or _robot_config["urdf_path"]

GRIPPER_OPEN_ANGLE = _config_data["gripper"]["open_angle"]
GRIPPER_CLOSED_ANGLE = _config_data["gripper"]["closed_angle"]

# IK Configuration
USE_REFERENCE_POSES = _config_data["ik"]["use_reference_poses"]
REFERENCE_POSES_FILE = _config_data["ik"]["reference_poses_file"]
IK_POSITION_ERROR_THRESHOLD = _config_data["ik"]["position_error_threshold"]
IK_HYSTERESIS_THRESHOLD = _config_data["ik"]["hysteresis_threshold"]
IK_MOVEMENT_PENALTY_WEIGHT = _config_data["ik"]["movement_penalty_weight"]

# Control Flags (from root or defaults)
ENABLE_ROBOT = _config_data.get("enable_robot", True)
ENABLE_PYBULLET = _config_data.get("enable_pybullet", True)
ENABLE_PYBULLET_GUI = _config_data.get("enable_pybullet_gui", True)
ENABLE_VR = _config_data.get("enable_vr", True)
ENABLE_KEYBOARD = _config_data.get("enable_keyboard", True)

# --- Joint Configuration ---
# Set dynamically based on robot type
JOINT_NAMES = _robot_config["joint_names"]
NUM_JOINTS = len(JOINT_NAMES)
NUM_IK_JOINTS = _robot_config["num_ik_joints"]
WRIST_FLEX_INDEX = _robot_config["wrist_flex_index"]
WRIST_ROLL_INDEX = _robot_config["wrist_roll_index"]
GRIPPER_INDEX = _robot_config["gripper_index"]

# Motor configuration for SO100 - Only relevant if using SO100 robot interface
COMMON_MOTORS = {
    "shoulder_pan": [1, "sts3215"],
    "shoulder_lift": [2, "sts3215"], 
    "elbow_flex": [3, "sts3215"],
    "wrist_flex": [4, "sts3215"],
    "wrist_roll": [5, "sts3215"],
    "gripper": [6, "sts3215"],
}

# URDF joint name mapping - Backward compatibility
URDF_TO_INTERNAL_NAME_MAP = _robot_config.get("joint_mapping")
if URDF_TO_INTERNAL_NAME_MAP is None:
    # If no mapping provided, assume simple mapping if using default so100 logic
    # But since we moved SO100 mapping to ROBOT_CONFIGS, this should handle it.
    # For others, we might default to empty or identity if code expects it.
    URDF_TO_INTERNAL_NAME_MAP = {}

# --- PyBullet Configuration ---
END_EFFECTOR_LINK_NAME = _robot_config["end_effector_link_name"]

# --- Keyboard Control ---
POS_STEP = 0.01  # meters
ANGLE_STEP = 5.0 # degrees
GRIPPER_STEP = 10.0 # degrees

# --- Device Ports ---
DEFAULT_FOLLOWER_PORTS = {
    "left": _config_data["robot"]["left_arm"]["port"],
    "right": _config_data["robot"]["right_arm"]["port"]
}

@dataclass
class TelegripConfig:
    """Main configuration class for the teleoperation system."""
    
    # Network settings
    https_port: int = None
    websocket_port: int = None
    host_ip: str = None
    
    # SSL settings
    certfile: str = None
    keyfile: str = None
    
    # Robot settings
    robot_type: str = None
    vr_to_robot_scale: float = None
    send_interval: float = None
    
    # Device ports
    follower_ports: Dict[str, str] = None
    
    # Control flags
    enable_pybullet: bool = None
    enable_pybullet_gui: bool = None
    enable_robot: bool = None
    enable_vr: bool = True
    enable_keyboard: bool = True
    autoconnect: bool = False
    log_level: str = "warning"
    
    # Arm control - which arms are enabled ("left", "right", or "dual")
    enabled_arms: str = "dual"  # Options: "left", "right", "dual"
    
    # Paths
    urdf_path: str = None
    webapp_dir: str = "webapp"
    
    # IK settings
    use_reference_poses: bool = None
    reference_poses_file: str = None
    ik_position_error_threshold: float = None
    ik_hysteresis_threshold: float = None
    ik_movement_penalty_weight: float = None
    
    # Gripper settings
    gripper_open_angle: float = None
    gripper_closed_angle: float = None
    
    # Keyboard control
    pos_step: float = None
    angle_step: float = None
    gripper_step: float = None
    
    def __post_init__(self):
        print(f"DEBUG: TelegripConfig.__post_init__ starting. Globals: ROBOT_TYPE={ROBOT_TYPE}, ENABLE_ROBOT={ENABLE_ROBOT}, URDF_PATH={URDF_PATH}")
        # Update defaults from globals (which might have been reloaded)
        if self.https_port is None: self.https_port = HTTPS_PORT
        if self.websocket_port is None: self.websocket_port = WEBSOCKET_PORT
        if self.host_ip is None: self.host_ip = HOST_IP
        if self.certfile is None: self.certfile = CERTFILE
        if self.keyfile is None: self.keyfile = KEYFILE
        if self.robot_type is None: self.robot_type = ROBOT_TYPE
        if self.vr_to_robot_scale is None: self.vr_to_robot_scale = VR_TO_ROBOT_SCALE
        if self.send_interval is None: self.send_interval = SEND_INTERVAL
        if self.urdf_path is None: self.urdf_path = URDF_PATH
        if self.reference_poses_file is None: self.reference_poses_file = REFERENCE_POSES_FILE
        if self.ik_position_error_threshold is None: self.ik_position_error_threshold = IK_POSITION_ERROR_THRESHOLD
        if self.ik_hysteresis_threshold is None: self.ik_hysteresis_threshold = IK_HYSTERESIS_THRESHOLD
        if self.ik_movement_penalty_weight is None: self.ik_movement_penalty_weight = IK_MOVEMENT_PENALTY_WEIGHT
        if self.gripper_open_angle is None: self.gripper_open_angle = GRIPPER_OPEN_ANGLE
        if self.gripper_closed_angle is None: self.gripper_closed_angle = GRIPPER_CLOSED_ANGLE
        if self.pos_step is None: self.pos_step = POS_STEP
        if self.angle_step is None: self.angle_step = ANGLE_STEP
        if self.gripper_step is None: self.gripper_step = GRIPPER_STEP

        # Boolean flags
        if self.enable_pybullet is None: self.enable_pybullet = ENABLE_PYBULLET
        if self.enable_pybullet_gui is None: self.enable_pybullet_gui = ENABLE_PYBULLET_GUI
        if self.enable_robot is None: self.enable_robot = ENABLE_ROBOT
        if self.enable_vr is None: self.enable_vr = ENABLE_VR
        if self.enable_keyboard is None: self.enable_keyboard = ENABLE_KEYBOARD
        if self.use_reference_poses is None: self.use_reference_poses = USE_REFERENCE_POSES

        # Initialize follower_ports if not set
        if self.follower_ports is None:
            self.follower_ports = {
                "left": _config_data["robot"]["left_arm"]["port"],
                "right": _config_data["robot"]["right_arm"]["port"]
            }
        
        # Ensure ports are not None
        if self.follower_ports["left"] is None:
            self.follower_ports["left"] = "/dev/ttyACM0"
        if self.follower_ports["right"] is None:
            self.follower_ports["right"] = "/dev/ttyACM1"
    
    @property
    def ssl_files_exist(self) -> bool:
        """Check if SSL certificate files exist."""
        cert_path = get_absolute_path(self.certfile)
        key_path = get_absolute_path(self.keyfile)
        return cert_path.exists() and key_path.exists()
    
    def ensure_ssl_certificates(self) -> bool:
        """Ensure SSL certificates exist, generating them if necessary."""
        from .utils import ensure_ssl_certificates
        return ensure_ssl_certificates(self.certfile, self.keyfile)
    
    @property
    def urdf_exists(self) -> bool:
        """Check if URDF file exists."""
        urdf_path = get_absolute_path(self.urdf_path)
        return urdf_path.exists()
    
    @property
    def webapp_exists(self) -> bool:
        """Check if webapp directory exists."""
        webapp_path = get_absolute_path(self.webapp_dir)
        return webapp_path.exists()
    
    def get_absolute_urdf_path(self) -> str:
        """Get absolute path to URDF file."""
        return str(get_absolute_path(self.urdf_path))
    
    def get_absolute_reference_poses_path(self) -> str:
        """Get absolute path to reference poses file."""
        return str(get_absolute_path(self.reference_poses_file))
    
    def get_absolute_ssl_paths(self) -> tuple:
        """Get absolute paths to SSL certificate files."""
        cert_path = str(get_absolute_path(self.certfile))
        key_path = str(get_absolute_path(self.keyfile))
        return cert_path, key_path
    
    def is_arm_enabled(self, arm: str) -> bool:
        """Check if a specific arm is enabled.
        
        Args:
            arm: "left" or "right"
            
        Returns:
            True if the arm is enabled, False otherwise
        """
        if self.enabled_arms == "dual":
            return True
        return self.enabled_arms == arm

def get_config_data():
    """Get the current configuration data."""
    return _config_data.copy()

def update_config_data(new_config: dict):
    """Update the global configuration data."""
    global _config_data
    _config_data = new_config
    
    # Save to file
    save_config(_config_data)

# Global configuration instance
config = TelegripConfig()
