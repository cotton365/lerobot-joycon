"""
JAKA Motors Bus implementation for LeRobot.

This module provides the JakaMotorsBus class for controlling JAKA Zu series 6-axis robot arms.
JAKA robots communicate via TCP/IP network connection rather than serial ports.
"""

import enum
import logging
import time
import traceback
from copy import deepcopy

import numpy as np

from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc

# JAKA specific constants
JAKA_DEFAULT_PORT = 10000  # Default JAKA control port
JAKA_TIMEOUT_MS = 1000
JAKA_NUM_JOINTS = 6  # JAKA Zu series is 6-axis

# Joint bounds for JAKA Zu series (in degrees)
# Note: These should be verified against your specific JAKA Zu model documentation
LOWER_BOUND_DEGREE = -270
UPPER_BOUND_DEGREE = 270
LOWER_BOUND_LINEAR = -10  # For gripper (if applicable)
UPPER_BOUND_LINEAR = 110

HALF_TURN_DEGREE = 180

# JAKA Zu series joint limits (example values - verify with your JAKA documentation)
# Format: [joint1, joint2, joint3, joint4, joint5, joint6]
JAKA_ZU_JOINT_LIMITS = {
    "min_degrees": [-170, -130, -135, -180, -125, -360],
    "max_degrees": [170, 130, 80, 180, 125, 360],
}

# Retry configuration
NUM_READ_RETRY = 10
NUM_WRITE_RETRY = 10


class CalibrationMode(enum.Enum):
    """Calibration mode for joint types."""
    # Joints with rotational motions are expressed in degrees in nominal range of [-180, 180]
    DEGREE = 0
    # Joints with linear motions (like gripper) are expressed in nominal range of [0, 100]
    LINEAR = 1


class JointOutOfRangeError(Exception):
    """Exception raised when joint position is out of safe range."""
    def __init__(self, message="Joint is out of range"):
        self.message = message
        super().__init__(self.message)


class JakaMotorsBus:
    """
    JakaMotorsBus class for controlling JAKA Zu series robot arms with LeRobot.

    Unlike Feetech or Dynamixel motors that use serial communication, JAKA robots
    communicate via TCP/IP network connection using the JAKA SDK.

    Example usage:
    ```python
    motors_bus = JakaMotorsBus(
        ip_address="192.168.1.100",
        port=10000,
        motors={
            "joint_1": (1, "jaka_zu"),
            "joint_2": (2, "jaka_zu"),
            "joint_3": (3, "jaka_zu"),
            "joint_4": (4, "jaka_zu"),
            "joint_5": (5, "jaka_zu"),
            "joint_6": (6, "jaka_zu"),
        },
    )
    motors_bus.connect()

    # Read current joint positions
    positions = motors_bus.read("Present_Position")

    # Write target positions
    motors_bus.write("Goal_Position", positions + 10.0)

    # Disconnect when done
    motors_bus.disconnect()
    ```
    """

    def __init__(
        self,
        ip_address: str,
        port: int = JAKA_DEFAULT_PORT,
        motors: dict[str, tuple[int, str]] | None = None,
        mock: bool = False,
    ):
        """
        Initialize JAKA motors bus.

        Args:
            ip_address: IP address of the JAKA robot controller
            port: TCP port for JAKA communication (default: 10000)
            motors: Dictionary mapping motor names to (index, model) tuples
                   Example: {"joint_1": (1, "jaka_zu"), ...}
            mock: If True, use mock connection for testing
        """
        self.ip_address = ip_address
        self.port = port
        self.motors = motors or {}
        self.mock = mock

        # JAKA SDK connection handle (will be initialized in connect())
        self.robot_handle = None
        self.is_connected = False

        # Calibration data
        self.calibration = None

        # Logging
        self.logs = {}

        # Track current positions for safety checks
        self.track_positions = {}

    def connect(self):
        """
        Establish connection to JAKA robot via TCP/IP.

        This method uses JAKA SDK functions to connect to the robot controller.
        """
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"JakaMotorsBus({self.ip_address}) is already connected."
            )

        if self.mock:
            # Mock connection for testing
            logging.info(f"[MOCK] Connecting to JAKA robot at {self.ip_address}:{self.port}")
            self.robot_handle = "mock_handle"
            self.is_connected = True
            return

        try:
            # ===============================================
            # PSEUDO CODE - Replace with actual JAKA SDK calls
            # ===============================================
            # from jaka_sdk import JAKARobot  # Import JAKA SDK
            #
            # # Create JAKA robot instance
            # self.robot_handle = JAKARobot()
            #
            # # Connect to robot controller
            # ret = self.robot_handle.login_in(self.ip_address, self.port)
            # if ret != 0:
            #     raise ConnectionError(f"Failed to connect to JAKA robot at {self.ip_address}:{self.port}, error code: {ret}")
            #
            # # Enable robot (power on)
            # ret = self.robot_handle.power_on()
            # if ret != 0:
            #     raise ConnectionError(f"Failed to power on JAKA robot, error code: {ret}")
            #
            # # Enable robot control
            # ret = self.robot_handle.enable_robot()
            # if ret != 0:
            #     raise ConnectionError(f"Failed to enable JAKA robot, error code: {ret}")
            # ===============================================

            logging.info(f"Successfully connected to JAKA robot at {self.ip_address}:{self.port}")
            self.is_connected = True

        except Exception as e:
            traceback.print_exc()
            raise ConnectionError(f"Failed to connect to JAKA robot: {e}")

    def reconnect(self):
        """Reconnect to JAKA robot after connection loss."""
        self.disconnect()
        time.sleep(1)
        self.connect()

    def disconnect(self):
        """
        Disconnect from JAKA robot and release resources.
        """
        if not self.is_connected:
            return

        if self.mock:
            logging.info("[MOCK] Disconnecting from JAKA robot")
            self.robot_handle = None
            self.is_connected = False
            return

        try:
            # ===============================================
            # PSEUDO CODE - Replace with actual JAKA SDK calls
            # ===============================================
            # # Disable robot
            # self.robot_handle.disable_robot()
            #
            # # Power off
            # self.robot_handle.power_off()
            #
            # # Logout
            # self.robot_handle.login_out()
            # ===============================================

            logging.info("Disconnected from JAKA robot")
            self.robot_handle = None
            self.is_connected = False

        except Exception as e:
            logging.error(f"Error during disconnect: {e}")
            self.is_connected = False

    @property
    def motor_names(self) -> list[str]:
        """Get list of motor names."""
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        """Get list of motor models."""
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        """Get list of motor indices (1-6 for JAKA)."""
        return [idx for idx, _ in self.motors.values()]

    def set_calibration(self, calibration: dict[str, list]):
        """
        Set calibration parameters for the motors.

        Args:
            calibration: Dictionary containing calibration data with keys:
                - "motor_names": List of motor names
                - "homing_offset": List of homing offsets in degrees
                - "drive_mode": List of drive modes (0=normal, 1=inverted)
                - "calib_mode": List of calibration modes ("DEGREE" or "LINEAR")
        """
        self.calibration = calibration

    def apply_calibration(self, values: np.ndarray | list, motor_names: list[str] | None = None):
        """
        Apply calibration to convert from JAKA joint angles to LeRobot standard range.

        JAKA SDK typically returns joint angles in degrees. This method converts them
        to the LeRobot standard range of [-180, 180] degrees with calibration offsets.

        Args:
            values: Raw joint values from JAKA (in degrees)
            motor_names: List of motor names (if None, uses all motors)

        Returns:
            Calibrated values in LeRobot standard range
        """
        if motor_names is None:
            motor_names = self.motor_names

        values = np.array(values, dtype=np.float32)

        if self.calibration is None:
            logging.warning("No calibration set, returning raw values")
            return values

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                drive_mode = self.calibration["drive_mode"][calib_idx]
                homing_offset = self.calibration["homing_offset"][calib_idx]

                # Apply homing offset
                values[i] -= homing_offset

                # Apply drive mode (invert if needed)
                if drive_mode:
                    values[i] *= -1

                # Normalize to [-180, 180] range
                values[i] = ((values[i] + 180) % 360) - 180

                # Check bounds
                if values[i] < LOWER_BOUND_DEGREE or values[i] > UPPER_BOUND_DEGREE:
                    raise JointOutOfRangeError(
                        f"Joint {name} at index {i} is out of range: {values[i]:.2f} degrees "
                        f"(bounds: [{LOWER_BOUND_DEGREE}, {UPPER_BOUND_DEGREE}])"
                    )

            elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                # For gripper or linear joints
                if values[i] < LOWER_BOUND_LINEAR or values[i] > UPPER_BOUND_LINEAR:
                    raise JointOutOfRangeError(
                        f"Joint {name} at index {i} is out of range: {values[i]:.2f}% "
                        f"(bounds: [{LOWER_BOUND_LINEAR}, {UPPER_BOUND_LINEAR}])"
                    )

        return values

    def revert_calibration(self, values: np.ndarray | list, motor_names: list[str] | None = None):
        """
        Revert calibration to convert from LeRobot standard range to JAKA joint angles.

        This is the inverse of apply_calibration, used before sending commands to JAKA.

        Args:
            values: Calibrated values in LeRobot standard range
            motor_names: List of motor names (if None, uses all motors)

        Returns:
            Raw joint values for JAKA (in degrees)
        """
        if motor_names is None:
            motor_names = self.motor_names

        values = np.array(values, dtype=np.float32)

        if self.calibration is None:
            logging.warning("No calibration set, returning raw values")
            return values

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                drive_mode = self.calibration["drive_mode"][calib_idx]
                homing_offset = self.calibration["homing_offset"][calib_idx]

                # Reverse drive mode inversion
                if drive_mode:
                    values[i] *= -1

                # Reverse homing offset
                values[i] += homing_offset

        return values

    def read(self, data_name: str, motor_names: list[str] | None = None) -> np.ndarray:
        """
        Read data from JAKA robot.

        Args:
            data_name: Type of data to read. Supported values:
                - "Present_Position": Current joint positions in degrees
                - "Present_Speed": Current joint velocities
                - "Present_Current": Current joint currents (if available)
            motor_names: List of motor names to read from (if None, reads all)

        Returns:
            NumPy array of values with timestamp
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "JakaMotorsBus is not connected. Call `motors_bus.connect()` first."
            )

        if motor_names is None:
            motor_names = self.motor_names

        num_motors = len(motor_names)

        # Initialize result array
        values = np.zeros(num_motors, dtype=np.float32)

        if self.mock:
            # Return mock data
            if data_name == "Present_Position":
                values = np.array([0.0] * num_motors, dtype=np.float32)
            return values

        try:
            # ===============================================
            # PSEUDO CODE - Replace with actual JAKA SDK calls
            # ===============================================
            if data_name == "Present_Position":
                # # Read current joint positions
                # ret, joint_positions = self.robot_handle.get_joint_position()
                # if ret != 0:
                #     raise ConnectionError(f"Failed to read joint positions, error code: {ret}")
                #
                # # Extract positions for requested motors
                # for i, name in enumerate(motor_names):
                #     motor_idx = self.motors[name][0]  # Get motor index (1-6)
                #     values[i] = joint_positions[motor_idx - 1]  # JAKA uses 0-indexed arrays

                pass  # Replace with actual implementation

            elif data_name == "Present_Speed":
                # # Read current joint velocities
                # ret, joint_speeds = self.robot_handle.get_joint_speed()
                # if ret != 0:
                #     raise ConnectionError(f"Failed to read joint speeds, error code: {ret}")
                #
                # for i, name in enumerate(motor_names):
                #     motor_idx = self.motors[name][0]
                #     values[i] = joint_speeds[motor_idx - 1]

                pass  # Replace with actual implementation

            elif data_name == "Present_Current":
                # # Read current joint currents (if supported by JAKA SDK)
                # ret, joint_currents = self.robot_handle.get_joint_current()
                # if ret != 0:
                #     logging.warning(f"Failed to read joint currents, error code: {ret}")
                #     return np.zeros(num_motors, dtype=np.float32)
                #
                # for i, name in enumerate(motor_names):
                #     motor_idx = self.motors[name][0]
                #     values[i] = joint_currents[motor_idx - 1]

                pass  # Replace with actual implementation

            else:
                raise NotImplementedError(f"Reading '{data_name}' is not implemented for JAKA")
            # ===============================================

            # Apply calibration if reading positions
            if data_name == "Present_Position" and self.calibration is not None:
                values = self.apply_calibration(values, motor_names)

            return values

        except Exception as e:
            logging.error(f"Error reading {data_name}: {e}")
            raise

    def write(
        self,
        data_name: str,
        values: int | float | np.ndarray,
        motor_names: list[str] | None = None,
    ):
        """
        Write data to JAKA robot.

        Args:
            data_name: Type of data to write. Supported values:
                - "Goal_Position": Target joint positions in degrees
                - "Goal_Speed": Target joint velocities (if applicable)
            values: Values to write (single value or array)
            motor_names: List of motor names to write to (if None, writes to all)
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "JakaMotorsBus is not connected. Call `motors_bus.connect()` first."
            )

        if motor_names is None:
            motor_names = self.motor_names

        # Convert to numpy array
        if isinstance(values, (int, float)):
            values = np.array([values] * len(motor_names), dtype=np.float32)
        else:
            values = np.array(values, dtype=np.float32)

        if len(values) != len(motor_names):
            raise ValueError(
                f"Number of values ({len(values)}) does not match number of motors ({len(motor_names)})"
            )

        if self.mock:
            logging.info(f"[MOCK] Writing {data_name}: {values}")
            return

        try:
            # ===============================================
            # PSEUDO CODE - Replace with actual JAKA SDK calls
            # ===============================================
            if data_name == "Goal_Position":
                # Revert calibration to get raw JAKA angles
                if self.calibration is not None:
                    values = self.revert_calibration(values, motor_names)

                # # Prepare full joint array (all 6 joints)
                # # First, read current positions
                # ret, current_positions = self.robot_handle.get_joint_position()
                # if ret != 0:
                #     raise ConnectionError(f"Failed to read current positions, error code: {ret}")
                #
                # # Update only the specified motors
                # target_positions = current_positions.copy()
                # for i, name in enumerate(motor_names):
                #     motor_idx = self.motors[name][0]  # Get motor index (1-6)
                #     target_positions[motor_idx - 1] = values[i]
                #
                # # Send joint motion command
                # ret = self.robot_handle.joint_move(
                #     target_positions,
                #     move_mode=0,  # 0=absolute, 1=relative
                #     is_block=False,  # Non-blocking motion
                #     speed=50.0  # Speed percentage (adjust as needed)
                # )
                # if ret != 0:
                #     raise ConnectionError(f"Failed to move joints, error code: {ret}")

                pass  # Replace with actual implementation

            elif data_name == "Goal_Speed":
                # # Set joint speed limits (if needed)
                # logging.warning("Setting joint speed via Goal_Speed is not standard for JAKA")

                pass  # Replace with actual implementation

            else:
                raise NotImplementedError(f"Writing '{data_name}' is not implemented for JAKA")
            # ===============================================

        except Exception as e:
            logging.error(f"Error writing {data_name}: {e}")
            raise

    def disconnect(self):
        """
        Disconnect from JAKA robot.
        Alias for consistency with other motor bus implementations.
        """
        if not self.is_connected:
            return

        if self.mock:
            logging.info("[MOCK] Disconnecting from JAKA robot")
            self.robot_handle = None
            self.is_connected = False
            return

        try:
            # ===============================================
            # PSEUDO CODE - Replace with actual JAKA SDK calls
            # ===============================================
            # # Disable robot
            # self.robot_handle.disable_robot()
            #
            # # Power off (optional - might want to keep powered for safety)
            # # self.robot_handle.power_off()
            #
            # # Logout
            # self.robot_handle.login_out()
            # ===============================================

            logging.info("Disconnected from JAKA robot")
            self.robot_handle = None
            self.is_connected = False

        except Exception as e:
            logging.error(f"Error during disconnect: {e}")
            self.is_connected = False

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        if self.is_connected:
            self.disconnect()
