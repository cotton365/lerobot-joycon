import enum
import logging
import math
import time
import traceback
from copy import deepcopy

import numpy as np
import tqdm

from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc

import jkrc


PROTOCOL_VERSION = 0
BAUDRATE = 1_000_000
TIMEOUT_MS = 1000

MAX_ID_RANGE = 252

# The following bounds define the lower and upper joints range (after calibration).
# For joints in degree (i.e. revolute joints), their nominal range is [-180, 180] degrees
# which corresponds to a half rotation on the left and half rotation on the right.
# Some joints might require higher range, so we allow up to [-270, 270] degrees until
# an error is raised.
LOWER_BOUND_DEGREE = -270
UPPER_BOUND_DEGREE = 270
# For joints in percentage (i.e. joints that move linearly like the prismatic joint of a gripper),
# their nominal range is [0, 100] %. For instance, for Aloha gripper, 0% is fully
# closed, and 100% is fully open. To account for slight calibration issue, we allow up to
# [-10, 110] until an error is raised.
LOWER_BOUND_LINEAR = -10
UPPER_BOUND_LINEAR = 110

HALF_TURN_DEGREE = 180

CALIBRATION_REQUIRED = ["Goal_Position", "Present_Position"]


class TorqueMode(enum.Enum):
    ENABLED = 1
    DISABLED = 0


class DriveMode(enum.Enum):
    NON_INVERTED = 0
    INVERTED = 1


class CalibrationMode(enum.Enum):
    # Joints with rotational motions are expressed in degrees in nominal range of [-180, 180]
    DEGREE = 0
    # Joints with linear motions (like gripper of Aloha) are experessed in nominal range of [0, 100]
    LINEAR = 1


class JointOutOfRangeError(Exception):
    def __init__(self, message="Joint is out of range"):
        self.message = message
        super().__init__(self.message)


class JakaMotorsBus:
    """

    ```bash
    python lerobot/scripts/find_motors_bus_port.py
    >>> Finding all available ports for the MotorsBus.
    >>> ['/dev/tty.usbmodem575E0032081', '/dev/tty.usbmodem575E0031751']
    >>> The port of this JakaMotorsBus is /dev/tty.usbmodem575E0031751.
    >>> Reconnect the usb cable.
    ```

    Example of usage for 1 motor connected to the bus:
    ```python
    motor_name = "gripper"
    motor_index = 6
    motor_model = "sts3215"

    motors_bus = JakaMotorsBus(
        port="/dev/tty.usbmodem575E0031751",
        motors={motor_name: (motor_index, motor_model)},
    )
    motors_bus.connect()

    position = motors_bus.read("Present_Position")

    # move from a few motor steps as an example
    few_steps = 30
    motors_bus.write("Goal_Position", position + few_steps)

    # when done, consider disconnecting
    motors_bus.disconnect()
    ```
    """

    def __init__(
        self,
        ip_address: str,
        motors: dict[str, tuple[int, str]],
        mock=False,
    ):
        self.robot = None
        self.ip_address = ip_address
        self.motors = motors
        self.mock = mock

        self.calibration = None
        self.is_connected = False
        self.logs = {}

        self.track_positions = {}

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"JakaMotorsBus({self.ip_address}) is already connected. Do not call `motors_bus.connect()` twice."
            )
        self.robot = jkrc.RC(self.ip_address)
        self.robot.login()
        self.is_connected = True

    def reconnect(self):
        self.robot.logout()
        self.robot.login()
        self.is_connected = True

    def are_motors_configured(self):
        # We assume the JAKA industrial robot is always configured and has 6 joints
        return True

    def find_motor_indices(self, possible_ids=None, num_retry=2):
        # Industrial arm has fixed joints, e.g., 6 joints indexed 0-5
        return list(range(len(self.motors)))

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def set_calibration(self, calibration: dict[str, list]):
        self.calibration = calibration

    def apply_calibration_autocorrect(self, values: np.ndarray | list, motor_names: list[str] | None):
        """This function apply the calibration, automatically detects out of range errors for motors values and attempt to correct.

        For more info, see docstring of `apply_calibration` and `autocorrect_calibration`.
        """
        try:
            values = self.apply_calibration(values, motor_names)
        except JointOutOfRangeError as e:
            print(e)
            self.autocorrect_calibration(values, motor_names)
            values = self.apply_calibration(values, motor_names)
        return values

    def apply_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """Convert from unsigned int32 joint position range [0, 2**32[ to the universal float32 nominal degree range ]-180.0, 180.0[ with
        a "zero position" at 0 degree.

        Note: We say "nominal degree range" since the motors can take values outside this range. For instance, 190 degrees, if the motor
        rotate more than a half a turn from the zero position. However, most motors can't rotate more than 180 degrees and will stay in this range.

        Joints values are original in [0, 2**32] (unsigned int32). Each motor are expected to complete a full rotation
        when given a goal position that is + or - their resolution. For instance, feetech xl330-m077 have a resolution of 4096, and
        at any position in their original range, let's say the position 56734, they complete a full rotation clockwise by moving to 60830,
        or anticlockwise by moving to 52638. The position in the original range is arbitrary and might change a lot between each motor.
        To harmonize between motors of the same model, different robots, or even models of different brands, we propose to work
        in the centered nominal degree range ]-180, 180[.
        """
        if motor_names is None:
            motor_names = self.motor_names

        # Convert from unsigned int32 original range [0, 2**32] to signed float32 range
        values = values.astype(np.float32)

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                drive_mode = self.calibration["drive_mode"][calib_idx]
                homing_offset = self.calibration["homing_offset"][calib_idx]
                _, model = self.motors[name]
                resolution = self.model_resolution[model]

                # Update direction of rotation of the motor to match between leader and follower.
                # In fact, the motor of the leader for a given joint can be assembled in an
                # opposite direction in term of rotation than the motor of the follower on the same joint.
                if drive_mode:
                    values[i] *= -1

                # Convert from range [-2**31, 2**31[ to
                # nominal range ]-resolution, resolution[ (e.g. ]-2048, 2048[)
                values[i] += homing_offset

                # Convert from range ]-resolution, resolution[ to
                # universal float32 centered degree range ]-180, 180[
                values[i] = values[i] / (resolution // 2) * HALF_TURN_DEGREE

                if (values[i] < LOWER_BOUND_DEGREE) or (values[i] > UPPER_BOUND_DEGREE):
                    raise JointOutOfRangeError(
                        f"Wrong motor position range detected for {name}. "
                        f"Expected to be in nominal range of [-{HALF_TURN_DEGREE}, {HALF_TURN_DEGREE}] degrees (a full rotation), "
                        f"with a maximum range of [{LOWER_BOUND_DEGREE}, {UPPER_BOUND_DEGREE}] degrees to account for joints that can rotate a bit more, "
                        f"but present value is {values[i]} degree. "
                        "This might be due to a cable connection issue creating an artificial 360 degrees jump in motor values. "
                        "You need to recalibrate by running: `python lerobot/scripts/control_robot.py calibrate`"
                    )

            elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]

                # Rescale the present position to a nominal range [0, 100] %,
                # useful for joints with linear motions like Aloha gripper
                values[i] = (values[i] - start_pos) / (end_pos - start_pos) * 100

                if (values[i] < LOWER_BOUND_LINEAR) or (values[i] > UPPER_BOUND_LINEAR):
                    raise JointOutOfRangeError(
                        f"Wrong motor position range detected for {name}. "
                        f"Expected to be in nominal range of [0, 100] % (a full linear translation), "
                        f"with a maximum range of [{LOWER_BOUND_LINEAR}, {UPPER_BOUND_LINEAR}] % to account for some imprecision during calibration, "
                        f"but present value is {values[i]} %. "
                        "This might be due to a cable connection issue creating an artificial jump in motor values. "
                        "You need to recalibrate by running: `python lerobot/scripts/control_robot.py calibrate`"
                    )

        return values

    def autocorrect_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """This function automatically detects issues with values of motors after calibration, and correct for these issues.

        Some motors might have values outside of expected maximum bounds after calibration.
        For instance, for a joint in degree, its value can be outside [-270, 270] degrees, which is totally unexpected given
        a nominal range of [-180, 180] degrees, which represents half a turn to the left or right starting from zero position.

        Known issues:
        #1: Motor value randomly shifts of a full turn, caused by hardware/connection errors.
        #2: Motor internal homing offset is shifted of a full turn, caused by using default calibration (e.g Aloha).
        #3: motor internal homing offset is shifted of less or more than a full turn, caused by using default calibration
            or by human error during manual calibration.

        Issues #1 and #2 can be solved by shifting the calibration homing offset by a full turn.
        Issue #3 will be visually detected by user and potentially captured by the safety feature `max_relative_target`,
        that will slow down the motor, raise an error asking to recalibrate. Manual recalibrating will solve the issue.

        Note: A full turn corresponds to 360 degrees but also to 4096 steps for a motor resolution of 4096.
        """
        if motor_names is None:
            motor_names = self.motor_names

        # Convert from unsigned int32 original range [0, 2**32] to signed float32 range
        values = values.astype(np.float32)

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                drive_mode = self.calibration["drive_mode"][calib_idx]
                homing_offset = self.calibration["homing_offset"][calib_idx]
                _, model = self.motors[name]
                resolution = self.model_resolution[model]

                if drive_mode:
                    values[i] *= -1

                # Convert from initial range to range [-180, 180] degrees
                calib_val = (values[i] + homing_offset) / (resolution // 2) * HALF_TURN_DEGREE
                in_range = (calib_val > LOWER_BOUND_DEGREE) and (calib_val < UPPER_BOUND_DEGREE)

                # Solve this inequality to find the factor to shift the range into [-180, 180] degrees
                # values[i] = (values[i] + homing_offset + resolution * factor) / (resolution // 2) * HALF_TURN_DEGREE
                # - HALF_TURN_DEGREE <= (values[i] + homing_offset + resolution * factor) / (resolution // 2) * HALF_TURN_DEGREE <= HALF_TURN_DEGREE
                # (- HALF_TURN_DEGREE / HALF_TURN_DEGREE * (resolution // 2) - values[i] - homing_offset) / resolution <= factor <= (HALF_TURN_DEGREE / 180 * (resolution // 2) - values[i] - homing_offset) / resolution
                low_factor = (
                    -HALF_TURN_DEGREE / HALF_TURN_DEGREE * (resolution // 2) - values[i] - homing_offset
                ) / resolution
                upp_factor = (
                    HALF_TURN_DEGREE / HALF_TURN_DEGREE * (resolution // 2) - values[i] - homing_offset
                ) / resolution

            elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]

                # Convert from initial range to range [0, 100] in %
                calib_val = (values[i] - start_pos) / (end_pos - start_pos) * 100
                in_range = (calib_val > LOWER_BOUND_LINEAR) and (calib_val < UPPER_BOUND_LINEAR)

                # Solve this inequality to find the factor to shift the range into [0, 100] %
                # values[i] = (values[i] - start_pos + resolution * factor) / (end_pos + resolution * factor - start_pos - resolution * factor) * 100
                # values[i] = (values[i] - start_pos + resolution * factor) / (end_pos - start_pos) * 100
                # 0 <= (values[i] - start_pos + resolution * factor) / (end_pos - start_pos) * 100 <= 100
                # (start_pos - values[i]) / resolution <= factor <= (end_pos - values[i]) / resolution
                low_factor = (start_pos - values[i]) / resolution
                upp_factor = (end_pos - values[i]) / resolution

            if not in_range:
                # Get first integer between the two bounds
                if low_factor < upp_factor:
                    factor = math.ceil(low_factor)

                    if factor > upp_factor:
                        raise ValueError(f"No integer found between bounds [{low_factor=}, {upp_factor=}]")
                else:
                    factor = math.ceil(upp_factor)

                    if factor > low_factor:
                        raise ValueError(f"No integer found between bounds [{low_factor=}, {upp_factor=}]")

                if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                    out_of_range_str = f"{LOWER_BOUND_DEGREE} < {calib_val} < {UPPER_BOUND_DEGREE} degrees"
                    in_range_str = f"{LOWER_BOUND_DEGREE} < {calib_val} < {UPPER_BOUND_DEGREE} degrees"
                elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                    out_of_range_str = f"{LOWER_BOUND_LINEAR} < {calib_val} < {UPPER_BOUND_LINEAR} %"
                    in_range_str = f"{LOWER_BOUND_LINEAR} < {calib_val} < {UPPER_BOUND_LINEAR} %"

                logging.warning(
                    f"Auto-correct calibration of motor '{name}' by shifting value by {abs(factor)} full turns, "
                    f"from '{out_of_range_str}' to '{in_range_str}'."
                )

                # A full turn corresponds to 360 degrees but also to 4096 steps for a motor resolution of 4096.
                self.calibration["homing_offset"][calib_idx] += resolution * factor

    def revert_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """Inverse of `apply_calibration`."""
        if motor_names is None:
            motor_names = self.motor_names

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                drive_mode = self.calibration["drive_mode"][calib_idx]
                homing_offset = self.calibration["homing_offset"][calib_idx]
                _, model = self.motors[name]
                resolution = self.model_resolution[model]

                # Convert from nominal 0-centered degree range [-180, 180] to
                # 0-centered resolution range (e.g. [-2048, 2048] for resolution=4096)
                values[i] = values[i] / HALF_TURN_DEGREE * (resolution // 2)

                # Substract the homing offsets to come back to actual motor range of values
                # which can be arbitrary.
                values[i] -= homing_offset

                # Remove drive mode, which is the rotation direction of the motor, to come back to
                # actual motor rotation direction which can be arbitrary.
                if drive_mode:
                    values[i] *= -1

            elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]

                # Convert from nominal lnear range of [0, 100] % to
                # actual motor range of values which can be arbitrary.
                values[i] = values[i] / 100 * (end_pos - start_pos) + start_pos

        values = np.round(values).astype(np.int32)
        return values

    def avoid_rotation_reset(self, values, motor_names, data_name):
        return values

    def read(self, data_name, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"JakaMotorsBus({self.ip_address}) is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()

        if motor_names is None:
            motor_names = self.motor_names

        if isinstance(motor_names, str):
            motor_names = [motor_names]

        if data_name == "Present_Position":
            ret, joint_pos = self.robot.get_joint_position()
            if ret != 0:
                raise ConnectionError(f"Failed to read joint positions from JAKA robot. Error code: {ret}")

            # `joint_pos` typically holds 6 joint angles
            values = []
            for name in motor_names:
                # Motor index from the kwargs
                motor_idx, _ = self.motors[name]
                # Fallback to appending 0 if idx out of range
                if motor_idx < len(joint_pos):
                    # Usually JAKA returns radians, we assume the conversion to steps logic
                    # expects values compatible with Lerobot's general mapping, or we just pass the values directly 
                    # and handle conversion in `apply_calibration`
                    values.append(joint_pos[motor_idx])
                else:
                    values.append(0.0)

            values = np.array(values)
        else:
            # Handle other fields if necessary
            values = np.zeros(len(motor_names))

        if data_name in CALIBRATION_REQUIRED and self.calibration is not None:
            values = self.apply_calibration_autocorrect(values, motor_names)

        # log the number of seconds it took to read the data from the motors
        # delta_ts_name = get_log_name("delta_timestamp_s", "read", data_name, motor_names)
        # self.logs[delta_ts_name] = time.perf_counter() - start_time

        # log the utc time at which the data was received
        # ts_utc_name = get_log_name("timestamp_utc", "read", data_name, motor_names)
        # self.logs[ts_utc_name] = capture_timestamp_utc()

        return values

    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"JakaMotorsBus({self.ip_address}) is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()

        if motor_names is None:
            motor_names = self.motor_names

        if isinstance(motor_names, str):
            motor_names = [motor_names]

        if isinstance(values, (int, float, np.integer)):
            values = [float(values)] * len(motor_names)

        values = np.array(values)

        if data_name in CALIBRATION_REQUIRED and self.calibration is not None:
            values = self.revert_calibration(values, motor_names)

        # Goal_Position handling
        if data_name == "Goal_Position":
            ret, current_pos = self.robot.get_joint_position()
            if ret != 0:
                raise ConnectionError(f"Failed to read intial joint positions. Error code: {ret}")

            # Assign new target pos
            target_pos = list(current_pos)
            for i, name in enumerate(motor_names):
                motor_idx, _ = self.motors[name]
                if motor_idx < len(target_pos):
                    target_pos[motor_idx] = values[i]

            # Typically joint_move(joint_pos, move_mode, is_block, speed)
            # Use appropriate parameters based on your configuration
            ret = self.robot.joint_move(target_pos, 0, False, 10.0)
            if ret != 0:
                raise ConnectionError(f"Failed to write joint positions to JAKA robot. Error code: {ret}")

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"JakaMotorsBus({self.ip_address}) is not connected. Try running `motors_bus.connect()` first."
            )

        if self.robot is not None:
            self.robot.logout()
            self.robot = None
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
