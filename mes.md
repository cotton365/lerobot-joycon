# UR5 遥操代码定位（本仓库 lerobot 版本）

结论先说：

- 在当前仓库代码中，没有找到**UR5 机械臂本体通信/驱动**（例如 RTDE/URScript `servoj/servol`）相关的“UR5 专用遥操实现”。对 `UR5/ur5/rtde/servoj/urscript` 等关键字的检索结果，仅出现于**数据集名称/文档**（见下文“UR5 相关检索结果”）。
- 但是，本仓库确实存在你记得的那类“**固定频率循环（可设 `--fps 30`）+ 每次循环持续下发伺服目标**”的遥操主逻辑：它是对任意实现了 `teleop_step()` 的 `Robot` 通用的控制环；“伺服模式持续执行”在这里体现为每个循环都对 follower 关节写入 `Goal_Position`（见 `ManipulatorRobot.teleop_step()`）。

---

## 1) 遥操入口（CLI / 脚本层）

**位置**

- `/home/runner/work/lerobot-joycon/lerobot-joycon/lerobot/scripts/control_robot.py:22`：示例写明 `--fps 30`
- `/home/runner/work/lerobot-joycon/lerobot-joycon/lerobot/scripts/control_robot.py:175`：`teleoperate()` 入口，调用 `control_loop(...)`

**逻辑**

1. `python lerobot/scripts/control_robot.py teleoperate --fps 30`
2. 进入 `teleoperate(robot, fps=30, ...)`
3. 转到通用控制环 `control_loop(..., teleoperate=True, fps=30)`

**关键代码**

```py
# lerobot/scripts/control_robot.py:175
@safe_disconnect
def teleoperate(robot: Robot, fps: int | None = None, teleop_time_s: float | None = None, display_cameras: bool = False):
    control_loop(
        robot,
        control_time_s=teleop_time_s,
        fps=fps,
        teleoperate=True,
        display_cameras=display_cameras,
    )
```

---

## 2) 30Hz 控制环（while 循环 + busy_wait）

**位置**

- `/home/runner/work/lerobot-joycon/lerobot-joycon/lerobot/common/robot_devices/control_utils.py:228`：`control_loop()` 定义
- `/home/runner/work/lerobot-joycon/lerobot-joycon/lerobot/common/robot_devices/control_utils.py:260`：`while timestamp < control_time_s:`
- `/home/runner/work/lerobot-joycon/lerobot-joycon/lerobot/common/robot_devices/control_utils.py:286`：`busy_wait(1 / fps - dt_s)`（`fps=30` 时就是 ~33ms 一次）

**逻辑思路**

- 这个控制环既支持“遥操”（`teleoperate=True`），也支持“策略推理控制”（`policy != None`）。
- 遥操路径下，每次循环都调用 `robot.teleop_step(record_data=True)`，让机器人实现把“leader/控制器输入”转换成“follower 目标”，并返回可记录的数据。
- 频率控制由 `busy_wait` 实现：循环用时 `dt_s` 小于 `1/fps` 时会自旋等待补足，达到近似固定 30Hz。

**关键代码**

```py
# lerobot/common/robot_devices/control_utils.py:228
while timestamp < control_time_s:
    start_loop_t = time.perf_counter()

    if teleoperate:
        observation, action = robot.teleop_step(record_data=True)
    else:
        observation = robot.capture_observation()
        ...

    if fps is not None:
        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)   # fps=30 -> 0.033s
```

---

## 3) “伺服模式持续执行”的核心：每轮写 follower `Goal_Position`

> 如果你记得的“servo 模式持续执行”指的是“每周期持续发送目标给机器人控制器/伺服”，在本仓库里对应的就是 `teleop_step()` 里每轮写入 `Goal_Position` 的逻辑（对电机总线/舵机而言，这就是持续伺服跟随）。

**位置**

- `/home/runner/work/lerobot-joycon/lerobot-joycon/lerobot/common/robot_devices/robots/manipulator.py:522`：`ManipulatorRobot.teleop_step()`
- `/home/runner/work/lerobot-joycon/lerobot-joycon/lerobot/common/robot_devices/robots/manipulator.py:538`：读取控制器（JoyCon）命令路径
- `/home/runner/work/lerobot-joycon/lerobot-joycon/lerobot/common/robot_devices/robots/manipulator.py:560`：写 follower `Goal_Position`

**逻辑思路**

- `leader_pos[name]` 来自两种来源之一：
  - 有 `self.controllers`：从控制器（此仓库是 JoyCon）读取目标，得到关节目标角（度）并塞进 `leader_pos`
  - 否则：读取 leader 机械臂的 `Present_Position` 作为 `leader_pos`
- 对每个 follower 臂：将 `leader_pos` 作为 `goal_pos`，必要时做安全裁剪，然后下发到硬件 `Goal_Position`
- 在 JoyCon 场景还额外读 `Present_Load` 做过载保护（夹爪/全关节）

**关键代码**

```py
# lerobot/common/robot_devices/robots/manipulator.py:522
if self.controllers is not None:
    present_pos = self.follower_arms[name].read("Present_Position")
    controller_command[name], button_control, self.gripper_state[name] = self.controllers[name].get_command(present_pos)
    leader_pos[name] = torch.from_numpy(np.array(controller_command[name]))

for name in self.follower_arms:
    goal_pos = leader_pos[name]
    goal_pos = goal_pos.numpy().astype(np.int32)
    self.follower_arms[name].write("Goal_Position", goal_pos)
```

---

## 4) JoyCon 遥操输入如何变成关节目标（IK + 映射）

**位置**

- `/home/runner/work/lerobot-joycon/lerobot-joycon/lerobot/common/robot_devices/controllers/joycon_controller.py:75`：`JoyConController.get_command()`

**逻辑思路**

1. `JoyconRobotics.get_control()` 得到一个 `target_pose`（含 xyz + rpy 等）与 `gripper_state/button_control`
2. 对 `target_pose` 做 workspace 限幅（`glimit`）
3. 组装 `target_gpos`（位置 + 姿态），用 `lerobot_IK(...)` 求逆解得到关节
4. 返回 `joint_angles`（转成角度制并做符号修正）供 `teleop_step()` 下发

**关键代码**

```py
# lerobot/common/robot_devices/controllers/joycon_controller.py:75
target_pose, gripper_state, button_control = self.joyconrobotics.get_control()
...
target_gpos = np.array([x, y, z, roll, pitch, 0.0])
qpos_inv_mujoco, IK_success = lerobot_IK(fd_qpos_mucojo, target_gpos, robot=self.robot)
...
joint_angles = np.rad2deg(self.target_qpos)
return joint_angles, button_control, gripper_state
```

---

## 5) UR5 相关检索结果（仅数据集/元数据）

**位置**

- `/home/runner/work/lerobot-joycon/lerobot-joycon/lerobot/__init__.py:171`：`"lerobot/berkeley_autolab_ur5"`（数据集列表）
- `/home/runner/work/lerobot-joycon/lerobot-joycon/lerobot/common/datasets/v2/batch_convert_dataset_v1_to_v2.py:285`：`berkeley_autolab_ur5`（数据集转换元信息）

如果你期望的是“UR5 通过 RTDE/URScript 的 `servoj`（典型 125Hz/500Hz）或其它伺服接口，以 ~30Hz 的上层 loop 持续发指令”的遥操实现：这份代码在当前仓库中未出现，可能在你记忆的另一份 UR5 驱动/遥操工程里，或在上游 `huggingface/lerobot` 的其它分支/插件包（例如某个 `gym_*` 扩展或独立 driver）中。

