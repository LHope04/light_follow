# originbot_traffic_light

ROS 2 node that detects traffic-light state from a camera image and publishes an
`originbot_msgs/TrafficLight` message on `/traffic_light/state`.

This package also ships an **intersection action manager** that subscribes to the
detection output and drives the robot through the intersection once a stable,
high-confidence actionable signal (`LEFT / RIGHT / STRAIGHT`) is received.

**Publish semantics:** a state message is emitted on *every* processed image
frame.  When detection fails or confidence is below threshold the message carries
`state = UNKNOWN` and `confidence = 0.0`, so downstream nodes and tools like
`ros2 topic echo` / `ros2 topic hz` never hang waiting for a first message.

## Detection modes

The node supports two independent detection modes, selected by the `mode` parameter.

### `roi` mode (default)

Fixed absolute-pixel regions-of-interest (ROIs) are evaluated on the full image.
Three zones (left / mid / right) are tested for green and red using HSV masks.
This is the original behaviour and is unchanged.

**Normalised (ratio) ROI** is an optional enhancement to the ROI mode.  Instead
of specifying ROI coordinates in absolute pixels you may provide each coordinate
as a fraction of the current frame dimension (range `[0, 1]`).  This makes the
configuration resolution-independent.

Precedence rules per band:

1. If `*_w_ratio` **and** `*_h_ratio` are both `> 0` the ratio set is *active*.
2. The set is then validated: `x/y_ratio ∈ [0, 1]`, `w/h_ratio ∈ (0, 1]`.
3. When valid the node multiplies each ratio by the current frame dimension and
   clamps the result to the image bounds.
4. If the set is *inactive* (default `0.0`) **or** *invalid* (value out of range),
   the absolute pixel parameters (`*_x/y/w/h`) are used instead.  A one-time
   warning is logged when an active set fails validation.

Migration note: existing configurations do **not** need to change.  All ratio
parameters default to `0.0` so the node falls back to the absolute pixel ROIs
without any modification.

### `card` mode

Automatically detects a **printed traffic-light card** (white rounded rectangle
containing red arrows and one green arrow/circle) anywhere in the camera frame.
No manual ROI tuning is required.

Algorithm:
1. Convert frame to grayscale → Gaussian blur → Canny edge detection.
2. Find external contours, approximate each with `approxPolyDP`.
3. Keep 4-point polygons whose bounding-rect area exceeds `min_card_area` and
   whose aspect ratio is within `[card_aspect_ratio_min, card_aspect_ratio_max]`.
4. Select the largest qualifying candidate.
5. Apply a perspective transform to warp the card to a canonical
   `card_warp_width × card_warp_height` rectangle.
6. Split the warped card into three equal horizontal bands (left / middle / right)
   with an optional pixel margin (`band_margin_px`) on each side.
7. Compute the green-pixel ratio in each band using the same HSV thresholds as
   the ROI mode.
8. Determine the output state:
   - green mostly in left band → `LEFT`
   - green mostly in middle band → `STRAIGHT`
   - green mostly in right band → `RIGHT`
   - no band exceeds `min_green_ratio` → `STOP`
   - no valid card found → `UNKNOWN` (confidence 0.0)

## Temporal debounce

The node includes a built-in temporal debounce filter that prevents
single-frame glitches from changing the published state.

| Parameter | Default | Description |
|---|---|---|
| `debounce_frames` | `3` | Consecutive frames a candidate must appear before becoming the stable state |
| `debounce_unknown` | `true` | Require the same streak for `UNKNOWN`; set `false` to let `UNKNOWN` take effect immediately |

**How it works:**
- Each frame the detector computes a *candidate* state using the normal detection
  logic.
- If the candidate matches the current stable state the streak counter is reset.
- If the candidate *differs* from the stable state its streak counter is
  incremented.
- The stable state only switches when the streak counter reaches `debounce_frames`.
- The published `confidence` value reflects the detection confidence at the time
  the state last committed; it is *not* updated during a held streak.

**Tuning guidance:**
- `debounce_frames: 1` → effectively no debounce (immediate transitions, legacy
  behaviour).
- `debounce_frames: 3` (default) → suppresses most single-frame noise at typical
  camera rates (30 fps corresponds to ~100 ms hold).
- `debounce_frames: 5` → more conservative; useful when the camera frame rate is
  high or the environment is noisy.
- `debounce_unknown: false` → good when `UNKNOWN` means "lost sight of sign" and
  you want the downstream controller to react immediately.

## Quickstart

```bash
# Source your workspace
source /opt/ros/foxy/setup.bash
source ~/ws_originbot/install/setup.bash

# Run with default roi mode (image_topic defaults to /image_raw)
ros2 launch originbot_traffic_light traffic_light_detector.launch.py

# Run in card mode (override via launch argument)
ros2 launch originbot_traffic_light traffic_light_detector.launch.py mode:=card

# Enable debug images in card mode
ros2 launch originbot_traffic_light traffic_light_detector.launch.py mode:=card debug:=true

# Override the image topic (e.g. when your camera publishes on a different topic)
ros2 launch originbot_traffic_light traffic_light_detector.launch.py image_topic:=/image_raw
```

To switch permanently, edit `config/params.yaml` and change `mode: roi` to
`mode: card`.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_topic` | string | `/image_raw` | Input image topic (override with `image_topic:=<topic>`) |
| `output_topic` | string | `/traffic_light/state` | Output state topic |
| `mode` | string | `roi` | Detection mode: `roi` or `card` |
| `left_x/y/w/h` | int | (see yaml) | Left ROI absolute pixels (roi mode) |
| `mid_x/y/w/h` | int | (see yaml) | Mid ROI absolute pixels (roi mode) |
| `right_x/y/w/h` | int | (see yaml) | Right ROI absolute pixels (roi mode) |
| `left_x/y/w/h_ratio` | float | `0.0` | Left ROI normalised [0,1]; w/h in (0,1]; 0 = disabled |
| `mid_x/y/w/h_ratio` | float | `0.0` | Mid ROI normalised; same rules |
| `right_x/y/w/h_ratio` | float | `0.0` | Right ROI normalised; same rules |
| `green_h/s/v_low/high` | int | (see yaml) | HSV green thresholds (both modes) |
| `red1_*/red2_*` | int | (see yaml) | HSV red thresholds (roi mode) |
| `min_green_ratio` | float | `0.01` | Min green pixel ratio to count as green |
| `min_red_ratio` | float | `0.01` | Min red pixel ratio to count as red (roi mode) |
| `card_warp_width` | int | `320` | Canonical warp width in pixels (card mode) |
| `card_warp_height` | int | `120` | Canonical warp height in pixels (card mode) |
| `min_card_area` | int | `5000` | Min contour area (px²) to consider as card |
| `card_aspect_ratio_min` | float | `1.5` | Min card width/height ratio |
| `card_aspect_ratio_max` | float | `6.0` | Max card width/height ratio |
| `band_margin_px` | int | `5` | Horizontal margin inside each band (card mode) |
| `debug` | bool | `false` | Publish annotated debug image (card mode) |
| `debug_topic` | string | `/traffic_light/debug` | Debug image topic |
| `publish_unknown` | bool | `true` | Publish `UNKNOWN` state when no confident detection; set `false` to revert to legacy publish-on-detection-only behaviour |
| `debounce_frames` | int | `3` | Consecutive frames required before switching stable state |
| `debounce_unknown` | bool | `true` | Apply debounce to `UNKNOWN` transitions; `false` = immediate UNKNOWN |

## Published topics

| Topic | Type | Description |
|---|---|---|
| `/traffic_light/state` | `originbot_msgs/TrafficLight` | Detected state + confidence (published every frame; `UNKNOWN` / `0.0` when no detection) |
| `/detect/traffic_decision` | `originbot_msgs/TrafficDecision` | Same detection data republished as `TrafficDecision`; consumed by `intersection_action_manager` |
| `/traffic_light/debug` | `sensor_msgs/Image` | Debug image (card mode, `debug:=true`) |

## Subscribed topics

| Topic | Type | Description |
|---|---|---|
| `/image_raw` | `sensor_msgs/Image` | Input camera frame (default; OriginBot camera publishes here) |

## Manual verification

After starting the robot with camera enabled and launching the detector:

```bash
# Confirm messages arrive at a non-zero rate
ros2 topic hz /traffic_light/state

# Inspect messages – should show UNKNOWN when no card/sign is visible
ros2 topic echo /traffic_light/state --qos-reliability best_effort
```

Expected output when no traffic-light card is in frame:
```
header: ...
state: 0        # UNKNOWN
confidence: 0.0
```

---

## Intersection Action Manager

`intersection_action_manager` is a companion node that closes the loop between
detection and robot movement.  It subscribes to `/detect/traffic_decision`,
applies a confidence threshold and an additional debounce filter, then issues a
single `/control/moving/state` (`MovingParam`) command when the conditions are
met.

### Message types (originbot_msgs)

| Message | Fields | Description |
|---|---|---|
| `TrafficDecision` | `header`, `uint8 state` (UNKNOWN/STOP/STRAIGHT/LEFT/RIGHT), `float32 confidence` | Traffic-light decision from the detector |
| `MovingParam` | `header`, `uint8 moving_type` (UNKNOWN/STOP/FORWARD/LEFT/RIGHT), `float32 moving_value` | Movement command sent to the robot executor |

### State machine

```
IDLE
 │  first /detect/traffic_decision message
 ▼
APPROACH
 │  (immediate pass-through — no distance sensor in this package)
 ▼
WAIT_PERMISSION
 │  stable_frames consecutive frames: same LEFT/RIGHT/STRAIGHT, confidence ≥ conf_th
 ▼
EXECUTE_MANEUVER          (one MovingParam published here)
 │  /control/moving/complete (Bool, data=True)  OR  execute_timeout exceeded
 ▼
DONE
 │  done_reset_delay elapsed
 ▼
IDLE
```

**STOP / UNKNOWN behaviour:** the node stays in `WAIT_PERMISSION` without
issuing any movement command.  `/control/max_vel` is held at `max_vel_approach`
(slow) so the robot idles safely.

**Timeout protection:** if `wait_timeout` elapses the stable-frame counter is
reset and `/control/max_vel` is set to `0.0`.  If `execute_timeout` elapses the
node transitions to `DONE` and logs an error.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `conf_th` | `0.6` | Minimum confidence; below this the detection is treated as UNKNOWN |
| `stable_frames` | `3` | Consecutive same-state frames required before issuing movement |
| `wait_timeout` | `30.0` | Seconds before resetting the frame counter in WAIT_PERMISSION |
| `execute_timeout` | `10.0` | Seconds before forcibly exiting EXECUTE_MANEUVER |
| `done_reset_delay` | `3.0` | Seconds in DONE before returning to IDLE |
| `max_vel_approach` | `0.1` | `/control/max_vel` (m/s) during APPROACH / WAIT_PERMISSION |
| `max_vel_execute` | `0.15` | `/control/max_vel` (m/s) during EXECUTE_MANEUVER |

### Topics (intersection_action_manager)

**Subscribed:**

| Topic | Type | Description |
|---|---|---|
| `/detect/traffic_decision` | `originbot_msgs/TrafficDecision` | Traffic-light decision from the detector |
| `/control/moving/complete` | `std_msgs/Bool` | Movement completion feedback from the executor |

**Published:**

| Topic | Type | Description |
|---|---|---|
| `/control/moving/state` | `originbot_msgs/MovingParam` | Movement command (one per EXECUTE_MANEUVER phase) |
| `/control/max_vel` | `std_msgs/Float64` | Speed ceiling for the approach / wait / execute phases |

### Quickstart with intersection pipeline

```bash
# Source workspace
source /opt/ros/humble/setup.bash
source ~/ws_originbot/install/setup.bash

# Launch detector + action manager together
ros2 launch originbot_traffic_light intersection_action.launch.py

# Launch in card mode
ros2 launch originbot_traffic_light intersection_action.launch.py mode:=card
```

### Minimum debugging steps

```bash
# 1. Confirm the detector publishes on both old and new topics
ros2 topic hz /traffic_light/state
ros2 topic hz /detect/traffic_decision

# 2. Inspect the traffic decision stream
ros2 topic echo /detect/traffic_decision

# 3. Simulate a LEFT signal (stable_frames=3 consecutive frames)
#    Replace the stamp if your middleware requires it.
for i in 1 2 3; do
  ros2 topic pub --once /detect/traffic_decision originbot_msgs/msg/TrafficDecision \
    "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, state: 3, confidence: 0.85}"
  sleep 0.1
done

# 4. Verify intersection_action_manager published a movement command
ros2 topic echo /control/moving/state --once

# 5. Simulate maneuver completion
ros2 topic pub --once /control/moving/complete std_msgs/msg/Bool "{data: true}"

# 6. Confirm no further /control/moving/state messages after completion
ros2 topic echo /control/moving/state
```

**Checking STOP / low-confidence behaviour:**

```bash
# STOP signal – manager must NOT publish a movement command
ros2 topic pub --once /detect/traffic_decision originbot_msgs/msg/TrafficDecision \
  "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, state: 1, confidence: 0.9}"

# Low-confidence signal – treated as UNKNOWN, no movement
ros2 topic pub --once /detect/traffic_decision originbot_msgs/msg/TrafficDecision \
  "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, state: 2, confidence: 0.3}"
```

## Intersection Action Manager（更新行为说明）

本节点根据 `/detect/traffic_decision` 执行十字路口动作，当前策略：

- `STOP(1)`：发布一次 `MOVING_STOP`，并将 `/control/max_vel` 置为 `0.0`（保持停车）
- `LEFT(3)` / `RIGHT(4)`：进入转向执行，退出条件采用 **角度限制为主**：
  1. （可选）达到最小转角后，重新识别到引导线，提前结束转向；
  2. 达到最大转角上限，强制结束；
  3. 超时仅作兜底，避免死锁。

### 新增参数（intersection_params.yaml）

- `turn_target_angle_deg`：目标转角（默认 90）
- `turn_max_angle_deg`：转角硬上限（默认 105）
- `turn_min_angle_deg_for_line_exit`：允许“识别到线即退出”的最小转角（默认 55）
- `right_turn_negative`：右转角符号配置
- `turn_use_line_reacquire`：是否启用线重捕获提前退出
- `line_reacquire_topic`：线重捕获信号 topic（`std_msgs/Bool`）
- `line_reacquire_true_frames`：连续为 true 的帧数阈值
- `odom_topic`：里程计 topic（用于 yaw 角估计）

### 调试建议

1. 先验证 STOP：
```bash
ros2 topic pub --once /detect/traffic_decision originbot_msgs/msg/TrafficDecision \
"{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, state: 1, confidence: 0.9}"
```

2. 验证 LEFT：
```bash
for i in 1 2 3; do
  ros2 topic pub --once /detect/traffic_decision originbot_msgs/msg/TrafficDecision \
  "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, state: 3, confidence: 0.9}"
done
```

3. 验证 RIGHT：
```bash
for i in 1 2 3; do
  ros2 topic pub --once /detect/traffic_decision originbot_msgs/msg/TrafficDecision \
  "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, state: 4, confidence: 0.9}"
done
```
