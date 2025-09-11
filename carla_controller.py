# ===========================================
# 文件: carla_controller.py
# ===========================================
import carla
import math

TIME_STEP = 0.05
TARGET_SPEED_KMH = 30
LOOKAHEAD_DIST = 5.0
KP_SPEED = 0.1
LANE_CHANGE_DISTANCE = 30.0
WAYPOINT_INTERVAL = 2.0

class LaneKeepAndChangeController:
    """
    简易车道保持 + 变道(区分加速/减速) + 加减速 + 刹车示例。
    run_step() 每帧返回 (control, signals)，其中 signals 用于标记各动作完成情况。
    """
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.map = self.world.get_map()

        # 状态: LaneKeep, LaneChangeAccel, LaneChangeDecel
        self.state = "LaneKeep"

        self.target_waypoints = []
        self.current_wp_idx = 0

        # 额外控制量
        self.extra_throttle = 0.0
        self.extra_brake = 0.0

    def run_step(self):
        """
        返回 (control, signals)
        signals = {
            "lane_change_done": bool,
            "acceleration_done": bool,
            "deceleration_done": bool,
            "brake_done": bool,
            "release_brake_done": bool,
        }
        """
        control = carla.VehicleControl()

        # 1. 用于标记动作完成情况
        signals = {
            "lane_change_done": False,
            "acceleration_done": False,
            "deceleration_done": False,
            "brake_done": False,
            "release_brake_done": False,
        }

        # 2. 纵向控制：基础PID + 状态修正 + extra_throttle / extra_brake
        vel = self.vehicle.get_velocity()
        speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        speed_kmh = speed_ms * 3.6
        speed_err = TARGET_SPEED_KMH - speed_kmh

        # 基本PID油门
        throttle = KP_SPEED * speed_err

        # 根据不同状态增加/减少油门
        if self.state == "LaneChangeAccel":
            throttle += 0.2
        elif self.state == "LaneChangeDecel":
            throttle -= 0.2

        # 叠加手动API提供的extra_throttle / extra_brake
        throttle += self.extra_throttle
        brake = self.extra_brake

        # 边界处理
        if throttle > 1.0:
            throttle = 1.0
        elif throttle < 0.0:
            # 如果油门<0，则转化为刹车
            brake = max(brake, -throttle)
            throttle = 0.0
            if brake > 1.0:
                brake = 1.0

        control.throttle = throttle
        control.brake = brake

        # 3. 横向控制：根据状态决定车道保持 or 变道
        if self.state == "LaneKeep":
            steer = self._lane_keep_steer()
        elif self.state in ["LaneChangeAccel", "LaneChangeDecel"]:
            steer, done = self._lane_change_steer()
            if done:
                # 变道完成 => 回到LaneKeep
                self.state = "LaneKeep"
                self.target_waypoints.clear()
                self.current_wp_idx = 0
                signals["lane_change_done"] = True
                steer = 0.0
        else:
            steer = 0.0

        steer = max(-1.0, min(1.0, steer))
        control.steer = steer

        # 4. 判断加速/减速/刹车是否到达边界
        if self.extra_throttle == 1.0:
            signals["acceleration_done"] = True
        if self.extra_throttle == -1.0:
            signals["deceleration_done"] = True
        if self.extra_brake == 1.0:
            signals["brake_done"] = True
        if self.extra_brake == 0.0:
            signals["release_brake_done"] = True

        return control, signals

    def _lane_keep_steer(self):
        transform = self.vehicle.get_transform()
        loc = transform.location
        yaw_rad = math.radians(transform.rotation.yaw)
        current_wp = self.map.get_waypoint(loc, lane_type=carla.LaneType.Driving)
        if not current_wp:
            return 0.0

        dist_accum = 0.0
        step = 1.0
        temp_wp = current_wp
        while dist_accum < LOOKAHEAD_DIST:
            nxt = temp_wp.next(step)
            if not nxt:
                break
            temp_wp = nxt[0]
            dist_accum += step

        tx = temp_wp.transform.location.x
        ty = temp_wp.transform.location.y
        dx = tx - loc.x
        dy = ty - loc.y
        fx = dx * math.cos(-yaw_rad) - dy * math.sin(-yaw_rad)
        fy = dx * math.sin(-yaw_rad) + dy * math.cos(-yaw_rad)
        heading_err = math.atan2(fy, fx)
        return heading_err

    def _lane_change_steer(self):
        transform = self.vehicle.get_transform()
        loc = transform.location
        yaw_rad = math.radians(transform.rotation.yaw)

        # 若已到达最后一个路点 => done
        if self.current_wp_idx >= len(self.target_waypoints):
            return (0.0, True)

        # 找最近的目标路点
        min_dist = 1e9
        closest_idx = self.current_wp_idx
        for i in range(self.current_wp_idx, len(self.target_waypoints)):
            wp = self.target_waypoints[i]
            dx = wp.transform.location.x - loc.x
            dy = wp.transform.location.y - loc.y
            dsq = dx*dx + dy*dy
            if dsq < min_dist:
                min_dist = dsq
                closest_idx = i
        self.current_wp_idx = closest_idx

        # 若离最后一个路点足够近 => done
        if self.current_wp_idx >= len(self.target_waypoints) - 1:
            return (0.0, True)

        # 向前查看一段距离
        look_idx = self.current_wp_idx
        dist_accum = 0.0
        tmp_wp = self.target_waypoints[look_idx]
        while look_idx < len(self.target_waypoints) - 1 and dist_accum < LOOKAHEAD_DIST:
            nxt_wp = self.target_waypoints[look_idx + 1]
            seg = tmp_wp.transform.location.distance(nxt_wp.transform.location)
            dist_accum += seg
            look_idx += 1
            tmp_wp = nxt_wp

        target_wp = self.target_waypoints[look_idx]
        tx = target_wp.transform.location.x
        ty = target_wp.transform.location.y
        dx = tx - loc.x
        dy = ty - loc.y
        fx = dx * math.cos(-yaw_rad) - dy * math.sin(-yaw_rad)
        fy = dx * math.sin(-yaw_rad) + dy * math.cos(-yaw_rad)
        heading_err = math.atan2(fy, fx)
        return (heading_err, False)

    # ========== 对外API：变道、加减速、刹车等 ==========

    def request_lane_change_accel(self, direction:str, distance=LANE_CHANGE_DISTANCE):
        """请求加速变道。若请求被接受则返回 True；若正在变道中则返回 False。"""
        if self.state in ["LaneChangeAccel", "LaneChangeDecel"]:
            return False

        side_wp = self._get_side_waypoint(direction)
        if not side_wp:
            return False

        wps = self._gen_lanechange_wps(side_wp, distance, WAYPOINT_INTERVAL)
        if len(wps) < 2:
            print("[WARN] 变道Waypoints不足")
            return False

        self.target_waypoints = wps
        self.current_wp_idx = 0
        self.state = "LaneChangeAccel"
        # print(f"[INFO] 开始向 {direction} 加速变道, lane_id={side_wp.lane_id}")
        return True

    def request_lane_change_decel(self, direction:str, distance=LANE_CHANGE_DISTANCE):
        """请求减速变道。若请求被接受则返回 True；若正在变道中则返回 False。"""
        if self.state in ["LaneChangeAccel", "LaneChangeDecel"]:
            return False

        side_wp = self._get_side_waypoint(direction)
        if not side_wp:
            return False

        wps = self._gen_lanechange_wps(side_wp, distance, WAYPOINT_INTERVAL)
        if len(wps) < 2:
            print("[WARN] 变道Waypoints不足")
            return False

        self.target_waypoints = wps
        self.current_wp_idx = 0
        self.state = "LaneChangeDecel"
        # print(f"[INFO] 开始向 {direction} 减速变道, lane_id={side_wp.lane_id}")
        return True

    def _get_side_waypoint(self, direction: str):
        """
        封装获取左右车道的逻辑，若不可行则返回None。
        """
        loc = self.vehicle.get_location()
        current_wp = self.map.get_waypoint(loc, lane_type=carla.LaneType.Driving)
        if not current_wp:
            return None

        if direction == "left":
            side_wp = current_wp.get_left_lane()
        else:
            side_wp = current_wp.get_right_lane()

        if side_wp is None:
            return None
        if side_wp.lane_type != carla.LaneType.Driving:
            return None

        # 如果 lane_id 异号 => 对向车道 => 放弃
        if side_wp.lane_id * current_wp.lane_id <= 0:
            return None

        return side_wp

    def _gen_lanechange_wps(self, start_wp, dist, interval):
        wps = []
        traveled = 0.0
        cur_wp = start_wp
        wps.append(cur_wp)

        while traveled < dist:
            nxt = cur_wp.next(interval)
            if not nxt:
                break
            cur_wp = nxt[0]
            wps.append(cur_wp)
            traveled += interval

        return wps

    def accelerate(self, val=0.1):
        """
        增加额外油门；可返回是否已到达油门上限 (True/False)
        """
        self.extra_throttle += val
        if self.extra_throttle > 1.0:
            self.extra_throttle = 1.0
        return (self.extra_throttle >= 1.0)

    def decelerate(self, val=0.1):
        """
        减少额外油门；可返回是否已到达油门下限 (True/False)
        """
        self.extra_throttle -= val
        if self.extra_throttle < -1.0:
            self.extra_throttle = -1.0
        return (self.extra_throttle <= -1.0)

    def brake(self, val=0.1):
        """
        增大刹车；可返回是否已刹车到底 (True/False)
        """
        self.extra_brake += val
        if self.extra_brake > 1.0:
            self.extra_brake = 1.0
        return (self.extra_brake >= 1.0)

    def release_brake(self, val=0.1):
        """
        减小刹车；可返回是否已完全松开 (True/False)
        """
        self.extra_brake -= val
        if self.extra_brake < 0.0:
            self.extra_brake = 0.0
        return (self.extra_brake <= 0.0)
