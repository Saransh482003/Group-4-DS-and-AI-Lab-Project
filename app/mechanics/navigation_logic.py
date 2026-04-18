import time

import cv2


class NavigationLogic:
	def __init__(
		self,
		frame_width,
		center_threshold=2.0,
		high_risk_threshold=4.0,
		safety_threshold=15.0,
		close_distance_threshold=1.5,
		far_distance_threshold=2.0,
		close_distance_weight=2.0,
		min_distance_for_risk=0.5,
		max_object_risk=8.0,
		significant_margin=0.6,
		bbox_overlap_threshold=0.1,
		ema_alpha=0.3,
		turn_hysteresis=0.5,
		command_hold_seconds=0.7,
		switch_confirm_frames=3,
		depth_hazard_danger_weight=4.0,
		depth_hazard_warning_weight=1.5,
	):
		# Tunable thresholds used by risk scoring and command selection.
		self.frame_width = float(frame_width)
		self.center_threshold = center_threshold
		self.high_risk_threshold = high_risk_threshold
		self.safety_threshold = safety_threshold
		self.close_distance_threshold = close_distance_threshold
		self.far_distance_threshold = far_distance_threshold
		self.close_distance_weight = close_distance_weight
		self.min_distance_for_risk = min_distance_for_risk
		self.max_object_risk = max_object_risk
		self.significant_margin = significant_margin
		self.bbox_overlap_threshold = bbox_overlap_threshold
		self.ema_alpha = ema_alpha
		self.turn_hysteresis = turn_hysteresis
		self.command_hold_seconds = command_hold_seconds
		self.switch_confirm_frames = switch_confirm_frames
		self.depth_hazard_danger_weight = depth_hazard_danger_weight
		self.depth_hazard_warning_weight = depth_hazard_warning_weight

		# Zone boundaries (left 30%, center 40%, right 30%).
		self.left_end = int(0.30 * frame_width)
		self.center_end = int(0.70 * frame_width)

		# State used by smoothing and command stabilization.
		self._smoothed_risks = {"left": 0.0, "center": 0.0, "right": 0.0}
		self._has_history = False
		self._last_command = "Searching for path. Turn back."
		self._last_command_ts = 0.0
		self._pending_command = None
		self._pending_count = 0
		self._first_decision = True

	def _zones_for_bbox(self, bbox):
		x1, _, x2, _ = bbox
		x1 = max(0.0, min(self.frame_width, float(x1)))
		x2 = max(0.0, min(self.frame_width, float(x2)))
		obj_width = x2 - x1
		if obj_width <= 0:
			return {}

		overlap_left = max(0.0, min(x2, float(self.left_end)) - x1)
		overlap_center = max(
			0.0,
			min(x2, float(self.center_end)) - max(x1, float(self.left_end)),
		)
		overlap_right = max(0.0, x2 - max(x1, float(self.center_end)))

		zone_weights = {
			"left": overlap_left / obj_width,
			"center": overlap_center / obj_width,
			"right": overlap_right / obj_width,
		}

		filtered = {
			zone: weight
			for zone, weight in zone_weights.items()
			if weight > self.bbox_overlap_threshold
		}
		if not filtered:
			best_zone = max(zone_weights, key=zone_weights.get)
			filtered = {best_zone: 1.0}

		total = sum(filtered.values())
		if total <= 0:
			return {}
		return {zone: weight / total for zone, weight in filtered.items()}

	def _object_risk(self, distance):
		if distance is None or distance <= 0:
			return 0.0
		if distance > self.far_distance_threshold:
			return 0.0

		safe_distance = max(float(distance), self.min_distance_for_risk)
		risk = 1.0 / safe_distance
		if distance < self.close_distance_threshold:
			risk *= self.close_distance_weight
		return min(risk, self.max_object_risk)

	def _is_left_command(self, command):
		return "left" in command.lower()

	def _is_right_command(self, command):
		return "right" in command.lower()

	def process_detections(self, detections, timestamp=None, depth_hazard=None):
		raw_zone_risks = {"left": 0.0, "center": 0.0, "right": 0.0}

		for det in detections:
			bbox = det.get("bbox")
			distance = det.get("distance")
			if bbox is None or len(bbox) != 4:
				continue

			risk = self._object_risk(distance)
			zone_weights = self._zones_for_bbox(bbox)
			for zone, weight in zone_weights.items():
				raw_zone_risks[zone] += risk * weight

		raw_zone_risks = self._blend_depth_hazard(raw_zone_risks, depth_hazard)

		zone_risks = self._smooth_zone_risks(raw_zone_risks)
		command = self._decide_stable_command(zone_risks, timestamp)
		return zone_risks, command

	def _blend_depth_hazard(self, zone_risks, depth_hazard):
		if not depth_hazard:
			return zone_risks

		if isinstance(depth_hazard, dict) and "zone_summary" in depth_hazard:
			zone_data = depth_hazard.get("zone_summary") or {}
		else:
			zone_data = depth_hazard

		if not isinstance(zone_data, dict):
			return zone_risks

		blended = zone_risks.copy()
		for zone in ("left", "center", "right"):
			zone_info = zone_data.get(zone) or {}
			danger_ratio = float(zone_info.get("danger_ratio", 0.0) or 0.0)
			warning_ratio = float(zone_info.get("warning_ratio", 0.0) or 0.0)

			hazard_risk = (
				(self.depth_hazard_danger_weight * danger_ratio)
				+ (self.depth_hazard_warning_weight * warning_ratio)
			)
			blended[zone] += hazard_risk

		return blended

	def _smooth_zone_risks(self, raw_zone_risks):
		if not self._has_history:
			self._smoothed_risks = raw_zone_risks.copy()
			self._has_history = True
			return self._smoothed_risks.copy()

		alpha = self.ema_alpha
		for zone in ("left", "center", "right"):
			self._smoothed_risks[zone] = (
				alpha * raw_zone_risks[zone] + (1.0 - alpha) * self._smoothed_risks[zone]
			)
		return self._smoothed_risks.copy()

	def _decide_stable_command(self, zone_risks, timestamp=None):
		if timestamp is None:
			timestamp = time.time()

		candidate = self._decide_command(zone_risks)
		if self._first_decision:
			self._first_decision = False
			self._last_command = candidate
			self._last_command_ts = timestamp
			return candidate

		left_cmd = self._is_left_command
		right_cmd = self._is_right_command
		is_turn_pair = (
			(left_cmd(self._last_command) or right_cmd(self._last_command))
			and (left_cmd(candidate) or right_cmd(candidate))
			and candidate != self._last_command
		)
		if is_turn_pair and (timestamp - self._last_command_ts) < self.command_hold_seconds:
			return self._last_command

		if candidate != self._last_command:
			if candidate == self._pending_command:
				self._pending_count += 1
			else:
				self._pending_command = candidate
				self._pending_count = 1

			if self._pending_count < self.switch_confirm_frames:
				return self._last_command

			self._last_command = candidate
			self._last_command_ts = timestamp
			self._pending_command = None
			self._pending_count = 0
			return candidate

		self._pending_command = None
		self._pending_count = 0
		return self._last_command

	def _decide_command(self, zone_risks):
		left_risk = zone_risks["left"]
		center_risk = zone_risks["center"]
		right_risk = zone_risks["right"]
		margin = self.significant_margin

		if center_risk < self.center_threshold:
			if (
				right_risk + margin < center_risk
				and right_risk + margin < left_risk
				and right_risk < self.safety_threshold
			):
				return "Turn right."

			if (
				left_risk + margin < center_risk
				and left_risk + margin < right_risk
				and left_risk < self.safety_threshold
			):
				return "Turn left."

			return "Go straight."

		if (
			left_risk + margin < center_risk
			and left_risk + margin < right_risk
			and left_risk < self.safety_threshold
		):
			return "Turn left."

		if (
			right_risk + margin < center_risk
			and right_risk + margin < left_risk
			and right_risk < self.safety_threshold
		):
			return "Turn right."

		if center_risk >= left_risk and center_risk >= right_risk:
			if left_risk <= right_risk and left_risk < self.safety_threshold:
				return "Turn left."
			if right_risk < self.safety_threshold:
				return "Turn right."

		if (
			left_risk > self.high_risk_threshold
			and center_risk > self.high_risk_threshold
			and right_risk > self.high_risk_threshold
			and min(left_risk, center_risk, right_risk) >= self.safety_threshold
		):
			return "Path blocked. Scan around."

		if right_risk - left_risk > self.turn_hysteresis:
			return "Move slightly left."
		if left_risk - right_risk > self.turn_hysteresis:
			return "Move slightly right."

		return "Searching for path. Turn back."

	def _wrap_text(self, text, max_width, font, font_scale, thickness):
		words = str(text).split()
		if not words:
			return [""]

		lines = []
		line = ""
		for word in words:
			candidate = word if not line else f"{line} {word}"
			candidate_w = cv2.getTextSize(candidate, font, font_scale, thickness)[0][0]
			if candidate_w <= max_width:
				line = candidate
				continue

			if line:
				lines.append(line)
				line = word
				continue

			# Handle very long single token.
			chunk = ""
			for ch in word:
				test_chunk = chunk + ch
				test_w = cv2.getTextSize(test_chunk, font, font_scale, thickness)[0][0]
				if test_w <= max_width or not chunk:
					chunk = test_chunk
				else:
					lines.append(chunk)
					chunk = ch
			if chunk:
				line = chunk

		if line:
			lines.append(line)

		return lines

	def draw_overlays(self, frame, zone_risks, command):
		h, w = frame.shape[:2]

		cv2.rectangle(frame, (0, 0), (self.left_end, h), (0, 0, 255), 2)
		cv2.rectangle(frame, (self.left_end, 0), (self.center_end, h), (0, 255, 0), 2)
		cv2.rectangle(frame, (self.center_end, 0), (w, h), (0, 0, 255), 2)

		cv2.putText(
			frame,
			f"Left Risk: {zone_risks['left']:.2f}",
			(10, 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			(255, 255, 255),
			2,
		)
		cv2.putText(
			frame,
			f"Center Risk: {zone_risks['center']:.2f}",
			(max(10, self.left_end + 10), 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			(255, 255, 255),
			2,
		)
		cv2.putText(
			frame,
			f"Right Risk: {zone_risks['right']:.2f}",
			(max(10, self.center_end + 10), 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			(255, 255, 255),
			2,
		)

		font = cv2.FONT_HERSHEY_SIMPLEX
		font_scale = 0.75
		thickness = 2
		padding = 8
		line_gap = 4
		lines = self._wrap_text(command, max(100, w - 20), font, font_scale, thickness)
		line_h = cv2.getTextSize("Ay", font, font_scale, thickness)[0][1]
		block_h = (len(lines) * line_h) + (max(0, len(lines) - 1) * line_gap) + (2 * padding)

		y2 = h - 8
		y1 = max(0, y2 - block_h)
		x1 = 8
		x2 = w - 8

		overlay = frame.copy()
		cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
		cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

		y = y1 + padding + line_h
		for line in lines:
			cv2.putText(
				frame,
				line,
				(x1 + padding, y),
				font,
				font_scale,
				(255, 255, 255),
				thickness,
			)
			y += line_h + line_gap

		return frame
