import time

import cv2


class NavigationLogic:
	def __init__(
		self,
		frame_width,
		center_threshold=1.5,
		high_risk_threshold=4.0,
		close_distance_threshold=1.5,
		close_distance_weight=2.0,
		ema_alpha=0.3,
		turn_hysteresis=0.5,
		command_hold_seconds=0.7,
		switch_confirm_frames=3,
	):
		self.center_threshold = center_threshold
		self.high_risk_threshold = high_risk_threshold
		self.close_distance_threshold = close_distance_threshold
		self.close_distance_weight = close_distance_weight
		self.ema_alpha = ema_alpha
		self.turn_hysteresis = turn_hysteresis
		self.command_hold_seconds = command_hold_seconds
		self.switch_confirm_frames = switch_confirm_frames

		self.left_end = int(0.30 * frame_width)
		self.center_end = int(0.70 * frame_width)

		self._smoothed_risks = {"left": 0.0, "center": 0.0, "right": 0.0}
		self._has_history = False
		self._last_command = "Searching for path. Turn back."
		self._last_command_ts = 0.0
		self._pending_command = None
		self._pending_count = 0
		self._first_decision = True

	def _zones_for_bbox(self, bbox):
		x1, _, x2, _ = bbox
		zones = []
		if x1 < self.left_end:
			zones.append("left")
		if x2 > self.left_end and x1 < self.center_end:
			zones.append("center")
		if x2 > self.center_end:
			zones.append("right")
		return zones

	def _object_risk(self, distance):
		if distance is None or distance <= 0:
			return 0.0
		risk = 1.0 / float(distance)
		if distance < self.close_distance_threshold:
			risk *= self.close_distance_weight
		return risk

	def process_detections(self, detections, timestamp=None):
		raw_zone_risks = {"left": 0.0, "center": 0.0, "right": 0.0}

		for det in detections:
			bbox = det.get("bbox")
			distance = det.get("distance")
			if bbox is None or len(bbox) != 4:
				continue

			risk = self._object_risk(distance)
			for zone in self._zones_for_bbox(bbox):
				raw_zone_risks[zone] += risk

		zone_risks = self._smooth_zone_risks(raw_zone_risks)
		command = self._decide_stable_command(zone_risks, timestamp)
		return zone_risks, command

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

		left_cmd = "Obstacle ahead. Move slightly left."
		right_cmd = "Obstacle ahead. Move slightly right."
		is_turn_pair = (
			self._last_command in (left_cmd, right_cmd)
			and candidate in (left_cmd, right_cmd)
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

		if center_risk < self.center_threshold:
			return "Path clear. Continue straight."
		if (
			left_risk > self.high_risk_threshold
			and center_risk > self.high_risk_threshold
			and right_risk > self.high_risk_threshold
		):
			return "Path blocked. Please scan left and right."

		if right_risk - left_risk > self.turn_hysteresis:
			return "Obstacle ahead. Move slightly left."
		if left_risk - right_risk > self.turn_hysteresis:
			return "Obstacle ahead. Move slightly right."

		return "Searching for path. Turn back."

	def draw_overlays(self, frame, zone_risks, command):
		h, w = frame.shape[:2]
		overlay = frame.copy()

		cv2.rectangle(overlay, (0, 0), (self.left_end, h), (0, 0, 255), -1)
		cv2.rectangle(overlay, (self.left_end, 0), (self.center_end, h), (0, 255, 0), -1)
		cv2.rectangle(overlay, (self.center_end, 0), (w, h), (0, 0, 255), -1)
		cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

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
		cv2.putText(
			frame,
			command,
			(10, h - 20),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.75,
			(255, 255, 255),
			2,
		)
		return frame
