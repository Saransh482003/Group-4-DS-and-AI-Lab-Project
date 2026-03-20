import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline


class NavigationSLMAugmentLogic:
	"""Hybrid navigation logic: Python spatial analysis + SLM command generation."""

	SYSTEM_INSTRUCTIONS = (
		"You are a spatial reasoning engine for a blind user. "
		"Read object classes, bounding boxes, depth values, and distance values. "
		"Use bbox positions to infer left/center/right and where objects are located. "
		"Give one decisive navigation decision that avoids the closest threat and moves "
		"toward the safest direction. Keep it imperative, natural, and under 10 words."
	)

	def __init__(
		self,
		hazard_distance_threshold=1.5,
		model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
		device=None,
		slm_pipeline=None,
		local_files_only=True,
	):
		self.hazard_distance_threshold = hazard_distance_threshold
		self.model_name = model_name

		if device is None:
			self.device = "cuda" if torch.cuda.is_available() else "cpu"
		else:
			self.device = device

		if slm_pipeline is not None:
			self.slm_pipeline = slm_pipeline
			self.tokenizer = None
			self.model = None
			return

		self.tokenizer = AutoTokenizer.from_pretrained(
			self.model_name,
			local_files_only=local_files_only,
		)
		self.model = AutoModelForCausalLM.from_pretrained(
			self.model_name,
			local_files_only=local_files_only,
		).to(self.device)
		self.model.eval()
		if self.tokenizer.pad_token_id is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token

		pipeline_device = 0 if self.device == "cuda" else -1
		self.slm_pipeline = pipeline(
			"text-generation",
			model=self.model,
			tokenizer=self.tokenizer,
			device=pipeline_device,
		)

	def _safest_path(self, zone_risks):
		ordered = ["center", "left", "right"]
		return min(ordered, key=lambda z: (float(zone_risks.get(z, float("inf"))), ordered.index(z)))

	def _zone_from_bbox(self, bbox, frame_width):
		x1, _, x2, _ = bbox
		center_x = (float(x1) + float(x2)) / 2.0
		if center_x < 0.30 * frame_width:
			return "Left"
		if center_x <= 0.70 * frame_width:
			return "Center"
		return "Right"

	def _valid_detection(self, det):
		bbox = det.get("bbox")
		distance = det.get("distance")
		if bbox is None or len(bbox) != 4 or distance is None:
			return None
		try:
			distance_val = float(distance)
		except (TypeError, ValueError):
			return None
		if distance_val <= 0:
			return None
		class_name = det.get("class_name") or det.get("class") or "unknown"
		depth_val = det.get("depth")
		if depth_val is not None:
			try:
				depth_val = float(depth_val)
			except (TypeError, ValueError):
				depth_val = None
		return {
			"class_name": class_name,
			"bbox": bbox,
			"depth": depth_val,
			"distance": distance_val,
		}

	def _immediate_hazards(self, detections):
		hazards = []
		seen = set()
		for det in detections:
			normalized = self._valid_detection(det)
			if normalized is None:
				continue
			if (
				normalized["distance"] < self.hazard_distance_threshold
				and normalized["class_name"] not in seen
			):
				class_name = normalized["class_name"]
				hazards.append(class_name)
				seen.add(class_name)
		return hazards

	def _infer_frame_width(self, detections):
		max_x = 0.0
		for det in detections:
			bbox = det.get("bbox")
			if bbox is None or len(bbox) != 4:
				continue
			max_x = max(max_x, float(bbox[2]))
		return max(1.0, max_x)

	def build_scene_context(self, zone_risks, detections, frame_width):
		normalized = []
		for det in detections:
			entry = self._valid_detection(det)
			if entry is not None:
				normalized.append(entry)

		normalized.sort(key=lambda d: d["distance"])
		safest_zone = self._safest_path(zone_risks).capitalize()

		if normalized:
			primary = normalized[0]
			primary_text = (
				f"{primary['class_name']} at {primary['distance']:.2f}m "
				f"({self._zone_from_bbox(primary['bbox'], frame_width)})"
			)
		else:
			primary_text = "none"

		top_three = normalized[:3]
		nearby = []
		for obj in top_three[1:]:
			nearby.append(
				f"{obj['class_name']} {obj['distance']:.2f}m {self._zone_from_bbox(obj['bbox'], frame_width)}"
			)
		nearby_text = "; ".join(nearby) if nearby else "none"

		return (
			"SCENE CONTEXT:\n"
			f"Primary threat: {primary_text}.\n"
			f"Risks -> Left:{float(zone_risks.get('left', 0.0)):.2f}, "
			f"Center:{float(zone_risks.get('center', 0.0)):.2f}, "
			f"Right:{float(zone_risks.get('right', 0.0)):.2f}.\n"
			f"Safest zone: {safest_zone}.\n"
			f"Nearby objects: {nearby_text}."
		)

	def prepare_rich_slm_context(self, zone_risks, detections, frame_width):
		"""Build a raw object-centric context (bbox + depth + distance) for SLM."""
		normalized = []
		for det in detections:
			entry = self._valid_detection(det)
			if entry is not None:
				normalized.append(entry)

		normalized.sort(key=lambda d: d["distance"])

		left_risk = float(zone_risks.get("left", 0.0))
		center_risk = float(zone_risks.get("center", 0.0))
		right_risk = float(zone_risks.get("right", 0.0))
		risk_map = {"left": left_risk, "center": center_risk, "right": right_risk}
		safest_zone = min(("center", "left", "right"), key=lambda z: (risk_map[z], ["center", "left", "right"].index(z)))

		bbox_lines = []
		distance_lines = []
		for det in normalized:
			x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
			bbox_lines.append(f"- {det['class_name']}: bbox=({x1}, {y1}, {x2}, {y2})")
			if det["depth"] is None:
				distance_lines.append(
					f"- {det['class_name']}: depth=N/A, distance={det['distance']:.4f}"
				)
			else:
				distance_lines.append(
					f"- {det['class_name']}: depth={det['depth']:.4f}, distance={det['distance']:.4f}"
				)

		bbox_section = "\n".join(bbox_lines) if bbox_lines else "- none"
		distance_section = "\n".join(distance_lines) if distance_lines else "- none"

		return (
			"SCENE CONTEXT:\n"
			"Detections (class + bbox):\n"
			f"{bbox_section}\n"
			"Object distances from camera (relative scale):\n"
			f"{distance_section}\n"
			f"Zone risks: Left={left_risk:.2f}, Center={center_risk:.2f}, Right={right_risk:.2f}.\n"
			f"Safest zone by risk: {safest_zone.capitalize()}.\n"
			"Decision task: Determine where everything is and choose the safest movement command."
		)

	def _fallback_instruction(self, safest_path, hazards):
		path_map = {
			"left": "Step left",
			"center": "Walk straight",
			"right": "Step right",
		}
		base = path_map.get(safest_path, "Move")
		if hazards:
			return f"{base}, avoid {hazards[0]}."
		if safest_path == "center":
			return "Walk straight now."
		return f"{base} now."

	def _sanitize_instruction(self, raw_text, safest_path, hazards):
		text = raw_text.strip().replace("\n", " ")
		text = re.sub(r"\s+", " ", text)
		text = text.strip("\"' ")
		text = re.sub(r"<\|[^>]+\|>", " ", text)
		text = re.sub(r"\b(system|assistant|user)\s*:\s*", "", text, flags=re.IGNORECASE)
		text = re.sub(r"\bSCENE CONTEXT\b\s*:?.*", "", text, flags=re.IGNORECASE)
		text = re.sub(r"\s+", " ", text).strip()

		if not text:
			text = self._fallback_instruction(safest_path, hazards)

		if hazards and hazards[0].lower() not in text.lower():
			text = self._fallback_instruction(safest_path, hazards)

		if not re.match(r"^(step|move|stop|turn|walk)\b", text, flags=re.IGNORECASE):
			text = self._fallback_instruction(safest_path, hazards)

		words = text.replace(".", "").split()
		if len(words) > 10:
			text = " ".join(words[:10]).rstrip(",") + "."
		elif not text.endswith("."):
			text += "."

		return text

	def _build_generation_prompt(self, context):
		messages = [
			{"role": "system", "content": self.SYSTEM_INSTRUCTIONS},
			{"role": "user", "content": f"{context}\n\nInstruction:"},
		]
		if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
			return self.tokenizer.apply_chat_template(
				messages,
				tokenize=False,
				add_generation_prompt=True,
			)
		return f"{self.SYSTEM_INSTRUCTIONS}\n\n{context}\n\nInstruction:"

	def _extract_generated_text(self, generation_output):
		if not generation_output:
			return ""

		first = generation_output[0]
		if isinstance(first, dict):
			if "generated_text" in first:
				return first.get("generated_text") or ""
			if "text" in first:
				return first.get("text") or ""

		if isinstance(first, str):
			return first

		return str(first)

	def _generate_navigation_text(self, context):
		prompt = self._build_generation_prompt(context)
		generation = self.slm_pipeline(
			prompt,
			temperature=0.1,
			max_new_tokens=25,
			do_sample=False,
			return_full_text=False,
		)
		return self._extract_generated_text(generation)

	def decide_instruction(self, zone_risks, detections, frame_width=None):
		if frame_width is None:
			frame_width = self._infer_frame_width(detections)

		safest_path = self._safest_path(zone_risks)
		hazards = self._immediate_hazards(detections)
		context = self.prepare_rich_slm_context(zone_risks, detections, frame_width)
		raw_instruction = self._generate_navigation_text(context)
		instruction = self._sanitize_instruction(raw_instruction, safest_path, hazards)
		return {
			"context": context,
			"instruction": instruction,
			"safest_path": safest_path,
			"hazards": hazards,
		}
