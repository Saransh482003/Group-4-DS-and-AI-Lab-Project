def build_short_tts_command(nav_command):
	cmd = str(nav_command or "").strip()
	if not cmd:
		return "Go straight."

	lower = cmd.lower()
	if "clear path on your left" in lower:
		return "Turn left."
	if "clear path on your right" in lower:
		return "Turn right."
	if "turn left" in lower:
		return "Turn left."
	if "turn right" in lower:
		return "Turn right."
	if "searching for path" in lower or "turn back" in lower:
		return "Searching for path. Turn back."
	if "blocked" in lower:
		return "Path blocked. Scan around."
	if "move slightly left" in lower:
		return "Move slightly left."
	if "move slightly right" in lower:
		return "Move slightly right."
	if (
		"continue straight" in lower
		or "keep going straight" in lower
		or "keep moving straight" in lower
		or "way ahead is clear" in lower
		or "go straight" in lower
	):
		return "Go straight."
	return cmd


def get_tts_command_priority(command_text):
	"""Return priority for queueing/preemption.

	Higher number means higher priority.
	- 3: stop/block/search commands
	- 2: turning/avoidance commands
	- 1: go-straight/clear-path commands
	"""
	text = str(command_text or "").strip().lower()
	if not text:
		return 1

	high_keywords = (
		"blocked",
		"turn back",
		"stop",
		"scan around",
		"searching for path",
	)
	if any(keyword in text for keyword in high_keywords):
		return 3

	turn_keywords = (
		"turn left",
		"turn right",
		"move slightly left",
		"move slightly right",
		"obstacle ahead",
	)
	if any(keyword in text for keyword in turn_keywords):
		return 2

	straight_keywords = (
		"go straight",
		"continue straight",
		"clear path",
		"keep going straight",
		"keep moving straight",
		"way ahead is clear",
	)
	if any(keyword in text for keyword in straight_keywords):
		return 1

	return 2