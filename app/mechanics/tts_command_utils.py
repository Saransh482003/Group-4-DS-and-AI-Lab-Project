def build_short_tts_command(nav_command):
	cmd = str(nav_command or "").strip()
	if not cmd:
		return "Go straight."

	lower = cmd.lower()
	if "turn left" in lower:
		return "Turn left."
	if "turn right" in lower:
		return "Turn right."
	if "blocked" in lower or "turn back" in lower:
		return "Path blocked. Scan around."
	if "move slightly left" in lower:
		return "Move slightly left."
	if "move slightly right" in lower:
		return "Move slightly right."
	if (
		"continue straight" in lower
		or "keep going straight" in lower
		or "keep moving straight" in lower
	):
		return "Go straight."
	return cmd