import queue
import threading
import time

from .nav_tts_piper import PiperTTS


class TextToSpeech:
	"""
	Wrapper for the Piper TTS engine, handling audio synthesis and playback
	of navigation instructions.
	"""
	def __init__(self, piper_executable, voice_model_path, voice_config_path=None):
		self.piper_executable = piper_executable
		self.voice_model_path = voice_model_path
		self.voice_config_path = voice_config_path
		self.engine = None

	def load_engine(self):
		self.engine = PiperTTS(
			piper_executable=self.piper_executable,
			voice_model_path=self.voice_model_path,
			voice_config_path=self.voice_config_path,
		)
		return self.engine

	def speak(self, text):
		if self.engine is None:
			self.load_engine()
		return self.engine.speak_direct_safe(text, fallback_play_audio=True)

	def synthesize(self, text, play_audio=False):
		if self.engine is None:
			self.load_engine()
		return self.engine.synthesize(text, play_audio=play_audio)


class _AsyncTTSWorker:
	def __init__(self, tts_engine, direct_playback=True, play_audio=False, queue_maxsize=5):
		self.tts_engine = tts_engine
		self.direct_playback = direct_playback
		self.play_audio = play_audio
		self.queue = queue.Queue(maxsize=max(1, int(queue_maxsize)))
		self._stop_event = threading.Event()
		self._thread = threading.Thread(target=self._run, daemon=True)
		self.last_error = None

	def start(self):
		self._thread.start()

	def stop(self):
		self._stop_event.set()
		try:
			self.queue.put_nowait(None)
		except queue.Full:
			pass
		self._thread.join(timeout=2.0)

	def enqueue(self, text):
		try:
			self.queue.put_nowait(text)
			return True
		except queue.Full:
			self.last_error = "tts_queue_full"
			return False

	def _run(self):
		while not self._stop_event.is_set():
			try:
				item = self.queue.get(timeout=0.1)
			except queue.Empty:
				continue
			if item is None:
				self.queue.task_done()
				continue
			try:
				if self.direct_playback:
					self.tts_engine.speak(item)
				else:
					self.tts_engine.synthesize(item, play_audio=self.play_audio)
			except Exception as exc:
				self.last_error = str(exc)
			finally:
				self.queue.task_done()


class TTSRuntimeController:
	"""
	Manages the timing and frequency of TTS announcements to ensure clear 
	communication without overwhelming the user.
	"""
	def __init__(
		self,
		tts_engine,
		enable_async=True,
		direct_playback=True,
		play_audio=False,
		speak_once_per_execution=False,
		speak_on_command_change=True,
		min_interval_seconds=1.2,
		queue_maxsize=5,
	):
		self.tts_engine = tts_engine
		self.enable_async = enable_async
		self.direct_playback = direct_playback
		self.play_audio = play_audio
		self.speak_once_per_execution = speak_once_per_execution
		self.speak_on_command_change = speak_on_command_change
		self.min_interval_seconds = min_interval_seconds
		self.worker = None

		self._state = {
			"spoken": False,
			"last_command": None,
			"last_spoken_at": 0.0,
		}

		if self.enable_async:
			self.worker = _AsyncTTSWorker(
				tts_engine=self.tts_engine,
				direct_playback=self.direct_playback,
				play_audio=self.play_audio,
				queue_maxsize=queue_maxsize,
			)
			self.worker.start()

	def stop(self):
		if self.worker is not None:
			self.worker.stop()

	def _evaluate_gate(self, nav_command, now):
		should_speak = (not self.speak_once_per_execution) or (not self._state["spoken"])
		if not should_speak:
			return False, "once_per_execution"

		if self.speak_on_command_change and self._state.get("last_command") == nav_command:
			return False, "command_unchanged"

		if (now - self._state.get("last_spoken_at", 0.0)) < self.min_interval_seconds:
			return False, "rate_limited"

		return True, None

	def _dispatch(self, nav_command):
		if self.enable_async and self.worker is not None:
			enqueue_ok = self.worker.enqueue(nav_command)
			result = {
				"tts_mode": "async_queue",
				"tts_enqueue_ok": enqueue_ok,
				"tts_error": None,
			}
			if not enqueue_ok and self.worker.last_error:
				result["tts_error"] = self.worker.last_error
			return result

		if self.direct_playback:
			self.tts_engine.speak(nav_command)
			return {"tts_mode": "direct", "tts_enqueue_ok": None, "tts_error": None}

		self.tts_engine.synthesize(nav_command, play_audio=self.play_audio)
		return {"tts_mode": "file", "tts_enqueue_ok": None, "tts_error": None}

	def handle_command(self, nav_command):
		now = time.time()
		should_speak, skip_reason = self._evaluate_gate(nav_command, now)
		result = {
			"tts_should_speak": should_speak,
			"tts_skip_reason": skip_reason,
			"tts_mode": None,
			"tts_error": None,
			"tts_enqueue_ok": None,
		}

		if not should_speak:
			return result

		dispatch = self._dispatch(nav_command)
		result.update(dispatch)

		self._state["spoken"] = True
		self._state["last_command"] = nav_command
		self._state["last_spoken_at"] = now
		return result
