import multiprocessing as mp
import queue
import threading
import time
import uuid

from .nav_tts_piper import PiperTTS
from .tts_config import DEFAULT_TTS_MIN_INTERVAL_SECONDS
from .tts_config import DEFAULT_TTS_PHRASE_CACHE_MAXSIZE
from .tts_config import DEFAULT_TTS_QUEUE_MAXSIZE
from .tts_phrase_cache import TtsPhraseCache


def _process_tts_worker_main(
	piper_executable,
	voice_model_path,
	voice_config_path,
	job_queue,
	result_queue,
	direct_playback,
	play_audio,
	cache_max_items,
):
	"""Long-lived process worker for TTS requests.

	Each process owns one PiperTTS engine instance and an in-memory phrase cache
	so repeated commands can be served without re-synthesis.
	"""
	engine = PiperTTS(
		piper_executable=piper_executable,
		voice_model_path=voice_model_path,
		voice_config_path=voice_config_path,
	)
	cache = TtsPhraseCache(cache_max_items)

	while True:
		job = job_queue.get()
		if job is None:
			break

		job_id = job.get("job_id")
		text = job.get("text")
		started = time.perf_counter()
		cache_hit = False

		try:
			if play_audio and not direct_playback:
				# Quality-first playback path: let Piper write/play standard WAV.
				engine.synthesize(text, play_audio=True)
				tts_mode = "process_file_playback"
			else:
				wav_bytes = cache.get(text)
				if wav_bytes is None:
					wav_bytes = engine.synthesize_wav_bytes(text)
					cache.put(text, wav_bytes)
				else:
					cache_hit = True

				tts_mode = "process_memory"
				if direct_playback:
					engine.play_bytes(wav_bytes)
					tts_mode = "process_direct_memory"

			result_queue.put(
				{
					"job_id": job_id,
					"ok": True,
					"error": None,
					"cache_hit": cache_hit,
					"tts_mode": tts_mode,
					"worker_latency_ms": (time.perf_counter() - started) * 1000.0,
				}
			)
		except Exception as exc:
			result_queue.put(
				{
					"job_id": job_id,
					"ok": False,
					"error": str(exc),
					"cache_hit": cache_hit,
					"tts_mode": "process_error",
					"worker_latency_ms": (time.perf_counter() - started) * 1000.0,
				}
			)


class TextToSpeech:
	def __init__(
		self,
		piper_executable,
		voice_model_path,
		voice_config_path=None,
		phrase_cache_maxsize=DEFAULT_TTS_PHRASE_CACHE_MAXSIZE,
	):
		self.piper_executable = piper_executable
		self.voice_model_path = voice_model_path
		self.voice_config_path = voice_config_path
		self.engine = None
		self._phrase_cache = TtsPhraseCache(phrase_cache_maxsize)

	def load_engine(self):
		self.engine = PiperTTS(
			piper_executable=self.piper_executable,
			voice_model_path=self.voice_model_path,
			voice_config_path=self.voice_config_path,
		)
		return self.engine

	def _cache_get(self, text):
		return self._phrase_cache.get(text)

	def _cache_put(self, text, wav_bytes):
		self._phrase_cache.put(text, wav_bytes)

	def synthesize_wav_bytes_cached(self, text):
		if self.engine is None:
			self.load_engine()

		cached = self._cache_get(text)
		if cached is not None:
			return cached, True

		wav_bytes = self.engine.synthesize_wav_bytes(text)
		self._cache_put(text, wav_bytes)
		return wav_bytes, False

	def speak(self, text):
		try:
			wav_bytes, cache_hit = self.synthesize_wav_bytes_cached(text)
			self.engine.play_bytes(wav_bytes)
			return "direct-memory-cache" if cache_hit else "direct-memory"
		except Exception:
			if self.engine is None:
				self.load_engine()
			return self.engine.speak_direct_safe(text, fallback_play_audio=True)

	def synthesize(self, text, play_audio=False):
		if self.engine is None:
			self.load_engine()

		if play_audio:
			# Quality-first playback path for runtime.
			self.engine.synthesize(text, play_audio=True)
			return None

		wav_bytes, _ = self.synthesize_wav_bytes_cached(text)
		return wav_bytes


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


class _ProcessTTSWorker:
	def __init__(
		self,
		tts_engine,
		direct_playback=True,
		play_audio=False,
		queue_maxsize=DEFAULT_TTS_QUEUE_MAXSIZE,
		cache_max_items=128,
	):
		self.tts_engine = tts_engine
		self.direct_playback = direct_playback
		self.play_audio = play_audio
		self.cache_max_items = max(0, int(cache_max_items))
		self.job_queue = mp.Queue(maxsize=max(1, int(queue_maxsize)))
		self.result_queue = mp.Queue()
		self.process = None
		self.last_error = None
		self.last_result = None

	def start(self):
		if self.process is not None and self.process.is_alive():
			return
		self.process = mp.Process(
			target=_process_tts_worker_main,
			args=(
				self.tts_engine.piper_executable,
				self.tts_engine.voice_model_path,
				self.tts_engine.voice_config_path,
				self.job_queue,
				self.result_queue,
				self.direct_playback,
				self.play_audio,
				self.cache_max_items,
			),
			daemon=True,
		)
		self.process.start()

	def stop(self):
		if self.process is None:
			return
		try:
			self.job_queue.put_nowait(None)
		except queue.Full:
			pass
		self.process.join(timeout=2.0)
		if self.process.is_alive():
			self.process.terminate()
			self.process.join(timeout=1.0)

	def enqueue(self, text):
		if self.process is None or not self.process.is_alive():
			self.last_error = "tts_process_not_running"
			return False
		job = {
			"job_id": str(uuid.uuid4()),
			"text": text,
		}
		try:
			self.job_queue.put_nowait(job)
			return True
		except queue.Full:
			self.last_error = "tts_queue_full"
			return False

	def poll_results(self):
		while True:
			try:
				result = self.result_queue.get_nowait()
			except queue.Empty:
				break
			self.last_result = result
			if not result.get("ok", False):
				self.last_error = result.get("error")


class TTSRuntimeController:
	def __init__(
		self,
		tts_engine,
		enable_async=True,
		use_process_worker=True,
		direct_playback=True,
		play_audio=False,
		speak_once_per_execution=False,
		speak_on_command_change=True,
		min_interval_seconds=DEFAULT_TTS_MIN_INTERVAL_SECONDS,
		queue_maxsize=DEFAULT_TTS_QUEUE_MAXSIZE,
		phrase_cache_maxsize=DEFAULT_TTS_PHRASE_CACHE_MAXSIZE,
	):
		self.tts_engine = tts_engine
		self.enable_async = enable_async
		self.use_process_worker = use_process_worker
		self.direct_playback = direct_playback
		self.play_audio = play_audio
		self.speak_once_per_execution = speak_once_per_execution
		self.speak_on_command_change = speak_on_command_change
		self.min_interval_seconds = min_interval_seconds
		self.phrase_cache_maxsize = max(0, int(phrase_cache_maxsize))
		self.worker = None

		self._state = {
			"spoken": False,
			"last_command": None,
			"last_spoken_at": 0.0,
		}

		if self.enable_async:
			if self.use_process_worker:
				self.worker = _ProcessTTSWorker(
					tts_engine=self.tts_engine,
					direct_playback=self.direct_playback,
					play_audio=self.play_audio,
					queue_maxsize=queue_maxsize,
					cache_max_items=self.phrase_cache_maxsize,
				)
			else:
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
			if isinstance(self.worker, _ProcessTTSWorker):
				self.worker.poll_results()
			enqueue_ok = self.worker.enqueue(nav_command)
			result = {
				"tts_mode": "async_process_queue" if isinstance(self.worker, _ProcessTTSWorker) else "async_queue",
				"tts_enqueue_ok": enqueue_ok,
				"tts_error": None,
				"tts_cache_hit": None,
				"tts_worker_latency_ms": None,
			}
			if isinstance(self.worker, _ProcessTTSWorker) and self.worker.last_result is not None:
				result["tts_cache_hit"] = self.worker.last_result.get("cache_hit")
				result["tts_worker_latency_ms"] = self.worker.last_result.get("worker_latency_ms")
			if not enqueue_ok and self.worker.last_error:
				result["tts_error"] = self.worker.last_error
			return result

		if self.direct_playback:
			mode = self.tts_engine.speak(nav_command)
			return {
				"tts_mode": mode,
				"tts_enqueue_ok": None,
				"tts_error": None,
				"tts_cache_hit": "cache" in str(mode),
				"tts_worker_latency_ms": None,
			}

		self.tts_engine.synthesize(nav_command, play_audio=self.play_audio)
		return {
			"tts_mode": "memory",
			"tts_enqueue_ok": None,
			"tts_error": None,
			"tts_cache_hit": None,
			"tts_worker_latency_ms": None,
		}

	def handle_command(self, nav_command):
		now = time.time()
		should_speak, skip_reason = self._evaluate_gate(nav_command, now)
		result = {
			"tts_should_speak": should_speak,
			"tts_skip_reason": skip_reason,
			"tts_mode": None,
			"tts_error": None,
			"tts_enqueue_ok": None,
			"tts_cache_hit": None,
			"tts_worker_latency_ms": None,
		}

		if not should_speak:
			return result

		dispatch = self._dispatch(nav_command)
		result.update(dispatch)

		self._state["spoken"] = True
		self._state["last_command"] = nav_command
		self._state["last_spoken_at"] = now
		return result
