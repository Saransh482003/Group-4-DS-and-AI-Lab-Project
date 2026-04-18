import hashlib
import multiprocessing as mp
import os
import queue
import threading
import time
import tempfile
import uuid
import wave
from pathlib import Path

from .nav_tts_piper import PiperTTS
from .tts_command_utils import get_tts_command_priority
from .tts_config import DEFAULT_TTS_MIN_INTERVAL_SECONDS
from .tts_config import DEFAULT_TTS_ENABLE_PRIORITY_PREEMPT
from .tts_config import DEFAULT_TTS_PHRASE_CACHE_MAXSIZE
from .tts_config import DEFAULT_TTS_QUEUE_MAXSIZE
from .tts_config import DEFAULT_TTS_USE_RUNTIME_SYNTHESIS
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
		self._local_phrase_paths = {}
		self._local_phrase_dir = None

	def load_engine(self):
		self.engine = PiperTTS(
			piper_executable=self.piper_executable,
			voice_model_path=self.voice_model_path,
			voice_config_path=self.voice_config_path,
		)
		self._local_phrase_dir = Path(self.engine.output_dir) / "common_phrase_store"
		self._local_phrase_dir.mkdir(parents=True, exist_ok=True)
		return self.engine

	def _ensure_engine(self):
		if self.engine is None:
			self.load_engine()

	def _local_store_dir_path(self):
		self._ensure_engine()
		return self._local_phrase_dir

	def _phrase_local_path(self, text):
		phrase = str(text or "").strip()
		if not phrase:
			raise ValueError("TTS phrase is empty.")

		digest = hashlib.sha1(phrase.encode("utf-8")).hexdigest()[:12]
		slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in phrase)
		slug = "_".join(part for part in slug.split("_") if part)[:48]
		if not slug:
			slug = "phrase"
		return self._local_store_dir_path() / f"{slug}_{digest}.wav"

	@staticmethod
	def _wav_duration_seconds_from_path(wav_path):
		try:
			with wave.open(str(wav_path), "rb") as wav_file:
				frames = float(wav_file.getnframes())
				rate = float(wav_file.getframerate())
				if rate <= 0:
					return 0.0
				return max(0.0, frames / rate)
		except Exception:
			return 0.0

	@staticmethod
	def _stop_windows_playback():
		if os.name != "nt":
			return
		import winsound  # pylint: disable=import-outside-toplevel

		winsound.PlaySound(None, 0)

	def _play_wav_interruptible(self, wav_path, interrupt_event=None, stop_event=None):
		self._ensure_engine()

		if os.name != "nt":
			self.engine.play(str(wav_path))
			return False

		import winsound  # pylint: disable=import-outside-toplevel

		winsound.PlaySound(str(wav_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
		duration_seconds = self._wav_duration_seconds_from_path(wav_path)
		deadline = time.perf_counter() + max(0.2, duration_seconds + 0.2)

		interrupted = False
		while time.perf_counter() < deadline:
			if stop_event is not None and stop_event.is_set():
				interrupted = True
				break
			if interrupt_event is not None and interrupt_event.is_set():
				interrupted = True
				break
			time.sleep(0.02)

		if interrupted:
			self._stop_windows_playback()
		return interrupted

	def ensure_local_phrase_wav(self, text):
		phrase = str(text or "").strip()
		if not phrase:
			raise ValueError("TTS phrase is empty.")

		cached_path = self._local_phrase_paths.get(phrase)
		if cached_path is not None and os.path.exists(cached_path):
			return cached_path

		wav_path = self._phrase_local_path(phrase)
		if not wav_path.exists():
			self._ensure_engine()
			self.engine.synthesize(phrase, output_wav_path=str(wav_path), play_audio=False)

		resolved = str(wav_path.resolve())
		self._local_phrase_paths[phrase] = resolved
		return resolved

	def prewarm_local_phrase_wavs(self, phrases):
		self._ensure_engine()
		warmed = 0
		skipped = 0
		path_map = {}

		for phrase in phrases or []:
			text = str(phrase or "").strip()
			if not text:
				skipped += 1
				continue

			wav_path = self._phrase_local_path(text)
			if wav_path.exists():
				skipped += 1
			else:
				self.engine.synthesize(text, output_wav_path=str(wav_path), play_audio=False)
				warmed += 1

			resolved = str(wav_path.resolve())
			self._local_phrase_paths[text] = resolved
			path_map[text] = resolved

		return {
			"warmed": warmed,
			"skipped": skipped,
			"path_map": path_map,
		}

	def play_local_phrase_interruptible(self, text, interrupt_event=None, stop_event=None):
		wav_path = self.ensure_local_phrase_wav(text)
		interrupted = self._play_wav_interruptible(
			wav_path,
			interrupt_event=interrupt_event,
			stop_event=stop_event,
		)
		return "local-wav-interrupted" if interrupted else "local-wav"

	def speak_runtime_interruptible(
		self,
		text,
		interrupt_event=None,
		stop_event=None,
		direct_playback=True,
		play_audio=False,
	):
		self._ensure_engine()

		# Always synthesize first, then use interruptible playback for preemption.
		tmp_dir = self._local_store_dir_path() / "runtime_queue"
		tmp_dir.mkdir(parents=True, exist_ok=True)
		fd, tmp_path = tempfile.mkstemp(suffix=".wav", dir=str(tmp_dir))
		os.close(fd)
		try:
			self.engine.synthesize(text, output_wav_path=tmp_path, play_audio=False)
			interrupted = self._play_wav_interruptible(
				tmp_path,
				interrupt_event=interrupt_event,
				stop_event=stop_event,
			)
			if interrupted:
				return "runtime-synth-interrupted"
			if direct_playback and not play_audio:
				return "runtime-synth-direct"
			return "runtime-synth"
		finally:
			try:
				os.remove(tmp_path)
			except OSError:
				pass

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

	def prewarm_phrase_cache(self, phrases):
		"""Pre-generate and cache WAV bytes for a fixed set of phrases."""
		if self.engine is None:
			self.load_engine()

		warmed = 0
		skipped = 0
		for phrase in phrases or []:
			text = str(phrase or "").strip()
			if not text:
				skipped += 1
				continue
			_, cache_hit = self.synthesize_wav_bytes_cached(text)
			if cache_hit:
				skipped += 1
			else:
				warmed += 1

		return {
			"warmed": warmed,
			"skipped": skipped,
		}

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
	def __init__(
		self,
		tts_engine,
		direct_playback=True,
		play_audio=False,
		queue_maxsize=5,
		latest_only_queue=True,
		use_runtime_synthesis=True,
		enable_priority_preempt=True,
	):
		self.tts_engine = tts_engine
		self.direct_playback = direct_playback
		self.play_audio = play_audio
		self.latest_only_queue = latest_only_queue
		self.use_runtime_synthesis = use_runtime_synthesis
		self.enable_priority_preempt = enable_priority_preempt
		self.queue = queue.PriorityQueue(maxsize=max(1, int(queue_maxsize)))
		self._job_seq = 0
		self._current_priority = -1
		self._interrupt_playback_event = threading.Event()
		self._stop_event = threading.Event()
		self._thread = threading.Thread(target=self._run, daemon=True)
		self.last_error = None
		self.last_mode = None

	def start(self):
		self._thread.start()

	def stop(self):
		self._stop_event.set()
		self._interrupt_playback_event.set()
		try:
			self.queue.put_nowait((999, self._job_seq, None, -1, True))
			self._job_seq += 1
		except queue.Full:
			pass
		self._thread.join(timeout=2.0)
		self.tts_engine._stop_windows_playback()

	def _drop_pending_jobs(self):
		dropped = 0
		while True:
			try:
				_ = self.queue.get_nowait()
			except queue.Empty:
				break
			dropped += 1
			try:
				self.queue.task_done()
			except ValueError:
				# Defensive: task_done can fail if queue internals are out of sync.
				pass
		return dropped

	def enqueue(self, text, priority=1):
		try:
			if self.enable_priority_preempt and int(priority) > self._current_priority:
				self._interrupt_playback_event.set()
			if self.latest_only_queue:
				self._drop_pending_jobs()
			self.queue.put_nowait((-int(priority), self._job_seq, text, int(priority), False))
			self._job_seq += 1
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

			_, _, text, priority, is_sentinel = item
			if is_sentinel:
				self.queue.task_done()
				continue

			self._current_priority = int(priority)
			self._interrupt_playback_event.clear()
			try:
				if self.use_runtime_synthesis:
					self.last_mode = self.tts_engine.speak_runtime_interruptible(
						text,
						interrupt_event=self._interrupt_playback_event,
						stop_event=self._stop_event,
						direct_playback=self.direct_playback,
						play_audio=self.play_audio,
					)
				else:
					self.last_mode = self.tts_engine.play_local_phrase_interruptible(
						text,
						interrupt_event=self._interrupt_playback_event,
						stop_event=self._stop_event,
					)
			except Exception as exc:
				self.last_error = str(exc)
			finally:
				self._current_priority = -1
				self._interrupt_playback_event.clear()
				self.queue.task_done()


class _ProcessTTSWorker:
	def __init__(
		self,
		tts_engine,
		direct_playback=True,
		play_audio=False,
		queue_maxsize=DEFAULT_TTS_QUEUE_MAXSIZE,
		cache_max_items=128,
		latest_only_queue=True,
	):
		self.tts_engine = tts_engine
		self.direct_playback = direct_playback
		self.play_audio = play_audio
		self.cache_max_items = max(0, int(cache_max_items))
		self.latest_only_queue = latest_only_queue
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

	def _drop_pending_jobs(self):
		while True:
			try:
				_ = self.job_queue.get_nowait()
			except queue.Empty:
				break

	def enqueue(self, text, priority=1):
		if self.process is None or not self.process.is_alive():
			self.last_error = "tts_process_not_running"
			return False
		if self.latest_only_queue:
			self._drop_pending_jobs()
		job = {
			"job_id": str(uuid.uuid4()),
			"text": text,
			"priority": int(priority),
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
		latest_only_queue=True,
		use_runtime_synthesis=DEFAULT_TTS_USE_RUNTIME_SYNTHESIS,
		enable_priority_preempt=DEFAULT_TTS_ENABLE_PRIORITY_PREEMPT,
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
		self.latest_only_queue = latest_only_queue
		self.use_runtime_synthesis = bool(use_runtime_synthesis)
		self.enable_priority_preempt = bool(enable_priority_preempt)
		self.worker = None

		self._state = {
			"spoken": False,
			"last_command": None,
			"last_spoken_at": 0.0,
		}

		if self.enable_async:
			can_use_process_worker = (
				self.use_process_worker
				and self.use_runtime_synthesis
				and (not self.enable_priority_preempt)
			)

			if can_use_process_worker:
				self.worker = _ProcessTTSWorker(
					tts_engine=self.tts_engine,
					direct_playback=self.direct_playback,
					play_audio=self.play_audio,
					queue_maxsize=queue_maxsize,
					cache_max_items=self.phrase_cache_maxsize,
					latest_only_queue=self.latest_only_queue,
				)
			else:
				self.worker = _AsyncTTSWorker(
					tts_engine=self.tts_engine,
					direct_playback=self.direct_playback,
					play_audio=self.play_audio,
					queue_maxsize=queue_maxsize,
					latest_only_queue=self.latest_only_queue,
					use_runtime_synthesis=self.use_runtime_synthesis,
					enable_priority_preempt=self.enable_priority_preempt,
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
		priority = get_tts_command_priority(nav_command)

		if self.enable_async and self.worker is not None:
			if isinstance(self.worker, _ProcessTTSWorker):
				self.worker.poll_results()
			enqueue_ok = self.worker.enqueue(nav_command, priority=priority)
			result = {
				"tts_mode": "async_process_queue" if isinstance(self.worker, _ProcessTTSWorker) else "async_priority_queue",
				"tts_enqueue_ok": enqueue_ok,
				"tts_error": None,
				"tts_cache_hit": None,
				"tts_worker_latency_ms": None,
				"tts_priority": priority,
			}
			if isinstance(self.worker, _ProcessTTSWorker) and self.worker.last_result is not None:
				result["tts_cache_hit"] = self.worker.last_result.get("cache_hit")
				result["tts_worker_latency_ms"] = self.worker.last_result.get("worker_latency_ms")
			if isinstance(self.worker, _AsyncTTSWorker):
				result["tts_mode"] = self.worker.last_mode or result["tts_mode"]
			if not enqueue_ok and self.worker.last_error:
				result["tts_error"] = self.worker.last_error
			return result

		if self.use_runtime_synthesis:
			mode = self.tts_engine.speak_runtime_interruptible(
				nav_command,
				direct_playback=self.direct_playback,
				play_audio=self.play_audio,
			)
			return {
				"tts_mode": mode,
				"tts_enqueue_ok": None,
				"tts_error": None,
				"tts_cache_hit": None,
				"tts_priority": priority,
				"tts_worker_latency_ms": None,
			}

		mode = self.tts_engine.play_local_phrase_interruptible(nav_command)
		return {
			"tts_mode": mode,
			"tts_enqueue_ok": None,
			"tts_error": None,
			"tts_cache_hit": None,
			"tts_priority": priority,
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
			"tts_priority": None,
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
