import io
import json
import os
import subprocess
import time
import wave
from pathlib import Path
from typing import Optional


class PiperTTS:
	"""Lightweight Piper-based TTS utility for navigation commands."""

	def __init__(
		self,
		piper_executable: str,
		voice_model_path: str,
		voice_config_path: Optional[str] = None,
		output_dir: Optional[str] = None,
	):
		self.piper_executable = Path(piper_executable)
		self.voice_model_path = Path(voice_model_path)
		self.voice_config_path = Path(voice_config_path) if voice_config_path else None

		if output_dir is None:
			default_dir = Path(__file__).resolve().parent / "audio_outputs"
			self.output_dir = default_dir
		else:
			self.output_dir = Path(output_dir)

		self.output_dir.mkdir(parents=True, exist_ok=True)
		self._validate_paths()

	def _validate_paths(self) -> None:
		if not self.piper_executable.exists():
			raise FileNotFoundError(f"Piper executable not found: {self.piper_executable}")
		if not self.voice_model_path.exists():
			raise FileNotFoundError(f"Piper voice model not found: {self.voice_model_path}")
		if self.voice_config_path is not None and not self.voice_config_path.exists():
			raise FileNotFoundError(f"Piper voice config not found: {self.voice_config_path}")

	@staticmethod
	def _normalize_text(text: str) -> str:
		cleaned = " ".join((text or "").strip().split())
		if not cleaned:
			raise ValueError("Navigation command text is empty.")
		return cleaned

	def _base_cmd(self):
		cmd = [
			str(self.piper_executable),
			"--model",
			str(self.voice_model_path),
		]
		if self.voice_config_path is not None:
			cmd.extend(["--config", str(self.voice_config_path)])
		return cmd

	def _base_cmd_raw(self):
		return self._base_cmd() + ["--output_raw"]

	def _sample_rate(self) -> int:
		if self.voice_config_path is None:
			return 22050
		try:
			with open(self.voice_config_path, "r", encoding="utf-8") as f:
				cfg = json.load(f)
			return int(cfg.get("audio", {}).get("sample_rate", 22050))
		except Exception:
			return 22050

	def _pcm_to_wav_bytes(self, pcm_bytes: bytes) -> bytes:
		buffer = io.BytesIO()
		with wave.open(buffer, "wb") as wav_file:
			wav_file.setnchannels(1)
			wav_file.setsampwidth(2)
			wav_file.setframerate(self._sample_rate())
			wav_file.writeframes(pcm_bytes)
		return buffer.getvalue()

	@staticmethod
	def _is_valid_wav_bytes(wav_bytes: bytes) -> bool:
		if not wav_bytes or wav_bytes[:4] != b"RIFF":
			return False
		try:
			with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
				channels = wav_file.getnchannels()
				width = wav_file.getsampwidth()
				frames = wav_file.getnframes()
				return channels >= 1 and width in (1, 2, 3, 4) and frames > 0
		except Exception:
			return False

	def synthesize_wav_bytes(self, text: str) -> bytes:
		"""Generate WAV bytes fully in memory (no file write)."""
		command_text = self._normalize_text(text)

		# Attempt 1: Ask Piper to write WAV bytes to stdout.
		cmd_wav = self._base_cmd() + ["--output_file", "-"]
		result_wav = subprocess.run(
			cmd_wav,
			input=command_text.encode("utf-8"),
			capture_output=True,
			check=False,
		)
		if result_wav.returncode == 0 and self._is_valid_wav_bytes(result_wav.stdout):
			return result_wav.stdout

		# Attempt 2: Get raw PCM bytes and wrap into WAV in memory.
		cmd_raw = self._base_cmd() + ["--output_raw"]
		result_raw = subprocess.run(
			cmd_raw,
			input=command_text.encode("utf-8"),
			capture_output=True,
			check=False,
		)
		if result_raw.returncode != 0:
			raise RuntimeError(
				"Piper in-memory synthesis failed. "
				f"stderr={result_raw.stderr.decode(errors='ignore').strip()}"
			)
		wav_bytes = self._pcm_to_wav_bytes(result_raw.stdout)
		if not self._is_valid_wav_bytes(wav_bytes):
			raise RuntimeError("Generated in-memory WAV bytes are invalid.")
		return wav_bytes

	def speak_direct(self, text: str) -> None:
		"""Speak directly from generated binary audio without saving WAV file."""
		if os.name != "nt":
			raise RuntimeError("Direct in-memory playback helper currently supports Windows only.")
		wav_bytes = self.synthesize_wav_bytes(text)
		self.play_bytes(wav_bytes)

	def speak_stream(self, text: str) -> None:
		"""Stream Piper raw PCM directly to speaker with sounddevice.

		This avoids WAV file IO and uses the correct sample rate from model config.
		"""
		command_text = self._normalize_text(text)
		try:
			import numpy as np  # pylint: disable=import-outside-toplevel
			import sounddevice as sd  # pylint: disable=import-outside-toplevel
		except ImportError as exc:
			raise RuntimeError(
				"sounddevice and numpy are required for raw stream playback. "
				"Install with: pip install sounddevice numpy"
			) from exc

		proc = subprocess.Popen(
			self._base_cmd_raw(),
			stdin=subprocess.PIPE,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
		)
		stdout_bytes, stderr_bytes = proc.communicate(input=command_text.encode("utf-8"))

		if proc.returncode != 0:
			raise RuntimeError(
				"Piper raw stream failed. "
				f"stderr={stderr_bytes.decode(errors='ignore').strip()}"
			)

		audio_data = np.frombuffer(stdout_bytes, dtype=np.int16)
		if audio_data.size == 0:
			raise RuntimeError("Piper returned empty raw audio buffer.")

		sd.play(audio_data, self._sample_rate())
		sd.wait()

	def speak_direct_safe(self, text: str, fallback_play_audio: bool = True) -> str:
		"""Try direct memory playback, fallback to file synthesis if needed.

		Returns playback mode: 'raw-stream', 'direct-memory', or 'file-fallback'.
		"""
		try:
			self.speak_stream(text)
			return "raw-stream"
		except Exception:
			pass

		try:
			self.speak_direct(text)
			return "direct-memory"
		except Exception:
			self.synthesize(text, play_audio=fallback_play_audio)
			return "file-fallback"

	def synthesize(
		self,
		text: str,
		output_wav_path: Optional[str] = None,
		play_audio: bool = False,
	) -> str:
		"""Generate speech from text and optionally play it.

		Returns absolute output WAV path.
		"""
		command_text = self._normalize_text(text)

		if output_wav_path is None:
			ts = int(time.time() * 1000)
			wav_path = self.output_dir / f"nav_command_{ts}.wav"
		else:
			wav_path = Path(output_wav_path)
			wav_path.parent.mkdir(parents=True, exist_ok=True)

		cmd = [
			str(self.piper_executable),
			"--model",
			str(self.voice_model_path),
			"--output_file",
			str(wav_path),
		]
		if self.voice_config_path is not None:
			cmd.extend(["--config", str(self.voice_config_path)])

		result = subprocess.run(
			cmd,
			input=command_text,
			capture_output=True,
			text=True,
			check=False,
		)
		if result.returncode != 0:
			raise RuntimeError(
				"Piper synthesis failed. "
				f"stdout={result.stdout.strip()} stderr={result.stderr.strip()}"
			)

		if play_audio:
			self.play(str(wav_path))

		return str(wav_path.resolve())

	@staticmethod
	def play(wav_path: str) -> None:
		"""Play a WAV file on Windows using winsound."""
		if os.name != "nt":
			raise RuntimeError("Audio playback helper currently supports Windows only.")

		import winsound  # pylint: disable=import-outside-toplevel

		winsound.PlaySound(str(Path(wav_path).resolve()), winsound.SND_FILENAME)

	@staticmethod
	def play_bytes(wav_bytes: bytes) -> None:
		"""Play in-memory WAV bytes on Windows using winsound."""
		if os.name != "nt":
			raise RuntimeError("In-memory playback helper currently supports Windows only.")

		import winsound  # pylint: disable=import-outside-toplevel

		winsound.PlaySound(wav_bytes, winsound.SND_MEMORY)


if __name__ == "__main__":
	cwd = os.getcwd()
	
	PIPER_EXE = os.path.abspath(os.path.join(cwd, "app", "piper", "piper.exe"))
	VOICE_MODEL = os.path.abspath(os.path.join(cwd, "app", "piper_voices", "en_US-amy-medium.onnx"))
	VOICE_CONFIG = os.path.abspath(os.path.join(cwd, "app", "piper_voices", "en_US-amy-medium.onnx.json"))

	tts = PiperTTS(
		piper_executable=PIPER_EXE,
		voice_model_path=VOICE_MODEL,
		voice_config_path=VOICE_CONFIG,
	)
	wav_file = tts.synthesize("Clear path on your left. Turn left.", play_audio=False)
	print(f"Generated: {wav_file}")
