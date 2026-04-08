import time
from app.pipeline.frame_context import FrameContext
from app.mechanics.text_to_speech import TextToSpeech, TTSRuntimeController

class TTSComponent:
    """
    Component responsible for managing the Text-To-Speech (TTS) system and announcing navigation commands.
    """
    def __init__(
        self,
        piper_exe: str,
        voice_model_path: str,
        voice_config_path: str,
        enable_async: bool = True,
        direct_playback: bool = True,
        play_audio: bool = False,
        speak_once_per_execution: bool = False,
        speak_on_command_change: bool = True,
        min_interval_seconds: float = 1.2,
        queue_maxsize: int = 5
    ):
        engine = TextToSpeech(
            piper_executable=piper_exe,
            voice_model_path=voice_model_path,
            voice_config_path=voice_config_path,
        )
        engine.load_engine()

        self.controller = TTSRuntimeController(
            tts_engine=engine,
            enable_async=enable_async,
            direct_playback=direct_playback,
            play_audio=play_audio,
            speak_once_per_execution=speak_once_per_execution,
            speak_on_command_change=speak_on_command_change,
            min_interval_seconds=min_interval_seconds,
            queue_maxsize=queue_maxsize,
        )

    def run(self, ctx: FrameContext) -> None:
        start_time = time.perf_counter()

        try:
            tts_result = self.controller.handle_command(ctx.nav_command)
            ctx.tts_result = tts_result

            ctx.metrics["tts_should_speak"] = tts_result.get("tts_should_speak", False)
            ctx.metrics["tts_skip_reason"] = tts_result.get("tts_skip_reason")
            ctx.metrics["tts_mode"] = tts_result.get("tts_mode")
            ctx.metrics["tts_error"] = tts_result.get("tts_error")
            ctx.metrics["tts_enqueue_ok"] = tts_result.get("tts_enqueue_ok")
        except Exception as exc:
            ctx.metrics["tts_error"] = str(exc)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        ctx.metrics["tts_latency_ms"] = latency_ms

    def stop(self):
        self.controller.stop()
