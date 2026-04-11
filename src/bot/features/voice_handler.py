"""Handle voice message transcription via AssemblyAI, Mistral (Voxtral), OpenAI (Whisper), or local whisper.cpp."""

import asyncio
import json
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import structlog
from telegram import Voice

from src.config.settings import Settings

logger = structlog.get_logger(__name__)


@dataclass
class ProcessedVoice:
    """Result of voice message processing."""

    prompt: str
    transcription: str
    duration: int


class VoiceHandler:
    """Transcribe Telegram voice messages using AssemblyAI, Mistral, OpenAI, or local whisper.cpp."""

    # Timeout (seconds) for ffmpeg and whisper.cpp subprocess calls.
    LOCAL_SUBPROCESS_TIMEOUT: int = 120

    # Timeout (seconds) for AssemblyAI polling.
    ASSEMBLYAI_POLL_TIMEOUT: int = 120
    ASSEMBLYAI_POLL_INTERVAL: float = 1.5

    def __init__(self, config: Settings):
        self.config = config
        self._mistral_client: Optional[Any] = None
        self._openai_client: Optional[Any] = None
        self._resolved_whisper_binary: Optional[str] = None
        self._assemblyai_keyterms: Optional[list[str]] = None

    def _ensure_allowed_file_size(self, file_size: Optional[int]) -> None:
        """Reject files that exceed the configured max size."""
        if (
            isinstance(file_size, int)
            and file_size > self.config.voice_max_file_size_bytes
        ):
            raise ValueError(
                "Voice message too large "
                f"({file_size / 1024 / 1024:.1f}MB). "
                f"Max allowed: {self.config.voice_max_file_size_mb}MB. "
                "Adjust VOICE_MAX_FILE_SIZE_MB if needed."
            )

    async def process_voice_message(
        self, voice: Voice, caption: Optional[str] = None
    ) -> ProcessedVoice:
        """Download and transcribe a voice message.

        1. Download .ogg bytes from Telegram
        2. Call the configured transcription provider (Mistral, OpenAI, or local)
        3. Build a prompt combining caption + transcription
        """
        initial_file_size = getattr(voice, "file_size", None)
        self._ensure_allowed_file_size(initial_file_size)

        # Resolve Telegram file metadata before downloading bytes.
        file = await voice.get_file()
        resolved_file_size = getattr(file, "file_size", None)
        self._ensure_allowed_file_size(resolved_file_size)

        # Refuse unknown-size payloads to avoid unbounded downloads.
        if not isinstance(initial_file_size, int) and not isinstance(
            resolved_file_size, int
        ):
            raise ValueError(
                "Unable to determine voice message size before download. "
                "Please retry with a smaller voice message."
            )

        # Download voice data
        voice_bytes = bytes(await file.download_as_bytearray())
        self._ensure_allowed_file_size(len(voice_bytes))

        logger.info(
            "Transcribing voice message",
            provider=self.config.voice_provider,
            duration=voice.duration,
            file_size=initial_file_size or resolved_file_size or len(voice_bytes),
        )

        if self.config.voice_provider == "assemblyai":
            transcription = await self._transcribe_assemblyai(voice_bytes)
        elif self.config.voice_provider == "local":
            transcription = await self._transcribe_local(voice_bytes)
        elif self.config.voice_provider == "openai":
            transcription = await self._transcribe_openai(voice_bytes)
        else:
            transcription = await self._transcribe_mistral(voice_bytes)

        logger.info(
            "Voice transcription complete",
            transcription_length=len(transcription),
            duration=voice.duration,
        )

        # Build prompt
        label = caption if caption else "Voice message transcription:"
        prompt = f"{label}\n\n{transcription}"

        dur = voice.duration
        duration_secs = int(dur.total_seconds()) if isinstance(dur, timedelta) else dur

        return ProcessedVoice(
            prompt=prompt,
            transcription=transcription,
            duration=duration_secs,
        )

    # -- AssemblyAI provider --

    def _load_keyterms(self) -> list[str]:
        """Load keyterms from JSON file for AssemblyAI vocabulary boost."""
        if self._assemblyai_keyterms is not None:
            return self._assemblyai_keyterms

        keyterms_path = self.config.assemblyai_keyterms_path
        if not keyterms_path or not Path(keyterms_path).is_file():
            self._assemblyai_keyterms = []
            return self._assemblyai_keyterms

        with open(keyterms_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # keyterms file format: {"term": frequency, ...}
        self._assemblyai_keyterms = list(data.keys())
        logger.info(
            "Loaded AssemblyAI keyterms",
            count=len(self._assemblyai_keyterms),
        )
        return self._assemblyai_keyterms

    async def _transcribe_assemblyai(self, voice_bytes: bytes) -> str:
        """Transcribe audio using AssemblyAI REST API with keyterms_prompt."""
        api_key = self.config.assemblyai_api_key_str
        if not api_key:
            raise RuntimeError("AssemblyAI API key is not configured.")

        headers = {"authorization": api_key}

        # Step 1: Upload audio
        upload_url = await self._assemblyai_upload(voice_bytes, headers)

        # Step 2: Request transcription
        keyterms = self._load_keyterms()
        transcript_request: dict[str, Any] = {
            "audio_url": upload_url,
            "speech_models": ["universal-3-pro"],
            "language_detection": True,
        }
        if keyterms:
            transcript_request["keyterms_prompt"] = keyterms

        req_data = json.dumps(transcript_request).encode("utf-8")
        req = urllib.request.Request(
            "https://api.assemblyai.com/v2/transcript",
            data=req_data,
            headers={**headers, "content-type": "application/json"},
            method="POST",
        )

        loop = asyncio.get_event_loop()
        response_data = await loop.run_in_executor(
            None, self._assemblyai_http_call, req
        )
        transcript_id = response_data["id"]

        # Step 3: Poll for completion
        text = await self._assemblyai_poll(transcript_id, headers)
        return text

    async def _assemblyai_upload(
        self, voice_bytes: bytes, headers: dict[str, str]
    ) -> str:
        """Upload audio bytes to AssemblyAI and return the upload URL."""
        req = urllib.request.Request(
            "https://api.assemblyai.com/v2/upload",
            data=voice_bytes,
            headers={**headers, "content-type": "application/octet-stream"},
            method="POST",
        )
        loop = asyncio.get_event_loop()
        response_data = await loop.run_in_executor(
            None, self._assemblyai_http_call, req
        )
        upload_url = response_data.get("upload_url")
        if not upload_url:
            raise RuntimeError("AssemblyAI upload did not return an upload_url.")
        return upload_url

    async def _assemblyai_poll(
        self, transcript_id: str, headers: dict[str, str]
    ) -> str:
        """Poll AssemblyAI until transcription completes or fails."""
        url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
        req = urllib.request.Request(url, headers=headers, method="GET")
        loop = asyncio.get_event_loop()

        elapsed = 0.0
        while elapsed < self.ASSEMBLYAI_POLL_TIMEOUT:
            response_data = await loop.run_in_executor(
                None, self._assemblyai_http_call, req
            )
            status = response_data.get("status")

            if status == "completed":
                text = (response_data.get("text") or "").strip()
                if not text:
                    raise ValueError(
                        "AssemblyAI transcription completed but returned empty text."
                    )
                return text
            elif status == "error":
                error_msg = response_data.get("error", "Unknown error")
                raise RuntimeError(
                    f"AssemblyAI transcription failed: {error_msg}"
                )

            await asyncio.sleep(self.ASSEMBLYAI_POLL_INTERVAL)
            elapsed += self.ASSEMBLYAI_POLL_INTERVAL

        raise RuntimeError(
            f"AssemblyAI transcription timed out after {self.ASSEMBLYAI_POLL_TIMEOUT}s."
        )

    def _assemblyai_http_call(self, req: urllib.request.Request) -> dict[str, Any]:
        """Execute a synchronous HTTP request and return parsed JSON."""
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:200]
            raise RuntimeError(
                f"AssemblyAI HTTP {exc.code}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"AssemblyAI request failed: {exc.reason}"
            ) from exc

    # -- Mistral provider --

    async def _transcribe_mistral(self, voice_bytes: bytes) -> str:
        """Transcribe audio using the Mistral API (Voxtral)."""
        client = self._get_mistral_client()
        try:
            response = await client.audio.transcriptions.complete_async(
                model=self.config.resolved_voice_model,
                file={
                    "content": voice_bytes,
                    "file_name": "voice.ogg",
                },
            )
        except Exception as exc:
            logger.warning(
                "Mistral transcription request failed",
                error_type=type(exc).__name__,
            )
            raise RuntimeError("Mistral transcription request failed.") from exc

        text = (getattr(response, "text", "") or "").strip()
        if not text:
            raise ValueError("Mistral transcription returned an empty response.")
        return text

    def _get_mistral_client(self) -> Any:
        """Create and cache a Mistral client on first use."""
        if self._mistral_client is not None:
            return self._mistral_client

        try:
            from mistralai import Mistral
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Optional dependency 'mistralai' is missing for voice transcription. "
                "Install voice extras: "
                'pip install "claude-code-telegram[voice]"'
            ) from exc

        api_key = self.config.mistral_api_key_str
        if not api_key:
            raise RuntimeError("Mistral API key is not configured.")

        self._mistral_client = Mistral(api_key=api_key)
        return self._mistral_client

    # -- OpenAI provider --

    async def _transcribe_openai(self, voice_bytes: bytes) -> str:
        """Transcribe audio using the OpenAI Whisper API."""
        client = self._get_openai_client()
        try:
            response = await client.audio.transcriptions.create(
                model=self.config.resolved_voice_model,
                file=("voice.ogg", voice_bytes),
            )
        except Exception as exc:
            logger.warning(
                "OpenAI transcription request failed",
                error_type=type(exc).__name__,
            )
            raise RuntimeError("OpenAI transcription request failed.") from exc

        text = (getattr(response, "text", "") or "").strip()
        if not text:
            raise ValueError("OpenAI transcription returned an empty response.")
        return text

    def _get_openai_client(self) -> Any:
        """Create and cache an OpenAI client on first use."""
        if self._openai_client is not None:
            return self._openai_client

        try:
            from openai import AsyncOpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Optional dependency 'openai' is missing for voice transcription. "
                "Install voice extras: "
                'pip install "claude-code-telegram[voice]"'
            ) from exc

        api_key = self.config.openai_api_key_str
        if not api_key:
            raise RuntimeError("OpenAI API key is not configured.")

        self._openai_client = AsyncOpenAI(api_key=api_key)
        return self._openai_client

    # -- Local whisper.cpp provider --

    async def _transcribe_local(self, voice_bytes: bytes) -> str:
        """Transcribe audio locally using whisper.cpp binary."""
        binary = self._resolve_whisper_binary()
        model_path = self.config.resolved_whisper_cpp_model_path

        if not Path(model_path).is_file():
            raise RuntimeError(
                f"whisper.cpp model not found at {model_path}. "
                "Download it with: "
                "curl -L -o ~/.cache/whisper-cpp/ggml-base.bin "
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"
            )

        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="voice_")
            ogg_path = Path(tmp_dir) / "voice.ogg"
            wav_path = Path(tmp_dir) / "voice.wav"

            ogg_path.write_bytes(voice_bytes)

            # Convert OGG/Opus -> WAV (16kHz mono PCM)
            await self._convert_ogg_to_wav(ogg_path, wav_path)

            # Run whisper.cpp
            text = await self._run_whisper_cpp(binary, model_path, wav_path)

        finally:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        text = text.strip()
        if not text:
            raise ValueError(
                "Local whisper.cpp transcription returned an empty response."
            )
        return text

    async def _convert_ogg_to_wav(self, ogg_path: Path, wav_path: Path) -> None:
        """Convert OGG/Opus to WAV (16kHz mono PCM) using ffmpeg."""
        try:
            process = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-i",
                str(ogg_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "wav",
                str(wav_path),
                "-y",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.LOCAL_SUBPROCESS_TIMEOUT,
            )

            if process.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg conversion failed (exit {process.returncode}): "
                    f"{stderr.decode()[:200]}"
                )
        except asyncio.TimeoutError:
            process.kill()
            raise RuntimeError(
                f"ffmpeg conversion timed out after {self.LOCAL_SUBPROCESS_TIMEOUT}s."
            )
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg is required for local voice transcription but was not found. "
                "Install it with: apt install ffmpeg"
            )

    async def _run_whisper_cpp(
        self, binary: str, model_path: str, wav_path: Path
    ) -> str:
        """Execute whisper.cpp binary and return transcription text."""
        try:
            process = await asyncio.create_subprocess_exec(
                binary,
                "-m",
                model_path,
                "-f",
                str(wav_path),
                "--no-timestamps",
                "-l",
                "auto",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.LOCAL_SUBPROCESS_TIMEOUT,
            )

            if process.returncode != 0:
                logger.warning(
                    "whisper.cpp transcription failed",
                    return_code=process.returncode,
                    stderr=stderr.decode()[:300],
                )
                raise RuntimeError("Local whisper.cpp transcription failed.")

            return stdout.decode()

        except asyncio.TimeoutError:
            process.kill()
            raise RuntimeError(
                f"whisper.cpp transcription timed out after "
                f"{self.LOCAL_SUBPROCESS_TIMEOUT}s."
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"whisper.cpp binary not found at '{binary}'. "
                "Set WHISPER_CPP_BINARY_PATH or install whisper.cpp."
            )
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning(
                "whisper.cpp transcription request failed",
                error_type=type(exc).__name__,
            )
            raise RuntimeError("Local whisper.cpp transcription failed.") from exc

    def _resolve_whisper_binary(self) -> str:
        """Resolve and validate the whisper.cpp binary path on first use."""
        if self._resolved_whisper_binary is not None:
            return self._resolved_whisper_binary

        binary = self.config.resolved_whisper_cpp_binary
        resolved = shutil.which(binary)
        if not resolved:
            raise RuntimeError(
                f"whisper.cpp binary '{binary}' not found on PATH. "
                "Set WHISPER_CPP_BINARY_PATH to the full path."
            )
        self._resolved_whisper_binary = resolved
        return resolved
