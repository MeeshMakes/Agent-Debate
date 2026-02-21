"""Text-to-Speech engine using Windows SAPI5 voices (Zira / David).

Design
------
* One pyttsx3 engine is created per utterance -- avoids SAPI5 loop-state
  corruption that causes silence after the first message.
* The worker runs a heartbeat loop: after exhausting the current queue it
  sleeps 400 ms and checks again, so messages added DURING a live debate
  are automatically picked up and read.
* Voices are discovered once at start (fast probe engine) and cached.
* Stop / pause are honoured between each utterance.
* Immediate interrupt: engine.stop() is called on the live engine ref.
"""
from __future__ import annotations

import threading
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal


class TTSPlaybackWorker(QThread):
    """Reads (agent, text) pairs using Windows SAPI5 voices.

    Astra  ->  Microsoft Zira  (female)
    Nova   ->  Microsoft David (male)
    """

    now_speaking = pyqtSignal(int, str)   # (message_index, agent_name)
    word_at      = pyqtSignal(int, int, int)  # (message_index, char_offset, word_len)
    finished_all = pyqtSignal()           # exhausted queue with no more heartbeat
    finished_one = pyqtSignal(int)        # message_index just completed
    error        = pyqtSignal(str)

    # agent name -> SAPI5 voice search token
    _VOICE_MAP: dict[str, str] = {"Astra": "zira", "Nova": "david"}

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._messages: list[tuple[str, str]] = []
        self._rate: int = 185
        self._start_index: int = 0

        self._stop_flag  = False
        self._pause_flag = False
        self._heartbeat  = True   # keep looping after queue exhausted
        self._lock = threading.Lock()

        self._live_engine: Optional[object] = None
        self._cur_msg_idx: int = 0   # which message is currently being spoken

    # ---------------------------------------------------------------- public API

    def load_messages(self, messages: list[tuple[str, str]], start_index: int = 0) -> None:
        """Replace the message queue (call before starting the thread)."""
        with self._lock:
            self._messages   = list(messages)
            self._start_index = start_index

    def add_message(self, agent: str, text: str) -> None:
        """Append a message while the worker is running; heartbeat picks it up."""
        with self._lock:
            self._messages.append((agent, text))

    def set_rate(self, rate: int) -> None:
        with self._lock:
            self._rate = max(50, min(400, rate))

    def request_stop(self) -> None:
        with self._lock:
            self._stop_flag  = True
            self._pause_flag = False
            self._heartbeat  = False
        engine = self._live_engine
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    def toggle_pause(self) -> None:
        with self._lock:
            self._pause_flag = not self._pause_flag

    @property
    def is_paused(self) -> bool:
        return self._pause_flag

    # ---------------------------------------------------------------- worker

    def run(self) -> None:  # noqa: C901
        try:
            import pyttsx3
        except ImportError:
            self.error.emit("pyttsx3 not installed -- TTS disabled")
            return

        # ---- discover SAPI5 voice IDs with a throw-away engine ----
        voice_ids: dict[str, str] = {}
        try:
            probe = pyttsx3.init()
            for v in probe.getProperty("voices"):
                nl = v.name.lower()
                if "zira"  in nl: voice_ids["zira"]  = v.id
                if "david" in nl: voice_ids["david"] = v.id
            try:    probe.stop()
            except Exception: pass
            del probe
        except Exception as exc:
            self.error.emit(f"TTS voice discovery failed: {exc}")
            return

        self._stop_flag = False
        cur = self._start_index

        while True:
            # --- pause gate ---
            while self._pause_flag:
                self.msleep(120)
                if self._stop_flag:
                    self.finished_all.emit()
                    return

            if self._stop_flag:
                break

            # --- check queue ---
            with self._lock:
                pending = len(self._messages)

            if cur >= pending:
                if not self._heartbeat:
                    break
                self.msleep(400)
                continue

            with self._lock:
                agent, text = self._messages[cur]
                rate = self._rate

            self.now_speaking.emit(cur, agent)
            self._cur_msg_idx = cur

            vid = voice_ids.get(self._VOICE_MAP.get(agent, "david"), "")
            ok  = self._speak_one(pyttsx3, text, vid, rate)

            if not ok or self._stop_flag:
                break

            self.finished_one.emit(cur)
            cur += 1

        self._live_engine = None
        self.finished_all.emit()

    # ---------------------------------------------------------------- internals

    @staticmethod
    def clean_text(text: str) -> str:
        """Return TTS-safe version of a message (same transform used for say())."""
        return (
            text.replace("TRUTH:",     "Truth:")
                .replace("PROBLEM:",   "Problem:")
                .replace("SUB-TOPIC:", "Sub topic:")
                .replace("VERIFY:",    "Verify:")
                .replace("**", "").replace("__", "")
                .replace("——", ",").replace("—", ",")
        )

    def _speak_one(self, pyttsx3_mod, text: str, voice_id: str, rate: int) -> bool:
        """Speak one utterance with a fresh pyttsx3 engine. Returns True on success."""
        engine = None
        try:
            engine = pyttsx3_mod.init()
            self._live_engine = engine

            if voice_id:
                try:    engine.setProperty("voice", voice_id)
                except Exception: pass
            try:    engine.setProperty("rate", rate)
            except Exception: pass

            clean = self.clean_text(text)

            # Connect word-boundary callback so we can emit word_at
            msg_idx = self._cur_msg_idx

            def _on_word(name, location, length,
                         _idx=msg_idx, _sig=self.word_at):
                try:
                    _sig.emit(_idx, location, length)
                except Exception:
                    pass

            try:
                engine.connect("started-word", _on_word)
            except Exception:
                pass

            engine.say(clean)
            engine.runAndWait()   # blocks QThread only -- UI stays responsive
            return True

        except Exception as exc:
            self.error.emit(f"TTS utterance error: {exc}")
            return False

        finally:
            if engine is not None:
                try:    engine.stop()
                except Exception: pass
            self._live_engine = None
