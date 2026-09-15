"""Share bounded NVENC video fragments from the existing local camera feeds."""

from collections import deque
import shutil
import struct
import subprocess
import threading
import time

CAMERAS = ("kitchen", "worktop", "side")
MAX_BOX_BYTES = 8 * 1024 * 1024


def boxes(stream):
    def exact(size):
        parts = bytearray()
        while len(parts) < size:
            part = stream.read(size - len(parts))
            if not part:
                raise EOFError("Video encoder closed its output")
            parts.extend(part)
        return bytes(parts)

    while True:
        header = exact(8)
        size, kind = struct.unpack(">I4s", header)
        if not 8 <= size <= MAX_BOX_BYTES:
            raise ValueError("Invalid video fragment size")
        yield kind, header + exact(size - 8)


def encoder_command(executable, origin, camera):
    if camera not in CAMERAS:
        raise ValueError("Unknown camera")
    return [executable, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-threads", "1", "-filter_threads", "1", "-rw_timeout", "7000000",
            "-probesize", "32768", "-analyzeduration", "0", "-r", "3",
            "-f", "mpjpeg", "-i", origin + "/stream/" + camera,
            "-an", "-c:v", "h264_nvenc", "-preset", "p4", "-tune", "ll",
            "-profile:v", "baseline", "-level:v", "3.1", "-pix_fmt", "yuv420p",
            "-b:v", "2M", "-maxrate", "2M", "-bufsize", "1M", "-rc", "cbr",
            "-g", "3", "-bf", "0", "-zerolatency", "1",
            "-movflags", "frag_keyframe+empty_moov+default_base_moof",
            "-f", "mp4", "pipe:1"]


class CameraVideo:
    def __init__(self, command):
        self.condition = threading.Condition()
        self.initialization = b""
        self.fragments = deque(maxlen=6)
        self.sequence = 0
        self.updated = time.monotonic()
        self.failed = False
        self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        initialization = bytearray()
        fragment = bytearray()
        try:
            for kind, data in boxes(self.process.stdout):
                if kind in (b"ftyp", b"moov"):
                    initialization.extend(data)
                    if kind == b"moov":
                        with self.condition:
                            self.initialization = bytes(initialization)
                            self.condition.notify_all()
                elif kind == b"moof":
                    fragment = bytearray(data)
                elif kind == b"mdat" and fragment:
                    fragment.extend(data)
                    with self.condition:
                        self.sequence += 1
                        self.fragments.append((self.sequence, bytes(fragment)))
                        self.updated = time.monotonic()
                        self.condition.notify_all()
                    fragment = bytearray()
        except (EOFError, ValueError, OSError):
            pass
        finally:
            with self.condition:
                self.failed = True
                self.condition.notify_all()
            if self.process.poll() is None:
                self.process.terminate()

    def chunks(self):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.failed or (self.initialization and self.fragments), timeout=12)
            if not ready or self.failed or time.monotonic() - self.updated > 5:
                raise TimeoutError("Camera video is unavailable")
            initial = self.initialization
            sequence, first = self.fragments[-1]
        yield initial
        yield first
        while True:
            with self.condition:
                ready = self.condition.wait_for(
                    lambda: self.failed or self.sequence > sequence, timeout=5)
                if not ready or self.failed:
                    raise TimeoutError("Camera video stopped advancing")
                pending = [item for item in self.fragments if item[0] > sequence]
            for sequence, fragment in pending:
                yield fragment

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        self.thread.join(timeout=2)
        self.process.stdout.close()


class VideoStreams:
    def __init__(self, origin, executable="ffmpeg"):
        self.executable = shutil.which(executable)
        if not self.executable:
            raise ValueError("NVENC video requires ffmpeg on the host")
        self.origin = origin
        self.streams = {}
        self.lock = threading.Lock()

    def get(self, camera):
        if camera not in CAMERAS:
            raise ValueError("Unknown camera")
        with self.lock:
            stream = self.streams.get(camera)
            if stream and (stream.failed or time.monotonic() - stream.updated > 12):
                stream.stop()
                stream = None
            if stream is None:
                stream = CameraVideo(encoder_command(self.executable, self.origin, camera))
                self.streams[camera] = stream
            return stream

    def stop(self):
        with self.lock:
            for stream in self.streams.values():
                stream.stop()
