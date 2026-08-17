from __future__ import annotations
import threading
from PySide6.QtCore import QThread

class WorkerThread(QThread):
    def __init__(self, loop) -> None:
        super().__init__()
        self._loop = loop
        self._stop = threading.Event()

    def run(self) -> None:  # executes in the new thread
        self._loop.run(self._stop)

    def stop(self) -> None:
        self._stop.set()
        # Stop the capturer too so a blocking grab() (bettercam waits for a changed
        # frame; a static screen never delivers one) unblocks and run() can exit
        # instead of hanging the 2 s join and aborting on a live QThread.
        try:
            self._loop.stop_capture()
        except Exception:
            pass
        self.wait(2000)
