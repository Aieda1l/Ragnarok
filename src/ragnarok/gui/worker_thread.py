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
        self.wait(2000)
