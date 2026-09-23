"""OpenCV HighGUI worker, intentionally isolated from MuJoCo on macOS."""
from __future__ import annotations

import queue


def preview_worker(frame_queue, stop_event, shutdown_event, window_name="Right-hand tracking"):
    import cv2
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            while not stop_event.is_set():
                try:
                    frame = frame_queue.get(timeout=.1)
                except queue.Empty:
                    continue
                if frame is None:
                    break
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    shutdown_event.set()
                    break
        except KeyboardInterrupt:
            pass
    finally:
        cv2.destroyAllWindows()


class CameraPreview:
    """Latest-only, non-blocking frame transport to a spawned UI process."""
    def __init__(self, context=None):
        import multiprocessing as mp
        self.context = context or mp.get_context("spawn")
        self.queue = self.context.Queue(maxsize=1)
        self.stop_event = self.context.Event()
        self.shutdown_event = self.context.Event()
        self.process = self.context.Process(
            target=preview_worker, args=(self.queue, self.stop_event, self.shutdown_event),
            name="aero-camera-preview", daemon=True,
        )
        self.process.start()

    @property
    def shutdown_requested(self):
        return self.shutdown_event.is_set()

    def publish(self, frame) -> bool:
        if not self.process.is_alive():
            return False
        try:
            self.queue.put_nowait(frame)
            return True
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(frame)
            except queue.Full:
                pass
            return True

    def close(self):
        self.stop_event.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.process.join(timeout=1.5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
        self.queue.close()
