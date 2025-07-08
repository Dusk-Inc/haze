from threading import RLock
from queue import Queue, Empty
from typing import Any, Callable

class Threader:
    def __init__(self):
        super().__init__()
        self._lock = RLock()
        self._queue = Queue()

    def enqueue(self, item: Any) -> None:
        self._queue.put(item)

    def dequeue(self) -> Any:
        if self._queue.empty():
            raise Empty("User tried to dequeue an empty queue.")
        
        return self._queue.get()

    def synchronized(self, func: Callable) -> Callable:
        """Decorator to make a method thread-safe."""
        def wrapper(*args, **kwargs):
            with self._lock:
                return func(*args, **kwargs)
        return wrapper