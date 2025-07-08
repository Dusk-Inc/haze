from uuid import UUID, uuid4
from typing import Callable
from ..threader.core import Threader
from ..entity.core import Entity

class Context(Threader, Entity):
    def __init__(
            self, 
            feature_id: UUID = None, 
            on_finish: Callable[[UUID], None] = None
        ):
        Threader.__init__(self)
        Entity.__init__(self, id=feature_id)
        self.active = True
        self.energy_log = []
        self.on_finish = on_finish

    def enqueue(self, func: Callable, *args: tuple, **kwargs: dict):
        super().enqueue((func, args, kwargs))

    def run(self):
        while not self._queue.empty():
            func, args, kwargs = self._queue.get()
            try:
                func(*args, **kwargs)
            finally:
                self._queue.task_done()

        self.active = False
        if self.on_finish:
            self.on_finish(self.get_id())