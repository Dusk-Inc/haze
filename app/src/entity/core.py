from uuid import UUID, uuid4
from typing import Union

class Entity:
    def __init__(self, id: UUID = None):
        if id is None:
            self._id = uuid4()
        elif type(id) == str:
            self._id = UUID(id)
        else:
            self._id = id

    def get_id(self, as_string=True):
        if as_string:
            return str(self._id)