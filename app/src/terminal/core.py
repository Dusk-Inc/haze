from ..connector.core import Connector
from .errors import NoMatchingConnectionError
from.errors import IdenticalConnectionError

class Terminal():
    def __init__(self):
        super().__init__()
        self._connections: list[Connector] = []

    def put_connection(self, connection: Connector) -> None:
        for x, c in enumerate(self._connections):
            if c.get_id() == connection.get_id():
                self._connections[x] = connection
                return

        raise NoMatchingConnectionError(f"Error: there is no connection with the id: {connection.get_id()} found in connection list.")

    def get_connections(self) -> list[Connector]:
        return self._connections

    def post_connection(self, connection: Connector) -> None:
        new_dendrite = connection.get_dendrite()

        for c in self._connections:
            if c.get_id() == connection.get_id():
                raise IdenticalConnectionError(f"Error: there is already a connection with the id of {connection.get_id()} registered to the connection list.")
            
            existing_dendrite = c.get_dendrite()
            if existing_dendrite.get_id() == new_dendrite.get_id():
                raise IdenticalConnectionError(f"Error: this connection would add a duplicate signal to an existing dendritic connection.")
        
        self._connections.append(connection)
        self.choose_default()

    def delete_connection(self, connection: Connector) -> None:
        for x, c in enumerate(self._connections):
            if c.get_id() == connection.get_id():
                del self._connections[x]
                return
        
        raise NoMatchingConnectionError(f"Error: there is no connection with the id: {connection.get_id()} found in connection list.")
    
    def clear_connections(self):
        self._connections = []
    
    def choose_default(self):
        x = 0

        while x < len(self._connections):
            if x > 0:
                if self._connections[x].get_strength() > self._connections[x-1].get_strength():
                    self._connections[x-1].is_default = False
                    self._connections[x].is_default = True
            else:
                self._connections[0].is_default = True
            x += 1