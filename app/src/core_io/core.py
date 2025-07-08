import json
import os

# sample file structure
# /core
# --/encoders
# --/--/numberic.json
# --/--/--/<sensor_id>.json
# --/decoders
# --/--/argmax
# --/--/--/<motor_id>.json
# --/meshes
# --/--/aperture
# --/--/--/<inter_id>.json
# --/--/nexus
# --/--/terminus
# --/connections
# --/--/<connection_id>.json

class CoreIO:
    def __init__(self, path: str = None, persist: bool = False):
        self._path = path or os.path.join(os.getcwd(), 'app', 'core')
        self._connection_path = os.path.join(self._path, "connections")
        self._mesh_path = os.path.join(self._path, 'meshes')
        self._aperture_path = os.path.join(self._mesh_path, 'aperture')
        self._nexus_path = os.path.join(self._mesh_path, 'nexus')
        self._terminus_path = os.path.join(self._mesh_path, 'terminus')
        self._encoder_path = os.path.join(self._path, 'encoders')
        self._decoder_path = os.path.join(self._path, 'decoders')
        self._persist = persist

    def get_neuron_data(self, path: str) -> list[dict]:
        neuron_files = os.listdir(path)
        return [self.load_from_file(os.path.join(path, neuron)) for neuron in neuron_files]

    def get_connection_data(self, connection_id: str) -> dict:
        return self.load_from_file(os.path.join(self._connection_path, f"{connection_id}.json"))
    
    def _create_directory(self, path: str):
        path, filename = os.path.split(path)
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as error:
            print(f"Error creating directory '{path}': {error}")

    def load_from_file(self, path):
            with open(path, 'r') as file:
                return json.load(file)

    def save_to_file(self, data: dict, path: str):
        if not self._persist:
            return 
        
        self._create_directory(path)
        with open(f"{path}.json", "w") as file:
            json.dump(data, file, indent=4)

    def remove_from_file(self, file_name: str, path: str):
        if not self._persist:
            return 
        
        file_path = os.path.join(path, f"{file_name}.json")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"File '{file_path}' has been removed.")
            except OSError as error:
                print(f"Error removing file '{file_path}': {error}")
        else:
            print(f"File '{file_path}' does not exist.")


    def _is_dir_empty(self, path) -> bool:
        return any(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))

    def is_empty(self) -> bool:
        if not self._persist:
            return True
        
        if os.path.exists(self._connection_path) and self._is_dir_empty(self._connection_path):
            return False
        if os.path.exists(self._aperture_path) and self._is_dir_empty(self._aperture_path):
            return False
        if os.path.exists(self._nexus_path) and self._is_dir_empty(self._nexus_path):
            return False
        if os.path.exists(self._terminus_path) and self._is_dir_empty(self._terminus_path):
            return False
        return True
