"""The Haze model: a torch module whose shape changes as it learns."""

from pathlib import Path
from typing import Any

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from ..errors import CheckpointCorruptError, HazeError, LearningDisabledError
from ..functions.serialize import (
    ensureCheckpointCoherent,
    loadPortImpl,
    loadStateTensors,
    readHazeConfig,
    readHazeLabels,
    readHazeTensors,
    saveHazeCheckpoint,
    toStateTensors,
)
from ..models import ChainSpec, HazeConfig, MeshCapacity, Stage
from ..tokens import PortRole, defaults
from .mesh import MeshState
from .ports import PortRegistry


class Haze(nn.Module, PyTorchModelHubMixin):
    """A continuously learning, self-organizing neural mesh.

    Its neuron and edge counts change as it learns, so its buffers have no fixed shape. The
    config records the capacity they were allocated at and is read before any tensor, which is
    what lets strict loading succeed on a variable-shape model. See specs/packaging.md.
    """

    def __init__(self, config: HazeConfig | dict | None = None) -> None:
        """Builds a mesh at the config's capacity, wiring its interneurons if it is fresh."""
        super().__init__()
        if config is None:
            config = HazeConfig()
        elif isinstance(config, dict):
            config = HazeConfig.model_validate(config)
        self.config = config

        self.mesh = MeshState(config)
        self.ports = PortRegistry(self.mesh)
        self.learning_enabled = True

        if config.counts.nexus == 0 and config.counts.terminus == 0:
            self.mesh.buildMesh(config.nexus_size, config.terminus_size)


    def registerEncoder(self, key: str, encoder: Any) -> int:
        """Registers an encoder under a key and returns its port slot."""
        slot = self.ports.registerPort(
            key, encoder, PortRole.ENCODER, width=getattr(encoder, "width", None)
        )
        encoder.key = key
        if isinstance(encoder, nn.Module):
            self.add_module(f"encoder_{key}", encoder)
        return slot

    def registerDecoder(self, key: str, decoder: Any) -> int:
        """Registers a decoder under a key, allocating motors for any labels it declares."""
        slot = self.ports.registerPort(key, decoder, PortRole.DECODER)
        decoder.key = key
        if isinstance(decoder, nn.Module):
            self.add_module(f"decoder_{key}", decoder)
        labels = getattr(decoder, "labels", None)
        if labels:
            self.ports.setLabelsActive(key, labels)
        return slot

    def setChain(
        self,
        stages: list[Stage] | list[tuple[str, str]],
        sequential: bool = False,
        end_token: str = defaults.END_TOKEN,
    ) -> None:
        """Sets the ordered stages a prediction runs through."""
        built: list[Stage] = []
        for stage in stages:
            if isinstance(stage, Stage):
                built.append(stage)
            else:
                encoders, decoders = stage
                built.append(
                    Stage(
                        encoders=[encoders] if isinstance(encoders, str) else list(encoders),
                        decoders=[decoders] if isinstance(decoders, str) else list(decoders),
                    )
                )
        self.config.chain = ChainSpec(
            stages=built, sequential=sequential, end_token=end_token
        )


    def train(self, mode: bool = True) -> "Haze":
        """Enables learning, the fired trace, and structural change; eval disables all three.

        Haze has no backpropagation, so the standard mode switch is given an explicit meaning
        rather than inherited without one. See specs/packaging.md.
        """
        super().train(mode)
        self.learning_enabled = mode
        return self

    def ensureLearningEnabled(self) -> None:
        """Raises if learning is attempted on a model in eval mode."""
        if not self.learning_enabled:
            raise LearningDisabledError(
                "this model is in eval mode, which records no fired trace, so there is nothing "
                "to learn from. Call train() first."
            )


    def _save_pretrained(self, save_directory: Path) -> None:
        """Writes config, labels, and tensors as one atomically-replaced directory."""
        self.config.counts = self.mesh.toCounts()
        self.config.capacity = self.mesh.capacity
        self.config.ports = self.ports.toPortSpecs()
        saveHazeCheckpoint(
            Path(save_directory),
            self.config,
            toStateTensors(self.mesh),
            self.ports.toLabelTable(),
        )

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "Haze":
        """Loads a checkpoint, reporting a malformed one as a Haze error.

        The hub mixin parses config.json itself before delegating, so without this a corrupt
        config surfaces as a raw JSON error rather than naming the checkpoint.
        """
        try:
            return super().from_pretrained(*args, **kwargs)
        except HazeError:
            raise
        except (ValueError, OSError) as cause:
            target = args[0] if args else kwargs.get("pretrained_model_name_or_path", "checkpoint")
            raise CheckpointCorruptError(
                f"{target} could not be read as a Haze checkpoint: {cause}"
            ) from cause

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None = None,
        cache_dir: str | None = None,
        force_download: bool = False,
        proxies: dict | None = None,
        resume_download: bool | None = None,
        local_files_only: bool = False,
        token: str | bool | None = None,
        trust_remote_code: bool = False,
        **model_kwargs: Any,
    ) -> "Haze":
        """Reads the config, allocates at its capacity, then loads and cross-checks the payload."""
        directory = Path(model_id)
        if not directory.is_dir():
            directory = Path(
                cls._download_hub_checkpoint(
                    model_id=model_id,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                )
            )

        config = readHazeConfig(directory)
        labels = readHazeLabels(directory)
        tensors = readHazeTensors(directory)

        model = cls(config)
        loadStateTensors(model.mesh, tensors)
        model.mesh.counts = config.counts.model_copy(deep=True)

        ensureCheckpointCoherent(model.mesh, labels, config.ports)
        model.ports.loadLabelTable(labels)
        for spec in config.ports:
            model.ports.specs[spec.key] = spec
            model.ports.impls[spec.key] = loadPortImpl(spec, trust_remote_code=trust_remote_code)
        model.ports.loadSensorIds()
        return model

    @staticmethod
    def _download_hub_checkpoint(
        model_id: str,
        revision: str | None,
        cache_dir: str | None,
        force_download: bool,
        proxies: dict | None,
        local_files_only: bool,
        token: str | bool | None,
    ) -> str:
        """Fetches a checkpoint's files from the hub and returns the local directory holding them."""
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            allow_patterns=[
                defaults.CONFIG_FILE,
                defaults.LABELS_FILE,
                defaults.WEIGHTS_FILE,
            ],
        )


    def calcMeshSize(self) -> dict[str, int]:
        """Returns the live neuron and edge counts, for logging and benchmarks."""
        return self.mesh.toCounts().model_dump()

    def extra_repr(self) -> str:
        """Returns the live mesh size, so printing a model says what it currently is."""
        counts = self.mesh.counts
        return (
            f"sensors={counts.sensors}, motors={counts.motors}, nexus={counts.nexus}, "
            f"terminus={counts.terminus}, edges={counts.edges}"
        )


def makeHaze(
    nexus_size: int = defaults.DEFAULT_NEXUS_SIZE,
    terminus_size: int = defaults.DEFAULT_TERMINUS_SIZE,
    seed: int = defaults.DEFAULT_SEED,
    **hyper: Any,
) -> Haze:
    """Builds a Haze model sized for a mesh, without the caller assembling a config by hand."""
    config = HazeConfig(
        seed=seed,
        nexus_size=nexus_size,
        terminus_size=terminus_size,
        capacity=MeshCapacity(
            nexus=max(nexus_size, 1),
            terminus=max(terminus_size, 1),
        ),
    )
    if hyper:
        config.hyper = config.hyper.model_copy(update=hyper)
        config.hyper.model_validate(config.hyper.model_dump())
    return Haze(config)
