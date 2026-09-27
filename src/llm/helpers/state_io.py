"""Atomic, model-bound serialization for one user's Titans fast-memory state."""

import hashlib
from pathlib import Path
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file

FORMAT = "titans-fast-memory-v1"


def checkpoint_digest(checkpoint: str | Path) -> str:
    """Bind a user state to the exact adapter weights, not just its architecture."""
    digest = hashlib.sha256()
    with (Path(checkpoint) / "adapter_model.safetensors").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_fast_memory_state(path: str | Path, states: dict, *, adapter_digest: str) -> None:
    """Save the completed batched state of a single user without overwriting in place."""
    path = Path(path)
    tensors = {}
    for layer, state in states.items():
        for kind in ("params", "surprise"):
            for name, value in state[kind].items():
                if value.shape[0] != 1:
                    msg = "Fast-memory state must contain exactly one user"
                    raise ValueError(msg)
                tensors[f"layers.{layer}.{kind}.{name}"] = value.detach().to("cpu").contiguous()
    if not tensors:
        msg = "Cannot save an empty fast-memory state"
        raise ValueError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".safetensors", delete=False) as file:
        temporary = Path(file.name)
    try:
        save_file(tensors, str(temporary), metadata={"format": FORMAT, "adapter": adapter_digest})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_fast_memory_state(path: str | Path, *, adapter_digest: str, device) -> dict:
    """Reject state from another adapter and restore tensors to the model device."""
    states = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != FORMAT or metadata.get("adapter") != adapter_digest:
            msg = "Fast-memory state does not match this adapter checkpoint"
            raise ValueError(msg)
        for key in sorted(handle.keys()):
            prefix, layer, kind, name = key.split(".", 3)
            if prefix != "layers" or kind not in ("params", "surprise"):
                msg = f"Invalid fast-memory tensor key: {key}"
                raise ValueError(msg)
            states.setdefault(int(layer), {"params": {}, "surprise": {}})[kind][name] = (
                handle.get_tensor(key).to(device)
            )
    if not states:
        msg = "Fast-memory state file is empty"
        raise ValueError(msg)
    return states
