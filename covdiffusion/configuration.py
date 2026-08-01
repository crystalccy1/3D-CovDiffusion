"""Configuration loading and serialization for 3D-CovDiffusion.

The training CLI uses a deliberately small configuration protocol:

1. ``default.yaml`` is always loaded.
2. YAML profiles named by ``config=[profile_a,profile_b]`` are merged in order.
3. Remaining ``key=value`` CLI arguments are applied last, including dotted keys.

The returned object is an :class:`omegaconf.DictConfig` because the training and
checkpoint code relies on attribute access and OmegaConf serialization.  File
discovery and merging are implemented here rather than delegated to a generic
command-line framework so missing profiles and malformed arguments fail with
actionable messages.
"""

from __future__ import annotations

import os
from pathlib import Path
import pprint
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence

from omegaconf import DictConfig, ListConfig, OmegaConf
import yaml


_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_LIST_FIELDS = ("dataset",)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration {path}: {exc}") from exc
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return value


def _parse_profile_names(raw_value: str) -> list[str]:
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid config profile list {raw_value!r}: {exc}") from exc

    if isinstance(value, str):
        names = [value]
    elif isinstance(value, list):
        names = value
    else:
        raise ValueError(
            "config must name one profile or a list, for example "
            "config=[covdiffusion,windows]"
        )

    for name in names:
        if not isinstance(name, str) or not _PROFILE_NAME.fullmatch(name):
            raise ValueError(f"Invalid configuration profile name: {name!r}")
    return names


def _split_cli(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    profile_names: list[str] | None = None
    overrides: list[str] = []
    for argument in argv:
        key, separator, value = argument.partition("=")
        if not separator or not key:
            raise ValueError(
                f"Invalid configuration argument {argument!r}; expected key=value"
            )
        if key == "config":
            if profile_names is not None:
                raise ValueError("config may be specified only once")
            profile_names = _parse_profile_names(value)
        else:
            overrides.append(argument)
    return profile_names or [], overrides


def _normalise_list_fields(config: DictConfig) -> None:
    for key in _LIST_FIELDS:
        value = config.get(key)
        if value is None:
            config[key] = []
        elif isinstance(value, str):
            config[key] = [value]
        elif isinstance(value, (list, tuple, ListConfig)):
            config[key] = list(value)
        else:
            raise ValueError(
                f"Configuration field {key!r} must be a string, list, or null"
            )


def load_args(
    root: str | os.PathLike[str], argv: Sequence[str] | None = None
) -> DictConfig:
    """Load the deterministic YAML stack and final CLI overrides.

    Args:
        root: Directory containing ``default.yaml`` and named profiles.
        argv: ``key=value`` arguments.  Defaults to ``sys.argv[1:]``.

    Returns:
        A merged OmegaConf ``DictConfig`` with list-like fields normalized.
    """

    config_root = Path(root).expanduser()
    if not config_root.is_dir():
        raise FileNotFoundError(f"Configuration directory not found: {config_root}")

    profile_names, overrides = _split_cli(
        sys.argv[1:] if argv is None else list(argv)
    )
    layers: list[Any] = [
        OmegaConf.create(_load_yaml_mapping(config_root / "default.yaml"))
    ]
    for name in profile_names:
        layers.append(
            OmegaConf.create(_load_yaml_mapping(config_root / f"{name}.yaml"))
        )
    if overrides:
        try:
            layers.append(OmegaConf.from_dotlist(overrides))
        except Exception as exc:
            raise ValueError(f"Invalid command-line configuration override: {exc}") from exc

    config = OmegaConf.merge(*layers)
    if not isinstance(config, DictConfig):
        raise ValueError("Merged configuration must be a mapping")
    _normalise_list_fields(config)
    return config


def _plain_value(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True, enum_to_str=True)
    if isinstance(value, Mapping):
        return dict(value)
    return value


def pformat_dict(value: Any) -> str:
    """Return a stable, human-readable representation of a configuration."""

    return pprint.pformat(_plain_value(value), sort_dicts=False, width=100)


def save_config(
    config: Any,
    output_dir: str | os.PathLike[str],
    filename: str = "config.yaml",
) -> Path:
    """Atomically save a configuration as UTF-8 YAML and return its path."""

    target_directory = Path(output_dir).expanduser()
    target_directory.mkdir(parents=True, exist_ok=True)
    target = target_directory / filename
    if target.name != filename or target.is_dir():
        raise ValueError(f"filename must be a plain file name, got {filename!r}")

    plain = _plain_value(config)
    if not isinstance(plain, dict):
        raise ValueError("Configuration must be a mapping")

    descriptor, temporary_name = tempfile.mkstemp(
        dir=target_directory, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(
                plain,
                stream,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return target
