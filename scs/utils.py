from __future__ import annotations

from typing import (
    Hashable,
    Literal,
    overload,
)

from ml_collections import config_dict


@overload
def make_config(
    config: dict[str, Hashable], frozen: Literal[True] = True
) -> config_dict.FrozenConfigDict: ...


@overload
def make_config(
    config: dict[str, Hashable], frozen: Literal[False]
) -> config_dict.ConfigDict: ...


def make_config(
    config: dict[str, Hashable], frozen: bool = True
) -> config_dict.FrozenConfigDict | config_dict.ConfigDict:
    """Creates a config dict from a built-in python dictionary.

    This function converts a standard Python dictionary into an `ml_collections`
    config dict, which allows for attribute-style access to keys.

    Args:
        config: The input dictionary to be converted. Its keys must be strings,
            and the values can be any hashable type.
        frozen: If True (the default), creates an immutable `FrozenConfigDict`.
            If False, creates a mutable `ConfigDict`.

    Returns:
        An instance of `config_dict.FrozenConfigDict` or
        `config_dict.ConfigDict`.
    """
    if frozen:
        return config_dict.FrozenConfigDict(config)
    return config_dict.ConfigDict(config)
