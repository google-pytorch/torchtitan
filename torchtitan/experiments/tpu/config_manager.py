# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fork command line for TorchTitan configs: ``--module/--config`` + overrides.

Upstream eeeb39b42 ("Replace Tyro CLI with Python config loading", #4908)
removed the tyro-generated ``--section.option`` flags; its ``ConfigLoader``
only accepts ``--module``, ``--config`` and a handful of launch options, and
expects every other setting to live in a Python recipe. The TPU lane's tests,
recipes and READMEs pass hundreds of dotted overrides
(``--parallelism.tensor_parallel_degree=2``, ``--training.steps 3``,
``--compile.no-enable``, ``activation-checkpoint:none`` ...), so this module
keeps that surface without tyro:

* ``--module`` / ``--config`` select the recipe exactly like upstream's
  ``ConfigLoader``. ``--module`` is an importable module, a module with a
  ``config_registry`` submodule (the TPU lane's layout), or a bare shorthand
  such as ``flux`` / ``alphabet_sort`` that is looked up under
  ``torchtitan_recipes``.
* ``--a.b.c=VALUE`` / ``--a.b.c VALUE`` set a nested dataclass field; dashes
  and underscores are interchangeable in the path. Values are coerced from
  the field's annotation (``bool``, ``int``, ``float``, ``str``, ``Literal``,
  ``Enum``, ``X | None``, ``list[...]`` as comma separated, registered custom
  types such as ``tpu_flags.FlagMap``).
* ``--a.flag`` sets a ``bool`` to ``True``; ``--a.no-flag`` sets it ``False``.
* ``a.b:choice`` (no leading dashes) selects a member of a dataclass union,
  tyro's "subcommand" spelling: ``activation-checkpoint:full`` picks
  ``FullAC.Config``, ``activation-checkpoint:none`` sets ``None``. Flags after
  it tune the chosen member (``--activation_checkpoint.mode ...``).
* ``--override.imports T1 T2 ...`` takes every following plain token.

Upstream #4913 moved ``optimizer`` / ``lr_scheduler`` / ``ema`` and
``training.max_norm`` under ``optim``; the old paths are accepted as aliases
so existing launch commands keep working.

After the overrides are applied, ``__post_init__`` re-runs on every modified
dataclass from the leaves up (tyro rebuilt each level, which ran its
validation on the final values).
"""

from __future__ import annotations

import dataclasses
import enum
import importlib
import json
import logging
import re
import sys
import types
import typing
from typing import Any, Callable

from torchtitan.config.configurable import Configurable

logger = logging.getLogger(__name__)

_NONE_TYPE = type(None)
_TRUE = frozenset({"true", "1", "yes", "y", "on"})
_FALSE = frozenset({"false", "0", "no", "n", "off"})
_SUBCOMMAND_RE = re.compile(r"^[A-Za-z_][\w.\-]*:[\w\-]+$")

# Root-level aliases for fields upstream moved (#4913: ``Optim`` owns the
# optimizer, the LR scheduler, the EMA and the gradient clipping norm).
_ROOT_ALIASES: dict[tuple[str, ...], tuple[str, ...]] = {
    ("optimizer",): ("optim", "optimizer"),
    ("lr_scheduler",): ("optim", "lr_scheduler"),
    ("ema",): ("optim", "ema"),
    ("training", "max_norm"): ("optim", "max_norm"),
}

# Shorthand ``--module`` names are searched here, in order; the first module
# that defines the requested ``--config`` function wins.
_SHORTHAND_PREFIXES = (
    "torchtitan.experiments.tpu.{name}.config_registry",
    "torchtitan_recipes.tests.models.{name}",
    "torchtitan_recipes.models.{name}",
    "torchtitan_recipes.rl.{name}",
    "torchtitan_recipes.tests.{name}",
    "torchtitan_recipes.{name}",
    "torchtitan.experiments.{name}.config_registry",
)

_PARSERS: dict[type, Callable[[str], Any]] = {}


def register_parser(tp: type, parse: Callable[[str], Any]) -> None:
  """Registers ``parse`` as the string constructor for leaf fields of ``tp``."""
  _PARSERS[tp] = parse


class ConfigError(ValueError):
  """A command-line override could not be applied."""


class ConfigManager:
  """Loads a recipe from ``--module/--config`` and applies CLI overrides."""

  def parse_args(self, args: list[str] | None = None) -> Configurable.Config:
    if args is None:
      args = sys.argv[1:]
    config, remaining = self._load_config(list(args))
    apply_cli_overrides(config, remaining)
    self.config = config
    return config

  @staticmethod
  def _load_config(args: list[str]) -> tuple[Configurable.Config, list[str]]:
    """Strips ``--module/--config`` and returns the loaded recipe + the rest."""
    module_name = None
    config_name = None
    remaining: list[str] = []
    i = 0
    while i < len(args):
      arg = args[i]
      for flag in ("--module", "--config"):
        if arg == flag or arg.startswith(flag + "="):
          if "=" in arg:
            value = arg.split("=", 1)[1]
          else:
            if i + 1 >= len(args):
              raise ConfigError(f"{flag} requires a value")
            i += 1
            value = args[i]
          if flag == "--module":
            module_name = value
          else:
            config_name = value
          break
      else:
        remaining.append(arg)
      i += 1
    if module_name is None or config_name is None:
      raise ConfigError(
          "--module and --config are required. Example: "
          "--module=torchtitan.experiments.tpu.llama3 --config=llama3_debugmodel"
      )
    config_fn = _resolve_config_fn(module_name, config_name)
    config = config_fn()
    if not isinstance(config, Configurable.Config):
      raise ConfigError(
          f"{module_name}.{config_name}() must return a Configurable.Config, "
          f"got {type(config).__qualname__}"
      )
    return config, remaining


def _resolve_config_fn(module_name: str, config_name: str) -> Callable[[], Any]:
  """Finds ``config_name`` in ``module_name`` (or its shorthand expansions)."""
  if "." in module_name:
    candidates = [f"{module_name}.config_registry", module_name]
  else:
    candidates = [p.format(name=module_name) for p in _SHORTHAND_PREFIXES]
  tried: list[str] = []
  import_errors: list[str] = []
  for candidate in candidates:
    try:
      module = importlib.import_module(candidate)
    except ImportError as error:
      # Only swallow "this candidate does not exist"; a broken module that
      # does exist should surface its real error.
      missing = error.name is not None and (
          candidate == error.name or candidate.startswith(error.name + ".")
      )
      if not missing:
        raise
      import_errors.append(f"{candidate}: {error}")
      continue
    tried.append(candidate)
    config_fn = getattr(module, config_name, None)
    if callable(config_fn):
      return config_fn
  if tried:
    available = sorted(
        name
        for candidate in tried
        for name in dir(importlib.import_module(candidate))
        if not name.startswith("_")
        and name[0].islower()
        and callable(getattr(importlib.import_module(candidate), name))
    )
    raise ConfigError(
        f"Config function {config_name!r} not found in {tried}. "
        f"Available config functions: {available}"
    )
  raise ConfigError(
      f"Cannot import a config module for --module={module_name!r}. Tried: "
      + "; ".join(import_errors)
  )


# ----------------------------------------------------------------------------
# Override application
# ----------------------------------------------------------------------------


def _norm(name: str) -> str:
  return name.replace("-", "_")


def _is_flag(token: str) -> bool:
  return token.startswith("--") and len(token) > 2


def _field_names(obj: Any) -> set[str]:
  return {f.name for f in dataclasses.fields(obj)}


def _type_hints(obj: Any) -> dict[str, Any]:
  try:
    return typing.get_type_hints(type(obj), include_extras=True)
  except Exception:  # pylint: disable=broad-except
    # Forward references to TYPE_CHECKING-only names; fall back to the raw
    # field annotations (strings are handled by _coerce from the value).
    return {f.name: f.type for f in dataclasses.fields(obj)}


def _strip_annotated(tp: Any) -> Any:
  while typing.get_origin(tp) is typing.Annotated:
    tp = typing.get_args(tp)[0]
  return tp


def _union_members(tp: Any) -> tuple[Any, ...] | None:
  tp = _strip_annotated(tp)
  origin = typing.get_origin(tp)
  if origin is typing.Union or origin is types.UnionType:
    return tuple(_strip_annotated(a) for a in typing.get_args(tp))
  return None


def _resolve_path(config: Any, raw_path: str) -> tuple[Any, str, list[Any]]:
  """Walks ``raw_path`` (dotted) to its parent object and leaf field name.

  Returns ``(parent, field_name, [objects along the way, root first])``.
  """
  parts = tuple(_norm(p) for p in raw_path.split(".") if p)
  if not parts:
    raise ConfigError(f"empty override path {raw_path!r}")
  root_fields = _field_names(config)
  for alias, target in _ROOT_ALIASES.items():
    if parts[: len(alias)] != alias:
      continue
    # Redirect only when the old path no longer exists on this config.
    exists = alias[0] in root_fields
    if exists and len(alias) > 1:
      section = getattr(config, alias[0])
      exists = dataclasses.is_dataclass(section) and alias[1] in _field_names(
          section
      )
    if not exists and target[0] in root_fields:
      parts = target + parts[len(alias) :]
    break
  obj = config
  chain = [config]
  for depth, part in enumerate(parts[:-1]):
    if not dataclasses.is_dataclass(obj):
      raise ConfigError(
          f"--{raw_path}: {'.'.join(parts[:depth])!r} is not a config section"
      )
    if part not in _field_names(obj):
      raise ConfigError(
          f"--{raw_path}: {type(obj).__qualname__} has no field {part!r} "
          f"(fields: {sorted(_field_names(obj))})"
      )
    obj = getattr(obj, part)
    if obj is None:
      joined = ".".join(parts[: depth + 1])
      raise ConfigError(
          f"--{raw_path}: {joined!r} is None in this recipe; pick a recipe that "
          f"configures it or select it with '{joined}:<choice>' first"
      )
    chain.append(obj)
  return obj, parts[-1], chain


def _leaf_type(parent: Any, name: str) -> Any:
  hints = _type_hints(parent)
  tp = hints.get(name)
  if tp is None:
    for f in dataclasses.fields(parent):
      if f.name == name:
        tp = f.type
  return _strip_annotated(tp)


def _literal_values(tp: Any) -> tuple[Any, ...] | None:
  if typing.get_origin(tp) is typing.Literal:
    return typing.get_args(tp)
  return None


def _coerce(value: str, tp: Any, *, current: Any, path: str) -> Any:
  """Converts the command-line string ``value`` to the field type ``tp``."""
  tp = _strip_annotated(tp)
  if isinstance(tp, str) or tp is None or tp is Any:
    # Unresolvable annotation: infer from the current value.
    if isinstance(current, bool):
      return _coerce(value, bool, current=current, path=path)
    if isinstance(current, int):
      return int(value)
    if isinstance(current, float):
      return float(value)
    return value
  if tp in _PARSERS:
    return _PARSERS[tp](value)
  members = _union_members(tp)
  if members is not None:
    if _NONE_TYPE in members and value in ("None", "none", "null"):
      return None
    errors = []
    # Literal members first: they are the most specific match.
    ordered = sorted(
        (m for m in members if m is not _NONE_TYPE),
        key=lambda m: 0 if _literal_values(m) is not None else 1,
    )
    for member in ordered:
      try:
        return _coerce(value, member, current=current, path=path)
      except (ValueError, TypeError, ConfigError) as error:
        errors.append(str(error))
    raise ConfigError(f"--{path}: cannot parse {value!r} as {tp}: {errors}")
  literals = _literal_values(tp)
  if literals is not None:
    for choice in literals:
      if value == str(choice) or (
          isinstance(choice, bool) and value.lower() in (_TRUE | _FALSE)
          and (value.lower() in _TRUE) == choice
      ):
        return choice
    raise ConfigError(f"--{path}: {value!r} is not one of {list(literals)}")
  origin = typing.get_origin(tp)
  if origin in (list, tuple, set, frozenset):
    args = typing.get_args(tp)
    item_tp = args[0] if args else str
    items = [v for v in value.split(",") if v != ""]
    coerced = [
        _coerce(item, item_tp, current=None, path=path) for item in items
    ]
    return origin(coerced)
  if origin is dict:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
      raise ConfigError(f"--{path}: expected a JSON object, got {value!r}")
    return parsed
  if tp is bool:
    lowered = value.lower()
    if lowered in _TRUE:
      return True
    if lowered in _FALSE:
      return False
    raise ConfigError(f"--{path}: expected a boolean, got {value!r}")
  if tp is int:
    return int(value)
  if tp is float:
    return float(value)
  if tp is str:
    return value
  if isinstance(tp, type) and issubclass(tp, enum.Enum):
    for member in tp:
      if value in (member.name, str(member.value)):
        return member
    raise ConfigError(
        f"--{path}: {value!r} is not one of {[m.name for m in tp]}"
    )
  if isinstance(tp, type) and issubclass(tp, dict):
    # e.g. ``tpu_flags.FlagMap`` when its parser was not registered.
    return tp(json.loads(value))
  if dataclasses.is_dataclass(tp):
    raise ConfigError(
        f"--{path}: {tp.__qualname__} is a config section; set one of its "
        f"fields (--{path}.<field>) or select it with '{path}:<choice>'"
    )
  raise ConfigError(f"--{path}: unsupported field type {tp!r}")


def _subcommand_choices(members: tuple[Any, ...]) -> dict[str, Any]:
  """Maps tyro-style names (``full``, ``selective`` ...) to union members."""
  choices: dict[str, Any] = {}
  for member in members:
    if member is _NONE_TYPE:
      choices["none"] = None
      continue
    if not dataclasses.is_dataclass(member):
      continue
    qualname = member.__qualname__
    owner = qualname.rsplit(".", 1)[0] if qualname.endswith(".Config") else qualname
    choices[owner.lower()] = member
  return choices


def _select_subcommand(config: Any, token: str) -> list[Any] | None:
  """Applies ``a.b:choice`` if ``a.b`` is a union field; else returns None."""
  raw_path, _, choice = token.partition(":")
  try:
    parent, name, chain = _resolve_path(config, raw_path)
  except ConfigError:
    return None
  if not dataclasses.is_dataclass(parent) or name not in _field_names(parent):
    return None
  tp = _leaf_type(parent, name)
  members = _union_members(tp)
  if members is None:
    if dataclasses.is_dataclass(tp):
      members = (tp,)
    else:
      return None
  choices = _subcommand_choices(members)
  key = _norm(choice).replace("_", "").lower()
  if key in choices:
    selected = choices[key]
  else:
    matches = [
        c
        for c in choices
        if c != "none" and (c.startswith(key) or c.removesuffix("ac") == key)
    ]
    if len(matches) != 1:
      raise ConfigError(
          f"{token}: {choice!r} is not one of {sorted(choices)}"
      )
    selected = choices[matches[0]]
  current = getattr(parent, name)
  if selected is None:
    setattr(parent, name, None)
  elif type(current) is not selected:
    setattr(parent, name, selected())
  return chain


def apply_cli_overrides(config: Any, tokens: list[str]) -> Any:
  """Applies dotted ``--path value`` overrides and ``path:choice`` selections."""
  modified: list[Any] = []

  def note(chain: list[Any]) -> None:
    for obj in chain:
      if not any(obj is m for m in modified):
        modified.append(obj)

  i = 0
  while i < len(tokens):
    token = tokens[i]
    i += 1
    if not _is_flag(token):
      if "=" not in token and _SUBCOMMAND_RE.match(token):
        chain = _select_subcommand(config, token)
        if chain is not None:
          note(chain)
          continue
      raise ConfigError(f"unexpected positional argument {token!r}")
    if token in ("--help", "-h"):
      print(__doc__)
      raise SystemExit(0)
    key, has_eq, value = token[2:].partition("=")
    parts = key.split(".")
    negated = False
    if parts[-1].startswith("no-") or (
        parts[-1].startswith("no_") and len(parts[-1]) > 3
    ):
      negated = True
      parts[-1] = parts[-1][3:]
    path = ".".join(parts)
    parent, name, chain = _resolve_path(config, path)
    if not dataclasses.is_dataclass(parent) or name not in _field_names(parent):
      raise ConfigError(
          f"--{key}: {type(parent).__qualname__} has no field {name!r} "
          f"(fields: {sorted(_field_names(parent)) if dataclasses.is_dataclass(parent) else []})"
      )
    tp = _leaf_type(parent, name)
    current = getattr(parent, name)
    members = _union_members(tp) or ()
    is_bool = tp is bool or (
        isinstance(current, bool) and (bool in members or isinstance(tp, str))
    )
    if negated:
      if has_eq:
        raise ConfigError(f"--{key}: a negated flag takes no value")
      new_value = False
    elif has_eq:
      new_value = _coerce(value, tp, current=current, path=path)
    elif path == "override.imports":
      from torchtitan.config.override import parse_cli_imports  # pylint: disable=g-import-not-at-top

      values = []
      while i < len(tokens) and not _is_flag(tokens[i]):
        values.append(tokens[i])
        i += 1
      new_value = list(current or []) + parse_cli_imports(values)
    elif is_bool:
      if i < len(tokens) and tokens[i].lower() in (_TRUE | _FALSE):
        new_value = _coerce(tokens[i], bool, current=current, path=path)
        i += 1
      else:
        new_value = True
    else:
      if i >= len(tokens) or _is_flag(tokens[i]):
        raise ConfigError(f"--{key} requires a value")
      new_value = _coerce(tokens[i], tp, current=current, path=path)
      i += 1
    setattr(parent, name, new_value)
    note(chain)

  # Re-run validation / derived defaults on every touched config, leaves
  # first, then the root (tyro rebuilt each level with the final values).
  for obj in reversed(modified):
    post_init = getattr(obj, "__post_init__", None)
    if post_init is not None and dataclasses.is_dataclass(obj):
      post_init()
  return config


__all__ = [
    "ConfigError",
    "ConfigManager",
    "apply_cli_overrides",
    "register_parser",
]
