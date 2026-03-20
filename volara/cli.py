import json
import logging
from pathlib import Path
from typing import Any, get_args, get_origin

import click
import yaml
from pydantic import BaseModel
from pydantic.fields import FieldInfo


def load_config_file(path: str) -> dict:
    """Load a YAML or JSON config file based on extension."""
    p = Path(path)
    with open(p) as f:
        if p.suffix in (".yaml", ".yml"):
            return yaml.safe_load(f) or {}
        elif p.suffix == ".json":
            return json.load(f)
        else:
            raise click.BadParameter(
                f"Unsupported config file format: {p.suffix} (use .yaml, .yml, or .json)"
            )


def merge_configs(base: dict, overrides: dict) -> dict:
    """Deep-merge overrides onto base dict. Overrides win on conflicts."""
    result = dict(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def _is_pydantic_model(annotation) -> bool:
    """Check if a type annotation is a Pydantic BaseModel subclass."""
    if annotation is None:
        return False
    origin = get_origin(annotation)
    if origin is not None:
        return False
    try:
        return isinstance(annotation, type) and issubclass(annotation, BaseModel)
    except TypeError:
        return False


def _get_first_required_field(model_class) -> str | None:
    """Get the name of the first required field of a Pydantic model (no default)."""
    for name, field_info in model_class.model_fields.items():
        if field_info.is_required():
            return name
    return None


def _get_model_candidates(annotation) -> list[type]:
    """Extract Pydantic model classes from a type annotation (handles unions)."""
    candidates = []
    origin = get_origin(annotation)
    if origin is not None:
        for arg in get_args(annotation):
            candidates.extend(_get_model_candidates(arg))
    elif _is_pydantic_model(annotation):
        candidates.append(annotation)
    return candidates


def resolve_shorthands(task_class, config: dict) -> dict:
    """Expand bare string values for nested-model fields into proper dicts.

    For example, if a field `frags_data` expects a `Labels` model and the config
    has `{"frags_data": "path/to/frags"}`, this expands it to
    `{"frags_data": {"store": "path/to/frags"}}` since `store` is the first
    required field of `Labels`.
    """
    result = dict(config)
    for field_name, field_info in task_class.model_fields.items():
        if field_name not in result:
            continue
        value = result[field_name]
        if not isinstance(value, str):
            continue

        # Get the model candidates for this field's annotation
        candidates = _get_model_candidates(field_info.annotation)
        if not candidates:
            continue

        # Try the first candidate that has a required field
        for model_cls in candidates:
            first_field = _get_first_required_field(model_cls)
            if first_field is not None:
                expanded = {first_field: value}
                # If the model has a discriminator-like type field with a default,
                # include it so Pydantic can resolve the type
                for mf_name, mf_info in model_cls.model_fields.items():
                    if mf_name != first_field and not mf_info.is_required():
                        if mf_info.default is not None and mf_name.endswith("_type"):
                            expanded[mf_name] = mf_info.default
                result[field_name] = expanded
                break

    return result


def _is_list_like_field(field_info: FieldInfo) -> bool:
    """Check if a field expects a list/sequence type."""
    ann = field_info.annotation
    if ann is None:
        return False
    # Handle Optional[X] -> X
    origin = get_origin(ann)
    args = get_args(ann)
    # Union with None (Optional)
    if origin is type(int | str):  # types.UnionType
        for arg in args:
            if arg is not type(None) and _annotation_is_list(arg):
                return True
        return False
    import types

    if isinstance(ann, types.UnionType):
        for arg in get_args(ann):
            if arg is not type(None) and _annotation_is_list(arg):
                return True
        return False
    return _annotation_is_list(ann)


def _annotation_is_list(ann) -> bool:
    """Check if annotation is a list type."""
    origin = get_origin(ann)
    if origin is list or ann is list:
        return True
    # Check Annotated types
    if origin is not None:
        args = get_args(ann)
        for arg in args:
            if _annotation_is_list(arg):
                return True
    # Check if it's an Annotated type
    try:
        from typing import Annotated

        if get_origin(ann) is Annotated:
            return _annotation_is_list(get_args(ann)[0])
    except ImportError:
        pass
    return False


def _is_coordinate_field(field_info: FieldInfo) -> bool:
    """Check if a field expects a PydanticCoordinate (list of ints)."""
    ann = field_info.annotation
    if ann is None:
        return False
    ann_str = str(ann)
    return "Coordinate" in ann_str or "PydanticCoordinate" in ann_str


def _looks_like_flag(token: str) -> bool:
    """Check if a token looks like a --flag."""
    return token.startswith("--")


def parse_cli_args(args: list[str], task_class=None) -> dict:
    """Parse flat CLI args into a nested dict.

    Supports:
    - Dotted keys: --db.path x -> {"db": {"path": "x"}}
    - Multi-value args: --block_size 20 100 100 -> {"block_size": [20, 100, 100]}
      (consumes non-flag tokens until the next --flag)
    - Single values: --bias -0.5 -0.5 -0.5 (negative numbers are not flags)

    Uses task_class field info to determine which fields are list-like.
    """
    result: dict[str, Any] = {}
    args_list = list(args)
    i = 0

    # Build a lookup of field types from task_class if available
    list_fields: set[str] = set()
    coordinate_fields: set[str] = set()
    if task_class is not None:
        for fname, finfo in task_class.model_fields.items():
            if _is_list_like_field(finfo) or _is_coordinate_field(finfo):
                list_fields.add(fname)
            if _is_coordinate_field(finfo):
                coordinate_fields.add(fname)

    while i < len(args_list):
        token = args_list[i]
        if not token.startswith("--"):
            i += 1
            continue

        key = token.lstrip("-")
        i += 1

        # Collect values until next --flag
        values: list[str] = []
        while i < len(args_list):
            next_token = args_list[i]
            if next_token.startswith("--"):
                # Could be a negative number
                try:
                    float(next_token)
                    values.append(next_token)
                    i += 1
                except ValueError:
                    break
            else:
                values.append(next_token)
                i += 1

        if not values:
            # Boolean flag
            _set_nested(result, key, True)
            continue

        # Determine the root field name (before any dots)
        root_field = key.split(".")[0]

        # If multiple values or field is known to be list-like, keep as list
        if len(values) > 1 or root_field in list_fields:
            parsed = [_try_parse_value(v) for v in values]
            _set_nested(result, key, parsed)
        else:
            _set_nested(result, key, _try_parse_value(values[0]))

    return result


def _try_parse_value(value: str) -> Any:
    """Try to parse a string value as int, float, bool, or JSON."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "null" or value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    # Try JSON for complex values like dicts
    if value.startswith("{") or value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def _set_nested(d: dict, key: str, value: Any) -> None:
    """Set a value in a nested dict using a dotted key."""
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in d:
            d[part] = {}
        d = d[part]
    d[parts[-1]] = value


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
)
def cli(log_level: str) -> None:
    logging.basicConfig(level=getattr(logging, log_level.upper()))


@cli.command()
@click.option(
    "-c", "--config-file", required=True, type=click.Path(exists=True, dir_okay=False)
)
def blockwise_worker(config_file: Path) -> None:
    from volara.blockwise import BlockwiseTask, get_blockwise_tasks_type

    config_file = Path(config_file)
    config_json = json.loads(config_file.open("r").read())

    BlockwiseTasks = get_blockwise_tasks_type()
    config = BlockwiseTasks.validate_python(config_json)
    assert isinstance(config, BlockwiseTask)
    config.process_blocks()


def _get_available_task_names() -> list[str]:
    """Discover and return sorted list of available task type names."""
    from volara.blockwise import BLOCKWISE_TASKS, get_blockwise_tasks_type

    get_blockwise_tasks_type()
    names = []
    for task in BLOCKWISE_TASKS:
        name = task.model_fields["task_type"].default
        if name is not None:
            names.append(name)
    return sorted(names)


class _RunCommand(click.Command):
    """Custom Command that appends available task names to the help text."""

    def format_help(self, ctx, formatter):
        super().format_help(ctx, formatter)
        task_names = _get_available_task_names()
        if task_names:
            with formatter.section("Available Tasks"):
                for name in task_names:
                    formatter.write("  " + name + "\n")


@cli.command(
    cls=_RunCommand,
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("task_name")
@click.option(
    "-c",
    "--config",
    "config_file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to a YAML or JSON config file.",
)
@click.pass_context
def run(ctx, task_name: str, config_file: str | None) -> None:
    """Run a blockwise task.

    TASK_NAME is the task type (see "Available Tasks").
    Additional arguments are passed as --field value pairs.
    """
    from volara.blockwise import BlockwiseTask, get_blockwise_tasks_type, get_task

    # Ensure tasks are discovered
    get_blockwise_tasks_type()

    # Resolve task class
    task_class = get_task(task_name)

    # Load config file if provided
    base_config: dict[str, Any] = {}
    if config_file is not None:
        base_config = load_config_file(config_file)

    # Parse CLI extra args
    cli_overrides = parse_cli_args(ctx.args, task_class=task_class)

    # Merge: CLI overrides config file
    config = merge_configs(base_config, cli_overrides)

    # Set task_type
    config["task_type"] = task_name

    # Resolve shorthands for nested model fields
    config = resolve_shorthands(task_class, config)

    # Validate via Pydantic
    try:
        task = task_class.model_validate(config)
    except Exception as e:
        raise click.ClickException(str(e)) from e

    assert isinstance(task, BlockwiseTask)

    # Run the task
    task.run_blockwise()
