import logging
import typing as ty
import re
from pathlib import Path
import inspect

from typing import dataclass_transform
from pydra.compose.base import (
    ensure_field_objects,
    build_task_class,
    check_explicit_fields_are_none,
    extract_fields_from_class,
)
from .fields import arg, out
from .task import MonaiTask as Task
from .task import MonaiOutputs as Outputs
from .task import BUNDLE_HELP
from .spec_parser import parse_monai_spec, name_from_spec


logger = logging.getLogger("pydra.compose.monai")


@dataclass_transform(
    kw_only_default=True,
    field_specifiers=(arg,),
)
def define(
    wrapped: type | str | Path | None = None,
    /,
    inputs: list[str | arg] | dict[str, arg | type] | None = None,
    outputs: list[str | out] | dict[str, out | type] | type | None = None,
    bases: ty.Sequence[type] = (),
    outputs_bases: ty.Sequence[type] = (),
    auto_attribs: bool = True,
    name: str | None = None,
    xor: ty.Sequence[str | None] | ty.Sequence[ty.Sequence[str | None]] = (),
) -> Task | ty.Callable[[str | type], Task[ty.Any]]:
    """
    Create an interface for a function or a class.

    Parameters
    ----------
    wrapped : type | callable | None
        The executable to run the app (or entrypoint if running inside a container) or
        class to create an interface for.
    inputs : list[str | Arg] | dict[str, Arg | type] | None
        The inputs to the function or class.
    outputs : list[str | base.Out] | dict[str, base.Out | type] | type | None
        The outputs of the function or class.
    image_tag : str
        the tag of the Docker image to use to run the container. If None, the executable
        is assumed to be in the native env.
    auto_attribs : bool
        Whether to use auto_attribs mode when creating the class.
    name: str | None
        The name of the returned class
    xor: Sequence[str | None] | Sequence[Sequence[str | None]], optional
        Names of args that are exclusive mutually exclusive, which must include
        the name of the current field. If this list includes None, then none of the
        fields need to be set.

    Returns
    -------
    Task
        The task class for the Python function
    """

    def make(wrapped: str | type) -> Task:
        if inspect.isclass(wrapped):
            klass = wrapped
            function = klass.function
            class_name = klass.__name__
            check_explicit_fields_are_none(klass, inputs, outputs)
            parsed_inputs, parsed_outputs = extract_fields_from_class(
                Task,
                Outputs,
                klass,
                arg,
                out,
                auto_attribs,
                skip_fields=["function"],
            )
        else:
            # wrapped is a Path or str pointing to a MONAI bundle dir or metadata.json
            spec_path = Path(wrapped) if not isinstance(wrapped, Path) else wrapped
            class_name = name or name_from_spec(spec_path)
            klass = None
            parsed_inputs, parsed_outputs = parse_monai_spec(spec_path)

            # Bundle root: metadata.json lives at <bundle_root>/configs/metadata.json,
            # so when spec_path is the file we go up two levels; when it's the dir we use it directly.
            bundle_root = spec_path if spec_path.is_dir() else spec_path.parent.parent

            # Add the `bundle` base field with a default pointing at the bundle root
            # used at define() time. Users override only if they want a different bundle.
            parsed_inputs["bundle"] = arg(
                name="bundle",
                type=ty.Any,
                help=BUNDLE_HELP,
                default=str(bundle_root),
            )

            parsed_inputs, parsed_outputs = ensure_field_objects(
                arg_type=arg,
                out_type=out,
                inputs=parsed_inputs,
                outputs=parsed_outputs,
                input_helps={},
                output_helps={},
            )

        defn = build_task_class(
            Task,
            Outputs,
            parsed_inputs,
            parsed_outputs,
            name=class_name,
            klass=klass,
            bases=bases,
            outputs_bases=outputs_bases,
            xor=xor,
        )

        return defn

    if wrapped is not None:
        if not isinstance(wrapped, (str, Path, type)):
            raise ValueError(f"wrapped must be a class or a str, not {wrapped!r}")
        return make(wrapped)
    return make
