import logging
import typing as ty
import re
from pathlib import Path
import inspect
import docker.errors
from typing import dataclass_transform
from pydra.compose.base import (
    ensure_field_objects,
    build_task_class,
    check_explicit_fields_are_none,
    extract_fields_from_class,
)
from .fields import arg, out
from .task import BidsAppTask as Task
from .task import BidsAppOutputs as Outputs


logger = logging.getLogger("pydra.compose.monai")


@dataclass_transform(
    kw_only_default=True,
    field_specifiers=(arg,),
)
def define(
    wrapped: type | str | None = None,
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
            app = klass.app
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
            if isinstance(wrapped, Path):
                wrapped = str(wrapped.absolute())
            elif not isinstance(wrapped, str):
                raise ValueError(
                    "wrapped must be a class or a str representing either the name of a "
                    "Docker image if executing the app as a Docker container, or the "
                    "name of the executable to run if extending the Docker image , not "
                    f"{wrapped!r}"
                )
            klass = None
            app = wrapped

            if name is None:
                if app.startswith("/"):
                    # Docker image name
                    class_name = Path(app).name.split(".")[0]
                    if class_name[0].isdigit():
                        class_name = DIGIT_TO_WORD[class_name[0]] + class_name[1:]
                    class_name = app.split("/")[-1].split(":")[0]
                else:
                    # Docker image name
                    class_name = app.split("/")[-1].split(":")[0]
                class_name = re.sub(r"[^a-zA-Z0-9]", "_", class_name)
            else:
                class_name = name

            parsed_inputs = (
                inputs if isinstance(inputs, dict) else {i.name: i for i in inputs}
            )
            parsed_outputs = (
                outputs if isinstance(outputs, dict) else {o.name: o for o in outputs}
            )

            # Add in fields from base classes
            parsed_inputs.update(
                {n: getattr(Task, n) for n in Task.BASE_ATTRS if n != "app"}
            )
            parsed_outputs.update({n: getattr(Outputs, n) for n in Outputs.BASE_ATTRS})

            parsed_inputs, parsed_outputs = ensure_field_objects(
                arg_type=arg,
                out_type=out,
                inputs=parsed_inputs,
                outputs=parsed_outputs,
                input_helps={},
                output_helps={},
            )
        if clashing := set(parsed_inputs) & set(["exectuable", "image_tag"]):
            raise ValueError(f"{list(clashing)} are reserved input names")

        parsed_inputs["app"] = arg(name="app", type=str, default=app, path=None)

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


DIGIT_TO_WORD = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def get_docker_entrypoint(image_tag: str) -> str:
    """Pulls a given Docker image tag and inspects the image to get its
    entrypoint/cmd

    IMAGE_TAG is the tag of the Docker image to inspect"""
    dc = docker.from_env()

    dc.images.pull(image_tag)

    image_attrs = dc.api.inspect_image(image_tag)["Config"]

    executable = image_attrs["Entrypoint"]
    if executable is None:
        executable = image_attrs["Cmd"]

    return executable
