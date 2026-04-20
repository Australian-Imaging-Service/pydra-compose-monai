import attrs
from pydra.compose import base


@attrs.define(kw_only=True)
class arg(base.Arg):
    """Argument of a Python task

    Parameters
    ----------
    help: str
        A short description of the input field.
    default : Any, optional
        the default value for the argument
    allowed_values: list, optional
        List of allowed values for the field.
    requires: list, optional
        Names of the inputs that are required together with the field.
    copy_mode: File.CopyMode, optional
        The mode of copying the file, by default it is File.CopyMode.any
    copy_collation: File.CopyCollation, optional
        The collation of the file, by default it is File.CopyCollation.any
    copy_ext_decomp: File.ExtensionDecomposition, optional
        The extension decomposition of the file, by default it is
        File.ExtensionDecomposition.single
    readonly: bool, optional
        If True the input field can’t be provided by the user but it aggregates other
        input fields (for example the fields with argstr: -o {fldA} {fldB}), by default
        it is False
    type: type, optional
        The type of the field, by default it is Any
    name: str, optional
        The name of the field, used when specifying a list of fields instead of a mapping
        from name to field, by default it is None
    path: str, optional
        The key path within the MONAI bundle config that this field corresponds to
        (e.g. "network_data_format/inputs/image"), by default it is None
    """

    path: str | None = attrs.field(default=None)


@attrs.define(kw_only=True)
class out(base.Out):
    """Output of a Python task

    Parameters
    ----------
    name: str, optional
        The name of the field, used when specifying a list of fields instead of a mapping
        from name to field, by default it is None
    type: type, optional
        The type of the field, by default it is Any
    help: str, optional
        A short description of the input field.
    requires: list, optional
        Names of the inputs that are required together with the field.
    converter: callable, optional
        The converter for the field passed through to the attrs.field, by default it is None
    validator: callable | iterable[callable], optional
        The validator(s) for the field passed through to the attrs.field, by default it is None
    position : int
        The position of the output in the output list, allows for tuple unpacking of
        outputs
    path: str, optional
        The key path within the MONAI bundle config that this field corresponds to
        (e.g. "network_data_format/outputs/pred"), by default it is None
    """

    path: str | None = attrs.field(default=None)
