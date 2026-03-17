import attrs
import re
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
    path: str | None
        The path to where the input is stored within the dataset: the modality of the
        file followed by the BIDS suffix in the form 'modality/suffix'.
    """

    # the BIDS suffix that identifies the input
    path: str | None = attrs.field()

    @path.validator
    def _path_validator(self, attribute, value):
        """Validate the path of the input field"""
        if value is None:
            return
        if not isinstance(value, str):
            raise TypeError(f"Path must be a string, got {type(value)}")
        if not re.match(r"\w+/\w+", value):
            raise ValueError(f"Path must be of the form 'modality/suffix', got {value}")


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
        The path to where the output is stored within the derivatives directory, the
        namespace of the derivative followed by suffix of the output file in the
        form 'modality/suffix'.
    """

    # the path the file will be stored within the deriavtives directory
    path: str = attrs.field(default="")

    @path.validator
    def _path_validator(self, attribute, value):
        """Validate the path of the input field"""
        if not isinstance(value, str):
            raise TypeError(f"Path must be a string, got {type(value)}")
