import attrs
import typing as ty
import logging
from pydra.compose import base
from . import fields


logger = logging.getLogger("pydra.compose.monai")

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job


@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class MonaiOutputs(base.Outputs):

    @classmethod
    def _from_job(cls, job: "Job[MonaiTask]") -> ty.Self:
        """Collect the outputs of a job from a combination of the provided inputs,
        the objects in the output directory, and the stdout and stderr of the process.

        Parameters
        ----------
        job : Job[Task]
            The job whose outputs are being collected.
        outputs_dict : dict[str, ty.Any]
            The outputs of the job, as a dictionary

        Returns
        -------
        outputs : Outputs
            The outputs of the job in dataclass
        """
        outputs = super()._from_job(job)
        raise NotImplementedError(
            "The job outputs need to be extracted from the Job object and populate the Outputs object"
        )
        return outputs


MonaiOutputsType = ty.TypeVar("MonaiOutputsType", bound=MonaiOutputs)


@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class MonaiTask(base.Task[MonaiOutputsType]):

    BASE_ATTRS = (
        "model_weights",
        "arch",
    )

    model_weights: str = fields.arg(
        name="model_weights",
        type=ty.Any,
        help="the weights of the model",
    )
    arch: list[tuple[str, str]] | None = fields.arg(
        name="arch", type=ty.Any, help="the architecture of the model"
    )

    def _run(self, job: "Job[MonaiTask]", rerun: bool = True) -> None:
        raise NotImplementedError(
            "Code to read a model weights + arch and execute in PyTorch"
        )
