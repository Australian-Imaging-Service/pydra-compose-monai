from pathlib import Path
from fileformats.generic import Directory
from pydra.compose import shell


@shell.define
class BidsApp(shell.Task["BidsApp.Outputs"]):

    executable = ""  # This should be changed to None once https://github.com/nipype/pydra/pull/822 is merged

    dataset_path: Directory = shell.arg(
        help="Path to BIDS dataset in the container",
        position=1,
        argstr="'{dataset_path}'",
    )

    output_path: Path = shell.arg(
        help="Directory where outputs will be written in the container",
        position=2,
        argstr="'{output_path}'",
    )

    analysis_level: str = shell.arg(
        help="The analysis level the app will be run at",
        position=3,
        argstr="",
        default="participant",
        allowed_values=["participant", "group"],
    )

    participant_label: str | None = shell.arg(
        help="The IDs to include in the analysis",
        argstr="--participant-label ",
        default=None,
        position=4,
    )

    flags: str | None = shell.arg(
        help="Additional flags to pass to the app",
        argstr="",
        default=None,
        position=-1,
    )

    work_dir: Path | None = shell.arg(
        help="Directory where the nipype temporary working directories will be stored",
        argstr="--work-dir '{work_dir}'",
        default=None,
    )

    class Outputs(shell.Outputs):

        pass
