import os
import pytest
import requests
from warnings import warn
import typing as ty
from tempfile import mkdtemp
from pathlib import Path
import docker
from fileformats.medimage import NiftiGzX

PKG_DIR = Path(__file__).parent


@pytest.fixture(scope="session")
def build_cache_dir() -> Path:
    # build_cache_dir = Path.home() / '.frametree-test-build-cache'
    # if build_cache_dir.exists():
    #     shutil.rmtree(build_cache_dir)
    # build_cache_dir.mkdir()
    return Path(mkdtemp())
    # return build_cache_dir


@pytest.fixture
def work_dir() -> Path:  # type: ignore[misc]
    # work_dir = Path.home() / '.frametree-tests'
    # work_dir.mkdir(exist_ok=True)
    # return work_dir
    work_dir = mkdtemp()
    yield Path(work_dir)
    # shutil.rmtree(work_dir)


@pytest.fixture(scope="session")
def pkg_dir() -> Path:
    return PKG_DIR


@pytest.fixture(scope="session")
def nifti_sample_dir(pkg_dir):
    return pkg_dir / "test-data" / "nifti"


# FIXME: should be converted to python script to be Windows compatible
@pytest.fixture(scope="session")
def mock_bids_app_script():
    file_tests = ""
    for inpt_path, datatype in [
        ("anat/T1w", NiftiGzX),
        ("anat/T2w", NiftiGzX),
        ("dwi/dwi", NiftiGzX),
    ]:
        subdir, suffix = inpt_path.split("/")
        fpath = (
            f"$BIDS_DATASET/sub-${{SUBJ_ID}}/{subdir}/"
            f"sub-${{SUBJ_ID}}_{suffix}{datatype.ext}"
        )
        file_tests += f"""
        if [ ! -f {fpath} ]; then
            echo "Did not find {suffix} file at {fpath}"
            exit 1;
        fi
        """

    return f"""#!/bin/sh
BIDS_DATASET=$1
OUTPUTS_DIR=$2
SUBJ_ID=$5
{file_tests}
# Write mock output files to 'derivatives' Directory
mkdir -p $OUTPUTS_DIR
echo 'file1' > $OUTPUTS_DIR/sub-${{SUBJ_ID}}_file1.txt
echo 'file2' > $OUTPUTS_DIR/sub-${{SUBJ_ID}}_file2.txt
"""


# For debugging in IDE's don't catch raised exceptions and let the IDE
# break at it
if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call: pytest.CallInfo[ty.Any]) -> None:
        if call.excinfo is not None:
            raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo: pytest.ExceptionInfo[BaseException]) -> None:
        raise excinfo.value

    CATCH_CLI_EXCEPTIONS = False
else:
    CATCH_CLI_EXCEPTIONS = True


@pytest.fixture(scope="session")
def bids_validator_app_image(
    bids_validator_app_script, bids_validator_docker, build_cache_dir
):
    return build_app_image(
        BIDS_VALIDATOR_APP_IMAGE,
        bids_validator_app_script,
        build_cache_dir,
        base_image=bids_validator_docker,
    )


def build_app_image(tag_name, script, build_cache_dir, base_image):
    dc = docker.from_env()

    # Create executable that runs validator then produces some mock output
    # files
    build_dir = build_cache_dir / tag_name.replace(":", "__i__")
    build_dir.mkdir()
    launch_sh = build_dir / "launch.sh"
    with open(launch_sh, "w") as f:
        f.write(script)

    # Build mock BIDS app image
    with open(build_dir / "Dockerfile", "w") as f:
        f.write(
            f"""FROM {base_image}
ADD ./launch.sh /launch.sh
RUN chmod +x /launch.sh
ENTRYPOINT ["/launch.sh"]"""
        )

    dc.images.build(path=str(build_dir), tag=tag_name)

    return tag_name


@pytest.fixture(scope="session")
def bids_validator_docker():
    dc = docker.from_env()
    try:
        dc.images.pull(BIDS_VALIDATOR_DOCKER)
    except requests.exceptions.HTTPError:
        warn("No internet connection, so couldn't download latest BIDS validator")
    return BIDS_VALIDATOR_DOCKER


@pytest.fixture(scope="session")
def bids_validator_app_script():
    return f"""#!/bin/sh
# Echo inputs to get rid of any quotes
BIDS_DATASET=$(echo $1)
OUTPUTS_DIR=$(echo $2)
SUBJ_ID=$5
# Run BIDS validator to check whether BIDS dataset is created properly
output=$(/usr/local/bin/bids-validator "$BIDS_DATASET")
if [[ "$output" != *"{SUCCESS_STR}"* ]]; then
    echo "BIDS validation was not successful, exiting:\n "
    echo $output
    exit 1;
fi
# Write mock output files to 'derivatives' Directory
mkdir -p $OUTPUTS_DIR
echo 'file1' > $OUTPUTS_DIR/sub-${{SUBJ_ID}}_file1.txt
echo 'file2' > $OUTPUTS_DIR/sub-${{SUBJ_ID}}_file2.txt
"""


BIDS_VALIDATOR_DOCKER = "bids/validator:latest"
SUCCESS_STR = "This dataset appears to be BIDS compatible"
BIDS_VALIDATOR_APP_IMAGE = "pydra-compose-monai-validator"
