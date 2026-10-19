"""The MuJoCo pair acceptance runs on CI and never passes by skipping (#407 review follow-up 6).

`tests/test_base_pose_mujoco_pair_pick.py` needs the so101 MJCF at
`assets/mjcf/so101` (`test_multi_arm_physics.MJCF`). The suite step fetches
the meshes outside the checkout on purpose (there the MJCF would turn on every
MuJoCo test), so the file skipped on every CI run. Now the `tests` job ends
with its own step: it fetches the MJCF where the test reads it (pinned to a
commit, like the meshes) and runs only this file with
`CASCADE_REQUIRE_MUJOCO_PAIR=1`, under which a missing requirement is a
collection failure instead of a skip.
"""

from __future__ import annotations

import os
import subprocess
import sys

import yaml

from conftest import REPO

PAIR_FILE = "tests/test_base_pose_mujoco_pair_pick.py"
REQUIRE_ENV = "CASCADE_REQUIRE_MUJOCO_PAIR"


def _tests_job_steps():
    ci = yaml.safe_load((REPO / ".github" / "workflows" / "ci.yml").read_text())
    return ci["jobs"]["tests"]["steps"]


def _index(steps, predicate, what):
    found = [i for i, step in enumerate(steps) if predicate(step)]
    assert found, f"no step {what}"
    return found[0]


def test_the_tests_job_runs_the_mujoco_pair_after_fetching_its_mjcf():
    """Both CI legs (the `tests` matrix) fetch the MJCF to the default
    `assets/` (no `--assets`) AFTER the suite step, then run only the pair
    file with the requirement switch on."""
    steps = _tests_job_steps()
    runs = [str(step.get("run", "")) for step in steps]
    suite = _index(steps, lambda s: "pytest tests/ " in str(s.get("run", "")), "running the suite")
    fetch = _index(steps, lambda s: any(
        line.strip() == "uv run python scripts/fetch_robot_assets.py so101"
        for line in str(s.get("run", "")).splitlines()), "fetching the so101 MJCF into assets/")
    pair = _index(steps, lambda s: PAIR_FILE in str(s.get("run", "")), f"running {PAIR_FILE}")
    assert suite < fetch <= pair, runs
    assert str((steps[pair].get("env") or {}).get(REQUIRE_ENV)) == "1", steps[pair]
    pair_runs = [line for line in runs[pair].splitlines() if PAIR_FILE in line]
    assert pair_runs == [f"uv run pytest {PAIR_FILE} -v -rs"], runs[pair]
    # nothing else runs with the MJCF in assets/: the pair step is the job's last
    assert pair == len(steps) - 1, runs


def _run_pair_file(tmp_path, required: bool):
    """The pair file in a subprocess where `import mujoco` fails (a plugin
    blocks it), so its requirements are missing whatever this host has."""
    (tmp_path / "no_mujoco.py").write_text("import sys\nsys.modules['mujoco'] = None\n")
    env = {k: v for k, v in os.environ.items() if k != REQUIRE_ENV}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(tmp_path), env.get("PYTHONPATH")]))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if required:
        env[REQUIRE_ENV] = "1"
    return subprocess.run(
        [sys.executable, "-m", "pytest", PAIR_FILE, "-q", "-rs", "-p", "no_mujoco", "-p", "no:cacheprovider"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=300)


def test_without_its_requirements_the_pair_file_fails_when_required(tmp_path):
    out = _run_pair_file(tmp_path, required=True)
    text = out.stdout + out.stderr
    assert out.returncode != 0, text
    assert f"{REQUIRE_ENV}=1 but this acceptance cannot run" in text, text
    assert "skipped" not in text.split("short test summary info")[-1], text


def test_without_its_requirements_the_pair_file_still_skips_by_default(tmp_path):
    """Local runs and the suite step: unchanged (2 skipped, exit 0)."""
    out = _run_pair_file(tmp_path, required=False)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "2 skipped" in out.stdout, out.stdout
