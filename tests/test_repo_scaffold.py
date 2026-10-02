"""F0 · Repository scaffold. See spec/2026-10-01-repository-scaffold/validation.md."""

from __future__ import annotations

import importlib
import importlib.resources
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from irwm.utils import paths
from irwm.utils import slurm as slurm_utils
from irwm.utils.paths import MissingEnvVarError, ROOT_ENV_VARS, get_root, load_env, repo_root
from irwm.utils.slurm import cluster_plugins, on_slurm, resume_ckpt_path, sbatch_args
from irwm.utils.tracking import git_info

REPO = repo_root()
SMOKE_CONF = REPO / "projects" / "smoke" / "conf"
SMOKE_TRAIN = REPO / "projects" / "smoke" / "train.py"
DINO_FORESIGHT_COMMIT = "d98f765d2b743ec90fb38bc657a39c07c7ca4ab0"


def _git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


# --------------------------------------------------------------------------- Group 1
class TestPackageAndEnvironment:
    def test_irwm_subpackages_import(self):
        import irwm

        assert isinstance(irwm.__version__, str)
        for sub in ["data", "features", "detection", "worldmodel", "eval", "utils"]:
            importlib.import_module(f"irwm.{sub}")

    def test_pinned_versions(self):
        import numpy
        import torch

        assert sys.version_info[:2] == (3, 11)
        assert torch.__version__.startswith("2.2.0")
        assert int(numpy.__version__.split(".")[0]) == 1

    def test_conf_shipped_with_package(self):
        conf = importlib.resources.files("irwm.conf")
        for rel in ["irwm_base.yaml", "paths/default.yaml", "wandb/default.yaml",
                    "cluster/local.yaml", "cluster/slurm.yaml", "cluster/example_site.yaml"]:
            assert conf.joinpath(rel).is_file(), rel


# --------------------------------------------------------------------------- Group 2
class TestPaths:
    def test_roots_resolve(self, irwm_env):
        wrappers = {"data": paths.data_root, "cache": paths.cache_root,
                    "ckpt": paths.ckpt_root, "output": paths.output_root}
        for name in ROOT_ENV_VARS:
            got = get_root(name)
            assert got.is_absolute()
            assert got == irwm_env[name].resolve()
            assert wrappers[name]() == got

    def test_unknown_root(self):
        with pytest.raises(ValueError):
            get_root("bogus")

    @pytest.mark.parametrize("value", [None, "   "])
    def test_missing_var(self, irwm_env, monkeypatch, value):
        if value is None:
            monkeypatch.delenv("IRWM_CACHE_ROOT")
        else:
            monkeypatch.setenv("IRWM_CACHE_ROOT", value)
        with pytest.raises(MissingEnvVarError) as exc:
            paths.cache_root()
        assert isinstance(exc.value, KeyError)
        assert "IRWM_CACHE_ROOT" in str(exc.value)

    def test_tilde_expanded(self, monkeypatch):
        monkeypatch.setenv("IRWM_DATA_ROOT", "~/x")
        got = paths.data_root()
        assert "~" not in str(got) and got.is_absolute()

    def test_repo_root(self):
        assert (REPO / "uv.lock").is_file()
        assert (REPO / "lib" / "irwm").is_dir()

    def test_load_env_does_not_override(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("IRWM_DATA_ROOT=/from/dotenv\nIRWM_CKPT_ROOT=/from/dotenv\n")
        monkeypatch.setenv("IRWM_DATA_ROOT", "/from/shell")
        monkeypatch.delenv("IRWM_CKPT_ROOT", raising=False)
        assert load_env(env) is True
        assert os.environ["IRWM_DATA_ROOT"] == "/from/shell"
        assert os.environ["IRWM_CKPT_ROOT"] == "/from/dotenv"
        monkeypatch.delenv("IRWM_CKPT_ROOT")  # load_dotenv set it outside monkeypatch

    def test_load_env_missing_file(self, tmp_path):
        assert load_env(tmp_path / "missing.env") is False

    def test_load_env_expands_tilde(self, tmp_path, monkeypatch):
        # Hydra's oc.env passes the raw string through, so ~ must be expanded in the env itself.
        monkeypatch.setenv("IRWM_OUTPUT_ROOT", "~/irwm/outputs")
        load_env(tmp_path / "missing.env")
        assert os.environ["IRWM_OUTPUT_ROOT"] == str(Path("~/irwm/outputs").expanduser())

    def test_no_hardcoded_paths_in_lib(self):
        pattern = re.compile(r"[A-Za-z]:\\\\|/home/|/scratch/|/cluster/")
        for f in (REPO / "lib" / "irwm").rglob("*.py"):
            assert not pattern.search(f.read_text(encoding="utf-8")), f


# --------------------------------------------------------------------------- Group 3
class TestHydraComposition:
    def _compose(self, overrides=(), hydra_cfg=False):
        with initialize_config_dir(version_base="1.3", config_dir=str(SMOKE_CONF)):
            return compose("train", overrides=list(overrides), return_hydra_config=hydra_cfg)

    def test_default_compose(self, irwm_env):
        cfg = self._compose()
        assert cfg.paths.output_root == str(irwm_env["output"])
        assert cfg.cluster.name == "local"
        assert cfg.wandb.group == "smoke"
        assert cfg.wandb.project == "ir-world-model"

    def test_missing_env_fails_loudly(self, irwm_env, monkeypatch):
        monkeypatch.delenv("IRWM_DATA_ROOT")
        cfg = self._compose()
        with pytest.raises(InterpolationResolutionError):
            _ = cfg.paths.data_root

    def test_cluster_slurm(self, irwm_env, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "4242")
        cfg = self._compose(["cluster=slurm"], hydra_cfg=True)
        assert cfg.cluster.name == "slurm"
        assert cfg.cluster.accelerator == "gpu"
        assert cfg.hydra.launcher._target_.endswith("SlurmLauncher")
        assert cfg.hydra.launcher.gres == cfg.cluster.gres
        assert cfg.hydra.launcher.partition == cfg.cluster.partition
        assert cfg.run_name == "slurm-4242"

    def test_site_config_inherits_slurm(self, irwm_env):
        cfg = self._compose(["cluster=example_site"], hydra_cfg=True)
        assert cfg.cluster.name == "example_site"
        assert cfg.cluster.accelerator == "gpu"          # inherited from cluster/slurm.yaml
        assert cfg.cluster.partition == "TODO_PARTITION"  # set by the site file
        assert cfg.hydra.launcher._target_.endswith("SlurmLauncher")
        assert cfg.hydra.launcher.partition == "TODO_PARTITION"
        assert any("slurm/env.sh" in s for s in cfg.hydra.launcher.setup)

    def test_run_dir_under_output_root(self, irwm_env):
        cfg = self._compose(hydra_cfg=True)
        run_dir = Path(cfg.hydra.run.dir)
        assert run_dir.is_relative_to(irwm_env["output"] / "smoke")


# --------------------------------------------------------------------------- Group 4
class TestTrackingAndSlurm:
    def test_git_info_in_repo(self):
        info = git_info()
        assert re.fullmatch(r"[0-9a-f]{40}", info["commit"])
        assert isinstance(info["dirty"], bool)

    def test_git_info_outside_repo(self, tmp_path):
        assert git_info(tmp_path) == {"commit": "unknown", "dirty": False}

    def test_on_slurm(self, monkeypatch):
        from lightning.pytorch.plugins.environments import SLURMEnvironment

        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        assert on_slurm() is False and cluster_plugins() == []
        monkeypatch.setenv("SLURM_JOB_ID", "1")
        plugins = cluster_plugins()
        assert on_slurm() is True and len(plugins) == 1
        assert isinstance(plugins[0], SLURMEnvironment) and plugins[0].auto_requeue

    def test_resume_ckpt_path(self, tmp_path):
        assert resume_ckpt_path(tmp_path) is None
        (tmp_path / "last.ckpt").touch()
        assert resume_ckpt_path(tmp_path) == str(tmp_path / "last.ckpt")
        for n in ["hpc_ckpt_1.ckpt", "hpc_ckpt_10.ckpt", "hpc_ckpt_2.ckpt", "hpc_ckpt_x.ckpt"]:
            (tmp_path / n).touch()
        assert resume_ckpt_path(tmp_path) == str(tmp_path / "hpc_ckpt_10.ckpt")


class TestSbatchFlags:
    def test_sbatch_args_mapping(self):
        cluster = OmegaConf.create({
            "partition": "gpu", "account": None, "qos": "", "constraint": None,
            "gres": "gpu:a100:2", "num_nodes": 1, "devices": 2, "cpus_per_task": 16,
            "mem_gb": 64, "timeout_min": 90, "extra_sbatch_args": ["--exclude=node1"],
        })
        assert sbatch_args(cluster) == [
            "--partition=gpu", "--gres=gpu:a100:2", "--nodes=1", "--ntasks-per-node=2",
            "--cpus-per-task=16", "--mem=64G", "--time=90", "--exclude=node1",
        ]

    def test_cli_refuses_todo_and_applies_cluster_overrides(self, capsys):
        assert slurm_utils._main(["example_site"]) == 1
        assert "TODO" in capsys.readouterr().err
        assert slurm_utils._main(["example_site", "cluster.partition=gpu",
                                  "cluster.timeout_min=30", "seed=3"]) == 0
        flags = capsys.readouterr().out.split()
        assert "--partition=gpu" in flags and "--time=30" in flags


# --------------------------------------------------------------------------- Shell scripts
def _find_bash() -> str | None:
    """Git Bash on Windows (never WSL's System32 bash), plain bash elsewhere."""
    if sys.platform != "win32":
        return shutil.which("bash")
    git = shutil.which("git")
    if git:
        # git.exe lives in <Git>/cmd or <Git>/mingw64/bin depending on how PATH was set up.
        for git_root in Path(git).resolve().parents[:3]:
            for cand in (git_root / "bin" / "bash.exe", git_root / "usr" / "bin" / "bash.exe"):
                if cand.is_file():
                    return str(cand)
    return None


BASH = _find_bash()


def _write_exe(path: Path, text: str) -> None:
    path.write_text(text, newline="\n")
    path.chmod(0o755)


@pytest.mark.skipif(BASH is None, reason="bash not available")
class TestShellScripts:
    @pytest.mark.parametrize("script", ["env.sh", "job.sbatch", "submit.sh", "setup_env.sh"])
    def test_syntax(self, script):
        subprocess.run([BASH, "-n", (REPO / "slurm" / script).as_posix()], check=True)

    def test_no_crlf(self):
        for f in (REPO / "slurm").iterdir():
            if f.suffix in {".sh", ".sbatch"}:
                assert b"\r\n" not in f.read_bytes(), f

    @staticmethod
    def _source_env_sh(tmp_path: Path, dotenv: str, **env_extra: str) -> dict[str, str]:
        """Source a copy of env.sh in an isolated fake repo (tmp/repo) with HOME=tmp/home."""
        repo, home = tmp_path / "repo", tmp_path / "home"
        (repo / "slurm").mkdir(parents=True, exist_ok=True)
        home.mkdir(exist_ok=True)
        shutil.copy(REPO / "slurm" / "env.sh", repo / "slurm" / "env.sh")
        (repo / ".env").write_bytes(dotenv.encode())
        env = {k: v for k, v in os.environ.items() if not k.startswith(("IRWM_", "WANDB_"))}
        env.update(HOME=home.as_posix(), **env_extra)
        names = "IRWM_DATA_ROOT IRWM_CACHE_ROOT IRWM_CKPT_ROOT WANDB_API_KEY"
        script = f'source "$1"; for v in {names}; do echo "$v=${{!v:-}}"; done'
        out = subprocess.run(
            [BASH, "-c", script, "_", (repo / "slurm" / "env.sh").as_posix()],
            env=env, capture_output=True, text=True, check=True,
        ).stdout
        return dict(line.split("=", 1) for line in out.splitlines())

    def test_env_sh_parses_dotenv_and_key_file(self, tmp_path):
        (tmp_path / "home").mkdir()
        (tmp_path / "home" / "mykey").write_text("abc123\n")
        dotenv = (
            'export IRWM_DATA_ROOT="/data/with space"\r\n'
            "IRWM_CACHE_ROOT=~/cache   # inline comment\r\n"
            "# a comment line\r\n"
            "IRWM_CKPT_ROOT=/from/dotenv\r\n"
            "WANDB_API_KEY_FILE=~/mykey\r\n"
        )
        got = self._source_env_sh(tmp_path, dotenv, IRWM_CKPT_ROOT="/from/shell")
        assert got["IRWM_DATA_ROOT"] == "/data/with space"
        assert got["IRWM_CACHE_ROOT"].endswith("/home/cache") and "~" not in got["IRWM_CACHE_ROOT"]
        assert got["IRWM_CKPT_ROOT"] == "/from/shell"   # shell wins over .env
        assert got["WANDB_API_KEY"] == "abc123"         # from WANDB_API_KEY_FILE

    def test_env_sh_default_key_file_and_no_override(self, tmp_path):
        (tmp_path / "home").mkdir()
        (tmp_path / "home" / ".wandb_api_key").write_text("  fromfile \n")
        assert self._source_env_sh(tmp_path, "")["WANDB_API_KEY"] == "fromfile"
        got = self._source_env_sh(tmp_path, "", WANDB_API_KEY="preset")
        assert got["WANDB_API_KEY"] == "preset"

    def test_submit_sh_builds_sbatch_call(self, tmp_path):
        stub, record, out_root = tmp_path / "stub", tmp_path / "sbatch_args.txt", tmp_path / "out"
        stub.mkdir()
        _write_exe(stub / "sbatch", f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "{record.as_posix()}"\n')
        # `uv run --frozen python <args>` -> this interpreter with <args>
        py = Path(sys.executable).as_posix()
        _write_exe(stub / "uv", f'#!/usr/bin/env bash\nshift 3\nexec "{py}" "$@"\n')
        env = {**os.environ, "IRWM_OUTPUT_ROOT": out_root.as_posix(), "HOME": tmp_path.as_posix(),
               "PATH": os.pathsep.join([str(stub), os.environ["PATH"]])}
        env.pop("SBATCH_EXTRA", None)
        proc = subprocess.run(
            [BASH, (REPO / "slurm" / "submit.sh").as_posix(), "example_site",
             "projects/smoke/train.py", "cluster.partition=gpu", "trainer.max_epochs=1"],
            env=env, capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        args = record.read_text().splitlines()
        assert "--job-name=irwm-smoke" in args
        assert "--partition=gpu" in args and "--gres=gpu:1" in args
        assert any(a.startswith("--output=") and a.endswith("slurm_logs/%x-%j.out") for a in args)
        i = args.index("slurm/job.sbatch")
        assert args[i + 1:] == ["projects/smoke/train.py", "cluster=example_site",
                                "cluster.partition=gpu", "trainer.max_epochs=1"]
        assert (out_root / "slurm_logs").is_dir()


# --------------------------------------------------------------------------- Smoke run
def _run_smoke(env_roots: dict[str, Path], *overrides: str) -> dict:
    env = {**os.environ, "WANDB_MODE": "disabled"}
    env.pop("SLURM_JOB_ID", None)
    before = set(env_roots["output"].rglob("result.json")) if env_roots["output"].exists() else set()
    proc = subprocess.run(
        [sys.executable, str(SMOKE_TRAIN), *overrides],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    new = set(env_roots["output"].rglob("result.json")) - before
    assert len(new) == 1, new
    return json.loads(new.pop().read_text())


class TestSmokeRun:
    def test_train_and_resume(self, irwm_env):
        status_before = _git("status", "--porcelain")

        r1 = _run_smoke(irwm_env, "trainer.max_epochs=2", "run_name=pytest")
        ckpt_dir = irwm_env["ckpt"] / "smoke" / "pytest"
        assert (ckpt_dir / "last.ckpt").is_file()
        assert math.isfinite(r1["final_val_loss"])
        assert r1["epochs_completed"] == 2 and r1["resumed_from"] is None
        assert re.fullmatch(r"[0-9a-f]{40}", r1["git"]["commit"])

        r2 = _run_smoke(irwm_env, "trainer.max_epochs=4", "run_name=pytest")
        assert r2["epochs_completed"] == 4
        assert r2["resumed_from"].endswith("last.ckpt")
        assert r2["final_val_loss"] < r2["initial_val_loss"]

        # Nothing written into the working tree outside git-ignored paths.
        assert _git("status", "--porcelain") == status_before


# --------------------------------------------------------------------------- Architecture
class TestArchitecture:
    def test_source_files_tracked_by_git(self):
        # Guards against .gitignore rules swallowing source (an unanchored `wandb/` once hid
        # lib/irwm/conf/wandb/, so a fresh clone had no wandb config group).
        sources = [p for d in ("lib", "projects", "slurm", "tests")
                   for p in (REPO / d).rglob("*")
                   if p.suffix in {".py", ".yaml", ".sh", ".sbatch", ".toml"}
                   and ".venv" not in p.parts and "__pycache__" not in p.parts]
        rel = [p.relative_to(REPO).as_posix() for p in sources]
        ignored = subprocess.run(["git", "check-ignore", "--no-index", *rel], cwd=REPO,
                                 capture_output=True, text=True).stdout.split()
        assert ignored == [], f"source files ignored by .gitignore: {ignored}"

    def test_lib_does_not_import_projects(self):
        pattern = re.compile(r"^\s*(from|import)\s+projects\b", re.M)
        for f in (REPO / "lib").rglob("*.py"):
            assert not pattern.search(f.read_text(encoding="utf-8")), f

    def test_dino_foresight_submodule_pinned(self):
        status = _git("submodule", "status", "third_party/DINO-Foresight")
        assert not status.startswith(("+", "-", "U")), "submodule not initialised or off its pin"
        assert status.strip().startswith(DINO_FORESIGHT_COMMIT), status
        assert _git("status", "--porcelain", cwd=REPO / "third_party" / "DINO-Foresight") == ""


# --------------------------------------------------------------------------- Group 5
@pytest.mark.data
def test_flir_dataset_present():
    """Real-data check (skipped by default): the raw dataset is where IRWM_DATA_ROOT says."""
    load_env()
    assert (paths.data_root() / "flir_adas_v2").is_dir()
