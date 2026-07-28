from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_stage_h3_environment.sh"


class StageH3CudaLauncherTest(unittest.TestCase):
    def make_fake_environment(self, root: Path, release: str) -> Path:
        bin_dir = root / "envs" / "h3_splatad" / "bin"
        bin_dir.mkdir(parents=True)
        python = bin_dir / "python"
        python.write_text(
            '#!/usr/bin/env bash\necho "fake-python:$*"\n'
        )
        python.chmod(0o755)
        nvcc = bin_dir / "nvcc"
        nvcc.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/env bash
                echo "Cuda compilation tools, release {release}, V{release}.0"
                """
            )
        )
        nvcc.chmod(0o755)
        return bin_dir

    def run_launcher(
        self, root: Path, bin_dir: Path, *arguments: str
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["H3_ROOT"] = str(root)
        environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
        return subprocess.run(
            ["bash", str(LAUNCHER), *arguments],
            cwd=REPO_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_accepts_cuda_11_8_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bin_dir = self.make_fake_environment(root, "11.8")
            result = self.run_launcher(root, bin_dir, "--print")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("release 11.8", result.stdout)
        self.assertIn(str(bin_dir / "nvcc"), result.stdout)

    def test_rejects_cuda_11_5_before_running_python(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bin_dir = self.make_fake_environment(root, "11.5")
            result = self.run_launcher(
                root, bin_dir, "python", "must_not_run.py"
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires CUDA 11.8", result.stderr)
        self.assertNotIn("must_not_run.py", result.stdout)

    def test_maps_python_to_the_h3_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bin_dir = self.make_fake_environment(root, "11.8")
            result = self.run_launcher(root, bin_dir, "python", "probe.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fake-python:probe.py", result.stdout)


if __name__ == "__main__":
    unittest.main()
