"""Package pinned planner source variants for the server-only ablation run."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / "freeze.json").read_text(encoding="utf-8"))
OUTPUT = HERE / "execution_bundle.tar.gz"
PACKAGE_REL = Path("输电线路智能路径规划系统V5.0/transmission_path_planner")


def git_bytes(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], check=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE).stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if OUTPUT.exists():
    raise SystemExit(f"preserving existing bundle; refusing to overwrite: {OUTPUT}")

head = git_bytes("rev-parse", "HEAD").decode("ascii").strip()
if head != PLAN["source_bundle_head"]:
    raise SystemExit(f"expected source HEAD {PLAN['source_bundle_head']}, got {head}")

stage = Path(tempfile.mkdtemp(prefix="nodata_ablation_sources_"))
shared = stage / "shared"
shared.mkdir()
source_archive = git_bytes(
    "archive", "--format=tar", head, PACKAGE_REL.as_posix())
with tarfile.open(fileobj=io.BytesIO(source_archive), mode="r:") as archive:
    archive.extractall(shared)
base_package = shared / PACKAGE_REL
if not (base_package / "__main__.py").is_file():
    raise SystemExit(f"could not find archived package: {base_package}")

source_manifest = {
    "experiment_id": PLAN["experiment_id"],
    "shared_package_commit": head,
    "variants": {},
}
sources = stage / "sources"
for arm in PLAN["arms"]:
    source_root = sources / arm["arm"]
    package = source_root / "transmission_path_planner"
    package.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base_package, package)
    for filename in ("data.py", "cost.py"):
        commit = arm["data_commit" if filename == "data.py" else "cost_commit"]
        revision = f"{commit}:" + (PACKAGE_REL / filename).as_posix()
        (package / filename).write_bytes(git_bytes("show", revision))
    source_manifest["variants"][arm["arm"]] = {
        "source_revision": arm["source_revision"],
        "python_file_sha256": {
            path.relative_to(package).as_posix(): sha256(path)
            for path in sorted(package.rglob("*.py"))
        },
    }

(stage / "freeze.json").write_bytes((HERE / "freeze.json").read_bytes())
(stage / "run_server_experiment.py").write_bytes(
    (HERE / "run_server_experiment.py").read_bytes())
(stage / "source_manifest.json").write_text(
    json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8")

with tarfile.open(OUTPUT, mode="w:gz") as bundle:
    for name in ("freeze.json", "run_server_experiment.py", "source_manifest.json", "sources"):
        bundle.add(stage / name, arcname=name)

print(json.dumps({
    "bundle": str(OUTPUT),
    "size_bytes": OUTPUT.stat().st_size,
    "sha256": sha256(OUTPUT),
    "source_head": head,
}, ensure_ascii=False, indent=2))
