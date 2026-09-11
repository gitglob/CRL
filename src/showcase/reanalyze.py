import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from ..utils.config import load_config
from ..utils.io import save_json
from .report import finalize, finalize_scratch
from .study import qualification, read_metrics


def refresh(root):
    root = Path(root)
    started = time.time()
    config = load_config(root / "selected_config.yaml")
    manifest = json.loads((root / "manifest.json").read_text())
    saved = json.loads((root / "summary.json").read_text())
    originals = root / "analysis_original"
    originals.mkdir(exist_ok=True)
    for name in ("summary.json", "REPORT.md"):
        if not (originals / name).exists():
            shutil.copy2(root / name, originals / name)
    selected = None
    for pilot in sorted((root / "pilot").iterdir()):
        path = pilot / "qualification.json"
        if not path.exists():
            continue
        old = json.loads(path.read_text())
        backup = originals / f"{pilot.name}_qualification.json"
        if not backup.exists():
            shutil.copy2(path, backup)
        runs = {str(file.parent.relative_to(pilot)): read_metrics(file.parent) for file in pilot.rglob("metrics.json")}
        current = qualification(pilot.name, runs, old["random_scores"])
        current["jobs"] = old.get("jobs", [])
        save_json(path, current)
        if pilot.name == config["suite"]:
            selected = current
    if selected is None:
        raise ValueError("The selected suite has no saved pilot qualification")
    source_files = [Path(__file__), Path(__file__).with_name("study.py"), Path(__file__).with_name("report.py")]
    source_hashes = {str(file): hashlib.sha256(file.read_bytes()).hexdigest() for file in source_files}
    raw_files = sorted((root / "main").rglob("metrics.json")) + sorted((root / "pilot").rglob("metrics.json"))
    raw_hashes = {str(file.relative_to(root)): hashlib.sha256(file.read_bytes()).hexdigest() for file in raw_files}
    analysis = {"started_at": started, "training_source_sha256": manifest["source_sha256"], "analysis_source_hashes": source_hashes, "raw_metric_hashes": raw_hashes, "additional_training_steps": 0, "additional_evaluation_steps": 0, "checkpoint_audits_reused": True, "note": "Post-run reporting correction: separate the raw probe deficit threshold from evidence based on a learned fresh reference. MinAtar uses its existing above-random learning threshold for the probe reference too. The original run was already inconclusive. Original qualification and report are retained in analysis_original."}
    save_json(root / "analysis.json", analysis)
    finalizer = finalize_scratch if config["arms"] == ["scratch"] else finalize
    report = finalizer(root, config, selected, manifest, float("inf"), saved_artifacts=saved)
    analysis["seconds"] = time.time() - started
    save_json(root / "analysis.json", analysis)
    print(f"Regenerated {root / 'REPORT.md'} from saved data: {report['outcome']}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Regenerate showcase analysis without training or evaluation")
    parser.add_argument("root", type=Path, nargs="?", default=Path("results/showcase"))
    refresh(parser.parse_args().root)


if __name__ == "__main__":
    main()
