import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from .config import load_config
from .io import save_json
from .report import finalize, finalize_scratch
from .study import qualification, read_metrics


def refresh(root):
    root = Path(root)
    started = time.time()
    config = load_config(root / "config.yaml")
    work = Path(config.get("workspace", root / "work"))
    work.mkdir(parents=True, exist_ok=True)
    saved = json.loads((root / "summary.json").read_text())
    manifest = saved.get("provenance") or json.loads((work / "manifest.json").read_text())
    manifest["workspace"] = str(work)
    originals = work / "analysis_original"
    originals.mkdir(exist_ok=True)
    for name in ("summary.json", "REPORT.md"):
        if not (originals / name).exists():
            shutil.copy2(root / name, originals / name)
    selected = saved["qualification"]
    history = manifest.setdefault("qualification_history", {})
    for path in sorted((work / "pilot").glob("*/qualification.json")):
        pilot = path.parent
        old = json.loads(path.read_text())
        backup = originals / f"{pilot.name}_qualification.json"
        if not backup.exists():
            shutil.copy2(path, backup)
        runs = {str(file.parent.relative_to(pilot)): read_metrics(file.parent) for file in sorted(pilot.rglob("run.json")) + sorted(pilot.rglob("metrics.json"))}
        current = qualification(pilot.name, runs, old["random_scores"])
        current["jobs"] = old.get("jobs", [])
        save_json(path, current)
        history[pilot.name] = current
        if pilot.name == config["suite"]:
            selected = current
    source_files = sorted(Path(__file__).resolve().parents[1].rglob("*.py"))
    source_hashes = {str(file): hashlib.sha256(file.read_bytes()).hexdigest() for file in source_files}
    patterns = ("metrics.json", "run.json", "episodes.csv", "evaluations.csv", "blocks.csv", "diagnostics.csv", "probes.csv", "probe_summary.csv")
    raw_files = sorted(file for base in (root, work / "pilot") for pattern in patterns for file in base.rglob(pattern))
    raw_hashes = {str(file): hashlib.sha256(file.read_bytes()).hexdigest() for file in raw_files}
    analysis = {"started_at": started, "training_source_sha256": manifest["source_sha256"], "analysis_source_hashes": source_hashes, "raw_metric_hashes": raw_hashes, "additional_training_steps": 0, "additional_evaluation_steps": 0, "checkpoint_audits_reused": True, "note": "Regenerated from saved data. Weak fresh-probe references cannot establish plasticity loss. Original qualification and report are retained in the temporary workspace. No training or evaluation is performed."}
    save_json(work / "analysis.json", analysis)
    finalizer = finalize_scratch if config["arms"] == ["scratch"] else finalize
    report = finalizer(root, config, selected, manifest, float("inf"), saved_artifacts=saved)
    report["profile"] = saved.get("profile", {})
    save_json(root / "summary.json", report)
    analysis["seconds"] = time.time() - started
    save_json(work / "analysis.json", analysis)
    print(f"Regenerated {root / 'REPORT.md'} from saved data: {report['outcome']}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Regenerate results without training or evaluation")
    parser.add_argument("--out", type=Path, default=Path("results"))
    refresh(parser.parse_args().out)


if __name__ == "__main__":
    main()
