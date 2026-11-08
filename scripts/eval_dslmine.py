"""Mines a DSL from one half of a real codebase and measures it on the other half.

Folders, not files, are split, so unseen code comes from packages the miner never read. Copy-paste
detection is the baseline: it can only reuse statements identical to ones in the mined half.
"""

import argparse
import json
import random
import time
from pathlib import Path

import transformers

from payrules.dslmine import apply_templates, exact_duplicates, load_corpus, mine

ROOT = Path(transformers.__file__).parent / "models"


def read_sources(folders: list[Path]):
    for folder in folders:
        for path in sorted(folder.glob("modeling_*.py")):
            yield str(path), path.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folders", type=int, default=120, help="Model folders per half")
    parser.add_argument("--min-size", type=int, default=15, help="Smallest statement mined, in tree nodes")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path, default=Path("runs/dslmine.json"))
    parser.add_argument("--templates-out", type=Path, default=Path("runs/dsl_templates.txt"))
    args = parser.parse_args()

    folders = sorted(p for p in ROOT.iterdir() if p.is_dir() and any(p.glob("modeling_*.py")))
    random.Random(args.seed).shuffle(folders)
    train_folders = folders[: args.folders]
    test_folders = folders[args.folders : 2 * args.folders]

    started = time.time()
    train = load_corpus(read_sources(train_folders), args.min_size)
    test = load_corpus(read_sources(test_folders), args.min_size)
    print(f"loaded {train.files}+{test.files} files in {time.time() - started:.0f}s "
          f"(skipped {train.skipped_files + test.skipped_files})", flush=True)

    results: dict = {
        "transformers_version": transformers.__version__,
        "available_model_folders": len(folders),
        "train": {"folders": len(train_folders), "files": train.files, "tree_nodes": train.total_nodes},
        "test": {"folders": len(test_folders), "files": test.files, "tree_nodes": test.total_nodes},
        "by_vocabulary_size": {},
    }
    baseline, entries = exact_duplicates(train, test)
    results["unlimited_copy_paste"] = {"entries": entries, "unseen_coverage": baseline.nodes / test.total_nodes}
    print("unlimited copy-paste:", json.dumps(results["unlimited_copy_paste"]), flush=True)

    last = None
    for size in (10, 50, 200, 1000):
        templates, covered = mine(train, size)
        unseen = apply_templates(templates, test)
        same_size, _ = exact_duplicates(train, test, max_entries=len(templates))
        results["by_vocabulary_size"][size] = {
            "operations": len(templates),
            "parameters_per_operation": sum(len(t.parameter_types) for t in templates) / max(len(templates), 1),
            "train_coverage": covered.nodes / train.total_nodes,
            "unseen_coverage": unseen.nodes / test.total_nodes,
            "unseen_rewrites": unseen.rewrites,
            "copy_paste_same_size_unseen_coverage": same_size.nodes / test.total_nodes,
        }
        print(size, json.dumps(results["by_vocabulary_size"][size]), flush=True)
        last = templates

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    args.templates_out.write_text("\n\n".join(t.render() for t in last[:30]), encoding="utf-8")
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
