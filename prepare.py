"""Export a pinned author checkout and apply this experiment's data patch."""

import argparse
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


BASE = Path(__file__).resolve().parent
LOCK = json.loads((BASE / "upstreams.json").read_text(encoding="utf-8"))


def prepare(method, checkout=None):
    spec = LOCK[method]
    dest = BASE / "upstream" / method
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        raise FileExistsError("Prepared folder already exists: " + str(dest))
    with TemporaryDirectory(dir=dest.parent) as tmp:
        tmp = Path(tmp)
        if checkout is None:
            checkout = tmp / "git"
            subprocess.run(["git", "init", "-q", str(checkout)], check=True)
            git = ["git", "-C", str(checkout)]
            subprocess.run(git + ["remote", "add", "origin", spec["url"]], check=True)
            subprocess.run(git + ["config", "remote.origin.promisor", "true"], check=True)
            subprocess.run(git + ["config", "remote.origin.partialclonefilter", "blob:none"], check=True)
            subprocess.run(git + ["fetch", "-q", "--filter=blob:none", "--depth=1",
                                  "origin", spec["commit"]], check=True)
        git = ["git", "-C", str(checkout)]
        files = subprocess.check_output(git + ["ls-tree", "-r", "--name-only", spec["commit"]],
                                        encoding="utf-8").splitlines()
        files = [f for f in files if f.endswith(".py") or Path(f).name.lower() in
                 ("readme.md", "requirements.txt", "license", "license.md", "license.txt")]
        staged = tmp / "source"
        # Read only selected blobs, including available upstream license notices.
        for name in files:
            path = staged / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as stream:
                subprocess.run(git + ["cat-file", "blob", spec["commit"] + ":" + name],
                               stdout=stream, check=True)
        # This is an exported tree inside a parent Git repo. Apply only within the export.
        env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(tmp))
        patch = str(BASE / "patches" / (spec.get("patch", method) + ".patch"))
        apply = ["git", "apply", "--no-index", "--ignore-space-change"]
        subprocess.run(apply + ["--check", patch], cwd=staged, env=env, check=True)
        subprocess.run(apply + [patch], cwd=staged, env=env, check=True)
        (staged / "UPSTREAM_COMMIT").write_text(spec["commit"] + "\n", encoding="ascii")
        staged.rename(dest)
    print(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=tuple(LOCK))
    parser.add_argument("--from-checkout", type=Path,
                        help="Use this local Git repository instead of fetching from the author.")
    args = parser.parse_args()
    prepare(args.method, args.from_checkout)
