"""Phase 1: harvest current, human-written GenAI SDK example code.

Clones each permissively-licensed source repo at a pinned ref (latest release
tag where the repo tags releases, otherwise HEAD with the SHA recorded), then
extracts Python snippets from example files, notebook code cells, and fenced
markdown blocks. Every snippet carries provenance: repo, ref, path, license.

Outputs:
  data/raw/<name>/            cloned repos (gitignored)
  data/processed/snippets.raw.jsonl
  data/sources.yaml           provenance + per-source snippet counts
  docs/LICENSES.md            license manifest for the dataset

Run:  .venv\\Scripts\\python.exe scripts\\01_harvest.py
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT_JSONL = ROOT / "data" / "processed" / "snippets.raw.jsonl"
SOURCES_YAML = ROOT / "data" / "sources.yaml"
LICENSES_MD = ROOT / "docs" / "LICENSES.md"

SOURCES = [
    {
        "name": "anthropic-cookbook",
        "url": "https://github.com/anthropics/anthropic-cookbook",
        "license": "MIT",
        "pin": "head",
        "sdk": "anthropic",
    },
    {
        "name": "openai-cookbook",
        "url": "https://github.com/openai/openai-cookbook",
        "license": "MIT",
        "pin": "head",
        "sdk": "openai",
        # examples/data contains dirs with trailing spaces — invalid paths on
        # Windows that abort a full checkout. Materialize code/doc files only.
        "code_only_checkout": True,
    },
    {
        "name": "openai-python",
        "url": "https://github.com/openai/openai-python",
        "license": "Apache-2.0",
        "pin": "tag",
        "sdk": "openai",
    },
    {
        "name": "anthropic-sdk-python",
        "url": "https://github.com/anthropics/anthropic-sdk-python",
        "license": "MIT",
        "pin": "tag",
        "sdk": "anthropic",
    },
    {
        "name": "ollama-python",
        "url": "https://github.com/ollama/ollama-python",
        "license": "MIT",
        "pin": "tag",
        "sdk": "ollama",
    },
]

MIN_LINES, MAX_LINES = 5, 400
FENCE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL)
HEADING_RE = re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE)
SECRET_RES = [
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"(?i)(api[_\-]?key|token|secret)\s*=\s*['\"][A-Za-z0-9_\-]{24,}['\"]"),
]
SEMVER_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def run_git(*args, cwd=None):
    res = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.strip()}")
    return res.stdout.strip()


def latest_tag(url):
    out = run_git("ls-remote", "--tags", "--refs", url)
    best, best_key = None, (-1, -1, -1)
    for line in out.splitlines():
        tag = line.split("refs/tags/")[-1]
        m = SEMVER_RE.match(tag)
        if m:
            key = tuple(int(g) for g in m.groups())
            if key > best_key:
                best, best_key = tag, key
    return best


def clone(source):
    dest = RAW / source["name"]
    if dest.exists():
        ref = run_git("rev-parse", "HEAD", cwd=dest)
        tag = run_git("describe", "--tags", "--exact-match", cwd=dest) if source["pin"] == "tag" else None
        return tag or ref
    args = ["clone", "--depth", "1"]
    tag = None
    if source["pin"] == "tag":
        tag = latest_tag(source["url"])
        if tag:
            args += ["--branch", tag]
    if source.get("code_only_checkout"):
        # Clone without checkout, then materialize only the file types we
        # harvest — sidesteps Windows-invalid paths in data directories.
        run_git(*args, "--no-checkout", source["url"], str(dest))
        run_git("checkout", "HEAD", "--", "*.py", "*.md", "*.mdx", "*.ipynb", cwd=dest)
    else:
        run_git(*args, source["url"], str(dest))
    return tag or run_git("rev-parse", "HEAD", cwd=dest)


def has_secret(code):
    return any(r.search(code) for r in SECRET_RES)


def usable(code):
    n = len(code.strip().splitlines())
    return MIN_LINES <= n <= MAX_LINES and not has_secret(code)


def last_heading(text_before):
    headings = HEADING_RE.findall(text_before)
    return headings[-1].strip() if headings else None


def extract_from_py(path):
    code = path.read_text(encoding="utf-8", errors="replace")
    return [(None, code)] if usable(code) else []


def extract_from_md(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    out = []
    for m in FENCE_RE.finditer(text):
        code = m.group(1)
        if usable(code):
            out.append((last_heading(text[: m.start()]), code))
    return out


def extract_from_ipynb(path):
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return []
    out, heading = [], None
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown":
            heading = last_heading(src) or heading
        elif cell.get("cell_type") == "code":
            src = re.sub(r"^[!%].*$", "", src, flags=re.MULTILINE)  # strip shell/magic lines
            if usable(src):
                out.append((heading, src))
    return out


def harvest(source, ref):
    repo_dir = RAW / source["name"]
    records = []
    for path in repo_dir.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        if path.suffix == ".py":
            if source["name"].endswith("cookbook"):
                if "examples" not in path.parts:
                    continue
            elif not {"examples", "tests"} & set(path.parts):
                continue  # SDK repos: examples/ and tests/ only — never library internals (src/)
            snippets = extract_from_py(path)
        elif path.suffix in (".md", ".mdx"):
            snippets = extract_from_md(path)
        elif path.suffix == ".ipynb":
            snippets = extract_from_ipynb(path)
        else:
            continue
        rel = path.relative_to(repo_dir).as_posix()
        for i, (heading, code) in enumerate(snippets):
            records.append({
                "id": f"{source['name']}:{rel}:{i}",
                "source": source["name"],
                "repo": source["url"],
                "ref": ref,
                "path": rel,
                "sdk": source["sdk"],
                "license": source["license"],
                "heading": heading,
                "code": code.strip(),
            })
    return records


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    all_records, seen_hashes, manifest = [], set(), []
    for source in SOURCES:
        print(f"--- {source['name']}")
        ref = clone(source)
        records = harvest(source, ref)
        fresh = []
        for r in records:
            h = hashlib.sha256(re.sub(r"\s+", " ", r["code"]).encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                fresh.append(r)
        all_records.extend(fresh)
        manifest.append({**source, "ref": ref, "snippets": len(fresh)})
        print(f"    ref={ref}  snippets={len(fresh)} (of {len(records)} pre-dedupe)")

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with SOURCES_YAML.open("w", encoding="utf-8") as f:
        f.write("# Dataset provenance — generated by scripts/01_harvest.py\n")
        f.write("sources:\n")
        for m in manifest:
            f.write(f"  - name: {m['name']}\n")
            f.write(f"    url: {m['url']}\n")
            f.write(f"    ref: {m['ref']}\n")
            f.write(f"    sdk: {m['sdk']}\n")
            f.write(f"    license: {m['license']}\n")
            f.write(f"    snippets: {m['snippets']}\n")

    with LICENSES_MD.open("w", encoding="utf-8") as f:
        f.write("# Dataset license manifest\n\n")
        f.write("All training answers are human-written code from these repos, ")
        f.write("harvested at the pinned refs below. No model-generated answers.\n\n")
        f.write("| Source | License | Ref | Snippets |\n|---|---|---|---|\n")
        for m in manifest:
            f.write(f"| [{m['name']}]({m['url']}) | {m['license']} | `{m['ref']}` | {m['snippets']} |\n")

    print(f"\nTOTAL: {len(all_records)} snippets -> {OUT_JSONL.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
