"""Download the resolved setup wheels concurrently; verify every official SHA256."""
import concurrent.futures
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".setup-downloads"


def fetch_range(url, start, end):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=30) as source:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", source.headers.get("Content-Range", ""))
        if source.status != 206 or not match or int(match[1]) != start:
            raise ValueError("Unexpected HTTP range response")
        data = source.read(262144)
        if len(data) != int(match[2]) - start + 1:
            raise ValueError("Incomplete HTTP range")
        return data, int(match[3])


def download(entry):
    info = entry["download_info"]
    url = info["url"]
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org":
        raise ValueError("Only resolved official PyPI HTTPS files are accepted")
    name = Path(urllib.parse.unquote(parsed.path)).name
    if not name.endswith(".whl"):
        raise ValueError(f"Expected a binary wheel: {name}")
    target = CACHE / name
    expected = info["archive_info"]["hashes"]["sha256"]
    if target.exists():
        with target.open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() == expected:
                return name, target.stat().st_size, "cached"
    partial = target.with_suffix(".whl.part")
    if partial.exists():
        with partial.open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() == expected:
                partial.replace(target)
                return name, target.stat().st_size, "resumed and verified"
    failures = 0
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            # Save a contiguous prefix only. Parallel batches never leave holes in partial files.
            data, total = fetch_range(url, offset, offset + 262143)
            with partial.open("ab") as destination:
                destination.write(data)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ranges:
                while partial.stat().st_size < total:
                    base = partial.stat().st_size
                    starts = list(range(base, min(total, base + 4*262144), 262144))
                    pending = [ranges.submit(fetch_range, url, start, min(total-1, start+262143)) for start in starts]
                    blocks = [future.result()[0] for future in pending]
                    with partial.open("ab") as destination:
                        for block in blocks:
                            destination.write(block)
            failures = 0
            if partial.stat().st_size < total:
                continue
            with partial.open("rb") as file:
                if hashlib.file_digest(file, "sha256").hexdigest() != expected:
                    raise ValueError(f"SHA256 mismatch: {name}")
            partial.replace(target)
            return name, target.stat().st_size, "verified"
        except Exception:
            failures += 1
            if failures >= 8:
                raise
            time.sleep(min(failures, 4))


if __name__ == "__main__":
    CACHE.mkdir(exist_ok=True)
    report_path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "evaluation/setup_resolution.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    entries = report["install"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download, entry) for entry in entries]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            name, size, state = future.result()
            print(f"{index}/{len(entries)} {state}: {name} ({size:,} bytes)", flush=True)
    pins = [f"{entry['metadata']['name']}=={entry['metadata']['version']}" for entry in entries]
    target_pins = "compatibility.txt" if "compatibility" in report_path.name else "resolved.txt"
    (CACHE / target_pins).write_text("\n".join(pins) + "\n", encoding="utf-8")
    print("Every setup wheel is downloaded and SHA256 verified.", flush=True)
