"""Disk-aware ZIP extraction restricted to a new destination directory."""
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath


def extract_dataset(archive, destination, reserve_bytes=1024**3):
    archive, destination = Path(archive).resolve(), Path(destination).resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Destination must be empty to avoid mixing datasets: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        size = 0
        targets = set()
        for entry in bundle.infolist():
            path = PurePosixPath(entry.filename.replace("\\", "/"))
            target = (destination / str(path)).resolve()
            if path.is_absolute() or ".." in path.parts or any(":" in part for part in path.parts) or not target.is_relative_to(destination):
                raise ValueError(f"Unsafe ZIP path: {entry.filename}")
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("ZIP symbolic links are not accepted")
            normalized = str(target).casefold()
            if normalized in targets:
                raise ValueError(f"Duplicate ZIP destination: {entry.filename}")
            targets.add(normalized)
            size += entry.file_size
        free = shutil.disk_usage(destination.parent).free
        if size + reserve_bytes > free:
            raise OSError(f"Need {size + reserve_bytes:,} free bytes including reserve; have {free:,}.")
        destination.mkdir(parents=True, exist_ok=True)
        # Write normalized, validated paths rather than trusting platform-specific ZIP handling.
        for entry in bundle.infolist():
            target = destination / entry.filename.replace("\\", "/")
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(entry) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024*1024)
    return destination
