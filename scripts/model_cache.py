"""Best-effort release of a loaded Spark model's clean, unused file cache."""
import ctypes
import os
from pathlib import Path
import platform
import stat
import sys


def cached_pages(fd):
    class Range(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ("offset", "length")]
    class Stats(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in
                    ("cached", "dirty", "writeback", "evicted", "recently_evicted")]
    region, result = Range(0, 0), Stats()
    libc = ctypes.CDLL(None, use_errno=True)
    # Linux aarch64 and x86_64 assign cachestat syscall number 451.
    if libc.syscall(ctypes.c_long(451), ctypes.c_uint(fd), ctypes.byref(region), ctypes.byref(result), ctypes.c_uint(0)):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return {name: int(getattr(result, name)) for name, _ in Stats._fields_}


def model_is_mapped(pid, info):
    for line in (Path("/proc") / str(pid) / "maps").read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) < 5:
            raise ValueError("incomplete process mapping")
        device = os.makedev(*(int(part, 16) for part in fields[3].split(":")))
        if int(fields[4]) == info.st_ino and device == info.st_dev:
            return True
    return False


def release_clean_model_cache(repo, record, owner):
    """Never read model bytes, flush dirty data, follow links, or change files."""
    skipped = {"status": "skipped", "reason": "unsupported or unverifiable"}
    if (not sys.platform.startswith("linux") or platform.machine() not in ("aarch64", "x86_64")
            or not hasattr(os, "posix_fadvise")):
        return skipped
    from cascade.apps.process_owner import is_live
    descriptors = []
    try:
        repo = Path(repo).resolve(strict=True)
        if record.get("role") != "qwen" or record.get("repo") != str(repo) or not is_live(record, owner):
            return skipped
        binding = record["model_binding"]["files"]["model"]
        path = Path(binding["path"])
        relative = path.relative_to(repo)
        if (not relative.parts or relative.parts[0] != "models" or ".." in relative.parts
                or path.suffix.lower() != ".gguf"):
            return skipped
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
        directory = os.open(repo, flags | os.O_DIRECTORY)
        descriptors.append(directory)
        for part in relative.parts[:-1]:
            directory = os.open(part, flags | os.O_DIRECTORY, dir_fd=directory)
            descriptors.append(directory)
        fd = os.open(relative.name, flags | getattr(os, "O_NOATIME", 0), dir_fd=directory)
        descriptors.append(fd)
        info = os.fstat(fd)
        expected = (binding["device"], binding["inode"], binding["size_bytes"], binding["mtime_ns"], binding["ctime_ns"])
        def identity(value):
            return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1
                or identity(info) != expected or model_is_mapped(record["pid"], info)):
            return skipped
        pages = cached_pages(fd)
        if pages["dirty"] or pages["writeback"]:
            return {"status": "skipped", "reason": "dirty or writeback pages"}
        if identity(os.fstat(fd)) != expected or not is_live(record, owner):
            return skipped
        if not pages["cached"]:
            return {"status": "skipped", "reason": "no cached pages"}
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return {"status": "advised", "clean_pages_before": pages["cached"],
                "note": "advisory only; reclaimed bytes are not guaranteed"}
    except (OSError, ValueError, KeyError, TypeError):
        return skipped
    finally:
        for fd in reversed(descriptors):
            os.close(fd)
