#!/usr/bin/env python3
"""Install the pinned Cocina Asier release, or verify it entirely offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
from scene_identity import (
    BUNDLE_MANIFEST, BUNDLE_ROOT, BUNDLE_URL, MANIFEST, MAX_ARCHIVE_BYTES,
    SCENE_NAME, bundle_manifest, bundle_problems, file_receipt, kitchen_manifest,
    kitchen_problems, local_file, relative_name, source_manifest, source_problems,  # noqa: F401 (re-exported)
    validate_bundle_manifest,
)

DOWNLOAD_BUDGET_S = 300
CHUNK_BYTES = 1024 * 1024
TIMEOUT_S = 30
USER_AGENT = "CASCADE-kitchen-installer/1"
GITHUB_API = "https://api.github.com"
GITHUB_HOSTS = frozenset({"github.com", "api.github.com", "release-assets.githubusercontent.com",
                          "objects.githubusercontent.com"})
# A private repository answers anonymous release downloads with 404; a token
# in either variable switches the installer to the authenticated API route.
TOKEN_VARIABLES = ("GH_TOKEN", "GITHUB_TOKEN")
# Any printable, space-free ASCII: GitHub App installation tokens (the Actions
# job token) are ~520 characters in the stateless format.
TOKEN_SHAPE = re.compile(r"[\x21-\x7e]{1,4096}")
RELEASE_METADATA_BYTES = 1024 * 1024
REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})


def github_token() -> str | None:
    """The first non-empty GH_TOKEN / GITHUB_TOKEN, refused unless header-safe.

    The shape check keeps the value out of http.client's "Invalid header value"
    errors (CR/LF), which would otherwise quote it.
    """
    for name in TOKEN_VARIABLES:
        value = os.environ.get(name, "").strip()
        if value:
            if not TOKEN_SHAPE.fullmatch(value):
                raise ValueError(f"{name} is not a well-formed GitHub token")
            return value
    return None


def redact(text: str) -> str:
    """Remove every configured token value from text bound for a log."""
    for name in TOKEN_VARIABLES:
        value = os.environ.get(name, "").strip()
        if value:
            text = text.replace(value, "[redacted]")
    return text


def release_asset(url: str) -> tuple[str, str, str, str]:
    """Owner, repository, tag and asset name of a github.com release download URL."""
    parts = urllib.parse.urlsplit(url)
    segments = parts.path.split("/")
    if (parts.scheme != "https" or parts.hostname != "github.com" or parts.port not in (None, 443)
            or parts.username or parts.query or parts.fragment or len(segments) != 7 or segments[0]
            or segments[3:5] != ["releases", "download"] or not all(segments[1:3] + segments[5:])):
        raise ValueError("Kitchen release URL is not a github.com release download")
    return segments[1], segments[2], segments[5], segments[6]


def require_github_https(url: str) -> None:
    final = urllib.parse.urlsplit(url)
    if (final.scheme != "https" or final.username or final.password or final.port not in (None, 443)
            or final.hostname not in GITHUB_HOSTS):
        raise ValueError("Kitchen release redirected outside GitHub HTTPS")


class ApiRefused(RuntimeError):
    """The GitHub API answered the authenticated release lookup with an HTTP error."""


class _KeepRedirect(urllib.request.HTTPRedirectHandler):
    """Surface a redirect as an HTTPError instead of following it.

    A followed urllib redirect copies the request's headers, so an API request
    never follows one: the storage host must not receive the token.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _api_request(url: str, token: str, accept: str) -> urllib.request.Request:
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Accept": accept, "X-GitHub-Api-Version": "2022-11-28"})
    # Belt and braces: urllib never copies an unredirected header onto a redirect.
    request.add_unredirected_header("Authorization", f"Bearer {token}")
    return request


def _open_api(request: urllib.request.Request):
    return urllib.request.build_opener(_KeepRedirect).open(request, timeout=TIMEOUT_S)


def _open_public(url: str, *, token_tried: bool = False):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        return urllib.request.urlopen(request, timeout=TIMEOUT_S)
    except urllib.error.HTTPError as exc:
        exc.close()
        owner, repo, tag, name = release_asset(url)
        advice = "" if token_tried else (
            f" If {owner}/{repo} is private, set GH_TOKEN (or GITHUB_TOKEN) to a token that can read"
            f" it, or fetch the archive with `gh release download {tag} -R {owner}/{repo} -p {name}`"
            f" and install it with --archive {name}.")
        raise RuntimeError(f"Kitchen release download failed: HTTP {exc.code} from {url}.{advice}") from None


def _open_with_token(url: str, token: str, expected: dict, deadline: float):
    """Resolve the asset through the API; fetch its signed storage URL without the token."""
    owner, repo, tag, name = release_asset(url)
    base = (f"{GITHUB_API}/repos/{urllib.parse.quote(owner, safe='')}/"
            f"{urllib.parse.quote(repo, safe='')}/releases")
    step = f"{base}/tags/{urllib.parse.quote(tag, safe='')}"
    try:
        with _open_api(_api_request(step, token, "application/vnd.github+json")) as response:
            raw = response.read(RELEASE_METADATA_BYTES + 1)
        if len(raw) > RELEASE_METADATA_BYTES:
            raise ValueError("GitHub release metadata exceeded its byte budget")
        release = json.loads(raw)
        assets = release.get("assets") if isinstance(release, dict) else None
        matches = [asset for asset in assets if isinstance(asset, dict) and asset.get("name") == name
                   ] if isinstance(assets, list) else []
        if len(matches) != 1 or type(matches[0].get("id")) is not int or matches[0]["id"] < 1:
            raise ValueError(f"GitHub release {tag} has no single asset named {name}")
        if matches[0].get("size") != expected["size_bytes"]:
            raise ValueError("Kitchen release asset size differs from its pinned size")
        if time.monotonic() >= deadline:
            raise TimeoutError("Kitchen release download exceeded 300 seconds")
        step = f"{base}/assets/{matches[0]['id']}"
        try:
            # 200 means the API streamed the bytes itself (final URL api.github.com).
            return _open_api(_api_request(step, token, "application/octet-stream"))
        except urllib.error.HTTPError as exc:
            location = exc.headers.get("Location") if exc.code in REDIRECT_CODES else None
            if not location:
                raise
            exc.close()
    except urllib.error.HTTPError as exc:
        exc.close()
        raise ApiRefused(f"Kitchen release lookup through the GitHub API failed: HTTP {exc.code} for "
                         f"{step}; check that GH_TOKEN/GITHUB_TOKEN can read {owner}/{repo}.") from None
    target = urllib.parse.urljoin(step, location)
    require_github_https(target)
    # A fresh request: the signed URL is the storage credential; the token stays behind.
    return urllib.request.urlopen(urllib.request.Request(target, headers={"User-Agent": USER_AGENT}),
                                  timeout=TIMEOUT_S)


def download(url: str, output: Path, expected: dict, *, token: str | None = None) -> None:
    """Bound both network time and bytes; never retain an unverified download.

    Without a token this fetches the anonymous public URL. With one, the asset
    is resolved through the GitHub API, which also serves a private repository;
    if the API refuses the token, the anonymous URL is still tried, so a stale
    token cannot break a public repository's install.
    """
    deadline = time.monotonic() + DOWNLOAD_BUDGET_S
    digest, size = hashlib.sha256(), 0
    if token:
        print("[cascade-install] Fetching the Cocina Asier release through the authenticated GitHub API.")
        try:
            opened = _open_with_token(url, token, expected, deadline)
        except ApiRefused as refused:
            try:
                opened = _open_public(url, token_tried=True)
            except RuntimeError as anonymous:
                raise RuntimeError(f"{refused} {anonymous}") from None
    else:
        opened = _open_public(url)
    with opened as response, output.open("xb") as stream:
        require_github_https(response.geturl())
        length = response.headers.get("Content-Length")
        if length is not None and int(length) != expected["size_bytes"]:
            raise ValueError("Kitchen release Content-Length differs from its pinned size")
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Kitchen release download exceeded 300 seconds")
            # read1 makes one socket read, so a slow trickle cannot hold a
            # buffered read open indefinitely between deadline checks.
            chunk = response.read1(min(CHUNK_BYTES, expected["size_bytes"] - size + 1))
            if not chunk:
                break
            size += len(chunk)
            if size > expected["size_bytes"]:
                raise ValueError("Kitchen release exceeded its pinned byte budget")
            digest.update(chunk)
            stream.write(chunk)
    if size != expected["size_bytes"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError("Kitchen release archive checksum or size mismatch")


def unpack(archive_path: Path, destination: Path, expected: dict) -> None:
    """Write only exact inventoried regular files, without ZipFile.extract."""
    deadline = time.monotonic() + DOWNLOAD_BUDGET_S
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("Kitchen release must contain exactly its pinned member inventory")
        for member in members:
            name = member.filename
            relative_name(name)
            mode = member.external_attr >> 16
            if (member.is_dir() or member.flag_bits & 1
                    or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                    or member.file_size != expected[name]["size_bytes"]):
                raise ValueError(f"Kitchen release member is not its pinned regular file: {name}")
        for member in members:
            target = destination / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            digest, size = hashlib.sha256(), 0
            with archive.open(member) as incoming, target.open("xb") as outgoing:
                while chunk := incoming.read(CHUNK_BYTES):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Kitchen release extraction exceeded 300 seconds")
                    size += len(chunk)
                    if size > expected[member.filename]["size_bytes"]:
                        raise ValueError("Kitchen release member exceeded its pinned byte budget")
                    digest.update(chunk)
                    outgoing.write(chunk)
            if size != member.file_size or digest.hexdigest() != expected[member.filename]["sha256"]:
                raise ValueError(f"Kitchen release member checksum mismatch: {member.filename}")
            target.chmod(0o644)


def prepare_kitchen(repo: Path, *, check: bool = False, archive: Path | None = None) -> None:
    """Validate sources first; install atomically only after archive and file checks."""
    repo = repo.resolve()
    problems = source_problems(repo)
    if problems:
        raise RuntimeError("Kitchen source verification failed: " + "; ".join(problems)
                           + ". Restore the matching source checkout before launching.")
    data = bundle_manifest(repo)
    problems = bundle_problems(repo)
    if not problems:
        print(f"[cascade-install] Cocina Asier: {len(kitchen_manifest(repo))} sources and "
              f"{len(data['files'])} release files verified.")
        return
    if check:
        raise RuntimeError("Kitchen release verification failed: " + "; ".join(problems)
                           + ". Run python3 scripts/kitchen_assets.py to install the pinned release.")
    destination = repo / BUNDLE_ROOT
    for path in (destination, *destination.parents):
        if path.is_symlink():
            raise ValueError("Kitchen installation destination must not contain symlinks")
        if path == repo:
            break
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The exclusive lock prevents simultaneous installers from replacing one
    # another's verified tree. A killed install leaves a clear, removable lock.
    lock = destination.with_name(".cocina-asier-install.lock")
    with lock.open("x"):
        try:
            with tempfile.TemporaryDirectory(prefix=".cocina-asier-", dir=destination.parent) as temporary:
                temporary = Path(temporary)
                if archive is None:
                    archive = temporary / "release.zip"
                    download(data["archive"]["url"], archive, data["archive"], token=github_token())
                elif archive.is_symlink() or not archive.is_file():
                    raise ValueError("Supplied kitchen archive must be a regular local file")
                if (archive.stat().st_size != data["archive"]["size_bytes"]
                        or file_receipt(archive) != {key: data["archive"][key]
                                                   for key in ("sha256", "size_bytes")}):
                    raise ValueError("Kitchen release archive checksum or size mismatch")
                staged = temporary / "verified"
                staged.mkdir()
                unpack(archive, staged, data["files"])
                # Preserve the old scene on every failure before replacement.
                backup = temporary / "previous"
                if destination.exists():
                    destination.rename(backup)
                try:
                    staged.rename(destination)
                except BaseException:
                    if backup.exists():
                        backup.rename(destination)
                    raise
        finally:
            lock.unlink()
    prepare_kitchen(repo, check=True)


def package_bundle(repo: Path, archive: Path) -> None:
    """Maintainer operation: deterministic ZIP and complete release byte receipt."""
    root = repo / BUNDLE_ROOT
    if archive.exists() or archive.is_symlink():
        raise ValueError("Choose a new archive path; existing release artifacts are immutable")
    if archive.resolve().is_relative_to(root.resolve()):
        raise ValueError("Release archive must be outside the kitchen asset directory")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Kitchen release members must not be symlinks")
        if path.is_dir():
            continue
        name = path.relative_to(root).as_posix()
        files[name] = file_receipt(local_file(repo, f"{BUNDLE_ROOT}/{name}"))
    archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
            for name in sorted(files):
                member = zipfile.ZipInfo(name, date_time=(2026, 9, 30, 0, 0, 0))
                member.create_system = 3
                member.external_attr = (stat.S_IFREG | 0o644) << 16
                member.compress_type = zipfile.ZIP_DEFLATED
                with output.open(member, "w") as outgoing, (root / name).open("rb") as incoming:
                    shutil.copyfileobj(incoming, outgoing, CHUNK_BYTES)
        if archive.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Kitchen archive exceeds its byte budget")
        data = {"schema": 1, "scene_name": SCENE_NAME, "root": BUNDLE_ROOT,
                "archive": {"url": BUNDLE_URL, **file_receipt(archive)}, "files": files}
        validate_bundle_manifest(data)
        # Recheck source files so a concurrent artwork rebuild cannot create a
        # receipt for different bytes from those actually placed in the archive.
        with tempfile.TemporaryDirectory(prefix=".cocina-package-", dir=archive.parent) as temp:
            unpack(archive, Path(temp), files)
        if bundle_problems(repo, manifest=data):
            raise ValueError("Kitchen artwork changed while packaging; rebuild the candidate archive")
        # Build both prospective receipts before publishing either one. Invalid
        # members, changed artwork, or missing sources leave the old bytes intact.
        bundle_bytes = (json.dumps(data, indent=2) + "\n").encode()
        sources = source_manifest(repo)
        sources["files"][BUNDLE_MANIFEST] = {
            "sha256": hashlib.sha256(bundle_bytes).hexdigest(), "size_bytes": len(bundle_bytes)}
        source_bytes = (json.dumps(sources, indent=2) + "\n").encode()
    except BaseException:
        archive.unlink(missing_ok=True)
        raise
    # The archive is valid and retained if a later filesystem write fails.
    # Each receipt replacement is atomic; no reader sees partially written JSON.
    with tempfile.TemporaryDirectory(prefix=".cocina-receipts-", dir=(repo / MANIFEST).parent) as temp:
        new_bundle, new_sources = Path(temp) / "bundle.json", Path(temp) / "sources.json"
        new_bundle.write_bytes(bundle_bytes)
        new_sources.write_bytes(source_bytes)
        new_bundle.replace(repo / BUNDLE_MANIFEST)
        new_sources.replace(repo / MANIFEST)
    print(f"[cascade-install] Prepared release archive {archive.name}: {data['archive']['sha256']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true", help="Offline, read-only verification; never download")
    parser.add_argument("--archive", type=Path, help="Install an already downloaded pinned archive offline")
    parser.add_argument("--write-manifest", action="store_true",
                        help="Maintainer operation after reviewing intentional source changes")
    parser.add_argument("--package-bundle", type=Path,
                        help="Maintainer operation: package installed artwork and update both receipts")
    args = parser.parse_args(argv)
    if args.check and (args.write_manifest or args.archive or args.package_bundle):
        parser.error("--check is read-only and cannot be combined with maintainer/install options")
    if args.package_bundle and (args.archive or args.write_manifest):
        parser.error("--package-bundle cannot be combined with --archive or --write-manifest")
    try:
        if args.package_bundle:
            package_bundle(args.repo, args.package_bundle)
        elif args.write_manifest:
            (args.repo / MANIFEST).write_text(json.dumps(source_manifest(args.repo), indent=2) + "\n")
        prepare_kitchen(args.repo, check=args.check or args.write_manifest or bool(args.package_bundle),
                        archive=args.archive)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        # The token is never formatted into a message; redact anyway, in case a
        # library error ever quotes a request header.
        print(redact(str(exc)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
