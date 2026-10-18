#!/usr/bin/env python3
"""Resume bounded HTTP datasets without packages, extraction, or credential caches.

The manifest is version 1 with max_bytes and files containing path, url, size,
source, group, and a sha256/md5/sha512 digest or explicitly verified S3 ETag
algorithm (etag_md5_part_size). The cap applies to the total
final payload (archives stay compressed). State and an exclusive process lock
live under ROOT/.download-state. Existing final files are never replaced.
Optional --hf-token-stdin keeps a user-supplied HF token in memory and sends it
only to the initial HTTPS huggingface.co request, never on redirects.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import errno
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import sys
import threading
import time
from typing import Any, Iterator
import urllib.error
import urllib.parse
import urllib.request


class DownloadError(RuntimeError):
    """A safe, actionable error containing no remote URLs or credentials."""


def validate_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.username or parsed.password or not parsed.hostname:
        raise DownloadError("URL must be public and must not contain credentials")
    if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
        raise DownloadError("Only HTTPS sources or localhost test HTTP are allowed")


class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int,
                         msg: str, headers: Any, newurl: str) -> urllib.request.Request | None:
        validate_url(newurl)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("Authorization")
            redirected.remove_header("Proxy-Authorization")
        return redirected


def open_url(url: str, headers: dict[str, str], hf_token: str | None = None) -> Any:
    validate_url(url)
    # No netrc, cookies, cached/environment tokens, or authorization handler.
    public_headers = {key: value for key, value in headers.items()
                      if key.lower() not in {"authorization", "proxy-authorization"}}
    request = urllib.request.Request(url, headers=public_headers)
    parsed = urllib.parse.urlsplit(url)
    if hf_token and parsed.scheme == "https" and parsed.hostname == "huggingface.co" and parsed.port in (None, 443):
        request.add_unredirected_header("Authorization", "Bearer " + hf_token)
    return urllib.request.build_opener(PublicRedirect()).open(request, timeout=60)


def safe_path(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if (not relative or value.is_absolute() or ".." in value.parts
            or str(value) != relative or "\\" in relative
            or value.parts[0] == ".download-state"):
        raise DownloadError("Unsafe or reserved manifest path")
    path = root / relative
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise DownloadError(f"Symlink in destination path: {relative}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise DownloadError("Destination escapes root")
    return path


def validate_manifest(manifest: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    if manifest.get("version") != 1 or type(manifest.get("max_bytes")) is not int:
        raise DownloadError("Expected manifest version 1 and integer max_bytes")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise DownloadError("Manifest must have a nonempty files array")
    paths = set()
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise DownloadError("Each entry requires a string path")
        path = entry["path"]
        safe_path(root, path)
        safe_path(root, path + ".part")
        if type(entry.get("size")) is not int or entry["size"] < 0:
            raise DownloadError(f"Invalid size: {path}")
        if not isinstance(entry.get("url"), str):
            raise DownloadError(f"Missing URL: {path}")
        validate_url(entry["url"])
        for kind, length in (("sha256", 64), ("md5", 32), ("sha512", 128)):
            if kind in entry and (not isinstance(entry[kind], str) or not re.fullmatch(f"[0-9a-fA-F]{{{length}}}", entry[kind])):
                raise DownloadError(f"Invalid {kind}: {path}")
        if "etag" in entry and (not isinstance(entry["etag"], str) or "\n" in entry["etag"] or "\r" in entry["etag"]):
            raise DownloadError(f"Invalid etag: {path}")
        if "etag_md5_part_size" in entry:
            part_size = entry["etag_md5_part_size"]
            if (type(part_size) is not int or not 0 < part_size <= 1024 ** 3
                    or not re.fullmatch(r'"?[0-9a-fA-F]{32}(?:-[1-9][0-9]*)?"?', entry.get("etag", ""))):
                raise DownloadError(f"Invalid explicit S3 ETag algorithm: {path}")
        if not any(name in entry for name in ("sha256", "md5", "sha512", "etag_md5_part_size")):
            raise DownloadError(f"Digest or verified S3 ETag algorithm required: {path}")
        for name in (path, path + ".part"):
            if name in paths:
                raise DownloadError(f"Manifest destination collision: {path}")
            paths.add(name)
    for path in paths:
        if any(str(parent) in paths for parent in PurePosixPath(path).parents):
            raise DownloadError(f"Manifest directory/file collision: {path}")
    if sum(entry["size"] for entry in files) > manifest["max_bytes"]:
        raise DownloadError("Manifest payload exceeds max_bytes cap")
    return files


def verify_file(path: Path, entry: dict[str, Any]) -> None:
    if not path.is_file() or path.stat().st_size != entry["size"]:
        raise DownloadError(f"Existing file size mismatch, preserved: {entry['path']}")
    hashes = {name: hashlib.new(name) for name in ("sha256", "md5", "sha512") if name in entry}
    part_size = entry.get("etag_md5_part_size")
    if not hashes and not part_size:
        raise DownloadError(f"Missing integrity algorithm: {entry['path']}")
    expected_etag = entry.get("etag", "").strip('"').lower()
    multipart = bool(part_size and "-" in expected_etag)
    if part_size and not multipart:
        hashes.setdefault("md5", hashlib.md5())
    part_digest, combined_digest = hashlib.md5(), hashlib.md5()
    part_bytes, part_count = 0, 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(min(8 * 1024 * 1024, part_size - part_bytes) if multipart else 8 * 1024 * 1024)
            if not chunk:
                break
            for digest in hashes.values():
                digest.update(chunk)
            if multipart:
                part_digest.update(chunk)
                part_bytes += len(chunk)
                if part_bytes == part_size:
                    combined_digest.update(part_digest.digest())
                    part_digest = hashlib.md5()
                    part_bytes = 0
                    part_count += 1
    if multipart and part_bytes:
        combined_digest.update(part_digest.digest())
        part_count += 1
    for name, digest in hashes.items():
        if name in entry and digest.hexdigest() != entry[name].lower():
            raise DownloadError(f"{name} mismatch, file preserved: {entry['path']}")
    if part_size:
        actual = f"{combined_digest.hexdigest()}-{part_count}" if multipart else hashes["md5"].hexdigest()
        if actual != expected_etag:
            raise DownloadError(f"S3 ETag checksum mismatch, file preserved: {entry['path']}")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def root_lock(root: Path) -> Iterator[Path]:
    state = root / ".download-state"
    if state.is_symlink():
        raise DownloadError("State directory must not be a symlink")
    state.mkdir(exist_ok=True)
    for name in ("lock", "status.json", "status.tmp", "identity.json", "identity.tmp"):
        if (state / name).is_symlink():
            raise DownloadError("State files must not be symlinks")
    with (state / "lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DownloadError("Another downloader owns this root") from None
        try:
            yield state
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class Progress:
    def __init__(self, root: Path, state: Path, files: list[dict[str, Any]], reserve_bytes: int,
                 hf_token: str | None = None) -> None:
        self.root, self.state = root, state
        self.reserve_bytes = reserve_bytes
        self.hf_token = hf_token
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.last_write = 0
        self.data = {"pid": os.getpid(), "status": "running", "started_at": time.time(),
                     "total_bytes": sum(f["size"] for f in files), "downloaded_bytes": 0,
                     "complete_bytes": 0, "complete_files": 0, "total_files": len(files), "files": {}}

    def check_space(self, upcoming: int = 0) -> None:
        if self.stop.is_set():
            raise DownloadError("Stopped after another download failed")
        if shutil.disk_usage(self.root).free < self.reserve_bytes + upcoming:
            self.stop.set()
            raise DownloadError("Free space below reserve; partials preserved")

    def update(self, path: str | None = None, status: str | None = None, bytes_: int = 0,
               transferred: int = 0, force: bool = False) -> None:
        with self.lock:
            self.data["downloaded_bytes"] += transferred
            if path:
                old = self.data["files"].get(path, {})
                if status == "complete" and old.get("status") != "complete":
                    self.data["complete_bytes"] += bytes_
                    self.data["complete_files"] += 1
                self.data["files"][path] = {"status": status, "bytes": bytes_}
            now = time.time()
            if force or now - self.last_write >= 10:
                self.data["updated_at"] = now
                self.data["partial_bytes"] = sum(f["bytes"] for f in self.data["files"].values() if f["status"] != "complete")
                atomic_json(self.state / "status.json", self.data)
                print(json.dumps({"event": status or "progress", "path": path, "bytes": bytes_,
                                  "complete_bytes": self.data["complete_bytes"],
                                  "downloaded_bytes": self.data["downloaded_bytes"], "time": now}), flush=True)
                self.last_write = now


def download_one(entry: dict[str, Any], progress: Progress, retry_delay: float) -> None:
    root = progress.root
    final = safe_path(root, entry["path"])
    part = safe_path(root, entry["path"] + ".part")
    progress.check_space()
    if final.exists():
        verify_file(final, entry)
        progress.update(entry["path"], "complete", entry["size"], force=True)
        return
    final.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        progress.check_space()
        offset = part.stat().st_size if part.exists() else 0
        if offset > entry["size"]:
            raise DownloadError(f"Oversized partial preserved: {entry['path']}")
        progress.update(entry["path"], "downloading", offset, force=True)
        if offset == entry["size"]:
            if not part.exists():
                part.touch(exist_ok=False)
            break
        headers = {"User-Agent": "PublicManifestDownloader/1", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            if entry.get("etag") and not entry["etag"].startswith("W/"):
                headers["If-Range"] = entry["etag"]
        try:
            with open_url(entry["url"], headers, hf_token=progress.hf_token) as response:
                if offset and response.status != 206:
                    raise DownloadError(f"Server ignored resume Range; partial preserved: {entry['path']}")
                if response.status not in (200, 206):
                    raise DownloadError(f"Unexpected HTTP status {response.status}: {entry['path']}")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise DownloadError(f"Unexpected encoded response: {entry['path']}")
                expected = entry["size"] - offset
                if response.status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or tuple(map(int, match.groups())) != (offset, entry["size"] - 1, entry["size"]):
                        raise DownloadError(f"Invalid Content-Range; partial preserved: {entry['path']}")
                length = response.headers.get("Content-Length")
                if length is not None and (not length.isdigit() or int(length) != expected):
                    raise DownloadError(f"Response length mismatch: {entry['path']}")
                if entry.get("etag") and response.headers.get("ETag") != entry["etag"]:
                    raise DownloadError(f"ETag mismatch: {entry['path']}")
                with part.open("ab") as handle:
                    while offset < entry["size"]:
                        chunk = response.read(min(1024 * 1024, entry["size"] - offset))
                        if not chunk:
                            break
                        progress.check_space(len(chunk))
                        handle.write(chunk)
                        handle.flush()
                        offset += len(chunk)
                        progress.update(entry["path"], "downloading", offset, transferred=len(chunk))
                    if offset == entry["size"] and response.read(1):
                        raise DownloadError(f"Response exceeds expected size: {entry['path']}")
                    os.fsync(handle.fileno())
            if offset == entry["size"]:
                break
        except urllib.error.HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504):
                raise DownloadError(f"HTTP {exc.code}: {entry['path']}") from None
        except (urllib.error.URLError, socket.timeout, ConnectionError, http.client.HTTPException):
            pass  # Never print exception URLs, redirect queries, or headers.
        except OSError as exc:
            if exc.errno in (errno.ENOSPC, errno.EDQUOT):
                progress.stop.set()
                raise DownloadError("Storage full or quota exceeded; partials preserved") from None
            raise DownloadError(f"Local I/O error {exc.errno}: {entry['path']}") from None
        if attempt == 4:
            raise DownloadError(f"Five attempts exhausted; partial preserved: {entry['path']}")
        if progress.stop.wait(retry_delay * 2 ** attempt):
            raise DownloadError("Stopped after another download failed")
    verify_file(part, entry)
    # link is atomic and refuses a concurrently created final; unlike replace,
    # it cannot overwrite another file. Unlink only our completed .part alias.
    try:
        os.link(part, final)
    except FileExistsError:
        raise DownloadError(f"Final appeared during download; both preserved: {entry['path']}") from None
    part.unlink()
    progress.update(entry["path"], "complete", entry["size"], force=True)


def run(manifest: dict[str, Any], root: Path | str, workers: int = 2,
        reserve_bytes: int = 20_000_000_000, retry_delay: float = 2,
        hf_token: str | None = None) -> dict[str, Any]:
    if workers not in (1, 2) or reserve_bytes < 0:
        raise DownloadError("workers must be 1 or 2 and reserve_bytes nonnegative")
    root = Path(root).absolute()
    if root.is_symlink():
        raise DownloadError("Root must not be a symlink")
    files = validate_manifest(manifest, root)
    root.mkdir(parents=True, exist_ok=True)
    with root_lock(root) as state:
        identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        identity_path = state / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text())["sha256"] != identity:
            raise DownloadError("Root belongs to a different manifest; choose a fresh root")
        if not identity_path.exists():
            atomic_json(identity_path, {"sha256": identity})
        progress = Progress(root, state, files, reserve_bytes, hf_token=hf_token)
        progress.update(force=True)
        heartbeat_done = threading.Event()
        heartbeat_errors = []

        def heartbeat() -> None:
            while not heartbeat_done.wait(10):
                try:
                    progress.update(force=True)
                except Exception as exc:
                    heartbeat_errors.append(exc)
                    progress.stop.set()
                    return

        reporter = threading.Thread(target=heartbeat, name="download-status", daemon=True)
        reporter.start()
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(download_one, entry, progress, retry_delay) for entry in files]
                try:
                    for future in as_completed(futures):
                        future.result()
                except BaseException:
                    progress.stop.set()
                    for other in futures:
                        other.cancel()
                    raise
            if heartbeat_errors:
                raise DownloadError("Status persistence failed; downloads stopped")
        except BaseException as exc:
            heartbeat_done.set()
            reporter.join()
            progress.data["status"] = "failed"
            progress.data["error"] = str(exc) if isinstance(exc, DownloadError) else type(exc).__name__
            progress.update(force=True)
            raise
        finally:
            heartbeat_done.set()
            reporter.join()
        progress.data["status"] = "complete"
        progress.update(force=True)
        return progress.data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    parser.add_argument("--reserve-bytes", type=int, default=20_000_000_000)
    parser.add_argument("--hf-token-stdin", action="store_true",
                        help="Read one user-supplied HF token from stdin; memory only, exact HF host only")
    args = parser.parse_args()
    try:
        hf_token = None
        if args.hf_token_stdin:
            hf_token = sys.stdin.readline(8193).strip()
            if not hf_token or len(hf_token) > 8192 or any(char.isspace() for char in hf_token):
                raise DownloadError("Expected one nonempty HF token line on stdin")
        run(json.loads(args.manifest.read_text()), args.root, args.workers, args.reserve_bytes,
            hf_token=hf_token)
    except Exception as exc:
        message = str(exc) if isinstance(exc, DownloadError) else type(exc).__name__
        print(json.dumps({"event": "failed", "error": message}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
