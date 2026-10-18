"""Offline integration checks for the public manifest downloader."""
import hashlib
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.request
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/data/download_manifest.py"
spec = importlib.util.spec_from_file_location("download_manifest", SCRIPT)
dm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dm)

PAYLOAD = b"public-dataset-shard\n" * 1000


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        start = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
        if self.path == "/ignore":
            start = 0
        ranged = bool(self.headers.get("Range")) and self.path != "/ignore"
        self.send_response(206 if ranged else 200)
        self.send_header("Content-Length", str(len(PAYLOAD) - start))
        if ranged:
            actual_start = start + 1 if self.path == "/wrong" else start
            self.send_header("Content-Range", f"bytes {actual_start}-{len(PAYLOAD)-1}/{len(PAYLOAD)}")
        self.end_headers()
        if self.path == "/short" and not self.headers.get("Range"):
            self.wfile.write(PAYLOAD[:101])
            self.close_connection = True
            return
        self.wfile.write(PAYLOAD[start:])


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "data"
        self.root.mkdir()

    def manifest(self, endpoint="/file", **overrides):
        entry = {"path": "source/shard.bin", "url": f"http://127.0.0.1:{self.server.server_port}{endpoint}",
                 "size": len(PAYLOAD), "sha256": hashlib.sha256(PAYLOAD).hexdigest(), "source": "test", "group": "g"}
        entry.update(overrides)
        return {"version": 1, "max_bytes": len(PAYLOAD), "files": [entry]}

    def run_download(self, manifest):
        return dm.run(manifest, self.root, workers=2, reserve_bytes=0, retry_delay=0)

    def test_full_download_and_verified_skip(self):
        self.run_download(self.manifest())
        final = self.root / "source/shard.bin"
        self.assertEqual(final.read_bytes(), PAYLOAD)
        with mock.patch.object(dm, "open_url", side_effect=AssertionError("must skip verified existing file")):
            self.run_download(self.manifest())
        status = json.loads((self.root / ".download-state/status.json").read_text())
        self.assertEqual(status["complete_bytes"], len(PAYLOAD))
        self.assertEqual(status["downloaded_bytes"], 0)

    def test_resume_valid_range(self):
        part = self.root / "source/shard.bin.part"
        part.parent.mkdir()
        part.write_bytes(PAYLOAD[:101])
        self.run_download(self.manifest())
        self.assertEqual(part.with_suffix("").read_bytes(), PAYLOAD)

    def test_ignored_range_preserves_partial(self):
        part = self.root / "source/shard.bin.part"
        part.parent.mkdir()
        part.write_bytes(PAYLOAD[:101])
        with self.assertRaises(dm.DownloadError):
            self.run_download(self.manifest("/ignore"))
        self.assertEqual(part.read_bytes(), PAYLOAD[:101])

    def test_wrong_range_preserves_partial(self):
        part = self.root / "source/shard.bin.part"
        part.parent.mkdir()
        part.write_bytes(PAYLOAD[:101])
        with self.assertRaises(dm.DownloadError):
            self.run_download(self.manifest("/wrong"))
        self.assertEqual(part.read_bytes(), PAYLOAD[:101])

    def test_bad_existing_final_is_not_overwritten(self):
        final = self.root / "source/shard.bin"
        final.parent.mkdir()
        final.write_bytes(b"other user's data")
        with self.assertRaises(dm.DownloadError):
            self.run_download(self.manifest())
        self.assertEqual(final.read_bytes(), b"other user's data")

    def test_validation_before_payload_writes(self):
        cases = [self.manifest(path="../escape"), self.manifest(path="/absolute"),
                 self.manifest(url="http://example.com/data"), self.manifest(path=".download-state/status.json")]
        over_cap = self.manifest()
        over_cap["max_bytes"] = 1
        cases.append(over_cap)
        collision = self.manifest()
        collision["files"].append(dict(collision["files"][0]))
        collision["max_bytes"] *= 2
        cases.append(collision)
        for manifest in cases:
            with self.subTest(manifest=manifest), self.assertRaises(dm.DownloadError):
                self.run_download(manifest)
        self.assertFalse((self.root / "source").exists())

    def test_symlink_escape_rejected(self):
        (self.root / "source").symlink_to(Path(self.temp.name), target_is_directory=True)
        with self.assertRaises(dm.DownloadError):
            self.run_download(self.manifest())
        self.assertFalse((Path(self.temp.name) / "shard.bin").exists())

    def test_low_free_space_stops_before_download(self):
        with mock.patch.object(dm.shutil, "disk_usage", return_value=mock.Mock(free=0)):
            with self.assertRaises(dm.DownloadError):
                dm.run(self.manifest(), self.root, reserve_bytes=100, retry_delay=0)
        self.assertFalse((self.root / "source/shard.bin").exists())

    def test_early_eof_retries_from_saved_prefix(self):
        self.run_download(self.manifest("/short"))
        self.assertEqual((self.root / "source/shard.bin").read_bytes(), PAYLOAD)

    def test_md5_and_sha512_verified_together(self):
        manifest = self.manifest(md5=hashlib.md5(PAYLOAD).hexdigest(), sha512=hashlib.sha512(PAYLOAD).hexdigest())
        self.run_download(manifest)
        self.assertEqual((self.root / "source/shard.bin").read_bytes(), PAYLOAD)

    def test_bad_hash_preserves_partial_and_never_publishes(self):
        with self.assertRaises(dm.DownloadError):
            self.run_download(self.manifest(sha256="0" * 64))
        self.assertFalse((self.root / "source/shard.bin").exists())
        self.assertEqual((self.root / "source/shard.bin.part").read_bytes(), PAYLOAD)

    def test_duplicate_process_lock_rejected(self):
        with dm.root_lock(self.root), self.assertRaises(dm.DownloadError):
            self.run_download(self.manifest())

    def test_changed_manifest_requires_fresh_root(self):
        self.run_download(self.manifest())
        with self.assertRaises(dm.DownloadError):
            self.run_download(self.manifest("/changed"))

    def test_wrong_etag_preserves_empty_destination(self):
        with self.assertRaises(dm.DownloadError):
            self.run_download(self.manifest(etag='"expected"'))
        self.assertFalse((self.root / "source/shard.bin").exists())

    def test_parent_file_and_partial_collisions_rejected(self):
        for path in ("source/shard.bin/child", "source/shard.bin.part"):
            manifest = self.manifest()
            manifest["files"].append(dict(manifest["files"][0], path=path))
            manifest["max_bytes"] *= 2
            with self.subTest(path=path), self.assertRaises(dm.DownloadError):
                self.run_download(manifest)

    def test_plain_http_redirect_rejected(self):
        with self.assertRaises(dm.DownloadError):
            dm.validate_url("http://remote.example/shard")
        with self.assertRaises(dm.DownloadError):
            dm.validate_url("https://name:secret@remote.example/shard")

    def test_missing_integrity_is_rejected(self):
        manifest = self.manifest()
        del manifest["files"][0]["sha256"]
        with self.assertRaises(dm.DownloadError):
            self.run_download(manifest)

    def test_multipart_s3_etag_checks_payload(self):
        part_size = 2048
        parts = [hashlib.md5(PAYLOAD[start:start + part_size]).digest()
                 for start in range(0, len(PAYLOAD), part_size)]
        etag = '"' + hashlib.md5(b"".join(parts)).hexdigest() + f'-{len(parts)}"'
        entry = self.manifest(etag=etag, etag_md5_part_size=part_size)["files"][0]
        del entry["sha256"]
        path = self.root / "part.bin"
        path.write_bytes(PAYLOAD)
        dm.verify_file(path, entry)
        path.write_bytes(b"X" + PAYLOAD[1:])
        with self.assertRaises(dm.DownloadError):
            dm.verify_file(path, entry)

    def test_single_part_etag_is_plain_md5(self):
        entry = self.manifest(etag='"' + hashlib.md5(PAYLOAD).hexdigest() + '"',
                              etag_md5_part_size=2048)["files"][0]
        del entry["sha256"]
        path = self.root / "part.bin"
        path.write_bytes(PAYLOAD)
        dm.verify_file(path, entry)

    def test_token_only_exact_hf_host_and_never_redirected(self):
        token = "hf_test_token_not_a_credential"
        with mock.patch.object(dm.urllib.request, "build_opener") as factory:
            for url, authorized in (("https://huggingface.co/repo/file", True),
                                    ("https://cdn-lfs.huggingface.co/file", False),
                                    ("https://huggingface.co.example/file", False),
                                    ("https://example.org/file", False)):
                dm.open_url(url, {}, hf_token=token)
                request = factory.return_value.open.call_args.args[0]
                self.assertEqual(request.get_header("Authorization"), "Bearer " + token if authorized else None)
                self.assertNotIn("Authorization", request.headers)
                redirected = dm.PublicRedirect().redirect_request(
                    request, None, 302, "Found", {}, "https://cdn.example/file")
                self.assertIsNone(redirected.get_header("Authorization"))


if __name__ == "__main__":
    unittest.main()
