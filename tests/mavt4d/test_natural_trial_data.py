"""Source splitting and truthful timestamps for natural RGB clips."""
import unittest
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile

from mavt4d.natural_trial.prepare import recording_key, split_for_source, select_frame_indices, prepare


class NaturalDataTests(unittest.TestCase):
    def test_segments_of_one_recording_share_partition(self):
        a = recording_key("epic_kitchen/P01/P01_01/video2/rgb")
        b = recording_key("P01/P01_01/video9/rgb/00000.png")
        self.assertEqual(a, "P01/P01_01")
        self.assertEqual(a, b)
        self.assertEqual(split_for_source(a), split_for_source(b))

    def test_selected_timestamps_preserve_original_indices(self):
        indices = [0, 3, 7, 10, 15, 19, 22, 28, 31, 35, 40, 45, 50]
        chosen = select_frame_indices(indices, count=12)
        self.assertEqual(len(chosen), 12)
        self.assertEqual(chosen[0], 0)
        self.assertEqual(chosen[-1], 50)
        self.assertEqual(chosen, sorted(set(chosen)))
        with self.assertRaises(ValueError):
            select_frame_indices(indices[:10], count=12)

    def test_interleaved_tar_export_timestamps_fps_split_and_no_overwrite(self):
        import numpy as np
        from PIL import Image
        all_sources = [f"P01/P01_{i:03d}" for i in range(1, 200)]
        sources = [next(s for s in all_sources if split_for_source(s) == split)
                   for split in ("train", "dev", "val")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "omniworld_epickitchen_0_3.tar.gz"
            metadata = root / "metadata.csv"
            with metadata.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("Index", "Video Path", "FPS"))
                writer.writeheader()
                for i, source in enumerate(sources):
                    writer.writerow({"Index": i, "Video Path": f"epic_kitchen/{source}/video1/rgb", "FPS": "25"})
                writer.writerow({"Index": 3, "Video Path": f"epic_kitchen/{sources[0]}/video2/rgb", "FPS": "X"})
            with tarfile.open(archive, "w:gz") as stream:
                for frame in reversed(range(51)):
                    for source in sources:
                        buffer = io.BytesIO()
                        Image.new("RGB", (6, 4), (frame, 0, 0)).save(buffer, format="PNG")
                        payload = buffer.getvalue()
                        member = tarfile.TarInfo(f"{source}/video1/rgb/{frame:05d}.png")
                        member.size = len(payload)
                        stream.addfile(member, io.BytesIO(payload))
            output = root / "export"
            summary = prepare(metadata, root, output, sources=2, height=4, width=6)
            self.assertEqual(summary["counts"], {"train": 1, "dev": 1})
            self.assertEqual(summary["excluded_metadata_invalid_fps"], 1)
            rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
            self.assertNotIn(sources[2], {row["source"] for row in rows})
            for row in rows:
                canonical = int(hashlib.sha1(("mavt-split-v1\0epic:" + row["source"]).encode()).hexdigest()[:15], 16) / 16**15
                self.assertGreaterEqual(canonical, .02)
                with np.load(output / row["input"], allow_pickle=False) as data:
                    np.testing.assert_allclose(data["frame_times"], np.asarray(row["frame_indices"]) / 25)
                    np.testing.assert_array_equal(data["images"][:, 0, 0, 0], row["frame_indices"])
                    self.assertEqual(data["images"].shape, (12, 3, 4, 6))
            with self.assertRaises(FileExistsError):
                prepare(metadata, root, output, sources=2)


if __name__ == "__main__":
    unittest.main()
