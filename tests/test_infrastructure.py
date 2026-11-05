"""Synthetic infrastructure tests. No dataset or model training is performed."""
import copy
import json
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from training.archive import extract_dataset
from training.data import discover, FolderImages, image_transforms
from training.train import Config, atomic_save, build_model, choose_device, evaluate, restore_rng, rng_state, run_epoch, topk_counts, train

torch.set_num_threads(2)


def make_images(root, count=6):
    for label, name in enumerate(["synthetic_a", "synthetic_b"]):
        folder = root / name
        folder.mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (40, 40), (label*100+index, index*10, 50)).save(folder / f"{index}.png")


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_reproducible_splits_and_real_transforms(self):
        make_images(self.root)
        a = discover(self.root)
        b = discover(self.root)
        self.assertEqual(a[2:], b[2:])
        self.assertEqual(sum(len(rows) for rows in a[1].values()), 12)
        hashes = [{row["sha256"] for row in a[1][name]} for name in ["train", "val", "test"]]
        self.assertFalse(hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2])
        image, label = FolderImages(a[1]["train"], image_transforms(64, True))[0]
        self.assertEqual(tuple(image.shape), (3, 64, 64))
        self.assertTrue(torch.isfinite(image).all())

    def test_identical_images_stay_in_same_split(self):
        make_images(self.root)
        original = self.root / "synthetic_a/0.png"
        (self.root / "synthetic_a/copy.png").write_bytes(original.read_bytes())
        _, splits, _, _ = discover(self.root)
        locations = [name for name, rows in splits.items() for row in rows if row["path"].name in {"0.png", "copy.png"} and row["label"] == 0]
        self.assertEqual(len(set(locations)), 1)

    def test_cross_class_duplicates_rejected(self):
        make_images(self.root)
        (self.root / "synthetic_b/copy.png").write_bytes((self.root / "synthetic_a/0.png").read_bytes())
        with self.assertRaisesRegex(ValueError, "Duplicate image"):
            discover(self.root)

    def test_explicit_split_leakage_rejected(self):
        for name in ["train", "val", "test"]:
            make_images(self.root / name)
        with self.assertRaisesRegex(ValueError, "Duplicate image"):
            discover(self.root)

    def test_incomplete_splits_rejected(self):
        make_images(self.root / "train")
        with self.assertRaisesRegex(ValueError, "Pre-split"):
            discover(self.root)

    def test_insufficient_unique_images_rejected(self):
        make_images(self.root, count=2)
        with self.assertRaisesRegex(ValueError, ">=3"):
            discover(self.root)


class ArchiveTests(unittest.TestCase):
    def test_safe_extraction_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("images/item.txt", "synthetic")
            extracted = extract_dataset(archive, root / "out", reserve_bytes=0)
            self.assertEqual((extracted / "images/item.txt").read_text(), "synthetic")
            with self.assertRaises(FileExistsError):
                extract_dataset(archive, root / "out", reserve_bytes=0)

    def test_unsafe_paths_rejected_before_writing(self):
        for name in ["../escape", "/absolute", "C:/escape", "..\\escape"]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                archive = root / "bad.zip"
                with zipfile.ZipFile(archive, "w") as output:
                    output.writestr(name, "bad")
                with self.assertRaises(ValueError):
                    extract_dataset(archive, root / "out", reserve_bytes=0)
                self.assertFalse((root / "out").exists())

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as output:
                info = zipfile.ZipInfo("link")
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                output.writestr(info, "/outside")
            with self.assertRaisesRegex(ValueError, "symbolic"):
                extract_dataset(archive, root / "out", reserve_bytes=0)

    def test_disk_capacity_checked_before_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("image", "test")
            with patch("training.archive.shutil.disk_usage") as usage:
                usage.return_value.free = 0
                with self.assertRaises(OSError):
                    extract_dataset(archive, root / "out", reserve_bytes=0)


class ModelTests(unittest.TestCase):
    def test_real_mobilenet_cpu_inference_without_weights_download(self):
        model = build_model(6, pretrained=False).eval()
        with torch.inference_mode():
            logits = model(torch.zeros(1, 3, 64, 64))
        self.assertEqual(tuple(logits.shape), (1, 6))
        self.assertTrue(torch.isfinite(logits).all())

    def test_top1_top5_and_small_class_count(self):
        logits = torch.tensor([[6., 5., 4., 3., 2., 1.], [6., 5., 4., 3., 2., 1.]])
        self.assertEqual(topk_counts(logits, torch.tensor([0, 5])), (1, 1, 5))
        self.assertEqual(topk_counts(logits[:, :2], torch.tensor([0, 1])), (1, 2, 2))

    def test_validation_forward_does_not_change_weights(self):
        model = nn.Sequential(nn.Flatten(), nn.Linear(12, 2))
        weights = copy.deepcopy(model.state_dict())
        loader = DataLoader([(torch.zeros(3, 2, 2), 0), (torch.ones(3, 2, 2), 1)], batch_size=2)
        metrics, actual, predicted = run_epoch(model, loader, torch.device("cpu"))
        self.assertEqual(len(actual), 2)
        self.assertTrue(0 <= metrics["top1"] <= 1)
        self.assertTrue(all(torch.equal(weights[k], model.state_dict()[k]) for k in weights))

    def test_rng_checkpoint_safe_loading_and_restoration(self):
        generator = torch.Generator().manual_seed(17)
        state = rng_state(generator)
        expected = torch.rand(3), np.random.rand(3), torch.rand(3, generator=generator)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.pt"
            atomic_save(state, path)
            restored = torch.load(path, weights_only=True)
        restore_rng(restored, generator)
        self.assertTrue(torch.equal(expected[0], torch.rand(3)))
        np.testing.assert_array_equal(expected[1], np.random.rand(3))
        self.assertTrue(torch.equal(expected[2], torch.rand(3, generator=generator)))

    def test_evaluation_writes_report_and_confusion_artifacts(self):
        model = nn.Sequential(nn.Flatten(), nn.Linear(12, 2))
        loader = DataLoader([(torch.zeros(3, 2, 2), 0), (torch.ones(3, 2, 2), 1)], batch_size=2)
        with tempfile.TemporaryDirectory() as temp:
            metrics = evaluate(model, loader, torch.device("cpu"), ["synthetic_a", "synthetic_b"], temp)
            root = Path(temp)
            self.assertEqual(metrics["top5_effective_k"], 2)
            report = json.loads((root / "classification_report.json").read_text())
            self.assertEqual(report["synthetic_a"]["support"], 1)
            self.assertTrue((root / "confusion_matrix.png").is_file())
            self.assertEqual(np.loadtxt(root / "confusion_matrix.csv", delimiter=",").shape, (2, 2))

    def test_cuda_request_fails_when_unavailable(self):
        with patch("training.train.torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                choose_device("cuda")


class OrchestrationTests(unittest.TestCase):
    """Exercise phase changes and resume using mocked epochs; no backpropagation."""
    @staticmethod
    def model(*args, **kwargs):
        model = nn.Module()
        model.features = nn.Linear(1, 1)
        model.classifier = nn.Linear(1, 2)
        return model

    @staticmethod
    def epoch(*args, **kwargs):
        return {"loss": 1.0, "top1": 0.0, "top5": 1.0, "top5_effective_k": 2}, [0], [0]

    def test_phase_transition_early_stopping_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_images(root / "data")
            cfg = Config(data_dir=str(root / "data"), output_dir=str(root / "outputs"),
                         evaluation_dir=str(root / "evaluation"), head_epochs=5, finetune_epochs=5,
                         patience=1, pretrained=False)
            with patch("training.train.build_model", side_effect=self.model), \
                 patch("training.train.run_epoch", side_effect=self.epoch), \
                 patch("training.train.evaluate", return_value={"synthetic": True}):
                train(cfg)
                checkpoint = torch.load(root / "outputs/last.pt", weights_only=True)
                self.assertEqual([h["stage"] for h in checkpoint["history"]], ["head", "head", "finetune"])
                self.assertEqual(checkpoint["epoch"], 5)
                (root / "outputs/best.pt").unlink()  # Test recovery from a self-contained last.pt.
                cfg.resume = str(root / "outputs/last.pt")
                train(cfg)
                self.assertTrue((root / "outputs/best.pt").is_file())
                resumed = torch.load(root / "outputs/last.pt", weights_only=True)
                self.assertEqual(resumed["epoch"], 5)  # No extra epochs after patience exhausted.
                cfg.batch_size = 16
                with self.assertRaisesRegex(ValueError, "same training"):
                    train(cfg)


if __name__ == "__main__":
    unittest.main()
