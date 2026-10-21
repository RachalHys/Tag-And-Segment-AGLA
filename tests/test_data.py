import json
from pathlib import Path
import tempfile
import unittest

from tas_runtime.artifacts import ArtifactError, atomic_json, atomic_jsonl
from tas_runtime.data import (
    build_coco_manifests,
    load_external_jsonl,
    pope_image_ids,
    prepare_coco_data,
    read_jsonl,
    verify_manifest_images,
)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.annotations = {}
        for split, ids in (("train2017", range(1, 11)), ("val2017", range(101, 121))):
            images = [
                {
                    "id": identifier,
                    "file_name": f"{identifier:012d}.jpg",
                    "width": 8,
                    "height": 6,
                }
                for identifier in ids
            ]
            for kind in ("captions", "instances"):
                data = {"images": images, "annotations": []}
                if kind == "captions":
                    data["annotations"] = [
                        {"id": i, "image_id": i, "caption": "A cat."} for i in ids
                    ]
                else:
                    data["categories"] = [{"id": 1, "name": "cat"}]
                    data["annotations"] = [
                        {
                            "id": i,
                            "image_id": i,
                            "category_id": 1,
                            "bbox": [1, 1, 3, 2],
                            "area": 6,
                            "iscrowd": 0,
                        }
                        for i in ids
                    ]
                path = self.root / "sources" / f"{kind}_{split}.json"
                atomic_json(path, data)
                self.annotations[f"{kind}_{split}"] = path
        self.sizes = {"A": 2, "B": 3, "C": 4, "D": 3}

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, **kwargs):
        return build_coco_manifests(
            {
                "train": self.annotations["captions_train2017"],
                "val": self.annotations["captions_val2017"],
            },
            self.root / "study",
            cohort_sizes=self.sizes,
            audit_count=2,
            **kwargs,
        )

    def test_locked_cohorts_disjoint_with_audit_subset_and_exclusions(self):
        result = self.build(
            exclusions=[101, 102], exclusion_provenance={"source": "explicit-test"}
        )
        ids = {name: set(info["image_ids"]) for name, info in result["cohorts"].items()}
        self.assertFalse(
            ids["B"] & ids["C"] or ids["C"] & ids["D"] or ids["B"] & ids["D"]
        )
        self.assertTrue(ids["audit"] <= ids["D"])
        self.assertEqual(len(ids["remaining"]), 18 - 10)
        self.assertNotIn(101, set.union(ids["B"], ids["C"], ids["D"], ids["remaining"]))
        row = read_jsonl(result["manifests"]["A"])[0]
        self.assertEqual(row["query"], "Describe this image.")
        self.assertTrue(row["image"].startswith("images/train2017/"))
        self.assertIn("unknown", result["configuration"]["training_overlap"])
        self.assertEqual(
            result["identity"],
            self.build(
                exclusions=[102, 101], exclusion_provenance={"source": "explicit-test"}
            )["identity"],
        )

    def test_seed_source_or_manifest_drift_is_rejected(self):
        result = self.build(seed=7)
        with self.assertRaises(ArtifactError):
            self.build(seed=8)
        Path(result["manifests"]["B"]).write_text("{}\n")
        with self.assertRaises(ArtifactError):
            self.build(seed=7)

    def test_offline_preparation_produces_small_annotations_and_portable_registry(self):
        registry = prepare_coco_data(
            self.root / "prepared",
            cohort_sizes=self.sizes,
            audit_count=2,
            annotation_files=self.annotations,
            download_images=False,
            progress=None,
        )
        selected = set(registry["cohorts"]["A"]["image_ids"])
        train = json.loads(
            Path(registry["annotations"]["instances_train2017"]).read_text()
        )
        self.assertEqual({image["id"] for image in train["images"]}, selected)
        self.assertEqual({row["image_id"] for row in train["annotations"]}, selected)
        portable = json.loads(
            (self.root / "prepared/dataset-manifest.json").read_text()
        )
        self.assertFalse(
            any(Path(path).is_absolute() for path in portable["annotations"].values())
        )

    def test_external_queries_keep_multiple_questions_on_same_image(self):
        path = self.root / "pope.jsonl"
        atomic_jsonl(
            path,
            [
                {
                    "question_id": 1,
                    "image": "COCO_val2014_000000000123.jpg",
                    "text": "Is there a cat?",
                    "label": "yes",
                },
                {
                    "question_id": 2,
                    "image": "COCO_val2014_000000000123.jpg",
                    "text": "Is there a dog?",
                    "label": "no",
                },
            ],
        )
        result = load_external_jsonl(path, dataset="pope", split="official-random")
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(pope_image_ids([path]), [123])
        self.assertFalse(result["project_heldout"])
        atomic_jsonl(
            path,
            [
                {"id": 1, "image": "a.jpg", "text": "query"},
                {"id": 1, "image": "b.jpg", "text": "query"},
            ],
        )
        with self.assertRaises(ArtifactError):
            load_external_jsonl(path, dataset="amber")

    def test_frozen_instances_annotations_cannot_change_behind_same_caption_manifest(
        self,
    ):
        options = {
            "cohort_sizes": self.sizes,
            "audit_count": 2,
            "annotation_files": self.annotations,
            "download_images": False,
            "progress": None,
        }
        prepare_coco_data(self.root / "prepared", **options)
        annotation = self.annotations["instances_train2017"]
        data = json.loads(annotation.read_text())
        data["annotations"][0]["category_id"] = 2
        atomic_json(annotation, data)
        with self.assertRaises(ArtifactError):
            prepare_coco_data(self.root / "prepared", **options)

    def test_image_verification_fails_on_missing_or_escaping_paths(self):
        with self.assertRaises(ArtifactError):
            verify_manifest_images([{"image": "missing.jpg"}], self.root, decode=False)
        with self.assertRaises(ArtifactError):
            verify_manifest_images(
                [{"image": "../outside.jpg"}], self.root, decode=False
            )

    def test_image_verification_checks_actual_dimensions_and_hashes(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")
        Image.new("RGB", (8, 6), "red").save(self.root / "image.jpg")
        result = verify_manifest_images(
            [{"image": "image.jpg", "width": 8, "height": 6}], self.root
        )
        self.assertEqual(result["count"], 1)
        self.assertEqual(len(result["files"]["image.jpg"]["sha256"]), 64)
        with self.assertRaises(ArtifactError):
            verify_manifest_images(
                [{"image": "image.jpg", "width": 9, "height": 6}], self.root
            )


if __name__ == "__main__":
    unittest.main()
