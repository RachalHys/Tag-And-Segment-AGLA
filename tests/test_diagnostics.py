import itertools

import numpy as np
from pycocotools import mask as coco_mask

from experiments.tas_v7.evaluation import ChairLexicon
from experiments.tas_v7.study_diagnostics import _mask, matched_annotation_instances
from tas_runtime.perception import PerceptionResult


def test_annotation_matching_is_one_to_one_and_excludes_crowds_and_unknown_labels():
    prediction = PerceptionResult(
        np.zeros((8, 8), bool),
        np.zeros((3, 8, 8), bool),
        np.array([[1, 1, 4, 4], [1, 1, 4, 4], [6, 6, 8, 8]], np.float32),
        np.ones(3),
        {"proposed_labels": ["cat", "cat", "object"]},
    )
    annotations = [
        {"id": 4, "category_id": 17, "bbox": [1, 1, 3, 3]},
        {"id": 6, "category_id": 17, "bbox": [2, 1, 3, 3]},
        {"id": 7, "category_id": 17, "bbox": [1, 1, 3, 3], "iscrowd": 1},
        {"id": 9, "category_id": 18, "bbox": [6, 6, 2, 2]},
    ]
    matches = matched_annotation_instances(
        prediction, annotations, {17: "cat", 18: "dog"}, ChairLexicon()
    )
    assert [(item["prediction_index"], item["annotation_id"]) for item in matches] == [
        (0, 4),
        (1, 6),
    ]
    assert len({item["annotation_id"] for item in matches}) == len(matches)


def test_coco_polygons_compressed_and_uncompressed_rle_give_the_same_mask():
    expected = np.zeros((7, 8), np.uint8)
    expected[1:4, 1:4] = 1
    polygon = [[1, 1, 4, 1, 4, 4, 1, 4]]
    compressed = coco_mask.encode(np.asfortranarray(expected))
    counts = [
        len(list(group)) for _, group in itertools.groupby(expected.ravel(order="F"))
    ]
    uncompressed = {"size": [7, 8], "counts": counts}
    for segmentation in (polygon, compressed, uncompressed):
        decoded = _mask({"segmentation": segmentation}, 7, 8)
        assert decoded.dtype == bool
        np.testing.assert_array_equal(decoded, expected)
