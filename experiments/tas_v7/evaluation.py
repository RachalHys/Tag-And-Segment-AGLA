"""Auditable CHAIR/COCO2017, POPE and paired image-cluster statistics.

CHAIR follows LisaAnne/Hallucination/utils/chair.py and data/synonyms.txt:
https://github.com/LisaAnne/Hallucination .  Ground truth is the UNION of
instance categories and object mentions in all reference captions.  Category
coverage below deliberately uses INSTANCE annotations only and is NOT AMBER
COVER.  COCO annotations are incomplete: a CHAIR false mention is annotation-
unsupported, not a proof that an entity is absent in the actual photograph.

The tokenizer is NLTK word_tokenize; TextBlob's Pattern-derived singularize
provides the Python-3 port of the original Pattern singularizer.  Package and
resource versions must be recorded by the runner.  No automatic resource or
annotation downloads happen while scoring.

POPE follows https://github.com/AoiDragon/POPE/blob/main/evaluate.py exactly
for binary predictions.  Its unusual parser maps every non-negative response
to YES, including malformed responses.  A separate strict invalid rate makes
this behavior visible without changing the official accuracy/F1.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


CHAIR_SOURCE = "https://github.com/LisaAnne/Hallucination"
POPE_SOURCE = "https://github.com/AoiDragon/POPE/blob/main/evaluate.py"
CHAIR_AUDITED_SOURCE_SHA256 = (
    "b05f701ce6ec0125dd700ffb80ed58bfcf2686df8c9bd62f70169cb928de4627"
)
CHAIR_AUDITED_SYNONYMS_SHA256 = (
    "b685537b36fe21b791c87151b96a991ceb582edff19f3da204dcc7a7905502f2"
)

# Canonical nodes and synonym rows from the original CHAIR data/synonyms.txt.
# Preserve its duplicate entries/whitespace quirks; corrections would constitute
# a separate evaluator version, rather than an invisible improvement.
_CHAIR_SYNONYMS = """person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, person, woman, guy, foreigner, child, gentleman, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter,  motor bike, motor cycle, motorbike, scooter, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, jetliner, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, lamb, goat, ewe
cow, cattle, oxen, ox, calf, cattle, holstein, heifer, buffalo, bull, zebu, bison
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake,  cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, toilet, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush"""


class ChairLexicon:
    """Original CHAIR object parser, independent of model-generated inventory."""

    version = "chair-original-mappings-textblob-pattern-v1"

    def __init__(
        self, *, tokenizer: Callable | None = None, singularizer: Callable | None = None
    ):
        if tokenizer is None:
            from nltk import word_tokenize

            tokenizer = word_tokenize
        if singularizer is None:
            from textblob.en.inflect import singularize

            singularizer = singularize
        self.tokenizer = tokenizer
        self.singularizer = singularizer
        self.inverse = {}
        for row in _CHAIR_SYNONYMS.splitlines():
            values = row.strip().split(", ")
            self.inverse.update({word: values[0] for word in values})
        self.canonical = frozenset(self.inverse.values())
        phrases = [
            "motor bike",
            "motor cycle",
            "air plane",
            "traffic light",
            "street light",
            "traffic signal",
            "stop light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "suit case",
            "sports ball",
            "baseball bat",
            "baseball glove",
            "tennis racket",
            "wine glass",
            "hot dog",
            "cell phone",
            "mobile phone",
            "teddy bear",
            "hair drier",
            "potted plant",
            "bow tie",
            "laptop computer",
            "stove top oven",
            "home plate",
            "train track",
        ]
        self.phrases = {phrase: phrase for phrase in phrases}
        for animal in [
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "animal",
            "cub",
        ]:
            self.phrases.update({f"baby {animal}": animal, f"adult {animal}": animal})
        self.phrases.update(
            {f"passenger {vehicle}": vehicle for vehicle in ["jet", "train"]}
        )
        self.phrases.update(
            {"bow tie": "tie", "toilet seat": "toilet", "wine glas": "wine glass"}
        )

    @property
    def metadata(self):
        from importlib.metadata import version

        return {
            "adapter_version": self.version,
            "source": CHAIR_SOURCE,
            "audited_original_parser_sha256": CHAIR_AUDITED_SOURCE_SHA256,
            "audited_original_synonyms_sha256": CHAIR_AUDITED_SYNONYMS_SHA256,
            "normalized_synonyms_sha256": hashlib.sha256(
                (_CHAIR_SYNONYMS + "\n").encode()
            ).hexdigest(),
            "synonyms_normalization": "strip each row then join with LF including final LF",
            "nltk": version("nltk"),
            "textblob": version("textblob"),
            "ground_truth": "instance categories UNION reference-caption objects",
            "coverage_scope": "instance-category annotations only",
        }

    @lru_cache(maxsize=65536)
    def caption_to_words(
        self, caption: str
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...], tuple[str, ...]]:
        words = [self.singularizer(word) for word in self.tokenizer(caption.lower())]
        merged, original_positions = [], []
        i = 0
        while i < len(words):
            phrase = " ".join(words[i : i + 2])
            original_positions.append(i)
            if phrase in self.phrases:
                merged.append(self.phrases[phrase])
                i += 2
            else:
                merged.append(words[i])
                i += 1
        # Original CHAIR special case for 'seat of the toilet'.
        if "toilet" in merged and "seat" in merged:
            retained = [
                (word, pos)
                for word, pos in zip(merged, original_positions)
                if word != "seat"
            ]
            merged = [word for word, _ in retained]
            original_positions = [pos for _, pos in retained]
        recognized = [
            (word, pos)
            for word, pos in zip(merged, original_positions)
            if word in self.inverse
        ]
        return (
            tuple(word for word, _ in recognized),
            tuple(self.inverse[word] for word, _ in recognized),
            tuple(pos for _, pos in recognized),
            tuple(merged),
        )

    def fixture_check(self) -> dict:
        fixtures = {
            "A person rides a bicycle beside a dog.": ("person", "bicycle", "dog"),
            "Two dogs sit beside a traffic light.": ("dog", "traffic light"),
            "A hot dog on a dining table.": ("hot dog", "dining table"),
            "A baby bird sits near a passenger jet.": ("bird", "airplane"),
            "A wine glass and a cell phone.": ("wine glass", "cell phone"),
            "A teddy bear sits on the seat of the toilet.": ("teddy bear", "toilet"),
            "A bow tie and a laptop computer.": ("tie", "laptop"),
            "A dog and another dog.": ("dog", "dog"),
        }
        failures = []
        for caption, expected in fixtures.items():
            observed = self.caption_to_words(caption)[1]
            if observed != expected:
                failures.append(
                    {"caption": caption, "expected": expected, "observed": observed}
                )
        if failures:
            raise ValueError(f"CHAIR parser fixture mismatch: {failures}")
        return {
            "passed": len(fixtures),
            "canonical_categories": len(self.canonical),
            **self.metadata,
        }


class CocoChairEvaluator:
    """Load explicitly named COCO2017/2014 annotations using their actual IDs.

    No filename-based synthetic image renumbering is performed: shared 2014/
    2017 photographs already share numeric COCO IDs.  Missing image IDs or
    category mappings fail immediately.  Reference captions are mandatory.
    """

    def __init__(
        self,
        instance_files: Sequence[str | Path],
        caption_files: Sequence[str | Path],
        *,
        image_ids: Iterable[int] | None = None,
        lexicon: ChairLexicon | None = None,
    ):
        if not instance_files or not caption_files:
            raise ValueError(
                "CHAIR requires both instance AND reference-caption annotations"
            )
        self.lexicon = lexicon or ChairLexicon()
        selected = (
            None if image_ids is None else {int(image_id) for image_id in image_ids}
        )
        self.instance_categories: dict[int, set[str]] = defaultdict(set)
        self.chair_categories: dict[int, set[str]] = defaultdict(set)
        seen_instances, seen_captions = set(), set()
        self.annotation_sources = {
            "instances": [str(Path(p).name) for p in instance_files],
            "captions": [str(Path(p).name) for p in caption_files],
        }
        for path in instance_files:
            with Path(path).open(encoding="utf-8") as handle:
                data = json.load(handle)
            categories = {}
            for category in data["categories"]:
                name = category["name"]
                if name not in self.lexicon.inverse:
                    raise ValueError(f"Unmapped COCO category {name!r}")
                categories[int(category["id"])] = self.lexicon.inverse[name]
            for image in data["images"]:
                image_id = int(image["id"])
                if selected is None or image_id in selected:
                    seen_instances.add(image_id)
                    self.instance_categories[image_id]
            for annotation in data["annotations"]:
                image_id = int(annotation["image_id"])
                if image_id in seen_instances:
                    self.instance_categories[image_id].add(
                        categories[int(annotation["category_id"])]
                    )
        for image_id, categories in self.instance_categories.items():
            self.chair_categories[image_id].update(categories)
        for path in caption_files:
            with Path(path).open(encoding="utf-8") as handle:
                data = json.load(handle)
            annotated_images = {
                int(annotation["image_id"]) for annotation in data["annotations"]
            }
            seen_captions.update(
                annotated_images if selected is None else annotated_images & selected
            )
            for annotation in data["annotations"]:
                image_id = int(annotation["image_id"])
                if image_id in seen_instances:
                    self.chair_categories[image_id].update(
                        self.lexicon.caption_to_words(annotation["caption"])[1]
                    )
        required = seen_instances if selected is None else selected
        missing_instances, missing_captions = (
            required - seen_instances,
            required - seen_captions,
        )
        if missing_instances or missing_captions:
            raise ValueError(
                f"Missing annotation IDs: instances={sorted(missing_instances)[:10]}, captions={sorted(missing_captions)[:10]}"
            )

    @classmethod
    def from_annotation_dir(
        cls,
        directory: str | Path,
        *,
        split: str = "val2017",
        image_ids: Iterable[int] | None = None,
        **kwargs,
    ):
        if split not in {"train2017", "val2017", "train2014", "val2014"}:
            raise ValueError("Use an explicit supported COCO split/year")
        root = Path(directory)
        return cls(
            [root / f"instances_{split}.json"],
            [root / f"captions_{split}.json"],
            image_ids=image_ids,
            **kwargs,
        )

    def score(self, image_id: int, caption: str, **identity) -> dict:
        image_id = int(image_id)
        if image_id not in self.chair_categories:
            raise KeyError(f"Image {image_id} was not loaded into this evaluator")
        words, nodes, positions, _ = self.lexicon.caption_to_words(caption)
        gt = self.chair_categories[image_id]
        annotated = self.instance_categories[image_id]
        mentions = [
            {
                "surface": word,
                "category": node,
                "token_position": pos,
                "annotation_supported": node in gt,
            }
            for word, node, pos in zip(words, nodes, positions)
        ]
        false = sum(node not in gt for node in nodes)
        covered = set(nodes) & annotated
        return {
            **identity,
            "image_id": image_id,
            "caption": caption,
            "CHAIRs": int(false > 0),
            "CHAIRi": false / len(nodes) if nodes else 0.0,
            "object_mentions": len(nodes),
            "true_mentions": len(nodes) - false,
            "false_mentions": false,
            "covered_categories": len(covered),
            "annotated_categories": len(annotated),
            "category_coverage": len(covered) / len(annotated) if annotated else None,
            "length_words": len(re.findall(r"\b[\w]+(?:['’-][\w]+)*\b", caption)),
            "mentions": mentions,
            "chair_gt_categories": sorted(gt),
            "instance_gt_categories": sorted(annotated),
            "evaluator_version": self.lexicon.version,
        }

    score_caption = score

    def evaluate(
        self, records: Iterable[Mapping], *, caption_key: str = "caption"
    ) -> dict:
        scored = [
            self.score(
                record["image_id"],
                record[caption_key],
                **{
                    key: value
                    for key, value in record.items()
                    if key not in {"image_id", caption_key}
                },
            )
            for record in records
        ]
        return {
            "records": scored,
            "overall": aggregate_caption_metrics(scored),
            "metadata": {
                **self.lexicon.metadata,
                "annotation_files": self.annotation_sources,
            },
        }


def aggregate_caption_metrics(records: Sequence[Mapping]) -> dict:
    if not records:
        raise ValueError("No scored captions; do not report an empty run as perfect")
    n = len(records)
    mentions = sum(record["object_mentions"] for record in records)
    false = sum(record["false_mentions"] for record in records)
    coverage = [
        record["category_coverage"]
        for record in records
        if record["category_coverage"] is not None
    ]
    annotated = sum(record["annotated_categories"] for record in records)
    return {
        "n_captions": n,
        "n_images": len({record["image_id"] for record in records}),
        "CHAIRs": sum(record["CHAIRs"] for record in records) / n,
        "CHAIRi": false / mentions if mentions else None,
        "object_mentions": mentions,
        "false_mentions_total": false,
        "true_mentions": sum(record["true_mentions"] for record in records) / n,
        "false_mentions": false / n,
        "category_coverage": float(np.mean(coverage)) if coverage else None,
        "category_recall_micro": sum(record["covered_categories"] for record in records)
        / annotated
        if annotated
        else None,
        "length_words": sum(record["length_words"] for record in records) / n,
        "zero_object_caption_rate": sum(
            record["object_mentions"] == 0 for record in records
        )
        / n,
        "coverage_scope": "COCO instance-annotation category recall; not AMBER COVER",
    }


def official_pope_prediction(answer: str) -> str:
    """Exact official first-sentence/comma/negation policy (case sensitive)."""
    text = answer.split(".", 1)[0].replace(",", "")
    words = text.split(" ")
    return "no" if any(word in words for word in ("No", "not", "no")) else "yes"


def score_pope(
    records: Sequence[Mapping],
    *,
    answer_key: str = "answer",
    label_key: str = "label",
    variant: str | None = None,
) -> dict:
    if not records:
        raise ValueError("POPE records must not be empty")
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    predicted_yes, invalid = 0, 0
    for record in records:
        label = str(record[label_key]).lower()
        if label not in {"yes", "no"}:
            raise ValueError(f"Invalid POPE ground truth {label!r}")
        answer = record[answer_key]
        if not isinstance(answer, str):
            raise ValueError(
                "Missing/invalid generation must be recorded as failure, not converted to an answer"
            )
        prediction = official_pope_prediction(answer)
        predicted_yes += prediction == "yes"
        invalid += re.match(r"^\s*(?:yes|no)\b", answer, re.IGNORECASE) is None
        key = ("t" if prediction == label else "f") + (
            "p" if prediction == "yes" else "n"
        )
        counts[key] += 1
    tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        **counts,
        "n": len(records),
        "accuracy": (tp + tn) / len(records),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "yes_rate": predicted_yes / len(records),
        "invalid_rate": invalid / len(records),
        "invalid_definition": "no leading explicit yes/no; diagnostic only, official binary mapping unchanged",
        "source": POPE_SOURCE,
        "variant": variant,
    }


def align_pope_answers(
    answers: Sequence[Mapping], labels: Sequence[Mapping], *, answer_key: str = "answer"
) -> list[dict]:
    """Fail on missing/misaligned queries instead of zip-truncating a run."""
    if len(answers) != len(labels):
        raise ValueError("POPE answer and manifest lengths differ")
    result = []
    for answer, label in zip(answers, labels):
        for key in ("question_id", "image_id"):
            if key in answer and key in label and str(answer[key]) != str(label[key]):
                raise ValueError(
                    f"POPE {key} mismatch: {answer[key]!r} vs {label[key]!r}"
                )
        result.append(
            {**label, **answer, "label": label["label"], "answer": answer[answer_key]}
        )
    return result


def _pairing_key(record: Mapping):
    return (
        str(record["image_id"]),
        str(record.get("seed", "none")),
        str(record.get("question_id", record.get("query_id", "caption"))),
    )


def paired_cluster_bootstrap(
    baseline_records: Sequence[Mapping],
    candidate_records: Sequence[Mapping],
    *,
    metric: str = "CHAIRi",
    repeats: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Paired percentile CI for candidate minus baseline, resampling IMAGES.

    All seeds and queries for a sampled image remain in that cluster.  CHAIRi
    is recomputed as a ratio of mention counts, never as a mean of image rates.
    Missing or duplicate sample/seed/query pairs are errors, not exclusions.
    """
    if repeats < 100 or not 0 < confidence < 1:
        raise ValueError("Need repeats>=100 and confidence in (0,1)")
    if not baseline_records or not candidate_records:
        raise ValueError("Both paired record collections must be nonempty")
    baseline, candidate = {}, {}
    for source, target in (
        (baseline_records, baseline),
        (candidate_records, candidate),
    ):
        for record in source:
            key = _pairing_key(record)
            if key in target:
                raise ValueError(f"Duplicate paired sample key {key}")
            target[key] = record
    if set(baseline) != set(candidate):
        raise ValueError(
            "Methods have different image/seed/query keys; resolve failures before paired inference"
        )
    image_ids = sorted({key[0] for key in baseline})
    image_to_index = {image: i for i, image in enumerate(image_ids)}

    def components(record):
        if metric == "CHAIRi":
            return record["false_mentions"], record["object_mentions"]
        if metric == "category_recall_micro":
            return record["covered_categories"], record["annotated_categories"]
        if metric == "category_coverage" and record[metric] is None:
            return 0.0, 0.0
        if metric not in {
            "CHAIRs",
            "category_coverage",
            "length_words",
            "true_mentions",
            "false_mentions",
        }:
            raise ValueError(f"Unsupported bootstrap metric {metric!r}")
        return record[metric], 1.0

    arrays = []
    for collection in (baseline, candidate):
        array = np.zeros((len(image_ids), 2), dtype=np.float64)
        for key, record in collection.items():
            array[image_to_index[key[0]]] += components(record)
        arrays.append(array)
    sums = [array.sum(axis=0) for array in arrays]
    if any(pair[1] == 0 for pair in sums):
        raise ValueError(f"Undefined {metric}: zero denominator")
    estimate = sums[1][0] / sums[1][1] - sums[0][0] / sums[0][1]
    rng = np.random.default_rng(seed)
    deltas = []
    # Bounded chunks avoid repeated Python aggregation of thousands of captions.
    for start in range(0, repeats, 128):
        indices = rng.integers(
            0, len(image_ids), size=(min(128, repeats - start), len(image_ids))
        )
        b, c = (array[indices].sum(axis=1) for array in arrays)
        valid = (b[:, 1] > 0) & (c[:, 1] > 0)
        deltas.extend((c[valid, 0] / c[valid, 1] - b[valid, 0] / b[valid, 1]).tolist())
    if not deltas:
        raise ValueError("All bootstrap draws have undefined metric denominators")
    lo, hi = np.quantile(deltas, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
    return {
        "metric": metric,
        "delta_candidate_minus_baseline": float(estimate),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "confidence": confidence,
        "bootstrap_repeats": repeats,
        "valid_draws": len(deltas),
        "n_image_clusters": len(image_ids),
        "n_paired_records": len(baseline),
        "resampling_unit": "image; all queries/seeds retained",
        "seed": seed,
    }


paired_bootstrap = paired_cluster_bootstrap
