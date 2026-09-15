# ImDUSTRY5-1.5K

DOI: https://doi.org/10.5281/zenodo.19045784

**ImDUSTRY5-1.5K** is a public benchmark dataset for industrial object detection
collected in a real production environment. It is designed for reproducible
evaluation of object detectors under realistic industrial conditions — clutter,
partial occlusion, scale variation and non-uniform illumination.

The DOI above is the **concept DOI** — it always resolves to the newest version on
Zenodo. This repository holds **v3**, which is being prepared for Zenodo: it blurs
faces that appeared incidentally in 33 images and removes 33 screen captures, as
described in [`docs/CHANGES_v3.md`](docs/CHANGES_v3.md). Do not use **v1** or
**v2**: both contain the images before anonymisation, and v1 also carries the
superseded first annotation pass (3,084 boxes).

Companion resource for the paper:

**ImDUSTRY5-1.5K: A Public Benchmark Dataset for Industrial Object Detection in
Real Production Environments**

---

## Contents

```
data.yaml              dataset definition
images/train  (1240)   images/val  (227)
labels/train           labels/val           YOLO format
splits/split.json      the official split, by file stem
weights/               released YOLOv8s checkpoint
docs/                  annotation protocol, label provenance, v3 changes
train.py               reproduce a baseline
evaluate.py            score a checkpoint on the validation split
audit_labels.py        re-run the label integrity checks
```

## Quick start

```bash
pip install ultralytics
python train.py                     # YOLOv8s baseline
python evaluate.py                  # score the released weights
python audit_labels.py              # check every label file
```

`train.py` fixes the seed and every setting that affects the result, so a clean
run reproduces the baseline numbers on the official split.

## Dataset at a glance

| | |
|---|---|
| Images | 1,467 |
| Annotated instances | **12,868** |
| Classes | 11 |
| Train / val images | 1,240 / 227 |
| Train / val instances | 10,572 / 2,296 |
| Objects per image | mean 8.77, median 6, max 88 |
| Sources | 976 smartphone photographs, 491 frames from a Pupil Labs Neon scene camera |
| Resolutions | 1536×2048, 2048×1536, 5712×4284 (smartphone); 1600×1200 (eye tracker) |

### Class distribution

| Class | Train | Val | Total |
|---|---|---|---|
| Nut | 3,555 | 843 | 4,398 |
| Wheel Support | 1,621 | 320 | 1,941 |
| Wheel | 1,300 | 235 | 1,535 |
| Box | 1,064 | 221 | 1,285 |
| Flange | 796 | 187 | 983 |
| Bolt | 734 | 216 | 950 |
| Washer | 672 | 122 | 794 |
| Frame | 433 | 78 | 511 |
| Wrench | 350 | 59 | 409 |
| Support | 23 | 10 | 33 |
| Table | 24 | 5 | 29 |

`Support` and `Table` are contextual categories annotated only when they are the
subject of the shot (see `docs/ANNOTATION_PROTOCOL.md`, rule R1). Their instance
counts are consequently low, and per-class AP for them is dominated by noise —
read those two rows with care.

## Annotation process and format

The original bounding-box annotations were created manually in
[makesense.ai](https://www.makesense.ai/). For v2, those labels were expanded in
a model-assisted re-annotation pass: detector outputs were used as candidates,
not as independently accepted ground truth. Validation candidates were reviewed
by a human against the written class definitions and inclusion rules, with
incorrect classes and boxes corrected or removed. Automated integrity checks
were subsequently applied to all label files to detect invalid coordinates,
zero-area or duplicate boxes, extreme aspect ratios and boundary-clipping issues;
`audit_labels.py` repeats them.

YOLO detection format, one row per instance, coordinates normalised to the image:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Class definitions, occlusion and truncation thresholds, minimum object size and
the treatment of dense fastener bins are specified in
[`docs/ANNOTATION_PROTOCOL.md`](docs/ANNOTATION_PROTOCOL.md). Two rules are worth
flagging because they are not what the class names suggest:

- **`Wrench`** is the blue powered nutrunner used on the line, not a hand wrench.
  The hand wrenches on the pegboard are unlabelled background.
- **`Wheel`** is the castor wheel component in any colour; conveyor rollers,
  furniture castors and the red emergency-stop button are not annotated.

## Baseline results

> **These are the v2 numbers.** The three baselines are being retrained on v3;
> this section, the checkpoint in `weights/` and the tables in `docs/` will be
> replaced when those runs finish.

Validation split, `imgsz=960`, seed 0.

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| YOLOv8n | 0.801 | 0.693 | 0.762 | 0.564 |
| YOLOv8s | 0.796 | 0.703 | 0.747 | 0.560 |
| Faster R-CNN MobileNetV3-320 | 0.567 | 0.356 | 0.490 | 0.329 |

The two YOLO scales land within 0.015 mAP of each other — close enough that the
gap is inside single-seed variation, so treat them as comparable rather than
ranked. YOLOv8s buys a little recall, YOLOv8n a little precision.

The Faster R-CNN row is a cross-architecture reference point, not a fair fight:
it ran 50 epochs at batch 16 in its native `torchvision` configuration against
100 epochs at batch 8 and `imgsz=960` for the YOLO runs, and its 320-pixel input
leaves most fasteners here spanning a handful of pixels. Same split, same
hardware, same evaluation code; everything else differs.

### Per-class, YOLOv8s

| Class | Precision | Recall | AP@0.5 | AP@0.5:0.95 |
|---|---|---|---|---|
| Wheel Support | 0.919 | 0.844 | 0.885 | 0.653 |
| Table | 0.725 | 0.800 | 0.795 | 0.641 |
| Wheel | 0.832 | 0.783 | 0.788 | 0.617 |
| Support | 1.000 | 0.580 | 0.783 | 0.628 |
| Nut | 0.791 | 0.734 | 0.782 | 0.536 |
| Box | 0.862 | 0.717 | 0.767 | 0.629 |
| Frame | 0.741 | 0.696 | 0.738 | 0.596 |
| Flange | 0.819 | 0.663 | 0.736 | 0.560 |
| Washer | 0.711 | 0.562 | 0.660 | 0.436 |
| Wrench | 0.614 | 0.783 | 0.658 | 0.492 |
| Bolt | 0.746 | 0.567 | 0.629 | 0.377 |

`Washer` and `Bolt` remain the hardest categories: both are small, low-contrast
and easily confused with `Nut` at the captured resolution.

## Official split

`splits/split.json` lists the exact train and validation file stems. Use it
rather than re-splitting — the reported numbers are only comparable on this
partition, and a random 80/20 draw will not reproduce them.

## Limitations

- Collected at a single industrial site, so the visual domain is narrow.
- Train/validation protocol only; no cross-domain evaluation.
- Class frequencies are naturally imbalanced, severely so for `Support` and
  `Table`.
- Bins holding loose fasteners cannot be annotated exhaustively at the captured
  resolution. Instances that stay visually separable are boxed individually; the
  rest are left unlabelled under the minimum-size rule rather than guessed. The
  56 images carrying 20 or more `Bolt`/`Nut`/`Washer` boxes (3.8% of the images,
  19.5% of all annotation) are the ones affected — recall measured on them is
  pessimistic, and they are not suitable for counting tasks. See rule R9 in the
  annotation protocol.
- Annotation was reviewed against the written protocol by a second annotator on
  20% of the images, but no formal inter-annotator agreement coefficient was
  computed for this release.

## License

Released under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
https://creativecommons.org/licenses/by/4.0/

## Citation

Cite the deposit, so that the version used is identifiable:

> Gurbanov, K., Gardashova, L., Bobenrieth, C., Al Makdessi, N., Chabrol, G.,
> Rima, S. & Amhaz, R. *ImDUSTRY5-1.5K: a public benchmark dataset for
> industrial object detection in real production environments.* Zenodo
> https://doi.org/10.5281/zenodo.19045784 (2026).

```bibtex
@dataset{gurbanov2026imdustry5,
  author    = {Gurbanov, Kanan and Gardashova, Latafat and Bobenrieth, Cedric
               and Al Makdessi, Nathalie and Chabrol, Gr\'egoire and
               Rima, Samy and Amhaz, Rabih},
  title     = {{ImDUSTRY5-1.5K}: A Public Benchmark Dataset for Industrial
               Object Detection in Real Production Environments},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19045784},
  note      = {Concept DOI; always resolves to the latest version}
}
```

## Contact

- kanan.gurbanov@ufaz.az
- l.qardashova@asoiu.edu.az
- amhaz@unistra.fr
