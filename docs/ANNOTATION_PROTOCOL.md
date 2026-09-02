# ImDUSTRY5 — Annotation Protocol v2

This document defines what counts as an annotatable object. It exists because
v1 of the dataset was annotated without a written protocol, which produced
systematic under-annotation (67.7% of images carry exactly one box, median = 1
object/image, 3,084 boxes over 1,500 images).

**Every rule below is binding for the v2 re-annotation pass.** Include this
document in the dataset release and summarise it in the paper — reviewers of a
dataset paper will ask for exactly this.

---

## R0 — Exhaustiveness (the rule that was broken in v1)

> On every image, **every visible instance** of the nine component classes must
> be annotated: Bolt, Frame, Wheel, Wheel Support, Wrench, Box, Flange, Nut,
> Washer.

There is no "label the interesting one" option. An unlabelled visible object is
not a neutral omission — YOLO treats every unlabelled region as background, so a
missed bolt actively teaches the detector to suppress bolts. This is the single
cause of the low precision seen in the v1 baselines (Nut: P=0.47 with 49 false
positives, most of which are real nuts that were never labelled).

If an image contains 18 bolts, it gets 18 boxes.

---

## R1 — Context classes: `Table` and `Support`

These two classes are **not** annotated exhaustively. They are annotated only
when the photograph is *about* them.

Binding test — annotate `Table` / `Support` only if **both** hold:

1. the object covers **≥ 40%** of the image area, **and**
2. no component-class object in the image covers **≥ 10%** of the image area.

In plain terms: a workbench that merely happens to be under a flange is not
annotated; a photograph taken of the workbench itself is.

This rule is machine-auditable — `05_audit_labels.py` checks it and reports
violations.

> **Disclose this in the paper.** `Table` and `Support` are a different
> annotation regime from the other nine classes, and after re-annotation they
> will still have few instances (v1: 31 and 29). With that support, per-class AP
> is dominated by noise. Report them separately from the nine exhaustively
> annotated classes, or state the limitation explicitly.

---

## R2 — Occlusion

- Annotate an object if **≥ 25%** of it is visible.
- Box the **full visible extent only** — do not extrapolate the hidden part.
- Objects occluding each other are each annotated in full-visible-extent terms;
  overlapping boxes are expected and correct.

## R3 — Truncation at the image border

- Annotate objects cut by the frame edge if ≥ 25% of the object is inside.
- Clip the box to the image bounds. No coordinate may fall outside `[0, 1]`.

## R4 — Minimum size

- Annotate down to **≥ 12 px** on the shorter box side, measured at native
  image resolution.
- Below that, skip. Do not annotate objects that are indistinguishable blurs,
  even if you know from context what they are.

## R5 — Assembled vs. loose parts

This was a major source of v1 inconsistency.

- A `Frame` with bolts fastened into it gets **one `Frame` box plus one `Bolt`
  box per individually visible bolt**. Assembly does not absorb its parts.
- The same applies to `Wheel Support` plates mounted on a `Frame`, and to
  `Nut` / `Washer` on a visible stud.
- If individual fasteners have merged into an unresolvable cluster at the
  captured resolution, apply R4 and skip them rather than guessing a count.

## R6 — Depth of field / background objects

- Annotate objects on the **working surface and its immediate surroundings**.
- Do **not** annotate objects inside distant storage bins, on far shelves, or
  in adjacent workstations across the room, even when recognisable.
- Operational boundary: if the object is out of focus, it is background.

## R7 — Box tightness

- The box is the tightest axis-aligned rectangle containing the visible extent.
- Target slack ≤ 2 px per side.

## R12 — The frame trolley is `Support`, never `Frame`

Ruled by the dataset author, 2026-07-29. The wheeled metal trolley that carries
stacked finished frames is annotated **`Support`**. It is not a `Frame`, even
though it is welded tube and even though frames are hanging on it — a detector
trained on frames will propose it as one, and that proposal is wrong.

R1's area test does not gate this case: the trolley is annotated whenever it is
identifiable, because the author ruled on the object directly rather than on its
prominence. R1 continues to govern the pegboard, racks and the workbench.

## R11 — `Wheel` is the castor wheel component, in any colour

`Wheel` means the **polyurethane castor wheel fitted to the frame** — the
component being assembled. It appears in green, red, grey and black; colour is
not part of the definition. (An early draft of this rule said "green only",
written before the red and grey stock showed up in the review sheets. Colour
was never the discriminator — provenance is.)

Not annotated, even though a detector reasonably fires on them:

- black conveyor rollers in the flow rack,
- black castors under trolleys, workbenches and drawer units,
- the red emergency-stop button, which the detector repeatedly proposes as a
  wheel on account of its shape.

Rationale: the taxonomy describes assembly components, and R6 already excludes
workstation infrastructure. Flag this back to the author if trolley castors are
meant to be in scope — it would change roughly 30 boxes in the validation set.

## R10 — What `Wrench` means here

Ruled by the dataset author, 2026-07-29, against `frame_0054_18.486`:

- **Annotate:** the **blue powered nutrunner** — blue body, rubber grip, torque
  LEDs, bit holder at the nose. It is the tool actually used on this line, and
  it is the only `Wrench` instance in that reference frame.
- **Do not annotate:** the small mechanical hand wrench hanging beside it on the
  pegboard, and any other hand tool.

Note the direction of this rule: it is the *powered* tool that counts and the
*hand* tool that does not. That is the opposite of what the class name suggests,
so anyone reusing these labels must read this rule before assuming the class
name means what it usually means.

## Scene reference — what is in the storage bins

Supplied by the dataset author, 2026-07-29. At the captured resolution the
contents of these bins are not reliably distinguishable by eye (a nut and a
washer are the same silver disc from above), so this mapping is the ground
truth for identifying them. On the standard overhead workstation shot, left to
right along the rail:

| # | Bin | Contents |
|---|-----|----------|
| 1 | blue  | **Bolt** |
| 2 | green | **Flange** |
| 3 | blue  | **Nut** |
| 4 | blue  | **Washer** |
| 5 | green | **Wheel** |

The bin itself is annotated `Box`; its contents are annotated per R9 below.

## R9 — Dense clusters of small parts — **RESOLVED: option (b)**

The v2 re-annotation pass surfaced a case v1 never confronted: bins holding
loose fasteners. `IMG_7589` alone draws 84 boxes, and the reviewer resolves
perhaps half the nuts actually in the bin; the rest are mutually occluded
beyond recovery. R0 exhaustiveness is physically unsatisfiable on these images.

**The rule applied in the released v2 labels is option (b) below.** Fasteners in
dense bins are boxed individually for every instance that stays visually
separable at the captured resolution. The remainder are left unannotated under
R4 (minimum size) rather than estimated, so these images are densely but **not
exhaustively** annotated. This is a documented departure from R0 and must be
stated in any paper describing the release.

Extent of the affected subset, measured on the released labels: **60 images
(4.0% of the collection) carry 20 or more `Bolt`, `Nut` or `Washer` boxes, and
those images account for 2,635 boxes — 19.8% of all annotation.** 50 of them are
in the training split and 10 in validation. The densest single image carries 88
boxes. Recall measured against this subset is pessimistic, and the subset is not
suitable material for counting tasks.

The options considered, for the record:

- **(a) Crowd region.** Draw one box over the pile, class `Box`/bin, and record
  the image in `reports/crowd_images.txt`. Exclude those regions from
  evaluation, the way COCO's `iscrowd=1` does. Reviewers recognise the
  convention, but YOLO's format has no `iscrowd` field, so the exclusion list
  has to ship as a side-car file that most consumers will silently ignore.
- **(b) Annotate only clearly separable instances** and state that dense bins
  are annotated non-exhaustively. **Chosen.** It keeps the label file readable
  by any standard loader with no side-car, and it preserves the dense-scene
  difficulty that makes these images interesting. The cost is that R0 does not
  hold on this subset, which is why the extent is quantified above rather than
  left vague.
- **(c) Drop the affected images.** Cleanest guarantee of exhaustiveness; costs
  4% of the data and removes a genuinely interesting difficulty from the
  benchmark.

Do **not** leave this to per-image improvisation — that is how v1 drifted.

## R8 — Ambiguity

- If class membership is genuinely unclear, **do not guess**. Leave the object
  unannotated and add the image to `reports/ambiguous.txt` for adjudication.
- Recurring ambiguities must be resolved once, written into this document as a
  new rule, and applied retroactively.

---

## Class definitions

| ID | Class | Definition |
|----|-------|------------|
| 0 | Bolt | Threaded fastener with a head; includes studs protruding from a plate. |
| 1 | Frame | The welded rectangular chassis assembly. |
| 2 | Wheel | Castor wheel, complete or as the wheel body alone. |
| 3 | Wheel Support | The stamped mounting plate / castor bracket. |
| 4 | Wrench | **The blue powered nutrunner only** — see R10. |
| 5 | Box | Cardboard or plastic container / bin. |
| 6 | Flange | Flat perforated connecting plate. |
| 7 | Nut | Internally threaded fastener. |
| 8 | Support | Workstation structure: pegboard, rack, upright, **and the mobile trolley that holds finished frames**. **R1 applies**, except for the trolley — see R12. |
| 9 | Table | Workbench surface. **R1 applies.** |
| 10 | Washer | Flat annular disc. |

---

## Annotation workflow and quality control for v2

1. The original annotations were created manually in
   [makesense.ai](https://www.makesense.ai/) and exported in YOLO format.
2. For v2, the earlier labels were expanded through model-assisted
   re-annotation. Detector outputs were treated as review candidates rather than
   independently accepted ground truth.
3. Validation candidates were reviewed by a human against this protocol;
   incorrect boxes were removed or reclassified, and boxes the candidate model
   had missed were added by hand during the same review.
4. A second annotator subsequently reviewed the resulting label set and
   spot-checked 20% of the images against this protocol.
5. `05_audit_labels.py` supplied automated integrity checks for all label files:
   coordinate range, degenerate boxes, duplicates, R1 violations, R3 clipping
   and R4 minimum size.
6. Recurring ambiguities were resolved once by the dataset author, written into
   this document as a new rule, and applied retroactively to the whole
   collection. R10, R11 and R12 arose this way.
7. A formal inter-annotator agreement score was not computed for this release:
   the second annotator reviewed an existing label set rather than annotating a
   subset independently, so what was measured is agreement with this protocol,
   not agreement between two independent annotations.

Candidate-sheet review can remove or reclassify proposed boxes but cannot recover
an object that was absent from the candidate set. The v2 annotations should
therefore not be described as a complete from-scratch manual re-annotation.
