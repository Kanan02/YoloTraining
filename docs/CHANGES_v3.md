# Changes in version v3

Version v3 of ImDUSTRY5-1.5K differs from v2 in two ways. No annotation box was
edited, added or moved between the training and validation subsets.

## 1. Identifiable faces were blurred

A screening of every image for identifiable people found faces that appeared
incidentally:

- a printed photograph of a young child pinned on a workstation pegboard, visible
  in 29 frames of the eye-tracker scene video;
- a person seen through a window in the background of four photographs of one
  workbench;
- a printed photograph of two adults, in one of the screen captures removed under
  point 2 below.

In the 33 remaining affected images, each region was pixelated and then
Gaussian-blurred, so its content cannot be recovered. No blurred region overlaps
any annotation box, so every label is unchanged. The affected files are listed in
`v3_anonymised_images.txt`.

The screening combined four methods: face detection over the whole image and
over overlapping tiles; person detection restricted to people who do not enter
the frame from its lower edge; feature matching to locate the pinned photograph
in every frame; and visual review of every candidate and of the images taken
immediately before and after each one. It reduces the chance that an
identifiable face remains, but cannot prove that none does. If you find one,
please report it through the issue tracker or to the corresponding authors, and
it will be removed in the next version.

## 2. The 33 PNG screen captures were removed

All 33 files in PNG format were screen captures of processed eye-tracker scene
video rather than camera images. Several carried hand-landmark skeletons, detector
prediction boxes with class names and scores, or video-player controls rendered
into the pixels. A painted prediction box is a shortcut a detector can learn in
place of the object, so the files were removed together with their 445 annotation
boxes: 28 images from the training subset and 5 from the validation subset. They
are listed in `v3_removed_screen_captures.txt`, and `splits/split.json` no longer
contains them.

## v3 at a glance

| | v2 | v3 |
|---|---|---|
| Images | 1,500 | **1,467** |
| Training / validation images | 1,268 / 232 | **1,240 / 227** |
| Annotation boxes | 13,313 | **12,868** |
| Training / validation boxes | 10,943 / 2,370 | **10,572 / 2,296** |
| Pixel dimensions | 37 | **4** |
| Smartphone photographs | 976 | 976 |
| Eye-tracker scene-camera frames | 491 | 491 |
| Screen captures | 33 | **0** |

The 37 pixel dimensions of v2 were an artefact of the screen captures, each of
which had its own dimensions. The camera images come in four: 1536×2048, 2048×1536
and 5712×4284 from smartphones, and 1600×1200 from the Pupil Labs Neon scene camera.

## Baselines

All three reference detectors are being retrained on v3 with the published
configurations. Until those runs finish, the checkpoint in `weights/` and the
per-class tables in `docs/` are still the v2 ones; they will be replaced in a
following commit.

## Label audit

`audit_labels.py` in the repository root runs every integrity check reported in
the paper on the released files.

## Earlier versions

Do not use v1 or v2. v1 carries the superseded first annotation pass, and both
contain the images before anonymisation.
