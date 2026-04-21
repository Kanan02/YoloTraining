# ImDUSTRY5-1.5K
DOI: https://doi.org/10.5281/zenodo.19045785

**ImDUSTRY5-1.5K** is a public benchmark dataset for industrial object detection collected in a real production environment. The dataset was designed to support reproducible evaluation of object detectors under realistic industrial conditions such as clutter, partial occlusion, scale variation, and non-uniform illumination.

This repository contains the dataset resources, annotation format, and baseline benchmark information associated with the paper:

**ImDUSTRY5-1.5K: A Public Benchmark Dataset for Industrial Object Detection in Real Production Environments**

## Overview

Public datasets for industrial object detection remain limited compared with general-purpose benchmarks, despite the growing need for robust vision systems in manufacturing, assembly support, and automated inspection. ImDUSTRY5-1.5K addresses this gap by providing a real-world dataset of industrial workspace objects annotated for object detection.

The dataset contains:

- **1,500 images**
- **11 object classes**
- **YOLO-format annotations**
- **Official train/validation split**
- **Reference baseline results using YOLOv8n and YOLOv8s**

The dataset was collected on the Icam Strasbourg-Europe Production Line 5.0 under natural operating conditions without artificial staging or controlled studio lighting.

## Object Classes

ImDUSTRY5-1.5K includes the following 11 categories:

- bolt
- frame
- wheel
- wheel_support
- wrench
- box
- flange
- nut
- support
- table
- washer

These classes represent recurring objects that are visually relevant in the industrial workspace used for acquisition. The dataset combines assembly-related components and salient contextual objects that repeatedly appear in the production environment.

## Acquisition Setting

Images were captured in a real industrial setting and reflect practical challenges encountered in industrial vision applications, including:

- cluttered scenes
- partial occlusions
- repeated metallic textures
- viewpoint variation
- illumination differences
- natural class imbalance

This design makes the dataset suitable for evaluating detector robustness in realistic deployment-oriented scenarios.

## Annotation Format

Annotations are provided in **YOLO object detection format**.

Each label file contains one row per object instance:

```text
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to the image size.

## Suggested Repository Structure

A typical structure may look like this:

```text
ImDUSTRY5-1.5K/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── data.yaml
├── README.md
└── LICENSE
```

## Example `data.yaml`

```yaml
path: .
train: images/train
val: images/val

names:
  0: bolt
  1: frame
  2: wheel
  3: wheel_support
  4: wrench
  5: box
  6: flange
  7: nut
  8: support
  9: table
  10: washer
```

## Baseline Benchmark Results

Two YOLOv8 baselines were trained under the same protocol on the official train/validation split.

### YOLOv8n

- Precision: **0.83**
- Recall: **0.74**
- mAP@0.5: **0.76**
- mAP@0.5:0.95: **0.54**

### YOLOv8s

- Precision: **0.80**
- Recall: **0.76**
- mAP@0.5: **0.79**
- mAP@0.5:0.95: **0.57**

These results indicate that the dataset is able to reveal meaningful trade-offs between compact and moderately larger detector variants. In our experiments, YOLOv8s improved recall and both mAP measures, while YOLOv8n retained slightly higher precision.

## Training and Validation

The dataset can be used directly with Ultralytics YOLO.

Example training command:

```bash
yolo detect train data=data.yaml model=yolov8n.pt imgsz=640 epochs=50 batch=16
```

Example validation command:

```bash
yolo detect val data=data.yaml model=path/to/best.pt imgsz=640
```

## Quality Control

Annotations were created manually and reviewed through a two-stage quality-control process:

1. automated checks for invalid boxes, duplicates, and clipping issues
2. manual spot-check review of approximately 20% of the dataset, with emphasis on dense and ambiguous scenes

## Intended Use

ImDUSTRY5-1.5K is intended for:

- benchmark evaluation of industrial object detectors
- comparative experiments across model families and scales
- deployment-oriented detector studies
- industrial computer vision research
- educational and reproducible benchmarking purposes

## Limitations

Users should be aware of the following limitations:

- the dataset was collected from a single industrial site
- the current benchmark uses a train/validation protocol rather than cross-domain evaluation
- class frequencies are naturally imbalanced
- some categories remain more difficult due to scale, clutter, and visual similarity

These limitations do not reduce the usefulness of the dataset, but they should be considered when interpreting results.

## License

This dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

You are free to share and adapt the material for any purpose, provided that appropriate credit is given.

More information:  
https://creativecommons.org/licenses/by/4.0/

## Citation

If you use ImDUSTRY5-1.5K in your research, please cite the associated article and, where available, the archival dataset release.

Suggested citation:

Gurbanov, K., Bobenrieth, C., Amhaz, R., and Gardashova, L.  
*ImDUSTRY5-1.5K: A Public Benchmark Dataset for Industrial Object Detection in Real Production Environments.*  
2025.

## Contact

For questions regarding the dataset, annotations, or benchmark setup, please contact:

- kanan.gurbanov@ufaz.az
- l.qardashova@asoiu.edu.az
- amhaz@unistra.fr
