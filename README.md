# COSMOduleS: Classification, Object detection, Segmentation MOduleS

<div align="center">
  <a href="https://pypi.python.org/pypi/cosmodules"><img src="https://img.shields.io/pypi/v/cosmodules.svg"></a><br>
  <a href="https://pypi.org/project/cosmodules"><img src="https://img.shields.io/pypi/pyversions/cosmodules.svg"></a><br>
  Github: <a href="https://github.com/bnbsking/COSMOduleS">source</a> <a href="https://github.com/bnbsking/COSMOduleS"><img src="https://github.com/bnbsking/COSMOduleS/blob/main/pictures/github-mark-white.png" width="20" height="20"></a>
  <!--<a href="https://github.com/bnbsking/COSMOduleS"><img src="https://img.shields.io/github/v/release/bnbsking/COSMOduleS"></a><br>-->
</div>

## **Introduction**
This repo provides comprehensive preprocessing and post-processing tools for common **Computer Vision** tasks.


| Tasks | Subtasks | Defined<br>Format | Visualization | Format<br>Conversion | Output<br>Analysis | Label<br>Merging | Active<br>Learning |
| - | - | - | - | - | - | - | - |
| Classification | binary<sup>1</sup><br> binary-bg<sup>2</sup><br> multi-class<sup>1</sup><br> multiclass-bg<sup>2</sup><br> multi-binary<sup>3</sup><br> | [single_label](https://github.com/bnbsking/COSMOduleS/blob/main/example/classification/data/single_label.json)<sup>1</sup><br> [single_label_bg](https://github.com/bnbsking/COSMOduleS/blob/main/example/classification/data/single_label_background.json)<sup>2</sup><br> [multi_label](https://github.com/bnbsking/COSMOduleS/blob/main/example/classification/data/multi_label.json)<sup>3</sup> | - | - | [metrics<br>plotting<br>export](https://github.com/bnbsking/COSMOduleS/blob/main/example/classification/s3_output_analysis.ipynb) | [ALL](https://github.com/bnbsking/COSMOduleS/blob/main/example/classification/s4_label_merging.ipynb) | [Entropy](https://github.com/bnbsking/COSMOduleS/blob/main/example/classification/s5_active_learning.ipynb) |
| Detection      | - | [coco](https://github.com/bnbsking/COSMOduleS/blob/main/example/detection/data/coco)<br> [voc](https://github.com/bnbsking/COSMOduleS/blob/main/example/detection/data/voc)<br> [yolo](https://github.com/bnbsking/COSMOduleS/blob/main/example/detection/data/yolo)<br> [**GENERAL**](https://github.com/bnbsking/COSMOduleS/blob/main/example/detection/data/general.json)<br> | [ALL](https://github.com/bnbsking/COSMOduleS/blob/main/example/detection/s1_visualization_gt_and_pd.ipynb) | [between ANY<br>two types](https://github.com/bnbsking/COSMOduleS/blob/main/example/detection/s2_format_conversion.ipynb) | [metrics<br>plotting<br>export](https://github.com/bnbsking/COSMOduleS/blob/main/example/detection/s3_output_analysis.ipynb) | [V](https://github.com/bnbsking/COSMOduleS/blob/main/example/detection/s4_label_merging.ipynb) | [horizontal<br>flip](https://github.com/bnbsking/COSMOduleS/blob/main/example/detection/s5_active_learning.ipynb) |
| Segmentation   | instance<sup>1</sup><br> semantic<sup>2</sup><br> | [coco](https://github.com/bnbsking/COSMOduleS/blob/main/example/segmentation/data/coco)<sup>1+2</sup><br> [**GENERAL**](https://github.com/bnbsking/COSMOduleS/blob/main/example/segmentation/data/general)<sup>1+2</sup> | [ALL](https://github.com/bnbsking/COSMOduleS/blob/main/example/segmentation/s1_visualization_gt_and_pd.ipynb) | [coco2general](https://github.com/bnbsking/COSMOduleS/blob/main/example/segmentation/s2_format_conversion.ipynb) | [metrics<br>plotting<br>export](https://github.com/bnbsking/COSMOduleS/blob/main/example/segmentation/s3_output_analysis.ipynb) | - | [instance<br>semantic<br>](https://github.com/bnbsking/COSMOduleS/blob/main/example/segmentation/s5_active_learning.ipynb) |


+ "bg" means background. If there is background class, it must be class 0 in this repo.
+ Adding prediction results after the defined format can use the visualization and output analysis. All the formats with predictions are in `example/*/prediction`, e.g. [here](https://github.com/bnbsking/COSMOduleS/blob/main/example/detection/prediction/general.json).


## **Motivation**

+ **[Classification]** Complicated tasks

| task                                          | label idx min | compute class-0 metrics | threshold optimization | data format |  
| -                                             | -             | -                       | -                      | -           |
| binary classification                         | 0             | V                       | V                      | [single_label](https://github.com/bnbsking/COSMOduleS/blob/main/example/classification/data/single_label.json)       |
| binary classification (cls-0 background)      | 1             |                         | V                      | [single_label_background](https://github.com/bnbsking/COSMOduleS/blob/main/example/classification/data/single_label_background.json) |
| multi-class classification                    | 0             | V                       |                        | single_label |
| multi-class classification (cls-0 background) | 1             |                         | V                      | single_label_background |
| multi-label classification (cls-0 background) | 0             |                         | V                      | [multi_label](https://github.com/bnbsking/COSMOduleS/blob/main/example/classification/data/multi_label.json)   |


+ **[Classification]** threshold optimization
    + `multi-class classification (cls-0 background)` checks whether prob-cls-0 < threshold, if yes, the pd-cls is pd[1:].argmax()
    + `multi-class classification (cls-0 background)` and `multi-label classification (cls-0 background)` take the mean of all optimized threshold for each foreground class


+ **[Object Detection]** Develop a [**GENERAL**](https://github.com/bnbsking/COSMOduleS/blob/main/example/detection/data/general.json) format to be the most convenient.

The formats can be summarized as following:
| format | extension | files     | type  | box                      | disadvantage |
| -      | -         | -         | -     | -                        | -            |
| coco   | .json     | 1         | int   | (xmin, ymin, w, h)       | get label of an image |
| yolo   | .txt      | len(imgs) | float | (cx, cy, w/2, h/2)       | visualization, compute metrics, etc. |
| voc    | .xml      | len(imgs) | int   | (xmin, ymin, xmax, ymax) | get class list |
| general| .json | 1 | int | (xmin, ymin, xmax, ymax) | **NO** |


+ **[Segmentation]** Develop a [**GENERAL**](https://github.com/bnbsking/COSMOduleS/blob/main/example/segmentation/data/general) format to be the most convenient.


| Includes         | Content | Advantage |
| -                | -       | -         |
| general.json     | Includes every imgs: path, contour, filled and boxes with class | Searching | 
| gt_contour_*.npy | (H, W) with {0, 1, ..., num_classes} int | Plotting |
| gt_filled_*.npy  | (num_classes, H, W) with 0 or 1 int values | Compute IOU for Metrics |
| *.jpg            | Raw data | - |


+ Segmentation prediction format: `(num_classes, H, W) with 0~1 float values (probability)`. e.g. [here](https://github.com/bnbsking/COSMOduleS/blob/main/example/segmentation/prediction/instance/pd_filled_img1.npy)


## **Installation**
```bash
pip install cosmodules
```

or

```bash
git clone https://github.com/bnbsking/COSMOduleS.git
pip install -e .
```

## **Quick Start - Classification**
+ Output Analysis:
    + Please conform your data format as either of one
        + [single_label](https://github.com/bnbsking/COSMOduleS/blob/main/example/classification/data/single_label.json)
        + [multilabel](https://github.com/bnbsking/COSMOduleS/blob/main/example/classification/data/multi_label.json)
        + [single_label_background](https://github.com/bnbsking/COSMOduleS/blob/main/example/classification/data/single_label_background.json)
    + The analysis pipeline is at [here](https://github.com/bnbsking/COSMOduleS/blob/main/cosmodules/classification/output_analysis.yaml)
    + See more in the [example](https://github.com/bnbsking/COSMOduleS/blob/main/example/classification/s3_output_analysis.ipynb)

```python
from cosmodules.classification import ClassificationAnalysis

ClassificationAnalysis(
    ant_path = "example/classification/data/single_label.json",
    save_folder = "example/classification/output/single_label",
)
```

+ Label Merging:
```python
from cosmodules.classification import ClassificationLabelMerging

ClassificationLabelMerging(
    cfg_path_list = [
        "example/classification/data/single_label.json",
        "example/classification/data_another_labeler/single_label.json",
    ],
    save_path = f"example/classification/output/label_merging/single_label.json"
)
```

+ Active Learning (see more in the [example](https://github.com/bnbsking/COSMOduleS/blob/main/example/classification/s5_active_learning.ipynb)):
```python
from cosmodules.classification import ClassificationActiveLearning

ClassificationActiveLearning(
    pred_path = "example/classification/prediction/single_label.json",
    save_path = "example/classification/output/active_learning/single_label.json",
    loss_name = "entropy"
)
```

## **Quick Start - Object detection**
+ Format Conversion (see more in the [example](https://github.com/bnbsking/COSMOduleS/blob/main/example/detection/s2_format_conversion.ipynb))

```python
from cosmodules.detection import coco2any

coco2any(
    tgt_foramt = "voc",
    img_folder = "example/detection/data/coco",
    ant_path = "example/detection/data/coco/coco.json",
    save_folder = "example/detection/output/visualization_gt_conversion/coco2voc"
)
```

or 

```python
from cosmodules.detection import coco2general

coco2general(
    img_folder = "example/detection/data/coco",
    ant_path = "example/detection/data/coco/coco.json",
    save_path = "example/detection/output/visualization_gt_conversion/coco2general/general.json"
)
```

+ Visualization (see more in the [example](https://github.com/bnbsking/COSMOduleS/blob/main/example/detection/s1_visualization_gt_and_pd.ipynb))

```python
from cosmodules.detection import show_coco

show_coco(
    img_name = "pic0.jpg",
    img_folder = "example/detection/data/coco",
    ant_path = "example/detection/data/coco/coco.json"
)
```

or

```python
from cosmodules.detection import show_general

show_general(
    img_name = "pic0.jpg",
    ant_path = "example/detection/data/general.json",
)  # when the anntotation includes predictions it will be shown!
```

+ Output Analysis
    + Please use the above `Format conversion` to change data format as [general](https://github.com/bnbsking/COSMOduleS/blob/main/example/detection/data/general.json)
    + The analysis pipeline is at [here](https://github.com/bnbsking/COSMOduleS/blob/main/cosmodules/detection/output_analysis.yaml)

```python
from cosmodules.detection import DetectionAnalysis

DetectionAnalysis(
    ant_path = "example/detection/data/general.json",
    save_folder = "example/detection/output/metrics"
)
```

+ Label Merging:
```python
from cosmodules.detection import DetectionLabelMerging

DetectionLabelMerging(
    cfg_path_list = [
        "example/detection/data/general.json",
        "example/detection/data_another_labeler/general.json",
    ],
    save_path = "example/detection/output/label_merging/general.json",
    ties_handling = "union"
)
```

+ Active Learning:
```python
from cosmodules.detection import DetectionActiveLearningByHFlip

DetectionActiveLearningByHFlip(
    pred_path_1 = f"{ROOT}/example/detection/prediction/general.json",
    pred_path_2 = f"{ROOT}/example/detection/prediction/general_horizontal_flip.json",
    save_path = f"{ROOT}/example/detection/output/active_learning/general.json"
)
```

## **Quick Start - Segmentation**
+ Format Conversion (see more in the [example](https://github.com/bnbsking/COSMOduleS/blob/main/example/segmentation/s2_format_conversion.ipynb))

```python
from cosmodules.segmentation import coco2general

coco2general(
    img_folder = "example/segmentation/data/coco",
    ant_path = "example/segmentation/data/coco/coco.json",
    save_folder = f"example/segmentation/output/visualization_gt_conversion/coco2general"
)
```

+ Visualization (see more in the [example](https://github.com/bnbsking/COSMOduleS/blob/main/example/segmentation/s1_visualization_gt_and_pd.ipynb))

```python
from cosmodules.segmentation import show_coco

show_coco(
    img_name = "img1.jpg",
    img_folder = "example/segmentation/data/coco",
    ant_path = "example/segmentation/data/coco/coco.json"
)   # when the anntotation includes predictions it will be shown!
```

or

```python
from cosmodules.segmentation import show_general

show_general(
    img_name = "img1.jpg",
    ant_path = "example/segmentation/data/general/general.json"
)
```

+ Output Analysis
    + Please use the above `Format conversion` to change data format as [general](https://github.com/bnbsking/COSMOduleS/blob/main/example/segmentation/data/general)
    + The analysis pipeline is at [here](https://github.com/bnbsking/COSMOduleS/blob/main/cosmodules/segmentation/output_analysis_instance.yaml)

```python
from cosmodules.segmentation import SegmentationAnalysis

SegmentationAnalysis(
    ant_path = "example/segmentation/prediction/instance/general.json",
    save_folder = "example/segmentation/output/metrics/instance",
    task = "instance",
)
```

or

```Python
from cosmodules.segmentation import SegmentationAnalysis

SegmentationAnalysis(
    ant_path = "example/segmentation/prediction/semantic/general.json",
    save_folder = "example/segmentation/output/metrics/semantic",
    task = "semantic"
)
```

+ Active Learning:
```python
from cosmodules.segmentation import (
    InstanceSegmentationActiveLearningByHFlip,
    SemanticSegmentationActiveLearning
)

InstanceSegmentationActiveLearningByHFlip(
    pred_path_1 = "example/segmentation/prediction/instance/general.json",
    pred_path_2 = "example/segmentation/prediction/instance_horizontal_flip/general.json",
    save_path = "example/segmentation/output/active_learning/instance.json"
)
```

or

```python
SemanticSegmentationActiveLearning(
    pred_path = "example/segmentation/prediction/semantic/general.json",
    save_path = "example/segmentation/output/active_learning/semantic.json",
    loss_name = "entropy"
)
```

## **Examples**
+ **[detection]**: format conversion workflow
![.](https://github.com/bnbsking/COSMOduleS/blob/main/pictures/detection_workflow.png)

+ detection visualization
![.](https://github.com/bnbsking/COSMOduleS/blob/main/pictures/detection_visualization.jpg)

+ confusion
![.](https://github.com/bnbsking/COSMOduleS/blob/main/pictures/confusion.jpg)

+ prf curves
![.](https://github.com/bnbsking/COSMOduleS/blob/main/pictures/prf_curves.jpg)


## **More**
+ Feel free to ask if you have any question.
+ Notice not supported
    + segmentation general2coco
    + segmentation label merging


## **Acknowledgement**
+ Confusion Matrix reference [here](https://github.com/kaanakan/object_detection_confusion_matrix/blob/master/confusion_matrix.py)
