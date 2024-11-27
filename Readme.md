# COSMOs: Classification, Object detection, Segmentation MOduleS

### **Introduction**
This repo provides tools for common computer vision tasks.
+ Classification
    + 
+ Object detection
    + `Format Conversion` between coco, voc, yolo and self-defined format [general](./example/detection/data/gt_and_pd.json).
    + `Visualization` of all the above formats for **labels** (and **predictions** if specified). The visualization tool automatically converts the data into general format in the cache directory in advance.
    + `Output Analysis`
        + Comprehensive `metrics` e.g. AP, PR-Curves, Confusion matrices, threshold optimization, etc.
        + `Plotting` of the above metrics
        + `Export` data for label correctness
+ Segmentation
    + 

### **Installation**
```bash
git clone https://github.com/bnbsking/COSMOs
pip install numpy matplotlib opencv-python
```

### **Quick start - Object detection**
+ Format Conversion (see more in the [example](./example/detection/s2_format_conversion.ipynb))

```python
from cosmos.detection import coco2any

coco2any(
    tgt_foramt = "voc",
    img_folder = "example/detection/data/coco",
    ant_path = "example/detection/data/coco/coco.json",
    save_folder = "example/detection/output/visualization_gt_conversion/coco2voc"
)
```

or 

```python
from cosmos.detection import coco2general

coco2general(
    img_folder = "example/detection/data/coco",
    ant_path = "example/detection/data/coco/coco.json",
    save_path = "example/detection/output/visualization_gt_conversion/coco2general/general.json"
)
```

+ Visualization (see more in the [example](./example/detection/s1_visualization_gt_and_pd.ipynb))

```python
from cosmos.detection import show_coco

show_coco(
    img_name = f"pic0.jpg",
    img_folder = "example/detection/data/coco",
    ant_path = "example/detection/data/coco/coco.json"
)
```

or

```python
from cosmos.detection import show_general

show_general(
    img_name = f"pic0.jpg",
    ant_path = "example/detection/data/gt_and_pd.json",
)  # when the anntotation includes predictions it will be shown!
```

+ Output Analysis
```python
from cosmos.detection import DetectionAnalysis

DetectionAnalysis(
    ant_path = f"example/detection/data/gt_and_pd.json",
    save_folder = f"example/detection/output/metrics"
)
```

### **Concepts**
+ Inconvenient formats in object detection

The formats can be summarized as following:
| format | extension | files     | type  | box                      | disadvantage |
| -      | -         | -         | -     | -                        | -            |
| coco   | .json     | 1         | int   | (xmin, ymin, w, h)       | get label of an image |
| yolo   | .txt      | len(imgs) | float | (cx, cy, w/2, h/2)       | visualization, compute metrics, etc. |
| voc    | .xml      | len(imgs) | int   | (xmin, ymin, xmax, ymax) | get class list |
| [general](./example/detection/data/gt_and_pd.json)| .json | 1 | int | (xmin, ymin, w, h) | **NO** |