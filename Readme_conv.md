### Introduction
Format conversion between voc, coco and yolo.
See full example with visualization in `example.ipynb`.

The formats can be summarized as following:
| format | extension | files     | box | class list info | image shape info | friendly search 1 img's label |
| -      | -         | -         | -   | -               | -                | - |
| coco   | .json     | 1         | (left_top_x:int, left_top_y:int, w:int, h:int) | V | V | |
| yolo   | .txt      | len(imgs) | (center_x:float, center_y:float, half_w:float, half_w:float) | V | | V |
| voc    | .xml      | len(imgs) | (left_top_x, left_top_y:int, left_btm_x:int, left_btm_y:int) | | V | V |

### Quick Start
```python
import os

import convert
```

1. voc2yolo
    + docs
        ```python
        def voc2yolo(sourceFolder, destFolder, classL):
            """
            Convert labels from voc to yolo.

            sourceFolder: str. Path of source that contains *.jpg and *.xml.
            destFolder: str. Path of destination.
            classL: list[str]. list of class names.
            """
        ```
    + example
        ```python
        os.makedirs("output/voc2yolo", exist_ok=True)
        convert.voc2yolo("data/voc", "output/voc2yolo", ['dog', 'cat'])
        ```

2. voc2coco
    + docs
        ```python
        def voc2coco(sourceFolder, destPath, classL):
            """
            Convert labels from voc to coco.

            sourceFolder: str. Path of source that contains *.jpg and *.xml.
            destFolder: str. Path of destination.
            classL: list[str]. list of class names.
            """
        ```
    + example
        ```python
        os.makedirs("output/voc2coco", exist_ok=True)
        convert.voc2coco("data/voc", "output/voc2coco/coco.json", ['dog', 'cat'])
        ```

3. yolo2voc
    + docs
        ```python
        def yolo2voc(sourceFolder, destFolder, classL=None, defaultAspect=None):
            """
            Convert labels from yolo to voc.

            sourceFolder: str. Path of source that contains *.jpg and *.txt.
            destFolder: str. Path of destination.
            classL: list[str] or None. list of class names. If 'classes.txt' is in sourceFolder, this arg can be None.
            defaultAspect: (int,int). Can be specified if all the images have same shape, so the box can be compute without reading all images. 
            """
        ```
    + example
        ```python
        os.makedirs("output/yolo2voc", exist_ok=True)
        convert.yolo2voc("data/yolo", "output/yolo2voc", ['dog', 'cat'])
        ```

4. yolo2coco
    + docs
        ```python
        def yolo2coco(sourceFolder, destPath, classL=None, defaultAspect=None):
            """
            Convert labels from yolo to coco.

            sourceFolder: str. Path of source that contains *.jpg and *.txt.
            destFolder: str. Path of destination.
            classL: list[str] or None. list of class names. If 'classes.txt' is in sourceFolder, this arg can be None.
            defaultAspect: (int,int). Can be specified if all the images have same shape, so the box can be compute without reading all images. 
            """
        ```
    + example
        ```python
        os.makedirs("output/yolo2coco", exist_ok=True)
        convert.yolo2coco("data/yolo", f"output/yolo2coco/coco.json", ['dog','cat'])
        ```

5. coco2voc
    + docs
        ```python
        def coco2voc(sourcePath, destFolder):
            """
            Convert labels from coco to voc.

            sourcePath: str. Path of the annotation file *.json.
            destFolder: str. Path of destination.
            """
        ```
    + example
        ```python
        os.makedirs("output/coco2voc", exist_ok=True)
        convert.coco2voc("data/coco/coco.json", "output/coco2voc")
        ```

6. coco2yolo
    + docs
        ```python
        def coco2yolo(sourcePath, destFolder):
            """
            Convert labels from coco to yolo.

            sourcePath: str. Path of the annotation file *.json.
            destFolder: str. Path of destination.
            """
        ```
    + example
        ```python
        os.makedirs("output/coco2yolo", exist_ok=True)
        convert.coco2yolo("data/coco/coco.json", "output/coco2yolo")
        ```

### File structure
+ data/ # shortcut to ../_data
    + coco/
        + pic*.jpg
        + coco.json
    + voc/
        + pic*.jpg
        + pic*.xml
    + yolo/
        + pic*.jpg
        + pic*.txt
+ convert.py # main program
+ example.ipynb # full example
+ visualization.py # shotcut to ../visulization.py
