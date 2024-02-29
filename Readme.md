### Introduction
This repo integrates the common useful tools for object detection.

### Quickstart
+  Visualization
    ```python
    import visualization as vis

    # args of vis.show
    # 1 class_list: list[str].
    # 2 img_path: str.
    # 3 ant_path: str or None.
    # 4 pd_boxes_type: '' or 'voc' or 'yolo' or 'yolo_int' or 'coco'
    # 5 pd_boxes: ndarray in shape (N,4) coco/voc/yolo
    # 6 pd_cids: ndarray in shape (N,) class index
    # 7 pd_cfs: ndarray in shape (N,) confidence

    # show img with coco annotation and coco prediction
    vis.show(["dog","cat"], f"data/coco/pic0.jpg", "data/coco/coco.json", "coco", pd_boxes, pd_cids, pd_cfs)

    # show img with voc annotation and voc prediction
    vis.show(["dog","cat"], f"data/voc/pic0.jpg", "data/voc/pic0.xml", "voc", pd_boxes, pd_cids, pd_cfs)

    # show img with coco annotation and coco prediction
    vis.show(["dog","cat"], f"data/coco/pic0.jpg", "data/yolo/pic0.txt", "yolo", pd_boxes, pd_cids, pd_cfs)
    ```

+ Format Conversion
    ```bash
    cd format_conversion
    ```
    ```python
    import os
    
    import convert
    
    # convert voc to yolo
    os.makedirs("output/voc2yolo", exist_ok=True)
    convert.voc2yolo("data/voc", "output/voc2yolo", ['dog', 'cat'])
    
    # convert voc to coco
    os.makedirs("output/voc2coco", exist_ok=True)
    convert.voc2coco("data/voc", "output/voc2coco/coco.json", ['dog', 'cat'])

    # convert yolo to voc
    os.makedirs("output/yolo2voc", exist_ok=True)
    convert.yolo2voc("data/yolo", "output/yolo2voc", ['dog', 'cat'])

    # convert yolo to coco
    os.makedirs("output/yolo2coco", exist_ok=True)
    convert.yolo2coco("data/yolo", f"output/yolo2coco/coco.json", ['dog','cat'])

    # convert coco to voc
    os.makedirs("output/coco2voc", exist_ok=True)
    convert.coco2voc("data/coco/coco.json", "output/coco2voc")

    # convert coco to yolo
    os.makedirs("output/coco2yolo", exist_ok=True)
    convert.coco2yolo("data/coco/coco.json", "output/coco2yolo")
    ```
    See more details and examples in `format_conversion/`

### Output examples

+ Visualization
![a](visualization/image.jpg)
