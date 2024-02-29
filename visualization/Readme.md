### Introduction
Visualization of images with any format ground truth and predictions
+ ground truth can be: coco, yolo, voc
+ predictions can be: coco, yolo, voc

### Quick start
+ docs
    ```python
    def show(class_list, img_path, ant_path="", pd_boxes_type="", pd_boxes=None, pd_cids=None, pd_cfs=None, \
        save_path="", box_width=4, value_ratios=(1,1)): # use help(show) for more details
        """
        This function helps visualize an image with varies format of label or prediction.
        
        + class_list: list[str]. List of class names
        + img_path: str. Path to the image (e.g. *.jpg, ... etc.)
        + ant_path: str or None. Path to the annotation (e.g. *.voc, *.txt, *.json). If None, show black img only.
        
        + pd_boxes_type: str. '' or 'voc' or 'yolo' or 'yolo_int' or 'coco'. If None, show black only, (pd_boxes, pd_cids, pd_cfs) are not used.
        + pd_boxes: None or ndarray in shape (N,4)
        + pd_cids: None or ndarray in shape (N,) class index
        + pd_cfs: None or ndarray in shape (N,) confidence
        
        + save_folder: str or None. Save at the folder with same filename. If None, show only but not save.
        + box_width: int. Predicted boxes width.
        + value_ratios: (int,int). Confidence patch size ratio.
        """
    ```

+ example
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
+ pred/ # shotcut to ../_pred
    + coco_box.json
    + voc_box.json
    + yolo_box.json
+ example.ipynb # full example
+ visualization.py # shotcut to ../visulization.py