### File structure
+ format_conversion/
    + convert.py
        + voc2yolo (src_folder, dst_folder, class_list)
        + voc2coco (src_folder, dst_path,   class_list)
        + yolo2voc (src_folder, dst_folder, class_list)
        + coco2voc (src_path,   dst_folder)
        + yolo2coco(src_folder, dst_path,   class_list)
        + coco2yolo(src_path,   dst_folder)
    + (img) pic*.jpg
    + (label) pic*.txt, classes.txt | pic*.xml | coco.json

+ visualization.py
    + show(class_list, img_path, ant_path="", \
        pd_boxes_type=None, pd_boxes=None, pd_cids=None, pd_cfs=None, \
        save_path="", box_width=4, valueRatios=(1,1))

+ example.ipynb
    + visualization & format_conversion