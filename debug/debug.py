import sys
ROOT="/home/james/Desktop/mygithub/COSMOs"
sys.path.append(ROOT)

from cosmos.classification import ClassificationAnalysis
from cosmos.detection import DetectionAnalysis
from cosmos.segmentation.format_conversion import coco_to_npy
from cosmos.segmentation.visualization import show_general


# DetectionAnalysis(
#     ant_path = f"{ROOT}/example/detection/data/general.json",
#     save_folder = f"{ROOT}/example/detection/output/metrics"
# )


# ClassificationAnalysis(
#     ant_path = f"{ROOT}/example/classification/data/single_label_background.json",
#     save_folder = f"{ROOT}/example/classification/output/single_label_background",
# )


# ClassificationAnalysis(
#     ant_path = f"{ROOT}/example/classification/data/single_label.json",
#     save_folder = f"{ROOT}/example/classification/output/single_label",
# )


# ClassificationAnalysis(
#     ant_path = f"{ROOT}/example/classification/data/multi_label.json",
#     save_folder = f"{ROOT}/example/classification/output/multi_label",
# )

coco_to_npy(
    f"{ROOT}/example/segmentation/data/instance_coco",
    f"{ROOT}/example/segmentation/data/instance_coco/coco.json",
    f"{ROOT}/example/segmentation/data/instance_general"
)

# show_npy(
#     f"{ROOT}/example/segmentation/data/instance_general/img1.jpg",
#     f"{ROOT}/example/segmentation/data/instance_general/contour_img1.npy",
#     ["c0", "c1"],
#     f"{ROOT}/example/segmentation/output/vis/exp1.jpg"
# )

# show_npy(
#     f"{ROOT}/example/segmentation/data/instance_general/img1.jpg",
#     f"{ROOT}/example/segmentation/data/instance_general/filled_img1.npy",
#     ["c0", "c1"],
#     f"{ROOT}/example/segmentation/output/vis/exp2.jpg"
# )

show_general(
    f"{ROOT}/example/segmentation/data/instance_general",
    "img1.jpg",
    f"{ROOT}/example/segmentation/output/visualization"
)