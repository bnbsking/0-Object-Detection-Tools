import sys
ROOT="/home/james/Desktop/mygithub/COSMOs"
sys.path.append(ROOT)

from cosmos.classification import ClassificationAnalysis
from cosmos.detection import DetectionAnalysis
from cosmos.segmentation.format_conversion import coco_to_general
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

# coco_to_general(
#     f"{ROOT}/example/segmentation/data/instance_coco",
#     f"{ROOT}/example/segmentation/data/instance_coco/coco.json",
#     "instance",
#     f"{ROOT}/example/segmentation/data/instance_general"
# )

# coco_to_general(
#     f"{ROOT}/example/segmentation/data/instance_coco",
#     f"{ROOT}/example/segmentation/data/instance_coco/coco.json",
#     "semantic",
#     f"{ROOT}/example/segmentation/data/semantic_general"
# )

show_general(
    f"{ROOT}/example/segmentation/data/instance_general",
    "img1.jpg",
    "instance",
    f"{ROOT}/example/segmentation/output/visualization/instance"
)

show_general(
    f"{ROOT}/example/segmentation/data/instance_general",
    "img1.jpg",
    "instance",
    f"{ROOT}/example/segmentation/output/visualization/semantic"
)