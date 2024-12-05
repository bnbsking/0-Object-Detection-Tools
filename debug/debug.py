import sys
ROOT="/home/james/Desktop/mygithub/COSMOs"
sys.path.append(ROOT)

from cosmos.classification import ClassificationAnalysis
from cosmos.detection import DetectionAnalysis, show_coco
from cosmos.segmentation.format_conversion import coco2general
from cosmos.segmentation.visualization import show_general


# show_coco(
#     img_name = f"pic0.jpg",
#     img_folder = f"{ROOT}/example/detection/data/coco",
#     ant_path = f"{ROOT}/example/detection/data/coco/coco.json",
#     use_cache = False
# )


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

# coco2general(
#     f"{ROOT}/example/segmentation/data/coco",
#     f"{ROOT}/example/segmentation/data/coco/coco.json",
#     f"{ROOT}/example/segmentation/data/general"
# )

# show_general(
#     "img1.jpg",
#     f"{ROOT}/example/segmentation/data/general/general.json",
#     f"{ROOT}/example/segmentation/output/visualization/gt"
# )

show_general(
    "img1.jpg",
    f"{ROOT}/example/segmentation/prediction/instance/general.json",
    f"{ROOT}/example/segmentation/output/visualization/pd_instance"
)

show_general(
    "img1.jpg",
    f"{ROOT}/example/segmentation/prediction/semantic/general.json",
    f"{ROOT}/example/segmentation/output/visualization/pd_sementic"
)