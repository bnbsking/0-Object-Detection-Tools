import sys
ROOT="/Users/james.chao/Desktop/codeMore/mygithub/COSMOs"
sys.path.append(ROOT)

from cosmos.classification import ClassificationAnalysis
from cosmos.detection import DetectionAnalysis
from cosmos.segmentation.format_conversion import coco_to_npy
from cosmos.segmentation.visualization import show_npy


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

# coco_to_npy(
#     "/Users/james.chao/Desktop/codeMore/mygithub/COSMOs/example/segmentation/data/instance_coco",
#     "/Users/james.chao/Desktop/codeMore/mygithub/COSMOs/example/segmentation/data/instance_coco/coco.json",
#     "/Users/james.chao/Desktop/codeMore/mygithub/COSMOs/example/segmentation/data/instance_npy"
# )

show_npy(
    "/Users/james.chao/Desktop/codeMore/mygithub/COSMOs/example/segmentation/data/instance_npy/img1.jpg",
    "/Users/james.chao/Desktop/codeMore/mygithub/COSMOs/example/segmentation/data/instance_npy/img1.npy",
    ["c0", "c1"],
    "/Users/james.chao/Desktop/codeMore/mygithub/COSMOs/example/segmentation/output/vis/exp.jpg"
)