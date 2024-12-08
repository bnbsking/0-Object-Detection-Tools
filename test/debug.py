import sys
ROOT="/home/james/Desktop/mygithub/COSMOduleS"
sys.path.append(ROOT)

from cosmodules.classification import ClassificationAnalysis
from cosmodules.classification import ClassificationLabelMerging
from cosmodules.classification import ClassificationActiveLearning

from cosmodules.detection import DetectionAnalysis, show_coco, show_general
from cosmodules.detection import DetectionActiveLearningByHFlip
from cosmodules.detection import DetectionLabelMerging

from cosmodules.segmentation import coco2general
#from cosmodules.segmentation.visualization import show_general
from cosmodules.segmentation import SegmentationAnalysis
from cosmodules.segmentation import (
    InstanceSegmentationActiveLearningByHFlip,
    SemanticSegmentationActiveLearning
)

from cosmodules.utils.detection.augmentation import horizontal_flip


# show_coco(
#     img_name = f"pic0.jpg",
#     img_folder = f"{ROOT}/example/detection/data/coco",
#     ant_path = f"{ROOT}/example/detection/data/coco/coco.json",
#     use_cache = False
# )


# DetectionAnalysis(
#     ant_path = f"{ROOT}/example/detection/prediction/general.json",
#     save_folder = f"{ROOT}/example/detection/output/metrics"
# )


# ClassificationAnalysis(
#     ant_path = f"{ROOT}/example/classification/prediction/single_label_background.json",
#     save_folder = f"{ROOT}/example/classification/output/single_label_background",
# )


# ClassificationAnalysis(
#     ant_path = f"{ROOT}/example/classification/prediction/single_label.json",
#     save_folder = f"{ROOT}/example/classification/output/single_label",
# )


# ClassificationAnalysis(
#     ant_path = f"{ROOT}/example/classification/prediction/multi_label.json",
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

# show_coco(
#     "img1.jpg",
#     f"{ROOT}/example/segmentation/data/coco",
#     f"{ROOT}/example/segmentation/data/coco/coco.json"
# )

# show_general(
#     "img1.jpg",
#     f"{ROOT}/example/segmentation/data/general/general.json"
# )

# show_general(
#     "img1.jpg",
#     f"{ROOT}/example/segmentation/prediction/instance/general.json",
# )

# show_general(
#     "img1.jpg",
#     f"{ROOT}/example/segmentation/prediction/semantic/general.json",
# )

# SegmentationAnalysis(
#     ant_path = f"{ROOT}/example/segmentation/prediction/instance/general.json",
#     save_folder = f"{ROOT}/example/segmentation/output/metrics/instance",
#     task = "instance"
# )

# SegmentationAnalysis(
#     ant_path = f"{ROOT}/example/segmentation/prediction/semantic/general.json",
#     save_folder = f"{ROOT}/example/segmentation/output/metrics/semantic",
#     task = "semantic"
# )

# ClassificationLabelMerging(
#     cfg_path_list = [
#         f"{ROOT}/example/classification/data/single_label.json",
#         f"{ROOT}/example/classification/data_another_labeler/single_label.json",
#     ],
#     save_path = f"{ROOT}/example/classification/output/label_merging/single_label.json"
# )

# ClassificationLabelMerging(
#     cfg_path_list = [
#         f"{ROOT}/example/classification/data/single_label_background.json",
#         f"{ROOT}/example/classification/data_another_labeler/single_label_background.json",
#     ],
#     save_path = f"{ROOT}/example/classification/output/label_merging/single_label_background.json"
# )

# ClassificationLabelMerging(
#     cfg_path_list = [
#         f"{ROOT}/example/classification/data/multi_label.json",
#         f"{ROOT}/example/classification/data_another_labeler/multi_label.json",
#     ],
#     save_path = f"{ROOT}/example/classification/output/label_merging/multi_label.json"
# )

# ClassificationActiveLearning(
#     pred_path = f"{ROOT}/example/classification/prediction/single_label.json",
#     save_path = f"{ROOT}/example/classification/output/active_learning/single_label.json",
#     loss_name = "entropy"
# )

# ClassificationActiveLearning(
#     pred_path = f"{ROOT}/example/classification/prediction/single_label_background.json",
#     save_path = f"{ROOT}/example/classification/output/active_learning/single_label_background.json",
#     loss_name = "entropy"
# )

# ClassificationActiveLearning(
#     pred_path = f"{ROOT}/example/classification/prediction/multi_label.json",
#     save_path = f"{ROOT}/example/classification/output/active_learning/multi_label.json",
#     loss_name = "entropy"
# )

# horizontal_flip(
#     ant_path=f"{ROOT}/example/detection/prediction/general.json",
#     save_path=f"{ROOT}/example/detection/prediction/general_horizontal_flip.json"
# )

# show_general(
#     img_name = "pic0.jpg",
#     ant_path = f"{ROOT}/example/detection/prediction/general_horizontal_flip.json",
#     #save_folder = f"{ROOT}/example/detection/output/visualization/horizontal_flip"
# )

# DetectionActiveLearningByHFlip(
#     pred_path_1 = f"{ROOT}/example/detection/prediction/general.json",
#     pred_path_2 = f"{ROOT}/example/detection/prediction/general_horizontal_flip.json",
#     save_path = f"{ROOT}/example/detection/output/active_learning/general.json"
# )

# DetectionLabelMerging(
#     cfg_path_list = [
#         f"{ROOT}/example/detection/data/general.json",
#         f"{ROOT}/example/detection/data_another_labeler/general.json",
#     ],
#     save_path = f"{ROOT}/example/detection/output/label_merging/general.json",
#     ties_handling = "union"
# )

InstanceSegmentationActiveLearningByHFlip(
    pred_path_1 = f"{ROOT}/example/segmentation/prediction/instance/general.json",
    pred_path_2 = f"{ROOT}/example/segmentation/prediction/instance_horizontal_flip/general.json",
    save_path = f"{ROOT}/example/segmentation/output/active_learning/instance.json"
)

SemanticSegmentationActiveLearning(
    pred_path = f"{ROOT}/example/segmentation/prediction/semantic/general.json",
    save_path = f"{ROOT}/example/segmentation/output/active_learning/semantic.json",
    loss_name = "entropy"
)