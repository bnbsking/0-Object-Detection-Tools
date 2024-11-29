import sys
ROOT="/Users/james.chao/Desktop/codeMore/mygithub/COSMOs"
sys.path.append(ROOT)

from cosmos.classification import ClassificationAnalysis
from cosmos.detection import DetectionAnalysis


# DetectionAnalysis(
#     ant_path = f"{ROOT}/example/detection/data/general.json",
#     save_folder = f"{ROOT}/example/detection/output/metrics"
# )


ClassificationAnalysis(
    ant_path = f"{ROOT}/example/classification/data/single_label_background.json",
    save_folder = f"{ROOT}/example/classification/output/single_label_background",
)


# ClassificationAnalysis(
#     ant_path = f"{ROOT}/example/classification/data/single_label.json",
#     save_folder = f"{ROOT}/example/classification/output/single_label",
# )


# ClassificationAnalysis(
#     ant_path = f"{ROOT}/example/classification/data/multi_label.json",
#     save_folder = f"{ROOT}/example/classification/output/multi_label",
# )