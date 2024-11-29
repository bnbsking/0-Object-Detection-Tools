import json
import os
import re
import shutil
from tqdm import tqdm
from typing import List, Literal, Optional, Union

import cv2


class VOCComponents:
    def get_xml(filename: str, path: str, width: int, height: int):
        return f"\
<annotation>\n\
    <folder>folder</folder>\n\
    <filename>{filename}</filename>\n\
    <path>{path}</path>\n\
    <source>\n\
        <database>Unknown</database>\n\
    </source>\n\
    <size>\n\
        <width>{width}</width>\n\
        <height>{height}</height>\n\
        <depth>3</depth>\n\
    </size>\n\
    <segmented>0</segmented>\n"
    
    def get_obj(name: str, xmin: int, ymin: int, xmax: int, ymax: int):
        return f"\
    <object>\n\
        <name>{name}</name>\n\
        <pose>Unspecified</pose>\n\
        <truncated>0</truncated>\n\
        <difficult>0</difficult>\n\
        <bndbox>\n\
            <xmin>{xmin}</xmin>\n\
            <ymin>{ymin}</ymin>\n\
            <xmax>{xmax}</xmax>\n\
            <ymax>{ymax}</ymax>\n\
        </bndbox>\n\
    </object>\n"

    def get_end():
        return "</annotation>"

class BoxConvert:
    def any2voc(
            src_type: Literal["voc", "yolo", "coco"],
            b1: Union[int, float],
            b2: Union[int, float],
            b3: Union[int, float],
            b4: Union[int, float],
            width_for_yolo: Optional[int] = None,
            height_for_yolo: Optional[int] = None,
        ):
        if src_type=="voc":  # b1, b2, b3, b4 = xmin, ymin, xmax, ymax
            xmin, ymin, xmax, ymax = int(b1), int(b2), int(b3), int(b4)
        elif src_type=="yolo":  # b1, b2, b3, b4 = cx, cy, w, h
            xmin = int((float(b1) - float(b3) / 2) * float(width_for_yolo))
            ymin = int((float(b2) - float(b4) / 2) * float(height_for_yolo))
            xmax = int((float(b1) + float(b3) / 2) * float(width_for_yolo))
            ymax = int((float(b2) + float(b4) / 2) * float(height_for_yolo))
        elif src_type=="coco":  # b1, b2, b3, b4 = xmin, ymin, w, h
            xmin, ymin, xmax, ymax = int(b1), int(b2), int(b1)+int(b3), int(b2)+int(b4)
        else:
            raise KeyError(f"{src_type} Not found")
        return xmin, ymin, xmax, ymax

    def voc2any(
            des_type: Literal["voc", "yolo", "coco"],
            xmin: int,
            ymin: int,
            xmax: int,
            ymax: int,
            width_for_yolo: Optional[int] = None,
            height_for_yolo: Optional[int] = None,
        ):
        if des_type=="voc":
            return int(xmin), int(ymin), int(xmax), int(ymax)
        elif des_type=="yolo":
            cx = round((int(xmin) + int(xmax)) / 2 / float(width_for_yolo), 6)
            cy = round((int(ymin) + int(ymax)) / 2 / float(height_for_yolo), 6)
            w  = round((int(xmax) - int(xmin)) / float(width_for_yolo), 6)
            h  = round((int(ymax) - int(ymin)) / float(height_for_yolo), 6)
            return cx, cy, w, h
        elif des_type=="coco":
            xmin = int(xmin)
            ymin = int(ymin)
            w    = int(xmax) - int(xmin)
            h    = int(ymax) - int(ymin)
            return xmin, ymin, w, h
        else:
            raise KeyError(f"{des_type} Not found")


class FormatConvertAny2General:
    def voc2general(
            img_path_list: List[str],
            ant_path_list: List[str],
            class_list: List[str],
            save_path: str,
        ):
        """
        Convert label from voc to general format.
        Args:
            img_path_list (List[str]): list of image path of the dataset
            ant_path_list (List[str]): list of annotation path of the dataset
            class_list (List[str]): list of class name
            save_path (str): target save path
        """
        # initialization
        out = {"categories": class_list, "data": []}
        class_list.insert(0, "__background__") if class_list[0] != "__background__" else None

        for img_path, ant_path in tqdm(zip(img_path_list, ant_path_list)):
            # extract
            with open(ant_path, "r", encoding="utf-8") as f:
                xml = f.read()
            img_width  = int(re.findall("<width>([0-9]*)</width>", xml)[0])
            img_height = int(re.findall("<height>([0-9]*)</height>", xml)[0])
            class_name_list = re.findall("<name>(.*)</name>", xml)
            gt_cls = [class_list.index(class_name) for class_name in class_name_list]
            xmin_list = re.findall("<xmin>(.*)</xmin>", xml)
            ymin_list = re.findall("<ymin>(.*)</ymin>", xml)
            xmax_list = re.findall("<xmax>(.*)</xmax>", xml)
            ymax_list = re.findall("<ymax>(.*)</ymax>", xml)
            gt_boxes = [
                [int(xmin), int(ymin), int(xmax), int(ymax)]
                for xmin, ymin, xmax, ymax in zip(xmin_list, ymin_list, xmax_list, ymax_list)
            ]

            # collect
            out["data"].append(
                {
                    "img_path": os.path.abspath(img_path),
                    "img_width": img_width,
                    "img_height": img_height,
                    "gt_boxes": gt_boxes,
                    "gt_cls": gt_cls,
                    "pd_boxes": [],
                    "pd_probs": [],
                }
            )

        # save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4)

    def yolo2general(
            img_path_list: List[str],
            ant_path_list: List[str],
            class_list: List[str],
            save_path: str,
            cat_index_start: int = 0
        ):
        """
        Convert label from yolo to general format.
        Args:
            img_path_list (List[str]): list of image path of the dataset
            ant_path_list (List[str]): list of annotation path of the dataset
            class_list (List[str]): list of class name
            save_path (str): target save path
            index_start (int): category index start from
        Notes:
            yolo class index usually starts from 0, this function will shift to 1.
        """
        # initialization
        out = {"categories": class_list, "data": []}
        class_list.insert(0, "__background__") if class_list[0] != "__background__" else None

        for img_path, ant_path in tqdm(zip(img_path_list, ant_path_list)):
            # extract
            img_height, img_width, _ = cv2.imread(img_path).shape
            with open(ant_path, "r", encoding="utf-8") as f:
                txt_lines = f.readlines()
            gt_boxes = []
            gt_cls = []
            for txt_line in txt_lines:
                if not txt_line.strip():
                    continue
                cid, cx, cy, w, h = txt_line.split(" ")
                xmin, ymin, xmax, ymax = BoxConvert.any2voc("yolo", cx, cy, w, h, img_width, img_height)
                gt_boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
                gt_cls.append(int(cid) - cat_index_start + 1)

            # collect
            out["data"].append(
                {
                    "img_path": os.path.abspath(img_path),
                    "img_width": img_width,
                    "img_height": img_height,
                    "gt_boxes": gt_boxes,
                    "gt_cls": gt_cls,
                    "pd_boxes": [],
                    "pd_probs": [],
                }
            )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4)

    def coco2general(
            img_folder: str,
            ant_path: str,
            save_path: str,
            cat_index_start: int = 1
        ):
        """
        Convert label from coco to general format.
        Args:
            img_folder (str): source folder of the images
            ant_path (str): source path of the label
            save_path (str): target save path
        Notes:
            coco class index usually starts from 1, this function remains.
        """
        # initialization
        with open(ant_path, "r", encoding="utf-8") as f:
            coco = json.load(f)
            class_list = ["__background__"] * (max(int(cat_dict["id"]) for cat_dict in coco['categories']) + 1)
            for cat_dict in coco['categories']:
                class_list[cat_dict["id"]] = cat_dict["name"]
        out = {"categories": class_list, "data": []}

        for img_dict in tqdm(coco['images']):
            # extract
            img_path = os.path.join(img_folder, img_dict['file_name'])
            img_height = img_dict['height']
            img_width = img_dict["width"]
            gt_boxes = []
            gt_cls = []
            for ant_dict in filter(lambda ant_dict: ant_dict['image_id']==img_dict['id'], coco['annotations']):
                xmin, ymin, w, h = ant_dict['bbox']
                xmin, ymin, xmax, ymax = BoxConvert.any2voc("coco", xmin, ymin, w, h)
                gt_boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
                gt_cls.append(int(ant_dict['category_id']) - cat_index_start + 1)
            
            # collect
            out["data"].append(
                {
                    "img_path": os.path.abspath(img_path),
                    "img_width": img_width,
                    "img_height": img_height,
                    "gt_boxes": gt_boxes,
                    "gt_cls": gt_cls,
                    "pd_boxes": [],
                    "pd_probs": [],
                }
            )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4)


class FormatConvertGeneral2Any:
    def general2voc(ant_path: str, save_folder: str):
        # initialization
        with open(ant_path, "r", encoding="utf-8") as f:
            general = json.load(f)
            categories = general["categories"]
            data = general["data"]
        
        vocc = VOCComponents
        os.makedirs(save_folder, exist_ok=True)
        for data_dict in data:
            # collect
            filename = os.path.basename(data_dict["img_path"])
            out = vocc.get_xml(
                filename,
                data_dict["img_path"],
                data_dict["img_width"],
                data_dict["img_height"]
            )
            for (xmin, ymin, xmax, ymax), gt_cls in zip(data_dict["gt_boxes"], data_dict["gt_cls"]):
                out += vocc.get_obj(categories[gt_cls], xmin, ymin, xmax, ymax)
            out += vocc.get_end()
            
            # save
            save_path = os.path.join(save_folder, f"{filename.split('.')[0]}.xml")
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(out)
            shutil.copy(data_dict["img_path"], save_folder)

    def general2yolo(ant_path: str, save_folder: str, cat_index_start: int = 0):
        # initialization
        with open(ant_path, "r", encoding="utf-8") as f:
            general = json.load(f)
            categories = general["categories"]
            data = general["data"]

        pad = lambda s: str(s) + '0'*(8-len(str(s)))
        os.makedirs(save_folder, exist_ok=True)
        for data_dict in data:
            # collect
            filename = os.path.basename(data_dict["img_path"])
            width = data_dict["img_width"]
            height = data_dict["img_height"]
            out = ""
            for (xmin, ymin, xmax, ymax), gt_cls in zip(data_dict["gt_boxes"], data_dict["gt_cls"]):
                cx, cy, w, h = BoxConvert.voc2any("yolo", xmin, ymin, xmax, ymax, width, height)
                out += f"{gt_cls - 1 + cat_index_start} {pad(cx)} {pad(cy)} {pad(w)} {pad(h)}\n"

            # save
            save_path = os.path.join(save_folder, f"{filename.split('.')[0]}.txt")
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(out)
            shutil.copy(data_dict["img_path"], save_folder)
        
        with open(os.path.join(save_folder, "classes.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(categories[1:]))

    def general2coco(ant_path: str, save_folder: str, cat_index_start: int = 1):
        # initialization
        with open(ant_path, "r", encoding="utf-8") as f:
            general = json.load(f)
            categories = general["categories"]
            data = general["data"]
        out = {
            "images":[],
            "annotations":[],
            "categories": [
                {
                    "supercategory": "none",
                    "id": i,
                    "name": class_name
                } for i, class_name in enumerate(categories, cat_index_start)
            ]
        }
        
        os.makedirs(save_folder, exist_ok=True)
        total_labels = 0
        for img_id, data_dict in enumerate(data):
            out["images"].append(
                {
                    "file_name": os.path.basename(data_dict["img_path"]),
                    "width": data_dict["img_width"],
                    "height": data_dict["img_height"],
                    "id": img_id
                }
            )
            for (xmin, ymin, xmax, ymax), gt_cls in zip(data_dict["gt_boxes"], data_dict["gt_cls"]):
                xmin, ymin, w, h = BoxConvert.voc2any("coco", xmin, ymin, xmax, ymax)
                out["annotations"].append(
                    {
                        "area": w * h,
                        "iscrowd": 0,
                        "bbox": [xmin, ymin, w, h],
                        "category_id": gt_cls - 1 + cat_index_start,
                        "ignore": 0,
                        "segmentation": [],
                        "image_id": img_id,
                        "id": total_labels
                    }
                )
                total_labels += 1
            shutil.copy(data_dict["img_path"], save_folder)

        save_path = os.path.join(save_folder, "coco.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4)


class FormatConvertAny2Any:
    def voc2any(
            tgt_foramt: Literal["voc", "yolo", "coco"],
            img_path_list: List[str],
            ant_path_list: List[str],
            class_list: List[str],
            save_folder: str,
        ):
        """
        Convert label from voc to other format.
        Args:
            tgt_foramt (str): target converted format
            img_path_list (List[str]): list of image path of the dataset
            ant_path_list (List[str]): list of annotation path of the dataset
            class_list (List[str]): list of class name
            save_folder (str): target save folder
        """
        FormatConvertAny2General.voc2general(
            img_path_list = img_path_list,
            ant_path_list = ant_path_list,
            class_list = class_list,
            save_path = os.path.join(save_folder, ".tmp_general.json")
        )
        getattr(FormatConvertGeneral2Any, f"general2{tgt_foramt}")(
            ant_path = os.path.join(save_folder, ".tmp_general.json"),
            save_folder = save_folder
        )
    
    def yolo2any(
            tgt_foramt: Literal["voc", "yolo", "coco"],
            img_path_list: List[str],
            ant_path_list: List[str],
            class_list: List[str],
            save_folder: str,
        ):
        """
        Convert label from yolo to other format.
        Args:
            tgt_foramt (str): target converted format
            img_path_list (List[str]): list of image path of the dataset
            ant_path_list (List[str]): list of annotation path of the dataset
            class_list (List[str]): list of class name
            save_folder (str): target save folder
        """
        FormatConvertAny2General.yolo2general(
            img_path_list = img_path_list,
            ant_path_list = ant_path_list,
            class_list = class_list,
            save_path = os.path.join(save_folder, ".tmp_general.json")
        )
        getattr(FormatConvertGeneral2Any, f"general2{tgt_foramt}")(
            ant_path = os.path.join(save_folder, ".tmp_general.json"),
            save_folder = save_folder
        )
    
    def coco2any(
            tgt_foramt: Literal["voc", "yolo", "coco"],
            img_folder: str,
            ant_path: str,
            save_folder: str,
        ):
        """
        Convert label from coco to other format.
        Args:
            tgt_foramt (str): target converted format
            img_folder (str): source folder of the images
            ant_path (str): source path of the label
            save_folder (str): target save folder
        """
        FormatConvertAny2General.coco2general(
            img_folder = img_folder,
            ant_path = ant_path,
            save_path = os.path.join(save_folder, ".tmp_general.json")
        )
        getattr(FormatConvertGeneral2Any, f"general2{tgt_foramt}")(
            ant_path = os.path.join(save_folder, ".tmp_general.json"),
            save_folder = save_folder
        )
