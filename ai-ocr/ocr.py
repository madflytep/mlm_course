import os
from pathlib import Path

import cv2
import numpy as np
from easyocr import Reader
from PIL import Image, ImageDraw
from ultralytics import YOLO


def yolo_to_easyocr_format(results) -> tuple[list, list, list]:
    horizontal_list = []
    free_list = [] # Keep it emp just for backward compatibility
    scores_list = list(results[0].boxes.conf.cpu().numpy()) # Detection scores
    for r in results[0].boxes:
        # Get coordinates in x1,y1,x2,y2 format
        x1, y1, x2, y2 = r.xyxy[0].cpu().numpy()
        # Convert to EasyOCR horizontal format [x_min, x_max, y_min, y_max]
        horizontal_box = [int(x1), int(x2), int(y1), int(y2)]
        horizontal_list.append(horizontal_box)
    return horizontal_list, free_list, scores_list


class CompositeOCR(object):
    """
    Class for custom integration of YOLO detector and EasyOCR recognizer.
    """
    def __init__(self, detector: YOLO, recognizer: Reader):
        self.detector = detector
        self.recognizer = recognizer
        return

    def forward(self, image: str| Path | np.ndarray, **kwargs) -> list[tuple]:
        """
        Read text from given image. 

        Backward compatible with easyocr.Reader.readtext method.
        """
        det_results = self.detector(image, verbose=False)
        horizontal_list, free_list, det_scores_list = yolo_to_easyocr_format(det_results)
        output = self.recognizer.recognize(
            image,
            horizontal_list=horizontal_list,
            free_list=free_list,
            **kwargs
        )
        results = [] # Final results with both recognition and detection scores
        for elem, det_score in zip(output, det_scores_list):
            elem_with_det_score = tuple(list(elem) + [det_score])
            results.append(elem_with_det_score)
        return results

    def readtext(self, image: str| Path | np.ndarray, **kwargs) -> list[tuple]:
        """An alias for 'forward' method."""
        return self.forward(image, **kwargs)


def sort_boxes(boxes, texts, recognition_scores, detection_scores, image_width, percentile=15):
    
    if len(boxes) == 0:
        return boxes, texts, recognition_scores, detection_scores
    
    # Find the minimux box height
    # boxes are like [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    heights = np.array([max(xy[1] for xy in box) - min(xy[1] for xy in box)
                        for box in boxes])
    tolerance = np.percentile(heights, percentile) + 1
    
    # Calculate ranks for the boxes
    ranks = []
    rough_image_width = image_width // tolerance
    for box in boxes:
        x_center = sum(xy[0] for xy in box) / 4
        y_center = sum(xy[1] for xy in box) / 4
        ranks.append(
            y_center * rough_image_width + x_center
        )
    
    # Sort the boxes by the ranks
    regions = zip(boxes, texts, recognition_scores, detection_scores, ranks)
    regions_sorted = sorted(regions, key=lambda x: x[4])
    
    # Unzip and return
    boxes_sorted, texts_sorted, recognition_scores_sorted, detection_scores_sorted, _ = zip(*regions_sorted)
    return boxes_sorted, texts_sorted, recognition_scores_sorted, detection_scores_sorted


class EasyOCRModel(object):
    def __init__(
        self,
        languages: list[str] = None,
        gpu: bool = False,
        custom_model_dir_path: str | None = None,
        custom_recognition_model_name: str | None = None,
        custom_detection_model_name_or_path: str | Path = None,
    ):
        """
        Creates default or custom EasyOCR detection+recognition model.
        For info about custom models, refer to https://github.com/JaidedAI/EasyOCR/blob/master/custom_model.md
        Args:
            languages (list, optional): Language codes (ISO 639) for languages to be recognized during analysis.
            gpu (bool, optional): Enable GPU support.
            custom_model_dir_path (str | None, optional): Path to custom model directory (with "model" and "user_network" subdirs).
            custom_recognition_model_name (str | None, optional): Filename of custom recognition model.
            custom_detection_model_name (str | None, optional): Path to custom detection model. Currently only ultralytics.YOLO is supported.
        """
        languages = languages or ["en", "ru"]
        recog_network = custom_recognition_model_name or "standard"
        if custom_model_dir_path is None:
            model_storage_directory = None
            user_network_directory = None
        else:
            model_storage_directory = os.path.join(custom_model_dir_path, "model")
            user_network_directory = os.path.join(custom_model_dir_path, "user_network")
            assert os.path.isdir(
                model_storage_directory
            ), f"Dir not found: {model_storage_directory=}; {os.getcwd()=}"
            assert os.path.isdir(
                user_network_directory
            ), f"Dir not found: {user_network_directory=}; {os.getcwd()=}"
        if custom_detection_model_name_or_path is not None:
            self.detector = YOLO(custom_detection_model_name_or_path)
            self.recognizer = Reader(
                languages,
                gpu=gpu,
                model_storage_directory=model_storage_directory,
                user_network_directory=user_network_directory,
                recog_network=recog_network,
            )
            self.reader = CompositeOCR(self.detector, self.recognizer)
        else:
            self.detector = None
            self.recognizer = None
            self.reader = Reader(
                languages,
                gpu=gpu,
                model_storage_directory=model_storage_directory,
                user_network_directory=user_network_directory,
                recog_network=recog_network,
            )
        return

    # TODO: rename to img_path_or_nparray
    def read_text(self, image_path):
        result = self.reader.readtext(image_path)
        return result

    @staticmethod
    def draw_boxes(img_path_or_nparray, boxes, color="green", thickness=2):
        """
        Draw bounding boxes on the image using PIL.

        Parameters:
        - img_path_or_nparray: Path to the image or a numpy array of the image.
        - boxes: List of bounding boxes to draw. Each box is a list of four points.
        - color: The color of the bounding box. Default is green.
        - thickness: The thickness of the bounding box. Default is 2.

        Returns:
        - image_with_boxes: Image with bounding boxes drawn.
        """
        if isinstance(img_path_or_nparray, str):
            image = Image.open(img_path_or_nparray)
        else:
            image = Image.fromarray(img_path_or_nparray)

        draw = ImageDraw.Draw(image)

        for box in boxes:
            # Draw the bounding box
            draw.line(
                [
                    tuple(box[0]),
                    tuple(box[1]),
                    tuple(box[2]),
                    tuple(box[3]),
                    tuple(box[0]),
                ],
                fill=color,
                width=thickness,
            )

        return image

    def get_img_array(self, img_path: str) -> np.array:
        image = cv2.imread(img_path)
        return image