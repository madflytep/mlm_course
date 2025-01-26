from pathlib import Path

import click
import numpy as np
import pandas as pd
from clearml import Dataset
from loguru import logger
from PIL import Image

from ai_ocr.ocr import EasyOCRModel, sort_boxes
from ai_ocr.utils.sys_utils import get_cml_task


@click.command()

@click.option("--dataset_name",
              type=str,
              help="ClearML dataset name")

@click.option("--ocr_toolkit",
              type=click.Choice(["easyocr"]),
              help="OCR toolkit to use")

@click.option("--custom_model_dir",
              type=Path,
              help="Path to the custom model directory")

@click.option("--custom_model_name",  # for recognition model
              type=str,
              help="Custom recognition model name")

@click.option("--custom_detection_model_name_or_path",  # for detection model
              type=str,
              help="Custom detection model name or path")

@click.option("--early_stop", type=int,
              help="Stop after this many images have been processed")

def main(
    dataset_name,
    ocr_toolkit,
    custom_model_dir,
    custom_model_name,
    early_stop,
    custom_detection_model_name_or_path
):

    cml_task = get_cml_task(task_name="ocr_step")
    logger.info("Starting ocr_step.py")

    # Print all arguments
    logger.info(f"Dataset name: {dataset_name}")
    logger.info(f"OCR toolkit: {ocr_toolkit}")
    logger.info(f"Custom model directory: {custom_model_dir}")
    logger.info(f"Custom model name: {custom_model_name}")
    logger.info(f"Early stop: {early_stop}")
    logger.info(f"Custom detection model name or path: {custom_detection_model_name_or_path}")

    if ocr_toolkit == "easyocr":
        ocr = EasyOCRModel(
            gpu=True,
            languages=["ru"],
            custom_model_dir_path=custom_model_dir,
            custom_recognition_model_name=custom_model_name,
            custom_detection_model_name_or_path=custom_detection_model_name_or_path
        )
        logger.info("EasyOCR model has been loaded")
    else:
        raise ValueError(f"Invalid OCR toolkit: {ocr_toolkit}")
    
    dataset_path = Path(Dataset.get(dataset_name=dataset_name).get_local_copy())
    logger.info(f"Dataset local path: {dataset_path}")

    # Get the list of images to process
    labels_df = pd.read_csv(dataset_path / "val.csv")
    image_filenames = labels_df["image_filename"].tolist()
    del labels_df
    
    results = []

    for idx, image_filename in enumerate(image_filenames):

        image_path = dataset_path / "images" / image_filename
        
        if early_stop and idx >= early_stop:
            logger.info(f"Early stopping after {early_stop} images")
            break

        logger.info(f"Processing image {image_path.name}")
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        ocr_results = ocr.read_text(image_np)
        if len(ocr_results) > 0:
            if custom_detection_model_name_or_path is None:
                boxes, texts, scores = zip(*ocr_results)
            else:
                # In this case we additionally need to sort the boxes by coordinates
                boxes, texts, scores, detection_scores = zip(*ocr_results)
                boxes, texts, scores, detection_scores = sort_boxes(boxes, texts, scores, detection_scores,
                                                                    image_np.shape[1])
        else:
            boxes, texts, scores = [], [], []

        results.append(
            {
                "image_filename": image_path.name,
                "texts": texts,
                "scores": scores,
                "boxes": boxes,
            }
        )
    
    logger.info("All images have been processed")
    cml_task.upload_artifact("results", results)
    cml_task.close()
    logger.info("Finished ocr_step.py")


if __name__ == "__main__":
    main()
