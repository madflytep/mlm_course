"""
Functions to convert Label Studio annotations to the recognition dataset
containing small text regions and corresponding transcriptions.

The top-level function is `labelstudio_to_recognition`.
"""


import json
import random
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from typing_extensions import Literal


def labelstudio_to_recognition(
    in_paths: list[tuple[Path, Path]],
    out_dir: Path,
    country: Literal["ru", "ua"],
    validation_size: float = 0.2,
    seed: int = 42,
    use_all_boxes: bool = False  # To output all boxes, not only containing text.
                                 # It's useful for training the detection model for lines.
):
    
    # Set the seed
    np.random.seed(seed)
    random.seed(seed)

    out_images_dir = out_dir / "images"
    if not out_images_dir.exists():
        out_images_dir.mkdir()
    out_train_annotation_path = out_dir / "train.tsv"
    out_val_annotation_path = out_dir / "val.tsv"
    
    # Cut out and write text regions
    labels = []
    for ann_path, imgs_dir_path in in_paths:
        with ann_path.open("r") as f:
            annotations = json.load(f)
        for annotation in annotations:
            new_labels = write_text_regions(annotation, imgs_dir_path, out_images_dir,
                                            use_all_boxes=use_all_boxes)
            if new_labels is not None:
                labels += new_labels
    
    labels_df = pd.DataFrame(labels)
    labels_df["text"] = labels_df["text"].str.replace("\n", "")  # drop newlines
    if country == "ua":  # Fix the Ukrainian "i" letter
        labels_df["text"] = labels_df["text"].apply(fix_ukranian_i)
        # For human validation, print fields with latin "i"
        for _, row in labels_df.iterrows():
            if "i" in row["text"] or "I" in row["text"]:
                print("Found latin 'i':", row["text"], "|", row["file"], row["annotation_id"])
    
    # Split into train and validation
    labels_val_df = labels_df.sample(frac=validation_size, random_state=seed)
    labels_train_df = labels_df.drop(labels_val_df.index)
    labels_val_df.reset_index(drop=True, inplace=True)
    labels_train_df.reset_index(drop=True, inplace=True)
    
    # Save the labels
    labels_train_df.to_csv(out_train_annotation_path, sep="\t", index=False)
    labels_val_df.to_csv(out_val_annotation_path, sep="\t", index=False)


def write_text_regions(
    annotation: dict,
    in_images_dir: Path,
    out_images_dir: Path,
    use_all_boxes: bool
) -> list[tuple[str, str]] | None:
    
    # Extract image path
    image_path = Path(annotation['data']['ocr'])
    if image_path.is_relative_to('/data/upload/'):  # Manually uploaded to Label Studio
        # In the case of a manual upload, Label Studio renames the file.
        # To access the file, we need to extract the relative path used in 
        # Label Studio's internal storage. This allows us to access the 
        # internal storage copy by the extracted relative path.
        # The unique part follows 'data/upload/'.
        relative_path = image_path.relative_to('/data/upload/')
    elif image_path.is_relative_to('s3://'):  # S3 connected to Label Studio
        # In the case of S3, Label Studio retains the original file name, 
        # allowing us to access images directly by name in the original image folder.
        relative_path = Path(image_path.name)
    else:
        raise ValueError(f"Unknown image path: {image_path}")
    
    # Open the image
    image_path = in_images_dir / relative_path
    if not image_path.exists():
        print(f"Warning: image not found: {image_path}")
        return None
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)  # Rotate the image according to EXIF
    
    if len(annotation['annotations']) > 1:
        print(f"Warning: multiple annotations for id={annotation['id']}")
    
    # Extract regions containing text
    text_regions = []
    for result in annotation['annotations'][0]['result']:
        if not use_all_boxes:  # Use only boxes containing text
            if result['type'] == 'textarea' and 'text' in result['value']:
                if len(result['value']['text']) == 0:
                    # NOTE: Also print the annotaion class ('text', 'recepient_account', etc.),
                    # because the empty text for such classes is likely an annotation error.
                    print(f"Warning: empty text found for id={result['id']}, type={result['type']} in id={annotation['id']}")
                    continue
                text_regions.append((
                    result['id'],               # bounding box id
                    result['value'],            # bounding box coordinates
                    result['value']['text'][0]  # text
                ))
        else:  # Use all boxes, don't take texts
            if result['type'] == 'rectangle':
                text_regions.append((
                    result['id'],               # bounding box id
                    result['value'],            # bounding box coordinates
                    "<<<EMPTY>>>"               # text
                ))
    
    # Process each region
    region_text_labels = []
    for i, (region_id, region, text) in enumerate(text_regions):
        try:
            region_image = crop_by_box(image, region)
        except KeyError:
            # It falls here if the 'textarea' is an annotator's comment
            print(f"Warning: comment for id={annotation['id']}:")
            print(f"{region['text'][0]}\n")
            continue
        
        if region_image is None:  # too rotated or too small
            print(f"Warning: skipped region for id={annotation['id']}: too rotated or too small")
            continue
        
        if region_image.width * 2 < region_image.height:  # too narrow
            print(f"Warning: skipped region for id={annotation['id']}: too narrow")
            continue
            
        # Generate output file name
        out_file_name = f"{relative_path.stem}_{i}.{relative_path.suffix[1:]}"
        out_path = out_images_dir / out_file_name
        
        # Save the region image
        region_image.save(out_path)
        
        # Add to the result dictionary
        region_text_labels.append({
            "file": out_file_name,
            "text": text,
            "annotation_id": annotation['id'],
            "project_id": annotation['project'],
            "original_image_path": str(relative_path),
            "top": round(region['y'] * image.height / 100),
            "left": round(region['x'] * image.width / 100),
            "bottom": round((region['y'] + region['height']) * image.height / 100),
            "right": round((region['x'] + region['width']) * image.width / 100),
        })
    
    return region_text_labels


def crop_by_box(
    image: Image.Image, box: dict[str, float], box_w_min=10, box_h_min=5
) -> Image.Image:
    """
    Crop the image by the box coordinates. The box is a dictionary with
    keys x, y, width, height, rotation, as it's given by Label Studio.
    
    NOTE: The function is quit slow
    """
    
    # Extract box coordinates
    left = box['x'] * image.width / 100
    top = box['y'] * image.height / 100
    right = left + (box['width'] * image.width / 100)
    bottom = top + (box['height'] * image.height / 100)
    rotation_degrees = box['rotation']
    
    # Skip too rotated boxes
    if 10 < rotation_degrees < 350:
        print(f"Warning: too rotated box: {box}")
        return None
    
    # Do not crop too small boxes
    w = right - left
    h = bottom - top
    if w < box_w_min or h < box_h_min:
        print(f"Warning: too small box: {box}")
        return None
    
    # Crop non-rotated boxes directly
    if not .1 < rotation_degrees < 359.9:
        return image.crop((left, top, right, bottom))
    
    # Rotate the box around the top-left corner
    origin = np.array([left, top])
    box = np.array([
        [left, top],
        [right, top],
        [right, bottom],
        [left, bottom]
    ])
    rotation_radians = np.radians(rotation_degrees)
    rotation_matrix = np.array([
        [np.cos(rotation_radians), -np.sin(rotation_radians)],
        [np.sin(rotation_radians), np.cos(rotation_radians)]
    ])
    rotated_box = np.dot(box - origin, rotation_matrix.T) + origin
    
    # Draw a mask from the rotated box
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(rotated_box.flatten().tolist(), outline=None, fill="white")
    
    # Rotate the image and the mask
    image = image.rotate(rotation_degrees, resample=Image.BICUBIC, expand=True)
    mask = mask.rotate(rotation_degrees, resample=Image.BICUBIC, expand=True)
    
    # Crop the image by the mask
    return image.crop(mask.getbbox())


def is_russian_alphabet(char: str) -> bool:
    return (ord("А") <= ord(char) <= ord("я")) or (char in "Ёё")


def fix_ukranian_i(text: str, threshold=.6) -> str:
    """
    Check if the text contains russian symbols more than `threshold` of
    all alphabetic symbols, and if so, apply substitution 
    latin "i", "I" -> ukranian "і", "І".
    """
    refined = str(text)  # copy
    refined = re.sub(r"[^\w]", "", refined)  # drop punctuation and spaces
    refined = re.sub(r"\d", "", refined)  # drop digits
    russian_count = sum(1 for char in refined if is_russian_alphabet(char))  # calculate the number of russian symbols
    if len(refined) and russian_count / len(refined) > threshold:
        return text.replace("i", "і").replace("I", "І")
    else:
        return text
