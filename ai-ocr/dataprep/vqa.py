"""
Function to convert Label Studio annotations to a VQA dataset.

The top-level function is `labelstudio_to_vqa`.
"""


import csv
import json
import random
import re
import shutil
from pathlib import Path
from typing import List, Literal, Tuple

import albumentations as A
import dateparser
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

from ai_ocr.ocr import EasyOCRModel, sort_boxes
from ai_ocr.parsing import parse_sum
from ai_ocr.utils.files_utils import create_empty_folder

question_common = \
"""<image>
Recognized text with errors: {ocr_text}
What are the operation_datetime, operation_sum, sender_bank, recepient_account, and recepient_telnum in the image?"""


# Which boxes to extract from the LS-annotation
fields = (
    "operation_datetime",
    "operation_date",
    "operation_time",
    "operation_sum",
    "recepient_account",
    "recepient_telnum",
)


"""
Pixel-level transformations:
    Exclimation mark denotes most useful transformations for our task.
    HSV/Brightness/Contrast/Gamma:
        (!) ColorJitter, RGBShift, RandomBrightnessContrast, RandomGamma,
        RandomBrightness, RandomContrast
    Blur:
        GaussianBlur, MotionBlur, MedianBlur, Blur, GlassBlur
    Noise:
        (!) ISONoise, (!) GaussNoise
    Additional elements:
        (!) RandomShadow (random dark figures on the image), ChromaticAberration
    Artefacts:
        (?) Sharpen, Emboss, (!) ImageCompression, RingingOvershoot
    (!) Strong but not blurring:
        ToGray, ToSepia, CLAHE, Solarize, InvertImg, Posterize, Equalize

Geometric transformations:
    Exclimation mark denotes valid transformations for our task.
    Rotations, flips, and transpositions:
        (!) RandomRotate90, Rotate, ShiftScaleRotate, HorizontalFlip,
        VerticalFlip, Transpose, D4
    Affine transformations:
        (!) ElasticTransform, (!) GridDistortion,
        (!) OpticalDistortion, Perspective, Affine, PiecewiseAffine,
        GridElasticDeform
    Crop and pad:
        CenterCrop, RandomCrop, RandomSizedCrop, CropNonEmptyMaskIfExists,
        PadIfNeeded, Crop, CropAndPad, BBoxSafeRandomCrop,
        RandomCropFromBorders, RandomSizedBBoxSafeCrop
    Resize:
        LongestMaxSize, SmallestMaxSize, Resize, RandomScale, RandomSizedCrop
    Mask, dropouts and morhological :
        CoarseDropout, GridDropout, MaskDropout, (!) PixelDropout,
        OverlayElemets, XYMasking, (?) Morphological, 
"""

# NOTE: TextImage is an interesting option for generating synthetic data

# Light transformations
p_lite = .3
transform_lite = A.Compose([
    A.RandomRotate90(p=p_lite),
    A.ColorJitter(p=p_lite),
])

# Strong transformations
p_strong = .2
transform_strong = A.Compose([
    A.RandomRotate90(p=p_strong),
    A.ColorJitter(p=p_strong),
    A.ISONoise(p=p_strong),
    A.ImageCompression(p=p_strong),
    A.ElasticTransform(p=p_strong),
    A.GridDistortion(p=p_strong),
    A.OpticalDistortion(p=p_strong),
    A.PixelDropout(p=p_strong),
    A.ToGray(p=p_strong),
    A.ToSepia(p=p_strong),
])


def merge_vqa_datasets(
    in_paths: list[Path],
    out_dir: Path,
    seed: int = 42,
    crop: int = None
) -> None:
    """Merge multiple VQA datasets into one and shuffle the samples.

    Args:
        in_paths (list[Path]): List of paths to the VQA datasets directories.
        out_dir (Path): Output directory for the merged dataset.
        seed (int, optional): Random seed. Defaults to 42.
    """

    print(f"Writing the merged dataset to {out_dir}")
    create_empty_folder(out_dir / "images")

    # Read annotations and copy images
    merged = []
    for in_path in in_paths:
        print(f"Copying dataset from {in_path}")
        dataset_name = in_path.name
        with open(in_path / "train.jsonl") as f:
            dataset = f.readlines()
        for sample_str in tqdm(dataset):
            sample = json.loads(sample_str)
            img_src = in_path / "images" / sample["image"]
            img_new_name = f"{dataset_name}_{img_src.name}"
            img_dst = out_dir / "images" / img_new_name
            shutil.copy(img_src, img_dst)
            sample["image"] = img_new_name
            new_sample_str = json.dumps(sample, ensure_ascii=False).strip()
            merged.append(new_sample_str)
    
    # Merge and shuffle
    random.seed(seed)
    random.shuffle(merged)

    # Write the merged dataset
    if isinstance(crop, int):
        merged = merged[:crop]
    with open(out_dir / "train.jsonl", "w") as f:
        f.write("\n".join(merged))
    

def labelstudio_to_vqa(
    in_paths: list[tuple[Path, Path]],
    fields: tuple[str],
    country: Literal["ru", "ua", "kg", "uz"],
    out_vqa_dir: Path,
    out_vqa_aug_dir: Path,
    validation_size: float = 0.2,
    seed: int = 42,
    bypass_ocr: bool = False,
    ocr_model_dir: Path = Path("../models/easyocr"),
    ocr_model_name: str = "cyrillic_g2_ft4",
    custom_detection_model_name_or_path: str = None,
    ocr_languages: List[str] = ["ru"],
):
    
    # Set the seed
    np.random.seed(seed)
    random.seed(seed)

    # Initialize the OCR model
    if not bypass_ocr:
        ocr = EasyOCRModel(
            gpu=True,
            languages=ocr_languages,
            custom_model_dir_path=ocr_model_dir,
            custom_recognition_model_name=ocr_model_name,
            custom_detection_model_name_or_path=custom_detection_model_name_or_path
        )
    else:
        class OcrStub:
            def read_text(self, _):
                return [(None, "OCR BYPASSED", None)]
        ocr = OcrStub()
    
    # Create a dataframe with extracted fields and OCR texts
    rows = []
    for annotations_file, images_dir in in_paths:
        with annotations_file.open("r") as f:
            annotations = json.load(f)
        for annotation in tqdm(annotations):
            new_fields = extract_fields(annotation, images_dir, fields, ocr)
            if new_fields is not None:
                rows.append(new_fields)
    labels_df = pd.DataFrame(rows)

    # Create the output directory
    create_empty_folder(out_vqa_dir)

    # For debugging, save the whole dataframe before cleaning
    labels_df.to_csv(out_vqa_dir / "dirty.csv", index=False)

    # Clean the extracted fields
    clean_column(labels_df, "operation_datetime", lambda x: clean_operation_datetime(x, country))
    clean_column(labels_df, "recepient_account", clean_recepient_account)
    clean_column(labels_df, "operation_sum", clean_operation_sum)
    clean_column(labels_df, "recepient_telnum", lambda x: clean_recepient_telnum(x, country))
    clean_column(labels_df, "operation_date", lambda x: clean_operation_date(x, country))
    clean_column(labels_df, "operation_time", lambda x: clean_operation_time(x, country))

    # Concatenate date and time where they are provided separately
    labels_df = concatenate_date_time(labels_df)

    # For debugging, save the whole dataframe
    labels_df.to_csv(out_vqa_dir / "all.csv", index=False)

    # Split into train and validation
    labels_val = labels_df.sample(frac=validation_size, random_state=seed)
    labels_train = labels_df.drop(labels_val.index)
    labels_val.reset_index(drop=True, inplace=True)
    labels_train.reset_index(drop=True, inplace=True)
    print(f"Train size: {len(labels_train)}, Validation size: {len(labels_val)}")

    # Write the litly augmented VQA dataset
    write_vqa_dataset(labels_train, out_vqa_dir, "train.jsonl", transform_lite)
    write_vqa_dataset(labels_val, out_vqa_dir, "val.jsonl")

    # Write the strongly augmented VQA dataset
    create_empty_folder(out_vqa_aug_dir)
    write_vqa_dataset(labels_train, out_vqa_aug_dir, "train.jsonl", transform_strong)

    # #
    # Additionally, write a CSV file wich is compatible with the current
    # test pipeline

    csv_columns = [
        "image_filename",
        "sender_bank",
        "operation_datetime",
        "operation_sum",
        "recepient_account",
        "recepient_telnum",
    ]

    csv_df = labels_val.copy()
    csv_df.drop(columns=csv_columns[-4:], inplace=True)
    for col in csv_columns[-4:]:
        csv_df.rename(columns={f"{col}_clean": col}, inplace=True)
    csv_df.rename(columns={"image": "image_filename"}, inplace=True)

    out_csv_path = out_vqa_dir / "val.csv"
    csv_df.to_csv(out_csv_path, columns=csv_columns, index=False, quoting=csv.QUOTE_ALL)


def extract_fields(
    annotation: dict,
    in_images_dir: Path,
    fields: tuple[str],
    ocr: EasyOCRModel
) -> dict[str, str|None]:
    
    if len(annotation['annotations']) > 1:
        print(f"Warning: multiple annotations for id={annotation['id']}")
    
    # Group rectangles, textareas and labels by their common ID
    grouped_objects = {}
    relations = []  # gather 'relations': connections between lines of a single text field
    for result in annotation['annotations'][0]['result']:
        if result['type'] == 'relation':
            relations.append(result)
            continue
        result_id = result['id']
        if result_id not in grouped_objects:
            grouped_objects[result_id] = []
        grouped_objects[result_id].append(result)
        
    def find_dict(dict_list, key, value):
        for dictionary in dict_list:
            if dictionary.get(key) == value:
                return dictionary
        return None
    
    def print_problematic_relation(relation, grouped_objects):
        print(f"~~~ from id={relation['from_id']} ~~~")
        for o in grouped_objects[relation['from_id']]:
            print(o)
        print(f"~~~ to id={relation['to_id']} ~~~")
        for o in grouped_objects[relation['to_id']]:
            print(o)
        print()
        
    # TODO: It might be chained relations for more than two lines
    
    # Merge text fields which are broken into two lines
    for rel in relations:
        # Process 'from_id' field
        try:
            objects_from = grouped_objects[rel['from_id']]
        except KeyError:
            print(f"Warning: objects with id=from_id={rel['from_id']} not found in id={annotation['id']}. Relation:")
            print(rel)
            continue
        text_from = find_dict(objects_from, 'type', 'textarea')
        label_from = find_dict(objects_from, 'type', 'labels')
        if text_from is None or label_from is None or len(text_from['value']['text']) == 0:
            print(f"Warning: text and label not simultaneously found by 'from' field in id={annotation['id']}."
                  " Details:")
            print_problematic_relation(rel, grouped_objects)
            continue
        # Process 'to_id' field
        try:
            objects_to = grouped_objects[rel['to_id']]
        except KeyError:
            print(f"Warning: objects with id=to_id={rel['to_id']} not found in id={annotation['id']}. Relation:")
            print(rel)
            continue
        text_to = find_dict(objects_to, 'type', 'textarea')
        label_to = find_dict(objects_to, 'type', 'labels')
        if text_to is None or label_to is None or len(text_to['value']['text']) == 0:
            print(f"Warning: text and label not simultaneously found by 'to' field in id={annotation['id']}."
                  " Details:")
            print_problematic_relation(rel, grouped_objects)
            continue
        # Extract necassary values
        label_from = label_from['value']['labels'][0]
        text_from = text_from['value']['text'][0]
        text_to = text_to['value']['text'][0]
        label_to = label_to['value']['labels'][0]
        # Merge the text fields
        if label_from != label_to:
            print(f"Warning: labels mismatch in id={annotation['id']}: {label_from} != {label_to}")
            continue
        if text_from == text_to:  # Label Studio sometimes merges the lines on its own
            text_merged = text_from
        else:
            if label_from in ["operation_number", "operation_sum"]:
                text_merged =  text_from + text_to  # don't add space
            else:
                text_merged = f"{text_from} {text_to}"
        # Modify from-objects and drop to-objects
        find_dict(objects_from, 'type', 'textarea')['value']['text'][0] = text_merged
        del grouped_objects[rel['to_id']]
    
    # Extract specified fields
    result = {f: [] for f in fields}
    result['id'] = annotation['id']
    result['project'] = annotation['project']
    for _, objects in grouped_objects.items():
        label_obj = find_dict(objects, 'type', 'labels')
        label = label_obj['value']['labels'][0] if label_obj else None
        textarea_obj = find_dict(objects, 'type', 'textarea')
        try:
            text = textarea_obj['value']['text'][0] if textarea_obj else None
        except IndexError:
            print(f"Warning: empty 'text' for id={annotation['id']}: {textarea_obj}")
            text = None
        if label is not None and text is not None:
            if label in fields:
                if label not in result:
                    result[label] = [text]
                else:
                    result[label].append(text)
    
    # Extract 'sender_bank'
    for _, objects in grouped_objects.items():
        choices_obj = find_dict(objects, 'type', 'choices')
        if choices_obj is not None and choices_obj['from_name'] == 'sender_bank':
            result["sender_bank"] = choices_obj['value']['choices'][0]
            break
    
    # Add the image info
    image_path = Path(annotation['data']['ocr'])
    if image_path.is_relative_to('/data/upload/'):  # Manually uploaded to Label Studio
        relative_path = image_path.relative_to('/data/upload/')
    elif image_path.is_relative_to('s3://'):  # S3 connected to Label Studio
        relative_path = Path(image_path.name)
    else:
        raise ValueError(f"Unknown image path: {image_path}")
    image_path = in_images_dir.resolve() / relative_path
    if not image_path.exists():
        print(f"Warning: image not found: {image_path}")
        return None
    result['image_orig_path'] = str(image_path)  # to copy the image later using augmentations
    result['image'] = str(image_path.name)  # to reference the image in the dataset during training

    # Open the image and OCR it
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    ocr_result = list(zip(*ocr.read_text(image_np)))
    if ocr_result:
        if len(ocr_result) == 3:
            _, texts, _ = ocr_result
        elif len(ocr_result) == 4:
            # In this case we additionally need to sort the boxes by coordinates
            boxes, texts, recognition_scores, detection_scores = ocr_result
            boxes, texts, recognition_scores, detection_scores = sort_boxes(boxes, texts, recognition_scores, detection_scores,
                                                                            image_np.shape[1])
        else:
            raise ValueError(f"Invalid number of elements in the OCR result: {ocr_result}")
        ocr_text = " ".join(texts)
    else:
        ocr_text = None
    result["ocr_text"] = ocr_text
    
    return result


def clean_column(annotations: pd.DataFrame, column: str, clean_fn: callable):
    # Create a new column with cleaned values
    column_clean = f"{column}_clean"
    annotations[column_clean] = annotations[column].apply(clean_fn)
    
    # Show erroneous values
    print(f"\n ~~~ errouneous values in '{column}' ~~~")
    for row in annotations[["project", "id", column]].loc[annotations[column_clean] == 'ERROR'].values:
        print(row)
    annotations[column_clean] = annotations[column_clean].replace('ERROR', None)
    
    def is_non_empty_list(value):
        if isinstance(value, list):
            return len(value) > 0
        else:
            print(f"Warning: non-list value passed to `is_empty_list()`: {value}")
            return False
    
    # Show lost values statistics
    print(f"\n ~~~ lost values in '{column}' ~~~")
    print(f"Total rows: {len(annotations)}")
    print(f"Values at input: {annotations[column].apply(is_non_empty_list).sum()}")
    print(f"Values at output: {annotations[column_clean].notna().sum()}")


def clean_operation_datetime(texts: list[str],
                             country: Literal["ru", "ua", "kg", "uz", "in"],
                             format_str="%B %d, %Y %H:%M:%S") -> str | None:
    # No texts
    if texts == []:
        return None
        
    if len(texts) == 1:  # there's only one text
        text = texts[0]
    else:  # multiple texts: choose the longest one
        text = max(texts, key=len)
        print(f"Warning: multiple texts for 'operation_datetime' field: {texts}")
    
    # Set country-specific parameters
    if country == "ru":
        garbage = ("(MCK)", "МСК", "мск", "GTM+03:00")
        languages = ['en', 'ru']
    elif country == "ua":
        languages = ['en', 'ru', 'uk']
        garbage = ("Дата та час операції:", "дата", "час", "вiд")
    elif country == "kg":
        languages = ['en', 'ru']
        garbage = []
    elif country == "uz":
        languages = ['en', 'ru']
        garbage = ['da']
    elif country == "in":
        languages = ['en']
        garbage = []
    else:
        raise ValueError("Invalid country code")
    
    # Drop garbage
    for g in garbage:
        text = text.replace(g, "")
    
    # Call the parser
    datetime = dateparser.parse(text, languages=languages,
                                settings={'DATE_ORDER': 'DMY'})
    
    if datetime is None and country == "uz":
        if not re.search(r"\d{4}-\d{2}-\d{2}", text):
            datetime = dateparser.parse(text, languages=['uz'])
        else:
            text = text.replace("-", ".")
            datetime = dateparser.parse(text, languages=['uz'], settings={'DATE_ORDER': 'YMD'})
    
    # Convert to `str` or return 'ERROR' if the parser failed
    if datetime is not None:
        datetime = datetime.replace(tzinfo=None)  # drop timezone
        datetime = datetime.strftime(format_str)
    else:
        datetime = 'ERROR'

    return datetime


def clean_recepient_account(texts: list[str] | None) -> str | None:

    # No texts
    if texts == []:
        return None
    
    if len(texts) == 1:  # there's only one text
        text = texts[0]
    else: # multiple texts: choose the longest one
        text = max(texts, key=len)
        print(f"Warning: multiple texts for 'recepient_account' field: {texts}. Selected: {text}")
        
    last_four = text[-4:]

    if last_four.isdigit():  # if the last four characters are digits, then use them
        text_cleaned = last_four
        if not len(text_cleaned) == 4:
            print(f"Error: cleaned account number length is not 4: {text}")
            return 'ERROR'
    else:  # for accounts like "516818********36" use all characters
        text_cleaned = text

    return text_cleaned


def clean_operation_sum(texts: list[str] | None) -> str | None:
    
    # No texts
    if texts == []:
        return None
    
    if len(texts) == 1:  # there's only one text
        text = texts[0]
    else:  # multiple texts: choose the shortest one
        text = min(texts, key=len)
        print(f"Warning: more than 2 texts for 'operation_sum' field: {texts}. Selected: {text}")
    
    # Drop garbage
    text_re = re.sub(r"\.$", "", text)
    text_re = re.sub(r",\d\d\b", "", text_re)
    text_re = re.sub(r"Rs\.", "", text_re)  # for Indian Rupees
    text_re = re.sub(r"₨\.", "", text_re)  # for Indian Rupees
    
    # Try to parse
    sum = parse_sum(text_re)
    text_cleaned = format(sum, ".2f") if sum is not None else 'ERROR'
    
    return text_cleaned


def clean_recepient_telnum(texts: list[str] | None,
                           country: Literal["ru", "ua", "kg", "uz", "in"]) -> str | None:
    
    # No texts
    if texts == []:
        return None
    
    if len(texts) == 1:  # there's only one text
        text = texts[0]
    else:  # multiple texts: choose the longest one
        text = max(texts, key=len)
        print(f"Warning: multiple texts for 'recepient_telnum' field: {texts}. Selected: {text}")
    
    # Leave only: digits, '*', '+'
    text_cleaned = re.sub(r"[^0-9*+]", "", text)

    if len(text_cleaned) == 0:
        return 'ERROR'
    
    if country == "ru":
        if text_cleaned.startswith('+7') and len(text_cleaned) != 12:
            return 'ERROR'
        if text_cleaned.startswith('7') and len(text_cleaned) != 11:
            return 'ERROR'
        if text_cleaned.startswith('9') and len(text_cleaned) != 10:
            return 'ERROR'
    elif country == "ua":
        pass
    elif country == "kg":
        pass
    elif country == "uz":
        pass
    elif country == "in":
        pass
    else:
        raise ValueError("Invalid country code")

    return text_cleaned


def clean_operation_date(texts: list[str] | None,
                         country: Literal["ru", "ua", "kg", "uz", "in"],
                         format_str="%B %d, %Y") -> str | None:
    # No texts
    if texts == []:
        return None
    
    if len(texts) == 1:  # there's only one text
        text = texts[0]
    else:  # multiple texts: choose the shortest one
        text = min(texts, key=len)
        print(f"Warning: multiple texts for 'operation_date' field: {texts}. Selected: {text}")
    
    if country == "ru":
        text_re = re.sub(r"(?i)дата операции:?\s*", "", text)
        text_re = re.sub(r"(?i)дата:?\s*", "", text_re)
        datetime = dateparser.parse(text_re, languages=['en', 'ru'],
                                    settings={'DATE_ORDER': 'DMY'})
    elif country == "ua":
        text_re = str(text)
        datetime = dateparser.parse(text_re, languages=['en', 'ru', 'uk'],
                                    settings={'DATE_ORDER': 'DMY'})
    elif country == "kg":
        text_re = str(text)
        datetime = dateparser.parse(text_re, languages=['en', 'ru'],
                                    settings={'DATE_ORDER': 'DMY'})
    elif country == "uz":
        text_re = str(text)
        datetime = dateparser.parse(text_re, languages=['en', 'ru'],
                                    settings={'DATE_ORDER': 'DMY'})
        if datetime is None:
            if not re.search(r"\d{4}-\d{2}-\d{2}", text):
                datetime = dateparser.parse(text_re, languages=['uz'])
            else:
                text_re = text_re.replace("-", ".")
                datetime = dateparser.parse(text_re, languages=['uz'], settings={'DATE_ORDER': 'YMD'})
    elif country == "in":
        text_re = str(text)
        datetime = dateparser.parse(text_re, languages=['en'],
                                    settings={'DATE_ORDER': 'DMY'})
    else:
        raise ValueError("Invalid country code")
    
    # Convert to `str` or return 'ERROR' if the parser failed
    if datetime is not None:
        datetime = datetime.replace(tzinfo=None)  # drop timezone
        datetime = datetime.strftime(format_str)
    else:
        datetime = 'ERROR'

    return datetime


def clean_operation_time(text: list[str] | None,
                         country: Literal["ru", "ua", "kg", "uz", "in"],
                         format_str="%H:%M:%S") -> str | None:
    # No texts
    if text == []:
        return None
    
    if len(text) == 1:  # there's only one text
        text = text[0]
    else:  # multiple texts: choose the shortest one
        text = min(text, key=len)
        print(f"Warning: multiple texts for 'operation_time' field: {text}. Selected: {text}")
    
    if country == "ru":
        text_re = re.sub(r"(?i)время операции \(мск\):?\s*", "", text)
        text_re = re.sub(r"(?i)время \(мск\):?\s*", "", text_re)
        text_re = re.sub(r"(?i)время:?\s*", "", text_re)
        text_re = re.sub(r"(?i)\s*\(мск\)\s*$", "", text_re)
        text_re = re.sub(r"(?i)\s*мск\s*$", "", text_re)
        datetime = dateparser.parse(text_re, languages=['en', 'ru'])
    elif country == "ua":
        text_re = str(text)
        datetime = dateparser.parse(text_re, languages=['en', 'ru', 'uk'])
    elif country == "kg":
        text_re = str(text)
        datetime = dateparser.parse(text_re, languages=['en', 'ru'])
    elif country == "uz":
        text_re = str(text)
        datetime = dateparser.parse(text_re, languages=['en', 'ru'])
    elif country == "in":
        text_re = str(text)
        datetime = dateparser.parse(text_re, languages=['en'])
    else:
        raise ValueError("Invalid country code")

    # Convert to `str` or return 'ERROR' if the parser failed
    if datetime is not None:
        datetime = datetime.replace(tzinfo=None)  # drop timezone
        datetime = datetime.strftime(format_str)
    else:
        datetime = 'ERROR'

    return datetime


def concatenate_date_time(labels: pd.DataFrame) -> pd.DataFrame:
    for idx, row in labels.iterrows():
        date = row["operation_date_clean"]
        time = row["operation_time_clean"]
        if row["operation_datetime_clean"] is None:
            if date is not None and time is not None:
                labels.at[idx, "operation_datetime_clean"] = f"{date} {time}"
            elif date is not None and time is None:
                labels.at[idx, "operation_datetime_clean"] = date
            else:
                print(f"Cannot concatenate date and time for id={row['id']}: ", end="")
                print(f"date: {date}, time: {time}, datetime: {row['operation_datetime_clean']}")
    return labels


def write_vqa_dataset(
    labels: pd.DataFrame,
    out_dir: Path,
    annotations_filename: str,
    augmentations=None
) -> None:

    # Create a common image directory
    out_images_dir = out_dir / "images"
    if not out_images_dir.exists():
        create_empty_folder(out_images_dir)
        
    # Write the train annotations
    write_vqa_annotations(labels, out_dir / annotations_filename, question_common)

    # Write the train images
    write_vqa_images(labels, out_images_dir, augmentations)


def write_vqa_annotations(labels: pd.DataFrame, out_dir: Path, question_template: str):

    def format_answer(row: pd.Series) -> str:
        answers = []
        answers.append(f"sender_bank: {row['sender_bank']}")
        for field in fields:
            if field in ["operation_date", "operation_time"]:  # we use only operation_datetime
                continue
            try:
                if row[f"{field}_clean"] is not None:
                    answers.append(f"{field}: {row[f'{field}_clean']}")
            except KeyError:
                pass
        return "\n".join(answers)
    
    lines = []
    for idx, row in labels.iterrows():
        sample = {
            "id": idx,
            "image": row["image"],
            "conversations": [
                {
                    "from": "human",
                    "value": question_template.format(ocr_text=row["ocr_text"])
                },
                {
                    "from": "gpt",
                    "value": format_answer(row)
                }
            ]
        }
        line = json.dumps(sample, ensure_ascii=False)
        lines.append(line)
    
    with out_dir.open("w") as f:
        f.write("\n".join(lines))


def write_vqa_images(
    labels: pd.DataFrame,
    out_dir: Path,
    augmentations=None
) -> None:
    for _, row in labels.iterrows():
        image_orig_path = Path(row["image_orig_path"])
        image = Image.open(image_orig_path)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        if augmentations is not None:
            image = augmentations(image=np.array(image))["image"]
            image = Image.fromarray(image)
        image_save_path = out_dir / image_orig_path.name
        if not image_save_path.exists():
            image.save(out_dir / image_orig_path.name)
        else:
            raise ValueError(f"Image already exists: {image_save_path}")
