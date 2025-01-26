from pathlib import Path

import click
from clearml import Dataset
from loguru import logger

from ai_ocr.dataprep.vqa import fields, labelstudio_to_vqa
from ai_ocr.utils.sys_utils import get_cml_task


@click.command()

@click.option("--in_data_paths",
              type=str,
              help="Input data paths in the format 'annotation_path, "
                   "images_dir_path, [annotation_path, images_dir_path, ...]'.")

@click.option("--out_dataset_dir",
              type=click.Path(exists=False, path_type=Path),
              help="Output dataset directory. If exists, will be overwritten.")

@click.option("--country",
              type=click.Choice(["ru", "ua", "kg", "uz", "in"]),
              help="Country code for the dataset.")

@click.option("--bypass_ocr",
              is_flag=True,
              help="Bypass OCR and use existing OCR results.")

@click.option("--ocr_model_dir",
              type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
              help="EasyOCR model directory.")

@click.option("--ocr_model_name",
              type=str,
              help="EasyOCR model name.")

@click.option("--custom_detection_model_name_or_path",  # for detection model
              type=str,
              help="Custom detection model name or path")

def main(
    in_data_paths,
    out_dataset_dir,
    country,
    bypass_ocr,
    ocr_model_dir,
    ocr_model_name,
    custom_detection_model_name_or_path
):
    
    cml_task = get_cml_task(task_name=f"vqa_step_{country}")
    logger.info("Starting vqa_step.py")

    # Print all arguments
    logger.info(f"Input data paths: {in_data_paths}")
    logger.info(f"Output dataset directory: {out_dataset_dir}")
    logger.info(f"Country: {country}")
    logger.info(f"Bypass OCR: {bypass_ocr}")
    logger.info(f"OCR model directory: {ocr_model_dir}")
    logger.info(f"OCR model name: {ocr_model_name}")
    logger.info(f"Custom detection model name or path: {custom_detection_model_name_or_path}")

    # Split the input data paths
    in_data_paths = in_data_paths.split(",")
    if len(in_data_paths) % 2 != 0:
        raise ValueError("Input data paths should be provided in pairs.")
    # Convert to list of tuples
    in_data_paths = [(Path(in_data_paths[i]), Path(in_data_paths[i + 1]))
                     for i in range(0, len(in_data_paths), 2)]
    
    logger.info("Input data paths:")
    for annotation_file, images_dir in in_data_paths:
        logger.info(f"{annotation_file} {images_dir}")
    
    # Format the output dataset directory for augmentated version
    dirname = out_dataset_dir.name
    out_aug_dataset_dir = out_dataset_dir.with_name(f"{dirname}-aug")

    logger.info(f"Output augmented dataset directory: {out_aug_dataset_dir}")
    
    # Call the main function
    labelstudio_to_vqa(
        in_data_paths,
        fields,
        country,
        out_dataset_dir,
        out_aug_dataset_dir,
        bypass_ocr=bypass_ocr,
        ocr_model_dir=ocr_model_dir,
        ocr_model_name=ocr_model_name,
        custom_detection_model_name_or_path=custom_detection_model_name_or_path
    )

    # Check if it's necessary to update the dataset in CML
    logger.info("Comparing a new dataset with the existing one in ClearML...")
    dataset_name = out_dataset_dir.name.replace("dataset_", "")
    try:
        cml_dataset = Dataset.get(dataset_name=dataset_name)
        is_new = False
        changed = cml_dataset.sync_folder(out_dataset_dir)
    except ValueError:  # Dataset does not exist yet
        is_new = True
        changed = [True]
    if any(changed):
        if not is_new:
            logger.info("Changes detected. Updating the dataset in ClearML.")
        else:
            logger.info("No existing dataset. Creating a new dataset in ClearML.")
            logger.info("Will be used the default project name: 'AI-OCR'")
        new_cml_dataset = Dataset.create(
            dataset_name=dataset_name,
            dataset_project=cml_dataset.project if not is_new else 'AI-OCR',
            parent_datasets=[cml_dataset.id] if not is_new else None
        )
        new_cml_dataset.sync_folder(out_dataset_dir)
        new_cml_dataset.upload()
        new_cml_dataset.finalize()
        if not is_new:
            logger.info(f"Dataset '{dataset_name}' has been updated.")
        else:
            logger.info(f"New dataset '{dataset_name}' has been created.")
    else:
        logger.info("No changes in the dataset.")

    cml_task.close()
    logger.info("Finished vqa_step.py")


if __name__ == "__main__":
    main()
