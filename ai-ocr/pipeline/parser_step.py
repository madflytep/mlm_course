import io
import json
import pickle
from decimal import Decimal
from pathlib import Path

import click
import pandas as pd
from clearml import Dataset, StorageManager
from loguru import logger

from ai_ocr.metrics import calculate_accuracy
from ai_ocr.parsing import check_named_fields, parse_response_named_fields
from ai_ocr.utils.sys_utils import get_cml_task


@click.command()

@click.option("--dataset_name",
              type=str,
              help="ClearML dataset name")

@click.option("--vlm_results_url",
              type=str,
              help="URL to the OCR results artifact")

def main(dataset_name, vlm_results_url):

    logger.info("Starting parser_step.py")
    cml_task = get_cml_task(task_name="parser_step")

    # Load the VLM results
    logger.info(f"VLM results URL: {vlm_results_url}")
    vlm_results_path = Path(StorageManager.get_local_copy(vlm_results_url))
    with open(vlm_results_path, "rb") as f:
        vlm_results = pickle.load(f)
    logger.info(f"VLM results local path: {vlm_results_path}")

    # Load labels from the dataset
    dataset_path = Path(Dataset.get(dataset_name=dataset_name).get_local_copy())
    logger.info(f"Dataset local path: {dataset_path}")
    dtype = {
        "operation_sum": str,
        "recepient_account": str,
        "recepient_telnum": str,
    }
    labels_df = pd.read_csv(dataset_path / "val.csv", dtype=dtype)
    labels_df["operation_sum"] = labels_df["operation_sum"].apply(Decimal)

    results = []
    
    for idx, vlm_result in enumerate(vlm_results):

        logger.info(f"Processing VLM result {vlm_result['image_filename']}")

        # Parse VLM response
        parsed = parse_response_named_fields(vlm_result["vlm_response"])

        # Get ground truth labels
        gt = labels_df.loc[
            labels_df["image_filename"] == vlm_result["image_filename"]
        ].values

        _, sender_bank, operation_datetime, operation_sum, recepient_account, recepient_telnum = gt.flatten()

        # Try to match the parsed fields with the ground truth
        matches = check_named_fields(parsed,
                                     sender_bank,
                                     operation_datetime,
                                     operation_sum,
                                     recepient_account,
                                     recepient_telnum)

        results.append(
            {
                "image_filename": vlm_result["image_filename"],
                "vlm_respone": vlm_result["vlm_response"],
                "vlm_response_parsed": parsed,
                "ground_truth": {
                    "sender_bank": sender_bank,
                    "operation_datetime": operation_datetime,
                    "operation_sum": operation_sum,
                    "recepient_account": recepient_account,
                    "recepient_telnum": recepient_telnum
                },
                "matches": matches
            }
        )
    
    # Build a JSON file
    buffer = io.StringIO()
    def decimal_to_string(obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return obj
    json.dump(results, buffer, indent=4, ensure_ascii=False,
              default=decimal_to_string)
    json_data = buffer.getvalue()
    buffer.close()

    # Calculate accuracy
    accuracy = calculate_accuracy(results)
    for k, v in accuracy.items():
        cml_task.get_logger().report_single_value(k, v)
    
    # Finalize the task
    logger.info("All images have been processed")
    cml_task.upload_artifact("results", json_data)
    cml_task.close()
    logger.info("Finished parser_step.py")
    
    
if __name__ == "__main__":
    main()