import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import click
import torch
from loguru import logger
from transformers import AutoModel

from ai_ocr.dataprep.vqa import merge_vqa_datasets
from ai_ocr.utils.files_utils import create_empty_folder
from ai_ocr.utils.sys_utils import get_cml_task

# sys.path.insert(0, "/home/nselivanov/InternVL/internvl_chat")

def run_script(script_path: str,
               bad_line: str = None) -> int:
    process = subprocess.Popen(
        [script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    # Print output in real-time
    is_bad = False  # Whether the script failed
    while True:
        output = process.stdout.readline()
        # In some cases, we cannot rely on return code,
        # so we need to check the output:
        if bad_line and bad_line in output:
            is_bad = True
        if output:
            print(output.strip())  # Mirror the output
        if process.poll() is not None:
            break  # Process has finished
    returncode = process.poll()
    if is_bad:
        returncode = 1
    return returncode


def count_lines(filepath):
    """
    Counts the number of lines in a file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return sum(1 for _ in file)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return 0
    except IOError as e:
        print(f"Error reading file '{filepath}': {e}")
        return 0


@click.command()

@click.option("--in_dataset_dirs",
              type=str,
              help="Input dataset directories separeted by comma.")

@click.option("--interim_dataset_dir",
              type=click.Path(exists=False, path_type=Path),
              help="Path to merged and shuffled interim dataset.")

@click.option("--repo_path",
              type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
              help="InternVL repository path.")

@click.option("--save_dir",
              type=click.Path(exists=False, path_type=Path),
              help="A path to the output directory.")

@click.option("--modify_dir",
              type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
              help="A path to the model weights to be replaced.")

@click.option("--early_stop", type=int,
              help="Stop after this many images have been processed")

def main(
    in_dataset_dirs,
    interim_dataset_dir,
    repo_path,
    save_dir,
    modify_dir,
    early_stop
):
    cml_task = get_cml_task(task_name="train_step")
    logger.info("Starting train_step.py")
    logger.info(f"Input dataset directories: {in_dataset_dirs}")
    logger.info(f"Interim dataset directory: {interim_dataset_dir}")
    logger.info(f"InternVL repository path: {repo_path}")
    logger.info(f"Output directory: {save_dir}")
    logger.info(f"Model weights directory: {modify_dir}")
    logger.info(f"Early stop: {early_stop}")

    if early_stop == 0:
        logger.info("Early stop is set to 0. Skipping the training.")
        logger.info("Closing CML task...")
        cml_task.close()
        logger.info("Finished train_step.py")
        return
        

    interim_dataset_dir = interim_dataset_dir.resolve()
    repo_path = repo_path.resolve()
    save_dir = save_dir.resolve()
    modify_dir = modify_dir.resolve()

    # Merge and shuffle the datasets, write result to `interim_dataset_dir`
    in_dataset_dirs = [Path(p) for p in in_dataset_dirs.split(",")]
    if len(str(interim_dataset_dir)) < 3:
        raise ValueError("Protection against accidental deletion of paths "
                         "like '.' or '..'.")
    create_empty_folder(interim_dataset_dir)
    merge_vqa_datasets(in_dataset_dirs, interim_dataset_dir,
                       crop=early_stop)

    # Prepare the datasets configuration
    datasets_config = {
        "merged_vqa": {
            "root": (interim_dataset_dir / "images").resolve().as_posix(),
            "annotation": (interim_dataset_dir / "train.jsonl").resolve().as_posix(),
            "data_augment": False,
            "repeat_time": 1,
            "length": count_lines(interim_dataset_dir / "train.jsonl")
        }
    }

    # Write pretty-printed JSON datasets configuration
    dataset_config_path = repo_path / "internvl_chat" / "shell" / "data" / "custom.json"
    with open(dataset_config_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(datasets_config, indent=4))

    # Compose the train script
    venv_path = repo_path / "venv"
    train_script = repo_path / "internvl_chat" / "custom_26b.sh"
    work_dir = repo_path / "internvl_chat"
    script = (
        f"#!/bin/bash\n"
        f"source {venv_path}/bin/activate\n"
        f"cd {work_dir}\n"
        f"bash {train_script}\n"
        f"exit_code=$?\n"
        f"exit $exit_code\n"
    )
    logger.info(f"Train script:\n{script}")
    
    # Save the train script
    script_path = Path("/tmp/train.sh")
    with open(script_path, 'w', encoding='ascii') as file:
        file.write(script)
    script_path.chmod(0o777)
    
    # Execute the train script
    logger.info(f"Executing the train script: {script_path}")
    returncode = run_script(script_path.resolve().as_posix(),
                            bad_line="exitcode  : 1")
    logger.info(f"Return code: {returncode}")
    if returncode != 0:
        logger.error("Training script failed.")
        sys.exit(returncode)

    # Compose the merge script
    train_dir = repo_path / "internvl_chat/work_dirs/internvl_chat_v2_0/internvl2_26b_internlm2_20b_dynamic_res_2nd_finetune_lora"
    if len(str(save_dir)) < 3:
        raise ValueError("Protection against accidental deletion of paths "
                         "like '.' or '..'.")
    script = (
        f"#!/bin/bash\n"
        f"source {venv_path}/bin/activate\n"
        f"cd {repo_path}\n"
        f"unset HF_HOME\n"
        f"HF_HOME=''\n"  # Set empty HF_HOME to avoid using the cache
        f"python internvl_chat/merge_lora.py --input-path {train_dir} --output-path {save_dir}\n"
        f"exit_code=$?\n"
        f"exit $exit_code\n"
    )
    logger.info(f"Merge script:\n{script}")
    
    # Save the merge script
    script_path = Path("/tmp/merge.sh")
    with open(script_path, 'w', encoding='ascii') as file:
        file.write(script)
    script_path.chmod(0o777)

    # Execute the merge script 
    logger.info(f"Executing the merge script: {script_path}")
    returncode = run_script(script_path.resolve().as_posix())
    logger.info(f"Return code: {returncode}")
    if returncode != 0:
        logger.error("Merge script failed.")
        sys.exit(returncode)

    # Replace the model weights
    for file in modify_dir.glob("*.safetensors"):
        file.unlink()
    for file in save_dir.glob("*.safetensors"):
        shutil.copy(file, modify_dir)

    logger.info("Closing CML task...")
    cml_task.close()
    logger.info("Finished train_step.py")


if __name__ == "__main__":
    main()
