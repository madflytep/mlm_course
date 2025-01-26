import asyncio
import io
import pickle
import time
from itertools import islice
from pathlib import Path

import click
from clearml import Dataset, StorageManager
from loguru import logger
from PIL import Image

from ai_ocr.utils.sys_utils import get_cml_task


async def run_tasks(vlm, patches_all, prompts):
    tasks = []
    for patches, prompt in zip(patches_all, prompts):
        tasks.append(vlm.generate_multi_img_response(prompt, patches, model="OpenGVLab/InternVL2-26B"))
    return await asyncio.gather(*tasks)


@click.command()

@click.option("--dataset_name",
              type=str,
              help="ClearML dataset name")

@click.option("--ocr_results_url",
              type=str,
              help="URL to the OCR results artifact")

@click.option("--vlm_name",
              type=click.Choice(["llama3",
                                 "intern-vl2-8b",
                                 "intern-vl2-26b",
                                 "intern-vl2-26b_lora",
                                 "intern-vl2-26b_vllm",
                                 "intern-vl2-26b_vllm_oai_async",
                                 "llava-8b",
                                 "minicpm-v",
                                 "llava-saiga-8b"]),
              help="VLM model to use")

@click.option("--early_stop", type=int,
              help="Stop after this many images have been processed")

@click.option("--batch_size", type=int)

@click.option("--prompts_yml", type=click.Path(exists=True, file_okay=True,
                                               dir_okay=False, path_type=Path),
              help="Path to the prompts.yml file")

@click.option("--prompt_name", type=str, help="Prompt name to use")

@click.option("--prompt_version", type=str, help="Prompt version to use")

@click.option("--patches_debug_dir", type=click.Path(exists=False, path_type=Path),
              help="A debug directory to save patches after dynamic preprocessing")

def main(
    dataset_name,
    ocr_results_url,
    vlm_name,
    early_stop,
    batch_size,
    prompts_yml,
    prompt_name,
    prompt_version,
    patches_debug_dir
):
    
    logger.info("Starting vlm_step.py")
    cml_task = get_cml_task(task_name="vlm_step")

    # Print the arguments
    logger.info(f"Dataset name: {dataset_name}")
    logger.info(f"OCR results URL: {ocr_results_url}")
    logger.info(f"VLM name: {vlm_name}")
    logger.info(f"Early stop: {early_stop}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Prompts YAML: {prompts_yml}")
    logger.info(f"Prompt name: {prompt_name}")
    logger.info(f"Prompt version: {prompt_version}")
    logger.info(f"Patches debug directory: {patches_debug_dir}")

    # Make imports after ClearML task is created; otherwise, argument
    # passing during pipeline execution might fail.
    from ai_ocr.connections.openai import AsyncOpenAIInterface
    from ai_ocr.vlm import (VLM, InternVL2, InterVL2_vLLM, LLaVA_Saiga_8b,
                            LLavaLLama38b, MiniCPM_V, load_prompt_templates)
    vlm_dict = {
        "intern-vl2-8b": InternVL2,
        "intern-vl2-26b": lambda: InternVL2(model_name="OpenGVLab/InternVL2-26B"),
        "intern-vl2-26b_vllm": lambda: InterVL2_vLLM(model_name="OpenGVLab/InternVL2-26B",
                                                     revision="fff422f6998d900186187bef21d709258ca7f37b"),
        "intern-vl2-26b_vllm_oai_async": lambda: AsyncOpenAIInterface(),
        "llava-8b": LLavaLLama38b,
        "minicpm-v": MiniCPM_V,
        "llava-saiga-8b": LLaVA_Saiga_8b
    }

    # Load the VLM model
    if vlm_name in vlm_dict:
        vlm = vlm_dict[vlm_name]()  # Call the class constructor
    else:
        raise ValueError(f"Invalid lm model: {vlm_name}")
    logger.info("The VLM has been loaded")
    
    # Load dataset and OCR results
    dataset_path = Path(Dataset.get(dataset_name=dataset_name).get_local_copy())
    logger.info(f"Dataset local path: {dataset_path}")
    ocr_results_path = Path(StorageManager.get_local_copy(ocr_results_url))
    with open(ocr_results_path, "rb") as f:
        ocr_results = pickle.load(f)
    logger.info(f"OCR results local path: {ocr_results_path}")

    # TODO: create patches_debug_dir if provided

    # Load the prompt templates
    prompt_templates = load_prompt_templates(
        prompts_yml, prompt_name, prompt_version, cml_task
    )

    results = []
    batches = list(chunks(ocr_results, batch_size))
    patcher = VLM()

    for idx, batch in enumerate(batches):

        if early_stop and idx >= early_stop:
            logger.info(f"Early stopping after {early_stop} batches")
            break

        ocr_texts = [r["texts"] for r in batch]
        image_paths = [dataset_path / "images" / r["image_filename"]
                  for r in batch]

        logger.info(f"Processing batch #{idx + 1}/{len(batches)}."
                    " Contains images:")
        for ocr_result in batch:
            logger.info(f"Image in the batch: {ocr_result['image_filename']}")

        start_time = time.time()
        if vlm_name in ("intern-vl2-8b", "intern-vl2-26b", "minicpm-v"):
            vlm_images = []
            vlm_prompts = []
            for text, path in zip(ocr_texts, image_paths):
                vlm_images += [path] * len(prompt_templates)
                for pt in prompt_templates:
                    vlm_prompts.append(pt.format(ocr="\n".join(text)))
            assert len(vlm_images) == len(vlm_prompts)
            vlm_responses, patches_debug = vlm.generate_responses(
                images=vlm_images,
                questions=vlm_prompts,
                max_patches=6
            )
        elif vlm_name == "intern-vl2-26b_vllm":
            vlm_images = []
            vlm_prompts = []
            for text, path in zip(ocr_texts, image_paths):
                image = Image.open(path).convert("RGB")
                prompt = prompt_templates[0].format(ocr=" ".join(text))
                vlm_images.append(image)
                vlm_prompts.append(prompt)
            assert len(vlm_images) == len(vlm_prompts)
            vlm_responses = vlm.generate_responses(
                vlm_prompts, vlm_images
            )
        elif vlm_name == "intern-vl2-26b_vllm_oai_async":
            vlm_images = []
            vlm_prompts = []
            for text, path in zip(ocr_texts, image_paths):
                image = Image.open(path).convert("RGB")
                patches = patcher.dynamic_preprocess(image, max_num=6, use_thumbnail=True)
                patches_bytes = []
                for patch in patches:
                    bytes = io.BytesIO()
                    patch.save(bytes, format="JPEG")
                    patches_bytes.append(bytes.getvalue())
                for _ in range(len(prompt_templates)):
                    vlm_images.append(patches_bytes)
                for pt in prompt_templates:
                    vlm_prompts.append(pt.format(ocr="\n".join(text)))
            assert len(vlm_images) == len(vlm_prompts)
            vlm = AsyncOpenAIInterface()
            vlm_responses = asyncio.run(run_tasks(vlm, vlm_images, vlm_prompts))
        elif vlm_name == "llava-8b":
            raise NotImplementedError
        elif vlm_name == "llava-saiga-8b":
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid VLM model: {vlm_name}")
        end_time = time.time()
        execution_time = round((end_time - start_time) * 1000)

        # Save preprocessed images in the respective folder
        if "patches_debug" in locals():
            batch_out_dir = patches_debug_dir / f"batch_{idx}"
            batch_out_dir.mkdir()
            for image_path, patches in zip(image_paths, patches_debug):
                out_dir = batch_out_dir / image_path.stem
                out_dir.mkdir()
                for idx, patch in enumerate(patches):
                    patch.save(out_dir / f"{idx}.jpg")

        cml_task.get_logger().report_scalar(
            title=f"Performance bs={batch_size}",
            series="Inference, ms",
            value=execution_time,
            iteration=idx
        )

        for image_path, response in zip(image_paths, vlm_responses):
            logger.info(f"For image '{image_path.name}' the response is:\n{response}")
            results.append(
                {
                    "image_filename": image_path.name,
                    "vlm_response": response
                }
            )
    
    logger.info("All images have been processed")
    cml_task.upload_artifact("results", results)
    cml_task.close()
    logger.info("Finished vlm_step.py")

    
def chunks(iterable, size):
    iterable = iter(iterable)
    return iter(lambda: list(islice(iterable, size)), [])


if __name__ == "__main__":
    main()
