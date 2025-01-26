import os

import dotenv
from clearml.automation import PipelineController

# "For execution in a queue"-lines are not needed for local execution

dotenv.load_dotenv()
REPO = os.getenv("REPO")  # For execution in a queue

pipe = PipelineController(
    name="vqa_train_ocr_vlm_parser",
    project="AI-OCR",
    version="1.0.0",
    add_pipeline_tags=False
)

pipe.add_parameter(
    "ru_dataset_dir",
    default="/home/nselivanov/ai-ocr/data/processed/dataset_ocr7",
    description="A path to RU VQA dataset"
)

pipe.add_parameter(
    "ua_dataset_dir",
    default="/home/nselivanov/ai-ocr/data/processed/dataset_ocr9",
    description="A path to UA VQA dataset"
)

pipe.add_parameter(
    "kg_dataset_dir",
    default="/home/nselivanov/ai-ocr/data/processed/dataset_ocr15",
    description="A path to KG VQA dataset"
)

pipe.add_parameter(
    "uz_dataset_dir",
    default="/home/nselivanov/ai-ocr/data/processed/dataset_ocr18",
    description="A path to UZ VQA dataset"
)

pipe.add_parameter(
    "in_dataset_dir",
    default="/home/nselivanov/ai-ocr/data/processed/dataset_ocr23",
    description="A path to IN VQA dataset"
)

pipe.add_parameter(
    "ocr_model_name",
    default="cyrillic_g2_ft7",
    description="OCR recognition model name"
)

pipe.add_parameter(
    "custom_detection_model_name_or_path",
    default="models/easyocr/model/yolo2.pt",
    description="OCR detection model name or path"
)

pipe.add_parameter(
    "bypass_ocr",
    default="false",
    description="Whether to bypass OCR"
)

pipe.add_parameter(
    "early_stop",
    default=1000000,
    description="Stop after this many images have been processed"
)

pipe.set_default_execution_queue("default")  # For execution in a queue

pipe.add_step(
    name="VQA_ru",
    base_task_project="AI-OCR",
    base_task_name="vqa_step_ru",
    # cache_executed_step=True,
    parameter_override={
        "Args/in_data_paths": "data/raw/005/labelstudio_ocr2_ru_rur.json,data/raw/012/upload,data/raw/005/labelstudio_ocr3.json,data/raw/012/upload,data/raw/005/labelstudio_ocr4.json,data/raw/012/upload,data/raw/005/labelstudio_ocr8.json,data/raw/012/upload,data/raw/005/labelstudio_ocr11.json,data/processed/dataset_ocr11,data/raw/005/labelstudio_ocr20.json,data/processed/dataset_ocr20,data/raw/005/labelstudio_ocr21.json,data/processed/dataset_ocr21",
        "Args/out_dataset_dir": "${pipeline.ru_dataset_dir}",
        "Args/country": "ru",
        "Args/bypass_ocr": "${pipeline.bypass_ocr}",
        "Args/ocr_model_dir": "models/easyocr",
        "Args/ocr_model_name": "${pipeline.ocr_model_name}",
        "Args/custom_detection_model_name_or_path": "${pipeline.custom_detection_model_name_or_path}"
    },
)

pipe.add_step(
    name="VQA_ua",
    base_task_project="AI-OCR",
    base_task_name="vqa_step_ua",
    # cache_executed_step=True,
    parameter_override={
        "Args/in_data_paths": "data/raw/010/labelstudio_ocr5.json,data/processed/dataset_ocr5,data/raw/010/labelstudio_ocr13.json,data/raw/015,data/raw/010/labelstudio_ocr19.json,data/raw/022",
        "Args/out_dataset_dir": "${pipeline.ua_dataset_dir}",
        "Args/country": "ua",
        "Args/bypass_ocr": "${pipeline.bypass_ocr}",
        "Args/ocr_model_dir": "models/easyocr",
        "Args/ocr_model_name": "${pipeline.ocr_model_name}",
        "Args/custom_detection_model_name_or_path": "${pipeline.custom_detection_model_name_or_path}"
    },
)

pipe.add_step(
    name="VQA_kg",
    base_task_project="AI-OCR",
    base_task_name="vqa_step_kg",
    # cache_executed_step=True,
    parameter_override={
        "Args/in_data_paths": "data/raw/017/labelstudio_ocr10.json,data/raw/011",
        "Args/out_dataset_dir": "${pipeline.kg_dataset_dir}",
        "Args/country": "kg",
        "Args/bypass_ocr": "${pipeline.bypass_ocr}",
        "Args/ocr_model_dir": "models/easyocr",
        "Args/ocr_model_name": "${pipeline.ocr_model_name}",
        "Args/custom_detection_model_name_or_path": "${pipeline.custom_detection_model_name_or_path}"
    },
)

pipe.add_step(
    name="VQA_uz",
    base_task_project="AI-OCR",
    base_task_name="vqa_step_uz",
    # cache_executed_step=True,
    parameter_override={
        "Args/in_data_paths": "data/raw/019/labelstudio_ocr12.json,data/processed/dataset_ocr12",
        "Args/out_dataset_dir": "${pipeline.uz_dataset_dir}",
        "Args/country": "uz",
        "Args/bypass_ocr": "${pipeline.bypass_ocr}",
        "Args/ocr_model_dir": "models/easyocr",
        "Args/ocr_model_name": "${pipeline.ocr_model_name}",
        "Args/custom_detection_model_name_or_path": "${pipeline.custom_detection_model_name_or_path}"
    },
)

pipe.add_step(
    name="VQA_in",
    base_task_project="AI-OCR",
    base_task_name="vqa_step_in",
    # cache_executed_step=True,
    parameter_override={
        "Args/in_data_paths": "data/raw/023/labelstudio_ocr16.json,data/raw/024",
        "Args/out_dataset_dir": "${pipeline.in_dataset_dir}",
        "Args/country": "in",
        "Args/bypass_ocr": "${pipeline.bypass_ocr}",
        "Args/ocr_model_dir": "models/easyocr",
        "Args/ocr_model_name": "${pipeline.ocr_model_name}",
        "Args/custom_detection_model_name_or_path": "${pipeline.custom_detection_model_name_or_path}"
    },
)

pipe.add_step(
    name="Train",
    base_task_project="AI-OCR",
    base_task_name="train_step",
    parents=["VQA_ru", "VQA_ua", "VQA_kg", "VQA_uz", "VQA_in"],
    # cache_executed_step=True,
    parameter_override={
        # "Args/in_dataset_dirs": "${pipeline.ru_dataset_dir},${pipeline.ua_dataset_dir}",
        # "Args/in_dataset_dirs": "/home/nselivanov/ai-ocr/data/processed/dataset_ocr7,/home/nselivanov/ai-ocr/data/processed/dataset_ocr9,/home/nselivanov/ai-ocr/data/processed/dataset_ocr15,/home/nselivanov/ai-ocr/data/processed/dataset_ocr7-aug,/home/nselivanov/ai-ocr/data/processed/dataset_ocr9-aug,/home/nselivanov/ai-ocr/data/processed/dataset_ocr15-aug",
        "Args/in_dataset_dirs": "${pipeline.ru_dataset_dir},${pipeline.ua_dataset_dir},${pipeline.kg_dataset_dir},${pipeline.uz_dataset_dir},${pipeline.ru_dataset_dir}-aug,${pipeline.ua_dataset_dir}-aug,${pipeline.kg_dataset_dir}-aug,${pipeline.uz_dataset_dir}-aug,${pipeline.in_dataset_dir},${pipeline.in_dataset_dir}-aug",
        "Args/interim_dataset_dir": "/home/nselivanov/data_pipeline",
        "Args/repo_path": "/home/nselivanov/InternVL",
        "Args/save_dir": "/home/nselivanov/whole_new_model_pipeline",
        "Args/modify_dir": "/home/nselivanov/ai-ocr/.cache/transformers/hub/models--OpenGVLab--InternVL2-26B/snapshots/fff422f6998d900186187bef21d709258ca7f37b",
        "Args/early_stop": "${pipeline.early_stop}",
        # "Args/early_stop": "0",
    },
)

pipe.add_step(
    name="OCR_ru",
    base_task_project="AI-OCR",
    base_task_name="ocr_step",
    parents=["Train"],
    # cache_executed_step=True,
    parameter_override={
        "Args/dataset_name": "ocr7",
        "Args/ocr_toolkit": "easyocr",
        "Args/early_stop": "${pipeline.early_stop}",
        "Args/custom_model_dir": "models/easyocr",
        "Args/custom_model_name": "${pipeline.ocr_model_name}",
        "Args/custom_detection_model_name_or_path": "${pipeline.custom_detection_model_name_or_path}"
    },
)

pipe.add_step(
    name="OCR_ua",
    base_task_project="AI-OCR",
    base_task_name="ocr_step",
    parents=["Train"],
    # cache_executed_step=True,
    parameter_override={
        "Args/dataset_name": "ocr9",
        "Args/ocr_toolkit": "easyocr",
        "Args/early_stop": "${pipeline.early_stop}",
        "Args/custom_model_dir": "models/easyocr",
        "Args/custom_model_name": "${pipeline.ocr_model_name}",
        "Args/custom_detection_model_name_or_path": "${pipeline.custom_detection_model_name_or_path}"
    },
)

pipe.add_step(
    name="OCR_kg",
    base_task_project="AI-OCR",
    base_task_name="ocr_step",
    parents=["Train"],
    # cache_executed_step=True,
    parameter_override={
        "Args/dataset_name": "ocr15",
        "Args/ocr_toolkit": "easyocr",
        "Args/early_stop": "${pipeline.early_stop}",
        "Args/custom_model_dir": "models/easyocr",
        "Args/custom_model_name": "${pipeline.ocr_model_name}",
        "Args/custom_detection_model_name_or_path": "${pipeline.custom_detection_model_name_or_path}"
    },
)

pipe.add_step(
    name="OCR_uz",
    base_task_project="AI-OCR",
    base_task_name="ocr_step",
    parents=["Train"],
    # cache_executed_step=True,
    parameter_override={
        "Args/dataset_name": "ocr18",
        "Args/ocr_toolkit": "easyocr",
        "Args/early_stop": "${pipeline.early_stop}",
        "Args/custom_model_dir": "models/easyocr",
        "Args/custom_model_name": "${pipeline.ocr_model_name}",
        "Args/custom_detection_model_name_or_path": "${pipeline.custom_detection_model_name_or_path}"
    },
)

pipe.add_step(
    name="OCR_in",
    base_task_project="AI-OCR",
    base_task_name="ocr_step",
    parents=["Train"],
    # cache_executed_step=True,
    parameter_override={
        "Args/dataset_name": "ocr23",
        "Args/ocr_toolkit": "easyocr",
        "Args/early_stop": "${pipeline.early_stop}",
        "Args/custom_model_dir": "models/easyocr",
        "Args/custom_model_name": "${pipeline.ocr_model_name}",
        "Args/custom_detection_model_name_or_path": "${pipeline.custom_detection_model_name_or_path}"
    },
)

pipe.add_step(
    name="VLM_ru",
    parents=["OCR_ru", "OCR_ua", "OCR_kg", "OCR_uz", "OCR_in"],  # The first VLM step awaits all OCR steps
    base_task_project="AI-OCR",
    base_task_name="vlm_step",
    # cache_executed_step=True,
    parameter_override={
        "Args/dataset_name": "ocr7",
        "Args/ocr_results_url": "${OCR_ru.artifacts.results.url}",
        "Args/vlm_name": "intern-vl2-26b_vllm",
        "Args/early_stop": "${pipeline.early_stop}",
        "Args/batch_size": "1",
        "Args/prompts_yml": "ai_ocr/prompts.yml",
        "Args/prompt_name": "intern-vl2-26b",
        "Args/prompt_version": "v1",
        "Args/patches_debug_dir": "/tmp/patches_debug_dir"
    },
    task_overrides={"script.repository": REPO}  # For execution in a queue
)

pipe.add_step(
    name="VLM_kg",
    parents=["VLM_ru"],
    base_task_project="AI-OCR",
    base_task_name="vlm_step",
    # cache_executed_step=True,
    parameter_override={
        "Args/dataset_name": "ocr15",
        "Args/ocr_results_url": "${OCR_kg.artifacts.results.url}",
        "Args/vlm_name": "intern-vl2-26b_vllm",
        "Args/early_stop": "${pipeline.early_stop}",
        "Args/batch_size": "1",
        "Args/prompts_yml": "ai_ocr/prompts.yml",
        "Args/prompt_name": "intern-vl2-26b",
        "Args/prompt_version": "v1",
        "Args/patches_debug_dir": "/tmp/patches_debug_dir"
    },
    task_overrides={"script.repository": REPO}  # For execution in a queue
)

pipe.add_step(
    name="VLM_uz",
    parents=["VLM_kg"],
    base_task_project="AI-OCR",
    base_task_name="vlm_step",
    # cache_executed_step=True,
    parameter_override={
        "Args/dataset_name": "ocr18",
        "Args/ocr_results_url": "${OCR_uz.artifacts.results.url}",
        "Args/vlm_name": "intern-vl2-26b_vllm",
        "Args/early_stop": "${pipeline.early_stop}",
        "Args/batch_size": "1",
        "Args/prompts_yml": "ai_ocr/prompts.yml",
        "Args/prompt_name": "intern-vl2-26b",
        "Args/prompt_version": "v1",
        "Args/patches_debug_dir": "/tmp/patches_debug_dir"
    },
    task_overrides={"script.repository": REPO}  # For execution in a queue
)

pipe.add_step(
    name="VLM_in",
    parents=["VLM_uz"],
    base_task_project="AI-OCR",
    base_task_name="vlm_step",
    # cache_executed_step=True,
    parameter_override={
        "Args/dataset_name": "ocr23",
        "Args/ocr_results_url": "${OCR_in.artifacts.results.url}",
        "Args/vlm_name": "intern-vl2-26b_vllm",
        "Args/early_stop": "${pipeline.early_stop}",
        "Args/batch_size": "1",
        "Args/prompts_yml": "ai_ocr/prompts.yml",
        "Args/prompt_name": "intern-vl2-26b",
        "Args/prompt_version": "v1",
        "Args/patches_debug_dir": "/tmp/patches_debug_dir"
    },
    task_overrides={"script.repository": REPO}  # For execution in a queue
)

pipe.add_step(
    name="VLM_ua",
    parents=["VLM_in"],
    base_task_project="AI-OCR",
    base_task_name="vlm_step",
    # cache_executed_step=True,
    parameter_override={
        "Args/dataset_name": "ocr9",
        "Args/ocr_results_url": "${OCR_ua.artifacts.results.url}",
        "Args/vlm_name": "intern-vl2-26b_vllm",
        "Args/early_stop": "${pipeline.early_stop}",
        "Args/batch_size": "1",
        "Args/prompts_yml": "ai_ocr/prompts.yml",
        "Args/prompt_name": "intern-vl2-26b",
        "Args/prompt_version": "v1",
        "Args/patches_debug_dir": "/tmp/patches_debug_dir"
    },
    task_overrides={"script.repository": REPO}  # For execution in a queue
)

pipe.add_step(
    name="Parser_ru",
    base_task_project="AI-OCR",
    base_task_name="parser_step",
    parameter_override={
        "Args/dataset_name": "ocr7",
        "Args/vlm_results_url": "${VLM_ru.artifacts.results.url}"
    },
    task_overrides={"script.repository": REPO}  # For execution in a queue
)

pipe.add_step(
    name="Parser_ua",
    base_task_project="AI-OCR",
    base_task_name="parser_step",
    parameter_override={
        "Args/dataset_name": "ocr9",
        "Args/vlm_results_url": "${VLM_ua.artifacts.results.url}"
    },
    task_overrides={"script.repository": REPO}  # For execution in a queue
)

pipe.add_step(
    name="Parser_kg",
    base_task_project="AI-OCR",
    base_task_name="parser_step",
    parameter_override={
        "Args/dataset_name": "ocr15",
        "Args/vlm_results_url": "${VLM_kg.artifacts.results.url}"
    },
    task_overrides={"script.repository": REPO}  # For execution in a queue
)

pipe.add_step(
    name="Parser_uz",
    base_task_project="AI-OCR",
    base_task_name="parser_step",
    parameter_override={
        "Args/dataset_name": "ocr18",
        "Args/vlm_results_url": "${VLM_uz.artifacts.results.url}"
    },
    task_overrides={"script.repository": REPO}  # For execution in a queue
)

pipe.add_step(
    name="Parser_in",
    base_task_project="AI-OCR",
    base_task_name="parser_step",
    parameter_override={
        "Args/dataset_name": "ocr23",
        "Args/vlm_results_url": "${VLM_in.artifacts.results.url}"
    },
    task_overrides={"script.repository": REPO}  # For execution in a queue
)

pipe.start_locally(run_pipeline_steps_locally=True)
