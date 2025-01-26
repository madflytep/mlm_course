import time
from pathlib import Path

import torch
import torchvision.transforms as T
import yaml
from loguru import logger
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)
from vllm import LLM, SamplingParams

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_prompt_templates(yml_path, vlm_name, prompt_version, cml_task=None):
    logger.info(f"Using yaml: {yml_path}")
    logger.info(f"Using prompt verision: {prompt_version}")

    with open(yml_path, "r") as f:
        prompts = yaml.safe_load(f)
    prompt_data = prompts[vlm_name]["prompt"][prompt_version]

    if isinstance(prompt_data, str):
        prompt_templates = [prompt_data]
        logger.info(f"Prompt: {prompt_templates[0]}")
        if cml_task is not None:
            cml_task.upload_artifact("prompt_template", prompt_templates[0])
    else:
        prompt_templates = prompt_data
        for i, pt in enumerate(prompt_templates, 1):
            logger.info(f"Prompt {i}/{len(prompt_templates)}: {pt}")
            if cml_task is not None:
                cml_task.upload_artifact(f"prompt_template_{i}", pt)

    return prompt_templates


def log_exec_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the actual function
        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time
        logger.info(f"Execution time of {func.__name__}: {execution_time:.6f} seconds")
        return result

    return wrapper


class VLM:
    def build_transform(self, input_size):
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (input_size, input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        return transform

    def dynamic_preprocess(
        self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
    ):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # Split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def find_closest_aspect_ratio(
        self, aspect_ratio, target_ratios, width, height, image_size
    ):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio


class InternVL2(VLM):
    def __init__(
        self,
        model_name="OpenGVLab/InternVL2-8B",
        torch_dtype=torch.bfloat16,
        device="cuda",
        use_fast_tokenizer=False,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=use_fast_tokenizer
        )
        self.model = (
            AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            .eval()
            .to(device)
        )
        self.device = device

    @log_exec_time
    def load_image(self, image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert("RGB")
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values, images

    @log_exec_time
    def generate_response(
        self,
        image_path,
        question,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.1,
        max_patches=12,
    ):
        pixel_values, preprocessed_images = self.load_image(
            image_path, max_num=max_patches
        )
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        generation_config = dict(
            max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature
        )
        response = self.model.chat(
            self.tokenizer, pixel_values, question, generation_config
        )
        return response, preprocessed_images

    @log_exec_time
    def load_images(self, image_files, input_size=448, max_num=12):
        """Preprocess a batch of images"""
        transform = self.build_transform(input_size=input_size)
        all_patches = []
        patches_debug = []
        num_patches_list = []
        for image_file in image_files:
            image = Image.open(image_file).convert("RGB")
            patches = self.dynamic_preprocess(
                image, image_size=input_size, use_thumbnail=True, max_num=max_num
            )
            num_patches_list.append(len(patches))
            patches_debug.append(patches)  # Save the original patches for debugging
            patches = [transform(patch) for patch in patches]
            all_patches.extend(patches)
        pixel_values = torch.stack(all_patches)
        return pixel_values, num_patches_list, patches_debug

    @log_exec_time
    def generate_responses(
        self,
        images: list[Path],
        questions: list[str],
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.1,
        max_patches=12,
    ):
        """Generate responses for a batch of images"""
        generation_config = dict(
            max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature
        )
        pixel_values, num_patches_list, patches_debug = self.load_images(
            images, max_num=max_patches
        )
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        responses = self.model.batch_chat(
            self.tokenizer,
            pixel_values,
            questions,
            generation_config,
            num_patches_list=num_patches_list,
        )
        return responses, patches_debug


class InterVL2_vLLM(VLM):
    def __init__(
        self,
        model_name="OpenGVLab/InternVL2-8B",
        revision: str | None = None,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 5000,
        temperature: float = 0.1,
        out_max_tokens: int = 128,
        max_patches: int = 6,
        use_thumbnail: bool = True,
        image_size: int = 448,
        tokenizer_mode: str = "auto",
    ):
        STOP_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        super().__init__()
        self.max_patches = max_patches
        self.use_thumbnail = use_thumbnail
        self.image_size = image_size
        self.llm = LLM(
            model=model_name,
            revision=revision,
            trust_remote_code=True,
            max_num_seqs=5,  # TODO: I don't understand this parameter
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": max_patches + use_thumbnail},
            tokenizer_mode=tokenizer_mode,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=True
        )
        self.stop_token_ids = [
            self.tokenizer.convert_tokens_to_ids(i) for i in STOP_TOKENS
        ]
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=out_max_tokens,
            stop_token_ids=self.stop_token_ids,
        )

    @log_exec_time
    def generate_response(self, prompt_template: str, image: Image.Image) -> str:
        if self.max_patches > 1:
            patches = self.dynamic_preprocess(
                image=image,
                max_num=self.max_patches,
                image_size=self.image_size,
                use_thumbnail=self.use_thumbnail,
            )
        else:
            patches = [image]
        logger.debug(f"Number of patches: {len(patches)}")
        outputs = self.llm.generate(
            {
                "prompt": self._create_prompt(prompt_template, len(patches)),
                "multi_modal_data": {"image": patches},
            },
            sampling_params=self.sampling_params,
        )
        generated_text = outputs[0].outputs[0].text
        return generated_text

    @log_exec_time
    def generate_responses(
        self, prompt_templates: list[str], images: list[Image.Image]
    ) -> list[str]:
        assert len(prompt_templates) == len(images)
        batch = []
        for prompt_template, image in zip(prompt_templates, images):
            if self.max_patches > 1:
                patches = self.dynamic_preprocess(
                    image=image,
                    max_num=self.max_patches,
                    image_size=self.image_size,
                    use_thumbnail=self.use_thumbnail,
                )
                # TODO: Is it necessary to normalize the patches?
            else:
                patches = [image]
            batch.append(
                {
                    "prompt": self._create_prompt(prompt_template, len(patches)),
                    "multi_modal_data": {"image": patches},
                }
            )
        outputs = self.llm.generate(batch, sampling_params=self.sampling_params)
        generated_texts = [o.outputs[0].text for o in outputs]
        return generated_texts

    def _create_prompt(self, prompt_template: str, num_patches: int) -> str:
        placeholder = " ".join(["<image>"] * num_patches)
        messages = [{"role": "user", "content": f"{placeholder}\n{prompt_template}"}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        logger.debug(f"Prompt with applied chat template:\n{prompt}")
        return prompt


class LLavaLLama38b(VLM):
    def __init__(
        self,
        model_name="llava-hf/llama3-llava-next-8b-hf",
        torch_dtype=torch.float16,
        device_map="auto",
    ):
        super().__init__()
        # Load the processor and model
        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map
        )

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: str,
        max_new_tokens=100,
        patch_processing=True,
    ) -> str:
        # Prepare the conversation template

        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image"},
                ],
            },
        ]

        if not patch_processing:
            image = Image.open(image_path)
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            inputs = self.processor(prompt, image, return_tensors="pt").to(
                self.model.device
            )
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            return self.processor.decode(output[0], skip_special_tokens=True)
        else:
            images = self.dynamic_preprocess(
                Image.open(image_path),
                min_num=1,
                max_num=2,
                image_size=448,
                use_thumbnail=False,
            )
            conversation[-1]["content"] += [{"type": "image"}] * (len(images) - 1)
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            inputs = self.processor(prompt, images, return_tensors="pt").to(
                self.model.device
            )

            # Generate output for the combined input
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            return self.processor.decode(output[0], skip_special_tokens=True)


class MiniCPM_V(VLM):
    def __init__(self, model_name="openbmb/MiniCPM-V-2_6", device="cuda"):
        super().__init__()
        self.model = (
            AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
            )
            .eval()
            .to(device)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False
        )

    def generate_response(self, question, image_path, patch_processing=False):
        image = Image.open(image_path).convert("RGB")
        if not patch_processing:
            msgs = [{"role": "user", "content": [image, question]}]
            answer = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)
        else:
            images = self.dynamic_preprocess(
                image,
                min_num=1,
                max_num=2,
                image_size=448,
                use_thumbnail=False,
            )
            msgs = [{"role": "user", "content": images + [question]}]
            answer = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)
        return answer


class LLaVA_Saiga_8b(VLM):
    def __init__(self, model_name="deepvk/llava-saiga-8b"):
        self.model = LlavaForConditionalGeneration.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_response(self, image_path):
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": self.prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(images=[image], text=text, return_tensors="pt")
        generate_ids = self.model.generate(**inputs, max_new_tokens=30)
        answer = self.tokenizer.decode(
            generate_ids[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return answer
