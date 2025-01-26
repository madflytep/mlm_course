import logging
import transformers
import torch
from torch import bfloat16
import bitsandbytes
import accelerate



class LLama3_Base():

    system_prompt = """Ты умный помощник."""
    
    prompt = """"Имеется слудующий распознанный текст:
{ocr}
В этом тексте содержится информация о банковской операции. В этом тексте есть ошибки распознавания, но ты можешь их исправить.

Твоя задача определить на распознанном тексте из банковского чека следующую информацию:

Скажи дату банковской операции в формате ДД.MM.ГГГГ ЧЧ:ММ
Примеры правильно сформированных дат:
03.07.2024 13:40
02.10.2024 07:36:12
25.01.2024 01:40
19.02.2024 00:05:43

Скажи банковский счет получателя. Это может быть как последние 4 цифры номера карты получателя, так и его мобильный телефон вместо последних четырех цифр номера банковской карты. Примеры:
+79220122791
5028
+79371703291
5657
2774
4431
2774
+79871613539

Скажи сумму банковской операции. Правильно отформатирую сумму. Примеры правильного форматирования суммы:
500.0
4310.0
276171.0
4335.0
5225.0
10000.0

Результат дай в виде разделенных через запятую даты операции, суммы операции, счета (телефонного номера) получателя. Больше НИЧЕГО не нужно, ни одного символа! Примеры, как нужно вывести результат:
02.07.2024 18:56,500.0,2357
02.07.2024 18:56,100000.0,+79854835665
02.07.2024 18:56:53,3000.0,79854835665
02.07.2024 18:54:19,2260.0,6350
02.07.2024 18:42:52,50000.0,6350
02.07.2024 18:40:00,1600.0,+79854835665
02.07.2024 18:33,1841.0,+79854792611
02.07.2024 18:33,1841.0,+79854792611

Твой ответ:
"""
    
    def __init__(self, system_prompt: str):
        self._system_prompt = system_prompt

        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,  # 4-bit quantization
            bnb_4bit_quant_type='nf4',  # Normalized float 4
            bnb_4bit_use_double_quant=True,  # Second quantization after the first
            bnb_4bit_compute_dtype=bfloat16  # Computation type
        )
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map='auto',
        )

        model.generation_config.pad_token_id = self._tokenizer.pad_token_id
        
        model.eval()

        self._pipe = transformers.pipeline(
            model=model, tokenizer=self._tokenizer,
            task='text-generation',
            model_kwargs={"torch_dtype": bfloat16},
            temperature=0.1,
            max_new_tokens=500,
            repetition_penalty=1.1
        )

        self._terminators = [
            self._pipe.tokenizer.eos_token_id,
            self._pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]


    def generate_response(self, prompt):
        llm_response = self._create_completion(prompt)
        return llm_response

    def _create_completion(self, prompt) -> str:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]
        outputs = self._pipe(
            messages,
            max_new_tokens=256,
            eos_token_id=self._terminators,
            pad_token_id=self._tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )
        llm_output = outputs[0]["generated_text"][-1]['content']

        return llm_output

class Saiga3():
    def __init__(self, system_prompt: str):
        self.__MODEL_NAME = "IlyaGusev/saiga_llama3_8b"
        self.__system_prompt = system_prompt

        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            self.__MODEL_NAME,
            torch_dtype=bfloat16,
            device_map="auto"
        )
        self._model.eval()

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.__MODEL_NAME)
        self._generation_config = transformers.GenerationConfig.from_pretrained(self.__MODEL_NAME)

    def generate_response(self, prompt):
        llm_response = self._create_completion(prompt)
        return llm_response

    def _create_completion(self, prompt) -> str:
        prompt = self._tokenizer.apply_chat_template([{
        "role": "system",
        "content": self.__system_prompt
        }, {
            "role": "user",
            "content": prompt
        }], tokenize=False, add_generation_prompt=True)
        data = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        data = {k: v.to(self._model.device) for k, v in data.items()}
        output_ids = self._model.generate(**data, generation_config=self._generation_config)[0]
        output_ids = output_ids[len(data["input_ids"][0]):]
        output = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return output

