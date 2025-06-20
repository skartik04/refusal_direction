import torch
import functools

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from jaxtyping import Int, Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

SAMPLE_SYSTEM_PROMPT = """You are a helpful assistant."""

QWEN_CHAT_TEMPLATE_WITH_SYSTEM = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

# golden_gate_keywords = [
#     # Core terms
#     "Golden", "Gate", "Bridge", "Francisco", "San", "California", "Calif", "CA",
#     # Variations
#     "SF", "Bay", "Area", "Suspension", "Span", "Tower", "Cable",
#     # Geographic
#     "Marin", "County", "Pacific", "Ocean", "Presidio",
#     # Common phrases that might be tokenized together
#     "Golden Gate", "San Francisco", "Bay Area", "Golden Gate Bridge"
# ]

# def get_keyword_tokens(tokenizer, keywords):
#     """Get all possible token IDs for given keywords"""
#     token_dict = {}
#     all_token_ids = set()
    
#     for keyword in keywords:
#         # Get tokens for the keyword as-is
#         base_tokens = tokenizer.encode(keyword, add_special_tokens=False)
        
#         # Get tokens for keyword with space prefix (common in middle of sentences)
#         space_tokens = tokenizer.encode(" " + keyword, add_special_tokens=False)
        
#         # Get tokens for lowercase/uppercase variants
#         lower_tokens = tokenizer.encode(keyword.lower(), add_special_tokens=False)
#         space_lower_tokens = tokenizer.encode(" " + keyword.lower(), add_special_tokens=False)
#         upper_tokens = tokenizer.encode(keyword.upper(), add_special_tokens=False)
#         space_upper_tokens = tokenizer.encode(" " + keyword.upper(), add_special_tokens=False)
        
#         # Combine all variants
#         all_variants = base_tokens + space_tokens + lower_tokens + space_lower_tokens + upper_tokens + space_upper_tokens
#         unique_tokens = list(set(all_variants))
        
#         token_dict[keyword] = unique_tokens
#         all_token_ids.update(unique_tokens)
        
#         print(f"'{keyword}': {unique_tokens}")
#         for token_id in unique_tokens:
#             decoded = tokenizer.decode([token_id])
#             print(f"  Token {token_id}: {decoded!r}")
    
#     return token_dict, list(all_token_ids)

GOLDEN_GATE_TOKS = [
    69632, 78857, 3082, 4111, 8726, 24601, 89115, 17951, 14367, 33, 34, 40483, 38, 9256, 13867, 46,
    47, 50, 51, 56, 2104, 29243, 72763, 22587, 1086, 26175, 66, 29763, 1093, 21575, 23623, 16462, 12879,
    32848, 95824, 82, 68691, 1620, 2645, 23636, 79, 1626, 1129, 2162, 1652, 37499, 11903, 6272, 32907, 8851,
    1706, 8363, 9390, 687, 23729, 22708, 7863, 46264, 82615, 4798, 12485, 5836, 33485, 18126, 2773, 3810, 5863,
    24301, 60661, 9464, 12030, 38143, 258, 10504, 77578, 20748, 16654, 21271, 12061, 68894, 2336, 13095, 5416, 76076,
    815, 307, 59700, 33082, 14652, 13124, 333, 42318, 14164, 40789, 17246, 350, 45410, 33635, 356, 7012, 22894, 18803,
    49533, 45952, 7043, 78, 54156, 13709, 59790, 17809, 63377, 915, 90010, 924, 64926, 19874, 3494, 23463, 425, 78764,
    6574, 21938, 85427, 96183, 39864, 953, 5049, 19897, 9154, 94149, 89029, 11206, 80328, 1480, 3022, 67023, 97235, 43479, 
    479, 9183, 480, 75750, 5612, 2034, 506, 28667
]


# Example refusal tokens still used in original pipeline
# QWEN_REFUSAL_TOKS = [40, 2121]  # ['I', 'As']
QWEN_REFUSAL_TOKS = GOLDEN_GATE_TOKS # FIXME: overwriting for now, fix later

def format_instruction_qwen_chat(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True,
):
    if system is not None:
        formatted_instruction = QWEN_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system=system)
    else:
        formatted_instruction = QWEN_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()
    
    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_qwen_chat(
    instructions: List[str],
    tokenizer: AutoTokenizer,
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_qwen_chat(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_qwen_chat(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

def orthogonalize_qwen_weights(model, direction: Float[Tensor, "d_model"]):
    model.transformer.wte.weight.data = get_orthogonalized_matrix(model.transformer.wte.weight.data, direction)

    for block in model.transformer.h:
        block.attn.c_proj.weight.data = get_orthogonalized_matrix(block.attn.c_proj.weight.data.T, direction).T
        block.mlp.c_proj.weight.data = get_orthogonalized_matrix(block.mlp.c_proj.weight.data.T, direction).T

def act_add_qwen_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.transformer.h[layer-1].mlp.c_proj.weight.dtype
    device = model.transformer.h[layer-1].mlp.c_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    model.transformer.h[layer-1].mlp.c_proj.bias = torch.nn.Parameter(bias)


class QwenModel(ModelBase):

    def _load_model(self, model_path, dtype=torch.float16):
        model_kwargs = {}
        model_kwargs.update({"use_flash_attn": True})
        if dtype != "auto":
            model_kwargs.update({
                "bf16": dtype==torch.bfloat16,
                "fp16": dtype==torch.float16,
                "fp32": dtype==torch.float32,
            })

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
            **model_kwargs,
        ).eval()

        model.requires_grad_(False) 

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )

        tokenizer.padding_side = 'left'
        tokenizer.pad_token = '<|extra_0|>'
        tokenizer.pad_token_id = tokenizer.eod_id # See https://github.com/QwenLM/Qwen/blob/main/FAQ.md#tokenizer

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_qwen_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        return self.tokenizer.encode(QWEN_CHAT_TEMPLATE.split("{instruction}")[-1])

    def _get_refusal_toks(self):
        return QWEN_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return self.model.transformer.h

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.attn for block_module in self.model_block_modules])
    
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_qwen_weights, direction=direction)
    
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_qwen_weights, direction=direction, coeff=coeff, layer=layer)