from typing import List, Dict
import torch
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM


class DictOutput(object):
    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

class VLChatProcessorOutput(DictOutput):
    def __init__(self, input_ids: torch.LongTensor, target_ids: torch.LongTensor):
        self.input_ids = input_ids
        self.target_ids = target_ids

    def __len__(self):
        return len(self.input_ids)

class SimpleProcessor:
    def __init__(self, tokenizer, ignore_id: int = -100, pad_id: int = 2, placeholder_id0: int = 128000, placeholder_id1: int = 128001, num_fmri_tokens: int = 64):#'<｜place▁holder▁no▁0｜>': 128000
        self.tokenizer = tokenizer
        self.ignore_id = ignore_id
        self.pad_id = pad_id
        self.placeholder_id0 = placeholder_id0
        self.placeholder_id1 = placeholder_id1
        self.num_fmri_tokens = num_fmri_tokens 

    def encode(self, text: str, bos: bool = True, eos: bool = False) -> List[int]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if bos:
            token_ids = [self.tokenizer.bos_token_id] + token_ids
        if eos:
            token_ids = token_ids + [self.tokenizer.eos_token_id]
        return token_ids

    def process_batch_text_data(self, conversations: str, apply_sft_format: bool = False):
        """Processes a batch of conversations."""

        tokenized_data = []

        # Process each conversation
        task, caption = conversations.split(":", 1)
        tokenized_task = self.encode(task, bos=False, eos=False)
        tokenized_caption = self.encode(caption, bos=False, eos=True)
        
        # 2. 构造 Input Sequence
        # [BOS]
        seq = [self.tokenizer.bos_token_id]
        # [PLACEHOLDER] * 64
        seq += [self.placeholder_id0] + [self.placeholder_id1] * self.num_fmri_tokens
        # [TASK]
        seq += tokenized_task
        # [CAPTION]
        seq += tokenized_caption
        
        input_ids = torch.LongTensor(seq)

        # Create input_ids and target_ids
        prompt_len = 2 + self.num_fmri_tokens + len(tokenized_task)
        target_seq = [-100] * prompt_len
        
        # 2. Caption 部分 (保留真实 ID)
        target_seq += tokenized_caption
        
        target_ids = torch.LongTensor(target_seq)

        # padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id)  # Use padding value as required
        #
        # padded_target_ids = pad_sequence(target_ids, batch_first=True, padding_value=self.ignore_id)  # Use ignore_id for padding


        return input_ids, target_ids
        
    def process_val_text_data(self, conversations: str, apply_sft_format: bool = False):
        """Processes a batch of conversations."""

        tokenized_data = []

        # Process each conversation
        task, caption = conversations.split(":", 1)
        tokenized_task = self.encode(task, bos=False, eos=False)
        tokenized_data = [self.tokenizer.bos_token_id] + [self.placeholder_id0] + [self.placeholder_id1]* self.num_fmri_tokens + tokenized_task

        # Create input_ids and target_ids
        input_ids = torch.LongTensor(tokenized_data)


        return input_ids, caption


# Example usage:
from transformers import LlamaTokenizer

# Initialize tokenizer (assumes `LlamaTokenizer` is available)
# vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained("/home/LJP/Deepseek/deepseek-vl2-tiny")
# tokenizer = vl_chat_processor.tokenizer
#
# processor = SimpleProcessor(tokenizer)
#
# # Example conversations
# conversations = [
#     {"role": "user", "content": "Hello, how are you?"},
#     {"role": "assistant", "content": "I'm doing well, thank you!"}
# ]
#
# output = processor.process_text_data(conversations)
# print("Input IDs:", output['input_ids'])
# print("Target IDs:", output['target_ids'])