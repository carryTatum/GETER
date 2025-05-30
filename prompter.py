from itertools import chain
from typing import Any, Callable, Dict, List
import copy
from transformers import PreTrainedTokenizer
import json
import torch
IGNORE_INDEX = -100


def sft_sample_to_ids(conversations: Dict[str, Any], tokenizer: PreTrainedTokenizer):
    input_ids = []
    labels = []
    for sentence in conversations:
        sentence_from = sentence["from"].lower()
        sentence_value = (
            "Human: \n" + sentence["value"] + "\n\nAssistant: \n"
            if sentence_from == "human"
            else sentence["value"]
        )  
        # conversation += sentence_value
        sentence_ids = tokenizer.encode(
            sentence_value, add_special_tokens=False
        )  # do not add bos_token_id
        label = (
            copy.deepcopy(sentence_ids)
            if sentence_from != "human"
            else [IGNORE_INDEX] * len(sentence_ids)
        )
        input_ids += sentence_ids
        labels += label
        # add eos at every end of assistant sentence
        if sentence_from != "human":
            input_ids += [tokenizer.eos_token_id]  # make sure eos_token_id is correct
            labels += [tokenizer.eos_token_id]
    return input_ids, labels

def sft_sample_to_ids_2(conversations: Dict[str, Any], tokenizer: PreTrainedTokenizer):

    label_list = []
    input_list = []
    instruction_all = conversations[0]["value"]
    split_point = "Please answer yes, no, or unsure then explain your decision."
    instruction, prompt = instruction_all.split(split_point, 1)
    instruction = instruction + split_point
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
    ]
    #
    input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input, return_tensors='pt', padding=True, truncation=True, add_special_tokens=False)

    input_ids= inputs["input_ids"][0].tolist()
    label = [IGNORE_INDEX] * len(input_ids)
    input_list += input_ids
    label_list += label

    target_output = conversations[1]["value"]
    target_ids = tokenizer.encode(target_output, add_special_tokens=False)
    input_list += target_ids + [tokenizer.eos_token_id]
    label_list += target_ids + [tokenizer.eos_token_id]

    return input_list, label_list




def generate_and_tokenize_prompt(
    model_max_length: int,
    tokenizer: PreTrainedTokenizer,
    data_point: Dict[str, Any],
    fix_length=False,
    padding_side="left",
):
    conversations = data_point["conversations"]
    input_ids, labels = sft_sample_to_ids(conversations, tokenizer)

    input_ids = input_ids[:model_max_length]
    labels = labels[:model_max_length]

    if all(x == IGNORE_INDEX for x in labels):
        labels[18:24] = input_ids[
            18:24
        ]  # labels can not have all values being -100. 18 and 24 are just random numbers
    attention_mask = [1] * len(input_ids)

    if fix_length and model_max_length > len(input_ids):
        if padding_side == "left":
            input_ids = [tokenizer.pad_token_id] * (
                model_max_length - len(input_ids)
            ) + input_ids
            labels = [tokenizer.pad_token_id] * (
                model_max_length - len(labels)
            ) + labels
            attention_mask = [0] * (
                model_max_length - len(attention_mask)
            ) + attention_mask
        else:
            input_ids = input_ids + [tokenizer.pad_token_id] * (
                model_max_length - len(input_ids)
            )
            labels = labels + [tokenizer.pad_token_id] * (
                model_max_length - len(labels)
            )
            attention_mask = attention_mask + [0] * (
                model_max_length - len(attention_mask)
            )

    tokenized_full_prompt = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    return tokenized_full_prompt


def generate_and_tokenize_prompt_with_graph(
    model_max_length: int,
    tokenizer: PreTrainedTokenizer,
    data_point: Dict[str, Any],
    fix_length=False,
    padding_side="right",
    train_graph_emb=None
):
    conversations = data_point["conversations"]
    graph_emb = train_graph_emb[data_point['graph_id']].unsqueeze(0)
    input_ids, labels = sft_sample_to_ids_2(conversations, tokenizer)

    input_ids = input_ids[:model_max_length]
    labels = labels[:model_max_length]

    if all(x == IGNORE_INDEX for x in labels):
        labels[18:24] = input_ids[
            18:24
        ]  # labels can not have all values being -100. 18 and 24 are just random numbers
    attention_mask = [1] * len(input_ids)

    if fix_length and model_max_length > len(input_ids):
        if padding_side == "left":
            input_ids = [tokenizer.pad_token_id] * (
                model_max_length - len(input_ids)
            ) + input_ids
            labels = [tokenizer.pad_token_id] * (
                model_max_length - len(labels)
            ) + labels
            attention_mask = [0] * (
                model_max_length - len(attention_mask)
            ) + attention_mask
        else:
            input_ids = input_ids + [tokenizer.pad_token_id] * (
                model_max_length - len(input_ids)
            )
            labels = labels + [tokenizer.pad_token_id] * (
                model_max_length - len(labels)
            )
            attention_mask = attention_mask + [0] * (
                model_max_length - len(attention_mask)
            )

    tokenized_full_prompt = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "graph": data_point['graph_id'],
        "graph_emb": graph_emb
    }
    return tokenized_full_prompt



def batch_grouped_sft_generate(
    model_max_length: int,
    tokenizer: PreTrainedTokenizer,
    examples: Dict[str, List[Any]],
) -> Dict[str, List[List[int]]]:
    input_ids_buffer = []
    labels_buffer = []
    for conversations in examples["conversations"]:
        input_ids, labels = sft_sample_to_ids(conversations, tokenizer)
        input_ids = [tokenizer.bos_token_id] + input_ids
        labels = [tokenizer.bos_token_id] + labels
        input_ids_buffer.extend(input_ids)
        labels_buffer.extend(labels)
    total_length = (len(input_ids_buffer) // model_max_length) * model_max_length
    input_ids_list: List[List[int]] = [
        input_ids_buffer[i : i + model_max_length]
        for i in range(0, total_length, model_max_length)
    ]
    labels_list: List[List[int]] = [
        labels_buffer[i : i + model_max_length]
        for i in range(0, total_length, model_max_length)
    ]
    for i, labels in enumerate(labels_list):
        if all(x == IGNORE_INDEX for x in labels):
            # labels can not have all values being -100. 18 and 24 are just random numbers
            labels[18:24] = input_ids_list[i][18:24]
    return {"input_ids": input_ids_list, "labels": labels_list}


def batch_grouped_pretrain_generate(
    model_max_length: int,
    tokenizer: PreTrainedTokenizer,
    examples: Dict[str, List[str]],
) -> Dict[str, List[List[int]]]:
    # build grouped texts with format `X1 X2 X3 ... <eos> X1 X2 X3 ... [<eos>]`
    token_ids_list: List[List[int]] = tokenizer(
        examples["text"], add_special_tokens=False
    )["input_ids"]
    token_ids_list = [
        token_ids + [tokenizer.eos_token_id] for token_ids in token_ids_list
    ]
    concatenated_ids = list(chain(*token_ids_list))
    # we drop the small remainder, and if the total_length < block_size, we exclude this batch
    total_length = (len(concatenated_ids) // model_max_length) * model_max_length
    result = [
        concatenated_ids[i : i + model_max_length]
        for i in range(0, total_length, model_max_length)
    ]
    return {"input_ids": result, "labels": result.copy()}

def inference_generate(
    model_max_length: int,
    tokenizer: PreTrainedTokenizer,
    model_prompt: Callable,
    data_point: Dict[str, Any],
):
    text = data_point['text']
    if model_prompt is not None:
        text = model_prompt(text)
    return {
        "input_ids": tokenizer.encode(text, add_special_tokens=False)[:model_max_length]
    }
