from LCDatasetItem import LCDatasetItem
import json
import random as rd
from typing import TypedDict
from LCDatasetItem import LCDatasetItem


class LoongItemInfo(TypedDict):
    level: str
    length: str
    type: str
    prompt_template: str
    instruction: str


def newLoongItem(question: dict) -> LCDatasetItem:
    question_specific_prompt = """
You are an expert research assistant, skilled in answering questions concisely and precisely,
using information provided by the user.

I’d like for you to answer questions about a context text that will be provided

Context:
{context}

Question:
{question}

Answer:
"""
    # TODO: figure out wtf to put here for rag
    ds_specific_info : LoongItemInfo = {
        "level": question["level"],
        "length": question["length"],
        "type": question["type"],
        "prompt_template": question["prompt_template"],
        "instruction": question["instruction"],
    }
    print(question["question"])
    exit()
    item = LCDatasetItem(
        ds_name = "loong",
        question_specific_prompt = question_specific_prompt,
        _id=question["id"],
        context=question["doc"],
        question=question["question"],
        answer=question["answer"], # TODO: debug this the outputs are weird here
        ds_specific_info=ds_specific_info
    )
    return item


def load_loong(loong_size, shuffle, difficulties = None, max_length = None, max_docs = None, only_english = True):
    # load json l from dir
    loong_json_path = "/Users/trevorarashiro/Documents/GitHub/datasets/datasets/loong_generate.jsonl"   
    results = []
    # Load and analyze prompts by level
    with open(loong_json_path, "r") as f:
        i = 0
        for line in f:
            i += 1
            if i > 10:
                exit()
            # dict_keys(['level', 'set', 'length', 'type', 'language', 'question', 'instruction',
            # 'prompt_template', 'doc', 'answer', 'shuffle_doc', 'id', 'docs', 'prompt', 'generate_response'])
            # level is how hard the question is
            # set is the length of the question
            # set 1 is 10k to 50k words
            # set 2 is 50k to 100k words
            # set 3 is 100k to 200k words
            # set 4 is 200k to 250k words
            # type is legal financial or paper
            # language is zh or en
            # there is exactly one instruction for each question type

            
            example = json.loads(line)
            
            if only_english and example["language"] != "en":
                continue
            
            if example["question"] != "":
                # print(example["question"])
                # print(len(example["doc"]))
                print("*****instruction:", example["instruction"])
                print("*****prompt_template:", example["prompt_template"])
                exit()
                # print("*****type:", example["type"])
                # print("*****prompt_template:", example["prompt_template"])
                # print("\n"*3)
                # print("*****prompt:", example["prompt"])
                # print("-"*200)
            assert(example["instruction"] in example["prompt"])
            assert(example["question"] in example["prompt"])
                
            
            
            if difficulties:
                if example["level"] not in difficulties:
                    continue
            if max_length:
                if example["length"] > max_length:
                    continue
            if max_docs:
                if len(example["doc"]) > max_docs:
                    continue
            results.append(example)
    
    if shuffle:
        rd.shuffle(results)
    return results[:loong_size]


if __name__ == "__main__":
    ds = load_loong(2, shuffle=True)
    # print(ds)