from datasets import load_dataset, concatenate_datasets
import os
from pydantic import BaseModel
import json

# Create a directory for the dataset (e.g., 'data')
os.makedirs("data_babi_cache", exist_ok=True)

# # need to select a length (up to 1m)
# # Rag also probably needs to be done on a per question basis as well
# babilong = load_dataset("RMT-team/babilong", "0k", cache_dir="./data_babi_cache")

# booydar_babilong = load_dataset("booydar/babilong", "0k", cache_dir="./data_babi_cache")

# booydar_babilong_1k = load_dataset("booydar/babilong-1k-samples", "2k", cache_dir="./data_babi_cache")

# for i in babilong:
#     print(i)
# exit()
# print(babilong["qa10"][0])
# print("-"*100)
# print(booydar_babilong["qa10"][0])


# # Print first example from the training set
# print("\nExample from dataset:")
# print(babilong["qa1"][5])


# class BabilongItem(BaseModel):
#     input: str
#     question: str
#     target: str


docfinqa = load_dataset("kensho/DocFinQA", cache_dir="./data_babi_cache")

print(docfinqa)
docfinqa = concatenate_datasets([docfinqa["train"], docfinqa["validation"], docfinqa["test"]])
print(docfinqa)
count = 0
for q in docfinqa:
    if q["Pfrogram"] in (None, ""):
        count += 1

print(count)

# print("\nExample from dataset:")
# first_example = docfinqa["train"][0]
# # print(first_example)
# print(len(first_example["Context"])) 


# # only 1.4k examples wihtout program field
# # Rag needs to be done on a question basis probably
# class DocFinQAItem(BaseModel):
#     context: str
#     question: str
#     answer: str
#     program: str  # keep ones requiring math?


# # Count items without program field or with empty program field
# # Do we want to have model write python code?
# program_missing_count = sum(
#     1
#     for item in docfinqa["train"]
#     if not item.get("Program") or not item.get("Program").strip()
# )
# print(
#     f"\nNumber of items without program field or with empty program: {program_missing_count}"
# )


# # load json l from dir
# loong_json_path = "/Users/george/Code/Paradome/datasets/data/loong/Loong/output/gpt4o/loong_generate.jsonl"

# # Load and analyze prompts by level
# with open(loong_json_path, "r") as f:
#     level_counts = {}
#     level_prompts = {}
#     level_templates = {}

#     for line in f:
#         example = json.loads(line)
#         level = example.get("level")
#         if level is None:
#             continue

#         # Initialize collections for this level if not exists
#         if level not in level_prompts:
#             level_prompts[level] = set()
#             level_templates[level] = set()
#             level_counts[level] = 0

#         # Add prompts and templates to their respective sets
#         level_prompts[level].add(example.get("prompt", ""))
#         level_templates[level].add(example.get("prompt_template", ""))
#         level_counts[level] += 1
    
#     print(level_counts)
#     print(level_prompts[0][0])

#     # # Print analysis results
#     # print("\nPrompt Analysis by Level:")
#     # for level in sorted(level_counts.keys()):
#     #     print(f"\nLevel {level}:")
#     #     print(f"Total items: {level_counts[level]}")
#     #     print(f"Unique prompts: {len(level_prompts[level])}")
#     #     print(f"Unique prompt templates: {len(level_templates[level])}")

#     #     # If there's only one template, print it as an example
#     #     if len(level_templates[level]) == 1:
#     #         template = next(iter(level_templates[level]))
#     #         print(f"Template used: {template[:200]}...")


# class LCDatasetItem(BaseModel):
#     context: str  # or just id to rag store
#     question: str
#     answer: str
#     prompt: str


# # # Load and analyze instructions by level
# # with open(loong_json_path, "r") as f:
# #     all_instructions = set()  # Track all unique instructions across levels
# #     level_instructions = {}

# #     for line in f:
# #         example = json.loads(line)
# #         level = example.get("level")
# #         if level is None:
# #             continue

# #         # Initialize collections for this level if not exists
# #         if level not in level_instructions:
# #             level_instructions[level] = set()

# #         instruction = example.get("instruction", "")
# #         level_instructions[level].add(instruction)
# #         all_instructions.add(instruction)

# #     # Print overall analysis
# #     print("\nOverall Instructions Analysis:")
# #     print(f"Total unique instructions across all levels: {len(all_instructions)}")
# #     if len(all_instructions) == 1:
# #         print("\nThe common instruction used across all levels:")
# #         print("=" * 80)
# #         print(next(iter(all_instructions)))
# #         print("=" * 80)
