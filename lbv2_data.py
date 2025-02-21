# https://cdn-lfs-us-1.hf.co/repos/8a/fd/8afd4d81a7452ac66cea7e23f6c3b453bbb9ee2d891f497de2973362063eb5da/15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27data.json%3B+filename%3D%22data.json%22%3B&response-content-type=application%2Fjson&Expires=1739982164&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczOTk4MjE2NH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zLzhhL2ZkLzhhZmQ0ZDgxYTc0NTJhYzY2Y2VhN2UyM2Y2YzNiNDUzYmJiOWVlMmQ4OTFmNDk3ZGUyOTczMzYyMDYzZWI1ZGEvMTVkNjFjMjJkOTJjOTY5MDBiM2M0OTQ4YjZhZWVhMjE4ZDMyMTRiNjc2YTY1ZGY0OGU3Yjg1NTU2MDRjN2ZlMj9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSomcmVzcG9uc2UtY29udGVudC10eXBlPSoifV19&Signature=XHtnZViQbZp7gCnWkNyXkZ0H9yWW4P2Klo3QMBaAUWBFJqKQEIJWBbkfmqygrgQAPslClNKmVW%7EuZIduGFYLHAR4g1aDgIBcuxhOp68v5bXxcyszPqKgiTdL%7ES6ZJA9Sg2Pyljl5JyyWm11YK54PA7S6A5wEDEnTGvsSIDIhPjWg4QaPNHoU-GOsRC%7Ekgr4uTpm5%7EFERyr%7EsB1B6K2ETqAwB7iFt93K4c0mVa5mROh14e2xgl0kZevyltqSY65ay9S6Ise%7ExCIZ02NJpAs7YA80vxFnf8YqcMjURshGvF6j-zZKkkDr5mrpUflstXrHdmD7vwuz-MEXl427fgyAHeQ__&Key-Pair-Id=K24J24Z295AEI9

from datasets import load_dataset
import time
from typing import TypedDict
from LCDatasetItem import LCDatasetItem, RewardType


class LBv2ItemInfo(TypedDict):
    domain: str
    sub_domain: str
    difficulty: str
    length: str


def newLBv2Item(question: dict) -> LCDatasetItem:
    question_specific_prompt = """Choose from the following options to answer this question.
Your answer should be based only on the information provided in the question and from any documents returned from search.  
Your final answer between the answer tags should be a single letter (A, B, C, or D) corresponding to the correct answer.
    
Question:
{question}

A. {choice_A}
B. {choice_B}
C. {choice_C}
D. {choice_D}
"""
    # map the multile choice answer to the actual answer
    ds_specific_info : LBv2ItemInfo = {
        "domain": question["domain"],
        "sub_domain": question["sub_domain"],
        "difficulty": question["difficulty"],
        "length": question["length"]
    }
    item = LCDatasetItem(
        ds_name="long bench v2",
        question_specific_prompt=question_specific_prompt.format(
            question=question["question"],
            choice_A=question["choice_A"],
            choice_B=question["choice_B"],
            choice_C=question["choice_C"],
            choice_D=question["choice_D"]
        ),
        _id=question["_id"],
        question=question["question"],
        answer=question["answer"],
        ds_specific_info=ds_specific_info,
        reward_type=RewardType.LETTER,
    )
    return item, question["context"]


def load_lbv2(lbv2_size, shuffle, difficulties = None, question_lengths = None):
    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    # '_id', 'domain', 'sub_domain', 'difficulty', 'length', 'question', 'choice_A', 'choice_B', 'choice_C', 'choice_D', 'answer', 'context'],
    # print(dataset["_id"])

    # I am skipping out on the long in-context learning for now, but should DEFINITELY look into these in the future
    # they dont serve the purpose of QA but can be useful for general tasks in the future
    dataset = dataset.filter(lambda x: x["domain"] != "Long In-context Learning")

    if difficulties:
        dataset = dataset.filter(lambda x: x["difficulty"] in difficulties)
    
    if question_lengths:
        dataset = dataset.filter(lambda x: x["length"] in question_lengths)

    if shuffle:
        dataset = dataset.shuffle(seed=int(time.time()))
    
    dataset = dataset.select(range(lbv2_size))

    return dataset

if __name__ == "__main__":
    ds = load_lbv2(400, shuffle=True, question_lengths=None)
    for q in ds:
        print("Answer: ", q["choice_" + q["answer"]])
        # print(q["context"][:1000])
        print("-"*100)