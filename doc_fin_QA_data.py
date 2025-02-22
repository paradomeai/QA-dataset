from datasets import load_dataset, concatenate_datasets
from LCDatasetItem import LCDatasetItem, RewardType, is_sorta_binary, binary_rm_prompt
import time


def get_docfinqa_q_id(q):
    return hash(q["Context"] + q["Question"] + q["Answer"])


def newDocFinQADatasetItem(question: dict) -> LCDatasetItem:
    question_specific_prompt: str = """Based on the documents you have access to, what is the correct answer to the question below?
    
Question:
{question}
"""
    
    item = LCDatasetItem(
        ds_name = "docfinqa",
        question_specific_prompt = question_specific_prompt.format(question=question["Question"]),
        _id=get_docfinqa_q_id(question),
        question=question["Question"],
        answer=question["Answer"],
        ds_specific_info=dict(),
        reward_type=RewardType.REWARD_MODEL if is_sorta_binary(question["Answer"]) else RewardType.MATHEMATICAL,
        rm_prompt=binary_rm_prompt.format(question=question["Question"], answer=question["Answer"]) if is_sorta_binary(question["Answer"]) else None
    )
    return item, question["Context"]


# heedle in a nay stack with very very random noise (like "kinds in the 1300's loved their jesters.  Jane went to the kitchen. Turtles run at approximately 0.5 mph")
def load_docfinqa(docfinqa_size, shuffle = True, exclude_programs = True):
    docfinqa = load_dataset("kensho/DocFinQA", cache_dir="./data_cache")
    # this is the same dataset 85% confident
    # booydar_babilong = load_dataset("booydar/babilong", "2k", cache_dir="./data_cache")
    # this is also the same dataset but only has the first 5 kinds of questions but 1k examples for each instead of 100
    # booydar_babilong_1k = load_dataset("booydar/babilong-1k-samples", "2k", cache_dir="./data_cache")

    # concatenate qa1,qa2,qa3,... (this is different for some noise levels)
    # ['Context', 'Question', 'Program', 'Answer']
    docfinqa = concatenate_datasets([docfinqa[k] for k in docfinqa])

    docfinqa = docfinqa.filter(lambda x: all(s not in x["Answer"] for s in ["\\", "respect", "and", "mill", "thous", "hun", "per", "a", "b", "r", "t"]))

    if exclude_programs:
        docfinqa = docfinqa.filter(lambda x: x["Program"] in (None, ""))

    if shuffle:
        docfinqa = docfinqa.shuffle(seed=int(time.time()))

    docfinqa = docfinqa.select(range(docfinqa_size))
    
    return docfinqa


if __name__ == "__main__":
    ds = load_docfinqa(1000, shuffle=True)
    count = 0
    for d in ds:
        count += 1
        if count > 3:
            break
        # print(d["Question"])
        print(d["Context"])
        print("-"*100)
    print(ds)