from datasets import load_dataset, concatenate_datasets
from LCDatasetItem import LCDatasetItem, RewardType, babilong_rm_prompt
import time


def get_babilong_q_id(question: dict) -> str:
    return hash(question["input"] + question["question"] + question["target"])
    

def newBabilongItem(question: dict, noise_level: str) -> LCDatasetItem:
    question_specific_prompt = """Search the documents for the correct answer to this question:
{question}
"""

    item = LCDatasetItem(
        ds_name = "babilong_" + noise_level,
        question_specific_prompt=question_specific_prompt.format(question=question["question"]),
        _id=get_babilong_q_id(question),
        question=question["question"],
        answer=question["target"],
        ds_specific_info=dict(),
        reward_type=RewardType.REWARD_MODEL,
        rm_prompt=babilong_rm_prompt.format(question=question["question"], answer=question["target"])
    )
    return item, question["input"]


# heedle in a nay stack with very very random noise (like "kinds in the 1300's loved their jesters.  Jane went to the kitchen. Turtles run at approximately 0.5 mph")
def load_babilong(babilong_size, noise_level, qa_sets = set(["qa1", "qa2", "qa3", "qa7", "qa8"]), shuffle = True):
    """
    noise_level: "0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1m"
    """
    babilong = load_dataset("RMT-team/babilong", noise_level, cache_dir="./data_babi_cache")
    # this is the same dataset 85% confident
    # booydar_babilong = load_dataset("booydar/babilong", "2k", cache_dir="./data_babi_cache")
    # this is also the same dataset but only has the first 5 kinds of questions but 1k examples for each instead of 100
    # booydar_babilong_1k = load_dataset("booydar/babilong-1k-samples", "2k", cache_dir="./data_babi_cache")
    # qa1, qa2, qa3, qa7, qa8 are the interesting ones that have multi step reasoning (qa1 is very basic)

    # concatenate qa1,qa2,qa3,... (this is different for some noise levels)
    # ['input', 'question', 'target'],
    # for k in ["qa8"]:
    #     for i in range(len(babilong[k])):
    #         print(babilong[k][i]["question"])
    #         print(babilong[k][i]["target"])
    #         print(babilong[k][i]["input"])
    #         print("-"*100)

    available_qa_sets = set(babilong.keys())
    qa_sets = qa_sets.intersection(available_qa_sets)
    
    babilong = concatenate_datasets([babilong[k] for k in qa_sets])

    if shuffle:
        babilong = babilong.shuffle(seed=int(time.time()))
    
    babilong = babilong.select(range(babilong_size))
    
    return babilong


if __name__ == "__main__":
    ds = load_babilong(400, noise_level="0k", shuffle=True)
    for d in ds:
        print(d["target"])
        print("-"*100)