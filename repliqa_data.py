from datasets import load_dataset, concatenate_datasets
import time
from typing import TypedDict
from LCDatasetItem import LCDatasetItem, RewardType, binary_rm_prompt, long_form_rm_prompt


class RepliqaItemInfo(TypedDict):
    document_topic: str
    document_id: str
    document_path: str
    long_answer: str


def newRepliqaItem(question: dict) -> LCDatasetItem:
    question_specific_prompt = """Search the documents for the correct answer to the question below.
If the answer is not found in the documents, your final answer between the answer tags should be "The answer is not found in the documents."

Question
{question}
"""
    ds_specific_info : RepliqaItemInfo = {
        "document_topic": question["document_topic"],
        "document_id": question["document_id"],
        "document_path": question["document_path"],
        "long_answer": question["long_answer"],
    }
    if len(question["answer"]) < 25:
        rm_prompt = binary_rm_prompt.format(question=question["question"], answer=question["answer"])
    else:
        rm_prompt = long_form_rm_prompt.format(question=question["question"], answer=question["answer"])
    item = LCDatasetItem(
        ds_name = "repliqa",
        question_specific_prompt = question_specific_prompt.format(question=question["question"]),
        _id=question["question_id"],
        question=question["question"],
        answer=question["answer"],
        ds_specific_info=ds_specific_info,
        reward_type=RewardType.REWARD_MODEL,
        rm_prompt=rm_prompt
    )
    return item, question["document_extracted"]


def load_repliqa(repliqa_size, shuffle):
    dataset = load_dataset("ServiceNow/repliqa")
    dataset = concatenate_datasets([dataset["repliqa_0"], dataset["repliqa_1"], dataset["repliqa_2"]])
    #['document_id', 'document_topic', 'document_path', 'document_extracted', 'question_id', 'question', 'answer', 'long_answer'],

    if shuffle:
        dataset = dataset.shuffle(seed=int(time.time()))

    # for i in range(len(dataset)):
    #     if dataset[i]["document_id"] == "jpwhpgtu":
    #         print(dataset[i]["document_extracted"])
    #         print("-"*100)
    
    dataset = dataset.select(range(repliqa_size))
    
    return dataset


if __name__ == "__main__":
    ds = load_repliqa(1000, shuffle=True)
    print(ds)
    for d in ds:
        print(d["answer"])
        print("-"*100)