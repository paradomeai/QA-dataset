# https://cdn-lfs-us-1.hf.co/repos/8a/fd/8afd4d81a7452ac66cea7e23f6c3b453bbb9ee2d891f497de2973362063eb5da/15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27data.json%3B+filename%3D%22data.json%22%3B&response-content-type=application%2Fjson&Expires=1739982164&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczOTk4MjE2NH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zLzhhL2ZkLzhhZmQ0ZDgxYTc0NTJhYzY2Y2VhN2UyM2Y2YzNiNDUzYmJiOWVlMmQ4OTFmNDk3ZGUyOTczMzYyMDYzZWI1ZGEvMTVkNjFjMjJkOTJjOTY5MDBiM2M0OTQ4YjZhZWVhMjE4ZDMyMTRiNjc2YTY1ZGY0OGU3Yjg1NTU2MDRjN2ZlMj9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSomcmVzcG9uc2UtY29udGVudC10eXBlPSoifV19&Signature=XHtnZViQbZp7gCnWkNyXkZ0H9yWW4P2Klo3QMBaAUWBFJqKQEIJWBbkfmqygrgQAPslClNKmVW%7EuZIduGFYLHAR4g1aDgIBcuxhOp68v5bXxcyszPqKgiTdL%7ES6ZJA9Sg2Pyljl5JyyWm11YK54PA7S6A5wEDEnTGvsSIDIhPjWg4QaPNHoU-GOsRC%7Ekgr4uTpm5%7EFERyr%7EsB1B6K2ETqAwB7iFt93K4c0mVa5mROh14e2xgl0kZevyltqSY65ay9S6Ise%7ExCIZ02NJpAs7YA80vxFnf8YqcMjURshGvF6j-zZKkkDr5mrpUflstXrHdmD7vwuz-MEXl427fgyAHeQ__&Key-Pair-Id=K24J24Z295AEI9

from datasets import load_dataset, concatenate_datasets
from LCDatasetItem import LCDatasetItem, RewardType, is_sorta_binary, long_form_rm_prompt, binary_rm_prompt
import time
from typing import TypedDict


class LBv1ItemInfo(TypedDict):
    length: str
    dataset: str


def newLBv1Item(question: dict) -> LCDatasetItem:
    if question["dataset"] == "passage_retrieval_en" or question["dataset"] == "passage_retrieval_en_e":
        question_specific_prompt = """In the paragraph documents you have access to, which one most closely fits the summary below?
Your final answer between the answer tags should be a single number corresponding to the correct paragraph.  For example, "Paragraph 15".  If no paragraph is related, return "Unanswerable".

Summary:
{question}
"""
    else:
        if is_sorta_binary(question["answers"][0]):
            question_specific_prompt = """Answer the following question based on the documents you have access to and any information provided in the question.
Your final answer between the answer tags should be only one of Yes, No, or Unanswerable): 

Question:
{question}
"""
        else:
            question_specific_prompt = """Based on the documents you have access to and any information provided in the question, what is the correct answer to the below question:
If the question is not related to any text or the answer is not in the text, return "Unanswerable".

Question:
{question}
"""
    # map the multile choice answer to the actual answer
    ds_specific_info : LBv1ItemInfo = {
        "length": question["length"],
        "dataset": question["dataset"]
    }
    assert(len(question["answers"]) == 1)
    if len(question["answers"][0]) < 25:
        rm_prompt = binary_rm_prompt.format(question=question["input"], answer=question["answers"][0])
    else:
        rm_prompt = long_form_rm_prompt.format(question=question["input"], answer=question["answers"][0])
    item = LCDatasetItem(
        ds_name="long bench v1",
        question_specific_prompt=question_specific_prompt.format(question=question["input"]),
        _id=question["_id"],
        question=question["input"],
        answer=question["answers"][0],
        ds_specific_info=ds_specific_info,
        reward_type=RewardType.REWARD_MODEL,
        rm_prompt=rm_prompt
    )
    return item, question["context"]


# this is for standardizing the triviaqa dataset
def remove_after_last_question(text: str) -> str:
    lines = text.split("\n")  # Split text into lines
    last_question_index = -1  # Track the last occurrence of "Question:"
    
    # Find the last occurrence of a line starting with "Question:"
    for i, line in enumerate(lines):
        if line.strip().startswith("Question:"):
            last_question_index = i
    
    # If "Question:" is found, return only the lines before and including it
    if last_question_index != -1:
        return "\n".join(lines[:last_question_index]), "\n".join(lines[last_question_index+1:])
    
    assert(False)


# _e datasets are similar to the original but are still reasonably different enough to be considered two separate examples
def load_lbv1(lbv1_size, shuffle, max_length = None, exclude_programs = True):
    # these configs include datasets with summaries
    # configs = ["narrativeqa","qasper","multifieldqa_en","multifieldqa_zh","hotpotqa","2wikimqa","musique","dureader","gov_report","qmsum","multi_news","vcsum","trec","triviaqa","samsum","lsht","passage_count","passage_retrieval_en","passage_retrieval_zh","lcc","repobench-p","qasper_e","multifieldqa_en_e","hotpotqa_e","2wikimqa_e","gov_report_e","multi_news_e","trec_e","triviaqa_e","samsum_e","passage_count_e","passage_retrieval_en_e","lcc_e","repobench-p_e"]
    # TODO: explore trivia qa and "triviaqa_e", might be worth it
    # "repobench-p" is programming and idk what to do with it 
    configs = ["qasper","multifieldqa_en","hotpotqa","2wikimqa","musique","qasper_e","multifieldqa_en_e","hotpotqa_e","2wikimqa_e","passage_retrieval_en_e", "passage_retrieval_en"]

    #     features: ['input', 'context', 'answers', 'length', 'dataset', 'language', 'all_classes', '_id'],
    datasets = []
    for config in configs:
        datasets.append(load_dataset("THUDM/LongBench", config, split="test"))

    # for d in datasets:
    #     print(d[0]["dataset"])
    #     for i in range(10):
    #         print(d[i]["context"])
    #         print("######################################################################################################################")
    #         print(d[i]["input"])
    #         print("######################################################################################################################")
    #         print(d[i]["answers"])
    #         # print(d[i]["context"])
    #         print("\n"*10)

    #     print("-"*100)
    #     print("-"*100)
    #     exit()

    datasets = concatenate_datasets(datasets)

    datasets = datasets.filter(lambda x: x["language"] != "zh" and \
                                x["input"] != "" and \
                                "column Ens Test in Table TABREF19" not in x["answers"] and \
                                len(x["answers"]) == 1 and \
                                "," not in x["answers"][0]
                            ) # the column Ens question is just wrong so remove it.  We remove any questions with multiple answers, or even commas cuz its too weird.  This gets rid of about 600 (25%) of the dataset
    
    if max_length:
        datasets = datasets.filter(lambda x: (int(x["length"]) < max_length))

    if exclude_programs:
        datasets = datasets.filter(lambda x: x["language"] == "en")

    # datasets = datasets.shuffle(seed=int(time.time()))

    # checkout the triviaqa dataset later
    # for d in datasets:
    #     if d["dataset"] == "qasper":
    #         # context, question = remove_after_last_question(d["context"])
    #         print(d["context"])
    #         print("asdf")
    #         print(d["input"])
    #         print("asdf")
    #         print(d["answers"])
    #         print("asdf")
    #         print("-"*100)
    # exit()

    if shuffle:
        datasets = datasets.shuffle(seed=int(time.time()))
    
    datasets = datasets.select(range(lbv1_size))
    
    return datasets


if __name__ == "__main__":
    ds = load_lbv1(20, shuffle=True, max_length=None, exclude_programs=True)
    for d in ds:
        if len(d["answers"]) > 1:
            print(d["input"])
        print(len(d["answers"]), d["answers"])
        print("-"*100)