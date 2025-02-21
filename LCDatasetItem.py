from pydantic import BaseModel
from typing import Dict, Any, Optional
import hashlib
from enum import Enum

class RewardType(str, Enum):
    REWARD_MODEL = "reward_model"
    MATHEMATICAL = "mathematical"
    LETTER = "letter"

class LCDatasetItem(BaseModel):
    ds_name: str
    _id: str
    context_uuid: Optional[str] = None
    question: str
    answer: Any 
    question_specific_prompt: str
    ds_specific_info: Dict
    reward_type: RewardType
    rm_prompt: Optional[str] = None

def get_context_hash(input_string: str) -> str:
    """
    Returns the SHA-512 hash of the input string.
    
    Args:
        input_string (str): The string to be hashed.
    
    Returns:
        str: The hexadecimal representation of the SHA-512 hash.
    """
    # Encode the string to bytes, then compute the SHA-512 hash
    hash_object = hashlib.sha512(input_string.encode('utf-8'))
    return hash_object.hexdigest()


def is_sorta_binary(s):
    if type(s) == str:
        return s.lower() in "yes." + "no." + "true" + "false" + "a." + "b." + "c." + "d." + "y" + "n" 
    elif type(s) == list:
        return any(is_sorta_binary(item) for item in s)
    else:
        return False

binary_rm_prompt = """You are a helpful assistant that determines if the student's answer to a factual question or yes/no question is correct.
If the provided answer has the same meaning as the ground truth answer, return 1. Otherwise return 0.

Question:
{question}

Ground Truth Answer:
{answer}

Student Answer:
"""

babilong_rm_prompt = """You are a helpful assistant that determines if the student's answer to a factual question is correct.
If the provided answer has the same meaning as the ground truth answer, return 1. Otherwise return 0.
Additional specifications:
    For questions with quantitative ground truth answers, only consider the quantity mentioned in the student's answer.  For example, if the answer is "one" and the student's answer is "1 football", return 1.
    For list answers, only consider the items mentioned and not their order.

Question:
{question}

Ground Truth Answer:
{answer}

Student Answer:
"""

long_form_rm_prompt = """You are a helpful assistant that determines if the student's answer to a long form question is correct.
If the provided answer shares the same meaning as the ground truth answer, return 1. 
If the provided answer partially matches the ground truth answer, return 0.5.
If the provided answer is correct but includes extra information beyond the ground truth answer, return 0.5.
Otherwise return 0.

Question:
{question}

Ground Truth Answer:
{answer}

Student Answer:
"""