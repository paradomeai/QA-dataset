import os
import random as rd
from lbv2_data import load_lbv2, newLBv2Item
from repliqa_data import load_repliqa, newRepliqaItem
from loong_data import load_loong, newLoongItem
from babilong_data import load_babilong, newBabilongItem
from doc_fin_QA_data import load_docfinqa, newDocFinQADatasetItem
from lbv1_data import load_lbv1, newLBv1Item
from LCDatasetItem import get_context_hash
import json
import uuid
os.makedirs("data_cache", exist_ok=True)

class RaggableDS:
    def __init__(self, shuffle=True, shuffle_all=True):
        self.shuffle = shuffle
        self.shuffle_all = shuffle_all
        self.ds = []
        self.hash_to_uuid = dict()
        self.contexts = dict()

    def _add_dataset(self, size, load_fn, item_fn, load_kwargs=None, item_kwargs=None):
        # Use empty dicts if no kwargs were provided
        load_kwargs = load_kwargs or {}
        item_kwargs = item_kwargs or {}

        # Load the dataset with the provided load function and parameters.
        ds = load_fn(size, shuffle=self.shuffle, **load_kwargs)
        for question in ds:
            # Create an item using the provided item function and parameters.
            item, context = item_fn(question, **item_kwargs)
            if context not in (None, ""):
                context_hash = get_context_hash(context)
                if context_hash not in self.hash_to_uuid:
                    self.hash_to_uuid[context_hash] = str(uuid.uuid4())
                item.context_uuid = self.hash_to_uuid[context_hash]
                self.contexts[item.context_uuid] = context
            self.ds.append(item)
            

        # Optionally shuffle the full dataset.
        if self.shuffle_all:
            rd.shuffle(self.ds)

    def add_lbv2(self, lbv2_size, difficulties=None, question_lengths=None):
        self._add_dataset(
            lbv2_size,
            load_fn=load_lbv2,
            item_fn=newLBv2Item,
            load_kwargs={'difficulties': difficulties, 'question_lengths': question_lengths}
        )

    def add_repliqa(self, repliqa_size):
        self._add_dataset(
            repliqa_size,
            load_fn=load_repliqa,
            item_fn=newRepliqaItem
        )

    # def add_loong(self, loong_size, difficulties=None, max_length=None, max_docs=None):
    #     self._add_dataset(
    #         loong_size,
    #         load_fn=load_loong,
    #         item_fn=newLoongItem,
    #         load_kwargs={'difficulties': difficulties, 'max_length': max_length, 'max_docs': max_docs}
    #     )

    def add_babilong(self, babilong_size, noise_level, qa_sets = set("qa" + str(i) for i in range(21))):
        self._add_dataset(
            babilong_size,
            load_fn=load_babilong,
            item_fn=newBabilongItem,
            item_kwargs={'noise_level': noise_level},
            load_kwargs={'noise_level': noise_level, 'qa_sets': qa_sets}
        )

    def add_docfinqa(self, docfinqa_size, exclude_programs=True):
        self._add_dataset(
            docfinqa_size,
            load_fn=load_docfinqa,
            item_fn=newDocFinQADatasetItem,
            load_kwargs={'exclude_programs': exclude_programs}
        )
    
    def add_lbv1(self, lbv1_size):
        self._add_dataset(
            lbv1_size,
            load_fn=load_lbv1,
            item_fn=newLBv1Item,
        )

if __name__ == "__main__":
    ds = RaggableDS(shuffle=True, shuffle_all=False)
    ds.add_lbv2(400)
    ds.add_repliqa(1600)
    # ds.add_loong(1)
    ds.add_babilong(200, "0k", set(["qa1", "qa2", "qa3", "qa7", "qa8"]))
    ds.add_babilong(350, "32k", set(["qa1", "qa2", "qa3", "qa7", "qa8"]))
    ds.add_babilong(350, "1M", set(["qa1", "qa2", "qa3", "qa7", "qa8"]))
    ds.add_docfinqa(1900)
    ds.add_lbv1(1700)

    # s = set()
    # for d in ds.ds:
    #     if not d.ds_specific_info["dataset"] in s:
    #         s.add(d.ds_specific_info["dataset"])
    #         print(d)
    #         print("--------------------------------")
    #         print(ds.contexts[d.context_uuid][:1000])
    #         print("\n"*20)
    # exit()
    # print(s)
    # for i in ds.ds:
    #     print(i)
    #     print("--------------------------------")
    
    # print(json.dumps(ds.contexts, indent=4))

    # Serialize the instance to a JSON string
    # Write the JSON string to a file
    file_info = []
    with open("questions.json", "w") as file:
        for d in ds.ds:
            json_data = d.model_dump()  # Get dictionary instead of JSON string
            file_info.append(json_data)
        
        json.dump(file_info, file, indent=4)  # Properly write as JSON
    
    print(json.dumps(ds.contexts, indent=4))
