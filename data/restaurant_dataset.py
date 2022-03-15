from torch.utils.data import Dataset

from utils.util import open_file


class Restaurant8kDataset(Dataset):

    def __init__(self, data_path, include_all=False):
        self.include_all = include_all

        self._init_data(data_path)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_examples
    
    def _init_data(self, dpath):
        self.data = []
        full_data = open_file(dpath)

        for raw_example in full_data:
            no_label = "labels" not in raw_example

            if not self.include_all and no_label:
                continue
            
            utterance = raw_example["userInput"]["text"]
            labels = []
            if no_label:
                labels.append({"slot": "none", "start_index": -1, "end_index": -1})
            else:
                for label in raw_example["labels"]:
                    slot = label["slot"]
                    end_index = label["valueSpan"]["endIndex"]

                    # If the utterance starts with slot value, start index is implicit
                    if "startIndex" not in label["valueSpan"]:
                        start_index = 0
                    else:
                        start_index = label["valueSpan"]["startIndex"]
                    
                    labels.append({"slot": slot, "start_index": start_index, "end_index": end_index})

            self.data.append({"utterance": utterance, "labels": labels})

        self.num_examples = len(self.data)
