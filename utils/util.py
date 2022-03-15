import json
import random

import torch
import numpy as np


def open_file(fpath):
    """ Helper function for reading data from files. """

    file_extension = fpath.split('.')[-1]

    if file_extension == "json":
        with open(fpath, 'r') as json_file:
            return json.load(json_file)
    
    print(f"Not supported file type: {file_extension}")

    return None


def set_seed(seed):
    """ Set a seed to allow reproducibility. """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def epoch_time(start_time, end_time):
    """ Calculate processing time in minutes and seconds. """

    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs
