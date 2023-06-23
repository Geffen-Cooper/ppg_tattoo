import datetime
import logging 
import time
import sys
import torch
import numpy as np
import random
import os

def init_logger(logname):
    # setup logging
    logname = logname + "_" + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H;%M;%S')
    fn = "saved_data/logs/"+str(logname)+".log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(fn),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger=logging.getLogger() 
    logger.setLevel(logging.INFO)
    
    return logger

def init_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False