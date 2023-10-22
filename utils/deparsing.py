import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#from datasets import load_datasets_and_vocabs, my_collate
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tree import *

import spacy
from transformers import AutoConfig, AutoTokenizer
from transformers import HfArgumentParser, PreTrainedTokenizer
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset, Dataset
import os
from spacy.tokens import Doc
from random_words import RandomWords
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from lemminflect import getInflection

import argparse
import json
import re
import sys

from allennlp.predictors.predictor import Predictor
from lxml import etree
from nltk.tokenize import TreebankWordTokenizer
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence
import codecs
import linecache
import logging
import pickle
from collections import Counter, defaultdict
from copy import copy, deepcopy

import simplejson as json
from allennlp.modules.elmo import batch_to_ids
from lxml import etree
from nltk import word_tokenize
from nltk.tokenize import TreebankWordTokenizer
from torch.utils.data import DataLoader, Dataset

import argparse
import json
import os
import re
import sys

from allennlp.predictors.predictor import Predictor
from lxml import etree
from nltk.tokenize import TreebankWordTokenizer
from tqdm import tqdm

import warnings
warnings.filterwarnings('always')


nltk.download('omw-1.4')
nltk.download("wordnet")

random.seed(12345)