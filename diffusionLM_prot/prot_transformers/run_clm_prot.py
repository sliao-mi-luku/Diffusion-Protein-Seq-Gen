#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import torch

# SL moved
import csv, json
from collections import Counter, defaultdict
from spacy.lang.en import English
import numpy as np

import argparse
from improved_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

import datasets
#import stanza
#import spacy_stanza
from datasets import Dataset, load_dataset, load_metric

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from custom_trainer import GPT2LMHeadModelCompress, BERTModelCompress, AutoEncoderWithNoise, GPT2VAE, AR_for_cont,\
    Classifier_GPT2, Classifier_Times, Classifier_POS, Classifier_Tree, Classifier_Consistency

from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0.dev0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


# all available model configs
MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())

# all available model types (abbreviations, ex. roberta, bert, gpt2)
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False, pad_mask_id=None):
    if pad_mask_id is None:
        pad_mask_id = pad_token_id
    result = torch.full([len(examples), max_length], pad_token_id).tolist()
    mask_ = torch.full([len(examples), max_length], pad_mask_id).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)
    experiment: Optional[str] = field(default='prot250-e2e-back')  # SL: default was 'compress'
    learned_emb: Optional[str] = field(default='no')
    padding_mode: Optional[str] = field(default='pad')  # SL: default was 'block'
    prot_data_dir: Optional[str] = field(default='./datasets/prot_data')  # SL added
    reduced_emb: Optional[int] = field(default=8)
    rounding_mode: Optional[str] = field(default='gpt2')
    sigma: Optional[float] = field(default=1.0)
    n_embd: Optional[int] = field(default=16)
    init_emb: Optional[str] = field(default="")
    task: Optional[str] = field(default=None)
    config_overrides: Optional[str] = field(default=None,
                                            metadata={"help": "Override some existing default config settings when a model is trained from scratch. Example: "
                                                                   "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"})
    config_name: Optional[str] = field(default=None,
                                       metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(default=None,
                                          metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(default=None,
                                     metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"})
    use_fast_tokenizer: bool = field(default=True,
                                     metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."})
    model_revision: str = field(default="main",
                                metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."})
    use_auth_token: bool = field(default=False,
                                 metadata={"help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                                           "with private models)."})
    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."})
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(default=None, metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."})
    synth_config:  Optional[str] = field(default='/juice/scr/xlisali/diffusion_lm/synthetic_data/configs/emnlp2020/experiments/difflm_seed0_m3_k32_trainc20000.yaml',
                                         metadata={"help": "The name of the dataset to use (via the datasets library)."})
    block_size: Optional[int] = field(default=None, metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    validation_split_percentage: Optional[int] = field(default=5, metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "The number of processes to use for the preprocessing."})
    keep_linebreaks: bool = field(default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."})

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def get_corpus_rocstory(data_args):

    assert data_args.experiment == 'prot250-e2e-back'
    
    # only predict tm
    ordered_ = ['tm']
    # predict both tm and pH
    # ordered_ = ['Type', 'tm', 'pH']

    full_dict = defaultdict(lambda:Counter())

    def ordered_fill(src_lst, mode='full', full_dict=None):
        pair_lst = {x.split(':')[0].lstrip().strip():x.split(':')[1].lstrip().strip() for x in src_lst.split('|')}
        result_lst = []
        if mode == 'full':
            for x in ordered_:
                v = pair_lst.get(x, 'none')
                result_lst.append(f"{x} : {v}")
            return "|".join(result_lst)
        else:  # SL: 'tm', 'pH'
            v = pair_lst.get(mode, 'none')
            full_dict[mode][v] += 1
            return f"{mode} : {v}"
    
    print('  >> Loading dataset from Prot250 dataset (e2e format)\n')

    nlp = English()
    tokenizer = nlp.tokenizer

    # path to prot250 (e2e format) data
    path = f'{data_args.prot_data_dir}/prot250_clf_train_e2e.txt'

    sentence_lst = []
    vocab_lst = []

    with open(path, 'r') as ff:
        for row in ff:
            # muti-labels and protein sequence
            src_lst, word_lst = row.split('||')

            # parse protein sequence into words
            word_lst = [x.text for x in tokenizer(word_lst)]

            # SL: only keep the tm part
            src_lst = [x for x in src_lst.split('|') if 'tm' in x][0].strip()
            assert 'tm' in src_lst
            
            for mode in ordered_:
                src_lst3 = ordered_fill(src_lst, mode, full_dict)  # ex. "tm : medium"
                src_lst2 = [x.text for x in tokenizer(src_lst3)]  # ex. ['tm', ':', 'medium']
                sentence_lst.append((word_lst, src_lst2)) 
            
            #vocab_lst.append(word_lst)
            vocab_lst.append(word_lst + [x.text for x in tokenizer(src_lst) if x.text != '|'])  # SL edited (model needs to know the vocabs from the prompts)

    print("sentence_lst[0]: {}\n".format(sentence_lst[0]))
    print("sentence_lst[1]: {}\n".format(sentence_lst[1]))
    print("full_dict: {}\n".format(full_dict))

    counter = Counter()
    for input_ids in vocab_lst:
        counter.update(input_ids)

    vocab_dict = {'START': 0, 'END': 1, 'UNK':2, 'PAD':3}

    for k, v in counter.items():
        if v > 10:
            vocab_dict[k] = len(vocab_dict)

    print("vocab_dict: {}\n".format(vocab_dict))
    print("len(vocab_dict): {}\n".format(len(vocab_dict)))

    return sentence_lst, vocab_dict


def main():
    # See all possible arguments in src/transformers/training_args.py

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print('\n' + '=' * 100)
    print("model_args: {}".format(model_args))
    print('=' * 100 + '\n')
    print("data_args: {}".format(data_args))
    print('=' * 100 + '\n')
    print("training_args: {}".format(training_args))
    print('=' * 100 + '\n')
    
    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout)]
                        )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)


    ###################### LOAD DATASETS & dictionary #########################
    print('\n' + '=' * 100)
    print("\tLoading dataset and dictionary")
    print('=' * 100 + '\n')

    if model_args.experiment.startswith('prot250-e2e-back'):

        # extract dataset and process vocab
        # SL: train_dataset is list of tuples (['A', 'A', 'A'], ['Tm', ':', 'low'])
        # SL: vocab is {token : id}, using only data from sequences (not relying on prompts)
        train_dataset, vocab = get_corpus_rocstory(model_args)

        # process dataset (30,000 entries)
        train_datasets = Dataset.from_dict({'text': train_dataset})
        
        # split 1% of data to be validation
        raw_datasets = train_datasets.train_test_split(0.01)
        raw_datasets['validation'] = raw_datasets['test']

        # attach vocab to dataset
        raw_datasets.vocab = vocab
        print(raw_datasets)
        print(raw_datasets.vocab)
    else:
        assert "HERE" == "Loading dataset and dictionary"

    # process other arguments
    config_kwargs = {"cache_dir": model_args.cache_dir,
                     "revision": model_args.model_revision,
                     "use_auth_token": True if model_args.use_auth_token else None,
                     }

    # SL: expect model_args.config_name to be None
    # SL: expect model_args.model_name_or_path to be 'gpt2'
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:  # SL: expected HERE
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        assert "HERE" == "model_args.model_name_or_path is not given"
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")


    # SL: overwrite config values
    print("config")
    print(config)

    tokenizer_kwargs = {"cache_dir": model_args.cache_dir,  # None
                        "use_fast": model_args.use_fast_tokenizer,  # True
                        "revision": model_args.model_revision,  # 'main'
                        "use_auth_token": True if model_args.use_auth_token else None}  # None


    ############# LOAD TOKENIZER ##############

    """
    SL: we cannot load the tokenizer directly from the LM because the prompt texts do not exist in the LM vocab
    """

    print('\n' + '=' * 100)
    print("\tLoading tokenizer")
    print('=' * 100 + '\n')

    assert model_args.experiment == 'prot250-e2e-back'

    print('loading from dataset-specific vocab')
    tokenizer = raw_datasets.vocab
    print(len(tokenizer))
    reverse_tokenizer = {v: k for k, v in tokenizer.items()}

    print("\n  >> Tokenizer: {}".format(tokenizer))


    if model_args.model_name_or_path:
        
        ############# LOAD MODELS for controllable classifier ##############
        if model_args.experiment in ['prot250-e2e-back']:

            print("\n  >> updating vocab_size from {} to {}".format(config.vocab_size, len(tokenizer)))
            config.vocab_size = len(tokenizer)

            # load training_args.json from a previously trained diffusion model
            config_path = os.path.join(model_args.init_emb, "training_args.json")
            print("\n  >> loaded training_args.json from {}".format(config_path))

            with open(config_path, 'rb', ) as f:
                training_args2 = json.load(f)
            

            print("\n  >> training_args2: {}".format(training_args2))
            

            training_args2['sigma_small'] = True
            training_args2['diffusion_steps'] = 200  # 500  # DEBUG

            temp_dict = model_and_diffusion_defaults()

            temp_dict.update(training_args2)

            print("\n  >> temp_dict: {}".format(temp_dict))

            _, diffusion = create_model_and_diffusion(**temp_dict)

            config.input_emb_dim = model_args.n_embd
            print("\n  >> config.input_emb_dim: {}".format(config.input_emb_dim))

            config.train_diff_steps = training_args2['diffusion_steps']
            print("\n  >> config.train_diff_steps: {}".format(config.train_diff_steps))

            print("\n  >> final clissifier model config: {}".format(config))
            

            ## Create classifier
            if model_args.experiment == 'e2e-back_t2':
                model = Classifier_Times(config=config, diffusion=diffusion,)
            elif model_args.experiment == 'e2e-back':
                model = Classifier_GPT2(config=config, diffusion=diffusion,)
            elif model_args.experiment == 'prot250-e2e-back':
                model = Classifier_GPT2(config=config, diffusion=diffusion,)


            print(model)

            ## SL rewrote
            filename = model_args.init_emb
            path_save = '{}/random_emb.torch'.format(filename)
            path_learned = '{}/ema_0.9999_200000.pt'.format(filename)
            path_lm_vocab = '{}/vocab.json'.format(filename)

            print("  >> model_args.learned_emb: {}".format(model_args.learned_emb))

            if model_args.experiment == 'prot250-e2e-back':
                
                # SL: added a new option 'when_exist'
                if model_args.learned_emb == 'when_exist':
                    print("  >> loading the trained word-to-embedding layer from LM: {}".format(path_learned))
                    # load the trained word embedding from LM
                    learned_embeddings = torch.load(path_learned)['word_embedding.weight']
                    # vocab size of LM
                    lm_vocab_size = learned_embeddings.shape[0]

                    assert model.transformer.wte.weight.shape[1] == learned_embeddings.shape[1]

                    # load the vocab from LM
                    with open(path_lm_vocab, 'r', ) as f:
                        lm_vocab = json.load(f)
                    
                    # update the embedding weights of the vocabs that exist in LM
                    print("  >> Vocab of the classifier: {}".format(tokenizer))
                    print("  >> Vocab of the LM: {}".format(lm_vocab))

                    print("  >> Size of vocab of the classifier: {}".format(len(tokenizer)))
                    print("  >> Size of vocab of the LM: {}".format(len(lm_vocab)))

                    # iterate over the vocabulary (vcb)
                    for vcb_token, vcb_idx in tokenizer.items():
                        if vcb_token in lm_vocab:
                            print("  >> Cloning the embedding of <{}>".format(vcb_token))
                            # update the word embedding layer weights
                            #print(model.transformer.wte.weight[vcb_idx, :])
                            #print(learned_embeddings[lm_vocab[vcb_token], :])
                            model.transformer.wte.weight[vcb_idx, :].data.copy_(learned_embeddings[lm_vocab[vcb_token], :].detach().clone())
                            #print(model.transformer.wte.weight[vcb_idx, :])
                            #print(learned_embeddings[lm_vocab[vcb_token], :])
                    # freeze the word embedding
                    model.transformer.wte.weight.requires_grad = False

                elif model_args.learned_emb == 'no':
                    model.transformer.wte.load_state_dict(torch.load(path_save))
                    model.transformer.wte.weight.requires_grad = False

                elif model_args.learned_emb == 'yes':
                    print('  >> loading the learned embeddings')
                    learned_embeddings = torch.load(path_learned)['word_embedding.weight']
                    print(learned_embeddings.shape)
                    print("before")
                    print(model.transformer)
                    model.transformer.wte.weight.data = learned_embeddings.clone()
                    model.transformer.wte.weight.requires_grad = False
                    print("after")
                    print(model.transformer)

        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names  # ['text']
    else:
        column_names = raw_datasets["validation"].column_names

    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    print('\n' + '=' * 100)
    print("\tClassifier Model")
    print('=' * 100 + '\n')
    print(model)
    print("# of trainable parameters: {}".format(sum(param.numel() for param in model.parameters() if param.requires_grad)))

    if model_args.experiment.startswith('prot250-e2e-back'):

        def tokenize_function(examples):
            vocab_dict = raw_datasets.vocab
            with CaptureLogger(tok_logger) as cl:
                if model_args.experiment == 'prot250-e2e-back':
                    input_ids = [[0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for (seq, _) in examples['text']]
                    src_ids = [ [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for (_, seq) in examples['text']]
                    result_dict = {'word_ids': input_ids, 'src_ids':src_ids}
                    

                elif model_args.experiment == 'e2e-back-gen':
                    input_strings = [
                        " ".join(attributes) + tokenizer.bos_token + " ".join(words) + tokenizer.eos_token
                        for (words, attributes) in examples['text']]
                    return tokenizer(input_strings, max_length=100, padding='max_length', truncation=True)
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
                )
            return result_dict

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        def pad_function(group_lst):
            if model_args.experiment in ['e2e-back', 'prot250-e2e-back']:
                vocab_dict = raw_datasets.vocab
                max_length = 256  # SL edited, was 64
                seqlen = 64  # SL edited, was 64

                group_lst['word_ids'] = _collate_batch_helper(group_lst['word_ids'], vocab_dict['PAD'], max_length)
                max_src_length = max([len(xx) for xx in group_lst['src_ids']])
                # print(max_src_length, seqlen)
                max_src_length = min(seqlen, max_src_length)
                group_lst['src_ids'], group_lst['src_mask'] = _collate_batch_helper(group_lst['src_ids'],
                                                                                    vocab_dict['PAD'],
                                                                                    max_src_length,
                                                                                    return_mask=True)

                group_lst['input_ids'] = [x + y  for (x,y) in zip(group_lst['word_ids'], group_lst['src_ids'])]
                group_lst['labels'] = [[-100] * len(x) + y for (x, y) in zip(group_lst['word_ids'], group_lst['src_ids'])]

            elif model_args.experiment == 'e2e-back-gen':
                group_lst['labels'] = group_lst['input_ids']
            return group_lst

        # def pad_function2(group_lst):
        #     vocab_dict = raw_datasets.vocab
        #     max_length = 64
        #     seqlen = 64
        #     group_lst['word_ids'] = _collate_batch_helper(group_lst['word_ids'], vocab_dict['PAD'], max_length)
        #     max_src_length = max([len(xx) for xx in group_lst['src_ids']])
        #     # print(max_src_length, seqlen)
        #     max_src_length = min(seqlen, max_src_length)
        #     group_lst['src_ids'], group_lst['src_mask'] = _collate_batch_helper(group_lst['src_ids'],
        #                                                                         vocab_dict['PAD'],
        #                                                                         max_src_length,
        #                                                                         return_mask=True)
        #     group_lst['input_ids'] = group_lst['word_ids']
        #     group_lst['tgt_ids'] = group_lst['src_ids']
        #     group_lst['labels'] = [[-100] * (len(x) * 2) + y for (x, y) in zip(group_lst['word_ids'], group_lst['src_ids'])]
        #
        #     return group_lst

        with training_args.main_process_first(desc="grouping texts together"):
            lm_datasets = tokenized_datasets.map(
                pad_function, #if model_args.experiment == 'e2e-back' else pad_function2,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )
    

    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            print(logits[0].shape, logits[1].shape)
            if type(logits) == tuple:
                return logits[0].argmax(dim=-1)
            else:
                return logits.argmax(dim=-1)

        metric = load_metric("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    trainer_tokenizer = None if ((model_args.experiment in ['prot250-e2e-back', 'e2e-back', 'e2e-back_t2']
                                 or model_args.experiment in ['synth_emb', 'pos_emb', 'roc_emb', 'simple-wiki_emb', 'e2e-tgt_emb'])
                                 and model_args.task not in ['data_teacher', 'finetune']) \
                        else tokenizer
    
    print(tokenizer)
    print(trainer_tokenizer)
    
    print(train_dataset)
    print(len(train_dataset))
    print(train_dataset[0])
    

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=trainer_tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        # compute_metrics=compute_metrics if training_args.do_eval else None,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint


        train_result = trainer.train(resume_from_checkpoint=checkpoint)


        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
