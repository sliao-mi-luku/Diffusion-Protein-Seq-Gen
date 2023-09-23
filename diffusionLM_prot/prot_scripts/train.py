import os
import json
import argparse
import torch
import numpy as np
from improved_diffusion import dist_util, logger
from improved_diffusion.text_datasets import load_data_text
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from transformers import set_seed
from functools import partial
from improved_diffusion.test_util import get_weights, compute_logp
from improved_diffusion.rounding import load_models
import torch.distributed as dist
import wandb


def main():

    # parse args
    args = create_argparser().parse_args()
    set_seed(args.seed)
    dist_util.setup_dist()
    logger.configure()

    assert args.modality.startswith('prot')


    """
    Create diffusion model and diffusion process
    """

    # SL: create model (BERT-based) and the diffusion process
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.to(dist_util.dev())

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'Total number of parameters: {pytorch_total_params}')


    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f'Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)


    wandb.init(project=os.getenv("WANDB_PROJECT", "diffusion_lm"), name=args.checkpoint_path)
    wandb.config.update(args.__dict__, allow_val_change=True)


    if args.experiment_mode == 'conditional_gen':
        assert args.modality in ['e2e', 'prot250'] and args.padding_mode == 'pad'

    

    """
    Load Dataset
    """

    logger.log("creating data loader...")

    data = load_data_text(data_dir=args.data_dir,  # dummy
                          batch_size=args.batch_size,      
                          image_size=args.image_size,      
                          class_cond=args.class_cond,  # dummy
                          data_args = args,
                          task_mode=args.modality,
                          padding_mode=args.padding_mode,
                          load_vocab=None,
                          model=None)
    next(data)


    ## Prepare validation data generator
    # SL: load_models() is exptected to produce a torch.embedding layer and load a tokenizer
    # SL: model2 is expected to be a single embedding layer of (d_in, d_out) = (vocab_size, in_channel)
    # SL: tokenizer is a id2token dict
    model2, tokenizer = load_models(modality=args.modality, 
                                    mode=args.experiment,
                                    model_name_or_path=args.model_name_or_path,    # SL: dummy variable
                                    emb_dim=args.in_channel,            
                                    file=args.checkpoint_path,
                                    extra_args=args)
    
    rev_tokenizer = {v: k for k, v in tokenizer.items()}  # token2id

    print("\n tokenizer")
    print(tokenizer)
    print("\n rev_tokenizer")
    print(rev_tokenizer)

    data_valid = load_data_text(data_dir=args.data_dir,  # dummy
                                batch_size=args.batch_size,      
                                image_size=args.image_size,      
                                class_cond=args.class_cond,  # dummy
                                data_args=args,
                                task_mode=args.modality,
                                padding_mode=args.padding_mode,
                                split='valid',
                                load_vocab=rev_tokenizer,  # SL: {'START': 0, ...}
                                model=model2,  # SL: Embedding layer
                                )                    
    

    # dist.barrier()
    # import time
    # while not os.path.exists(os.path.join(args.checkpoint_path, 'vocab.json')):
    #     time.sleep(1)


    def get_mapping_func(args, diffusion, data):

        # SL: model2 = Embedding(vocab_size, in_channel), tokenizer = {0: 'START', ...}
        model2, tokenizer = load_models(modality=args.modality,
                                        mode=args.experiment,
                                        model_name_or_path=args.model_name_or_path,
                                        emb_dim=args.in_channel,
                                        file=args.checkpoint_path,
                                        extra_args=args)
        
        # SL TODO: understand what happened
        model3 = get_weights(model2, args)

        mapping_func = partial(compute_logp, args, model3.cuda())
        diffusion.mapping_func = mapping_func
        return mapping_func


    get_mapping_func(args, diffusion, data)

    print("\ndiffusion.mapping_func")
    print(diffusion.mapping_func)

    print("\nmodel")
    print(model)

    """
    Train the model
    """

    logger.log("training...")
    
    TrainLoop(
        model=model,               # SL: transformer-based model
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval
    ).run_loop()



def create_argparser():
    defaults = dict(data_dir="",
                    schedule_sampler="uniform",
                    lr=1e-4,
                    weight_decay=0.0,
                    lr_anneal_steps=0,
                    batch_size=1,
                    microbatch=-1,  # -1 disables microbatches
                    ema_rate="0.9999",  # comma-separated list of EMA values
                    log_interval=50,
                    save_interval=50000,
                    resume_checkpoint="",
                    use_fp16=False,
                    fp16_scale_growth=1e-3,
                    seed=101,
                    gradient_clipping=-1.0,
                    eval_interval=2000,
                    checkpoint_path='diff_models'
                    )
    
    text_defaults = dict(modality='prot250',  # SL: was 'text'
                         dataset_name='',  # SL: was 'wikitext'
                         dataset_config_name=None,  # SL: was 'wikitext-2-raw-v1'
                         config=None,  # SL: was 'diffusion_lm/synthetic_data/configs/emnlp2020/experiments/difflm_seed0_m3_k128_trainc20000.yaml'
                         model_name_or_path=None,  # SL: was 'predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None'
                         experiment='random',  # was 'gpt2_pre_compress'
                         model_arch='conv-unet',
                         roc_train='diffusion_lm/ROCstory',#'diffusion_lm/ROCstory/ROCstory17.csv',
                         wiki_train='',  # SL: was 'diffusion_lm/simple_wiki/data.v1.split/simple.training.txt'
                         e2e_train='e2e_data',
                         yelp_train='',  # SL: was 'diffusion_lm/yelpnlg-resources/yelpnlg-corpus'
                         commonGen_train = '',  # SL: was 'diffusion_lm/common-gen/commongen_data'
                         prot_data_dir='../datasets/prot_data',  # SL added
                         emb_scale_factor=1.0,
                         noise_level=0.0,
                         cache_mode='no',
                         use_bert_tokenizer='no',
                         padding_mode='block',
                         preprocessing_num_workers=1)
    
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
