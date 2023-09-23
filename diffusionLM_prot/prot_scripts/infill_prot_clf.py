"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json, sys
#import stanza
#import spacy_stanza
import numpy as np
import torch as th
from transformers import set_seed
import torch.distributed as dist
from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer
from improved_diffusion.test_util import get_weights, denoised_fn_round
from functools import partial
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
sys.path.insert(0, '../transformers/examples/pytorch/language-modeling')

from custom_trainer import Classifier_GPT2, Classifier_Times, Classifier_POS, Classifier_Tree
from infill_util import langevin_fn3, get_score, langevin_fn3_compose, langevin_fn1, langevin_fn4, langevin_fn_tree, langevin_fn_length, langevin_fn_prot
from spacy.lang.en import English



def main():

    print("\n")

    set_seed(101)
    
    # parse args
    args = create_argparser().parse_args()

    # load configurations of diffusion LM
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print("  >> loading diffusion model config from :{}".format(config_path))
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    
    # update args with diffusion params
    args.__dict__.update(training_args)

    args.noise_level = 0.0
    args.sigma_small = True

    # use 200 diffusion steps
    if args.eval_task_.startswith('control_'):
        args.diffusion_steps = 200

    print("  >> update args: {}".format(args))

    dist_util.setup_dist()
    logger.configure()
    print(args.clip_denoised, 'clip_denoised')

    
    ## Load diffusion model, expected to be TransformerNetModel2()
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # load the weights
    model.load_state_dict(th.load(args.model_path))
    model.to(dist_util.dev())
    model.eval()


    ## Load word embedding and tokenizer (for LM)
    # SL: model_embs expected to be Embedding(vocab_size, in_channel), where vocab_size is dataset-wise
    # SL: tokenizer is id2token
    logger.log("load embedding models")
    print(os.path.split(args.model_path)[0])
    model_embs, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                   os.path.split(args.model_path)[0])
    
    print("  >> model_embs: {}".format(model_embs))
    print("  >> tokenizer: {}".format(tokenizer))
    
    if args.training_mode.startswith('e2e'):
        print('e2e, load the right model embeddings', '*'*80)
        model_embs.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())

    model_embs = model_embs.cuda()

    # SL: model3 is cloning model_embs, i.e., Embedding(vocab_size, in_channel), with LM's vocab
    model3 = get_weights(model_embs, args)



    ## Process the partial sequence
    # SL: partial_seq expected to be "START"
    logger.log('load the partial sequences')
    if args.partial_seq:
        partial_seq = [args.partial_seq]
        partial_seq_idx = ['0']    



    if args.modality == 'e2e-tgt' or args.modality.startswith('prot'):

        tokens2id = {v:k for k, v in tokenizer.items()}
        todo_pad_token = -1
        pad_token = tokens2id['PAD']   # SL: PAD token expected to be 3

        # SL: encoded_partial_seq expected to be [Tensor([0])]
        # SL: encoded_partial_seq will be overwritten later.
        encoded_partial_seq = [th.LongTensor([tokens2id.get(x, tokens2id['UNK']) for x in seq.split()]) for seq in partial_seq]


        ## Length control (Method 1)
        if args.eval_task_ == 'length':
            right_length = args.image_size ** 2 - len(encoded_partial_seq[0])
            # right_length = args.tgt_len - len(encoded_partial_seq[0])
            # assert args.tgt_len > len(encoded_partial_seq[0])
            right_pad = th.empty(right_length).fill_(todo_pad_token).long()
            encoded_partial_seq = [th.cat([seq, right_pad], dim=0) for seq in encoded_partial_seq]
            encoded_partial_seq[0][args.tgt_len-1] = tokens2id['END']
            encoded_partial_seq[0][args.tgt_len] = tokens2id['START']
            # encoded_partial_seq[0][args.tgt_len+1:] = tokens2id['PAD']

        ## Attribute control
        elif args.eval_task_.startswith('control'):
            # right_pad = th.empty(args.tgt_len+2).fill_(pad_token).long()
            # TO FIX... IMPORTANT.

            # SL: no constraint on length, use default
            if 'length' not in args.eval_task_:
                print("  >> Generating sequences with default length = {}".format(args.tgt_len))  # args.tgt_len includes START, \n and END
                
                if False:
                    # create a template of size image_size**2
                    right_pad = th.empty(256).fill_(pad_token).long()  # SL: (256, ). Changed from 64 to 256
                    # convert into a tensor
                    encoded_partial_seq = [th.cat([right_pad], dim=0)]  # SL: list(Tensor([3, 3, 3, 3, ...]))
                    # fill in START, \n, and END tokens
                    encoded_partial_seq[0][0] = tokens2id['START']
                    encoded_partial_seq[0][args.tgt_len-2] = tokens2id['\n']  # SL added
                    encoded_partial_seq[0][args.tgt_len-1] = tokens2id['END']  # SL edited
                    print(encoded_partial_seq)
                else:
                    right_length = args.image_size ** 2 - len(encoded_partial_seq[0])  # expected to be 16**2 - 1 = 255
                    right_pad = th.empty(right_length).fill_(todo_pad_token).long()  # expected to be [-1, -1, ..., -1]. len = 255
                    encoded_partial_seq = [th.cat([seq, right_pad], dim=0) for seq in encoded_partial_seq]
                    encoded_partial_seq[0][args.tgt_len-1] = tokens2id['END']
                    encoded_partial_seq[0][args.tgt_len] = tokens2id['START']

                    
            if args.eval_task_ == 'control_attribute':
                # load trained classifier (original GPT2 implementation)
                model_control = Classifier_GPT2.from_pretrained(args.clf_path).cuda()
                print("  >> Classifier ({}) loaded".format(type(model_control)))
                print(model_control)
                
                # controling labels (ex. ["Type", ":", "enzyme"] for 1st line, ["tm", ":", "low"] for 2nd line, ...)
                control_label_lst = []
                
                #with open('../datasets/control_target/target_attribute_prot.json', 'r') as controlf:
                with open('../datasets/control_target/target_attribute_prot_tm.json', 'r') as controlf:
                    for line in controlf:
                        control_label_lst.append(json.loads(line))
                print("Control_label_lst: {}".format(control_label_lst))

                control_constraints = []
                for label_class in control_label_lst:

                    print("  >> label_class = {}".format(label_class))
                    
                    # SL changed from 64 to 256. length of label = 256 + 3
                    label = [-100] * 256 + [tokens2id.get(x, tokens2id['UNK']) for x in label_class]  # [25, 26, 27]
                    label_ids = th.tensor(label).unsqueeze(0)  # SL: (3, )
                    debug_lst = []
                    langevin_fn_selected = partial(langevin_fn_prot, debug_lst, model_control, model3.cuda(),  # SL replaced langevin_fn3
                                                   label_ids.expand(args.batch_size, -1), 0.1)
                    control_constraints.append((langevin_fn_selected, label_class))
                
                print("  >> Using batch_size = {}".format(args.batch_size))

                # pre-fill the sequence
                partial_seq = control_constraints
                encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(partial_seq))]  # SL len(partial_seq) = 7
                assert len(partial_seq) == len(encoded_partial_seq)


            elif args.eval_task_ == 'control_length':
                control_length_lst = [args.tgt_len]  # control_length_lst = list(range(10, 41)) #[40] #[10, 20, 30]
                control_constraints = []
                for target_length in control_length_lst:
                    encoded_partial_seq = [th.LongTensor([0])]
                    right_length = args.image_size ** 2 - len(encoded_partial_seq[0])
                    # right_length = args.tgt_len - len(encoded_partial_seq[0])
                    # assert args.tgt_len > len(encoded_partial_seq[0])
                    right_pad = th.empty(right_length).fill_(todo_pad_token).long()
                    encoded_partial_seq = [th.cat([seq, right_pad], dim=0) for seq in encoded_partial_seq]
                    encoded_partial_seq[0][target_length - 1] = tokens2id['END']
                    # encoded_partial_seq[0][target_length] = tokens2id['START']
                    #print(encoded_partial_seq[0], todo_pad_token)
                    #print(tokens2id)  # SL added
                    partial_mask = (encoded_partial_seq[0] == todo_pad_token).unsqueeze(0).expand(args.batch_size, -1)
                    # print(partial_mask[0])
                    label = encoded_partial_seq[0]
                    label_ids = th.tensor(label).unsqueeze(0)  # SL: [1, seqlen]
                    label_ids = label_ids.masked_fill(label_ids == todo_pad_token, 3)
                    tgt_embs = model3.cuda()(label_ids.cuda())
                    langevin_fn_selected = partial(langevin_fn_length, 0.01, diffusion, partial_mask, model,
                                                   tgt_embs.expand(args.batch_size, -1, -1), 0.1)
                    control_constraints.append((langevin_fn_selected, (str(target_length),)))
                partial_seq = control_constraints
                # print(control_constraints)
                encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(partial_seq))]
                assert len(partial_seq) == len(encoded_partial_seq)

    logger.log("sampling...")


    print('\n========================\n')
    print("Sampling ")
    print("Generating {} samples".format(args.num_samples))
    print('\n========================\n')


    sample_dict = {}
    # model3 = get_weights(model_embs, args)
    if True:
        for (encoded_seq, control_helper) in zip(encoded_partial_seq, partial_seq):

            # SL: encoded_seq is a 1-D tensor of length args.tgt_len, encoded_seq[0] = 0, encoded_seq[-1] = 1, otherwise = 3
            # SL: control_helper is tuple(langevin_fn_selected,  ['Type', ':', 'enzyme'])

            all_images = []
            all_labels = []
            
            while len(all_images) * args.batch_size < args.num_samples:

                model_kwargs = {}

                ## SL: original implementation - error may occur because encoded_seq changes sizes
                #encoded_seq = encoded_seq.unsqueeze(0).expand(args.batch_size,-1)  # SL: [bsz, 256]
                #print(model_embs.weight.device, encoded_seq.device)  # SL: [cuda, cpu]
                # SL: partial_mak_temp is all FALSE if args.eval_task_ == 'control_attribute'
                #partial_mask_temp = (encoded_seq == todo_pad_token).view(args.batch_size, -1)  # SL: [bsz, 256]
                # # encoded_seq[encoded_seq == todo_pad_token] = 0
                # SL: encoded_seq doesn't change if args.eval_task_ == 'control_attribute'
                #encoded_seq.masked_fill_(encoded_seq == todo_pad_token, 3)  # SL: [bsz, 256]
                #encoded_seq_hidden = model_embs(encoded_seq.cuda())  # SL: [bsz, 256, emb_dim]
                #seqlen = encoded_seq.size(1)  # SL: 256


                ### SL: new implementation - clone encoded_seq to avoid errors
                encoded_seq_copy = encoded_seq.clone()
                encoded_seq_copy = encoded_seq_copy.unsqueeze(0).expand(args.batch_size,-1)  # SL: [bsz, seq_len]
                partial_mask_temp = (encoded_seq_copy == todo_pad_token).view(args.batch_size, -1)  # SL: [bsz, seq_len]
                #  # encoded_seq[encoded_seq == todo_pad_token] = 0
                encoded_seq_copy.masked_fill_(encoded_seq_copy == todo_pad_token, 3)  # SL: pending edit to clear warning message.
                encoded_seq_hidden = model_embs(encoded_seq_copy.cuda())  # SL: [bsz, seq_len, emb_dim]
                seqlen = encoded_seq_copy.size(1)

                if args.model_arch == '1d-unet':
                    encoded_seq_hidden = encoded_seq_hidden.permute(0, 2, 1)
                    partial_mask = partial_mask_temp.unsqueeze(1).expand(-1, args.in_channel, -1)
                    sample_shape = (args.batch_size, args.in_channel, seqlen)

                else:
                    partial_mask = partial_mask_temp.unsqueeze(-1).expand(-1, -1, args.in_channel)  # SL: [bsz, 256, emb_dim]
                    sample_shape = (args.batch_size, seqlen, args.in_channel, )  # SL: (bsz, 256, emb_dim)
                    
                
                
                if args.eval_task_.startswith('control'):

                    # SL: control_helper is tuple(langevin_fn_selected,  ['Type', ':', 'enzyme']), where langevin_fn_selected
                    #     is a tuple(langevin_fn, input_ids+src_ids, step_size=0.1)

                    langevin_fn_selected, label_class_attributes = control_helper  

                    print("  >> Condition ----> {}".format(label_class_attributes))

                    # loop_func_ = diffusion.p_sample_loop_langevin_progressive

                    if args.use_ddim:
                        loop_func_ = diffusion.ddim_sample_loop_progressive
                    else:
                        loop_func_ = diffusion.p_sample_loop_progressive

                    for sample in loop_func_(
                            model,
                            sample_shape,
                            denoised_fn=partial(denoised_fn_round, args, model3.cuda()),
                            # denoised_fn=partial(langevin_early, model_control, model3.cuda(),
                            #                     label_ids.expand(args.batch_size, -1), 0.1),
                            clip_denoised=args.clip_denoised,
                            model_kwargs=model_kwargs,
                            device=encoded_seq_hidden.device,
                            langevin_fn=langevin_fn_selected,
                            eta=args.eta,
                            # langevin_func=partial(langevin_func, model_control,
                            #                       label_ids.expand(args.batch_size, -1), 0.01),
                            ):
                        final = sample["sample"]  # SL: [bsz, seqlen, emb_dim]


                # SL: for `length` task
                else:
                    label_class_attributes = control_helper  # SL: expected to be 'START'
                    loop_func_ = diffusion.p_sample_loop_progressive_infill

                    for sample in loop_func_(
                            model,
                            sample_shape,
                            encoded_seq_hidden,
                            partial_mask,
                            denoised_fn=partial(denoised_fn_round, args, model3.cuda()),
                            clip_denoised=args.clip_denoised,
                            model_kwargs=model_kwargs,
                            device=encoded_seq_hidden.device,
                            greedy=False,
                    ):
                        final = sample["sample"]
    
                sample = final  # SL: [bsz, seqlen, emb_dim]
    
                if args.model_arch == '1d-unet':
                    print(sample.shape)
                    sample = sample.permute(0, 2, 1)
                    print(sample.shape)

    
                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

                if args.class_cond:
                    gathered_labels = [
                        th.zeros_like(classes) for _ in range(dist.get_world_size())
                    ]
                    dist.all_gather(gathered_labels, classes)
                    all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
                logger.log(f"created {len(all_images) * args.batch_size} samples")


                # SL added a forced break here (original issue of the shape change of encoded_seq)
                #break
    
            arr = np.concatenate(all_images, axis=0)
            arr = arr[: args.num_samples]  # SL: [bsz, seqlen, emb_dim]

            if args.verbose == 'pipe':
                sample_dict[tuple(label_class_attributes)] = arr
                print(f'writing to sample_dict, for class {" ".join(label_class_attributes)}')


    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

    if dist.get_rank() == 0:  # SL: expected here
        shape_str = "x".join([str(x) for x in arr.shape])
        model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'

    dist.barrier()
    logger.log("sampling complete")


    # SL commented: rounding function to map from embedding to discrete tokens
    def decode_helper(args, sample_dict, diff_model=None):
        result_dict = {}
        if not diffusion.training_mode.startswith('e2e'):
            logger.log('decode by rounding. ')
            set_seed(101)
            model, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                           os.path.split(args.model_path)[0])

        for k, v in sample_dict.items():
            arr = v
            if diffusion.training_mode.startswith('e2e'):
                word_lst_e2e = []
                print('decoding for e2e', )
                x_t = th.tensor(arr).cuda()
                print(x_t.shape)
                if args.model_arch == 'conv-unet':
                    reshaped_x_t = x_t.view(x_t.size(0), -1, x_t.size(-1))
                else:
                    reshaped_x_t = x_t
                logits = diff_model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
                cands = th.topk(logits, k=1, dim=-1)
                tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
                for seq in cands.indices:
                    tokens = " ".join([tokenizer[x[0].item()] for x in seq])
                    word_lst_e2e.append(tokens)
                word_lst = word_lst_e2e
            else:
                word_lst = rounding_func(args.experiment, arr, model, tokenizer)
            result_dict[k] = word_lst
        return result_dict
    

    # SL commented: save to json
    if args.verbose == 'pipe':

        print(f'sampled for {len(sample_dict)} control tasks')

        # SL added 
        if not os.path.isdir(args.out_dir):
            print("{} does not exist. Creating the folder".format(args.out_dir))
            os.mkdir(args.out_dir)

        # decode
        result_dict = decode_helper(args, sample_dict, diff_model=model)

        # json file
        out_path_pipe = os.path.join(args.out_dir, f"{model_base_name}.infill_{args.eval_task_}_{args.notes}.json")
        fout = open(out_path_pipe, 'w')
        for k, word_lst in result_dict.items():
            print({k:word_lst}, file=fout)
        fout.close()
        print(f'written the decoded output to {out_path_pipe}')
        out_path2 = out_path_pipe
        
    args.out_path2 = out_path2
    return args


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=False,
        use_ddim=False,
        eta=1.0,
        num_samples=50,
        batch_size=1,
        model_path="",
        out_dir="clf_out_gen",   # SL edited. Was "diffusion_lm/improved_diffusion/out_gen"
        emb_scale_factor=1.0,
        split='train',
        debug_path='',
        eval_task_='infill',
        partial_seq="",
        partial_seq_file="",
        verbose='yes',
        tgt_len=15,
        t_merge=200,
        interp_coef=0.5,
        notes='',
        start_idx=0,
        end_idx=0,
        clf_path = '',  # SL added
        )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def eval(args):
    if args.modality == 'e2e-tgt':
        model_name_path = "predictability/diff_models/e2e-tgt_e=15_b=20_m=gpt2_wikitext-103-raw-v1_101_None"

        COMMAND = f"python scripts/ppl_under_ar.py " \
              f"--model_path {args.model_path} " \
              f"--modality {args.modality}  --experiment random " \
              f"--model_name_or_path {model_name_path} " \
              f"--input_text {args.out_path2}  --mode eval"
        print(COMMAND)
        os.system(COMMAND)


if __name__ == "__main__":
    args = main()
    import numpy as np
    if args.verbose != 'pipe':
        eval(args)

