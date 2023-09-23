import sys
import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training args.')
    parser.add_argument('--experiment', type=str, default='random', help='no-rep=gpt2gen, no-zipfs, has-rep=regular, rm-window-rep')

    parser.add_argument('--model_arch', type=str, default='conv-unet', help='')

    parser.add_argument('--modality', type=str, default='prot250', help='')  # SL: modified, default was 'synth'

    parser.add_argument('--noise_schedule', type=str, default='cosine', help='')
    parser.add_argument('--loss_type', type=str, default='Lsimple', help='')
    parser.add_argument('--dropout', type=str, default='0.1', help='')
    parser.add_argument('--weight_decay', type=str, default=0.0, help='')

    parser.add_argument('--image_size', type=int, default=8, help='')
    parser.add_argument('--hidden_size', type=int, default=128, help='')
    parser.add_argument('--in_channel', type=int, default=16, help='')
    parser.add_argument('--m', type=int, default=3, help='')
    parser.add_argument('--k', type=int, default=32, help='')
    parser.add_argument('--lr_anneal_steps', type=int, default=400000, help='')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='')

    parser.add_argument('--lr', type=float, default=1e-04, help='')
    parser.add_argument('--bsz', type=int, default=8, help='')
    parser.add_argument('--diff_steps', type=int, default=4000, help='')
    parser.add_argument('--padding_mode', type=str, default='block', help='')
    parser.add_argument('--seed', type=int, default=101, help='') # old is 42

    parser.add_argument('--notes', type=str, default=None, help='')
    parser.add_argument('--submit', type=str, default='no', help='')
    parser.add_argument('--use_big', type=str, default='no', help='')
    parser.add_argument('--app', type=str, default='', help='')


    args = parser.parse_args()

    folder_name = "diffusion_models/"

    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)


    if args.loss_type == 'Lsimple':
        train_setup = " --use_kl False --learn_sigma False "
    elif args.loss_type == 'Lhybrid':
        train_setup = " --use_kl False --learn_sigma True "
    elif args.loss_type == 'Lvlb':
        train_setup = " --use_kl True --learn_sigma True "
    else:
        assert False


    if args.experiment == 'random':
        exp_m = 'rand'
    if args.experiment == 'random1':
        exp_m = 'pred'
    elif args.experiment == 'gpt2_pre_compress':
        exp_m = 'emb'
    elif args.experiment == 'glove':
        exp_m = 'glo'



    # (SL) create a new modality for protein sequence
    if args.modality.startswith('prot'):
        Model_FILE = f"diff_{args.modality}_{args.padding_mode}_{exp_m}{args.in_channel}_{args.model_arch}_lr{args.lr}_{args.weight_decay}" \
                     f"_{args.diff_steps}_{args.noise_schedule}_{args.loss_type}_h{args.hidden_size}" \
                     f"_s{args.num_res_blocks}_d{args.dropout}_sd{args.seed}"


    if args.notes:
        Model_FILE = Model_FILE + f'_{args.notes}'


    # logging_dir = os.path.join(folder_name, 'runs', Model_FILE)
    Model_FILE = os.path.join(folder_name, Model_FILE)  # folder_name = "diffusion_models/"


    # (SL) create folder if not exist
    if not os.path.isdir(Model_FILE):
        os.mkdir(Model_FILE)

    app = " " + args.app

    # (SL) split command lines
    COMMANDLINE_A = f"set OPENAI_LOGDIR={Model_FILE}  "
    os.system(COMMANDLINE_A)

    # (SL) split command lines
    COMMANDLINE_B = f"set TOKENIZERS_PARALLELISM=false  "
    os.system(COMMANDLINE_B)

    COMMANDLINE = f"python scripts/train.py   " \
                  f"--checkpoint_path {Model_FILE} " \
                  f"--model_arch {args.model_arch} " \
                  f"--modality {args.modality} " \
                  f"--save_interval 50000 --lr {args.lr} " \
                  f"--batch_size {args.bsz}  " \
                  f"--diffusion_steps {args.diff_steps} " \
                  f"--noise_schedule {args.noise_schedule} {train_setup} " \
                  f"--image_size {args.image_size} --num_channels {args.hidden_size} --seed {args.seed} " \
                  f"--dropout {args.dropout} --in_channel {args.in_channel} --out_channel {args.in_channel} --padding_mode {args.padding_mode} " \
                  f"--experiment {args.experiment}  --lr_anneal_steps {args.lr_anneal_steps} --weight_decay {args.weight_decay} " \
                  f"--num_res_blocks {args.num_res_blocks} "

    # SL commented: add appendix info to the command line and save to bash file
    COMMANDLINE += app

    with open(Model_FILE + '.sh', 'w') as f:
        print(COMMANDLINE, file=f)

    print(COMMANDLINE)


    # SL commented: execute command (will run scripts/train.py)
    if args.submit == 'no':
        os.system(COMMANDLINE)  # textattack/roberta-base-ag-news # textattack/roberta-base-imdb
    # #
    elif args.submit == 'yes':
        if args.use_big == 'no':
            full_command = "nlprun  -g 1 -n {} -x jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard20 \'{}\'".format(
                Model_FILE, COMMANDLINE)
        elif True:
            full_command = "nlprun  -g 1 -n {} -x " \
                           "jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18," \
                           "jagupard19,jagupard20,jagupard21,jagupard22,jagupard23," \
                           "jagupard24,jagupard25\'{}\'".format(Model_FILE, COMMANDLINE)

        print(full_command)
        os.system(full_command)
