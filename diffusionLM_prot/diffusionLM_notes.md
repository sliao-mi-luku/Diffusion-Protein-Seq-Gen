# Diffusion-LM for Protein Sequence Generation

Steps for running Diffusion-LM (Li et al., 2022) to generation protein sequences.

*Last Updated: 2023-09-22*

**References**
1. https://github.com/XiangLi1999/Diffusion-LM
2. https://arxiv.org/abs/2205.14217


This document shows how to apply the Diffusion-LM framework to generating protein sequences. We will:

1. Clone the original Diffusion-LM repository.
2. Prepare protein sequences data.
3. Replace original codes with revised codes
4. Generate protein sequences with Diffusion-LM (our main goal) by:
    - Classifier-free conditional generation
    - Classifier-guided conditional generation



## Clone the original Diffusion-LM repository

1. Clone the [original Diffusion-LM repository](https://github.com/XiangLi1999/Diffusion-LM).
2. In the example, the original repository is clone to the directory `orig/Diffusion-LM`
3. There should be three subfolders:
    - `orig/Diffusion-LM/datasets`
    - `orig/Diffusion-LM/improved-diffusion`
    - `orig/Diffusion-LM/tranformers`
4. Follow the instructions from the original repository and setup the Conda environment.

## Prepare protein sequences data

1. Download data from [Kaggle](https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/data) (Novozymes Enzyme Stability Prediction)
2. (Optional) Use the file `train_updates_20220929.csv` to update `train.csv`
3. Create a subfolder `prot_data` under `orig/Diffusion-LM/datasets`
4. Copy the files `train.csv` and `test.csv` to the folder `orig/Diffusion-LM/datasets/prot_data`


## Replace original codes with revised codes

1. Move (and replace) all files from `prot_improved_diffusion` to `orig/Diffusion-LM/improved-diffusion`
2. Move (and replace) all files from `prot_scripts` to `orig/Diffusion-LM/scripts`


## Generate protein sequences with Diffusion-LM 

Activate the Conda environment (instructed by the original Diffusion-LM repository) and run the following commands.

### Classifier-Free Conditional Sequence Generation

#### Step 1: Learn the language model with diffusion 

```terminal
# cd Diffusion-LM/improved-diffusion
python scripts/run_train.py --modality prot250-tm-prefix --diff_steps 2000 --model_arch transformer --image_size 16 \
                            --hidden_size 128 --bsz 8 --lr 0.0001 --lr_anneal_steps 200000 --seed 102 \
                            --noise_schedule sqrt --in_channel 16 --submit no --padding_mode pad \
                            --app "--experiment_mode lm --vocab_size 29 --predict_xstart True --training_mode e2e " \
                            --notes MMDD-bert96
```
After training, diffusion model will be saved in:
`diffusion/diffusion_models/`

#### Step 2: Classifier-free conditional generation with diffusion

```terminal
# cd Diffusion-LM/improved-diffusion
python ./scripts/infill_prot_length.py --eval_task_ length --model_path "./diffusion_models/ema_0.9999_050000.pt" \
        --partial_seq "START medium |" --tgt_len 200 --num_samples 400 --use_ddim True \
        --eta 1. --verbose pipe --out_dir "infill_length_out" --notes MMDD-tm-medium-L200-N400
```
Generated sequences are saved in `Diffusion-LM/improved-diffusion/infill_length_out`


### Classifier-Guided Conditional Sequence Generation

To run in this mode, set `modality = 'prot250'`

#### Step 1: Learn the language model with diffusion 

Example of training a BERT-based model with hidden dimension `hidden_dim = 96`.

```terminal
# cd Diffusion-LM/improved-diffusion
python scripts/run_train.py --modality prot250 --diff_steps 2000 --model_arch transformer --image_size 16 \
                            --hidden_size 128 --bsz 8 --lr 0.0001 --lr_anneal_steps 200000 --seed 102 \
                            --noise_schedule sqrt --in_channel 16 --submit no --padding_mode pad \
                            --app "--experiment_mode lm --vocab_size 25 --predict_xstart True --training_mode e2e " \
                            --notes MMDD-bert96
```

After training, diffusion model will be saved as:
`diffusion/diffusion_models/diff_prot250_pad_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_MMDD-bert96`

#### Step 2: Train a classifier

Example ot training a gpt2 for tm classification.

```terminal
# cd Diffusion-LM
python train_run.py --experiment prot250-e2e-back \
    --app "--init_emb ./improved-diffusion/diffusion_models/diff_prot250_pad_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_MMDD-bert96 \
           --n_embd 16 --learned_emb when_exist " \
    --epoch 10 --bsz 4 --notes MMDD-tm
```

After training, the classifier model will be saved as:
`Diffusion-LM/classifier_models/prot250-e2e-back_e=10_b=4_m=gpt2_wikitext-103-raw-v1_101_wp_MMDD-tm`

#### Step 3: Classifier-guided conditional generation with diffusion

Use the trained classifier (**GPT2** in this example) to genertate protein sequences with desired proterty (`tm` in this example).

```terminal
# cd Diffusion-LM
python ./scripts/infill_prot_clf.py --model_path ./diffusion_models/diff_prot250_pad_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_MMDD-bert96/ema_0.9999_200000.pt \
    --eval_task_ control_attribute --partial_seq START --tgt_len 200 --num_samples 400 --use_ddim True --eta 1.0 --verbose pipe \
    --out_dir clf_out_gen --clf_path ../classifier_models/prot250-e2e-back_e=10_b=4_m=gpt2_wikitext-103-raw-v1_101_wp_MMDD-tm --notes MMDD-bert96-tm
```

Generated sequences will be saved to:
`Diffusion-LM/improved-diffusion/clf_out_gen/diff_prot250_pad_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_MMDD-bert96.ema_0.9999_200000.pt.infill_control_attribute_MMDD-bert96-tm`
