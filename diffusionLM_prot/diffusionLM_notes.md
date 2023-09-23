# Diffusion-LM for Protein Sequence Generation

Steps for running Diffusion-LM (Li et al., 2022) to generation protein sequences.

*Last Updated: 2023-09-22*

**References**
1. https://github.com/XiangLi1999/Diffusion-LM
2. https://arxiv.org/abs/2205.14217


## Setting Up

### Original Diffusion-LM Repository

Clone the [original Diffusion-LM repository](https://github.com/XiangLi1999/Diffusion-LM).

### Data

Down data from [Kaggle](https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/data) (Novozymes Enzyme Stability Prediction).


###  

## Classifier-Free Conditional Sequence Generation

## Classifier-Guided Conditional Sequence Generation

To run in this mode, set `modality = 'prot250'`

### Step 1: Learn the language model with diffusion 

Example of training a BERT-based model with hidden dimension `hidden_dim = 96`.

```terminal
# cd Diffusion-LM/improved-diffusion
python scripts/run_train.py --modality prot250 --diff_steps 2000 --model_arch transformer --image_size 16 \
                            --hidden_size 128 --bsz 8 --lr 0.0001 --lr_anneal_steps 200000 --seed 102 \
                            --noise_schedule sqrt --in_channel 16 --submit no --padding_mode pad \
                            --app "--experiment_mode lm --vocab_size 25 --predict_xstart True --training_mode e2e " \
                            --notes 0528-bert96
```

After training, diffusion model will be saved as:
`diffusion/diffusion_models/diff_prot250_pad_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_0528-bert96`

### Step 2: Train a classifier

Example ot training a gpt2 for tm classification.

```terminal
# cd Diffusion-LM
python train_run.py --experiment prot250-e2e-back \
    --app "--init_emb ./improved-diffusion/diffusion_models/diff_prot250_pad_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_0528-bert96 \
           --n_embd 16 --learned_emb when_exist " \
    --epoch 10 --bsz 4 --notes 0529-tm
```

After training, the classifier model will be saved as:
`Diffusion-LM/classifier_models/prot250-e2e-back_e=10_b=4_m=gpt2_wikitext-103-raw-v1_101_wp_0529-tm`

### Step 3: Classifier-guided diffusion generation

Use the trained classifier (**GPT2** in this example) to genertate protein sequences with desired proterty (`tm` in this example).

```terminal
# cd Diffusion-LM
python ./scripts/infill_prot_clf.py --model_path ./diffusion_models/diff_prot250_pad_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_0528-bert96/ema_0.9999_200000.pt \
    --eval_task_ control_attribute --partial_seq START --tgt_len 200 --num_samples 400 --use_ddim True --eta 1.0 --verbose pipe \
    --out_dir clf_out_gen --clf_path ../classifier_models/prot250-e2e-back_e=10_b=4_m=gpt2_wikitext-103-raw-v1_101_wp_0529-tm --notes 0529-bert96-tm
```

Generated sequences will be saved to:
`Diffusion-LM/improved-diffusion/clf_out_gen/diff_prot250_pad_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_0528-bert96.ema_0.9999_200000.pt.infill_control_attribute_0529-bert96-tm`
