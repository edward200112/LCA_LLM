---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:1284600
- loss:BinaryCrossEntropyLoss
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model trained using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** 256 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'BertForSequenceClassification'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['Laziza Karahi Fry Meat Masala 90Gram Boxes Pack of 6', 'Meat Processed from Carcasses  This U.S. industry comprises establishments primarily engaged in processing or preserving meat and meat byproducts (except poultry and small game) from purchased meats.  This industry includes establishments primarily engaged in assembly cutting and packing of meats (i.e., boxed meats) from purchased meats.\n\nCross-References. Establishments primarily engaged in--'],
    ['Laziza Karahi Fry Meat Masala 90Gram Boxes Pack of 6', 'Rendering and Meat Byproduct Processing  This U.S. industry comprises establishments primarily engaged in rendering animal fat, bones, and meat scraps.\n\nCross-References.'],
    ['Laziza Karahi Fry Meat Masala 90Gram Boxes Pack of 6', 'Broilers and Other Meat Type Chicken Production  This industry comprises establishments primarily engaged in raising broilers, fryers, roasters, and other meat type chickens.\n\nCross-References.'],
    ['Laziza Karahi Fry Meat Masala 90Gram Boxes Pack of 6', 'Poultry Processing  This U.S. industry comprises establishments primarily engaged in (1) slaughtering poultry and small game and/or (2) preparing processed poultry and small game meat and meat byproducts.\n\nCross-References. Establishments primarily engaged in--'],
    ['Laziza Karahi Fry Meat Masala 90Gram Boxes Pack of 6', 'Turkey Production This industry comprises establishments primarily engaged in raising turkeys for meat or egg production.'],
]
scores = model.predict(pairs)
print(scores)
# [-0.8008 -9.     -5.8125 -5.0938 -8.625 ]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Laziza Karahi Fry Meat Masala 90Gram Boxes Pack of 6',
    [
        'Meat Processed from Carcasses  This U.S. industry comprises establishments primarily engaged in processing or preserving meat and meat byproducts (except poultry and small game) from purchased meats.  This industry includes establishments primarily engaged in assembly cutting and packing of meats (i.e., boxed meats) from purchased meats.\n\nCross-References. Establishments primarily engaged in--',
        'Rendering and Meat Byproduct Processing  This U.S. industry comprises establishments primarily engaged in rendering animal fat, bones, and meat scraps.\n\nCross-References.',
        'Broilers and Other Meat Type Chicken Production  This industry comprises establishments primarily engaged in raising broilers, fryers, roasters, and other meat type chickens.\n\nCross-References.',
        'Poultry Processing  This U.S. industry comprises establishments primarily engaged in (1) slaughtering poultry and small game and/or (2) preparing processed poultry and small game meat and meat byproducts.\n\nCross-References. Establishments primarily engaged in--',
        'Turkey Production This industry comprises establishments primarily engaged in raising turkeys for meat or egg production.',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 1,284,600 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                           | sentence_1                                                                          | label                                                          |
  |:--------|:-------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                               | string                                                                              | float                                                          |
  | details | <ul><li>min: 15 tokens</li><li>mean: 113.05 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 19 tokens</li><li>mean: 104.3 tokens</li><li>max: 237 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.02</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                        | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                      | label            |
  |:------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Laziza Karahi Fry Meat Masala 90Gram Boxes Pack of 6</code> | <code>Meat Processed from Carcasses  This U.S. industry comprises establishments primarily engaged in processing or preserving meat and meat byproducts (except poultry and small game) from purchased meats.  This industry includes establishments primarily engaged in assembly cutting and packing of meats (i.e., boxed meats) from purchased meats.<br><br>Cross-References. Establishments primarily engaged in--</code> | <code>1.0</code> |
  | <code>Laziza Karahi Fry Meat Masala 90Gram Boxes Pack of 6</code> | <code>Rendering and Meat Byproduct Processing  This U.S. industry comprises establishments primarily engaged in rendering animal fat, bones, and meat scraps.<br><br>Cross-References.</code>                                                                                                                                                                                                                                   | <code>0.0</code> |
  | <code>Laziza Karahi Fry Meat Masala 90Gram Boxes Pack of 6</code> | <code>Broilers and Other Meat Type Chicken Production  This industry comprises establishments primarily engaged in raising broilers, fryers, roasters, and other meat type chickens.<br><br>Cross-References.</code>                                                                                                                                                                                                            | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 128
- `learning_rate`: 2e-05
- `warmup_steps`: 3010
- `bf16`: True
- `tf32`: True
- `auto_find_batch_size`: True
- `per_device_eval_batch_size`: 256
- `dataloader_num_workers`: 8
- `dataloader_persistent_workers`: True
- `dataloader_prefetch_factor`: 4

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 128
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 2e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 3010
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1.0
- `label_smoothing_factor`: 0.0
- `bf16`: True
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: True
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: True
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: trackio
- `eval_strategy`: no
- `per_device_eval_batch_size`: 256
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 8
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: True
- `dataloader_prefetch_factor`: 4
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step  | Training Loss |
|:------:|:-----:|:-------------:|
| 0.0050 | 50    | 0.1486        |
| 0.0100 | 100   | 0.1580        |
| 0.0149 | 150   | 0.1330        |
| 0.0199 | 200   | 0.1259        |
| 0.0249 | 250   | 0.1255        |
| 0.0299 | 300   | 0.1309        |
| 0.0349 | 350   | 0.0972        |
| 0.0399 | 400   | 0.0877        |
| 0.0448 | 450   | 0.0972        |
| 0.0498 | 500   | 0.0811        |
| 0.0548 | 550   | 0.0918        |
| 0.0598 | 600   | 0.0898        |
| 0.0648 | 650   | 0.0934        |
| 0.0697 | 700   | 0.0852        |
| 0.0747 | 750   | 0.0768        |
| 0.0797 | 800   | 0.0820        |
| 0.0847 | 850   | 0.0829        |
| 0.0897 | 900   | 0.0834        |
| 0.0947 | 950   | 0.0892        |
| 0.0996 | 1000  | 0.0786        |
| 0.1046 | 1050  | 0.0692        |
| 0.1096 | 1100  | 0.0771        |
| 0.1146 | 1150  | 0.0811        |
| 0.1196 | 1200  | 0.0783        |
| 0.1246 | 1250  | 0.0721        |
| 0.1295 | 1300  | 0.0767        |
| 0.1345 | 1350  | 0.0860        |
| 0.1395 | 1400  | 0.0739        |
| 0.1445 | 1450  | 0.0882        |
| 0.1495 | 1500  | 0.0802        |
| 0.1544 | 1550  | 0.0667        |
| 0.1594 | 1600  | 0.0784        |
| 0.1644 | 1650  | 0.0720        |
| 0.1694 | 1700  | 0.0688        |
| 0.1744 | 1750  | 0.0756        |
| 0.1794 | 1800  | 0.0793        |
| 0.1843 | 1850  | 0.0712        |
| 0.1893 | 1900  | 0.0790        |
| 0.1943 | 1950  | 0.0806        |
| 0.1993 | 2000  | 0.0738        |
| 0.2043 | 2050  | 0.0817        |
| 0.2092 | 2100  | 0.0616        |
| 0.2142 | 2150  | 0.0854        |
| 0.2192 | 2200  | 0.0709        |
| 0.2242 | 2250  | 0.0759        |
| 0.2292 | 2300  | 0.0687        |
| 0.2342 | 2350  | 0.0832        |
| 0.2391 | 2400  | 0.0753        |
| 0.2441 | 2450  | 0.0693        |
| 0.2491 | 2500  | 0.0730        |
| 0.2541 | 2550  | 0.0729        |
| 0.2591 | 2600  | 0.0697        |
| 0.2640 | 2650  | 0.0723        |
| 0.2690 | 2700  | 0.0807        |
| 0.2740 | 2750  | 0.0740        |
| 0.2790 | 2800  | 0.0762        |
| 0.2840 | 2850  | 0.0790        |
| 0.2890 | 2900  | 0.0717        |
| 0.2939 | 2950  | 0.0679        |
| 0.2989 | 3000  | 0.0758        |
| 0.3039 | 3050  | 0.0740        |
| 0.3089 | 3100  | 0.0761        |
| 0.3139 | 3150  | 0.0638        |
| 0.3189 | 3200  | 0.0656        |
| 0.3238 | 3250  | 0.0713        |
| 0.3288 | 3300  | 0.0699        |
| 0.3338 | 3350  | 0.0713        |
| 0.3388 | 3400  | 0.0737        |
| 0.3438 | 3450  | 0.0714        |
| 0.3487 | 3500  | 0.0792        |
| 0.3537 | 3550  | 0.0852        |
| 0.3587 | 3600  | 0.0676        |
| 0.3637 | 3650  | 0.0736        |
| 0.3687 | 3700  | 0.0761        |
| 0.3737 | 3750  | 0.0724        |
| 0.3786 | 3800  | 0.0730        |
| 0.3836 | 3850  | 0.0691        |
| 0.3886 | 3900  | 0.0660        |
| 0.3936 | 3950  | 0.0753        |
| 0.3986 | 4000  | 0.0731        |
| 0.4035 | 4050  | 0.0660        |
| 0.4085 | 4100  | 0.0819        |
| 0.4135 | 4150  | 0.0704        |
| 0.4185 | 4200  | 0.0716        |
| 0.4235 | 4250  | 0.0792        |
| 0.4285 | 4300  | 0.0715        |
| 0.4334 | 4350  | 0.0736        |
| 0.4384 | 4400  | 0.0656        |
| 0.4434 | 4450  | 0.0753        |
| 0.4484 | 4500  | 0.0719        |
| 0.4534 | 4550  | 0.0754        |
| 0.4583 | 4600  | 0.0658        |
| 0.4633 | 4650  | 0.0741        |
| 0.4683 | 4700  | 0.0665        |
| 0.4733 | 4750  | 0.0768        |
| 0.4783 | 4800  | 0.0609        |
| 0.4833 | 4850  | 0.0619        |
| 0.4882 | 4900  | 0.0758        |
| 0.4932 | 4950  | 0.0719        |
| 0.4982 | 5000  | 0.0685        |
| 0.5032 | 5050  | 0.0694        |
| 0.5082 | 5100  | 0.0657        |
| 0.5132 | 5150  | 0.0617        |
| 0.5181 | 5200  | 0.0713        |
| 0.5231 | 5250  | 0.0713        |
| 0.5281 | 5300  | 0.0634        |
| 0.5331 | 5350  | 0.0691        |
| 0.5381 | 5400  | 0.0671        |
| 0.5430 | 5450  | 0.0726        |
| 0.5480 | 5500  | 0.0797        |
| 0.5530 | 5550  | 0.0794        |
| 0.5580 | 5600  | 0.0715        |
| 0.5630 | 5650  | 0.0726        |
| 0.5680 | 5700  | 0.0740        |
| 0.5729 | 5750  | 0.0759        |
| 0.5779 | 5800  | 0.0756        |
| 0.5829 | 5850  | 0.0608        |
| 0.5879 | 5900  | 0.0642        |
| 0.5929 | 5950  | 0.0659        |
| 0.5978 | 6000  | 0.0694        |
| 0.6028 | 6050  | 0.0629        |
| 0.6078 | 6100  | 0.0751        |
| 0.6128 | 6150  | 0.0729        |
| 0.6178 | 6200  | 0.0656        |
| 0.6228 | 6250  | 0.0746        |
| 0.6277 | 6300  | 0.0761        |
| 0.6327 | 6350  | 0.0725        |
| 0.6377 | 6400  | 0.0697        |
| 0.6427 | 6450  | 0.0673        |
| 0.6477 | 6500  | 0.0742        |
| 0.6527 | 6550  | 0.0648        |
| 0.6576 | 6600  | 0.0733        |
| 0.6626 | 6650  | 0.0701        |
| 0.6676 | 6700  | 0.0725        |
| 0.6726 | 6750  | 0.0703        |
| 0.6776 | 6800  | 0.0637        |
| 0.6825 | 6850  | 0.0724        |
| 0.6875 | 6900  | 0.0739        |
| 0.6925 | 6950  | 0.0727        |
| 0.6975 | 7000  | 0.0676        |
| 0.7025 | 7050  | 0.0601        |
| 0.7075 | 7100  | 0.0657        |
| 0.7124 | 7150  | 0.0716        |
| 0.7174 | 7200  | 0.0686        |
| 0.7224 | 7250  | 0.0756        |
| 0.7274 | 7300  | 0.0724        |
| 0.7324 | 7350  | 0.0677        |
| 0.7373 | 7400  | 0.0766        |
| 0.7423 | 7450  | 0.0650        |
| 0.7473 | 7500  | 0.0731        |
| 0.7523 | 7550  | 0.0674        |
| 0.7573 | 7600  | 0.0679        |
| 0.7623 | 7650  | 0.0629        |
| 0.7672 | 7700  | 0.0679        |
| 0.7722 | 7750  | 0.0724        |
| 0.7772 | 7800  | 0.0599        |
| 0.7822 | 7850  | 0.0710        |
| 0.7872 | 7900  | 0.0746        |
| 0.7921 | 7950  | 0.0701        |
| 0.7971 | 8000  | 0.0747        |
| 0.8021 | 8050  | 0.0707        |
| 0.8071 | 8100  | 0.0648        |
| 0.8121 | 8150  | 0.0760        |
| 0.8171 | 8200  | 0.0775        |
| 0.8220 | 8250  | 0.0661        |
| 0.8270 | 8300  | 0.0687        |
| 0.8320 | 8350  | 0.0748        |
| 0.8370 | 8400  | 0.0665        |
| 0.8420 | 8450  | 0.0711        |
| 0.8470 | 8500  | 0.0666        |
| 0.8519 | 8550  | 0.0680        |
| 0.8569 | 8600  | 0.0619        |
| 0.8619 | 8650  | 0.0721        |
| 0.8669 | 8700  | 0.0640        |
| 0.8719 | 8750  | 0.0694        |
| 0.8768 | 8800  | 0.0604        |
| 0.8818 | 8850  | 0.0657        |
| 0.8868 | 8900  | 0.0778        |
| 0.8918 | 8950  | 0.0770        |
| 0.8968 | 9000  | 0.0630        |
| 0.9018 | 9050  | 0.0659        |
| 0.9067 | 9100  | 0.0737        |
| 0.9117 | 9150  | 0.0661        |
| 0.9167 | 9200  | 0.0730        |
| 0.9217 | 9250  | 0.0629        |
| 0.9267 | 9300  | 0.0656        |
| 0.9316 | 9350  | 0.0686        |
| 0.9366 | 9400  | 0.0607        |
| 0.9416 | 9450  | 0.0716        |
| 0.9466 | 9500  | 0.0737        |
| 0.9516 | 9550  | 0.0765        |
| 0.9566 | 9600  | 0.0710        |
| 0.9615 | 9650  | 0.0670        |
| 0.9665 | 9700  | 0.0717        |
| 0.9715 | 9750  | 0.0805        |
| 0.9765 | 9800  | 0.0706        |
| 0.9815 | 9850  | 0.0664        |
| 0.9864 | 9900  | 0.0670        |
| 0.9914 | 9950  | 0.0714        |
| 0.9964 | 10000 | 0.0759        |
| 1.0014 | 10050 | 0.0758        |
| 1.0064 | 10100 | 0.0560        |
| 1.0114 | 10150 | 0.0616        |
| 1.0163 | 10200 | 0.0570        |
| 1.0213 | 10250 | 0.0703        |
| 1.0263 | 10300 | 0.0656        |
| 1.0313 | 10350 | 0.0689        |
| 1.0363 | 10400 | 0.0675        |
| 1.0413 | 10450 | 0.0679        |
| 1.0462 | 10500 | 0.0642        |
| 1.0512 | 10550 | 0.0688        |
| 1.0562 | 10600 | 0.0608        |
| 1.0612 | 10650 | 0.0677        |
| 1.0662 | 10700 | 0.0666        |
| 1.0711 | 10750 | 0.0646        |
| 1.0761 | 10800 | 0.0671        |
| 1.0811 | 10850 | 0.0639        |
| 1.0861 | 10900 | 0.0738        |
| 1.0911 | 10950 | 0.0624        |
| 1.0961 | 11000 | 0.0607        |
| 1.1010 | 11050 | 0.0717        |
| 1.1060 | 11100 | 0.0681        |
| 1.1110 | 11150 | 0.0637        |
| 1.1160 | 11200 | 0.0648        |
| 1.1210 | 11250 | 0.0628        |
| 1.1259 | 11300 | 0.0673        |
| 1.1309 | 11350 | 0.0649        |
| 1.1359 | 11400 | 0.0705        |
| 1.1409 | 11450 | 0.0655        |
| 1.1459 | 11500 | 0.0614        |
| 1.1509 | 11550 | 0.0646        |
| 1.1558 | 11600 | 0.0631        |
| 1.1608 | 11650 | 0.0705        |
| 1.1658 | 11700 | 0.0653        |
| 1.1708 | 11750 | 0.0664        |
| 1.1758 | 11800 | 0.0671        |
| 1.1807 | 11850 | 0.0648        |
| 1.1857 | 11900 | 0.0721        |
| 1.1907 | 11950 | 0.0647        |
| 1.1957 | 12000 | 0.0666        |
| 1.2007 | 12050 | 0.0676        |
| 1.2057 | 12100 | 0.0665        |
| 1.2106 | 12150 | 0.0586        |
| 1.2156 | 12200 | 0.0703        |
| 1.2206 | 12250 | 0.0696        |
| 1.2256 | 12300 | 0.0608        |
| 1.2306 | 12350 | 0.0778        |
| 1.2356 | 12400 | 0.0635        |
| 1.2405 | 12450 | 0.0679        |
| 1.2455 | 12500 | 0.0731        |
| 1.2505 | 12550 | 0.0627        |
| 1.2555 | 12600 | 0.0602        |
| 1.2605 | 12650 | 0.0707        |
| 1.2654 | 12700 | 0.0599        |
| 1.2704 | 12750 | 0.0635        |
| 1.2754 | 12800 | 0.0610        |
| 1.2804 | 12850 | 0.0648        |
| 1.2854 | 12900 | 0.0701        |
| 1.2904 | 12950 | 0.0669        |
| 1.2953 | 13000 | 0.0602        |
| 1.3003 | 13050 | 0.0593        |
| 1.3053 | 13100 | 0.0639        |
| 1.3103 | 13150 | 0.0651        |
| 1.3153 | 13200 | 0.0793        |
| 1.3202 | 13250 | 0.0673        |
| 1.3252 | 13300 | 0.0651        |
| 1.3302 | 13350 | 0.0691        |
| 1.3352 | 13400 | 0.0674        |
| 1.3402 | 13450 | 0.0674        |
| 1.3452 | 13500 | 0.0650        |
| 1.3501 | 13550 | 0.0673        |
| 1.3551 | 13600 | 0.0636        |
| 1.3601 | 13650 | 0.0671        |
| 1.3651 | 13700 | 0.0685        |
| 1.3701 | 13750 | 0.0748        |
| 1.3750 | 13800 | 0.0671        |
| 1.3800 | 13850 | 0.0663        |
| 1.3850 | 13900 | 0.0695        |
| 1.3900 | 13950 | 0.0674        |
| 1.3950 | 14000 | 0.0606        |
| 1.4000 | 14050 | 0.0527        |
| 1.4049 | 14100 | 0.0659        |
| 1.4099 | 14150 | 0.0653        |
| 1.4149 | 14200 | 0.0655        |
| 1.4199 | 14250 | 0.0598        |
| 1.4249 | 14300 | 0.0619        |
| 1.4299 | 14350 | 0.0690        |
| 1.4348 | 14400 | 0.0668        |
| 1.4398 | 14450 | 0.0708        |
| 1.4448 | 14500 | 0.0656        |
| 1.4498 | 14550 | 0.0674        |
| 1.4548 | 14600 | 0.0554        |
| 1.4597 | 14650 | 0.0766        |
| 1.4647 | 14700 | 0.0599        |
| 1.4697 | 14750 | 0.0641        |
| 1.4747 | 14800 | 0.0667        |
| 1.4797 | 14850 | 0.0670        |
| 1.4847 | 14900 | 0.0641        |
| 1.4896 | 14950 | 0.0674        |
| 1.4946 | 15000 | 0.0630        |
| 1.4996 | 15050 | 0.0624        |
| 1.5046 | 15100 | 0.0682        |
| 1.5096 | 15150 | 0.0631        |
| 1.5145 | 15200 | 0.0657        |
| 1.5195 | 15250 | 0.0570        |
| 1.5245 | 15300 | 0.0745        |
| 1.5295 | 15350 | 0.0618        |
| 1.5345 | 15400 | 0.0653        |
| 1.5395 | 15450 | 0.0668        |
| 1.5444 | 15500 | 0.0637        |
| 1.5494 | 15550 | 0.0601        |
| 1.5544 | 15600 | 0.0555        |
| 1.5594 | 15650 | 0.0621        |
| 1.5644 | 15700 | 0.0750        |
| 1.5694 | 15750 | 0.0629        |
| 1.5743 | 15800 | 0.0681        |
| 1.5793 | 15850 | 0.0645        |
| 1.5843 | 15900 | 0.0671        |
| 1.5893 | 15950 | 0.0577        |
| 1.5943 | 16000 | 0.0666        |
| 1.5992 | 16050 | 0.0720        |
| 1.6042 | 16100 | 0.0625        |
| 1.6092 | 16150 | 0.0721        |
| 1.6142 | 16200 | 0.0693        |
| 1.6192 | 16250 | 0.0661        |
| 1.6242 | 16300 | 0.0725        |
| 1.6291 | 16350 | 0.0666        |
| 1.6341 | 16400 | 0.0667        |
| 1.6391 | 16450 | 0.0651        |
| 1.6441 | 16500 | 0.0694        |
| 1.6491 | 16550 | 0.0585        |
| 1.6540 | 16600 | 0.0706        |
| 1.6590 | 16650 | 0.0626        |
| 1.6640 | 16700 | 0.0680        |
| 1.6690 | 16750 | 0.0721        |
| 1.6740 | 16800 | 0.0691        |
| 1.6790 | 16850 | 0.0632        |
| 1.6839 | 16900 | 0.0662        |
| 1.6889 | 16950 | 0.0647        |
| 1.6939 | 17000 | 0.0642        |
| 1.6989 | 17050 | 0.0604        |
| 1.7039 | 17100 | 0.0551        |
| 1.7088 | 17150 | 0.0630        |
| 1.7138 | 17200 | 0.0750        |
| 1.7188 | 17250 | 0.0582        |
| 1.7238 | 17300 | 0.0601        |
| 1.7288 | 17350 | 0.0598        |
| 1.7338 | 17400 | 0.0682        |
| 1.7387 | 17450 | 0.0615        |
| 1.7437 | 17500 | 0.0631        |
| 1.7487 | 17550 | 0.0645        |
| 1.7537 | 17600 | 0.0658        |
| 1.7587 | 17650 | 0.0603        |
| 1.7637 | 17700 | 0.0675        |
| 1.7686 | 17750 | 0.0586        |
| 1.7736 | 17800 | 0.0605        |
| 1.7786 | 17850 | 0.0652        |
| 1.7836 | 17900 | 0.0597        |
| 1.7886 | 17950 | 0.0603        |
| 1.7935 | 18000 | 0.0627        |
| 1.7985 | 18050 | 0.0597        |
| 1.8035 | 18100 | 0.0694        |
| 1.8085 | 18150 | 0.0585        |
| 1.8135 | 18200 | 0.0716        |
| 1.8185 | 18250 | 0.0704        |
| 1.8234 | 18300 | 0.0664        |
| 1.8284 | 18350 | 0.0713        |
| 1.8334 | 18400 | 0.0678        |
| 1.8384 | 18450 | 0.0633        |
| 1.8434 | 18500 | 0.0635        |
| 1.8483 | 18550 | 0.0658        |
| 1.8533 | 18600 | 0.0692        |
| 1.8583 | 18650 | 0.0603        |
| 1.8633 | 18700 | 0.0659        |
| 1.8683 | 18750 | 0.0632        |
| 1.8733 | 18800 | 0.0644        |
| 1.8782 | 18850 | 0.0640        |
| 1.8832 | 18900 | 0.0620        |
| 1.8882 | 18950 | 0.0554        |
| 1.8932 | 19000 | 0.0668        |
| 1.8982 | 19050 | 0.0613        |
| 1.9031 | 19100 | 0.0641        |
| 1.9081 | 19150 | 0.0655        |
| 1.9131 | 19200 | 0.0721        |
| 1.9181 | 19250 | 0.0673        |
| 1.9231 | 19300 | 0.0651        |
| 1.9281 | 19350 | 0.0641        |
| 1.9330 | 19400 | 0.0656        |
| 1.9380 | 19450 | 0.0607        |
| 1.9430 | 19500 | 0.0610        |
| 1.9480 | 19550 | 0.0694        |
| 1.9530 | 19600 | 0.0662        |
| 1.9580 | 19650 | 0.0576        |
| 1.9629 | 19700 | 0.0657        |
| 1.9679 | 19750 | 0.0542        |
| 1.9729 | 19800 | 0.0589        |
| 1.9779 | 19850 | 0.0598        |
| 1.9829 | 19900 | 0.0612        |
| 1.9878 | 19950 | 0.0604        |
| 1.9928 | 20000 | 0.0546        |
| 1.9978 | 20050 | 0.0630        |
| 2.0028 | 20100 | 0.0625        |
| 2.0078 | 20150 | 0.0654        |
| 2.0128 | 20200 | 0.0658        |
| 2.0177 | 20250 | 0.0582        |
| 2.0227 | 20300 | 0.0626        |
| 2.0277 | 20350 | 0.0577        |
| 2.0327 | 20400 | 0.0624        |
| 2.0377 | 20450 | 0.0604        |
| 2.0426 | 20500 | 0.0679        |
| 2.0476 | 20550 | 0.0612        |
| 2.0526 | 20600 | 0.0666        |
| 2.0576 | 20650 | 0.0470        |
| 2.0626 | 20700 | 0.0626        |
| 2.0676 | 20750 | 0.0706        |
| 2.0725 | 20800 | 0.0618        |
| 2.0775 | 20850 | 0.0556        |
| 2.0825 | 20900 | 0.0548        |
| 2.0875 | 20950 | 0.0713        |
| 2.0925 | 21000 | 0.0585        |
| 2.0974 | 21050 | 0.0679        |
| 2.1024 | 21100 | 0.0588        |
| 2.1074 | 21150 | 0.0604        |
| 2.1124 | 21200 | 0.0625        |
| 2.1174 | 21250 | 0.0556        |
| 2.1224 | 21300 | 0.0522        |
| 2.1273 | 21350 | 0.0589        |
| 2.1323 | 21400 | 0.0561        |
| 2.1373 | 21450 | 0.0580        |
| 2.1423 | 21500 | 0.0558        |
| 2.1473 | 21550 | 0.0662        |
| 2.1523 | 21600 | 0.0562        |
| 2.1572 | 21650 | 0.0619        |
| 2.1622 | 21700 | 0.0629        |
| 2.1672 | 21750 | 0.0638        |
| 2.1722 | 21800 | 0.0586        |
| 2.1772 | 21850 | 0.0665        |
| 2.1821 | 21900 | 0.0717        |
| 2.1871 | 21950 | 0.0631        |
| 2.1921 | 22000 | 0.0595        |
| 2.1971 | 22050 | 0.0731        |
| 2.2021 | 22100 | 0.0560        |
| 2.2071 | 22150 | 0.0564        |
| 2.2120 | 22200 | 0.0711        |
| 2.2170 | 22250 | 0.0647        |
| 2.2220 | 22300 | 0.0552        |
| 2.2270 | 22350 | 0.0622        |
| 2.2320 | 22400 | 0.0668        |
| 2.2369 | 22450 | 0.0672        |
| 2.2419 | 22500 | 0.0603        |
| 2.2469 | 22550 | 0.0615        |
| 2.2519 | 22600 | 0.0577        |
| 2.2569 | 22650 | 0.0701        |
| 2.2619 | 22700 | 0.0635        |
| 2.2668 | 22750 | 0.0562        |
| 2.2718 | 22800 | 0.0577        |
| 2.2768 | 22850 | 0.0679        |
| 2.2818 | 22900 | 0.0622        |
| 2.2868 | 22950 | 0.0638        |
| 2.2917 | 23000 | 0.0591        |
| 2.2967 | 23050 | 0.0650        |
| 2.3017 | 23100 | 0.0561        |
| 2.3067 | 23150 | 0.0651        |
| 2.3117 | 23200 | 0.0662        |
| 2.3167 | 23250 | 0.0633        |
| 2.3216 | 23300 | 0.0628        |
| 2.3266 | 23350 | 0.0618        |
| 2.3316 | 23400 | 0.0622        |
| 2.3366 | 23450 | 0.0649        |
| 2.3416 | 23500 | 0.0553        |
| 2.3466 | 23550 | 0.0628        |
| 2.3515 | 23600 | 0.0545        |
| 2.3565 | 23650 | 0.0585        |
| 2.3615 | 23700 | 0.0672        |
| 2.3665 | 23750 | 0.0617        |
| 2.3715 | 23800 | 0.0591        |
| 2.3764 | 23850 | 0.0544        |
| 2.3814 | 23900 | 0.0681        |
| 2.3864 | 23950 | 0.0559        |
| 2.3914 | 24000 | 0.0613        |
| 2.3964 | 24050 | 0.0517        |
| 2.4014 | 24100 | 0.0556        |
| 2.4063 | 24150 | 0.0581        |
| 2.4113 | 24200 | 0.0542        |
| 2.4163 | 24250 | 0.0529        |
| 2.4213 | 24300 | 0.0664        |
| 2.4263 | 24350 | 0.0570        |
| 2.4312 | 24400 | 0.0541        |
| 2.4362 | 24450 | 0.0635        |
| 2.4412 | 24500 | 0.0632        |
| 2.4462 | 24550 | 0.0634        |
| 2.4512 | 24600 | 0.0639        |
| 2.4562 | 24650 | 0.0574        |
| 2.4611 | 24700 | 0.0640        |
| 2.4661 | 24750 | 0.0622        |
| 2.4711 | 24800 | 0.0626        |
| 2.4761 | 24850 | 0.0527        |
| 2.4811 | 24900 | 0.0608        |
| 2.4861 | 24950 | 0.0599        |
| 2.4910 | 25000 | 0.0519        |
| 2.4960 | 25050 | 0.0626        |
| 2.5010 | 25100 | 0.0562        |
| 2.5060 | 25150 | 0.0568        |
| 2.5110 | 25200 | 0.0623        |
| 2.5159 | 25250 | 0.0552        |
| 2.5209 | 25300 | 0.0635        |
| 2.5259 | 25350 | 0.0581        |
| 2.5309 | 25400 | 0.0650        |
| 2.5359 | 25450 | 0.0582        |
| 2.5409 | 25500 | 0.0613        |
| 2.5458 | 25550 | 0.0612        |
| 2.5508 | 25600 | 0.0572        |
| 2.5558 | 25650 | 0.0688        |
| 2.5608 | 25700 | 0.0493        |
| 2.5658 | 25750 | 0.0545        |
| 2.5707 | 25800 | 0.0580        |
| 2.5757 | 25850 | 0.0561        |
| 2.5807 | 25900 | 0.0578        |
| 2.5857 | 25950 | 0.0529        |
| 2.5907 | 26000 | 0.0603        |
| 2.5957 | 26050 | 0.0579        |
| 2.6006 | 26100 | 0.0558        |
| 2.6056 | 26150 | 0.0597        |
| 2.6106 | 26200 | 0.0610        |
| 2.6156 | 26250 | 0.0602        |
| 2.6206 | 26300 | 0.0626        |
| 2.6255 | 26350 | 0.0562        |
| 2.6305 | 26400 | 0.0580        |
| 2.6355 | 26450 | 0.0559        |
| 2.6405 | 26500 | 0.0578        |
| 2.6455 | 26550 | 0.0593        |
| 2.6505 | 26600 | 0.0631        |
| 2.6554 | 26650 | 0.0610        |
| 2.6604 | 26700 | 0.0564        |
| 2.6654 | 26750 | 0.0622        |
| 2.6704 | 26800 | 0.0569        |
| 2.6754 | 26850 | 0.0641        |
| 2.6804 | 26900 | 0.0612        |
| 2.6853 | 26950 | 0.0620        |
| 2.6903 | 27000 | 0.0632        |
| 2.6953 | 27050 | 0.0623        |
| 2.7003 | 27100 | 0.0617        |
| 2.7053 | 27150 | 0.0572        |
| 2.7102 | 27200 | 0.0600        |
| 2.7152 | 27250 | 0.0601        |
| 2.7202 | 27300 | 0.0599        |
| 2.7252 | 27350 | 0.0652        |
| 2.7302 | 27400 | 0.0577        |
| 2.7352 | 27450 | 0.0657        |
| 2.7401 | 27500 | 0.0571        |
| 2.7451 | 27550 | 0.0628        |
| 2.7501 | 27600 | 0.0575        |
| 2.7551 | 27650 | 0.0589        |
| 2.7601 | 27700 | 0.0598        |
| 2.7650 | 27750 | 0.0576        |
| 2.7700 | 27800 | 0.0623        |
| 2.7750 | 27850 | 0.0596        |
| 2.7800 | 27900 | 0.0668        |
| 2.7850 | 27950 | 0.0665        |
| 2.7900 | 28000 | 0.0658        |
| 2.7949 | 28050 | 0.0649        |
| 2.7999 | 28100 | 0.0663        |
| 2.8049 | 28150 | 0.0666        |
| 2.8099 | 28200 | 0.0589        |
| 2.8149 | 28250 | 0.0660        |
| 2.8198 | 28300 | 0.0695        |
| 2.8248 | 28350 | 0.0540        |
| 2.8298 | 28400 | 0.0667        |
| 2.8348 | 28450 | 0.0577        |
| 2.8398 | 28500 | 0.0616        |
| 2.8448 | 28550 | 0.0549        |
| 2.8497 | 28600 | 0.0648        |
| 2.8547 | 28650 | 0.0660        |
| 2.8597 | 28700 | 0.0543        |
| 2.8647 | 28750 | 0.0585        |
| 2.8697 | 28800 | 0.0708        |
| 2.8747 | 28850 | 0.0617        |
| 2.8796 | 28900 | 0.0641        |
| 2.8846 | 28950 | 0.0590        |
| 2.8896 | 29000 | 0.0579        |
| 2.8946 | 29050 | 0.0625        |
| 2.8996 | 29100 | 0.0568        |
| 2.9045 | 29150 | 0.0637        |
| 2.9095 | 29200 | 0.0639        |
| 2.9145 | 29250 | 0.0548        |
| 2.9195 | 29300 | 0.0579        |
| 2.9245 | 29350 | 0.0678        |
| 2.9295 | 29400 | 0.0603        |
| 2.9344 | 29450 | 0.0650        |
| 2.9394 | 29500 | 0.0730        |
| 2.9444 | 29550 | 0.0619        |
| 2.9494 | 29600 | 0.0582        |
| 2.9544 | 29650 | 0.0636        |
| 2.9593 | 29700 | 0.0615        |
| 2.9643 | 29750 | 0.0576        |
| 2.9693 | 29800 | 0.0624        |
| 2.9743 | 29850 | 0.0661        |
| 2.9793 | 29900 | 0.0619        |
| 2.9843 | 29950 | 0.0688        |
| 2.9892 | 30000 | 0.0556        |
| 2.9942 | 30050 | 0.0595        |
| 2.9992 | 30100 | 0.0596        |

</details>

### Training Time
- **Training**: 1.6 hours

### Framework Versions
- Python: 3.13.5
- Sentence Transformers: 5.4.0
- Transformers: 5.5.3
- PyTorch: 2.8.0+cu128
- Accelerate: 1.13.0
- Datasets: 4.8.4
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->