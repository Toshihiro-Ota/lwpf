# Learning with Partial Forgetting in Modern Hopfield Networks

This is the official PyTorch implementation of *learning with partial forgetting* (LwPF). The paper is ***accepted in AISTATS 2023*** and available [here](https://proceedings.mlr.press/v206/ota23a.html).

## Abstract

It has been known by neuroscience studies that partial and transient forgetting of memory often plays an important role in the brain to improve performance for certain intellectual activities. In machine learning, associative memory models such as classical and modern Hopfield networks have been proposed to express memories as attractors in the feature space of a closed recurrent network. In this work, we propose learning with partial forgetting (LwPF), where a partial forgetting functionality is designed by element-wise non-bijective projections, for memory neurons in modern Hopfield networks to improve model performance. We incorporate LwPF into the attention mechanism also, whose process has been shown to be identical to the update rule of a certain modern Hopfield network, by modifying the corresponding Lagrangian. We evaluated the effectiveness of LwPF on three diverse tasks such as bit-pattern classification, immune repertoire classification for computational biology, and image classification for computer vision, and confirmed that LwPF consistently improves the performance of existing neural networks including DeepRC and vision transformers.

## Usage

This instruction shows how to apply LwPF to attention mechanism.

### 1. Define partial forgetting module

The two partial forgetting modules, ReLU and PFU, are implemented. The following example imports them from `lagrangian_units.py`.

```python
import lagrangian_units as lu
func_d = lu.ReLU()
# func_d = lu.PFU(bias = 'mu', mu = 'dmedian', sigma = 0.01)
```

Note that `func_d` returns a pair of $(D(x), D'(x))$, where $D$ is the partial forgetting function introduced in the paper and $D'$ is its derivative (i.e., the indicator function).

### 2. Apply LwPF to attention

LwPF can be applied to any networks with attention modules. For example, let us say we have the following code to compute an attention,

```python
attn = q @ k
attn = attn.softmax(dim=-1)
outs = attn @ v
```

where `q`, `k` and `v` are the query matrix, the key matrix and the value matrix, respectively. Only two lines are needed to apply LwPF:

```python
  attn = q @ k
+ attn, mask = func_d(attn)
  attn = attn.softmax(dim=-1)
+ attn = mask * attn
  outs = attn @ v
```

where `+` indicates inserted lines. A pseudo code as an example is provided in `lagrangian_vision_transformer.py`.

## Examples

### (1) Bit pattern classification

The first example in the paper uses `hopfield-layers`:

> H. Ramsauer et al., Hopfield Networks is All You Need, ICLR, 2021.<br>  https://github.com/ml-jku/hopfield-layers

To run experiments with LwPF, modify `hflayers/functional.py` as follows:

```python
+ attn_output_weights, xi_mask = func_d(attn_output_weights)
  if xi is None:
      xi = nn.functional.softmax(attn_output_weights, dim=-1)
```

```python
+ attn_output_weights = xi_mask * attn_output_weights
  attn_output = torch.bmm(attn_output_weights, v)
```

See **Usage** for the definition of `func_d`.

### (2) Immune repertoire classification

The second example uses DeepRC:

> M. Widrich et al., Modern Hopfield Networks and Attention for Immune Repertoire Classification, NeurIPS, 2020.<br>  https://github.com/ml-jku/DeepRC

To run experiments with LwPF, modify `deeprc/architectures.py` as follows:

```python
  # Calculate attention activations ...
+ attention_weights, xi_mask = self.func_d(attention_weights)
  attention_weights = torch.softmax(attention_weights, dim=0)
```

```python
  # Apply attention weights to sequence features ...
+ attention_weights = xi_mask * attention_weights
  emb_seqs_after_attention = emb_seqs * attention_weights
```

### (3) Image classification

The third example uses timm:

> Ross Wightman, PyTorch Image Models.<br>  https://github.com/huggingface/pytorch-image-models

To run experiments with LwPF, modify `Attention` in `timm/models/vision_transformer.py` as follows:

```python
  attn = (q @ k.transpose(-2, -1)) * self.scale
+ attn, mask = self.func_d(attn)
  attn = attn.softmax(dim=-1)
  attn = self.attn_drop(attn)
+ attn = mask * attn
  x = (attn @ v).transpose(1, 2).reshape(B, N, C)
```

## Citation

If you use our code please cite:

```tex
@InProceedings{pmlr-v206-ota23a,
  title     = {Learning with Partial Forgetting in Modern Hopfield Networks},
  author    = {Ota, Toshihiro and Sato, Ikuro and Kawakami, Rei and Tanaka, Masayuki and Inoue, Nakamasa},
  booktitle = {Proceedings of The 26th International Conference on Artificial Intelligence and Statistics},
  pages     = {6661--6673},
  year      = {2023},
  volume    = {206},
  publisher = {PMLR}
}
```
