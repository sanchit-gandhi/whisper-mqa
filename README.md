# Whisper Multi-Query Attention

This is a repository for benchmarking the [Whisper Model](https://arxiv.org/abs/2212.04356) with Multi-Query 
Attention (MQA) from the paper [_Fast Transformer Decoding: One Write-Head is All You Need_](https://arxiv.org/abs/1911.02150) by 
Noam Shazeer.

The modelling code is split into two parts:
* [multiquery_attention.py](whisper_mqa/multiquery_attention.py): implements MQA from the Fast Transformer Decoding paper
* [modeling_whisper_mqa.py](whisper_mqa/modeling_whisper_mqa.py): augments the Hugging Face Transformers Whisper model with MQA

The benchmark can be run from the Python script in [benchmark_mqa.py](benchmark_mqa.py). By default, it uses:
* Batch size = 1: we benchmark the performance of the model for single-batch inference with greedy decoding
* Num batches = 100: we benchmark over multiple batches for a good estimate of runtime
* Generated tokens = 25: this is the typical output seq len for speech

## Results

Multi-Head Attention (MHA) vs Multi-Query Attention (MQA) for the default benchmark settings:
* Batch size = 1
* Num batches = 100 
* Generated tokens = 25

| Checkpoint | VRAM MHA / GB | VRAM MQA / GB | Time MHA / s | Time MQA / s |
|------------|---------------|---------------|--------------|--------------|
| tiny       | 1.38          | 1.37          | 17.0         | 15.1         |
| base       | 1.52          | 1.49          | 19.4         | 20.4         |
| small      | 2.27          | 2.1           | 39.4         | 38.7         |
| medium     | 4.21          | 3.67          | 76.0         | 75.9         |
| large      | 7.52          | 6.29          | 107.2        | 91.2         |

We see very little difference between MHA and MQA, likely because we are memory access limited for low batch sizes 
(see [Analysis](#analysis)). We should see improvements using MQA with larger batch sizes. Re-running the benchmark 
with a 32x larger batch size:
* Batch size = 32
* Num batches = 100 
* Generated tokens = 25

| Checkpoint | VRAM MHA / GB | VRAM MQA / GB | Time MHA / s | Time MQA / s |
|------------|---------------|---------------|--------------|--------------|
| tiny       | 3.22          | 3.21          | 47.2         | 49.6         |
| base       | 3.98          | 3.94          | 51.4         | 55.6         |
| small      | 5.98          | 5.76          | 77.9         | 83.3         |
| medium     | 9.19          | 8.55          | 181.0        | 173.4        |
| large      | 15.28         | 12.40         | 303.1        | 277.5        |

## Analysis

Assuming we have a query length of $n$, a batch size of $b$, and a model dimension of $d$, the total number of 
_algorithmic operations_ is $\mathcal{O} \left(bnd^{2}\right)$ (the complexity of the quadratic attention operation).

### Multi-Head Attention

Across $n$ calls, the total amount of memory access is $\mathcal{O} \left(bn^{2}d + nd^{2}\right)$ (see Section 2.4.1 of the [MQA paper](https://arxiv.org/abs/1911.02150) for a derivation). The ratio of algorithmic 
operations to memory access is then:

$$ \mathcal{O} \left(\frac{n}{d} + \frac{1}{b}\right) $$

When $n \approx d$ or $b \approx 1$, the ratio is close to 1, causing memory bandwidth to be the major
performance bottleneck. Otherwise, the algorithmic operations are the limiting factor. 

This means that for low batch sizes, we are memory limited ($b \approx 1$). To circumvent this, we can just use a larger 
batch size $b$, memory size permitting.

For very low sequence lengths $n << d$, the $n/d$ term is small and so the offensive comes from the $1/b$. This is the 
typical setting in speech, where the output sequence length is much less than the model dim.

### Multi-Query Attention

Across $n$ calls, the total amount of memory access is $\mathcal{O} \left(bnd + bn^{2}k + nd^{2}\right)$ (see Section 3.1 of the [MQA paper](https://arxiv.org/abs/1911.02150) for a derivation). The ratio of algorithmic 
operations to memory access is then:

$$ \mathcal{O} \left(\frac{1}{d} + \frac{n}{dh} + \frac{1}{b}\right)$$

We have reduced the $n / d$ (sequence length : model dim) term by a factor of $h$. However, since we have small sequence lengths,
this term was already small. Thus, the offensive still comes from $1/b$, and so we expect that our ratio is still dominated by the batch size $b$.
