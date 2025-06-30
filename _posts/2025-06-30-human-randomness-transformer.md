---
layout: post
title: visualizing human randomness with transformers
categories: [neural networks, transformers, randomness]
description: 
keywords: 
mermaid: false
sequence: false
flow: false
mathjax: true
mindmap: false
mindmap2: false
image: /images/blog/humanrand/banner.png
---
# Can we really be random? 
#### No. My robot can tell.

  <div style="text-align: center; font-size: 2rem; letter-spacing: 0.1em; color: #333;">
    27549235168
  </div>

Pop quiz: how *random* is the sequence of numbers above? If you're a human like me, chances are that it looks very random to you. No patterns, no repeating: random, *right?*

As you can probably gather from the nature of this post, though, it isn't. In fact, it falls victim to many of the common biases that humans have when creating "random" sequences. For instance, every number is used just once! 

I recently had the idea to make some models to detect these "patterns in masking patterns", as well as human sequences in general, and this post will catalog my journey of doing so, alongside exploring the research of the human biases that create them.

![banner](/images/blog/humanrand/banner.png)

# The Problem

Think back to the sequence at the beginning of this post. Could you detect any "non-randomness" in it? Chances are, probably not.

This is the problem: Humans are bad at creating "random" sequences, but are even worse at detecting them.

## Human Biases

If you were to create a "random" set of numbers, you'd likely fall victim to a couple of these biases:
1. Evenly distributing the digits
2. Not repeating numbers
3. Cycling patterns (e.g., **143**531**143**)


There is a trove of research on the human RNG task, and we know of a lot of biases in the generation of this data. So with the right method, we should be able to detect it.

## The Traditional Solution
There is no shortage of methods of measuring randomness, and many methods have been proposed for the human RNG task specifically, so I will list some here, and a short description of each.

1. Derivatives of [Shannon entropy](https://www.wikipedia.com/wiki/Shannon_entropy)
   1. Block entropy: how uniform is the frequency of $$n$$ sized blocks of digits (n-grams)
   2. Redundancy index: how uniform is the frequency of digits? (i.e, block entropy with $$n=1$$)
2. Coupon score: how long does it take for all possible digits to occur?
3. Algorithmic complexity[^1]: the shortest computer program that produces the string[^2]
1. Compression algorithms: As simple as it sounds! Pass it through gzip, lz, etc and compute $$\frac{\text{compressed}}{\text{uncompressed}}$$. For true random files, this should be $$100\%$$.

# My solution: Why machine learning?

All of the traditional solutions have solid foundations and are very tractable and explainable. But, there is a wide variety of them and there is no standard, making it harder to compare results.

The first thing I thought of was to use a neural network to naturally sniff out the biases and provide a single source, although I do not propose this as a real measure for any research in the field.

Before we get into my final architecture, I thought it would be good to explain why machine learning even works here, and what it has to do with randomness.

# Learning = Compression

Think of a line in a 2D plane, say, $$y=x$$.
![line](/images/blog/humanrand/line.png)
<p style="text-align: center; margin-top: -1rem;">Some points on the line y = x</p>

Every point along this line is of the form $$(x,x)$$. Seems redundant, right? Well, that's because it is!
## Subspaces

The set of points that make up the line $$y=x$$ is only a subset of the points that make up the entire second dimension, $$\mathbb{R}^2$$. In fact, it only uses one number!

We could compress the entire line down to a single dimension ($$\mathbb{R}$$), the number line, and a simple rule:

> For any number $$x$$, its corresponding point is $$(x,x)$$.

This means that the line is a **subspace** of the entire second dimension. It doesn't "use up" all of the space it can.

But, what does this have to do with neural networks?

Any input to a machine-learning task is going to be a **subspace**:
- The space of all cat pictures is a subspace of all images
- The space of all bird calls is a subspace of all sounds.
- The space of all cancerous CT scans is a subspace of all CT scans.
- *The space of human sequences is a subspace of all sequences.*

Neural networks come up with a "custom compression algorithm" that learns what is important to the task it's being trained for, and compresses accordingly. This is beneficial if we know there are biases we can exploit in the data, like we can in the human RNG task.

## What's incompressible?

You might not know that randomness and neural networks are intertwined, but they are, precisely for the reason above.

Take for example, a bunch of random points in the 2D plane.
![rand](/images/blog/humanrand/rand.png)

This is a case where, in fact, the space of infinite random points spans the entire second dimension. There's no underlying structure we can exploit to represent these points more efficiently, other than memorizing. Thus,

**True randomness is incompressible.**

# The Data
There was not a lot of freely available data for this task, although I am sure there is a lot of it in the academic world I could not find. I sourced my data from the following sources:

- Humans participating in the RNG task (Angelike, Musch)[^3]
- Brian Ellis' HumanRandom project [^4]
- "Keyboard-smash" sequences from me and friends.

The non-human sequences came from the following pseudo-RNG sources:
- SFC64
- Philox
- PCG64
- MT19937

With a sequence size of 20, I had about 12,000 training pairs of (sequence, label). In partitioning, we do lose the opportunity to examine if there is any bias of starting/ending on a particular digit, but I don't believe this is a common bias.

I have to acknowledge that this is not a lot of data for a transformer, and as such I had to combat overfitting more, but the real solution of course is more data. My data pool is quite limited and I would love to see more research on this front.

# The Architecture
Since we're dealing with sequences, my first thought was to build a RNN. Although light, with a bi-directional gated recurrent network I had a hard time pushing past 20% error, and I realized I didn't have a great opportunity to visualize and interpret the data with that method.

So, my alternative thought was to go with an *attention* base, where every digit pays attention to each other one in the sequence, hoping that we would see for example in a counting sequence:

> 1 pays attention to 2 who pays attention to 3, etc.

The canonical form of this is a transformer, laid out in the famous paper "Attention is all you need"[^5], and so this is what I built.

## Embedding
![1748978354446](/images/blog/humanrand/embed.png)
<details markdown="1">

<summary>Code</summary>
JAX (flax.nnx)
```py
class Embed(nnx.Module):
    def __init__(self, rngs, vocab_size=11, embed_dim=64, max_seq_len=128):
        self.cls = vocab_size - 1
        self.embed1 = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.embed2 = nnx.Embed(max_seq_len + 1, embed_dim, rngs=rngs)

    def __call__(self, x):
        batch_size, seq_len = x.shape
        cls_tokens = jnp.full((batch_size, 1), self.cls)
        seq = jnp.concatenate([cls_tokens, x], axis=1)
        pos = jnp.broadcast_to(
            jnp.arange(seq_len + 1)[None, :], (batch_size, seq_len + 1)
        )
        embed_seq = self.embed1(seq)
        embed_pos = self.embed2(pos)
        return embed_seq + embed_pos
```
Mathematica
```mathematica
embedding = NetGraph[
  <|
   "pos" -> SequenceIndicesLayer[],
   "embed_seq" -> EmbeddingLayer[16, 11],
   "embed_pos" -> EmbeddingLayer[16, 26],
   "add" -> ThreadingLayer[Plus],
   "cls" -> NetArrayLayer["Output" -> {}, "Array" -> {10}],
   "prepend" -> PrependLayer[]
   |>,
  {
   NetPort["Input"] -> "prepend" -> "pos" -> "embed_pos" -> "add",
   "prepend" -> "embed_seq" -> "add",
   "cls" -> NetPort["prepend", "Element"]
   }
  ];
```

</details>

## Transformer
### Attention
Initially, I tried single head attention at first. I didn't get great performance, and stacking more didn't help. After I switched to multi-head attention, my performance saw a massive boost. This is likely because splitting into multiple heads allows each blcok to work with smaller, richer subspaces of the projections which are specific to each head.


Early on, I also had an issue with saturated weights after my first transformer block. When I inspected the attention weights, they would all be close to either 1 or 0, even with score rescaling.

Eventually I discovered that switching from post-norm to pre-norm fixed the issue and the attention weights were well-distributed throughout the network afterwards.
![1748978429241](/images/blog/humanrand/attn.png)

<details markdown="1">

<summary>Code</summary>

JAX (flax.nnx)
```py
class Attention(nnx.Module):
    def __init__(self, rngs, num_heads=4, embed_dim=64, dropout_rate=0.5):
        self.mha = nnx.MultiHeadAttention(
            num_heads,
            embed_dim,
            embed_dim,
            embed_dim,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )
        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)

    def __call__(self, x, decode=False, deterministic=False):
        normed = self.norm(x)
        attn = self.mha(
            normed, normed, normed, decode=decode, deterministic=deterministic
        )
        return attn + x
```
Mathematica

```mathematica
attn = NetGraph[
   <|
    "w_q" -> NetMapOperator[LinearLayer[16]],
    "scaled_q" -> ElementwiseLayer[0.25 #1 &],
    "w_v" -> NetMapOperator[LinearLayer[16]],
    "w_k" -> NetMapOperator[LinearLayer[16]],
    "attend" -> AttentionLayer[],
    "out" -> NetMapOperator[LinearLayer[16]]
    |>,
   {
    NetPort["Query"] -> 
     "w_q" -> "scaled_q" -> NetPort["attend", "Query"],
    NetPort["Input"] -> "w_v" -> NetPort["attend", "Value"],
    NetPort["Input"] -> "w_k" -> NetPort["attend", "Key"],
    "attend" -> "out" -> NetPort["Output"]
    }
   ];

attnFull = NetGraph[
   <|
    "attention" -> attn,
    "norm" -> NormalizationLayer[1],
    "add" -> ThreadingLayer[Plus]
    |>,
   {
    NetPort["Input"] -> "add",
    NetPort["Input"] -> "norm",
    "norm" -> NetPort["attention", "Query"],
    "norm" -> NetPort["attention", "Input"],
    "attention" -> "add",
    "add" -> NetPort["Output"]
    }
   ];
```

</details>

### Feed-Forward
![1748978450041](/images/blog/humanrand/ffn.png)

<details markdown="1">

<summary>Code</summary>
JAX (flax.nnx)

```py
class FFN(nnx.Module):
    def __init__(self, rngs, embed_dim=64, dropout_rate=0.5):
        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.ffn_up = nnx.Linear(embed_dim, embed_dim * 4, rngs=rngs)
        self.ffn_down = nnx.Linear(embed_dim * 4, embed_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x):
        normed = self.norm(x)
        normed = self.ffn_up(normed)
        normed = nnx.gelu(normed)
        normed = self.dropout(normed)
        normed = self.ffn_down(normed)
        return x + normed
```
Mathematica

```mathematica
ffn = NetGraph[
   <|
    "norm" -> NormalizationLayer[1],
    "ffn_up" -> NetMapOperator[LinearLayer[16*4]],
    "ffn_down" -> NetMapOperator[LinearLayer[16]],
    "gelu" -> 
     ElementwiseLayer[0.5 #1 (1 + Tanh[0.798 (#1 + 0.0447 #1^3)]) &],
    "dropout" -> DropoutLayer[0.1692],
    "add" -> ThreadingLayer[Plus]
    |>,
   {
    NetPort["Input"] -> 
     "norm" -> "ffn_up" -> "gelu" -> "dropout" -> "ffn_down" -> "add",
    NetPort["Input"] -> "add",
    "add" -> NetPort["Output"]
    }
   ]
   ```
</details>

## Classification
![Classification](/images/blog/humanrand/clswbar.png)
<p style="text-align: center; margin-top: -1rem;">Attention weights as a graph, colored by their attention to [CLS].</p>

Originally, I just used mean pooling, but I wanted more interpretability, so I went for a [CLS] token. With this, I could see which tokens affected the final classification the most by checking the attention to the [CLS] token.

The logits were passed to a standard sigmoid binary cross entropy loss.
![1748978503648](/images/blog/humanrand/class.png)


<details markdown="1">

<summary>Code</summary>
JAX (flax.nnx)

```py
class Classify(nnx.Module):
    def __init__(self, rngs, embed_dim=64):
        self.fc = nnx.Linear(embed_dim, 1, rngs=rngs)

    def __call__(self, x):
        return self.fc(x[:, 0]).squeeze(-1)
```
Mathematica

```mathematica
classify = NetGraph[
   <|
    "extract_cls" -> PartLayer[{All, 1}],
    "fc" -> LinearLayer[1],
    "squeeze" -> FlattenLayer[]
    |>,
   {
    NetPort["Input"] -> "extract_cls" -> "fc" -> "squeeze" -> NetPort["Output"]
    }
   ];
  ```

</details>

# Training

To determine good hyperparameters for this model, I conducted a hyperparameter sweep using Bayesian search. All runs were on a cosine decay learning rate schedule with $$\alpha = 0$$ and exponent 1.
<details markdown="1">
<summary>Sweep parameters</summary>

```yaml
  batch_size:
    distribution: int_uniform
    max: 50
    min: 24
  dropout_rate:
    distribution: uniform
    max: 1
    min: 0.07504860929737157
  embed_dim:
    values:
    - 16
    - 24
  learning_rate:
    distribution: uniform
    max: 0.01
    min: 0.007
  num_epochs:
    value: 80
  num_heads:
    values:
    - 2
    - 4
  num_transformer_layers:
    value: 2
  patience:
    distribution: int_uniform
    max: 30
    min: 10
  sequence_length:
    distribution: int_uniform
    max: 28
    min: 10
```
</details>
Here are the charts from my best-performing run, which achieved a 90% validation accuracy:
![Validation accuracy](/images/blog/humanrand/valacc.png)
![Learning rate](/images/blog/humanrand/learnrate.png)
![Rounds since improvement](/images/blog/humanrand/patience.png)
# Interpretability
Unfortunately, contrary to what I hoped, there didn't appear to be a "counting head", or a "repetition head" that identified biases in a nice way. As I found out, "Attention is not explanation"[^6]

Or, is it? Many papers have used attention for explainability[^7], and even though peeking at individual attention weights might not yield human explanations, there are some methods that might work, like attention flow and attention rollout [^8]. I may do a post in the future exploring these, or update this post with an implementation of these.

What I did observe, though, was that there appeared to be "sinks" of attention, where many of the tokens paid attention to a single one, usually in the middle of a recognizable signal (e.g., in the middle of a counting sequence, or in the middle of a repeated pattern). This may indicate the kind of behavior I was looking for, but it's hard to tell.

Ablating heads did not yield great results in terms of interpretability, aside from degrading the overall predictive accuracy of the model. More data would make this work much easier, as the low data makes it hard to know if it might just be memorizing common patterns in the data.

# Applications

## Neuropsychology
The human RNG task has been used to investigate several disorders, like schizophrenia, autism, depression, and more [^1]. What processes underly the creation of numbers seem to indicate things about brain health and cognitive ability.

## Identification

Research has shown that humans exhibit a kind of "randomness fingerprint". For example, a short paper by Schulz et al. [^9] re-identified participants with a simple Euclidian distance between the points in n-dimensional space. This suggests that humans generate random numbers in predictable, unique ways, and has application in cryptography as well.

## Cryptography

One of the motivations for me starting this little project was the existance of "gibberish" in ciphers. Sometimes, ciphers will be mostly deciphered, except for some chunk. Much speculation is made about these chunks, mostly whether they are padding of gibberish or undeciphered codetext. With a reliable way to determine if a sequence is human-random, we may get an answer to that.

Indeed, I generated synthetic data for this task, and I was able to pick out the enciphered English with high probability. More research in this area would be great.


# Further Reading

I refer you to any of the papers I have linked in the footnotes, they are all wonderful.

# Acknowledgements

My little curiosity about this would be much harder to do without the work of Tim Angelike et al, and Brian Ellis' HumanRandom project.

[^1]:[Gauvrit, N. Singmann et al., Algorithmic complexity for psychology: A user-friendly implementation of the coding theorem method.](https://doi.org/10.3758/s13428-015-0574-3)

[^2]: This is not computable, but an upper bound can be made and is generally obtained by $$K(s) \approx -\log_2 m(s) + \mathcal{O}(1)$$ where $$m(s)$$ is the probability that a random deterministic Turing machine program would generate a string $$s$$. See the paper by Gauvrit for more information.

[^3]: [Angelike, T., Musch, J. A comparative evaluation of measures to assess randomness in human-generated sequences](https://doi.org/10.3758/s13428-024-02456-7)

[^4]:[https://github.com/kitchWWW/HumanRandom/](https://github.com/kitchWWW/HumanRandom/)

[^5]: [Vaswani, Ashish, et al., Attention is all you need.](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)

[^6]: [Jain, S., Wallace, B. Attention is not Explanation](https://arxiv.org/abs/1902.10186)

[^7]: [Wiegreffe, S., Pinter Y. Attention is not not Explanation](https://doi.org/10.18653/v1/D19-1002)

[^8]: [Abnar, S., Zuidema, W. Quantifying Attention Flow in Transformers](https://arxiv.org/pdf/2005.00928)

[^9]: [Schulz, M., Baier, S., Böhme, B., Bzdok, D., & Witt, K. A Cognitive Fingerprint in Human Random Number Generation.](https://doi.org/10.31234/osf.io/gub8e)