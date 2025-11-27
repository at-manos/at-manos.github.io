---
layout: post
title: how to think about AI historically
categories: [history, neural networks]
description: The wars and winters of AI that led to a data center boom
keywords: ai, ethics, neural networks, history
mermaid: false
sequence: false
flow: false
mathjax: true
mindmap: false
mindmap2: false
image: /images/blog/social-ai/dartmouth-workshop.webp
---


# The wars and winters of AI
#### that led to a data center boom

# What is "AI"?

I see a lot of confusion around AI nowadays, and I thought I would put together a post to offer the story of AI in a way that will equip you to think about how this technology grew and why it is the way it is today.

Some confusions around AI I hope to address:
- How did AI start?
- Is today's "AI" really AI?
- What is "generative AI"?
- What is "machine learning"?
- Why does AI need so much power?

To address these, I think it would be helpful to start at the beginning of the field of "Artificial Intelligence", and trace through how we got to its current titanic scale, and the downstream effects of that scale.


## Historical

Artificial Intelligence, as a field, was the product of some early computing research that saw computers as more than just "number-crunchers". These researchers believed computers had an ability to learn anything that can be completely described.

> To proceed on the basis of the conjecture that every aspect of learning or any other feature of intelligence can in principle be so precisely described that a machine can be made to simulate it.
>
> *Original Dartmouth Workshop Proposal* [^dartmouth]

[John McCarthy](https://en.wikipedia.org/wiki/John_McCarthy_(computer_scientist)) organized the [Dartmouth Workshop](https://en.wikipedia.org/wiki/Dartmouth_workshop) in 1956 alongside Marvin Minsky, Nathaniel Rochester, and Claude Shannon to explore this emerging body of work around what McCarthy called "artificial intelligence". Later on, Minsky also co-founded the [MIT AI Lab](https://en.wikipedia.org/wiki/MIT_Computer_Science_and_Artificial_Intelligence_Laboratory).

![Dartmouth Workshop attendees, 1956](/images/blog/social-ai/dartmouth-workshop.webp)
<p style="text-align: center; margin-top: -1rem;">Dartmouth Workshop attendees, 1956. Minsky is third from the left.</p>

Minsky defined artificial intelligence as follows:

> Artificial intelligence is the science of making machines do things that would require intelligence if done by men.[^sip]

[Oliver Selfridge](https://en.wikipedia.org/wiki/Oliver_Selfridge), another prominent computer scientist around that time, gave a similar description of the field:

> I don't mean that machines will behave like men. [...] and I don't think my daughter will ever marry a computer. But I think that computers will be doing the things that men do when they say they are "thinking"[^thinkingmachine]


Definitions of "AI" vary, with some people considering today's models not truly artificial intelligence. However, the idea that this young body of scattered research was part of a new field came precisely with using the term "Artificial Intelligence".

## Machine Learning

Three years after the Dartmouth christening of "Artificial Intelligence", Arthur Samuel at IBM coined "machine learning" in 1959. Through his demonstrations of a self‑improving Checkers program, the term was linked to algorithms that improve through experience.

![Arthur Samuel working on his Checkers program in 1959](/images/blog/social-ai/samuel.jpg)
<p style="text-align: center; margin-top: -1rem;">Arthur Samuel working on his Checkers program in 1959</p>

While Minsky's style of AI encountered several funding ["winters"](https://en.wikipedia.org/wiki/AI_winter) (a period of reduced funding or interest), researchers persisted with data-driven statistical approaches. By the mid-1980s, machine learning (ML) had become its own field, marked by the launch of the journal *Machine Learning* in 1986.

By 1997, Tom M. Mitchell's textbook *Machine Learning* became the first comprehensive introduction for undergraduates, graduates, and practitioners, and thus the term was cemented forever.

Today, ML is considered a core subfield of AI that focuses on algorithms that learn patterns from data. However colloquially, when people say "machine learning" they often think of traditional statistical methods, whereas "AI" evokes neural networks and deep learning (though technically they're all part of the same field).

Put broadly by Charu C. Aggarwal in his fantastic book *Linear Algebra and Optimization for Machine Learning*:
> Machine learning builds mathematical models from data [...] in order to predict some variables from others


## "Generative AI"

While we're on the topic of "what is AI?", I thought I would fast-forward to a modern usage and clear up a term I've been hearing a lot lately: "generative AI". This term is used sort of loosely, and so I thought I would clarify what it actually means in different places.
### In research / marketing
Generative AI is a real phrase used in the field, and it is broadly used to mean models that learn the underlying patterns of a dataset so they can create brand new examples that are like the originals.

For example,
- a text model (e.g., ChatGPT) outputs entirely new sentences based on the sentences it has been trained on.
- an image model (e.g., Stable Diffusion) outputs entirely new images based on the images it has been trained on.

#### In contrast to discriminative

In Generative Adversarial Networks (GANs), we pit two models against each other as adversaries: a discriminative net, and a generative net. Take an image model for example:

|Network|Function|
|----|--------|
|Generator|Generates images|
|Discriminator|Determines if images are fake|

In a GAN like this, the generative component learns to "fool" the discriminative one, thus producing better images.


<div id="face-generator-widget">
  <img
    id="face-img"
    src="/images/blog/social-ai/stylegan.png"
    alt="StyleGAN‑generated human face"
    width="400"
    height="400"
    style="    display: block !important;
    margin-left: auto !important;
    margin-right: auto !important;"
    onerror="this.onerror=null; this.parentElement.classList.remove('loading'); this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjQwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxOCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkZhaWxlZCB0byBsb2FkPC90ZXh0Pjwvc3ZnPg==';"
  />
  
  <style>
    #face-generator-widget {
      display: block;
      margin: 20px auto;
      cursor: pointer;
      width: 400px;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 8px 24px rgba(0,0,0,0.12);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
      background: #f8f9fa;
    }
    
    #face-generator-widget:hover {
      transform: translateY(-2px);
      box-shadow: 0 12px 32px rgba(0,0,0,0.15);
    }
    
    #face-generator-widget img {
      display: block;
      width: 400px;
      height: 400px;
      object-fit: cover;
      transition: opacity 0.3s ease;
    }
    
    #face-generator-widget.loading::after {
      content: "";
      position: absolute;
      top: 0; left: 0;
      width: 400px; height: 400px;
      background: rgba(255,255,255,0.5);
      backdrop-filter: blur(4px);
      pointer-events: none;
      border-radius: 12px;
    }
    
    #face-generator-widget.loading::before {
      content: "";
      position: absolute;
      top: 200px; left: 200px;
      width: 48px; height: 48px;
      margin: -24px 0 0 -24px;
      border: 4px solid rgba(99, 102, 241, 0.2);
      border-top-color: rgba(91, 172, 223, 0.8);
      border-radius: 50%;
      animation: face-spin 1s linear infinite;
      pointer-events: none;
      z-index: 10;
    }
    
    @keyframes face-spin {
      to { transform: rotate(360deg); }
    }
    
    #face-generator-widget.loading img {
      opacity: 0.7;
    }
  </style>

  <script>
    (function() {
      const widget = document.getElementById('face-generator-widget');
      const img = document.getElementById('face-img');
      
      function showLoading() {
        widget.classList.add('loading');
      }
      
      function hideLoading() {
        widget.classList.remove('loading');
      }
      
      img.addEventListener('click', () => {
        showLoading();
        img.src = 'https://thispersondoesnotexist.com/?' + Date.now();
      });
      
      img.addEventListener('load', hideLoading);
      
      if (img.complete && img.naturalHeight !== 0) {
        hideLoading();
      }
    })();
  </script>
</div>


<p markdown="1" style="text-align: center; margin-top: -1rem;"><b>Click</b> to see a new fake face, generated[^tpdne] by thispersondoesnotexist.com</p>



Famously, the website [thispersondoesnotexist.com](https://thispersondoesnotexist.com) used [StyleGAN](https://en.wikipedia.org/wiki/StyleGAN) to viral effect, producing generated images of faces that are hard to discriminate from real ones.


# Understanding the scale of AI

Why does AI use so much power? Does it? These are fair questions, and I do not think it is well understood by the general public, and even people in the tech space.

## Winters, resurgences, winters.

To understand why today's AI is so large and computationally expensive, it helps to understand the paths not taken, and why certain dumb ideas ended up scaling better than the smart ones.

![The broad differences between connectionist AI and symbolic AI](/images/blog/social-ai/con-v-sym.png)


Let's think back to Minsky. In that time of early AI research, there were two prominent camps, the "connectionists", and the "symbolicists". Symbolicists, like some Dartmouth attendees, argued that 'machine thinking' could be produced by rigid structure and manipulation of symbols.

Connectionists, like [Frank Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt), argued that "machine thinking" could be produced by the connections between many simple units. Based off of the human neuron, Rosenblatt proposed the "perceptron", a model made of simple units able to connected into arbitrarily large networks.

![A figure of an asymptote from Frank Rosenblatt's paper introducing the perceptron](/images/blog/social-ai/percep-asym.png)

> This asymptote can be made arbitrarily close to unity by increasing the number of association cells in the system. [^perceptron]

Each of the connections between these "neurons" had a weight associated to it. By updating these weights, Rosenblatt hoped that the perceptron could approximate any function.

Rosenblatt's simple model wasn't without fault, however, and it wouldn't go unnoticed. In their 1969 book *Perceptrons*, Marvin Minsky and Seymour Papert rigorously exposed the limitations of single‑layer perceptrons; this critique not only contributed to funding cuts that helped contribute to the first [AI Winter](https://en.wikipedia.org/wiki/AI_winter) (1974 - 1980) but also cast a shadow of doubt over neural network research for nearly two decades.

What the symbolicists didn't know then, was that neural networks would come to be the most dominant, powerful sorts of artificial intelligence.[^convsym]


## The storm brews

Before we get into the societal impact, it’s worth tracing how we got from Rosenblatt’s perceptron to today’s trillion‑parameter models.

Early connectionist approaches couldn't handle deep networks with lots of connections, which didn't help on top of the brutal critique from Minsky and Papert. Early perceptron models could only "learn" if they were tiny (and thus, not very useful).

Because of these limitations, there were a few more slumps in research until David Rumelhart, Geoffrey Hinton, and Ronald J. Williams published a paper in 1986, popularizing "backpropagation", an algorithm that solved this very problem.

> **Backpropagation**
> 
> a way for models to "learn from their mistakes" by comparing the error between the real measurement and the model's prediction
> 
> Crucially, it updated *every* parameter in the model in one pass.


With backpropagation and a few more innovations[^nn-lims], models could perform better and better at the task they were being trained for just by adding more connections and more data.
    
The door had swung open for huge, highly interconnected networks à la Rosenblatt's perceptron that could "learn" efficiently, something Rosenblatt unfortunately did not live to see, dying in 1971.

After this, there would be one more long winter (1987-1993), with machine learning quietly jogging along in the meantime. In the background, though, a powerful storm was brewing between 2000 and 2010. Chief among the ingredients were: 

1. The networked, wide proliferation of very large datasets
2. The availability of fast commodity hardware (importantly, GPUs)

## ImageNet
One of these *very large datasets* was ImageNet, a dataset of over 14 million labeled images crowdsourced by Fei-Fei Li and her colleagues. With this data, they launched an image classification challenge with PASCAL VOC in 2010.

As computer scientists, not really known for our marketing skills, this challenge was given the unwieldy name of the "ImageNet Large Scale Visual Recognition Challenge" (ILSVRC).

The idea was simple: here's some images, write a program that tells me what is in them.

### Traditional Methods

At the inaugural ILSVRC in 2010, the popular methods were bespoke, complicated, and time-consuming. The winner (NEC-UIUC), used a combination of many features, such as color histograms, local patterns, and more, and fed it into a Support Vector Machine (an algorithm in machine learning). They achieved 72% top-5 accuracy, which meant the real answer was in their top-five 72% of the time.

In 2011, there were only marginal improvements. Despite tuning hyper-parameters and changing the features they fed into the SVM, the winners (XRCE) achieved just 74% top-5 accuracy, a 2% increase.

### AlexNet

In 2012, a submission was made that shocked everyone. A team consisting of Alex Krizhevsky, Ilya Sutskever (future OpenAI co-founder), and *Geoffrey Hinton* (co-author of the 1986 backpropagation paper) submitted a model that involved nothing bespoke, nothing complicated, and comparatively it didn't even take that long to develop. It was called "AlexNet".

The creators realized that with such a large dataset, backpropagation gave neural networks the unique scaling potential to learn a classification function based only on the vast amounts of data given, and that's exactly what they did.

AlexNet worked by taking in images, and feeding them through a bunch of image filters (formally, convolutions) learned by a Convolutional Neural Network (CNN). While the technology had been around for a while in academia[^lenet], it wasn't obvious that you could scale it end-to-end over a million images and train it in just days.

<img src="/images/blog/social-ai/alexnet.png" alt="AlexNet predictions on 8 ImageNet images" width="700">
<p markdown="1" style="text-align: center; margin-top: -1rem;">AlexNet predictions on 8 ImageNet images [^alexnet]</p>


There were no explicit patterns or recognition programmed in; it was end to end, image to prediction. It was trained on two Nvidia GTX580 GPUs, powered by a vast dataset of 1.2 million images, for 5 to 6 days. They achieved a staggering ~85% top-5 accuracy rate, an almost 11% increase and **half** the error rate of last year's winner.[^alexnet]

Within a year, pretty much every single top entry in ILSVRC was a deep CNN. Manual feature engineering from the years previous vanished.

### Big Data

This was the sea-change that brought about the modern rise in using **big data** to solve problems, and with big data, came big compute.

We traded developer time, bespoke solutions, and low memory for deep neural networks trained off of lots of data, which in turn, required teraFLOPs of GPU compute to update the model. With the door labeled "scale" now undeniably blasted off its hinges, all that was left was to keep pushing it higher and higher in order to solve bigger and bigger problems, and that's exactly what happened.

# Data Centers

A little before the time of AlexNet, the Internet was growing rapidly and companies needed a centralized hub to store servers, applications, and data. This led to the creation of the data center industry, networks of buildings purpose-built for fast compute and big data.

Data centers were the natural choice for the burgeoning new growth in neural networks. They offered fast, cheap, distributed clusters; if you needed only a days worth of compute, that's all you needed to pay for. While AI has pushed demand higher for data centers in recent years, data centers have been the fundamental hubs of data infrastructure for a long time.

Because of this, AI started finding its home in cloud data centers instead of university walls. What was once relegated to academic research just became very physical, and would end up driving the construction and usage of these structures.

As the problems AI was being used to solve got bigger, its new home in the cloud gave AI the unprecedented space to grow unbounded. ChatGPT (2022) famously used transformers trained on terabytes of human language. Image models like Stable Diffusion (2022) needed to store billions of images and train on them. 

However, large buildings with lots of computers working very hard don't come out of thin air. There are many considerations to take when examining data centers from an efficiency and sustainability standpoint. If you would like a deep dive into the effects of these systems in the real world, I am currently working on a sister article to this one, titled **"how to think about AI practically"**. I will update this article with a link there when it's up.

[^thinkingmachine]: [*The Thinking Machine*](https://www.youtube.com/watch?v=aygSMgK3BEM), Interview. YouTube.

[^sip]: Minsky, Marvin (ed.). *Semantic Information Processing*. Cambridge, MA: MIT Press, 1968.


[^nn-lims]: Better initialization, regularization, and preventing vanishing gradients, among others.

[^lenet]: Famously, [LeNet](https://en.wikipedia.org/wiki/LeNet) by Yann LeCun et al.

[^perceptron]: The Perceptron: A Probabilistic Model For Information Storage and Organization in the Brain, Frank Rosenblatt

[^alexnet]: Krizhevsky, A., Sutskever, I., & Hinton, G. E. [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). *Advances in Neural Information Processing Systems*, 25, 2012.

[^tpdne]: No requests are made unless you click, in case you are worried about generative AI.
[^convsym]: Both approaches are used today, including hybrid mixes of both (e.g., [neuro-symbolic](https://en.wikipedia.org/wiki/Neuro-symbolic_AI)) but connectionist models dominate the field of "AI" today.
[^dartmouth]: [Artificial Intelligence Coined at Dartmouth](https://home.dartmouth.edu/about/artificial-intelligence-ai-coined-dartmouth)