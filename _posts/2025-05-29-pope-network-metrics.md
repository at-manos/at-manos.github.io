---
layout: post
title: the network metrics that predicted the pope(*)
categories: [graph theory, social networks]
description: what can graph connections tell us about human connections?
keywords: pope, robert prevost, graphs, eigenvector centrality, betweenness centrality
mermaid: false
sequence: false
flow: false
mathjax: true
mindmap: false
mindmap2: false
---
# Habemus Papam!

There's a new pope, if you haven't heard. [Robert Francis Prevost](https://en.wikipedia.org/wiki/Pope_Leo_XIV), or as he's now known, [Pope Leo XIV](https://en.wikipedia.org/wiki/Pope_Leo_XIV). What you might not have heard was his name in the media swarm of people in contention to be pope.

Was this really such a surprise? Some Italian researchers at the University of Bocconi might not think so. Prior to the conclave, they calculated some network centrality metrics and found him at the top of one list, "eigenvector centrality".[^1]

![The network graph scaled by eigenvector centrality, where R. Prevost is the largest node.](/images/blog/pope/1748542033056.png)
<p style="text-align: center; margin-top: -1rem;">The College of Cardinals, scaled by eigenvector centrality</p>

But what is that? And how do we even measure "importance"?

# Measuring Importance

In any network of people, be it business, politics, or friendships, there are many notions of "importance". If we want to quantify it, the first thing that often jumps to mind is
> How many connections do you have?

## Degree Centrality

*Degree Centrality* is just that: a measure of how many nodes someone is connected to. 

Great! A simple, quick way to measure how important someone is. 

But, there's a problem. Have you spotted it? Over in the corner, Joe, who is friends with the entire janitorial staff, has just been promoted to the C-suite for his immense power and importance in the company. 

*Degree Centrality* doesn't take into account how important Joe's connections are. If only there was a way to do that. Wait. What's that below me?

## Eigenvector Centrality

What we would like, then, is for a node's importance to be proportional to the sum of their neighbors' importance. In the College of Cardinals, being connected to influential cardinals from major archdioceses matters more than knowing many auxiliary bishops.

Let's go back to the basics of graphs to find a way we can calculate such a thing.

Graphs are commonly stored as adjacency matrices. In the simplest case, we have a matrix $$A$$ where

$$
A[i][j] = \begin{cases}
0 & \text{No connection between i,j,}\\
1 & \text{Connection between i,j.}
\end{cases}
$$

![An example of a simple graph and its adjacency matrix](/images/blog/pope/1748543692014.png)
<p style="text-align: center; margin-top: -1rem;">An example of a simple graph and its adjacency matrix</p>

This matrix representation lends itself naturally to linear algebra, and in fact this is where the insight of *eigenvector centrality* comes from.

Again, we want any node ($$i$$)'s importance/centrality ($$c_i$$) to be proportional to the sum of its neighbors. 

This can be given by

$$
c_i \propto \sum_j A_{ij} c_j
$$

$$
c_i \propto \sum_j A_{ij} c_j
$$ 

implies that there is some *proportional constant* $$k$$ where

$$
c_i = k\sum_j A_{ij} c_j
$$

given recursively in lazy pseudocode by
```python
def centrality(node):
    # base case omitted
    c_node = 0
    for neighbor in node.neighbors:
        c_node += proportion * centrality(neighbor)
    return c_node
```

Observe that $$A_{ij}$$ is $$0$$ when there is no connection to $$c_j$$, so it is not counted in the sum.


$$
c_i = k\sum_j A_{ij} c_j
$$

If we pack our scores in a vector $$c=(c_1,c_2,\dots, c_n)^T$$, we can rewrite the above as

$$
c= kAc
$$

$$
\frac1kc = Ac
$$

$$
Ac = \frac1kc
$$

This *is* the eigenvector equation $$Ac = \lambda c$$, just using $$\frac1k$$ instead of $$\lambda$$.

Now that we've connected this problem to a well-known concept in linear algebra, we've now climbed on top of the shoulders of giants and can use all of the results that come with it.

The result $$Ac = \lambda c$$ means that when the adjacency matrix transforms all of the centralities, it is only scaled by $$\lambda$$ (the *eigenvalue*), not rotated. Inside this vector $$c$$ we can find the importance of every node $$i$$ at $$c_i$$.

But... which $$\lambda$$? Every matrix $$A$$ can have multiple eigenvectors, so which one do we choose?
### The Principal Eigenvector

For a strongly-connected graph, the [Perron-Frobenius theorem](https://en.wikipedia.org/wiki/Perron%E2%80%93Frobenius_theorem) guarantees that its adjacency matrix $$A$$ has a unique, largest, positive, real eigenvalue. Its corresponding eigenvector is called the **principal eigenvector**. This is the one we are interested in. Again, in that vector $$c$$ we can find the importance of every node $$i$$ in our network at $$c_i$$.
As a note, typically the eigenvector $$c$$ is normalized such that $$c_1+c_2+\dots+c_n=1$$ so we can interpret them as percentages.

We can find this principal eigenvector in many ways. For small graphs, you can probably get away with solving the textbook

$$
det(A - \lambda I) = 0
$$

to get all the possible eigenvalues, then getting the corresponding vector $$c$$ by

$$
(A-\lambda I)c = 0
$$

However, this is **terribly inefficient**, computing the full set of solutions is $$\mathcal{O}(n^4)$$ or worse. So, generally a process called [power iteration](https://en.wikipedia.org/wiki/Power_iteration) is used.

Sergey Brin and Larry Page realized that web pages shouldn't be ranked by just their content, but also the **importance of pages linking to them**. This lead to the development of PageRank, which is essentially just eigenvector centrality applied to the web graph.

The University of Bocconi arrived at the following top-5 for the eigenvector centrality scores.


1. Robert Prevost (US)
2. Lazzaro You Heung-sik (South Korea)
3. Arthur Roche (UK)
4. Jean-Marc Aveline (France)
5. Claudio Gugerotti (Italy)

I am no expert on Vatican politics, but it does make sense that Robert Prevost ranks the highest here. He was the Prefect of the Dicastery for Bishops, and likely influenced who became bishops, and thus cardinals, and worked closely in the appointment process.

## Betweenness Centrality

Another metric that they used, was *betweenness centrality*.

The idea is that nodes are important if they lie on many shortest paths **between** nodes; they are a bridge or bottleneck.

Think of working in a company, and to do anything with a database, you have to go talk to George, who's the only guy who knows COBOL and has been working there since 1968. He is the bottleneck, and thus is more important by betweenness centrality.

The University of Bocconi arrived at the following top-5 for the betweenness centrality scores.

1. Anders Arborelius (Sweden)
2. Pietro Parolin (Italy)
3. Víctor Fernández (Argentina)
4. Gérald Lacroix (Canada)
5. Joseph Tobin (USA)


If you would like to read more, please head over to the original press publication[^1]. I don't believe any technical paper has been put out by them as of yet, but I could be wrong.

[^1]: [https://www.unibocconi.it/en/news/network-conclave](https://www.unibocconi.it/en/news/network-conclave)

[^2]: [https://graphicmaths.com/computer-science/graph-theory/adjacency-matrices/](https://graphicmaths.com/computer-science/graph-theory/adjacency-matrices/)