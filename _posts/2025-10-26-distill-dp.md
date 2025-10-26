---
layout: distill
title: Teaching AI to Whisper, Not Shout
description: How privacy-preserving deep learning (DP-SGD) works in practice.
tags: [Differential Privacy, DP-SGD, Deep Learning]
giscus_comments: true
date: 2025-10-26
featured: true
mermaid:
  enabled: true
  zoomable: true
code_diff: true
map: true
chart:
  chartjs: true
  echarts: true
  vega_lite: true
tikzjax: true
typograms: true

authors:
  - name: YAN SHIYU
    url: "https://siashiay.github.io/"
affiliations:
  - name: Yonsei University
    url: https://www.yonsei.ac.kr/en_sc/

scholar:
  style: apa
  locale: en
  source: /_bibliography/
  bibliography: papers.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Equations
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Why Whisper?
  - name: What Does DP-SGD Actually Do?
  - name: Algorithm: Making SGD Private
  - name: Implementing DP-SGD: Where Theory Meets TensorFlow
  - name: When Math Meets Reality
  - name: Lessons from Experiments
  - name: Beyond the Noise
  - name: References

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Motivation

Deep learning showed me how models can learn powerful patterns from massive amounts of data.  
But data privacy reminded me that these datasets often come from **real people** — each record a small fragment of someone’s life.  

That made me wonder:  
*Can we build models that learn, without exposing the people who taught them?*  

This question led me to explore **DP-SGD (Differentially Private Stochastic Gradient Descent)** —  
an elegant training method that allows neural networks to **learn responsibly** —  
keeping the insights, but forgetting the identities.

---

Just like how attention helps models focus on **important regions** in an image,  
differential privacy helps models **forget unnecessary details** about individual samples.  
Take the two brains in Fig. 1 as an example.

![Illustration of DP-SGD vs SGD](assets/img/dpsgd_brain.png)

**Figure 1:** A visual comparison between **SGD** and **DP-SGD**.  
The brain on the left (SGD) memorizes every tiny detail — faces, words, and data points — 
while the brain on the right (DP-SGD) only retains the general patterns after adding noise.  
This simple idea makes models *learn like humans — remembering what matters, and forgetting what doesn’t.*

---

## Why Whisper?
*Why should a neural network learn to whisper instead of shout?*

Modern deep learning thrives on **data — lots of it.**  
The more data a model sees, the smarter it becomes.  
But there’s a catch: much of that data is **personal** — medical records, voice recordings, photos, chat logs.  

> “The more a model learns, the more it remembers.”

Imagine training a speech recognition system.  
It might not only learn to recognize words,  
but also **remember the speaker’s voice** so well  
that someone could later reconstruct what that person said.  
That’s not intelligence — that’s a privacy leak.

### The Problem
Deep neural networks have millions (or even billions) of parameters.  
These parameters don’t just learn _patterns_ — they can memorize _examples_.  
Even without malicious intent, attackers can exploit this fact.

Researchers have shown **model inversion attacks**,  
where adversaries, given access to a model’s outputs,  
can reconstruct sensitive parts of the training data —  
like recreating someone’s face from a classifier’s memory.

In short:
> The bigger and more capable our models become,  
> the more they risk revealing what they were trained on.

### The Goal
The main goal of this paper is to train deep learning models  that can **learn from data without revealing anyone’s secrets**.  

The authors introduced **DP-SGD**, a clever twist on the classic SGD algorithm.  
It adds a bit of random noise during training,  so the model learns general patterns — but forgets the fine details about any individual.  

That is to say, they wanted to prove that **privacy and performance don’t have to be enemies** —  
you can build smart models and keep people’s data safe at the same time.

---

## What Does DP-SGD Actually Do?

In deep learning, we usually train models with **stochastic gradient descent (SGD)**, letting them adjust their weights based on every tiny detail in the data.  
That’s great for accuracy — but sometimes the model *remembers too much*, even bits of someone’s private information.

That’s where **DP-SGD** (Differentially Private Stochastic Gradient Descent) steps in.  
It takes the same learning process and adds a bit of mathematical “privacy dust.”

---

### The Math Behind the Privacy

Under the hood, DP-SGD relies on the formal definition of **differential privacy**.  
A randomized mechanism \( \mathcal{M} : \mathcal{D} \to \mathcal{R} \) is said to satisfy  
\( (\varepsilon, \delta) \)-differential privacy if, for any two adjacent datasets \( d, d' \in \mathcal{D} \) and any subset of outputs \( S \subseteq \mathcal{R} \),

\[
\Pr[\mathcal{M}(d) \in S] \le e^{\varepsilon} \Pr[\mathcal{M}(d') \in S] + \delta
\]

This means that whether or not your data is included, the algorithm’s behavior doesn’t change much —  
no single individual can drastically affect the outcome.

---

### The Gaussian Mechanism

To achieve this, DP uses noise drawn from a Gaussian distribution:

\[
\mathcal{M}(d) \triangleq f(d) + \mathcal{N}(0, S_f^2 \cdot \sigma^2)
\]


where  
- \( S_f \) is the **sensitivity** (the largest possible change in \( f \) when one data point changes), and  
- \( \sigma \) controls the noise scale — larger \( \sigma \) means stronger privacy.  

DP-SGD effectively applies this mechanism at every training step, ensuring that each gradient update respects the same privacy guarantee.

---

So to me, DP-SGD is a way of teaching models to **learn from the crowd, not from individuals**. It keeps the essence of the data, but forgets who said what — which feels like a much more human way to learn.


---

## Algorithm: Making SGD Private

Algorithm 1 outlines our basic method for training a model with parameters \(\theta\) by minimizing the empirical loss function \(\mathcal{L}(\theta)\).  
At each step of SGD, we compute the gradient \(\nabla_\theta \mathcal{L}(\theta, x_i)\) for a random subset of examples, clip the ℓ₂ norm of each gradient, compute the average, add Gaussian noise to protect privacy, and take a step in the opposite direction of this average noisy gradient.  
At the end of training, we output the final model \(\theta_T\) and compute the overall privacy cost \((\varepsilon, \delta)\) using a privacy accountant.

---

### **Algorithm 1: Differentially Private SGD (Outline)**

**Input:**  
Examples \(\{x_1, \dots, x_N\}\), loss function  
\[
\mathcal{L}(\theta) = \frac{1}{N} \sum_i \mathcal{L}(\theta, x_i)
\]  
Parameters: learning rate \(\eta_t\), noise scale \(\sigma\), group size \(L\), gradient norm bound \(C\).  

**Initialize** \(\theta_0\) randomly  

**for** \(t \in [T]\) **do**  
&nbsp;&nbsp;&nbsp;&nbsp;Take a random sample \(L_t\) with sampling probability \(L/N\)  
&nbsp;&nbsp;&nbsp;&nbsp;**Compute gradient:**  
\[
g_t(x_i) \leftarrow \nabla_\theta \mathcal{L}(\theta_t, x_i)
\]  

&nbsp;&nbsp;&nbsp;&nbsp;**Clip gradient:**  
\[
\bar{g}_t(x_i) \leftarrow g_t(x_i) / \max\Big(1, \frac{||g_t(x_i)||_2}{C}\Big)
\]  

&nbsp;&nbsp;&nbsp;&nbsp;**Add noise:**  
\[
\tilde{g}_t \leftarrow \frac{1}{L} \Big(\sum_i \bar{g}_t(x_i) + \mathcal{N}(0, \sigma^2 C^2 \mathbf{I})\Big)
\]  

&nbsp;&nbsp;&nbsp;&nbsp;**Descent:**  
\[
\theta_{t+1} \leftarrow \theta_t - \eta_t \tilde{g}_t
\]  

**Output:** \(\theta_T\) and compute the overall privacy cost \((\varepsilon, \delta)\) using a privacy accounting method.

---

### The Moments Accountant: Keeping Track of Privacy

The **Moments Accountant** keeps a tight bound on how much privacy loss has accumulated during training.  
It’s based on the idea that every noisy gradient update “spends” a small privacy cost —  
and this accountant tracks those costs precisely over all steps.

The accountant defines a *privacy loss random variable* that compares how likely a given output is when training on two adjacent datasets (differing by one person’s data):

\[
c(o; \mathcal{M}, d, d') = \log \frac{\Pr[\mathcal{M}(d) = o]}{\Pr[\mathcal{M}(d') = o]}
\]

This measures how much the inclusion or exclusion of one example can influence the output.  
We then compute the **log moments** of this variable:

\[
\alpha_{\mathcal{M}}(\lambda; d, d') = \log \mathbb{E}_{o \sim \mathcal{M}(d)} [\exp(\lambda c(o; \mathcal{M}, d, d'))]
\]

and take the maximum over all possible pairs of neighboring datasets.  
These moments can then be converted into an overall privacy guarantee \((\varepsilon, \delta)\).

In short, the Moments Accountant works like a **ledger** — it records exactly how much privacy has been spent,  
ensuring that the final model doesn’t exceed the allowed budget.

---

### Hyperparameter Tuning: Balancing Privacy and Accuracy

In deep learning, tuning **hyperparameters** like batch size, noise level, and learning rate can dramatically affect both accuracy and privacy.  
Through experiments, the paper finds that model accuracy is more sensitive to these parameters than to network architecture itself.

When we try multiple hyperparameter configurations, each run technically adds to the total privacy cost.  
However, using results from theory (e.g., Gupta et al.), we can control this accumulation and reuse information across runs more efficiently.  

Empirically, the best settings tend to be:
- Small to moderate **batch sizes** — too large a batch increases privacy loss.  
- A relatively **large learning rate** at the beginning, which decays over time.  
- Carefully chosen **noise scale** \(\sigma\) to balance accuracy and privacy.

---

### That’s What I Think

To me, this section is where the algorithm becomes “human.”  
DP-SGD isn’t just about math or bounds — it’s about *teaching a model to learn responsibly*.  
The gradient clipping feels like setting social boundaries, and the Gaussian noise is like gentle confusion that protects everyone’s secrets.  

The Moments Accountant then plays the role of a careful observer, making sure the learning process never crosses the line of privacy.  
And hyperparameter tuning is where art meets science — finding that sweet spot  
where the model learns *just enough* without remembering too much.

Here is the lightweight implementation of the code.

---

## Implementing DP-SGD: Where Theory Meets TensorFlow

We have implemented the differentially private SGD algorithms in Tensor Flow. The source code is available under an Apache 2.0 license from github.com/tensorflow/models.
In practice, it works like this:
- **Sanitizer** trims each sample’s gradient so no single datapoint can shout too loudly.  
- **Accountant** keeps track of the privacy “budget” — how much (ε, δ) you’ve spent so far.  
- **Optimizer** updates the model using a noisy, clipped gradient — learning from the crowd, not from individuals.

---

<details>
  <summary><strong> Click to view full DP-SGD code (Python-like)</strong></summary>

{% highlight python linenos %}
# --- DP-SGD Optimizer, Sanitizer, and Accountant -----------------------------

class PrivacyAccountant:
    """Toy moments accountant for demonstration."""
    def __init__(self, eps_budget=8.0, delta=1e-5):
        self.eps_budget = eps_budget
        self.delta = delta
        self._eps_spent = 0.0

    def accumulate_privacy_spending(self, batch_size, noise_sigma, sampling_prob):
        step_eps = sampling_prob / max(noise_sigma, 1e-6)
        self._eps_spent += step_eps

    def get_spent_privacy(self):
        return self._eps_spent, self.delta

    def within_limit(self):
        return self._eps_spent < self.eps_budget


class Sanitizer:
    """Clips per-example gradients and adds Gaussian noise."""
    def __init__(self, clipping_C, noise_sigma):
        self.C = float(clipping_C)
        self.sigma = float(noise_sigma)

    def _l2_norm(self, g):
        return (sum(x * x for x in g) ** 0.5)

    def clip_per_example(self, per_example_grads):
        clipped = []
        for g in per_example_grads:
            norm = self._l2_norm(g)
            scale = min(1.0, self.C / max(norm, 1e-12))
            clipped.append([scale * x for x in g])
        return clipped

    def add_noise_to_batch_mean(self, clipped_grads):
        L = len(clipped_grads)
        d = len(clipped_grads[0])
        mean_grad = [sum(g[k] for g in clipped_grads) / L for k in range(d)]
        import random
        noisy = [m + (self.sigma * self.C) * random.gauss(0.0, 1.0) for m in mean_grad]
        return noisy


class DPSGDOptimizer:
    """Integrates Sanitizer and Accountant into a training step."""
    def __init__(self, accountant: PrivacyAccountant, sanitizer: Sanitizer, lr=0.05):
        self.accountant = accountant
        self.sanitizer = sanitizer
        self.lr = lr

    def per_example_gradients(self, loss_fn, params, batch):
        grads = []
        eps = 1e-4
        for x in batch:
            base = loss_fn(params, x)
            g = [0.0 for _ in params]
            for j in range(len(params)):
                p = params[:]
                p[j] += eps
                g[j] = (loss_fn(p, x) - base) / eps
            grads.append(g)
        return grads

    def step(self, loss_fn, params, batch, sampling_prob):
        per_ex_grads = self.per_example_gradients(loss_fn, params, batch)
        clipped = self.sanitizer.clip_per_example(per_ex_grads)
        noisy_grad = self.sanitizer.add_noise_to_batch_mean(clipped)
        self.accountant.accumulate_privacy_spending(
            batch_size=len(batch),
            noise_sigma=self.sanitizer.sigma,
            sampling_prob=sampling_prob,
        )
        return [p - self.lr * g for p, g in zip(params, noisy_grad)]


# ------------------------------- Training Loop -------------------------------

def dp_train(dataset, init_params, loss_fn, *,
             lr=0.05, clipping_C=1.0, noise_sigma=1.0,
             batch_size=64, epochs=5, delta=1e-5, eps_budget=8.0):
    N = len(dataset)
    sampling_prob = batch_size / N
    accountant = PrivacyAccountant(eps_budget=eps_budget, delta=delta)
    sanitizer = Sanitizer(clipping_C=clipping_C, noise_sigma=noise_sigma)
    opt = DPSGDOptimizer(accountant, sanitizer, lr=lr)

    import random
    params = init_params[:]

    for epoch in range(epochs):
        random.shuffle(dataset)
        for i in range(0, N, batch_size):
            if not accountant.within_limit():
                break
            batch = dataset[i:i + batch_size]
            params = opt.step(loss_fn, params, batch, sampling_prob)

        eps, delt = accountant.get_spent_privacy()
        print(f"[epoch {epoch+1}] θ={params}  |  privacy spent ε≈{eps:.3f}, δ={delt}")

        if not accountant.within_limit():
            print("Stopping early: privacy budget exhausted.")
            break

    return params, accountant.get_spent_privacy()
{% endhighlight %}

</details>

---

> Algorithm 1 becomes a working loop — each step clips, adds noise, and updates the model, while a privacy accountant keeps score of (ε, δ).  
> When the budget runs out, training stops — your model learns *from the crowd, not the individuals*.

---

## Does Privacy Hurt Accuracy?(Results)

After building our differentially private SGD system, it’s time to see **how well it actually works**.  
The experiments follow the original paper setup — evaluating **the moments accountant**, and testing on **MNIST** and **CIFAR-10**.

---

#### Applying the Moments Accountant

The **moments accountant** gives us a much **tighter privacy bound** than the old “strong composition theorem”.  
Instead of overspending the privacy budget too quickly, it keeps the noise–privacy tradeoff well balanced.

Here’s the key idea:

> The overall privacy loss \((\varepsilon, \delta)\) depends on  
> the sampling rate \(q = L/N\), the number of steps \(T = E/q\),  
> and the noise scale \(\sigma\).

When using the same training settings (q = 0.01, σ = 4, δ = 10⁻⁵),  
we can compare the two accounting methods directly.

<figure>
  <img src="/assets/img/moments_vs_composition.png" alt="Privacy accountant vs strong composition" width="480">
  <figcaption><strong>Figure 2.</strong> The ε-value as a function of training epochs, comparing the strong composition theorem and the moments accountant. The latter achieves much tighter bounds.</figcaption>
</figure>

Using the moments accountant, we achieve roughly  
**(2.55, 1e−5)-differential privacy**,  
while the old method only achieves around **(24.22, 1e−5)** —  
a dramatic improvement in privacy without hurting training.

---

#### MNIST — Privacy Still Learns

We start with the classic **MNIST digit classification** task:  
60,000 training samples and 10,000 test samples of 28×28 grayscale images.

We use:
- a **60-dimensional PCA projection**,  
- a **single hidden layer with 1,000 ReLU units**,  
- and a **lot size of 600**.

The model reaches about **98.3% accuracy** after 100 epochs —  
almost the same as the non-private baseline (98.5%),  
showing that DP-SGD learns *almost as well as regular SGD* when the noise is tuned properly.

We tested three noise scales:
- **Small noise** (σ = 2, δ = 1e−5) → ε ≈ 8  
- **Medium noise** (σ = 4, δ = 1e−5) → ε ≈ 4  
- **Large noise** (σ = 8, δ = 1e−5) → ε ≈ 2

And the results are surprisingly stable — around **90–97% accuracy** depending on σ.  
That’s a great privacy–accuracy tradeoff.

---

> *Fun takeaway:*  
> Differential privacy doesn’t mean “worse learning”.  
> It just means “learning quietly” — your model still finds patterns,  
> but it forgets who the individual datapoints were.

---

### What This Tells Us

DP-SGD with a moments accountant proves that **privacy-preserving deep learning is feasible**.  
You can train neural networks that remember trends, not people — and do it with quantifiable (ε, δ) guarantees.

In short:
- The **moments accountant** reduces overestimation of ε.  
- **MNIST** accuracy stays high (~98%).  
- **Training stability** improves as ε shrinks smoothly.

It’s a beautiful example of math meeting engineering — and both protecting privacy *and* achieving state-of-the-art accuracy.


---

## Diff2Html

This theme also supports integrating [Diff2Html](https://github.com/rtfpessoa/diff2html), a tool that beautifully renders code differences (diffs) directly in markdown. Diff2Html is ideal for showcasing code changes, allowing you to clearly present additions, deletions, and modifications. It’s perfect for code reviews, documentation, and tutorials where step-by-step code changes need to be highlighted—you can even introduce changes across multiple files at once.

````markdown
```diff2html
diff --git a/utils/mathUtils.js b/utils/mathUtils.js
index 3b5f3d1..c7f9b2e 100644
--- a/utils/mathUtils.js
+++ b/utils/mathUtils.js
@@ -1,8 +1,12 @@
-// Basic math utilities
+// Extended math utilities with additional functions

-export function calculateArea(radius) {
-    const PI = 3.14159;
+export function calculateCircleMetrics(radius) {
+    const PI = Math.PI;
     const area = PI * radius ** 2;
+    const circumference = 2 * PI * radius;
+
+    if (!isValidRadius(radius)) throw new Error("Invalid radius");
+
     return { area, circumference };
 }

-export function validateRadius(radius) {
+export function isValidRadius(radius) {
     return typeof radius === 'number' && radius > 0;
 }

diff --git a/main.js b/main.js
index 5f6a9c3..b7d4e8f 100644
--- a/main.js
+++ b/main.js
@@ -2,9 +2,12 @@
 import { calculateCircleMetrics } from './utils/mathUtils';

-function displayCircleMetrics(radius) {
-    const { area } = calculateCircleMetrics(radius);
+function displayCircleMetrics(radius) {
+    const { area, circumference } = calculateCircleMetrics(radius);
     console.log(`Area: ${area}`);
+    console.log(`Circumference: ${circumference}`);
 }

-displayCircleMetrics(5);
+try {
+    displayCircleMetrics(5);
+} catch (error) {
+    console.error("Error:", error.message);
+}
```
````

Here’s how it will look when rendered with Diff2Html:

```diff2html
diff --git a/utils/mathUtils.js b/utils/mathUtils.js
index 3b5f3d1..c7f9b2e 100644
--- a/utils/mathUtils.js
+++ b/utils/mathUtils.js
@@ -1,8 +1,12 @@
-// Basic math utilities
+// Extended math utilities with additional functions

-export function calculateArea(radius) {
-    const PI = 3.14159;
+export function calculateCircleMetrics(radius) {
+    const PI = Math.PI;
     const area = PI * radius ** 2;
+    const circumference = 2 * PI * radius;
+
+    if (!isValidRadius(radius)) throw new Error("Invalid radius");
+
     return { area, circumference };
 }

-export function validateRadius(radius) {
+export function isValidRadius(radius) {
     return typeof radius === 'number' && radius > 0;
 }

diff --git a/main.js b/main.js
index 5f6a9c3..b7d4e8f 100644
--- a/main.js
+++ b/main.js
@@ -2,9 +2,12 @@
 import { calculateCircleMetrics } from './utils/mathUtils';

-function displayCircleMetrics(radius) {
-    const { area } = calculateCircleMetrics(radius);
+function displayCircleMetrics(radius) {
+    const { area, circumference } = calculateCircleMetrics(radius);
     console.log(`Area: ${area}`);
+    console.log(`Circumference: ${circumference}`);
 }

-displayCircleMetrics(5);
+try {
+    displayCircleMetrics(5);
+} catch (error) {
+    console.error("Error:", error.message);
+}
```

---

## Leaflet

[Leaflet](https://leafletjs.com/) is created by Ukrainian software engineer [Volodymyr Agafonkin](https://agafonkin.com/), allowing interactive maps to be embedded in webpages. With support for [GeoJSON data](https://geojson.org/), Leaflet allows you to highlight specific regions, making it easy to visualize geographical information in detail.

You can use the following code to load map information on [OpenStreetMap](https://www.openstreetmap.org/):

````markdown
```geojson
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "name": "Crimea",
        "popupContent": "Occupied Crimea"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              33.9,
              45.3
            ],
            [
              36.5,
              45.3
            ],
            [
              36.5,
              44.4
            ],
            [
              33.9,
              44.4
            ],
            [
              33.9,
              45.3
            ]
          ]
        ]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Donetsk",
        "popupContent": "Occupied Donetsk"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              37.5,
              48.5
            ],
            [
              39.5,
              48.5
            ],
            [
              39.5,
              47.5
            ],
            [
              37.5,
              47.5
            ],
            [
              37.5,
              48.5
            ]
          ]
        ]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Luhansk",
        "popupContent": "Occupied Luhansk"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              38.5,
              49.5
            ],
            [
              40.5,
              49.5
            ],
            [
              40.5,
              48.5
            ],
            [
              38.5,
              48.5
            ],
            [
              38.5,
              49.5
            ]
          ]
        ]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Kherson",
        "popupContent": "Occupied Kherson"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              32.3,
              47.3
            ],
            [
              34.3,
              47.3
            ],
            [
              34.3,
              46.3
            ],
            [
              32.3,
              46.3
            ],
            [
              32.3,
              47.3
            ]
          ]
        ]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Zaporizhzhia",
        "popupContent": "Occupied Zaporizhzhia"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              34.3,
              48
            ],
            [
              36.3,
              48
            ],
            [
              36.3,
              47
            ],
            [
              34.3,
              47
            ],
            [
              34.3,
              48
            ]
          ]
        ]
      }
    }
  ]
}
```
````

The rendered map below highlights the regions of Ukraine that have been illegally occupied by Russia over the years, including Crimea and the four eastern regions:

```geojson
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "name": "Crimea",
        "popupContent": "Occupied Crimea"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              33.9,
              45.3
            ],
            [
              36.5,
              45.3
            ],
            [
              36.5,
              44.4
            ],
            [
              33.9,
              44.4
            ],
            [
              33.9,
              45.3
            ]
          ]
        ]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Donetsk",
        "popupContent": "Occupied Donetsk"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              37.5,
              48.5
            ],
            [
              39.5,
              48.5
            ],
            [
              39.5,
              47.5
            ],
            [
              37.5,
              47.5
            ],
            [
              37.5,
              48.5
            ]
          ]
        ]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Luhansk",
        "popupContent": "Occupied Luhansk"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              38.5,
              49.5
            ],
            [
              40.5,
              49.5
            ],
            [
              40.5,
              48.5
            ],
            [
              38.5,
              48.5
            ],
            [
              38.5,
              49.5
            ]
          ]
        ]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Kherson",
        "popupContent": "Occupied Kherson"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              32.3,
              47.3
            ],
            [
              34.3,
              47.3
            ],
            [
              34.3,
              46.3
            ],
            [
              32.3,
              46.3
            ],
            [
              32.3,
              47.3
            ]
          ]
        ]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Zaporizhzhia",
        "popupContent": "Occupied Zaporizhzhia"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              34.3,
              48
            ],
            [
              36.3,
              48
            ],
            [
              36.3,
              47
            ],
            [
              34.3,
              47
            ],
            [
              34.3,
              48
            ]
          ]
        ]
      }
    }
  ]
}
```

---

## Chartjs, Echarts and Vega-Lite

[Chart.js](https://www.chartjs.org/) is a versatile JavaScript library for creating responsive and interactive charts. Supporting multiple chart types like bar, line, pie, and radar, it’s an ideal tool for visualizing data directly in webpages.

Here’s an example of a JSON-style configuration that creates a bar chart in Chart.js:

````
```chartjs
{
  "type": "bar",
  "data": {
    "labels": ["2017", "2018", "2019", "2020", "2021"],
    "datasets": [
      {
        "label": "Population (millions)",
        "data": [12, 15, 13, 14, 16],
        "backgroundColor": "rgba(54, 162, 235, 0.6)",
        "borderColor": "rgba(54, 162, 235, 1)",
        "borderWidth": 1
      }
    ]
  },
  "options": {
    "scales": {
      "y": {
        "beginAtZero": true
      }
    }
  }
}
```
````

The rendered bar chart illustrates population data from 2017 to 2021:

```chartjs
{
  "type": "bar",
  "data": {
    "labels": ["2017", "2018", "2019", "2020", "2021"],
    "datasets": [
      {
        "label": "Population (millions)",
        "data": [12, 15, 13, 14, 16],
        "backgroundColor": "rgba(54, 162, 235, 0.6)",
        "borderColor": "rgba(54, 162, 235, 1)",
        "borderWidth": 1
      }
    ]
  },
  "options": {
    "scales": {
      "y": {
        "beginAtZero": true
      }
    }
  }
}
```

---

[ECharts](https://echarts.apache.org/) is a powerful visualization library from [Apache](https://www.apache.org/) that supports a wide range of interactive charts, including more advanced types such as scatter plots, heatmaps, and geographic maps.

The following JSON configuration creates a visually enhanced line chart that displays monthly sales data for two products.

````
```echarts
{
  "title": {
    "text": "Monthly Sales Comparison",
    "left": "center"
  },
  "tooltip": {
    "trigger": "axis",
    "backgroundColor": "rgba(50, 50, 50, 0.7)",
    "borderColor": "#777",
    "borderWidth": 1,
    "textStyle": {
      "color": "#fff"
    }
  },
  "legend": {
    "data": ["Product A", "Product B"],
    "top": "10%"
  },
  "xAxis": {
    "type": "category",
    "data": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "axisLine": {
      "lineStyle": {
        "color": "#888"
      }
    }
  },
  "yAxis": {
    "type": "value",
    "axisLine": {
      "lineStyle": {
        "color": "#888"
      }
    },
    "splitLine": {
      "lineStyle": {
        "type": "dashed"
      }
    }
  },
  "series": [
    {
      "name": "Product A",
      "type": "line",
      "smooth": true,
      "data": [820, 932, 901, 934, 1290, 1330, 1320, 1400, 1450, 1500, 1600, 1650],
      "itemStyle": {
        "color": "#5470C6"
      },
      "lineStyle": {
        "width": 3
      },
      "areaStyle": {
        "color": {
          "type": "linear",
          "x": 0,
          "y": 0,
          "x2": 0,
          "y2": 1,
          "colorStops": [
            { "offset": 0, "color": "rgba(84, 112, 198, 0.5)" },
            { "offset": 1, "color": "rgba(84, 112, 198, 0)" }
          ]
        }
      },
      "emphasis": {
        "focus": "series"
      }
    },
    {
      "name": "Product B",
      "type": "line",
      "smooth": true,
      "data": [620, 732, 701, 734, 1090, 1130, 1120, 1200, 1250, 1300, 1400, 1450],
      "itemStyle": {
        "color": "#91CC75"
      },
      "lineStyle": {
        "width": 3
      },
      "areaStyle": {
        "color": {
          "type": "linear",
          "x": 0,
          "y": 0,
          "x2": 0,
          "y2": 1,
          "colorStops": [
            { "offset": 0, "color": "rgba(145, 204, 117, 0.5)" },
            { "offset": 1, "color": "rgba(145, 204, 117, 0)" }
          ]
        }
      },
      "emphasis": {
        "focus": "series"
      }
    }
  ]
}
```
````

The rendered output is shown below, and you can also interact with it using your mouse:

```echarts
{
  "title": {
    "text": "Monthly Sales Comparison",
    "left": "center"
  },
  "tooltip": {
    "trigger": "axis",
    "backgroundColor": "rgba(50, 50, 50, 0.7)",
    "borderColor": "#777",
    "borderWidth": 1,
    "textStyle": {
      "color": "#fff"
    }
  },
  "legend": {
    "data": ["Product A", "Product B"],
    "top": "10%"
  },
  "xAxis": {
    "type": "category",
    "data": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "axisLine": {
      "lineStyle": {
        "color": "#888"
      }
    }
  },
  "yAxis": {
    "type": "value",
    "axisLine": {
      "lineStyle": {
        "color": "#888"
      }
    },
    "splitLine": {
      "lineStyle": {
        "type": "dashed"
      }
    }
  },
  "series": [
    {
      "name": "Product A",
      "type": "line",
      "smooth": true,
      "data": [820, 932, 901, 934, 1290, 1330, 1320, 1400, 1450, 1500, 1600, 1650],
      "itemStyle": {
        "color": "#5470C6"
      },
      "lineStyle": {
        "width": 3
      },
      "areaStyle": {
        "color": {
          "type": "linear",
          "x": 0,
          "y": 0,
          "x2": 0,
          "y2": 1,
          "colorStops": [
            { "offset": 0, "color": "rgba(84, 112, 198, 0.5)" },
            { "offset": 1, "color": "rgba(84, 112, 198, 0)" }
          ]
        }
      },
      "emphasis": {
        "focus": "series"
      }
    },
    {
      "name": "Product B",
      "type": "line",
      "smooth": true,
      "data": [620, 732, 701, 734, 1090, 1130, 1120, 1200, 1250, 1300, 1400, 1450],
      "itemStyle": {
        "color": "#91CC75"
      },
      "lineStyle": {
        "width": 3
      },
      "areaStyle": {
        "color": {
          "type": "linear",
          "x": 0,
          "y": 0,
          "x2": 0,
          "y2": 1,
          "colorStops": [
            { "offset": 0, "color": "rgba(145, 204, 117, 0.5)" },
            { "offset": 1, "color": "rgba(145, 204, 117, 0)" }
          ]
        }
      },
      "emphasis": {
        "focus": "series"
      }
    }
  ]
}
```

---

[Vega-Lite](https://vega.github.io/vega-lite/) is a declarative visualization grammar that allows users to create, share, and customize a wide range of interactive data visualizations. The following JSON configuration generates a straightforward bar chart:

````
```vega_lite
{
  "$schema": "https://vega.github.io/schema/vega/v5.json",
  "width": 400,
  "height": 200,
  "padding": 5,

  "data": [
    {
      "name": "table",
      "values": [
        {"category": "A", "value": 28},
        {"category": "B", "value": 55},
        {"category": "C", "value": 43},
        {"category": "D", "value": 91},
        {"category": "E", "value": 81},
        {"category": "F", "value": 53},
        {"category": "G", "value": 19},
        {"category": "H", "value": 87}
      ]
    }
  ],

  "scales": [
    {
      "name": "xscale",
      "type": "band",
      "domain": {"data": "table", "field": "category"},
      "range": "width",
      "padding": 0.1
    },
    {
      "name": "yscale",
      "type": "linear",
      "domain": {"data": "table", "field": "value"},
      "nice": true,
      "range": "height"
    }
  ],

  "axes": [
    {"orient": "bottom", "scale": "xscale"},
    {"orient": "left", "scale": "yscale"}
  ],

  "marks": [
    {
      "type": "rect",
      "from": {"data": "table"},
      "encode": {
        "enter": {
          "x": {"scale": "xscale", "field": "category"},
          "width": {"scale": "xscale", "band": 0.8},
          "y": {"scale": "yscale", "field": "value"},
          "y2": {"scale": "yscale", "value": 0},
          "fill": {"value": "steelblue"}
        },
        "update": {
          "fillOpacity": {"value": 1}
        },
        "hover": {
          "fill": {"value": "orange"}
        }
      }
    }
  ]
}
```
````

The rendered output shows a clean and simple bar chart with a hover effect：

```vega_lite
{
  "$schema": "https://vega.github.io/schema/vega/v5.json",
  "width": 400,
  "height": 200,
  "padding": 5,

  "data": [
    {
      "name": "table",
      "values": [
        {"category": "A", "value": 28},
        {"category": "B", "value": 55},
        {"category": "C", "value": 43},
        {"category": "D", "value": 91},
        {"category": "E", "value": 81},
        {"category": "F", "value": 53},
        {"category": "G", "value": 19},
        {"category": "H", "value": 87}
      ]
    }
  ],

  "scales": [
    {
      "name": "xscale",
      "type": "band",
      "domain": {"data": "table", "field": "category"},
      "range": "width",
      "padding": 0.1
    },
    {
      "name": "yscale",
      "type": "linear",
      "domain": {"data": "table", "field": "value"},
      "nice": true,
      "range": "height"
    }
  ],

  "axes": [
    {"orient": "bottom", "scale": "xscale"},
    {"orient": "left", "scale": "yscale"}
  ],

  "marks": [
    {
      "type": "rect",
      "from": {"data": "table"},
      "encode": {
        "enter": {
          "x": {"scale": "xscale", "field": "category"},
          "width": {"scale": "xscale", "band": 0.8},
          "y": {"scale": "yscale", "field": "value"},
          "y2": {"scale": "yscale", "value": 0},
          "fill": {"value": "steelblue"}
        },
        "update": {
          "fillOpacity": {"value": 1}
        },
        "hover": {
          "fill": {"value": "orange"}
        }
      }
    }
  ]
}
```

---

## TikZ

[TikZ](https://tikz.net/) is a powerful LaTeX-based drawing tool powered by [TikZJax](https://tikzjax.com/). You can easily port TikZ drawings from papers, posters, and notes. For example, we can use the following code to illustrate Euler’s formula $ e^{i \theta} = \cos \theta + i \sin \theta $:

```markdown
<script type="text/tikz">
\begin{tikzpicture}
    \filldraw[fill=cyan!10, draw=blue, thick] (0,0) circle (2cm);

    \draw[->, thick] (-2.5,0) -- (2.5,0) node[right] {Re};
    \draw[->, thick] (0,-2.5) -- (0,2.5) node[above] {Im};

    \draw[->, thick, color=purple] (0,0) -- (1.5,1.5);
    \node[color=purple] at (1.1, 1.7) {$e^{i\theta}$};

    \draw[thick] (0.7,0) arc (0:45:0.7);
    \node at (0.9, 0.3) {$\theta$};

    \draw[dashed, color=gray] (1.5,1.5) -- (1.5,0) node[below, black] {$\cos \theta$};
    \draw[dashed, color=gray] (1.5,1.5) -- (0,1.5) node[left, black] {$\sin \theta$};
    \node at (2.2, 0) [below] {1};
    \node at (0, 2.2) [left] {$i$};
    \node at (1.5,1.5) [above right, color=blue] {$(\cos \theta \, \sin \theta)$};
\end{tikzpicture}
</script>
```

The rendered output is shown below, displayed as a vector graphic：

<script type="text/tikz">
\begin{tikzpicture}
    \filldraw[fill=cyan!10, draw=blue, thick] (0,0) circle (2cm);

    \draw[->, thick] (-2.5,0) -- (2.5,0) node[right] {Re};
    \draw[->, thick] (0,-2.5) -- (0,2.5) node[above] {Im};

    \draw[->, thick, color=purple] (0,0) -- (1.5,1.5);
    \node[color=purple] at (1.1, 1.7) {$e^{i\theta}$};

    \draw[thick] (0.7,0) arc (0:45:0.7);
    \node at (0.9, 0.3) {$\theta$};

    \draw[dashed, color=gray] (1.5,1.5) -- (1.5,0) node[below, black] {$\cos \theta$};
    \draw[dashed, color=gray] (1.5,1.5) -- (0,1.5) node[left, black] {$\sin \theta$};
    \node at (2.2, 0) [below] {1};
    \node at (0, 2.2) [left] {$i$};
    \node at (1.5,1.5) [above right, color=blue] {$(\cos \theta \, \sin \theta)$};
\end{tikzpicture}
</script>

---

## Typograms

[Typograms](https://google.github.io/typograms/) are a way of combining text and graphics to convey information in a clear and visually engaging manner. Typograms are particularly effective for illustrating simple diagrams, charts, and concept visuals where text and graphics are closely integrated. The following example demonstrates a simple Typogram:

````
```typograms
             ___________________
            /                  /|
           /__________________/ |
          |                  |  |
          |     Distill      |  |
          |                  |  |
          |                  | /
          |__________________|/
```
````

The rendered output is shown below：

```typograms
             ___________________
            /                  /|
           /__________________/ |
          |                  |  |
          |     Distill      |  |
          |                  |  |
          |                  | /
          |__________________|/
```

---

## Layouts

The main text column is referred to as the body.
It is the assumed layout of any direct descendants of the `d-article` element.

<div class="fake-img l-body">
  <p>.l-body</p>
</div>

For images you want to display a little larger, try `.l-page`:

<div class="fake-img l-page">
  <p>.l-page</p>
</div>

All of these have an outset variant if you want to poke out from the body text a little bit.
For instance:

<div class="fake-img l-body-outset">
  <p>.l-body-outset</p>
</div>

<div class="fake-img l-page-outset">
  <p>.l-page-outset</p>
</div>

Occasionally you’ll want to use the full browser width.
For this, use `.l-screen`.
You can also inset the element a little from the edge of the browser by using the inset variant.

<div class="fake-img l-screen">
  <p>.l-screen</p>
</div>
<div class="fake-img l-screen-inset">
  <p>.l-screen-inset</p>
</div>

The final layout is for marginalia, asides, and footnotes.
It does not interrupt the normal flow of `.l-body` sized text except on mobile screen sizes.

<div class="fake-img l-gutter">
  <p>.l-gutter</p>
</div>

---

## Sidenotes

Distill supports sidenotes, which are like footnotes but placed in the margin of the page.
They are useful for providing additional context or references without interrupting the flow of the main text.

There are two main ways to create a sidenote:

**Using the `<aside>` tag:**

The following code creates a sidenote with **_distill's styling_** in the margin:

```html
<aside><p>This is a sidenote using aside tag.</p></aside>
```

<aside><p> This is a sidenote using `&lt;aside&gt;` tag</p> </aside>

We can also add images to sidenotes (click on the image to zoom in for a larger version):
{% raw %}

```html
<aside>
  {% include figure.liquid loading="eager" path="assets/img/rhino.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  <p>
    F.J. Cole, “The History of Albrecht Dürer’s Rhinoceros in Zoological Literature,” Science, Medicine, and History: Essays on the Evolution of
    Scientific Thought and Medical Practice (London, 1953), ed. E. Ashworth Underwood, 337-356. From page 71 of Edward Tufte’s Visual Explanations.
  </p>
</aside>
```

{% endraw %}

<aside>
  {% include figure.liquid loading="eager" path="assets/img/rhino.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  <p>F.J. Cole, “The History of Albrecht Dürer’s Rhinoceros in Zoological Literature,” Science, Medicine, and History: Essays on the Evolution of Scientific Thought and Medical Practice (London, 1953), ed. E. Ashworth Underwood, 337-356. From page 71 of Edward Tufte’s Visual Explanations.</p>
</aside>

Sidenotes can also contain equations and links:

In physics, mass–energy equivalence is the relationship between mass and energy in a system's rest frame. The two differ only by a multiplicative constant and the units of measurement.

<aside>
  <p>This principle is defined by Einstein's famous equation: $E = mc^2$ <a href="https://en.wikipedia.org/wiki/Mass%E2%80%93energy_equivalence" target="_blank">(Source: Wikipedia)</a></p>
</aside>

**Using the `l-gutter` class:**

The following code creates a sidenote with **_al-folio's styling_** in the margin:

```html
<div class="l-gutter"><p>This is a sidenote using l-gutter class.</p></div>
```

<div class="l-gutter">
  <p> This is a sidenote using `l-gutter` class. </p>
</div>

---

## Other Typography?

Emphasis, aka italics, with _asterisks_ (`*asterisks*`) or _underscores_ (`_underscores_`).

Strong emphasis, aka bold, with **asterisks** or **underscores**.

Combined emphasis with **asterisks and _underscores_**.

Strikethrough uses two tildes. ~~Scratch this.~~

1. First ordered list item
2. Another item
   ⋅⋅\* Unordered sub-list.
3. Actual numbers don't matter, just that it's a number
   ⋅⋅1. Ordered sub-list
4. And another item.

⋅⋅⋅You can have properly indented paragraphs within list items. Notice the blank line above, and the leading spaces (at least one, but we'll use three here to also align the raw Markdown).

⋅⋅⋅To have a line break without a paragraph, you will need to use two trailing spaces.⋅⋅
⋅⋅⋅Note that this line is separate, but within the same paragraph.⋅⋅
⋅⋅⋅(This is contrary to the typical GFM line break behaviour, where trailing spaces are not required.)

- Unordered list can use asterisks

* Or minuses

- Or pluses

[I'm an inline-style link](https://www.google.com)

[I'm an inline-style link with title](https://www.google.com "Google's Homepage")

[I'm a reference-style link][Arbitrary case-insensitive reference text]

[You can use numbers for reference-style link definitions][1]

Or leave it empty and use the [link text itself].

URLs and URLs in angle brackets will automatically get turned into links.
http://www.example.com or <http://www.example.com> and sometimes
example.com (but not on Github, for example).

Some text to show that the reference links can follow later.

[arbitrary case-insensitive reference text]: https://www.mozilla.org
[1]: http://slashdot.org
[link text itself]: http://www.reddit.com

Here's our logo (hover to see the title text):

Inline-style:
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

Reference-style:
![alt text][logo]

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2"

Inline `code` has `back-ticks around` it.

```javascript
var s = "JavaScript syntax highlighting";
alert(s);
```

```python
s = "Python syntax highlighting"
print s
```

```
No language indicated, so no syntax highlighting.
But let's throw in a <b>tag</b>.
```

Colons can be used to align columns.

| Tables        |      Are      |  Cool |
| ------------- | :-----------: | ----: |
| col 3 is      | right-aligned | $1600 |
| col 2 is      |   centered    |   $12 |
| zebra stripes |   are neat    |    $1 |

There must be at least 3 dashes separating each header cell.
The outer pipes (|) are optional, and you don't need to make the
raw Markdown line up prettily. You can also use inline Markdown.

| Markdown | Less      | Pretty     |
| -------- | --------- | ---------- |
| _Still_  | `renders` | **nicely** |
| 1        | 2         | 3          |

> Blockquotes are very handy in email to emulate reply text.
> This line is part of the same quote.

Quote break.

> This is a very long line that will still be quoted properly when it wraps. Oh boy let's keep writing to make sure this is long enough to actually wrap for everyone. Oh, you can _put_ **Markdown** into a blockquote.

Here's a line for us to start with.

This line is separated from the one above by two newlines, so it will be a _separate paragraph_.

This line is also a separate paragraph, but...
This line is only separated by a single newline, so it's a separate line in the _same paragraph_.
