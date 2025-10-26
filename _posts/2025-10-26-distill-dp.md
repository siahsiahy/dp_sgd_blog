---
layout: post
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
    url: "https://siahsiahy.github.io/"
affiliations:
  - name: Yonsei University
    url: https://www.yonsei.ac.kr/en_sc/


# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: "Why Whisper?"
  - name: "What Does DP–SGD Actually Do?"
  - name: "Algorithm: Making SGD Private"
  - name: "Implementing DP–SGD: Where Theory Meets TensorFlow"
  - name: "Does Privacy Hurt Accuracy? (Results)"
  - name: "Who Else Is Thinking About Privacy?"
  - name: "Conclusions"
  - name: "References"

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

### Applying the Moments Accountant

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

### MNIST Results: Small Noise, Big Insights

We trained models on MNIST using different noise levels while keeping other hyperparameters fixed. Below are training and testing accuracy plots for three noise scales:

{{< figure src="assets/img/result1.jpg" title="Figure 3: Accuracy over epochs for different noise levels on MNIST (Large, Medium, Small). Each model uses 60D PCA, 1000 hidden units, and clip threshold 4." >}}

Key takeaways:
- **Small noise** gives the best accuracy — almost non-private level!
- As noise increases, accuracy drops slightly, but not catastrophically.
- The model still generalizes well under moderate noise.

---

### Privacy-Accuracy Tradeoff

To analyze how privacy parameters directly affect accuracy, we varied ε (epsilon) while fixing δ (delta). Each line in the plot below represents a different (ε, δ) pair:

{{< figure src="assets/img/result2.jpg" title="Figure 4: Accuracy vs epsilon on MNIST for various delta values." >}}

What’s happening here:
- Smaller epsilon = stronger privacy = slightly lower accuracy.
- But even with ε=1 and δ=1e-5, the drop is tolerable.
- There’s a clear "diminishing returns" zone — cranking up ε too high doesn’t boost accuracy much more.

---

### What Happens When We Tweak One Hyperparameter?

To see how robust the model is, we varied each parameter individually — like projection dimension, hidden units, lot size, etc. This gives a sense of which knobs matter most.

{{< figure src="assets/img/result3.jpg" title="Figure 5: MNIST accuracy when one parameter varies and others stay at reference values." >}}

- **Projection Dimension**: Accuracy is best around 60 dimensions (PCA). Too low or no PCA hurts generalization.
- **Hidden Units**: More units help until about 1000–1200; after that, gains plateau.
- **Lot Size**: There’s a sweet spot near 600.
- **Learning Rate**: 0.05 seems optimal — too high and things crash.
- **Clipping Norm**: Between 4–6 is stable.
- **Noise Level**: Moderate noise still yields strong results!

---

### CIFAR-10 Results: What About Real Images?

MNIST is nice, but what about a more complex dataset like CIFAR-10? We use a 2-layer convolutional network and vary the noise level (ε = 2, 4, 8). Here's what we get:

{{< figure src="assets/img/result4.jpg" title="Figure 6: CIFAR-10 accuracy under different noise levels with lot size = 2000 or 4000." >}}

- With increasing ε (weaker privacy), accuracy improves: from **67% → 70% → 73%**.
- Compared to state-of-the-art non-private models (~86%), we lose performance — but it’s still **remarkably good** for a DP-trained model!

---

### Does Privacy Kill Accuracy?

I don't think it's true.

- **Smart tuning** (like PCA, optimal clip norm, lot size) helps a lot.
- With **moderate noise**, you can get surprisingly strong performance.
- Even on real-world datasets like CIFAR-10, the tradeoff is manageable.

If you want both privacy AND performance, **you don’t have to choose**

---
## Who Else Is Thinking About Privacy?
Let’s be real! privacy in machine learning isn’t exactly new. Since the late 90s, researchers have been scratching their heads about how to learn useful stuff from data without leaking personal info. But the ways people have tackled this vary a lot.
Depending on what kind of model they’re using, how the learning is done, and how privacy is guaranteed.

### Privacy Guarantees

Early privacy-preserving methods used Secure Function Evaluation (SFE) and Multi-Party Computation (MPC). These split the data between parties and compute the result together while keeping everyone’s input secret. Cool, but not really applicable here—we assume all data is held centrally, and we care about what leaks from the trained model.

Another classic idea is k-anonymity, where data is generalized or suppressed (like saying someone was “born in the 90s” instead of giving their birthday). Unfortunately, this breaks down when dealing with high-dimensional data like images.

Instead, most recent work—including ours—leans on Differential Privacy (DP). The idea is to inject carefully calibrated noise into the training process so that it's hard to tell whether any particular person’s data was used. Techniques like the moments accountant and Rényi DP help track how much "privacy budget" we're spending during training.

### Learning Algorithms and Training Techniques

Most DP work tries to solve convex optimization problems—think logistic regression or SVMs—where the math is easier and DP guarantees are cleaner. But training deep neural networks is non-convex and trickier. This paper pushes forward by applying DP-SGD (stochastic gradient descent with differential privacy) to deep models, even if the theory is messier.

### Model Classes

Other researchers have also tackled DP learning:

**Netflix Prize (McSherry & Mironov):** The first full-stack DP recommender system using Gaussian noise on sufficient statistics. But their problem had cleaner structure than deep learning.
**Distributed DP (Shokri & Shmatikov):** Each user trains on their own data and sends noisy gradients to a central server. But privacy loss per parameter can easily exceed thousands—yikes.
**Autoencoders (Phan et al.):** Recent work adds DP constraints to the loss functions of autoencoders.

Our approach differs by applying end-to-end DP training using standard neural networks and SGD, without assuming any special structure in the data or model.

---

## Conclusions

We trained deep neural networks *with strong privacy guarantees*—and guess what? The performance is still pretty solid.

- On **MNIST**, our model hit **97%** accuracy.
- On **CIFAR-10**, we got **73%**—not bad at all considering we’re adding noise!

All of this while ensuring **(ε = 8, δ = 10⁻⁵)** differential privacy. 🎉
 We’re using a private version of **stochastic gradient descent (SGD)** that plugs right into **TensorFlow**. So it’s easy to work with and pretty versatile.

### Keypoint
A big part of what made this work is something called the Moments Accountant.
Think of it as a super-smart privacy tracker that knows exactly how much of your privacy budget is being used during training. It’s way more accurate than traditional methods, and it gives us tight control over how noisy we need to be.

> **Privacy doesn’t have to come at the cost of usefulness.**
> With the right tools (👋 hello, moments accountant) and some clever training tricks, we can get the best of both worlds: solid performance and user privacy.

---

## References

<details>
<summary>Show References</summary>

1. Abadi et al., *Deep Learning with Differential Privacy* [https://arxiv.org/abs/1607.00133](https://arxiv.org/abs/1607.00133)  
2. TensorFlow Privacy: [https://github.com/tensorflow/privacy](https://github.com/tensorflow/privacy)  
3. McSherry, F. & Mironov, I. *Differentially Private Recommender Systems: Building Privacy into the Netflix Prize Contenders.* In *Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD’09)*, pp. 627–636. ACM, 2009.  
4. Dwork, C. & Roth, A. *The Algorithmic Foundations of Differential Privacy.* *Foundations and Trends in Theoretical Computer Science.* Now Publishers, 2014.  

</details>

---
