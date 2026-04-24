# Big-picture overview

  

## What are INRs?

  

An *implicit neural representation* (INR) represents a single signal as

a neural function $$f_{\theta} : \mathcal{X} \to \mathcal{Y},$$ where

$\mathcal{X}$ is usually a coordinate domain (for example, pixel

coordinates $(x,y)$, 3D coordinates $(x,y,z)$, time, or space-time), and

$\mathcal{Y}$ is the signal value at that coordinate (for example,

grayscale intensity, RGB, signed distance, occupancy, or physical field

values) . Instead of storing the signal as an array, one stores

parameters $\theta$ of a neural network that can be evaluated

continuously at arbitrary coordinates.

  

A typical INR fitting objective is

$$\theta^*(s) = \arg\min_{\theta}\; \frac{1}{|\mathcal{I}|}\sum_{i\in\mathcal{I}} \ell\big(f_{\theta}(x_i),y_i\big),$$

where $\{(x_i,y_i)\}$ are sampled coordinate-value pairs from signal $s$

. The fitted weights $\theta^*(s)$ or a lower-dimensional learned

representation derived from them become the representation of the

signal.

  

## Why use SIREN?

  

SIREN (*Sinusoidal Representation Networks*) uses sine activations in

place of standard ReLU or tanh nonlinearities . A SIREN layer has the

form $$\phi_i(h_i)=\sin(W_i h_i + b_i),$$ and the overall network is a

composition of such layers. The motivation is that periodic activations

make coordinate MLPs much better at representing high-frequency detail

and spatial derivatives. This is especially valuable for images, audio,

signed distance fields, and physics problems where derivative

information matters.

  

Two points from SIREN are central for this thesis:

  

1. The activation is smooth and bounded, which makes it attractive from

a stability perspective.

  

2. The linear maps $W_i$ can still have large operator norms and

therefore can amplify perturbations substantially unless controlled.

  

## What are parameter-space classifiers?

  

A *parameter-space classifier* first maps an input signal $x$ into an

INR representation and then classifies in that representation space:

$$x \xrightarrow{\text{fit INR}} z^*(x) \xrightarrow{g(\cdot)} \hat{y}.$$

Here $z^*(x)$ may be:

  

- the full INR weights,

  

- a compact code that modulates a shared INR backbone,

  

- or another learned representation of the fitted model.

  

The *Functa* framework is the main example in this literature: it treats

each datapoint as a function and uses low-dimensional *modulations* of a

shared INR backbone as the representation on which downstream learning

is performed .

  

## Why might they seem robust?

  

At first glance, these systems can look robust to standard white-box

attacks because the adversary is no longer attacking a simple

feedforward classifier $x\mapsto h(x)$. Instead, inference includes an

*inner optimization loop* that fits the INR or its modulations. That

loop can make gradients hard to compute, noisy, truncated, or

misleading. In addition, the INR fitting process can act as a kind of

*spectral filter*: if it preferentially fits global or low-frequency

structure, then some high-frequency adversarial perturbations may be

attenuated before the downstream classifier sees them .

  

## Why is that robustness not necessarily real?

  

This is the central caution of the project. In adversarial ML, apparent

robustness often comes from *gradient masking* or *obfuscated

gradients*, not from true stability . The recent paper *Adversarial

Attacks in Weight-Space Classifiers* argues exactly this for

parameter-space classifiers: standard gradient attacks underestimate

vulnerability because they do not fully handle the inner optimization,

while stronger adaptive attacks reduce the apparent robustness

substantially . Therefore, robustness in this setting must be evaluated

against the *full composed system*, including the fitting loop.

  

## Why are Lipschitz constraints relevant?

  

If the goal is to move from accidental robustness to structural

robustness, then one natural route is to control sensitivity explicitly.

For a feedforward map with 1-Lipschitz activations, a standard upper

bound on the Lipschitz constant is

$$\mathrm{Lip}(f) \le \prod_{i=1}^{L}\|W_i\|_2,$$ where $\|W_i\|_2$ is

the spectral norm of layer $i$. Since sine is bounded and smooth, the

uncontrolled amplification in SIREN arises mainly from the linear layers

and from explicit frequency scaling such as $\omega_0$ . This makes

spectral constraints, orthogonality constraints, Jacobian control, and

related methods natural candidates.

  

## Why might Fourier features matter for scalability?

  

A major limitation of vanilla coordinate MLPs is spectral bias: they

learn low-frequency structure more easily than high-frequency structure.

Fourier feature mappings change the effective input basis so that

high-frequency functions become easier to learn . This is highly

relevant here because current parameter-space classifier demonstrations

are mostly on simple datasets such as MNIST, Fashion-MNIST, and

ModelNet10 . Richer encodings may be necessary for scaling to CIFAR-like

or ImageNet-like data, but they may also remove some of the incidental

“scrubbing” effect that currently contributes to apparent robustness.

  

# Paper-by-paper explanation

  

## Overview table

  

| Paper | Main contribution | Why it matters here | What to prioritize |

|:-------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|

| Paper | Main contribution | Why it matters here | What to prioritize |

| SIREN | Defines sinusoidal INR architecture, analyzes initialization, emphasizes representation of fine detail and derivatives | Gives the foundational INR model and the key architectural knobs that interact with stability | Architecture, initialization, role of $\omega_0$, derivative arguments |

| Functa | Formalizes “datapoints as functions” and trains downstream models on compact INR modulations | Blueprint for parameter-space classification and for the inner optimization bottleneck | Modulations, latent modulations, meta-learning procedure, downstream classification |

| SVCCA | Compares learned representations up to affine transformations | Not weight-space learning itself, but useful as a diagnostic tool for representation stability | Method definition and invariance ideas |

| Adversarial Attacks in Weight-Space Classifiers | Studies attacks on parameter-space classifiers and attributes much of the robustness to gradient obfuscation | Most directly aligned with the thesis problem | Threat model, attack suite, BPDA/implicit-diff logic, conclusions and limitations |

  

## Paper 1: SIREN — *Implicit Neural Representations with Periodic Activation Functions*

  

#### Core idea in plain language.

  

Replace standard activations by sine functions in a coordinate MLP so

that the network can represent detailed continuous signals and their

derivatives more effectively .

  

#### Main technical contributions.

  

The paper contributes:

  

1. a concrete INR architecture based on periodic activations,

  

2. an initialization scheme that keeps activations and gradients in a

usable regime,

  

3. the observation that derivatives of the representation remain well

behaved and can themselves be represented/supervised.

  

#### Key equations and concepts.

  

The core building block is $$\phi_i(h_i)=\sin(W_i h_i+b_i).$$ The

complete model is

$$\Phi(x)=W_n\big(\phi_{n-1}\circ\cdots\circ\phi_0\big)(x)+b_n.$$ A

practical detail that matters a lot is the first-layer and hidden-layer

frequency scaling parameter $\omega_0$, which directly affects gradient

magnitudes and the frequencies that can be represented .

  

#### What matters most for this thesis.

  

For this thesis, the most important ideas are:

  

- why sine activations help with high-frequency representation;

  

- how initialization is coupled to stable optimization;

  

- how $\omega_0$ affects sensitivity;

  

- why bounded periodic activations do *not* automatically imply

robustness if linear maps are unconstrained.

  

#### What can be skimmed.

  

Most modality-specific applications can be skimmed on a first pass. The

thesis-relevant content is in the architecture, motivation,

initialization analysis, and derivative discussion.

  

#### Connection to the thesis.

  

This paper provides the candidate INR family you want to regularize. It

gives the right intuition for why controlling linear layers may matter,

but it does *not* itself study robustness.

  

#### Limitations / unanswered questions.

  

The paper is about representation power and optimization, not

adversarial stability. It does not answer:

  

- whether SIRENs are robust under adaptive attacks,

  

- whether controlling layer norms preserves enough expressivity,

  

- or how SIREN behaves inside a bilevel classification pipeline.

  

## Paper 2: Functa — *Your Data Point Is a Function and You Can Treat It Like One*

  

#### Core idea in plain language.

  

Turn a dataset of arrays into a dataset of functions. Instead of

learning directly on pixels or voxels, fit each datapoint with an INR

and then perform downstream learning on compact representations of that

fitted function .

  

#### Main technical contributions.

  

The paper introduces:

  

1. *functa*: datapoints represented as continuous functions,

  

2. *modulations*: compact parameters that adapt a shared INR backbone

to each datapoint,

  

3. a meta-learning procedure so that per-sample fitting is fast,

  

4. downstream generative and discriminative learning directly on

modulation space.

  

#### Key equations and concepts.

  

At the base level, each signal is fitted by minimizing reconstruction

loss over sampled coordinates. But instead of storing the full fitted

network, the paper learns a shared backbone and only stores

low-dimensional modulation parameters. In practice, the paper reports

that *shift-only* modulations often perform similarly to shift+scale

modulations at lower dimensional cost .

  

#### What matters most for this thesis.

  

The most important parts are:

  

- the modulation representation,

  

- latent modulation compression,

  

- the meta-learning setup that reduces inner-loop fitting to a few

steps,

  

- the idea of downstream classification directly in modulation space.

  

#### What can be skimmed.

  

The modality-specific application details can be skimmed if the goal is

only to understand parameter-space classification and inner-loop

fitting.

  

#### Connection to the thesis.

  

This is the foundational design pattern for the thesis. It provides the

exact setting in which robustness becomes nontrivial:

$$x \mapsto z^*(x) \mapsto g(z^*(x)).$$ The inner optimization that

makes this framework practical is also what creates the gradient

bottleneck that the attack paper later exploits.

  

#### Limitations / unanswered questions.

  

Important limitations for your project are:

  

- the representation and classification quality does not obviously

outperform conventional signal-domain deep learning on realistic

datasets;

  

- robustness is not a design goal of the paper;

  

- the inner optimization creates a difficult evaluation problem for

adversarial attacks.

  

## Paper 3: arXiv:1706.05806 — actually SVCCA, not HyperNetworks

  

#### Important correction.

  

The arXiv identifier `1706.05806` corresponds to *SVCCA: Singular Vector

Canonical Correlation Analysis for Deep Learning Dynamics and

Interpretability*, not to HyperNetworks . If the goal was background on

learning in parameter space, then *HyperNetworks* is likely the intended

paper.

  

#### Core idea in plain language.

  

SVCCA compares learned neural representations in a way that is invariant

to affine transformations. It is not itself a weight-space classifier,

but it is useful for studying how similar two representations are across

training, architectures, or perturbations .

  

#### Why it may still be useful for this thesis.

  

Even though it is not the core background paper for parameter-space

learning, it can still help as a diagnostic tool:

  

- compare clean and adversarial representations,

  

- compare modulation spaces across training runs,

  

- study whether a proposed robustness method stabilizes the learned

parameter-space embedding.

  

#### If HyperNetworks was the intended background.

  

*HyperNetworks* introduces the idea of using one network to generate the

weights of another network . This is highly relevant because many

INR-related systems either generate weights directly or amortize the

fitting process using hypernetwork-like machinery. It is a cleaner

conceptual background paper if your goal is to understand what is

special about learning directly in weight space.

  

#### What to focus on.

  

If you keep SVCCA in your reading list, focus on the method and

invariance viewpoint. If you instead switch to HyperNetworks, focus on

the genotype/phenotype idea and how it enables efficient parameter

generation.

  

#### Connection to the thesis.

  

SVCCA helps as an analysis tool; HyperNetworks helps as a conceptual

background paper. The thesis itself is more directly related to

HyperNetworks than to SVCCA.

  

## Paper 4: *Adversarial Attacks in Weight-Space Classifiers*

  

#### Core idea in plain language.

  

Study adversarial attacks on classifiers that operate on INR-derived

parameter-space representations, and show that much of their apparent

robustness to standard white-box attacks comes from gradient obfuscation

caused by the INR fitting loop rather than from true principled

robustness .

  

#### Main technical contributions.

  

This paper is the most important one for the thesis because it:

  

1. formalizes the threat model for parameter-space classifiers,

  

2. develops attack strategies that account for the inner optimization,

  

3. compares naive and stronger attack settings,

  

4. argues that the apparent robustness is largely not structural.

  

#### Key concepts to understand very carefully.

  

- The adversary perturbs the signal in input space, but success is

measured after INR fitting and downstream classification.

  

- The pipeline is bilevel and expensive to differentiate through.

  

- Standard white-box attacks can fail if they do not handle the fitting

loop correctly.

  

- Stronger attacks, including optimization-aware and gradient-free

methods, reveal that the robustness is limited.

  

#### What matters most for your thesis.

  

You should read this paper particularly carefully for:

  

- the threat model,

  

- the exact attack definitions,

  

- where gradient obfuscation comes from,

  

- how BPDA-like logic and alternative attacks are used,

  

- which conclusions remain strong after adaptive evaluation.

  

#### What can be skimmed.

  

Once the attack taxonomy is clear, some implementation details and

qualitative visualizations can be skimmed on the first pass.

  

#### Connection to the thesis.

  

This paper essentially defines the problem statement of your thesis. It

tells you what *not* to do: do not claim robustness based only on naive

PGD or standard white-box attacks that ignore the bilevel structure.

  

#### Limitations / unanswered questions.

  

The paper is diagnostic, not constructive. It identifies the fragility

of current robustness claims but leaves open:

  

- how to achieve genuine representation stability,

  

- whether constraining the INR can help,

  

- and how robustness behaves when richer INR representations are used.

  

# Mathematical foundations to review

  

## Operator norms and spectral norm

  

For a linear map $W:\mathbb{R}^n\to\mathbb{R}^m$, the operator norm

induced by the $\ell_2$ norm is

$$\|W\|_2=\max_{\|x\|_2=1}\|Wx\|_2=\sigma_{\max}(W),$$ where

$\sigma_{\max}(W)$ is the largest singular value . This is the tightest

bound on how much $W$ can amplify Euclidean perturbations:

$$\|W\delta\|_2 \le \|W\|_2 \|\delta\|_2.$$ This is the basic

linear-algebra fact behind spectral normalization and Parseval

constraints.

  

## Singular values and condition number

  

The singular values of $W$ are the square roots of the eigenvalues of

$W^\top W$. The condition number in $\ell_2$ is

$$\kappa_2(W)=\frac{\sigma_{\max}(W)}{\sigma_{\min}(W)}.$$ This measures

sensitivity of linear system solutions and invertibility. A crucial

point for your thesis is that *condition number alone does not control

amplification*. A matrix can have condition number $1$ and still have

very large singular values. Therefore:

  

- controlling $\kappa(W)$ helps with numerical conditioning,

  

- but it does *not* by itself guarantee a small Lipschitz constant.

  

To control amplification, you must also control scale, especially

$\sigma_{\max}(W)$.

  

## Lipschitz continuity of neural networks

  

A function $f$ is $L$-Lipschitz if $$\|f(x)-f(y)\| \le L\|x-y\|

\qquad \forall x,y.$$ If $f$ is a feedforward network with 1-Lipschitz

activations, then $$\mathrm{Lip}(f)\le \prod_{i=1}^L \|W_i\|_2.$$ This

is usually loose but conceptually important. It shows why unconstrained

linear layers can destroy stability even when activations are bounded or

smooth.

  

## Bounded activations versus unconstrained linear maps

  

A common misconception is that bounded activations imply robustness.

They do not. The output of $\sin(\cdot)$ is bounded in $[-1,1]$, but the

sensitivity of the network still depends on the chain of linear

transforms and on any frequency scaling inside the activation arguments.

In SIREN, $\omega_0$ and the norms of $W_i$ matter directly .

  

## Composition of sensitivity in the bilevel pipeline

  

In your actual system, the classifier is not directly $x\mapsto f(x)$.

It is $$x \mapsto z^*(x) \mapsto g(z^*(x)).$$ By the chain rule,

$$\frac{d}{dx}g(z^*(x))

=

\frac{\partial g}{\partial z}\frac{dz^*}{dx}.$$ So the total sensitivity

has two factors:

  

1. how sensitive the downstream classifier $g$ is to changes in

representation space,

  

2. how sensitive the fitted representation $z^*(x)$ is to changes in

the input signal.

  

Even if $g$ is smooth, the overall pipeline can be unstable if $z^*(x)$

changes abruptly.

  

## Implicit differentiation viewpoint

  

If $z^*(x)$ is the solution of an inner optimization problem

$$z^*(x)=\arg\min_z \mathcal{L}(z,x),$$ and if the first-order condition

$\nabla_z\mathcal{L}(z^*,x)=0$ holds with nonsingular Hessian

$H=\nabla^2_{zz}\mathcal{L}(z^*,x)$, then the implicit function theorem

gives $$\frac{dz^*}{dx}

=

- H^{-1}\nabla^2_{xz}\mathcal{L}(z^*,x).$$ This is one of the most

important equations for the thesis. It explains why conditioning

matters: if $H$ is ill-conditioned, then small perturbations in input

space can induce large changes in the fitted representation.

  

## How robustness bounds are usually derived

  

In standard adversarial ML, a common logic is:

  

- if the classifier has margin $m(x)$ at input $x$,

  

- and if the score function is $L$-Lipschitz,

  

- then perturbations smaller than roughly $m(x)/L$ cannot flip the

prediction.

  

This idea motivates Lipschitz control and certified robustness methods.

In your setting, the challenge is that the relevant $L$ is the Lipschitz

constant of the *composed bilevel map*, not only of the outer

classifier.

  

## Functional-analysis viewpoint

  

The useful functional-analysis perspective is that INRs replace finite

arrays by elements of a function space. Instead of saying “this image is

a vector in $\mathbb{R}^{H\times W \times C}$,” one says “this datapoint

is represented by a function $f_\theta$ defined on a continuous domain.”

This viewpoint is valuable because:

  

- it clarifies why derivatives and continuity matter,

  

- it suggests studying norms on functions or Jacobians, not just on

arrays,

  

- and it connects robustness to stability of the mapping from signals to

functions and from functions to downstream representations.

  

# Literature on constraining linear layers

  

## Why this literature matters

  

If the hypothesis is that uncontrolled linear maps inside SIREN amplify

perturbations, then the relevant literature is the literature on

controlling operator norms, Jacobians, or conditioning in deep networks.

The critical question is *not* whether these methods help in ordinary

feedforward networks; it is whether they help in the *bilevel

INR-fitting pipeline* without simply increasing gradient obfuscation.

  

## Method comparison

  

| Method | What it constrains | Controls $\|W\|_2$? | Controls $\kappa(W)$? | Comments for SIRENs |

|:--------------------------------------|:------------------------------------|:-------------------------------------|:-----------------------------|:------------------------------------------------------------------------|

| Method | What it constrains | Controls $\|W\|_2$? | Controls $\kappa(W)$? | Comments for SIRENs |

| Spectral normalization | Largest singular value | Yes | No | Simple and lightweight; natural first baseline |

| Parseval / orthogonality constraints | Near-isometry / orthonormality | Yes (usually $\le 1$) | Partly | Attractive because they control amplification and collapse jointly |

| Singular value regularization | Large and/or small singular values | Yes | Yes if both ends constrained | Potentially expensive but SIREN MLPs are often small enough |

| Jacobian regularization | Norm of input-output Jacobian | Indirectly | Indirectly | More function-level than layer-level; may fit bilevel robustness better |

| Condition number regularization | Ratio $\sigma_{\max}/\sigma_{\min}$ | No, unless paired with scale control | Yes | Useful for optimization stability, insufficient alone for robustness |

| Certified methods (e.g. smoothing) | End-to-end robustness guarantees | Not directly | Not directly | Potentially relevant later, but hard in bilevel settings |

  

## Spectral normalization

  

Spectral normalization rescales a weight matrix by an estimate of its

largest singular value so that its spectral norm is bounded . It is

attractive because:

  

- it is simple to implement,

  

- it directly controls $\|W\|_2$,

  

- it introduces relatively low overhead.

  

For your thesis, it is the most defensible first method to test.

However, it does not control the smaller singular values, so it does not

by itself prevent ill-conditioning.

  

## Parseval networks and orthogonality constraints

  

Parseval networks constrain linear maps to behave approximately like

Parseval tight frames, which bounds Lipschitz constants and was

explicitly motivated by adversarial robustness . This is attractive for

SIRENs because near-orthogonality can simultaneously:

  

- prevent large amplification,

  

- reduce collapse of dimensions,

  

- improve conditioning.

  

This may be a stronger candidate than spectral normalization alone if

expressivity remains acceptable.

  

## Singular value regularization

  

A more direct but more expensive route is to regularize singular values

explicitly. For example, one can penalize large $\sigma_{\max}(W)$ and

small $\sigma_{\min}(W)$, or regularize the entire spectrum. The

advantage is conceptual clarity:

  

- $\sigma_{\max}$ controls worst-case amplification,

  

- $\sigma_{\min}$ controls collapse,

  

- the ratio controls conditioning.

  

The downside is cost. Still, compared with CNNs, SIREN layers are often

small enough that this may be feasible.

  

## Jacobian regularization

  

Jacobian regularization penalizes the norm of the input-output Jacobian

. This is appealing because it works at the *function* level rather than

only at the layer level. In your setting, that may matter more, since

the final object of interest is the map from signal to representation or

from representation to prediction. A limitation is computational cost,

especially if applied to the full composed bilevel system.

  

## Condition number control

  

Condition number regularization targets numerical stability more

directly than adversarial robustness. It can be useful for:

  

- stabilizing the optimization dynamics of the INR,

  

- improving the conditioning of the inner problem,

  

- making implicit differentiation more numerically reliable.

  

But by itself, it is not enough for robustness because it does not bound

absolute sensitivity unless scale is also controlled.

  

## Certified robustness methods

  

Certified methods such as randomized smoothing provide formal guarantees

under certain norms . These are relevant conceptually, but in the

INR-fitting setting they are challenging because:

  

- the full pipeline is computationally expensive,

  

- certification usually assumes a better-behaved forward map than an

inner optimization loop provides,

  

- guarantees may become too loose to be useful.

  

Still, they may be worth revisiting once you have a simplified or

amortized version of the pipeline.

  

## Critical evaluation for the thesis

  

The most important critical point is this: a layer constraint may

improve true stability, but it may also make the fitting loop harder to

differentiate through or slower to optimize. If that happens, apparent

robustness can go up while real robustness does not. Therefore:

  

> No layer-constraining method should be evaluated only through naive

> white-box attacks in this project.

  

# Fourier features and richer INR representations

  

## What Fourier features are

  

Fourier features map coordinates $x$ into a higher-dimensional

sinusoidal feature space:

$$\gamma(x)=\big[\sin(2\pi Bx), \cos(2\pi Bx)\big],$$ where $B$ is a

matrix of frequencies . The transformed input is then fed into a

standard MLP.

  

## How they differ from or complement SIREN

  

SIREN uses sinusoidal *activations* internally. Fourier features instead

use sinusoidal *input encodings*. These are different levers:

  

- SIREN changes the network nonlinearity.

  

- Fourier features change the basis in which the signal is presented to

the network.

  

They can be viewed as complementary rather than mutually exclusive.

  

## Why they help with complex natural images

  

Fourier features were introduced partly to address spectral bias in

standard MLPs . In NeRF-style models, positional encodings are crucial

for fitting fine detail . Multi-resolution encodings, as in Instant-NGP,

push this even further and make neural fields practical at scale . This

suggests that richer encodings may be necessary if parameter-space

classifiers are to move beyond toy benchmarks.

  

## Will they make parameter-space learning easier or harder?

  

There are arguments both ways.

  

#### Why easier:

  

- Better fitting quality may produce more informative representations.

  

- Fewer optimization steps may be needed.

  

- More realistic datasets may become accessible.

  

#### Why harder:

  

- The representation may become more sensitive to perturbations because

high-frequency content is preserved rather than suppressed.

  

- The downstream parameter-space classifier may need to handle a richer,

less compressed representation.

  

- Some of the current incidental robustness may disappear.

  

## Robustness implications

  

This is one of the most interesting thesis questions. If current

apparent robustness is partly due to the inner INR acting as a low-pass

filter, then improving the INR’s ability to capture high-frequency

content may *reduce* that apparent robustness unless a new robustness

mechanism is added. So Fourier features are not just a scaling trick;

they are also a stress test of the current robustness story.

  

## Papers most relevant beyond the core list

  

The most relevant papers for this direction are:

  

- Fourier Features ,

  

- NeRF positional encoding ,

  

- Instant-NGP / multi-resolution encoding ,

  

- Trans-INR ,

  

- End-to-End INR Classification .

  

# Robustness and attacks in this setting

  

## Attack surface

  

The adversary perturbs the original signal $x$ subject to a norm budget,

but the decision is made after fitting and downstream classification:

$$\hat{y}(x)=g(z^*(x)).$$ So the attack surface includes:

  

1. the input signal,

  

2. the inner fitting procedure,

  

3. the downstream classifier.

  

## Differentiability bottlenecks

  

If the fitting procedure is iterative, then exact gradients through the

pipeline may require:

  

- unrolling many optimization steps,

  

- implicit differentiation,

  

- or approximations such as BPDA-like backward surrogates.

  

This is why naive attacks often underestimate vulnerability.

  

## Where gradient obfuscation comes from

  

Gradient obfuscation arises because:

  

- the fitting process is iterative and truncated,

  

- the optimizer may be nondifferentiable in practice or badly

conditioned,

  

- the INR may suppress certain perturbations, especially high-frequency

ones,

  

- naive backpropagation through the full process may be too weak, too

noisy, or too expensive.

  

This mirrors the classical gradient-masking patterns identified by

Athalye et al. .

  

## Why BPDA is relevant

  

BPDA (*Backward Pass Differentiable Approximation*) is conceptually

relevant whenever a component is hard to differentiate through usefully.

The idea is to use the true forward pass but replace the backward pass

with a differentiable approximation . In your setting, BPDA-like

reasoning matters because the fitting loop or reconstruction stage may

behave like a non-usefully-differentiable defense when attacked naively.

  

## What black-box attacks reveal

  

Black-box or gradient-free attacks are a crucial sanity check. If

robustness vanishes under black-box attacks while naive white-box

attacks fail, then the likely explanation is not true robustness but bad

gradients. The recent weight-space attack paper explicitly emphasizes

the importance of alternative attacks in this setting .

  

## What a strong evaluation protocol should look like

  

A credible evaluation protocol for this thesis should include:

  

- standard gradient attacks only as a baseline,

  

- optimization-aware attacks that account for the fitting loop,

  

- implicit-differentiation-based attacks when possible,

  

- BPDA-style approximations when appropriate,

  

- gradient-free attacks,

  

- robustness sanity checks inspired by AutoAttack methodology .

  

## How to test whether a Lipschitz-bounded SIREN helps for real

  

Suppose you impose spectral or orthogonality constraints on the SIREN.

To test whether this improves *real* robustness rather than merely

making gradients worse, you should check:

  

1. Does robust accuracy still improve under stronger adaptive attacks?

  

2. Does the fitted representation become more stable under small input

perturbations?

  

3. Does the inner optimization remain well-conditioned and convergent?

  

4. Does clean reconstruction quality remain acceptable?

  

5. Do attack success rates keep rising as you strengthen the attack, or

do they saturate?

  

If gains disappear under stronger adaptive evaluation, then the method

likely improved obfuscation, not robustness.

  

# Research opportunities

  

## Ranked candidate directions

  

| Direction | Novelty | Feasibility | Pub. potential | One-line rationale |

|:------------------------------------------------------|:--------|:------------|:---------------|:-------------------------------------------------------------------------------|

| Direction | Novelty | Feasibility | Pub. potential | One-line rationale |

| Representation-stability regularization | High | Medium | High | Targets the real object that matters: stability of $z^*(x)$ under perturbation |

| Spectral / Parseval SIRENs with adaptive evaluation | Medium | High | Medium–High | Clean, testable hypothesis with strong baselines |

| Fourier-feature INR scaling + robustness reevaluation | High | Medium | High | Probes whether scaling kills accidental robustness |

| Implicit-diff attacks + stability training | Medium | Medium | Medium | Makes both attacks and defenses more principled |

| Conditioning-aware inner-loop design | Medium | Medium | Medium | Focuses on conditioning of the fitting problem rather than only layer norms |

| Symmetry-aware weight-space classifiers | High | Low–Medium | Medium | More theoretical and architectural, less directly robustness-focused |

  

## Direction 1: Representation-stability regularization

  

#### Rationale.

  

The most direct way to seek structural robustness is to stabilize the

fitted representation itself: $$\|z^*(x)-z^*(x+\delta)\|$$ should remain

small for allowable perturbations.

  

#### Key experiment.

  

Add a regularizer such as

$$\mathcal{L}_{\text{stab}} = \mathbb{E}_{x,\delta}\|z^*(x)-z^*(x+\delta)\|_2^2$$

while preserving reconstruction and classification quality.

  

#### What could go wrong.

  

You may accidentally train a system whose gradients become harder to

compute without becoming genuinely stable. This is why adaptive

evaluation is essential.

  

#### Baselines.

  

Functa baseline, unconstrained SIREN, and the attack paper’s evaluated

pipeline.

  

## Direction 2: Spectral / Parseval constraints on SIREN layers

  

#### Rationale.

  

This is the cleanest version of the advisor’s idea. It directly tests

whether controlling worst-case amplification in the INR backbone

improves end-to-end robustness.

  

#### Key experiment.

  

Compare:

  

- unconstrained SIREN,

  

- spectrally normalized SIREN,

  

- Parseval-constrained SIREN,

  

- perhaps spectral + Jacobian regularization.

  

Measure clean accuracy, reconstruction fidelity, representation

stability, robust accuracy under adaptive attacks, and attack cost.

  

#### What could go wrong.

  

- clean fitting quality may degrade,

  

- richer perturbations may still break the system,

  

- observed gains may disappear under implicit-diff or black-box attacks.

  

#### Baselines.

  

Unconstrained SIREN, Jacobian-regularized outer classifier, adversarial

training in representation space.

  

## Direction 3: Conditioning-aware inner-loop design

  

#### Rationale.

  

A subtle but important point is that the stability of $z^*(x)$ depends

not only on layer norms but also on the conditioning of the *inner

optimization landscape*. That suggests targeting Hessian conditioning or

optimizer stability directly.

  

#### Key experiment.

  

Track approximate Hessian condition surrogates, convergence rate, and

representation stability as you vary:

  

- condition-number regularization,

  

- damping,

  

- trust-region-like updates,

  

- or reparameterizations that improve conditioning.

  

#### What could go wrong.

  

This direction may become optimization-heavy and drift away from

adversarial ML unless kept tightly tied to attack outcomes.

  

#### Baselines.

  

Standard optimizer-based fitting, spectral-only control, and

unregularized fitting.

  

## Direction 4: Fourier-feature or richer INR pipelines

  

#### Rationale.

  

This direction addresses the scalability bottleneck directly and also

tests a strong scientific hypothesis: improving the INR’s fidelity may

weaken its incidental robustness mechanism.

  

#### Key experiment.

  

Replace or augment vanilla SIREN with Fourier-feature input encoding or

multi-resolution encodings, then repeat the robustness evaluation on

increasingly realistic datasets.

  

#### What could go wrong.

  

- the representation becomes too large or too unstable,

  

- accidental robustness disappears,

  

- the downstream weight-space model becomes the new bottleneck.

  

#### Baselines.

  

Vanilla SIREN pipeline, end-to-end INR classification models, and

standard pixel-space classifiers.

  

## Direction 5: Implicit differentiation as both attack and training tool

  

#### Rationale.

  

If the core issue is the differentiability of the inner loop, then

better implicit differentiation may help both in attacking the model

correctly and in training it to be stable.

  

#### Key experiment.

  

Implement a stable implicit differentiation pipeline for $dz^*/dx$ and

use it for:

  

- stronger adaptive attacks,

  

- representation-stability regularization,

  

- sensitivity diagnostics.

  

#### What could go wrong.

  

Implicit differentiation can be numerically fragile if the inner problem

is badly conditioned, which may force you back into the conditioning

direction above.

  

# Experimental blueprint

  

## Small proof-of-concept setup

  

A realistic proof-of-concept should stay very close to the literature:

  

- **Datasets:** MNIST, Fashion-MNIST, and optionally ModelNet10 .

  

- **INR fitting:** Functa-style shared SIREN backbone with per-sample

latent modulations.

  

- **Classifier:** simple MLP on modulation vectors as the first

baseline.

  

- **Main intervention:** one of spectral normalization, Parseval-style

constraint, or stability regularization.

  

#### Metrics.

  

Report:

  

- clean accuracy,

  

- robust accuracy across multiple $\epsilon$ values,

  

- reconstruction fidelity,

  

- representation shift $\|z^*(x)-z^*(x+\delta)\|$,

  

- attack compute cost.

  

#### Attacks.

  

At minimum:

  

- naive PGD-style baseline,

  

- unrolled attack through the fitting process,

  

- implicit-differentiation attack,

  

- BPDA-style approximation where appropriate,

  

- gradient-free attack,

  

- AutoAttack-style sanity-check philosophy for the differentiable parts

.

  

#### Ablations.

  

The most important ablations are:

  

- number of inner-loop steps,

  

- spectral bound / regularization strength,

  

- $\omega_0$ choices,

  

- attack strength and restarts,

  

- representation dimensionality.

  

## Ambitious setup

  

A more ambitious version could extend along one axis at a time:

  

- **Dataset scale:** CIFAR-10, Imagenette, or small ImageNet subsets .

  

- **Richer representations:** Fourier features or multi-resolution

encodings .

  

- **Stronger weight-space models:** Transformer-based classifiers over

weights or modulations .

  

- **End-to-end training:** learned initialization or learned step

schedules to reduce dependence on long optimization loops.

  

## Failure cases to watch for

  

1. “Robustness” that disappears as attack strength increases.

  

2. Better robust accuracy but much worse reconstruction quality.

  

3. Constrained INRs that fit too poorly to be meaningful

representations.

  

4. Gains that exist only under one threat model or one perturbation

frequency regime.

  

5. Excessive compute that prevents reliable attack evaluation.

  

# Final synthesis

  

## The 10 most important things to understand first

  

1. What an INR is and how it differs from array representations.

  

2. Why SIREN uses sine activations and why $\omega_0$ matters.

  

3. How Functa uses modulations instead of full weights.

  

4. Why parameter-space classification is inherently bilevel.

  

5. What operator norm and spectral norm mean geometrically.

  

6. Why condition number is not the same as robustness.

  

7. How Lipschitz constants compose across layers.

  

8. What gradient obfuscation is and why it creates false confidence.

  

9. Why BPDA and gradient-free attacks are essential sanity checks.

  

10. Why scaling the INR with Fourier features may change both fidelity

and robustness.

  

## Five important papers beyond the main four

  

1. Athalye et al., *Obfuscated Gradients Give a False Sense of

Security* .

  

2. Croce and Hein, *Reliable Evaluation of Adversarial Robustness with

an Ensemble of Diverse Parameter-Free Attacks* .

  

3. Tancik et al., *Fourier Features Let Networks Learn High Frequency

Functions in Low Dimensional Domains* .

  

4. Cisse et al., *Parseval Networks: Improving Robustness to

Adversarial Examples* .

  

5. Gielisse et al., *End-to-End Implicit Neural Representations for

Classification* .

  

## The three most promising thesis directions

  

1. **Representation-stability regularization with adaptive

evaluation.**

  

2. **Spectral / Parseval SIRENs evaluated under full adaptive attack

suites.**

  

3. **Fourier-feature INR pipelines to test the tradeoff between

scalability and accidental robustness.**

  

## The most likely pitfalls

  

1. Claiming robustness based only on naive PGD.

  

2. Improving obfuscation rather than improving stability.

  

3. Treating condition number alone as a robustness guarantee.

  

4. Ignoring reconstruction quality and representational usefulness.

  

5. Scaling to richer INR representations without rethinking the

robustness mechanism.

  

# Conclusion

  

The literature suggests that parameter-space classifiers based on INRs

are scientifically interesting precisely because they sit at the

intersection of representation learning, continuous signal modeling, and

adversarial robustness. But the existing evidence does *not* support the

claim that their current robustness is principled. The strongest thesis

direction is therefore not to defend the existing robustness story, but

to replace it with a more rigorous one: identify which parts of the

pipeline should be stable, impose constraints or training objectives

that target that stability, and evaluate them under attacks that fully

account for the inner fitting loop.

  

<div class="thebibliography">

  

99

  

Vincent Sitzmann, Julien N. P. Martel, Alexander W. Bergman, David B.

Lindell, and Gordon Wetzstein. Implicit Neural Representations with

Periodic Activation Functions. *NeurIPS*, 2020.

<https://arxiv.org/abs/2006.09661>

  

Emilien Dupont, Hyunjik Kim, S. M. Ali Eslami, Danilo J. Rezende, and

Dan Rosenbaum. Your Data Point Is a Function and You Can Treat It Like

One. *ICLR*, 2022. <https://arxiv.org/abs/2201.12204>

  

Maithra Raghu, Justin Gilmer, Jason Yosinski, and Jascha Sohl-Dickstein.

SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning

Dynamics and Interpretability. *NeurIPS*, 2017.

<https://arxiv.org/abs/1706.05806>

  

David Ha, Andrew Dai, and Quoc V. Le. HyperNetworks. *ICLR*, 2017.

<https://arxiv.org/abs/1609.09106>

  

Tamir Shor, Amit Port, and colleagues. Adversarial Attacks in

Weight-Space Classifiers. TMLR / OpenReview, 2025–2026.

<https://openreview.net/forum?id=eOLybAlili>

  

Anish Athalye, Nicholas Carlini, and David Wagner. Obfuscated Gradients

Give a False Sense of Security: Circumventing Defenses to Adversarial

Examples. *ICML*, 2018.

<https://proceedings.mlr.press/v80/athalye18a.html>

  

Francesco Croce and Matthias Hein. Reliable Evaluation of Adversarial

Robustness with an Ensemble of Diverse Parameter-Free Attacks. *ICML*,

2020. <https://arxiv.org/abs/2003.01690>

  

Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida.

Spectral Normalization for Generative Adversarial Networks. *ICLR*,

2018. <https://arxiv.org/abs/1802.05957>

  

Moustapha Cisse, Piotr Bojanowski, Edouard Grave, Yann Dauphin, and

Nicolas Usunier. Parseval Networks: Improving Robustness to Adversarial

Examples. *ICML*, 2017. <https://arxiv.org/abs/1704.08847>

  

Daniel Jakubovitz and Raja Giryes. Improving DNN Robustness to

Adversarial Attacks using Jacobian Regularization. *ECCV Workshops*,

2018.

<https://openaccess.thecvf.com/content_ECCV_2018/html/Daniel_Jakubovitz_Improving_DNN_Robustness_ECCV_2018_paper.html>

  

Judy Hoffman, Daniel A. Roberts, and Sho Yaida. Robust Learning with

Jacobian Regularization. arXiv, 2019. <https://arxiv.org/abs/1908.02729>

  

Jeremy M. Cohen, Elan Rosenfeld, and Zico Kolter. Certified Adversarial

Robustness via Randomized Smoothing. *ICML*, 2019.

<https://arxiv.org/abs/1902.02918>

  

Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara

Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi,

Jonathan T. Barron, and Ren Ng. Fourier Features Let Networks Learn High

Frequency Functions in Low Dimensional Domains. *NeurIPS*, 2020.

<https://arxiv.org/abs/2006.10739>

  

Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T.

Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing Scenes as

Neural Radiance Fields for View Synthesis. *ECCV*, 2020.

<https://arxiv.org/abs/2003.08934>

  

Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller.

Instant Neural Graphics Primitives with a Multiresolution Hash Encoding.

*SIGGRAPH*, 2022. <https://arxiv.org/abs/2201.05989>

  

Yunseok Lee, Hae Beom Lee, Hyeokjun Kwon, and Jinwoo Shin. Transformers

as Meta-Learners for Implicit Neural Representations. *ECCV*, 2022.

<https://arxiv.org/abs/2208.02801>

  

A. Gielisse, and collaborators. End-to-End Implicit Neural

Representations for Classification. *CVPR*, 2025.

<https://arxiv.org/abs/2503.18123>

  

</div>