# LSTM-H5: a recurrent neural network on the five-event history (experiment 6C)

**Status.** Methodology and results note, 2026-11-25.

**Task.** `actions15_360current_v1` (`docs/canonical_prediction_protocol.md`). The notation
follows `02_mathematical_framework.md`, `07_logistic_regression_h5.md` and `08_mlp_h5.md`.

| Item | Location |
|---|---|
| Implementation | `src/sequence_models/lstm_h5_experiment.py`, `scripts/run_lstm_h5_experiment6c.py` |
| Lifecycle supervisor | `scripts/supervise_lstm_h5_experiment6c.py` |
| Independent audit | `scripts/verify_lstm_h5_experiment6c.py` |
| Configuration (frozen before the run) | `configs/lstm_h5_experiment6c_v1.yaml` |
| Tests | `tests/test_lstm_h5_experiment6c.py` |
| Results | `results/lstm_h5_experiment6c_actions15_360current_v1_2026-11-25/README.md` |

## 1. Research question and motivation

**Question.** Given the same five-event history, does a recurrent architecture designed for
sequential processing estimate next-event probabilities better than the feed-forward network
(MLP-H5), the additive softmax regression (MLR-H5), the Markov chains and gradient-boosted
trees (XGB-H5)?

**What is held fixed, and what is not.**
- **Identical to experiments 5, 5B, 6A and 6B.** The 690,147 evaluation rows, targets, outer
  folds, the five-position history (the input information) and the metric definitions.
- **Identical to MLP-H5 (experiment 6B).**
  - Every inner and outer training, stopping and validation match set (verified against 6B's
    data-access audit).
  - The optimiser settings, batch size, epoch limit, patience, stopping rule, the number of
    refit replicates, and the seed-derivation convention.
  - The same seed integers do **not** imply the same random draws: initialisation, batch order
    and dropout masks depend on the model's shape.
- **Not held fixed.**
  - The architecture: a shared-parameter recurrence vs a feed-forward network on the
    position-specific encoding.
  - The parameter counts, the candidate hyperparameter lists (two different sets of 8 configurations),
    the selected configurations, and the optimisation trajectories that result from them.

  The LSTM-H5 vs MLP-H5 comparison is therefore a comparison of two complete model-plus-search
  procedures under shared data partitions and a shared training protocol. It does not isolate
  the recurrence alone.

**No expected result.** A negative result is as informative as a positive one. The protocol,
search space, seeds and evaluation were fixed before any outer-validation result existed.

## 2. The H5 prediction task (unchanged)

\[
X_t = (E_{t-4}, E_{t-3}, E_{t-2}, E_{t-1}, E_t), \qquad Y_t = E_{t+1},\qquad
\text{estimate } P(Y_t = c \mid X_t),\ c = 1,\dots,K,\ K = 15 .
\]
- **What \(t\) indexes.** \(t\) indexes positions in the canonical event sequence of one
  (match, period). H5 means five **event positions**, including the current one. It does not
  mean five seconds: the model is event-history based, not a continuous-time model.
- **`<START>`.** Positions before the start of the sequence hold the input symbol `<START>`
  (id 15). It is never a target and never an output class.
- **No boundary crossing.** Histories never cross a period or match boundary.
- **Rows.** All protocol rules apply (Protocol B rows, K1–K3, K4 inactive, StatsBomb `index`
  order, period 5 excluded).
- **Final test.** The 83 final-test matches were never loaded.

## 3. Recurrent neural networks and the motivation for LSTM

**Recurrent networks.** A recurrent network processes a sequence \(x_1,\dots,x_S\) with one
transition function \(h_s = \Phi(x_s, h_{s-1}; \theta)\), whose parameters \(\theta\) are shared
across positions. The final state \(h_S\) summarises the sequence.
- **Contrast with MLP-H5.** In MLP-H5, each history position has its own first-layer columns.
  An RNN instead reuses the same weights at every position and conditions on order through the
  state.

**The LSTM.**
- **Motivation.** A simple tanh RNN multiplies gradients through \(\partial h_s/\partial h_{s-1}\)
  at every step, which makes long-range credit assignment hard.
- **Origin.** The LSTM (Hochreiter and Schmidhuber, 1997) introduced an additively updated
  memory cell with multiplicative gates.
- **The forget gate.** The variant used here, which PyTorch implements, has a forget gate. That
  gate was added later by Gers, Schmidhuber and Cummins (2000). The equations below are therefore
  the modern formulation, not the unchanged 1997 equations.

**Caution for this task.** The window has only five positions. Long-range dependencies beyond
five events are **not available** to the model. Using an LSTM here tests a sequential
inductive bias on short windows. It does not test long-term memory.

## 4. Input representation

Each position \(s = 1,\dots,5\) of \(X_t\) is mapped, oldest first, to
\[
x_s = \mathrm{onehot}_{16}(E_{t-5+s}) \in \{0,1\}^{16},\qquad s = 1 \leftrightarrow E_{t-4},\ \dots,\ s = 5 \leftrightarrow E_t ,
\]
over the shared vocabulary of 15 classes plus `<START>`. A mini-batch of \(B\) windows is a
tensor of shape \((B, 5, 16)\).
- **Same information as before.** The ids are those of MLR-H5 and MLP-H5. The position-specific
  79-column encoding of 6A/6B is a bijective re-coding of the same five ids (verified on all
  rows). The difference between the models is architectural, not informational.
- **Not used.** No embedding layer, no pretrained vectors and no coordinates, scores, team,
  player, time or target-derived information.

## 5. Model equations

**Symbols.** The event notation is that of `02_mathematical_framework.md`: \(E_t\) is the
current retained canonical event (written \(e_t\) in some project notes), \(Y_t = E_{t+1}\) the
target, \(X_t = H_t = (E_{t-4},\dots,E_t)\) the history, and \(K = 15\) the number of target
classes (written \(V\) in some notes).

| Symbol | Meaning |
|---|---|
| \(s = 1,\dots,5\) | position inside the recurrence (not the event index \(t\)) |
| \(m = 16\) | input dimension of layer 1 |
| \(H\) | hidden dimension (= cell dimension); written \(d\) in some texts, but \(d\) already denotes the number of input features in `02` |
| \(L \in \{1,2\}\) | number of stacked recurrent layers |
| \(h_s^{(\ell)}, c_s^{(\ell)} \in \mathbb{R}^H\) | hidden and cell state of layer \(\ell\) after position \(s\) |
| \(i\) | index of an observation (row) |
| \(c\) | class index; in code 0-based |
| \(\sigma(a) = 1/(1+e^{-a})\) | logistic function; \(\tanh\) the hyperbolic tangent; \(\odot\) element-wise product |

**Layer 1** (\(\ell = 1\), input \(x_s\)); \(h_0 = c_0 = 0\) for **every** window:
\[
\begin{aligned}
i_s &= \sigma(W_i x_s + U_i h_{s-1} + b_i) &&\text{(input gate)}\\
f_s &= \sigma(W_f x_s + U_f h_{s-1} + b_f) &&\text{(forget gate)}\\
g_s &= \tanh(W_g x_s + U_g h_{s-1} + b_g) &&\text{(candidate cell state)}\\
o_s &= \sigma(W_o x_s + U_o h_{s-1} + b_o) &&\text{(output gate)}\\
c_s &= f_s \odot c_{s-1} + i_s \odot g_s &&\text{(cell-state update)}\\
h_s &= o_s \odot \tanh(c_s) &&\text{(hidden-state update)}
\end{aligned}
\]
with \(W_\bullet \in \mathbb{R}^{H\times 16}\), \(U_\bullet \in \mathbb{R}^{H\times H}\) and
\(b_\bullet \in \mathbb{R}^H\).

**Layer 2** (when \(L = 2\)). The same equations with its own parameters. Its input is
\(h_s^{(1)}\), so \(W^{(2)}_\bullet \in \mathbb{R}^{H\times H}\).

**PyTorch storage.**
- **Weights.** `torch.nn.LSTM` stores `weight_ih_l{k}` \(= [W_i; W_f; W_g; W_o]\)
  (\(4H \times m\)) and `weight_hh_l{k}` \(= [U_i; U_f; U_g; U_o]\) (\(4H\times H\)), in the
  block order input, forget, candidate, output.
- **Biases.** It stores two bias vectors, `bias_ih` and `bias_hh`. The effective gate bias is
  their sum, \(b_\bullet = b^{ih}_\bullet + b^{hh}_\bullet\); the parameterisation is redundant
  but the function is not changed.
- **Verified.** Both conventions were checked against an independent NumPy implementation (§15).

**Output.** Only the final hidden state of the last layer is used:
\[
z_t = W_{\text{out}}\,\mathrm{drop}_p\!\left(h_5^{(L)}\right) + b_{\text{out}} \in \mathbb{R}^{15},\quad
W_{\text{out}}\in\mathbb{R}^{15\times H},\quad
P(Y_t = c \mid X_t) = \frac{\exp z_{t,c}}{\sum_{j=1}^{15}\exp z_{t,j}} .
\]
- **Dropout.** \(\mathrm{drop}_p\) is inverted dropout with rate \(p\), active only in training.
- **Inter-layer dropout.** The built-in inter-layer dropout of `nn.LSTM` is set to 0: it would
  act only between stacked layers and has no effect when \(L = 1\). The only dropout in this
  experiment is the one on \(h_5^{(L)}\).
- **Softmax.** It is applied once, at inference, in float64 on the float32 logits. It is never
  applied before the loss (§6).
- **Classification type.** The 15 outputs are mutually exclusive classes, not independent
  sigmoids.

**Statelessness.** Every window is an independent example. `nn.LSTM` is called without an
initial state, so \(h_0 = c_0 = 0\) for every window. No state is carried between windows,
batches, matches, periods, folds, or training and evaluation.
- **Tests.** A window gives the same logits alone, inside any batch and after any permutation of
  the batch. Changing all other windows of a batch leaves it unchanged.
- **Why it matters.** It is what keeps the model's information identical to H5.

## 6. Objective and optimisation

**Loss.** The loss is mean multiclass cross-entropy, i.e. the negative log-likelihood, on logits:
\[
L_{CE} = \frac1N\sum_{i=1}^N \ell_i,\qquad \ell_i = -z_{i,y_i} + \log\sum_{j}\exp z_{i,j} ,
\]
computed by `F.cross_entropy`, which applies log-softmax internally.
- **Not used:** class weights, resampling, focal loss, label smoothing or decision thresholds.
- **Epoch training loss.** It is row-weighted: \(\sum_b |b|\,L_b / N\), so the final smaller
  batch is weighted correctly. This is tested.

**Optimiser.** AdamW, as in 6B: \(\eta\) from the configuration, \(\beta = (0.9, 0.999)\),
\(\epsilon = 10^{-8}\), mini-batches of 512 rows in a seeded per-epoch permutation. With
\(g_q\) the mini-batch gradient at step \(q\):
\[
m_q = \beta_1 m_{q-1} + (1-\beta_1) g_q,\quad v_q = \beta_2 v_{q-1} + (1-\beta_2) g_q^2,\quad
\hat m_q = \frac{m_q}{1-\beta_1^q},\quad \hat v_q = \frac{v_q}{1-\beta_2^q},
\]
\[
\theta_q = \theta_{q-1} - \eta\lambda\,\theta_{q-1} - \eta\,\frac{\hat m_q}{\sqrt{\hat v_q}+\epsilon}
\quad\text{(PyTorch AdamW: decay applied first, then the Adam step).}
\]
- **Decoupled weight decay.** The decay \(\lambda\) is decoupled (Loshchilov and Hutter, 2019).
  It is applied to the weight matrices `weight_ih`, `weight_hh` and \(W_\text{out}\), and not to
  biases.
- **Not an L2 penalty.** Under Adam it is **not** equivalent to adding \(\tfrac{\lambda}{2}\|\theta\|^2\)
  to the loss: the adaptive rescaling acts on the gradient, but not on the decay term.
- **Reported losses.** The reported log loss never contains a regularisation term.
- **Test.** With a zero gradient, one step multiplies exactly the decayed tensors by
  \((1-\eta\lambda)\).

**Gradient clipping.** None, as in 6B (prespecified). The global norm \(\|g_q\|_2\) of every
mini-batch is recorded as a diagnostic and doubles as the finiteness check. Any non-finite loss,
gradient or parameter stops the fit with an error.

## 7. Backpropagation through time

**Unrolled graph.** The loss of window \(i\) depends on \(\theta = (W, U, b)\) through all five
positions. Unrolling the recurrence gives a feed-forward graph with tied weights. Autograd
differentiates it exactly (backpropagation through time; Werbos, 1990). For a shared parameter
the total gradient is the sum of the contributions of its five uses:
\[
\frac{\partial \ell_i}{\partial U} = \sum_{s=1}^{5} \frac{\partial \ell_i}{\partial a_s}\, h_{s-1}^{\top},
\qquad a_s = W x_s + U h_{s-1} + b,
\]
where \(\partial \ell_i/\partial a_s\) itself collects the paths through \(h_s\) and \(c_s\) to
all later positions.
- **The cell path.** Along it, \(\partial c_s/\partial c_{s-1} = \mathrm{diag}(f_s)\). This is the
  additive route that motivates the LSTM.
- **No truncation.** Nothing is detached inside the window, so there is no truncation. A test
  confirms that the gradient reaches the input of every position, including \(E_{t-4}\).
- **Between windows.** No state is propagated between windows, so no gradient flows between
  them either.

**Gradient verification.** Autograd gradients of all parameter tensors (`weight_ih`,
`weight_hh`, both biases, \(W_\text{out}\), \(b_\text{out}\); one and two layers) were compared
with central finite differences of the **independent NumPy** loss.
- **Setup.** float64, step \(h = 10^{-6}\).
- **Tolerance.** The truncation error is \(O(h^2)\approx 10^{-12}\) and the rounding error
  \(\approx 10^{-16}/h = 10^{-10}\), hence a tolerance of \(10^{-7}\).

## 8. Parameter counts

For a layer with input dimension \(m\):
\[
P_{\text{LSTM}} = 4Hm + 4H^2 + 8H \quad(\text{two bias vectors}),\qquad P_{\text{out}} = 15H + 15 .
\]
The formula equals `sum(p.numel() for p in model.parameters())` for every candidate (tested).

| Config | H | L | Parameters | | MLP-H5 config | Parameters |
|---|---:|---:|---:|---|---|---:|
| A | 32 | 1 | 6,895 | | A [32] | 3,055 |
| B, C, G | 64 | 1 | 21,967 | | B, H [64] | 6,095 |
| D, H | 128 | 1 | 76,687 | | C, E [64, 32] | 7,695 |
| E | 32 | 2 | 15,343 | | D, F, G [128, 64] | 19,471 |
| F | 64 | 2 | 55,247 | | | |

**Larger is not assumed better.** The LSTM candidates have between 0.35× (A) and 3.9× (D, H)
the parameters of the MLP-H5 configuration selected in four of five folds (F, 19,471). A larger
parameter count is not assumed to imply better prediction; the results compare the models at
their selected sizes.

## 9. Architecture search, nested cross-validation and early stopping

**Candidates.** Eight prespecified configurations, the researcher's proposed table, adopted
unchanged and frozen before the run:

| | A | B | C | D | E | F | G | H |
|---|---|---|---|---|---|---|---|---|
| H | 32 | 64 | 64 | 128 | 32 | 64 | 64 | 128 |
| layers | 1 | 1 | 1 | 1 | 2 | 2 | 1 | 1 |
| dropout on \(h_5\) | 0 | 0 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| weight decay | 0 | 0 | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| learning rate | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 3e-4 | 3e-4 |

**Nested CV.** The protocol of 6B is reused unchanged, for each of the 5 fixed outer folds:
1. **Inner CV.** Leave one inner fold out over the other 4 folds. A fit trains on 3 inner folds
   minus a seeded 10 % match-level stopping subset, and stops on that subset (patience 5, at most
   60 epochs, strict improvement).
2. **Scoring.** The fit is scored on the held inner fold. That fold never influences the
   stopping epoch.
3. **Selection.** The smallest pooled inner log loss is selected: the sum of NLL over the 4 held
   folds divided by their rows. Ties go to the earlier configuration.
4. **Refit.** The selected configuration is refitted on the outer-training matches minus a
   seeded 10 % stopping subset.
5. **Prediction.** The refit predicts the outer-validation rows once.

**Early stopping details.**
- **Epoch 0.** The initial model (epoch 0) is the incumbent.
- **Returned state.** The returned state is the stored best-epoch state, and its recomputed
  stopping loss equals the recorded best loss (tested). A test also covers the case where epoch 0
  is the best.
- **Outer-validation labels.** They are never used for selection, stopping, epochs, seeds or any
  other choice. An adversarial test changes all outer-validation labels of one fold and finds
  that fold's selection, stopping epochs, weights and predictions bit-identical.

**Same stopping subsets as MLP-H5.** Every inner and outer data set (training, stopping, held
inner fold, outer validation) consists of the same matches as in experiment 6B. This is
verified by the SHA-256 of each match set against 6B's `data_access_audit.csv`. LSTM-H5 and
MLP-H5 therefore see the same training and stopping data.

**Different training data than XGB-H5.** Both neural networks update parameters on about 90 % of
the outer-training rows; XGB-H5 used 100 % (audit finding F1 of 6A/6B). The comparison with
XGB-H5 inherits this difference. The magnitude, and even the direction, of its effect on
predictive performance are unknown without a dedicated controlled refit.

## 10. Reproducibility and execution

- **Seeds.** As in 6B: SeedSequence of (base 20261123, outer, inner + 1 or 0 for refits,
  configuration, replicate) per fit. Each epoch is re-seeded per stream (permutation, dropout,
  NumPy, Python). The stopping subsets use (base, 7, outer, inner).
- **Replicates.** Replicate 0 is the primary prediction. Replicates 1–2 are prespecified
  robustness refits of the selected configuration; they are reported separately and never
  averaged or chosen by result.
- **Device.** It was chosen by benchmark before the run:
  - CPU with 1 thread took 2.1–8.6 s per inner epoch; 2 and 4 threads were slower.
  - MPS was up to 1.4× faster for the largest configurations, but its training determinism was
    not verified, and CPU and MPS logits differed by 8e-8.
  - Choice: CPU, 1 thread, `torch.use_deterministic_algorithms(True)`.
- **Parallel folds.** The five outer folds were computed in separate single-threaded worker
  processes, at most 4 at a time. This is the number of performance cores; a worker's peak RSS
  was 0.8 GB on an 8 GB machine.
  - **Why the result is unchanged.** Each fold is a deterministic function of the data,
    configuration and seeds, so this equals a sequential run.
  - **Check.** The main process retrained fold 0's primary refit from scratch. It was checked
    for bitwise identity with the worker's weights and predictions.
- **Checkpoints.** Epoch-boundary checkpoints hold the model, optimiser, RNG states, early-stopping
  state, best state, and identifiers (configuration, dataset, folds, rows, features, sources,
  software, device, threads). Completed fits and folds are committed atomically.
  - **Tests.** Interrupt-and-resume reproduces an uninterrupted fit bit for bit, with dropout
    active. Incompatible or corrupted checkpoints are refused. Folds computed in separate runs
    assemble to exactly the sequential result.
- **Lifecycle.** A detached supervisor runs the experiment and then the independent audit. It
  holds `caffeinate -d -i -w <supervisor pid>`, writes `run_status.json` and creates a
  `COMPLETED` marker only if both stages exit 0. Duplicate runners, workers and supervisors are
  prevented by locks.
- **Scope.** Bitwise reproducibility is claimed only for this machine, thread count and software
  (Python 3.11.1, torch 2.15.0, NumPy 2.4.6), not across platforms.

## 11. Evaluation

Metrics and the paired match-level bootstrap are unchanged from experiments 5–6B:
- **Log loss** \(-\frac1N\sum_i \log \hat p_{i,y_i}\) is the primary metric.
- **Brier score** \(\frac1N\sum_i\sum_c(\hat p_{i,c}-\mathbb{1}[y_i=c])^2\) has range [0, 2].
- **Argmax metrics.** Accuracy and macro-/weighted-F1 (\(\frac1K\sum_c F1_c\),
  \(\sum_c \frac{N_c}{N}F1_c\)).
- **Per class.** Precision, recall, F1 and support.
- **Shot.** Shot average precision (step-wise \(\sum_n (R_n - R_{n-1})P_n\), the scikit-learn
  definition, not a trapezoidal PR-AUC); Shot argmax recall and precision.
- **Pooled metrics** are computed on all 690,147 out-of-fold rows, never as averages of fold
  metrics.
- **Paired differences.** \(d_i = \ell_i^{\text{LSTM}} - \ell_i^{\text{ref}}\) (negative favours
  LSTM-H5), with the 95 % percentile CI of the match-level bootstrap. It resamples the 304 matches
  with replacement, 1,000 replicates, seed 7, keeping both models' predictions of a match
  together, as a ratio of resampled sums.

**What the CIs describe.** They describe uncertainty under resampling of the evaluation matches
for **fixed** predictions. They do not include refitting, the overlap of training sets between
folds, hyperparameter selection, initialisation randomness (reported separately via seeds) or the
choice of competitions. No equivalence margin was prespecified, so "no detectable difference" is
never a claim of equivalence.

## 12. Training behaviour

| Item | Value |
|---|---|
| Fits | 175 (160 inner, 5 primary refits, 10 robustness refits), 6,085 epochs, 4,541,996 mini-batches |
| Time | 55,782 s of fitting; wall time 18,600 s (5.2 h) with fold workers, then 23 s for the independent audit |
| Peak memory | Main process 1.96 GB (holding the canonical table during verification); workers 0.60–0.85 GB |
| Inner fits | Stopped after 16–60 epochs (median 32) |
| Primary refits | Best epochs 17–31 |
| Max-epoch cap | It bound in 13 inner fits, all of the lr-3e-4 configurations (G: 11, H: 2). Their inner scores may be pessimistic, but they were 0.0017–0.0029 (G) and 0.0009–0.0015 (H) above the selected configuration |
| Selection | Two-layer networks in every fold: F (H = 64) in folds 1–3, E (H = 32) in folds 0 and 4. E and F were the two best configurations in four folds; in fold 2, C (1-layer, H = 64) was second and E third. G (lr 3e-4) was worst in four folds and second-worst in fold 4 |
| Gradients | Largest per-batch global norm 2.59; 718 batches (0.016 %) above 1, none above 5. No clipping was applied (prespecified); no non-finite value and no warning occurred |

**Inner-CV log loss** (`inner_cv_scores.csv`). The spread between configurations is small
(≈ 0.003 within a fold); the selected configuration is marked *.

| Config | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---:|---:|---:|---:|---:|
| A | 1.05321 | 1.05475 | 1.05260 | 1.05345 | 1.05512 |
| B | 1.05265 | 1.05388 | 1.05272 | 1.05272 | 1.05368 |
| C | 1.05283 | 1.05384 | 1.05183 | 1.05458 | 1.05327 |
| D | 1.05270 | 1.05462 | 1.05273 | 1.05331 | 1.05448 |
| E | **1.05195*** | 1.05339 | 1.05226 | 1.05263 | **1.05298*** |
| F | 1.05207 | **1.05280*** | **1.05171*** | **1.05244*** | 1.05302 |
| G | 1.05442 | 1.05481 | 1.05337 | 1.05538 | 1.05513 |
| H | 1.05289 | 1.05422 | 1.05264 | 1.05390 | 1.05424 |

## 13. Results

**Pooled out-of-fold results on 690,147 rows from 304 matches.**

| Model | Log loss | Brier | Accuracy | Macro-F1 | Weighted-F1 | Shot AP | Shot recall (TP / 7,212) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic F0 | 1.15850 | 0.52249 | 0.6505 | 0.1004 | 0.5497 | 0.0336 | 0 |
| Markov 2 | 1.06807 | 0.49375 | 0.6659 | 0.1204 | 0.5731 | 0.0527 | 0 |
| XGB-H2 | 1.06761 | 0.49365 | 0.6659 | 0.1208 | 0.5731 | 0.0529 | 0 |
| XGB-H5 | 1.05185 | **0.48798** | 0.6668 | **0.1325** | **0.5778** | 0.0769 | 12 |
| MLR-H5 | 1.10297 | 0.50569 | 0.6491 | 0.1219 | 0.5579 | 0.0585 | 1 |
| MLP-H5 | 1.05267 | 0.48838 | 0.6668 | 0.1302 | 0.5762 | 0.0757 | 13 |
| **LSTM-H5** | **1.05147** | 0.48803 | **0.6669** | 0.1298 | 0.5767 | **0.0808** | **28** (of 75 predicted) |

**Reproduced comparators.** The comparators' metrics were recomputed from their stored
predictions on the identical rows. They equal the values stored in experiment 6B to 1e-12.

**Per-class average precision.** Close to MLP-H5 and XGB-H5 for every class (within ±0.003),
with Shot the largest difference: 0.0808 vs 0.0757 (MLP-H5) and 0.0769 (XGB-H5).

**Argmax decisions for rare classes.** For six classes they are never made: Dribble, Miscontrol,
Dispossessed, Interception, Foul Committed and 50/50. This holds for every model in this
comparison and reflects unweighted maximum-likelihood training without decision thresholds.

## 14. Statistical comparisons

**Definition.** \(d_i = \ell_i^{\text{LSTM}} - \ell_i^{\text{ref}}\); negative values favour LSTM-H5.

| Reference | \(\bar d\) | rel. | 95 % CI | Folds LSTM-H5 better |
|---|---:|---:|---|---:|
| **MLP-H5** | −0.00120 | −0.11 % | [−0.00153, −0.00090] | 5 / 5 |
| **XGB-H5** | −0.00038 | −0.04 % | [−0.00069, −0.00009] | 4 / 5 |
| **MLR-H5** | −0.05150 | −4.67 % | [−0.05300, −0.05011] | 5 / 5 |
| Markov 2 | −0.01661 | −1.56 % | [−0.01740, −0.01589] | 5 / 5 |
| XGB-H2 | −0.01614 | −1.51 % | [−0.01693, −0.01543] | 5 / 5 |
| Logistic F0 | −0.10704 | −9.24 % | [−0.10969, −0.10470] | 5 / 5 |

**Per fold.**
- **vs MLP-H5.** −0.00098, −0.00053, −0.00178, −0.00083, −0.00190.
- **vs XGB-H5.** −0.00065, −0.00006, **+0.00018** (fold 2), −0.00059, −0.00082.

**Secondary metrics.**
- **vs MLP-H5.**
  - Brier −0.00036 [−0.00049, −0.00024].
  - Shot AP +0.0052 [+0.0025, +0.0081].
  - Accuracy and macro-F1: no detectable difference; both CIs contain 0.
- **vs XGB-H5.**
  - Accuracy and Brier: no detectable difference.
  - Macro-F1 −0.0027 [−0.0037, −0.0018], i.e. lower.
  - Shot AP +0.0039 [+0.0013, +0.0066].

**Seeds** (prespecified; replicate 0 primary):

| Replicate | LSTM-H5 log loss | − MLP-H5 (same replicate) | − XGB-H5 |
|---:|---:|---:|---:|
| 0 | 1.051465 | −0.00120 | −0.00038 |
| 1 | 1.051218 | −0.00094 | −0.00063 |
| 2 | 1.051344 | −0.00124 | −0.00050 |

- **Range.** The three LSTM-H5 seeds span 0.00025.
- **Against MLP-H5.** Every LSTM-H5 seed is better than every MLP-H5 seed (1.05215–1.05267).
- **Against XGB-H5.** Every seed is better than XGB-H5. The primary replicate is the least
  favourable of the three.

## 15. Independent validation

**Run integrity audit.** 70 checks in total (66 PASS, 4 INFO, 0 FAIL), covering:
- **Inputs and row identity.** Row identity, folds and manifest hashes against experiments 5,
  5B, 6A and 6B. The 5 × 16 one-hot input carries exactly the ids of the 79-column encoding.
- **Raw JSON.** A rebuild of 707 stratified rows and of 12 complete matches (27,265 rows) with
  the audited K1–K3 implementation of the 6A/6B audit. Histories and targets are identical.
- **Pre-run mathematics on real windows.** For all 8 architectures, the NumPy reference agrees
  with `nn.LSTM`: float32 ≤ 5.2e-8, float64 ≤ 6.9e-17. Parameter counts agree; finite
  differences agree to 3.0e-10.
- **Training records and access.** The early-stopping bookkeeping of all 175 fits. The data-access
  audit: no outer-validation match in any training, stopping or inner set; final test never used.
  All data sets are identical to experiment 6B.
- **Saved models.** A float64 NumPy forward pass of the saved weights reproduces all rows of all
  three replicates (≤ 5.6e-7). A gate-level audit of every fold's primary model gives
  \(h_s, c_s\) for \(s = 1..5\), the recovered output gate and the logits ≤ 5.3e-15.
- **Reproducibility.** A bitwise retraining of fold 0's primary refit in the main process
  (max |Δp| = 0).
- **History.** Historical checksums are unchanged.

**Independent verifier** (`scripts/verify_lstm_h5_experiment6c.py`). 59 checks, PASS. It uses
no PyTorch and no experiment module:
- an own NumPy LSTM on an independently built history (pandas `shift`) for all rows and seeds;
- shapes and parameter counts;
- the early-stopping bookkeeping and the selection recomputed;
- scikit-learn metrics;
- an exact reproduction of all six paired differences and CIs with an own seed-7 bootstrap,
  plus an independent 2,000-replicate bootstrap;
- the Pressure decomposition;
- the raw-JSON rebuilds;
- the historical checksums.

**Unit tests.** `tests/test_lstm_h5_experiment6c.py` has 39 tests:
- the equations, gate order, effective bias, gradients and BPTT reach;
- statelessness and the optimiser and dropout semantics;
- early stopping including epoch 0, checkpoint resume, and the equivalence of parallel and
  sequential folds;
- adversarial label mutation;
- synthetic order-dependence: a window permutation task that no bag-of-events model can solve,
  and copying \(E_{t-4}\). These are learnt above 90 % held-out accuracy. They show
  representational capacity only and say nothing about football.

**Final independent audit** (`scripts/audit_experiment6c.py`, `results/audit_experiment6c_2026-11-26/`,
2026-11-26): 66 checks, all PASS; 65 were executed with own code and 1 is by inspection.
- **Raw JSON.** A third, separately written K1–K3 reconstruction of 60 complete matches
  (138,830 rows) gives identical histories and targets.
- **Stored models.** An own NumPy LSTM reproduces all 15 stored models on all rows and seeds.
  Layer-wise states agree with PyTorch, and finite differences agree on the trained E and F
  models.
- **Optimiser.** The closed-form AdamW step matches the production optimiser.
- **Fits and statistics.** All 175 fits were re-scored from their stored best-epoch weights, and
  all data sets were checked against 6B. Metrics and the bootstrap were recomputed exactly.
- **Findings.** None affects results. Overstated wording in this chapter and in the result README
  was corrected (finding F1).

## 16. Scientific interpretation

**What the results support.**
- **vs MLP-H5.** With identical information, evaluation rows, training and stopping partitions,
  seed convention and training protocol, the selected stateless LSTM procedure is slightly but
  consistently better than the selected MLP procedure.
  - The difference is −0.00120 log loss (approximately −0.11 %): all 5 folds and all 3
    seed-matched replicates agree, and the CI excludes 0.
  - It is also better in Brier score and Shot AP.
- **What differs between the two.** The architecture, the parameter count, the candidate
  configurations and the selected configuration, and with them the optimisation. Which of these
  produces the difference is not identified by this experiment.
- **vs XGB-H5.** LSTM-H5 also has a lower pooled log loss than XGB-H5. The difference is
  −0.00038, CI [−0.00069, −0.00009], in 4 of 5 folds and for every seed. XGB-H5 has no detectable
  advantage in accuracy or Brier, but a higher macro-F1.
- **Where the difference arises (descriptive decomposition, §14).**
  - **vs XGB-H5.** The 85,965 rows whose current event is Pressure (12.46 % of the rows) have a
    conditional mean difference of −0.00642 and contribute −0.00080 to the pooled −0.00038. The
    other 14 current classes together contribute +0.00042, i.e. LSTM-H5 is marginally worse on
    them in aggregate.
  - **Pressure in other comparisons.** It is also where the additive MLR-H5 lost most (−0.0296
    of its −0.0515 deficit here). This is consistent with, but does not demonstrate, a role of
    interactions between history positions.
  - **Status.** This is descriptive evidence about *where* the models differ, not an
    explanation of *why*.
- **vs the Markov chains and MLR-H5.** The improvement over these models is of the same size as
  that of the other nonlinear H5 models, about −0.0166 vs Markov 2 and −0.0515 vs MLR-H5.

**Practical relevance.** The gains over MLP-H5 (approximately 0.11 %; 0.114 %) and over XGB-H5
(approximately 0.04 %; 0.036 %) are statistically detectable under the bootstrap, but very small:
- **Compared with other gaps.** The gap between the additive and the nonlinear H5 models is
  4.7 %, and the gain from H2 to H5 information about 1.5 %.
- **Among the nonlinear models.** On this input the three nonlinear H5 procedures (boosted trees,
  MLP, LSTM) differ by at most about 0.11 % in log loss. The choice of information (history
  length) matters far more than the choice among them.

**Shot.** The LSTM ranks Shot outcomes better (AP 0.0808 vs 0.0769 for XGB-H5 and 0.0757 for
MLP-H5).
- **Argmax recall is tiny.** It predicts Shot 75 times, 28 of them correctly, against 12 (XGB-H5)
  and 13 (MLP-H5). That is still only 28 of 7,212 Shot targets, a recall of 0.388 %.
- **Not a practical Shot detector.** This is not useful Shot detection: AP measures ranking
  quality over all thresholds, whereas argmax recall reflects a single decision rule that almost
  never selects a rare class.

**Possible explanations of the LSTM-H5 vs MLP-H5 difference** (hypotheses only, not tested here):
- **Inductive bias.** The recurrence applies one transition to every position and conditions
  each update on the current state. That is a natural bias for position-dependent interactions,
  for example the meaning of \(E_{t-1}\) given \(E_t\).
- **Capacity.** The selected LSTMs are of similar or larger size than the MLP (15,343 and 55,247
  vs 19,471 and 7,695 parameters).
- **Optimisation and search.** The training dynamics and the candidate configurations differ.

No dedicated experiment separates these. The model does **not** learn long-term football
dependencies: it sees five events.

**What the results support, overall.** A small predictive advantage of the selected LSTM-H5
procedure on this development population and task.

**Not established.**
- That recurrence is generally superior, or superior on other populations or tasks.
- That the ranking against XGB-H5 would hold with the same training population; see the
  limitations below.
- Equivalence or practical superiority.
- Any interpretation of gate activations as football behaviour.

## 17. Limitations

- **Training population vs XGB-H5.** This is audit finding F1, inherited from 6B. LSTM-H5's
  refits update on 89.8–90.5 % of the outer-training rows; XGB-H5 used 100 %. The LSTM-H5 vs
  XGB-H5 difference is a difference between complete procedures. This establishes a difference
  in data availability, not its effect: neither the magnitude nor the direction of that effect is
  known without a controlled refit.
  - **Possible follow-up.** A full-population refit of MLP-H5 and LSTM-H5 is possible as a
    separate experiment (not authorised, not run).
- **Five-event, event-type-only input.** No spatial, temporal or contextual information.
- **Restricted search.** 8 prespecified configurations, and the lr-3e-4 configurations were
  partly capped at 60 epochs. Neither larger networks nor other sequence architectures were
  explored, by design.
- **Seed variation.** The LSTM seed spread is 0.00025 and MLP-H5's 0.0005, comparable to the
  LSTM-H5 vs XGB-H5 difference (0.00038). The CIs do not include it.
- **Comparison with MLP-H5.** It shares partitions and protocol but not architecture, search
  space or parameter count (§1), so it is not an isolated test of recurrence.
- **Bootstrap scope.** The CIs cover match resampling of fixed out-of-fold predictions only.
  They do not cover refitting, the overlap of training sets between folds, hyperparameter
  selection, initialisation or the dataset's competition mix.
- **Reproducibility.** Bitwise on this machine, thread count and software only (Apple M1 CPU,
  Python 3.11.1, torch 2.15.0, NumPy 2.4.6). It is not claimed across platforms or versions.
- **Development data only.** The final test is untouched.

## 18. Reproduction

```bash
python scripts/run_lstm_h5_experiment6c.py --smoke
python scripts/supervise_lstm_h5_experiment6c.py --detach      # runner (fold workers) + verifier; caffeinate -d -i -w <supervisor>
python scripts/supervise_lstm_h5_experiment6c.py --status
python -m pytest tests/test_lstm_h5_experiment6c.py
```

**Seeds.** Training base 20261123, folds 20261027, bootstrap 7, permutation 11, raw
reconstruction 20261125.

**Source provenance.** The run started from commit `46d0e2d` with the 6C files uncommitted. The
executed sources are identified by `environment.json → source_files_sha256`, and were verified
byte-identical to the current files after the run.

## 19. Integrity verdict and conclusion

**Verdict.** **PASS WITH LIMITATIONS** (§17). Every audit passed with 0 FAIL:
- the run audit: 70 checks;
- the independent verifier: 59 checks;
- the final independent audit: 66 checks.

The experiment and the audit stages both exited 0.

**Conclusion.** Given the same five-event history, evaluation rows, folds, training and stopping
partitions and seed convention:
- **vs MLP-H5.** The selected stateless LSTM procedure is consistently, if slightly, better than
  the selected feed-forward MLP procedure: −0.00120 log loss, approximately 0.11 %.
- **vs XGB-H5.** It is marginally better than XGB-H5 (−0.00038, approximately 0.04 %). That
  comparison is subject to the difference in training-data availability, whose effect is unknown.
- **Implication.** On this development population, the LSTM procedure gives a small,
  statistically detectable improvement in next-event probability estimation. It does not change
  the overall picture: the nonlinear H5 procedures differ by at most about 0.11 %, far less than
  the gains from nonlinearity (≈ 4.7 %) and from a longer history (≈ 1.5 %).
- **Scope.** No claim of formal equivalence, universal superiority or long-term memory is made.

## References

- Gers, F. A., Schmidhuber, J., and Cummins, F. (2000). Learning to forget: Continual prediction
  with LSTM. *Neural Computation*, 12(10), 2451–2471. doi:10.1162/089976600300015015
- Hochreiter, S., and Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*,
  9(8), 1735–1780. doi:10.1162/neco.1997.9.8.1735
- Kingma, D. P., and Ba, J. (2015). Adam: A method for stochastic optimization. *3rd
  International Conference on Learning Representations (ICLR)*. arXiv:1412.6980
- Loshchilov, I., and Hutter, F. (2019). Decoupled weight decay regularization. *International
  Conference on Learning Representations (ICLR)*. arXiv:1711.05101
- PyTorch documentation, `torch.nn.LSTM`, `torch.optim.AdamW`,
  `torch.nn.functional.cross_entropy` (version 2.15.0, as installed).
- Werbos, P. J. (1990). Backpropagation through time: What it does and how to do it.
  *Proceedings of the IEEE*, 78(10), 1550–1560. doi:10.1109/5.58337
