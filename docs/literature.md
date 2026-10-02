# Gradient Inversion Against Speech Models in Federated Learning: Verified Literature Survey

Prepared 2026-10-02 for the rewrite of "Long-Form Speech Reconstruction from Gradients in Federated ASR using CTC Loss" (Khoo, Bui, Chin).

## How this was verified

- Every entry has a URL I opened in this session. Numbers come from the opened page, or from an ar5iv full-text fetch where I say so.
- I read two PDFs directly, page by page: Zeng and Rudzicz (Interspeech 2025) and Graves et al. (ICML 2006).
- Most other pages were read through a fetch tool that summarizes page text. Abstract-level facts are reliable. Check full-text details marked "(ar5iv)" against the PDF before quoting them in the paper.
- Venues are listed only when the opened page (or a Crossref / Semantic Scholar record) stated them. Otherwise the entry is cited as an arXiv preprint.
- Three pages would not open: IEEE Xplore for Ovi and Gangopadhyay, SpringerLink for Bui et al., and OpenReview. For the first two I verified bibliographic metadata through Crossref and say so in the entry.
- Passages labeled "my analysis" are my own derivations and are not claims from any paper.

---

## 1. Gradient inversion foundations

### 1.1 Core optimization attacks

**[DLG]** L. Zhu, Z. Liu, S. Han. "Deep Leakage from Gradients." NeurIPS 2019. https://arxiv.org/abs/1906.08935
Optimizes dummy inputs and labels so their gradient matches the shared gradient. The abstract reports pixel-wise accurate image recovery and token-wise matching text, and names gradient pruning as the most effective defense.
*Relevance:* the objective your attack instantiates. It needs the gradient of a gradient, which is why CTC must be twice differentiable.

**[iDLG]** B. Zhao, K. R. Mopuri, H. Bilen. "iDLG: Improved Deep Leakage from Gradients." arXiv:2001.02610, 2020. https://arxiv.org/abs/2001.02610
Shows the ground-truth label can be extracted analytically from the gradient for any differentiable model trained with cross-entropy over one-hot labels (single sample).
*Relevance:* the cross-entropy sign rule has a CTC analog through the occupancy term (Section 6). This is the starting point for transcript recovery.

**[InvGrad]** J. Geiping, H. Bauermeister, H. Dröge, M. Moeller. "Inverting Gradients - How easy is it to break privacy in federated learning?" NeurIPS 2020. https://arxiv.org/abs/2003.14053
Uses a magnitude-invariant (cosine) loss with adversarial-attack-style optimization. Proposition 3.1 (ar5iv) proves the input to a biased fully connected layer preceded only by fully connected layers is uniquely reconstructible. The paper (ar5iv) reports reconstruction quality "unimpeded" at 100 local steps, and that averaging 100 CIFAR-100 images still leaks.
*Relevance:* the cosine objective and the FC-layer proposition both apply to DeepSpeech1's per-frame FC stack.

**[GradInversion]** H. Yin, A. Mallya, A. Vahdat, J. M. Alvarez, J. Kautz, P. Molchanov. "See through Gradients: Image Batch Recovery via GradInversion." CVPR 2021. https://arxiv.org/abs/2104.07586
Recovers batches of 8 to 48 ImageNet images from ResNet-50 gradients, using fidelity regularization, batch label recovery, and group consistency across random seeds.
*Relevance:* group consistency across seeds is a cheap add-on for your overlapping temporal grids, since overlapped regions give free agreement constraints.

**[R-GAP]** J. Zhu, M. Blaschko. "R-GAP: Recursive Gradient Attack on Privacy." ICLR 2021. https://arxiv.org/abs/2010.07733
A closed-form recursive, layer-by-layer procedure to recover data from gradients, plus a rank analysis that estimates architecture-level risk.
*Relevance:* rank analysis is the right tool to predict at which utterance length each DeepSpeech layer stops leaking.

**[Phong17]** L. T. Phong, Y. Aono, T. Hayashi, L. Wang, S. Moriai. "Privacy-Preserving Deep Learning via Additively Homomorphic Encryption." IEEE TIFS 2018 (ePrint 2017/715). https://eprint.iacr.org/2017/715
The abstract points out that local data may leak to an honest-but-curious server through shared gradients, then proposes an encrypted aggregation fix. Usually cited as the origin of the single-layer leakage argument.
*Relevance:* historical anchor for the linear-layer leak.

**[CPL]** W. Wei, L. Liu, M. Loper, K.-H. Chow, M. E. Gursoy, S. Truex, Y. Wu. "A Framework for Evaluating Gradient Leakage Attacks in Federated Learning." arXiv:2004.10397, 2020. https://arxiv.org/abs/2004.10397
A framework for comparing client privacy leakage attacks across FL hyperparameters, attack settings, and gradient compression ratios.
*Relevance:* template for a hyperparameter sweep section (batch, local steps, compression).

**[Melis19]** L. Melis, C. Song, E. De Cristofaro, V. Shmatikov. "Exploiting Unintended Feature Leakage in Collaborative Learning." IEEE S&P 2019. https://arxiv.org/abs/1805.04049
Shows model updates leak membership and properties unrelated to the task, with passive and active attacks.
*Relevance:* foundation for the speaker and attribute inference line in Section 2.

### 1.2 Evaluation frameworks and realism critiques

**[Huang21]** Y. Huang, S. Gupta, Z. Song, K. Li, S. Arora. "Evaluating Gradient Inversion Attacks and Defenses in Federated Learning." NeurIPS 2021. https://arxiv.org/abs/2112.00059
Finds that several attacks rely on strong assumptions and weaken sharply when these are relaxed. Evaluates three defenses and shows that combining them works with minor utility loss.
*Relevance:* reviewers will apply this checklist. State which assumptions you make (known transcript, known length, model state, single utterance).

**[Hatamizadeh22]** A. Hatamizadeh, H. Yin, P. Molchanov, A. Myronenko, W. Li, P. Dogra, A. Feng, M. G. Flores, J. Kautz, D. Xu, H. R. Roth. "Do Gradient Inversion Attacks Make Federated Learning Unsafe?" IEEE Transactions on Medical Imaging. https://arxiv.org/abs/2202.06924
Challenges practical threat claims when clients update batch-norm statistics, and proposes reproducible leakage measures.
*Relevance:* DeepSpeech2 uses batch norm in its usual form, so say whether BN runs in train or eval mode during the attack.

**[SoK25]** J. Du, J. Hu, Z. Wang, P. Sun, N. Z. Gong, K. Ren, C. Chen. "SoK: On Gradient Leakage in Federated Learning." USENIX Security 2025. https://arxiv.org/abs/2404.05403
Systematizes gradient inversion attacks and concludes they are "notably constrained, fragile, and easily defensible" under realistic training setups, architectures, and post-processing.
*Relevance:* the strongest skeptical position you must answer.

**[FedLeak]** M. Fan, F. Wang, C. Chen, J. Zhou. "Boosting Gradient Leakage Attacks: Data Reconstruction in Realistic FL Settings." USENIX Security 2025. https://arxiv.org/abs/2506.08435
A counterpoint to the skeptics. Partial gradient matching plus gradient regularization gives high-fidelity reconstruction under a practical evaluation protocol.
*Relevance:* partial gradient matching (a subset of layers) directly reduces double-backward memory on large ASR models.

**[Valadi25]** V. Valadi, M. Åkesson, J. Östman, F. Hoseini, S. Toor, A. Hellander. "Practical Feasibility of Gradient Inversion Attacks in Federated Learning." arXiv:2508.19819, 2025 (revised 2026). https://arxiv.org/abs/2508.19819
Finds that modern performance-optimized vision models resist visually meaningful reconstruction, and that many reported successes use upper-bound settings such as inference mode.
*Relevance:* motivates testing a modern architecture (wav2vec2 or Conformer CTC) and not only DeepSpeech.

**[Zhu24]** H. Zhu, L. Huang, Z. Xie. "Privacy Attack in Federated Learning is Not Easy: An Experimental Study." arXiv:2409.19301, 2024. https://arxiv.org/abs/2409.19301
Empirical study concluding that no evaluated state-of-the-art attack breaches client data in realistic FL settings.
*Relevance:* same as above. Cite when justifying your threat model.

### 1.3 FedAvg, multi-step, and batch settings

**[Dimitrov22]** D. I. Dimitrov, M. Balunović, N. Konstantinov, M. Vechev. "Data Leakage in Federated Averaging." TMLR 2022 (venue per Semantic Scholar). https://arxiv.org/abs/2206.12395
Attacks FedAvg by differentiating through a simulation of the client's local updates, with a permutation-invariant prior across epochs and per-step label estimation. On FEMNIST it recovers more than 45% of client images from 10 local epochs of 10 batches of 5 images, against under 10% for the baseline.
*Relevance:* the direct recipe for a FedAvg variant of your attack.

**[Geng21]** J. Geng, Y. Mou, F. Li, Q. Li, O. Beyan, S. Decker, C. Rong. "Towards General Deep Leakage in Federated Learning." arXiv:2110.09074, 2021. https://arxiv.org/abs/2110.09074
Reconstructs from FedSGD gradients and FedAvg weights, with zero-shot label restoration that handles duplicate labels. Reports that image restoration fails with even one wrong label in the batch.
*Relevance:* a warning that transcript errors may break feature reconstruction. Worth quantifying for CTC.

**[NL-SME]** L. Xia, J. Yu, Z. Liu, S. Huang, W. Tang, X. Liu. "Trajectory-Aware Information Matching for Multi-Step Gradient Inversion in Federated Learning." arXiv:2509.22082, 2025. https://arxiv.org/abs/2509.22082
Fits a learnable nonlinear surrogate trajectory to approximate hidden local states under multi-step FedAvg, and reports better reconstruction than endpoint matching.
*Relevance:* a cheaper alternative to full unrolling for multi-step ASR updates.

### 1.4 Generative and learned priors

**[GIAS]** J. Jeon, J. Kim, K. Lee, S. Oh, J. Ok. "Gradient Inversion with Generative Image Prior." NeurIPS 2021. https://arxiv.org/abs/2110.14962
Optimizes in the latent and parameter space of a pretrained generator, and shows a prior can also be learned from the sequence of gradients seen during FL.
*Relevance:* blueprint for searching a vocoder or codec latent in place of raw MFCC frames.

**[GGL]** Z. Li, J. Zhang, L. Liu, J. Liu. "Auditing Privacy Defenses in Federated Learning via Generative Gradient Leakage." CVPR 2022. https://arxiv.org/abs/2203.15696
Uses a GAN latent prior with gradient-free optimizers (evolution strategies, Bayesian optimization) to reconstruct under noise and compression defenses.
*Relevance:* gradient-free search over a low-dimensional latent avoids the CTC double backward entirely.

**[GIFD]** H. Fang, B. Chen, X. Wang, Z. Wang, S.-T. Xia. "GIFD: A Generative Gradient Inversion Method with Feature Domain Optimization." ICCV 2023. https://arxiv.org/abs/2308.04699
Moves optimization progressively from GAN latent space to intermediate generator features, with regularization against unrealistic outputs.
*Relevance:* the same schedule applies to vocoder intermediate features.

**[Li24-diff]** Z. Li, A. Lowy, J. Liu, T. Koike-Akino, B. Malin, K. Parsons, Y. Wang. "Exploring User-level Gradient Inversion with a Diffusion Prior." arXiv:2409.07291 (NeurIPS 2023 FL workshop). https://arxiv.org/abs/2409.07291
Uses a denoising diffusion prior to recover a representative image that captures user-level semantics in large batches.
*Relevance:* a user-level goal ("recover the speaker's voice") may be more attainable than exact utterances at large batch.

**[Meng25]** J. Meng, T. Huang, H. Chen, C. Hou, G. Zheng. "Enhanced Privacy Leakage from Noise-Perturbed Gradients via Gradient-Guided Conditional Diffusion Models." arXiv:2511.10423, 2025. https://arxiv.org/abs/2511.10423
Gradient-guided conditional diffusion reconstructs from Gaussian-noised gradients, with error bounds.
*Relevance:* an adaptive attacker for your DP noise sweep.

**[GUIDE]** V. Carletti, P. Foggia, C. Mazzocca, G. Parrella, M. Vento. "GUIDE: Enhancing Gradient Inversion Attacks in Federated Learning with Denoising Models." arXiv:2510.17621, 2025. https://arxiv.org/abs/2510.17621
Post-processes noisy reconstructions with a diffusion denoiser. Reports up to 46% higher perceptual similarity (DreamSim).
*Relevance:* the lowest-effort prior. Run a speech enhancement or diffusion model on reconstructed features after the attack.

**[L2I]** R. Wu, X. Chen, C. Guo, K. Q. Weinberger. "Learning to Invert: Simple Adaptive Attacks for Gradient Inversion in Federated Learning." UAI 2023 (PMLR v216). https://arxiv.org/abs/2210.10880
Trains a model to map gradients to inputs using auxiliary data, and breaks compression-style defenses on vision and language tasks.
*Relevance:* a learned inverter over LibriSpeech gradients is feasible for short chunks and adapts to defenses.

**[Bayes]** M. Balunović, D. I. Dimitrov, R. Staab, M. Vechev. "Bayesian Framework for Gradient Leakage." arXiv:2111.04706, 2021. https://arxiv.org/abs/2111.04706
Frames existing attacks as approximations of a Bayes-optimal adversary. Shows several heuristic defenses fail against stronger attacks, especially early in training.
*Relevance:* justifies evaluating defenses with an adaptive attacker and at several training stages.

### 1.5 Analytic, linear-layer, and malicious-server attacks

**[RobFed]** L. Fowl, J. Geiping, W. Czaja, M. Goldblum, T. Goldstein. "Robbing the Fed: Directly Obtaining Private Data in Federated Learning with Modified Models." arXiv:2110.13057, 2021. https://arxiv.org/abs/2110.13057
A malicious server makes minimal architecture modifications that let it read verbatim user data from gradient updates, even for large batches.
*Relevance:* DeepSpeech1 already starts with FC plus ReLU on MFCC windows, so an imprint-style readout needs no architectural change.

**[Fishing]** Y. Wen, J. Geiping, L. Fowl, M. Goldblum, T. Goldstein. "Fishing for User Data in Large-Batch Federated Learning via Gradient Magnification." ICML 2022. https://arxiv.org/abs/2202.00580
Parameter-only manipulation that magnifies one sample's gradient so existing attacks work on arbitrarily large batches.
*Relevance:* a route to batch > 1 without solving the mixing problem.

**[Boenisch]** F. Boenisch, A. Dziedzic, R. Schuster, A. S. Shamsabadi, I. Shumailov, N. Papernot. "When the Curious Abandon Honesty: Federated Learning Is Not Private." arXiv:2112.02918, 2021. https://arxiv.org/abs/2112.02918
"Trap weights" make individual samples extractable with near-zero cost. Perfectly reconstructs more than 50% of ImageNet points from mini-batches of 100.
*Relevance:* same as RobFed. A strong malicious-server baseline for the first FC layer.

**[CPA]** S. Kariyappa, C. Guo, K. Maeng, W. Xiong, G. E. Suh, M. K. Qureshi, H.-H. S. Lee. "Cocktail Party Attack: Breaking Aggregation-Based Privacy in Federated Learning using Independent Component Analysis." ICML (venue per Semantic Scholar; arXiv 2022). https://arxiv.org/abs/2209.05578
Treats the aggregate FC-layer gradient as a linear mixture of inputs and applies ICA to unmix. Works at batch sizes up to 1024.
*Relevance:* frames in an utterance are the "batch" for a per-frame FC layer. ICA on the first-layer gradient is a candidate for separating frames.

**[SPEAR]** D. I. Dimitrov, M. Baader, M. N. Müller, M. Vechev. "SPEAR: Exact Gradient Inversion of Batches in Federated Learning." NeurIPS 2024. https://arxiv.org/abs/2403.03945
Uses the factorization dL/dW = (dL/dZ) X^T (rank at most b) plus ReLU-induced sparsity of dL/dZ to recover whole batches exactly in the honest-but-curious setting. Reports exact recovery up to b ≲ 25 on ImageNet-sized inputs.
*Relevance:* the low-rank factorization holds for DeepSpeech's per-frame FC plus ReLU layers. Exact disaggregation is exponential in b, so it suits short chunks.

**[LOKI]** J. C. Zhao, A. Sharma, A. R. Elkordy, Y. H. Ezzeldin, S. Avestimehr, S. Bagchi. "LOKI: Large-scale Data Reconstruction Attack against Federated Learning through Model Manipulation." IEEE S&P 2024. https://arxiv.org/abs/2303.12233
Manipulated convolution parameters leak data through FedAvg and secure aggregation. Recovers 76 to 86% of samples with 100 aggregated clients.
*Relevance:* the secure-aggregation threat model for DeepSpeech2's conv front end.

**[SEER]** K. Garov, D. I. Dimitrov, N. Jovanović, M. Vechev. "Hiding in Plain Sight: Disguising Data Stealing Attacks in Federated Learning." arXiv:2306.03013, 2023. https://arxiv.org/abs/2306.03013
A malicious-server attack with a secret decoder trained jointly with the shared model. Works at batch sizes up to 512 and under secure aggregation.
*Relevance:* a hard-to-detect upper bound for batch settings.

**[ARES]** Z. Gong, L. Y. Zhang, Y. Zhang, V. Vo, T. Zhu, S. Pan, C. Wang. "ARES: Scalable and Practical Gradient Inversion Attack in Federated Learning through Activation Recovery." IEEE S&P 2026. https://arxiv.org/abs/2603.17623
An active attack with no architecture change. Casts activation recovery as noisy sparse recovery solved with generalized Lasso.
*Relevance:* sparse recovery of intermediate activations is another formulation for separating frames.

**[VGIA]** F. Diana, C. Xu, A. Nusser, G. Neglia. "No More Guessing: a Verifiable Gradient Inversion Attack in Federated Learning." arXiv:2604.15063, 2026. https://arxiv.org/abs/2604.15063
Gives an algebraic subspace test that certifies when a ReLU hyperplane region isolates exactly one record, then recovers it analytically (tabular data).
*Relevance:* a certificate of correctness for a recovered frame would be new for audio.

**[LeakLearn]** J. C. Zhao, A. Dabholkar, A. Sharma, S. Bagchi. "Leak and Learn: An Attacker's Cookbook to Train Using Leaked Data from Federated Learning." CVPR 2024. https://arxiv.org/abs/2403.18144
Evaluates whether data leaked through gradient inversion and linear-layer leakage is good enough to train models on.
*Relevance:* a downstream-utility metric. Train a speaker or ASR model on your reconstructions.

### 1.6 Label recovery

**[LLG]** A. Wainakh, F. Ventola, T. Müßig, J. Keim, C. Garcia Cordero, E. Zimmer, T. Grube, K. Kersting, M. Mühlhäuser. "User-Level Label Leakage from Gradients in Federated Learning." PETS 2022. https://arxiv.org/abs/2105.09369
Uses direction and magnitude of last-layer gradients to infer which labels are present and how often, for arbitrary batch sizes. Works best early in training. Gradient compression mitigates it.
*Relevance:* the magnitude-to-count idea is what a CTC character-histogram attack needs.

### 1.7 Surveys

**[Zhang22]** R. Zhang, S. Guo, J. Wang, X. Xie, D. Tao. "A Survey on Gradient Inversion: Attacks, Defenses and Future Directions." IJCAI 2022. https://arxiv.org/abs/2206.07284
Taxonomy of iteration-based and recursion-based attacks, with defenses grouped into data obscuration, model improvement, and gradient protection.

**[Guo25]** P. Guo, R. Wang, S. Zeng, J. Zhu, H. Jiang, Y. Wang, Y. Zhou, F. Wang, H. Xiong, L. Qu. "Exploring the Vulnerabilities of Federated Learning: A Deep Dive into Gradient Inversion Attacks." IEEE TPAMI (accepted; arXiv 2025). https://arxiv.org/abs/2503.11514
Groups attacks into optimization-based, generation-based, and analytics-based. Finds the optimization-based type the most practical setting despite weaker performance, and proposes a three-stage defense pipeline.

**[Li24-inf]** Z. Li, A. Lowy, J. Liu, T. Koike-Akino, K. Parsons, B. Malin, Y. Wang. "Analyzing Inference Privacy Risks Through Gradients in Machine Learning." arXiv:2408.16913, 2024. https://arxiv.org/abs/2408.16913
A game-based framework for attribute, property, distributional, and user inference from gradients across five datasets. Evaluates five defenses under static and adaptive adversaries.

*Relevance of the surveys:* none of the abstracts mention speech, which supports the claim that speech is under-studied in this literature.

---

## 2. Speech-specific gradient leakage

### 2.1 Reconstruction of speech features or audio from gradients

**[Dang-Spk]** T. Dang, O. Thakkar, S. Ramaswamy, R. Mathews, P. Chin, F. Beaufays. "A Method to Reveal Speaker Identity in Distributed ASR Training, and How to Counter It." ICASSP 2022. https://arxiv.org/abs/2104.07815
- **Method:** Hessian-Free Gradients Matching reconstructs DeepSpeech input features from a gradient without second derivatives of the loss.
- **Headline:** speaker identity revealed with 34% top-1 (51% top-5) accuracy on LibriSpeech. Dropout 0.2 reduces this to 0% top-1 (0.5% top-5).
- **Details (ar5iv):** 26-dimensional MFCCs, CTC loss, zeroth-order direct search over random unit directions, utterances of 1 to 4 s, and the attacker knows the transcript and length.
- **Why zeroth-order (ar5iv):** backpropagating twice through the CTC dynamic program was found "intractable", and CTC second derivatives were not implemented in TensorFlow or PyTorch.
- **DP-SGD (ar5iv):** noise at σ ≥ 1e-3 drives top-1 to about 0% but nearly doubles WER.

*Relevance:* your closest predecessor. A twice-differentiable CTC replaces its zeroth-order search with exact first-order matching and extends past 4 s. Its known-transcript and known-length assumptions are the ones to remove.

**[Bui25]** M. N. Bui, T. Dang, P. Cherian, T. D. Tran, P. Chin. "Reconstructing Speech Features of Automatic Speech Recognition Systems in Federated Learning by Gradient Descent." Complex Networks & Their Applications XIII (COMPLEX NETWORKS 2024), Studies in Computational Intelligence, Springer, 2025, pp. 125-134. https://doi.org/10.1007/978-3-031-82427-2_11
Metadata verified through Crossref (https://api.crossref.org/works/10.1007/978-3-031-82427-2_11). SpringerLink required a login, so I did not read the abstract or body. Content is unverified beyond the title.
*Relevance:* from your own group. Differentiate the new paper from it explicitly (long-form, overlapping grids, the CTC formulation).

**[Li23]** Z. Li, J. Zhang, J. Liu. "Speech Privacy Leakage from Shared Gradients in Distributed Learning." ICASSP 2023. https://arxiv.org/abs/2302.10441
Gradient matching (Euclidean loss, anisotropic TV, Adam for 8,000 iterations) on a small keyword-spotting CNN over roughly 1 s Speech Commands clips. Waveforms come from Griffin-Lim. Results (ar5iv):

| Feature | PESQ | STOI | Speaker re-ID vs original signal |
|---|---|---|---|
| Mel-spectrogram | 2.04 | 0.80 | 90.5% |
| MFCC | 1.39 | 0.43 | 2.5% |

*Relevance:* MFCC inversion through Griffin-Lim is the weak link for perceptual quality and speaker leakage. This is the gap a neural vocoder prior should close.

**[Ovi24]** P. R. Ovi, A. Gangopadhyay. "Gradient Inversion Attacks on Acoustic Signals: Revealing Security Risks in Audio Recognition Systems." ICASSP 2024, pp. 4835-4839. https://doi.org/10.1109/ICASSP48485.2024.10445809
Metadata verified through Crossref. IEEE Xplore returned no content, so I did not read the abstract. Zeng and Rudzicz use it as a baseline and report it at 15.1 dB SNR on AudioMNIST and 12.2 dB on UrbanSound8K (their Table 1).
*Relevance:* a short-clip classifier baseline. Cite with Li23 as the "1 s clips" prior art.

**[Zeng25]** X. Zeng, F. Rudzicz. "How to Recover Long Audio Sequences Through Gradient Inversion Attack With Dynamic Segment-based Reconstruction." Interspeech 2025, pp. 5118-5122. https://www.isca-archive.org/interspeech_2025/zeng25_interspeech.html
Read in full from the PDF.
- **Setup:** a 4-layer CNN classifier with cross-entropy and a dummy label, 128-bin mel-spectrograms, L-BFGS, Griffin-Lim plus a low-pass filter.
- **Method:** splits the mel-spectrogram into dynamically sized segments reconstructed in parallel. The batch variant adds spectral and smoothness losses.
- **Results:** SNR 47.7 dB against 32.5 dB for Li23 on Speech Commands. A 139.43 s CommonVoice clip at 40.1 dB SNR (mel cosine 0.992). A 34.96 s LibriSpeech clip at 37.6 dB.
- **Batch results:** up to 16 for CommonVoice and LibriSpeech. Whisper WER 0.189 (CommonVoice, B=16) and 0.409 (LibriSpeech, B=16).
- **Claim:** "the first successful high-quality reconstruction of clips up to 140 seconds".
- **Stated future work:** MFCC features and defenses.

*Relevance:* this is the paper your "long-form" claim collides with. Your differentiators are a real ASR model with a sequence loss (CTC), recurrent layers, MFCC inputs, and no dummy-label classifier. A head-to-head comparison is mandatory.

**[Arasteh25]** S. Tayebi Arasteh, M. Lotfinia, P. A. Perez-Toro, T. Arias-Vergara, M. Ranji, J. R. Orozco-Arroyave, M. Schuster, A. Maier, S. H. Yang. "Differential privacy enables fair and accurate AI-based analysis of speech disorders while protecting patient data." npj Artificial Intelligence 1, 37 (2025). https://arxiv.org/abs/2409.19078
Trains pathological-speech classifiers with DP (maximum accuracy reduction of 3.85% at high privacy levels). Demonstrates that non-private models are vulnerable to gradient inversion that reconstructs identifiable speech samples, and that DP mitigates it. I did not read the attack details.
*Relevance:* one of the few papers pairing a speech gradient inversion demo with a DP utility number, though on a classifier.

### 2.2 Transcript and label leakage

**[Dang-RLG]** T. Dang, O. Thakkar, S. Ramaswamy, R. Mathews, P. Chin, F. Beaufays. "Revealing and Protecting Labels in Distributed Training." NeurIPS 2021. https://arxiv.org/abs/2111.00556
- **Method:** recovers the set of labels in a batch from only the last-layer gradient, using ΔW = h^T g, an SVD, and a per-class linear program.
- **Assumption (ar5iv):** S < min{d, C}, where S is the number of label positions, d the hidden dimension, and C the number of classes.
- **ASR results (ar5iv):** demonstrated on an attention-based encoder-decoder with 16k word pieces. Bag-of-words exact match of 98% (untrained), 95.5% (4k steps), and 93.3% (10k steps).
- **Effect on reconstruction (ar5iv):** with the bag of words, gradient matching reconstructs over 50% of utterance transcripts on the first run.
- **Defenses (ar5iv):** Sign-SGD drops ASR exact match to 10.4%. 90% gradient dropping leaves it at 86.6%.

*Relevance:* gives the token set but neither counts nor order, and it was shown on an attention model. For character CTC the condition S < min{d, C} fails (C is about 29, and T is far larger). That is my analysis, and it leaves CTC transcript recovery open.

### 2.3 Speaker and attribute leakage from updates

**[Tomashenko22]** N. Tomashenko, S. Mdhaffar, M. Tommasi, Y. Estève, J.-F. Bonastre. "Privacy attacks for automatic speech recognition acoustic models in a federated learning framework." ICASSP 2022, pp. 6972-6976. https://arxiv.org/abs/2111.03777
Compares personalized acoustic models through their "footprint" on an indicator dataset. Two attack models reach a speaker-verification EER of 1 to 2% on TED-LIUM 3 with no access to user speech.
*Relevance:* speaker identity leaks from weight deltas without reconstruction. A baseline to beat or complement with speaker ID on reconstructed audio.

**[Mdhaffar21]** S. Mdhaffar, J.-F. Bonastre, M. Tommasi, N. Tomashenko, Y. Estève. "Retrieving Speaker Information from Personalized Acoustic Models for Speech Recognition." arXiv:2111.04194, 2021. https://arxiv.org/abs/2111.04194
Uses weight-matrix changes of locally adapted HMM/TDNN models on TED-LIUM 3. Reports 95% gender accuracy and 9.07% speaker-verification EER. Gender sits in the first layers and identity in the middle-to-upper layers.
*Relevance:* layer-wise localization suggests which layers' gradients to weight in the matching loss.

**[Nguyen23]** T. Nguyen, S. Mdhaffar, N. Tomashenko, J.-F. Bonastre, Y. Estève. "Federated Learning for ASR based on Wav2vec 2.0." ICASSP 2023. https://arxiv.org/abs/2302.10790
Federated fine-tuning of wav2vec 2.0 reaches 10.92% WER on TED-LIUM 3 with no language model. A layer-wise footprint analysis shows which exchanged layers carry speaker identity.
*Relevance:* establishes wav2vec2 FL as a realistic target and provides the utility baseline for an attack on wav2vec2-CTC.

**[Feng21]** T. Feng, H. Hashemi, R. Hebbar, M. Annavaram, S. S. Narayanan. "Attribute Inference Attack of Speech Emotion Recognition in Federated Learning Settings." arXiv:2112.13416, 2021. https://arxiv.org/abs/2112.13416
Infers client gender from shared gradients (FedSGD) or parameters (FedAvg) on IEMOCAP, CREMA-D, and MSP-Improv. Most leakage comes from the first layer.
*Relevance:* shows a FedSGD-versus-FedAvg comparison in speech, and the first-layer finding matches the first-FC-layer argument in Section 6.

**[Feng22]** T. Feng, R. Peri, S. Narayanan. "User-Level Differential Privacy against Attribute Inference Attack of Speech Emotion Recognition in Federated Learning." Interspeech 2022. https://arxiv.org/abs/2204.02500
User-level DP reduces attribute leakage while keeping utility when the adversary sees one update. Protection weakens as more updates leak.
*Relevance:* multi-round observation is an attacker lever you can also test, since the same utterance is seen across rounds.

**[Tan24]** C. Tan, S. Li, Y. Cao, Z. Ren, T. Schultz. "Investigating Effective Speaker Property Privacy Protection in Federated Learning for Speech Emotion Recognition." arXiv:2410.13221, 2024. https://arxiv.org/abs/2410.13221
Decomposes speech properties and perturbs them to defend against property inference, with better privacy-utility trade-offs than the baselines.
*Relevance:* an input-side defense to include in the sweep.

**[AlAli25]** H. Al-Ali, A. R. Ghavamipour, T. Caselli, F. Turkmen, Z. Talat, H. Aldarmaki. "Personal Attribute Leakage in Federated Speech Models." arXiv:2510.13357, 2025. https://arxiv.org/abs/2510.13357
A passive, non-parametric white-box attribute inference attack on weight differentials of Wav2Vec2, HuBERT, and Whisper. Covers gender, age, accent, emotion, and dysarthria. Accent is reliably inferable in all three models.
*Relevance:* the only work I found attacking modern speech foundation models in FL, and it stops at attributes.

### 2.4 Related training-data leakage in ASR (not gradient-based)

**[Amid22]** E. Amid, O. Thakkar, A. Narayanan, R. Mathews, F. Beaufays. "Extracting Targeted Training Data from ASR Models, and How to Mitigate It." Interspeech 2022. https://arxiv.org/abs/2204.08345
Noise Masking extracts names from LibriSpeech-trained Conformer models: 11.8% correct-name accuracy, and some training-set name 55.2% of the time. Word Dropout plus multistyle training mitigates it.
*Relevance:* a memorization threat to contrast with gradient leakage in related work.

**[Wang23]** L. Wang, O. Thakkar, R. Mathews. "Unintended Memorization in Large ASR Models, and How to Mitigate It." arXiv:2310.11739, 2023. https://arxiv.org/abs/2310.11739
Audits memorization with sped-up random utterances. Per-example gradient clipping mitigates memorization for up to 16 repetitions, and per-core average clipping keeps quality neutral.
*Relevance:* clipping-only (no noise) is a deployed mitigation, so include it in the defense sweep.

**[Jagielski24]** M. Jagielski, O. Thakkar, L. Wang. "Noise Masking Attacks and Defenses for Pretrained Speech Models." ICASSP 2024. https://arxiv.org/abs/2404.02052
Extends noise masking to pretrained speech encoders by fine-tuning them into ASR models.
*Relevance:* context for SSL speech privacy.

**[Tseng22]** W.-C. Tseng, W.-T. Kao, H.-y. Lee. "Membership Inference Attacks Against Self-supervised Speech Models." Interspeech 2022. https://arxiv.org/abs/2111.05113
Black-box membership inference at utterance and speaker level against SSL speech models.
*Relevance:* context only.

---

## 3. Text and sequence gradient inversion that could transfer

**[TAG]** J. Deng, Y. Wang, J. Li, C. Shang, H. Liu, S. Rajasekaran, C. Ding. "TAG: Gradient Attack on Transformer-based Language Models." Findings of EMNLP 2021. https://arxiv.org/abs/2103.06819
The first gradient-matching attack formulated for transformer language models, evaluated on BERT variants and GLUE.
*Relevance:* its distance choice is a baseline alongside cosine and L2 for CTC models.

**[LAMP]** M. Balunović, D. I. Dimitrov, N. Jovanović, M. Vechev. "LAMP: Extracting Text from Gradients with Language Model Priors." NeurIPS 2022. https://arxiv.org/abs/2202.08827
Alternates continuous embedding optimization with discrete transformations scored by a language-model prior. Reconstructs 5x more bigrams and 23% longer subsequences than prior work, and is the first to handle text batch sizes above 1.
*Relevance:* the template for jointly recovering the transcript with an LM prior while optimizing features.

**[FILM]** S. Gupta, Y. Huang, Z. Zhong, T. Gao, K. Li, D. Chen. "Recovering Private Text in Federated Learning of Language Models." NeurIPS 2022. https://arxiv.org/abs/2205.08514
First recovers the bag of words from gradients, then orders it with beam search and a prior-based reordering, for batches up to 128 sentences. Freezing word embeddings defends with minimal utility loss.
*Relevance:* bag-then-order is the pipeline to copy, starting from an RLG-style or bias-gradient character set.

**[Decepticons]** L. Fowl, J. Geiping, S. Reich, Y. Wen, W. Czaja, M. Goldblum, T. Goldstein. "Decepticons: Corrupted Transformers Breach Privacy in Federated Learning for Language Models." ICLR 2023. https://arxiv.org/abs/2201.12675
Malicious parameter vectors separately extract tokens and positional embeddings, working with mini-batches, multiple users, and long sequences.
*Relevance:* a malicious-server route for transformer speech encoders.

**[Panning]** H.-M. Chu, J. Geiping, L. H. Fowl, M. Goldblum, T. Goldstein. "Panning for Gold in Federated Learning: Targeted Text Extraction under Arbitrarily Large-Scale Aggregation." ICLR 2023. https://iclr.cc/virtual/2023/poster/10906
Maliciously modified parameters make the transformer filter sequences containing target phrases and encode them in the update, under very large aggregation.
*Relevance:* targeted extraction (for example, utterances containing a keyword) is a speech analog nobody has built.

**[APRIL]** J. Lu, X. S. Zhang, T. Zhao, X. He, J. Cheng. "APRIL: Finding the Achilles' Heel on Privacy for Vision Transformers." arXiv:2112.14087, 2021. https://arxiv.org/abs/2112.14087
Analyzes gradient leakage of self-attention in theory and practice, and proposes an attack on ViT-style models.
*Relevance:* carries over to transformer speech encoders with learned or convolutional positional embeddings.

**[DAGER]** I. Petrov, D. I. Dimitrov, M. Baader, M. N. Müller, M. Vechev. "DAGER: Exact Gradient Inversion for Large Language Models." NeurIPS 2024. https://arxiv.org/abs/2405.15586
- **Theory (ar5iv):** Theorem 3.1 gives dL/dW = X^T dL/dY with rank at most b. Theorem 5.1 says the row span of a layer's inputs equals the column span of dL/dW^Q when b < d and dL/dQ is full rank.
- **Method:** a span check tests each vocabulary token for membership in the gradient's column space.
- **Results:** exact recovery of batches up to 128, ROUGE-1/2 above 0.99, and 20x faster than prior attacks at the same batch size.
- **Limitation (ar5iv):** degrades when total tokens exceed the hidden dimension.

*Relevance:* the rank theorem holds for any linear layer shared across time. The discrete span check does not transfer to continuous frames, which is where TIGER comes in.

**[TIGER]** W. Kalikman, I. Petrov, D. I. Dimitrov, M. Vechev. "TIGER: Inverting Transformer Gradients via Embedding-Subspace Distance Optimization." arXiv:2606.18312, 2026. https://arxiv.org/abs/2606.18312
Turns the DAGER subspace signal into a differentiable objective by optimizing embeddings to minimize distance to the gradient subspace. More robust to numerical noise, with the first successful reconstructions under DP-defended FL for decoder models.
*Relevance:* the most transferable recent idea. A subspace-distance loss works for continuous inputs such as MFCC frames and needs no CTC double backward.

**[Li24-partial]** W. Li, Q. Xu, M. Dras. "Seeing the Forest through the Trees: Data Leakage from Partial Transformer Gradients." EMNLP 2024. https://arxiv.org/abs/2406.00999
Gradients from a single transformer layer, or a single linear component with 0.54% of parameters, suffice for text reconstruction. DP offers limited protection.
*Relevance:* justifies matching only a few layers to cut memory on wav2vec2-scale models.

**[Li23-pooler]** J. Li, S. Liu, Q. Lei. "Beyond Gradient and Priors in Privacy Attacks: Leveraging Pooler Layer Inputs of Language Models in Federated Learning." arXiv:2312.05720, 2023. https://arxiv.org/abs/2312.05720
A two-stage attack that first recovers intermediate feature directions analytically, then uses them as extra supervision for gradient matching.
*Relevance:* the same two-stage pattern (analytic intermediate features, then optimization) fits the FC layers of DeepSpeech.

**[FET]** Y. Gao, Y. Xie, H. Deng, Z. Zhu. "Gradient Inversion Attack in Federated Learning: Exposing Text Data through Discrete Optimization." COLING 2025, pp. 2582-2591. https://aclanthology.org/2025.coling-main.176/
Searches directly over discrete token sequences with global and local search. Reports exact-match gains of 39% (TinyBERT-6), 20% (BERT-base), and 15% (BERT-large).
*Relevance:* discrete search over transcripts scored by gradient distance is a fallback when bias-gradient recovery is ambiguous.

---

## 4. Modern speech models under FL, and generative priors for audio

### 4.1 Target models and FL practice

**[DS1]** A. Hannun, C. Case, J. Casper, B. Catanzaro, G. Diamos, E. Elsen, R. Prenger, S. Satheesh, S. Sengupta, A. Coates, A. Y. Ng. "Deep Speech: Scaling up end-to-end speech recognition." arXiv:1412.5567, 2014. https://arxiv.org/abs/1412.5567

**[DS2]** D. Amodei et al. (34 authors, full list in BibTeX). "Deep Speech 2: End-to-End Speech Recognition in English and Mandarin." arXiv:1512.02595, 2015. https://arxiv.org/abs/1512.02595

**[w2v2]** A. Baevski, H. Zhou, A. Mohamed, M. Auli. "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." arXiv:2006.11477, 2020. https://arxiv.org/abs/2006.11477

**[HuBERT]** W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, A. Mohamed. "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units." arXiv:2106.07447, 2021. https://arxiv.org/abs/2106.07447

**[Whisper]** A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, I. Sutskever. "Robust Speech Recognition via Large-Scale Weak Supervision." ICML 2023. https://arxiv.org/abs/2212.04356

**[Conformer]** A. Gulati, J. Qin, C.-C. Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han, S. Wang, Z. Zhang, Y. Wu, R. Pang. "Conformer: Convolution-augmented Transformer for Speech Recognition." arXiv:2005.08100 (submitted to Interspeech 2020). https://arxiv.org/abs/2005.08100

*Relevance of the six model papers:* DS1 and DS2 are your current targets. wav2vec2 and HuBERT with a CTC head are the natural next targets. Whisper (attention decoder) and Conformer cover the non-CTC and production cases.

**[Guliani21]** D. Guliani, F. Beaufays, G. Motta. "Training Speech Recognition Models with Federated Learning: A Quality/Cost Framework." ICASSP 2021. https://arxiv.org/abs/2010.15965
A framework that varies the degree of non-IID-ness and trades quality against cost. Variational noise and hyperparameter tuning compensate for non-IID effects.
*Relevance:* variational (weight) noise is a standard FL-ASR ingredient that perturbs the model the client differentiates. Test whether it hurts your attack.

**[Gao21]** Y. Gao, T. Parcollet, S. Zaiem, J. Fernandez-Marques, P. P. B. de Gusmao, D. J. Beutel, N. D. Lane. "End-to-End Speech Recognition from Federated Acoustic Models." arXiv:2104.14297, 2021. https://arxiv.org/abs/2104.14297
A realistic CommonVoice FL setup with an attention-based seq2seq model and three aggregation weightings, cross-silo (10 clients) and cross-device (2K and 4K clients).
*Relevance:* source for realistic client counts and local-epoch settings.

**[Gao22]** Y. Gao, J. Fernandez-Marques, T. Parcollet, A. Mehrotra, N. D. Lane. "Federated Self-supervised Speech Representations: Are We There Yet?" arXiv:2204.02804, 2022. https://arxiv.org/abs/2204.02804
A systems study of SSL speech pretraining under FL. Concludes that current constraints make it nearly impossible today.
*Relevance:* federated fine-tuning (not pretraining) is the realistic wav2vec2 scenario to attack.

**[Azam23]** S. S. Azam, T. Likhomanenko, M. Pelikan, J. Silovsky. "Importance of Smoothness Induced by Optimizers in FL4ASR: Towards Understanding Federated Learning for End-to-End ASR." ASRU 2023. https://arxiv.org/abs/2309.13102
Studies adaptive optimizers, CTC weight, seed models, normalization placement, local epochs, and cohort size for FL ASR.
*Relevance:* tells you which FL hyperparameters (local epochs, seed-model start) are realistic for the attack setting.

**[Xiao24]** Y. Xiao, Y. Ding, C. Ryu, P. Zadrazil, F. Beaufays. "Federated Learning of Large ASR Models in the Real World." arXiv:2408.10443, 2024. https://arxiv.org/abs/2408.10443
Trains a 130M-parameter Conformer ASR model with FL in production. The authors state it is the first real-world FL application of the Conformer.
*Relevance:* evidence that FL for full-size ASR is deployed, which strengthens the motivation.

**[Ali26]** M. N. Ali, D. Falavigna, A. Brutti. "SpeechLLM Meets Federated Learning for End-to-End ASR: English and Italian Case Studies." FLICS 2026 (arXiv:2607.25716). https://arxiv.org/abs/2607.25716
The first federated training study of SpeechLLM-based ASR, with communication-efficient strategies.
*Relevance:* shows where FL ASR is heading. Future-work material.

### 4.2 Generative priors and feature inversion

**[HiFiGAN]** J. Kong, J. Kim, J. Bae. "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis." NeurIPS 2020. https://arxiv.org/abs/2010.05646

**[BigVGAN]** S.-g. Lee, W. Ping, B. Ginsburg, B. Catanzaro, S. Yoon. "BigVGAN: A Universal Neural Vocoder with Large-Scale Training." ICLR 2023. https://arxiv.org/abs/2206.04658

**[Vocos]** H. Siuzdak. "Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis." arXiv:2306.00814, 2023. https://arxiv.org/abs/2306.00814

**[DiffWave]** Z. Kong, W. Ping, J. Huang, K. Zhao, B. Catanzaro. "DiffWave: A Versatile Diffusion Model for Audio Synthesis." ICLR 2021. https://arxiv.org/abs/2009.09761

**[EnCodec]** A. Défossez, J. Copet, G. Synnaeve, Y. Adi. "High Fidelity Neural Audio Compression." arXiv:2210.13438, 2022. https://arxiv.org/abs/2210.13438

*Relevance of the vocoder and codec group:* mel-to-waveform vocoders replace Griffin-Lim after the attack. A codec latent gives a low-dimensional search space, and a differentiable decoder followed by an MFCC front end lets gradient matching run directly in latent space, as GIAS does for images.

**[Juvela18]** L. Juvela, B. Bollepalli, X. Wang, H. Kameoka, M. Airaksinen, J. Yamagishi, P. Alku. "Speech Waveform Synthesis from MFCC Sequences with Generative Adversarial Networks." arXiv:1804.00920, 2018. https://arxiv.org/abs/1804.00920
Generates waveforms from MFCCs alone by predicting F0 with an RNN, converting the spectral envelope to an all-pole filter, and modeling excitation with a GAN.
*Relevance:* direct evidence that MFCC-to-waveform is learnable. MFCCs discard pitch, so a learned prior must supply it.

**[Polyak21]** A. Polyak, Y. Adi, J. Copet, E. Kharitonov, K. Lakhotia, W.-N. Hsu, A. Mohamed, E. Dupoux. "Speech Resynthesis from Discrete Disentangled Self-Supervised Representations." Interspeech 2021. https://arxiv.org/abs/2104.00355
Resynthesizes speech from separate discrete content, prosody, and speaker representations, at 365 bits per second.
*Relevance:* if an attack recovers intermediate SSL features (for example through the subspace leak), a unit vocoder converts them to audio.

**[CQTDiff]** E. Moliner, J. Lehtinen, V. Välimäki. "Solving Audio Inverse Problems with a Diffusion Model." ICASSP 2023. https://arxiv.org/abs/2210.15228
An unconditional audio diffusion model solves bandwidth extension, inpainting, and declipping without retraining.

**[DPS]** H. Chung, J. Kim, M. T. McCann, M. L. Klasky, J. C. Ye. "Diffusion Posterior Sampling for General Noisy Inverse Problems." ICLR 2023. https://arxiv.org/abs/2209.14687
Posterior sampling with diffusion priors for noisy nonlinear inverse problems.

*Relevance of the diffusion pair:* gradient inversion is a nonlinear inverse problem with measurement operator "features to gradient". Posterior sampling with a speech diffusion prior is the principled version of a generative-prior attack and should handle DP noise.

---

## 5. Defenses, with emphasis on speech FL

**[DPSGD]** M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, L. Zhang. "Deep Learning with Differential Privacy." ACM CCS 2016, pp. 308-318. https://arxiv.org/abs/1607.00133
Per-example clipping plus Gaussian noise with a moments accountant.

**[SecAgg]** K. Bonawitz, V. Ivanov, B. Kreuter, A. Marcedone, H. B. McMahan, S. Patel, D. Ramage, A. Segal, K. Seth. "Practical Secure Aggregation for Privacy Preserving Machine Learning." Cryptology ePrint 2017/281. https://eprint.iacr.org/2017/281
The server learns only the sum of client updates.

**[FedAvg]** H. B. McMahan, E. Moore, D. Ramage, S. Hampson, B. Agüera y Arcas. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017. https://arxiv.org/abs/1602.05629
Local multi-step training with weight averaging.

*Relevance of the three:* they define the defense axes (noise, clip, aggregation, local steps) for the sweep.

**[Pelikan25]** M. Pelikan, S. S. Azam, V. Feldman, J. Silovsky, K. Talwar, C. G. Brinton, T. Likhomanenko. "Enabling Differentially Private Federated Learning for Speech Recognition: Benchmarks, Adaptive Optimizers and Gradient Clipping." NeurIPS 2025. https://arxiv.org/abs/2310.00098
The first benchmark for FL with DP on end-to-end transformer ASR. Uses per-layer clipping and layer-wise gradient normalization to handle gradient heterogeneity. Reports a 1.3% absolute WER drop at user-level (7.2, 1e-9)-DP and 4.6% at (4.5, 1e-9)-DP, extrapolated to high and low population scales (the ε values come from the v1 abstract as surfaced in search results).
*Relevance:* the reference utility cost for DP in FL ASR. Per-layer clipping changes relative gradient norms across layers, which matters for cosine versus L2 matching.

**[Shoemate22]** M. Shoemate, K. Jett, E. Cowan, S. Colbath, J. Honaker, P. Muthukumar. "Sotto Voce: Federated Speech Recognition with Differential Privacy Guarantees." arXiv:2207.07816, 2022. https://arxiv.org/abs/2207.07816
Cross-organization FL with DP for a senone classification prototype. The model improves with added private data.
*Relevance:* an early DP FL speech reference.

**[Chauhan24]** G. Chauhan, S. Chien, O. Thakkar, A. Thakurta, A. Narayanan. "Training Large ASR Encoders with Differential Privacy." IEEE SLT 2024. https://arxiv.org/abs/2409.13953
DP pre-training of a Conformer encoder (BEST-RQ) with gradient-based layer freezing. Reports LibriSpeech test-clean/other WER of 3.78/8.41 at (10, 1e-9)-DP extrapolated to low dataset scales, and 2.81/5.89 at (10, 7.9e-11)-DP extrapolated to high scales.
*Relevance:* layer freezing shrinks the gradient the attacker sees. Test an attack with only a subset of layers shared.

**[Liu24]** H. Liu, L. Wang, O. Thakkar, A. Thakurta, A. Narayanan. "Differentially Private Parameter-Efficient Fine-tuning for Large ASR Models." arXiv:2410.01948, 2024. https://arxiv.org/abs/2410.01948
DP parameter-efficient fine-tuning reaches 4.6%/8.1% WER on LibriSpeech clean/other at (10, 3.52e-6)-DP for a model with over 600M parameters.
*Relevance:* parameter-efficient FL shares only adapter gradients. Whether adapters alone leak speech is untested.

**[Luque26]** J. Luque, F. López, A. Sant. "Component-Aware Differential Privacy for Federated Multilingual Speech-LLMs." SLT 2026 (arXiv:2609.11762). https://arxiv.org/abs/2609.11762
Per-layer DP clipping breaks when encoder and LLM update norms differ by an order of magnitude. A two-pool allocation fixes this. The paper argues the acoustic encoder is better protected against gradient inversion (it states an adversary "learns 4.47× less") but runs no attack, per the HTML full text.
*Relevance:* a defense paper making an inversion-resistance claim with no empirical attack. Your attack could test it.

**[Soteria]** J. Sun, A. Li, B. Wang, H. Yang, H. Li, Y. Chen. "Provable Defense against Privacy Leakage in Federated Learning from Representation Perspective." arXiv:2012.06043, 2020. https://arxiv.org/abs/2012.06043
Perturbs the data representation in one layer to degrade reconstruction, with certified robustness and convergence guarantees.

**[PRECODE]** D. Scheliga, P. Mäder, M. Seeland. "PRECODE - A Generic Model Extension to Prevent Deep Gradient Leakage." WACV 2022. https://arxiv.org/abs/2108.04725
A variational bottleneck module that prevents reconstruction from gradients.

**[ATS]** W. Gao, S. Guo, T. Zhang, H. Qiu, Y. Wen, Y. Liu. "Privacy-preserving Collaborative Learning with Automatic Transformation Search." CVPR 2021. https://arxiv.org/abs/2011.12505
Searches augmentation policies that make gradient-based reconstruction fail.

*Relevance of the three:* SpecAugment-style masking is the speech analog of ATS. A stochastic bottleneck is the analog of dropout, which Dang-Spk found effective.

**[Yue23]** K. Yue, R. Jin, C.-W. Wong, D. Baron, H. Dai. "Gradient Obfuscation Gives a False Sense of Security in Federated Learning." USENIX Security 2023. https://arxiv.org/abs/2206.04055
Shows quantization, sparsification, and perturbation give inadequate protection, using an attack that reconstructs at the semantic level.
*Relevance:* evaluate pruning and quantization with an adaptive attacker and not only the vanilla attack.

Speech-specific defense numbers already listed above:
- Dang-Spk: dropout 0.2, and DP-SGD roughly doubling WER (ar5iv).
- Dang-RLG: Sign-SGD against 90% gradient dropping.
- Feng22: user-level DP weakens with more observed updates.
- Wang23: clipping only.
- Arasteh25: DP stops inversion at a maximum 3.85% accuracy cost.

---

## 6. CTC-specific math useful to an attacker

**[Graves06]** A. Graves, S. Fernández, F. Gomez, J. Schmidhuber. "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks." ICML 2006. https://www.cs.toronto.edu/~graves/icml_2006.pdf
Read from the PDF. Eq. (16) gives the error signal with respect to the unnormalized outputs:

∂O/∂u_k^t = y_k^t − (1 / (y_k^t Z_t)) Σ_{s ∈ lab(z,k)} α̂_t(s) β̂_t(s)

So the logit gradient at frame t is softmax output minus posterior occupancy of label k at time t. Figure 4 shows the error is determined by the target sequence at initialization and "virtually disappears" once the network predicts the labelling strongly.
*Relevance:* the starting point for every closed form below.

**[Hannun17]** A. Hannun. "Sequence Modeling with CTC." Distill, 2017. doi:10.23915/distill.00008. https://distill.pub/2017/ctc/
States that the CTC loss is differentiable in the per-time-step output probabilities "since it's just sums and products of them", and recommends log-space computation with log-sum-exp.
*Relevance:* supports the argument that a log-space, autograd-only CTC is smooth and admits a second backward pass.

**[Zeyer21]** A. Zeyer, R. Schlüter, H. Ney. "Why does CTC result in peaky behavior?" arXiv:2105.14849, 2021. https://arxiv.org/abs/2105.14849
A formal analysis of CTC's convergence to peaky, blank-dominated outputs, and of why this does not occur with a label prior.
*Relevance:* on trained models, occupancy concentrates on a few non-blank frames. The logit gradient is then informative mainly at those frames, which predicts where reconstruction is sharp and where it is poorly constrained.

**[Zeyer26]** A. Zeyer, R. Schlüter, H. Ney. "Gradient-Based Speech-to-Text Alignment for Any ASR Model: From CTC to Speech LLMs." arXiv:2607.06831, 2026. https://arxiv.org/abs/2607.06831
Takes the gradient of each token's log-probability with respect to the input, reduces it to per-frame saliency, and decodes word boundaries with one dynamic-programming pass, across sixteen models.
*Relevance:* input gradients localize tokens in time. This can initialize or regularize the alignment of a hypothesized transcript in long-form attacks.

**Twice-differentiable CTC: implementation notes (verified sources)**
- PyTorch forum thread "Higher order gradients of CTCLoss" (2019): https://discuss.pytorch.org/t/higher-order-gradients-of-ctcloss/35019. Reports `RuntimeError: derivative for _ctc_loss_backward is not implemented`. The explanation given is that CTCLoss implements its own backward and does not rely on autograd. No workaround was posted in that thread.
- Dang-Spk (ar5iv) states the same limitation for TensorFlow and PyTorch and avoids it with zeroth-order search.
- `vadimkantorov/ctc`: https://github.com/vadimkantorov/ctc. A pure-PyTorch CTC whose only loop is over time, with gradients from autograd. The README says "It might support double-backwards (not checked)".
- `alexeytochin/tf_seq2seq_losses`: https://github.com/alexeytochin/tf_seq2seq_losses. A TensorFlow CTC that "supports second-order derivatives without using TensorFlow's autogradient". The README lists O(l^4) Hessian complexity.

*Relevance:* your log-space twice-differentiable CTC fills a documented hole. Cite the forum thread and Dang-Spk as evidence that the hole exists, and the two repos as the nearest prior implementations.

**Closed forms (my analysis, derived from Graves Eq. 16 and DAGER/SPEAR Theorem 3.1; not taken from any paper)**

Let g_t = y_t − γ_t ∈ R^C be the logit gradient at frame t (γ_t is the occupancy), and h_t the last hidden vector.

1. **Last-layer gradients.** dL/dW_out = Σ_t g_t h_t^T and dL/db_out = Σ_t g_t. Each g_t sums to zero over classes, so the bias gradient also sums to zero and does not reveal T directly.
2. **Character presence and soft counts.** For a character k absent from the transcript, γ_t(k) = 0 for all t, so the bias gradient entry is Σ_t y_t(k) > 0. For present characters the entry is pulled down by total occupancy Σ_t γ_t(k). Near initialization (y roughly uniform) the entry is about T/C minus the expected dwell time of k, which grows with the count of k. This is the CTC analog of the iDLG/LLG sign-and-magnitude rule and should yield a character histogram.
3. **Why RLG does not apply directly.** With C around 29 characters and T in the hundreds, dL/dW_out has rank at most C, so the condition S < min{d, C} fails. Word-piece CTC with large C relaxes this.
4. **Frame-subspace leak in per-frame FC layers.** For any linear layer applied per frame with shared weights, dL/dW = Σ_t δ_t x_t^T has rank at most min(T, d_in, d_out). When T is below both dimensions and the δ_t are linearly independent, the row space of the gradient equals span{x_t}, and its rank reveals T (the utterance length in frames). For DS1's 2048-unit hidden layers this holds up to T < 2048 frames. For the first layer it holds while T is below the input dimension (context window times MFCC dimension; check your config). For wav2vec2-base (d = 768, 50 frames per second) it holds up to roughly 15 s.
5. **Consequence.** A TIGER-style loss, distance of each candidate frame (or its layer-1 activation) to the leaked subspace, is a first-order objective that never differentiates through CTC twice. Overlapping context windows add shift-consistency constraints between adjacent x_t.
6. **Vanishing signal.** Since g_t tends to 0 for well-fit utterances (Graves Fig. 4c), attacks weaken on converged models. This matches RLG's drop from 98% to 93.3% exact match over training (ar5iv).

---

## (a) Gap analysis: what has not been done as of October 2026

Based on the searches above. The arXiv full-text searches for "gradient inversion" with "speech" and with "audio" each returned only one or two hits, and "gradient leakage" with "speech" returned none.

1. **First-order gradient matching through CTC.** The only CTC ASR attack (Dang-Spk) used zeroth-order search because CTC had no double backward, and stopped at 1 to 4 s. I could not verify what Bui25 does internally.
2. **Long-form reconstruction on a real ASR model.** Zeng25 reaches 139 s, but on a CNN classifier with cross-entropy and a dummy label, on mel-spectrograms. No published long-form result uses a sequence loss, recurrent or attention layers, or MFCC input. Zeng25 lists MFCC as future work.
3. **Transcript recovery for CTC.** Dang-Spk assumes the transcript and length are known. RLG recovers only a token set, on an attention model, and its rank condition fails for character CTC. No work recovers character counts, order, or utterance length from CTC gradients.
4. **Analytic or low-rank attacks on speech.** DAGER, SPEAR, and TIGER have not been applied to frame sequences. Nobody has reported the frame-subspace leak or rank-reveals-length observation for ASR.
5. **Feature reconstruction from modern speech models.** For wav2vec 2.0, HuBERT, Whisper, and Conformer, only speaker and attribute inference from weight updates exists (Tomashenko22, Nguyen23, AlAli25). I found no RNN-T or attention-decoder feature inversion either.
6. **Generative priors for audio gradient inversion.** Every speech attack found uses TV or smoothness regularization and Griffin-Lim. None uses a vocoder, codec latent, or diffusion prior, though Li23 shows MFCC-plus-Griffin-Lim output is poor (PESQ 1.39, STOI 0.43).
7. **FedAvg multi-step and batch > 1 for ASR.** Zeng25 does batches up to 16 or 32 on its CNN. No ASR result exists for multiple local steps or batches of variable-length utterances.
8. **Defense evaluation against reconstruction on ASR.** DP utility costs for FL ASR are measured (Pelikan25, Chauhan24, Liu24), and Luque26 claims inversion resistance without an attack. Empirical attack-versus-defense results exist only for speaker ID on DS1 (Dang-Spk) and a pathology classifier (Arasteh25). No adaptive-attacker evaluation exists for speech.
9. **No shared benchmark or metric set.** Dang-Spk uses MFCC MAE and speaker top-k. Li23 uses PESQ and STOI. Zeng25 uses SNR, mel cosine, Whisper WER, and ECAPA similarity. Nobody reports all of them on the same utterances.
10. **Trained-model regime.** Most speech attacks use untrained or lightly trained models. Systematic results across training stages are missing, although CTC's error signal vanishes as training converges.

---

## (b) Experiment ideas, ranked by expected payoff and feasibility

Budget assumed: a few 48 GB GPUs, under a day each.

**1. Transcript and length recovery from last-layer and rank structure (high payoff, hours, mostly CPU).**
- Recover the character set and soft counts from dL/db_out using the occupancy closed form.
- Recover T from the numerical rank of FC-layer gradients.
- Order characters with an LM-guided beam search scored by gradient distance.
- Report character histogram error, transcript exact match and CER, and length error against training stage.
- Builds on: Graves06 Eq. 16, iDLG, LLG, Dang-RLG, FILM, LAMP, FET.
- Removes the two strongest assumptions inherited from Dang-Spk.

**2. Subspace-constrained inversion for per-frame FC layers (high payoff, under a day).**
- Extract the row space of dL/dW for DS1 layers 1 to 3.
- Verify numerically that true frames (or activations) lie in it for T below the layer dimension.
- Add a TIGER-style subspace-distance loss with shift-consistency across context windows, alone and combined with gradient matching.
- Hypothesis: faster convergence and better long-form accuracy, since each chunk's frames must lie in a known T-dimensional subspace.
- For short chunks (T ≲ 25), try SPEAR-style exact disaggregation or CPA-style ICA.
- Builds on: DAGER Thm 3.1/5.1, TIGER, SPEAR, CPA, InvGrad Prop. 3.1, R-GAP rank analysis, Li23-pooler.

**3. Head-to-head with Zeng25, plus a unified metric suite (medium-high payoff, under a day; needed for the rewrite).**
- Reimplement their segment method on their CNN, and run your method on DS1/DS2, on the same LibriSpeech and CommonVoice clips. Include their 35 s and 139 s lengths.
- Report SNR, mel cosine, MFCC MAE, PESQ, STOI, Whisper WER, and ECAPA speaker similarity.
- Run their segmentation against a CTC model to show where it breaks (no dummy label, a temporally coupled loss).
- Builds on: Zeng25, Li23, Dang-Spk, Huang21, SoK25.

**4. Attack wav2vec2-CTC (and HuBERT-CTC) fine-tuning (high payoff, medium feasibility).**
- Plug the twice-differentiable CTC into a wav2vec2-base CTC head. Optimize either the waveform or the conv-encoder output.
- Match a subset of layers to fit double backward in 48 GB (CTC head plus the first transformer layers).
- Add the subspace loss on query-projection gradients, valid for clips under about 15 s.
- Start at 2 to 5 s.
- Builds on: Nguyen23, Li24-partial, FedLeak, DAGER/TIGER, APRIL, AlAli25.
- This answers the "legacy architecture" critique (Valadi25, SoK25).

**5. Neural vocoder and generative priors for MFCC inversion (medium-high payoff, about a day).**
- Stage A (cheap): train a small MFCC-to-mel regressor on LibriSpeech and vocode with pretrained HiFi-GAN, BigVGAN, or Vocos. Compare PESQ, STOI, and speaker ID against Griffin-Lim.
- Stage B: GUIDE-style post-hoc denoising of reconstructed features.
- Stage C: optimize in an EnCodec latent, or run diffusion posterior sampling with gradient distance as the measurement term.
- Builds on: Li23, Juvela18, HiFiGAN, BigVGAN, Vocos, EnCodec, GIAS, GGL, GIFD, GUIDE, DPS, CQTDiff, Meng25.

**6. Batch > 1 and FedAvg multi-step (medium payoff, a day of compute).**
- Sweep B ∈ {1, 2, 4, 8} with variable-length utterances, local steps E ∈ {1, 2, 5, 10}, and client learning rate.
- Use Dimitrov22's simulated-update objective, and the cheaper NL-SME surrogate.
- Check whether the rank of FC gradients still reveals the total frame count ΣT for a batch.
- Builds on: Dimitrov22, Geng21, NL-SME, InvGrad, CPL, Zeng25 (batch results).

**7. Defense sweep with utility cost and an adaptive attacker (medium payoff, a day).**
- Defenses: DP-SGD (clip only, and clip plus noise at several σ), per-layer clipping, top-k pruning, Sign-SGD and quantization, dropout, variational noise, SpecAugment.
- Report attack metrics next to WER after a fixed fine-tuning budget.
- Rerun the best attack adaptively: known mask for pruning, the subspace loss under noise, a denoising prior.
- Builds on: DPSGD, Pelikan25, Dang-Spk, Dang-RLG, Guliani21, Wang23, Yue23, Bayes, L2I, TIGER, Meng25, FILM.

**8. Training-stage and model-state study (medium payoff, cheap).**
- Attack checkpoints from random init through convergence, in train and eval mode (dropout, BN for DS2).
- Relate success to CTC loss value and occupancy peakiness.
- Builds on: Graves06 Fig. 4, Zeyer21, Dang-RLG, Hatamizadeh22, Bayes.

**9. Malicious-server readout on DS1's first FC layer (lower priority, a few hours).**
- Imprint or trap-weight parameters for the first FC plus ReLU layer to read out individual MFCC context windows. Stitch windows using their overlap.
- A different threat model, useful as an upper bound.
- Builds on: RobFed, Boenisch, Fishing, LOKI, ARES, VGIA.

---

## (c) BibTeX for all verified papers

```bibtex
@inproceedings{zhu2019dlg, title={Deep Leakage from Gradients}, author={Zhu, Ligeng and Liu, Zhijian and Han, Song}, booktitle={Advances in Neural Information Processing Systems}, volume={32}, year={2019}, note={arXiv:1906.08935}}
@article{zhao2020idlg, title={{iDLG}: Improved Deep Leakage from Gradients}, author={Zhao, Bo and Mopuri, Konda Reddy and Bilen, Hakan}, journal={arXiv preprint arXiv:2001.02610}, year={2020}}
@inproceedings{geiping2020inverting, title={Inverting Gradients -- How easy is it to break privacy in federated learning?}, author={Geiping, Jonas and Bauermeister, Hartmut and Dr{\"o}ge, Hannah and Moeller, Michael}, booktitle={Advances in Neural Information Processing Systems}, year={2020}, note={arXiv:2003.14053}}
@inproceedings{yin2021gradinversion, title={See through Gradients: Image Batch Recovery via {GradInversion}}, author={Yin, Hongxu and Mallya, Arun and Vahdat, Arash and Alvarez, Jose M. and Kautz, Jan and Molchanov, Pavlo}, booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, year={2021}, note={arXiv:2104.07586}}
@inproceedings{zhu2021rgap, title={{R-GAP}: Recursive Gradient Attack on Privacy}, author={Zhu, Junyi and Blaschko, Matthew}, booktitle={International Conference on Learning Representations (ICLR)}, year={2021}, note={arXiv:2010.07733}}
@article{phong2018privacy, title={Privacy-Preserving Deep Learning via Additively Homomorphic Encryption}, author={Phong, Le Trieu and Aono, Yoshinori and Hayashi, Takuya and Wang, Lihua and Moriai, Shiho}, journal={IEEE Transactions on Information Forensics and Security}, year={2018}, doi={10.1109/TIFS.2017.2787987}, note={Cryptology ePrint 2017/715}}
@article{wei2020framework, title={A Framework for Evaluating Gradient Leakage Attacks in Federated Learning}, author={Wei, Wenqi and Liu, Ling and Loper, Margaret and Chow, Ka-Ho and Gursoy, Mehmet Emre and Truex, Stacey and Wu, Yanzhao}, journal={arXiv preprint arXiv:2004.10397}, year={2020}}
@inproceedings{melis2019exploiting, title={Exploiting Unintended Feature Leakage in Collaborative Learning}, author={Melis, Luca and Song, Congzheng and De Cristofaro, Emiliano and Shmatikov, Vitaly}, booktitle={IEEE Symposium on Security and Privacy (S\&P)}, year={2019}, note={arXiv:1805.04049}}
@inproceedings{huang2021evaluating, title={Evaluating Gradient Inversion Attacks and Defenses in Federated Learning}, author={Huang, Yangsibo and Gupta, Samyak and Song, Zhao and Li, Kai and Arora, Sanjeev}, booktitle={Advances in Neural Information Processing Systems}, year={2021}, note={arXiv:2112.00059}}
@article{hatamizadeh2023gradient, title={Do Gradient Inversion Attacks Make Federated Learning Unsafe?}, author={Hatamizadeh, Ali and Yin, Hongxu and Molchanov, Pavlo and Myronenko, Andriy and Li, Wenqi and Dogra, Prerna and Feng, Andrew and Flores, Mona G. and Kautz, Jan and Xu, Daguang and Roth, Holger R.}, journal={IEEE Transactions on Medical Imaging}, year={2023}, note={arXiv:2202.06924}}
@inproceedings{du2025sok, title={{SoK}: On Gradient Leakage in Federated Learning}, author={Du, Jiacheng and Hu, Jiahui and Wang, Zhibo and Sun, Peng and Gong, Neil Zhenqiang and Ren, Kui and Chen, Chun}, booktitle={USENIX Security Symposium}, year={2025}, note={arXiv:2404.05403}}
@inproceedings{fan2025boosting, title={Boosting Gradient Leakage Attacks: Data Reconstruction in Realistic {FL} Settings}, author={Fan, Mingyuan and Wang, Fuyi and Chen, Cen and Zhou, Jianying}, booktitle={USENIX Security Symposium}, year={2025}, note={arXiv:2506.08435}}
@article{valadi2025practical, title={Practical Feasibility of Gradient Inversion Attacks in Federated Learning}, author={Valadi, Viktor and {\AA}kesson, Mattias and {\"O}stman, Johan and Hoseini, Fazeleh and Toor, Salman and Hellander, Andreas}, journal={arXiv preprint arXiv:2508.19819}, year={2025}}
@article{zhu2024privacy, title={Privacy Attack in Federated Learning is Not Easy: An Experimental Study}, author={Zhu, Hangyu and Huang, Liyuan and Xie, Zhenping}, journal={arXiv preprint arXiv:2409.19301}, year={2024}}
@article{dimitrov2022data, title={Data Leakage in Federated Averaging}, author={Dimitrov, Dimitar I. and Balunovi{\'c}, Mislav and Konstantinov, Nikola and Vechev, Martin}, journal={Transactions on Machine Learning Research}, year={2022}, note={arXiv:2206.12395}}
@article{geng2021towards, title={Towards General Deep Leakage in Federated Learning}, author={Geng, Jiahui and Mou, Yongli and Li, Feifei and Li, Qing and Beyan, Oya and Decker, Stefan and Rong, Chunming}, journal={arXiv preprint arXiv:2110.09074}, year={2021}}
@article{xia2025trajectory, title={Trajectory-Aware Information Matching for Multi-Step Gradient Inversion in Federated Learning}, author={Xia, Li and Yu, Jing and Liu, Zheng and Huang, Sili and Tang, Wei and Liu, Xuan}, journal={arXiv preprint arXiv:2509.22082}, year={2025}}
@inproceedings{jeon2021gradient, title={Gradient Inversion with Generative Image Prior}, author={Jeon, Jinwoo and Kim, Jaechang and Lee, Kangwook and Oh, Sewoong and Ok, Jungseul}, booktitle={Advances in Neural Information Processing Systems}, year={2021}, note={arXiv:2110.14962}}
@inproceedings{li2022auditing, title={Auditing Privacy Defenses in Federated Learning via Generative Gradient Leakage}, author={Li, Zhuohang and Zhang, Jiaxin and Liu, Luyang and Liu, Jian}, booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, year={2022}, note={arXiv:2203.15696}}
@inproceedings{fang2023gifd, title={{GIFD}: A Generative Gradient Inversion Method with Feature Domain Optimization}, author={Fang, Hao and Chen, Bin and Wang, Xuan and Wang, Zhi and Xia, Shu-Tao}, booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)}, year={2023}, note={arXiv:2308.04699}}
@article{li2024exploring, title={Exploring User-level Gradient Inversion with a Diffusion Prior}, author={Li, Zhuohang and Lowy, Andrew and Liu, Jing and Koike-Akino, Toshiaki and Malin, Bradley and Parsons, Kieran and Wang, Ye}, journal={arXiv preprint arXiv:2409.07291}, year={2024}, note={NeurIPS 2023 Workshop on Federated Learning in the Age of Foundation Models}}
@article{meng2025enhanced, title={Enhanced Privacy Leakage from Noise-Perturbed Gradients via Gradient-Guided Conditional Diffusion Models}, author={Meng, Jiayang and Huang, Tao and Chen, Hong and Hou, Chen and Zheng, Guolong}, journal={arXiv preprint arXiv:2511.10423}, year={2025}}
@article{carletti2025guide, title={{GUIDE}: Enhancing Gradient Inversion Attacks in Federated Learning with Denoising Models}, author={Carletti, Vincenzo and Foggia, Pasquale and Mazzocca, Carlo and Parrella, Giuseppe and Vento, Mario}, journal={arXiv preprint arXiv:2510.17621}, year={2025}}
@inproceedings{wu2023learning, title={Learning to Invert: Simple Adaptive Attacks for Gradient Inversion in Federated Learning}, author={Wu, Ruihan and Chen, Xiangyu and Guo, Chuan and Weinberger, Kilian Q.}, booktitle={Uncertainty in Artificial Intelligence (UAI), PMLR 216}, year={2023}, note={arXiv:2210.10880}}
@article{balunovic2021bayesian, title={Bayesian Framework for Gradient Leakage}, author={Balunovi{\'c}, Mislav and Dimitrov, Dimitar I. and Staab, Robin and Vechev, Martin}, journal={arXiv preprint arXiv:2111.04706}, year={2021}}
@article{fowl2021robbing, title={Robbing the Fed: Directly Obtaining Private Data in Federated Learning with Modified Models}, author={Fowl, Liam and Geiping, Jonas and Czaja, Wojtek and Goldblum, Micah and Goldstein, Tom}, journal={arXiv preprint arXiv:2110.13057}, year={2021}}
@inproceedings{wen2022fishing, title={Fishing for User Data in Large-Batch Federated Learning via Gradient Magnification}, author={Wen, Yuxin and Geiping, Jonas and Fowl, Liam and Goldblum, Micah and Goldstein, Tom}, booktitle={International Conference on Machine Learning (ICML)}, year={2022}, note={arXiv:2202.00580}}
@article{boenisch2021curious, title={When the Curious Abandon Honesty: Federated Learning Is Not Private}, author={Boenisch, Franziska and Dziedzic, Adam and Schuster, Roei and Shamsabadi, Ali Shahin and Shumailov, Ilia and Papernot, Nicolas}, journal={arXiv preprint arXiv:2112.02918}, year={2021}}
@article{kariyappa2022cocktail, title={Cocktail Party Attack: Breaking Aggregation-Based Privacy in Federated Learning using Independent Component Analysis}, author={Kariyappa, Sanjay and Guo, Chuan and Maeng, Kiwan and Xiong, Wenjie and Suh, G. Edward and Qureshi, Moinuddin K. and Lee, Hsien-Hsin S.}, journal={arXiv preprint arXiv:2209.05578}, year={2022}, note={Published at ICML per Semantic Scholar record}}
@inproceedings{dimitrov2024spear, title={{SPEAR}: Exact Gradient Inversion of Batches in Federated Learning}, author={Dimitrov, Dimitar I. and Baader, Maximilian and M{\"u}ller, Mark Niklas and Vechev, Martin}, booktitle={Advances in Neural Information Processing Systems}, year={2024}, note={arXiv:2403.03945}}
@inproceedings{zhao2024loki, title={{LOKI}: Large-scale Data Reconstruction Attack against Federated Learning through Model Manipulation}, author={Zhao, Joshua C. and Sharma, Atul and Elkordy, Ahmed Roushdy and Ezzeldin, Yahya H. and Avestimehr, Salman and Bagchi, Saurabh}, booktitle={IEEE Symposium on Security and Privacy (S\&P)}, year={2024}, note={arXiv:2303.12233}}
@article{garov2023hiding, title={Hiding in Plain Sight: Disguising Data Stealing Attacks in Federated Learning}, author={Garov, Kostadin and Dimitrov, Dimitar I. and Jovanovi{\'c}, Nikola and Vechev, Martin}, journal={arXiv preprint arXiv:2306.03013}, year={2023}}
@inproceedings{gong2026ares, title={{ARES}: Scalable and Practical Gradient Inversion Attack in Federated Learning through Activation Recovery}, author={Gong, Zirui and Zhang, Leo Yu and Zhang, Yanjun and Vo, Viet and Zhu, Tianqing and Pan, Shirui and Wang, Cong}, booktitle={IEEE Symposium on Security and Privacy (S\&P)}, year={2026}, note={arXiv:2603.17623}}
@article{diana2026no, title={No More Guessing: a Verifiable Gradient Inversion Attack in Federated Learning}, author={Diana, Francesco and Xu, Chuan and Nusser, Andr{\'e} and Neglia, Giovanni}, journal={arXiv preprint arXiv:2604.15063}, year={2026}}
@inproceedings{zhao2024leak, title={Leak and Learn: An Attacker's Cookbook to Train Using Leaked Data from Federated Learning}, author={Zhao, Joshua C. and Dabholkar, Ahaan and Sharma, Atul and Bagchi, Saurabh}, booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, year={2024}, note={arXiv:2403.18144}}
@article{wainakh2022user, title={User-Level Label Leakage from Gradients in Federated Learning}, author={Wainakh, Aidmar and Ventola, Fabrizio and M{\"u}{\ss}ig, Till and Keim, Jens and Garcia Cordero, Carlos and Zimmer, Ephraim and Grube, Tim and Kersting, Kristian and M{\"u}hlh{\"a}user, Max}, journal={Proceedings on Privacy Enhancing Technologies (PETS)}, year={2022}, note={arXiv:2105.09369}}
@inproceedings{zhang2022survey, title={A Survey on Gradient Inversion: Attacks, Defenses and Future Directions}, author={Zhang, Rui and Guo, Song and Wang, Junxiao and Xie, Xin and Tao, Dacheng}, booktitle={International Joint Conference on Artificial Intelligence (IJCAI-ECAI)}, year={2022}, note={arXiv:2206.07284}}
@article{guo2025exploring, title={Exploring the Vulnerabilities of Federated Learning: A Deep Dive into Gradient Inversion Attacks}, author={Guo, Pengxin and Wang, Runxi and Zeng, Shuang and Zhu, Jinjing and Jiang, Haoning and Wang, Yanran and Zhou, Yuyin and Wang, Feifei and Xiong, Hui and Qu, Liangqiong}, journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, year={2025}, note={Accepted; arXiv:2503.11514}}
@article{li2024analyzing, title={Analyzing Inference Privacy Risks Through Gradients in Machine Learning}, author={Li, Zhuohang and Lowy, Andrew and Liu, Jing and Koike-Akino, Toshiaki and Parsons, Kieran and Malin, Bradley and Wang, Ye}, journal={arXiv preprint arXiv:2408.16913}, year={2024}}

@inproceedings{dang2022speaker, title={A Method to Reveal Speaker Identity in Distributed {ASR} Training, and How to Counter It}, author={Dang, Trung and Thakkar, Om and Ramaswamy, Swaroop and Mathews, Rajiv and Chin, Peter and Beaufays, Fran{\c{c}}oise}, booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, year={2022}, note={arXiv:2104.07815}}
@inproceedings{dang2021revealing, title={Revealing and Protecting Labels in Distributed Training}, author={Dang, Trung and Thakkar, Om and Ramaswamy, Swaroop and Mathews, Rajiv and Chin, Peter and Beaufays, Fran{\c{c}}oise}, booktitle={Advances in Neural Information Processing Systems}, volume={34}, year={2021}, note={arXiv:2111.00556}}
@incollection{bui2025reconstructing, title={Reconstructing Speech Features of Automatic Speech Recognition Systems in Federated Learning by Gradient Descent}, author={Bui, Minh N. and Dang, Trung and Cherian, Paul and Tran, Trac D. and Chin, Peter}, booktitle={Complex Networks \& Their Applications XIII (COMPLEX NETWORKS 2024)}, series={Studies in Computational Intelligence}, publisher={Springer Nature Switzerland}, pages={125--134}, year={2025}, doi={10.1007/978-3-031-82427-2_11}}
@inproceedings{li2023speech, title={Speech Privacy Leakage from Shared Gradients in Distributed Learning}, author={Li, Zhuohang and Zhang, Jiaxin and Liu, Jian}, booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, year={2023}, note={arXiv:2302.10441}}
@inproceedings{ovi2024gradient, title={Gradient Inversion Attacks on Acoustic Signals: Revealing Security Risks in Audio Recognition Systems}, author={Ovi, Pretom Roy and Gangopadhyay, Aryya}, booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, pages={4835--4839}, year={2024}, doi={10.1109/ICASSP48485.2024.10445809}}
@inproceedings{zeng2025recover, title={How to Recover Long Audio Sequences Through Gradient Inversion Attack With Dynamic Segment-based Reconstruction}, author={Zeng, Xijie and Rudzicz, Frank}, booktitle={Proc. Interspeech}, pages={5118--5122}, year={2025}, doi={10.21437/Interspeech.2025-244}}
@article{arasteh2025differential, title={Differential privacy enables fair and accurate {AI}-based analysis of speech disorders while protecting patient data}, author={Tayebi Arasteh, Soroosh and Lotfinia, Mahshad and Perez-Toro, Paula Andrea and Arias-Vergara, Tomas and Ranji, Mahtab and Orozco-Arroyave, Juan Rafael and Schuster, Maria and Maier, Andreas and Yang, Seung Hee}, journal={npj Artificial Intelligence}, volume={1}, number={37}, year={2025}, doi={10.1038/s44387-025-00040-8}, note={arXiv:2409.19078}}
@inproceedings{tomashenko2022privacy, title={Privacy attacks for automatic speech recognition acoustic models in a federated learning framework}, author={Tomashenko, Natalia and Mdhaffar, Salima and Tommasi, Marc and Est{\`e}ve, Yannick and Bonastre, Jean-Fran{\c{c}}ois}, booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, pages={6972--6976}, year={2022}, note={arXiv:2111.03777}}
@article{mdhaffar2021retrieving, title={Retrieving Speaker Information from Personalized Acoustic Models for Speech Recognition}, author={Mdhaffar, Salima and Bonastre, Jean-Fran{\c{c}}ois and Tommasi, Marc and Tomashenko, Natalia and Est{\`e}ve, Yannick}, journal={arXiv preprint arXiv:2111.04194}, year={2021}}
@inproceedings{nguyen2023federated, title={Federated Learning for {ASR} based on Wav2vec 2.0}, author={Nguyen, Tuan and Mdhaffar, Salima and Tomashenko, Natalia and Bonastre, Jean-Fran{\c{c}}ois and Est{\`e}ve, Yannick}, booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, year={2023}, note={arXiv:2302.10790}}
@article{feng2021attribute, title={Attribute Inference Attack of Speech Emotion Recognition in Federated Learning Settings}, author={Feng, Tiantian and Hashemi, Hanieh and Hebbar, Rajat and Annavaram, Murali and Narayanan, Shrikanth S.}, journal={arXiv preprint arXiv:2112.13416}, year={2021}}
@inproceedings{feng2022user, title={User-Level Differential Privacy against Attribute Inference Attack of Speech Emotion Recognition in Federated Learning}, author={Feng, Tiantian and Peri, Raghuveer and Narayanan, Shrikanth}, booktitle={Proc. Interspeech}, year={2022}, note={arXiv:2204.02500}}
@article{tan2024investigating, title={Investigating Effective Speaker Property Privacy Protection in Federated Learning for Speech Emotion Recognition}, author={Tan, Chao and Li, Sheng and Cao, Yang and Ren, Zhao and Schultz, Tanja}, journal={arXiv preprint arXiv:2410.13221}, year={2024}}
@article{alali2025personal, title={Personal Attribute Leakage in Federated Speech Models}, author={Al-Ali, Hamdan and Ghavamipour, Ali Reza and Caselli, Tommaso and Turkmen, Fatih and Talat, Zeerak and Aldarmaki, Hanan}, journal={arXiv preprint arXiv:2510.13357}, year={2025}}
@inproceedings{amid2022extracting, title={Extracting Targeted Training Data from {ASR} Models, and How to Mitigate It}, author={Amid, Ehsan and Thakkar, Om and Narayanan, Arun and Mathews, Rajiv and Beaufays, Fran{\c{c}}oise}, booktitle={Proc. Interspeech}, year={2022}, note={arXiv:2204.08345}}
@article{wang2023unintended, title={Unintended Memorization in Large {ASR} Models, and How to Mitigate It}, author={Wang, Lun and Thakkar, Om and Mathews, Rajiv}, journal={arXiv preprint arXiv:2310.11739}, year={2023}}
@inproceedings{jagielski2024noise, title={Noise Masking Attacks and Defenses for Pretrained Speech Models}, author={Jagielski, Matthew and Thakkar, Om and Wang, Lun}, booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, year={2024}, note={arXiv:2404.02052}}
@inproceedings{tseng2022membership, title={Membership Inference Attacks Against Self-supervised Speech Models}, author={Tseng, Wei-Cheng and Kao, Wei-Tsung and Lee, Hung-yi}, booktitle={Proc. Interspeech}, year={2022}, note={arXiv:2111.05113}}
@inproceedings{luque2026component, title={Component-Aware Differential Privacy for Federated Multilingual Speech-{LLMs}}, author={Luque, Jordi and L{\'o}pez, Fernando and Sant, Aleix}, booktitle={IEEE Spoken Language Technology Workshop (SLT)}, year={2026}, note={arXiv:2609.11762}}

@inproceedings{deng2021tag, title={{TAG}: Gradient Attack on Transformer-based Language Models}, author={Deng, Jieren and Wang, Yijue and Li, Ji and Shang, Chao and Liu, Hang and Rajasekaran, Sanguthevar and Ding, Caiwen}, booktitle={Findings of EMNLP}, year={2021}, note={arXiv:2103.06819}}
@inproceedings{balunovic2022lamp, title={{LAMP}: Extracting Text from Gradients with Language Model Priors}, author={Balunovi{\'c}, Mislav and Dimitrov, Dimitar I. and Jovanovi{\'c}, Nikola and Vechev, Martin}, booktitle={Advances in Neural Information Processing Systems}, year={2022}, note={arXiv:2202.08827}}
@inproceedings{gupta2022recovering, title={Recovering Private Text in Federated Learning of Language Models}, author={Gupta, Samyak and Huang, Yangsibo and Zhong, Zexuan and Gao, Tianyu and Li, Kai and Chen, Danqi}, booktitle={Advances in Neural Information Processing Systems}, year={2022}, note={arXiv:2205.08514}}
@inproceedings{fowl2023decepticons, title={Decepticons: Corrupted Transformers Breach Privacy in Federated Learning for Language Models}, author={Fowl, Liam and Geiping, Jonas and Reich, Steven and Wen, Yuxin and Czaja, Wojtek and Goldblum, Micah and Goldstein, Tom}, booktitle={International Conference on Learning Representations (ICLR)}, year={2023}, note={arXiv:2201.12675}}
@inproceedings{chu2023panning, title={Panning for Gold in Federated Learning: Targeted Text Extraction under Arbitrarily Large-Scale Aggregation}, author={Chu, Hong-Min and Geiping, Jonas and Fowl, Liam H. and Goldblum, Micah and Goldstein, Tom}, booktitle={International Conference on Learning Representations (ICLR)}, year={2023}}
@article{lu2021april, title={{APRIL}: Finding the Achilles' Heel on Privacy for Vision Transformers}, author={Lu, Jiahao and Zhang, Xi Sheryl and Zhao, Tianli and He, Xiangyu and Cheng, Jian}, journal={arXiv preprint arXiv:2112.14087}, year={2021}}
@inproceedings{petrov2024dager, title={{DAGER}: Exact Gradient Inversion for Large Language Models}, author={Petrov, Ivo and Dimitrov, Dimitar I. and Baader, Maximilian and M{\"u}ller, Mark Niklas and Vechev, Martin}, booktitle={Advances in Neural Information Processing Systems}, year={2024}, note={arXiv:2405.15586}}
@article{kalikman2026tiger, title={{TIGER}: Inverting Transformer Gradients via Embedding-Subspace Distance Optimization}, author={Kalikman, William and Petrov, Ivo and Dimitrov, Dimitar I. and Vechev, Martin}, journal={arXiv preprint arXiv:2606.18312}, year={2026}}
@inproceedings{li2024seeing, title={Seeing the Forest through the Trees: Data Leakage from Partial Transformer Gradients}, author={Li, Weijun and Xu, Qiongkai and Dras, Mark}, booktitle={Conference on Empirical Methods in Natural Language Processing (EMNLP)}, year={2024}, note={arXiv:2406.00999}}
@article{li2023beyond, title={Beyond Gradient and Priors in Privacy Attacks: Leveraging Pooler Layer Inputs of Language Models in Federated Learning}, author={Li, Jianwei and Liu, Sheng and Lei, Qi}, journal={arXiv preprint arXiv:2312.05720}, year={2023}}
@inproceedings{gao2025gradient, title={Gradient Inversion Attack in Federated Learning: Exposing Text Data through Discrete Optimization}, author={Gao, Ying and Xie, Yuxin and Deng, Huanghao and Zhu, Zukun}, booktitle={International Conference on Computational Linguistics (COLING)}, pages={2582--2591}, year={2025}}

@article{hannun2014deep, title={Deep Speech: Scaling up end-to-end speech recognition}, author={Hannun, Awni and Case, Carl and Casper, Jared and Catanzaro, Bryan and Diamos, Greg and Elsen, Erich and Prenger, Ryan and Satheesh, Sanjeev and Sengupta, Shubho and Coates, Adam and Ng, Andrew Y.}, journal={arXiv preprint arXiv:1412.5567}, year={2014}}
@article{amodei2015deep, title={Deep Speech 2: End-to-End Speech Recognition in English and Mandarin}, author={Amodei, Dario and Anubhai, Rishita and Battenberg, Eric and Case, Carl and Casper, Jared and Catanzaro, Bryan and Chen, Jingdong and Chrzanowski, Mike and Coates, Adam and Diamos, Greg and Elsen, Erich and Engel, Jesse and Fan, Linxi and Fougner, Christopher and Han, Tony and Hannun, Awni and Jun, Billy and LeGresley, Patrick and Lin, Libby and Narang, Sharan and Ng, Andrew and Ozair, Sherjil and Prenger, Ryan and Raiman, Jonathan and Satheesh, Sanjeev and Seetapun, David and Sengupta, Shubho and Wang, Yi and Wang, Zhiqian and Wang, Chong and Xiao, Bo and Yogatama, Dani and Zhan, Jun and Zhu, Zhenyao}, journal={arXiv preprint arXiv:1512.02595}, year={2015}}
@article{baevski2020wav2vec, title={wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations}, author={Baevski, Alexei and Zhou, Henry and Mohamed, Abdelrahman and Auli, Michael}, journal={arXiv preprint arXiv:2006.11477}, year={2020}}
@article{hsu2021hubert, title={{HuBERT}: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units}, author={Hsu, Wei-Ning and Bolte, Benjamin and Tsai, Yao-Hung Hubert and Lakhotia, Kushal and Salakhutdinov, Ruslan and Mohamed, Abdelrahman}, journal={arXiv preprint arXiv:2106.07447}, year={2021}}
@inproceedings{radford2023robust, title={Robust Speech Recognition via Large-Scale Weak Supervision}, author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya}, booktitle={International Conference on Machine Learning (ICML), PMLR 202}, year={2023}, note={arXiv:2212.04356}}
@article{gulati2020conformer, title={Conformer: Convolution-augmented Transformer for Speech Recognition}, author={Gulati, Anmol and Qin, James and Chiu, Chung-Cheng and Parmar, Niki and Zhang, Yu and Yu, Jiahui and Han, Wei and Wang, Shibo and Zhang, Zhengdong and Wu, Yonghui and Pang, Ruoming}, journal={arXiv preprint arXiv:2005.08100}, year={2020}}
@inproceedings{guliani2021training, title={Training Speech Recognition Models with Federated Learning: A Quality/Cost Framework}, author={Guliani, Dhruv and Beaufays, Fran{\c{c}}oise and Motta, Giovanni}, booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, year={2021}, note={arXiv:2010.15965}}
@article{gao2021end, title={End-to-End Speech Recognition from Federated Acoustic Models}, author={Gao, Yan and Parcollet, Titouan and Zaiem, Salah and Fernandez-Marques, Javier and de Gusmao, Pedro P. B. and Beutel, Daniel J. and Lane, Nicholas D.}, journal={arXiv preprint arXiv:2104.14297}, year={2021}}
@article{gao2022federated, title={Federated Self-supervised Speech Representations: Are We There Yet?}, author={Gao, Yan and Fernandez-Marques, Javier and Parcollet, Titouan and Mehrotra, Abhinav and Lane, Nicholas D.}, journal={arXiv preprint arXiv:2204.02804}, year={2022}}
@inproceedings{azam2023importance, title={Importance of Smoothness Induced by Optimizers in {FL4ASR}: Towards Understanding Federated Learning for End-to-End {ASR}}, author={Azam, Sheikh Shams and Likhomanenko, Tatiana and Pelikan, Martin and Silovsky, Jan}, booktitle={IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)}, year={2023}, note={arXiv:2309.13102}}
@article{xiao2024federated, title={Federated Learning of Large {ASR} Models in the Real World}, author={Xiao, Yonghui and Ding, Yuxin and Ryu, Changwan and Zadrazil, Petr and Beaufays, Fran{\c{c}}oise}, journal={arXiv preprint arXiv:2408.10443}, year={2024}}
@inproceedings{ali2026speechllm, title={{SpeechLLM} Meets Federated Learning for End-to-End {ASR}: English and Italian Case Studies}, author={Ali, Mohamed Nabih and Falavigna, Daniele and Brutti, Alessio}, booktitle={International Conference on Federated Learning and Intelligent Computing Systems (FLICS)}, year={2026}, note={arXiv:2607.25716}}

@inproceedings{kong2020hifigan, title={{HiFi-GAN}: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis}, author={Kong, Jungil and Kim, Jaehyeon and Bae, Jaekyoung}, booktitle={Advances in Neural Information Processing Systems}, year={2020}, note={arXiv:2010.05646}}
@inproceedings{lee2023bigvgan, title={{BigVGAN}: A Universal Neural Vocoder with Large-Scale Training}, author={Lee, Sang-gil and Ping, Wei and Ginsburg, Boris and Catanzaro, Bryan and Yoon, Sungroh}, booktitle={International Conference on Learning Representations (ICLR)}, year={2023}, note={arXiv:2206.04658}}
@article{siuzdak2023vocos, title={Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis}, author={Siuzdak, Hubert}, journal={arXiv preprint arXiv:2306.00814}, year={2023}}
@inproceedings{kong2021diffwave, title={{DiffWave}: A Versatile Diffusion Model for Audio Synthesis}, author={Kong, Zhifeng and Ping, Wei and Huang, Jiaji and Zhao, Kexin and Catanzaro, Bryan}, booktitle={International Conference on Learning Representations (ICLR)}, year={2021}, note={arXiv:2009.09761}}
@article{defossez2022high, title={High Fidelity Neural Audio Compression}, author={D{\'e}fossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi}, journal={arXiv preprint arXiv:2210.13438}, year={2022}}
@article{juvela2018speech, title={Speech Waveform Synthesis from {MFCC} Sequences with Generative Adversarial Networks}, author={Juvela, Lauri and Bollepalli, Bajibabu and Wang, Xin and Kameoka, Hirokazu and Airaksinen, Manu and Yamagishi, Junichi and Alku, Paavo}, journal={arXiv preprint arXiv:1804.00920}, year={2018}}
@inproceedings{polyak2021speech, title={Speech Resynthesis from Discrete Disentangled Self-Supervised Representations}, author={Polyak, Adam and Adi, Yossi and Copet, Jade and Kharitonov, Eugene and Lakhotia, Kushal and Hsu, Wei-Ning and Mohamed, Abdelrahman and Dupoux, Emmanuel}, booktitle={Proc. Interspeech}, year={2021}, note={arXiv:2104.00355}}
@inproceedings{moliner2023solving, title={Solving Audio Inverse Problems with a Diffusion Model}, author={Moliner, Eloi and Lehtinen, Jaakko and V{\"a}lim{\"a}ki, Vesa}, booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, year={2023}, note={arXiv:2210.15228}}
@inproceedings{chung2023diffusion, title={Diffusion Posterior Sampling for General Noisy Inverse Problems}, author={Chung, Hyungjin and Kim, Jeongsol and McCann, Michael T. and Klasky, Marc L. and Ye, Jong Chul}, booktitle={International Conference on Learning Representations (ICLR)}, year={2023}, note={arXiv:2209.14687}}

@inproceedings{abadi2016deep, title={Deep Learning with Differential Privacy}, author={Abadi, Mart{\'i}n and Chu, Andy and Goodfellow, Ian and McMahan, H. Brendan and Mironov, Ilya and Talwar, Kunal and Zhang, Li}, booktitle={ACM SIGSAC Conference on Computer and Communications Security (CCS)}, pages={308--318}, year={2016}, note={arXiv:1607.00133}}
@misc{bonawitz2017practical, title={Practical Secure Aggregation for Privacy Preserving Machine Learning}, author={Bonawitz, Keith and Ivanov, Vladimir and Kreuter, Ben and Marcedone, Antonio and McMahan, H. Brendan and Patel, Sarvar and Ramage, Daniel and Segal, Aaron and Seth, Karn}, howpublished={Cryptology ePrint Archive, Paper 2017/281}, year={2017}}
@inproceedings{mcmahan2017communication, title={Communication-Efficient Learning of Deep Networks from Decentralized Data}, author={McMahan, H. Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and Ag{\"u}era y Arcas, Blaise}, booktitle={International Conference on Artificial Intelligence and Statistics (AISTATS)}, year={2017}, note={arXiv:1602.05629}}
@inproceedings{pelikan2025enabling, title={Enabling Differentially Private Federated Learning for Speech Recognition: Benchmarks, Adaptive Optimizers and Gradient Clipping}, author={Pelikan, Martin and Azam, Sheikh Shams and Feldman, Vitaly and Silovsky, Jan and Talwar, Kunal and Brinton, Christopher G. and Likhomanenko, Tatiana}, booktitle={Advances in Neural Information Processing Systems}, year={2025}, note={arXiv:2310.00098}}
@article{shoemate2022sotto, title={Sotto Voce: Federated Speech Recognition with Differential Privacy Guarantees}, author={Shoemate, Michael and Jett, Kevin and Cowan, Ethan and Colbath, Sean and Honaker, James and Muthukumar, Prasanna}, journal={arXiv preprint arXiv:2207.07816}, year={2022}}
@inproceedings{chauhan2024training, title={Training Large {ASR} Encoders with Differential Privacy}, author={Chauhan, Geeticka and Chien, Steve and Thakkar, Om and Thakurta, Abhradeep and Narayanan, Arun}, booktitle={IEEE Spoken Language Technology Workshop (SLT)}, year={2024}, note={arXiv:2409.13953}}
@article{liu2024differentially, title={Differentially Private Parameter-Efficient Fine-tuning for Large {ASR} Models}, author={Liu, Hongbin and Wang, Lun and Thakkar, Om and Thakurta, Abhradeep and Narayanan, Arun}, journal={arXiv preprint arXiv:2410.01948}, year={2024}}
@article{sun2020provable, title={Provable Defense against Privacy Leakage in Federated Learning from Representation Perspective}, author={Sun, Jingwei and Li, Ang and Wang, Binghui and Yang, Huanrui and Li, Hai and Chen, Yiran}, journal={arXiv preprint arXiv:2012.06043}, year={2020}}
@inproceedings{scheliga2022precode, title={{PRECODE} - A Generic Model Extension to Prevent Deep Gradient Leakage}, author={Scheliga, Daniel and M{\"a}der, Patrick and Seeland, Marco}, booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}, year={2022}, note={arXiv:2108.04725}}
@inproceedings{gao2021privacy, title={Privacy-preserving Collaborative Learning with Automatic Transformation Search}, author={Gao, Wei and Guo, Shangwei and Zhang, Tianwei and Qiu, Han and Wen, Yonggang and Liu, Yang}, booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, year={2021}, note={arXiv:2011.12505}}
@inproceedings{yue2023gradient, title={Gradient Obfuscation Gives a False Sense of Security in Federated Learning}, author={Yue, Kai and Jin, Richeng and Wong, Chau-Wai and Baron, Dror and Dai, Huaiyu}, booktitle={USENIX Security Symposium}, year={2023}, note={arXiv:2206.04055}}

@inproceedings{graves2006connectionist, title={Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks}, author={Graves, Alex and Fern{\'a}ndez, Santiago and Gomez, Faustino and Schmidhuber, J{\"u}rgen}, booktitle={International Conference on Machine Learning (ICML)}, year={2006}}
@article{hannun2017sequence, title={Sequence Modeling with {CTC}}, author={Hannun, Awni}, journal={Distill}, year={2017}, doi={10.23915/distill.00008}}
@article{zeyer2021does, title={Why does {CTC} result in peaky behavior?}, author={Zeyer, Albert and Schl{\"u}ter, Ralf and Ney, Hermann}, journal={arXiv preprint arXiv:2105.14849}, year={2021}}
@article{zeyer2026gradient, title={Gradient-Based Speech-to-Text Alignment for Any {ASR} Model: From {CTC} to Speech {LLMs}}, author={Zeyer, Albert and Schl{\"u}ter, Ralf and Ney, Hermann}, journal={arXiv preprint arXiv:2607.06831}, year={2026}}
@misc{pytorch2019ctc, title={Higher order gradients of {CTCLoss}}, howpublished={PyTorch Forums thread, \url{https://discuss.pytorch.org/t/higher-order-gradients-of-ctcloss/35019}}, year={2019}}
@misc{kantorov_ctc, title={ctc: A primer on {CTC} implementation in pure {Python} {PyTorch} code}, author={Kantorov, Vadim}, howpublished={\url{https://github.com/vadimkantorov/ctc}}}
@misc{tochin_tfseq2seq, title={tf\_seq2seq\_losses: {CTC} loss for {TensorFlow} with second-order derivatives}, author={Tochin, Alexey}, howpublished={\url{https://github.com/alexeytochin/tf_seq2seq_losses}}}
```

---

## Caveats to carry into the rewrite

1. **Zeng25 changes your novelty framing.** "First long-form" is no longer safe. "First long-form reconstruction from a CTC ASR model" and "first first-order gradient matching through CTC" look defensible from what I found, pending your own check of Bui25.
2. **Two entries have metadata-only verification:** Ovi24 and Bui25 (Crossref). Read both before characterizing their methods.
3. **Some venues are left as arXiv on purpose,** because the opened page did not state one: Bayesian Framework, Robbing the Fed, Boenisch et al., APRIL, Soteria, wav2vec 2.0, HuBERT, Vocos, EnCodec, Wang23, Mdhaffar21, Feng21. Fill them in from the official proceedings pages.
4. **The Section 6 closed forms and the RLG rank-condition remark are my derivations.** Check them numerically (rank of FC gradients against T, sign pattern of the output bias gradient) before putting them in the paper.
5. **Full-text details tagged "(ar5iv)" came through an automated page summarizer.** Confirm the Dang-Spk DP-SGD figure and the RLG defense numbers against the PDFs before quoting them.