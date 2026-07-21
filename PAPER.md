<div style="page-break-after: always; min-height: 190mm; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;">
  <h1>Real-Document Evaluation of Text-Only and Layout-Aware LLM Parsing on FUNSD</h1>
  <h2>Choi Seok Won</h2>
  <p>Independent Researcher</p>
  <h3>Independent Preprint · July 2026</h3>
</div>

<div style="page-break-after: always;">
  <h1>Table of Contents</h1>
  <ul>
    <li>1. Introduction</li>
    <li>2. Background and Related Work</li>
    <li>3. Methods
      <ul>
        <li>3.1 Research Questions</li>
        <li>3.2 Real-Document Dataset and Fixed Sample</li>
        <li>3.3 OCR and Layout Boundary</li>
        <li>3.4 Model and Prompt Conditions</li>
        <li>3.5 Hybrid Routing Rule</li>
        <li>3.6 Repeated-Trial Protocol and Metrics</li>
        <li>3.7 Reproduction Procedure</li>
      </ul>
    </li>
    <li>4. Results</li>
    <li>5. Discussion</li>
    <li>6. Threats to Validity</li>
    <li>7. Conclusion</li>
    <li>Research Transparency Statements</li>
    <li>References</li>
  </ul>
</div>

<div style="page-break-after: always;">
  <h1>List of Tables</h1>
  <ul>
    <li>Table 3-1. Fixed FUNSD Test Sample</li>
    <li>Table 3-2. Experimental Conditions</li>
    <li>Table 4-1. Aggregate Results Across Three Trials</li>
    <li>Table 4-2. Mean Page-Level Results</li>
    <li>Table 4-3. Result-to-Artifact Traceability</li>
  </ul>
</div>

<div style="page-break-after: always;">
  <h1>Abstract</h1>
  <p>This paper reports a repeated real-document experiment on semantic entity classification in scanned forms. Four fixed pages from the official FUNSD test split provide 163 nonempty OCR entities with real text, bounding boxes, and gold labels. The same locally served language model, <code>openai/qwable-3.6-27b-mtp</code>, classified each entity as header, question, answer, or other under text-only and layout-aware prompts. Three trials used seeds 17, 29, and 43 with temperature 0.2 and top-p 0.9. Layout-aware prompting increased pooled entity accuracy from 0.7791 ± 0.0061 to 0.8262 ± 0.0035 and mean page macro-F1 from 0.6900 ± 0.0044 to 0.7247 ± 0.0090. A label-blind hybrid rule used layout on two of four pages and achieved accuracy 0.8119 ± 0.0035 and mean page macro-F1 0.7077 ± 0.0059. Layout-aware prompts consumed a mean 7,710 input tokens, compared with 2,915 for text-only prompts. Six uniform successful comparison calls were selected from eleven recorded attempts; four earlier attempts were truncated by the completion limit, and one otherwise complete pilot used a different reasoning setting and was excluded. The public package includes source identifiers and hashes, prompt code, page-level metrics, aggregate results, and an execution log. The official FUNSD archive and raw OCR/model responses remain in the local audit archive and are not redistributed. The small fixed sample and locally served model limit external validity, but the reported observations are directly auditable.</p>
  <p><b>Keywords:</b> document understanding, FUNSD, layout-aware prompting, large language models, semantic entity classification, reproducibility</p>
</div>

# 1. Introduction

Scanned forms encode meaning through both language and position. A phrase such as “Account number” is likely a field name, while nearby text may be its filled value. A parser that receives OCR text alone can use lexical cues, but a parser that also receives bounding boxes can exploit alignment, proximity, and page structure. Layout-aware document models formalize this intuition by combining textual and spatial representations (Xu et al., 2020; Wang et al., 2024).

An empirical comparison must use actual documents, an identified model, repeated executions, and inspectable run records. The previous version of this manuscript used authored layout descriptors and deterministic parser surrogates. That experiment was useful as a software self-check but could not support a claim about real documents or a real language model. The present study replaces it in full.

The revised experiment uses four scanned forms from the official FUNSD test split (Jaume et al., 2019). FUNSD supplies real page images, OCR entity text, bounding boxes, and semantic labels. A locally served model is evaluated under matched text-only and layout-aware conditions across three sampling seeds. A simple hybrid policy selects the layout-aware result for pages with at least 40 nonempty entities and the text-only result otherwise. Every reported number is derived from locally retained raw responses and the publicly released page-level scores.

The contribution is deliberately bounded. This is not a new pretrained document model, an OCR benchmark, or a claim of state-of-the-art performance. It is an auditable case study of whether adding FUNSD geometry to the prompt changes entity classification by one concrete model, and whether a fixed routing rule can recover part of that change while using the layout condition on only half of the sampled pages.

# 2. Background and Related Work

FUNSD was introduced as a dataset for form understanding in noisy scanned documents. Its semantic entities have identifiers, OCR text, bounding boxes, one of four labels—header, question, answer, or other—and links between related entities (Jaume et al., 2019). The original release contains 199 annotated images and defines an official split of 149 training and 50 testing images. Its real scans make it more informative than authored geometric examples, while its small size and domain concentration remain important limitations.

LayoutLM jointly models words and two-dimensional positions and established a widely used layout-aware pretraining approach for document image understanding (Xu et al., 2020). LayoutLMv2 further integrated text, layout, and visual information in a multimodal architecture (Xu et al., 2021). More recent work has connected layout signals to generative language models. DocLLM, for example, incorporates bounding-box information without requiring a vision encoder, demonstrating that spatial coordinates can be useful to language-model-based document understanding (Wang et al., 2024). LayoutLLM likewise studies instruction tuning for visually rich documents (Fujitake, 2024).

The present work differs from those trained architectures. It does not modify model weights or process page pixels. It serializes published OCR entities into an OpenAI-compatible chat request, with or without normalized boxes, and evaluates the returned labels. Consequently, its conclusions concern prompt-level access to layout for one model endpoint rather than a comparison among document foundation models.

Conditional computation motivates the hybrid condition. Systems such as BranchyNet allocate different computation according to input difficulty (Teerapittayanon et al., 2016). Here, a fixed page-level rule uses entity count as a label-blind proxy for document complexity. This rule is intentionally simple and is evaluated without tuning on gold outcomes.

# 3. Methods

## 3.1 Research Questions

The study addresses three questions.

* **RQ1:** On the fixed real-document sample, how does providing normalized bounding boxes change semantic entity classification relative to OCR text alone?
* **RQ2:** How much trial-to-trial variation is observed across three seeded sampling runs with otherwise identical settings?
* **RQ3:** What accuracy and macro-F1 are obtained by a fixed hybrid rule that uses layout-aware outputs on pages with at least 40 nonempty entities?

The study does not evaluate OCR recognition, visual encoders, training, fine-tuning, wall-clock production latency, accelerator memory, monetary cost, or cross-model generalization.

## 3.2 Real-Document Dataset and Fixed Sample

The experiment uses the original FUNSD archive from the official project website. The downloaded archive is 16,838,830 bytes and has SHA-256 digest `c31735649e4f441bcbb4fd0f379574f7520b42286e80b01d80b445649d54761f`. It was retrieved on 20 July 2026. The exact source URL, terms URL, archive digest, internal paths, page dimensions, and per-file hashes are recorded in `artifacts/source_manifest.json`.

The sample comprises the first four annotation filenames in lexicographic order within the official test split. This selection rule and the four identifiers were fixed before model execution. Entities with empty OCR text were excluded because neither prompt condition contained usable language for them. Seven entities were excluded, leaving 163 evaluated entities.

**Table 3-1. Fixed FUNSD Test Sample**

| Page identifier | Nonempty entities | Empty-text exclusions | Hybrid source |
|---|---:|---:|---|
| `82092117` | 27 | 1 | Text only |
| `82200067_0069` | 59 | 2 | Layout aware |
| `82250337_0338` | 36 | 2 | Text only |
| `82251504` | 41 | 2 | Layout aware |
| **Total** | **163** | **7** | **2 of 4 layout aware** |

The FUNSD terms restrict use to non-commercial research and educational purposes, identify the images as copyrighted, and make the licensee responsible for additional permissions. The local archive is therefore an audit input, not an artifact to redistribute. `artifacts/DATASET_LICENSE_NOTICE.md` directs future users to the official source and terms. The manifest and hashes allow an independently downloaded copy to be checked against the experimental input.

## 3.3 OCR and Layout Boundary

Tesseract was not installed in the experimental environment. OCR was therefore not rerun. Both conditions use FUNSD’s published semantic-entity text. The layout-aware condition additionally uses FUNSD’s entity boxes. Each box `[left, top, right, bottom]` is normalized independently by the page width and height:

$$
\tilde{b}_i = \left[\frac{x_{1i}}{W},\frac{y_{1i}}{H},\frac{x_{2i}}{W},\frac{y_{2i}}{H}\right].
$$

This design uses real OCR annotations and real document geometry but does not measure OCR engine quality. The page images remain part of the source archive and are hashed in the manifest, yet the language model receives no pixels.

## 3.4 Model and Prompt Conditions

All successful comparison calls used the OpenAI-compatible endpoint `http://private-lan-llm-host:11440/v1` and model identifier `openai/qwable-3.6-27b-mtp`. The returned model identifier matches the requested identifier, and the successful responses share system fingerprint `b9894-a8cfdbb9e`. No public model card or immutable weight digest was supplied by the endpoint; the identifier and fingerprint are therefore the strongest model identity available for this run.

Each request batches all four pages and requires one nested JSON object mapping page identifiers and entity identifiers to FUNSD labels. Both conditions use the same system instruction and dataset ID order.

**Table 3-2. Experimental Conditions**

| Condition | Input per entity | Successful calls | Mean prompt tokens | Layout-aware pages |
|---|---|---:|---:|---:|
| Text only | Identifier and OCR text | 3 | 2,915 | 0 |
| Layout aware | Identifier, OCR text, and normalized box | 3 | 7,710 | 4 |
| Hybrid | Matched seeded output selected by fixed route | Post hoc | Not reported | 2 |

The embedding endpoint and `bge-m3` were not invoked because this experiment contains no embedding, retrieval, clustering, or similarity operation. Adding an unused embedding stage would not test any research question.

## 3.5 Hybrid Routing Rule

Let $n(P)$ be the number of nonempty entities on page $P$. The hybrid source is

$$
h(P)=
\begin{cases}
\text{layout-aware}, & n(P)\geq 40,\\
\text{text-only}, & n(P)<40.
\end{cases}
$$

For each seed, the hybrid score selects the corresponding page result from the matched text-only or layout-aware call. The threshold uses no gold labels or measured outcomes. It routes pages `82200067_0069` and `82251504` to the layout-aware source and the remaining two pages to text only.

No separate hybrid API request was issued. Consequently, hybrid prompt tokens and end-to-end latency are not reported. The defensible resource observation is limited to route selection: layout information is used on two of four pages. The hybrid results should be interpreted as a post-hoc policy evaluation over matched seeded predictions, not as a seventh model configuration.

## 3.6 Repeated-Trial Protocol and Metrics

Three trials used seeds 17, 29, and 43, temperature 0.2, top-p 0.9, JSON response mode, and thinking-disabled generation. The endpoint’s seed implementation is not independently documented, so “seeded trials” describes the submitted requests rather than a guarantee of statistically independent random streams. Observed variation is reported without an inferential significance test because three trials and four pages do not justify one.

For a page with $N$ evaluated entities, accuracy is

$$
\mathrm{Accuracy}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\hat{y}_i=y_i].
$$

For each of the four labels, precision, recall, and F1 are computed at entity level. Page macro-F1 is the unweighted mean of the four label F1 values; if a label has neither a true nor predicted instance on a page, its F1 contribution is defined as zero. All four labels are present in the gold annotations of every sampled page, so that convention does not change the reported values. The reported accuracy is pooled across all 163 entities within a trial. The reported macro-F1 is the mean of the four page macro-F1 values. Table entries use the arithmetic mean and sample standard deviation across the three seeds.

The first execution phase used the model’s default reasoning behavior and an 8,192-token completion limit. Four responses exhausted that limit without a complete result. One complete pilot response used that default reasoning setting. To make the comparison uniform, it was excluded and repeated with thinking disabled, as were the failed batches. The artifact therefore records eleven attempts: six uniform successful calls selected for analysis, four truncated calls, and one excluded nonuniform pilot. No missing or invalid label remains in a selected call.

## 3.7 Reproduction Procedure

From the paper directory, run:

```text
python artifacts/run_experiment.py
```

The script uses only the Python standard library. It validates the FUNSD archive digest, loads the fixed test pages, constructs both prompt conditions, calls the specified endpoint sequentially with transport and structured-output retry plus per-attempt checkpointing, requires exact page/entity coverage with no additional identifiers, calculates page-level and aggregate metrics, and writes the manifest and log. In the complete local audit archive, existing uniform responses can be reused and `--verify-only` prohibits network recovery. The public rights-safe package omits those OCR-bearing responses, so public reproduction requires the officially obtained FUNSD archive and a fresh compatible endpoint run.

Because the official FUNSD terms govern the archive, a reproducer must first read and accept those terms and place the official `dataset.zip` at `artifacts/funsd_dataset.zip`. The expected digest is embedded in the script and notice. The endpoint address is configurable through `PAPER2_LLM_URL`, but changing it or the served weights constitutes a replication rather than an exact reproduction.

# 4. Results

All six uniform comparison calls returned complete label mappings for all four pages and all 163 evaluated entities. Table 4-1 summarizes the final aggregate file.

**Table 4-1. Aggregate Results Across Three Trials**

| Condition | Entity accuracy, mean ± SD | Mean page macro-F1, mean ± SD | Mean prompt tokens | Layout-aware pages |
|---|---:|---:|---:|---:|
| Text only | 0.7791 ± 0.0061 | 0.6900 ± 0.0044 | 2,915 | 0/4 |
| Layout aware | 0.8262 ± 0.0035 | 0.7247 ± 0.0090 | 7,710 | 4/4 |
| Hybrid | 0.8119 ± 0.0035 | 0.7077 ± 0.0059 | Not reported | 2/4 |

For RQ1, layout-aware prompting improved mean pooled entity accuracy by 0.0470, or 4.70 percentage points, relative to text only. Mean page macro-F1 increased by 0.0347. The additional layout serialization also increased mean input length from 2,915 to 7,710 prompt tokens, a factor of approximately 2.65. These observations show a quality–input-size trade-off on the fixed sample; they do not establish that boxes will help every page or model.

For RQ2, the sample standard deviation of accuracy was 0.0061 for text only and 0.0035 for layout aware. Macro-F1 variation was 0.0044 and 0.0090, respectively. The narrow ranges describe these three requests only.

For RQ3, the hybrid rule used layout-aware predictions on two pages. Its mean accuracy of 0.8119 lies between text only and layout aware: 3.27 percentage points above text only and 1.43 points below layout aware. Its mean page macro-F1 of 0.7077 is likewise intermediate.

**Table 4-2. Mean Page-Level Results**

| Page identifier | Text-only accuracy | Layout-aware accuracy | Hybrid accuracy | Text-only macro-F1 | Layout-aware macro-F1 | Hybrid macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| `82092117` | 0.7901 | 0.8889 | 0.7901 | 0.7156 | 0.8026 | 0.7156 |
| `82200067_0069` | 0.8475 | 0.8475 | 0.8475 | 0.6045 | 0.5804 | 0.5804 |
| `82250337_0338` | 0.8611 | 0.8519 | 0.8611 | 0.8451 | 0.8262 | 0.8451 |
| `82251504` | 0.6016 | 0.7317 | 0.7317 | 0.5950 | 0.6897 | 0.6897 |

The page-level results qualify the aggregate improvement. Layout increased accuracy on two pages, left one unchanged, and decreased one; macro-F1 decreased on pages `82200067_0069` and `82250337_0338`. This can occur because macro-F1 weights rare labels equally, whereas pooled accuracy is dominated by more frequent question and answer entities. The hybrid rule selects layout for the two pages with 59 and 41 entities, explaining why it captures part, but not all, of the aggregate accuracy gain.

**Table 4-3. Result-to-Artifact Traceability**

| Claim or material | Authoritative artifact |
|---|---|
| Source URL, terms, test IDs, dimensions, and per-file hashes | `artifacts/source_manifest.json` |
| Dataset-use restriction and expected archive digest | `artifacts/DATASET_LICENSE_NOTICE.md` |
| Model, prompts, seeds, routing, validation, metrics, and checkpoints | `artifacts/run_experiment.py` |
| Raw requests and API responses | Locally retained; excluded publicly because they contain restricted FUNSD OCR text |
| Page/seed/condition scores and class-level F1 | `artifacts/per_page_metrics.csv` |
| Trial values, selected response IDs, attempt dispositions, means, standard deviations, and boundaries | `artifacts/aggregate_results.json` |
| Compact run identity and pass status | `artifacts/execution.log` |
| Public-package file digests | `PUBLIC_RELEASE_MANIFEST.json` |

# 5. Discussion

The principal observation is that adding real FUNSD bounding boxes improved pooled entity accuracy for the evaluated model and pages. This is consistent with the document-understanding literature: form semantics often depend on spatial relations that are absent from plain OCR text. The result is not uniform across metrics or pages, however. Layout-aware macro-F1 was slightly lower on two pages even when accuracy increased or remained unchanged. Geometry can help the model recognize dominant field/value patterns without necessarily improving every rare class.

The prompt-token difference is material. Serializing four normalized coordinates per entity raised the input from about 2.9 thousand to 7.7 thousand tokens. A production system must therefore decide where layout is valuable enough to justify the longer context. The simple entity-count route illustrates that decision without claiming an optimal policy. It used layout on half the pages and produced scores between the two baselines.

The hybrid threshold should not be generalized from four pages. It was chosen as a transparent label-blind rule, not fitted by optimization. A larger study should preregister candidate routing features—such as entity density, alignment ambiguity, OCR confidence, and table likelihood—then learn or validate thresholds on a development split separate from the final test set. It should also issue the hybrid request directly so that token use, latency, and any cross-page generation effects are measured rather than inferred from selected baseline outputs.

The recovery history is itself informative for reproducibility. A reasoning-capable model can consume the full completion budget before emitting structured content. Recording only successful calls would conceal that operational failure mode. The raw artifact retains all eleven attempts and the final protocol explicitly disables optional thinking. Future structured-output experiments should pin this setting and validate full identifier coverage before accepting a response.

# 6. Threats to Validity

**Sample size and selection.** The sample contains four pages and 163 nonempty entities. Lexicographic selection is reproducible but not random or stratified, and it cannot characterize the full 50-page FUNSD test split or other document domains.

**Model identity.** The endpoint supplied a model identifier and system fingerprint but no public model card, weight checksum, training-data statement, or server build. The same network address may serve different weights later. The experiment is therefore reproducible at the artifact level and repeatable against the current service, but long-term model identity is not guaranteed.

**Potential benchmark exposure.** The model’s training corpus is undisclosed. FUNSD or derivatives may have appeared in training data, so the scores cannot be assumed to measure uncontaminated generalization.

**OCR boundary.** The experiment uses FUNSD-provided OCR text and entity boxes. It does not include errors introduced by a newly run OCR engine, entity grouping system, or layout detector. The supplied boxes are annotations available at inference time only for this controlled comparison.

**Prompt representation.** Entity ID order is held constant, but the layout-aware JSON is longer and may alter attention independently of geometry. The study does not isolate coordinate utility from generic prompt-length effects.

**Repeated trials.** Three seeds are sufficient to expose some variation but not to estimate a stable sampling distribution. The endpoint’s handling of `seed` is not independently documented. No hypothesis test or confidence interval is claimed.

**Hybrid evaluation.** Hybrid predictions are selected post hoc from matched baseline calls. This isolates the routing policy but does not measure a directly executed hybrid prompt, latency, or token count.

**Licensing and privacy.** FUNSD contains historical scanned forms derived from RVL-CDIP. The official terms restrict redistribution and place responsibility for image permissions on the licensee. The experiment does not assert that the source images are free of personal or sensitive-looking text.

# 7. Conclusion

This study replaces a synthetic parser demonstration with an auditable real-document experiment. Four official FUNSD test forms, 163 nonempty OCR entities, a named local language model, two prompt conditions, and three seeded trials are fully recorded. Layout-aware prompting achieved 0.8262 ± 0.0035 accuracy compared with 0.7791 ± 0.0061 for text only, while using approximately 2.65 times as many prompt tokens. A fixed two-of-four-page hybrid achieved an intermediate 0.8119 ± 0.0035 accuracy.

The evidence supports a narrow conclusion: normalized layout boxes improved aggregate entity classification for this model on this fixed sample, at a substantial input-length cost. Larger licensed samples, direct hybrid execution, frozen public model weights, OCR reruns, and more repeated trials are required before making general performance or deployment claims.

# Research Transparency Statements

**Data and materials availability.** Source identifiers, URLs, hashes, prompt code, page-level metrics, aggregate statistics, code, and logs are included in the public `artifacts` directory. Raw responses remain local because they contain restricted FUNSD OCR text. The FUNSD archive is governed by its official non-commercial research and educational terms and must be obtained from the official source.

**Reproducibility.** Run `python artifacts/run_experiment.py` from the paper directory after placing the digest-matched official FUNSD archive in `artifacts`. The public package requires a fresh live replication through the configured endpoint; no restricted raw response is redistributed.

**Funding.** No external funding supported this study.

**Competing interests.** The author declares no competing interests.

**Ethics and privacy.** No new human-subject data were collected. The study uses an existing licensed dataset of historical scanned forms. Users must follow the dataset terms and assess their own legal and privacy obligations.

**Generative-AI disclosure.** OpenAI ChatGPT/Codex assisted with experiment code, execution orchestration, artifact validation, and English-language editing. The evaluated predictions were produced by the separately identified local endpoint. AI systems are not authors. Choi Seok Won reviewed and remains responsible for the design, code, interpretation, references, and manuscript.

# References

Fujitake, M. (2024). LayoutLLM: Large language model instruction tuning for visually rich document understanding. In *Proceedings of LREC-COLING 2024* (pp. 10219–10224). ELRA and ICCL. https://aclanthology.org/2024.lrec-main.892/

Jaume, G., Ekenel, H. K., & Thiran, J.-P. (2019). FUNSD: A dataset for form understanding in noisy scanned documents. In *2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)* (pp. 1–6). IEEE. https://doi.org/10.1109/ICDARW.2019.10029

Teerapittayanon, S., McDanel, B., & Kung, H. T. (2016). BranchyNet: Fast inference via early exiting from deep neural networks. In *Proceedings of the 23rd International Conference on Pattern Recognition* (pp. 2464–2469). IEEE. https://doi.org/10.1109/ICPR.2016.7900006

Wang, D., Raman, N., Sibue, M., Ma, Z., Babkin, P., Kaur, S., Pei, Y., Nourbakhsh, A., & Liu, X. (2024). DocLLM: A layout-aware generative language model for multimodal document understanding. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 8529–8548). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.acl-long.463

Xu, Y., Li, M., Cui, L., Huang, S., Wei, F., & Zhou, M. (2020). LayoutLM: Pre-training of text and layout for document image understanding. In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 1192–1200). Association for Computing Machinery. https://doi.org/10.1145/3394486.3403172

Xu, Y., Xu, Y., Lv, T., Cui, L., Wei, F., Wang, G., Lu, Y., Florêncio, D., Zhang, C., Che, W., Zhang, M., & Zhou, L. (2021). LayoutLMv2: Multi-modal pre-training for visually-rich document understanding. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)* (pp. 2579–2591). Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.acl-long.201
