# Forecasting Conversation Derailment with Chain-of-Thought and Social-Signal Prompting

Can a small open LLM tell that an online discussion is about to turn into a personal attack, **without any training of its weights**, only by changing the prompt? This repository tests five prompting strategies on Reddit conversations and analyses where the model's reasoning helps and where it breaks.

*Individual contribution to a group project on LLMs for human–artificial agent interaction (generative AI × social signal processing).*

## At a glance

| | |
|---|---|
| Model | [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), FP16, **no fine-tuning** |
| Data | ConvoKit **CGA-CMV** (Reddit r/ChangeMyView), official split |
| Methods | 5 prompts: base, social-signal checklist, chain-of-thought (CoT), checklist + CoT, few-shot CoT |
| Best ROC-AUC | 0.624 (few-shot CoT); no variant is significantly better than the base prompt (McNemar) |
| Main qualitative finding | CoT reasoning often sounds plausible but is **unfaithful** to the conversation; few-shot CoT fails the sarcasm crash test |

## Application and design

- **Application:** a moderation assistant that forecasts whether a discussion thread will derail, so that human moderators can look at risky threads before a personal attack happens.
- **Agent:** non-embodied, text only. Input: the conversation so far. Output: a risk score from 0 to 1 and, for CoT prompts, a written explanation.
- **Intended users:** forum moderators. The assistant only flags threads; a human decides whether and how to intervene, and handles appeals. It never removes content by itself.

## Why this model

- **Open weights** (Apache 2.0), so every step can be reproduced and inspected.
- **Small enough for a free Google Colab T4 GPU** (about 3 GB of memory in FP16), which matters because CoT means generating text for every conversation.
- **Instruction-tuned with a chat template**, so system prompts, few-shot turns and step-by-step reasoning can be tested directly.

## Data

- **Corpus:** `conversations-gone-awry-cmv-corpus` from [ConvoKit](https://convokit.cornell.edu/documentation/awry_cmv.html), downloaded automatically by the notebook (not stored in this repository).
- **Split:** official train / validation / test = 4,106 / 1,368 / 1,368 conversations.
- **Label:** `has_removed_comment`, whether the final comment was removed by moderators under CMV Rule 2 (rude or hostile behaviour).
- **Balance:** each derailing conversation is paired with a civil one, so every split is 50/50. Guessing gives 0.5 accuracy and 0.5 ROC-AUC.
- **Input:** every comment **except the last** (`all_but_last`), at most 600 characters per comment, usernames replaced by `Speaker 1`, `Speaker 2`, …

## Method

### Five prompt variants

All variants use the same model, data, question (*"Will this conversation derail?"*) and evaluation. Only the system prompt (and, for E, the examples) changes.

| Variant | Social-signal checklist | Step-by-step reasoning | Examples from train set |
|---|---|---|---|
| `A_base` | – | – | – |
| `B_social` | yes | – | – |
| `C_cot` | – | yes | – |
| `D_social_cot` | yes | yes | – |
| `E_fewshot_cot` | yes | yes | 4 |

- **Social-signal checklist:** hostility or rude language; criticism of the person rather than the argument; sarcasm or mocking; confrontational questions and accusatory "you"; tension that rises from turn to turn. Politeness markers (hedging, acknowledging the other side, gratitude) lower the risk. Strong disagreement alone is not a personal attack. Motivated by Zhang et al. (2018).
- **Few-shot examples (E):** 4 short conversations from the **training split** (2 derailed, 2 civil, never from the same pair), shown as earlier chat turns with a short reasoning and an answer. The reasoning was first written by the model given the correct label (label-conditioned rationales, as in STaR), then **checked by hand: two of the four contained invented details and were rewritten.** The final examples are in `results/fewshot_examples.json`.

### From reasoning to a risk score

1. Direct variants (A, B): one forward pass; score = probability of `yes` relative to `no` for the next token.
2. CoT variants (C, D, E): the assistant turn is pre-filled with `Reasoning:` (without this, Qwen tended to answer first and explain afterwards), the reasoning is generated greedily (at most 256 tokens), then `Answer:` is appended and the yes/no probability of the next token is read.

This gives every variant a continuous score, so all of them are evaluated the same way.

### Evaluation

- Threshold chosen on **validation** accuracy from {0.05, 0.10, …, 0.95}, then fixed and applied **once** to the test set.
- Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix. **ROC-AUC is the main metric**: it does not depend on the threshold, and on balanced data F1 is misleading (always answering "yes" already gives F1 = 0.667).
- **McNemar tests** (exact binomial) compare each variant's test decisions with the base prompt.
- Prompts were checked on a 100-conversation validation sample before the full run; they were never tuned on the test set.

## Results

### Test set (n = 1,368)

| Variant | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| A_base | 0.85 | 0.575 | 0.598 | 0.456 | 0.517 | 0.603 |
| B_social | 0.85 | 0.595 | 0.592 | 0.610 | 0.601 | 0.619 |
| C_cot | 0.15 | 0.568 | 0.582 | 0.481 | 0.527 | 0.592 |
| D_social_cot | 0.95 | 0.599 | 0.621 | 0.509 | 0.560 | 0.611 |
| E_fewshot_cot | 0.80 | 0.591 | 0.570 | 0.741 | 0.645 | 0.624 |

### Significance

| Comparison | Errors fixed | Errors introduced | McNemar p |
|---|---|---|---|
| B_social vs. A_base | 125 | 97 | 0.070 |
| C_cot vs. A_base | 210 | 219 | 0.699 |
| D_social_cot vs. A_base | 240 | 206 | 0.118 |
| E_fewshot_cot vs. A_base | 269 | 246 | 0.332 |
| E_fewshot_cot vs. D_social_cot | 245 | 256 | 0.655 |

### Findings

1. **Naming the social signals helps a little; reasoning alone does not.** The checklist (B) gives the most consistent gain (p = 0.07). CoT alone (C) changes about a third of the decisions but fixes as many errors as it introduces.
2. **Few-shot CoT reaches the highest AUC (0.624) but is not significantly better.** It also leans towards "yes": recall 0.741 with 382 false positives, compared with 210 for the base prompt.
3. **CoT makes scores extreme** (most near 0 or 1), which pushes the chosen thresholds to 0.15–0.95 and reduces how much the ranking can say.
4. **Few-shot examples made the model over-sensitive.** E has the most false positives on the test set and was the only variant to flag the sarcasm crash test.
5. **Small samples mislead.** On the 100-conversation check, C had the best AUC and E the worst; on the full test set it was the other way round.
6. **The reasoning is often unfaithful.** It summarises who said what instead of judging the signals, mixes up speakers, and invents details (see below). Even when given the correct answer, the model invented details in two of four example rationales.

### Scenario tests and crash test

Risk scores; a score at or above the variant's threshold means "flag".

| Scenario | Expected | A | B | C | D | E |
|---|---|---|---|---|---|---|
| Polite disagreement | no derail | 0.05 | 0.03 | 0.00 | 0.00 | 0.00 |
| Escalating tension | derail | 0.88 | 0.95 | 0.88 | 1.00 | 1.00 |
| Crash test: friendly sarcasm | no derail | 0.16 | 0.35 | 0.00 | 0.00 | **1.00 (wrong)** |

A–D made every decision correctly, but the explanations were not always right, and E failed the crash test:

- **Few-shot CoT (E) flagged the friendly banter as an attack** with a score of 1.00. Its reasoning says Speaker 1 "calls Speaker 2 'guilty'", although Speaker 2 said "Haha guilty" about himself, and concludes that the light-hearted tone is "critical and dismissive". This matches E's high false-positive count on the test set.

- **Crash test**, "You absolute genius, you broke the build again :)": C read the sarcasm as a sincere compliment and invented a promise to "fix it soon" (the reply was "buying coffee tomorrow").
- The checklist raised B's score for the friendly banter from 0.16 to 0.35: naming sarcasm makes teasing look riskier.
- **Real test case `df3b8t7`** (derailed): D said Speaker 4 criticised an animal analogy that Speaker 4 had made himself, claimed a position that Speaker 4 explicitly rejected, and missed clear signals such as profanity and an ableist slur.

### Early-warning test (optional, A and B only)

The same test conversations, but the model sees only the first 1, 2 or 3 comments (never the last one), with the full-context validation threshold kept fixed.

| Context given | A_base AUC | B_social AUC |
|---|---|---|
| First comment | 0.528 | 0.535 |
| First 2 comments | 0.545 | 0.560 |
| First 3 comments | 0.559 | 0.585 |
| All but the last (main task) | 0.603 | 0.619 |

With only the opening comments the model is close to chance: warning very early is much harder than forecasting the final comment.

## Human–AI teaming

The assistant ranks threads by risk and flags those above the threshold; the CoT explanation is shown as a pointer, not as evidence. Because the explanations can be fluent but wrong, moderators should always check them against the original conversation before acting.

## Ethical considerations

- **Errors have costs:** false positives may silence legitimate heated debate; false negatives miss real harm.
- **Automation bias:** convincing but unfaithful explanations can make moderators over-trust the model.
- **Base rates:** derailment is rare in practice, not 50/50, so precision in deployment would be far lower than here and the threshold would need recalibrating.
- **Bias and privacy:** labels reflect CMV moderators' Rule 2 decisions, not a universal definition of a personal attack; conversations may contain personal information.
- The score should never be used for automatic removal.

## Limitations

- One model (1.5B parameters) and one CoT style; results may differ for larger or reasoning-trained models.
- `all_but_last` context can already contain hostility, so this is forecasting the final comment, not warning before any harm.
- Yes/no scores are relative probabilities, not calibrated.
- 100–134 long conversations per variant were shortened by dropping the oldest comments.
- Only three hand-written scenarios, all easier than real data.

## How to run

1. Open `CoT_Conversation_Derailment.ipynb` in [Google Colab](https://colab.research.google.com) and choose **Runtime → Change runtime type → T4 GPU**.
2. Run the cells in order. Section 2 mounts Google Drive; every result is saved batch by batch to `MyDrive/CoT_Project/`, so a disconnected run resumes where it stopped.
3. Approximate time on a T4 per split (1,368 conversations): A and B about 2 minutes, C and D about 12 minutes, E about 28 minutes.

| Section | What it does |
|---|---|
| 1–4 | Install packages, settings, data preparation (identical split checks), load the model |
| 5–7 | Prompt variants, scoring functions, run/resume/evaluate helpers |
| 8 | 100-conversation check of the prompts |
| 9 | Full run of A–D: threshold on validation, test once |
| 10 | Error analysis, A vs. D |
| 11 | Scenario tests and crash test |
| 12 | Optional early-warning test (A and B, first 1–3 comments) |
| 13 | Few-shot CoT (E): example selection, rationales, full run, McNemar tests |

## Repository structure

```
├── CoT_Conversation_Derailment.ipynb   notebook with outputs
├── requirements.txt
└── results/
    ├── results_summary_test_with_E.csv  metrics of A–E on the test set
    ├── fewshot_examples.json            the 4 examples used by E
    ├── scenario_results.csv             scenario tests and crash test
    ├── early_warning_results_cot.csv    early-warning test
    ├── error_analysis_A_base_vs_D_social_cot.csv
    └── predictions/                     per-conversation scores, reasoning and raw output (val and test)
```

## References

- Chang, J. P., & Danescu-Niculescu-Mizil, C. (2019). Trouble on the Horizon: Forecasting the Derailment of Online Conversations as they Develop. *EMNLP*.
- Kojima, T., et al. (2022). Large Language Models are Zero-Shot Reasoners. *NeurIPS*.
- Qwen Team (2024). Qwen2.5 Technical Report. arXiv:2412.15115.
- Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS*.
- Zelikman, E., et al. (2022). STaR: Bootstrapping Reasoning With Reasoning. *NeurIPS*.
- Zhang, J., et al. (2018). Conversations Gone Awry: Detecting Early Signs of Conversational Failure. *ACL*.
