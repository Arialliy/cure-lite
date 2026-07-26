# Search Notes

Date: 2026-07-24  
Mode: standard  
Scope: read-only novelty scan; no CURE code or experiment state was modified

## Safe Queries Used

Only public topic phrases and public paper titles were used. No local paths, repository-specific unpublished names, private protocol text, or unpublished numerical results were included in any query.

- `infrared small target detection residual correction completion post-processing`
- `infrared small target detection missed detection false alarm`
- `infrared small target target-level insensitivity posterior distribution`
- `infrared small target plug-and-play loss cross-backbone false alarm`
- `infrared small target lightweight plug-in existing frameworks`
- `model-agnostic segmentation refinement coarse mask`
- `universal mask refinement SAMRefiner ICLR`
- `self-repairing segmentation masks rNCA MIDL PMLR`
- `frozen detector lightweight mask decoder`
- `before after mask difference objective segmentation`
- `same image counterfactual mask removal segmentation correction`
- Exact-title verification queries for the final candidate papers

## Sources Checked

- CVF Open Access for ICCV/CVPR papers
- ICLR official proceedings and OpenReview
- NeurIPS official proceedings
- PMLR for MIDL
- ECVA for ECCV
- arXiv for stable preprint records and recent 2026 work
- DBLP and DOI/publisher records for venue and bibliographic cross-checking
- ACM MM OpenReview record for the generic-segmentation IRSTD baseline

Discovery snippets were not used as the sole evidence for papers in the final table. Final entries point to a primary or stable paper record.

## Candidate Screening

Twenty-six candidates were screened; fifteen were retained in the scored table.

| Candidate | Decision | Reason |
| --- | --- | --- |
| Miss Detection vs. False Alarm, ICCV 2019 | Retain | Direct IRSTD problem-framing prior |
| Self-Supervised Difference Detection, ICCV 2019 | Retain | Direct before/after mask-difference prior |
| CascadePSP, CVPR 2020 | Retain | Class-agnostic appended refinement |
| SegFix, ECCV 2020 | Retain | Model-agnostic post-processing/boundary prior |
| ISNet, CVPR 2022 | Retain | Strong IRSTD method and IRSTD-1K source |
| OSCAR, IEEE TGRS 2023 | Retain | IRSTD refinement terminology and cascade mechanism |
| SegRefiner, NeurIPS 2023 | Retain | Strong model-agnostic refinement baseline |
| Mask Frozen-DETR, CoRR 2023 | Retain | Frozen detector plus lightweight mask-network pattern |
| MSHNet/SLS, CVPR 2024 | Retain | Cross-detector loss/generalization and baseline relevance |
| IRSTD-Diff, arXiv 2024 | Retain | Closest target-level-insensitivity problem framing |
| Generic Segmentation Model for IRSTD, ACM MM 2024 | Retain | Generic/lightweight IRSTD claim boundary |
| SAMRefiner, ICLR 2025 | Retain | Universal coarse-mask refinement; mandatory closest work |
| rNCA, MIDL 2026 | Retain | Arbitrary base masks, image context, synthetic corruptions; mandatory closest work |
| NS-FPN, CVPR 2026 | Retain | Plug-in IRSTD feature module and false-alarm focus |
| AC-SLSIoU, arXiv 2026 | Retain | Very recent cross-backbone, logit-domain, false-alarm loss |
| DNANet, IEEE TIP 2023 | Screened, not final | Important backbone baseline but farther from the correction mechanism |
| UIU-Net, IEEE TIP 2023 | Screened, not final | Important nested IRSTD architecture but not a generic correction operator |
| SCTransNet, IEEE TGRS 2024 | Screened, not final | Important spatial-channel architecture baseline but not a post-detector plugin |
| ALCNet, IEEE TGRS 2021 | Screened, not final | Model-driven local-contrast backbone; background reference |
| AGPCNet, IEEE TAES 2023 | Screened, not final | Context-architecture baseline; weaker mechanism overlap |
| SAIST, CVPR 2025 | Screened, not final | Foundation/text-guided IRSTD line; different problem and mechanism |
| Text-IRSTD, ICCV 2025 | Screened, not final | Cross-modal IRSTD; different mechanism |
| SIRST-5K, arXiv 2024 | Screened, not final | Dataset/self-supervision relevance but not close to paired correction |
| Adjustable Sensitivity post-processing, arXiv 2024 | Screened, not final | Relevant post-processing signal but weaker source/status and less direct than retained work |
| FrozenSeg, CoRR/OpenReview 2024 | Screened, not final | Frozen foundation-model segmentation pattern, already represented more directly |
| RefineMask, CVPR 2021 | Screened, not final | Integrated high-quality instance-mask refinement, less model-agnostic than retained refiners |

## Excluded Sources

- MDPI-domain papers, reviews, journals, proceedings, and PDFs were policy-excluded and do not appear in the scored paper table.
- Search-result-only pages without a stable title/author/venue record were excluded.
- Blog summaries, commercial paper summaries, and social-media discussions were not used as evidence.
- Duplicate workshop/arXiv versions were collapsed in favor of the strongest official or stable record.

## Searched Findings

The following statements are directly supported by retrieved papers:

1. Universal/model-agnostic segmentation refinement is already a well-developed research line.
2. Refiners can already consume coarse masks from different base models, use image context, and train on imperfect or synthetically corrupted masks.
3. Freezing a detector and training a lightweight mask network is established as an efficient architecture pattern.
4. IRSTD literature already frames miss detection, false alarm, weak-target discrimination, target-level objective insensitivity, and cross-backbone improvement as explicit research problems.
5. Predicting before/after mask differences is an established segmentation-learning mechanism.

## Inference From The Bounded Search

The following statements are interpretations, not direct literature facts:

1. No exact retrieved method combines all of:
   - an arbitrary frozen IRSTD detector;
   - a separate residual completion decoder;
   - a same-source target-coverage intervention;
   - a coupled finite-difference objective on pre-hard-mask detector/decoder response;
   - zero-order factual-miss and no-miss anchors;
   - natural-miss recovery under frozen false-alarm and retention constraints;
   - cross-backbone validation with only the correction plugin retrained.
2. The likely CURE novelty is therefore **conjunctive**: the learning object, controlled intervention, paired objective, and evidence protocol must function together.
3. A new decoder architecture alone is unlikely to establish ICLR-level novelty; a generic refiner framing is directly exposed to SAMRefiner, SegRefiner, and rNCA.
4. The search does not prove global priority. Differently named, unpublished, or poorly indexed work may exist.

## Innovation Risks

### High-risk claims

- “the first universal/model-agnostic segmentation refiner”
- “the first detector-agnostic post-processing plugin”
- “the first frozen detector with a lightweight correction head”
- “the first segmentation method learning from before/after masks”
- “the first IRSTD method balancing missed detections and false alarms”
- “the first plug-and-play or cross-backbone IRSTD method”
- “the first method to address target-level insensitivity”

### Defensible working boundary

The current line can remain differentiated if implementation and experiments support:

> A post-frozen-detector completion operator learned from coupled same-source target-coverage responses, with absolute-state anchors and natural-miss recovery evaluated under fixed false-alarm and detected-target retention constraints.

This sentence should remain a working description, not a priority claim, until direct comparisons and controls are complete.

## Model-Oriented Consequence

The literature scan should now end and feed the code plan. The next implementation gate should not be another open-ended proof exercise. It should establish:

1. the exact paired training sample and intervention builder;
2. a decoder with sufficient small-target spatial resolution;
3. a coupled loss that directly consumes both states;
4. absolute-state anchors and a frozen-base composition rule;
5. generic-refiner and separable-loss controls;
6. a minimal real training run before cross-backbone expansion.

If the coupled pair can be removed without changing performance, or a generic refiner matches CURE under the same frozen-base inputs and constraints, the proposed novelty boundary is not supported and the model must be redesigned.

## Unknowns

- **Papers not accessible:** no final-table paper was inaccessible at the metadata/abstract level; full appendices were not exhaustively re-read for every candidate.
- **Venue status not verified:** AC-SLSIoU and IRSTD-Diff remain preprints in this report. Mask Frozen-DETR is recorded only as CoRR/arXiv. No accepted-venue claim is made for them.
- **Missing benchmark details:** metric implementations and dataset splits differ across papers; reported values were not pooled or compared numerically.
- **Search limitation:** terminology around interventions, counterfactual masks, completion, repair, and refinement is inconsistent, so an exact-mechanism paper could be indexed under other language.
- **Current evidence limitation:** no CURE performance result was assumed or invented.

## Handoff Notes

- **For writing:** position against universal mask refinement, synthetic-corruption repair, target-level-insensitivity, and plug-and-play IRSTD work; avoid all unsafe priority claims listed above.
- **For idea optimization:** keep the same-source coupled finite-difference learning object central. Freezing, lightweight decoding, and generic compatibility are constraints, not standalone innovations.
- **For direction scouting:** only reopen search when the final mathematical objective and model input contract are frozen; then run an exact formula/mechanism search.
- **For experiment design:** include SAMRefiner/SegRefiner/rNCA-style generic refinement controls where technically compatible, plus separable endpoint loss, shuffled pair, occupancy-only, feature-only, and no-intervention controls.
- **For review:** require a claim-to-prior table and verify that every “different from” statement is supported by code and matched evidence rather than terminology.
