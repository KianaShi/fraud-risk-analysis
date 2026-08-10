# Deferred Phase 5: TabPrep representation ablation

Phase 5 is deferred and is not part of the Fraud v1.0 Phases 1–4 reproduction path.

## Completed exploratory work

- Independently verified the TabPrep paper and official repository.
- Inspected the official wrapper and its AutoGluon feature-generator dependencies.
- Created an isolated AutoGluon environment outside the Git repository.
- Passed a leakage-safe synthetic smoke test using 80 Train rows and 24 Validation rows.
- Expanded a synthetic 14-feature representation to 964 transformed features, identifying substantial dimensional-expansion risk.
- Used no real fraud Train, Validation, or Test data during this exploration.

Pinned source revisions:

- TabPrep: `fee7f189764070836b59e8cf29207050468589f0`
- AutoGluon: `07786b4b7bf4349fb0d5c06701c00fb4ce09dab0`

## Local exploratory artifacts

The verified TabPrep and AutoGluon source checkouts, isolated Python environment, and synthetic-only smoke script were local exploratory artifacts outside Git. Their workstation paths are intentionally not tracked. Reconstruct source identity from the pinned revisions above rather than relying on a private cache location.

## Not completed

- Real-data paired cross-validation
- Fixed-Validation TabPrep comparison
- Generator-contribution ablation
- Representation-specific hyperparameter optimization
- Test evaluation

The planned primary experiment was a controlled frozen-parameter ablation: it would ask how changing representation affects the already-frozen XGBoost and CatBoost models. Those configurations were tuned for the original 14 features, not the high-dimensional TabPrep representation. Consequently, that experiment would not determine TabPrep's maximum achievable performance after representation-specific tuning.

If resumed, first reconstruct the isolated environment from the pinned revisions, rerun the synthetic leakage checks, and define the real-data protocol before accessing any data. A secondary Train-only representation-specific HPO experiment may be considered, but it must remain separate from the primary frozen-parameter ablation. Test must remain inaccessible until a new protocol is frozen.
