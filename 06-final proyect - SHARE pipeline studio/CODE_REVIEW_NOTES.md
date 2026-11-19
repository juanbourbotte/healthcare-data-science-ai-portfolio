# Code review notes

The supplied Colab-exported script contains EDA, clustering, several supervised
models, SHAP and DCA. The refactored starter in `src/main.py` focuses on a
single reproducible supervised baseline rather than silently preserving
multiple competing workflows.

Key issues to resolve in the original analysis before a scientific release:

- `Y_deterioro` is described as derived from the lower quartile of `SGC`.
  The exact threshold and target construction must be documented and verified.
- Some earlier preprocessing blocks impute or encode before train/test split.
  Those operations can leak information across validation boundaries.
- The Random Forest block selects numeric columns only, whereas later model
  blocks use separate feature definitions and preprocessing, so comparisons
  are not necessarily like-for-like.
- Threshold selection is performed using `y_test`/`test_probs` in one block.
  This makes the test set part of model selection. The refactored workflow
  selects the threshold using out-of-fold training predictions.
- SHAP is calculated on transformed/scaled inputs, while later dependence
  plots appear to combine SHAP values with untransformed `X_test`. Feature
  spaces and column ordering must match exactly.
- DCA is only meaningful when probabilities are sufficiently calibrated and
  the decision threshold range has a defensible interpretation. The current
  plot should not be described as clinical utility without this justification.
- Cross-sectional random splitting is not adequate for future-risk prediction,
  repeated observations, or external generalization.
- The custom KMeans implementation is unnecessary for a production analysis;
  use the tested scikit-learn implementation if clustering remains in scope.

This starter has not been run against the user's dataset because the SHARE
analysis CSV was not included. Runtime validation and scientific validation
remain necessary.
