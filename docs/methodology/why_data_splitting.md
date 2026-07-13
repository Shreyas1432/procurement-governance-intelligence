# Why Data Splitting

Procurement data is temporal. A random split can make a model look stronger than it is by allowing future procurement patterns to influence training.

RQ2 prefers temporal holdout windows so the classifier is tested on later observations. When configured windows are unavailable in a constrained local sample, the code falls back to an adaptive temporal split before using stratified random splitting as a last resort.

RQ3 currently uses a deterministic random holdout for price regression. The next implementation phase should evaluate temporal price holdouts where CPV coverage remains sufficient.
