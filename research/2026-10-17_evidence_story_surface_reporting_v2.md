# Surface-control reporting repair v2

The final independent audit detected that report rendering refitted and rewrote
surface_controls/*.joblib, invalidating the older auxiliary surface-route source
hash. The main frozen module/config/data, both models' probes/directions/dev
selections, and every primary prediction remained unchanged and passed the prior
independent audit. The old surface-route plan, selections and predictions remain.

Repair report-only code to load existing surface models. A separately versioned
surface_routes_v2 pins the currently saved matched2200 source-fit classifiers and
reruns the unchanged source-dev2%FPR selector and direct routing. No model training
budget, classifier objective, selector, main method or primary statistic changes.
Compare old/new surface predictions and probability differences explicitly. The
previous surface-route fingerprint failure remains in the original final audit
log; it is not removed or relabeled as success. Both auxiliary versions remain
exploratory post-hoc diagnostics, not new confirmatory experiments.
