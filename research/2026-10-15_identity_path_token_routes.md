# Supplement: fixed token-group K/V transmission at answer receivers

Declared after receiver identities were frozen, before Q/K/V decomposition
outcomes. Never changes receiver selection.
The contextual source can reach a final head through other token positions.
Source-only K/V patches cannot identify those routes by themselves.

For the frozen decision-position heads only (at most four), use the same64fixed
historical pairs, both orders, and same-input A/B layer7half contrast. Partition
all unmasked prompt tokens into contextual prefix, contextual source pronoun,
context suffix, question pronoun, other question, option A, option B, answer
marker and other formatting by public text offsets. No gold-dependent spans.
With baseline queries and baseline values, replace baseline K with B K only in
one fixed group; independently, with baseline queries/keys, replace V only in
that group. This identifies where a receiver reads changed keys or content.
All groups are reported; no posthoc best-group deployment or target tuning.
Prefix tokens preceding the source are a numerical negative control. Fixed-Q/K
V contributions are additive at the head output, but downstream answer effects
need not add. Report individual score effects, family uncertainty and norm-free
alignment with the source-half score contrast, never a percentage of bias.

