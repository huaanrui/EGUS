# Exploratory frozen-receiver graph after Q/K/V results

This supplement is explicitly proposed AFTER inspecting the main Q/K/V panel:
question readers showed source-V sensitivity, final readers showed Q/K sensitivity,
and the predeclared question13->decision21 K/V relay was near zero. It is not a
preregistered confirmation and does not change any main result/receiver selection.

Test ALL temporally ordered pairs among the four already frozen heads. At most
six pairs. For each, project that sender head's B-minus-A change through its own
W_O columns to obtain its residual contribution at its actual receiver token.
Add only that contribution to the later head's BASELINE residual at the sender
token, apply actual receiver RMSNorm/projections/K-norm/RoPE, and transplant only
the later query head's K/V for that token. If sender and receiver occupy the same
token, additionally transplant its Q (baseline K/V). Other contributions fixed.
This tests controlled additive edges, not necessity or completeness of a circuit.
All pairs reported, no best-pair deployment or target selection, source contrast
still layer7half, same64historical pairs with both choice orders.

