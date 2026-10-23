# Supplement: keep the question-pronoun receiver active on utility controls

Declared during the rank scan, before head selection or control-panel outputs.
The native literal/gender/binding/record questions often have no gendered pronoun,
so question-pronoun receiver interventions are inactive there. A high native
control score alone cannot prove those particular receiver edits preserve utility.

Use the same fixed64historical pairs, both choice orders, same frozen selected
receivers. For each utility question prepend: "Considering the pronoun 'him/her'
in the sentence, ..." (the public observed pronoun, never the gold entity).
The question-pronoun receiver now exists in every item; no remapping or selection
uses target labels. This changes the prompt and is a supplementary diagnostic.
Literal and gender tasks become easy with repeated public pronoun information;
explicit binding and record reading still require reading the relevant facts.

Compare baseline, sourcehalf at contextual pronoun, selected receiver-half, and
sourcehalf+receiver restoration on literal, gender classification, explicit gender
binding, and record reading. In the record task also swap only the record-source
MLP output, then restore/transplant the same frozen receivers to compare evidence
transmission while holding receiver presence constant. Report two-way fidelity,
actual control accuracy, receiver-presence audit and family bootstrap.
All comparisons exploratory, historical, no reselection. Supplement script and
protocol hashes are saved before this panel runs; main frozen assets unchanged.

