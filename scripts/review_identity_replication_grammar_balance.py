"""Persist agent text review before any derived-probe inference."""
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mechanism.fairsteer_author import read,write,sha
from scripts.identity_replication_grammar_balance import OUT,SOURCE

if __name__=='__main__':
    assert not (OUT/'declaration.json').exists()
    values=read(OUT/'candidates.json')
    excluded={
        'wbdev:6af49c8790be01100b88':'mixed present/past likes ... demanded in native text',
        'wbdev:79d3e478d70528c7306f':'native sold ... ask tense disagreement; unchanged typo would carry into relative clause',
    }
    write(OUT/'text_review.json',{'created':time.strftime('%Y-%m-%d %H:%M:%S'),
        'reviewer':'agent textual review, not independent human annotation',
        'source_sha256':sha(SOURCE),'candidate_sha256':sha(OUT/'candidates.json'),
        'rule':'two native occupations once each; unique object pronoun; nearest antecedent at end of first clause; who is relative-clause subject, nonreflexive him/her refers to first occupation; exclude native tense errors, retain unusual occupational events without score filtering',
        'excluded':excluded,'accepted_pair_ids':[r['pair_id'] for r in values if r['pair_id'] not in excluded]})
    print('Agent text review saved before inference; accepted',len(values)-len(excluded))
