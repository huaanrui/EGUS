import unittest
import numpy as np
from mechanism.identity_replication import matched_groups,derived_rows,extended_metrics,site_id
from mechanism.identity_causal import variants


class ReplicationTests(unittest.TestCase):
    def base(self,g):
        return {'source_id':'s:'+g,'context':'The lawyer thanked the nurse and gave '+('him' if g=='male' else 'her')+' a card.',
                'entities':['lawyer','nurse'],'gold':'nurse','pronoun':'him' if g=='male' else 'her','gender':g,
                'pair_id':'p','family_id':'f','occupation_pair':'lawyer|nurse','stereotype':g=='male'}

    def test_matching_budget_exclusion_and_reproducibility(self):
        rms=np.linspace(.1,2,100);wn=np.linspace(2,.2,100);groups={'p':np.arange(8),'n':np.arange(8,16)}
        a=matched_groups(groups,rms,wn,1701);b=matched_groups(groups,rms,wn,1701);c=matched_groups(groups,rms,wn,1702)
        self.assertEqual(len(a['p']),8);self.assertEqual(len(a['n']),8)
        self.assertFalse(set(a['p'])&set(a['n']))
        self.assertFalse((set(a['p'])|set(a['n']))&set(range(16)))
        self.assertTrue(all(np.array_equal(a[k],b[k]) for k in a))
        self.assertFalse(all(np.array_equal(a[k],c[k]) for k in a))

    def test_irrelevant_identity_changes_only_profiles(self):
        a,b=[derived_rows(self.base(g),'explicit') for g in ('male','female')]
        self.assertEqual(a['gold'],b['gold']);self.assertEqual(a['pronoun'],b['pronoun'])
        self.assertEqual(a['context'].split('Event record: ')[1],b['context'].split('Event record: ')[1])
        self.assertNotEqual(a['context'],b['context'])
        self.assertEqual(a['choices'][a['label']],a['gold'])

    def test_required_identity_gold_changes_and_order_is_valid(self):
        a,b=[derived_rows(self.base(g),'identity_required',1) for g in ('male','female')]
        self.assertNotEqual(a['gold'],b['gold'])
        self.assertEqual(a['choices'][a['label']],a['gold']);self.assertEqual(b['choices'][b['label']],b['gold'])

    def test_neutral_control_has_identical_pair_input(self):
        a,b=[derived_rows(self.base(g),'explicit',neutral=True) for g in ('male','female')]
        self.assertEqual(a['context'],b['context']);self.assertEqual(a['question'],b['question'])

    def test_reject_subject_pronoun_to_avoid_grammar_changes(self):
        r=self.base('male');r['pronoun']='he'
        with self.assertRaises(AssertionError):derived_rows(r,'explicit')

    def test_both_wrong_is_not_both_correct_or_success(self):
        vs=[variants(self.base(g)) for g in ('male','female')]
        score=np.array([[10.,0.],[10.,0.]])
        result=extended_metrics(vs,score)
        self.assertEqual(result['both_wrong'],1);self.assertEqual(result['both_correct'],0)
        self.assertEqual(result['flip'],0);self.assertEqual(result['identity_gap']['rms'],0)

    def test_correct_to_wrong_direction_is_distinguished(self):
        vs=[variants(self.base(g)) for g in ('male','female')]
        result=extended_metrics(vs,np.array([[0.,10.],[10.,0.]]))
        self.assertEqual(result['male_correct_female_wrong'],1)
        self.assertEqual(result['male_wrong_female_correct'],0)

    def test_site_ids_include_head_and_position(self):
        self.assertEqual(site_id({'layer':21,'module':'head','head':1,'position':'decision'}),'head21_decision_h1')


if __name__=='__main__':unittest.main()
