import unittest
from pathlib import Path
import tomllib

import numpy as np

from mechanism.fairsteer_author import ROOT, read, rows
from mechanism.fairsteer_improve import CONFIG, Pilot, balanced_indices, choose_actions, family, full_mmlu_prompt, make_splits, outcomes


class ImprovementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=tomllib.loads(CONFIG.read_text(encoding='utf-8'))
        cls.source=rows(ROOT/cls.c['parent_data']/'bbq.jsonl')
        cls.mapping=read(ROOT/'configs/bbq_template_families.json')['mapping']

    def test_families_and_shot_settings_cannot_cross_splits(self):
        splits,assignment,protected=make_splits(self.source,self.mapping,self.c)
        for k,indices in splits.items():
            self.assertTrue(all(assignment[self.source[i]['family']]==k for i in indices))
        for k in ('controller','dev','test'):
            self.assertFalse({self.source[i]['family'] for i in splits[k]}&set(protected))
            counts={}
            for i in splits[k]:
                r=self.source[i];key=(r['category'],r['context_condition']);counts[key]=counts.get(key,0)+1
            self.assertEqual(set(counts.values()),{self.c[f'{k}_per_condition_category']})
        repeated,_,_=make_splits(self.source,self.mapping,self.c)
        self.assertEqual(splits,repeated)

    def test_known_family_equivalence_is_preserved(self):
        a={'category':'Gender_identity','question_index':'26'}
        b={'category':'Gender_identity','question_index':'1'}
        self.assertEqual(family(a,self.mapping),family(b,self.mapping))

    def test_actions_need_positive_predicted_benefit(self):
        predictions=np.array([[0.,0.,0.],[-1.,-.1,-.2],[.1,.2,.3],[.3,.3,.3]])
        np.testing.assert_array_equal(choose_actions(predictions,[.5,1.,1.5]),[0.,0.,1.5,.5])
        np.testing.assert_array_equal(choose_actions(predictions,[.5,1.,1.5],.3),[0.,0.,0.,0.])

    def test_outcome_labels_penalize_destroying_correct_answers(self):
        before=np.array([[3.,0.,0.],[0.,3.,0.],[0.,3.,0.]])
        after=np.array([[0.,3.,0.],[3.,0.,0.],[0.,3.,0.]])
        target=outcomes(before,after,[0,0,0])
        self.assertLess(target[0],-2.)
        self.assertGreater(target[1],1.)
        self.assertEqual(target[2],0.)

    def test_evidence_gate_depends_only_on_features(self):
        dummy=object.__new__(Pilot)
        dummy.head_probabilities=lambda x:{'original':np.array([.1,.1]),'expanded':np.array([.1,.1]),'evidence':np.array([.9,.1])}
        action=dummy.actions({'gate':'evidence','threshold':.5,'alpha':1.},np.zeros((2,4)))
        np.testing.assert_array_equal(action,[1.,0.])

    def test_neutral_eval_retains_four_choices(self):
        text=full_mmlu_prompt({'question':'Question?','choices':['one','two','three','four']})
        self.assertEqual(text,'Question?\nA. one\nB. two\nC. three\nD. four\nAnswer:')

    def test_controller_stage_dispatches_to_training_method(self):
        dummy=object.__new__(Pilot);dummy.state={'stages':{}};dummy.controller=None
        called=[];dummy.train_controller=lambda:called.append('trained')
        dummy.run('controller')
        self.assertEqual(called,['trained'])

    def test_calibration_edits_only_identifiable_unknown_option(self):
        from scripts.fairsteer_improve_calibration import calibrate
        scores=np.array([[1.,2.,3.],[4.,5.,6.]])
        edited=calibrate(scores,[0,2],.5)
        np.testing.assert_array_equal(edited,[[1.5,2.,3.],[4.,5.,6.5]])
        np.testing.assert_array_equal(scores,[[1.,2.,3.],[4.,5.,6.]])


if __name__=='__main__':unittest.main()
