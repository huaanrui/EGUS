import unittest
import numpy as np
import torch
from mechanism.identity_combo import make_candidates,edit_output,edit_channels,SCOPES,evaluate_feasibility,ALL_TASKS,task_row


class CombinationTests(unittest.TestCase):
    def positions(self):return torch.tensor([[1,3,0,2,-1],[2,3,0,1,-1]])

    def test_bounded_unique_candidates_and_combination_size(self):
        cs=make_candidates();self.assertEqual(len(cs),46);self.assertEqual(len({c['name'] for c in cs}),46)
        self.assertTrue(all(2<=len(c['edits'])<=3 for c in cs if c['category']=='combo'))

    def test_distinct_position_edits_compose_without_overwrite(self):
        x=torch.ones(2,4,3);p=self.positions()
        y=edit_output(x,p,{'position':'identity','operation':'scale','strength':.5})
        z=edit_output(y,p,{'position':'decision','operation':'scale','strength':.25})
        self.assertEqual(z[0,1,0].item(),.5);self.assertEqual(z[0,3,0].item(),.75)
        self.assertEqual(z[0,0,0].item(),1);self.assertTrue(torch.equal(x,torch.ones_like(x)))

    def test_missing_question_pronoun_is_noop(self):
        x=torch.ones(2,4,3);y=edit_output(x,self.positions(),{'position':'question_identity','operation':'scale','strength':1})
        self.assertTrue(torch.equal(x,y))

    def test_subspace_and_complement_reconstruct_whole_scale(self):
        x=torch.arange(24).reshape(2,4,3).float();p=self.positions();basis=np.array([[1.],[0.],[0.]],np.float32)
        a=edit_output(x,p,{'position':'identity','operation':'subspace','strength':1},basis)
        b=edit_output(x,p,{'position':'identity','operation':'complement','strength':1},basis)
        idx=torch.arange(2);ps=p[:,0]
        self.assertTrue(torch.allclose(a[idx,ps]+b[idx,ps],x[idx,ps]))

    def test_neuron_edits_only_change_chosen_units_and_tokens(self):
        x=torch.ones(2,4,6);y=edit_channels(x,self.positions(),{'position':'identity','strength':.5},[1,4])
        self.assertEqual(y[0,1].tolist(),[1,.5,1,1,.5,1])
        self.assertTrue(torch.equal(y[:,3],x[:,3]));self.assertTrue(torch.equal(x,torch.ones_like(x)))

    def test_full_channel_half_scale_matches_output_linear(self):
        x=torch.arange(48).reshape(2,4,6).float();w=torch.randn(3,6)
        a=edit_channels(x,self.positions(),{'position':'identity','strength':.5},np.arange(6))@w.T
        b=edit_output(x@w.T,self.positions(),{'position':'identity','operation':'scale','strength':.5})
        self.assertTrue(torch.allclose(a,b))

    def test_feasibility_checks_identity_controls(self):
        b={'tasks':{t:{'accuracy':.9} for t in ALL_TASKS}};a={'tasks':{t:{'accuracy':.95} for t in ALL_TASKS}}
        a['tasks']['literal']['accuracy']=.85
        self.assertFalse(evaluate_feasibility(a,b,ALL_TASKS,.02))

    def test_literal_changes_gold_with_identity_and_not_permutation(self):
        base={'source_id':'a','context':'The developer thanked the designer and helped him.',
              'entities':['designer','developer'],'gold':'designer','pronoun':'him','gender':'male','pair_id':'p'}
        a=task_row(base,'literal',0);b=task_row(base,'literal',1)
        self.assertNotEqual(a['label'],b['label']);self.assertEqual(a['gold'],b['gold'])
        female={**base,'pronoun':'her','gender':'female','context':base['context'].replace('him','her')}
        self.assertNotEqual(task_row(female,'literal')['gold'],a['gold'])

    def test_anonymization_preserves_identity_and_referent_binding(self):
        from scripts.identity_combo_mechanism_panel import anonymize
        base={'source_id':'a','pair_id':'p','context':'The carpenter thanked the nurse and helped her.',
              'entities':['carpenter','nurse'],'gold':'nurse','pronoun':'her','gender':'female'}
        a=anonymize(base)
        self.assertNotIn('carpenter',a['context']);self.assertNotIn('nurse',a['context'])
        self.assertTrue(a['context'].endswith('helped her.'))
        self.assertIn('thanked the '+a['gold'],a['context'])
        male=anonymize({**base,'pronoun':'him','gender':'male','context':base['context'].replace('her.','him.')})
        self.assertEqual(a['gold'],male['gold']);self.assertEqual(a['entities'],male['entities'])


if __name__=='__main__':unittest.main()
