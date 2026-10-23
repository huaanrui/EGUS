import unittest
import numpy as np
from mechanism.identity_causal import family,swap_entities,variants,render,metrics,feasible_choice,TASKS


class IdentityCausalTests(unittest.TestCase):
    def base(self,g='male'):
        return {'source_id':g,'pair_id':'p','family_id':'f','occupation_pair':'developer|designer',
                'entities':['designer','developer'],'gold':'designer','pronoun':'him' if g=='male' else 'her',
                'gender':g,'stereotype':g=='male','context':'The developer thanked the designer and helped '+('him' if g=='male' else 'her')+'.'}

    def test_family_keeps_gender_and_role_variants_together(self):
        a=self.base();b=self.base('female')
        self.assertEqual(family(a['context'],a['entities']),family(b['context'],b['entities']))
        self.assertEqual(family(a['context'],a['entities']),family(swap_entities(a['context'],a['entities']),a['entities']))

    def test_role_swap_is_involution_and_changes_gold(self):
        a=self.base();s=swap_entities(a['context'],a['entities'])
        self.assertEqual(swap_entities(s,a['entities']),a['context'])
        self.assertEqual(variants(a,'role_swap')['gold'],'developer')
        self.assertNotEqual(variants(a)['gold'],variants(a,'role_swap')['gold'])

    def test_identity_reading_and_binding_must_change_gold(self):
        for t in ('reading','binding'):
            self.assertNotEqual(variants(self.base(),t)['gold'],variants(self.base('female'),t)['gold'])
        self.assertIn('designer is a man',variants(self.base(),'binding')['context'])

    def test_permutation_changes_letter_not_semantic_gold(self):
        a=variants(self.base(),permutation=0);b=variants(self.base(),permutation=1)
        self.assertNotEqual(a['label'],b['label']);self.assertEqual(a['gold'],b['gold'])

    def test_constant_wrong_entity_does_not_improve_both_correct(self):
        vs=[variants(self.base()),variants(self.base('female'))]
        wrong=1-vs[0]['label'];m=metrics(vs,predictions=[wrong,wrong])
        self.assertEqual(m['flip'],0);self.assertEqual(m['both_correct'],0)

    def test_prompt_excludes_metadata(self):
        r=variants(self.base());before=render(r)
        r.update(gold='SECRET',label=999,family_id='SECRET',stereotype='SECRET')
        self.assertEqual(render(r),before)

    def test_selection_rejects_gain_that_breaks_identity_binding(self):
        def candidate(name,op,acc,both,binding):
            return {'name':name,'spec':{'operation':op,'strength':0 if op=='none' else 1},
                    'tasks':{t:{'accuracy':binding if t=='binding' else acc,'both_correct':both} for t in TASKS}}
        cs=[candidate('base','none',.8,.7,.9),candidate('bad','project',.9,.8,.7),candidate('good','mean',.85,.75,.9)]
        self.assertEqual(feasible_choice(cs,.02)['name'],'good')

    def test_bootstrap_clusters_repeated_option_orders(self):
        from mechanism.identity_causal import comparison
        vs=[variants(self.base(g),permutation=p) for g in ('male','female') for p in (0,1)]
        gold=np.array([r['label'] for r in vs]);m=comparison(vs,gold,1-gold,'both_correct',100,1)
        self.assertEqual(m['clusters'],1);self.assertEqual(m['delta'],1);self.assertEqual(m['ci95'],[1.,1.])

    def test_numpy_donor_patch_only_changes_aligned_positions(self):
        import torch
        from mechanism.identity_causal import Experiment
        e=Experiment.__new__(Experiment);site={'position':'identity'}
        pos=torch.tensor([[1,3],[2,3]]);h=torch.arange(32).reshape(2,4,4).float()
        donor=np.zeros((2,4),np.float32)
        hook=e.edit_hook(site,pos,'patch',1,donor=donor);edited=hook(None,None,h)
        self.assertTrue(torch.equal(edited[0,1],torch.zeros(4)))
        self.assertTrue(torch.equal(edited[1,2],torch.zeros(4)))
        self.assertTrue(torch.equal(edited[:,3],h[:,3]));self.assertTrue(torch.equal(h,torch.arange(32).reshape(2,4,4).float()))

    def test_projection_retains_orthogonal_information(self):
        import torch
        from mechanism.identity_causal import Experiment
        e=Experiment.__new__(Experiment);h=torch.tensor([[[3.,4.],[5.,6.]]]);pos=torch.tensor([[0,1]])
        edit=e.edit_hook({'position':'identity'},pos,'project',1,direction=np.array([1.,0.],np.float32))
        y=edit(None,None,h)
        self.assertEqual(y[0,0].tolist(),[0.,4.]);self.assertTrue(torch.equal(y[:,1],h[:,1]))


if __name__=='__main__':unittest.main()
