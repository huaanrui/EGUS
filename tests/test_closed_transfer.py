import unittest
import numpy as np

from mechanism.closed_transfer import (UNKNOWN,ROOT,with_choices,task_prompt,apply_policy,
    metric,bootstrap_difference,calibrate,canonical,template,winobias_data,winogender_data)


class ClosedTransferTests(unittest.TestCase):
    def row(self,condition='clear'):
        return {'source_id':'x','pair_id':'p','family_id':'f','context':'A nurse met a doctor.',
                'question':'Who arrived?','condition':condition,'stereotype':'true'}

    def test_rotated_options_preserve_gold_and_unknown(self):
        variants=[with_choices(self.row(),['nurse','doctor'],'doctor',11,True,p) for p in range(3)]
        self.assertEqual({v['choices'][v['label']] for v in variants},{'doctor'})
        self.assertEqual({v['unknown_index'] for v in variants},{0,1,2})
        self.assertEqual({v['label'] for v in variants},{0,1,2})

    def test_supervision_is_not_in_prompt(self):
        r=with_choices(self.row(),['nurse','doctor'],'doctor',11,True)
        r.update(label=999,stereotype='SECRET',task_type='SECRET',condition='SECRET',family_id='SECRET')
        self.assertNotIn('SECRET',task_prompt(r));self.assertNotIn('999',task_prompt(r))

    def test_same_trigger_direct_follows_current_unknown_position(self):
        vs=[with_choices(self.row(),['nurse','doctor'],'doctor',11,True,p) for p in range(3)]
        base=np.zeros((3,3));sig={'g':np.array([.8,.9,1.])}
        p,t,_=apply_policy(base,base,base,sig,vs,{'action':'direct','gate':'g','threshold':.5})
        self.assertEqual(p.tolist(),[v['unknown_index'] for v in vs]);self.assertTrue(t.all())

    def test_direct_is_inapplicable_without_unknown(self):
        vs=[with_choices(self.row(),['nurse','doctor'],'doctor',11)]
        base=np.array([[0.,1.,-1e9]])
        p,t,_=apply_policy(base,base,base,{'g':np.array([1.])},vs,{'action':'direct','gate':'g','threshold':.1})
        self.assertEqual(p.tolist(),[1]);self.assertFalse(t.any())

    def test_abstaining_everywhere_is_not_perfect(self):
        vs=[with_choices(self.row('ambig'),['nurse','doctor'],UNKNOWN,11,True),
            with_choices(self.row('clear'),['nurse','doctor'],'doctor',11,True)]
        p=np.array([v['unknown_index'] for v in vs]);m=metric(vs,p)
        self.assertEqual(m['balanced_accuracy'],.5);self.assertEqual(m['clear_false_unknown'],1.)

    def test_pair_comparison_is_semantic_not_letter(self):
        vs=[with_choices(self.row(),['nurse','doctor'],'doctor',11,True,p) for p in range(2)]
        m=metric(vs,np.array([v['label'] for v in vs]))
        self.assertEqual(m['pair_answer_flip'],0.);self.assertEqual(m['pair_all_correct'],1.)

    def test_calibration_can_choose_noop(self):
        vs=[with_choices({**self.row(cond),'pair_id':str(i)},['nurse','doctor'],UNKNOWN if cond=='ambig' else 'doctor',11,True)
            for i,cond in enumerate(['ambig','ambig','clear','clear'])]
        base=np.zeros((4,3))
        for i,v in enumerate(vs):base[i,v['label']]=1.
        sig={'g':np.array([.2,.3,.9,.95])};cache={'base':base,'cf1':-base,'cf15':-base}
        best,_=calibrate(vs,cache,sig,{'action':'direct','gate':'g','threshold':.1},
                         {'max_calibration_fpr':.02,'calibration_accuracy_tolerance':.01})
        self.assertEqual(best['metrics']['trigger_rate'],0.)

    def test_cluster_bootstrap_binds_repeated_family(self):
        vs=[{**self.row(),'label':0,'family_id':str(i//2)} for i in range(8)]
        r=bootstrap_difference(vs,np.zeros(8,int),np.ones(8,int),100,3)
        self.assertEqual(r['clusters'],4);self.assertEqual(r['ci95'],[1.,1.])

    def test_template_binds_gender_and_occupation_variants(self):
        a=template('The nurse thanked the doctor because she was kind.',['nurse','doctor'])
        b=template('The writer thanked the analyst because he was kind.',['writer','analyst'])
        self.assertEqual(a,b)
        self.assertEqual(canonical('He paid his bill.'),canonical('She paid her bill.'))

    def test_native_winobias_multitoken_occupations_and_multiple_pronouns(self):
        folder=ROOT/'data/raw/closed_transfer/winobias/WinoBias/wino/data'
        if not folder.exists():self.skipTest('Native data not downloaded')
        values=winobias_data(folder,{'seed':11})
        self.assertEqual(len(values),3168)
        self.assertTrue(any('construction worker' in v['choices'] for v in values))
        self.assertTrue(all(v['unknown_index']==-1 for v in values))
        self.assertTrue(all('[' not in v['context'] for v in values))
        self.assertEqual(len({v['source_id'] for v in values}),1584)

    def test_native_winogender_gold_including_someone(self):
        folder=ROOT/'data/raw/closed_transfer/winogender/data'
        if not folder.exists():self.skipTest('Native data not downloaded')
        values=winogender_data(folder,{'seed':11})
        self.assertEqual(len(values),1440)
        self.assertEqual(len({v['source_id'] for v in values}),720)
        for v in values:
            occ,part,gold,_,_=v['source_id'].removeprefix('winogender:').split('.')
            self.assertEqual(v['choices'][v['label']],occ if gold=='0' else part)

    def test_margin_calibration_uses_true_noop_and_only_actual_boundaries(self):
        from scripts.closed_transfer_stable_calibration_v2 import calibrate as stable_calibrate
        vs=[with_choices({**self.row(cond),'pair_id':str(i)},['nurse','doctor'],UNKNOWN if cond=='ambig' else 'doctor',11,True)
            for i,cond in enumerate(['ambig','ambig','clear','clear'])]
        base=np.zeros((4,3))
        for i,v in enumerate(vs):base[i,v['label']]=1.
        sig={'g':np.array([2.,3.,4.,5.])};cache={'base':base,'cf1':-base,'cf15':-base}
        best,candidates=stable_calibrate(vs,cache,sig,{'action':'direct','gate':'g','threshold':.1},
                                         {'max_calibration_fpr':.02,'calibration_accuracy_tolerance':.01})
        self.assertEqual({x['spec']['threshold'] for x in candidates},{2.,3.,4.,5.,np.finfo(np.float64).max})
        self.assertEqual(best['spec']['threshold'],np.finfo(np.float64).max)
        self.assertFalse(6.>best['spec']['threshold'])


if __name__=='__main__':unittest.main()
