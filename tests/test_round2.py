import unittest
import numpy as np
import torch

from mechanism.round2 import Operator, eligibility, extended_metrics
from scripts.inspect_native_pairs import canonical
from scripts.prepare_round2 import aligned
from scripts.calibration_round2 import calibrated


class Round2Tests(unittest.TestCase):
    def test_calibration_only_changes_unknown_relative_preference(self):
        rows = [dict(uid='x', unknown_index=2)]
        pred = [dict(uid='x', rotations=[dict(prediction=0, semantic_logprobs=list(np.log([.6, .3, .1])), candidate_mass=.9) for _ in range(3)])]
        self.assertEqual(calibrated(rows, pred, 0.), pred)
        c = calibrated(rows, pred, 3.)
        for p in c[0]['rotations']:
            self.assertEqual(p['prediction'], 2)
            self.assertAlmostEqual(p['semantic_logprobs'][0]-p['semantic_logprobs'][1], np.log(2.))
            self.assertAlmostEqual(p['semantic_logprobs'][2]-p['semantic_logprobs'][0], np.log(.1/.6)+3.)
            self.assertAlmostEqual(sum(np.exp(p['semantic_logprobs'])), 1.)
            self.assertTrue(.9 < p['candidate_mass'] < 1.)

    def test_native_alignment_preserves_ground_truth_and_flips_identity_role(self):
        a = dict(uid='x', template_group_id='g', question_polarity='neg', context='A woman and a man arrived.', question='Who led?', answers=['The man', 'Unknown', 'The woman'], unknown_index=1, stereotype_index=2, counterstereotype_index=0, correct_index=1, target_person_index=2, source_answer_info={'ans0': ['man', 'man'], 'ans1': ['Unknown', 'unknown'], 'ans2': ['woman', 'woman']})
        b = dict(a, uid='y', context='A man and a woman arrived.')
        ca, ma = canonical(a)
        cb, mb = canonical(b)
        self.assertEqual(ca, cb)
        aa, bb = aligned(a, 'p', 0, ma), aligned(b, 'p', 1, mb)
        self.assertEqual(aa['stereotype_index'], 0)
        self.assertEqual(bb['stereotype_index'], 1)
        self.assertEqual(aa['correct_index'], bb['correct_index'])
        self.assertEqual(aa['source_answer_info']['ans0'][0], 'woman')

    def test_pair_key_rejects_other_lexical_changes(self):
        a = dict(template_group_id='g', question_polarity='neg', context='A woman and a man arrived.', question='Who led?', answers=['man', 'Unknown', 'woman'], unknown_index=1, source_answer_info={'ans0': ['man', 'man'], 'ans1': ['Unknown', 'unknown'], 'ans2': ['woman', 'woman']})
        b = dict(a, context='A man and a woman left.')
        self.assertNotEqual(canonical(a)[0], canonical(b)[0])

    def test_donor_reference_is_rotation_specific(self):
        spec = dict(method='layer_update', layers=[0], scope='last1', operation='patch', strength=1.)
        donor = {'residual': {0: np.array([[[1., 2.]], [[3., 4.]], [[5., 6.]]])}}
        op = Operator(spec, {}, 'cpu', donor)
        h = torch.tensor([[[10., 20.]]])
        for perm, expected in zip(((0, 1, 2), (1, 2, 0), (2, 0, 1)), ([1., 2.], [3., 4.], [5., 6.])):
            p = dict(masks={'last1': [0]}, permutation=perm)
            changed = op.apply('residual', 0, h, p)
            torch.testing.assert_close(changed[0, 0], torch.tensor(expected))

    def test_neuron_fraction_and_retain_are_complements(self):
        fit = dict(rankings={'paired_relp:last1': {0: np.array([1, 0, 2])}}, suffix_means={'mlp': {0: np.zeros((8, 3))}})
        spec = dict(method='paired_relp', layers=[0], pool='last1', scope='last1', operation='remove', strength=.5, k=1)
        p = dict(masks={'last1': [0]})
        h = torch.tensor([[[2., 4., 6.]]])
        removed = Operator(spec, fit, 'cpu').apply('mlp', 0, h, p)
        retained = Operator(dict(spec, operation='retain'), fit, 'cpu').apply('mlp', 0, h, p)
        torch.testing.assert_close(removed[0, 0], torch.tensor([2., 2., 6.]))
        torch.testing.assert_close(removed + retained, 1.5*h)

    def test_all_scope_reference_tracks_each_prompt_length(self):
        fit = dict(role_means={'update': {0: np.array([[1., 2.], [3., 4.]])}})
        s = dict(method='layer_update', layers=[0], scope='all', operation='remove', strength=1.)
        op = Operator(s, fit, 'cpu')
        for n in (2, 4):
            h = torch.zeros((1, n, 2))
            roles = np.array([i%2 for i in range(n)])
            p = dict(masks={'all': list(range(n))}, roles=roles)
            result = op.apply('residual', 0, h, p, update=h)
            torch.testing.assert_close(result[0], torch.tensor(fit['role_means']['update'][0][roles], dtype=torch.float32))

    def test_macro_stereotype_equalizes_categories(self):
        rows = [dict(uid=str(i), template_group_id=str(i), category='a' if i < 2 else 'b', context_condition='ambig', correct_index=2, unknown_index=2, stereotype_index=0, counterstereotype_index=1) for i in range(3)]
        pred = [dict(uid=str(i), rotations=[dict(prediction=0 if i < 2 else 2, semantic_logprobs=[-1., -2., -3.], candidate_mass=1.)]*3) for i in range(3)]
        m = extended_metrics(rows, pred)
        self.assertAlmostEqual(m['stereotype_rate'], 2/3)
        self.assertEqual(m['macro_stereotype'], .5)

    def test_selection_rejects_counterstereotype_substitution(self):
        base = dict(disambiguated_accuracy=.8, categories={'c': dict(disambiguated_accuracy=.8)}, counterstereotype_rate=.1, macro_counterstereotype=.1, macro_ambiguous_accuracy=.7)
        changed = dict(base, counterstereotype_rate=.12, macro_counterstereotype=.12)
        neutral = dict(accuracy=.8, disambiguated_overabstention=.1)
        c = dict(constraints=dict(disambiguated_accuracy_drop=.02, category_disambiguated_accuracy_drop=.05, neutral_accuracy_drop=.02, neutral_overabstention_increase=.02, counterstereotype_increase=.005, ambiguous_accuracy_drop=0.))
        ok, reasons = eligibility(changed, neutral, base, neutral, c)
        self.assertFalse(ok)
        self.assertIn('macro_counterstereotype', reasons)


if __name__ == '__main__':
    unittest.main()
