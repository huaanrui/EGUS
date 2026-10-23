import json,unittest
import numpy as np
from scripts.identity_replication_norm_controls import calibrate


class ToyEnergy:
    def infer(self,key,vs,spec):
        amplitude=sum(e['amount']**2 for e in spec['edits'])
        return {'edit_energy':np.array([amplitude]*len(vs))}


class NormTests(unittest.TestCase):
    def test_calibration_matches_scalar_energy_without_target_answers(self):
        spec={'name':'toy','edits':[{'amount':.5}]}
        result=calibrate(ToyEnergy(),'toy',[{},{}],spec,4.)
        self.assertAlmostEqual(result['energy'],4)
        self.assertEqual(spec['edits'][0]['amount'],.5)
        json.dumps(result)

    def test_zero_target_produces_zero_edit(self):
        result=calibrate(ToyEnergy(),'zero',[{}],{'edits':[{'amount':.5}]},0.)
        self.assertEqual(result['energy'],0)
        self.assertEqual(result['spec']['edits'][0]['amount'],0)


if __name__=='__main__':unittest.main()
