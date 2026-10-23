import unittest
from types import SimpleNamespace
from scripts.resume_identity_replication_execution_v2 import trained_sites


class ExecutionTests(unittest.TestCase):
    def test_extra_baseline_site_is_not_a_fitted_diagnostic_component(self):
        fitted={'layer':9,'module':'mlp','position':'identity'}
        extra={'layer':7,'module':'mlp','position':'identity'}
        e=SimpleNamespace(meta={'groups':{'mlp9_identity':{}}},sites={'mlp9_identity':fitted,'mlp7_identity':extra},
            params={'mlp9_identity_transfer_pos':[],'mlp9_identity_transfer_neg':[]})
        self.assertEqual(trained_sites(e),[fitted])

    def test_missing_fitted_parameters_fail_before_any_model_inference(self):
        e=SimpleNamespace(meta={'groups':{'mlp9_identity':{}}},sites={},params={})
        with self.assertRaises(AssertionError):trained_sites(e)


if __name__=='__main__':unittest.main()
