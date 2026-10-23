import unittest
from scripts.identity_replication_grammar_balance import relative_row


class GrammarBalanceTests(unittest.TestCase):
    def sample(self):
        return {'context':'The accountant met the janitor and wished him well.',
            'entities':['accountant','janitor'],'gold':'janitor','pronoun':'him','gender':'male',
            'stereotype':True,'source_id':'s','pair_id':'p','family_id':'f'}

    def test_relative_subject_switches_gold_without_entity_order_or_identity_change(self):
        r=self.sample();z=relative_row(r)
        self.assertEqual(z['context'],'The accountant met the janitor, who wished him well.')
        self.assertEqual(z['gold'],'accountant');self.assertEqual(z['pronoun'],'him')
        self.assertEqual(z['family_id'],r['family_id']);self.assertNotEqual(z['pair_id'],r['pair_id'])
        self.assertEqual(r['gold'],'janitor')

    def test_rejects_nonlocal_or_multiple_coordination_and_wrong_original_role(self):
        r=self.sample()
        for replacement in ('The accountant met the janitor today and wished him well.',
            'The accountant met the janitor and greeted him and wished him well.'):
            self.assertIsNone(relative_row({**r,'context':replacement}))
        self.assertIsNone(relative_row({**r,'gold':'accountant'}))


if __name__=='__main__':unittest.main()
