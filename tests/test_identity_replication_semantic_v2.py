import unittest
from scripts.identity_replication_semantic_controls_v2 import make_row
from tests import test_identity_replication as fixtures


class SemanticTests(unittest.TestCase):
    def test_neutral_swaps_and_avoids_color_overlap_in_profiles(self):
        base=fixtures.ReplicationTests()
        a,b=[make_row(base.base(g),'neutral_explicit_direct',0) for g in ('male','female')]
        self.assertNotEqual(a['context'],b['context']);self.assertEqual(a['gold'],b['gold'])
        self.assertEqual(a['context'].split('Event record: ')[1],b['context'].split('Event record: ')[1])
        for r in (a,b):
            prefix=r['context'].split('Event record: ')[0]
            self.assertNotIn('blue',prefix);self.assertNotIn('green',prefix)
            self.assertNotIn('Considering',r['question'])

    def test_required_gender_and_code_answers_agree_but_swap(self):
        base=fixtures.ReplicationTests();answers=[]
        for g in ('male','female'):
            a=make_row(base.base(g),'gender_required_direct',1);b=make_row(base.base(g),'neutral_required_direct',1)
            self.assertEqual(a['gold'],b['gold']);self.assertEqual(b['choices'][b['label']],b['gold']);answers.append(b['gold'])
        self.assertNotEqual(*answers)


if __name__=='__main__':unittest.main()
