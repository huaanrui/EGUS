import unittest
from scripts.audit_identity_replication_records_v3 import restore_metadata


class AuditTests(unittest.TestCase):
    def test_reconstruct_original_record_and_anonymous_metadata(self):
        for pid in ('p','p:record0','p:record1','p:anonymous'):
            original={'pair_id':pid,'prediction':1,'scores':[0.,1.]}
            restored=restore_metadata(original,{'p':'chef|guard'})
            self.assertEqual(restored['occupation_pair'],'chef|guard');self.assertNotIn('occupation_pair',original)
            self.assertEqual(restored['scores'],original['scores'])

    def test_unknown_pair_and_conflicting_saved_metadata_are_not_silently_accepted(self):
        with self.assertRaises(AssertionError):restore_metadata({'pair_id':'missing'},{'p':'a|b'})
        with self.assertRaises(AssertionError):restore_metadata({'pair_id':'p','occupation_pair':'wrong'},{'p':'a|b'})


if __name__=='__main__':unittest.main()
