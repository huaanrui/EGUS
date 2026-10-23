import unittest
from mechanism.identity_answer_parser import parse_answer_line


class AnswerLineTests(unittest.TestCase):
    def test_valid_first_answer_followed_by_question_copy(self):
        self.assertEqual(parse_answer_line(' accountant\nThe carpenter made a lion for the accountant',['accountant','carpenter']),0)

    def test_anchored_pronoun_explanation(self):
        self.assertEqual(parse_answer_line("\nThe pronoun 'her' refers to the clerk.\n",['clerk','supervisor']),0)

    def test_does_not_extract_mentions_from_refusal_or_negation(self):
        for text in ['It is not the clerk.','I cannot decide between clerk and supervisor.','The clerk thanked the supervisor.']:
            self.assertEqual(parse_answer_line(text,['clerk','supervisor']),-1)

    def test_does_not_search_later_lines_for_a_candidate(self):
        self.assertEqual(parse_answer_line('I do not know.\nclerk',['clerk','supervisor']),-1)

    def test_article_case_and_multiword_occupation(self):
        self.assertEqual(parse_answer_line('The Construction Worker.',['nurse','construction worker']),1)


if __name__=='__main__':unittest.main()
