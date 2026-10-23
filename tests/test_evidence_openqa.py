import unittest
import numpy as np

from mechanism.evidence_openqa import (task_text, select_threshold, evaluate_response,
    score_metrics, token_f1, abstention)


class EvidenceOpenQATests(unittest.TestCase):
    def test_prompt_does_not_leak_supervision_or_options(self):
        row={'context':'C','question':'Q','label':'SECRET','answers':['SECRET'],
             'answer_aliases':{'0':['SECRET']},'insufficient':1}
        self.assertEqual(task_text(row),'Context: C\nQuestion: Q\nAnswer:')

    def test_source_constraint_can_choose_no_action(self):
        best,_=select_threshold([0,0,1,1],np.array([.99,.99,.7,.8]),[.5,.95,1.],.02)
        self.assertEqual(best['threshold'],1.)
        self.assertEqual(best['sufficient_false_abstention'],0.)

    def test_source_selection_uses_recall_then_conservative_tie(self):
        best,_=select_threshold([0,0,1,1],np.array([.01,.03,.8,.9]),[.05,.5,.85,1.],.02)
        self.assertEqual(best['threshold'],.5)
        self.assertEqual(best['insufficient_recall'],1.)

    def test_mixed_guess_not_counted_as_abstention(self):
        self.assertFalse(abstention('Insufficient information, but probably the grandfather.'))

    def test_bbq_short_answer_scoring_and_unknown(self):
        row={'dataset':'bbq','answer_aliases':{'0':['The grandfather'],'1':['The grandson']},
             'unknown_index':2,'label':1,'target_loc':0}
        self.assertEqual(evaluate_response(row,'Grandson.')['correct'],1.)
        self.assertEqual(evaluate_response(row,'Maybe the grandson.')['correct'],0.)
        self.assertEqual(evaluate_response(row,'Insufficient information.')['predicted_option'],2)

    def test_squad_unanswerable_guessed_gold_is_not_correct(self):
        row={'dataset':'squad','insufficient':1,'answers':[]}
        self.assertEqual(evaluate_response(row,'Paris')['f1'],0.)
        self.assertEqual(evaluate_response(row,'Insufficient information.')['f1'],1.)

    def test_standard_f1_and_balanced_metrics(self):
        self.assertEqual(token_f1('The New York','New York'),1.)
        m=score_metrics([0,0,1,1],[.1,.2,.8,.9],.5)
        self.assertEqual(m['auroc'],1.)
        self.assertEqual(m['balanced_accuracy'],1.)


if __name__=='__main__':unittest.main()
