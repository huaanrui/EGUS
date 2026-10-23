import itertools
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from mechanism.fairsteer import add_direction, calibrated, detector_label, summarize
from mechanism.fairsteer_data import FairPromptBuilder, PERMUTATIONS6, ROOT, direction_rows


class ContextTokenizer:
    """Small tokenizer fixture: the answer must be a distinct appended token."""
    def encode(self, text, add_special_tokens=False):
        for i, letter in enumerate("ABC"):
            if text.endswith(" " + letter):
                return [ord(c) for c in text[:-2]] + [1000+i]
        return [ord(c) for c in text]


class FairSteerTests(unittest.TestCase):
    def row(self):
        return dict(uid="x",template_group_id="g",category="c",context_condition="ambig",
                    context="One person met another.",question="Who knows?",
                    answers=["First person","Unknown","Second person"],
                    correct_index=1,unknown_index=1,stereotype_index=2,counterstereotype_index=0)

    def test_teacher_forced_pair_preserves_question_and_permutation(self):
        row=self.row()
        builder=FairPromptBuilder(ContextTokenizer(),"Demo\n",2048)
        for i,perm in enumerate(PERMUTATIONS6):
            prefix=builder.build(row,i,stage="direction",shots=True)
            correct=builder.build(row,i,stage="direction",shots=True,answer=row["correct_index"])
            biased=builder.build(row,i,stage="direction",shots=True,answer=row["stereotype_index"])
            self.assertEqual(correct["ids"][:-1],prefix["ids"])
            self.assertEqual(biased["ids"][:-1],prefix["ids"])
            self.assertEqual(correct["ids"][-1],1000+perm.index(1))
            self.assertEqual(biased["ids"][-1],1000+perm.index(2))
            self.assertEqual(correct["extraction_position"],prefix["extraction_position"]+1)
            self.assertEqual(correct["position_kind"],"teacher_forced_answer")

    def test_gate_is_per_sample_and_addition_retains_raw_length(self):
        h=torch.tensor([[[0.,0.],[-1.,2.]],[[0.,0.],[1.,2.]]])
        vector=torch.tensor([3.,4.])
        changed,trigger=add_direction(h,vector,.5,torch.tensor([1.,0.]))
        self.assertEqual(trigger.tolist(),[True,False])
        torch.testing.assert_close(changed[0,-1]-h[0,-1],torch.tensor([1.5,2.]))
        self.assertTrue(torch.equal(changed[1],h[1]))
        self.assertTrue(torch.equal(changed[:,:-1],h[:,:-1]))
        self.assertEqual(float(torch.linalg.norm(changed[0,-1]-h[0,-1])),2.5)
        self.assertTrue(torch.equal(h[0,-1],torch.tensor([-1.,2.])))

    def test_zero_strength_is_exact_and_boundary_uses_intercept(self):
        h=torch.tensor([[[1.,2.]]])
        vector=torch.tensor([3.,4.])
        changed,trigger=add_direction(h,vector,0.,torch.tensor([1.,0.]),-2.)
        self.assertIs(changed,h)
        self.assertTrue(bool(trigger[0]))
        changed,trigger=add_direction(h,vector,1.,torch.tensor([1.,0.]),-1.)
        self.assertIs(changed,h)
        self.assertFalse(bool(trigger[0]))

    def test_detector_correct_precedence_and_other_error_exclusion(self):
        row=self.row()
        self.assertEqual(detector_label(row,1),1)
        self.assertEqual(detector_label(row,2),0)
        self.assertIsNone(detector_label(row,0))
        row["correct_index"]=2
        self.assertEqual(detector_label(row,2),1)

    def test_ambiguous_direction_selection_never_fills_with_evidence_rows(self):
        config=dict(direction_context="ambig",direction_examples=4,categories=["a","b"],seed=7)
        fit=[]
        for cat in config["categories"]:
            for i in range(4):
                fit.append(dict(self.row(),uid=f"{cat}{i}",category=cat,template_group_id=f"{cat}{i}",question_polarity="neg",context_condition="ambig" if i<2 else "disambig"))
        chosen=direction_rows(fit,config)
        self.assertEqual(len(chosen),4)
        self.assertEqual({r["context_condition"] for r in chosen},{"ambig"})
        self.assertEqual({r["uid"] for r in chosen},{"a0","a1","b0","b1"})
        with self.assertRaisesRegex(ValueError,"Not enough"):
            direction_rows(fit,dict(config,direction_examples=5))

    def test_ambiguous_direction_requires_unknown_ground_truth(self):
        row=dict(self.row(),category="a",question_polarity="neg",correct_index=0)
        with self.assertRaisesRegex(ValueError,"unknown"):
            direction_rows([row],dict(direction_context="ambig",direction_examples=1,categories=["a"],seed=1))

    def test_calibration_keeps_stereotype_counter_gap_and_updates_partition(self):
        row=self.row()
        p=[dict(uid="x",rotations=[dict(prediction=2,semantic_logprobs=np.log([.2,.1,.7]).tolist(),candidate_mass=.8)])]
        result=calibrated([row],p,2.)[0]["rotations"][0]
        self.assertAlmostEqual(result["semantic_logprobs"][2]-result["semantic_logprobs"][0],np.log(.7/.2))
        self.assertAlmostEqual(sum(np.exp(result["semantic_logprobs"])),1.)
        factor=1+.1*np.expm1(2.)
        self.assertAlmostEqual(result["candidate_mass"],.8*factor/(.2+.8*factor))
        self.assertEqual(p[0]["rotations"][0]["prediction"],2)

    def test_signed_bias_can_hide_both_wrong_answer_types(self):
        row=self.row()
        rotations=[dict(prediction=x,semantic_logprobs=[-1.,-1.,-1.],candidate_mass=1.) for x in (0,2,0,2,1,1)]
        m=summarize([row],[dict(uid="x",rotations=rotations)])
        self.assertAlmostEqual(m["signed_ambiguous_bbq_bias"],0.)
        self.assertAlmostEqual(m["macro_stereotype"],1/3)
        self.assertAlmostEqual(m["macro_counterstereotype"],1/3)
        self.assertAlmostEqual(m["ambiguous_accuracy"],1/3)

    def test_prepared_data_has_no_family_or_question_leakage(self):
        directory=ROOT/"data/prepared/bbq_fairsteer_20261012_v1"
        if not directory.exists():
            self.skipTest("Prepared local pilot not present")
        sets={}
        for split in ("fit","probe_validation","selection","test"):
            rows=[json.loads(line) for line in (directory/f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
            sets[split]={r["template_group_id"] for r in rows}
        for a,b in itertools.combinations(sets,2):
            self.assertFalse(sets[a]&sets[b])
        test=[json.loads(s) for s in (directory/"test.jsonl").read_text(encoding="utf-8").splitlines()]
        fit=[json.loads(s) for s in (directory/"fit.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue({r["category"] for r in test}<={r["category"] for r in fit})
        self.assertEqual({r["category"] for r in test},{"Nationality","Race_ethnicity","Religion"})
        mmlu=[]
        for split in ("fit","probe_validation","selection","test"):
            rows=[json.loads(s) for s in (directory/f"mmlu_{split}.jsonl").read_text(encoding="utf-8").splitlines()]
            mmlu.append({r["template_group_id"] for r in rows})
        for a,b in itertools.combinations(mmlu,2):
            self.assertFalse(a&b)


if __name__=="__main__":
    unittest.main()
