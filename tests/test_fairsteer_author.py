import ast
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from mechanism.fairsteer_author import AUTHOR, RAW, META, ROOT, prompt, metrics, label_for, rows
from mechanism.fairsteer_data import source_fewshot


def source_functions(filename,names):
    tree=ast.parse((AUTHOR/filename).read_text(encoding='utf-8'))
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    env={'pd':pd,'np':np}
    exec(compile(ast.Module(body=selected,type_ignores=[]),filename,'exec'),env)
    return env


class AuthorParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=rows(RAW)
        cls.enriched=rows(ROOT/'data/prepared/fairsteer_author_qwen3_v1/bbq.jsonl')

    def test_prompt_byte_parity(self):
        u=source_functions('utils.py',{'format_bbqa','format_bbq','format_mmlu'})
        select=source_functions('select_layer.py',{'format_bbq'})['format_bbq']
        evaluate=source_functions('evaluate.py',{'format_bbq'})['format_bbq']
        for r in self.raw[::251]:
            args=[r[k] for k in ('context','question','ans0','ans1','ans2')]
            self.assertEqual(prompt(r),u['format_bbq'](*args))
            self.assertEqual(prompt(r),evaluate([r],0))
            self.assertEqual(prompt(r,'selection'),select([r],0))
            for answer in range(3):
                self.assertEqual(prompt(r,answer=answer),u['format_bbqa'](*args,answer))
                self.assertEqual(prompt(r,shots=True,answer=answer),source_fewshot()+u['format_bbqa'](*args,answer))
        for r in rows(AUTHOR/'generate_data/mmlu.json')[::127]:
            self.assertEqual(prompt(r),u['format_mmlu'](r['question'],r['choices']))

    def test_metrics_against_author_functions(self):
        funcs=source_functions('evaluate.py',{'bbq_bias_evaluate','bbq_total_bias_score'})
        metadata=pd.read_csv(META,dtype=str)
        pred=np.random.RandomState(71).randint(0,3,len(self.raw))
        df=pd.DataFrame(self.raw)
        df['pred_label']=pred
        df['pred_cat']=[r['answer_info'][f'ans{p}'][1] for r,p in zip(self.raw,pred)]
        actual=metrics(self.enriched,pred)
        with patch.object(pd,'read_csv',return_value=metadata):
            total=funcs['bbq_total_bias_score'](df)
            cats=funcs['bbq_bias_evaluate'](df)
        for i,condition in total['context_condition'].items():
            self.assertAlmostEqual(actual[condition]['bias_score'],total['acc_bias'][i],places=12)
        for i,category in cats['category'].items():
            condition=cats['context_condition'][i]
            self.assertAlmostEqual(actual['categories'][category][condition]['bias_score'],cats['acc_bias'][i],places=12)

    def test_label_precedence_and_exclusion(self):
        r={'label':1,'target_loc':2}
        self.assertEqual([label_for(r,i) for i in range(3)],[-1,1,0])
        r['label']=2
        self.assertEqual(label_for(r,2),1)
        r['target_loc']=None
        self.assertEqual(label_for(r,0),-1)

    def test_all_unknown_has_explicit_undefined_bias(self):
        predictions=[r['unknown_index'] for r in self.enriched]
        result=metrics(self.enriched,predictions)
        self.assertEqual(result['ambig']['accuracy'],1.)
        self.assertIsNone(result['ambig']['bias_score'])


if __name__=='__main__':unittest.main()
