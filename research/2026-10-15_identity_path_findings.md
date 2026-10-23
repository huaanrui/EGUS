# Layer-7 transmission findings

All behavior evaluation sets had been inspected previously. This is an exploratory mechanistic follow-up, not fresh confirmation.
Fidelity measures reproduction of the layer7half effect; it is not a percentage of bias explained. Continuous contrasts are not multiplicity adjusted.

## Frozen receivers

{
  "created": "2026-10-15 17:15:29",
  "heads": [
    {
      "layer": 21,
      "position": "decision",
      "head": 1
    },
    {
      "layer": 17,
      "position": "decision",
      "head": 1
    },
    {
      "layer": 13,
      "position": "question_identity",
      "head": 6
    },
    {
      "layer": 11,
      "position": "question_identity",
      "head": 13
    }
  ],
  "random_heads": [
    {
      "layer": 25,
      "position": "decision",
      "head": 0
    },
    {
      "layer": 25,
      "position": "decision",
      "head": 13
    },
    {
      "layer": 25,
      "position": "question_identity",
      "head": 15
    },
    {
      "layer": 24,
      "position": "question_identity",
      "head": 5
    }
  ],
  "norm_matches": [
    {
      "target_norm": 90.77320413797516,
      "control_norm": 88.23862838749587
    },
    {
      "target_norm": 20.044983267172796,
      "control_norm": 20.069572208080192
    },
    {
      "target_norm": 19.142341952404067,
      "control_norm": 19.22813945349369
    },
    {
      "target_norm": 15.829060155532696,
      "control_norm": 15.814979935633207
    }
  ],
  "candidates": [
    {
      "k": 1,
      "sites": [
        {
          "layer": 21,
          "position": "decision",
          "head": 1
        }
      ],
      "transmission": {
        "fidelity": 0.6163458515806789,
        "restore_alignment": 0.39797980470003946,
        "transplant_alignment": 0.40421869204579286,
        "total_rms": 0.8653901556939889,
        "restore_rms": 0.3708129475999734,
        "transplant_rms": 0.3753237202863234,
        "restore_mean": 0.08091517857142858,
        "transplant_mean": 0.07900191326530612
      },
      "random_transmission": {
        "fidelity": 0.0049631671126973975,
        "restore_alignment": 0.0020690951367947985,
        "transplant_alignment": 0.0031352446405612504,
        "total_rms": 0.8653901556939889,
        "restore_rms": 0.013623387963988508,
        "transplant_rms": 0.013252613936426247,
        "restore_mean": 0.0005580357142857143,
        "transplant_mean": 0.0013552295918367347
      }
    },
    {
      "k": 2,
      "sites": [
        {
          "layer": 21,
          "position": "decision",
          "head": 1
        },
        {
          "layer": 17,
          "position": "decision",
          "head": 1
        }
      ],
      "transmission": {
        "fidelity": 0.7549020423166547,
        "restore_alignment": 0.5164737562559358,
        "transplant_alignment": 0.5294338762767598,
        "total_rms": 0.8653901556939889,
        "restore_rms": 0.4613737234513107,
        "transplant_rms": 0.47223143805824835,
        "restore_mean": 0.09582270408163265,
        "transplant_mean": 0.09566326530612244
      },
      "random_transmission": {
        "fidelity": 0.0019817739995908745,
        "restore_alignment": 0.0017996470562797204,
        "transplant_alignment": 0.00040916190004141516,
        "total_rms": 0.8653901556939889,
        "restore_rms": 0.013531646934131853,
        "transplant_rms": 0.012527870714644894,
        "restore_mean": -0.0005580357142857143,
        "transplant_mean": 0.0009566326530612245
      }
    },
    {
      "k": 4,
      "sites": [
        {
          "layer": 21,
          "position": "decision",
          "head": 1
        },
        {
          "layer": 17,
          "position": "decision",
          "head": 1
        },
        {
          "layer": 13,
          "position": "question_identity",
          "head": 6
        },
        {
          "layer": 11,
          "position": "question_identity",
          "head": 13
        }
      ],
      "transmission": {
        "fidelity": 0.87217599284133,
        "restore_alignment": 0.6943843360849195,
        "transplant_alignment": 0.6970721638510452,
        "total_rms": 0.8653901556939889,
        "restore_rms": 0.6220253670378085,
        "transplant_rms": 0.6251903010666685,
        "restore_mean": 0.12069515306122448,
        "transplant_mean": 0.1171077806122449
      },
      "random_transmission": {
        "fidelity": 0.00415731776078665,
        "restore_alignment": 0.002395094049022918,
        "transplant_alignment": 0.0020108810453254916,
        "total_rms": 0.8653901556939889,
        "restore_rms": 0.013669027582495413,
        "transplant_rms": 0.013623387963988508,
        "restore_mean": 0.0014349489795918368,
        "transplant_mean": 0.0008769132653061224
      }
    }
  ],
  "selected": {
    "k": 4,
    "sites": [
      {
        "layer": 21,
        "position": "decision",
        "head": 1
      },
      {
        "layer": 17,
        "position": "decision",
        "head": 1
      },
      {
        "layer": 13,
        "position": "question_identity",
        "head": 6
      },
      {
        "layer": 11,
        "position": "question_identity",
        "head": 13
      }
    ],
    "transmission": {
      "fidelity": 0.87217599284133,
      "restore_alignment": 0.6943843360849195,
      "transplant_alignment": 0.6970721638510452,
      "total_rms": 0.8653901556939889,
      "restore_rms": 0.6220253670378085,
      "transplant_rms": 0.6251903010666685,
      "restore_mean": 0.12069515306122448,
      "transplant_mean": 0.1171077806122449
    },
    "random_transmission": {
      "fidelity": 0.00415731776078665,
      "restore_alignment": 0.002395094049022918,
      "transplant_alignment": 0.0020108810453254916,
      "total_rms": 0.8653901556939889,
      "restore_rms": 0.013669027582495413,
      "transplant_rms": 0.013623387963988508,
      "restore_mean": 0.0014349489795918368,
      "transplant_mean": 0.0008769132653061224
    }
  },
  "criterion": "two-way transmission fidelity only; not target accuracy",
  "target_sets_previously_inspected": true
}

## Behavior and transmission

|Dataset|Method|Accuracy|Both correct|Flip|
|---|---|---:|---:|---:|
|historical|baseline|76.71%|65.75%|21.92%|
|historical|source_half|79.11%|70.21%|17.81%|
|historical|selected_restore|77.23%|66.44%|21.58%|
|historical|selected_transplant|79.28%|69.52%|19.52%|
|historical|selected_half|71.75%|60.27%|22.95%|
|historical|matched_restore|79.11%|70.21%|17.81%|
|historical|matched_transplant|76.71%|65.75%|21.92%|
|historical|matched_half|76.71%|65.75%|21.92%|
|new_type1|baseline|57.50%|41.56%|31.87%|
|new_type1|source_half|58.13%|43.12%|30.00%|
|new_type1|selected_restore|57.19%|41.25%|31.87%|
|new_type1|selected_transplant|57.19%|41.56%|31.25%|
|new_type1|selected_half|58.59%|43.75%|29.69%|
|new_type1|matched_restore|57.81%|42.81%|30.00%|
|new_type1|matched_transplant|57.50%|41.56%|31.87%|
|new_type1|matched_half|57.50%|41.56%|31.87%|
|winogender|baseline|57.71%|48.54%|18.33%|
|winogender|source_half|57.81%|48.12%|19.38%|
|winogender|selected_restore|57.50%|48.12%|18.75%|
|winogender|selected_transplant|58.02%|48.54%|18.96%|
|winogender|selected_half|56.88%|47.92%|17.92%|
|winogender|matched_restore|57.81%|48.12%|19.38%|
|winogender|matched_transplant|57.71%|48.54%|18.33%|
|winogender|matched_half|57.71%|48.54%|18.33%|

Two-way mediation estimates and uncertainty:

{
  "historical": {
    "selected": {
      "transmission": {
        "fidelity": 0.8778855873579159,
        "restore_alignment": 0.7178974713303289,
        "transplant_alignment": 0.7144746454184319,
        "total_rms": 0.8720886409244732,
        "restore_rms": 0.6513546622048755,
        "transplant_rms": 0.647420975068533,
        "restore_mean": -0.007651969178082192,
        "transplant_mean": -0.005217251712328767
      },
      "bootstrap": {
        "clusters": 129,
        "cluster": "family_id",
        "ci95": {
          "fidelity": [
            0.843271269153089,
            0.9054526439506645
          ],
          "restore_alignment": [
            0.6651656579268515,
            0.7629180120568052
          ],
          "transplant_alignment": [
            0.6638594601735919,
            0.7594564346387054
          ]
        }
      }
    },
    "matched": {
      "transmission": {
        "fidelity": 0.0018551529552999213,
        "restore_alignment": 0.0022822503912311456,
        "transplant_alignment": -0.00018743915785400303,
        "total_rms": 0.8720886409244732,
        "restore_rms": 0.013791747605433703,
        "transplant_rms": 0.013203273814716176,
        "restore_mean": -0.000508347602739726,
        "transplant_mean": -0.0006153681506849315
      },
      "bootstrap": {
        "clusters": 129,
        "cluster": "family_id",
        "ci95": {
          "fidelity": [
            -0.000117862088431514,
            0.003969039346484891
          ],
          "restore_alignment": [
            0.0009836889135208663,
            0.0036399127928870762
          ],
          "transplant_alignment": [
            -0.0015169282412098824,
            0.001334037760114498
          ]
        }
      }
    }
  },
  "new_type1": {
    "selected": {
      "transmission": {
        "fidelity": 0.8710036092123004,
        "restore_alignment": 0.662826924932098,
        "transplant_alignment": 0.6924080633670613,
        "total_rms": 0.917752841962728,
        "restore_rms": 0.6250036620986462,
        "transplant_rms": 0.6519799228689629,
        "restore_mean": 0.04345703125,
        "transplant_mean": 0.04697265625
      },
      "bootstrap": {
        "clusters": 160,
        "cluster": "family_id",
        "ci95": {
          "fidelity": [
            0.8406131874086167,
            0.8898828989897077
          ],
          "restore_alignment": [
            0.6323994317144517,
            0.6901596707392637
          ],
          "transplant_alignment": [
            0.654660479099574,
            0.7203779873427785
          ]
        }
      }
    },
    "matched": {
      "transmission": {
        "fidelity": 0.0015919665247710046,
        "restore_alignment": 0.001163516928061682,
        "transplant_alignment": 0.0006322575444040904,
        "total_rms": 0.917752841962728,
        "restore_rms": 0.013189017981354792,
        "transplant_rms": 0.013014321288248188,
        "restore_mean": 0.000341796875,
        "transplant_mean": -0.0001953125
      },
      "bootstrap": {
        "clusters": 160,
        "cluster": "family_id",
        "ci95": {
          "fidelity": [
            0.0004778803857360764,
            0.0026972815942351847
          ],
          "restore_alignment": [
            0.0001816960002623259,
            0.00198146518037318
          ],
          "transplant_alignment": [
            -0.00033244432358642983,
            0.0018601643860741228
          ]
        }
      }
    }
  },
  "winogender": {
    "selected": {
      "transmission": {
        "fidelity": 0.8292302533289193,
        "restore_alignment": 0.6050708665739275,
        "transplant_alignment": 0.6136425230318509,
        "total_rms": 1.0890869721664604,
        "restore_rms": 0.6739515285551305,
        "transplant_rms": 0.6853680470700845,
        "restore_mean": 0.006005859375,
        "transplant_mean": 0.006689453125
      },
      "bootstrap": {
        "clusters": 120,
        "cluster": "family_id",
        "ci95": {
          "fidelity": [
            0.807527163728313,
            0.8480747517289506
          ],
          "restore_alignment": [
            0.582102238459751,
            0.6299291488028249
          ],
          "transplant_alignment": [
            0.5889818707914378,
            0.6448162894248886
          ]
        }
      }
    },
    "matched": {
      "transmission": {
        "fidelity": 0.0015168388516579867,
        "restore_alignment": 0.000753648818089973,
        "transplant_alignment": 0.0009123117271615463,
        "total_rms": 1.0890869721664604,
        "restore_rms": 0.013150397055158957,
        "transplant_rms": 0.013446808185983207,
        "restore_mean": 0.00013020833333333333,
        "transplant_mean": 0.0002766927083333333
      },
      "bootstrap": {
        "clusters": 120,
        "cluster": "family_id",
        "ci95": {
          "fidelity": [
            0.00011012341947224392,
            0.0029164833038737518
          ],
          "restore_alignment": [
            -0.00016944875010665685,
            0.0016991096364728149
          ],
          "transplant_alignment": [
            0.00013188068853189192,
            0.0017291710372209462
          ]
        }
      }
    }
  }
}

Causal behavior contrasts:

{
  "historical": {
    "source_half_vs_selected_restore": {
      "accuracy": {
        "delta": 0.018835616438356163,
        "ci95": [
          0.003424657534246575,
          0.03523489932885906
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.0427393151712072
      },
      "both_correct": {
        "delta": 0.03767123287671233,
        "ci95": [
          0.013513513513513514,
          0.06462585034013606
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.012246938265433641
      },
      "flip": {
        "delta": -0.03767123287671233,
        "ci95": [
          -0.0684931506849315,
          -0.0070921985815602835
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.02324418895276181
      }
    },
    "selected_transplant_vs_source_half": {
      "accuracy": {
        "delta": 0.0017123287671232876,
        "ci95": [
          -0.008445945945945946,
          0.01206896551724138
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 1.0
      },
      "both_correct": {
        "delta": -0.00684931506849315,
        "ci95": [
          -0.023809523809523808,
          0.007195547909498493
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.6928267933016746
      },
      "flip": {
        "delta": 0.017123287671232876,
        "ci95": [
          0.0,
          0.03793103448275862
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.17720569857535617
      }
    },
    "selected_restore_vs_baseline": {
      "accuracy": {
        "delta": 0.005136986301369863,
        "ci95": [
          -0.001736111111111111,
          0.012502248201438857
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.37640589852536865
      },
      "both_correct": {
        "delta": 0.00684931506849315,
        "ci95": [
          0.0,
          0.017361111111111112
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.4886278430392402
      },
      "flip": {
        "delta": -0.003424657534246575,
        "ci95": [
          -0.017985611510791366,
          0.01056338028169014
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 1.0
      }
    }
  },
  "new_type1": {
    "source_half_vs_selected_restore": {
      "accuracy": {
        "delta": 0.009375,
        "ci95": [
          -0.00625,
          0.0265625
        ],
        "cluster": "family_id",
        "clusters": 160,
        "p_sign": 0.34841289677580606
      },
      "both_correct": {
        "delta": 0.01875,
        "ci95": [
          -0.003125,
          0.040625
        ],
        "cluster": "family_id",
        "clusters": 160,
        "p_sign": 0.18645338665333666
      },
      "flip": {
        "delta": -0.01875,
        "ci95": [
          -0.05,
          0.0125
        ],
        "cluster": "family_id",
        "clusters": 160,
        "p_sign": 0.3141714571357161
      }
    },
    "selected_transplant_vs_source_half": {
      "accuracy": {
        "delta": -0.009375,
        "ci95": [
          -0.0203125,
          3.906250000014211e-05
        ],
        "cluster": "family_id",
        "clusters": 160,
        "p_sign": 0.15046238440389903
      },
      "both_correct": {
        "delta": -0.015625,
        "ci95": [
          -0.03125,
          -0.003125
        ],
        "cluster": "family_id",
        "clusters": 160,
        "p_sign": 0.06273431642089478
      },
      "flip": {
        "delta": 0.0125,
        "ci95": [
          -0.009375,
          0.034375
        ],
        "cluster": "family_id",
        "clusters": 160,
        "p_sign": 0.38240439890027494
      }
    },
    "selected_restore_vs_baseline": {
      "accuracy": {
        "delta": -0.003125,
        "ci95": [
          -0.0140625,
          0.0078125
        ],
        "cluster": "family_id",
        "clusters": 160,
        "p_sign": 0.7865533616595851
      },
      "both_correct": {
        "delta": -0.003125,
        "ci95": [
          -0.015625,
          0.00625
        ],
        "cluster": "family_id",
        "clusters": 160,
        "p_sign": 1.0
      },
      "flip": {
        "delta": 0.0,
        "ci95": [
          -0.01875,
          0.01875
        ],
        "cluster": "family_id",
        "clusters": 160,
        "p_sign": 1.0
      }
    }
  },
  "winogender": {
    "source_half_vs_selected_restore": {
      "accuracy": {
        "delta": 0.003125,
        "ci95": [
          -0.008333333333333333,
          0.014583333333333334
        ],
        "cluster": "family_id",
        "clusters": 120,
        "p_sign": 0.7243189202699325
      },
      "both_correct": {
        "delta": 0.0,
        "ci95": [
          -0.014583333333333334,
          0.014583333333333334
        ],
        "cluster": "family_id",
        "clusters": 120,
        "p_sign": 1.0
      },
      "flip": {
        "delta": 0.00625,
        "ci95": [
          -0.014583333333333334,
          0.025
        ],
        "cluster": "family_id",
        "clusters": 120,
        "p_sign": 0.7020744813796551
      }
    },
    "selected_transplant_vs_source_half": {
      "accuracy": {
        "delta": 0.0020833333333333333,
        "ci95": [
          -0.005208333333333333,
          0.009375
        ],
        "cluster": "family_id",
        "clusters": 120,
        "p_sign": 0.7960509872531867
      },
      "both_correct": {
        "delta": 0.004166666666666667,
        "ci95": [
          -0.004166666666666667,
          0.0125
        ],
        "cluster": "family_id",
        "clusters": 120,
        "p_sign": 0.6288427893026743
      },
      "flip": {
        "delta": -0.004166666666666667,
        "ci95": [
          -0.01875,
          0.010416666666666666
        ],
        "cluster": "family_id",
        "clusters": 120,
        "p_sign": 0.7998000499875031
      }
    },
    "selected_restore_vs_baseline": {
      "accuracy": {
        "delta": -0.0020833333333333333,
        "ci95": [
          -0.010416666666666666,
          0.00625
        ],
        "cluster": "family_id",
        "clusters": 120,
        "p_sign": 0.8032991752061984
      },
      "both_correct": {
        "delta": -0.004166666666666667,
        "ci95": [
          -0.014583333333333334,
          0.00625
        ],
        "cluster": "family_id",
        "clusters": 120,
        "p_sign": 0.6873281679580105
      },
      "flip": {
        "delta": 0.004166666666666667,
        "ci95": [
          -0.010416666666666666,
          0.01875
        ],
        "cluster": "family_id",
        "clusters": 120,
        "p_sign": 0.7680579855036241
      }
    }
  }
}

## Source controls and utility

{
  "source_controls": {
    "identity_swap": {
      "transmission": {
        "fidelity": 0.854594521820458,
        "restore_alignment": 0.678141052096586,
        "transplant_alignment": 0.6704529779998948,
        "total_rms": 1.8611444340596333,
        "restore_rms": 1.31918087830944,
        "transplant_rms": 1.2969389750316145,
        "restore_mean": -0.1370849609375,
        "transplant_mean": -0.104736328125
      },
      "source_mean_norm": 28.75114477316985,
      "receiver_response_per_source_norm": [
        0.3617329630657162,
        0.14545784578525084,
        0.2561876524310194,
        0.20158249267475425
      ],
      "margin_absolute_response_per_source_norm": 0.04424819088040912,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.796875,
        "both_correct": 0.7109375,
        "flip": 0.171875,
        "gap": 0.09375,
        "by_permutation": {
          "0": {
            "accuracy": 0.8671875,
            "both_correct": 0.78125,
            "flip": 0.171875,
            "gap": 0.078125
          },
          "1": {
            "accuracy": 0.7265625,
            "both_correct": 0.640625,
            "flip": 0.171875,
            "gap": 0.109375
          }
        },
        "gold_margin": 3.5667724609375
      }
    },
    "neutral_prefix": {
      "transmission": {
        "fidelity": 0.8854725955821412,
        "restore_alignment": 0.7305538539151742,
        "transplant_alignment": 0.711177975568448,
        "total_rms": 0.5144454874385608,
        "restore_rms": 0.38832206895287297,
        "transplant_rms": 0.378996848060566,
        "restore_mean": 0.05621337890625,
        "transplant_mean": 0.0513916015625
      },
      "source_mean_norm": 8.404080864983019,
      "receiver_response_per_source_norm": [
        0.3227045814817442,
        0.13659260850356908,
        0.2543780789842448,
        0.206069704314774
      ],
      "margin_absolute_response_per_source_norm": 0.03746195927820094,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.75,
        "both_correct": 0.6171875,
        "flip": 0.265625,
        "gap": 0.109375,
        "by_permutation": {
          "0": {
            "accuracy": 0.8046875,
            "both_correct": 0.6875,
            "flip": 0.234375,
            "gap": 0.109375
          },
          "1": {
            "accuracy": 0.6953125,
            "both_correct": 0.546875,
            "flip": 0.296875,
            "gap": 0.109375
          }
        },
        "gold_margin": 3.49713134765625
      }
    },
    "neutral_norm_matched": {
      "transmission": {
        "fidelity": 0.8639106076718855,
        "restore_alignment": 0.7169003951239041,
        "transplant_alignment": 0.7000842135277097,
        "total_rms": 1.5169489475579165,
        "restore_rms": 1.1430533700466057,
        "transplant_rms": 1.1130246586154482,
        "restore_mean": -0.02301025390625,
        "transplant_mean": -0.04693603515625
      },
      "source_mean_norm": 28.75114477316985,
      "receiver_response_per_source_norm": [
        0.2961313246575561,
        0.12766990296150776,
        0.24684970498392672,
        0.1982997425713548
      ],
      "margin_absolute_response_per_source_norm": 0.03619913152413359,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.72265625,
        "both_correct": 0.578125,
        "flip": 0.2890625,
        "gap": 0.1015625,
        "by_permutation": {
          "0": {
            "accuracy": 0.7578125,
            "both_correct": 0.609375,
            "flip": 0.296875,
            "gap": 0.078125
          },
          "1": {
            "accuracy": 0.6875,
            "both_correct": 0.546875,
            "flip": 0.28125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.28497314453125
      }
    },
    "random_norm_0": {
      "transmission": {
        "fidelity": 0.8616063620436176,
        "restore_alignment": 0.7134291930810677,
        "transplant_alignment": 0.7010860928740834,
        "total_rms": 1.3814538865623573,
        "restore_rms": 1.0375741035929806,
        "transplant_rms": 1.0167591511502387,
        "restore_mean": 0.11444091796875,
        "transplant_mean": 0.10748291015625
      },
      "source_mean_norm": 28.75114477316985,
      "receiver_response_per_source_norm": [
        0.281485668784927,
        0.115901972063254,
        0.22512247490105963,
        0.19821644997295546
      ],
      "margin_absolute_response_per_source_norm": 0.03315907106321406,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.73046875,
        "both_correct": 0.59375,
        "flip": 0.2734375,
        "gap": 0.1484375,
        "by_permutation": {
          "0": {
            "accuracy": 0.796875,
            "both_correct": 0.65625,
            "flip": 0.28125,
            "gap": 0.09375
          },
          "1": {
            "accuracy": 0.6640625,
            "both_correct": 0.53125,
            "flip": 0.265625,
            "gap": 0.203125
          }
        },
        "gold_margin": 3.33441162109375
      }
    },
    "random_norm_1": {
      "transmission": {
        "fidelity": 0.8359019663476833,
        "restore_alignment": 0.6521947303257758,
        "transplant_alignment": 0.6527670303593077,
        "total_rms": 1.3321801602982186,
        "restore_rms": 0.912810160559791,
        "transplant_rms": 0.9119551403657697,
        "restore_mean": -0.05377197265625,
        "transplant_mean": -0.05657958984375
      },
      "source_mean_norm": 28.75114477316985,
      "receiver_response_per_source_norm": [
        0.26728582582603067,
        0.11737487224413899,
        0.22818561695571377,
        0.1970831608405933
      ],
      "margin_absolute_response_per_source_norm": 0.03301030347357266,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.78125,
        "both_correct": 0.6796875,
        "flip": 0.203125,
        "gap": 0.078125,
        "by_permutation": {
          "0": {
            "accuracy": 0.8359375,
            "both_correct": 0.71875,
            "flip": 0.234375,
            "gap": 0.046875
          },
          "1": {
            "accuracy": 0.7265625,
            "both_correct": 0.640625,
            "flip": 0.171875,
            "gap": 0.109375
          }
        },
        "gold_margin": 3.46966552734375
      }
    },
    "random_norm_2": {
      "transmission": {
        "fidelity": 0.8242523183453803,
        "restore_alignment": 0.6413110474348996,
        "transplant_alignment": 0.6249861265745029,
        "total_rms": 1.6530148561413547,
        "restore_rms": 1.110436589752355,
        "transplant_rms": 1.0875057176461584,
        "restore_mean": -0.04827880859375,
        "transplant_mean": -0.053955078125
      },
      "source_mean_norm": 28.75114477316985,
      "receiver_response_per_source_norm": [
        0.2837565493533535,
        0.12275603546939035,
        0.2303579042198048,
        0.19758986360214714
      ],
      "margin_absolute_response_per_source_norm": 0.03504401946515785,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.7578125,
        "both_correct": 0.6171875,
        "flip": 0.28125,
        "gap": 0.109375,
        "by_permutation": {
          "0": {
            "accuracy": 0.796875,
            "both_correct": 0.671875,
            "flip": 0.25,
            "gap": 0.09375
          },
          "1": {
            "accuracy": 0.71875,
            "both_correct": 0.5625,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.26361083984375
      }
    }
  },
  "zero_neutral_delta_rows": 0,
  "record": {
    "transmission": {
      "fidelity": 0.6344571841220514,
      "restore_alignment": 0.4275387480665425,
      "transplant_alignment": 0.4274888649605865,
      "total_rms": 1.106148123495249,
      "restore_rms": 0.5198757778858071,
      "transplant_rms": 0.5191272512856012,
      "restore_mean": -0.047149658203125,
      "transplant_mean": -0.04888916015625
    },
    "baseline": {
      "rows": 512,
      "pairs": 128,
      "accuracy": 0.9140625,
      "both_correct": 0.890625,
      "flip": 0.046875,
      "gap": 0.03125,
      "by_permutation": {
        "0": {
          "accuracy": 0.921875,
          "both_correct": 0.90625,
          "flip": 0.03125,
          "gap": 0.03125
        },
        "1": {
          "accuracy": 0.90625,
          "both_correct": 0.875,
          "flip": 0.0625,
          "gap": 0.03125
        }
      },
      "gold_margin": 6.721954345703125
    },
    "record_source_swap": {
      "rows": 512,
      "pairs": 128,
      "accuracy": 0.916015625,
      "both_correct": 0.8984375,
      "flip": 0.03515625,
      "gap": 0.02734375,
      "by_permutation": {
        "0": {
          "accuracy": 0.91796875,
          "both_correct": 0.8984375,
          "flip": 0.0390625,
          "gap": 0.0234375
        },
        "1": {
          "accuracy": 0.9140625,
          "both_correct": 0.8984375,
          "flip": 0.03125,
          "gap": 0.03125
        }
      },
      "gold_margin": 6.68560791015625
    },
    "selected_half": {
      "rows": 512,
      "pairs": 128,
      "accuracy": 0.921875,
      "both_correct": 0.8984375,
      "flip": 0.046875,
      "gap": 0.0390625,
      "by_permutation": {
        "0": {
          "accuracy": 0.93359375,
          "both_correct": 0.9140625,
          "flip": 0.0390625,
          "gap": 0.0390625
        },
        "1": {
          "accuracy": 0.91015625,
          "both_correct": 0.8828125,
          "flip": 0.0546875,
          "gap": 0.0390625
        }
      },
      "gold_margin": 5.446319580078125
    }
  },
  "utilities": {
    "literal": {
      "baseline": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.96875,
        "both_correct": 0.9375,
        "flip": 0.9375,
        "gap": 0.03125,
        "by_permutation": {
          "0": {
            "accuracy": 0.9375,
            "both_correct": 0.875,
            "flip": 0.875,
            "gap": 0.0625
          },
          "1": {
            "accuracy": 1.0,
            "both_correct": 1.0,
            "flip": 1.0,
            "gap": 0.0
          }
        },
        "gold_margin": 7.602783203125
      },
      "source_half": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.95703125,
        "both_correct": 0.9140625,
        "flip": 0.9140625,
        "gap": 0.0078125,
        "by_permutation": {
          "0": {
            "accuracy": 0.9140625,
            "both_correct": 0.828125,
            "flip": 0.828125,
            "gap": 0.015625
          },
          "1": {
            "accuracy": 1.0,
            "both_correct": 1.0,
            "flip": 1.0,
            "gap": 0.0
          }
        },
        "gold_margin": 7.57666015625
      },
      "selected_half": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.96875,
        "both_correct": 0.9375,
        "flip": 0.9375,
        "gap": 0.03125,
        "by_permutation": {
          "0": {
            "accuracy": 0.9375,
            "both_correct": 0.875,
            "flip": 0.875,
            "gap": 0.0625
          },
          "1": {
            "accuracy": 1.0,
            "both_correct": 1.0,
            "flip": 1.0,
            "gap": 0.0
          }
        },
        "gold_margin": 6.3917236328125
      }
    },
    "reading": {
      "baseline": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.70703125,
        "both_correct": 0.4140625,
        "flip": 0.4140625,
        "gap": -0.0234375,
        "by_permutation": {
          "0": {
            "accuracy": 0.671875,
            "both_correct": 0.34375,
            "flip": 0.34375,
            "gap": 0.0
          },
          "1": {
            "accuracy": 0.7421875,
            "both_correct": 0.484375,
            "flip": 0.484375,
            "gap": -0.046875
          }
        },
        "gold_margin": 0.76739501953125
      },
      "source_half": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.7109375,
        "both_correct": 0.421875,
        "flip": 0.421875,
        "gap": -0.015625,
        "by_permutation": {
          "0": {
            "accuracy": 0.6640625,
            "both_correct": 0.328125,
            "flip": 0.328125,
            "gap": 0.015625
          },
          "1": {
            "accuracy": 0.7578125,
            "both_correct": 0.515625,
            "flip": 0.515625,
            "gap": -0.046875
          }
        },
        "gold_margin": 0.7060546875
      },
      "selected_half": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.5703125,
        "both_correct": 0.140625,
        "flip": 0.140625,
        "gap": 0.0,
        "by_permutation": {
          "0": {
            "accuracy": 0.5390625,
            "both_correct": 0.078125,
            "flip": 0.078125,
            "gap": -0.015625
          },
          "1": {
            "accuracy": 0.6015625,
            "both_correct": 0.203125,
            "flip": 0.203125,
            "gap": 0.015625
          }
        },
        "gold_margin": 0.53875732421875
      }
    },
    "binding": {
      "baseline": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.98828125,
        "both_correct": 0.9765625,
        "flip": 0.9765625,
        "gap": 0.0078125,
        "by_permutation": {
          "0": {
            "accuracy": 0.9921875,
            "both_correct": 0.984375,
            "flip": 0.984375,
            "gap": -0.015625
          },
          "1": {
            "accuracy": 0.984375,
            "both_correct": 0.96875,
            "flip": 0.96875,
            "gap": 0.03125
          }
        },
        "gold_margin": 6.99957275390625
      },
      "source_half": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.9921875,
        "both_correct": 0.984375,
        "flip": 0.984375,
        "gap": 0.0,
        "by_permutation": {
          "0": {
            "accuracy": 0.9921875,
            "both_correct": 0.984375,
            "flip": 0.984375,
            "gap": -0.015625
          },
          "1": {
            "accuracy": 0.9921875,
            "both_correct": 0.984375,
            "flip": 0.984375,
            "gap": 0.015625
          }
        },
        "gold_margin": 6.986328125
      },
      "selected_half": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.98046875,
        "both_correct": 0.96875,
        "flip": 0.9765625,
        "gap": 0.0234375,
        "by_permutation": {
          "0": {
            "accuracy": 0.984375,
            "both_correct": 0.984375,
            "flip": 1.0,
            "gap": 0.0
          },
          "1": {
            "accuracy": 0.9765625,
            "both_correct": 0.953125,
            "flip": 0.953125,
            "gap": 0.046875
          }
        },
        "gold_margin": 5.29998779296875
      }
    },
    "role_swap": {
      "baseline": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.7421875,
        "both_correct": 0.6171875,
        "flip": 0.25,
        "gap": -0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.7109375,
            "both_correct": 0.578125,
            "flip": 0.265625,
            "gap": -0.140625
          },
          "1": {
            "accuracy": 0.7734375,
            "both_correct": 0.65625,
            "flip": 0.234375,
            "gap": -0.109375
          }
        },
        "gold_margin": 3.203125
      },
      "source_half": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.79296875,
        "both_correct": 0.703125,
        "flip": 0.1796875,
        "gap": -0.0703125,
        "by_permutation": {
          "0": {
            "accuracy": 0.7578125,
            "both_correct": 0.671875,
            "flip": 0.171875,
            "gap": -0.078125
          },
          "1": {
            "accuracy": 0.828125,
            "both_correct": 0.734375,
            "flip": 0.1875,
            "gap": -0.0625
          }
        },
        "gold_margin": 3.50885009765625
      },
      "selected_half": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.71484375,
        "both_correct": 0.6171875,
        "flip": 0.1953125,
        "gap": -0.0859375,
        "by_permutation": {
          "0": {
            "accuracy": 0.6640625,
            "both_correct": 0.5625,
            "flip": 0.203125,
            "gap": -0.109375
          },
          "1": {
            "accuracy": 0.765625,
            "both_correct": 0.671875,
            "flip": 0.1875,
            "gap": -0.0625
          }
        },
        "gold_margin": 1.85125732421875
      }
    }
  },
  "anonymous": {
    "baseline": {
      "rows": 256,
      "pairs": 64,
      "accuracy": 0.80078125,
      "both_correct": 0.65625,
      "flip": 0.2890625,
      "gap": 0.0078125,
      "by_permutation": {
        "0": {
          "accuracy": 0.765625,
          "both_correct": 0.609375,
          "flip": 0.3125,
          "gap": 0.03125
        },
        "1": {
          "accuracy": 0.8359375,
          "both_correct": 0.703125,
          "flip": 0.265625,
          "gap": -0.015625
        }
      },
      "gold_margin": 4.2603759765625
    },
    "source_half": {
      "rows": 256,
      "pairs": 64,
      "accuracy": 0.80859375,
      "both_correct": 0.671875,
      "flip": 0.2734375,
      "gap": 0.0234375,
      "by_permutation": {
        "0": {
          "accuracy": 0.7578125,
          "both_correct": 0.609375,
          "flip": 0.296875,
          "gap": 0.046875
        },
        "1": {
          "accuracy": 0.859375,
          "both_correct": 0.734375,
          "flip": 0.25,
          "gap": 0.0
        }
      },
      "gold_margin": 4.57440185546875
    },
    "selected_half": {
      "rows": 256,
      "pairs": 64,
      "accuracy": 0.68359375,
      "both_correct": 0.5546875,
      "flip": 0.2578125,
      "gap": 0.0390625,
      "by_permutation": {
        "0": {
          "accuracy": 0.625,
          "both_correct": 0.53125,
          "flip": 0.1875,
          "gap": 0.03125
        },
        "1": {
          "accuracy": 0.7421875,
          "both_correct": 0.578125,
          "flip": 0.328125,
          "gap": 0.046875
        }
      },
      "gold_margin": 1.86383056640625
    }
  },
  "limits": "Derived record task measures fact reading, not native disambiguation. Norm matching is at source only. All historical exploratory."
}

## Q/K/V and controlled direct edges

{
  "heads": [
    {
      "layer": 21,
      "position": "decision",
      "head": 1
    },
    {
      "layer": 17,
      "position": "decision",
      "head": 1
    },
    {
      "layer": 13,
      "position": "question_identity",
      "head": 6
    },
    {
      "layer": 11,
      "position": "question_identity",
      "head": 13
    }
  ],
  "results": {
    "baseline": {
      "alignment": 0.0,
      "effect_rms": 0.0,
      "reproduction_fidelity": 0.0,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58428955078125
      }
    },
    "source_half": {
      "alignment": 1.0,
      "effect_rms": 0.8783412897761801,
      "reproduction_fidelity": 1.0,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.79296875,
        "both_correct": 0.6953125,
        "flip": 0.1953125,
        "gap": 0.1015625,
        "by_permutation": {
          "0": {
            "accuracy": 0.8359375,
            "both_correct": 0.75,
            "flip": 0.171875,
            "gap": 0.078125
          },
          "1": {
            "accuracy": 0.75,
            "both_correct": 0.640625,
            "flip": 0.21875,
            "gap": 0.125
          }
        },
        "gold_margin": 3.7357177734375
      }
    },
    "h0_v_source": {
      "alignment": -0.0012015442068139424,
      "effect_rms": 0.01284467425582608,
      "reproduction_fidelity": -0.002616943503935376,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58349609375
      }
    },
    "h0_k_source": {
      "alignment": -0.0009159920342069252,
      "effect_rms": 0.013389950391408289,
      "reproduction_fidelity": -0.002064381507591806,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58355712890625
      }
    },
    "h0_kv_source": {
      "alignment": -0.0008356418557677213,
      "effect_rms": 0.012918707573557571,
      "reproduction_fidelity": -0.001887611115025667,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5831298828125
      }
    },
    "h0_q_receiver": {
      "alignment": 0.2498989441986553,
      "effect_rms": 0.25560205930259355,
      "reproduction_fidelity": 0.4151137449487551,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.77734375,
        "both_correct": 0.65625,
        "flip": 0.2421875,
        "gap": 0.1171875,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.71875,
            "flip": 0.21875,
            "gap": 0.09375
          },
          "1": {
            "accuracy": 0.7265625,
            "both_correct": 0.59375,
            "flip": 0.265625,
            "gap": 0.140625
          }
        },
        "gold_margin": 3.60968017578125
      }
    },
    "h0_all_k": {
      "alignment": 0.12761091724055237,
      "effect_rms": 0.14695562219602057,
      "reproduction_fidelity": 0.22722906846947744,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.76953125,
        "both_correct": 0.6328125,
        "flip": 0.2734375,
        "gap": 0.1171875,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.7109375,
            "both_correct": 0.5625,
            "flip": 0.296875,
            "gap": 0.109375
          }
        },
        "gold_margin": 3.6297607421875
      }
    },
    "h0_all_v": {
      "alignment": 0.003942103369886484,
      "effect_rms": 0.0284546919615888,
      "reproduction_fidelity": 0.006834709793697824,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5806884765625
      }
    },
    "h0_weights": {
      "alignment": 0.3848439784958199,
      "effect_rms": 0.37317595320870106,
      "reproduction_fidelity": 0.5891781907364897,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.78515625,
        "both_correct": 0.671875,
        "flip": 0.2265625,
        "gap": 0.1171875,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.71875,
            "flip": 0.21875,
            "gap": 0.09375
          },
          "1": {
            "accuracy": 0.7421875,
            "both_correct": 0.625,
            "flip": 0.234375,
            "gap": 0.140625
          }
        },
        "gold_margin": 3.66064453125
      }
    },
    "h0_all_qkv": {
      "alignment": 0.3910111637301767,
      "effect_rms": 0.3773387458422068,
      "reproduction_fidelity": 0.5974629122118673,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.78515625,
        "both_correct": 0.671875,
        "flip": 0.2265625,
        "gap": 0.1171875,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.71875,
            "flip": 0.21875,
            "gap": 0.09375
          },
          "1": {
            "accuracy": 0.7421875,
            "both_correct": 0.625,
            "flip": 0.234375,
            "gap": 0.140625
          }
        },
        "gold_margin": 3.6573486328125
      }
    },
    "h0_direct_v": {
      "alignment": -0.0009567852017222134,
      "effect_rms": 0.01317454840159379,
      "reproduction_fidelity": -0.0021385509030742433,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58441162109375
      }
    },
    "h0_direct_k": {
      "alignment": -0.0010235376576563213,
      "effect_rms": 0.01295556558732578,
      "reproduction_fidelity": -0.002264638875394187,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58453369140625
      }
    },
    "h0_direct_kv": {
      "alignment": -0.0015266533903448753,
      "effect_rms": 0.01284467425582608,
      "reproduction_fidelity": -0.0032671618709971817,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5826416015625
      }
    },
    "h1_v_source": {
      "alignment": 0.0027195445010192113,
      "effect_rms": 0.018295892573432806,
      "reproduction_fidelity": 0.005005198038466774,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5860595703125
      }
    },
    "h1_k_source": {
      "alignment": -0.0007763063393818475,
      "effect_rms": 0.0146484375,
      "reproduction_fidelity": -0.0018307479118224723,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5838623046875
      }
    },
    "h1_kv_source": {
      "alignment": 0.0026404304791713797,
      "effect_rms": 0.018399847345133568,
      "reproduction_fidelity": 0.004842025368405611,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5849609375
      }
    },
    "h1_q_receiver": {
      "alignment": 0.015057623439506822,
      "effect_rms": 0.06266001245327463,
      "reproduction_fidelity": 0.025025990192333647,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.76171875,
        "both_correct": 0.6171875,
        "flip": 0.2890625,
        "gap": 0.1171875,
        "by_permutation": {
          "0": {
            "accuracy": 0.8203125,
            "both_correct": 0.6875,
            "flip": 0.265625,
            "gap": 0.109375
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5836181640625
      }
    },
    "h1_all_k": {
      "alignment": 0.17633650160267703,
      "effect_rms": 0.17266663394962034,
      "reproduction_fidelity": 0.314028275845871,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.76953125,
        "both_correct": 0.6328125,
        "flip": 0.2734375,
        "gap": 0.1328125,
        "by_permutation": {
          "0": {
            "accuracy": 0.8203125,
            "both_correct": 0.703125,
            "flip": 0.234375,
            "gap": 0.109375
          },
          "1": {
            "accuracy": 0.71875,
            "both_correct": 0.5625,
            "flip": 0.3125,
            "gap": 0.15625
          }
        },
        "gold_margin": 3.60955810546875
      }
    },
    "h1_all_v": {
      "alignment": 0.04571183459235882,
      "effect_rms": 0.08050571814658183,
      "reproduction_fidelity": 0.08302274898975104,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.76171875,
        "both_correct": 0.625,
        "flip": 0.2734375,
        "gap": 0.1171875,
        "by_permutation": {
          "0": {
            "accuracy": 0.8203125,
            "both_correct": 0.703125,
            "flip": 0.234375,
            "gap": 0.109375
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.57867431640625
      }
    },
    "h1_weights": {
      "alignment": 0.1944029301855842,
      "effect_rms": 0.18933227459965224,
      "reproduction_fidelity": 0.3423412064146638,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.6328125,
        "flip": 0.265625,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.8203125,
            "both_correct": 0.703125,
            "flip": 0.234375,
            "gap": 0.109375
          },
          "1": {
            "accuracy": 0.7109375,
            "both_correct": 0.5625,
            "flip": 0.296875,
            "gap": 0.140625
          }
        },
        "gold_margin": 3.61383056640625
      }
    },
    "h1_all_qkv": {
      "alignment": 0.2353506172747939,
      "effect_rms": 0.224167363994133,
      "reproduction_fidelity": 0.40556567143699496,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.6328125,
        "flip": 0.265625,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.8203125,
            "both_correct": 0.703125,
            "flip": 0.234375,
            "gap": 0.109375
          },
          "1": {
            "accuracy": 0.7109375,
            "both_correct": 0.5625,
            "flip": 0.296875,
            "gap": 0.140625
          }
        },
        "gold_margin": 3.60821533203125
      }
    },
    "h1_direct_v": {
      "alignment": -0.0018517625738758083,
      "effect_rms": 0.014218964627501012,
      "reproduction_fidelity": -0.0039655903451225605,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58343505859375
      }
    },
    "h1_direct_k": {
      "alignment": -0.0015081110414742898,
      "effect_rms": 0.014285877771804486,
      "reproduction_fidelity": -0.0032807595935022693,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58184814453125
      }
    },
    "h1_direct_kv": {
      "alignment": -0.0022485688397063386,
      "effect_rms": 0.014484762670108717,
      "reproduction_fidelity": -0.004769092129514707,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58197021484375
      }
    },
    "h2_v_source": {
      "alignment": 0.18887483790896695,
      "effect_rms": 0.23966204690879953,
      "reproduction_fidelity": 0.3032984366327589,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.7734375,
        "both_correct": 0.640625,
        "flip": 0.265625,
        "gap": 0.140625,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.71875,
            "both_correct": 0.578125,
            "flip": 0.28125,
            "gap": 0.15625
          }
        },
        "gold_margin": 3.64263916015625
      }
    },
    "h2_k_source": {
      "alignment": -0.04889493781514267,
      "effect_rms": 0.19324590262625477,
      "reproduction_fidelity": -0.14619529543524457,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.57147216796875
      }
    },
    "h2_kv_source": {
      "alignment": 0.24762070760075602,
      "effect_rms": 0.30885323346893867,
      "reproduction_fidelity": 0.37159608830608226,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.76953125,
        "both_correct": 0.6328125,
        "flip": 0.2734375,
        "gap": 0.1328125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.7109375,
            "both_correct": 0.5625,
            "flip": 0.296875,
            "gap": 0.140625
          }
        },
        "gold_margin": 3.65447998046875
      }
    },
    "h2_q_receiver": {
      "alignment": 0.07241776159236747,
      "effect_rms": 0.1388746042323203,
      "reproduction_fidelity": 0.11983672843741155,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5750732421875
      }
    },
    "h2_all_k": {
      "alignment": -0.03898590657870176,
      "effect_rms": 0.1679403591558576,
      "reproduction_fidelity": -0.1145299081906499,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.57916259765625
      }
    },
    "h2_all_v": {
      "alignment": 0.2354445651757382,
      "effect_rms": 0.2858026572208728,
      "reproduction_fidelity": 0.3650110821438417,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.7734375,
        "both_correct": 0.640625,
        "flip": 0.265625,
        "gap": 0.140625,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.71875,
            "both_correct": 0.578125,
            "flip": 0.28125,
            "gap": 0.15625
          }
        },
        "gold_margin": 3.6546630859375
      }
    },
    "h2_weights": {
      "alignment": -0.006916296128728403,
      "effect_rms": 0.16903278533920943,
      "reproduction_fidelity": -0.050867843734972995,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.76171875,
        "both_correct": 0.625,
        "flip": 0.2734375,
        "gap": 0.1171875,
        "by_permutation": {
          "0": {
            "accuracy": 0.8203125,
            "both_correct": 0.703125,
            "flip": 0.234375,
            "gap": 0.109375
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.55914306640625
      }
    },
    "h2_all_qkv": {
      "alignment": 0.31087854885105426,
      "effect_rms": 0.3534211933613945,
      "reproduction_fidelity": 0.45985272430370394,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.76953125,
        "both_correct": 0.640625,
        "flip": 0.2578125,
        "gap": 0.1328125,
        "by_permutation": {
          "0": {
            "accuracy": 0.8203125,
            "both_correct": 0.703125,
            "flip": 0.234375,
            "gap": 0.109375
          },
          "1": {
            "accuracy": 0.71875,
            "both_correct": 0.578125,
            "flip": 0.28125,
            "gap": 0.15625
          }
        },
        "gold_margin": 3.64849853515625
      }
    },
    "h2_direct_v": {
      "alignment": -0.006331594061009273,
      "effect_rms": 0.0657132962549247,
      "reproduction_fidelity": -0.018260505167752683,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.60723876953125
      }
    },
    "h2_direct_k": {
      "alignment": -0.001438886272357437,
      "effect_rms": 0.04789140560640324,
      "reproduction_fidelity": -0.005850729146965383,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5849609375
      }
    },
    "h2_direct_kv": {
      "alignment": -0.006716038760926079,
      "effect_rms": 0.0808308279167003,
      "reproduction_fidelity": -0.02190098632934423,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.6080322265625
      }
    },
    "h3_v_source": {
      "alignment": 0.23597487635343695,
      "effect_rms": 0.3215961322305121,
      "reproduction_fidelity": 0.33789104268572323,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.7734375,
        "both_correct": 0.640625,
        "flip": 0.265625,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.71875,
            "both_correct": 0.578125,
            "flip": 0.28125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.65020751953125
      }
    },
    "h3_k_source": {
      "alignment": 0.00025093978804859084,
      "effect_rms": 0.03836038933867648,
      "reproduction_fidelity": -0.001405510044390379,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58447265625
      }
    },
    "h3_kv_source": {
      "alignment": 0.23894165217273064,
      "effect_rms": 0.325011972060141,
      "reproduction_fidelity": 0.3409616556586922,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.7734375,
        "both_correct": 0.640625,
        "flip": 0.265625,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.71875,
            "both_correct": 0.578125,
            "flip": 0.28125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.65155029296875
      }
    },
    "h3_q_receiver": {
      "alignment": -0.0008022656278006673,
      "effect_rms": 0.020344401031249668,
      "reproduction_fidelity": -0.0021410232162568654,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58575439453125
      }
    },
    "h3_all_k": {
      "alignment": 0.000489518010183458,
      "effect_rms": 0.03778426195733028,
      "reproduction_fidelity": -0.0008714903969175847,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58544921875
      }
    },
    "h3_all_v": {
      "alignment": 0.23653856375910276,
      "effect_rms": 0.3215961322305121,
      "reproduction_fidelity": 0.33901841749705486,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.77734375,
        "both_correct": 0.640625,
        "flip": 0.2734375,
        "gap": 0.1328125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.7265625,
            "both_correct": 0.578125,
            "flip": 0.296875,
            "gap": 0.140625
          }
        },
        "gold_margin": 3.65252685546875
      }
    },
    "h3_weights": {
      "alignment": -0.0015686827144515358,
      "effect_rms": 0.037960530714017206,
      "reproduction_fidelity": -0.005005198038466663,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.586181640625
      }
    },
    "h3_all_qkv": {
      "alignment": 0.23854731822008285,
      "effect_rms": 0.3242716918371894,
      "reproduction_fidelity": 0.3407960106754483,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.7734375,
        "both_correct": 0.640625,
        "flip": 0.265625,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.71875,
            "both_correct": 0.578125,
            "flip": 0.28125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.65484619140625
      }
    },
    "h3_direct_v": {
      "alignment": 0.020033153719780608,
      "effect_rms": 0.10816945126970129,
      "reproduction_fidelity": 0.024899902220013592,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.59033203125
      }
    },
    "h3_direct_k": {
      "alignment": 0.003939631056703739,
      "effect_rms": 0.03092790551193968,
      "reproduction_fidelity": 0.00663939705226102,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5792236328125
      }
    },
    "h3_direct_kv": {
      "alignment": 0.0194027138581807,
      "effect_rms": 0.107355270544904,
      "reproduction_fidelity": 0.023866475309626267,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.586181640625
      }
    },
    "relay_kv": {
      "alignment": -0.00039680626583053037,
      "effect_rms": 0.01313830473347042,
      "reproduction_fidelity": -0.0010173568746993933,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58349609375
      }
    }
  },
  "relay_eligible": true,
  "limits": "Per-query GQA incoming-edge interventions; controlled additive edges are not full natural mediation. All recipes exploratory."
}

## Native chat short generation

{
  "metrics": {
    "baseline": {
      "strict": {
        "rows": 128,
        "pairs": 64,
        "accuracy": 0.8203125,
        "both_correct": 0.734375,
        "flip": 0.171875,
        "gap": 0.140625,
        "by_permutation": {
          "0": {
            "accuracy": 0.8203125,
            "both_correct": 0.734375,
            "flip": 0.171875,
            "gap": 0.140625
          }
        }
      },
      "unparsed": 0.0546875,
      "line": {
        "rows": 128,
        "pairs": 64,
        "accuracy": 0.8359375,
        "both_correct": 0.75,
        "flip": 0.171875,
        "gap": 0.140625,
        "by_permutation": {
          "0": {
            "accuracy": 0.8359375,
            "both_correct": 0.75,
            "flip": 0.171875,
            "gap": 0.140625
          }
        }
      }
    },
    "source_half": {
      "strict": {
        "rows": 128,
        "pairs": 64,
        "accuracy": 0.84375,
        "both_correct": 0.78125,
        "flip": 0.125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.84375,
            "both_correct": 0.78125,
            "flip": 0.125,
            "gap": 0.125
          }
        }
      },
      "unparsed": 0.0625,
      "line": {
        "rows": 128,
        "pairs": 64,
        "accuracy": 0.859375,
        "both_correct": 0.796875,
        "flip": 0.125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.859375,
            "both_correct": 0.796875,
            "flip": 0.125,
            "gap": 0.125
          }
        }
      }
    },
    "selected_half": {
      "strict": {
        "rows": 128,
        "pairs": 64,
        "accuracy": 0.796875,
        "both_correct": 0.71875,
        "flip": 0.15625,
        "gap": 0.15625,
        "by_permutation": {
          "0": {
            "accuracy": 0.796875,
            "both_correct": 0.71875,
            "flip": 0.15625,
            "gap": 0.15625
          }
        }
      },
      "unparsed": 0.0703125,
      "line": {
        "rows": 128,
        "pairs": 64,
        "accuracy": 0.8125,
        "both_correct": 0.734375,
        "flip": 0.15625,
        "gap": 0.15625,
        "by_permutation": {
          "0": {
            "accuracy": 0.8125,
            "both_correct": 0.734375,
            "flip": 0.15625,
            "gap": 0.15625
          }
        }
      }
    },
    "source_half_restore": {
      "strict": {
        "rows": 128,
        "pairs": 64,
        "accuracy": 0.828125,
        "both_correct": 0.75,
        "flip": 0.15625,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.75,
            "flip": 0.15625,
            "gap": 0.125
          }
        }
      },
      "unparsed": 0.0546875,
      "line": {
        "rows": 128,
        "pairs": 64,
        "accuracy": 0.84375,
        "both_correct": 0.765625,
        "flip": 0.15625,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.84375,
            "both_correct": 0.765625,
            "flip": 0.15625,
            "gap": 0.125
          }
        }
      }
    }
  },
  "pairs": 64,
  "status": "hash-selected historical native chat short answers; prefill-only edits"
}

{
  "source_half_vs_baseline": {
    "accuracy": {
      "delta": 0.0234375,
      "ci95": [
        -0.008064516129032258,
        0.056451612903225805
      ],
      "cluster": "family_id",
      "clusters": 61,
      "p_sign": 0.38665333666583357
    },
    "both_correct": {
      "delta": 0.046875,
      "ci95": [
        0.0,
        0.10606060606060606
      ],
      "cluster": "family_id",
      "clusters": 61,
      "p_sign": 0.2539365158710322
    },
    "flip": {
      "delta": -0.046875,
      "ci95": [
        -0.12121212121212122,
        0.016129032258064516
      ],
      "cluster": "family_id",
      "clusters": 61,
      "p_sign": 0.3839040239940015
    }
  },
  "selected_half_vs_baseline": {
    "accuracy": {
      "delta": -0.0234375,
      "ci95": [
        -0.056451612903225805,
        0.007936507936507936
      ],
      "cluster": "family_id",
      "clusters": 61,
      "p_sign": 0.38540364908772806
    },
    "both_correct": {
      "delta": -0.015625,
      "ci95": [
        -0.07352941176470588,
        0.03225806451612903
      ],
      "cluster": "family_id",
      "clusters": 61,
      "p_sign": 1.0
    },
    "flip": {
      "delta": -0.015625,
      "ci95": [
        -0.08196721311475409,
        0.046875
      ],
      "cluster": "family_id",
      "clusters": 61,
      "p_sign": 1.0
    }
  },
  "source_half_restore_vs_source_half": {
    "accuracy": {
      "delta": -0.015625,
      "ci95": [
        -0.047619047619047616,
        0.015384615384615385
      ],
      "cluster": "family_id",
      "clusters": 61,
      "p_sign": 0.6368407898025493
    },
    "both_correct": {
      "delta": -0.03125,
      "ci95": [
        -0.07936507936507936,
        0.0
      ],
      "cluster": "family_id",
      "clusters": 61,
      "p_sign": 0.5028742814296426
    },
    "flip": {
      "delta": 0.03125,
      "ci95": [
        -0.03076923076923077,
        0.09523809523809523
      ],
      "cluster": "family_id",
      "clusters": 61,
      "p_sign": 0.6393401649587603
    }
  }
}

## Independent audit

{
  "records_recomputed": 41024,
  "source_inputs_unchanged": true,
  "implementation_unchanged": true,
  "target_selection_forbidden": true,
  "historical_status": "all external behavior sets previously inspected; exploratory",
  "independent_selection_and_metric_recomputation": true,
  "independent_records_recomputed": 92632,
  "selection_sha256": "f2908257dc4427298788d4af9f1a6f016a26bde668fcbbde3daece87d2bb3390"
}

## Controls with active question-pronoun receiver

{
  "status": "supplementary historical; receiver present in every item; no reselection",
  "results": {
    "literal": {
      "metrics": {
        "baseline": {
          "rows": 256,
          "pairs": 64,
          "accuracy": 1.0,
          "both_correct": 1.0,
          "flip": 1.0,
          "gap": 0.0,
          "by_permutation": {
            "0": {
              "accuracy": 1.0,
              "both_correct": 1.0,
              "flip": 1.0,
              "gap": 0.0
            },
            "1": {
              "accuracy": 1.0,
              "both_correct": 1.0,
              "flip": 1.0,
              "gap": 0.0
            }
          },
          "gold_margin": 7.40264892578125
        },
        "source_half": {
          "rows": 256,
          "pairs": 64,
          "accuracy": 1.0,
          "both_correct": 1.0,
          "flip": 1.0,
          "gap": 0.0,
          "by_permutation": {
            "0": {
              "accuracy": 1.0,
              "both_correct": 1.0,
              "flip": 1.0,
              "gap": 0.0
            },
            "1": {
              "accuracy": 1.0,
              "both_correct": 1.0,
              "flip": 1.0,
              "gap": 0.0
            }
          },
          "gold_margin": 7.40289306640625
        },
        "selected_half": {
          "rows": 256,
          "pairs": 64,
          "accuracy": 1.0,
          "both_correct": 1.0,
          "flip": 1.0,
          "gap": 0.0,
          "by_permutation": {
            "0": {
              "accuracy": 1.0,
              "both_correct": 1.0,
              "flip": 1.0,
              "gap": 0.0
            },
            "1": {
              "accuracy": 1.0,
              "both_correct": 1.0,
              "flip": 1.0,
              "gap": 0.0
            }
          },
          "gold_margin": 6.06640625
        },
        "source_half_restore": {
          "rows": 256,
          "pairs": 64,
          "accuracy": 1.0,
          "both_correct": 1.0,
          "flip": 1.0,
          "gap": 0.0,
          "by_permutation": {
            "0": {
              "accuracy": 1.0,
              "both_correct": 1.0,
              "flip": 1.0,
              "gap": 0.0
            },
            "1": {
              "accuracy": 1.0,
              "both_correct": 1.0,
              "flip": 1.0,
              "gap": 0.0
            }
          },
          "gold_margin": 7.3826904296875
        },
        "selected_transplant": {
          "rows": 256,
          "pairs": 64,
          "accuracy": 1.0,
          "both_correct": 1.0,
          "flip": 1.0,
          "gap": 0.0,
          "by_permutation": {
            "0": {
              "accuracy": 1.0,
              "both_correct": 1.0,
              "flip": 1.0,
              "gap": 0.0
            },
            "1": {
              "accuracy": 1.0,
              "both_correct": 1.0,
              "flip": 1.0,
              "gap": 0.0
            }
          },
          "gold_margin": 7.42059326171875
        }
      },
      "transmission": {
        "fidelity": 0.49259395725865884,
        "restore_alignment": 0.47111274871039055,
        "transplant_alignment": 0.4513633014001474,
        "total_rms": 0.1137601005345583,
        "restore_rms": 0.07560006731625744,
        "transplant_rms": 0.0735605061924472,
        "restore_mean": -0.00262451171875,
        "transplant_mean": -0.0042724609375
      },
      "selected_half_vs_baseline": {
        "accuracy": {
          "delta": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "cluster": "family_id",
          "clusters": 61,
          "p_sign": 1.0
        },
        "both_correct": {
          "delta": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "cluster": "family_id",
          "clusters": 61,
          "p_sign": 1.0
        }
      },
      "question_receiver_present": 256
    },
    "reading": {
      "metrics": {
        "baseline": {
          "rows": 256,
          "pairs": 64,
          "accuracy": 0.8984375,
          "both_correct": 0.796875,
          "flip": 0.796875,
          "gap": 0.0625,
          "by_permutation": {
            "0": {
              "accuracy": 0.859375,
              "both_correct": 0.71875,
              "flip": 0.71875,
              "gap": 0.09375
            },
            "1": {
              "accuracy": 0.9375,
              "both_correct": 0.875,
              "flip": 0.875,
              "gap": 0.03125
            }
          },
          "gold_margin": 4.8702392578125
        },
        "source_half": {
          "rows": 256,
          "pairs": 64,
          "accuracy": 0.88671875,
          "both_correct": 0.7734375,
          "flip": 0.7734375,
          "gap": 0.0546875,
          "by_permutation": {
            "0": {
              "accuracy": 0.8359375,
              "both_correct": 0.671875,
              "flip": 0.671875,
              "gap": 0.078125
            },
            "1": {
              "accuracy": 0.9375,
              "both_correct": 0.875,
              "flip": 0.875,
              "gap": 0.03125
            }
          },
          "gold_margin": 4.80633544921875
        },
        "selected_half": {
          "rows": 256,
          "pairs": 64,
          "accuracy": 0.59375,
          "both_correct": 0.1875,
          "flip": 0.1875,
          "gap": 0.03125,
          "by_permutation": {
            "0": {
              "accuracy": 0.515625,
              "both_correct": 0.03125,
              "flip": 0.03125,
              "gap": 0.0
            },
            "1": {
              "accuracy": 0.671875,
              "both_correct": 0.34375,
              "flip": 0.34375,
              "gap": 0.0625
            }
          },
          "gold_margin": 2.8631591796875
        },
        "source_half_restore": {
          "rows": 256,
          "pairs": 64,
          "accuracy": 0.89453125,
          "both_correct": 0.7890625,
          "flip": 0.7890625,
          "gap": 0.0546875,
          "by_permutation": {
            "0": {
              "accuracy": 0.8515625,
              "both_correct": 0.703125,
              "flip": 0.703125,
              "gap": 0.078125
            },
            "1": {
              "accuracy": 0.9375,
              "both_correct": 0.875,
              "flip": 0.875,
              "gap": 0.03125
            }
          },
          "gold_margin": 4.7762451171875
        },
        "selected_transplant": {
          "rows": 256,
          "pairs": 64,
          "accuracy": 0.88671875,
          "both_correct": 0.7734375,
          "flip": 0.7734375,
          "gap": 0.0390625,
          "by_permutation": {
            "0": {
              "accuracy": 0.8359375,
              "both_correct": 0.671875,
              "flip": 0.671875,
              "gap": 0.046875
            },
            "1": {
              "accuracy": 0.9375,
              "both_correct": 0.875,
              "flip": 0.875,
              "gap": 0.03125
            }
          },
          "gold_margin": 4.89599609375
        }
      },
      "transmission": {
        "fidelity": 0.8433767035637283,
        "restore_alignment": 0.7515797021346519,
        "transplant_alignment": 0.7500211622823397,
        "total_rms": 0.4376362179819568,
        "restore_rms": 0.35552374445813767,
        "transplant_rms": 0.3545930577045546,
        "restore_mean": 0.06903076171875,
        "transplant_mean": 0.068603515625
      },
      "selected_half_vs_baseline": {
        "accuracy": {
          "delta": -0.3046875,
          "ci95": [
            -0.3387096774193548,
            -0.2734375
          ],
          "cluster": "family_id",
          "clusters": 61,
          "p_sign": 0.00024993751562109475
        },
        "both_correct": {
          "delta": -0.609375,
          "ci95": [
            -0.6774193548387096,
            -0.546875
          ],
          "cluster": "family_id",
          "clusters": 61,
          "p_sign": 0.00024993751562109475
        }
      },
      "question_receiver_present": 256
    },
    "binding": {
      "metrics": {
        "baseline": {
          "rows": 256,
          "pairs": 64,
          "accuracy": 0.98828125,
          "both_correct": 0.9765625,
          "flip": 0.9765625,
          "gap": 0.0078125,
          "by_permutation": {
            "0": {
              "accuracy": 0.984375,
              "both_correct": 0.96875,
              "flip": 0.96875,
              "gap": 0.0
            },
            "1": {
              "accuracy": 0.9921875,
              "both_correct": 0.984375,
              "flip": 0.984375,
              "gap": 0.015625
            }
          },
          "gold_margin": 4.68658447265625
        },
        "source_half": {
          "rows": 256,
          "pairs": 64,
          "accuracy": 0.99609375,
          "both_correct": 0.9921875,
          "flip": 0.9921875,
          "gap": 0.0078125,
          "by_permutation": {
            "0": {
              "accuracy": 1.0,
              "both_correct": 1.0,
              "flip": 1.0,
              "gap": 0.0
            },
            "1": {
              "accuracy": 0.9921875,
              "both_correct": 0.984375,
              "flip": 0.984375,
              "gap": 0.015625
            }
          },
          "gold_margin": 4.69683837890625
        },
        "selected_half": {
          "rows": 256,
          "pairs": 64,
          "accuracy": 0.98046875,
          "both_correct": 0.9609375,
          "flip": 0.9609375,
          "gap": 0.0390625,
          "by_permutation": {
            "0": {
              "accuracy": 0.984375,
              "both_correct": 0.96875,
              "flip": 0.96875,
              "gap": 0.03125
            },
            "1": {
              "accuracy": 0.9765625,
              "both_correct": 0.953125,
              "flip": 0.953125,
              "gap": 0.046875
            }
          },
          "gold_margin": 3.2799072265625
        },
        "source_half_restore": {
          "rows": 256,
          "pairs": 64,
          "accuracy": 0.98828125,
          "both_correct": 0.9765625,
          "flip": 0.9765625,
          "gap": 0.0078125,
          "by_permutation": {
            "0": {
              "accuracy": 0.984375,
              "both_correct": 0.96875,
              "flip": 0.96875,
              "gap": 0.0
            },
            "1": {
              "accuracy": 0.9921875,
              "both_correct": 0.984375,
              "flip": 0.984375,
              "gap": 0.015625
            }
          },
          "gold_margin": 4.65582275390625
        },
        "selected_transplant": {
          "rows": 256,
          "pairs": 64,
          "accuracy": 0.99609375,
          "both_correct": 0.9921875,
          "flip": 0.9921875,
          "gap": 0.0078125,
          "by_permutation": {
            "0": {
              "accuracy": 1.0,
              "both_correct": 1.0,
              "flip": 1.0,
              "gap": 0.0
            },
            "1": {
              "accuracy": 0.9921875,
              "both_correct": 0.984375,
              "flip": 0.984375,
              "gap": 0.015625
            }
          },
          "gold_margin": 4.7216796875
        }
      },
      "transmission": {
        "fidelity": 0.7947691085886956,
        "restore_alignment": 0.6805291907178553,
        "transplant_alignment": 0.6930132103978616,
        "total_rms": 0.3137973558167406,
        "restore_rms": 0.2383452782827625,
        "transplant_rms": 0.23911028615028482,
        "restore_mean": -0.0108642578125,
        "transplant_mean": -0.00933837890625
      },
      "selected_half_vs_baseline": {
        "accuracy": {
          "delta": -0.0078125,
          "ci95": [
            -0.027777777777777776,
            0.011363636363636364
          ],
          "cluster": "family_id",
          "clusters": 61,
          "p_sign": 0.683579105223694
        },
        "both_correct": {
          "delta": -0.015625,
          "ci95": [
            -0.05555555555555555,
            0.022727272727272728
          ],
          "cluster": "family_id",
          "clusters": 61,
          "p_sign": 0.683579105223694
        }
      },
      "question_receiver_present": 256
    },
    "record": {
      "metrics": {
        "baseline": {
          "rows": 512,
          "pairs": 128,
          "accuracy": 0.71484375,
          "both_correct": 0.640625,
          "flip": 0.1484375,
          "gap": -0.0546875,
          "by_permutation": {
            "0": {
              "accuracy": 0.703125,
              "both_correct": 0.6328125,
              "flip": 0.140625,
              "gap": -0.046875
            },
            "1": {
              "accuracy": 0.7265625,
              "both_correct": 0.6484375,
              "flip": 0.15625,
              "gap": -0.0625
            }
          },
          "gold_margin": 2.466949462890625
        },
        "source_half": {
          "rows": 512,
          "pairs": 128,
          "accuracy": 0.712890625,
          "both_correct": 0.63671875,
          "flip": 0.15234375,
          "gap": -0.04296875,
          "by_permutation": {
            "0": {
              "accuracy": 0.71875,
              "both_correct": 0.6484375,
              "flip": 0.140625,
              "gap": -0.046875
            },
            "1": {
              "accuracy": 0.70703125,
              "both_correct": 0.625,
              "flip": 0.1640625,
              "gap": -0.0390625
            }
          },
          "gold_margin": 2.528717041015625
        },
        "selected_half": {
          "rows": 512,
          "pairs": 128,
          "accuracy": 0.744140625,
          "both_correct": 0.69140625,
          "flip": 0.10546875,
          "gap": -0.02734375,
          "by_permutation": {
            "0": {
              "accuracy": 0.74609375,
              "both_correct": 0.6796875,
              "flip": 0.1328125,
              "gap": -0.0390625
            },
            "1": {
              "accuracy": 0.7421875,
              "both_correct": 0.703125,
              "flip": 0.078125,
              "gap": -0.015625
            }
          },
          "gold_margin": 1.939483642578125
        },
        "source_half_restore": {
          "rows": 512,
          "pairs": 128,
          "accuracy": 0.71875,
          "both_correct": 0.65234375,
          "flip": 0.1328125,
          "gap": -0.03125,
          "by_permutation": {
            "0": {
              "accuracy": 0.7109375,
              "both_correct": 0.640625,
              "flip": 0.140625,
              "gap": -0.03125
            },
            "1": {
              "accuracy": 0.7265625,
              "both_correct": 0.6640625,
              "flip": 0.125,
              "gap": -0.03125
            }
          },
          "gold_margin": 2.488922119140625
        },
        "selected_transplant": {
          "rows": 512,
          "pairs": 128,
          "accuracy": 0.70703125,
          "both_correct": 0.6328125,
          "flip": 0.1484375,
          "gap": -0.046875,
          "by_permutation": {
            "0": {
              "accuracy": 0.70703125,
              "both_correct": 0.6328125,
              "flip": 0.1484375,
              "gap": -0.0703125
            },
            "1": {
              "accuracy": 0.70703125,
              "both_correct": 0.6328125,
              "flip": 0.1484375,
              "gap": -0.0234375
            }
          },
          "gold_margin": 2.511199951171875
        }
      },
      "transmission": {
        "fidelity": 0.9126704978049927,
        "restore_alignment": 0.7952902512513972,
        "transplant_alignment": 0.7999009816785732,
        "total_rms": 0.6862726526742806,
        "restore_rms": 0.565775342318905,
        "transplant_rms": 0.5681479149153469,
        "restore_mean": -0.0469970703125,
        "transplant_mean": -0.04815673828125
      },
      "selected_half_vs_baseline": {
        "accuracy": {
          "delta": 0.029296875,
          "ci95": [
            0.005952380952380952,
            0.05303030303030303
          ],
          "cluster": "family_id",
          "clusters": 61,
          "p_sign": 0.026743314171457137
        },
        "both_correct": {
          "delta": 0.05078125,
          "ci95": [
            0.015873015873015872,
            0.08955223880597014
          ],
          "cluster": "family_id",
          "clusters": 61,
          "p_sign": 0.01374656335916021
        }
      },
      "question_receiver_present": 512,
      "record_evidence_transmission": {
        "estimate": {
          "fidelity": 0.7610060361086959,
          "restore_alignment": 0.534762615260258,
          "transplant_alignment": 0.5469247962661502,
          "total_rms": 1.0295954059637646,
          "restore_rms": 0.5765024045313427,
          "transplant_rms": 0.5895186785326375,
          "restore_mean": -0.016204833984375,
          "transplant_mean": -0.01708984375
        },
        "bootstrap": {
          "clusters": 61,
          "cluster": "family_id",
          "ci95": {
            "fidelity": [
              0.7242872719757436,
              0.7918007820530152
            ],
            "restore_alignment": [
              0.49866532351588166,
              0.566204609393098
            ],
            "transplant_alignment": [
              0.5101370023780317,
              0.5786050468271625
            ]
          }
        }
      }
    }
  }
}

## Fixed token-group transmission

{
  "heads": [
    {
      "layer": 21,
      "position": "decision",
      "head": 1
    },
    {
      "layer": 17,
      "position": "decision",
      "head": 1
    }
  ],
  "groups": [
    "context_prefix",
    "context_pronoun",
    "context_suffix",
    "question_pronoun",
    "question_other",
    "option_A",
    "option_B",
    "answer_marker",
    "format_other"
  ],
  "results": {
    "baseline": {
      "alignment": 0.0,
      "effect_rms": 0.0,
      "reproduction_fidelity": 0.0,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58428955078125
      }
    },
    "source_half": {
      "alignment": 1.0,
      "effect_rms": 0.8783412897761801,
      "reproduction_fidelity": 1.0,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.79296875,
        "both_correct": 0.6953125,
        "flip": 0.1953125,
        "gap": 0.1015625,
        "by_permutation": {
          "0": {
            "accuracy": 0.8359375,
            "both_correct": 0.75,
            "flip": 0.171875,
            "gap": 0.078125
          },
          "1": {
            "accuracy": 0.75,
            "both_correct": 0.640625,
            "flip": 0.21875,
            "gap": 0.125
          }
        },
        "gold_margin": 3.7357177734375
      }
    },
    "h0_k_context_prefix": {
      "alignment": 0.0,
      "effect_rms": 0.0,
      "reproduction_fidelity": 0.0,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58428955078125
      }
    },
    "h0_v_context_prefix": {
      "alignment": 0.0,
      "effect_rms": 0.0,
      "reproduction_fidelity": 0.0,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58428955078125
      }
    },
    "h0_k_context_pronoun": {
      "alignment": -0.0009159920342069252,
      "effect_rms": 0.013389950391408289,
      "reproduction_fidelity": -0.002064381507591806,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58355712890625
      }
    },
    "h0_v_context_pronoun": {
      "alignment": -0.0012015442068139424,
      "effect_rms": 0.01284467425582608,
      "reproduction_fidelity": -0.002616943503935376,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58349609375
      }
    },
    "h0_k_context_suffix": {
      "alignment": -0.0006959561609426436,
      "effect_rms": 0.012467915366019242,
      "reproduction_fidelity": -0.0015934058462789835,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5853271484375
      }
    },
    "h0_v_context_suffix": {
      "alignment": -0.002277000441307903,
      "effect_rms": 0.01295556558732578,
      "reproduction_fidelity": -0.004771564442697329,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58441162109375
      }
    },
    "h0_k_question_pronoun": {
      "alignment": -0.0028048393058239043,
      "effect_rms": 0.014938533731228863,
      "reproduction_fidelity": -0.0058989392540289565,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58160400390625
      }
    },
    "h0_v_question_pronoun": {
      "alignment": -0.0017800654915762109,
      "effect_rms": 0.0126953125,
      "reproduction_fidelity": -0.0037690414470943345,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5841064453125
      }
    },
    "h0_k_question_other": {
      "alignment": -0.001713313035642103,
      "effect_rms": 0.011759369705851851,
      "reproduction_fidelity": -0.003605868777033283,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.583251953125
      }
    },
    "h0_v_question_other": {
      "alignment": -0.0013276321791339239,
      "effect_rms": 0.012158105076160872,
      "reproduction_fidelity": -0.0028468686299305546,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.584228515625
      }
    },
    "h0_k_option_A": {
      "alignment": -0.0014153992971213621,
      "effect_rms": 0.013845162967536939,
      "reproduction_fidelity": -0.003079266069108577,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5821533203125
      }
    },
    "h0_v_option_A": {
      "alignment": -0.0003424153758101461,
      "effect_rms": 0.013028968812623373,
      "reproduction_fidelity": -0.0009048666248845372,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58282470703125
      }
    },
    "h0_k_option_B": {
      "alignment": 0.05939732421544232,
      "effect_rms": 0.09128117277095757,
      "reproduction_fidelity": 0.10799434829206422,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.6011962890625
      }
    },
    "h0_v_option_B": {
      "alignment": -0.002712127561470977,
      "effect_rms": 0.018885819927552457,
      "reproduction_fidelity": -0.00588657768811518,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.57952880859375
      }
    },
    "h0_k_answer_marker": {
      "alignment": 0.06912464043295148,
      "effect_rms": 0.09969550504203255,
      "reproduction_fidelity": 0.1253660568706202,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.61578369140625
      }
    },
    "h0_v_answer_marker": {
      "alignment": 0.0043401457923083865,
      "effect_rms": 0.023233158697988903,
      "reproduction_fidelity": 0.007980626953900005,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58343505859375
      }
    },
    "h0_k_format_other": {
      "alignment": 0.0,
      "effect_rms": 0.0,
      "reproduction_fidelity": 0.0,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58428955078125
      }
    },
    "h0_v_format_other": {
      "alignment": 0.0,
      "effect_rms": 0.0,
      "reproduction_fidelity": 0.0,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58428955078125
      }
    },
    "h1_k_context_prefix": {
      "alignment": 0.0,
      "effect_rms": 0.0,
      "reproduction_fidelity": 0.0,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58428955078125
      }
    },
    "h1_v_context_prefix": {
      "alignment": 0.0,
      "effect_rms": 0.0,
      "reproduction_fidelity": 0.0,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58428955078125
      }
    },
    "h1_k_context_pronoun": {
      "alignment": -0.0007763063393818475,
      "effect_rms": 0.0146484375,
      "reproduction_fidelity": -0.0018307479118224723,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5838623046875
      }
    },
    "h1_v_context_pronoun": {
      "alignment": 0.0027195445010192113,
      "effect_rms": 0.018295892573432806,
      "reproduction_fidelity": 0.005005198038466774,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5860595703125
      }
    },
    "h1_k_context_suffix": {
      "alignment": -0.002162037878310273,
      "effect_rms": 0.019604355370981184,
      "reproduction_fidelity": -0.0048222468629435244,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5830078125
      }
    },
    "h1_v_context_suffix": {
      "alignment": -0.0009073389380673186,
      "effect_rms": 0.0171107572929055,
      "reproduction_fidelity": -0.002194177949685905,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.582763671875
      }
    },
    "h1_k_question_pronoun": {
      "alignment": -0.0006156059825034396,
      "effect_rms": 0.016773011755193034,
      "reproduction_fidelity": -0.0015958781594618277,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5850830078125
      }
    },
    "h1_v_question_pronoun": {
      "alignment": 0.01225896491663978,
      "effect_rms": 0.023396774513107302,
      "reproduction_fidelity": 0.023808375949831873,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58660888671875
      }
    },
    "h1_k_question_other": {
      "alignment": -0.001237392747963741,
      "effect_rms": 0.01648627247669349,
      "reproduction_fidelity": -0.0028270901244686897,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5831298828125
      }
    },
    "h1_v_question_other": {
      "alignment": 0.00233262748791966,
      "effect_rms": 0.01568591641036036,
      "reproduction_fidelity": 0.004346326575265236,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58343505859375
      }
    },
    "h1_k_option_A": {
      "alignment": 0.022955427901784887,
      "effect_rms": 0.042421497322213814,
      "reproduction_fidelity": 0.04357822831565006,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5859375
      }
    },
    "h1_v_option_A": {
      "alignment": -0.0017293830713299438,
      "effect_rms": 0.023109686652732876,
      "reproduction_fidelity": -0.004151013833828321,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.57989501953125
      }
    },
    "h1_k_option_B": {
      "alignment": 0.04963539561337472,
      "effect_rms": 0.05774945094520515,
      "reproduction_fidelity": 0.09494795162672032,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5870361328125
      }
    },
    "h1_v_option_B": {
      "alignment": 0.0026577366714505926,
      "effect_rms": 0.03072682172373491,
      "reproduction_fidelity": 0.004091678317442504,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58026123046875
      }
    },
    "h1_k_answer_marker": {
      "alignment": 0.10466043396513297,
      "effect_rms": 0.11709387536063813,
      "reproduction_fidelity": 0.19154864461610543,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.8203125,
            "both_correct": 0.703125,
            "flip": 0.234375,
            "gap": 0.109375
          },
          "1": {
            "accuracy": 0.7109375,
            "both_correct": 0.546875,
            "flip": 0.328125,
            "gap": 0.140625
          }
        },
        "gold_margin": 3.60498046875
      }
    },
    "h1_v_answer_marker": {
      "alignment": 0.018538640400811414,
      "effect_rms": 0.05974615440602511,
      "reproduction_fidelity": 0.03245034668011604,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5809326171875
      }
    },
    "h1_k_format_other": {
      "alignment": 0.0,
      "effect_rms": 0.0,
      "reproduction_fidelity": 0.0,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58428955078125
      }
    },
    "h1_v_format_other": {
      "alignment": 0.0,
      "effect_rms": 0.0,
      "reproduction_fidelity": 0.0,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58428955078125
      }
    }
  },
  "status": "historical fixed groups; all unadjusted, no reselection; score effects are not additive"
}

## Post-QKV exploratory receiver edges

{
  "heads": [
    {
      "layer": 21,
      "position": "decision",
      "head": 1
    },
    {
      "layer": 17,
      "position": "decision",
      "head": 1
    },
    {
      "layer": 13,
      "position": "question_identity",
      "head": 6
    },
    {
      "layer": 11,
      "position": "question_identity",
      "head": 13
    }
  ],
  "pairs": [
    [
      1,
      0
    ],
    [
      2,
      0
    ],
    [
      2,
      1
    ],
    [
      3,
      0
    ],
    [
      3,
      1
    ],
    [
      3,
      2
    ]
  ],
  "results": {
    "baseline": {
      "alignment": 0.0,
      "effect_rms": 0.0,
      "reproduction_fidelity": 0.0,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58428955078125
      }
    },
    "source_half": {
      "alignment": 1.0,
      "effect_rms": 0.8783412897761801,
      "reproduction_fidelity": 1.0,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.79296875,
        "both_correct": 0.6953125,
        "flip": 0.1953125,
        "gap": 0.1015625,
        "by_permutation": {
          "0": {
            "accuracy": 0.8359375,
            "both_correct": 0.75,
            "flip": 0.171875,
            "gap": 0.078125
          },
          "1": {
            "accuracy": 0.75,
            "both_correct": 0.640625,
            "flip": 0.21875,
            "gap": 0.125
          }
        },
        "gold_margin": 3.7357177734375
      }
    },
    "h1_to_h0_kv": {
      "alignment": -0.001605767412192707,
      "effect_rms": 0.012467915366019242,
      "reproduction_fidelity": -0.0034130283487792124,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5843505859375
      }
    },
    "h1_to_h0_q": {
      "alignment": 0.04259424766891771,
      "effect_rms": 0.04551242264738689,
      "reproduction_fidelity": 0.08250356322137464,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.59283447265625
      }
    },
    "h2_to_h0_kv": {
      "alignment": -0.00039680626583053037,
      "effect_rms": 0.01313830473347042,
      "reproduction_fidelity": -0.0010173568746993933,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58349609375
      }
    },
    "h2_to_h1_kv": {
      "alignment": 0.00605345882795049,
      "effect_rms": 0.01648627247669349,
      "reproduction_fidelity": 0.011754613027359806,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5843505859375
      }
    },
    "h3_to_h0_kv": {
      "alignment": -0.0014475393684970437,
      "effect_rms": 0.012807497117777344,
      "reproduction_fidelity": -0.0031076976707100634,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58465576171875
      }
    },
    "h3_to_h1_kv": {
      "alignment": -2.101466205333027e-05,
      "effect_rms": 0.013810679320049757,
      "reproduction_fidelity": -0.0002892606423812172,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58428955078125
      }
    },
    "h3_to_h2_kv": {
      "alignment": 0.010221778854058117,
      "effect_rms": 0.028117539157984314,
      "reproduction_fidelity": 0.01941878389386853,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.58935546875
      }
    },
    "h3_to_h2_q": {
      "alignment": 0.01791438132216837,
      "effect_rms": 0.04474105933185667,
      "reproduction_fidelity": 0.0332340699590461,
      "metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.765625,
        "both_correct": 0.625,
        "flip": 0.28125,
        "gap": 0.125,
        "by_permutation": {
          "0": {
            "accuracy": 0.828125,
            "both_correct": 0.703125,
            "flip": 0.25,
            "gap": 0.125
          },
          "1": {
            "accuracy": 0.703125,
            "both_correct": 0.546875,
            "flip": 0.3125,
            "gap": 0.125
          }
        },
        "gold_margin": 3.5906982421875
      }
    }
  },
  "status": "post-QKV exploratory additive edges; no necessity/completeness claim"
}

## Subset documentation clarification

{
  "created": "2026-10-15 17:16:26",
  "status": "documentation clarification; implementation and chosen subsets unchanged",
  "detail": "The controls/generation subsets are chosen by deterministic pair_id hash, not family_id hash. Main rank/screen/dev/historical partitions remain family-isolated; all uncertainty clusters by family_id. Clarified after selection while target inference was starting, before inspection of target outcomes; chosen subsets unchanged."
}


Final fingerprint/numerical validation:

{
  "records_recomputed": 41024,
  "source_inputs_unchanged": true,
  "implementation_unchanged": true,
  "target_selection_forbidden": true,
  "historical_status": "all external behavior sets previously inspected; exploratory",
  "independent_selection_and_metric_recomputation": true,
  "independent_records_recomputed": 92632,
  "selection_sha256": "f2908257dc4427298788d4af9f1a6f016a26bde668fcbbde3daece87d2bb3390",
  "supplement_source_hashes_match": {
    "active_controls": true,
    "token_routes": true,
    "receiver_edges": true,
    "qkv_validation": true
  },
  "manual_QKV_node_patch_equivalent": true,
  "manual_QKV_max_logit_error": 0.0,
  "manual_QKV_validated_rows_times_heads": 1024
}
