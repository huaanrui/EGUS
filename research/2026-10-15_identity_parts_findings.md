# Signed parts and local reprocessing results

All external behavior sets had been inspected previously. This is exploratory. Source-effect sign and identity-gap sign are distinct; neither inherently defines a bias neuron.

## Frozen selections

{
  "overall": {
    "name": "combo_source_terminal",
    "edits": [
      {
        "type": "unit",
        "site": "mlp9_identity",
        "mode": "centered",
        "groups": [
          [
            "gap_pos",
            0.5
          ]
        ],
        "group": "whole",
        "amount": 0.5
      },
      {
        "type": "unit",
        "site": "head21_1_decision",
        "mode": "centered",
        "groups": [
          [
            "gap_pos",
            -0.5
          ],
          [
            "gap_neg",
            0.5
          ]
        ],
        "group": "whole",
        "amount": 0.5
      }
    ],
    "category": "combo",
    "site": "combination",
    "burden": 0.13020833333333334
  },
  "unit": {
    "name": "mlp9_identity_gap_pos_damp_0.5",
    "edits": [
      {
        "type": "unit",
        "site": "mlp9_identity",
        "mode": "centered",
        "groups": [
          [
            "gap_pos",
            0.5
          ]
        ],
        "group": "whole",
        "amount": 0.5
      }
    ],
    "category": "unit",
    "site": "mlp9_identity",
    "burden": 0.005208333333333333
  },
  "qk": {
    "name": "q21_decision_amplify",
    "edits": [
      {
        "type": "qk",
        "field": "q21_decision",
        "rank": 1,
        "amount": -0.25,
        "preserve": true,
        "random": false
      }
    ],
    "category": "qk",
    "site": "q21_decision",
    "burden": 0.001953125
  },
  "combo": {
    "name": "combo_source_terminal",
    "edits": [
      {
        "type": "unit",
        "site": "mlp9_identity",
        "mode": "centered",
        "groups": [
          [
            "gap_pos",
            0.5
          ]
        ],
        "group": "whole",
        "amount": 0.5
      },
      {
        "type": "unit",
        "site": "head21_1_decision",
        "mode": "centered",
        "groups": [
          [
            "gap_pos",
            -0.5
          ],
          [
            "gap_neg",
            0.5
          ]
        ],
        "group": "whole",
        "amount": 0.5
      }
    ],
    "category": "combo",
    "site": "combination",
    "burden": 0.13020833333333334
  }
}

## Main coreference retests

|Dataset|Method|Accuracy|Both correct|Flip|Margin gap RMS|
|---|---|---:|---:|---:|---:|
|historical|baseline|76.71%|65.75%|21.92%|3.660|
|historical|source_half|79.11%|70.21%|17.81%|3.306|
|historical|whole_four_heads|71.75%|60.27%|22.95%|2.263|
|historical|overall|81.68%|72.60%|18.15%|3.620|
|historical|unit|79.62%|69.52%|20.21%|3.318|
|historical|qk|76.54%|65.41%|22.26%|3.680|
|historical|combo|81.68%|72.60%|18.15%|3.620|
|new_type1|baseline|57.50%|41.56%|31.87%|5.739|
|new_type1|source_half|58.13%|43.12%|30.00%|5.530|
|new_type1|whole_four_heads|58.59%|43.75%|29.69%|3.908|
|new_type1|overall|59.53%|42.50%|34.06%|6.501|
|new_type1|unit|58.44%|41.56%|33.75%|5.966|
|new_type1|qk|57.50%|41.56%|31.87%|5.764|
|new_type1|combo|59.53%|42.50%|34.06%|6.501|
|winogender|baseline|57.71%|48.54%|18.33%|4.069|
|winogender|source_half|57.81%|48.12%|19.38%|3.857|
|winogender|whole_four_heads|56.88%|47.92%|17.92%|2.733|
|winogender|overall|58.65%|48.54%|20.21%|4.207|
|winogender|unit|57.81%|48.12%|19.38%|3.880|
|winogender|qk|57.81%|48.54%|18.54%|4.080|
|winogender|combo|58.65%|48.54%|20.21%|4.207|

## Behavior uncertainty vs old L7

{
  "historical": {
    "overall": {
      "accuracy": {
        "delta": 0.025684931506849314,
        "ci95": [
          0.007296644436275797,
          0.044964928057553956
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.014496375906023495
      },
      "both_correct": {
        "delta": 0.023972602739726026,
        "ci95": [
          -0.003472222222222222,
          0.0528169014084507
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.16795801049737566
      },
      "flip": {
        "delta": 0.003424657534246575,
        "ci95": [
          -0.028368794326241134,
          0.03793103448275862
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 1.0
      }
    },
    "unit": {
      "accuracy": {
        "delta": 0.005136986301369863,
        "ci95": [
          -0.011668624161073825,
          0.022887323943661973
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.7030742314421394
      },
      "both_correct": {
        "delta": -0.00684931506849315,
        "ci95": [
          -0.03356271539996372,
          0.02027027027027027
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.7970507373156711
      },
      "flip": {
        "delta": 0.023972602739726026,
        "ci95": [
          -0.00684931506849315,
          0.05629255666449026
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.19645088727818044
      }
    },
    "qk": {
      "accuracy": {
        "delta": -0.025684931506849314,
        "ci95": [
          -0.043921298058622,
          -0.008620689655172414
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.011247188202949263
      },
      "both_correct": {
        "delta": -0.04794520547945205,
        "ci95": [
          -0.07643895966029723,
          -0.020833333333333332
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.0022494376405898524
      },
      "flip": {
        "delta": 0.04452054794520548,
        "ci95": [
          0.013422818791946308,
          0.07692953020134231
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.01324668832791802
      }
    },
    "combo": {
      "accuracy": {
        "delta": 0.025684931506849314,
        "ci95": [
          0.007296644436275797,
          0.044964928057553956
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.014496375906023495
      },
      "both_correct": {
        "delta": 0.023972602739726026,
        "ci95": [
          -0.003472222222222222,
          0.0528169014084507
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.16795801049737566
      },
      "flip": {
        "delta": 0.003424657534246575,
        "ci95": [
          -0.028368794326241134,
          0.03793103448275862
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 1.0
      }
    }
  },
  "new_type1": {
    "overall": {
      "accuracy": {
        "delta": 0.0140625,
        "ci95": [
          -0.01875,
          0.0484375
        ],
        "cluster": "family_id",
        "clusters": 160,
        "p_sign": 0.47813046738315423
      },
      "both_correct": {
        "delta": -0.00625,
        "ci95": [
          -0.046875,
          0.034375
        ],
        "cluster": "family_id",
        "clusters": 160,
        "p_sign": 0.8900274931267184
      },
      "flip": {
        "delta": 0.040625,
        "ci95": [
          -0.0125,
          0.096875
        ],
        "cluster": "family_id",
        "clusters": 160,
        "p_sign": 0.1802049487628093
      }
    },
    "unit": {
      "accuracy": {
        "delta": 0.003125,
        "ci95": [
          -0.0296875,
          0.0359375
        ],
        "cluster": "family_id",
        "clusters": 160,
        "p_sign": 0.9267683079230192
      },
      "both_correct": {
        "delta": -0.015625,
        "ci95": [
          -0.05625,
          0.025
        ],
        "cluster": "family_id",
        "clusters": 160,
        "p_sign": 0.5538615346163459
      },
      "flip": {
        "delta": 0.0375,
        "ci95": [
          -0.0125,
          0.090625
        ],
        "cluster": "family_id",
        "clusters": 160,
        "p_sign": 0.20669832541864533
      }
    },
    "qk": {
      "accuracy": {
        "delta": -0.00625,
        "ci95": [
          -0.0265625,
          0.0140625
        ],
        "cluster": "family_id",
        "clusters": 160,
        "p_sign": 0.6513371657085728
      },
      "both_correct": {
        "delta": -0.015625,
        "ci95": [
          -0.04375,
          0.009375
        ],
        "cluster": "family_id",
        "clusters": 160,
        "p_sign": 0.37265683579105224
      },
      "flip": {
        "delta": 0.01875,
        "ci95": [
          -0.015625,
          0.053125
        ],
        "cluster": "family_id",
        "clusters": 160,
        "p_sign": 0.3841539615096226
      }
    },
    "combo": {
      "accuracy": {
        "delta": 0.0140625,
        "ci95": [
          -0.01875,
          0.0484375
        ],
        "cluster": "family_id",
        "clusters": 160,
        "p_sign": 0.47813046738315423
      },
      "both_correct": {
        "delta": -0.00625,
        "ci95": [
          -0.046875,
          0.034375
        ],
        "cluster": "family_id",
        "clusters": 160,
        "p_sign": 0.8900274931267184
      },
      "flip": {
        "delta": 0.040625,
        "ci95": [
          -0.0125,
          0.096875
        ],
        "cluster": "family_id",
        "clusters": 160,
        "p_sign": 0.1802049487628093
      }
    }
  },
  "winogender": {
    "overall": {
      "accuracy": {
        "delta": 0.008333333333333333,
        "ci95": [
          -0.010416666666666666,
          0.027083333333333334
        ],
        "cluster": "family_id",
        "clusters": 120,
        "p_sign": 0.4651337165708573
      },
      "both_correct": {
        "delta": 0.004166666666666667,
        "ci95": [
          -0.016666666666666666,
          0.025
        ],
        "cluster": "family_id",
        "clusters": 120,
        "p_sign": 0.8545363659085229
      },
      "flip": {
        "delta": 0.008333333333333333,
        "ci95": [
          -0.01875,
          0.035416666666666666
        ],
        "cluster": "family_id",
        "clusters": 120,
        "p_sign": 0.6415896025993502
      }
    },
    "unit": {
      "accuracy": {
        "delta": 0.0,
        "ci95": [
          -0.01875,
          0.01875
        ],
        "cluster": "family_id",
        "clusters": 120,
        "p_sign": 1.0
      },
      "both_correct": {
        "delta": 0.0,
        "ci95": [
          -0.020833333333333332,
          0.022916666666666665
        ],
        "cluster": "family_id",
        "clusters": 120,
        "p_sign": 1.0
      },
      "flip": {
        "delta": 0.0,
        "ci95": [
          -0.027083333333333334,
          0.025
        ],
        "cluster": "family_id",
        "clusters": 120,
        "p_sign": 1.0
      }
    },
    "qk": {
      "accuracy": {
        "delta": 0.0,
        "ci95": [
          -0.015625,
          0.016666666666666666
        ],
        "cluster": "family_id",
        "clusters": 120,
        "p_sign": 1.0
      },
      "both_correct": {
        "delta": 0.004166666666666667,
        "ci95": [
          -0.014583333333333334,
          0.022916666666666665
        ],
        "cluster": "family_id",
        "clusters": 120,
        "p_sign": 0.8252936765808548
      },
      "flip": {
        "delta": -0.008333333333333333,
        "ci95": [
          -0.03333333333333333,
          0.016666666666666666
        ],
        "cluster": "family_id",
        "clusters": 120,
        "p_sign": 0.6323419145213697
      }
    },
    "combo": {
      "accuracy": {
        "delta": 0.008333333333333333,
        "ci95": [
          -0.010416666666666666,
          0.027083333333333334
        ],
        "cluster": "family_id",
        "clusters": 120,
        "p_sign": 0.4651337165708573
      },
      "both_correct": {
        "delta": 0.004166666666666667,
        "ci95": [
          -0.016666666666666666,
          0.025
        ],
        "cluster": "family_id",
        "clusters": 120,
        "p_sign": 0.8545363659085229
      },
      "flip": {
        "delta": 0.008333333333333333,
        "ci95": [
          -0.01875,
          0.035416666666666666
        ],
        "cluster": "family_id",
        "clusters": 120,
        "p_sign": 0.6415896025993502
      }
    }
  }
}

## Exact signed effects

{
  "mlp9_identity_transfer_pos": {
    "alignment": 0.38419190589387103,
    "ci95": [
      0.2596870297264051,
      0.47493445411641577
    ],
    "clusters": 61,
    "p_sign": 0.00024993751562109475,
    "largest_family_energy_share": 0.15976582249533042,
    "p_holm": 0.0024993751562109477
  },
  "mlp9_identity_transfer_neg": {
    "alignment": -0.10483720435769922,
    "ci95": [
      -0.14344249979715648,
      -0.05972931447085745
    ],
    "clusters": 61,
    "p_sign": 0.0004998750312421895,
    "largest_family_energy_share": 0.15976582249533042,
    "p_holm": 0.0024993751562109477
  },
  "mlp18_decision_transfer_pos": {
    "alignment": 0.2877463505567031,
    "ci95": [
      0.23876313283495385,
      0.3321146292824349
    ],
    "clusters": 61,
    "p_sign": 0.00024993751562109475,
    "largest_family_energy_share": 0.15976582249533042,
    "p_holm": 0.0024993751562109477
  },
  "mlp18_decision_transfer_neg": {
    "alignment": -0.10169427622413497,
    "ci95": [
      -0.13536267568962876,
      -0.06919748879832739
    ],
    "clusters": 61,
    "p_sign": 0.00024993751562109475,
    "largest_family_energy_share": 0.15976582249533042,
    "p_holm": 0.0024993751562109477
  },
  "head17_1_decision_transfer_pos": {
    "alignment": 0.09377298478662083,
    "ci95": [
      0.07474177838793913,
      0.11447041717236227
    ],
    "clusters": 61,
    "p_sign": 0.00024993751562109475,
    "largest_family_energy_share": 0.15976582249533042,
    "p_holm": 0.0024993751562109477
  },
  "head17_1_decision_transfer_neg": {
    "alignment": -0.014943278954804878,
    "ci95": [
      -0.026623688926920942,
      -0.002781797448050226
    ],
    "clusters": 61,
    "p_sign": 0.02324418895276181,
    "largest_family_energy_share": 0.15976582249533042,
    "p_holm": 0.02324418895276181
  },
  "head21_1_decision_transfer_pos": {
    "alignment": 0.23440495748239404,
    "ci95": [
      0.1983472566275217,
      0.2666301195591376
    ],
    "clusters": 61,
    "p_sign": 0.00024993751562109475,
    "largest_family_energy_share": 0.15976582249533042,
    "p_holm": 0.0024993751562109477
  },
  "head21_1_decision_transfer_neg": {
    "alignment": -0.010264426256460464,
    "ci95": [
      -0.012749943078779568,
      -0.008337207412055132
    ],
    "clusters": 61,
    "p_sign": 0.00024993751562109475,
    "largest_family_energy_share": 0.15976582249533042,
    "p_holm": 0.0024993751562109477
  },
  "attention18_heads_positive": {
    "alignment": 0.0831179330472867,
    "ci95": [
      0.0631506364820782,
      0.10163836301910399
    ],
    "clusters": 61,
    "p_sign": 0.00024993751562109475,
    "largest_family_energy_share": 0.15976582249533042,
    "p_holm": 0.0024993751562109477
  },
  "attention18_heads_negative": {
    "alignment": -0.17294943254231673,
    "ci95": [
      -0.20599261359523585,
      -0.1434920796539547
    ],
    "clusters": 61,
    "p_sign": 0.00024993751562109475,
    "largest_family_energy_share": 0.15976582249533042,
    "p_holm": 0.0024993751562109477
  }
}

## Local and destination localization

{
  "results": [
    {
      "index": 0,
      "site": {
        "layer": 8,
        "module": "attention",
        "position": "identity"
      },
      "transmission": {
        "fidelity": 0.16652282328903967,
        "restore_alignment": 0.15503128907989724,
        "transplant_alignment": 0.13030700286596397,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.2984097612681549,
        "transplant_rms": 0.3017973148699228,
        "restore_mean": -0.0025634765625,
        "transplant_mean": -0.0069580078125
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.08337381610789074,
          "restore_alignment": 0.10401619664652548,
          "transplant_alignment": 0.09005959390455533,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.09447319556150979,
          "restore_alignment": 0.11387389926989508,
          "transplant_alignment": 0.10925841440586875,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.10405486978780765,
          "restore_alignment": 0.11258508479826072,
          "transplant_alignment": 0.1007946530895493,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.12760364415408532,
          "restore_alignment": 0.12927344140694214,
          "transplant_alignment": 0.11792803394751827,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.11508704271329595,
          "restore_alignment": 0.12619767602069953,
          "transplant_alignment": 0.11296697453721719,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.08564851885124225,
          "restore_alignment": 0.1124392709568674,
          "transplant_alignment": 0.10608689893815355,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.08143295298350184,
          "restore_alignment": 0.10699316627945972,
          "transplant_alignment": 0.09902573578999968,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.08816352936920357,
          "restore_alignment": 0.09716368502399658,
          "transplant_alignment": 0.09408835048102476,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": 0.14266914597293062
    },
    {
      "index": 1,
      "site": {
        "layer": 8,
        "module": "mlp",
        "position": "identity"
      },
      "transmission": {
        "fidelity": -0.012553375119205512,
        "restore_alignment": 0.268613039507024,
        "transplant_alignment": 0.27653658696149946,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.6804349450737661,
        "transplant_rms": 0.6184763910325506,
        "restore_mean": -0.0626220703125,
        "transplant_mean": -0.0140380859375
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.2029747024859242,
          "restore_alignment": 0.36631910923406436,
          "transplant_alignment": 0.3748470632601665,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.13061336390737288,
          "restore_alignment": 0.35127535668366444,
          "transplant_alignment": 0.3498358945180525,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.17129733485280074,
          "restore_alignment": 0.34826929962992315,
          "transplant_alignment": 0.3381328588713333,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.045184119534764644,
          "restore_alignment": 0.28763491371784466,
          "transplant_alignment": 0.2779158243802637,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.07641833012202937,
          "restore_alignment": 0.31059463636189544,
          "transplant_alignment": 0.3031327566615946,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.04088070467092053,
          "restore_alignment": 0.3094852867022053,
          "transplant_alignment": 0.33444212840157467,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.20377813237688047,
          "restore_alignment": 0.39439790206384556,
          "transplant_alignment": 0.41327468097767317,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.1656063390803194,
          "restore_alignment": 0.3541407510952843,
          "transplant_alignment": 0.359184188065325,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": 0.2725748132342617
    },
    {
      "index": 2,
      "site": {
        "layer": 9,
        "module": "attention",
        "position": "identity"
      },
      "transmission": {
        "fidelity": -0.03665615729336258,
        "restore_alignment": 0.03898365217916427,
        "transplant_alignment": 0.046844294366455895,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.27619977537639084,
        "transplant_rms": 0.33076504890650854,
        "restore_mean": -0.0009765625,
        "transplant_mean": -0.037841796875
      },
      "destinations": {
        "q17_decision": {
          "fidelity": -0.06789056744735356,
          "restore_alignment": 0.04159465981992048,
          "transplant_alignment": 0.020364928626967618,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": -0.03011114651874358,
          "restore_alignment": 0.04833200821892013,
          "transplant_alignment": 0.058195284493528766,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": -0.09019433720887293,
          "restore_alignment": 0.005658502102961929,
          "transplant_alignment": 0.03382041334365126,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": -0.0965477401149617,
          "restore_alignment": 0.014723115351218466,
          "transplant_alignment": 0.03264702290139887,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": -0.0777383240627656,
          "restore_alignment": 0.023954894298021116,
          "transplant_alignment": 0.04163613530689211,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": -0.05204719383406853,
          "restore_alignment": 0.03273637932938986,
          "transplant_alignment": 0.06934280392537714,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": -0.06034747759167436,
          "restore_alignment": 0.06391396213777538,
          "transplant_alignment": 0.06761554321401174,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": -0.07813896785062124,
          "restore_alignment": 0.056883387036686925,
          "transplant_alignment": 0.05611542181210971,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": 0.04291397327281008
    },
    {
      "index": 3,
      "site": {
        "layer": 9,
        "module": "mlp",
        "position": "identity"
      },
      "transmission": {
        "fidelity": 0.15935036472473918,
        "restore_alignment": 0.25241367510184365,
        "transplant_alignment": 0.3740925915208732,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.5908243478684504,
        "transplant_rms": 0.5992975573584957,
        "restore_mean": -0.0416259765625,
        "transplant_mean": -0.037109375
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.2375406048301305,
          "restore_alignment": 0.26222869997035175,
          "transplant_alignment": 0.31309767773307734,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.161501912858657,
          "restore_alignment": 0.22278054707999367,
          "transplant_alignment": 0.2855271680884622,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.18609590918774643,
          "restore_alignment": 0.23760013905647512,
          "transplant_alignment": 0.28410362960211616,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.14956250043362318,
          "restore_alignment": 0.2269606292555503,
          "transplant_alignment": 0.31928522862777253,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.17067251499099134,
          "restore_alignment": 0.22291738502251635,
          "transplant_alignment": 0.30943104292347895,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.20600005969241753,
          "restore_alignment": 0.2730203076905549,
          "transplant_alignment": 0.3353449766490587,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.21013318921095914,
          "restore_alignment": 0.3384522696847134,
          "transplant_alignment": 0.3470713065434169,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.20882765727021835,
          "restore_alignment": 0.34058421616725004,
          "transplant_alignment": 0.35721396374820313,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": 0.3132531333113584
    },
    {
      "index": 4,
      "site": {
        "layer": 10,
        "module": "attention",
        "position": "identity"
      },
      "transmission": {
        "fidelity": 0.041728838039066685,
        "restore_alignment": 0.13447636397490834,
        "transplant_alignment": 0.14375866520055458,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.36737964553725366,
        "transplant_rms": 0.472849908673791,
        "restore_mean": 0.017822265625,
        "transplant_mean": 0.034912109375
      },
      "destinations": {
        "q17_decision": {
          "fidelity": -0.004180705940984408,
          "restore_alignment": 0.09134482385117995,
          "transplant_alignment": 0.07542205242069992,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": -0.06358987571429164,
          "restore_alignment": 0.09349457826154285,
          "transplant_alignment": 0.08134276768954886,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": -0.05315798873799049,
          "restore_alignment": 0.07778160256159089,
          "transplant_alignment": 0.049702257212759546,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": -0.0393193876134621,
          "restore_alignment": 0.07962739164723923,
          "transplant_alignment": 0.07891560123299353,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": -0.036562874077198826,
          "restore_alignment": 0.08924881955109944,
          "transplant_alignment": 0.09020956337137476,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": -0.0285682214127434,
          "restore_alignment": 0.07319363808419933,
          "transplant_alignment": 0.06873961150428513,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": -0.20644247317692344,
          "restore_alignment": 0.05650570726468254,
          "transplant_alignment": 0.05606635138415998,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": -0.035935684972238535,
          "restore_alignment": 0.06560937692969245,
          "transplant_alignment": 0.05585773357755555,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": 0.13911751458773147
    },
    {
      "index": 5,
      "site": {
        "layer": 10,
        "module": "mlp",
        "position": "identity"
      },
      "transmission": {
        "fidelity": 0.12429451742853337,
        "restore_alignment": 0.2551336988548725,
        "transplant_alignment": 0.208420247444159,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.4644168207531221,
        "transplant_rms": 0.5464894706919324,
        "restore_mean": 0.007080078125,
        "transplant_mean": 0.0152587890625
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.24895841316030676,
          "restore_alignment": 0.29714176789081653,
          "transplant_alignment": 0.2781409495666622,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.21384504662702142,
          "restore_alignment": 0.27622352762764807,
          "transplant_alignment": 0.27999375513442365,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.24240988665440932,
          "restore_alignment": 0.2902755863094935,
          "transplant_alignment": 0.3092957832001696,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.1926962056657865,
          "restore_alignment": 0.2679345951531081,
          "transplant_alignment": 0.273645183401469,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.2410941572714298,
          "restore_alignment": 0.2945959510623461,
          "transplant_alignment": 0.2949184483133551,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.19764331905322152,
          "restore_alignment": 0.2819217833575611,
          "transplant_alignment": 0.3382854966093146,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.29276418124108894,
          "restore_alignment": 0.36741349994263206,
          "transplant_alignment": 0.3600101057520049,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.1601621302709656,
          "restore_alignment": 0.3142216566049451,
          "transplant_alignment": 0.3094547766767807,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": 0.23177697314951576
    },
    {
      "index": 6,
      "site": {
        "layer": 11,
        "module": "attention",
        "position": "identity"
      },
      "transmission": {
        "fidelity": 0.24711956781565236,
        "restore_alignment": 0.1789805824021257,
        "transplant_alignment": 0.22336150650566272,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.32058034723482837,
        "transplant_rms": 0.3640784264220245,
        "restore_mean": -0.00146484375,
        "transplant_mean": 0.00634765625
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.20845441634267559,
          "restore_alignment": 0.15928625190228035,
          "transplant_alignment": 0.18017147876425924,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.1243613989287442,
          "restore_alignment": 0.11914683915402713,
          "transplant_alignment": 0.1352047999428578,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.14421740796286886,
          "restore_alignment": 0.13451404677740925,
          "transplant_alignment": 0.15369595289625795,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.18435778729345476,
          "restore_alignment": 0.15558370596697785,
          "transplant_alignment": 0.18889142906504736,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.1701686625460167,
          "restore_alignment": 0.14358069447990052,
          "transplant_alignment": 0.17835942923537068,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.16702054496914054,
          "restore_alignment": 0.18622399713346796,
          "transplant_alignment": 0.19824384616602986,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.11910096566731343,
          "restore_alignment": 0.16469721707532495,
          "transplant_alignment": 0.17561363576909833,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": 0.20117104445389422
    },
    {
      "index": 7,
      "site": {
        "layer": 11,
        "module": "mlp",
        "position": "identity"
      },
      "transmission": {
        "fidelity": 0.1902368509120006,
        "restore_alignment": 0.19765841433434905,
        "transplant_alignment": 0.1881119109865307,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.3529513581995157,
        "transplant_rms": 0.41456429647064774,
        "restore_mean": -0.0303955078125,
        "transplant_mean": -0.041259765625
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.13693969750018486,
          "restore_alignment": 0.16449910260019254,
          "transplant_alignment": 0.1764134482208856,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.1152749721028622,
          "restore_alignment": 0.14392678835623338,
          "transplant_alignment": 0.15432440186577775,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.1503089961662628,
          "restore_alignment": 0.20252507997167749,
          "transplant_alignment": 0.19833931550397155,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.15698542200530907,
          "restore_alignment": 0.17545289511780177,
          "transplant_alignment": 0.18903210640395218,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.14049916209734925,
          "restore_alignment": 0.17106927453030812,
          "transplant_alignment": 0.1807659798173866,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.21608476000149324,
          "restore_alignment": 0.23517081444832427,
          "transplant_alignment": 0.2559244982905314,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.17894618516369054,
          "restore_alignment": 0.3278998780575572,
          "transplant_alignment": 0.33264010172863634,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": 0.19288516266043987
    },
    {
      "index": 8,
      "site": {
        "layer": 12,
        "module": "attention",
        "position": "identity"
      },
      "transmission": {
        "fidelity": 0.08567194148300428,
        "restore_alignment": 0.08356587313768957,
        "transplant_alignment": 0.06199439891500995,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.21143874906102758,
        "transplant_rms": 0.21467944711743703,
        "restore_mean": -0.0374755859375,
        "transplant_mean": -0.0382080078125
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.06657924074119637,
          "restore_alignment": 0.07547949430551874,
          "transplant_alignment": 0.07327316751820392,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.02815857762300933,
          "restore_alignment": 0.058177155238463875,
          "transplant_alignment": 0.04843948251568325,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.034857347737298316,
          "restore_alignment": 0.050297861901147115,
          "transplant_alignment": 0.04715846519600491,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.05388653829848178,
          "restore_alignment": 0.06601488763677643,
          "transplant_alignment": 0.060377202586833156,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.036245003323806424,
          "restore_alignment": 0.055239298865059544,
          "transplant_alignment": 0.050337099885060195,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.08222094491998866,
          "restore_alignment": 0.09235631895149728,
          "transplant_alignment": 0.09850813808912778,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.05282418538009759,
          "restore_alignment": 0.13666387806430141,
          "transplant_alignment": 0.12865582203482911,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": 0.07278013602634975
    },
    {
      "index": 9,
      "site": {
        "layer": 12,
        "module": "mlp",
        "position": "identity"
      },
      "transmission": {
        "fidelity": 0.011702895906880117,
        "restore_alignment": 0.05542960523170526,
        "transplant_alignment": 0.04980839055615805,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.24830049779859736,
        "transplant_rms": 0.2831088205055556,
        "restore_mean": -0.004150390625,
        "transplant_mean": -0.00927734375
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.08587385207027376,
          "restore_alignment": 0.12624524312598034,
          "transplant_alignment": 0.11844516912879531,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.07663129736515983,
          "restore_alignment": 0.11967835227687727,
          "transplant_alignment": 0.11788004692409607,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.06523927224306514,
          "restore_alignment": 0.10549396247964692,
          "transplant_alignment": 0.09934518878564079,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.05313215711622343,
          "restore_alignment": 0.09322711401676095,
          "transplant_alignment": 0.09095011040441514,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.05821127452294672,
          "restore_alignment": 0.09403280812546166,
          "transplant_alignment": 0.0978840307732906,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.07472246098755819,
          "restore_alignment": 0.13213364981451267,
          "transplant_alignment": 0.12369275819685548,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.2253600203595969,
          "restore_alignment": 0.38330842063450027,
          "transplant_alignment": 0.373844307895579,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": 0.05261899789393165
    },
    {
      "index": 10,
      "site": {
        "layer": 14,
        "module": "attention",
        "position": "decision"
      },
      "transmission": {
        "fidelity": 0.028375307921282844,
        "restore_alignment": 0.013972517946872728,
        "transplant_alignment": 0.01589490291201063,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.03259437357085671,
        "transplant_rms": 0.034637007521422264,
        "restore_mean": -0.0052490234375,
        "transplant_mean": -0.0098876953125
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.09320447534345588,
          "restore_alignment": 0.06287023576715284,
          "transplant_alignment": 0.05813502540716005,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.14225912890553039,
          "restore_alignment": 0.10227742947361536,
          "transplant_alignment": 0.09417133547242042,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.03630809603331153,
          "restore_alignment": 0.02431213644238972,
          "transplant_alignment": 0.0225333101131995,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.0762762585282174,
          "restore_alignment": 0.05085903407378218,
          "transplant_alignment": 0.04977155899129777,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": 0.01493371042944168
    },
    {
      "index": 11,
      "site": {
        "layer": 14,
        "module": "mlp",
        "position": "decision"
      },
      "transmission": {
        "fidelity": 0.00022771706720547957,
        "restore_alignment": -0.002216781615558234,
        "transplant_alignment": 0.003414497902738392,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.028168369339562415,
        "transplant_rms": 0.026021313661082478,
        "restore_mean": 0.00146484375,
        "transplant_mean": -0.0010986328125
      },
      "destinations": {
        "q17_decision": {
          "fidelity": -0.010937418032975321,
          "restore_alignment": 0.0031265016007077577,
          "transplant_alignment": 0.0015949452223074517,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": -0.013721926260476192,
          "restore_alignment": 0.011099650369966052,
          "transplant_alignment": 0.0071055452574473685,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": -0.0004452080511516687,
          "restore_alignment": 0.004377165727394328,
          "transplant_alignment": 0.0038754440747460622,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": -0.000627967061787249,
          "restore_alignment": 0.009499500965361488,
          "transplant_alignment": 0.008040807059166325,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": 0.0005988581435900791
    },
    {
      "index": 12,
      "site": {
        "layer": 15,
        "module": "attention",
        "position": "decision"
      },
      "transmission": {
        "fidelity": 0.026568668647763194,
        "restore_alignment": 0.023644831829058712,
        "transplant_alignment": 0.01308177936354967,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.09021450335844529,
        "transplant_rms": 0.0852131224936464,
        "restore_mean": -0.0001220703125,
        "transplant_mean": -0.0035400390625
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.49318848338736243,
          "restore_alignment": 0.3831307550127578,
          "transplant_alignment": 0.3837707508656528,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.5644223419195287,
          "restore_alignment": 0.4635448406230956,
          "transplant_alignment": 0.462143659317059,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.1350960352596724,
          "restore_alignment": 0.11546495648730042,
          "transplant_alignment": 0.11253307221502148,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.2236416410441222,
          "restore_alignment": 0.1850434450273762,
          "transplant_alignment": 0.18694989626174727,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": 0.01836330559630419
    },
    {
      "index": 13,
      "site": {
        "layer": 15,
        "module": "mlp",
        "position": "decision"
      },
      "transmission": {
        "fidelity": 0.027942519683058054,
        "restore_alignment": 0.013902064047626835,
        "transplant_alignment": 0.016951711400699003,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.043934460465632784,
        "transplant_rms": 0.04983340164911357,
        "restore_mean": -0.00439453125,
        "transplant_mean": -0.009033203125
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.021591804479300425,
          "restore_alignment": 0.0631481138498651,
          "transplant_alignment": 0.061110420064982164,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.09190775772568138,
          "restore_alignment": 0.12336382044446784,
          "transplant_alignment": 0.12550836277386146,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.022392018391684365,
          "restore_alignment": 0.030908564306690735,
          "transplant_alignment": 0.03260337457527918,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.08061293029131555,
          "restore_alignment": 0.07746806522716412,
          "transplant_alignment": 0.08142855204188203,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": 0.015426887724162919
    },
    {
      "index": 14,
      "site": {
        "layer": 16,
        "module": "attention",
        "position": "decision"
      },
      "transmission": {
        "fidelity": 0.11657100872370241,
        "restore_alignment": 0.06516734059176243,
        "transplant_alignment": 0.06888881619835792,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.11431205309764338,
        "transplant_rms": 0.11593569689888308,
        "restore_mean": -0.0010986328125,
        "transplant_mean": -0.0057373046875
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.6200358435123606,
          "restore_alignment": 0.5141997208431016,
          "transplant_alignment": 0.5114507878729063,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.48493614621837045,
          "restore_alignment": 0.4011000487757936,
          "transplant_alignment": 0.39913128165949674,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.2725474330743928,
          "restore_alignment": 0.2096179312868181,
          "transplant_alignment": 0.2078437330596817,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.26006520912337017,
          "restore_alignment": 0.19447370734782826,
          "transplant_alignment": 0.19693794075096377,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": 0.06702807839506017
    },
    {
      "index": 15,
      "site": {
        "layer": 16,
        "module": "mlp",
        "position": "decision"
      },
      "transmission": {
        "fidelity": 0.03723991817282846,
        "restore_alignment": 0.020766286802726568,
        "transplant_alignment": 0.0207537057492898,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.05734343068712187,
        "transplant_rms": 0.05657323577328218,
        "restore_mean": 0.003173828125,
        "transplant_mean": 0.00146484375
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.190178236151324,
          "restore_alignment": 0.18687985580700203,
          "transplant_alignment": 0.18529836630512928,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.1999983254100438,
          "restore_alignment": 0.22409979791433768,
          "transplant_alignment": 0.22429779806867103,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.01607884311339436,
          "restore_alignment": 0.049020585268943144,
          "transplant_alignment": 0.049271903290534796,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.10957818032602273,
          "restore_alignment": 0.10497508809272649,
          "transplant_alignment": 0.10700310226344613,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": 0.020759996276008184
    },
    {
      "index": 16,
      "site": {
        "layer": 17,
        "module": "attention",
        "position": "decision"
      },
      "transmission": {
        "fidelity": 0.7419525291691724,
        "restore_alignment": 0.5085664392850943,
        "transplant_alignment": 0.5003661086550099,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.45363824650420104,
        "transplant_rms": 0.4460571487830553,
        "restore_mean": -0.044677734375,
        "transplant_mean": -0.044677734375
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.6408942261988547,
          "restore_alignment": 0.6061671424734304,
          "transplant_alignment": 0.6068919465022814,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.5665620453794851,
          "restore_alignment": 0.4766433646709763,
          "transplant_alignment": 0.47807413380945074,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": 0.5044662739700521
    },
    {
      "index": 17,
      "site": {
        "layer": 17,
        "module": "mlp",
        "position": "decision"
      },
      "transmission": {
        "fidelity": 0.20385206694126912,
        "restore_alignment": 0.1161055097465421,
        "transplant_alignment": 0.10779949826758894,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.13093223251636227,
        "transplant_rms": 0.11514330352512771,
        "restore_mean": -0.025634765625,
        "transplant_mean": -0.0189208984375
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.16121861466955856,
          "restore_alignment": 0.16373950702225584,
          "transplant_alignment": 0.1676270585900167,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.21865560751686275,
          "restore_alignment": 0.204647509320388,
          "transplant_alignment": 0.208248095401803,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": 0.11195250400706552
    },
    {
      "index": 18,
      "site": {
        "layer": 18,
        "module": "attention",
        "position": "decision"
      },
      "transmission": {
        "fidelity": -0.338026485633695,
        "restore_alignment": -0.14767640524076361,
        "transplant_alignment": -0.14109399808264744,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.19302369852782164,
        "transplant_rms": 0.19343334285971178,
        "restore_mean": 0.02392578125,
        "transplant_mean": 0.0220947265625
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": -0.20816676421459146,
          "restore_alignment": -0.01260361368560341,
          "transplant_alignment": -0.010777254896113869,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": -0.012079308497449759,
          "restore_alignment": 0.045236292758145696,
          "transplant_alignment": 0.05000895083112333,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": -0.14438520166170554
    },
    {
      "index": 19,
      "site": {
        "layer": 18,
        "module": "mlp",
        "position": "decision"
      },
      "transmission": {
        "fidelity": 0.4145620661109196,
        "restore_alignment": 0.24224063529287435,
        "transplant_alignment": 0.2375755806785214,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.22355817190517982,
        "transplant_rms": 0.22124691028916088,
        "restore_mean": -0.0198974609375,
        "transplant_mean": -0.015869140625
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.4945847571873986,
          "restore_alignment": 0.3887507678141706,
          "transplant_alignment": 0.3899788375606897,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.37953076927710794,
          "restore_alignment": 0.30876383360309495,
          "transplant_alignment": 0.31155617527788665,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": 0.23990810798569787
    },
    {
      "index": 20,
      "site": {
        "layer": 19,
        "module": "attention",
        "position": "decision"
      },
      "transmission": {
        "fidelity": 0.12833177747639168,
        "restore_alignment": 0.06762316222261923,
        "transplant_alignment": 0.06791252645166485,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.0738451741226936,
        "transplant_rms": 0.0739484180376662,
        "restore_mean": -0.0015869140625,
        "transplant_mean": 0.0003662109375
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.12493552913990524,
          "restore_alignment": 0.09784335727717272,
          "transplant_alignment": 0.09724526331889848,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.09898873280202025,
          "restore_alignment": 0.08883710431183792,
          "transplant_alignment": 0.09035049378576303,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": 0.06776784433714203
    },
    {
      "index": 21,
      "site": {
        "layer": 19,
        "module": "mlp",
        "position": "decision"
      },
      "transmission": {
        "fidelity": 0.2665069711617093,
        "restore_alignment": 0.14321767990277362,
        "transplant_alignment": 0.148270230962979,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.1367675694088046,
        "transplant_rms": 0.1384447413632342,
        "restore_mean": -0.0108642578125,
        "transplant_mean": -0.0079345703125
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.40734475731456754,
          "restore_alignment": 0.2937561202121377,
          "transplant_alignment": 0.29344432019866507,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.43635676694272274,
          "restore_alignment": 0.35982497051249934,
          "transplant_alignment": 0.3627525838635164,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": 0.14574395543287633
    },
    {
      "index": 22,
      "site": {
        "layer": 20,
        "module": "attention",
        "position": "decision"
      },
      "transmission": {
        "fidelity": 0.005867803322907816,
        "restore_alignment": 0.0015701154689084426,
        "transplant_alignment": 0.005450112348807191,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.02962060720332637,
        "transplant_rms": 0.029491540762776366,
        "restore_mean": 0.001953125,
        "transplant_mean": 0.00146484375
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": -0.024432164205682616,
          "restore_alignment": -0.005123484812182292,
          "transplant_alignment": -0.004393894053036618,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.00011643285286089267,
          "restore_alignment": 0.009367423636683995,
          "transplant_alignment": 0.011281214584894816,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": 0.0035101139088578167
    },
    {
      "index": 23,
      "site": {
        "layer": 20,
        "module": "mlp",
        "position": "decision"
      },
      "transmission": {
        "fidelity": 0.18813329877737328,
        "restore_alignment": 0.10406544160755668,
        "transplant_alignment": 0.10629983669792639,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.13353577186718196,
        "transplant_rms": 0.12598791736617296,
        "restore_mean": -0.0130615234375,
        "transplant_mean": -0.006591796875
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.12060402797642034,
          "restore_alignment": 0.10385319622636158,
          "transplant_alignment": 0.10481008618496454,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.16586618585321666,
          "restore_alignment": 0.1546323020381812,
          "transplant_alignment": 0.1554819757616236,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      },
      "signed_alignment": 0.10518263915274154
    }
  ],
  "unit_sites": [
    {
      "layer": 9,
      "module": "mlp",
      "position": "identity",
      "id": "mlp9_identity",
      "coarse_index": 3
    },
    {
      "layer": 18,
      "module": "mlp",
      "position": "decision",
      "id": "mlp18_decision",
      "coarse_index": 19
    },
    {
      "layer": 17,
      "head": 1,
      "position": "decision",
      "module": "head",
      "id": "head17_1_decision"
    },
    {
      "layer": 21,
      "head": 1,
      "position": "decision",
      "module": "head",
      "id": "head21_1_decision"
    }
  ],
  "selection": "largest absolute source-effect alignment per source/terminal MLP, fit only"
}

## Active utilities

{
  "reading": {
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
    "overall": {
      "rows": 256,
      "pairs": 64,
      "accuracy": 0.96875,
      "both_correct": 0.9375,
      "flip": 0.9375,
      "gap": 0.03125,
      "by_permutation": {
        "0": {
          "accuracy": 0.96875,
          "both_correct": 0.9375,
          "flip": 0.9375,
          "gap": 0.03125
        },
        "1": {
          "accuracy": 0.96875,
          "both_correct": 0.9375,
          "flip": 0.9375,
          "gap": 0.03125
        }
      },
      "gold_margin": 5.76007080078125
    },
    "whole_four_heads": {
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
    }
  },
  "literal": {
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
    "overall": {
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
      "gold_margin": 7.8436279296875
    },
    "whole_four_heads": {
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
    "overall": {
      "rows": 256,
      "pairs": 64,
      "accuracy": 0.984375,
      "both_correct": 0.96875,
      "flip": 0.96875,
      "gap": 0.03125,
      "by_permutation": {
        "0": {
          "accuracy": 0.9921875,
          "both_correct": 0.984375,
          "flip": 0.984375,
          "gap": 0.015625
        },
        "1": {
          "accuracy": 0.9765625,
          "both_correct": 0.953125,
          "flip": 0.953125,
          "gap": 0.046875
        }
      },
      "gold_margin": 5.15045166015625
    },
    "whole_four_heads": {
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
    "overall": {
      "rows": 256,
      "pairs": 64,
      "accuracy": 0.79296875,
      "both_correct": 0.703125,
      "flip": 0.1796875,
      "gap": -0.0859375,
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
          "gap": -0.09375
        }
      },
      "gold_margin": 4.005126953125
    },
    "whole_four_heads": {
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
  },
  "record": {
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
    "overall": {
      "rows": 512,
      "pairs": 128,
      "accuracy": 0.708984375,
      "both_correct": 0.63671875,
      "flip": 0.14453125,
      "gap": -0.01953125,
      "by_permutation": {
        "0": {
          "accuracy": 0.70703125,
          "both_correct": 0.625,
          "flip": 0.1640625,
          "gap": -0.0390625
        },
        "1": {
          "accuracy": 0.7109375,
          "both_correct": 0.6484375,
          "flip": 0.125,
          "gap": 0.0
        }
      },
      "gold_margin": 2.79608154296875
    },
    "whole_four_heads": {
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
    }
  }
}

## Float16 orientation control checks

{
  "q17_decision": {
    "max_relative_norm_error": 0.00012988544767722487,
    "mean_learned_edit_norm": 0.683234453201294,
    "mean_random_edit_norm": 0.68313068151474,
    "max_edit_norm_difference_after_float16": 0.004018114879727364
  },
  "k17_decision": {
    "max_relative_norm_error": 0.0002387005224591121,
    "mean_learned_edit_norm": 0.42442989349365234,
    "mean_random_edit_norm": 0.4243580102920532,
    "max_edit_norm_difference_after_float16": 0.013353079557418823
  },
  "k17_option_b": {
    "max_relative_norm_error": 0.00021848754840902984,
    "mean_learned_edit_norm": 0.8819663524627686,
    "mean_random_edit_norm": 0.8824865818023682,
    "max_edit_norm_difference_after_float16": 0.02056998573243618
  },
  "q21_decision": {
    "max_relative_norm_error": 0.00010342879977542907,
    "mean_learned_edit_norm": 0.41150104999542236,
    "mean_random_edit_norm": 0.4114685654640198,
    "max_edit_norm_difference_after_float16": 0.002028812188655138
  },
  "k21_decision": {
    "max_relative_norm_error": 0.00017004311666823924,
    "mean_learned_edit_norm": 0.3299739360809326,
    "mean_random_edit_norm": 0.32997432351112366,
    "max_edit_norm_difference_after_float16": 0.00987571757286787
  },
  "k21_option_b": {
    "max_relative_norm_error": 0.00023235213302541524,
    "mean_learned_edit_norm": 0.8134570717811584,
    "mean_random_edit_norm": 0.8134120106697083,
    "max_edit_norm_difference_after_float16": 0.014329907484352589
  }
}

## Actual native-chat short generation

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
      },
      "unparsed": 0.0546875
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
      },
      "unparsed": 0.0625
    },
    "overall": {
      "strict": {
        "rows": 128,
        "pairs": 64,
        "accuracy": 0.8515625,
        "both_correct": 0.796875,
        "flip": 0.109375,
        "gap": 0.078125,
        "by_permutation": {
          "0": {
            "accuracy": 0.8515625,
            "both_correct": 0.796875,
            "flip": 0.109375,
            "gap": 0.078125
          }
        }
      },
      "line": {
        "rows": 128,
        "pairs": 64,
        "accuracy": 0.8671875,
        "both_correct": 0.8125,
        "flip": 0.109375,
        "gap": 0.078125,
        "by_permutation": {
          "0": {
            "accuracy": 0.8671875,
            "both_correct": 0.8125,
            "flip": 0.109375,
            "gap": 0.078125
          }
        }
      },
      "unparsed": 0.046875
    },
    "whole_four_heads": {
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
      },
      "unparsed": 0.0703125
    }
  },
  "coverage": {
    "baseline": [],
    "source_half": [
      {
        "edit": {
          "type": "source"
        },
        "active_on_chat": true
      }
    ],
    "overall": [
      {
        "edit": {
          "type": "unit",
          "site": "mlp9_identity",
          "mode": "centered",
          "groups": [
            [
              "gap_pos",
              0.5
            ]
          ],
          "group": "whole",
          "amount": 0.5
        },
        "active_on_chat": true
      },
      {
        "edit": {
          "type": "unit",
          "site": "head21_1_decision",
          "mode": "centered",
          "groups": [
            [
              "gap_pos",
              -0.5
            ],
            [
              "gap_neg",
              0.5
            ]
          ],
          "group": "whole",
          "amount": 0.5
        },
        "active_on_chat": true
      }
    ],
    "whole_four_heads": [
      {
        "edit": {
          "type": "whole_head",
          "layer": 21,
          "position": "decision",
          "head": 1,
          "amount": 0.5
        },
        "active_on_chat": true
      },
      {
        "edit": {
          "type": "whole_head",
          "layer": 17,
          "position": "decision",
          "head": 1,
          "amount": 0.5
        },
        "active_on_chat": true
      },
      {
        "edit": {
          "type": "whole_head",
          "layer": 13,
          "position": "question_identity",
          "head": 6,
          "amount": 0.5
        },
        "active_on_chat": true
      },
      {
        "edit": {
          "type": "whole_head",
          "layer": 11,
          "position": "question_identity",
          "head": 13,
          "amount": 0.5
        },
        "active_on_chat": true
      }
    ]
  },
  "pairs": 64,
  "status": "historical native chat short-answer; prefill-only"
}
{
  "overall": {
    "accuracy": {
      "delta": 0.0078125,
      "ci95": [
        -0.024193548387096774,
        0.04413410886742763
      ],
      "cluster": "family_id",
      "clusters": 61,
      "p_sign": 1.0
    },
    "both_correct": {
      "delta": 0.015625,
      "ci95": [
        -0.03225806451612903,
        0.07249040920716122
      ],
      "cluster": "family_id",
      "clusters": 61,
      "p_sign": 1.0
    },
    "flip": {
      "delta": -0.015625,
      "ci95": [
        -0.08196721311475409,
        0.04838709677419355
      ],
      "cluster": "family_id",
      "clusters": 61,
      "p_sign": 1.0
    }
  },
  "baseline": {
    "accuracy": {
      "delta": -0.0234375,
      "ci95": [
        -0.056451612903225805,
        0.008064516129032258
      ],
      "cluster": "family_id",
      "clusters": 61,
      "p_sign": 0.38665333666583357
    },
    "both_correct": {
      "delta": -0.046875,
      "ci95": [
        -0.10606060606060606,
        0.0
      ],
      "cluster": "family_id",
      "clusters": 61,
      "p_sign": 0.2539365158710322
    },
    "flip": {
      "delta": 0.046875,
      "ci95": [
        -0.016129032258064516,
        0.12121212121212122
      ],
      "cluster": "family_id",
      "clusters": 61,
      "p_sign": 0.3839040239940015
    }
  },
  "whole_four_heads": {
    "accuracy": {
      "delta": -0.046875,
      "ci95": [
        -0.09701492537313433,
        -0.007692307692307693
      ],
      "cluster": "family_id",
      "clusters": 61,
      "p_sign": 0.11972006998250437
    },
    "both_correct": {
      "delta": -0.0625,
      "ci95": [
        -0.13846153846153847,
        0.0
      ],
      "cluster": "family_id",
      "clusters": 61,
      "p_sign": 0.23094226443389151
    },
    "flip": {
      "delta": 0.03125,
      "ci95": [
        -0.046875,
        0.10606060606060606
      ],
      "cluster": "family_id",
      "clusters": 61,
      "p_sign": 0.686078480379905
    }
  }
}

## routes

{
  "option_sites": [
    {
      "site": {
        "layer": 14,
        "module": "attention",
        "position": "option_b"
      },
      "transmission": {
        "fidelity": 0.2609335644892218,
        "restore_alignment": 0.15634978348007036,
        "transplant_alignment": 0.155421301736437,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.2003500735650952,
        "transplant_rms": 0.19217697983147836,
        "restore_mean": 0.0045166015625,
        "transplant_mean": 0.0032958984375
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.05518299014137784,
          "restore_alignment": 0.04600367467804757,
          "transplant_alignment": 0.0419381766657377,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.05726649434086517,
          "restore_alignment": 0.04143777029762255,
          "transplant_alignment": 0.037859390316750716,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.2760712494452421,
          "restore_alignment": 0.24036322642351923,
          "transplant_alignment": 0.23942157682096302,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.17960617976542914,
          "restore_alignment": 0.11715958523892553,
          "transplant_alignment": 0.12121852443082749,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.16107567170496417,
          "restore_alignment": 0.10421293070647972,
          "transplant_alignment": 0.102502724798561,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.1602997195231194,
          "restore_alignment": 0.16093218864902567,
          "transplant_alignment": 0.16840427904685976,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      }
    },
    {
      "site": {
        "layer": 14,
        "module": "mlp",
        "position": "option_b"
      },
      "transmission": {
        "fidelity": 0.12393847361627286,
        "restore_alignment": 0.08508566439285094,
        "transplant_alignment": 0.0856694252723169,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.18638277991653107,
        "transplant_rms": 0.19036206732120947,
        "restore_mean": -0.0089111328125,
        "transplant_mean": -0.0047607421875
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.06535118727934275,
          "restore_alignment": 0.06037162426184834,
          "transplant_alignment": 0.056136915240474476,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.03746552456938934,
          "restore_alignment": 0.03744498793666271,
          "transplant_alignment": 0.03180705351640214,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.16885323647347927,
          "restore_alignment": 0.22702037109492304,
          "transplant_alignment": 0.21793254000431703,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.10977303161047458,
          "restore_alignment": 0.0841370350042859,
          "transplant_alignment": 0.0910643422503413,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.0801005572628416,
          "restore_alignment": 0.06289583909537491,
          "transplant_alignment": 0.064590006638671,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.12750812955797874,
          "restore_alignment": 0.1522581888436687,
          "transplant_alignment": 0.15413518625302566,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      }
    },
    {
      "site": {
        "layer": 15,
        "module": "attention",
        "position": "option_b"
      },
      "transmission": {
        "fidelity": 0.029356630089350633,
        "restore_alignment": 0.012367175528341339,
        "transplant_alignment": 0.018718091303221,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.03399784215923674,
        "transplant_rms": 0.038273277230987154,
        "restore_mean": 0.00048828125,
        "transplant_mean": -0.002197265625
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.004314350351732443,
          "restore_alignment": 0.007313561976962848,
          "transplant_alignment": 0.006591845083853843,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.0031388878906866147,
          "restore_alignment": 0.004740141950683361,
          "transplant_alignment": 0.004040482881508834,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.26190460497421175,
          "restore_alignment": 0.1889569641201513,
          "transplant_alignment": 0.18995385129748654,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.023115686616597686,
          "restore_alignment": 0.014672877028855844,
          "transplant_alignment": 0.014424903782103697,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.016653692566443445,
          "restore_alignment": 0.010940638737958521,
          "transplant_alignment": 0.012331788759564444,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.16242623480379492,
          "restore_alignment": 0.11283479142280026,
          "transplant_alignment": 0.12308196567625809,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      }
    },
    {
      "site": {
        "layer": 15,
        "module": "mlp",
        "position": "option_b"
      },
      "transmission": {
        "fidelity": 0.0465826084549712,
        "restore_alignment": 0.025786127123996347,
        "transplant_alignment": 0.024787191481117098,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.05305889729377392,
        "transplant_rms": 0.05687585851000405,
        "restore_mean": -0.000244140625,
        "transplant_mean": -0.004638671875
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.013473219460311991,
          "restore_alignment": 0.022304995689399604,
          "transplant_alignment": 0.02137558945843898,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.007385050235740209,
          "restore_alignment": 0.010743604876649096,
          "transplant_alignment": 0.008897557656712306,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.06565125248617432,
          "restore_alignment": 0.15094058591214354,
          "transplant_alignment": 0.1523432708280742,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.04838419881543832,
          "restore_alignment": 0.03697862819751008,
          "transplant_alignment": 0.036018416301371856,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.03785055535621418,
          "restore_alignment": 0.02559388471622907,
          "transplant_alignment": 0.025578466902767574,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.09407037222043069,
          "restore_alignment": 0.1350798271847801,
          "transplant_alignment": 0.13653956858806324,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      }
    },
    {
      "site": {
        "layer": 16,
        "module": "attention",
        "position": "option_b"
      },
      "transmission": {
        "fidelity": 0.00014719832521015164,
        "restore_alignment": -0.0012253946047410442,
        "transplant_alignment": 0.001985290232321733,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.020156022891162396,
        "transplant_rms": 0.02286074201312427,
        "restore_mean": 0.0010986328125,
        "transplant_mean": -0.002197265625
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.12282966938208706,
          "restore_alignment": 0.11520949794836804,
          "transplant_alignment": 0.11553674788412308,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.004702256708956298,
          "restore_alignment": 0.003777104151228902,
          "transplant_alignment": 0.003169705019613713,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.005475243140491082,
          "restore_alignment": 0.004910618331851011,
          "transplant_alignment": 0.004300737798948925,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.24954352384912204,
          "restore_alignment": 0.23844591392810607,
          "transplant_alignment": 0.2411959444381137,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      }
    },
    {
      "site": {
        "layer": 16,
        "module": "mlp",
        "position": "option_b"
      },
      "transmission": {
        "fidelity": -0.005490371719804932,
        "restore_alignment": -0.003303784632494848,
        "transplant_alignment": -0.0006340850932130249,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.03377268840974723,
        "transplant_rms": 0.03482920800806056,
        "restore_mean": 0.005615234375,
        "transplant_mean": 0.003662109375
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.21365516045553357,
          "restore_alignment": 0.20605951745932197,
          "transplant_alignment": 0.20437303179023317,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.01725484336990002,
          "restore_alignment": 0.009315024385606385,
          "transplant_alignment": 0.011229945108160578,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.013839144087222643,
          "restore_alignment": 0.007943717355946981,
          "transplant_alignment": 0.01098018334183413,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.3442482552213355,
          "restore_alignment": 0.25160216314963757,
          "transplant_alignment": 0.26183737813468716,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      }
    },
    {
      "site": {
        "layer": 17,
        "module": "attention",
        "position": "option_b"
      },
      "transmission": {
        "fidelity": 0.0014354981971350655,
        "restore_alignment": -0.0007624118382680418,
        "transplant_alignment": 0.0024583378415441483,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.013879560940968648,
        "transplant_rms": 0.014218964627501012,
        "restore_mean": 0.0013427734375,
        "transplant_mean": -0.001708984375
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": -0.00036253676926301104,
          "restore_alignment": 0.0002066850032040008,
          "transplant_alignment": 0.00011491671319116504,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 1.0434228729927497e-05,
          "restore_alignment": 2.410440725780495e-05,
          "transplant_alignment": 0.0018847284909284598,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.1059172140900615,
          "restore_alignment": 0.06724280580560293,
          "transplant_alignment": 0.06842842656046579,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      }
    },
    {
      "site": {
        "layer": 17,
        "module": "mlp",
        "position": "option_b"
      },
      "transmission": {
        "fidelity": -0.005748283315258451,
        "restore_alignment": -0.003854834773025215,
        "transplant_alignment": -0.0014141104062925396,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.018936249443032536,
        "transplant_rms": 0.019186408891004394,
        "restore_mean": 0.00048828125,
        "transplant_mean": -0.0010986328125
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": -0.001232038328877083,
          "restore_alignment": 0.00022478407744487043,
          "transplant_alignment": -0.000633076675863087,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": -0.0003901883050914101,
          "restore_alignment": 0.00014922011221151418,
          "transplant_alignment": 0.0015050416897029665,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.21642515240303783,
          "restore_alignment": 0.1515849558726595,
          "transplant_alignment": 0.15335089005621252,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      }
    },
    {
      "site": {
        "layer": 18,
        "module": "attention",
        "position": "option_b"
      },
      "transmission": {
        "fidelity": -0.0020456792888181585,
        "restore_alignment": -0.0031427471485042384,
        "transplant_alignment": 0.0014115941956051865,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.015502449088269086,
        "transplant_rms": 0.015378921628929319,
        "restore_mean": 0.000244140625,
        "transplant_mean": -0.00390625
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.00014602196641955167,
          "restore_alignment": 0.0002311938235422899,
          "transplant_alignment": 0.00030406724240952316,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.0009833272589400144,
          "restore_alignment": 0.0005561189371818101,
          "transplant_alignment": 0.0018490626079048446,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.07896086405114433,
          "restore_alignment": 0.05432447923851834,
          "transplant_alignment": 0.0542948846880765,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      }
    },
    {
      "site": {
        "layer": 18,
        "module": "mlp",
        "position": "option_b"
      },
      "transmission": {
        "fidelity": -0.003970580464643492,
        "restore_alignment": -0.002891126079768911,
        "transplant_alignment": -0.0007674442596427484,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.015440808887540914,
        "transplant_rms": 0.015316784317732542,
        "restore_mean": 0.0013427734375,
        "transplant_mean": -0.0001220703125
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": -0.0009998582248544885,
          "restore_alignment": -0.00041135461768829174,
          "transplant_alignment": -0.0001468954617188347,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": -0.0010629996913482653,
          "restore_alignment": -0.0009888158477403455,
          "transplant_alignment": 0.0013719442401654244,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.1942709800681648,
          "restore_alignment": 0.13401512658473355,
          "transplant_alignment": 0.1332599360226068,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      }
    },
    {
      "site": {
        "layer": 19,
        "module": "attention",
        "position": "option_b"
      },
      "transmission": {
        "fidelity": -0.0009762897466931708,
        "restore_alignment": -0.001748766427710525,
        "transplant_alignment": 0.0010618409100630814,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.014285877771804486,
        "transplant_rms": 0.015316784317732542,
        "restore_mean": -0.0015869140625,
        "transplant_mean": -0.0010986328125
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.000377328853408021,
          "restore_alignment": 0.0002885501782478675,
          "transplant_alignment": 0.00026012245380082716,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 7.586679502991078e-05,
          "restore_alignment": 0.00022385315545898077,
          "transplant_alignment": 0.000544453353449137,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": -0.0734735590230744,
          "restore_alignment": -0.02518593714177359,
          "transplant_alignment": -0.024998531804114998,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      }
    },
    {
      "site": {
        "layer": 19,
        "module": "mlp",
        "position": "option_b"
      },
      "transmission": {
        "fidelity": -0.0006265364611510726,
        "restore_alignment": -0.001351205139108708,
        "transplant_alignment": 0.001011516696316016,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.015254393898255184,
        "transplant_rms": 0.014218964627501012,
        "restore_mean": 0.001708984375,
        "transplant_mean": 0.0
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.00036848783065934754,
          "restore_alignment": 0.0002576996458611784,
          "transplant_alignment": 0.0003076889875953808,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.00021016811922058842,
          "restore_alignment": 0.00017580343739942626,
          "transplant_alignment": 0.0008525590702234688,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.2729986116498314,
          "restore_alignment": 0.17944117642433383,
          "transplant_alignment": 0.178341535529327,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      }
    },
    {
      "site": {
        "layer": 20,
        "module": "attention",
        "position": "option_b"
      },
      "transmission": {
        "fidelity": -0.00027552507026529227,
        "restore_alignment": -0.0006290526718383183,
        "transplant_alignment": 0.0005913095115280193,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.014285877771804486,
        "transplant_rms": 0.012506102026236032,
        "restore_mean": 0.0015869140625,
        "transplant_mean": 0.00048828125
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.020258905703094188,
          "restore_alignment": 0.01692064594563923,
          "transplant_alignment": 0.017529493153188535,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      }
    },
    {
      "site": {
        "layer": 20,
        "module": "mlp",
        "position": "option_b"
      },
      "transmission": {
        "fidelity": 0.0008743832138552454,
        "restore_alignment": -0.0011247461772469131,
        "transplant_alignment": 0.002264589618617946,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.013948102399497755,
        "transplant_rms": 0.01441877251975918,
        "restore_mean": 0.001220703125,
        "transplant_mean": -0.0006103515625
      },
      "destinations": {
        "q17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.6436095489631011
        },
        "k17_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.5073273645971015
        },
        "k17_option_b": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 1.119203951240851
        },
        "q21_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.42308948139883945
        },
        "k21_decision": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 0.4836611714693561
        },
        "k21_option_b": {
          "fidelity": 0.05517040765755277,
          "restore_alignment": 0.04292639086372647,
          "transplant_alignment": 0.043595257274193146,
          "total_rms": 0.6253180298214283
        },
        "v11_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 3.7643307283321485
        },
        "v13_identity": {
          "fidelity": 0.0,
          "restore_alignment": 0.0,
          "transplant_alignment": 0.0,
          "total_rms": 5.597426489876519
        }
      }
    }
  ],
  "attention_layer": 18,
  "head_scan": [
    {
      "head": 0,
      "transmission": {
        "fidelity": 0.0007485726794875802,
        "restore_alignment": 0.0009888708001298365,
        "transplant_alignment": 7.5486320620598205e-06,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.013810679320049757,
        "transplant_rms": 0.013601941676937616,
        "restore_mean": -0.000244140625,
        "transplant_mean": 0.0006103515625
      },
      "signed_alignment": 0.0004982097160959482
    },
    {
      "head": 1,
      "transmission": {
        "fidelity": 0.0018154460109254256,
        "restore_alignment": -9.813221680677766e-05,
        "transplant_alignment": 0.0021941357193720544,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.014550453540772793,
        "transplant_rms": 0.014615849167085708,
        "restore_mean": -0.0001220703125,
        "transplant_mean": -0.001220703125
      },
      "signed_alignment": 0.0010480017512826383
    },
    {
      "head": 2,
      "transmission": {
        "fidelity": -0.27999637665661026,
        "restore_alignment": -0.12985660115292774,
        "transplant_alignment": -0.12693779675559794,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.13405610324919703,
        "transplant_rms": 0.13116510461235012,
        "restore_mean": 0.019287109375,
        "transplant_mean": 0.015625
      },
      "signed_alignment": -0.12839719895426283
    },
    {
      "head": 3,
      "transmission": {
        "fidelity": -0.0059495801702467155,
        "restore_alignment": -0.0040586478387008304,
        "transplant_alignment": -0.0016028262078440353,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.015440808887540914,
        "transplant_rms": 0.014084184669781208,
        "restore_mean": 0.0023193359375,
        "transplant_mean": 0.001220703125
      },
      "signed_alignment": -0.002830737023272433
    },
    {
      "head": 4,
      "transmission": {
        "fidelity": 0.0035969231775715427,
        "restore_alignment": 0.0009536438505068906,
        "transplant_alignment": 0.0029087395545803842,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.014285877771804486,
        "transplant_rms": 0.014084184669781208,
        "restore_mean": 0.0023193359375,
        "transplant_mean": -0.001220703125
      },
      "signed_alignment": 0.0019311917025436375
    },
    {
      "head": 5,
      "transmission": {
        "fidelity": 0.0028168978644920184,
        "restore_alignment": 0.0016128910505934482,
        "transplant_alignment": 0.001499661569662551,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.014484762670108717,
        "transplant_rms": 0.015440808887540914,
        "restore_mean": 0.001953125,
        "transplant_mean": -0.0018310546875
      },
      "signed_alignment": 0.0015562763101279998
    },
    {
      "head": 6,
      "transmission": {
        "fidelity": -0.13159027031651416,
        "restore_alignment": -0.06372303565722165,
        "transplant_alignment": -0.06120934118055573,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.07043446453225376,
        "transplant_rms": 0.07164269098029775,
        "restore_mean": 0.0125732421875,
        "transplant_mean": 0.0128173828125
      },
      "signed_alignment": -0.06246618841888869
    },
    {
      "head": 7,
      "transmission": {
        "fidelity": -0.05460051381022235,
        "restore_alignment": -0.02803310326780282,
        "transplant_alignment": -0.02452047314825765,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.04066535705792385,
        "transplant_rms": 0.03807341540941001,
        "restore_mean": 0.0057373046875,
        "transplant_mean": 0.00537109375
      },
      "signed_alignment": -0.026276788208030234
    },
    {
      "head": 8,
      "transmission": {
        "fidelity": 0.003189297046220285,
        "restore_alignment": 0.0008857061619483523,
        "transplant_alignment": 0.0025539538476635724,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.013460985109463107,
        "transplant_rms": 0.014084184669781208,
        "restore_mean": 0.0001220703125,
        "transplant_mean": -0.00146484375
      },
      "signed_alignment": 0.0017198300048059624
    },
    {
      "head": 9,
      "transmission": {
        "fidelity": 0.0013436565070465978,
        "restore_alignment": -0.000294396650420333,
        "transplant_alignment": 0.001912320122388488,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.014218964627501012,
        "transplant_rms": 0.014615849167085708,
        "restore_mean": 0.00146484375,
        "transplant_mean": 0.00048828125
      },
      "signed_alignment": 0.0008089617359840775
    },
    {
      "head": 10,
      "transmission": {
        "fidelity": 0.004712862617412661,
        "restore_alignment": 0.002380335310236197,
        "transplant_alignment": 0.0025967294293485783,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.014484762670108717,
        "transplant_rms": 0.013810679320049757,
        "restore_mean": 0.0009765625,
        "transplant_mean": 0.00048828125
      },
      "signed_alignment": 0.0024885323697923876
    },
    {
      "head": 11,
      "transmission": {
        "fidelity": 0.00208719676515956,
        "restore_alignment": 0.0009964194321918963,
        "transplant_alignment": 0.0013914645101063602,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.016046559301909163,
        "transplant_rms": 0.014084184669781208,
        "restore_mean": 0.0025634765625,
        "transplant_mean": -0.00048828125
      },
      "signed_alignment": 0.0011939419711491281
    },
    {
      "head": 12,
      "transmission": {
        "fidelity": -0.0013902064047626173,
        "restore_alignment": -0.0012077811299295712,
        "transplant_alignment": 0.00010568084886883749,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.014615849167085708,
        "transplant_rms": 0.014938533731228863,
        "restore_mean": 0.00048828125,
        "transplant_mean": -0.0015869140625
      },
      "signed_alignment": -0.0005510501405303668
    },
    {
      "head": 13,
      "transmission": {
        "fidelity": -0.002528791740790126,
        "restore_alignment": -0.000855511633700113,
        "transplant_alignment": -0.001195200076492805,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.019236050394133015,
        "transplant_rms": 0.018835255392564365,
        "restore_mean": -0.00146484375,
        "transplant_mean": -0.000244140625
      },
      "signed_alignment": -0.0010253558550964588
    },
    {
      "head": 14,
      "transmission": {
        "fidelity": 0.01782609461455431,
        "restore_alignment": 0.0076895398605516035,
        "transplant_alignment": 0.010605828047194048,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.0185803687406645,
        "transplant_rms": 0.019136638615493577,
        "restore_mean": -0.0008544921875,
        "transplant_mean": -0.0029296875
      },
      "signed_alignment": 0.009147683953872826
    },
    {
      "head": 15,
      "transmission": {
        "fidelity": 0.110250287477071,
        "restore_alignment": 0.058713260178701286,
        "transplant_alignment": 0.06376077881753195,
        "total_rms": 0.8706458612422402,
        "restore_rms": 0.09486237847328821,
        "transplant_rms": 0.09763671679648428,
        "restore_mean": -0.0078125,
        "transplant_mean": -0.0078125
      },
      "signed_alignment": 0.06123701949811662
    }
  ],
  "groups": {
    "positive": [
      15,
      14,
      10,
      4
    ],
    "negative": [
      2,
      6,
      7,
      3
    ]
  },
  "validation": {
    "positive": {
      "heads": [
        15,
        14,
        10,
        4
      ],
      "transmission": {
        "fidelity": 0.1522079610956798,
        "restore_alignment": 0.08441342515504494,
        "transplant_alignment": 0.08182244093952845,
        "total_rms": 0.8783412897761801,
        "restore_rms": 0.10476108071799911,
        "transplant_rms": 0.10329427986076056,
        "restore_mean": -0.0098876953125,
        "transplant_mean": -0.0107421875
      },
      "restore_metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.796875,
        "both_correct": 0.6875,
        "flip": 0.21875,
        "gap": 0.109375,
        "by_permutation": {
          "0": {
            "accuracy": 0.8359375,
            "both_correct": 0.734375,
            "flip": 0.203125,
            "gap": 0.078125
          },
          "1": {
            "accuracy": 0.7578125,
            "both_correct": 0.640625,
            "flip": 0.234375,
            "gap": 0.140625
          }
        },
        "gold_margin": 3.7357177734375
      },
      "transplant_metrics": {
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
        "gold_margin": 3.58197021484375
      }
    },
    "negative": {
      "heads": [
        2,
        6,
        7,
        3
      ],
      "transmission": {
        "fidelity": -0.3872717900412752,
        "restore_alignment": -0.16694912844779525,
        "transplant_alignment": -0.17894973663683822,
        "total_rms": 0.8783412897761801,
        "restore_rms": 0.17313271625826485,
        "transplant_rms": 0.18401661325172985,
        "restore_mean": 0.00372314453125,
        "transplant_mean": 0.00469970703125
      },
      "restore_metrics": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.796875,
        "both_correct": 0.703125,
        "flip": 0.1875,
        "gap": 0.09375,
        "by_permutation": {
          "0": {
            "accuracy": 0.84375,
            "both_correct": 0.765625,
            "flip": 0.15625,
            "gap": 0.0625
          },
          "1": {
            "accuracy": 0.75,
            "both_correct": 0.640625,
            "flip": 0.21875,
            "gap": 0.125
          }
        },
        "gold_margin": 3.76153564453125
      },
      "transplant_metrics": {
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
        "gold_margin": 3.5556640625
      }
    }
  },
  "status": "supplementary exploratory; no new deployment selection"
}

## part_controls

{
  "metrics": {
    "coref": {
      "baseline": {
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
      },
      "source_half": {
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
      },
      "selected_unit": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.79296875,
        "both_correct": 0.6796875,
        "flip": 0.2265625,
        "gap": 0.1171875,
        "by_permutation": {
          "0": {
            "accuracy": 0.8515625,
            "both_correct": 0.765625,
            "flip": 0.171875,
            "gap": 0.078125
          },
          "1": {
            "accuracy": 0.734375,
            "both_correct": 0.59375,
            "flip": 0.28125,
            "gap": 0.15625
          }
        },
        "gold_margin": 3.87811279296875
      },
      "matched_unit": {
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
        "gold_margin": 3.6181640625
      },
      "whole_raw": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.74609375,
        "both_correct": 0.59375,
        "flip": 0.3046875,
        "gap": 0.1328125,
        "by_permutation": {
          "0": {
            "accuracy": 0.8125,
            "both_correct": 0.6875,
            "flip": 0.25,
            "gap": 0.09375
          },
          "1": {
            "accuracy": 0.6796875,
            "both_correct": 0.5,
            "flip": 0.359375,
            "gap": 0.171875
          }
        },
        "gold_margin": 3.293212890625
      },
      "whole_centered": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.7890625,
        "both_correct": 0.6796875,
        "flip": 0.21875,
        "gap": 0.109375,
        "by_permutation": {
          "0": {
            "accuracy": 0.84375,
            "both_correct": 0.75,
            "flip": 0.1875,
            "gap": 0.09375
          },
          "1": {
            "accuracy": 0.734375,
            "both_correct": 0.609375,
            "flip": 0.25,
            "gap": 0.125
          }
        },
        "gold_margin": 3.90155029296875
      }
    },
    "reading": {
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
      "selected_unit": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.89453125,
        "both_correct": 0.7890625,
        "flip": 0.7890625,
        "gap": 0.0390625,
        "by_permutation": {
          "0": {
            "accuracy": 0.8515625,
            "both_correct": 0.703125,
            "flip": 0.703125,
            "gap": 0.046875
          },
          "1": {
            "accuracy": 0.9375,
            "both_correct": 0.875,
            "flip": 0.875,
            "gap": 0.03125
          }
        },
        "gold_margin": 4.8978271484375
      },
      "matched_unit": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.90234375,
        "both_correct": 0.8046875,
        "flip": 0.8046875,
        "gap": 0.0546875,
        "by_permutation": {
          "0": {
            "accuracy": 0.8671875,
            "both_correct": 0.734375,
            "flip": 0.734375,
            "gap": 0.078125
          },
          "1": {
            "accuracy": 0.9375,
            "both_correct": 0.875,
            "flip": 0.875,
            "gap": 0.03125
          }
        },
        "gold_margin": 4.86883544921875
      },
      "whole_raw": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.90234375,
        "both_correct": 0.8046875,
        "flip": 0.8046875,
        "gap": 0.0390625,
        "by_permutation": {
          "0": {
            "accuracy": 0.859375,
            "both_correct": 0.71875,
            "flip": 0.71875,
            "gap": 0.0625
          },
          "1": {
            "accuracy": 0.9453125,
            "both_correct": 0.890625,
            "flip": 0.890625,
            "gap": 0.015625
          }
        },
        "gold_margin": 5.115234375
      },
      "whole_centered": {
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
        "gold_margin": 4.92724609375
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
      "selected_unit": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.99609375,
        "both_correct": 0.9921875,
        "flip": 0.9921875,
        "gap": 0.0078125,
        "by_permutation": {
          "0": {
            "accuracy": 0.9921875,
            "both_correct": 0.984375,
            "flip": 0.984375,
            "gap": 0.015625
          },
          "1": {
            "accuracy": 1.0,
            "both_correct": 1.0,
            "flip": 1.0,
            "gap": 0.0
          }
        },
        "gold_margin": 4.76904296875
      },
      "matched_unit": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.984375,
        "both_correct": 0.96875,
        "flip": 0.96875,
        "gap": 0.015625,
        "by_permutation": {
          "0": {
            "accuracy": 0.9765625,
            "both_correct": 0.953125,
            "flip": 0.953125,
            "gap": 0.015625
          },
          "1": {
            "accuracy": 0.9921875,
            "both_correct": 0.984375,
            "flip": 0.984375,
            "gap": 0.015625
          }
        },
        "gold_margin": 4.68341064453125
      },
      "whole_raw": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.98828125,
        "both_correct": 0.9765625,
        "flip": 0.9765625,
        "gap": 0.0234375,
        "by_permutation": {
          "0": {
            "accuracy": 0.9921875,
            "both_correct": 0.984375,
            "flip": 0.984375,
            "gap": 0.015625
          },
          "1": {
            "accuracy": 0.984375,
            "both_correct": 0.96875,
            "flip": 0.96875,
            "gap": 0.03125
          }
        },
        "gold_margin": 4.60382080078125
      },
      "whole_centered": {
        "rows": 256,
        "pairs": 64,
        "accuracy": 0.9921875,
        "both_correct": 0.984375,
        "flip": 0.984375,
        "gap": 0.0,
        "by_permutation": {
          "0": {
            "accuracy": 0.984375,
            "both_correct": 0.96875,
            "flip": 0.96875,
            "gap": 0.0
          },
          "1": {
            "accuracy": 1.0,
            "both_correct": 1.0,
            "flip": 1.0,
            "gap": 0.0
          }
        },
        "gold_margin": 4.771240234375
      }
    }
  },
  "groups": {
    "gap_pos": {
      "selected_indices": [
        4075,
        105,
        5027,
        5252,
        3714,
        1692,
        5761,
        5243,
        3110,
        2721,
        1024,
        1852,
        5626,
        827,
        2451,
        5771,
        5703,
        714,
        2137,
        4590,
        1719,
        23,
        178,
        3700,
        4546,
        5463,
        3243,
        349,
        3830,
        3252,
        1498,
        2213,
        3089,
        901,
        362,
        1168,
        5168,
        4163,
        1229,
        4483,
        2769,
        2862,
        4280,
        957,
        2487,
        298,
        2498,
        172,
        5090,
        2924,
        5208,
        2107,
        1823,
        219,
        4927,
        2108,
        3363,
        56,
        4934,
        635,
        5037,
        4151,
        6108,
        4563
      ],
      "matched_indices": [
        5257,
        118,
        6085,
        5071,
        1892,
        6072,
        3223,
        3279,
        2011,
        4313,
        1050,
        2221,
        2942,
        5067,
        6083,
        1435,
        5370,
        2291,
        5192,
        2121,
        3302,
        2954,
        3641,
        5420,
        2357,
        3962,
        1490,
        2455,
        5648,
        1146,
        3378,
        3014,
        5491,
        411,
        4222,
        3042,
        1777,
        216,
        4461,
        4678,
        3282,
        4114,
        2971,
        851,
        2392,
        1817,
        6021,
        4631,
        3168,
        2400,
        1833,
        95,
        2740,
        3811,
        5162,
        1295,
        2752,
        3181,
        1598,
        4627,
        766,
        1693,
        2945,
        1909
      ],
      "amount": 0.5,
      "selected_rms": 0.8630609097135358,
      "matched_rms": 0.8197059879656136,
      "selected_column_norm": 1.3317204713821411,
      "matched_column_norm": 1.3041129112243652
    }
  },
  "status": "same-site, same-budget fit-matched control; no target selection"
}

## Independent verification

{
  "records_recomputed": 29368,
  "source_inputs_unchanged": true,
  "implementation_unchanged": true,
  "parameters_unchanged": true,
  "historical_exploratory": true,
  "independent_records_recomputed": 85604,
  "fit_signs_and_subspaces_recomputed": true,
  "selection_recomputed": true,
  "all_recorded_behavior_metrics_match": true,
  "routes_source_hashes_match": true
}


Final diagnostic verification:

{
  "records_recomputed": 29368,
  "source_inputs_unchanged": true,
  "implementation_unchanged": true,
  "parameters_unchanged": true,
  "historical_exploratory": true,
  "independent_records_recomputed": 85604,
  "fit_signs_and_subspaces_recomputed": true,
  "selection_recomputed": true,
  "all_recorded_behavior_metrics_match": true,
  "routes_source_hashes_match": true,
  "diagnostic_records_recomputed": 14848,
  "diagnostic_metrics_and_transmission_match": true,
  "parameter_metadata_unchanged": true,
  "supplementary_head_group_ranking_recomputed": true
}
