# Identity interference pilot results

Agent text review; not independent human annotation. Native WinoBias development families are newly evaluated; Winogender is previously inspected exploratory validation.

Data: {'fit': 96, 'dev': 49, 'confirmation': 146, 'external': 240}
Selected: site2_scale_0.5 {'site': {'layer': 7, 'module': 'mlp', 'position': 'identity'}, 'site_index': 2, 'operation': 'scale', 'strength': 0.5}

| Method | Native accuracy | Both genders correct | Identity flip | Reading | Explicit identity binding | Role-swap accuracy | Both native and swapped evidence correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 76.71% | 65.75% | 21.92% | 67.98% | 98.97% | 74.83% | 36.64% |
| selected | 79.11% | 70.21% | 17.81% | 66.95% | 99.49% | 78.60% | 44.86% |
| instruction | 96.40% | 92.81% | 7.19% | 63.18% | 100.00% | 96.23% | 85.62% |
| fairsteer | 78.42% | 67.47% | 21.92% | 67.81% | 99.49% | 76.20% | 39.38% |
| evidence | 76.71% | 65.75% | 21.92% | 67.98% | 98.97% | 74.66% | 36.64% |
| random_0 | 76.71% | 65.75% | 21.92% | 67.98% | 98.97% | 74.66% | 36.64% |
| random_1 | 76.71% | 65.75% | 21.92% | 67.98% | 98.97% | 74.83% | 36.64% |
| random_2 | 77.40% | 66.10% | 22.60% | 68.66% | 99.14% | 75.68% | 36.64% |

Selected versus baseline, family cluster bootstrap:
{
  "coref": {
    "accuracy": {
      "delta": 0.023972602739726026,
      "ci95": [
        0.00684931506849315,
        0.04245360093723024
      ],
      "cluster": "family_id",
      "clusters": 129,
      "p_sign": 0.01949512621844539
    },
    "both_correct": {
      "delta": 0.04452054794520548,
      "ci95": [
        0.017479482323232324,
        0.07291666666666667
      ],
      "cluster": "family_id",
      "clusters": 129,
      "p_sign": 0.00424893776555861
    },
    "flip": {
      "delta": -0.0410958904109589,
      "ci95": [
        -0.0738255033557047,
        -0.01020408163265306
      ],
      "cluster": "family_id",
      "clusters": 129,
      "p_sign": 0.02024493876530867
    },
    "gap": {
      "delta": -0.0410958904109589,
      "ci95": [
        -0.07394366197183098,
        -0.010344827586206896
      ],
      "cluster": "family_id",
      "clusters": 129,
      "p_sign": 0.021494626343414147
    },
    "occupation_cluster_both_correct": {
      "delta": 0.04452054794520548,
      "ci95": [
        0.016556291390728478,
        0.07516787536932583
      ],
      "cluster": "occupation_pair",
      "clusters": 110,
      "p_sign": 0.008247938015496127
    }
  },
  "reading": {
    "accuracy": {
      "delta": -0.010273972602739725,
      "ci95": [
        -0.023809523809523808,
        0.0034965034965034965
      ],
      "cluster": "family_id",
      "clusters": 129,
      "p_sign": 0.20094976255936017
    }
  },
  "binding": {
    "accuracy": {
      "delta": 0.005136986301369863,
      "ci95": [
        0.0,
        0.011824324324324325
      ],
      "cluster": "family_id",
      "clusters": 129,
      "p_sign": 0.2551862034491377
    }
  },
  "role_swap": {
    "accuracy": {
      "delta": 0.03767123287671233,
      "ci95": [
        0.01883241310222719,
        0.05650684931506849
      ],
      "cluster": "family_id",
      "clusters": 129,
      "p_sign": 0.0007498125468632841
    }
  }
}

Short actual generation:
{
  "baseline": {
    "strict": {
      "rows": 292,
      "pairs": 146,
      "accuracy": 0.0,
      "both_correct": 0.0,
      "flip": 0.0,
      "gap": 0.0,
      "by_permutation": {
        "0": {
          "accuracy": 0.0,
          "both_correct": 0.0,
          "flip": 0.0,
          "gap": 0.0
        }
      }
    },
    "unique_entity": {
      "rows": 292,
      "pairs": 146,
      "accuracy": 0.1232876712328767,
      "both_correct": 0.08904109589041095,
      "flip": 0.08904109589041095,
      "gap": 0.0273972602739726,
      "by_permutation": {
        "0": {
          "accuracy": 0.1232876712328767,
          "both_correct": 0.08904109589041095,
          "flip": 0.08904109589041095,
          "gap": 0.0273972602739726
        }
      }
    },
    "strict_unparsed": 1.0
  },
  "selected": {
    "strict": {
      "rows": 292,
      "pairs": 146,
      "accuracy": 0.0,
      "both_correct": 0.0,
      "flip": 0.0,
      "gap": 0.0,
      "by_permutation": {
        "0": {
          "accuracy": 0.0,
          "both_correct": 0.0,
          "flip": 0.0,
          "gap": 0.0
        }
      }
    },
    "unique_entity": {
      "rows": 292,
      "pairs": 146,
      "accuracy": 0.1095890410958904,
      "both_correct": 0.07534246575342465,
      "flip": 0.08904109589041095,
      "gap": 0.0136986301369863,
      "by_permutation": {
        "0": {
          "accuracy": 0.1095890410958904,
          "both_correct": 0.07534246575342465,
          "flip": 0.08904109589041095,
          "gap": 0.0136986301369863
        }
      }
    },
    "strict_unparsed": 1.0
  }
}

Short generation differences:
{
  "strict": {
    "accuracy": {
      "delta": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "cluster": "family_id",
      "clusters": 129,
      "p_sign": 1.0
    },
    "both_correct": {
      "delta": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "cluster": "family_id",
      "clusters": 129,
      "p_sign": 1.0
    }
  },
  "unique_entity": {
    "accuracy": {
      "delta": -0.0136986301369863,
      "ci95": [
        -0.027972027972027972,
        -0.003289473684210526
      ],
      "cluster": "family_id",
      "clusters": 129,
      "p_sign": 0.12521869532616844
    },
    "both_correct": {
      "delta": -0.0136986301369863,
      "ci95": [
        -0.034482758620689655,
        0.0
      ],
      "cluster": "family_id",
      "clusters": 129,
      "p_sign": 0.4936265933516621
    }
  }
}

Independent recomputation checked 26368 item predictions. Identity-fact clause order: {'woman_first': 308, 'woman_last': 276}.

Limits: one model, agent annotation, simple explicit fact controls; native coreference role exchange does not establish universal evidence following. Oracle donor recovery cannot establish a deployable bias correction.

## Answer-line format correction (post-hoc, exploratory)
The original whole-completion strict scores are all zero because generations continue after the first answer line; they are NOT evidence of zero semantic task accuracy. Preserve original metrics, and separately apply exact first-nonempty-line occupation matching or an anchored pronoun-refers-to sentence. The parser never scans subsequent lines or uses gold.

| Method | Answer-line accuracy | Both genders correct | Unparsed |
|---|---:|---:|---:|
| baseline | 79.79% | 71.92% | 9.59% |
| selected | 81.85% | 76.03% | 9.93% |
| instruction | 62.33% | 57.53% | 21.92% |

{
  "selected": {
    "accuracy": {
      "delta": 0.02054794520547945,
      "ci95": [
        0.003472222222222222,
        0.039735099337748346
      ],
      "cluster": "family_id",
      "clusters": 129,
      "p_sign": 0.06923269182704324
    },
    "both_correct": {
      "delta": 0.0410958904109589,
      "ci95": [
        0.0136986301369863,
        0.07534246575342465
      ],
      "cluster": "family_id",
      "clusters": 129,
      "p_sign": 0.03424143964008998
    },
    "flip": {
      "delta": -0.0410958904109589,
      "ci95": [
        -0.08053691275167785,
        -0.006896551724137931
      ],
      "cluster": "family_id",
      "clusters": 129,
      "p_sign": 0.06973256685828542
    }
  },
  "instruction": {
    "accuracy": {
      "delta": -0.17465753424657535,
      "ci95": [
        -0.24482758620689654,
        -0.10810810810810811
      ],
      "cluster": "family_id",
      "clusters": 129,
      "p_sign": 0.00024993751562109475
    },
    "both_correct": {
      "delta": -0.14383561643835616,
      "ci95": [
        -0.22377622377622378,
        -0.0657870726085286
      ],
      "cluster": "family_id",
      "clusters": 129,
      "p_sign": 0.0017495626093476631
    },
    "flip": {
      "delta": -0.0410958904109589,
      "ci95": [
        -0.11564625850340136,
        0.03448874521072799
      ],
      "cluster": "family_id",
      "clusters": 129,
      "p_sign": 0.387903024243939
    }
  }
}

Literal pronoun reading: secondary control added after selection-set metalinguistic performance inspection, with no reselection.
{
  "status": "secondary control, added after selection-set metalinguistic baseline inspection; no reselection",
  "metrics": {
    "baseline": {
      "rows": 584,
      "pairs": 146,
      "accuracy": 0.964041095890411,
      "both_correct": 0.928082191780822,
      "flip": 0.928082191780822,
      "gap": 0.030821917808219176,
      "by_permutation": {
        "0": {
          "accuracy": 0.934931506849315,
          "both_correct": 0.8698630136986302,
          "flip": 0.8698630136986302,
          "gap": 0.04794520547945205
        },
        "1": {
          "accuracy": 0.9931506849315068,
          "both_correct": 0.9863013698630136,
          "flip": 0.9863013698630136,
          "gap": 0.0136986301369863
        }
      },
      "gold_margin": 7.339014530181885
    },
    "selected": {
      "rows": 584,
      "pairs": 146,
      "accuracy": 0.958904109589041,
      "both_correct": 0.9178082191780822,
      "flip": 0.9178082191780822,
      "gap": 0.02054794520547945,
      "by_permutation": {
        "0": {
          "accuracy": 0.9246575342465754,
          "both_correct": 0.8493150684931506,
          "flip": 0.8493150684931506,
          "gap": 0.0273972602739726
        },
        "1": {
          "accuracy": 0.9931506849315068,
          "both_correct": 0.9863013698630136,
          "flip": 0.9863013698630136,
          "gap": 0.0136986301369863
        }
      },
      "gold_margin": 7.320794105529785
    },
    "instruction": {
      "rows": 584,
      "pairs": 146,
      "accuracy": 0.952054794520548,
      "both_correct": 0.9041095890410958,
      "flip": 0.9041095890410958,
      "gap": 0.0410958904109589,
      "by_permutation": {
        "0": {
          "accuracy": 0.9041095890410958,
          "both_correct": 0.8082191780821918,
          "flip": 0.8082191780821918,
          "gap": 0.0821917808219178
        },
        "1": {
          "accuracy": 1.0,
          "both_correct": 1.0,
          "flip": 1.0,
          "gap": 0.0
        }
      },
      "gold_margin": 5.93910551071167
    },
    "fairsteer": {
      "rows": 584,
      "pairs": 146,
      "accuracy": 0.964041095890411,
      "both_correct": 0.928082191780822,
      "flip": 0.928082191780822,
      "gap": 0.030821917808219176,
      "by_permutation": {
        "0": {
          "accuracy": 0.934931506849315,
          "both_correct": 0.8698630136986302,
          "flip": 0.8698630136986302,
          "gap": 0.04794520547945205
        },
        "1": {
          "accuracy": 0.9931506849315068,
          "both_correct": 0.9863013698630136,
          "flip": 0.9863013698630136,
          "gap": 0.0136986301369863
        }
      },
      "gold_margin": 7.339014530181885
    },
    "evidence": {
      "rows": 584,
      "pairs": 146,
      "accuracy": 0.964041095890411,
      "both_correct": 0.928082191780822,
      "flip": 0.928082191780822,
      "gap": 0.030821917808219176,
      "by_permutation": {
        "0": {
          "accuracy": 0.934931506849315,
          "both_correct": 0.8698630136986302,
          "flip": 0.8698630136986302,
          "gap": 0.04794520547945205
        },
        "1": {
          "accuracy": 0.9931506849315068,
          "both_correct": 0.9863013698630136,
          "flip": 0.9863013698630136,
          "gap": 0.0136986301369863
        }
      },
      "gold_margin": 7.339014530181885
    },
    "random_0": {
      "rows": 584,
      "pairs": 146,
      "accuracy": 0.964041095890411,
      "both_correct": 0.928082191780822,
      "flip": 0.928082191780822,
      "gap": 0.030821917808219176,
      "by_permutation": {
        "0": {
          "accuracy": 0.934931506849315,
          "both_correct": 0.8698630136986302,
          "flip": 0.8698630136986302,
          "gap": 0.04794520547945205
        },
        "1": {
          "accuracy": 0.9931506849315068,
          "both_correct": 0.9863013698630136,
          "flip": 0.9863013698630136,
          "gap": 0.0136986301369863
        }
      },
      "gold_margin": 7.3392014503479
    },
    "random_1": {
      "rows": 584,
      "pairs": 146,
      "accuracy": 0.964041095890411,
      "both_correct": 0.928082191780822,
      "flip": 0.928082191780822,
      "gap": 0.030821917808219176,
      "by_permutation": {
        "0": {
          "accuracy": 0.934931506849315,
          "both_correct": 0.8698630136986302,
          "flip": 0.8698630136986302,
          "gap": 0.04794520547945205
        },
        "1": {
          "accuracy": 0.9931506849315068,
          "both_correct": 0.9863013698630136,
          "flip": 0.9863013698630136,
          "gap": 0.0136986301369863
        }
      },
      "gold_margin": 7.338452339172363
    },
    "random_2": {
      "rows": 584,
      "pairs": 146,
      "accuracy": 0.9674657534246576,
      "both_correct": 0.934931506849315,
      "flip": 0.934931506849315,
      "gap": 0.030821917808219176,
      "by_permutation": {
        "0": {
          "accuracy": 0.9417808219178082,
          "both_correct": 0.8835616438356164,
          "flip": 0.8835616438356164,
          "gap": 0.04794520547945205
        },
        "1": {
          "accuracy": 0.9931506849315068,
          "both_correct": 0.9863013698630136,
          "flip": 0.9863013698630136,
          "gap": 0.0136986301369863
        }
      },
      "gold_margin": 7.555516719818115
    },
    "selected_difference": {
      "delta": -0.005136986301369863,
      "ci95": [
        -0.013422818791946308,
        0.0017605633802816902
      ],
      "cluster": "family_id",
      "clusters": 129,
      "p_sign": 0.38315421144713824
    }
  }
}

## Native-chat short generation (exploratory)
Native tokenizer chat template, enable_thinking=False, greedy max32, frozen component/strength, prefill-only intervention. Primary parsing is whole-completion exact occupation. This adapter condition was added after raw-generation format problems; it is not a newly untouched confirmation dataset.

| Method | Strict accuracy | Both genders correct | Identity flip | Unparsed |
|---|---:|---:|---:|---:|
| baseline | 86.30% | 80.14% | 13.01% | 3.77% |
| selected | 88.01% | 84.25% | 8.90% | 4.45% |
| instruction | 75.34% | 69.86% | 10.96% | 2.74% |

{
  "selected": {
    "strict": {
      "accuracy": {
        "delta": 0.017123287671232876,
        "ci95": [
          -0.003246753246753247,
          0.03767123287671233
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.18345413646588352
      },
      "both_correct": {
        "delta": 0.0410958904109589,
        "ci95": [
          0.013513513513513514,
          0.07534246575342465
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.03749062734316421
      },
      "flip": {
        "delta": -0.0410958904109589,
        "ci95": [
          -0.08109487038058466,
          -0.006802721088435374
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.0757310672331917
      }
    },
    "answer_line": {
      "accuracy": {
        "delta": 0.017123287671232876,
        "ci95": [
          -0.003246753246753247,
          0.03767123287671233
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.18345413646588352
      },
      "both_correct": {
        "delta": 0.0410958904109589,
        "ci95": [
          0.013513513513513514,
          0.07534246575342465
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.03749062734316421
      },
      "flip": {
        "delta": -0.0410958904109589,
        "ci95": [
          -0.08109487038058466,
          -0.006802721088435374
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.0757310672331917
      }
    }
  },
  "instruction": {
    "strict": {
      "accuracy": {
        "delta": -0.1095890410958904,
        "ci95": [
          -0.17123287671232876,
          -0.05066777658953915
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.000999750062484379
      },
      "both_correct": {
        "delta": -0.10273972602739725,
        "ci95": [
          -0.176056338028169,
          -0.0328893618851049
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.011747063234191453
      },
      "flip": {
        "delta": -0.02054794520547945,
        "ci95": [
          -0.08333333333333333,
          0.04081632653061224
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.6540864783804049
      }
    },
    "answer_line": {
      "accuracy": {
        "delta": -0.11301369863013698,
        "ci95": [
          -0.1724137931034483,
          -0.056328178863390095
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.00024993751562109475
      },
      "both_correct": {
        "delta": -0.10273972602739725,
        "ci95": [
          -0.176056338028169,
          -0.02919179096583093
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.012246938265433641
      },
      "flip": {
        "delta": -0.02054794520547945,
        "ci95": [
          -0.08333333333333333,
          0.04081632653061224
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.6540864783804049
      }
    }
  }
}

Oracle patch confirmation:
{
  "baseline": {
    "rows": 292,
    "pairs": 146,
    "accuracy": 0.791095890410959,
    "both_correct": 0.6917808219178082,
    "flip": 0.19863013698630136,
    "gap": 0.07534246575342465,
    "by_permutation": {
      "0": {
        "accuracy": 0.791095890410959,
        "both_correct": 0.6917808219178082,
        "flip": 0.19863013698630136,
        "gap": 0.07534246575342465
      }
    },
    "gold_margin": 3.867990255355835
  },
  "discordant_pairs": 29,
  "results": [
    {
      "site": {
        "layer": 0,
        "module": "mlp",
        "position": "identity"
      },
      "top_index": 0,
      "control": "top",
      "donor": "counterpart",
      "bad_rescue_margin": 3.618534564971924,
      "good_reverse_margin": -3.409482717514038,
      "bad_fixed": 18,
      "good_broken": 12,
      "prediction_change": 0.18493150684931506,
      "metrics": {
        "rows": 292,
        "pairs": 146,
        "accuracy": 0.7363013698630136,
        "both_correct": 0.6232876712328768,
        "flip": 0.22602739726027396,
        "gap": 0.07534246575342465,
        "by_permutation": {
          "0": {
            "accuracy": 0.7363013698630136,
            "both_correct": 0.6232876712328768,
            "flip": 0.22602739726027396,
            "gap": 0.07534246575342465
          }
        },
        "gold_margin": 2.66668438911438
      },
      "both_correct_difference": {
        "delta": -0.0684931506849315,
        "ci95": [
          -0.1486558366877516,
          0.0070934650455927095
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.137215696075981
      }
    },
    {
      "site": {
        "layer": 0,
        "module": "mlp",
        "position": "identity"
      },
      "top_index": 0,
      "control": "top",
      "donor": "neutral",
      "bad_rescue_margin": 0.0,
      "good_reverse_margin": 0.0,
      "bad_fixed": 0,
      "good_broken": 0,
      "prediction_change": 0.0,
      "metrics": {
        "rows": 292,
        "pairs": 146,
        "accuracy": 0.791095890410959,
        "both_correct": 0.6917808219178082,
        "flip": 0.19863013698630136,
        "gap": 0.07534246575342465,
        "by_permutation": {
          "0": {
            "accuracy": 0.791095890410959,
            "both_correct": 0.6917808219178082,
            "flip": 0.19863013698630136,
            "gap": 0.07534246575342465
          }
        },
        "gold_margin": 3.867990255355835
      },
      "both_correct_difference": {
        "delta": 0.0,
        "ci95": [
          0.0,
          0.0
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 1.0
      }
    },
    {
      "site": {
        "layer": 25,
        "module": "mlp",
        "position": "identity"
      },
      "top_index": 0,
      "control": "random",
      "donor": "counterpart",
      "bad_rescue_margin": -0.004310344811528921,
      "good_reverse_margin": 0.0016163793625310063,
      "bad_fixed": 0,
      "good_broken": 0,
      "prediction_change": 0.0,
      "metrics": {
        "rows": 292,
        "pairs": 146,
        "accuracy": 0.791095890410959,
        "both_correct": 0.6917808219178082,
        "flip": 0.19863013698630136,
        "gap": 0.07534246575342465,
        "by_permutation": {
          "0": {
            "accuracy": 0.791095890410959,
            "both_correct": 0.6917808219178082,
            "flip": 0.19863013698630136,
            "gap": 0.07534246575342465
          }
        },
        "gold_margin": 3.866919994354248
      },
      "both_correct_difference": {
        "delta": 0.0,
        "ci95": [
          0.0,
          0.0
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 1.0
      }
    },
    {
      "site": {
        "layer": 25,
        "module": "mlp",
        "position": "identity"
      },
      "top_index": 0,
      "control": "random",
      "donor": "neutral",
      "bad_rescue_margin": 0.0,
      "good_reverse_margin": 0.0,
      "bad_fixed": 0,
      "good_broken": 0,
      "prediction_change": 0.0,
      "metrics": {
        "rows": 292,
        "pairs": 146,
        "accuracy": 0.791095890410959,
        "both_correct": 0.6917808219178082,
        "flip": 0.19863013698630136,
        "gap": 0.07534246575342465,
        "by_permutation": {
          "0": {
            "accuracy": 0.791095890410959,
            "both_correct": 0.6917808219178082,
            "flip": 0.19863013698630136,
            "gap": 0.07534246575342465
          }
        },
        "gold_margin": 3.8680436611175537
      },
      "both_correct_difference": {
        "delta": 0.0,
        "ci95": [
          0.0,
          0.0
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 1.0
      }
    },
    {
      "site": {
        "layer": 21,
        "module": "attention",
        "position": "decision"
      },
      "top_index": 1,
      "control": "top",
      "donor": "counterpart",
      "bad_rescue_margin": 3.975215435028076,
      "good_reverse_margin": -4.364762783050537,
      "bad_fixed": 28,
      "good_broken": 17,
      "prediction_change": 0.15753424657534246,
      "metrics": {
        "rows": 292,
        "pairs": 146,
        "accuracy": 0.8253424657534246,
        "both_correct": 0.7671232876712328,
        "flip": 0.11643835616438356,
        "gap": -0.02054794520547945,
        "by_permutation": {
          "0": {
            "accuracy": 0.8253424657534246,
            "both_correct": 0.7671232876712328,
            "flip": 0.11643835616438356,
            "gap": -0.02054794520547945
          }
        },
        "gold_margin": 3.8225064277648926
      },
      "both_correct_difference": {
        "delta": 0.07534246575342465,
        "ci95": [
          0.031188174460431674,
          0.125
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.003749062734316421
      }
    },
    {
      "site": {
        "layer": 21,
        "module": "attention",
        "position": "decision"
      },
      "top_index": 1,
      "control": "top",
      "donor": "neutral",
      "bad_rescue_margin": -0.4482758641242981,
      "good_reverse_margin": -0.14762930572032928,
      "bad_fixed": 1,
      "good_broken": 2,
      "prediction_change": 0.0410958904109589,
      "metrics": {
        "rows": 292,
        "pairs": 146,
        "accuracy": 0.7773972602739726,
        "both_correct": 0.6575342465753424,
        "flip": 0.23972602739726026,
        "gap": 0.07534246575342465,
        "by_permutation": {
          "0": {
            "accuracy": 0.7773972602739726,
            "both_correct": 0.6575342465753424,
            "flip": 0.23972602739726026,
            "gap": 0.07534246575342465
          }
        },
        "gold_margin": 3.7509632110595703
      },
      "both_correct_difference": {
        "delta": -0.03424657534246575,
        "ci95": [
          -0.06711409395973154,
          -0.006944444444444444
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.06573356660834791
      }
    },
    {
      "site": {
        "layer": 22,
        "module": "attention",
        "position": "decision"
      },
      "top_index": 1,
      "control": "random",
      "donor": "counterpart",
      "bad_rescue_margin": -0.1923491358757019,
      "good_reverse_margin": 0.13038793206214905,
      "bad_fixed": 0,
      "good_broken": 0,
      "prediction_change": 0.00684931506849315,
      "metrics": {
        "rows": 292,
        "pairs": 146,
        "accuracy": 0.7842465753424658,
        "both_correct": 0.678082191780822,
        "flip": 0.21232876712328766,
        "gap": 0.08904109589041095,
        "by_permutation": {
          "0": {
            "accuracy": 0.7842465753424658,
            "both_correct": 0.678082191780822,
            "flip": 0.21232876712328766,
            "gap": 0.08904109589041095
          }
        },
        "gold_margin": 3.8470141887664795
      },
      "both_correct_difference": {
        "delta": -0.0136986301369863,
        "ci95": [
          -0.03424657534246575,
          0.0
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.48962759310172455
      }
    },
    {
      "site": {
        "layer": 22,
        "module": "attention",
        "position": "decision"
      },
      "top_index": 1,
      "control": "random",
      "donor": "neutral",
      "bad_rescue_margin": -0.009159483015537262,
      "good_reverse_margin": 0.058728449046611786,
      "bad_fixed": 1,
      "good_broken": 0,
      "prediction_change": 0.003424657534246575,
      "metrics": {
        "rows": 292,
        "pairs": 146,
        "accuracy": 0.7945205479452054,
        "both_correct": 0.6986301369863014,
        "flip": 0.1917808219178082,
        "gap": 0.0684931506849315,
        "by_permutation": {
          "0": {
            "accuracy": 0.7945205479452054,
            "both_correct": 0.6986301369863014,
            "flip": 0.1917808219178082,
            "gap": 0.0684931506849315
          }
        },
        "gold_margin": 3.9172730445861816
      },
      "both_correct_difference": {
        "delta": 0.00684931506849315,
        "ci95": [
          0.0,
          0.02142857142857143
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 1.0
      }
    },
    {
      "site": {
        "layer": 7,
        "module": "mlp",
        "position": "identity"
      },
      "top_index": 2,
      "control": "top",
      "donor": "counterpart",
      "bad_rescue_margin": 2.379310369491577,
      "good_reverse_margin": -1.1174569129943848,
      "bad_fixed": 14,
      "good_broken": 4,
      "prediction_change": 0.09246575342465753,
      "metrics": {
        "rows": 292,
        "pairs": 146,
        "accuracy": 0.8082191780821918,
        "both_correct": 0.7328767123287672,
        "flip": 0.1506849315068493,
        "gap": 0.0547945205479452,
        "by_permutation": {
          "0": {
            "accuracy": 0.8082191780821918,
            "both_correct": 0.7328767123287672,
            "flip": 0.1506849315068493,
            "gap": 0.0547945205479452
          }
        },
        "gold_margin": 3.853114366531372
      },
      "both_correct_difference": {
        "delta": 0.0410958904109589,
        "ci95": [
          -0.013513513513513514,
          0.0958904109589041
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.2026993251687078
      }
    },
    {
      "site": {
        "layer": 7,
        "module": "mlp",
        "position": "identity"
      },
      "top_index": 2,
      "control": "top",
      "donor": "neutral",
      "bad_rescue_margin": 0.0,
      "good_reverse_margin": 0.0,
      "bad_fixed": 0,
      "good_broken": 0,
      "prediction_change": 0.0,
      "metrics": {
        "rows": 292,
        "pairs": 146,
        "accuracy": 0.791095890410959,
        "both_correct": 0.6917808219178082,
        "flip": 0.19863013698630136,
        "gap": 0.07534246575342465,
        "by_permutation": {
          "0": {
            "accuracy": 0.791095890410959,
            "both_correct": 0.6917808219178082,
            "flip": 0.19863013698630136,
            "gap": 0.07534246575342465
          }
        },
        "gold_margin": 3.8681507110595703
      },
      "both_correct_difference": {
        "delta": 0.0,
        "ci95": [
          0.0,
          0.0
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 1.0
      }
    },
    {
      "site": {
        "layer": 12,
        "module": "mlp",
        "position": "identity"
      },
      "top_index": 2,
      "control": "random",
      "donor": "counterpart",
      "bad_rescue_margin": -0.1298491358757019,
      "good_reverse_margin": -0.3195043206214905,
      "bad_fixed": 1,
      "good_broken": 2,
      "prediction_change": 0.017123287671232876,
      "metrics": {
        "rows": 292,
        "pairs": 146,
        "accuracy": 0.7808219178082192,
        "both_correct": 0.678082191780822,
        "flip": 0.2054794520547945,
        "gap": 0.0821917808219178,
        "by_permutation": {
          "0": {
            "accuracy": 0.7808219178082192,
            "both_correct": 0.678082191780822,
            "flip": 0.2054794520547945,
            "gap": 0.0821917808219178
          }
        },
        "gold_margin": 3.7959654331207275
      },
      "both_correct_difference": {
        "delta": -0.0136986301369863,
        "ci95": [
          -0.03424657534246575,
          0.0
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 0.4921269682579355
      }
    },
    {
      "site": {
        "layer": 12,
        "module": "mlp",
        "position": "identity"
      },
      "top_index": 2,
      "control": "random",
      "donor": "neutral",
      "bad_rescue_margin": 0.0,
      "good_reverse_margin": 0.0,
      "bad_fixed": 0,
      "good_broken": 0,
      "prediction_change": 0.0,
      "metrics": {
        "rows": 292,
        "pairs": 146,
        "accuracy": 0.791095890410959,
        "both_correct": 0.6917808219178082,
        "flip": 0.19863013698630136,
        "gap": 0.07534246575342465,
        "by_permutation": {
          "0": {
            "accuracy": 0.791095890410959,
            "both_correct": 0.6917808219178082,
            "flip": 0.19863013698630136,
            "gap": 0.07534246575342465
          }
        },
        "gold_margin": 3.8680436611175537
      },
      "both_correct_difference": {
        "delta": 0.0,
        "ci95": [
          0.0,
          0.0
        ],
        "cluster": "family_id",
        "clusters": 129,
        "p_sign": 1.0
      }
    }
  ],
  "interpretation": "Counterpart donor is oracle and cannot be used as a deployable method; bidirectional rescue/damage may reflect ordinary answer or identity representations."
}
