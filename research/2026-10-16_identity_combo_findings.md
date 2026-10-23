# Bounded combination and identity-mechanism results

Historical WinoBias type2 / Winogender have been inspected previously. Their results are exploratory retests. New native WinoBias TYPE1 development160 is an unseen-output transfer check with native contextual/commonsense labels; it is not uniformly unambiguous explicit evidence.

Counts: {'rank': 64, 'screen': 32, 'dev': 49, 'historical': 146, 'winogender': 240, 'new_type1': 160}
Selected configurations: {"overall": {"name": "l7_scale_0.5", "edits": [{"layer": 7, "module": "mlp", "position": "identity", "operation": "scale", "strength": 0.5}], "category": "single"}, "single": {"name": "l7_scale_0.5", "edits": [{"layer": 7, "module": "mlp", "position": "identity", "operation": "scale", "strength": 0.5}], "category": "single"}, "combo": {"name": "combo_l7_question_identity_0.25", "edits": [{"layer": 7, "module": "mlp", "position": "identity", "operation": "scale", "strength": 0.5}, {"layer": 7, "module": "mlp", "position": "question_identity", "operation": "scale", "strength": 0.25}], "category": "combo"}}

| Dataset | Method | Accuracy | Both genders correct | Identity flip | Pro-anti gap |
|---|---|---:|---:|---:|---:|
| historical | baseline | 76.71% | 65.75% | 21.92% | 8.90% |
| historical | old_l7 | 79.11% | 70.21% | 17.81% | 4.79% |
| historical | best_single | 79.11% | 70.21% | 17.81% | 4.79% |
| historical | instruction | 96.40% | 92.81% | 7.19% | 6.51% |
| historical | best_combo | 78.94% | 69.52% | 18.84% | 5.82% |
| historical | combo_instruction | 97.26% | 94.52% | 5.48% | 3.42% |
| historical | random_combo | 78.08% | 66.78% | 22.60% | 10.96% |
| new_type1 | baseline | 57.50% | 41.56% | 31.87% | 25.62% |
| new_type1 | old_l7 | 58.13% | 43.12% | 30.00% | 26.25% |
| new_type1 | best_single | 58.13% | 43.12% | 30.00% | 26.25% |
| new_type1 | instruction | 54.84% | 28.12% | 53.44% | 45.31% |
| new_type1 | best_combo | 57.81% | 42.81% | 30.00% | 26.88% |
| new_type1 | combo_instruction | 54.53% | 28.75% | 51.56% | 45.31% |
| new_type1 | random_combo | 56.56% | 41.25% | 30.63% | 25.62% |
| winogender | baseline | 57.71% | 48.54% | 18.33% | 9.58% |
| winogender | old_l7 | 57.81% | 48.12% | 19.38% | 11.46% |
| winogender | best_single | 57.81% | 48.12% | 19.38% | 11.46% |
| winogender | instruction | 55.94% | 41.88% | 28.12% | 16.88% |
| winogender | best_combo | 57.92% | 47.92% | 20.00% | 10.42% |
| winogender | combo_instruction | 56.35% | 41.67% | 29.38% | 14.37% |
| winogender | random_combo | 58.23% | 47.71% | 21.04% | 10.21% |

Native chat short generation (historical, frozen configs):
| Method | Strict accuracy | Both correct | Flip | Unparsed |
|---|---:|---:|---:|---:|
| baseline | 86.30% | 80.14% | 13.01% | 3.77% |
| old_l7 | 88.01% | 84.25% | 8.90% | 4.45% |
| best_single | 88.01% | 84.25% | 8.90% | 4.45% |
| instruction | 75.34% | 69.86% | 10.96% | 2.74% |
| best_combo | 87.67% | 84.25% | 8.22% | 4.79% |
| combo_instruction | 77.74% | 73.29% | 8.90% | 2.74% |

Identity/control accuracies:
| Method | Literal reading | Gender classification | Identity binding | Role exchange |
|---|---:|---:|---:|---:|
| baseline | 96.40% | 67.98% | 98.97% | 74.83% |
| old_l7 | 95.89% | 66.95% | 99.49% | 78.60% |
| best_single | 95.89% | 66.95% | 99.49% | 78.60% |
| instruction | 95.21% | 63.18% | 100.00% | 96.23% |
| best_combo | 95.89% | 66.95% | 99.49% | 78.25% |
| combo_instruction | 95.55% | 60.10% | 100.00% | 97.26% |
| random_combo | 95.38% | 67.47% | 98.97% | 75.68% |

Layerwise normalized male-female differences (rank only):
| Layer | Before MLP | After MLP | MLP half-scaled |
|---|---:|---:|---:|
| 0 | 0.6971 | 0.7697 | 0.7277 |
| 5 | 0.5512 | 0.6680 | 0.5958 |
| 6 | 0.6587 | 0.6445 | 0.6357 |
| 7 | 0.6391 | 0.7835 | 0.6919 |
| 8 | 0.7606 | 0.8329 | 0.7837 |
| 9 | 0.7999 | 0.7436 | 0.7600 |

Norm growth and identity differences are not semantic bias attribution. The old suffix-neutral donor was causally inaccessible to the pronoun. Corrected prefix-neutral diagnostics:
{
  "screen_only": true,
  "discordant_pairs": 3,
  "suffix_control_was_causally_inaccessible": true,
  "baseline": {
    "rows": 64,
    "pairs": 32,
    "accuracy": 0.859375,
    "both_correct": 0.8125,
    "flip": 0.09375,
    "gap": -0.03125,
    "by_permutation": {
      "0": {
        "accuracy": 0.859375,
        "both_correct": 0.8125,
        "flip": 0.09375,
        "gap": -0.03125
      }
    },
    "gold_margin": 3.62939453125
  },
  "results": {
    "gender_counterpart": {
      "metrics": {
        "rows": 64,
        "pairs": 32,
        "accuracy": 0.859375,
        "both_correct": 0.78125,
        "flip": 0.15625,
        "gap": -0.09375,
        "by_permutation": {
          "0": {
            "accuracy": 0.859375,
            "both_correct": 0.78125,
            "flip": 0.15625,
            "gap": -0.09375
          }
        },
        "gold_margin": 3.464599609375
      },
      "bad_fixed": 2,
      "good_broken": 1,
      "bad_margin_change": 1.2083333730697632,
      "all_mean_abs_margin_change": 1.177978515625,
      "edit_norm": 29.219989776611328
    },
    "neutral_prefix": {
      "metrics": {
        "rows": 64,
        "pairs": 32,
        "accuracy": 0.875,
        "both_correct": 0.84375,
        "flip": 0.0625,
        "gap": -0.0625,
        "by_permutation": {
          "0": {
            "accuracy": 0.875,
            "both_correct": 0.84375,
            "flip": 0.0625,
            "gap": -0.0625
          }
        },
        "gold_margin": 3.572021484375
      },
      "bad_fixed": 1,
      "good_broken": 0,
      "bad_margin_change": 0.2083333283662796,
      "all_mean_abs_margin_change": 0.244384765625,
      "edit_norm": 7.986980438232422
    },
    "neutral_norm_matched": {
      "metrics": {
        "rows": 64,
        "pairs": 32,
        "accuracy": 0.875,
        "both_correct": 0.8125,
        "flip": 0.125,
        "gap": 0.0625,
        "by_permutation": {
          "0": {
            "accuracy": 0.875,
            "both_correct": 0.8125,
            "flip": 0.125,
            "gap": 0.0625
          }
        },
        "gold_margin": 3.237548828125
      },
      "bad_fixed": 2,
      "good_broken": 0,
      "bad_margin_change": 0.6510416865348816,
      "all_mean_abs_margin_change": 1.129150390625,
      "edit_norm": 29.219989776611328
    }
  },
  "zero_neutral_norm_rows": 0
}

Comparisons against previous layer7 single:
{
  "historical": {
    "coref": {
      "best_single": {
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
        },
        "flip": {
          "delta": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 1.0
        },
        "gap": {
          "delta": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 1.0
        },
        "occupation_both_correct": {
          "delta": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "cluster": "occupation_pair",
          "clusters": 110,
          "p_sign": 1.0
        }
      },
      "best_combo": {
        "accuracy": {
          "delta": -0.0017123287671232876,
          "ci95": [
            -0.008930177286742034,
            0.00528169014084507
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 1.0
        },
        "both_correct": {
          "delta": -0.00684931506849315,
          "ci95": [
            -0.020689655172413793,
            0.006757905865048727
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.6210947263184204
        },
        "flip": {
          "delta": 0.010273972602739725,
          "ci95": [
            -0.003472222222222222,
            0.025974025974025976
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.3789052736815796
        },
        "gap": {
          "delta": 0.010273972602739725,
          "ci95": [
            -0.0034965034965034965,
            0.024822695035460994
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.3781554611347163
        },
        "occupation_both_correct": {
          "delta": -0.00684931506849315,
          "ci95": [
            -0.02055148795465281,
            0.0066236203090507765
          ],
          "cluster": "occupation_pair",
          "clusters": 110,
          "p_sign": 0.6248437890527369
        }
      }
    },
    "reading": {
      "best_single": {
        "accuracy": {
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
      "best_combo": {
        "accuracy": {
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
    },
    "binding": {
      "best_single": {
        "accuracy": {
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
      "best_combo": {
        "accuracy": {
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
    },
    "role_swap": {
      "best_single": {
        "accuracy": {
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
      "best_combo": {
        "accuracy": {
          "delta": -0.003424657534246575,
          "ci95": [
            -0.008620689655172414,
            0.0
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.5128717820544864
        }
      }
    },
    "literal": {
      "best_single": {
        "accuracy": {
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
      "best_combo": {
        "accuracy": {
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
    }
  },
  "new_type1": {
    "coref": {
      "best_single": {
        "accuracy": {
          "delta": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "cluster": "family_id",
          "clusters": 160,
          "p_sign": 1.0
        },
        "both_correct": {
          "delta": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "cluster": "family_id",
          "clusters": 160,
          "p_sign": 1.0
        },
        "flip": {
          "delta": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "cluster": "family_id",
          "clusters": 160,
          "p_sign": 1.0
        },
        "gap": {
          "delta": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "cluster": "family_id",
          "clusters": 160,
          "p_sign": 1.0
        },
        "occupation_both_correct": {
          "delta": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "cluster": "occupation_pair",
          "clusters": 127,
          "p_sign": 1.0
        }
      },
      "best_combo": {
        "accuracy": {
          "delta": -0.003125,
          "ci95": [
            -0.0125,
            0.00625
          ],
          "cluster": "family_id",
          "clusters": 160,
          "p_sign": 0.7735566108472882
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
            -0.015625,
            0.015625
          ],
          "cluster": "family_id",
          "clusters": 160,
          "p_sign": 1.0
        },
        "gap": {
          "delta": 0.00625,
          "ci95": [
            -0.009375,
            0.021875
          ],
          "cluster": "family_id",
          "clusters": 160,
          "p_sign": 0.6890777305673582
        },
        "occupation_both_correct": {
          "delta": -0.003125,
          "ci95": [
            -0.013247240618101544,
            0.00641025641025641
          ],
          "cluster": "occupation_pair",
          "clusters": 127,
          "p_sign": 1.0
        }
      }
    }
  },
  "winogender": {
    "coref": {
      "best_single": {
        "accuracy": {
          "delta": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "cluster": "family_id",
          "clusters": 120,
          "p_sign": 1.0
        },
        "both_correct": {
          "delta": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "cluster": "family_id",
          "clusters": 120,
          "p_sign": 1.0
        },
        "flip": {
          "delta": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "cluster": "family_id",
          "clusters": 120,
          "p_sign": 1.0
        },
        "gap": {
          "delta": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "cluster": "family_id",
          "clusters": 120,
          "p_sign": 1.0
        },
        "occupation_both_correct": {
          "delta": 0.0,
          "ci95": [
            0.0,
            0.0
          ],
          "cluster": "occupation_pair",
          "clusters": 120,
          "p_sign": 1.0
        }
      },
      "best_combo": {
        "accuracy": {
          "delta": 0.0010416666666666667,
          "ci95": [
            -0.007291666666666667,
            0.009375
          ],
          "cluster": "family_id",
          "clusters": 120,
          "p_sign": 1.0
        },
        "both_correct": {
          "delta": -0.0020833333333333333,
          "ci95": [
            -0.0125,
            0.00625
          ],
          "cluster": "family_id",
          "clusters": 120,
          "p_sign": 1.0
        },
        "flip": {
          "delta": 0.00625,
          "ci95": [
            -0.010416666666666666,
            0.022916666666666665
          ],
          "cluster": "family_id",
          "clusters": 120,
          "p_sign": 0.6088477880529868
        },
        "gap": {
          "delta": -0.010416666666666666,
          "ci95": [
            -0.027083333333333334,
            0.004166666666666667
          ],
          "cluster": "family_id",
          "clusters": 120,
          "p_sign": 0.2994251437140715
        },
        "occupation_both_correct": {
          "delta": -0.0020833333333333333,
          "ci95": [
            -0.010416666666666666,
            0.00625
          ],
          "cluster": "occupation_pair",
          "clusters": 120,
          "p_sign": 1.0
        }
      }
    }
  }
}

Native chat differences:
{
  "best_single": {
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
    },
    "flip": {
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
  "best_combo": {
    "accuracy": {
      "delta": -0.003424657534246575,
      "ci95": [
        -0.01079136690647482,
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
    },
    "flip": {
      "delta": -0.00684931506849315,
      "ci95": [
        -0.02158273381294964,
        0.0
      ],
      "cluster": "family_id",
      "clusters": 129,
      "p_sign": 1.0
    }
  }
}

Constituent interaction:
{
  "status": "descriptive; final score interaction does not establish a circuit edge",
  "combined": {
    "rows": 584,
    "pairs": 146,
    "accuracy": 0.7893835616438356,
    "both_correct": 0.6952054794520548,
    "flip": 0.18835616438356165,
    "gap": 0.05821917808219178,
    "by_permutation": {
      "0": {
        "accuracy": 0.8047945205479452,
        "both_correct": 0.7191780821917808,
        "flip": 0.17123287671232876,
        "gap": 0.02054794520547945
      },
      "1": {
        "accuracy": 0.773972602739726,
        "both_correct": 0.6712328767123288,
        "flip": 0.2054794520547945,
        "gap": 0.0958904109589041
      }
    },
    "gold_margin": 3.7427761554718018
  },
  "baseline": {
    "rows": 584,
    "pairs": 146,
    "accuracy": 0.7671232876712328,
    "both_correct": 0.6575342465753424,
    "flip": 0.2191780821917808,
    "gap": 0.08904109589041095,
    "by_permutation": {
      "0": {
        "accuracy": 0.791095890410959,
        "both_correct": 0.6917808219178082,
        "flip": 0.19863013698630136,
        "gap": 0.07534246575342465
      },
      "1": {
        "accuracy": 0.7431506849315068,
        "both_correct": 0.6232876712328768,
        "flip": 0.23972602739726026,
        "gap": 0.10273972602739725
      }
    },
    "gold_margin": 3.619755983352661
  },
  "constituents": [
    {
      "rows": 584,
      "pairs": 146,
      "accuracy": 0.791095890410959,
      "both_correct": 0.702054794520548,
      "flip": 0.1780821917808219,
      "gap": 0.04794520547945205,
      "by_permutation": {
        "0": {
          "accuracy": 0.8116438356164384,
          "both_correct": 0.7328767123287672,
          "flip": 0.15753424657534246,
          "gap": 0.02054794520547945
        },
        "1": {
          "accuracy": 0.7705479452054794,
          "both_correct": 0.6712328767123288,
          "flip": 0.19863013698630136,
          "gap": 0.07534246575342465
        }
      },
      "gold_margin": 3.7955639362335205
    },
    {
      "rows": 584,
      "pairs": 146,
      "accuracy": 0.7671232876712328,
      "both_correct": 0.6541095890410958,
      "flip": 0.22602739726027396,
      "gap": 0.08904109589041095,
      "by_permutation": {
        "0": {
          "accuracy": 0.7773972602739726,
          "both_correct": 0.6712328767123288,
          "flip": 0.21232876712328766,
          "gap": 0.07534246575342465
        },
        "1": {
          "accuracy": 0.7568493150684932,
          "both_correct": 0.636986301369863,
          "flip": 0.23972602739726026,
          "gap": 0.10273972602739725
        }
      },
      "gold_margin": 3.5592358112335205
    }
  ],
  "mean_margin_interaction": 0.007732234429568052,
  "both_correct_interaction": -0.003424657534246589
}

Fixed descriptive mechanism contrasts:
{
  "historical_exploratory": true,
  "never_selects": true,
  "metrics": {
    "coref": {
      "baseline": {
        "rows": 584,
        "pairs": 146,
        "accuracy": 0.7671232876712328,
        "both_correct": 0.6575342465753424,
        "flip": 0.2191780821917808,
        "gap": 0.08904109589041095,
        "by_permutation": {
          "0": {
            "accuracy": 0.791095890410959,
            "both_correct": 0.6917808219178082,
            "flip": 0.19863013698630136,
            "gap": 0.07534246575342465
          },
          "1": {
            "accuracy": 0.7431506849315068,
            "both_correct": 0.6232876712328768,
            "flip": 0.23972602739726026,
            "gap": 0.10273972602739725
          }
        },
        "gold_margin": 3.619755983352661
      },
      "l7_scale_0.5": {
        "rows": 584,
        "pairs": 146,
        "accuracy": 0.791095890410959,
        "both_correct": 0.702054794520548,
        "flip": 0.1780821917808219,
        "gap": 0.04794520547945205,
        "by_permutation": {
          "0": {
            "accuracy": 0.8116438356164384,
            "both_correct": 0.7328767123287672,
            "flip": 0.15753424657534246,
            "gap": 0.02054794520547945
          },
          "1": {
            "accuracy": 0.7705479452054794,
            "both_correct": 0.6712328767123288,
            "flip": 0.19863013698630136,
            "gap": 0.07534246575342465
          }
        },
        "gold_margin": 3.7955639362335205
      },
      "subspace_rank1": {
        "rows": 584,
        "pairs": 146,
        "accuracy": 0.785958904109589,
        "both_correct": 0.684931506849315,
        "flip": 0.20205479452054795,
        "gap": 0.08561643835616438,
        "by_permutation": {
          "0": {
            "accuracy": 0.8082191780821918,
            "both_correct": 0.7191780821917808,
            "flip": 0.1780821917808219,
            "gap": 0.0547945205479452
          },
          "1": {
            "accuracy": 0.7636986301369864,
            "both_correct": 0.6506849315068494,
            "flip": 0.22602739726027396,
            "gap": 0.11643835616438356
          }
        },
        "gold_margin": 3.8183326721191406
      },
      "complement_rank1": {
        "rows": 584,
        "pairs": 146,
        "accuracy": 0.7722602739726028,
        "both_correct": 0.6712328767123288,
        "flip": 0.20205479452054795,
        "gap": 0.07191780821917808,
        "by_permutation": {
          "0": {
            "accuracy": 0.791095890410959,
            "both_correct": 0.6986301369863014,
            "flip": 0.18493150684931506,
            "gap": 0.04794520547945205
          },
          "1": {
            "accuracy": 0.7534246575342466,
            "both_correct": 0.6438356164383562,
            "flip": 0.2191780821917808,
            "gap": 0.0958904109589041
          }
        },
        "gold_margin": 3.6077964305877686
      },
      "subspace_rank4": {
        "rows": 584,
        "pairs": 146,
        "accuracy": 0.7876712328767124,
        "both_correct": 0.6883561643835616,
        "flip": 0.19863013698630136,
        "gap": 0.07534246575342465,
        "by_permutation": {
          "0": {
            "accuracy": 0.815068493150685,
            "both_correct": 0.7328767123287672,
            "flip": 0.1643835616438356,
            "gap": 0.0410958904109589
          },
          "1": {
            "accuracy": 0.7602739726027398,
            "both_correct": 0.6438356164383562,
            "flip": 0.2328767123287671,
            "gap": 0.1095890410958904
          }
        },
        "gold_margin": 3.8150951862335205
      },
      "complement_rank4": {
        "rows": 584,
        "pairs": 146,
        "accuracy": 0.7756849315068494,
        "both_correct": 0.6746575342465754,
        "flip": 0.20205479452054795,
        "gap": 0.06506849315068493,
        "by_permutation": {
          "0": {
            "accuracy": 0.797945205479452,
            "both_correct": 0.7054794520547946,
            "flip": 0.18493150684931506,
            "gap": 0.03424657534246575
          },
          "1": {
            "accuracy": 0.7534246575342466,
            "both_correct": 0.6438356164383562,
            "flip": 0.2191780821917808,
            "gap": 0.0958904109589041
          }
        },
        "gold_margin": 3.6045055389404297
      },
      "neurons_top128_0.5": {
        "rows": 584,
        "pairs": 146,
        "accuracy": 0.7893835616438356,
        "both_correct": 0.6883561643835616,
        "flip": 0.20205479452054795,
        "gap": 0.07876712328767123,
        "by_permutation": {
          "0": {
            "accuracy": 0.8082191780821918,
            "both_correct": 0.7191780821917808,
            "flip": 0.1780821917808219,
            "gap": 0.0547945205479452
          },
          "1": {
            "accuracy": 0.7705479452054794,
            "both_correct": 0.6575342465753424,
            "flip": 0.22602739726027396,
            "gap": 0.10273972602739725
          }
        },
        "gold_margin": 3.83855938911438
      },
      "neurons_matched128_0.5": {
        "rows": 584,
        "pairs": 146,
        "accuracy": 0.7688356164383562,
        "both_correct": 0.660958904109589,
        "flip": 0.21575342465753425,
        "gap": 0.07876712328767123,
        "by_permutation": {
          "0": {
            "accuracy": 0.791095890410959,
            "both_correct": 0.6986301369863014,
            "flip": 0.18493150684931506,
            "gap": 0.06164383561643835
          },
          "1": {
            "accuracy": 0.7465753424657534,
            "both_correct": 0.6232876712328768,
            "flip": 0.2465753424657534,
            "gap": 0.0958904109589041
          }
        },
        "gold_margin": 3.6164650917053223
      },
      "l7_position_subject": {
        "rows": 584,
        "pairs": 146,
        "accuracy": 0.7808219178082192,
        "both_correct": 0.678082191780822,
        "flip": 0.2054794520547945,
        "gap": 0.0547945205479452,
        "by_permutation": {
          "0": {
            "accuracy": 0.8013698630136986,
            "both_correct": 0.7123287671232876,
            "flip": 0.1780821917808219,
            "gap": 0.0273972602739726
          },
          "1": {
            "accuracy": 0.7602739726027398,
            "both_correct": 0.6438356164383562,
            "flip": 0.2328767123287671,
            "gap": 0.0821917808219178
          }
        },
        "gold_margin": 3.6743900775909424
      },
      "l7_position_object": {
        "rows": 584,
        "pairs": 146,
        "accuracy": 0.7636986301369864,
        "both_correct": 0.6506849315068494,
        "flip": 0.22602739726027396,
        "gap": 0.07534246575342465,
        "by_permutation": {
          "0": {
            "accuracy": 0.7876712328767124,
            "both_correct": 0.678082191780822,
            "flip": 0.2191780821917808,
            "gap": 0.0547945205479452
          },
          "1": {
            "accuracy": 0.7397260273972602,
            "both_correct": 0.6232876712328768,
            "flip": 0.2328767123287671,
            "gap": 0.0958904109589041
          }
        },
        "gold_margin": 3.4953179359436035
      },
      "l7_position_question_identity": {
        "rows": 584,
        "pairs": 146,
        "accuracy": 0.7654109589041096,
        "both_correct": 0.660958904109589,
        "flip": 0.2089041095890411,
        "gap": 0.07876712328767123,
        "by_permutation": {
          "0": {
            "accuracy": 0.773972602739726,
            "both_correct": 0.6712328767123288,
            "flip": 0.2054794520547945,
            "gap": 0.0547945205479452
          },
          "1": {
            "accuracy": 0.7568493150684932,
            "both_correct": 0.6506849315068494,
            "flip": 0.21232876712328766,
            "gap": 0.10273972602739725
          }
        },
        "gold_margin": 3.5271832942962646
      }
    },
    "literal": {
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
      "l7_scale_0.5": {
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
      "subspace_rank1": {
        "rows": 584,
        "pairs": 146,
        "accuracy": 0.9674657534246576,
        "both_correct": 0.934931506849315,
        "flip": 0.934931506849315,
        "gap": 0.017123287671232876,
        "by_permutation": {
          "0": {
            "accuracy": 0.9417808219178082,
            "both_correct": 0.8835616438356164,
            "flip": 0.8835616438356164,
            "gap": 0.02054794520547945
          },
          "1": {
            "accuracy": 0.9931506849315068,
            "both_correct": 0.9863013698630136,
            "flip": 0.9863013698630136,
            "gap": 0.0136986301369863
          }
        },
        "gold_margin": 7.4601616859436035
      },
      "complement_rank1": {
        "rows": 584,
        "pairs": 146,
        "accuracy": 0.9503424657534246,
        "both_correct": 0.9006849315068494,
        "flip": 0.9006849315068494,
        "gap": 0.030821917808219176,
        "by_permutation": {
          "0": {
            "accuracy": 0.9075342465753424,
            "both_correct": 0.815068493150685,
            "flip": 0.815068493150685,
            "gap": 0.04794520547945205
          },
          "1": {
            "accuracy": 0.9931506849315068,
            "both_correct": 0.9863013698630136,
            "flip": 0.9863013698630136,
            "gap": 0.0136986301369863
          }
        },
        "gold_margin": 7.137307167053223
      },
      "subspace_rank4": {
        "rows": 584,
        "pairs": 146,
        "accuracy": 0.9674657534246576,
        "both_correct": 0.934931506849315,
        "flip": 0.934931506849315,
        "gap": 0.017123287671232876,
        "by_permutation": {
          "0": {
            "accuracy": 0.9417808219178082,
            "both_correct": 0.8835616438356164,
            "flip": 0.8835616438356164,
            "gap": 0.02054794520547945
          },
          "1": {
            "accuracy": 0.9931506849315068,
            "both_correct": 0.9863013698630136,
            "flip": 0.9863013698630136,
            "gap": 0.0136986301369863
          }
        },
        "gold_margin": 7.497137069702148
      },
      "complement_rank4": {
        "rows": 584,
        "pairs": 146,
        "accuracy": 0.952054794520548,
        "both_correct": 0.9041095890410958,
        "flip": 0.9041095890410958,
        "gap": 0.03424657534246575,
        "by_permutation": {
          "0": {
            "accuracy": 0.910958904109589,
            "both_correct": 0.821917808219178,
            "flip": 0.821917808219178,
            "gap": 0.0547945205479452
          },
          "1": {
            "accuracy": 0.9931506849315068,
            "both_correct": 0.9863013698630136,
            "flip": 0.9863013698630136,
            "gap": 0.0136986301369863
          }
        },
        "gold_margin": 7.102766513824463
      },
      "neurons_top128_0.5": {
        "rows": 584,
        "pairs": 146,
        "accuracy": 0.9674657534246576,
        "both_correct": 0.934931506849315,
        "flip": 0.934931506849315,
        "gap": 0.010273972602739725,
        "by_permutation": {
          "0": {
            "accuracy": 0.9417808219178082,
            "both_correct": 0.8835616438356164,
            "flip": 0.8835616438356164,
            "gap": 0.00684931506849315
          },
          "1": {
            "accuracy": 0.9931506849315068,
            "both_correct": 0.9863013698630136,
            "flip": 0.9863013698630136,
            "gap": 0.0136986301369863
          }
        },
        "gold_margin": 7.51819372177124
      },
      "neurons_matched128_0.5": {
        "rows": 584,
        "pairs": 146,
        "accuracy": 0.9606164383561644,
        "both_correct": 0.9212328767123288,
        "flip": 0.9212328767123288,
        "gap": 0.023972602739726026,
        "by_permutation": {
          "0": {
            "accuracy": 0.928082191780822,
            "both_correct": 0.8561643835616438,
            "flip": 0.8561643835616438,
            "gap": 0.03424657534246575
          },
          "1": {
            "accuracy": 0.9931506849315068,
            "both_correct": 0.9863013698630136,
            "flip": 0.9863013698630136,
            "gap": 0.0136986301369863
          }
        },
        "gold_margin": 7.22816801071167
      },
      "l7_position_subject": {
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
        "gold_margin": 7.334572792053223
      },
      "l7_position_object": {
        "rows": 584,
        "pairs": 146,
        "accuracy": 0.964041095890411,
        "both_correct": 0.928082191780822,
        "flip": 0.928082191780822,
        "gap": 0.03767123287671233,
        "by_permutation": {
          "0": {
            "accuracy": 0.934931506849315,
            "both_correct": 0.8698630136986302,
            "flip": 0.8698630136986302,
            "gap": 0.06164383561643835
          },
          "1": {
            "accuracy": 0.9931506849315068,
            "both_correct": 0.9863013698630136,
            "flip": 0.9863013698630136,
            "gap": 0.0136986301369863
          }
        },
        "gold_margin": 7.308406352996826
      },
      "l7_position_question_identity": {
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
      }
    }
  }
}

Occupation-context diagnostics (post-selection exploratory):
{
  "status": "post-selection historical exploratory diagnostics, never reselects",
  "gender_delta_context_changes": {
    "role_exchange": [
      {
        "layer": 0,
        "mean_gender_delta_cosine": 0.9992382526397705,
        "relative_gender_delta_change": 0.034949831664562225
      },
      {
        "layer": 5,
        "mean_gender_delta_cosine": 0.9012579917907715,
        "relative_gender_delta_change": 0.42951589822769165
      },
      {
        "layer": 6,
        "mean_gender_delta_cosine": 0.8268575668334961,
        "relative_gender_delta_change": 0.5761632919311523
      },
      {
        "layer": 7,
        "mean_gender_delta_cosine": 0.8666079640388489,
        "relative_gender_delta_change": 0.5068715810775757
      },
      {
        "layer": 8,
        "mean_gender_delta_cosine": 0.7803328037261963,
        "relative_gender_delta_change": 0.6523888111114502
      },
      {
        "layer": 9,
        "mean_gender_delta_cosine": 0.711383581161499,
        "relative_gender_delta_change": 0.7592275142669678
      }
    ],
    "anonymous_occupations": [
      {
        "layer": 0,
        "mean_gender_delta_cosine": 0.9829057455062866,
        "relative_gender_delta_change": 0.18360522389411926
      },
      {
        "layer": 5,
        "mean_gender_delta_cosine": 0.8355377316474915,
        "relative_gender_delta_change": 0.5841556787490845
      },
      {
        "layer": 6,
        "mean_gender_delta_cosine": 0.6940573453903198,
        "relative_gender_delta_change": 0.7918983697891235
      },
      {
        "layer": 7,
        "mean_gender_delta_cosine": 0.7851670980453491,
        "relative_gender_delta_change": 0.70201176404953
      },
      {
        "layer": 8,
        "mean_gender_delta_cosine": 0.6289900541305542,
        "relative_gender_delta_change": 0.9003115892410278
      },
      {
        "layer": 9,
        "mean_gender_delta_cosine": 0.527076244354248,
        "relative_gender_delta_change": 1.0345107316970825
      }
    ]
  },
  "anonymized_occupation_behavior": {
    "baseline": {
      "rows": 584,
      "pairs": 146,
      "accuracy": 0.7962328767123288,
      "both_correct": 0.6575342465753424,
      "flip": 0.2773972602739726,
      "gap": 0.017123287671232876,
      "by_permutation": {
        "0": {
          "accuracy": 0.7534246575342466,
          "both_correct": 0.6095890410958904,
          "flip": 0.2876712328767123,
          "gap": 0.0410958904109589
        },
        "1": {
          "accuracy": 0.839041095890411,
          "both_correct": 0.7054794520547946,
          "flip": 0.2671232876712329,
          "gap": -0.00684931506849315
        }
      },
      "gold_margin": 4.223566055297852
    },
    "old_l7": {
      "rows": 584,
      "pairs": 146,
      "accuracy": 0.815068493150685,
      "both_correct": 0.6917808219178082,
      "flip": 0.2465753424657534,
      "gap": 0.0273972602739726,
      "by_permutation": {
        "0": {
          "accuracy": 0.7602739726027398,
          "both_correct": 0.6301369863013698,
          "flip": 0.2602739726027397,
          "gap": 0.0410958904109589
        },
        "1": {
          "accuracy": 0.8698630136986302,
          "both_correct": 0.7534246575342466,
          "flip": 0.2328767123287671,
          "gap": 0.0136986301369863
        }
      },
      "gold_margin": 4.6151275634765625
    },
    "best_combo": {
      "rows": 584,
      "pairs": 146,
      "accuracy": 0.8133561643835616,
      "both_correct": 0.6746575342465754,
      "flip": 0.2773972602739726,
      "gap": 0.023972602739726026,
      "by_permutation": {
        "0": {
          "accuracy": 0.7671232876712328,
          "both_correct": 0.6164383561643836,
          "flip": 0.3013698630136986,
          "gap": 0.0136986301369863
        },
        "1": {
          "accuracy": 0.8595890410958904,
          "both_correct": 0.7328767123287672,
          "flip": 0.2534246575342466,
          "gap": 0.03424657534246575
        }
      },
      "gold_margin": 4.669172763824463
    },
    "instruction": {
      "rows": 584,
      "pairs": 146,
      "accuracy": 0.8116438356164384,
      "both_correct": 0.6815068493150684,
      "flip": 0.2602739726027397,
      "gap": 0.06164383561643835,
      "by_permutation": {
        "0": {
          "accuracy": 0.7602739726027398,
          "both_correct": 0.6027397260273972,
          "flip": 0.3150684931506849,
          "gap": 0.0410958904109589
        },
        "1": {
          "accuracy": 0.863013698630137,
          "both_correct": 0.7602739726027398,
          "flip": 0.2054794520547945,
          "gap": 0.0821917808219178
        }
      },
      "gold_margin": 2.0235979557037354
    }
  },
  "limits": "Anonymization changes lexical form/task difficulty as well as occupation priors. Gender-difference norms are not bias-specific attribution."
}

Secondary comparisons:
{
  "status": "secondary descriptive comparisons, not extra model selection",
  "comparisons": {
    "historical": {
      "old_l7_vs_baseline": {
        "accuracy": {
          "delta": 0.023972602739726026,
          "ci95": [
            0.006578947368421052,
            0.04310344827586207
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.02174456385903524
        },
        "both_correct": {
          "delta": 0.04452054794520548,
          "ci95": [
            0.017361111111111112,
            0.07299909340276166
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.00424893776555861
        },
        "flip": {
          "delta": -0.0410958904109589,
          "ci95": [
            -0.07395317853064332,
            -0.010273972602739725
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.021994501374656337
        }
      },
      "best_combo_vs_baseline": {
        "accuracy": {
          "delta": 0.02226027397260274,
          "ci95": [
            0.00510204081632653,
            0.04021125686394143
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.02649337665583604
        },
        "both_correct": {
          "delta": 0.03767123287671233,
          "ci95": [
            0.01048951048951049,
            0.06551724137931035
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.015496125968507872
        },
        "flip": {
          "delta": -0.030821917808219176,
          "ci95": [
            -0.061224489795918366,
            0.0
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.07848037990502374
        }
      },
      "combo_instruction_vs_instruction": {
        "accuracy": {
          "delta": 0.008561643835616438,
          "ci95": [
            -0.0016894764662621806,
            0.02040816326530612
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.23144213946513373
        },
        "both_correct": {
          "delta": 0.017123287671232876,
          "ci95": [
            -0.0033789529325243613,
            0.04081632653061224
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.23144213946513373
        },
        "flip": {
          "delta": -0.017123287671232876,
          "ci95": [
            -0.04081632653061224,
            0.0033789529325243635
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.23144213946513373
        }
      }
    },
    "new_type1": {
      "old_l7_vs_baseline": {
        "accuracy": {
          "delta": 0.00625,
          "ci95": [
            -0.0140625,
            0.0265625
          ],
          "cluster": "family_id",
          "clusters": 160,
          "p_sign": 0.6520869782554362
        },
        "both_correct": {
          "delta": 0.015625,
          "ci95": [
            -0.009375,
            0.04375
          ],
          "cluster": "family_id",
          "clusters": 160,
          "p_sign": 0.3709072731817046
        },
        "flip": {
          "delta": -0.01875,
          "ci95": [
            -0.053125,
            0.015625
          ],
          "cluster": "family_id",
          "clusters": 160,
          "p_sign": 0.3781554611347163
        }
      },
      "best_combo_vs_baseline": {
        "accuracy": {
          "delta": 0.003125,
          "ci95": [
            -0.01875,
            0.02347656250000014
          ],
          "cluster": "family_id",
          "clusters": 160,
          "p_sign": 0.8910272431892027
        },
        "both_correct": {
          "delta": 0.0125,
          "ci95": [
            -0.0125,
            0.040625
          ],
          "cluster": "family_id",
          "clusters": 160,
          "p_sign": 0.5191202199450138
        },
        "flip": {
          "delta": -0.01875,
          "ci95": [
            -0.05625,
            0.01875
          ],
          "cluster": "family_id",
          "clusters": 160,
          "p_sign": 0.4136465883529118
        }
      },
      "combo_instruction_vs_instruction": {
        "accuracy": {
          "delta": -0.003125,
          "ci95": [
            -0.01875,
            0.0125
          ],
          "cluster": "family_id",
          "clusters": 160,
          "p_sign": 0.8575356160959761
        },
        "both_correct": {
          "delta": 0.00625,
          "ci95": [
            -0.015625,
            0.028125
          ],
          "cluster": "family_id",
          "clusters": 160,
          "p_sign": 0.7990502374406399
        },
        "flip": {
          "delta": -0.01875,
          "ci95": [
            -0.05,
            0.0125
          ],
          "cluster": "family_id",
          "clusters": 160,
          "p_sign": 0.3594101474631342
        }
      }
    },
    "winogender": {
      "old_l7_vs_baseline": {
        "accuracy": {
          "delta": 0.0010416666666666667,
          "ci95": [
            -0.015625,
            0.016666666666666666
          ],
          "cluster": "family_id",
          "clusters": 120,
          "p_sign": 1.0
        },
        "both_correct": {
          "delta": -0.004166666666666667,
          "ci95": [
            -0.022916666666666665,
            0.014583333333333334
          ],
          "cluster": "family_id",
          "clusters": 120,
          "p_sign": 0.8295426143464134
        },
        "flip": {
          "delta": 0.010416666666666666,
          "ci95": [
            -0.0125,
            0.03333333333333333
          ],
          "cluster": "family_id",
          "clusters": 120,
          "p_sign": 0.48137965508622843
        }
      },
      "best_combo_vs_baseline": {
        "accuracy": {
          "delta": 0.0020833333333333333,
          "ci95": [
            -0.015625,
            0.017708333333333333
          ],
          "cluster": "family_id",
          "clusters": 120,
          "p_sign": 0.9070232441889527
        },
        "both_correct": {
          "delta": -0.00625,
          "ci95": [
            -0.025,
            0.0125
          ],
          "cluster": "family_id",
          "clusters": 120,
          "p_sign": 0.6683329167708073
        },
        "flip": {
          "delta": 0.016666666666666666,
          "ci95": [
            -0.010416666666666666,
            0.04375
          ],
          "cluster": "family_id",
          "clusters": 120,
          "p_sign": 0.2551862034491377
        }
      },
      "combo_instruction_vs_instruction": {
        "accuracy": {
          "delta": 0.004166666666666667,
          "ci95": [
            -0.0125,
            0.020859375000000093
          ],
          "cluster": "family_id",
          "clusters": 120,
          "p_sign": 0.7223194201449638
        },
        "both_correct": {
          "delta": -0.0020833333333333333,
          "ci95": [
            -0.029166666666666667,
            0.027083333333333334
          ],
          "cluster": "family_id",
          "clusters": 120,
          "p_sign": 1.0
        },
        "flip": {
          "delta": 0.0125,
          "ci95": [
            -0.020833333333333332,
            0.04583333333333333
          ],
          "cluster": "family_id",
          "clusters": 120,
          "p_sign": 0.535616095976006
        }
      }
    },
    "chat": {
      "old_l7_vs_baseline": {
        "accuracy": {
          "delta": 0.017123287671232876,
          "ci95": [
            -0.003312369438641717,
            0.03819444444444445
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.18370407398150462
        },
        "both_correct": {
          "delta": 0.0410958904109589,
          "ci95": [
            0.013422818791946308,
            0.07534246575342465
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.03524118970257436
        },
        "flip": {
          "delta": -0.0410958904109589,
          "ci95": [
            -0.07947019867549669,
            -0.00684931506849315
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.0709822544363909
        }
      },
      "best_combo_vs_baseline": {
        "accuracy": {
          "delta": 0.0136986301369863,
          "ci95": [
            -0.00684931506849315,
            0.03496503496503497
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.3346663334166458
        },
        "both_correct": {
          "delta": 0.0410958904109589,
          "ci95": [
            0.013422818791946308,
            0.07534246575342465
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.03524118970257436
        },
        "flip": {
          "delta": -0.04794520547945205,
          "ci95": [
            -0.09027777777777778,
            -0.013333333333333334
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.04423894026493377
        }
      },
      "combo_instruction_vs_instruction": {
        "accuracy": {
          "delta": 0.023972602739726026,
          "ci95": [
            0.0070921985815602835,
            0.04285714285714286
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.014996250937265684
        },
        "both_correct": {
          "delta": 0.03424657534246575,
          "ci95": [
            0.006944444444444444,
            0.06711409395973154
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.05998500374906274
        },
        "flip": {
          "delta": -0.02054794520547945,
          "ci95": [
            -0.05673758865248227,
            0.0136986301369863
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.45188702824293925
        }
      }
    }
  }
}

Mechanistic descriptive contrast uncertainty:
{
  "status": "descriptive, multiple contrasts unadjusted; RMS matching is approximate",
  "comparisons": {
    "coref": {
      "subspace_rank1_vs_complement_rank1": {
        "accuracy": {
          "delta": 0.0136986301369863,
          "ci95": [
            -0.004935027448588358,
            0.031914893617021274
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.2131967008247938
        },
        "both_correct": {
          "delta": 0.0136986301369863,
          "ci95": [
            -0.013986013986013986,
            0.0425584406494961
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.4943764058985254
        }
      },
      "neurons_top128_0.5_vs_neurons_matched128_0.5": {
        "accuracy": {
          "delta": 0.02054794520547945,
          "ci95": [
            0.00827678415824329,
            0.034722222222222224
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.008747813046738315
        },
        "both_correct": {
          "delta": 0.0273972602739726,
          "ci95": [
            0.006711409395973154,
            0.05102040816326531
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.04048987753061734
        }
      }
    },
    "literal": {
      "subspace_rank1_vs_complement_rank1": {
        "accuracy": {
          "delta": 0.017123287671232876,
          "ci95": [
            0.006756756756756757,
            0.02910958904109589
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.007998000499875032
        },
        "both_correct": {
          "delta": 0.03424657534246575,
          "ci95": [
            0.013513513513513514,
            0.05821917808219178
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.007998000499875032
        }
      },
      "neurons_top128_0.5_vs_neurons_matched128_0.5": {
        "accuracy": {
          "delta": 0.00684931506849315,
          "ci95": [
            0.0,
            0.015625
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.22969257685578606
        },
        "both_correct": {
          "delta": 0.0136986301369863,
          "ci95": [
            0.0,
            0.03125
          ],
          "cluster": "family_id",
          "clusters": 129,
          "p_sign": 0.22969257685578606
        }
      }
    }
  }
}

Neuron-control RMS: top128=0.9251, matched128=0.5178. Nearest-neighbor control is only approximate; this mismatch limits claims of channel specificity.

Limits: one model; repeated old selection splits; historical retests not independent; agent data audit; no verified circuit edges or broad free-generation claim.
