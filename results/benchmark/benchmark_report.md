
# Smart Contract Vulnerability Detection Benchmark Report

Generated on: 2025-11-16 00:38:51

## Executive Summary

This report compares the performance of our CodeBERT-based vulnerability detection model against traditional static analysis tools (Slither and Mythril) on a test set of Solidity smart contract functions.

### Dataset Overview
- Total test functions analyzed: 612
- Ground truth vulnerability rate: 7.2%
- Vulnerability categories covered: 7 types

## Tool Performance Comparison


### Performance Metrics

                tool  n_predictions  accuracy  precision  recall    f1  true_positives  false_positives  true_negatives  false_negatives
Our Model (CodeBERT)            612     0.933      0.588   0.227 0.328              10                7             561               34
             Slither            612     0.928      0.000   0.000 0.000               0                0             568               44
             Mythril            612     0.928      0.000   0.000 0.000               0                0             568               44


### Key Findings

- Best F1 Score: Our Model (CodeBERT) with 0.328
- Highest Precision: Our Model (CodeBERT) with 0.588
- Highest Recall: Our Model (CodeBERT) with 0.227
- Our Model (CodeBERT) Coverage: 100.0% (612/612 functions)
- Slither Coverage: 100.0% (612/612 functions)
- Mythril Coverage: 100.0% (612/612 functions)

### Tool Agreement Analysis

- Our Model vs Slither: 97.2% agreement (612 overlapping predictions)
- Our Model vs Mythril: 97.2% agreement (612 overlapping predictions)
- Slither vs Mythril: 100.0% agreement (612 overlapping predictions)


## Methodology Notes

1. **Static Tool Analysis**: Slither and Mythril were run on contract files with a timeout of 60-120 seconds per contract.
2. **Function-Level Mapping**: Tool outputs were mapped to function-level predictions using heuristic approaches.
3. **Evaluation Metrics**: Standard classification metrics (Accuracy, Precision, Recall, F1) were computed.
4. **Limitations**: 
   - Limited sample size for static tool analysis due to computational constraints
   - Function-level granularity mapping may introduce noise
   - Static tools may detect different vulnerability types than our training data

## Conclusions

This benchmark provides insights into the relative strengths and weaknesses of different vulnerability detection approaches. Machine learning models like our CodeBERT implementation may offer advantages in terms of consistency and scalability, while static analysis tools provide rule-based detection with different coverage patterns.

For production use, a hybrid approach combining multiple detection methods may yield the best results.
