# Supplementary-materials-for-LIGNN

## About the paper

We have updated the writing of the paper, including:
- Clarifying notations in Section 2.
- Added discussions about link prediction literature in Section 6.
- Added discussions about 2-FWL in Appendix H.
- Added discussions about limitations and social impacts in Section 8.
  
Other updates include
- Fixed citations.
- Title: link level $\rightarrow$ link-level
- Section 2.2 renamed to be "Message passing for link prediction"
- Added motivations for linearizing the layers in Section 3.2.
- Added more citations of related works
- ...

Major additions to the paper are colored blue.


## About the additional experiments

As reviewers suggested, we provide two additional experiments, including NBFNet for knowledge graph completion and EdgeTransformer for PCQM-Contact. All the models are trained with their official implementations, and tested with the evaluation routine in the LIGNN codes.

### NBFNet
NBFNet's evaluation details are different from here, e.g. when computing rankings, it samples negative triples without replacement. To run the evaluation, first follow the NBFNet's instructions to train a NBFNet model. Follow the scripts `NBFNet/evaluation.ipynb` to load and evaluate the model. The evaluation process in `NBFNet/evaluation.ipynb` is directly taken from the LIGNN code, which provides a fair comparison.

The results are as follows. **Best results** are bold, and <u>secondary results</u> are underlined.

| Dataset     | Metric      | Ours   | NBFNet  | INDIGO
| ----------- | ----------- | ------ | ------- | -------- |
| wn18rr-v1   | e-hits@3    | **86.7**   | <u>81.3</u>    | 12.5 |
|             | r-hits@3    | **99.5**   | 72.7    | <u>98.4</u> |
|             | ACC         | **88.8**   | 84.0    | <u>85.7</u> |
|             | AUC         | **97.5**   | 89.9    | <u>91.2</u> |
| nell995-v1  | e-hits@3    | **54.3**   | <u>51.5</u>    | 39.5 |
|             | r-hits@3    | **95.0**   | 46.0    | <u>80.0</u> |
|             | ACC         | <u>90.5</u>   | **98.5**    | 85.6 |
|             | AUC         | **99.7**   | <u>99.2</u>    | 94.5 |
| fb15k237-v1 | e-hits@3    | <u>60.6</u>   | **62.1**    | 45.1 |
|             | r-hits@3    | **57.4**   | 39.0    | <u>53.1</u> |
|             | ACC         | <u>77.8</u>   | 62.2    | **84.3** |
|             | AUC         | 81.9   | <u>82.5</u>    | **93.4** |

We are sorry that we are only able to run these experiments due to the limited time.

### EdgeTransformer
Run `EdgeTransformer/train_ef.py` to evaluate EdgeTransformer on PCQM-Contact. The codes of the EdgeTransformer model is taken from its official implementation, and the training & evaluation routine is taken from the LIGNN codes.

We have just made the codes work properly and are still running the experiments. We will release the results as soon as the results are out.
