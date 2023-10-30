# FITs_Algorithm_Model
The source code and models for paper "Federated Intelligent Terminals Facilitate Stuttering Monitoring"

## Abstract
Stuttering is a complicated language disorder. The most common  form of stuttering is developmental stuttering, which begins  in childhood. Early monitoring and intervention are essential  for the treatment of children with stuttering. Automatic  speech recognition technology has shown its great potential  for non-fluent disorder identification, whereas the previous  work has not considered the privacy of users  data. To  this end, we propose federated intelligent terminals for automatic  monitoring of stuttering speech in different contexts.  Experimental results demonstrate that the proposed federated  intelligent terminals model can analyse symptoms of stammering  speech by taking the personal privacy protection into  account. Furthermore, the study has explored that the Shapley  value approach in the federated learning setting has comparable  performance to data-centralised learning.

#### Index Terms— Stuttering Monitoring, Federated Learning, Computer Audition, Healthcare

![](/figures/FL.jpg)

Fig. 1. Basic structure of the proposed privacy-preserving FL framework.
 
## Results
 * Concentrated learning with XGBoost(Non-Federated) 
  * Federated Learning with XGBoost (Homogeneous SecureBoost)
  
 ![](/figures/results.jpg)
 
 Fig. 2. Model performance variation (UAR and UF1 in [%]) between centralised learning and federated learning for multiclassification of stuttering data.
 
Table 2. Optimal values for the depth and number of trees; summary of experimental results (in [%]).

| Model      | XGBoost     | FL-SecureBoost |
| -----      | -----       | ----           |
| Tree depth |  5          | 3              |
| Tree number|  50         | 50             |
| UF1        |  33.1%      | 30.8%          |
| UAR        |  31.3%      | 29.5%          |

The important parameters are set, e. g., learning rate=0.3, subsample feature rate=1.0, and other parameters to their default values.

 ![](/figures/matrix.jpg)
 
 Fig. 3. Normalised confusion matrix (in [%]) of the FL.
 
![](/figures/shap1.jpg)

(a) The contribution of significant auDeep features from all class predictions for the XGBoost model (average feature importance).
  
![](/figures/shap2.jpg)

(b) The contribution of significant auDeep features from all class predictions for the FL model (average feature importance).

Fig. 4. The plot sorts the features by the mean of Shapley values for all class predictions and uses the Shapley values to show the average impact on the model output magnitude of the features. Top 10 most impactful features are shown above.
  
## Availability
1. KSoF: Access to the data can be requested from the Kassel State of Fluency (KSoF) dataset at https://zenodo.org/record/6801844.

2. SHAP (SHapley Additive exPlanations) is a game-theoretic method to explain the output of ML models. https://shap.readthedocs.io.

3. FATE (Federated AI Technology Enabler) supports the FL architecture, as well as the secure computation and development of various ML algorithms. https://github.com/FederatedAI/FATE.

## References
[1] Bjoern Schuller, Anton Batliner, Shahin Amiriparian, Christian Bergler, Maurice Gerczuk, Natalie Holz, Pauline Larrouy-Maestri, Sebastien Bayerl, Korbinian  Riedhammer, Adria Mallol-Ragolta, et al., The ACM Multimedia 2022 Computational Paralinguistics Challenge: Vocalisations, Stuttering, Activity, &amp; Mosquitoes, in Proceedings of the 30th ACM International Conference on Multimedia, 2022, pp. 7120 7124.

[2] Sebastian P Bayerl, Alexander Wolff von Gudenberg, Florian H onig, Elmar N oth, and Korbinian Riedhammer, KSoF: The Kassel State of Fluency Dataset A Therapy Centered Dataset of Stuttering, arXiv preprint arXiv:2203.05383, pp. 1 8, 2022.

[3] Kun Qian, Zixing Zhang, Yoshiharu Yamamoto, and Bjoern W Schuller, Artificial Intelligence Internet of Things for the Elderly: From Assisted Living to Healthcare  Monitoring, IEEE Signal Processing Magazine, vol. 38, no. 4, pp. 78 88, 2021.

[4] Wanyong Qiu, Kun Qian, Zhihua Wang, Yi Chang, Zhihao Bao, Bin Hu, Bjoern W Schuller, and Yoshiharu Yamamoto, A Federated Learning Paradigm for Heart Sound Classification, in Proceedings of the Engineering in Medicine &amp; Biology Society (EMBC). IEEE, 2022, pp. 1045 1048.

## Cite As
Yongzi Yu, Wanyong Qiu, Chen Quan, Kun Qian, Zhihua Wang, Yu Ma, Bin Hu∗, Bjoern W. Schuller and Yoshiharu Yamamoto, “Federated intelligent terminals facilitate stuttering monitoring”, in Proceedings of ICASSP, pp. 1-5, October 2022.


