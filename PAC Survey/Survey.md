# Survey

### [An Introduction to Computational Learning Theory](https://direct.mit.edu/books/book/2604/An-Introduction-to-Computational-Learning-Theory)에 대한 리뷰
<details>

<summary>요약</summary>

- ch1 : PAC Learning의 definintion에 대한 내용   
    - Rectangle Learning Game ([링크](https://www.cs.purdue.edu/homes/egrigore/590ST15/lec2DensityEstimation.pdf))
        - 가장 기초적인 pac learnable 한 game에 대한 소개
    - Definition of PAC Learning ([링크](http://sanghyukchun.github.io/66/))
        - PAC Learning의 definition에 필요한 개념들에 대한 소개 (concept, hypothesis, error, confidence, distribution)
    - 3-DNF & 3-CNF ([링크](https://www.cse.wustl.edu/~bjuba/cse513t/s18/notes/l5.pdf))
        - Rectangle Learning과 마찬가지로 가장 기초적인 pac learnable 한 구조에 대한 소개
    - Graph 3-coloring ([링크](https://vatsalsharan.github.io/lecture_notes/lec9_final.pdf))
        - Non PAC learnable에 대한 예시로 3-CNF에 대응되는 NP-Complete 문제인 Graph 3-coloring에 대한 소개
- ch2 : Occam's Razor, Occam Learning에 대한 내용   
    - Occam's Razor ([링크](https://www.cs.princeton.edu/courses/archive/spring18/cos511/scribe_notes/0214.pdf))
        - 오컴의 면도날에서 생각하는 개념으로 hypothesis representation에 대한 compact한 representation을 요구함
    - Set cover problem([링크](https://en.wikipedia.org/wiki/Set_cover_problem))
        - Subset Collection의 union으로 만들어지는 Set cover problem이 Occam's Learning의 아이디어를 통해 PAC learnalbe 함을 보임
    - Learning decision list([링크](https://www.cse.wustl.edu/~bjuba/cse513t/f16/notes/l7.pdf))
        - DNF, CNF가 Decision list의 구조에 포함되며 이것이 PAC-learnable 함을 보임
- ch3 : VC dimension에 대한 내용   
관련 링크 : [VC dimension](https://www.cs.umb.edu/~dsim/cs671-17/S7-VCD.pdf), [Lower bound](http://www.cs.cmu.edu/~ninamf/ML11/lect0927.pdf)
- ch4 : Boosting에 대한 내용  
관련 링크 : [Boosting](https://www.cs.princeton.edu/courses/archive/spring18/cos511/scribe_notes/0312.pdf)
- ch5 : Statistical Query에 대한 내용  
관련 링크 : [Statistical Query](http://www.cs.cmu.edu/~avrim/ML07/lect1207.pdf), [Statistical Query Learnability](https://www.cs.ox.ac.uk/people/varun.kanade/teaching/AML-HT2017/lectures/lecture07.pdf), [SQ algorithms](https://arxiv.org/pdf/2004.00557.pdf)
- ch6 : Discrete Cube Root problem과 관련한 PAC learnablity에 대한 내용   
관련 링크 : [Discrete Cube Root Problem](https://www.cs.ox.ac.uk/people/varun.kanade/teaching/CLT-MT2018/lectures/lecture05.pdf)
- ch7 : Concept의 reducing에 대한 내용   
관련 링크 : [Monotone DNF](http://www.cs.cmu.edu/afs/cs/user/srallen/www/papers/dnfeval.pdf), [Finite Automata](https://lutter.cc/thesis.pdf)   
- ch8 : Exact Learning과 관련된 내용   
관련 링크 : [Membership and Equivalence Queries](https://www.cs.ox.ac.uk/people/varun.kanade/teaching/CLT-MT2018/lectures/lecture06.pdf)

</details>


### 논문 찾은 것들

교수님이 말씀하신 논문들, 제가 검색하여 찾은 논문들, 그리고 [링크](https://docs.google.com/spreadsheets/d/10-lexYyn9TEy9R5KZHv3qLNAfGO8ubRVTSQoEfqj05s/edit#gid=0)에 있는 논문들을 우선적으로 살펴보았습니다

<details>
<summary>2020</summary>

#### 2020
- Statistical Queries and Statistical Algorithms: Foundations and Applications ([링크](https://arxiv.org/pdf/2004.00557.pdf))   
    - **Statistical Query와 Statistical Algorithm : 기초와 응용**
        - Statistical Query의 기초와 응용에 대한 survey 논문
        - 모델과 정의, 이론과 이것이 다양한 개념에 어떻게 적용되는지를 살펴본다
        - Statistical Query가 최적화, 진화, 차등 프라이버시에 대해 응용되는 것에 대해 요약되어있다

- Reducing Adversarially Robust Learning to Non-Robust PAC Learning ([링크](https://arxiv.org/abs/2010.12039.pdf))
    - **Adversarially Robust Learning을 Non Robust PAC Learning으로 reduce하는 것**
        - Adversarially Robust Learning을 standard PAC learnable한 문제로 reduce하는 것에 대해 연구하였다
        - Black-Box-Non-Robust Learner에 대한 access만 허용한다
        - Hypothesis class C에 대해 non-robust learner A를 이용하여 robust하게 learn 할 수 있도록 reduction을 줄 수 있다
        - A에 대한 호출 수는 adverarial perturbation per example에 log scale로 변하며, 이것에 대한 하한이 존재한다

- On the Sample Complexity of Adversarial Multi-Source PAC Learning
 ([링크](http://proceedings.mlr.press/v119/konstantinov20a/konstantinov20a.pdf))
    - **Adversarial Multi-source PAC Learning의 Sample Complexity**
        - 크라우드 소싱, 협업 학습 패러다임에서 발생하는 신뢰할 수 없는 데이터에 대한 학습에 대한 연구이다
        - 구체적으로, 다양한 소스에서 데이터를 얻지만 편향되어있거나 적대적으로 교란된 경우를 연구한다
        - 단일 소스의 경우 훈련데이터의 고정된 부분을 손상시키면 PAC Learnablity를 방지할 수 있다
        - 데이터의 개수와 관계없이, 학습 시스템이 optimal test error에 다다를 수없다
        - 하지만 이는 다양한 소스의 경우 다른 결과를 나타낸다
        - 연구에서 finite-sample quarantee 뿐만 아니라 일반적인 lowewr bound에 대한 결과를 얻었다
        - 악의적인 참가자도 데이터를 공유하는 과정에서 benefit을 가짐을 보인다
- Algorithms and SQ Lower Bounds for PAC Learning One-Hidden-Layer ReLU Networks ([링크](http://proceedings.mlr.press/v125/diakonikolas20d/diakonikolas20d.pdf))

    - **PAC Learning One-Hidden-Layer ReLU Network에 대한 algorithm과 SQ Lower Bounds**
        - Gaussian marginal을 label noise로 가지고 k개의 hidden unit이 있는 PAC Learning One-Hidden-Layer에 대한 연구이다
        - k가 O(sqrt(logd))임을 보였다
        - k=3과 같은 간단한 경우에도 이를 구하는 알고리즘은 알려지지 않았다
        - 연구에서 찾은 알고리즘은 weigh matrix의 rank에 대한 가정이 없으며 복잡성은 condition number와 무관하다
        - Statistical Query의 lower bound가 dΩ(k) 인 것 또한 보였다
        - 이번 연구를 통해 상한과 하한이 확장되었다
</details>

<details>
<summary>2021</summary>

#### 2021

- Sequential prediction under log-loss and misspecification ([링크](https://arxiv.org/pdf/2102.00050.pdf))
    - **log-loss와 misspecifiation하에서의 sequential prediction**
        - cumulative regret의 log-loss term에 대한 sequential predicton 문제에 대한 연구이다
        - 분포에 대한 가설이 주어졌을 때, 학습자는 순차적으로 예측을 진행하고 가장 좋은 예측과 비교된다
        - well-specified case는 데이터의 분포가 가설에 속한다는 추가적인 가정이 있는 것이다
        - 더 general한 misspecified case에 대해 연구해본다
        - log-loss의 특징으로 인해 밀도 추정 및 모델 선택에서 동일한 문제가 발생한다
        - d차원 Gaussian location 가설에 대해 specified case와 misspecified case가 점근적으로 일치함을 보인다
        - 즉, 이 경우 PAC regret의 O(1) characterization이 존재한다
        - 최악의 경우 regret은 d2 + o(1) 보다 크다
        - 전통적인 Bayesian estimator나 Shtarkov의 정규화된 최대우도는 PAC regret을 달성하지 못하고 heavy-tailed data에 대해 robustification이 요구된다
        - 추가적으로, optimal estimator에 대해 존재성과 유일성, misspecified regret의 bound에 대한 결과를 보였다 
- Efficient Competitions and Online Learning with Strategic Forecasters ([링크](https://arxiv.org/abs/2102.08358))
    - **Strategic Forecaster를 이용한 효율적인 경쟁과 online learning**
        - 승자독식경쟁의 예측은 왜곡된 incentive로 인해 학습이 어렵다
        - 이에 대한 해결을 위해 2018년 ELF라는 매커니즘이 제안되었다
        - n명의 forecaster가 있을 때 ELF가 높은 확률로 최적의 forecaster를 찾기 위해서는 Θ(nlogn)의 event나 데이터가 필요하다
        - 표준 online-learning algoritm에서는 O(log(n)/ϵ2)의 event를 사용하여 ϵ명의 최적화된 forecaster를 선택할 수 있다
        - 이 범위는 비 전락적인 설정에서도 잘 일치한다
        - 이후 이러한 매커니즘을 통해 forecater에 대한 non-regret guarantee를 얻는다
- PAC-Learning for Strategic Classification ([링크](https://arxiv.org/pdf/2012.03310.pdf))
    - **Strategic Classification을 위한 PAC Learning**
        - Classifier를 속이기 위해 testing data에 대한 adversarial manupulation에 대한 연구가 관심을 많이 받았다
        - 이전의 작업은 모든 테스트 데이터가 adverarial하거나 모두 positive 한 극단적인 상황을 가정하였다
        - 이번 연구에서는 이 두가지를 일반화하고 strategic vc-dimension의 개념을 통해 PAC-learnability를 확인한다
        - SVC는 adversarial vc-dimension 의 개념인 AVC를 일반화한다
        - startegic linear classification에 대한 framework을  instance화 한다
        - SVC를 고정하여 linear classifier의 statistical learnablility를 구한다
        - empirical risk minimization의 complexity를 고정하여 computational tractability를 구한다
        - SVC는 VC의 상한이다
        - 이는 AVC의 bound를 일반화한다
- Incentive-Aware PAC Learning ([링크](https://ojs.aaai.org/index.php/AAAI/article/view/16726))
    - **Incentive Aware PAC Learning**
        - staregic manipulation이 있는 PAC Learning에 대한 연구이다
        - ERM principle는 non trivial guarantee에 다다를 수 없다
        - 대신에 Incentive Aware version의 ERM principle을 생각해서 최적의 샘플 복잡성을 점근적으로 가지도록 한다
        - 이후 incentive compatible classifier를 통해 전략적 조작을 방지한다
        - incentive-compatible classifer로 제한된 ERM principle에 대해 sample complexity bound를 제공한다
- Symbolic Abstractions From Data: A PAC Learning Approach
 ([링크](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9683316))
    - **PAC Learning Approach로 구한 데이터의 symbolic abstraction**
        - 기호 제어 기술은 높은 logic specification을 만족시키는 것을 목표로 한다
        - 중요한 단계는 주어진 continuous state system을 모방하는 finite state system의 추상화를 구성하는 것이다
        - 그러나 이를 위해서는 정확한 closedform model에 대한 지식이 필요하다
        - unknown dynamics를 일반화 하기 위해서, closedform model이 필요 없고 후속 작업을 평가하는 기능에 의존하는 방식이 제시된다
        - 학습에 대한 보장을 제공하기 위해 PAC Learning framework을 사용한다
        - 필요한 데이터의 수를 지정하는 PAC 범위를 제공한다
</details>

<details>
<summary>2022</summary>

#### 2022   
- On the Power of Differentiable Learning versus PAC and SQ Learning ([링크](https://arxiv.org/pdf/2108.04190.pdf))
    - **Differentiable Learning vs PAC & SQ Learning**
        - SGD, GD와 같은 방법으로 어떠한 문제를 학습할 수 있는지 연구하였다
        - SGD와 GD는 SQ를 학습할 수 있지만, 그 가능성은 minibatch size나 sample size에 연관되어있는 precision에 영향을 받는다
        - precision * minibatch size가 충분히 작을 때, SGD는 SQ Learning을 넘어 PAC Learnable함을 알 수 있다
        - 이는 기존의 연구 결과인 batch size = 1인 경우를 확장 시킨 결과이다
        - sample size m에 대해서도 GD를 이용하여 모든 sample 기반 학습이 가능하다
        - ρ가 매우 작은 경우, minibatch size와 관계 없이 SGD와 GD 모두 PAC Learning이 가능하다
        - bρ^2이 충분히 큰 경우, SGD는 SQ Learning과 비슷한 성능을 보인다
- Hardness of Noise-Free Learning for Two-Hidden-Layer Neural Networks ([링크](https://arxiv.org/pdf/2202.05258.pdf))
    - **2층 신경망에 대한  noise-free learning hardness**
        - Gaussian input을 넣은 2층 ReLU 신경망의 형태를 가지고 있는 noise-free model에 대한 SQ lower bound를 구하였다
        - 모든 깊이의 ReLU 신경망에 대해 일반적으로 알려진 SQ lower bound는 없다
        - 기존의 SQ lower bound는 adversarial noise model이나 correlational SQ와 같은 제한적인 조건에서만 구해진다
        - 기존의 방법에서 생기는 문제점을 해결하기 위해 Boolean PAC 문제를 Gaussian 문제로 변형해서 lifting procedure를 개선한다
        - 이를 이용해서 2층 ReLU 신경망에 대한 새로운 SQ lower bound를 구하였다.
- Near-Optimal SQ Lower Bounds for Agnostically Learning Halfspaces and ReLUs under Gaussian Marginals ([링크](https://arxiv.org/pdf/2006.16200.pdf))
    - **Gaussian marginal 하에서 halfspace와 ReLU를 모델에 관계없이 학습할때의 Near-Optimal SQ Lower Bound**
        - Gaussian marginal 하에서 halfspace와 ReLU를 모델에 관계없이 학습하는 문제에 대한 연구이다
        - 기존 문제는 R^d×{±1} distribution과  (x, y) labeled data가 주어졌을 때 0-1 loss OPT hypothesis를 찾는 것이다
        - 이를 확장해서 R^d×{R} distribution과  (x, y) labeled data가 주어졌을 때 square loss OPT hypothesis를 찾는 문제를 생각한다
        - 두 문제에 대해 SQ lower bound가 d poly(1/ǫ)임을 보였다
        - 이는 현재의 upper bound가 essentially best possible임을 보인다
- Conformal Inference for Online Prediction with Arbitrary Distribution Shifts ([링크](https://arxiv.org/pdf/2208.08401.pdf))
    - **무작위로 분포가 변화하는 online prediciton에 대한 conformal inference**
        - conformal inference는 blackbox model로 만든 prediction의 성능을 정량화하는 강력한 도구이다
        - 이 방법은 training set과 test set이 exchangeable, iid인 경우 prediction에 대한 target level을 줄 수 있다는 것이다
        - 이 연구에서는 데이터의 분포가 시간에 따라 변하는 online predciton setting인 경우에 대해 다루고 있다
        - non-exchangeability가 있기 때문에, protective layer를 만들어 prediction을 보정한다
        - 분포변화에 대한 가정이나 분포유형에 대한 가정 없이 조건부 분포의 추정치를 생성한다
        - 주식시장의 변동성과 covid 사례 수에 대한 예측에 대해 test해보고 이것이 실제로 적응력이 있음을 확인했다
- Integral Probability Metrics PAC-Bayes Bounds ([링크](https://arxiv.org/pdf/2207.00614.pdf))
    - **IPM(Integral Probability Metrics) PAC-Bayes Bounds**
        - KL-Divergence를 다양한 IPM을 이용하여 대체하는 PAC-Bayes Bound를 제시한다
        - 이 경우 IPM은 total variation과 Wasserstein distance이다
        - 이 bound는 분포가 멀리 떨어진 경우와 적당히 멀어진 경우 사이의 경계를 자연스럽게 나타낸다
        - 큰 가설공간을 사용하는 알고리즘을 분석하는 것에 더 적합하다
- Group symmetry in PAC learning ([링크](https://openreview.net/pdf?id=HxeTEZJaxq))
    - **PAC learning에서의 group symmetry**
        - 이 연구에서는 invariant 또는 equivaraint hypothesis가 orbit representative로 어떻게 reduce 되는지 보여준다
        - 이 결과는 compact group에 대해 모두 적용되며 rotation과 같은 infinite group에서도 적용된다
        - 이를 이용해서 invarant/equivaraint model에 대한 일반화를 도출한다
        - 이는 현재까지의 가장 일반화된 결과이다
- Fairness-Aware PAC Learning from Corrupted Data ([링크](https://dl.acm.org/doi/pdf/10.5555/3586589.3586749))
    - **손상데이터를 사용한 Fairness-Aware PAC Learning**
        - Fariness 문제를 생각하는 것은 실제 automated system을 채택하기 위한 중요한 단계이다
        - 데이터 손상에 대한 robustness는 아직 많이 연구되지 않았다
        - 이 연구에서는 최악의 데이터에서 fairness-aware learning이 가능한지 확인한다
        - 이 연구에서는 fairness와 accuracy가 최적화된 2가지 알고리즘을 제시한다
</details>

<details>
<summary>2023</summary>

#### 2023   
- Revisiting Fair-PAC Learning and the Axioms of Cardinal Welfare
 ([링크](https://proceedings.mlr.press/v206/cousins23a/cousins23a.pdf))
     - **Fair-PAC Learning과 Axioms of cardinal welfarism에 대한 재정의**
        - 사회적 목표는 효용과 비효용을 전체 그룹에서 요악하여 fair machine learning에서의 목표를 정의한다
        - Standard axiom 하에서 welfare function은 p ≤ 1, malfare fuction은  p ≥ 1를 만족한다
        - p를 제한하는 더 강한 공리를 찾아냈다
        - power-mean malfare function는 lipschitz 연속이므로 학습하기 쉽다
        - 모든 power mean이 locally holder continuous함을 보였다

- Probably Approximately Correct Federated Learning
 ([링크](https://arxiv.org/abs/2304.04641.pdf))
    - **PAC Federated Learning**
        - Federated Learning(FL)은 분산 학습 프레임워크이다
        - 기존의 연구에 따르면 개인정보 보호와 유용성,효율성을 동시에 달성하기는 어렵다
        - 세 가지 요소에 대한 절충점을 찾는 방법은 신뢰할 수 있는 FL에 대한 중요한 문제이다
        - 이를 다목적 최적화 문제로 생각할 수 있다
        - 기존의 프레임워크는 시간이 많이 걸리고 존재성을 알 수 없기에 적합하지 않다
        - 이를 위해 pac learning을 이용하여 여러가지를 정량화하는 FedPAC를 제안한다
        - 이는 solution sapce의 dimension을 낮추고, 이에 대한 단일 목표 최적화 알고리즘을 사용하여 문제를 해결한다
</details>

<details>
<summary>etc</summary>

#### 2020   

- Assumption-lean inference for generalised linear model parameters ([링크](https://arxiv.org/pdf/2006.08402.pdf))
    - **일반화된 선형모델에 변수에 대한 가정의존성 추론**
        - 일반적으로 선형모델의 변수에 대한 추론은 일반적으로 모델이 정확하고 선험적으로 지정된다는 가정을 기반으로 한다
        - 선택되는 모델은 데이터에 적응한 모델을 고르는 형태의 process로 선택되어지기에 불확실성을 가질 수 있다
        - 모델에 주어진 가정은 선험적이지 않은 편향된 추론을 만들고, 데이터의 정보를 순수하게 반영하지 못하게 한다
        - 주효과 추정치와 효과 수정 추정치에 대한 새로운 비모수적인 정의를 제안한다
        - 이는 projection parameter와 같이 가정이 없는 추론에서 영감을 받은 것이다
        - 모델이 잘 지정되면 주효과 추정치와 효과 수정 추정치는 감소된다
        - 그러나 모델이 잘못 지정된 경우에도 두 변수가 상호작용하는 정도를 잘 찾아낸다
        - 비모수적 모델에서 influenc curve를 도출하고 유연한 데이터 적응 절차를 호출하여 가정에 기대지 않는 추론을 얻을 수 있다

- Asymptotics of Cross-Validation ([링크](https://arxiv.org/pdf/2001.11111.pdf))
    - **cross-validation의 점근성**
        - cross-validation은 머신러닝 모델의 성능을 평가하는 핵심 도구이다
        - 그럼에도 불구하고, 이의 이론적 속성은 아직 잘 이해되지 않았다
        - 다양한 모델의 cross-validated risk의 점근적 특성에 대해 연구했다
        - 안정적인 조건하에, 중심극한정리와 Berry-Esseen Bound의 설정으로 신뢰구간을 점근적으로 계산할 수 있다
        - 결과를 통해 cross-validation의 통계적 속도 향상에 대한 아이디어를 얻을 수 있다
        - M-estimator가 trian loss 관점에서 cross-validation의 속도향상에 도움이 된다
        - cross-validation risk는 분산 감소와 함께 복잡하며, 모델의 크기나 기본 분포에 따라 다르다
        - Kn-fold의 Kn이 관찰수에 따라 증가하도록 한다

#### 2021   
- The Price of Tolerance in Distribution Testing ([링크](https://arxiv.org/pdf/2106.13414.pdf))
    - **distribution testing에서의 tolerance price**
        - distribution testing에서의 tolerance 문제에 대해 연구한다
        - unknown distribution p에서 나온 smaple은 reference distribution q로부터 얼마나 떨어져있는지 알 수 있을까?
        - 이 문제는 극단적인 경우에만 설명이 가능하다
        - 예를 들어 noiseless setting의 경우 복잡도는 Θ(√n)이고, 선형적이지 않다
        - ε1 = ε2/2일 때, 복잡도는 Θ(n/ log n)이다
        - 그 사이의 단계에 대해서는 알려진 것이 거의 없다
        - n, ε1, ε2, log(n)으로 tolerance를 표현할 수 있다
        - p와 q를 모두 모르는 경우에도 비슷한 결과를 보인다
        - 복잡도는 직관적인 ε1/ε2가 아닌 ε1/ε2^2에 의해 크게 결정된다
        - ε1과 ε2의 비대칭성을 이용하여 하한을 구하는 framework을 찾아냈다

- Minimax bounds for estimating multivariate Gaussian location mixtures([링크](https://arxiv.org/pdf/2012.00444.pdf))
    - **multivariate Gaussian location의 minimax bound**
        - L2^2과 Hellinger loss function^2에서 Gaussian location의 minimax bound를 찾았다
        - L2^2 loss에서 minimax rate는 n^−1(log n)^d/2의 상수배로 bound된다
        - Hellinger loss function^2에서는 tail을 기준으로 2개의 하위 클래스를 고려한다
        - subGaussian tail이 있는 경우, (log n)^d/n로 bounded below 된다
        - bounded pth moment를 가지는 경우, n^−p/(p+d)(log n)^−3d/2로 bounded below 된다
        - 이들은 log scale로 나타나진다
- Threshold Martingales and the Evolution of Forecasts ([링크](https://arxiv.org/pdf/2105.06834.pdf))
     - **Threshold Martingale과 Evolving Forecast**
        - Evolving Forecast distribution의 두 가지 속성을 설명하는 martingale에 대한 연구이다
        - 이상적인 Forecast는 martingale과 같이 사용가능한 정보를 활용하기 위해 forecast를 순차적으로 진행한다
        - Threshold Martingale은 threshold 밑의 예측분포의 비율을 측정한다
        - 보정 조정은 잘 알려져 있으며 martingale filter를 통해 변동성을 개선하여 더 작은 mse를 보장한다
        - 시뮬레이션 모델의 예측에 threshold martingale을 적용하여 농구 경기의 승자를 예측하는 모델에 적용한다
- Higher Order Kernel Mean Embeddings to Capture Filtrations of Stochastic Processes ([링크](https://arxiv.org/pdf/2109.03582.pdf))
    - **고차원 kernel mean embedding을 이용한 Stochastic Process의 filtration capture**
        - Stochastic Process는 경로의 일부 공간에 값이 있는 random 변수이다
        - 그러나 Stochastic Process를 path-valued random varaible로 reduce하면 시간에 따라 전달되는 정보의 흐름이 무시된다
        - 필터링에 대한 process를 조절하여 KME의 개념을 일반화하고 추가적인 정보를 얻는 고차원 KME를 도입한다
        - MMD에 대한 empirical estimate를 도출하고 일관성을 증명한다
        - 이후 MMD에서 놓친 정보를 선택한 filtration-sensitive kernel two-sample test를 구성한다
        - 추가적으로 고차원 MMD를 활용하여 kernel 기반 회귀모형을 이용하여 파생상품의 pricing같은 문제에서 optimal stopping problem을 해결한다
        - 조건부 독립에 대한 기존 테스트를 적용하여 다차원 궤적의 관찰에서 상호작용하는 인과 발견 알고리즘을 설계한다
- Incentive-Compatible Forecasting Competitions ([링크](https://arxiv.org/abs/2101.01816.pdf))
    - **incentive-compatible 예측 경쟁**
        - 여러 예측자가 1개 이상의 event에 대해 예측하고 결과에 대해 경쟁하는 incentive-compatible 예측 경쟁에 대한 연구이다
        - 2가지 목표가 있다 - 예측자가 진실하게 report하고 정확한 예측자에게 award가 이루어지는 것
        - 적절한 점수 규칙은 예측자가 올바르게 incentive를 받는 경우 진실된 report를 장려한다
        - 그러나 가장 높은 점수를 받는 사람만 상을 받으면 incentive가 왜곡되고, 높은 점수를 받기 위해 극단적인 결과를 보고한다
        - 이를 막기 위해 2가지 새로운 예측 경쟁 매커니즘을 제시한다
        - 1번째 매커니즘은 인센티브와 호환되어 가장 정확한 예측자를 선택한다
        - 더 높은 확률로 가장 정확한 예측자츨 선택하는 매커니즘이 없음을 보일 수 있다
        - 2번째 매커니즘은 이벤트에 대한 정보가 다른 이벤트에 대해 업데이트 되지 않을 때 인센티브와 호환되며, 이벤트 수가 증가함에 따라 최상의 예측가를 선택한다
        - incentive-compatible에 대한 이 논문의 개념은 dominant strategy incentive compatibility에 비해 일반적이다
        - 2가지 매커니즘은 구현하기 쉽고 예측가에 대한 순위를 매기거나 높은 정확도를 가지는 예측가를 고용하는 문제에 사용될 수 있다

#### 2022
- Conditional Versus Unconditional Approaches to Selective Inference ([링크](https://arxiv.org/pdf/2207.13480.pdf))
    - **선택적 추론에 대한 조건부 대 무조건부 접근**
        - 선택 이벤트를 조건으로 하는 선택적 추론을 위한 방법 클래스를 조사합니다
        - 이러한 방법은 2단계 프로세스로 작동합니다
        - 첫째, hypothesis의 모음은 hyphothesis space에서 데이터 기반 방식으로 결정됩니다
        - 둘째, 선택에 사용된 정보에 따라 데이터 기반 수집 내에서 추론이 수행됩니다
        - 이러한 방법의 예로는 기본 데이터 분할, 최신 데이터 조각 방법 및 다면체 기본형을 기반으로 한 사후 선택 추론 방법이 있습니다
        - 이 백서에서는 이러한 방법에 대한 전체적인 관점을 채택하여 선택, 조건 지정 및 최종 오류 제어 단계를 단일 방법으로 함께 봅니다
        - 이러한 관점에서 우리는 선택 및 조건화에 기반한 선택적 추론 방법이 항상 hyphothesis space에서 직접 정의된 여러 테스트 방법에 의해 지배된다는 것을 보여줍니다
        - 이 결과는 hyphothesis space가 잠재적으로 무한하고 데이터 분할과 같이 암시적으로만 정의된 경우에도 유지됩니다
        - 우리는 비선택적 및/또는 무조건적 관점으로 전환함으로써 얻을 수 있는 잠재적인 힘의 네 가지 사례 연구를 조사한다
- Permutation tests using arbitrary permutation distributions ([링크](https://arxiv.org/pdf/2204.13581.pdf))   
- Debiased Machine Learning without Sample-Splitting for Stable Estimators
 ([링크](https://arxiv.org/pdf/2206.01825.pdf))
    - **안정 추정을 위한 Sample-splitting을 사용하지 않은 debiased machine learning**
        - 인과추론은 일반적인 moment problem 문제의 일반화이며, regression이나 classification 문제를 포함한다
        - debiased machine learning의 최근 연구는 이러한 문제를 풀기 위해서는 알고리즘에 mse 조건이 필요하다
        - 선행연구에서는 문제를 풀기 위해서는 sample splitting이 주어지거나 corss-fitting 방식이 주어져야 한다고 주장한다
        - 이 연구에서는 leave-one-out stability가 있는 경우 sample splitting이 필요 없음을 보인다
        - 이를 통해 sample을 재사용할 수 있으며, 중간 크기의 sample을 사용할 경우 유용할 수 있다
- Deploying the Conditional Randomization Test in High Multiplicity Problems ([링크](https://arxiv.org/pdf/2110.02422.pdf))
    - **High Multiplicity Problem에 대한 CRT(Conditional Randomization Test)**
        - 이 연구에서는 CRT와 selective seqstep+를 결합한 Sequential CRT를 연구하였다
        - p값은 flexivale CRT에 의해 구성되고, 이후 정렬된 후 SeqStep+ 필터를 통과하여 discoveries를 생성한다
        - 연구에서는 p값이 독립적이지 않더라도 FDR에 대한 제어가 보장되는 이론적 근거를 찾았다
        - 시뮬레이션을 통해 새로운 절차가 실제로 FDR을 제어하고 전력 측면에서 sota 모델을 능가하는 것을 보였다
        - 유방암 데이터셋에 대한 biomaker identification을 통해 결과를 확인한다
- The Hardness of Conditional Independence Testing and the Generalised Covariance Measure
 ([링크](https://arxiv.org/pdf/1804.07203.pdf))
- Local permutation tests for conditional independence ([링크](https://arxiv.org/pdf/2112.11666.pdf))
- The edge of discovery: Controlling the local false discovery rate at the margin ([링크](https://arxiv.org/pdf/2207.07299.pdf))
- A regret-variance tradeoff in online learning ([링크](https://arxiv.org/abs/2206.02656.pdf))
- Continuous prediction with experts' advice ([링크](https://arxiv.org/abs/2206.00236.pdf))
- Post-Selection Inference via Algorithmic Stability ([링크](https://arxiv.org/pdf/2011.09462.pdf))


#### 2023
- Training-conditional coverage for distribution-free predictive inference ([링크](https://arxiv.org/pdf/2205.03647.pdf))
- Efficient Concentration with Gaussian Approximation ([링크](https://arxiv.org/pdf/2208.09922.pdf))
- Reconciling Individual Probability Forecasts ([링크](https://arxiv.org/pdf/2209.01687.pdf))
    - **개별 확률 예측 조정**
        - 개별 확률은 한 번만 실현되는 결과의 확률을 나타냅니다
        - 내일 비가 올 확률, Alice가 향후 12개월 이내에 사망할 확률, Bob이 향후 18개월 이내에 폭력 범죄로 체포될 확률 등
        - 개별 확률은 근본적으로 알 수 없습니다
        - 그럼에도 불구하고 우리는 데이터 또는 데이터 분포에서 샘플링하는 방법에 대해 동의하는 두 당사자가 개별 확률을 모델링하는 방법에 동의하지 않을 수 있음을 보여줍니다
        - 이것은 실질적으로 동의하지 않는 개별 확률의 두 모델이 함께 사용되어 두 모델 중 적어도 하나를 경험적으로 위조하고 개선할 수 있기 때문입니다
        - 이것은 "조정" 프로세스에서 효율적으로 반복될 수 있으며, 그 결과 양 당사자가 동의한 모델이 시작 모델보다 우수하고 자체적으로 (거의) 모든 곳에서 개별 확률의 예측에 (거의) 동의합니다
        - 우리는 개별 확률을 알 수 없지만 계산 및 데이터 효율적인 프로세스를 통해 합의에 도달해야 한다는 결론을 내립니다
        - 따라서 우리는 때때로 예측 또는 모델 다중성 문제라고 불리는 문제에 대한 답을 제공하면서 예측에서 크게 동의하지 않는 똑같이 정확하고 개선 불가능한 두 모델이 있는 상황에 처할 수 없습니다
- Improved Online Conformal Prediction via Strongly Adaptive Online Learning ([링크](https://arxiv.org/pdf/2302.07869.pdf))

</details>

### code 구현된 것들

- Improved Online Conformal Prediction via Strongly Adaptive Online Learning ([링크](https://github.com/salesforce/online_conformal))
- Integral Probability Metrics PAC-Bayes Bounds code ([링크](https://github.com/ron-amit/pac_bayes_reg))
-  Higher Order Kernel Mean Embeddings to Capture Filtrations of Stochastic Processes ([링크](https://github.com/maudl3116/higherOrderKME))
- Deploying the Conditional Randomization Test in High Multiplicity Problems ([링크](https://github.com/lsn235711/sequential-CRT))
- Training-conditional coverage for distribution-free predictive inference ([링크](https://rinafb.github.io/research/))




### **앞으로 할 것들**

- code가 구현되어있는 논문에 대해서 코드 리뷰해보기
- 관심있는 몇몇 논문들 관련해서 살펴보기
- 책 관련 읽은 내용 관련해서 정리해보기

