# NLP 주요 개념

##  개요
이 문서는 자연어 처리(NLP)와 대형 언어 모델(LLM)의 핵심 개념인 코퍼스, 임베딩, RAG(Retrieval-Augmented Generation)에 대해 설명합니다.

죄송합니다. 제가 내용을 충분히 포함시키지 않았네요. 원본 문서의 모든 내용을 유지하면서 챕터 1에 대한 내용을 1.1, 1.2 등의 세부 항목으로 완전히 정리해드리겠습니다.

# 1. 코퍼스(Corpus)의 이론적 기반과 활용

## 1.1 코퍼스의 정의와 기본 개념

코퍼스(Corpus, 복수형: Corpora)는 자연어 처리(NLP)와 언어 모델 학습에 사용되는 대규모 텍스트 데이터 모음을 의미합니다. 이는 언어 모델이 학습하는 기본 자료로서, 모델의 성능과 지식 범위를 결정짓는 핵심 요소입니다.

## 1.2 코퍼스의 수학적 정의

코퍼스 $C$는 일련의 문서 집합으로 정의할 수 있습니다:

$C = \{D_1, D_2, ..., D_n\}$

여기서 각 문서 $D_i$는 토큰(단어, 부분단어, 문자 등)의 시퀀스입니다:

$D_i = (t_1, t_2, ..., t_m)$

토큰 집합 $V$는 코퍼스의 어휘(vocabulary)를 구성합니다:

$V = \{t | t \in D_i \text{ for some } i \in \{1,2,...,n\}\}$

## 1.3 코퍼스 유형 분류

### 1.3.1 규모에 따른 분류
- **소규모 코퍼스**: 수백만 단어 미만
- **중규모 코퍼스**: 수백만~수억 단어
- **대규모 코퍼스**: 수억~수조 단어 (예: Common Crawl, The Pile)

### 1.3.2 구성에 따른 분류
- **균형 코퍼스(Balanced Corpus)**: 다양한 주제, 장르, 스타일을 균형 있게 포함
- **전문 코퍼스(Specialized Corpus)**: 특정 도메인이나 장르에 초점
- **참조 코퍼스(Reference Corpus)**: 언어의 대표적 사용을 포괄적으로 포함

### 1.3.3 시간적 범위에 따른 분류
- **통시적 코퍼스(Diachronic Corpus)**: 시간에 따른 언어 변화 연구용
- **공시적 코퍼스(Synchronic Corpus)**: 특정 시점의 언어 상태 연구용

### 1.3.4 언어적 특성에 따른 분류
- **단일어 코퍼스(Monolingual Corpus)**: 단일 언어로 구성
- **병렬 코퍼스(Parallel Corpus)**: 동일 내용의 여러 언어 버전 포함
- **비교 코퍼스(Comparable Corpus)**: 유사 주제의 다언어 텍스트 모음

## 1.4 코퍼스 구축과 처리의 수학적 기반

### 1.4.1 코퍼스 구축 방법론

코퍼스 구축은 다음과 같은 단계를 포함합니다:

1. **데이터 수집**: 웹 크롤링, 디지털 아카이브, 출판물 등에서 텍스트 획득
2. **데이터 정제**: 노이즈 제거, 중복 제거, 포맷 정규화
3. **데이터 주석(Annotation)**: 언어학적 정보 추가 (품사 태깅, 구문 분석 등)
4. **데이터 표준화**: 일관된 형식으로 변환

### 1.4.2 코퍼스 전처리 및 정규화

전처리는 원시 텍스트를 모델 학습에 적합한 형태로 변환하는 과정입니다:

- **토큰화(Tokenization)**: 텍스트를 토큰 시퀀스로 분할
  
  $Tokenize(T) = (t_1, t_2, ..., t_n)$

- **표준화(Normalization)**: 대소문자 통일, 철자 수정 등
  
  $Normalize(t) = StandardForm(t)$

- **불용어 제거(Stopword Removal)**: 빈번하지만 정보가 적은 단어 제거
  
  $FilterStopwords(T) = \{t \in T | t \notin Stopwords\}$

- **어간 추출(Stemming)** 및 **표제어 추출(Lemmatization)**: 단어의 기본 형태로 변환
  
  $Lemmatize(t) = BaseForm(t)$

### 1.4.3 코퍼스 표현 기법

#### 1.4.3.1 분포 표현(Distributional Representation)

단어의 의미를 그 맥락(주변 단어)으로 표현하는 방식입니다:

- **동시 출현 행렬(Co-occurrence Matrix)**: 단어 간 동시 출현 빈도를 행렬로 표현

  $M_{i,j} = count(w_i, w_j)$

- **PMI(Pointwise Mutual Information)**: 단어 간 연관성 측정
  
  $PMI(w_i, w_j) = \log \frac{P(w_i, w_j)}{P(w_i)P(w_j)}$

#### 1.4.3.2 희소 표현(Sparse Representation)

- **One-hot 인코딩**: 각 단어를 어휘 크기의 벡터로 표현
  
  $OneHot(w_i)[j] = \begin{cases} 
  1 & \text{if } j = i \\
  0 & \text{otherwise}
  \end{cases}$

- **Bag-of-Words(BoW)**: 문서를 단어 빈도 벡터로 표현
  
  $BoW(D)[i] = \text{count of word } w_i \text{ in document } D$

- **TF-IDF(Term Frequency-Inverse Document Frequency)**: 단어의 중요도 측정
  
  $TF\text{-}IDF(w, D, C) = TF(w, D) \times IDF(w, C)$
  
  여기서 $TF(w, D)$는 문서 $D$에서 단어 $w$의 빈도이고, $IDF(w, C)$는 단어 $w$가 등장하는 문서의 역빈도입니다:
  
  $IDF(w, C) = \log \frac{|C|}{|\{D \in C : w \in D\}|}$

## 1.5 코퍼스와 언어 모델링

### 1.5.1 통계적 언어 모델

통계적 언어 모델은 코퍼스에서 관찰된 단어 시퀀스의 확률 분포를 학습합니다:

#### 1.5.1.1 N-gram 모델

단어 시퀀스 확률을 이전 N-1개 단어의 조건부 확률로 근사:

$P(w_1, w_2, ..., w_m) \approx \prod_{i=1}^{m} P(w_i | w_{i-n+1}, ..., w_{i-1})$

조건부 확률은 코퍼스에서의 빈도로 추정:

$P(w_i | w_{i-n+1}, ..., w_{i-1}) = \frac{count(w_{i-n+1}, ..., w_i)}{count(w_{i-n+1}, ..., w_{i-1})}$

#### 1.5.1.2 스무딩(Smoothing) 기법

희소성 문제를 해결하기 위한 방법:

- **라플라스 스무딩(Laplace Smoothing)**:
  
  $P(w_i | w_{i-n+1}, ..., w_{i-1}) = \frac{count(w_{i-n+1}, ..., w_i) + \alpha}{count(w_{i-n+1}, ..., w_{i-1}) + \alpha|V|}$

- **백오프(Backoff) 모델**: 고차 N-gram이 없을 때 저차 N-gram으로 후퇴
  
  $P_{BO}(w_i | w_{i-n+1}, ..., w_{i-1}) = \begin{cases}
  P_{ML}(w_i | w_{i-n+1}, ..., w_{i-1}) & \text{if } count(w_{i-n+1}, ..., w_i) > 0 \\
  \alpha(w_{i-n+1}, ..., w_{i-1}) \cdot P_{BO}(w_i | w_{i-n+2}, ..., w_{i-1}) & \text{otherwise}
  \end{cases}$

### 1.5.2 신경망 기반 언어 모델

#### 1.5.2.1 피드포워드 신경망 언어 모델

$P(w_t | w_{t-n+1}, ..., w_{t-1}) = softmax(W_o \cdot h + b_o)$

여기서 $h$는 은닉층 표현:

$h = f(W_h \cdot x + b_h)$

$x$는 이전 단어들의 연결(concatenation):

$x = [E(w_{t-n+1}); E(w_{t-n+2}); ...; E(w_{t-1})]$

#### 1.5.2.2 순환 신경망 언어 모델

RNN은 가변 길이의 맥락을 처리할 수 있습니다:

$h_t = f(W_{hx} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$

$P(w_t | w_1, ..., w_{t-1}) = softmax(W_o \cdot h_t + b_o)$

#### 1.5.2.3 트랜스포머 언어 모델

자기 주의(self-attention) 메커니즘을 사용:

$\text{Attention}(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}}) \cdot V$

여기서 $Q$, $K$, $V$는 각각 쿼리, 키, 값 행렬이며, $d_k$는 키의 차원입니다.

## 1.6 코퍼스 기반 임베딩 학습

### 1.6.1 단어 임베딩 모델

#### 1.6.1.1 Word2Vec

두 가지 접근법:

- **CBOW(Continuous Bag of Words)**: 주변 단어로 중심 단어 예측
  
  $p(w_c | w_{c-m}, ..., w_{c-1}, w_{c+1}, ..., w_{c+m}) = \frac{\exp(v'^T_c \cdot \bar{v})}{\sum_{j=1}^{|V|} \exp(v'^T_j \cdot \bar{v})}$
  
  여기서 $\bar{v}$는 맥락 단어 벡터의 평균:
  
  $\bar{v} = \frac{1}{2m} \sum_{-m \leq j \leq m, j \neq 0} v_{c+j}$

- **Skip-gram**: 중심 단어로 주변 단어 예측
  
  $p(w_{c+j} | w_c) = \frac{\exp(v'^T_{c+j} \cdot v_c)}{\sum_{k=1}^{|V|} \exp(v'^T_k \cdot v_c)}$

#### 1.6.1.2 GloVe(Global Vectors)

동시 출현 통계를 직접 활용:

$J = \sum_{i,j=1}^{|V|} f(X_{ij}) (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$

여기서 $X_{ij}$는 단어 $i$와 $j$의 동시 출현 빈도이고, $f(X_{ij})$는 가중치 함수입니다.

#### 1.6.1.3 FastText

단어를 문자 n-gram의 집합으로 표현:

$s(w, c) = \sum_{g \in G_w} z_g^T v_c$

여기서 $G_w$는 단어 $w$의 n-gram 집합이고, $z_g$는 n-gram $g$의 벡터 표현입니다.

### 1.6.2 문맥화된 임베딩 모델

#### 1.6.2.1 ELMo(Embeddings from Language Models)

양방향 LSTM을 사용한 문맥화된 표현:

$\text{ELMo}_k^{task} = E(R_k; \Theta^{task}) = \gamma^{task} \sum_{j=0}^L s_j^{task} h_{k,j}^{LM}$

여기서 $h_{k,j}^{LM}$은 단어 $k$에 대한 $j$번째 층의 표현이고, $s_j^{task}$는 가중치입니다.

#### 1.6.2.2 BERT(Bidirectional Encoder Representations from Transformers)

마스킹된 언어 모델 목표를 사용한 양방향 표현:

$L_{MLM} = \mathbb{E}_{(x,y) \sim D} \log p(y|x) = \mathbb{E}_{(x,y) \sim D} \log \prod_{i \in M} p(y_i | \tilde{x})$

여기서 $M$은 마스킹된 토큰 위치 집합이고, $\tilde{x}$는 마스킹된 입력입니다.

## 1.7 코퍼스 품질과 편향 문제

### 1.7.1 코퍼스 품질 평가 지표

#### 1.7.1.1 크기와 다양성

- **어휘 크기(Vocabulary Size)**: $|V|$
- **토큰-타입 비율(Type-Token Ratio)**: $TTR = \frac{|V|}{N}$ (여기서 $N$은 코퍼스의 총 토큰 수)
- **해픽스 레지스터(Hapax Legomena)**: 코퍼스에서 한 번만 나타나는 단어의 비율

#### 1.7.1.2 대표성과 균형

- **장르/도메인 분포**: 각 장르나 도메인의 상대적 비율
- **시간적 범위 및 분포**: 연대별 텍스트 분포

### 1.7.2 코퍼스 편향(Bias) 분석 및 완화

#### 1.7.2.1 편향 유형

- **샘플링 편향(Sampling Bias)**: 데이터 수집 과정에서 발생
- **선택 편향(Selection Bias)**: 코퍼스 구성 과정에서 발생
- **내용 편향(Content Bias)**: 텍스트 내용 자체의 편향

#### 1.7.2.2 편향 측정

- **통계적 편향 측정**: 
  
  $Bias(a, X) = \frac{1}{|X|} \sum_{x \in X} s(a, x) - \frac{1}{|Y|} \sum_{y \in Y} s(a, y)$
  
  여기서 $s(a, x)$는 속성 $a$와 단어 $x$ 사이의 연관성 점수입니다.

- **WEAT(Word Embedding Association Test)**:
  
  $WEAT(X, Y, A, B) = \frac{\text{mean}_{x \in X} s(x, A, B) - \text{mean}_{y \in Y} s(y, A, B)}{\text{std\_dev}_{w \in X \cup Y} s(w, A, B)}$

#### 1.7.2.3 편향 완화 기법

- **균형 재조정(Rebalancing)**: 과소 표현된 집단의 데이터 추가
- **다양성 증진(Diversity Enhancement)**: 다양한 소스에서 데이터 수집
- **후처리(Post-processing)**: 임베딩 공간에서 편향 방향 제거
  
  $\hat{w} = w - \frac{w \cdot b}{||b||^2} b$
  
  여기서 $b$는 식별된 편향 방향입니다.

## 1.8 코퍼스와 대규모 언어 모델(LLM)

### 1.8.1 LLM 학습을 위한 코퍼스 요구사항

#### 1.8.1.1 규모적 요구사항

- **토큰 수**: 수천억~수조 토큰
- **계산 효율성**: $O(L \cdot d^2)$ (여기서 $L$은 시퀀스 길이, $d$는 모델 차원)

#### 1.8.1.2 품질적 요구사항

- **다양성 지수(Diversity Index)**: 
  
  $D = 1 - \sum_{i=1}^{C} p_i^2$
  
  여기서 $p_i$는 카테고리 $i$의 비율입니다.

- **중복 제거율(Deduplication Rate)**: 
  
  $DR = 1 - \frac{|T_{dedup}|}{|T|}$
  
  여기서 $T_{dedup}$은 중복 제거 후 토큰 수입니다.

### 1.8.2 대규모 코퍼스 구축 사례

#### 1.8.2.1 Common Crawl

웹 크롤링 데이터:
- 수조 웹 페이지
- 매월 갱신
- 다양한 언어 및 도메인 포함

전처리 및 필터링:
- 품질 필터링: 휴리스틱 규칙 및 언어 모델 기반 점수
- 중복 제거: MinHash, SimHash 등의 알고리즘 활용
- 자동 언어 식별: fastText, CLD3 등 활용

#### 1.8.2.2 The Pile

다양한 도메인의 텍스트 모음:
- 학술 논문, 코드, 도서, 웹 텍스트 등
- 825GB, 약 8000억 토큰
- 22개 다양한 데이터셋 포함

품질 관리:
- 문서 단위 필터링
- 도메인별 맞춤 전처리
- 중복 제거 및 품질 평가

### 1.8.3 코퍼스 기반 LLM 성능 평가

#### 1.8.3.1 내재적 평가(Intrinsic Evaluation)

- **퍼플렉시티(Perplexity)**: 
  
  $PPL(W) = P(w_1, w_2, ..., w_N)^{-\frac{1}{N}} = \sqrt[N]{\frac{1}{P(w_1, w_2, ..., w_N)}}$

- **비트 당 정보량(Bits Per Character, BPC)**: 
  
  $BPC = -\frac{1}{N} \sum_{i=1}^{N} \log_2 P(c_i | c_1, ..., c_{i-1})$

#### 1.8.3.2 외재적 평가(Extrinsic Evaluation)

- **벤치마크 테스트**: GLUE, SuperGLUE, MMLU 등
- **태스크 기반 평가**: 분류, 요약, 번역, 질의응답 등

## 1.9 코퍼스 기반 지식 추출 및 활용

### 1.9.1 명시적 지식 추출

#### 1.9.1.1 정보 추출(Information Extraction)

- **개체명 인식(NER)**: 
  
  $P(y_i | X, y_{1:i-1}) = \frac{\exp(W_{y_i} \cdot h_i + b_{y_i})}{\sum_{y'} \exp(W_{y'} \cdot h_i + b_{y'})}$

- **관계 추출(Relation Extraction)**: 
  
  $P(r | e_1, e_2, X) = \text{softmax}(W_r \cdot f(e_1, e_2, X) + b_r)$

#### 1.9.1.2 지식 그래프 구축

- **트리플 추출(Triple Extraction)**: 
  
  $(subject, relation, object)$ 형태의 지식 단위 추출

- **온톨로지 매핑(Ontology Mapping)**: 
  
  추출된 지식을 구조화된 온톨로지에 매핑

### 1.9.2 암묵적 지식 저장

LLM은 코퍼스의 지식을 매개변수로 인코딩합니다:

- **인코딩된 지식 측정**: 프로빙 작업과 사실 검색 정확도로 평가
  
  $Accuracy = \frac{\text{# correct predictions}}{\text{# total predictions}}$

- **지식 밀도(Knowledge Density)**: 
  
  $KD = \frac{\text{# facts recalled}}{\text{# parameters}}$

### 1.9.3 코퍼스 큐레이션(Curation)과 증강(Augmentation)

#### 1.9.3.1 지식 강화 코퍼스 구축

- **사실 집중 텍스트 선별**: 사실 밀도 점수 기반 필터링
  
  $FactDensity(D) = \frac{\text{# verifiable facts in }D}{\text{# tokens in }D}$

- **지식 기반 가중치 부여**: 중요 도메인 텍스트에 더 높은 샘플링 확률 할당
  
  $P(D) \propto \exp(\lambda \cdot ImportanceScore(D))$

#### 1.9.3.2 지식 증강 기법

- **합성 데이터 생성**: 기존 지식을 바탕으로 새로운 학습 데이터 생성
  
  $D_{synth} = Generator(z; \theta)$ (여기서 $z$는 잠재 변수, $\theta$는 생성기 매개변수)

- **지식 증류(Knowledge Distillation)**: 대규모 모델의 지식을 소규모 모델로 전달
  
  $L_{KD} = \alpha L_{CE}(y, \hat{y}_S) + (1-\alpha) L_{KL}(\hat{y}_T, \hat{y}_S)$

## 1.10 코퍼스의 윤리적, 법적 고려사항

### 1.10.1 저작권 및 라이선스 이슈

- **페어 유스(Fair Use) 분석**: 4가지 요소 평가
  1. 사용 목적과 성격
  2. 저작물의 성격
  3. 사용된 부분의 양과 질
  4. 시장 영향

- **라이선스 호환성(License Compatibility)**: 
  
  $L_1 \rightarrow L_2$ (라이선스 $L_1$이 $L_2$와 호환됨)

### 1.10.2 개인정보 보호와 익명화

- **개인식별정보(PII) 감지 및 제거**:
  
  $PII\_Detection(T) = \{(start_i, end_i, type_i) | i \in \{1,2,...,n\}\}$

- **익명화 기법**: 대체, 삭제, 일반화 등
  
  $Anonymize(T, PII) = T'$ (여기서 $T'$는 익명화된 텍스트)

### 1.10.3 사회적 영향과 책임

- **다양성 및 대표성 메트릭(Diversity and Representation Metrics)**:
  
  $Representation(g) = \frac{\text{# mentions of group }g}{\text{total # of group mentions}}$

- **유해 콘텐츠 필터링**:
  
  $ToxicityScore(T) = Classifier(T; \theta)$

## 1.11 코퍼스와 다국어 및 도메인 적응

### 1.11.1 다국어 코퍼스 구축 및 활용

#### 1.11.1.1 교차 언어 정렬(Cross-lingual Alignment)

- **병렬 코퍼스 구축**:
  
  $PC = \{(S_i^{L1}, S_i^{L2}) | i \in \{1,2,...,n\}\}$

- **문장 정렬 알고리즘**:
  
  $Align(D^{L1}, D^{L2}) = \{(S_i^{L1}, S_j^{L2}) | sim(S_i^{L1}, S_j^{L2}) > \tau\}$

#### 1.11.1.2 언어 균형(Language Balance) 전략

- **언어별 토큰 비율**:
  
  $R_L = \frac{N_L}{\sum_{l} N_l}$ (여기서 $N_L$은 언어 $L$의 토큰 수)

- **온도 샘플링(Temperature Sampling)**:
  
  $P(L) \propto \left(\frac{N_L}{\sum_{l} N_l}\right)^{1/T}$

### 1.11.2 도메인 적응 및 전이 학습

#### 1.11.2.1 도메인 특화 코퍼스 구축

- **도메인 관련성 점수**:
  
  $Relevance(D, Domain) = cosine(E(D), E(Domain))$

- **도메인 적응 사전 학습**:
  
  $L_{adapt} = L_{MLM} + \lambda L_{domain}$

#### 1.11.2.2 도메인 전이 기법

- **점진적 미세 조정(Gradual Fine-tuning)**:
  
  $\theta_{target} = FT(FT(\theta_{general}, D_{intermediate}), D_{target})$

- **도메인 혼합 비율 최적화**:
  
  $\lambda^* = \arg\min_{\lambda} L_{dev}(\theta_{\lambda})$ (여기서 $\theta_{\lambda}$는 혼합 비율 $\lambda$로 학습된 모델)

## 1.12 코퍼스 기반 연구의 최신 동향과 과제 (계속)

### 1.12.1 동적 및 진화하는 코퍼스 (계속)

#### 1.12.1.2 시간적 편향(Temporal Bias) 관리 (계속)

- **시간 인식 샘플링(Time-aware Sampling)**:
  
  $P(D_t) \propto \exp(-\beta (T_{now} - t))$

### 1.12.2 멀티모달 코퍼스

#### 1.12.2.1 텍스트-이미지 코퍼스

- **이미지-텍스트 쌍 구축**:
  
  $MTCorpus = \{(I_i, T_i) | i \in \{1,2,...,n\}\}$

- **정렬 품질 점수**:
  
  $Alignment(I, T) = sim(E_I(I), E_T(T))$

#### 1.12.2.2 멀티모달 사전 학습

- **대조 학습 목표(Contrastive Learning Objective)**:
  
  $L_{contrastive} = -\log \frac{\exp(sim(E_I(I), E_T(T))/\tau)}{\sum_{j} \exp(sim(E_I(I), E_T(T_j))/\tau)}$

- **결합 표현 학습**:
  
  $L_{joint} = L_{MLM} + L_{ITM} + L_{contrastive}$
  
  여기서 $L_{ITM}$은 이미지-텍스트 매칭 손실입니다.

### 1.12.3 효율적인 코퍼스 활용 기법

#### 1.12.3.1 데이터 증류(Data Distillation)

- **모델 기반 데이터 큐레이션**:
  
  $Score(D) = ModelUtility(D; \theta)$

- **학습 동적 가중치**:
  
  $w_i = f(L_i, g_i)$ (여기서 $L_i$는 손실, $g_i$는 그래디언트 노름)

#### 1.12.3.2 교사 필터링(Curriculum Filtering)

- **난이도 추정**:
  
  $Difficulty(D) = -\log P_{\theta}(D)$

- **진행적 샘플링**:
  
  $P(D_i, t) \propto \exp(-\alpha |Difficulty(D_i) - c(t)|)$
  
  여기서 $c(t)$는 시간 $t$에서의 목표 난이도입니다.

## 1.13 결론: 코퍼스와 언어 모델의 미래

### 1.13.1 코퍼스 구축의 진화 방향

- **자기 개선 코퍼스(Self-improving Corpus)**:
  모델이 생성한 고품질 데이터로 코퍼스 강화
  
  $C_{t+1} = C_t \cup Filter(Generated(M_t))$

- **협력적 코퍼스 큐레이션**:
  인간-AI 협업 기반 코퍼스 구축

### 1.13.2 한계와 도전 과제

- **언어 자원 불균형(Language Resource Imbalance)**:
  저자원 언어에 대한 코퍼스 구축의 어려움

- **지식 최신성 유지**:
  빠르게 변화하는 지식을 코퍼스에 반영하는 문제

- **설명 가능성과 추적성**:
  모델의 출력이 코퍼스의 어떤 부분에서 유래했는지 추적하는 문제

### 1.13.3 미래 연구 방향

- **지속 가능한 코퍼스 생태계 구축**:
  오픈 소스 협력 및 표준화

- **개인화 및 맥락화된 코퍼스**:
  개인 또는 그룹별 맞춤형 코퍼스 구축 방법론

- **지식 통합 프레임워크**:
  명시적 지식 베이스와 코퍼스 기반 암묵적 지식의 통합

## 1.14 코퍼스의 특징 및 중요성

### 1.14.1 코퍼스의 주요 특징

- LLM이 배우는 교과서와 같은 역할
- 위키백과, 뉴스 기사, 논문, 기술 문서 등 다양한 텍스트가 포함됨
- 사용된 코퍼스에 따라 모델이 잘 이해하는 분야가 결정됨

### 1.14.2 코퍼스의 중요성

코퍼스의 품질과 다양성은 LLM의 성능과 지식의 폭에 직접적인 영향을 미칩니다. 편향되거나 제한된 코퍼스는 모델의 한계로 이어질 수 있습니다.

# 2. 임베딩(Embedding)의 이론적 기반과 응용

## 2.1. 임베딩의 정의와 수학적 기초

임베딩은 고차원 또는 비수치적 데이터(텍스트, 이미지, 그래프 등)를 저차원 연속 벡터 공간으로 변환하는 기법입니다. 이는 원본 데이터의 의미적 관계와 구조를 보존하면서 기계 학습 알고리즘이 처리할 수 있는 형태로 변환하는 과정입니다.

### 2.1.1 임베딩의 수학적 정의

임베딩 함수 $f$는 다음과 같이 정의됩니다:

$f: X \rightarrow \mathbb{R}^d$

여기서:
- $X$는 입력 공간(단어, 문서, 그래프 등)
- $\mathbb{R}^d$는 $d$차원 실수 벡터 공간
- $d$는 일반적으로 원본 표현보다 낮은 차원($d \ll |X|$)

임베딩의 핵심 목표는 원본 공간의 관계를 임베딩 공간에서 보존하는 것입니다:

$sim(x_i, x_j) \approx sim(f(x_i), f(x_j))$

여기서 $sim$은 유사도 함수로, 코사인 유사도, 유클리드 거리의 역수 등이 사용됩니다.

### 2.1.2 임베딩 공간의 특성

이상적인 임베딩 공간은 다음 특성을 가집니다:

1. **선형성(Linearity)**: 의미적 관계가 벡터 연산으로 표현 가능
   
   $\vec{v}_{king} - \vec{v}_{man} + \vec{v}_{woman} \approx \vec{v}_{queen}$

2. **등거리성(Isometry)**: 원본 공간의 거리 관계가 임베딩 공간에서 보존
   
   $d_X(x_i, x_j) \approx c \cdot d_{\mathbb{R}^d}(f(x_i), f(x_j))$
   
   여기서 $c$는 스케일 상수입니다.

3. **군집성(Clustering)**: 유사한 항목들이 임베딩 공간에서 가까이 위치

4. **분리성(Separability)**: 서로 다른 범주의 항목들이 임베딩 공간에서 잘 분리됨

## 2.2. 텍스트 임베딩 모델의 수학적 근거

### 2.2.1 희소 표현(Sparse Representations)

#### 2.2.1.1 원-핫 인코딩(One-Hot Encoding)

가장 기본적인 텍스트 표현 방식으로, 각 단어를 어휘 크기의 벡터로 표현합니다:

$OneHot(w_i)[j] = \begin{cases} 
1 & \text{if } j = i \\
0 & \text{otherwise}
\end{cases}$

원-핫 벡터의 문제점:
- 고차원성 ($\mathbb{R}^{|V|}$, 여기서 $|V|$는 어휘 크기)
- 단어 간 의미적 관계 캡처 불가능 (모든 단어 쌍의 코사인 유사도가 0)

#### 2.2.1.2 희소 피처 벡터(Sparse Feature Vectors)

문맥 정보를 포착하기 위해 동시 출현(co-occurrence) 통계를 사용합니다:

**단어-문서 행렬(Word-Document Matrix)** $M_{WD}$:
$M_{WD}[i,j] = \text{TF-IDF}(w_i, d_j)$

**단어-문맥 행렬(Word-Context Matrix)** $M_{WC}$:
$M_{WC}[i,j] = count(w_i, c_j)$

여기서 $c_j$는 문맥 윈도우 내의 단어입니다.

**PMI(Pointwise Mutual Information)** 행렬:
$PMI(w_i, c_j) = \log \frac{P(w_i, c_j)}{P(w_i)P(c_j)}$

여기서:
- $P(w_i, c_j)$: 단어 $w_i$와 문맥 $c_j$의 공동 확률
- $P(w_i)$, $P(c_j)$: 각각의 주변 확률

### 2.2.2 밀집 벡터 임베딩(Dense Vector Embeddings)

#### 2.2.2.1 SVD(특이값 분해) 기반 임베딩

희소 행렬 $M$을 저차원 공간으로 사영하기 위해 SVD를 적용합니다:

$M \approx U\Sigma V^T$

단어 임베딩은 다음과 같이 계산됩니다:
$W = U_k \Sigma_k$

여기서 $U_k$는 $U$의 처음 $k$개 열, $\Sigma_k$는 $\Sigma$의 상위 $k$개 특이값으로 구성된 대각 행렬입니다.

**LSA(Latent Semantic Analysis)** 또는 **LSI(Latent Semantic Indexing)**는 이 방법론을 문서-단어 행렬에 적용한 사례입니다.

#### 2.2.2.2 Word2Vec

**CBOW(Continuous Bag of Words)** 모델:
주변 단어들로 중심 단어 예측

목적 함수:
$J_{\text{CBOW}} = \frac{1}{T} \sum_{t=1}^{T} \log p(w_t | w_{t-c}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+c})$

확률 계산:
$p(w_t | \text{context}) = \frac{\exp(v'^T_{w_t} \cdot \bar{v})}{\sum_{j=1}^{|V|} \exp(v'^T_j \cdot \bar{v})}$

여기서 $\bar{v}$는 문맥 단어 벡터들의 평균입니다:
$\bar{v} = \frac{1}{2c} \sum_{-c \leq j \leq c, j \neq 0} v_{t+j}$

**Skip-gram** 모델:
중심 단어로 주변 단어들 예측

목적 함수:
$J_{\text{skip-gram}} = \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log p(w_{t+j} | w_t)$

확률 계산:
$p(w_{t+j} | w_t) = \frac{\exp(v'^T_{w_{t+j}} \cdot v_{w_t})}{\sum_{k=1}^{|V|} \exp(v'^T_k \cdot v_{w_t})}$

**네거티브 샘플링(Negative Sampling)**:
계산 효율성을 위해 전체 어휘 대신 소수의 네거티브 샘플만 사용

목적 함수:
$J_{\text{neg}} = \log \sigma(v'^T_{w_O} \cdot v_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)}[\log \sigma(-v'^T_{w_i} \cdot v_{w_I})]$

여기서:
- $w_I$: 입력 단어
- $w_O$: 출력(타겟) 단어
- $\sigma$: 시그모이드 함수
- $P_n(w)$: 네거티브 샘플링 분포
- $k$: 네거티브 샘플 수

#### 2.2.2.3 GloVe(Global Vectors)

단어 벡터와 문맥 벡터를 동시에 학습하는 방법입니다:

목적 함수:
$J_{\text{GloVe}} = \sum_{i,j=1}^{|V|} f(X_{ij}) (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$

여기서:
- $X_{ij}$: 단어 $i$와 단어 $j$의 동시 출현 횟수
- $f(X_{ij})$: 가중치 함수 (일반적으로 $\min(1, (X_{ij}/x_{\max})^{\alpha})$)
- $w_i$, $\tilde{w}_j$: 각각 단어와 문맥 임베딩 벡터
- $b_i$, $\tilde{b}_j$: 각각의 편향 항

최종 임베딩:
$\hat{w}_i = \frac{w_i + \tilde{w}_i}{2}$

#### 2.2.2.4 FastText

FastText는 Word2Vec을 확장하여 부분 단어(subword) 정보를 포함합니다:

단어 표현:
$v_{w} = \frac{1}{|G_w|} \sum_{g \in G_w} z_g$

여기서:
- $G_w$: 단어 $w$의 n-gram 집합
- $z_g$: n-gram $g$의 벡터 표현

목적 함수는 Skip-gram과 유사하지만 단어 대신 n-gram 집합을 사용합니다.

## 2.3. 문맥화된 임베딩 모델(Contextual Embedding Models)

### 2.3.1 ELMo(Embeddings from Language Models)

양방향 LSTM에 기반한 문맥화된 단어 표현입니다:

$ELMo_k^{task} = \gamma^{task} \sum_{j=0}^{L} s_j^{task} h_{k,j}^{LM}$

여기서:
- $h_{k,j}^{LM}$: 단어 $k$에 대한 biLM의 $j$번째 층 표현
- $s_j^{task}$: 태스크별 softmax-normalized 가중치
- $\gamma^{task}$: 태스크별 스케일 파라미터

biLM의 목적 함수:
$\mathcal{L} = \sum_{t=1}^{N} (\log p(t_t | t_1, \ldots, t_{t-1}; \Theta_x, \overrightarrow{\Theta}_{LSTM}, \Theta_s) + \log p(t_t | t_{t+1}, \ldots, t_N; \Theta_x, \overleftarrow{\Theta}_{LSTM}, \Theta_s))$

### 2.3.2 BERT(Bidirectional Encoder Representations from Transformers)

트랜스포머 인코더에 기반한 양방향 문맥 표현 모델입니다:

**마스크드 언어 모델링(Masked Language Modeling, MLM)** 목적 함수:
$\mathcal{L}_{MLM} = \mathbb{E}_{(x,y) \sim D} \log p(y|x) = \mathbb{E}_{(x,y) \sim D} \log \prod_{i \in M} p(y_i | \tilde{x})$

여기서:
- $M$: 마스킹된 토큰 위치 집합
- $\tilde{x}$: 마스킹된 입력 시퀀스

**다음 문장 예측(Next Sentence Prediction, NSP)** 목적 함수:
$\mathcal{L}_{NSP} = \mathbb{E}_{(x,y) \sim D} \log p(y|x)$

여기서 $y$는 두 번째 문장이 첫 번째 문장 다음에 오는지 여부입니다.

**자기 주의(Self-attention)** 메커니즘:
$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

여기서 $Q$, $K$, $V$는 각각 쿼리, 키, 값 행렬이며, $d_k$는 키의 차원입니다.

### 2.3.3 GPT(Generative Pre-trained Transformer)

단방향(왼쪽에서 오른쪽) 트랜스포머에 기반한 자기회귀적 언어 모델입니다:

사전 학습 목적 함수:
$\mathcal{L}_1 = \sum_{i} \log P(w_i | w_{i-k}, \ldots, w_{i-1}; \Theta)$

미세 조정 목적 함수:
$\mathcal{L}_2 = \mathcal{L}_1 + \lambda \mathcal{L}_{task}$

여기서 $\mathcal{L}_{task}$는 특정 다운스트림 태스크의 목적 함수입니다.

### 2.3.4 문맥화된 임베딩과 정적 임베딩의 수학적 비교

정적 임베딩(Static Embeddings):
$E: V \rightarrow \mathbb{R}^d$, 여기서 $E(w)$는 단어 $w$에 대해 항상 동일한 벡터입니다.

문맥화된 임베딩(Contextual Embeddings):
$E: V \times S \rightarrow \mathbb{R}^d$, 여기서 $E(w, s)$는 문장 $s$ 내의 단어 $w$에 대한 벡터입니다.

주요 차이점:
- 다의어 처리: 문맥화된 임베딩은 단어의 서로 다른 의미를 구별할 수 있습니다.
- 문맥 의존성: 문맥화된 임베딩은 동일한 단어도 문맥에 따라 다른 벡터로 표현합니다.

## 2.4. 임베딩 평가 방법론

### 2.4.1 내재적 평가(Intrinsic Evaluation)

#### 2.4.1.1 단어 유사도(Word Similarity)

골드 스탠다드 데이터셋(예: WordSim353, SimLex-999)을 사용하여 평가합니다:

스피어만 상관계수:
$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2-1)}$

여기서:
- $d_i$: 순위 차이
- $n$: 평가 쌍 수

#### 2.4.1.2 단어 유추(Word Analogy)

관계 쌍 사이의 벡터 차이를 계산합니다:

정확도:
$Accuracy = \frac{\# \text{ correct predictions}}{\# \text{ total questions}}$

예측 방법:
$\hat{w_d} = \arg\max_{w \in V \setminus \{w_a, w_b, w_c\}} \cos(E(w), E(w_b) - E(w_a) + E(w_c))$

여기서 문제는 "$w_a$는 $w_b$와 같고, $w_c$는 $\hat{w_d}$와 같다"의 형태입니다.

#### 2.4.1.3 개념 범주화(Concept Categorization)

군집 품질 평가:

순수도(Purity):
$Purity = \frac{1}{N} \sum_{i=1}^{k} \max_j |c_i \cap t_j|$

여기서:
- $c_i$: $i$번째 군집
- $t_j$: $j$번째 실제 클래스
- $N$: 총 데이터 포인트 수

### 2.4.2 외재적 평가(Extrinsic Evaluation)

임베딩을 다운스트림 태스크에 적용하여 평가합니다:

- **감성 분석(Sentiment Analysis)**
- **개체명 인식(Named Entity Recognition)**
- **질의응답(Question Answering)**
- **자연어 추론(Natural Language Inference)**

성능 지표:
- 정확도(Accuracy)
- 정밀도(Precision), 재현율(Recall), F1 점수
- 평균 제곱 오차(MSE)

## 2.5. 임베딩의 차원 축소 및 시각화

### 2.5.1 차원 축소 기법

#### 2.5.1.1 PCA(Principal Component Analysis)

데이터의 주요 변동성을 캡처하는 직교 축으로 투영합니다:

공분산 행렬:
$\Sigma = \frac{1}{n} \sum_{i=1}^n (x_i - \mu)(x_i - \mu)^T$

고유값 분해:
$\Sigma = Q \Lambda Q^T$

차원 축소된 표현:
$z_i = Q_k^T (x_i - \mu)$

여기서 $Q_k$는 상위 $k$개 고유벡터로 구성된 행렬입니다.

#### 2.5.1.2 t-SNE(t-distributed Stochastic Neighbor Embedding)

고차원의 유사성을 저차원에서 보존하는 확률적 임베딩 기법입니다:

고차원 공간에서의 조건부 확률:
$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$

대칭화된 유사도:
$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$

저차원 공간에서의 유사도:
$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}$

목적 함수(KL 발산):
$C = KL(P||Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$

#### 2.5.1.3 UMAP(Uniform Manifold Approximation and Projection)

위상 구조를 보존하는 다양체 학습 알고리즘입니다:

지역 퍼지 단체화(local fuzzy simplicial set) 표현:
$\mu_X(x_i, x_j) = \exp\left(-\frac{\max(0, d(x_i, x_j) - \rho_i)}{\sigma_i}\right)$

저차원 공간에서의 유사도:
$\nu_Y(y_i, y_j) = (1 + a \cdot d(y_i, y_j)^{2b})^{-1}$

목적 함수(크로스 엔트로피):
$C = \sum_{i,j} \mu_X(x_i, x_j) \log \frac{\mu_X(x_i, x_j)}{\nu_Y(y_i, y_j)} + (1 - \mu_X(x_i, x_j)) \log \frac{1 - \mu_X(x_i, x_j)}{1 - \nu_Y(y_i, y_j)}$

### 2.5.2 임베딩 공간 분석

#### 2.5.2.1 임베딩 공간의 이방성(Anisotropy)

임베딩 벡터들이 공간 전체에 균등하게 분포하지 않고 특정 영역에 집중되는 현상입니다:

이방성 측정:
$Anisotropy = \frac{1}{n(n-1)} \sum_{i \neq j} \cos(v_i, v_j)$

여기서 값이 0에 가까울수록 등방성(isotropy)이 높습니다.

#### 2.5.2.2 임베딩 공간의 선형 구조

선형 구조 평가를 위한 선형성 척도:
$Linearity = \frac{1}{|S|} \sum_{(a,b,c,d) \in S} \cos\left((v_b - v_a), (v_d - v_c)\right)$

여기서 $S$는 유추 관계 집합입니다.

## 2.6. 임베딩 편향과 공정성

### 2.6.1 임베딩에서의 사회적 편향 측정

#### 2.6.1.1 WEAT(Word Embedding Association Test)

임베딩 공간에서 두 개념 집합과 두 속성 집합 간의 연관성 차이를 측정합니다:

$s(X, Y, A, B) = \frac{\text{mean}_{x \in X} s(x, A, B) - \text{mean}_{y \in Y} s(y, A, B)}{\text{std\_dev}_{w \in X \cup Y} s(w, A, B)}$

여기서 $s(w, A, B)$는 단어 $w$와 속성 집합 $A$, $B$ 사이의 연관성 차이입니다:
$s(w, A, B) = \text{mean}_{a \in A} \cos(w, a) - \text{mean}_{b \in B} \cos(w, b)$

#### 2.6.1.2 편향 방향(Bias Direction) 계산

편향 방향은 특정 개념 쌍 간의 차이 벡터의 평균으로 계산됩니다:
$\vec{b} = \frac{1}{|C|} \sum_{(x, y) \in C} (\vec{x} - \vec{y})$

여기서 $C$는 반대되는 개념 쌍의 집합입니다(예: 남성-여성, 흑인-백인 등).

### 2.6.2 임베딩 편향 완화 기법

#### 2.6.2.1 후처리 접근법(Post-processing Approaches)

**하드 편향 제거(Hard Debiasing)**:
1. 편향 방향 식별
2. 중립화(Neutralization): 편향 방향과 직교하는 성분만 보존
   $\vec{v}_{neutralized} = \vec{v} - (\vec{v} \cdot \vec{b}) \vec{b}$
3. 균등화(Equalization): 쌍을 이루는 단어들에 대해 편향 방향 이외의 모든 방향에서 동일하게 만듦

**소프트 편향 제거(Soft Debiasing)**:
목적 함수에 공정성 제약 조건을 추가:
$\min_{\hat{E}} ||E - \hat{E}||_F^2 \quad \text{s.t.} \quad \hat{E}B = 0$

여기서 $B$는 보호받아야 할 속성을 표현하는 행렬입니다.

#### 2.6.2.2 학습 기반 접근법(Training-based Approaches)

**적대적 학습(Adversarial Learning)**:
주 모델이 보호 속성에 대해 불변인 표현을 학습하도록 함:

$\min_{\theta_E} \max_{\theta_A} \mathcal{L}_{main}(\theta_E) - \lambda \mathcal{L}_{adv}(\theta_A, \theta_E)$

여기서:
- $\theta_E$: 임베딩 모델 파라미터
- $\theta_A$: 적대적 분류기 파라미터
- $\mathcal{L}_{main}$: 주 태스크 손실
- $\mathcal{L}_{adv}$: 적대적 태스크 손실

## 2.7. 고급 임베딩 기법 및 응용

### 2.7.1 그래프 임베딩(Graph Embeddings)

#### 2.7.1.1 Node2Vec

임의 보행(random walk)을 사용하여 그래프 노드를 벡터로 임베딩합니다:

목적 함수:
$\max_f \sum_{u \in V} \log P(N_S(u) | f(u))$

여기서:
- $f$: 노드에서 임베딩으로의 매핑
- $N_S(u)$: 전략 $S$를 통해 생성된 노드 $u$의 네트워크 이웃

확률 계산:
$P(N_S(u) | f(u)) = \prod_{n_i \in N_S(u)} P(n_i | f(u))$

$P(n_i | f(u)) = \frac{\exp(f(n_i) \cdot f(u))}{\sum_{v \in V} \exp(f(v) \cdot f(u))}$

#### 2.7.1.2 TransE

지식 그래프에서 엔티티와 관계를 임베딩합니다:

에너지 함수:
$d(h + r, t)$

여기서:
- $h$: 헤드 엔티티 임베딩
- $r$: 관계 임베딩
- $t$: 테일 엔티티 임베딩
- $d$: 거리 함수(일반적으로 L1 또는 L2 노름)

목적 함수:
$\mathcal{L} = \sum_{(h,r,t) \in S} \sum_{(h',r,t') \in S'} [\gamma + d(h + r, t) - d(h' + r, t')]_+$

여기서 $S'$은 손상된(corrupted) 트리플 집합입니다.

### 2.7.2 다중 모달 임베딩(Multimodal Embeddings)

#### 2.7.2.1 이미지-텍스트 공동 임베딩

대조적 손실 함수를 사용하여 이미지와 텍스트를 공통 임베딩 공간에 매핑합니다:

$\mathcal{L}_{contrastive} = \sum_{i} \sum_{j \neq i} [\alpha - s(i_i, t_i) + s(i_i, t_j)]_+ + \sum_{i} \sum_{j \neq i} [\alpha - s(i_i, t_i) + s(i_j, t_i)]_+$

여기서:
- $i_i$: $i$번째 이미지의 임베딩
- $t_i$: $i$번째 텍스트의 임베딩
- $s$: 유사도 함수(일반적으로 코사인 유사도)
- $\alpha$: 여백(margin) 파라미터

#### 2.7.2.2 CLIP(Contrastive Language-Image Pre-training)

대규모 이미지-텍스트 쌍으로 사전 학습된 대조적 모델입니다:

목적 함수:
$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(sim(i_i, t_i) / \tau)}{\sum_{j=1}^{N} \exp(sim(i_i, t_j) / \tau)}$

여기서:
- $i_i$: 이미지 인코더에 의해 인코딩된 이미지
- $t_i$: 텍스트 인코더에 의해 인코딩된 텍스트
- $sim$: 유사도 함수(일반적으로 코사인 유사도)
- $\tau$: 온도 파라미터

## 2.8. 쿼리 임베딩과 코퍼스 임베딩

쿼리 임베딩과
코퍼스 임베딩은 RAG(Retrieval-Augmented Generation) 시스템의 핵심 요소입니다.

### 2.8.1 코퍼스 임베딩

- 미리 준비된 모든 문서(코퍼스)의 문단들을 벡터로 변환하여 저장
- 기술 매뉴얼, 논문, 사내 문서 등 다양한 자료가 대상이 될 수 있음
- 문서를 효율적으로 검색하기 위한 기초 작업

### 2.8.2 쿼리 임베딩

- 사용자의 질문(쿼리)을 동일한 벡터 공간으로 변환
- 질문의 의미를 수치화하여 코퍼스 임베딩과 비교 가능하게 함

### 2.8.3 매칭 프로세스

- 쿼리 벡터와 코퍼스 벡터 간의 유사도를 계산
- 코사인 유사도($cos(\theta) = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}||\vec{b}|}$) 또는 유클리드 거리 등의 방법으로 계산
- 가장 유사도가 높은 문서 또는 문단을 찾아 활용

## 2.9. LLM의 임베딩 관련 한계점

LLM은 훈련된 데이터(학습 코퍼스) 외의 정보가 필요한 전문적이거나 특정 분야에 한정된 질문에 취약점을 보입니다.

### 2.9.1 잘 처리하는 케이스: 일반적인 질문

- 예시: "지구는 왜 자전을 하나요?"
- 이유: 과학 교과서나 인터넷에 널리 퍼진 지식으로, 학습 데이터에 충분히 포함됨

### 2.9.2 처리가 어려운 케이스: 도메인 특화 쿼리

- 예시: "ROS2에서 rmw_fastrtps_cpp와 rmw_cyclonedds_cpp의 성능 차이는 어떤 상황에서 뚜렷하게 나타나나요?"
- 문제점: 매우 전문적이고 최신 정보일 가능성이 높으며, 테스트 환경에 따라 결과가 달라질 수 있음

### 2.9.3 한계의 원인

- LLM은 "이미 존재하는 데이터"를 기반으로 학습함
- 다음과 같은 정보는 LLM이 정확히 처리하기 어려움:
  - 극히 드물거나 폐쇄적인 정보
  - 빠르게 변화하는 최신 정보
  - 실제 경험, 실험, 특수 장비 없이 알 수 없는 정보

# 3. RAG의 주요 컴포넌트 최적화

## 3.1 문서 처리 및 인덱싱

### 3.1.1 청킹(Chunking) 전략

문서를 최적의 단위로 분할하는 수학적 모델:

**고정 크기 청킹**:
$C = \{z_1, z_2, ..., z_n\}$ 여기서 $|z_i| \approx k$ (토큰 수)

**의미 기반 청킹**:
$C = \{z_1, z_2, ..., z_n\}$ 여기서 각 $z_i$는 코히런스 함수 $\phi(z_i) > \theta$를 만족

**계층적 청킹**:
$C = \{z^1, z^2, ..., z^L\}$ 여기서 $z^l = \{z^l_1, z^l_2, ..., z^l_{n_l}\}$는 $l$ 레벨의 청크 집합

### 3.1.2 벡터 양자화(Vector Quantization)

$Q: \mathbb{R}^d \rightarrow \mathcal{C}$ 여기서 $\mathcal{C} = \{c_1, c_2, ..., c_K\}$는 코드북

양자화 손실:
$\mathcal{L}_{\text{quant}} = \|v - Q(v)\|_2^2$

곱 양자화(Product Quantization):
$v \approx [q_1(v_1), q_2(v_2), ..., q_M(v_M)]$

여기서 $v = [v_1, v_2, ..., v_M]$은 벡터의 하위 공간 분해입니다.

## 3.2 검색 최적화

### 3.2.1 퀴리 확장(Query Expansion)

원본 쿼리 $x$를 확장하여 검색 성능을 향상시킵니다:

$x' = f_{\text{expand}}(x)$

예를 들어, 유의어 확장:
$x' = x \cup \{s : s \in \text{Synonyms}(t), t \in x\}$

### 3.2.2 쿼리 분해(Query Decomposition)

복잡한 쿼리를 하위 쿼리로 분해합니다:

$\{x_1, x_2, ..., x_m\} = f_{\text{decompose}}(x)$

최종 검색 결과는 하위 쿼리 결과의 집합이 됩니다:
$Z = \cup_{i=1}^{m} \text{Retriever}(x_i, C)$

### 3.2.3 순위 재조정(Re-ranking)

1차 검색 결과에 대한 정교한 순위 재조정:

$\text{score}_{\text{rerank}}(x, z) = f_{\text{rerank}}(x, z)$

교차 인코더 기반 재순위 모델:
$\text{score}_{\text{cross-encoder}}(x, z) = f_{\text{CE}}([x; z])$

여기서 $[x; z]$는 쿼리와 문서의 연결입니다.

## 3.3 생성 최적화

### 3.3.1 검색 증강 디코딩(Retrieval-Augmented Decoding)

생성 중에 동적으로 검색을 수행합니다:

$P(y_i|y_{<i}, x) = \sum_{z \in \text{Retriever}(y_{<i}, x)} P(z|y_{<i}, x) P(y_i|y_{<i}, x, z)$

여기서 검색은 부분적으로 생성된 출력 $y_{<i}$를 고려합니다.

### 3.3.2 검색-생성 반복(Retrieval-Generation Iteration)

생성 결과를 기반으로 검색을 반복하는 과정:

초기화: $y^0 = \emptyset$
반복 $t = 1, 2, ..., T$:
  - $z^t = \text{Retriever}(x, y^{t-1}, C)$
  - $y^t = \text{Generator}(x, z^t, y^{t-1})$
최종 출력: $y^T$

## 3.4 RAG 학습 및 미세 조정

### 3.4.1 엔드-투-엔드 학습(End-to-End Training)

RAG 시스템의 전체 파라미터 집합은 $\theta = \{\theta_R, \theta_G\}$이며, 여기서:
- $\theta_R$: 검색기 파라미터
- $\theta_G$: 생성기 파라미터

log-likelihood 손실:
$\mathcal{L}(\theta) = -\log P_{\theta}(y|x) = -\log \sum_{z \in Z} P_{\theta_R}(z|x) P_{\theta_G}(y|x,z)$

그래디언트는 다음과 같이 계산됩니다:
$\nabla_{\theta} \mathcal{L}(\theta) = -\nabla_{\theta} \log \sum_{z \in Z} P_{\theta_R}(z|x) P_{\theta_G}(y|x,z)$

### 3.4.2 단계별 학습(Step-wise Training)

#### 3.4.2.1 검색기 학습

대조 손실을 사용한 검색기 최적화:
$\mathcal{L}_R(\theta_R) = -\log \frac{\exp(\text{sim}(E_Q(x), E_D(z^+))/\tau)}{\sum_{z \in \{z^+\} \cup Z^-} \exp(\text{sim}(E_Q(x), E_D(z))/\tau)}$

여기서 $z^+$는 관련 문서, $Z^-$는 관련 없는 문서 집합입니다.

#### 3.4.2.2 생성기 학습

검색기를 고정한 상태에서 생성기 학습:
$\mathcal{L}_G(\theta_G) = -\log P_{\theta_G}(y|x, z) = -\sum_{i=1}^{|y|} \log P_{\theta_G}(y_i|y_{<i}, x, z)$

여기서 $z$는 고정된 검색기에서 얻은 검색 결과입니다.

### 3.4.3 강화 학습(Reinforcement Learning) 접근법

#### 3.4.3.1 정책 경사법(Policy Gradient)

시스템을 정책으로 모델링하고 보상에 따라 최적화합니다:

$\mathcal{L}_{RL}(\theta) = -\mathbb{E}_{z \sim P_{\theta_R}(z|x), y \sim P_{\theta_G}(y|x,z)}[R(y, y^*)]$

여기서 $R(y, y^*)$는 생성된 응답 $y$와 참조 응답 $y^*$ 사이의 보상 함수입니다.

REINFORCE 알고리즘을 적용한 그래디언트:
$\nabla_{\theta} \mathcal{L}_{RL}(\theta) = -\mathbb{E}_{z, y}[\nabla_{\theta} \log P_{\theta}(z, y|x) \cdot R(y, y^*)]$

#### 3.4.3.2 인간 피드백 기반 강화 학습(RLHF)

인간 평가자의 선호도를 모델링한 보상 함수 사용:

$R_{\phi}(x, z, y) = f_{\phi}(x, z, y)$

이 보상 모델은 인간 선호 데이터 $\mathcal{D} = \{(x_i, z_i, y_i^w, y_i^l)\}$에서 학습됩니다:
$\mathcal{L}_{\phi} = -\mathbb{E}_{(x, z, y^w, y^l) \sim \mathcal{D}}[\log \sigma(R_{\phi}(x, z, y^w) - R_{\phi}(x, z, y^l))]$

정책 최적화는 PPO(Proximal Policy Optimization)와 같은 알고리즘을 통해 수행됩니다.

## 3.5 RAG 평가 방법론

### 3.5.1 검색 품질 평가

#### 3.5.1.1 표준 IR 평가 지표

검색된 문서의 관련성을 평가하는 지표들:

**정밀도@k**:
$\text{Precision@k} = \frac{|\{z_1, z_2, ..., z_k\} \cap \text{Relevant}|}{k}$

**재현율@k**:
$\text{Recall@k} = \frac{|\{z_1, z_2, ..., z_k\} \cap \text{Relevant}|}{|\text{Relevant}|}$

**평균 정밀도(AP)**:
$\text{AP} = \sum_{k=1}^{n} P(k) \cdot \text{rel}(k)$

여기서 $P(k)$는 $k$까지의 정밀도, $\text{rel}(k)$는 $k$번째 문서의 관련성 지표입니다.

**정규화된 할인 누적 이득(NDCG)**:
$\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$

여기서:
$\text{DCG@k} = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i+1)}$

#### 3.5.1.2 RAG 특화 검색 평가

**응답 기여도(Answer Relevance)**:
$\text{AR}(z, y) = \text{sim}(E(z), E(y))$

**알람 메커니즘(Alarm Mechanism)**:
$\text{Alarm}(x, C) = \begin{cases}
1 & \text{if } \max_{z \in C} \text{sim}(E_Q(x), E_D(z)) < \tau \\
0 & \text{otherwise}
\end{cases}$

여기서 $\tau$는 임계값 파라미터입니다.

### 3.5.2 생성 품질 평가

#### 3.5.2.1 참조 기반 지표

참조 응답과 생성된 응답을 비교하는 지표들:

**BLEU**:
$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$

여기서:
- $\text{BP}$: 간결성 페널티(brevity penalty)
- $p_n$: n-gram 정밀도
- $w_n$: 가중치

**ROUGE-L**:
$\text{ROUGE-L} = \frac{\text{LCS}(y, y^*)}{\text{len}(y^*)}$

여기서 $\text{LCS}$는 최장 공통 부분 수열의 길이입니다.

#### 3.5.2.2 참조 없는 지표

참조 응답 없이 생성된 응답의 품질을 평가하는 지표들:

**퍼플렉서티(Perplexity)**:
$\text{PPL}(y) = \exp\left(-\frac{1}{|y|}\sum_{i=1}^{|y|} \log P(y_i|y_{<i})\right)$

**지식 충실도(Knowledge Faithfulness)**:
$\text{KF}(y, z) = \frac{|\{f_i : f_i \in \text{Facts}(y) \cap \text{Facts}(z)\}|}{|\text{Facts}(y)|}$

여기서 $\text{Facts}(\cdot)$는 텍스트에서 추출된 사실 집합입니다.

**질의-응답 기반 평가**:
$\text{QA-Eval}(y, z) = \frac{1}{|Q|}\sum_{q \in Q} \text{Acc}(\text{QA}(q, y), \text{QA}(q, z))$

여기서 $Q$는 검색된 문서에서 생성된 질문 집합입니다.

### 3.5.3 엔드-투-엔드 시스템 평가

#### 3.5.3.1 통합 평가 프레임워크

**RAGAS 점수**:
$\text{RAGAS} = \alpha \cdot \text{Faithfulness} + \beta \cdot \text{Context Relevance} + \gamma \cdot \text{Answer Relevance}$

여기서 $\alpha$, $\beta$, $\gamma$는 각 요소의 가중치입니다.

#### 3.5.3.2 인간 평가 프로토콜

**RAG-specific 평가 기준**:
1. 사실적 정확성(Factual Accuracy): 1-5 점수
2. 인용 정확성(Citation Accuracy): 1-5 점수
3. 응답 완전성(Response Completeness): 1-5 점수

**쌍별 비교(Pairwise Comparison)**:
$\text{Win Rate} = \frac{\# \text{Wins}}{(\# \text{Wins} + \# \text{Losses})}$

## 3.6 RAG의 고급 아키텍처와 변형

### 3.6.1 다단계 검색(Multi-stage Retrieval)

#### 3.6.1.1 콜라이더(Colander) 아키텍처

첫 번째 단계에서 높은 재현율 검색, 두 번째 단계에서 정밀한 재순위:

$Z_1 = \text{Retriever}_1(x, C)$ (재현율 최적화)
$Z_2 = \text{Ranker}(x, Z_1)$ (정밀도 최적화)

목적 함수:
$\mathcal{L} = \mathcal{L}_1 + \lambda \mathcal{L}_2$

여기서 $\mathcal{L}_1$과 $\mathcal{L}_2$는 각 단계의 손실 함수입니다.

#### 3.6.1.2 계층적 검색(Hierarchical Retrieval)

문서 컬렉션이 계층적으로 구성된 경우:

$Z_1 = \text{Retriever}_1(x, C_1)$ (상위 레벨 검색)
$Z_2 = \text{Retriever}_2(x, \cup_{z \in Z_1} \text{Children}(z))$ (하위 레벨 검색)

### 3.6.2 다중 쿼리 확장(Multi-Query Expansion)

#### 3.6.2.1 자율 쿼리 생성(Autonomous Query Generation)

LLM을 사용하여 원본 쿼리에서 다양한 검색 쿼리 생성:

$\{x_1, x_2, ..., x_n\} = \text{LLM}_{\text{query-gen}}(x)$

각 쿼리에 대한 검색 결과를 집계:
$Z = \cup_{i=1}^{n} \text{Retriever}(x_i, C)$

#### 3.6.2.2 하이퍼쿼리(HyperQuery)

쿼리 표현의 불확실성을 고려한 다중 표현 생성:

$Q = \{q_1, q_2, ..., q_m\} = f_{\text{hyperquery}}(x)$

여기서 각 $q_i$는 원본 쿼리 $x$의 다른 표현입니다.

검색 점수는 다음과 같이 집계됩니다:
$\text{score}(x, z) = \max_{q \in Q} \text{sim}(q, E_D(z))$

### 3.6.3 지식 증류 기반 RAG(Knowledge Distillation-based RAG)

#### 3.6.3.1 FiD(Fusion-in-Decoder)

인코더-디코더 아키텍처에서 검색된 여러 문서를 독립적으로 인코딩:

$h_i = \text{Encoder}([x; z_i])$ for $i = 1, 2, ..., k$

디코더는 모든 인코딩 결과를 연결하여 사용:
$y = \text{Decoder}([h_1; h_2; ...; h_k])$

#### 3.6.3.2 REALM(Retrieval-Augmented Language Model Pre-training)

사전 학습 중에 잠재 변수(검색 문서)를 통합:

$\mathcal{L}_{\text{REALM}} = -\log \sum_{z \in Z} P(z|x) P(y|x, z)$

최대우도추정(MLE)을 위한 EM 알고리즘:
- E-step: $P(z|x, y) \propto P(z|x) P(y|x, z)$
- M-step: $\theta^* = \arg\max_{\theta} \mathbb{E}_{z \sim P(z|x, y)}[\log P_{\theta}(z|x) + \log P_{\theta}(y|x, z)]$

## 3.7 RAG의 실제 응용과 시스템 설계

### 3.7.1 산업 응용 사례

#### 3.7.1.1 지식 기반 질의응답 시스템

RAG 기반 질의응답 시스템의 형식적 모델:

$\text{QA}(q) = \text{Generator}(q, \text{Retriever}(q, KB))$

여기서 $KB$는 지식 베이스입니다.

#### 3.7.1.2 대화 시스템 및 챗봇

대화 이력을 고려한 RAG 모델:

$\text{Response}(q_t, h_t) = \text{Generator}(q_t, h_t, \text{Retriever}(q_t, h_t, KB))$

여기서:
- $q_t$: 현재 질문
- $h_t$: 대화 이력
- $KB$: 지식 베이스

### 3.7.2 확장성 및 효율성 고려사항

#### 3.7.2.1 인덱스 구축 및 유지보수

**점진적 인덱싱(Incremental Indexing)**:
$I_{t+1} = f_{\text{update}}(I_t, \Delta C_t)$

여기서:
- $I_t$: 시간 $t$에서의 인덱스
- $\Delta C_t$: 시간 $t$에서 추가된 새로운 문서

**분산 인덱싱(Distributed Indexing)**:
$I = \cup_{i=1}^{m} I_i$ 여기서 $I_i$는 샤드 $i$의 인덱스입니다.

#### 3.7.2.2 지연 시간 최적화

**캐싱 전략(Caching Strategies)**:
$\text{Cache}(q) = \begin{cases}
\text{CachedResult}(q) & \text{if } \exists q' \in \text{Keys(Cache)}: \text{sim}(q, q') > \tau \\
\text{Retriever}(q, C) & \text{otherwise}
\end{cases}$

여기서 $\tau$는 캐시 히트를 결정하는 유사도 임계값입니다.

**병렬 검색(Parallel Retrieval)**:
$Z = \cup_{i=1}^{m} \text{Retriever}_i(q, C_i) \text{ (in parallel)}$

여기서 각 $\text{Retriever}_i$는 동시에 실행됩니다.

### 3.7.3 멀티모달 RAG

#### 3.7.3.1 이미지-텍스트 RAG

이미지와 텍스트를 모두 포함하는 검색 시스템:

$Z = \text{Retriever}(q, C_{\text{text}} \cup C_{\text{image}})$

검색 점수는 모달리티별 인코더를 사용하여 계산됩니다:
$\text{score}(q, z) = \begin{cases}
\text{sim}(E_Q(q), E_T(z)) & \text{if } z \in C_{\text{text}} \\
\text{sim}(E_Q(q), E_I(z)) & \text{if } z \in C_{\text{image}}
\end{cases}$

#### 3.7.3.2 구조화된 지식(Structured Knowledge) 통합

테이블, 그래프 등 구조화된 지식 검색:

$Z = \text{Retriever}(q, C_{\text{text}} \cup C_{\text{struct}})$

구조화된 데이터의 선형화(linearization):
$z_{\text{linear}} = f_{\text{linearize}}(z_{\text{struct}})$

## 3.8 RAG의 도전 과제와 연구 방향

### 3.8.1 환각(Hallucination) 완화

#### 3.8.1.1 사실 검증(Fact Verification) 모듈

생성 후 검증을 통한 환각 감소:

$V(y, Z) = \prod_{f \in \text{Facts}(y)} \max_{z \in Z} V_f(f, z)$

여기서 $V_f(f, z)$는 사실 $f$가 문서 $z$에 의해 지지되는 확률입니다.

#### 3.8.1.2 불확실성 인식 생성(Uncertainty-Aware Generation)

모델의 불확실성을 명시적으로 표현:

$P(y|x, Z, \mathcal{U}) = \begin{cases}
P(y|x, Z) & \text{if } \mathcal{U}(x, Z) < \tau \\
\text{"I'm uncertain about: "} + x & \text{otherwise}
\end{cases}$

여기서 $\mathcal{U}(x, Z)$는 검색 결과에 대한 불확실성 측정입니다.

### 3.8.2 다국어 및 도메인 특화 RAG

#### 3.8.2.1 교차 언어(Cross-lingual) RAG

다국어 환경에서의 RAG 모델:

$\text{score}(q^{L_1}, z^{L_2}) = \text{sim}(E_{L_1}(q^{L_1}), E_{L_2}(z^{L_2}))$

여기서:
- $q^{L_1}$: 언어 $L_1$의 쿼리
- $z^{L_2}$: 언어 $L_2$의 문서
- $E_{L_1}, E_{L_2}$: 각 언어별 인코더

#### 3.8.2.2 도메인 적응(Domain Adaptation)

특정 도메인에 RAG 모델을 적응시키는 방법:

$\mathcal{L}_{\text{adapt}} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{domain}}$

$\mathcal{L}_{\text{domain}} = \text{DJS}(P(Z|X_{\text{source}}), P(Z|X_{\text{target}}))$

여기서 $\text{DJS}$는 Jensen-Shannon 발산입니다.

### 3.8.3 윤리적, 책임 있는 RAG

#### 3.8.3.1 인용 및 출처 검증(Citation and Source Verification)

생성된 응답에 인용 추가:

$y_{\text{cited}} = f_{\text{cite}}(y, Z) = y + \text{GenerateCitations}(y, Z)$

인용 정확도 측정:
$\text{CitationAccuracy}(y_{\text{cited}}, Z) = \frac{|\{\text{c} \in \text{Citations}(y_{\text{cited}}) : \text{IsSupported}(\text{c}, Z)\}|}{|\text{Citations}(y_{\text{cited}})|}$

#### 3.8.3.2 편향 감지 및 완화(Bias Detection and Mitigation)

검색 및 생성에서의 편향 측정:

$\text{Bias}(q, r) = \text{Disparity}(Z_{q,r=1}, Z_{q,r=0})$

여기서:
- $r$: 보호 속성(예: 성별, 인종)
- $Z_{q,r}$: 속성 $r$이 있는 쿼리 $q$에 대한 검색 결과

편향 완화를 위한 규제화:
$\mathcal{L}_{\text{debiased}} = \mathcal{L}_{\text{original}} + \lambda \mathcal{L}_{\text{fairness}}$

## 3.9 통합 RAG 프레임워크 및 이론적 전망

### 3.9.1 통합 수학적 프레임워크

#### 3.9.1.1 베이지안 RAG 모델

베이지안 관점에서의 RAG:

$P(y|x) = \int_Z P(y|x,Z) P(Z|x) dZ$

파라미터에 대한 불확실성 포함:
$P(y|x, \mathcal{D}) = \int_{\theta} \int_Z P(y|x,Z,\theta) P(Z|x,\theta) P(\theta|\mathcal{D}) d\theta dZ$

여기서 $\mathcal{D}$는 학습 데이터입니다.

#### 3.9.1.2 정보 이론적 해석

정보 이론 관점에서의 RAG 최적화:

$I(Y;Z|X) = H(Y|X) - H(Y|X,Z)$

여기서:
- $I(Y;Z|X)$: $X$가 주어졌을 때 $Y$와 $Z$ 사이의 조건부 상호 정보
- $H(Y|X)$: $X$가 주어졌을 때 $Y$의 조건부 엔트로피
- $H(Y|X,Z)$: $X$와 $Z$가 주어졌을 때 $Y$의 조건부 엔트로피

### 3.9.2 미래 연구 방향

#### 3.9.2.1 자율 RAG(Autonomous RAG)

스스로 검색 전략을 결정하는 RAG 시스템:

$\text{Strategy}(x) = f_{\text{meta}}(x, \text{History})$

$\text{Response}(x) = \text{ExecuteStrategy}(\text{Strategy}(x), x)$

#### 3.9.2.2 인과적 RAG(Causal RAG)

인과 관계를 고려한 RAG 모델:

$P(y|do(x)) = \sum_Z P(y|x,Z) P(Z|do(x))$

여기서 $do(x)$는 외부 개입을 통해 $x$를 설정하는 작업입니다.

### 3.9.3 실용적 구현 지침

#### 3.9.3.1 RAG 파이프라인 설계 원칙

효과적인 RAG 파이프라인을 위한 수학적 설계 원칙:

1. **검색-생성 균형(Retrieval-Generation Balance)**:
   $\alpha = \frac{\text{Influence}(\text{Retriever})}{\text{Influence}(\text{Retriever}) + \text{Influence}(\text{Generator})}$

   최적의 $\alpha$는 태스크와 데이터에 따라 다릅니다.

2. **컴포넌트 간 정보 흐름(Information Flow)**:
   $I(X \rightarrow Z \rightarrow Y) = I(Z;Y) - I(Z;Y|X)$

#### 3.9.3.2 벤치마킹 및 평가 체계

표준화된 RAG 평가 프레임워크:

$\text{RAGScore} = w_1 \cdot \text{RetrievalQuality} + w_2 \cdot \text{GenerationQuality} + w_3 \cdot \text{Faithfulness}$

이상적인 가중치 결정:
$\{w_1, w_2, w_3\} = \arg\max_{w_1, w_2, w_3} \text{Corr}(\text{RAGScore}, \text{HumanEvaluation})$

## 3.10 결론 및 전망

### 3.10.1 RAG의 이론적 의의

RAG는 기존의 검색 시스템과 생성 모델을 통합하는 새로운 패러다임을 제시합니다. 수학적으로는 다음과 같이 요약할 수 있습니다:

$P(y|x) = \int_Z P(y|x,Z) P(Z|x) dZ \approx \sum_{z \in \text{top-k}(P(z|x))} P(z|x) P(y|x,z)$

이 공식은 RAG가 생성과 검색 사이의 확률적 브릿지 역할을 함을 나타냅니다.

### 3.10.2 응용 및 영향

RAG는 다음과 같은 영역에서 중요한 영향을 미칠 것으로 예상됩니다:

1. **개방형 도메인 QA**: 
   $\lim_{|C| \rightarrow \infty} P(y|x,C) \rightarrow P(y|x,\text{World Knowledge})$

2. **지식 집약적 작업**:
   $\text{Performance} \propto I(Y;Z|X)$ (검색된 정보와 출력 사이의 상호 정보)

3. **지속적 학습(Continual Learning)**:
   $\text{Knowledge}(t+1) = \text{Knowledge}(t) \cup \text{NewKnowledge}(t)$

### 3.10.3 미래 도전 과제

RAG의 미래 발전을 위한 핵심 연구 문제들은 다음과 같습니다:

1. **최적의 검색-생성 통합**:
   $P(y|x) = \int_Z P(y|x,Z) P(Z|x) dZ$ vs. $P(y|x) = P(y|x,\text{Retrieve}(x))$

2. **효율적인 지식 표현과 검색**:
   $\text{Efficiency} = \frac{\text{RetrievalPerformance}}{\text{ComputationalCost} \times \text{MemoryCost}}$

3. **인과적 및 반사적 추론**:
   $P(y|x,Z) = \int_{\text{reasoning}} P(y|x,Z,r) P(r|x,Z) dr$

RAG는 단순한 기술적 접근법을 넘어, 인공지능 시스템이 어떻게 지식을 획득하고, 활용하며, 추론하는지에 대한 근본적인 질문을 다루는 연구 영역으로 발전하고 있습니다.  
  
  

# 4. Retriever의 이론적 기반과 최적화

## 4.1 Retriever의 정의와 기본 개념

Retriever는 RAG(Retrieval-Augmented Generation) 시스템에서 사용자 쿼리와 관련된 문서를 코퍼스에서 효율적으로 검색하는 핵심 컴포넌트입니다. 검색의 정확도와 효율성은 전체 시스템의 성능에 직접적인 영향을 미칩니다.

### 4.1.1 Retriever의 수학적 정의

형식적으로, Retriever $R$은 다음과 같이 정의됩니다:

$R: Q \times C \rightarrow Z^k$

여기서:
- $Q$는 가능한 모든 쿼리의 집합
- $C$는 문서 코퍼스
- $Z^k$는 상위 $k$개의 관련 문서 집합
- $R(q, C)$는 쿼리 $q$에 대해 코퍼스 $C$에서 검색된 $k$개의 문서 집합

검색 과정의 핵심은 쿼리-문서 쌍에 대한 관련성 점수 함수 $S$입니다:

$S: Q \times D \rightarrow \mathbb{R}$

최종적으로, 검색 결과는 다음과 같이 정의됩니다:

$R(q, C) = \arg\text{top-}k_{d \in C} S(q, d)$

여기서 $\arg\text{top-}k$ 연산자는 점수 함수 $S$에 따라 상위 $k$개의 문서를 반환합니다.

### 4.1.2 Retriever의 유형 분류

#### 4.1.2.1 표현 방식에 따른 분류

**희소 검색기(Sparse Retrievers)**:
텍스트를 고차원 희소 벡터로 표현합니다. 이러한 방식은 용어의 출현 여부나 빈도를 기반으로 합니다.

수학적 표현:
$v_{\text{sparse}}(d) \in \mathbb{R}^{|V|}$ 여기서 $|V|$는 어휘 크기입니다.

**밀집 검색기(Dense Retrievers)**:
텍스트를 저차원 밀집 벡터(임베딩)로 표현합니다. 이러한 방식은 의미적 유사성을 포착할 수 있습니다.

수학적 표현:
$v_{\text{dense}}(d) \in \mathbb{R}^{d}$ 여기서 $d$는 임베딩 차원이며, 일반적으로 $d \ll |V|$입니다.

**하이브리드 검색기(Hybrid Retrievers)**:
희소 표현과 밀집 표현을 모두 활용합니다.

수학적 표현:
$S_{\text{hybrid}}(q, d) = \lambda S_{\text{sparse}}(q, d) + (1-\lambda) S_{\text{dense}}(q, d)$

여기서 $\lambda \in [0, 1]$는 두 방식의 가중치를 조절하는 파라미터입니다.

#### 4.1.2.2 검색 패러다임에 따른 분류

**용어 기반 검색(Term-based Retrieval)**:
쿼리와 문서 간의 용어 일치를 기반으로 합니다.

**의미론적 검색(Semantic Retrieval)**:
단어의 표면적 일치가 아닌 의미적 유사성을 고려합니다.

**신경망 검색(Neural Retrieval)**:
딥러닝 기반 모델을 사용하여 쿼리와 문서 간의 복잡한 관계를 모델링합니다.

## 4.2 희소 검색 모델(Sparse Retrieval Models)

### 4.2.1 불리언 모델(Boolean Model)

가장 기본적인 정보 검색 모델로, 불리언 논리를 사용하여 쿼리와 문서의 관련성을 결정합니다.

$R_{\text{boolean}}(q, d) = \begin{cases} 
1 & \text{if } d \text{ satisfies the boolean query } q \\
0 & \text{otherwise}
\end{cases}$

### 4.2.2 벡터 공간 모델(Vector Space Model, VSM)

쿼리와 문서를 벡터로 표현하고, 코사인 유사도를 사용하여 관련성을 계산합니다.

$\text{sim}(q, d) = \frac{v(q) \cdot v(d)}{||v(q)|| \cdot ||v(d)||}$

여기서 $v(q)$와 $v(d)$는 각각 쿼리와 문서의 벡터 표현입니다.

### 4.2.3 BM25(Best Matching 25)

BM25는 TF-IDF의 확장으로, 문서 길이를 고려합니다:

$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{avgdl})}$

여기서:
- $t$는 쿼리 $q$의 용어
- $f(t, d)$는 문서 $d$에서 용어 $t$의 빈도
- $\text{IDF}(t)$는 용어 $t$의 역문서 빈도: $\log\frac{N - n_t + 0.5}{n_t + 0.5}$
- $|d|$는 문서 $d$의 길이
- $avgdl$은 모든 문서의 평균 길이
- $k_1$과 $b$는 각각 용어 빈도 스케일링과 문서 길이 정규화를 제어하는 파라미터

### 4.2.4 신경망 기반 희소 검색 모델

#### 4.2.4.1 SPLADE(Sparse Lexical and Expansion)

SPLADE는 딥러닝을 사용하여 어휘 확장을 수행하는 희소 검색 모델입니다:

$v_{\text{SPLADE}}(x)[i] = \log(1 + \sum_{j=1}^{|x|} w_j \cdot MLMW_{j,i})$

여기서:
- $x$는 입력 텍스트
- $w_j$는 위치 $j$에서의 중요도 가중치
- $MLMW_{j,i}$는 마스크드 언어 모델(MLM)의 출력 로짓

#### 4.2.4.2 DeepCT(Deep Contextualized Term Weighting)

BERT와 같은 사전 학습된 언어 모델을 사용하여 문맥 기반 용어 가중치를 학습합니다:

$w_i = \sigma(W \cdot h_i + b)$

여기서:
- $h_i$는 토큰 $i$의 BERT 임베딩
- $W$와 $b$는 학습 가능한 파라미터
- $\sigma$는 시그모이드 함수

## 4.3 밀집 검색 모델(Dense Retrieval Models)

### 4.3.1 이중 인코더 아키텍처(Dual Encoder Architecture)

쿼리와 문서를 독립적으로 인코딩하는 가장 일반적인 아키텍처입니다:

$S(q, d) = E_Q(q)^T \cdot E_D(d)$

여기서:
- $E_Q$: 쿼리 인코더
- $E_D$: 문서 인코더

학습 목적 함수는 일반적으로 대조 손실(contrastive loss)입니다:

$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(S(q, d^+)/\tau)}{\exp(S(q, d^+)/\tau) + \sum_{j=1}^{n} \exp(S(q, d_j^-)/\tau)}$

여기서:
- $d^+$는 관련 문서
- $d_j^-$는 관련 없는 문서
- $\tau$는 온도 파라미터

### 4.3.2 DPR(Dense Passage Retrieval)

DPR은 이중 인코더 아키텍처를 사용하는 대표적인 밀집 검색 모델입니다:

$E_Q(q) = \text{BERT}_Q(q)_{[\text{CLS}]}$
$E_D(d) = \text{BERT}_D(d)_{[\text{CLS}]}$

여기서 $[\text{CLS}]$는 BERT의 [CLS] 토큰 표현을 나타냅니다.

### 4.3.3 ANCE(Approximate Nearest Neighbor Negative Contrastive Estimation)

ANCE는 어려운 네거티브 샘플을 선택하여 학습 효율성을 향상시킵니다:

$\mathcal{L}_{\text{ANCE}} = -\log \frac{\exp(S(q, d^+)/\tau)}{\exp(S(q, d^+)/\tau) + \sum_{j=1}^{n} \exp(S(q, d_j^{\text{hard}})/\tau)}$

여기서 $d_j^{\text{hard}}$는 현재 모델에 따라 선택된 어려운 네거티브 샘플입니다.

### 4.3.4 ColBERT(Contextualized Late Interaction)

ColBERT는 늦은 상호작용(late interaction)을 통해 토큰 수준의 세밀한 매칭을 수행합니다:

$S_{\text{ColBERT}}(q, d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} E_Q(q_i)^T \cdot E_D(d_j)$

여기서:
- $q_i$는 쿼리의 $i$번째 토큰
- $d_j$는 문서의 $j$번째 토큰

### 4.3.5 BEIR(Benchmark for IR)에서의 일반화 문제

밀집 검색기가 직면하는 주요 과제는 특정 도메인이나 작업에 훈련된 후 다른 도메인으로 일반화하는 능력입니다:

$\Delta_{\text{generalization}} = \text{Performance}(\text{test domain}) - \text{Performance}(\text{training domain})$

## 4.4 하이브리드 및 앙상블 접근법

### 4.4.1 희소-밀집 결합(Sparse-Dense Combination)

희소 및 밀집 검색 결과의 선형 결합:

$S_{\text{hybrid}}(q, d) = \lambda \cdot S_{\text{sparse}}(q, d) + (1-\lambda) \cdot S_{\text{dense}}(q, d)$

### 4.4.2 CLARA(Cross-Lingual Retrieval Augmentation)

CLARA는 다양한 검색 시스템의 강점을 결합하는 앙상블 방법입니다:

$S_{\text{CLARA}}(q, d) = \sum_{i=1}^{m} w_i \cdot S_i(q, d)$

여기서:
- $m$은 결합된 검색기의 수
- $w_i$는 각 검색기의 가중치
- $S_i$는 $i$번째 검색기의 점수 함수

### 4.4.3 RRF(Reciprocal Rank Fusion)

순위 기반 앙상블 방법:

$\text{RRF}(d) = \sum_{i=1}^{m} \frac{1}{k + r_i(d)}$

여기서:
- $r_i(d)$는 $i$번째 검색기에서 문서 $d$의 순위
- $k$는 상수 (일반적으로 60)

## 4.5 쿼리 확장 및 재구성 기법

### 4.5.1 PRF(Pseudo-Relevance Feedback)

초기 검색 결과를 사용하여 쿼리를 확장합니다:

$q' = q + \gamma \sum_{d \in R_{\text{top-k}}(q, C)} w(d) \cdot d$

여기서:
- $R_{\text{top-k}}(q, C)$는 초기 쿼리로 검색된 상위 $k$개 문서
- $w(d)$는 문서 $d$의 가중치
- $\gamma$는 확장 정도를 제어하는 파라미터

### 4.5.2 RM3(Relevance Model 3)

언어 모델링 기반의 쿼리 확장 방법:

$P_{\text{RM3}}(t|q) = (1-\lambda) \cdot P_{\text{MLE}}(t|q) + \lambda \cdot \sum_{d \in R_{\text{top-k}}(q, C)} P(d|q) \cdot P_{\text{MLE}}(t|d)$

여기서:
- $P_{\text{MLE}}(t|q)$는 쿼리에서 용어 $t$의 최대 가능도 추정
- $P_{\text{MLE}}(t|d)$는 문서 $d$에서 용어 $t$의 최대 가능도 추정
- $P(d|q)$는 문서 $d$가 쿼리 $q$와 관련될 확률

### 4.5.3 Doc2Query/DocT5Query

Doc2Query는 문서에서 가능한 쿼리를 생성하여 문서를 확장합니다:

$d' = d + \sum_{i=1}^{n} q_i$

여기서 $q_i$는 문서 $d$에서 생성된 $i$번째 가상 쿼리입니다.

### 4.5.4 LLM 기반 쿼리 확장

대규모 언어 모델을 사용하여 쿼리를 재구성합니다:

$q' = \text{LLM}(\text{prompt}(q))$

예를 들어, 프롬프트는 다음과 같은 형태를 취할 수 있습니다:
"다음 질문을 검색 엔진에 더 효과적인 쿼리로 재구성하세요: {$q$}"

## 4.6 인덱싱 및 검색 최적화

### 4.6.1 역색인(Inverted Index) 구조

역색인은 용어에서 문서로의 매핑을 제공합니다:

$I = \{(t, L_t) | t \in V\}$

여기서:
- $V$는 어휘
- $L_t$는 용어 $t$를 포함하는 문서 목록: $L_t = \{(d, f_{t,d}) | t \in d\}$
- $f_{t,d}$는 문서 $d$에서 용어 $t$의 빈도

### 4.6.2 ANNS(Approximate Nearest Neighbor Search)

#### 4.6.2.1 LSH(Locality-Sensitive Hashing)

유사한 벡터가 높은 확률로 동일한 해시 값을 갖도록 하는 해싱 함수를 사용합니다:

$\text{Pr}[h(v_1) = h(v_2)] \sim \text{sim}(v_1, v_2)$

여기서 $h$는 LSH 함수입니다.

#### 4.6.2.2 HNSW(Hierarchical Navigable Small World)

그래프 기반 ANNS 방법으로, 계층적 구조를 사용하여 검색 효율성을 향상시킵니다:

$G = (V, E, L)$

여기서:
- $V$는 벡터 집합
- $E$는 에지 집합
- $L$은 계층 수준

검색 복잡성: $O(\log N)$ (여기서 $N$은 벡터 수)

#### 4.6.2.3 IVF(Inverted File Index)

벡터 공간을 보로노이 셀로 분할합니다:

$C = \{c_1, c_2, ..., c_k\}$

쿼리 벡터 $q$에 대해 가장 가까운 센트로이드 $c_i$를 찾고, 해당 셀 내에서만 검색을 수행합니다.

### 4.6.3 벡터 압축 및 양자화 기법

#### 4.6.3.1 스칼라 양자화(Scalar Quantization)

각 차원을 독립적으로 양자화합니다:

$Q(v)[i] = \text{round}(\frac{v[i] - \min_i}{\max_i - \min_i} \cdot (2^b - 1))$

여기서 $b$는 비트 깊이입니다.

#### 4.6.3.2 곱 양자화(Product Quantization, PQ)

벡터를 하위 벡터로 분할하고 각 하위 벡터를 독립적으로 양자화합니다:

$v \approx [q_1(v_1), q_2(v_2), ..., q_M(v_M)]$

여기서:
- $v = [v_1, v_2, ..., v_M]$은 벡터의 분할
- $q_i$는 $i$번째 하위 공간에 대한 양자화 함수

#### 4.6.3.3 OPQ(Optimized Product Quantization)

데이터 분포에 맞게 회전을 최적화한 PQ 확장:

$v \approx R^T[q_1(R v_1), q_2(R v_2), ..., q_M(R v_M)]$

여기서 $R$은 회전 행렬입니다.

## 4.7 검색 모델 학습 및 미세 조정

### 4.7.1 대조 학습(Contrastive Learning)

#### 4.7.1.1 인-배치 네거티브(In-Batch Negatives)

계산 효율성을 위해 배치 내 다른 샘플을 네거티브로 사용합니다:

$\mathcal{L}_{\text{in-batch}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(S(q_i, d_i^+)/\tau)}{\sum_{j=1}^{B} \exp(S(q_i, d_j)/\tau)}$

여기서 $B$는 배치 크기입니다.

#### 4.7.1.2 어려운 네거티브 샘플링(Hard Negative Sampling)

모델에 더 도전적인 네거티브 샘플을 선택합니다:

$\mathcal{L}_{\text{hard}} = -\log \frac{\exp(S(q, d^+)/\tau)}{\exp(S(q, d^+)/\tau) + \sum_{j=1}^{n} \exp(S(q, d_j^{\text{hard}})/\tau)}$

어려운 네거티브는 다음과 같이 선택될 수 있습니다:
$d_j^{\text{hard}} = \arg\max_{d \in D \setminus \{d^+\}} S(q, d)$

### 4.7.2 지식 증류(Knowledge Distillation)

복잡한 교차 인코더(cross-encoder)의 지식을 효율적인 이중 인코더로 전달합니다:

$\mathcal{L}_{\text{KD}} = -\sum_{d \in C} S_{\text{CE}}(q, d) \log \frac{\exp(S_{\text{DE}}(q, d)/\tau)}{\sum_{d' \in C} \exp(S_{\text{DE}}(q, d')/\tau)}$

여기서:
- $S_{\text{CE}}$는 교차 인코더의 점수
- $S_{\text{DE}}$는 이중 인코더의 점수

### 4.7.3 준지도 및 자기 지도 학습

#### 4.7.3.1 PAL(Pseudo-Annotated Label)

레이블이 지정되지 않은 데이터에 대한 가상 레이블 생성:

$\mathcal{L}_{\text{PAL}} = \mathcal{L}_{\text{supervised}} + \lambda \mathcal{L}_{\text{pseudo-labeled}}$

#### 4.7.3.2 ICT(Inverse Cloze Task)

자기 지도 방식으로, 문서의 한 부분을 쿼리로, 나머지를 관련 문서로 사용합니다:

$q = d[i:i+l]$
$d^+ = d \setminus d[i:i+l]$

여기서 $d[i:i+l]$은 문서 $d$의 $i$부터 $i+l$까지의 부분입니다.

## 4.8 컨텍스트 인식 검색(Context-Aware Retrieval)

### 4.8.1 대화 검색(Conversational Search)

대화 이력을 고려한 검색:

$S_{\text{conv}}(q_t, h_t, d) = f(E_Q([q_t; h_t]), E_D(d))$

여기서:
- $q_t$는 현재 쿼리
- $h_t$는 대화 이력
- $[q_t; h_t]$는 쿼리와 이력의 연결

### 4.8.2 개인화 검색(Personalized Search)

사용자 프로필이나 선호도를 고려한 검색:

$S_{\text{pers}}(q, u, d) = f(E_Q(q), E_U(u), E_D(d))$

여기서 $E_U(u)$는 사용자 $u$의 임베딩입니다.

### 4.8.3 다중 홉 검색(Multi-hop Retrieval)

복잡한 쿼리를 여러 단계로 나누어 처리합니다:

$q_1, q_2, ..., q_n = \text{Decompose}(q)$

$d_1 = R(q_1, C)$
$d_2 = R(q_2, C|d_1)$
...
$d_n = R(q_n, C|d_{n-1})$

최종 응답: $a = f(q, [d_1, d_2, ..., d_n])$

## 4.9 멀티모달 검색(Multimodal Retrieval)

### 4.9.1 이미지-텍스트 검색(Image-Text Retrieval)

이미지와 텍스트 간의 검색:

$S_{\text{img-txt}}(i, t) = E_I(i)^T \cdot E_T(t)$

여기서:
- $E_I$는 이미지 인코더
- $E_T$는 텍스트 인코더

### 4.9.2 텍스트에서 테이블 검색(Text-to-Table Retrieval)

구조화된 데이터(테이블)에 대한 텍스트 쿼리 검색:

$S_{\text{txt-tbl}}(q, t) = f(E_Q(q), E_T(t))$

여기서 $E_T(t)$는 테이블 $t$의 인코딩입니다.

### 4.9.3 크로스모달 검색(Cross-modal Retrieval)

서로 다른 모달리티 간의 검색을 위한 공유 임베딩 공간 학습:

$\mathcal{L}_{\text{cross}} = \mathcal{L}_{\text{align}} + \lambda \mathcal{L}_{\text{uniform}}$

여기서:
- $\mathcal{L}_{\text{align}}$은 관련 항목을 가깝게 정렬하는 손실
- $\mathcal{L}_{\text{uniform}}$은 임베딩 공간을 균일하게 분포시키는 손실

## 4.10 Retriever 평가 방법론 및 벤치마크

### 4.10.1 검색 품질 평가 지표

#### 4.10.1.1 정밀도-재현율 기반 지표

**정밀도@k(Precision@k)**:
$P@k = \frac{|\{d_1, d_2, ..., d_k\} \cap \text{Relevant}|}{k}$

**재현율@k(Recall@k)**:
$R@k = \frac{|\{d_1, d_2, ..., d_k\} \cap \text{Relevant}|}{|\text{Relevant}|}$

**F1 점수@k**:
$F1@k = \frac{2 \cdot P@k \cdot R@k}{P@k + R@k}$

#### 4.10.1.2 순위 기반 지표

**평균 정밀도(Average Precision, AP)**:
$AP = \sum_{k=1}^{n} P@k \cdot \text{rel}@k$

여기서 $\text{rel}@k$는 $k$번째 결과가 관련 있으면 1, 아니면 0입니다.

**정규화된 할인 누적 이득(NDCG@k)**:
$NDCG@k = \frac{DCG@k}{IDCG@k}$

여기서:
$DCG@k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i+1)}$

#### 4.10.1.3 RAG 특화 지표

**RAG 관련성 점수(RAG Relevance Score)**:
$RRS = \alpha \cdot \text{Retrieval Precision} + (1-\alpha) \cdot \text{Generation Quality}$

### 4.10.2 주요 검색 벤치마크

#### 4.10.2.1 MS MARCO

마이크로소프트의 대규모 질의응답 및 검색 데이터셋:

성능 지표: MRR@10, NDCG@10

#### 4.10.2.2 BEIR(Benchmark for IR)

다양한 도메인과 태스크에서의 검색 일반화 능력을 평가:

$\text{BEIR score} = \frac{1}{|D|} \sum_{d \in D} \text{NDCG@10}_d$

여기서 $D$는 다양한 데이터셋의 집합입니다.

#### 4.10.2.3 MTEB(Massive Text Embedding Benchmark)

다양한 임베딩 태스크에 대한 종합적인 벤치마크:

$\text{MTEB score} = \frac{1}{|T|} \sum_{t \in T} \text{Performance}_t$

여기서 $T$는 태스크 집합입니다.

### 4.10.3 블랙박스 및 효율성 평가

**지연 시간(Latency)**:
$L = \text{Time}(R(q, C))$

**처리량(Throughput)**:
$T = \frac{\text{Number of queries}}{\text{Time}}$


