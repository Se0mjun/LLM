# Chapter 1: Understanding Large Language Models

## 1.1 모델 구조에 따른 분류

- **인코딩(Encoding)**: 입력을 이해하고 분석하는 데 특화됨
- **디코딩(Decoding)**: 텍스트 생성에 특화됨

## 1.2 주요 언어 모델 유형

### 🔹 자기 회귀 언어 모델 (Autoregressive Language Model)
- 이전 토큰들을 기반으로 다음 토큰을 예측
- 대표: GPT 계열, LLaMA

### 🔹 자동 인코딩 언어 모델 (Autoencoding Language Model)
- 입력 문장을 변형한 후 원래 문장을 복원하는 방식으로 학습
- 대표: BERT

### 🔹 인코더-디코더 모델 (Encoder-Decoder Transformer)
- 다양한 NLP 작업 수행 가능 (요약, 번역, 질의응답 등)
- 대표: T5

## 1.3 대표 모델 소개

- **BERT**  
  - 자동 인코딩 기반  
  - 문장 분류, 토큰 분류에 적합  

- **GPT**  
  - 자기회귀 기반  
  - 어텐션 메커니즘을 통해 이전 토큰 기반으로 다음 토큰 예측  

- **T5 (Text-to-Text Transfer Transformer)**  
  - 인코더-디코더 구조  
  - 텍스트 분류, 요약, 생성 등 범용 NLP 작업에 활용  

- **LLaMA**  
  - Meta에서 공개한 오픈소스 자기회귀 모델  
  - 다양한 크기 제공, 경량화 및 효율성 강조

## 1.4 전이 학습 (Transfer Learning)

- 한 작업(Task)에서 얻은 지식을 다른 관련 작업에 활용
- LLM에서의 전이학습: 사전 학습(pretraining) → 미세 조정(fine-tuning)
- **사전 학습**: 대규모 코퍼스를 통해 일반적인 언어 패턴을 학습 (비지도 학습)
- **미세 조정**: 소규모의 도메인 특화 데이터로 모델을 특정 작업에 적합하도록 조정 (지도 학습)

## 1.5 미세 조정 (Fine-Tuning)

미세 조정은 다음과 같은 절차로 수행됨:

1. **모델 및 파라미터 선택**
2. **학습 데이터 준비**
3. **손실 함수(Loss) 및 기울기(Gradient) 계산**
4. **역전파(Backpropagation)**를 통해 모델 업데이트

- 목적: 사전 학습된 모델의 지식을 바탕으로 특정 태스크에 성능 최적화

## 1.6 핵심 개념

- **어텐션 (Attention)**: 입력 시퀀스 내에서 중요한 부분에 집중하는 메커니즘
- **임베딩 (Embedding)**: 단어를 고차원 벡터로 표현하는 방법
- **토큰화 (Tokenization)**: 문장을 모델이 처리할 수 있는 단위로 나누는 과정
- **정렬 (Alignment)**: 모델 출력을 인간 의도와 일치시키기 위한 기법
- **RLHF (Reinforcement Learning with Human Feedback)**: 인간 피드백을 활용한 보상 학습
- **도메인 특화 LLM**: 특정 분야(예: 법률, 의학 등)에 최적화된 LLM

## 1.7 LLM 기반 애플리케이션

- **텍스트 분류 (Text Classification)**
- **기계 번역 (Machine Translation)**
- **요약 (Summarization)**
- **질의응답 (Question Answering)**
- **대화 시스템 (Conversational AI)**

# Chapter 2: Foundations of Transformer Architectures

## 2.1 트랜스포머의 등장

- 기존 RNN(Recurrent Neural Network)과 LSTM(Long Short-Term Memory)의 한계를 극복하기 위해 트랜스포머가 도입됨
- **병렬처리 효율**, **장기 의존성 문제 해결**, **스케일 확장성** 측면에서 우수

## 2.2 트랜스포머 기본 구조

### 🔹 인코더와 디코더

- **인코더**: 입력 시퀀스를 처리하여 벡터 표현 생성
- **디코더**: 해당 표현을 바탕으로 출력 시퀀스 생성
- GPT 계열은 **디코더만 사용**, BERT는 **인코더만 사용**, T5는 **인코더-디코더 구조**

### 🔹 구성 요소

- **멀티헤드 어텐션 (Multi-Head Attention)**
- **포지션 와이즈 피드포워드 네트워크**
- **포지셔널 인코딩 (Positional Encoding)**
- **레이어 정규화 & 잔차 연결**

## 2.3 어텐션 메커니즘

### 🔸 Scaled Dot-Product Attention

- 입력 쿼리(Query), 키(Key), 값(Value)로부터 다음 계산 수행:

  $$
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
  $$

- $Q, K, V$는 모두 임베딩 벡터에서 선형변환된 행렬

### 🔸 Self-Attention

- 시퀀스 내 모든 토큰이 서로를 참조할 수 있게 함
- 문맥을 고려한 표현 학습 가능

## 2.4 포지셔널 인코딩

- 트랜스포머는 순서를 인식하지 못하기 때문에 **위치 정보(Position)**를 인코딩하여 입력에 추가
- 대표 방식: 사인/코사인 함수 기반 포지셔널 인코딩

## 2.5 학습 방식

### 🔸 사전학습 (Pretraining)

- 일반 텍스트 코퍼스를 기반으로 모델에 언어적 지식 주입
- 대표적 사전학습 목표:
  - **MLM (Masked Language Modeling)**: BERT
  - **CLM (Causal Language Modeling)**: GPT
  - **Span Corruption**: T5

### 🔸 미세 조정 (Fine-Tuning)

- 태스크 특화 데이터셋을 사용하여 모델을 특정 작업에 적합하게 조정
- 사전학습된 가중치를 초기값으로 활용

## 2.6 트랜스포머의 장점과 한계

### ✅ 장점

- 병렬처리 용이
- 확장성과 일반화 능력 우수
- 다양한 NLP 태스크에 적용 가능

### ⚠️ 한계

- 연산량이 많고 메모리 요구도 큼
- 입력 길이가 늘어날수록 **어텐션 계산 비용**이 **제곱**으로 증가

## 2.7 트랜스포머 기반 모델 예시

| 모델 | 구조 | 특징 |
|------|------|------|
| BERT | 인코더 | 문장 이해에 최적 |
| GPT  | 디코더 | 텍스트 생성에 특화 |
| T5   | 인코더-디코더 | 텍스트 입력 → 텍스트 출력 방식 통일 |
| DistilBERT | 인코더 | 경량화된 BERT |
| LLaMA | 디코더 | Meta 개발, 오픈소스, 경량-고성능 |
| PaLM | 디코더 | 대규모 데이터에서 학습된 멀티태스크 모델 |

---

# Chapter 3: Training and Adapting Large Language Models

## 3.1 사전 학습 (Pretraining)

- 대규모 말뭉치를 활용해 언어의 통계적 패턴을 학습
- **비지도 학습** 또는 **자기 지도 학습(self-supervised learning)** 형태로 수행
- 대표적인 목표 함수:
  - **Causal LM**: 다음 토큰 예측 (GPT)
  - **Masked LM**: 일부 마스킹된 토큰 복원 (BERT)
  - **Span Corruption**: 연속된 토큰 덩어리 예측 (T5)

## 3.2 미세 조정 (Fine-Tuning)

- **지도 학습 데이터셋**을 이용하여 특정 작업에 맞게 사전 학습된 모델 조정
- 일반적인 미세 조정 절차:
  1. 태스크별 입력/출력 정의
  2. 사전 학습된 모델 불러오기
  3. 출력층 수정 (e.g., 분류기 추가)
  4. 학습률 설정 후 전체 네트워크 업데이트

## 3.3 파라미터 효율적 미세 조정 (PEFT)

- 모든 파라미터를 업데이트하지 않고, **일부만 조정**하여 효율성을 높이는 전략

### 🔹 대표 기법

#### ✅ LoRA (Low-Rank Adaptation)
- 선형 계층에 저랭크 행렬을 삽입하여 파라미터 수를 줄이고 학습 효율성 향상
- 기존 파라미터는 고정 (frozen), 새로운 적응 파라미터만 업데이트

#### ✅ Prefix Tuning
- 입력 앞에 학습 가능한 "프리픽스 벡터"를 추가하여 맥락을 조정

#### ✅ Adapter
- 각 Transformer 레이어 사이에 소형 MLP 모듈을 추가하여 작은 파라미터만 학습

## 3.4 RLHF (Reinforcement Learning with Human Feedback)

- 인간 피드백을 통해 LLM 출력을 **정렬(alignment)**시키는 과정

### 🔸 구성 단계

1. **사전 학습 (Supervised Fine-Tuning)**: 일부 정답 레이블로 초기 미세 조정
2. **보상 모델 학습 (Reward Model)**: 사람이 더 선호한 응답 쌍을 비교하여 학습
3. **강화 학습 (PPO 등)**: 보상 모델을 기준으로 응답을 개선

## 3.5 Instruct 모델

- 자연어 명령(instruction)에 반응하도록 미세 조정된 모델
- 예시: `text-davinci-003`, `ChatGPT`, `Flan-T5`

### 특징:
- Prompt에 따라 문장 생성, 분류, 추론 등 다양한 작업 수행
- "Do what I mean" 스타일의 직관적 제어 가능

## 3.6 LLM을 도메인에 맞게 적응시키기

### 방법:
- **도메인 데이터셋 수집**
- **사전 학습 또는 LoRA 기반 미세 조정**
- **분야 전문가 피드백 수렴 (RLHF 가능)**

### 예시:
- **의료**: Med-PaLM
- **법률**: LawGPT
- **연구 논문 요약**: SciSummary

---

# Chapter 4: Serving, Scaling, and Using LLMs

## 4.1 LLM 서빙(Serving) 개요

- 학습된 LLM을 사용자나 애플리케이션이 접근 가능하도록 **API 형태로 제공**
- LLM은 계산 자원이 크고 응답 시간이 길기 때문에 **최적화된 서빙 전략**이 중요

### 서빙 방식 예시:
- **Batch Inference**: 여러 요청을 동시에 처리해 GPU 활용도 향상
- **Streaming Inference**: 응답을 점진적으로 반환하여 체감 속도 개선
- **Quantized Inference**: 모델 크기를 줄여 CPU 또는 경량 GPU에서 추론 가능

## 4.2 모델 최적화 전략

### 🔹 양자화 (Quantization)

- 모델의 weight와 activation을 16bit 또는 8bit로 줄임
- 추론 속도 향상 및 메모리 절감
- 예: INT8 quantization (GPTQ, AWQ)

### 🔹 지연 로딩 (Lazy Loading)

- 요청이 들어올 때만 모델을 메모리에 불러옴
- 서버 부하 감소

### 🔹 혼합 정밀도 (Mixed Precision)

- float16 (FP16), bfloat16 (BF16)을 이용해 메모리 사용량 감소
- A100 GPU 등에서 성능 극대화 가능

## 4.3 LLM 배포 도구 및 프레임워크

| 도구 | 설명 |
|------|------|
| **Hugging Face Transformers** | 다양한 사전학습 모델과 API 추론 지원 |
| **FastAPI / Flask** | LLM을 REST API로 서빙하는 데 사용 |
| **vLLM** | 빠르고 대규모 배치를 지원하는 LLM 추론 프레임워크 |
| **DeepSpeed-Inference** | 효율적인 분산 추론 지원 |
| **ONNX Runtime / TensorRT** | 모델을 추론 전용 엔진으로 변환해 속도 최적화 |

## 4.4 클라우드 기반 서비스

- Hugging Face Inference Endpoints
- OpenAI API (ChatGPT, GPT-4)
- AWS SageMaker, GCP Vertex AI, Azure ML 등

## 4.5 사용자 정의 애플리케이션 구축

- **LLM + UI**: Streamlit, Gradio로 간단한 챗봇 만들기
- **LangChain / LlamaIndex**: 문서 기반 질의응답 시스템 구축
- **멀티모달 파이프라인**: 이미지/음성 + 텍스트 입력 처리

## 4.6 비용과 성능 고려사항

| 항목 | 고성능 GPU | 경량화 모델 |
|------|------------|--------------|
| 비용 | 매우 높음 | 낮음 |
| 응답 시간 | 빠름 | 중간 |
| 설치/유지관리 | 복잡 | 단순 |
| 활용 예 | 실시간 대화형 AI | 임베디드 응용 |

## 4.7 LLM API 호출 예시 (Python)

# Chapter 5: Prompt Engineering and Interaction Techniques

## 5.1 프롬프트 엔지니어링(Prompt Engineering) 개요

- LLM에 원하는 동작을 유도하기 위해 **입력 텍스트(prompt)**를 구조적으로 설계하는 기법
- 사전 학습된 모델을 **fine-tuning 없이도** 다양한 작업에 활용 가능하게 함
- 다양한 문장 구조나 지시어를 실험하여 모델 성능을 극대화

## 5.2 프롬프트 방식 종류

| 유형 | 설명 | 예시 |
|------|------|------|
| Zero-shot | 예시 없이 직접 질문 | "Translate to French: How are you?" |
| One-shot | 1개의 예시 제공 | "Hi → Salut\nBye →" |
| Few-shot | 여러 개의 예시 제공 | "Hi → Salut\nBye → Au revoir\nThanks →" |
| Instruction-based | 명령문 형태로 작업 지시 | "Summarize this paragraph in one sentence." |
| Chain-of-Thought (CoT) | 중간 추론 과정을 유도 | "Let's think step by step..." |

## 5.3 프롬프트 구성 전략

- **명확한 작업 설명**: 번역, 요약, 분류 등 원하는 작업을 명확히 기술
- **역할 부여**: "You are a legal advisor."와 같이 시스템 프롬프트로 맥락 설정
- **출력 형식 명시**: JSON, 마크다운, 표 등 원하는 결과 포맷을 제시
- **제약조건 부여**: 글자 수 제한, 키워드 포함 여부 등

## 5.4 인-컨텍스트 학습 (In-Context Learning)

- 프롬프트 내에서 **작업 예시**를 제공하여 모델이 맥락을 파악하고 따라하게 함
- 모델 파라미터는 고정된 채 동작 → 파인튜닝 없이도 특정 태스크에 대응 가능

### 특징:
- Few-shot prompting 기반
- 빠른 테스트와 반복 가능
- GPT-3/4, Claude, PaLM 등 대부분의 LLM에서 지원

## 5.5 프롬프트 튜닝과 파인튜닝 비교

| 항목 | 프롬프트 튜닝 | 파인튜닝 |
|------|----------------|------------|
| 방식 | 입력 문장 구조 조정 | 모델 가중치 수정 |
| 장점 | 빠르고 간단, 자원 적게 사용 | 성능 극대화 가능 |
| 단점 | 복잡한 작업엔 한계 | 리소스 및 시간 많이 소요 |
| 예시 | P-tuning, Prefix-tuning | LoRA, Full fine-tuning |

# Chapter 6: Advanced Topics, Limitations, and the Future of LLMs

## 6.1 LLM의 한계

| 항목 | 설명 |
|------|------|
| 지식 업데이트 | 고정된 시점 이후의 지식 반영 불가 (사전 학습 기반) |
| 환각 현상 (Hallucination) | 실제와 다른 정보 생성 가능성 |
| 수학적 계산 | 기본적인 산술 연산에서 오류 발생 가능 |
| 긴 문맥 처리 | 컨텍스트 윈도우 제한 존재 (e.g. 4K, 8K, 최대 128K) |
| 데이터 편향 | 학습 데이터의 편향이 출력에 반영될 수 있음 |
| 높은 비용 | 훈련과 추론 모두에서 큰 자원 소모 |

## 6.2 LLM과 윤리

- **프라이버시 문제**: 개인 정보가 포함된 데이터로 학습 시 민감 정보 출력 가능
- **저작권 이슈**: 웹 기반 학습 데이터 내 저작권 문제
- **악용 가능성**: 허위 정보 생성, 피싱 이메일 생성, 자동화된 스팸 등
- **책임 주체 부재**: 모델이 내놓는 응답에 대한 법적·윤리적 책임의 불명확성

### 대응 방안

- 데이터 필터링, RLHF 통한 정렬(alignment)
- 수동 검토, 콘텐츠 안전 필터 적용
- 투명한 모델 카드와 사용 가이드라인 제공

## 6.3 멀티모달 LLM

### 정의
- 텍스트 외에도 **이미지, 오디오, 비디오** 등의 다양한 입력을 처리하는 모델
- GPT-4, Gemini, Claude, Kosmos, Flamingo 등이 대표적

### 활용 분야
| 분야 | 예시 |
|------|------|
| 시각 질의응답 | 이미지 기반 질문 응답 (Visual QA) |
| 의학 영상 해석 | X-ray, MRI 등에서 진단 지원 |
| 로봇 제어 | 이미지 + 명령어 → 동작 결정 (Embodied AI) |
| 디자인 자동화 | 스케치 → 설명, 설명 → 디자인 생성 |

## 6.4 LLM의 미래

### 🔹 기술적 발전 방향

- **더 긴 컨텍스트**: 128K 이상의 문서 길이 처리
- **메모리 기반 LLM**: 장기 기억 시스템 통합 (e.g. ReAct + DB)
- **에이전트화**: LLM이 스스로 작업 계획, 도구 실행 수행
- **지속적 학습(Continual Learning)**: 새로운 지식을 온라인 업데이트

### 🔹 연구 트렌드

| 주제 | 설명 |
|------|------|
| Retrieval-Augmented Generation (RAG) | 실시간 정보 검색 후 생성 결합 |
| Mixture-of-Experts (MoE) | 모델 일부만 활성화해 추론 효율화 |
| Parameter-Efficient Tuning | LoRA, Adapter, Prompt-Tuning의 확장 |
| Alignment 연구 | 인간 가치에 부합하는 응답 생성 기법 |
| Open-Weight Movement | 공개 LLM (LLaMA, Falcon, Mistral 등)의 활발한 생태계 |

## 6.5 책임 있는 LLM 사용을 위한 제언

- 모델의 한계에 대해 사용자 교육
- 출력물에 대한 검증 및 인간 리뷰 권장
- 투명한 문서화와 사용 범위 정의
- 윤리 가이드라인 및 정책 마련

---

# Appendix A: Glossary of Terms

| 용어 | 정의 |
|------|------|
| **LLM (Large Language Model)** | 대규모 말뭉치를 학습한 언어 생성 및 이해 인공지능 모델 (ex: GPT-4, LLaMA, PaLM) |
| **Transformer** | 어텐션 기반의 시퀀스 모델 아키텍처로, 대부분의 최신 LLM의 기반 구조 |
| **Self-Attention** | 입력 시퀀스 내 각 토큰이 다른 토큰을 참조해 문맥을 파악하는 메커니즘 |
| **Positional Encoding** | 트랜스포머에서 순서를 고려하기 위한 벡터 표현 |
| **Embedding** | 단어 또는 문장을 고차원 벡터로 변환한 것 |
| **Token / Tokenization** | 모델 입력 단위로, 문장을 서브워드 단위로 쪼개는 과정 |
| **Autoregressive Model** | 이전 토큰을 기반으로 다음 토큰을 예측하는 생성 중심 모델 (ex: GPT, LLaMA) |
| **Autoencoding Model** | 손상된 입력을 원래대로 복원하는 데 초점을 맞춘 모델 (ex: BERT) |
| **Encoder / Decoder** | 입력을 벡터로 인코딩하거나, 그 벡터로부터 출력을 생성하는 트랜스포머 구성 요소 |
| **Fine-Tuning** | 사전 학습된 모델을 특정 태스크에 맞게 재훈련하는 과정 |
| **PEFT (Parameter-Efficient Fine-Tuning)** | 전체 모델이 아닌 일부 파라미터만 업데이트하는 효율적인 미세 조정 기법 |
| **LoRA (Low-Rank Adaptation)** | 선형 계층의 일부에만 적은 수의 파라미터를 추가해 학습하는 PEFT 방식 |
| **Prompt Engineering** | 모델의 출력 결과를 유도하기 위해 입력을 정교하게 설계하는 기술 |
| **In-Context Learning** | 학습 없이 프롬프트 내 예시만으로 모델이 태스크를 수행하도록 유도하는 방식 |
| **Chain-of-Thought (CoT)** | 중간 추론 단계를 언어로 기술하여 복잡한 문제 해결 정확도를 높이는 기법 |
| **RLHF (Reinforcement Learning with Human Feedback)** | 인간의 선호도에 기반해 보상 모델을 훈련하고 LLM 응답을 개선하는 강화 학습 방식 |
| **Alignment** | 모델의 출력이 인간의 기대, 가치, 윤리에 부합하도록 조정하는 과정 |
| **Hallucination** | LLM이 사실이 아닌 내용을 사실처럼 생성하는 오류 현상 |
| **RAG (Retrieval-Augmented Generation)** | 검색 결과를 활용해 더 정확한 텍스트를 생성하는 프레임워크 |
| **Multimodal** | 텍스트 외에 이미지, 오디오 등 다양한 입력 모달리티를 처리할 수 있는 모델 |
| **Context Window** | LLM이 한 번에 고려할 수 있는 입력 길이 (예: GPT-4는 최대 128K 토큰까지 지원) |
| **Quantization** | 모델 파라미터를 낮은 비트 수로 표현하여 속도 및 메모리 최적화하는 기법 |
| **Zero-shot / Few-shot Prompting** | 예시 없이 또는 적은 수의 예시로 태스크를 해결하는 프롬프트 설계 방식 |


# Appendix B: Model Comparison Table

| 모델 | 구조 | 학습 방식 | 용도 | 공개 여부 | 특징 |
|------|------|-----------|------|-----------|------|
| **BERT** | 인코더 | Masked Language Modeling (MLM) | 문장 이해, 분류 | ✅ 오픈소스 | 문맥 기반 이해, 양방향 처리 |
| **GPT-2 / 3 / 4** | 디코더 | Causal Language Modeling (CLM) | 텍스트 생성, 대화 | ❌ (GPT-3/4) | 자연스러운 생성, Instruct형 fine-tuning |
| **T5** | 인코더-디코더 | Text-to-Text | 다목적(NLP 전반) | ✅ 오픈소스 | 모든 작업을 텍스트 → 텍스트로 처리 |
| **DistilBERT** | 인코더 | MLM | 경량화된 NLP 처리 | ✅ 오픈소스 | BERT보다 40% 작고 60% 빠름 |
| **LLaMA** | 디코더 | CLM | 연구, 생성 | ✅ (LLaMA 2부터) | 고성능 경량 모델, Meta 제공 |
| **PaLM** | 디코더 | CLM + Few-shot | 대규모 생성, reasoning | ❌ | 구글의 초대형 언어 모델 |
| **OPT** | 디코더 | CLM | 생성, 연구용 | ✅ 오픈소스 | Meta 공개, GPT 대안으로 활용 |
| **Falcon** | 디코더 | CLM | 생성, 코드 등 | ✅ (Falcon-7B, 40B) | HuggingFace에서 제공 |
| **Mistral** | 디코더 | CLM | 고속 생성 | ✅ 오픈소스 | Grouped Query Attention 적용 |
| **Claude** | 디코더 | RLHF 기반 fine-tuning | 대화형 AI | ❌ | Anthropic 제공, 안전성 강조 |
| **Gemini (Bard)** | 멀티모달 | CLM + Retrieval | 검색, 생성 | ❌ | Google DeepMind의 최신 모델 |
| **GPT-Neo / GPT-J / GPT-NeoX** | 디코더 | CLM | 오픈 GPT 대체 | ✅ 오픈소스 | EleutherAI 커뮤니티 주도 |
| **Flan-T5** | 인코더-디코더 | Instruction fine-tuning | 텍스트 분류/생성 | ✅ 오픈소스 | 구글의 지시어 기반 튜닝 모델 |
| **ChatGLM** | 디코더 | Instruction fine-tuning | 중국어 중심 대화형 모델 | ✅ 제한적 공개 | bilingual 모델 (중·영) |
| **BLOOM** | 디코더 | Multilingual CLM | 다국어 생성 | ✅ 오픈소스 | BigScience 프로젝트 결과물 |

---

## 범례
- **CLM**: Causal Language Modeling (자기회귀 생성)
- **MLM**: Masked Language Modeling (마스크 복원)
- ✅ 공개된 모델 / ❌ 비공개 모델

