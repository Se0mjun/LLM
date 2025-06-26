# LLM 심화 학습 로드맵

## ✅ 개요

이 문서는 기본적인 LLM 이해를 바탕으로 더 심화된 개념과 실습을 학습하고자 하는 분들을 위한 가이드입니다. 목적별로 주제를 분류하고, 추천 자료 및 실습 방향을 제시합니다.

---

## 🎯 1. Transformer 구조 내부 이해

### 학습 주제
- Transformer 아키텍처 수식적 분석
- Self-Attention 연산의 수학적 의미
- Positional Encoding 종류 (Sinusoidal, Rotary, ALiBi)
- Feedforward Network (FFN)
- Residual Connection과 LayerNorm
- Causal vs Bidirectional Attention 차이

### 추천 자료
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- 《Transformers for Natural Language Processing》 (Denis Rothman)
- Hugging Face `transformers` GitHub 코드 분석

---

## 🎯 2. Fine-tuning 및 LoRA 기반 실습

### 학습 주제
- Hugging Face `Trainer` 사용법
- LoRA / Adapter / Prompt-Tuning 등 PEFT 기법
- 학습률, 손실 함수 튜닝
- 커스텀 데이터셋 준비 및 텍스트 전처리

### 실습 키워드
- `transformers`, `peft`, `datasets`, `trl`
- Alpaca, DPO, Self-Instruct
- TinyLLaMA, Mistral 등 Colab에서 실습

### 추천 자료
- Hugging Face 공식 튜토리얼 (PEFT)
- [llama-recipes](https://github.com/facebookresearch/llama-recipes)
- DeepLearning.AI - Fine-Tuning LLMs 과정

---

## 🎯 3. 멀티모달 / LLM 기반 에이전트

### 학습 주제
- CLIP, Flamingo, GPT-4o 등 멀티모달 모델 구조
- LangChain / LangGraph로 툴 사용 및 에이전트 구성
- Retrieval-Augmented Generation (RAG)
- Vision-Language-Action 모델

### 추천 자료
- GPT-4 Technical Report
- OpenFlamingo 프로젝트
- LangChain 공식 문서
- 논문: NaViLA, ViperGPT 등

---

## 🎯 4. 최신 아키텍처 및 연구 트렌드

### 학습 주제
- SSM 기반 모델 (Mamba, RWKV, Hyena)
- Long-context modeling (GPT-4 128K, Gemini)
- Mixture-of-Experts (MoE), Sparse Transformer
- Efficient Transformer variants (RetNet 등)

### 추천 자료
- 논문: Attention is All You Need
- Retentive Network, Mamba, LLaMA 3 등
- Hugging Face Papers with Code
- ArXiv Sanity, arxiv-daily newsletter

---

## 📌 추천 학습 순서 (실전 중심)

1. 🤖 Hugging Face로 GPT 또는 T5 Fine-tuning 실습  
2. 🧪 LoRA / Adapter 등 PEFT 실험  
3. 📂 LangChain 기반 문서 QA 또는 에이전트 구성  
4. ✍️ Transformer 내부 연산 수식적 분석  
5. 📈 최신 논문 분석 및 아키텍처 비교  

---

## 🎁 추가 도움 가능 항목

- 실전 Fine-tuning 코드 템플릿
- 최신 논문 요약 (예: Mamba, GPT-4, LLaMA 3)
- LangChain / RAG 프로젝트 설계
- 멀티모달 LLM 파이프라인 설계 지원

---

> 이 로드맵은 실제 프로젝트 응용, 연구 준비, 고급 튜닝 등에 필요한 주제를 모두 포함하며, 실습과 이론을 병행할 수 있도록 설계되어 있습니다.
