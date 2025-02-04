# Transformer Study

![Transformer](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Transformer_Attention_Head.svg/800px-Transformer_Attention_Head.svg.png)

## 📌 소개
이 저장소는 **어텐션 메커니즘(Attention Mechanism)** 및 **트랜스포머(Transformer)** 구조를 쉽게 이해하기 위한 학습 자료를 제공합니다. 복잡한 개념을 단순화하여 설명하고 구현하는 것이 목표입니다.

## 📚 다루는 내용

- Attention Mechanism (1): **Bahdanau Attention**
- Attention Mechanism (2): **Luong Attention**
- **Scaled Dot-Product Attention**
- **Single-Head Attention과 인코더**
- **Multi-Head Attention과 인코더**
- **Masked Multi-Head Attention과 디코더**
- **Positional Encoding**

## 🔥 왜 어텐션 메커니즘을 공부해야 할까요?
어텐션 메커니즘은 딥러닝, 특히 **자연어 처리(NLP)** 및 **컴퓨터 비전(CV)** 분야에서 혁신을 가져왔습니다. **트랜스포머, BERT, GPT** 등의 최신 모델을 이해하고 활용하기 위해 필수적인 개념입니다.

## 📁 저장소 구조
```
transformer-study/
│── notebooks/        # Jupyter 노트북 파일 (설명 및 구현 포함)
│── src/              # 주요 구성 요소를 포함한 Python 스크립트
│── data/             # 샘플 데이터셋 (필요한 경우)
│── README.md         # 이 문서
│── requirements.txt  # 의존성 목록
```

## ⚡ 시작하기
### 1️⃣ 저장소 클론하기
```bash
git clone https://github.com/YOUR_USERNAME/transformer-study.git
cd transformer-study
```

### 2️⃣ 의존성 설치하기
```bash
pip install -r requirements.txt
```

### 3️⃣ 노트북 실행하기
Jupyter Notebook을 열어 단계별 구현을 확인하세요.
```bash
jupyter notebook
```

## 🛠 사용된 기술 및 라이브러리
- Python 3.10.4
- PyTorch
- NumPy, Matplotlib
- Jupyter Notebook

## 📌 참고 자료
- [Attention is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- [CS224N: NLP with Deep Learning (Stanford)](http://web.stanford.edu/class/cs224n/)

## 🤝 기여하기
이 프로젝트에 기여하고 싶다면 저장소를 포크하고 풀 리퀘스트(PR)를 제출해주세요!

## 📜 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.

---
✨ 즐겁게 학습하세요! ✨

