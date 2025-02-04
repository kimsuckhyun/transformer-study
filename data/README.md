# 📁 데이터 디렉토리

이 디렉토리는 모델 학습 및 실험을 위한 데이터셋을 저장하는 공간입니다.

## 📥 OpenSubtitles 데이터 다운로드 및 압축 해제

아래 명령어를 실행하여 OpenSubtitles 영어-한국어 병렬 데이터를 다운로드하고 압축을 해제할 수 있습니다.

```bash
wget -O opensubtitles_en_ko.zip "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-ko.txt.zip"
unzip opensubtitles_en_ko.zip -d opensubtitles_en_ko
