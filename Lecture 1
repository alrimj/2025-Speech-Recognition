#Speech signal processing and analysis

https://ocw.mit.edu/courses/6-345-automatic-speech-recognition-spring-2003/pages/lecture-notes/

## 음성 신호의 기본 개념
음성신호 : 시간에 따라 연속적으로 변화하는 공기의 압력 파동으로 연속(Continuous)적인 신호이다.
구성요소 = 
- 기본 주파수(Fundamental Frequency, F0) : 성대 진동에 의해 생성되는 주된 주파수 성분
- 포먼트(Formants) : 공명에 의해 증폭되는 특정 주파수 영역(주로 모음의 특징을 결정)
- 잡음(Noise), 배경 신호 : 비음성 요소(호흡이나 환경적 소음과 같은)

음성신호를 컴퓨터에서 처리하기 위해서는 디지털 변환이 필요하다.
이 과정은 샘플링(Sampling), 양자화(Quantization)가 포함된다.

(1) 샘플링
이는 연속적인 아날로그 신호를 일정한 시간의 간격으로 측정하여 이산(discrete) 신호로 변환하는 과정이다.

**샘플링 정리** (Nyquist Theorem)
- 주파수 f의 신호를 디지털화 하기 위해서는 최소한 2f 이상의 샘플링 주파수가 필요하다.
  - ex) 사람이 들을 수 있는 주파수의 대역: 20Hz~20kHz 이므로 최소 40kHz 이상의 샘플링 주파수가 필요.
(2) 양자화
이는 샘플링된 신호를 유한한 비트(bit) 수로 표현하는 과정이다.

**양자화 비트수와 해상도**
- 8-bit 오디오 → 256 단계
- 16-bit 오디오 → 65,536 단계 (CD 품질)
- 24-bit 오디오 → 16,777,216 단계 (스튜디오 품질)

## 음성 신호 분석

### 푸리에 변환(Fourier Transform, FT)
음성 신호는 여러 개의 주파수 성분으로 구성되어 있다.
푸리에 변환은 시간 영역 신호를 주파수 영역 신호로 변환하는 도구이다.

**DTFT (Discrete-Time Fourier Transform) **
- X(f)= n=−∞ ∑∞ x[n]e ^ −j2πfn

> 문제점 : 연속적인 주파수 변환이 필요하여 실질적인 계산이 불가능하다.
'''
import numpy as np
import matplotlib.pyplot as plt

# 푸리에 변환 수행
D = np.abs(librosa.stft(y))  # Short-Time Fourier Transform (STFT)
log_D = librosa.amplitude_to_db(D, ref=np.max)

# 주파수 스펙트로그램 출력
plt.figure(figsize=(10, 4))
librosa.display.specshow(log_D, sr=sr, x_axis="time", y_axis="log")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (FFT)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.show()
'''
밝은 부분: 강한 주파수 성분
어두운 부분: 약한 주파수 성분
시간에 따른 주파수 변화를 볼 수 있음

### 멜 스펙트로그램(Mel Spectrogram)
사람의 청각 인식 방식을 반영하여 로그 스케일로 변환한 주파수 분석 기법이다.
- Mel(f)=2595log 10(1+f/700)
'''
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis="time", y_axis="mel")
plt.colorbar(format="%+2.0f dB")
plt.title("Mel Spectrogram")
plt.xlabel("Time (s)")
plt.ylabel("Mel Frequency")
plt.show()
'''

### MFCC(Mel-Frequency Cepstral Coefficients)
음성 인식에서 가장 많이 사용되는 특징 값이다.
음성의 포먼트(Formants) 정보를 추출하여 AI 모델에서 활용한다.

아래와 같이 음성의 주된 특징을 벡터로 변환하여 머신러닝 모델에서 활용할 수 있다.
'''
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, sr=sr, x_axis="time")
plt.colorbar()
plt.title("MFCC")
plt.xlabel("Time (s)")
plt.ylabel("MFCC Coefficients")
plt.show()
'''
