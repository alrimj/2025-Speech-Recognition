#Speech Features 

https://ocw.mit.edu/courses/6-345-automatic-speech-recognition-spring-2003/pages/lecture-notes/

## Speech Features 란?
음성신호는 비정상(non-stationary) 신호이기 때문에 
단순한 파형(waveform) 분석만으로는 '의미있는' 정보를 추출하기 어렵다.

따라서, 음성 데이터를 수학적으로 분석 및 변환하여 
**음성의 주요 정보를 압축적으로 표현하는 특성(Feature)**을 추출하는 것이 필요하다.

음성 feature의 목표 : 
- 주어진 신호에서 중요한 정보만을 추출 (잡음 제거, 차원 축)
- 음소(phoneme)와 단어(word)의 패턴 인식
- 딥러닝 모델의 입력 데이터로 활용

📌 음성 특징의 분류 :
|Feature |Type	|설명	|활용|
|    --: |  ::    |    ::  |   ::   |
|Spectral Features	|주파수 영역에서 음성 정보를 추출	|MFCC, Mel Spectrogram|
|Prosodic Features	|억양, 강세, 길이 등	|Emotion Recognition|
|Phonetic Features	|음소 정보 추출	|HMM-GMM 모델|
|Temporal Features	|시간 도메인에서 특징 추출	|Zero Crossing Rate|


## 음성 신호의 기본 특징 (기초 Feature)
### Zero Crossing Rate(ZCR)
- 정의: 신호가 0을 기준으로 바뀌는 횟수
- 특징: 잡음(noise)와 조음 방식(자음 vs 모음) 분석에 사용됨
- 활용: 음성-비음성(Voiced-Unvoiced) 분류

'''
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 오디오 로드
y, sr = librosa.load("sample.wav")

# ZCR 계산
zcr = librosa.feature.zero_crossing_rate(y)

# 시각화
plt.figure(figsize=(10, 4))
plt.plot(zcr[0], label="Zero Crossing Rate")
plt.legend()
plt.show()

'''
📌 해석
- 높은 ZCR → 무성음(자음) / 노이즈
- 낮은 ZCR → 유성음(모음) / 지속적인 소리

### Short-Time Energy (STE)
- 정의: 음성 신호의 에너지를 작은 프레임 단위로 측정
- 특징: 음성-비음성(Voiced-Unvoiced) 구분에 사용

'''
frame_size = 1024
frame_stride = 512
energy = np.array([sum(abs(y[i:i+frame_size]**2)) for i in range(0, len(y), frame_stride)])

plt.figure(figsize=(10, 4))
plt.plot(energy, label="Short-Time Energy")
plt.legend()
plt.show()

'''
📌 해석
- 높은 에너지 → 강한 발성 구간
- 낮은 에너지 → 정적 구간(침묵, 소음)

## Spectral Features (주파수 기반 특징 분석)
고급 음성 특징은 주로 주파수 변환을 통해 추출되는 특징들로 다음과 같다.

- Mel Spectrogram
- MFCC (Mel-Frequency Cepstral Coefficients)
- Spectral Centroid
- Spectral Bandwidth

### Spectrogram

시간에 따른 주파수 화를 시각적으로 표현하기 :
'''
D = np.abs(librosa.stft(y))  # STFT 변환
log_D = librosa.amplitude_to_db(D, ref=np.max)  # dB 변환

plt.figure(figsize=(10, 4))
librosa.display.specshow(log_D, sr=sr, x_axis="time", y_axis="log")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram")
plt.show()
'''
📌 해석
- X축: 시간 (Time)
- Y축: 주파수 (Hz)
- 색상: 에너지 크기 (강한 신호는 밝은 색)

### 멜 스펙트로그램(Mel Spectrogram)
사람의 청각 인식 방식을 반영한 주파수 변환.

'''
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis="time", y_axis="mel")
plt.colorbar(format="%+2.0f dB")
plt.title("Mel Spectrogram")
plt.show()

'''
📌 해석
- 멜 주파수 변환을 통해 인간 청각 시스템을 모방
- 높은 주파수(고음)는 압축적으로 표현

### MFCC(Mel-Frequency Cepstral Coefficients)
음성 인식에서 가장 많이 사용되는 특징 값이다.
음성의 포먼트(Formants) 정보를 추출하여 AI 모델에서 활용한다.

'''
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, sr=sr, x_axis="time")
plt.colorbar()
plt.title("MFCC")
plt.show()

'''
📌 해석
- 음성의 주된 특징을 벡터로 변환하여 머신러닝 모델에서 사용
- CNN/RNN 기반 음성 인식 모델의 입력 데이터로 활용 가능
