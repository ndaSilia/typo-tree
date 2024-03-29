# Typo Tree

## 개요
NLP 인공지능 모델 Typo는 대화 하나 없는 소설, 대화만으로 구성된 소설, 중력이 없는 세상을 배경으로 한 소설, 한 성별만 존재하는 소설, 자신이 회귀자인 줄 모르는 회귀 소설 등의 독특하고 창의적인 단편 소설을 작성하여 사용자에게 신비로운 아이디어를 제공합니다.

## 목적
### 1. 단편 소설 제작
앞써 소개되었던 것처럼 Typo-tree 프로젝트의 목적은 독특하고 특이한 단편 소설을 작성하여 사용자의 창의력을 자극하는 것을 목적으로 하고 있습니다. 본래에는 장편 소설 작성을 목표로 기획되었으나, 설정 오류, 개연성 파괴, 고유 명사 오사용 등의 문제로 단편 소설을 목표로 잡게 되었습니다.
### 2. 아이디어 제공
Typo는 유저 입력값, 즉 시드(Seed)에 이어지는 이야기를 만듭니다. 만일 중력이 없는 세상에 대한 아이디어를 얻고 싶다면, '허공을 유영하여 다른 건물로 나아갔다' 혹은 '세상에 중력이 사라졌다' 등의 문장을 입력하여 인간의 상식과 편견, 고정 관념과 제한적인 경험으로는 쉽사리 생각할 수 없는 이야기에 대한 아이디어를 얻어갈 수 있습니다.

## 구현
### 모델
소설 특성상 문맥을 읽고 문장 간 논리의 지속성을 위해 LSTM을 차용하였습니다.
### 하이퍼 파라미터
- seq_len = 60
- step = 1
- hidden_size = 128
- dropout = 0.2
- batch_size = 128
- epochs = 100

## 개발 일지
### 3/14
시퀀스 길이에 맞춰 유저 입력값을 변환하는 것에 대한 영향력을 최소화하고 모델이 제대로 논리를 끝맺기 위해 START, END 태그를 추가하였습니다.
START와 END는 각각 한 화의 시작과 끝을 의미하며, 이 작업을 통해 모델은 이야기의 끝을 매끄럽게 구현할 수 있을 것입니다.
