``` resnet50
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def resnet_block(x, filters, stride=1):
    # Shortcut connection
    shortcut = x

    # First convolutional layer
    x = Conv2D(filters, kernel_size=1, strides=stride, padding='valid')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second convolutional layer
    x = Conv2D(filters, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Third convolutional layer
    x = Conv2D(4*filters, kernel_size=1, strides=1, padding='valid')(x)
    x = BatchNormalization()(x)

    # Adjust shortcut connection
    if stride != 1:
        shortcut = Conv2D(4*filters, kernel_size=1, strides=stride, padding='valid')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Add shortcut connection
    x = Add()([x, shortcut])
    x = ReLU()(x)

    return x

def resnet50():
    inputs = Input(shape=(224, 224, 3))

    # Initial convolutional layers
    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # ResNet blocks
    x = resnet_block(x, 64)
    x = resnet_block(x, 64)
    x = resnet_block(x, 64)

    x = resnet_block(x, 128, stride=2)
    x = resnet_block(x, 128)
    x = resnet_block(x, 128)
    x = resnet_block(x, 128)

    x = resnet_block(x, 256, stride=2)
    x = resnet_block(x, 256)
    x = resnet_block(x, 256)
    x = resnet_block(x, 256)
    x = resnet_block(x, 256)
    x = resnet_block(x, 256)

    x = resnet_block(x, 512, stride=2)
    x = resnet_block(x, 512)
    x = resnet_block(x, 512)

    # Final layers
    x = AveragePooling2D(pool_size=2)(x)
    x = Flatten()(x)
    outputs = Dense(1000, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Create the model
model = resnet50()
model.summary()
```




``` resnet 34
def resnet_block(inputs, filters, strides=1):
    # Convolutional block
    x = Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Convolutional block
    x = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    # Shortcut connection
    if strides > 1:
        inputs = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(inputs)
    x = Add()([x, inputs])
    x = Activation('relu')(x)
    return x

def ResNet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, kernel_size=(7, 7), strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    x = resnet_block(x, filters=64)
    x = resnet_block(x, filters=64)
    x = resnet_block(x, filters=64)

    x = resnet_block(x, filters=128, strides=2)
    x = resnet_block(x, filters=128)
    x = resnet_block(x, filters=128)
    x = resnet_block(x, filters=128)

    x = resnet_block(x, filters=256, strides=2)
    x = resnet_block(x, filters=256)
    x = resnet_block(x, filters=256)
    x = resnet_block(x, filters=256)
    x = resnet_block(x, filters=256)
    x = resnet_block(x, filters=256)

    x = resnet_block(x, filters=512, strides=2)
    x = resnet_block(x, filters=512)
    x = resnet_block(x, filters=512)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
```




# Steen작업
기간: 빠른시일내
사다리타기 기능 구현

# AI경진대회 PPT수정, 케글 작성
6/12일 월요일까지

# 메디치 시험
6/13

# 2종소형면허
6/14

# 졸업작품진행도발표
6/15

# 인공지능AI 마무리 글 작성, 영상 제작
미정

# Unity 천체관측 기술 블로그 작성
미정

# 졸작보고서작성
6/25일

# 토익시험
6/25일

# 암호분석경진대회
기간:  2023/08/31(목)
https://cryptocontest.kr/

# HTP심리진단 앱 만들기
미정

# 치아모형생성AI만들기
미정

<br>
<br>

------------

# 끝난 일정

<br>
<br>
# 인공지능 1차 발표자료
https://docs.google.com/presentation/d/1jIv9hyrzdZgW7ljEf-TimC9mCWrsBO9lImue-kz4m6s/edit?usp=sharing

# 졸작진척도발표
 https://docs.google.com/presentation/d/1BDdHf1CuZSNxVJaeaejFFSeaQdHEeOdmymcN07mIbrg/edit?usp=sharing
 
# 중국경제지리 기말 메모
 
1. 중국 경제의 주요 이슈에 대해 최소 8개 사항을 제시하고 이를 구체적으로 논하라
경제 연착륙
위안화 절상
시장개방 확대
로컬기업들의 약진
젊은 리더의 부상
경제,사회 불균형
글로벌 영향력 확대

2. 중국이 의욕적으로 추진하고 있는 일대일로 전략의 내용 및 특징에 대해 구체적으로 논하라
일대일로(하나의 띠, 하나의 길) Belt and Road Initiative, 경제 교류벨트, 시진핑 주석 2013년 제안, 60개국 150년
실크로드를 따라 육로, 말레이시아,인도, 수에즈 운하를 이으며 해로, 도로,철도,해로 등 교통 인프라를 직접 투자
가장 큰 대상국 이라크
항만을 이어서 전략적 거점을 만든다는 진주목걸이 전략
전쟁같은 상황 속에서, 중동에서 안정적으로 석유 공급망을 확보하기 위한 전략
전체적인 이어진 국가가 미국과 완만하지 않은 관계나 제3세계
인도는 반대, 계속 견제중
견제하는 집단 자유롭고 열린 인도-태평양(Free and Open Indo Pacifc,foip)
중국이 차관을 빌려줘 인프라를 지으면, 지을때 중국 기업이 수주하고, 자국 노동자, 원자재를 사용. 해당국가는 차관만 남음
프로젝트 이후에도 유지보수를 위해 중국 기업을 이용



3. 중국 경제가 당면한 문제의 원인은 무엇이고 그에 대한 해법으로 제시된 정책은 어떠한 것이 있는지 구체적으로 논하라


5. 중국 경제에서 3농 문제는 무엇이며 그것이 중요한 이유와 원인, 현황 정책에 대해 구체적으로 논하라
3농: 농촌,농민,농업 균부론 선부론
산업화로 GDP의 농업 비중은 줄어듬
농업: 생존에서 그치는 것이 아닌 고부가가치창출
농촌: 빈곤탈피를 위해 국가,지방,기업이 힘을 합침
농민: 농촌 시장을 개선하여 농산품 보호제도 수립
흐루쇼프의 처녀지 개간 운동
문화대혁명 상산하향, 산을 오르고 고향으로 돌아가는 운동으로, 마오쩌둥이 귀향을 시킴


5. 중국의 지역개발 정책 중 '경진기 개발정책'의 배경과 내용 및 성과 그리고 그 특징에 대해서 구체적으로 논하라
경진기(베이징/톈진/허베이) 
베이징-정치, 문화, 국제교류 중심지, 과학기술혁신 중심지, 현대서비스업, 문화콘텐츠산업, 첨단기술, 연구개발 등이 발달한 국제 대도시로 
건설
톈진-국제 해상운송, 국제항구
허베이성-녹색산업 및 첨단산업, 중화학공업과 장비제조업, 신재생에너지, 전자정보산업, 전략적 신흥산업 및 일반제조업 육성

6. 중국위협론이 나타난 배경과 그 니용 및 변화와 그 특징에 대해 구체적으로 논하라
황화론, 19세기 말 20세기 초 일본,중국의 황인종에게 정복당할지도 모른다는 유럽인의 위기론

7. 2023년 양회(전국인민대표대회와 중국인민정치협상회의)
작년: 중앙에 집중된 통일 영도와 사회주의 현대화 국가 건설을 위해 재정과 화폐 정책의 효율성 제고, 과학기술의 자립자강 및 산업정책과 안보와의 연계, 금융리스크 예방
베이징과 톈징을 제외하고 경제성장목표치 5%, 굉장히 낮게 잡음
시진핑 3연임 확정된 후에 열렸기 때문에, 긴장감 없음
시진핑은 사상통일 강조
중국의 꿈, 샤오캉 이미 달성, 

8. 위안화 국제화

# 메디치수업 3분발표
https://docs.google.com/presentation/d/1HqNft6kqN8as9YGrDNU1pw_sJTxIRIENgCoRnQ-cnE8/edit?usp=sharing
