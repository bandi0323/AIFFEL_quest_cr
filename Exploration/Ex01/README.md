# AIFFEL Campus Online Code Peer Review Templete

- 코더 : 송반디
- 리뷰어 : 김혜원


# PRT(Peer Review Template)

- [ ]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - PyTorch의 deeplabv3_resnet101 모델을 활용해 고양이 이미지를 세그멘테이션하고
    - NumPy의 np.where 연산을 통해 사막 배경 이미지(sand_img)와 성공적으로 합성하여 최종 결과물(result_img)을 출력하는 전체 파이프라인이 누락 없이 구현되어 있습니다.
    - 해당 조건을 만족하는 부분을 캡쳐해 근거로 첨부
    - 

- [ ]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
      # 마스크를 원본 크기로 Resize
output_predictions_resized = cv2.resize(output_predictions, (cat_img.shape[1], cat_img.shape[0]), interpolation=cv2.INTER_NEAREST)
...
# 고양이 부분만 남기고 배경 적용
img_mask_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
result_img = np.where(img_mask_color == 255, cat_img, sand_img_resized)
    - 해당 코드 블럭을 왜 핵심적이라고 생각하는지 : 모델의 출력 마스크 크기를 원본 이미지 크기로 복원할 때 왜곡을 방지하기 위해 INTER_NEAREST를 사용한 점과 np.where를 활용해 픽셀 단위 조각 합성을 처리한 흐름이 주석 덕분에 직관적으로 잘 이해되었습니다. target_class_id = unique_classes[-1] 부분은 고양이 클래스 ID를 동적으로 가져오려는 의도로 보이나, 이미지에 다른 객체가 섞일 경우 오작동할 수 있으므로 이에 대한 짤막한 선택 이유나 설명 주석이 보강되면 더 좋을 것 같습니다.
    - doc string 또는 annotation이 작성되어 있는지 확인
    - 코드의 기능, 존재 이유, 작동 원리 등을 설명했는지 확인
    - 주석을 보고 코드 이해가 잘 되었는지 확인

- [ ]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나 새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 문제 원인 및 해결 과정을 기록했는지 확인 : 현재 제출된 코드 내에는 명시적인 디버깅 로그나 에러 발생 시의 수정 기록(Try-Catch 등), 혹은 다른 세그멘테이션 모델(예: 후속 버전인 미디어파이프 등)과의 비교 실험 기록은 포함되어 있지 않습니다. 추후 프로젝트 발전 시, 런타임에서 발생할 수 있는 채널 수 불일치 에러 등을 해결한 과정이 기술된다면 완성도가 더 높아질 것입니다.
    - 추가적으로 수행한 시도나 실험이 기록되어 있는지 확인 : 

- [ ]  **4. 회고를 잘 작성했나요?**
    - 배운 점, 아쉬운 점, 느낀 점이 기록되어 있는지 확인 : 아래 리뷰어의 회고 세션에서 코드 개선 방향과 함께 종합 의견을 정리해 두었습니다.
    - 전체 코드 실행 흐름을 이해할 수 있도록 정리했는지 확인 : 이미지 로드 / 모델 추론 / 마스크 가공 / 이미지 합성)은 코드 순서대로 깔끔하게 정리되어 있어 흐름을 파악하기 쉽습니다.

- [ ]  **5. 코드가 간결하고 효율적인가요?**
    - Python 스타일 가이드, PEP8을 준수했는지 확인 : 대안적으로 공백 분리나 변수 명명법(cat_img_path, result_img) 등 파이썬 스타일 가이드를 잘 준수하여 가독성이 높습니다.
    - 코드 중복을 줄이고 함수화 또는 모듈화를 했는지 확인 : 현재는 주피터 노트북 스타일의 순차적 스크립트 형식으로 작성되어 있습니다. 단일 실행에는 문제가 없으나, 향후 다양한 이미지에 재사용하기 위해 전처리-추론-합성 단계를 함수(def)화하여 모듈성을 높이는 것을 추천합니다.


# 회고

리뷰어의 회고를 작성합니다.
코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
import matplotlib.pyplot as plt

# 1. 이미지 로드 및 RGB 변환
cat_img = cv2.imread("./images/cat.jpeg")
sand_img = cv2.imread("./images/sand.jpeg")
cat_img = cv2.cvtColor(cat_img, cv2.COLOR_BGR2RGB)
sand_img = cv2.cvtColor(sand_img, cv2.COLOR_BGR2RGB)

# 2. 모델 및 전처리 설정 (정규화 추가로 추론 정확도 향상)
model = deeplabv3_resnet101(pretrained=True).eval()
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((520, 520)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 💡추가: 모델 요구 정규화
])

input_tensor = transform(cat_img).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)["out"][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()

# 3. 마스크 복원 (INTER_NEAREST 유지)
output_predictions_resized = cv2.resize(
    output_predictions, 
    (cat_img.shape[1], cat_img.shape[0]), 
    interpolation=cv2.INTER_NEAREST
)

# 4. 고양이 클래스(COCO ID: 15) 마스크 추출
# unique_classes[-1] 대신 안전하게 고양이 타깃 ID 지정 또는 매칭 확인
CAT_CLASS_ID = 15 
img_mask = (output_predictions_resized == CAT_CLASS_ID).astype(np.uint8) * 255

# 5. 배경 이미지 리사이즈
sand_img_resized = cv2.resize(sand_img, (cat_img.shape[1], cat_img.shape[0]))

# 6. 이미지 합성 (np.where 차원 일치를 위해 마스크에 np.newaxis 적용)
# 💡수정: 3채널 컬러 변환 대신 차원 확장을 통해 메모리를 아끼고 에러를 방지합니다.
img_mask_3d = img_mask[:, :, np.newaxis] 
result_img = np.where(img_mask_3d == 255, cat_img, sand_img_resized)

# 7. 최종 시각화
plt.figure(figsize=(6, 6))
plt.imshow(result_img)
plt.axis('off')
plt.show()

# np.where 연산 시 조건문 배열과 참/거짓일 때의 배열 크기가 완벽히 호환되어야 합니다. 개선된 코드에서는 img_mask[:, :, np.newaxis]를 사용해 (H, W, 1) 차원으로 확장함으로써, (H, W, 3) 크기인 두 이미지
#(cat_img, sand_img_resized)와 완벽하게 브로드캐스팅(Broadcasting) 합성이 에러 없이 이루어지도록 안전성을 확보했습니다.

