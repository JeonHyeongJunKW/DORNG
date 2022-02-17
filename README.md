# DRNG

Dynamic Object Removing with neighbor patch geometry
동적물체를 제거하기 위한 RGB-D카메라 기반의 알고리즘

## 개발일정

1. 데이터셋에서 정보를 읽어오고, 전처리
2. RAY를 사용하여 RGB-D 데이터에 대한 세그먼트 처리
3. DenseSIFT를 활용하여 flow간에 변화 및 상호관계를 활용하여 제약조건 만족으로 동적물체 전처리
4. 그림자나 빛에 의한 심한 변화 인지를 위한 매칭 성능기반의 평가 지표추가
5. 결과 출력 및 비교
