# DORNPG

Dynamic Object Removing with Neighbor Patch Geometry

동적물체를 제거하기 위한 RGB-D카메라 기반의 알고리즘

## 결과 이미지 

- 왼쪽 : 원본이미지
- 중간 : groudtruth
- 오른쪽 : 마스킹 결과(빨간색은 배경, 나머지 색은 동적물체)

![image](https://user-images.githubusercontent.com/63538314/156711148-982a6ed6-092b-4a83-bfb6-7561d8af50b4.png)

## 개발일정

1. 데이터셋에서 정보를 읽어오고, 전처리
2. RGB-D 데이터에 대한 세그먼트 처리
3. DenseFlow를 활용하여 flow간에 변화 및 상호관계를 활용하여 제약조건을 기반으로 각 세그먼트 사이의 유사관계를찾음
4. 3차원 Depth합을 기반으로 배경을 찾는다. 
5. 이웃하지 않은 배경사이에서 다시 한번 결합하여 정적인요소를 합친다.




## References

https://github.com/hmorimitsu/sift-flow-gpu

[1] C. Liu; Jenny Yuen; Antonio Torralba. "SIFT Flow: Dense correspondence across scenes and its applications." IEEE Transactions on Pattern Analysis and Machine Intelligence 33.5 (2010): 978-994.