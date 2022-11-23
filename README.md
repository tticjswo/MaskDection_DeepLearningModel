주어진 INPUT을 활용하여 사진의 마스크 착용 여부를 판단하는 모델구현

( 학습데이터의 양이 매우 커서 업로드 하지 않았습니다.)

_________________________________________________________________________________________________

제안 배경

	 처음 생각은 코로나 시대가 굉장히 길어지면서 마스크를 다들 쓰고 다니지만

	아직도 어르신 분들이나 몇몇 분들은 정확한 마스크 착용방법을 지키지 않으십니다.

	게다가 정확하지 않은 마스크 착용은 타인에게 불쾌감을 줄 수 있다고 생각하여 

	실내 출입 시 마스크를 착용했는지, 제대로 착용했는지를 파악할 수 있는 마스크 식별 모델을 

	만들어보자 생각하게 되어 시도해보았습니다.


_________________________________________________________________________________________________


개발 환경

	◾ Operating system : Window 10 
	◾ Language : Python
	◾ Development Tools : Colab
	◾ Library : opencv, tensorflow, keras, pillow, os ....
  
  
______________________________________________________________________________________________


설계 내용

(1) Data selection & Preperation ( 데이터 선택 및 준비 )

필요 데이터 : 마스크를 쓰지 않은 인물 사진 , 마스크를 착용한 인물 사진


 - 해당 사진들을 직접 구하기 어렵다고 판단했고 직접 데이터셋을 만들어보자 판단.
 - 마스크를 쓰지 않은 인물 데이터를 Google 과 ‘AI가 만든 저작권 없는 인물사진 ‘에서 찾았다 
![image](https://user-images.githubusercontent.com/66824080/202888695-31b5570b-0cb2-4f8c-aecd-2652f809e37a.png)

-> Ai가 만든 인물사진들은 dataset, 즉 여러장을 한번에 무료로 사용할 수 없었고 상업적 이용이 아닌 경우 한장씩 다운로드가 가능했다. 즉 400장 정도를 직접 한장, 한장 다운받았다.


![image](https://user-images.githubusercontent.com/66824080/202888733-3157bd33-3081-4fee-a93a-43708c4ff8e3.png)
 -> Facial landmark 추출을 위해  shape_predictor_68_face_landmark를 이용해서  얼굴 부위별 좌표를 파악.!


마스크가 착용 되는 얼굴에서 중요 포인트를 4가지
3번 : 왼쪽  턱
8번 : 밑 턱
13번 : 오른쪽 턱
29번 : 코 중앙
이미지에 삽입될 마스크 이미지 크기 조절을 위해 위 4가지 좌표들을 사용

![image](https://user-images.githubusercontent.com/66824080/202888752-1f7a6d75-aeff-4296-ae83-7af466b35267.png)


(그림 2)와 같이 마스크 png 사진을 색깔, 모양별로 여러 장 준비.


(1) Data selection & Preperation ( 마스크 데이터 만들기(2) )

마스크 이미지를그냥 사용하여 이미지에 붙히면 얼굴에 맞게 하기 어렵다고 판단,
 마스크를 반으로 쪼개어 얼굴의 반쪽 별로 길이를 설정하여 얼굴에 맞게 마스크가 부착되도록 설정

마스크의 높이는 8번과 29번 사이의 거리에 특정 비율을 곱해서 높이를 설정, 
왼쪽의 얼굴부분은 8번과 29번을 연결하는 직선과 3번 점 사이의 거리를 이용해 마스크의 넓이를 설정하고 
오른쪽의 얼굴부분도 마찬가지로 직선과 13번 사이의 거리를 이용해 넓이를 설정.
  
반으로 쪼갠 마스크는 다시 합친 후 8번과 29번의 x, y 좌표를 이용하여 ArcTan 값을 도출하여 
인물 얼굴의 기울어짐을 각도로 구하여 합친 마스크를 회전시켜 얼굴에 부착.

![image](https://user-images.githubusercontent.com/66824080/202888777-10b78f5a-fee0-4c62-a281-f547afea0d2f.png)

 -> 결과값 예시. 완벽하진 않지만 얼굴에 기울어진 각도에 맞춰 완성된 걸 볼 수 있다.
 
 
 (2) Data Annotation ( Preprocessing )

데이터를 수집하였지만 위의 사진을 그대로 사용해서 학습하는 것보다 
얼굴부분만을 도출하여 모델에 학습시키는 것이 더 낫다고 생각하여 얼굴부분만 도려낸 이미지들을 저장하는 과정 추가

models/res10_300x300_ssd_iter_140000.caffemodel 얼굴 인식 모델을 사용하여 얼굴 부분만을 추출 성공.

![image](https://user-images.githubusercontent.com/66824080/203550538-54dbd85f-bb16-4fee-9cdd-270e429cbbd3.png)  얼굴 부분 추출과 200 x 200 px 사이즈로 이미지를 통일시켰다.


(3) Data Deeplearning( 데이터 학습 )

얼굴 이미지로부터 인물의 마스크 착용여부를 분류이라고 판단, Supervised learning의 Classification 기법으로 모델을 설계.
 
모델은 기본적인 CNN 계층으로 층을 구축하였고 다음과 같이 설계

![image](https://user-images.githubusercontent.com/66824080/203550812-0460ddbc-938c-4778-8de2-d8f50f9d1604.png)

 
Convolution Base 영역에서는 Convolution, Activation function, Maxpooling을 이용하여 네트워크를 형성
Activation function은 기본적으로 ReLU를 사용, Maxpooling은 모두 크기를 절반으로 줄이는 방식을 사용
Dropout : 0.25.
 
Classifier 계층은  Convolution Base 계층의 마지막 데이터를 1차원으로 바꿔준 후 Fully-Connected로 연결. 
이 후 마지막 출력은 인물의 마스크를 착용한 확률과 착용하지 않은 확률을 나타내기 위해서 Activation function을 Softmax로 사용 


![image](https://user-images.githubusercontent.com/66824080/203550918-9dbac170-971f-4c6e-a271-189d51dec877.png)



(4) Training

◾ Loss function : binary_crossentropy
◾ Optimizer : adam
◾ Epoch : 32 
◾ Batch : 40
![image](https://user-images.githubusercontent.com/66824080/203551041-9c756a08-cd9b-48e3-8ef4-61775bfcd48c.png)


(5) Test 및 result

![image](https://user-images.githubusercontent.com/66824080/203551068-73d048bb-eb1d-4aae-a84e-65dc86ea3986.png)

----------------------------------------------------------------------------------------------------------------

     실험결과

	◾ 시간이 부족하여 처음 목표로 했던 [mask , no-mask, not-valid mask] 3가지 클래스로 구분을 
	하지 못하였다. Not – valid mask에 대한 자료를 찾기 매우 어려웠고 정확한 예측이 안되어 결론을 
	얼른 내야하는 입장으로써 2가지 클래스로 구현하였다.

	◾ 다양한 테스트를 진행해본 결과 역시 data generation을 통해 만들어진 마스크 착용 인물 사진에는
	 높은 정확도를 보여줬지만 실제 인물 사진들에 대해는 정확도가 많이 부족했다. 
		(보고서에서 몇가지 예측 실패 예를 보이겠다. )

	◾ 약 2가지 클래스 각각 400개 정도의 사진으로 학습시켰는데 많이 부족했다는 생각이 든다.
	 마스크의 종류, 색상, 빛의 양 등 오차가 많기 때문에 데이터 수를 더 늘릴 수 있었다면 더 높은 정확도를 보여주었을 것이다.





