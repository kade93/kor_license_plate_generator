# 딥러닝을 위한 번호판 생성기 (흑백)

기존의
 https://github.com/qjadud1994/Korean-license-plate-Generator/tree/e4c7386a6e5954bd9ffaf34abb102ac49f2bdf48 의 번호판 생성기를 참고하였습니다 

* 2020.08.29 추가할 사항들

  * '하' 와 '호' 글자 생성 및 label로 적용될 수 있는 번호판 추가 예정.

  * ㅇ 과 0 의 구별,  숫자 0  6 8 9 에 대한 흐릿한 이미지 추가 및 perspective 변경 등을 이용하여 학습 데이터의 다양성을 향상할 필요가 있음.

* **2020.08.20 진행상황**

  현재 대부분의 타입의 번호판에 대해 생성이 가능합니다.

  실제 사이즈와 차이가 있을 수 있어 학습 활용시 부정확 할 수 있습니다.

  추후 augmentation 작업을 통해, 각도 조절과 배경 변경을 해보려고 합니다.

* **파워포인트를 이용하여 번호판 plate 를 생성하여 이미지로 저장하였습니다.**



- **저장된 이미지를 모두 흑백 또는 흑백에서 다시 역전 시켜 코드에서 사용하였습니다.**

<img src="https://user-images.githubusercontent.com/58680436/90756872-d4132980-e317-11ea-9799-2b57256580e1.PNG" alt="슬라이드1" style="zoom:33%;" /><img src="https://user-images.githubusercontent.com/58680436/90756875-d5445680-e317-11ea-9f7c-c2901a96f5dd.PNG" alt="슬라이드2" style="zoom:33%;" />
<img src="https://user-images.githubusercontent.com/58680436/90756878-d5445680-e317-11ea-90ed-4bd953e28bc8.PNG" alt="슬라이드3" style="zoom:33%;" /><img src="https://user-images.githubusercontent.com/58680436/90756884-d6758380-e317-11ea-8cb7-d1b7af50b9ee.PNG" alt="슬라이드5" style="zoom:33%;" />
<img src="https://user-images.githubusercontent.com/58680436/90756881-d5dced00-e317-11ea-8307-eebdcfd1d7e6.PNG" alt="슬라이드4" style="zoom:33%;" /><img src="https://user-images.githubusercontent.com/58680436/90756885-d6758380-e317-11ea-9958-b18e900c4cc4.PNG" alt="슬라이드6" style="zoom:33%;" />



### 코드로 생성된 샘플 이미지들 

* 밝기조절이 랜덤으로 되어있습니다. 

<img src="https://user-images.githubusercontent.com/58680436/90756820-c8276780-e317-11ea-8a67-aacff2e88d0a.jpg" alt="Z428ej8595" style="zoom:50%;" /> <img src="https://user-images.githubusercontent.com/58680436/90756827-c8bffe00-e317-11ea-9a53-6b0d4e2d4ed1.jpg" alt="Z512dn5296" style="zoom:50%;" />

<img src="https://user-images.githubusercontent.com/58680436/90756830-c9589480-e317-11ea-9443-3650eaef0169.jpg" alt="Z39an4372X" style="zoom:50%;" /> <img src="https://user-images.githubusercontent.com/58680436/90756831-c9589480-e317-11ea-9d1c-c0507d9fa0e3.jpg" alt="Z04ak6042X" style="zoom:50%;" />

<img src="https://user-images.githubusercontent.com/58680436/90756832-c9f12b00-e317-11ea-848c-803d7d632f17.jpg" alt="Z29eh2144X" style="zoom: 33%;" /> <img src="https://user-images.githubusercontent.com/58680436/90756833-c9f12b00-e317-11ea-9986-b6ab5fe35497.jpg" alt="Z93ak0431X" style="zoom: 33%;" /> <img src="https://user-images.githubusercontent.com/58680436/90756835-ca89c180-e317-11ea-8716-e1b18ed42d0a.jpg" alt="G29dk5454X" style="zoom: 33%;" />


<img src="https://user-images.githubusercontent.com/58680436/90756837-ca89c180-e317-11ea-8223-a668c44fc1ce.jpg" alt="Z439dn4863" style="zoom: 50%;" /> <img src="https://user-images.githubusercontent.com/58680436/90756839-cb225800-e317-11ea-98cf-1c85377dd664.jpg" alt="Z912an3351" style="zoom: 50%;" />

<img src="https://user-images.githubusercontent.com/58680436/90756841-cbbaee80-e317-11ea-9055-a531ec88e497.jpg" alt="Z739ah1425" style="zoom: 50%;" /> <img src="https://user-images.githubusercontent.com/58680436/90756845-cbbaee80-e317-11ea-8920-337fbc7a26aa.jpg" alt="Z44an1643X" style="zoom: 50%;" />

<img src="https://user-images.githubusercontent.com/58680436/90756846-cc538500-e317-11ea-98f0-7be1654a8f9f.jpg" alt="Z854dn6989" style="zoom:50%;" /> <img src="https://user-images.githubusercontent.com/58680436/90756848-cc538500-e317-11ea-9a13-53bbb3a3d97b.jpg" alt="Z59ah7204X" style="zoom:50%;" />

<img src="https://user-images.githubusercontent.com/58680436/90756849-ccec1b80-e317-11ea-87c7-5326867daa5e.jpg" alt="Z55dh5345X" style="zoom:50%;" /> <img src="https://user-images.githubusercontent.com/58680436/90756850-ccec1b80-e317-11ea-9765-bbccbe30e8d5.jpg" alt="Z081dn8816" style="zoom:50%;" />

<img src="https://user-images.githubusercontent.com/58680436/90756853-cd84b200-e317-11ea-940e-aa2abc18dba3.jpg" alt="Z97eh3984X" style="zoom:50%;" /><img src="https://user-images.githubusercontent.com/58680436/90756858-ce1d4880-e317-11ea-9dca-1033790f2810.jpg" alt="Z54aj7307X" style="zoom:50%;" />



### 코드 사용법

1. **깃헙에서 코드 pull**
2. **vs code 를 사용하던, python 컴파일러 기능이 있는 IDE를 실행**
3. **실행하려는 파일 (Generator_original_customby_ydh.py) 의 경로와 터미널 경로를 일치 확인**
4. **하단 처럼 py 실행 ( -n 뒤에 원하는 이미지 갯수 , -i 뒤에는 저장할 경로 )**

```python
python Generator_original_customby_ydh.py -n 10 -i "./test_generate/"
```

![이미지 189](https://user-images.githubusercontent.com/58680436/90756981-fc028d00-e317-11ea-89b5-6474b515aaf9.png)



---

### Labeling 관련

**파일명에 알파뱃 대문자 한글자와, 소문자가 이루어져 있습니다.**

* 대문자 알파벳

  Z = 지역명 없음 

  A = 서울 B = 경기 C = 인천 D = 강원 E = 충남, F = 대전 G = 충북 H = 부산
  I = 울산  J  =대구 K = 경북 L = 경남 M = 전남 N = 광주 O = 전북 P = 제주

  X = 세자리 번호판이 생김에 따라, 최대 글자수를 10으로 늘려주었고, 학습상에 편의를 위해 
  2자리 번호판에 한하여 맨 끝에 X를 붙여줍니다.
  딥러닝 학습 prediction 시 label 데이터에서 X는 제거합니다. 

* 소문자 알파벳

  rk = 가 , sk = 나, ek = 다... 키보드 자판을 그대로 반영합니다. 