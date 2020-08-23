import os, random
import cv2, argparse
import numpy as np

def image_augmentation(img,type2 = False):
    # perspective
    w, h, _ = img.shape
    pts1 = np.float32([[0, 0], [0, w], [h, 0], [h, w]])
    # 좌표의 이동점
    begin, end = 30, 90
    pts2 = np.float32([[random.randint(begin, end), random.randint(begin, end)],
                       [random.randint(begin, end), w -
                                       random.randint(begin, end)],
                       [h - random.randint(begin, end),
                                           random.randint(begin, end)],
                       [h - random.randint(begin, end), w - random.randint(begin, end)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    img = cv2.warpPerspective(img, M, (h, w))

    # Brightness
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.array(img, dtype=np.float64)
    random_bright = .4 + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * random_bright
    img[:, :, 2][img[:, :, 2] > 255] = 255
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    # Blur
    blur_value = random.randint(0, 4) * 2 + 1
    img = cv2.blur(img, (blur_value, blur_value))
    if type2:
        return img[130:280, 180:600, :]
    return img[130:280, 120:660, :]

class ImageGenerator:
    def __init__(self, save_path):
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Plate
        self.new_plate1 = cv2.imread("new_plate1.png")
        self.new_plate2 = cv2.imread("new_plate2.png")
        self.new_plate3 = cv2.imread("new_plate3.png")
        self.new_plate4 = cv2.imread("new_plate4.png")
        self.new_plate5 = cv2.imread("new_plate5.png")
        self.new_plate6 = cv2.imread("new_plate6.png")
        self.new_plate7 = cv2.imread("new_plate7.png")
        self.new_plate8 = cv2.imread("new_plate8.png")

        # loading Number black number
        file_path = "./num/"
        file_list = os.listdir(file_path)
        self.Number = list()
        self.number_list = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Number.append(img)
            self.number_list.append(file[0:-4])

        #load reverse Number white number (위와 동일하지만 색 역전 시켰음. )
        file_path = "./num_r/"
        file_list = os.listdir(file_path)
        self.Number_r = list()
        self.number_list_r = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            img_r = cv2.bitwise_not(img) # bitwise 를 이용한 색 역전 
            self.Number_r.append(img_r)
            self.number_list_r.append(file[0:-4])

        
        # loading Char 흰색 바탕 까만 글씨 
        file_path = "./char1/" 
        file_list = os.listdir(file_path)
        self.char_list = list()
        self.Char1 = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Char1.append(img)
            self.char_list.append(file[0:-4])

        # loading reverse char 동일한 흰색 바탕 까만 글씨, bitwise 이용 역전
        file_path = "./char1_r/"
        file_list = os.listdir(file_path)
        self.char_list_r = list()
        self.Char1_r = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            img_r = cv2.bitwise_not(img)
            self.Char1_r.append(img_r)
            self.char_list_r.append(file[0:-4])

        # loading reverse Region 지역 이름 , 까만배경에 흰색 글씨. ( 초록색 번호판 변환 시 사용 )
        file_path = "./region_r/"
        file_list = os.listdir(file_path)
        self.Region_r = list()
        self.region_list_r = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Region_r.append(img)
            self.region_list_r.append(file[0:-4])
       
        # loading Region 지역 이름, 흰색 배경에 까만 글씨 (노란색 번호판 변환 시 사용 reverse yellow 줄여서 ry )
        file_path = "./region_ry/"
        file_list = os.listdir(file_path)
        self.Region_ry = list()
        self.region_list_ry = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Region_ry.append(img)
            self.region_list_ry.append(file[0:-4])
        
    #=====================================================================================================================#

    # 일반 흰색 2자리 번호판, plate 형태 : 링컨 번호판 사진에서 뽑아옴, 왜곡되어있는 번호판.
    def Type_1(self, num, save=False):
        number = [cv2.resize(number, (56, 83)) for number in self.Number]
        char = [cv2.resize(char1, (60, 83)) for char1 in self.Char1]
        Plate = cv2.resize(self.new_plate1,(520,110))

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.new_plate1, (520,110))
            b_width, b_height = 400, 800
            random_R, random_G, random_B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            background = np.zeros((b_width, b_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (b_height, b_width),
                          (random_R, random_G, random_B), -1)


            label = "Z"
            # row -> y , col -> x
            row, col = 13, 35  # row + 83, col + 56
            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # character 3
            label += self.char_list[i%37]
            Plate[row:row + 83, col:col + 60, :] = char[i%37]
            col += (60 + 36)

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            s_width, s_height = int((400-110)/2), int((800-520)/2)
            background[s_width:110 + s_width,s_height:520 + s_height, :] = Plate
            background = image_augmentation(background)

            # 2자리 번호판 맨뒤에label 전용 X 삽입
            if save:
                cv2.imwrite(self.save_path + label + "X.jpg", Plate)
                
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

  # 일반 흰색 3자리 번호판, plate 형태 : 링컨 번호판 사진에서 뽑아옴, 왜곡되어있는 번호판.
    def Type_1_1(self, num, save=False):
        number = [cv2.resize(number, (56, 83)) for number in self.Number]
        char = [cv2.resize(char1, (60, 83)) for char1 in self.Char1]
        Plate = cv2.resize(self.new_plate1,(520+56,110))

        for i, Iter in enumerate(range(num)):
            
            Plate = cv2.resize(self.new_plate1, (520+56,110))
            b_width, b_height = 400, 800
            random_R, random_G, random_B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            background = np.zeros((b_width, b_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (b_height, b_width),
                          (random_R, random_G, random_B), -1)
            label = "Z"
            # row -> y , col -> x
            row, col = 13, 35  # row + 83, col + 56
            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 3
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # character 4
            label += self.char_list[i%37]
            Plate[row:row + 83, col:col + 60, :] = char[i%37]
            col += (60 + 36)

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 8
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56


            s_width, s_height = int((400-110)/2), int((800-520-56)/2)
            background[s_width:110 + s_width,s_height:520+56 + s_height, :] = Plate
            background = image_augmentation(background)

            # 3자리 번호판은 X 안넣어도 됨.
            if save:
                cv2.imwrite(self.save_path + label + ".jpg", Plate)
                
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    

    # 흰색 2자리 번호판 ,현대, 실제 이미지 
    def Type_2(self, num, save=False):
        number = [cv2.resize(number, (56, 83)) for number in self.Number]
        char = [cv2.resize(char1, (60, 83)) for char1 in self.Char1]
        Plate =  Plate = cv2.resize(self.new_plate2,(520,130))

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.new_plate2, (520,130))
            b_width, b_height = 400, 800
            random_R, random_G, random_B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            background = np.zeros((b_width, b_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (b_height, b_width),(random_R, random_G, random_B), -1)
            label = "Z"
            # row -> y 세로 길이 , col -> x 가로 이동 축 
            row, col = 20, 45  # row + 83, col + 56
            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # character 3
            label += self.char_list[i%37]
            Plate[row:row + 83, col:col + 60, :] = char[i%37]
            col += (60 + 36)

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            s_width, s_height = int((400-130)/2), int((800-520)/2)
            background[s_width:130 + s_width,s_height:520 + s_height, :] = Plate
            background = image_augmentation(background)

            # 2자리 번호판 맨뒤에 X 삽입
            if save:
                cv2.imwrite(self.save_path + label + "X.jpg", Plate)
                
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # 흰색 3자리 번호판 ,현대, 실제 이미지 
    def Type_2_1(self, num, save=False):
        number = [cv2.resize(number, (56, 83)) for number in self.Number]
        char = [cv2.resize(char1, (60, 83)) for char1 in self.Char1]
        Plate =  Plate = cv2.resize(self.new_plate2,(520+56,130))

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.new_plate2, (520+56,130))
            b_width, b_height = 400, 800
            random_R, random_G, random_B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            background = np.zeros((b_width, b_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (b_height, b_width),(random_R, random_G, random_B), -1)
            label = "Z"
            # row -> y , col -> x
            row, col = 20, 45  # row + 80, col + 56
            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 3
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # character 3
            label += self.char_list[i%37]
            Plate[row:row + 83, col:col + 60, :] = char[i%37]
            col += (60 + 36)

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            s_width, s_height = int((400-130)/2), int((800-520-56)/2)
            background[s_width:130 + s_width,s_height:520+56 + s_height, :] = Plate
            background = image_augmentation(background)

            # 3자리 번호판 맨뒤에 X 불필요
            if save:
                cv2.imwrite(self.save_path + label + ".jpg", Plate)
                
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    
    # 흰색 2자리 번호판, 현대, 정면 이미지 
    def Type_3(self, num, save=False):
        number = [cv2.resize(number, (56, 83)) for number in self.Number]
        char = [cv2.resize(char1, (60, 83)) for char1 in self.Char1]
        Plate =  Plate = cv2.resize(self.new_plate3,(520,130))

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.new_plate3, (520,130))
            b_width, b_height = 400, 800
            random_R, random_G, random_B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            background = np.zeros((b_width, b_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (b_height, b_width),(random_R, random_G, random_B), -1)

            label = "Z"
            # row -> y , col -> x
            row, col = 20, 45  # row + 83, col + 56
            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # character 3
            label += self.char_list[i%37]
            Plate[row:row + 83, col:col + 60, :] = char[i%37]
            col += (60 + 36)

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            s_width, s_height = int((400-130)/2), int((800-520)/2)
            background[s_width:130 + s_width,s_height:520 + s_height, :] = Plate
            background = image_augmentation(background)

            # 2자리 번호판 맨뒤에 X 삽입
            if save:
                cv2.imwrite(self.save_path + label + "X.jpg", Plate)
                
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # 흰색 3자리 번호판, 현대, 정면 이미지 
    def Type_3_1(self, num, save=False):
        number = [cv2.resize(number, (56, 83)) for number in self.Number]
        char = [cv2.resize(char1, (60, 83)) for char1 in self.Char1]
        Plate =  Plate = cv2.resize(self.new_plate3,(520+56,130))

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.new_plate3, (520+56,130))
            b_width, b_height = 400, 800
            random_R, random_G, random_B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            background = np.zeros((b_width, b_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (b_height, b_width),(random_R, random_G, random_B), -1)

            label = "Z"
            # row -> y , col -> x
            row, col = 20, 45  # row + 83, col + 56
            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 3
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # character 4
            label += self.char_list[i%37]
            Plate[row:row + 83, col:col + 60, :] = char[i%37]
            col += (60 + 36)

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 8
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            s_width, s_height = int((400-130)/2), int((800-520-56)/2)
            background[s_width:130 + s_width,s_height:520+56 + s_height, :] = Plate
            background = image_augmentation(background)

            # 3자리 번호판 맨뒤에 X 불필요
            if save:
                cv2.imwrite(self.save_path + label + ".jpg", Plate)
                
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    
    # 흰색 2자리 번호판 (기아)
    def Type_4(self, num, save=False):
        number = [cv2.resize(number, (56, 83)) for number in self.Number]
        char = [cv2.resize(char1, (60, 83)) for char1 in self.Char1]
        Plate =  Plate = cv2.resize(self.new_plate4,(520,130))

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.new_plate4, (520,130))
            b_width, b_height = 400, 800
            random_R, random_G, random_B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            background = np.zeros((b_width, b_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (b_height, b_width),(random_R, random_G, random_B), -1)

            label = "Z"
            # row -> y , col -> x
            row, col = 20, 40  # row + 83, col + 56
            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # character 3
            label += self.char_list[i%37]
            Plate[row:row + 83, col:col + 60, :] = char[i%37]
            col += (60 + 36)

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            s_width, s_height = int((400-130)/2), int((800-520)/2)
            background[s_width:130 + s_width,s_height:520 + s_height, :] = Plate
            background = image_augmentation(background)

            # 2자리 번호판 맨뒤에 X 삽입
            if save:
                #cv2.imwrite(self.save_path + label + "X.jpg", Plate)
                cv2.imwrite(self.save_path + label + "X.jpg", Plate)
                
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

   # 흰색 3자리 번호판 (기아)
    def Type_4_1(self, num, save=False):
        number = [cv2.resize(number, (56, 83)) for number in self.Number]
        char = [cv2.resize(char1, (60, 83)) for char1 in self.Char1]
        Plate =  Plate = cv2.resize(self.new_plate4,(520+56,130))

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.new_plate4, (520+56,130))
            b_width, b_height = 400, 800
            random_R, random_G, random_B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            background = np.zeros((b_width, b_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (b_height, b_width),(random_R, random_G, random_B), -1)
            label = "Z"
            # row -> y , col -> x
            row, col = 20, 40  # row + 83, col + 56
            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 3
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # character 4
            label += self.char_list[i%37]
            Plate[row:row + 83, col:col + 60, :] = char[i%37]
            col += (60 + 36)

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 8
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            s_width, s_height = int((400-130)/2), int((800-520-56)/2)
            background[s_width:130 + s_width,s_height:520+56 + s_height, :] = Plate
            background = image_augmentation(background)

            # 3자리 번호판 맨뒤에 X 불필요
            if save:
                cv2.imwrite(self.save_path + label + ".jpg", Plate)
                
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # 흰색 배경 ,까만 글씨, 위 두줄 ( 트럭 형태 번호판 )
    def Type_5(self, num, save=False):
        # 38,83 / 40, 70 -> 
        number = [cv2.resize(number, (35, 78)) for number in self.Number]
        char = [cv2.resize(char1, (37, 65)) for char1 in self.Char1]
        Plate = cv2.resize(self.new_plate5, (355, 155))

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.new_plate5, (355, 155))
            b_width, b_height = 400, 800
            random_R, random_G, random_B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            background = np.zeros((b_width, b_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (b_height, b_width),(random_R, random_G, random_B), -1)
            label = "Z"
            row, col = 46, 30  # row + 83, col + 56

            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 78, col:col + 35, :] = number[rand_int]
            col += 40

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 78, col:col + 35, :] = number[rand_int]
            col += 40

            # character 3
            label += self.char_list[i%37]
            Plate[row + 12:row + 77, col + 2:col + 37 + 2, :] = char[i%37]
            col += 42 + 2

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 78, col + 2:col + 35 + 2, :] = number[rand_int]
            col += 40 + 2

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 78, col:col + 35, :] = number[rand_int]
            col += 40

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 78, col:col + 35, :] = number[rand_int]
            col += 40

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 78, col:col + 35, :] = number[rand_int]
            col += 40

            s_width, s_height = int((400-155)/2), int((800-355)/2)
            background[s_width:155 + s_width,s_height:355 + s_height, :] = Plate
            background = image_augmentation(background)


            if save:
                cv2.imwrite(self.save_path + label + "X.jpg", Plate)
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    
    #초록색 배경 -> 까만색 배경에 흰색 숫자, 지역이름과 번호까지 있음, 옛날 번호판 1.
    def Type_6(self, num, save=False):
        '''
        number1 = [cv2.resize(number, (44, 60)) for number in self.Number_r]
        number2 = [cv2.resize(number, (64, 90)) for number in self.Number_r]
        region = [cv2.resize(region, (88, 60)) for region in self.Region_r]
        char = [cv2.resize(char1, (64, 66)) for char1 in self.Char1_r]
        '''
        number1 = [cv2.resize(number, (40, 60)) for number in self.Number_r]
        number2 = [cv2.resize(number, (70, 95)) for number in self.Number_r]
        region = [cv2.resize(region, (84, 60)) for region in self.Region_r]
        char = [cv2.resize(char1, (70, 70)) for char1 in self.Char1_r]
        for i, Iter in enumerate(range(num)):
            #  336, 170 기존. 
            Plate = cv2.resize(self.new_plate6, (440, 240))
            b_width, b_height = 400, 800
            random_R, random_G, random_B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            background = np.zeros((b_width, b_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (b_height, b_width),(random_R, random_G, random_B), -1)
            label = str()
            # row -> y , col -> x
            row, col = 27, 130

            # region
            label += self.region_list_r[i % 16]
            Plate[row:row + 60, col:col + 84, :] = region[i % 16]
            col += 88 

            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list_r[rand_int]
            Plate[row:row + 60, col:col + 40, :] = number1[rand_int]
            col += 44

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 60, col:col + 40, :] = number1[rand_int]

            row, col = 90, 30

            # character 3
            label += self.char_list[i % 37]
            Plate[row:row + 70, col:col + 70, :] = char[i % 37]
            col += 70

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 95, col:col + 70, :] = number2[rand_int]
            col += 70

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 95, col:col + 70, :] = number2[rand_int]
            col += 70

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 95, col:col + 70, :] = number2[rand_int]
            col += 70

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 95, col:col + 70, :] = number2[rand_int]

            s_width, s_height = int((400-240)/2), int((800-440)/2)
            background[s_width:240 + s_width,s_height:440 + s_height, :] = Plate
            background = image_augmentation(background)

            if save:
                cv2.imwrite(self.save_path + label + "X.jpg", Plate)
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def Type_7(self, num, save=False):
        '''
        number1 = [cv2.resize(number, (60, 65)) for number in self.Number_r]
        number2 = [cv2.resize(number, (80, 90)) for number in self.Number_r]
        char = [cv2.resize(char1, (60, 65)) for char1 in self.Char1_r]

        '''
        number1 = [cv2.resize(number, (55, 60)) for number in self.Number_r]
        number2 = [cv2.resize(number, (80, 90)) for number in self.Number_r]
        char = [cv2.resize(char1, (60, 65)) for char1 in self.Char1_r]

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.new_plate7, (420, 210))
            b_width, b_height = 400, 800
            random_R, random_G, random_B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            background = np.zeros((b_width, b_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (b_height, b_width),(random_R, random_G, random_B), -1)
            random_width, random_height = 420, 210
            label = "Z"

            # row -> y , col -> x
            row, col = 25, 115

            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list_r[rand_int]
            Plate[row:row + 60, col:col + 55, :] = number1[rand_int]
            col += 60

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list_r[rand_int]
            Plate[row:row + 60, col:col + 55, :] = number1[rand_int]
            col += 60

            # character 3
            label += self.char_list_r[i%37]
            Plate[row:row + 65, col:col + 60, :] = char[i%37]
            row, col = 85, 40

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list_r[rand_int]
            Plate[row:row + 90, col:col + 80, :] = number2[rand_int]
            col += 80


            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list_r[rand_int]
            Plate[row:row + 90, col:col + 80, :] = number2[rand_int]
            col += 80

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list_r[rand_int]
            Plate[row:row + 90, col:col + 80, :] = number2[rand_int]
            col += 80

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list_r[rand_int]
            Plate[row:row + 90, col:col + 80, :] = number2[rand_int]


            s_width, s_height = int((400-210)/2), int((800-420)/2)
            background[s_width:210 + s_width,s_height:420 + s_height, :] = Plate
            background = image_augmentation(background)

            if save:
                cv2.imwrite(self.save_path + label + "X.jpg", Plate)
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

# 전기차 번호판 만들기
    # 2자리 Type_8 전기차
    def Type_8(self, num, save=False):
        number = [cv2.resize(number, (50, 83)) for number in self.Number]
        char = [cv2.resize(char1, (60, 83)) for char1 in self.Char1]
        Plate =  Plate = cv2.resize(self.new_plate8,(560,140))

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.new_plate8, (560,140))
            b_width, b_height = 400, 800
            random_R, random_G, random_B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            background = np.zeros((b_width, b_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (b_height, b_width),(random_R, random_G, random_B), -1)

            label = "Z"
            # row -> y , col -> x
            row, col = 20, 75  # row + 83, col + 56
            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 50, :] = number[rand_int]
            col += 56

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 50, :] = number[rand_int]
            col += 56

            # character 3
            label += self.char_list[i%37]
            Plate[row:row + 83, col:col + 60, :] = char[i%37]
            col += (60 + 36)

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 50, :] = number[rand_int]
            col += 56

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 50, :] = number[rand_int]
            col += 56

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 50, :] = number[rand_int]
            col += 56

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 50, :] = number[rand_int]
            col += 56

            s_width, s_height = int((400-140)/2), int((800-560)/2)
            background[s_width:140 + s_width,s_height:560 + s_height, :] = Plate
            background = image_augmentation(background)

            # 2자리 번호판 맨뒤에 X 삽입
            if save:
                #cv2.imwrite(self.save_path + label + "X.jpg", Plate)
                cv2.imwrite(self.save_path + label + "X.jpg", Plate)
                
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # 3자리 Type_8_1 전기차
    def Type_8_1(self, num, save=False):
        number = [cv2.resize(number, (50, 83)) for number in self.Number]
        char = [cv2.resize(char1, (60, 83)) for char1 in self.Char1]
        Plate =  Plate = cv2.resize(self.new_plate8,(560+56,140))

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.new_plate8, (560+56,140))
            b_width, b_height = 400, 800
            random_R, random_G, random_B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            background = np.zeros((b_width, b_height, 3), np.uint8)
            cv2.rectangle(background, (0, 0), (b_height, b_width),(random_R, random_G, random_B), -1)

            label = "Z"
            # row -> y , col -> x
            row, col = 20, 75  # row + 83, col + 56
            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 50, :] = number[rand_int]
            col += 56

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 50, :] = number[rand_int]
            col += 56

            # number 3
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 50, :] = number[rand_int]
            col += 56

            # character 4
            label += self.char_list[i%37]
            Plate[row:row + 83, col:col + 60, :] = char[i%37]
            col += (60 + 36)

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 50, :] = number[rand_int]
            col += 56

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 50, :] = number[rand_int]
            col += 56

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 50, :] = number[rand_int]
            col += 56

            # number 8
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 50, :] = number[rand_int]
            col += 56

            s_width, s_height = int((400-140)/2), int((800-560-56)/2)
            background[s_width:140 + s_width,s_height:560+56 + s_height, :] = Plate
            background = image_augmentation(background)

            # 3자리 번호판 맨뒤에 X 불필요
            if save:
                cv2.imwrite(self.save_path + label + ".jpg", Plate)
                
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--img_dir", help="save image directory",
                    type=str, default="../DB/train/")
parser.add_argument("-n", "--num", help="number of image",
                    type=int)
parser.add_argument("-s", "--save", help="save or imshow",
                    type=bool, default=True)
args = parser.parse_args()


img_dir = args.img_dir
A = ImageGenerator(img_dir)

num_img = args.num
Save = args.save

A.Type_1(num_img, save=Save)
print("Type 1 finish")
A.Type_1_1(num_img, save=Save)
print("Type 1_2 finish")
A.Type_2(num_img, save=Save)
print("Type 2 finish")
A.Type_2_1(num_img, save=Save)
print("Type 2_1 finish")
A.Type_3(num_img, save=Save)
print("Type 3 finish")
A.Type_3_1(num_img, save=Save)
print("Type 3_1 finish")

A.Type_4(num_img, save=Save)
print("Type 4 finish")
A.Type_4_1(num_img, save=Save)
print("Type 4_1 finish")
A.Type_5(num_img, save=Save)
print("Type 5 finish")
A.Type_6(num_img, save=Save)
print("Type 6 finish")
A.Type_7(num_img, save=Save)
print("Type 7 finish")
A.Type_8(num_img, save=Save)
print("Type 8 finish")
A.Type_8_1(num_img, save=Save)
print("Type 8_1 finish")

# 2020.08.23 perspective 추가