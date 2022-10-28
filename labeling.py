import sys
from cv2 import destroyWindow
import numpy as np
import cv2
import json
from collections import OrderedDict
from argparse import ArgumentParser as ArgParse
import copy
import glob
import os
import ctypes

green = (0, 255, 0)  # 이미지 내에 선이나 글자의 색상 : 녹색
red = (0, 0, 255)  # 이미지 내에 선이나 글자의 색상 : 빨강
blue = (255, 0, 0)  # 이미지 내에 선이나 글자의 색상 : 파랑
yellow = (0, 255, 255)  # 이미지 내에 선이나 글자의 색상 : 노랑
sky_blue = (255, 255, 0)  # 이미지 내에 선이나 글자의 색상 : 하늘
purple = (255, 0, 139)  # 이미지 내에 선이나 글자의 색상 : 자주
magenta = (255, 0, 255)  # 이미지 내에 선이나 글자의 색상 : 핑크

NEXT_PAGE = 32  # space key ascii code
h_samples = list(range(320, 720, 10))
print(h_samples)
json_file_path = './train_cart.json'  # train_cart.json 파일 경로
#train.txt 기준으로 train 이미지 불러오기
# f = open("./train.txt", 'r')  # train image의 경로가 적힌 train.txt 파일 로드
# lines = f.readlines()  # train.txt를 라인별로 읽음
# file_data = OrderedDict()  # json파일 저장을 위한 file_data 선언
#train.txt 기준으로 train 이미지 불러오기

#clips 폴더 기준으로 train 이미지 불러오기(경윤)
jpg_images = glob.glob('./clips/*.jpg') #원본 파일
png_images = glob.glob('./clips/*.png') #labelling 된 파일
#clips 폴더 기준으로 train 이미지 불러오기(경윤)

#글자 출력에 관한 global variable
font = cv2.FONT_HERSHEY_SIMPLEX  # 폰트 종류
org = (10, 60)  # 폰트를 찍는 위치
file_count = 0  # labeling하는 파일 차례(순서)

#lane에 대한 global variable
lane_class = 1
left_lane = 0  # left lain은 0으로 표시함 (내가 지정했음)
right_lane = 1  # right lain은 1로 표시함 (내가 지정했음)
lane_count = 0  # lane의 점을 몇개 찍었는지 count

#이전의 라벨링 불러올때 쓰는 global variable
l_index = -1  # left line mouse point index value
r_index = -1  # right line mouse point index value
pre_label = False  # 이전의 label 불러왔는지 확인 변수
more_point = False # 7개의 포인터를 찍게 하기위한 변수

class MyLane:
    def __init__(self):
        self.points = []

    def points_append(self, x1, y1):
        self.points.append((x1, y1))

    def point_change(self, index, val):
        self.points[index] = val


def calc_inclination(lane_coordi, lane, param, h_anchor):
    """
    x1 : lane_coordi.points[0][0]
    y1 : lane_coordi.points[0][1]
    x2 : lane_coordi.points[1][0]
    y2 : lane_coordi.points[1][1]
    """
    global pre_label, more_point

    m = []  # inclination
    b = []  # y-intercept
    lane_x_axis = []

    for i in range(0, len(lane_coordi.points)-1):
        if (lane_coordi.points[i+1][0]-lane_coordi.points[i][0]) == 0:
            m.append(0)
        else:
            m.append((lane_coordi.points[i+1][1]-lane_coordi.points[i][1])/(
                lane_coordi.points[i+1][0]-lane_coordi.points[i][0]))

        b.append(lane_coordi.points[i][1]-(m[i]*lane_coordi.points[i][0]))

    count_index = 0  # count index는 몇번째 anchor 인지 확인하는 flag

    # print('file_data["h_samples"] : ' + str(h_anchor))
    # 직선의 방정식에 따른 x좌표값 추출하는 식 lane_x_axis = [int((num-b)/m) for num in h_samples]
    # 기울기 0일 경우 바로 click point의 x좌표로 설정
    if len(lane_coordi.points) != 0:
        if more_point == False:
            for num in h_samples:
                if num < lane_coordi.points[3][1]:
                    lane_x_axis.append(-2)

                elif num < lane_coordi.points[2][1]:
                    if m[2] == 0:
                        lane_x_axis.append(lane_coordi.points[2][0])
                    else:
                        lane_x_axis.append(int((num-b[2])/m[2]))

                elif num < lane_coordi.points[1][1]:
                    if m[1] == 0:
                        lane_x_axis.append(lane_coordi.points[1][0])
                    else:
                        lane_x_axis.append(int((num-b[1])/m[1]))

                elif num < lane_coordi.points[0][1]:
                    if m[0] == 0:
                        lane_x_axis.append(lane_coordi.points[0][0])
                    else:
                        lane_x_axis.append(int((num-b[0])/m[0]))

                else:
                    lane_x_axis.append(-2)
        else:
            for num in h_samples:
                if num < lane_coordi.points[6][1]:
                    lane_x_axis.append(-2)

                elif num < lane_coordi.points[5][1]:
                    if m[5] == 0:
                        lane_x_axis.append(lane_coordi.points[5][0])
                    else:
                        lane_x_axis.append(int((num-b[5])/m[5]))

                elif num < lane_coordi.points[4][1]:
                    if m[4] == 0:
                        lane_x_axis.append(lane_coordi.points[4][0])
                    else:
                        lane_x_axis.append(int((num-b[4])/m[4]))

                elif num < lane_coordi.points[3][1]:
                    if m[3] == 0:
                        lane_x_axis.append(lane_coordi.points[3][0])
                    else:
                        lane_x_axis.append(int((num-b[3])/m[3]))

                elif num < lane_coordi.points[2][1]:
                    if m[2] == 0:
                        lane_x_axis.append(lane_coordi.points[2][0])
                    else:
                        lane_x_axis.append(int((num-b[2])/m[2]))

                elif num < lane_coordi.points[1][1]:
                    if m[1] == 0:
                        lane_x_axis.append(lane_coordi.points[1][0])
                    else:
                        lane_x_axis.append(int((num-b[1])/m[1]))

                elif num < lane_coordi.points[0][1]:
                    if m[0] == 0:
                        lane_x_axis.append(lane_coordi.points[0][0])
                    else:
                        lane_x_axis.append(int((num-b[0])/m[0]))

                else:
                    lane_x_axis.append(-2)

    print('lane_x_axis : ' + str(lane_x_axis))
    h, w, _ = param.shape  # 이미지 사이즈의 h, w

    for i in h_samples:
        # 추출한 x의 좌표가 영상 안에 존재하거나, y축의 값이 anchor 값 사이일경우만 입력하고 아니면 -2 입력할것
        if len(file_data['lanes'][lane]) >= len(h_samples):
            del file_data['lanes'][lane][:]

        if lane_x_axis[count_index] >= 0 and lane_x_axis[count_index] <= w:
            file_data["lanes"][lane].append(lane_x_axis[count_index])
        else:
            file_data["lanes"][lane].append(-2)
        count_index += 1
    print('lanes : ', end='')
    print(file_data["lanes"])
    x_axis_index = 0

    #사용자가 노란색선안에 점들이 제대로 찍혔는지 확인하기 위함
    for i in h_samples:
        if file_data["lanes"][lane][x_axis_index] != -2:
            if pre_label == True:
                cv2.circle(param, (file_data["lanes"]
                                   [lane][x_axis_index], i), 5, purple, -1)
            else:
                cv2.circle(param, (file_data["lanes"]
                                   [lane][x_axis_index], i), 5, green, -1)
        x_axis_index += 1
    #cv2.line(param, (x1, y1), (x2, y2), (0, 0, 255), 4, cv2.LINE_AA)
    cv2.imshow('labeling_tusimple', param)
    return m, b


def on_mouse(event, x, y, flags, param):
    # event는 마우스 동작 상수값, 클릭, 이동 등등
    # x, y는 내가 띄운 창을 기준으로 좌측 상단점이 0,0이 됌
    # flags는 마우스 이벤트가 발생할 때 키보드 또는 마우스 상태를 의미, Shif+마우스 등 설정가능
    # param은 영상 일수도 있도 전달하고 싶은 데이타, 안쓰더라도 넣어줘야함
    global lane_count, left_lane_coordi, right_lane_coordi, h_samples  # 밖에 있는 oldx, oldy 불러옴
    global pre_left_lane_coordi, pre_right_lane_coordi, l_index, r_index, pre_label, more_point

    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽이 눌러지면 실행
        pre_label = False
        if more_point == False:
            if lane_count >= 0 and lane_count < 4:
                # 마우스가 눌렀을 때 좌표 저장, 띄워진 영상에서의 좌측 상단 기준
                left_lane_coordi.points_append(x, y)
                param[1] = param[0].copy()
                if lane_count == 3:
                    calc_inclination(left_lane_coordi, left_lane,
                                    param[0], h_samples)
                print('count : ' + str(lane_count), end=' -> ')
                cv2.circle(param[0], (x, y), 5, blue, -1)
                cv2.circle(param[1], (x, y), 5, blue, -1)
                cv2.imshow('labeling_tusimple', param[0])
                print('EVENT_LBUTTONDOWN: %d, %d' % (x, y))  # 좌표 출력
                lane_count += 1

            elif lane_count >= 4 and lane_count < 8:
                # 마우스가 눌렀을 때 좌표 저장, 띄워진 영상에서의 좌측 상단 기준
                right_lane_coordi.points_append(x, y)
                param[1] = param[0].copy()
                if lane_count == 7:
                    calc_inclination(right_lane_coordi,
                                    right_lane, param[0], h_samples)
                print('count : ' + str(lane_count), end=' -> ')
                cv2.circle(param[0], (x, y), 5, blue, -1)
                cv2.circle(param[1], (x, y), 5, blue, -1)
                cv2.imshow('labeling_tusimple', param[0])
                print('EVENT_LBUTTONDOWN: %d, %d' % (x, y))  # 좌표 출력
                lane_count += 1
            else:
                pass
        else:
            if lane_count >= 0 and lane_count < 7:
                # 마우스가 눌렀을 때 좌표 저장, 띄워진 영상에서의 좌측 상단 기준
                left_lane_coordi.points_append(x, y)
                param[1] = param[0].copy()
                if lane_count == 6:
                    calc_inclination(left_lane_coordi, left_lane,
                                    param[0], h_samples)
                print('count : ' + str(lane_count), end=' -> ')
                cv2.circle(param[0], (x, y), 5, blue, -1)
                cv2.circle(param[1], (x, y), 5, blue, -1)
                cv2.imshow('labeling_tusimple', param[0])
                print('EVENT_LBUTTONDOWN: %d, %d' % (x, y))  # 좌표 출력
                lane_count += 1

            elif lane_count >= 7 and lane_count < 14:
                # 마우스가 눌렀을 때 좌표 저장, 띄워진 영상에서의 좌측 상단 기준
                right_lane_coordi.points_append(x, y)
                param[1] = param[0].copy()
                if lane_count == 13:
                    calc_inclination(right_lane_coordi,
                                    right_lane, param[0], h_samples)
                print('count : ' + str(lane_count), end=' -> ')
                cv2.circle(param[0], (x, y), 5, blue, -1)
                cv2.circle(param[1], (x, y), 5, blue, -1)
                cv2.imshow('labeling_tusimple', param[0])
                print('EVENT_LBUTTONDOWN: %d, %d' % (x, y))  # 좌표 출력
                lane_count += 1
            else:
                pass

    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스가 움직일 때 발생
        # if flags & cv2.EVENT_FLAG_LBUTTON: # ==를 쓰면 다른 키도 입력되었을 때 작동안하므로 &(and) 사용
        if flags == cv2.EVENT_FLAG_RBUTTON:
            """
            오른쪽 클릭후 마우스 움직이면 좌표값 이동
            마우스 이동마다 직선 다시 그리기
            단, 이전의 라벨링을 가져왔을때만 적용
            """
            param[0] = param[1].copy()
            if more_point==False:
                if l_index != -1:
                    if l_index > 0 and l_index < 3:
                        cv2.line(param[0], (left_lane_coordi.points[l_index-1][0],
                                left_lane_coordi.points[l_index-1][1]), (x, y), sky_blue, 4, cv2.LINE_AA)
                        cv2.line(param[0], (x, y), (left_lane_coordi.points[l_index+1][0],
                                left_lane_coordi.points[l_index+1][1]), sky_blue, 4, cv2.LINE_AA)
                        cv2.circle(param[0], (x, y), 5, magenta, -1)
                        cv2.imshow('labeling_tusimple', param[0])

                    elif l_index == 0:
                        cv2.line(param[0], (x, y), (left_lane_coordi.points[l_index+1][0],
                                left_lane_coordi.points[l_index+1][1]), sky_blue, 4, cv2.LINE_AA)
                        cv2.circle(param[0], (x, y), 5, magenta, -1)
                        cv2.imshow('labeling_tusimple', param[0])

                    elif l_index == 3:
                        cv2.line(param[0], (left_lane_coordi.points[l_index-1][0],
                                left_lane_coordi.points[l_index-1][1]), (x, y), sky_blue, 4, cv2.LINE_AA)
                        cv2.circle(param[0], (x, y), 5, magenta, -1)
                        cv2.imshow('labeling_tusimple', param[0])

                if r_index != -1:
                    if r_index > 0 and r_index < 3:
                        cv2.line(param[0], (right_lane_coordi.points[r_index-1][0],
                                right_lane_coordi.points[r_index-1][1]), (x, y), sky_blue, 4, cv2.LINE_AA)
                        cv2.line(param[0], (x, y), (right_lane_coordi.points[r_index+1][0],
                                right_lane_coordi.points[r_index+1][1]), sky_blue, 4, cv2.LINE_AA)
                        cv2.circle(param[0], (x, y), 5, magenta, -1)
                        cv2.imshow('labeling_tusimple', param[0])

                    elif r_index == 0:
                        cv2.line(param[0], (x, y), (right_lane_coordi.points[r_index+1][0],
                                right_lane_coordi.points[r_index+1][1]), sky_blue, 4, cv2.LINE_AA)
                        cv2.circle(param[0], (x, y), 5, magenta, -1)
                        cv2.imshow('labeling_tusimple', param[0])

                    elif r_index == 3:
                        cv2.line(param[0], (right_lane_coordi.points[r_index-1][0],
                                right_lane_coordi.points[r_index-1][1]), (x, y), sky_blue, 4, cv2.LINE_AA)
                        cv2.circle(param[0], (x, y), 5, magenta, -1)
                        cv2.imshow('labeling_tusimple', param[0])
            else:
                if l_index != -1:
                    if l_index > 0 and l_index < 6:
                        cv2.line(param[0], (left_lane_coordi.points[l_index-1][0],
                                left_lane_coordi.points[l_index-1][1]), (x, y), sky_blue, 4, cv2.LINE_AA)
                        cv2.line(param[0], (x, y), (left_lane_coordi.points[l_index+1][0],
                                left_lane_coordi.points[l_index+1][1]), sky_blue, 4, cv2.LINE_AA)
                        cv2.circle(param[0], (x, y), 5, magenta, -1)
                        cv2.imshow('labeling_tusimple', param[0])

                    elif l_index == 0:
                        cv2.line(param[0], (x, y), (left_lane_coordi.points[l_index+1][0],
                                left_lane_coordi.points[l_index+1][1]), sky_blue, 4, cv2.LINE_AA)
                        cv2.circle(param[0], (x, y), 5, magenta, -1)
                        cv2.imshow('labeling_tusimple', param[0])

                    elif l_index == 6:
                        cv2.line(param[0], (left_lane_coordi.points[l_index-1][0],
                                left_lane_coordi.points[l_index-1][1]), (x, y), sky_blue, 4, cv2.LINE_AA)
                        cv2.circle(param[0], (x, y), 5, magenta, -1)
                        cv2.imshow('labeling_tusimple', param[0])

                if r_index != -1:
                    if r_index > 0 and r_index < 6:
                        cv2.line(param[0], (right_lane_coordi.points[r_index-1][0],
                                right_lane_coordi.points[r_index-1][1]), (x, y), sky_blue, 4, cv2.LINE_AA)
                        cv2.line(param[0], (x, y), (right_lane_coordi.points[r_index+1][0],
                                right_lane_coordi.points[r_index+1][1]), sky_blue, 4, cv2.LINE_AA)
                        cv2.circle(param[0], (x, y), 5, magenta, -1)
                        cv2.imshow('labeling_tusimple', param[0])

                    elif r_index == 0:
                        cv2.line(param[0], (x, y), (right_lane_coordi.points[r_index+1][0],
                                right_lane_coordi.points[r_index+1][1]), sky_blue, 4, cv2.LINE_AA)
                        cv2.circle(param[0], (x, y), 5, magenta, -1)
                        cv2.imshow('labeling_tusimple', param[0])

                    elif r_index == 6:
                        cv2.line(param[0], (right_lane_coordi.points[r_index-1][0],
                                right_lane_coordi.points[r_index-1][1]), (x, y), sky_blue, 4, cv2.LINE_AA)
                        cv2.circle(param[0], (x, y), 5, magenta, -1)
                        cv2.imshow('labeling_tusimple', param[0])

        else:
            if more_point == False:
                if lane_count == 0 or lane_count == 4:
                    pass

                elif lane_count > 0 and lane_count < 4:
                    param[0] = param[1].copy()
                    cv2.line(param[0], (left_lane_coordi.points[lane_count-1][0],
                            left_lane_coordi.points[lane_count-1][1]), (x, y), red, 4, cv2.LINE_AA)
                    cv2.imshow('labeling_tusimple', param[0])

                elif lane_count > 4 and lane_count < 8:
                    param[0] = param[1].copy()
                    cv2.line(param[0], (right_lane_coordi.points[lane_count-5][0],
                            right_lane_coordi.points[lane_count-5][1]), (x, y), red, 4, cv2.LINE_AA)
                    cv2.imshow('labeling_tusimple', param[0])
                else:
                    pass
            else :
                if lane_count == 0 or lane_count == 7:
                    pass

                elif lane_count > 0 and lane_count < 7:
                    param[0] = param[1].copy()
                    cv2.line(param[0], (left_lane_coordi.points[lane_count-1][0],
                            left_lane_coordi.points[lane_count-1][1]), (x, y), red, 4, cv2.LINE_AA)
                    cv2.imshow('labeling_tusimple', param[0])

                elif lane_count > 7 and lane_count < 14:
                    param[0] = param[1].copy()
                    cv2.line(param[0], (right_lane_coordi.points[lane_count-8][0],
                            right_lane_coordi.points[lane_count-8][1]), (x, y), red, 4, cv2.LINE_AA)
                    cv2.imshow('labeling_tusimple', param[0])
                else:
                    pass

    elif event == cv2.EVENT_RBUTTONDOWN:  # 오른쪽 버튼으로 마우스 좌표 옮기기
        print("x : {0}, y : {1}".format(x, y))

        index = 0
        for pre_left in pre_left_lane_coordi.points:
            if (x >= pre_left[0] - 3 and x <= pre_left[0] + 3) and (y >= pre_left[1] - 3 and y <= pre_left[1] + 3):
                l_index = index
                break
            index += 1

        index = 0
        for pre_right in pre_right_lane_coordi.points:
            if (x >= pre_right[0] - 3 and x <= pre_right[0] + 3) and (y >= pre_right[1] - 2 and y <= pre_right[1] + 3):
                r_index = index
                break
            index += 1

    elif event == cv2.EVENT_LBUTTONDBLCLK:  # 이전의 프레임을 불러왔다면, 더블클릭으로 프레임 저장
        param[1] = param[0].copy()

    elif event == cv2.EVENT_RBUTTONUP:
        # 기존의 포인터 바꾸고, 새롭게 기울기 저장 및 그리기
        if l_index != -1:
            left_lane_coordi.point_change(l_index, (x, y))
            pre_label = True
            calc_inclination(left_lane_coordi, left_lane, param[0], h_samples)
            l_index = -1

        if r_index != -1:
            right_lane_coordi.point_change(r_index, (x, y))
            pre_label = True
            calc_inclination(right_lane_coordi, right_lane,
                             param[0], h_samples)
            r_index = -1

    elif event == cv2.EVENT_RBUTTONDBLCLK:  # 오른쪽 마우스 더블클릭 이전 라벨링 불러오기
        #왼쪽 point, line 불러오기
        left_lane_coordi = copy.deepcopy(pre_left_lane_coordi)
        if len(left_lane_coordi.points) != 0:
            calc_inclination(left_lane_coordi, left_lane, param[0], h_samples)
            for count in range(0, len(left_lane_coordi.points)):
                cv2.circle(param[0], (left_lane_coordi.points[count]
                           [0], left_lane_coordi.points[count][1]), 5, blue, -1)
            if more_point == False : lane_count = 4
            else : lane_count = 7

        #오른쪽 point, line 불러오기
        right_lane_coordi = copy.deepcopy(pre_right_lane_coordi)
        if len(right_lane_coordi.points) != 0:
            calc_inclination(right_lane_coordi, right_lane,
                             param[0], h_samples)
            for count in range(0, len(right_lane_coordi.points)):
                cv2.circle(param[0], (right_lane_coordi.points[count]
                           [0], right_lane_coordi.points[count][1]), 5, blue, -1)
            if more_point == False : lane_count = 8  # 이전 라벨링을 불러왔기 때문에 lane_count 8로 설정
            else : lane_count = 14

        cv2.imshow('labeling_tusimple', param[0])
        param[1] = param[0].copy()


def labeling(imagenum, auto_bright):
    cv2.namedWindow('labeling_tusimple', cv2.WND_PROP_FULLSCREEN)
    # file_count = imagenum-1
    file_count = len(png_images) #labelling을 하게되면 png 파일이 무조건 생성 -> png 파일로 인덱스 기록 

    # 이전의 값들을 받아오기 위해 global 변수로 받아옴
    global file_data, lane_count, pre_label, b_imgnum, more_point
    global left_lane_coordi, right_lane_coordi, pre_left_lane_coordi, pre_right_lane_coordi

    pre_left_lane_coordi.__init__()
    pre_right_lane_coordi.__init__()

    while (file_count < len(jpg_images)): #jpg 파일과 png 파일 비교해서 실행
        if b_imgnum == False:
            line = jpg_images[file_count]
            line = line.replace("\\", "/")
            line = line[2:]
        else: 
            line = 'clips/' + str(imagenum).zfill(8) + '.jpg'
            file_count=0
            for i in jpg_images:
                if i.replace("\\", "/")[2:] == line:
                    print(file_count)
                    break
                file_count+=1

            b_imgnum = False

        if more_point == False: point_flag = 'off'
        else :point_flag='on'
        left_lane_coordi.__init__()
        right_lane_coordi.__init__()

        file_data = OrderedDict()
        lane_count = 0
        #print('lines : ' + str(len(lines)))
        line = line.strip()
        file_data["lanes"] = [[], []]
        file_data["h_samples"] = h_samples
        #print('file num : ' + str(len(lines)), end=' ')
        #print(line)
        img = cv2.imread(line, cv2.IMREAD_COLOR)
        img_h, img_w, _ = img.shape

        if auto_bright == 1:
            img_crop_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            yyy, cr, cb = cv2.split(img_crop_ycrcb)

            # 밝기 성분에 대해서만 히스토그램 평활화 수행
            dst_y = cv2.equalizeHist(yyy)
            dst_ycrcb = cv2.merge([dst_y, cr, cb])
            crop = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)

        else:
            crop = img.copy()
        #crop2는 마우스 이동시 직선을 잘 그려주기 위해서 임시로 이전 frame을 저장하기 위한것

        crop2 = crop.copy()
        img_crop = [crop, crop2]

        seg_gt_png_path = (line.rstrip('.jpg ')) + '.png'
        train_gt_str = line + ' ' + seg_gt_png_path
        train_cart_classes = ''
        file_data["raw_file"] = line
        pre_label = False

        #아래 cv2.line은 사용자가 노란색선안에 점들이 제대로 찍혔는지 확인하기 위함
        cv2.line(crop, (0, h_samples[0]), (img_w,
                 h_samples[0]), yellow, 5, cv2.LINE_AA)
        cv2.line(crop, (0, h_samples[len(h_samples)-1]), (img_w,
                 h_samples[len(h_samples)-1]), yellow, 4, cv2.LINE_AA)
        #위 cv2.line은 사용자가 노란색선안에 점들이 제대로 찍혔는지 확인하기 위함

        # 라벨링 정보 확인
        jpg_image_name = os.path.splitext(os.path.basename(seg_gt_png_path))[0] + '.jpg'
        with open(json_file_path, "r+") as json_file: #json file 내용 삭제
            new_json = json_file.readlines()
            for data in new_json:
                json_data = json.loads(data)
                if json_data['raw_file'].split('/')[-1] == jpg_image_name:
                    for count in range(0, len(json_data['lanes'][0])):
                        if json_data['lanes'][0][count] != -2:
                            cv2.circle(
                                crop, (json_data['lanes'][0][count], json_data['h_samples'][count]), 5, green, -1)
                        if json_data['lanes'][1][count] != -2:
                            cv2.circle(
                                crop, (json_data['lanes'][1][count], json_data['h_samples'][count]), 5, green, -1)
        # 라벨링 정보 확인
        text = str(file_count+1) + ' / ' + str(len(jpg_images)) + ', filename : ' + line + ', More point : ' + point_flag
        cv2.putText(crop, text, org, font, 1, red, 4)
        # 윈도우 창
        # 마우스 입력, namedWIndow or imshow가 실행되어 창이 떠있는 상태에서만 사용가능
        # 마우스 이벤트가 발생하면 on_mouse 함수 실행
        cv2.setMouseCallback('labeling_tusimple', on_mouse, param=img_crop)

        # 영상 출력
        cv2.imshow('labeling_tusimple', crop)
        waitKey = cv2.waitKey()
        # print(waitKey)

        if NEXT_PAGE == waitKey or waitKey == 13:  # space, enter key 라벨링 저장
            file_count += 1
            label_img = np.zeros((img_h, img_w), dtype=np.uint8)

            if lane_count < 3:
                for i in range(1, 40 + 1, 1):
                    file_data["lanes"][0].append(-2)
                    file_data["lanes"][1].append(-2)

                train_gt_str = train_gt_str + ' ' + '0 0\n'
                train_cart_classes = train_cart_classes + '0 0\n'

            if lane_count >= 3:  # lane이 2개 이상 선택 되었을때 json으로 저장할것
                #여기에 json 파일 추가해야함!!!!!!!!!!!!!
                ##################################################cv2.imsave 해줘야함~!!!

                if len(file_data["lanes"][0]) > 0:
                    lane_class = 1  # 왼쪽
                    train_gt_str = train_gt_str + ' ' + '1'
                    train_cart_classes = train_cart_classes + '1'

                    for i in range(0, len(file_data["lanes"][0])-1):
                        if file_data["lanes"][0][i] != -2 and file_data["lanes"][0][i+1] != -2:
                            cv2.line(label_img, (file_data["lanes"][0][i], h_samples[i]), (
                                file_data["lanes"][0][i+1], h_samples[i+1]), lane_class, 24, cv2.LINE_8)
                else:
                    train_gt_str = train_gt_str + ' ' + '0'
                    train_cart_classes = train_cart_classes + '0'

                if len(file_data["lanes"][1]) > 0:
                    lane_class = 2  # 오른쪽
                    train_gt_str = train_gt_str + ' ' + '1'
                    train_cart_classes = train_cart_classes + ' 1'

                    for i in range(0, len(file_data["lanes"][1])-1):
                        if file_data["lanes"][1][i] != -2 and file_data["lanes"][1][i+1] != -2:
                            cv2.line(label_img, (file_data["lanes"][1][i], h_samples[i]), (
                                file_data["lanes"][1][i+1], h_samples[i+1]), lane_class, 24, cv2.LINE_8)
                else:
                    if len(file_data['lanes'][1])==0:
                        for i in range(1, 40 + 1, 1):
                            file_data["lanes"][1].append(-2)

                    train_gt_str = train_gt_str + ' ' + '0'
                    train_cart_classes = train_cart_classes + ' 0'

                train_gt_str = train_gt_str + '\n'
                if train_cart_classes != None:
                    train_cart_classes = train_cart_classes+'\n'

            # print('h_samples : ', end='')
            # print(file_data["h_samples"])
            print('lanes : ', end='')
            print(file_data["lanes"])

            with open("./train_cart_classes.txt", 'r+') as gt_classes_txt:
                for line in gt_classes_txt:
                    pass
                gt_classes_txt.write(train_cart_classes)

            with open("./train_gt.txt", 'r+') as gt_txt:
                for line in gt_txt:
                    pass
                gt_txt.write(train_gt_str)
                
            # cv2.imshow('label_img', label_img)     #seg label 보고싶으면 이걸 활성화 할것
            cv2.imwrite(seg_gt_png_path, label_img)

            #현재 label 좌표 저장
            pre_left_lane_coordi = copy.deepcopy(left_lane_coordi)
            pre_right_lane_coordi = copy.deepcopy(right_lane_coordi)

            with open(json_file_path, "r+") as json_file:
                for line in json_file:  # 파일의 맨 끝으로 가는 코드
                    pass
                json_file.write(json.dumps(file_data) + '\n')
            continue

        #'a'(97) 왼쪽 line만 존재
        elif waitKey == ord('a') or waitKey == ord('A'):
            file_count += 1
            label_img = np.zeros((img_h, img_w), dtype=np.uint8)

            for i in range(1, 40 + 1, 1):
                file_data["lanes"][1].append(-2)

            train_gt_str = train_gt_str + ' 1 0\n'
            train_cart_classes = train_cart_classes + '1 0\n'

            if lane_count >= 3:  # lane이 2개 이상 선택 되었을때 json으로 저장할것
                if len(file_data["lanes"][0]) > 0:
                    lane_class = 1  # 왼쪽
                    for i in range(0, len(file_data["lanes"][0])-1):
                        if file_data["lanes"][0][i] != -2 and file_data["lanes"][0][i+1] != -2:
                            cv2.line(label_img, (file_data["lanes"][0][i], h_samples[i]), (
                                file_data["lanes"][0][i+1], h_samples[i+1]), lane_class, 24, cv2.LINE_8)

            # print('h_samples : ', end='')
            # print(file_data["h_samples"])
            print('lanes : ', end='')
            print(file_data["lanes"])

            with open("./train_cart_classes.txt", 'r+') as gt_classes_txt:
                for line in gt_classes_txt:
                    pass
                gt_classes_txt.write(train_cart_classes)

            with open("./train_gt.txt", 'r+') as gt_txt:
                for line in gt_txt:
                    pass
                gt_txt.write(train_gt_str)

            cv2.imwrite(seg_gt_png_path, label_img)

            #현재 label 좌표 저장
            pre_left_lane_coordi = copy.deepcopy(left_lane_coordi)
            pre_right_lane_coordi = copy.deepcopy(right_lane_coordi)

            with open(json_file_path, "r+") as json_file:
                for line in json_file:  # 파일의 맨 끝으로 가는 코드
                    pass
                json_file.write(json.dumps(file_data) + '\n')
            continue
        
        #'s'(115) 오른쪽 line 만 존재
        elif waitKey == ord('s') or waitKey == ord('S'):
            file_count += 1
            label_img = np.zeros((img_h, img_w), dtype=np.uint8)

            for i in range(1, 40 + 1, 1):
                file_data["lanes"][1].append(-2)

            file_data["lanes"][0], file_data["lanes"][1] = file_data["lanes"][1], file_data["lanes"][0]

            train_gt_str = train_gt_str + ' 0 1\n'
            train_cart_classes = train_cart_classes + '0 1\n'

            if lane_count >= 3:  # lane이 2개 이상 선택 되었을때 json으로 저장할것                
                if len(file_data["lanes"][1]) > 0:
                    lane_class = 2 # 오른쪽
                    for i in range(0, len(file_data["lanes"][1])-1):
                        if file_data["lanes"][1][i] != -2 and file_data["lanes"][1][i+1] != -2:
                            cv2.line(label_img, (file_data["lanes"][1][i], h_samples[i]), (
                                file_data["lanes"][1][i+1], h_samples[i+1]), lane_class, 24, cv2.LINE_8)

            # print('h_samples : ', end='')
            # print(file_data["h_samples"])
            print('lanes : ', end='')
            print(file_data["lanes"])

            with open("./train_cart_classes.txt", 'r+') as gt_classes_txt:
                for line in gt_classes_txt:
                    pass
                gt_classes_txt.write(train_cart_classes)

            with open("./train_gt.txt", 'r+') as gt_txt:
                for line in gt_txt:
                    pass
                gt_txt.write(train_gt_str)

            cv2.imwrite(seg_gt_png_path, label_img)

            #현재 label 좌표 저장
            pre_left_lane_coordi = copy.deepcopy(left_lane_coordi)
            pre_right_lane_coordi = copy.deepcopy(right_lane_coordi)

            with open(json_file_path, "r+") as json_file:
                for line in json_file:  # 파일의 맨 끝으로 가는 코드
                    pass
                json_file.write(json.dumps(file_data) + '\n')
            continue

        elif waitKey == 3 or waitKey == 54:  # -> 방향키 눌렀을때 다음 이미지로 그냥 넘어가고 라벨링은 안됨
            file_count += 1

        elif waitKey == 2 or waitKey == 52:  # <- 방향키 눌렀을때 이전 이미지로 그냥 넘어가고 라벨링은 안됨
            if (file_count > 0):
                file_count -= 1
            else:
                file_count = 0

        # 키보드 1을 누르면 auto_bright가 바뀜
        elif waitKey == 49:  
            if auto_bright == 1:
                auto_bright = 0
                print('auto_bright off')
            else:
                auto_bright = 1
                print('auto_bright on')

        elif waitKey == 50: 
            if more_point == False:
                more_point = True
                print('more point on')
            else:
                more_point = False
                print('more point off')

        # Backspace 해당 labelling 정보 지우기
        elif waitKey == 8:
            # jpg_image_name = os.path.splitext(os.path.basename(seg_gt_png_path))[0] + '.jpg'
            check = messageBox("경고", "{} labelling 정보를 삭제 하시겠습니까?".format(jpg_image_name), 49)
            index = 0 #txt 파일 행 확인

            if check == 1:
                if os.path.isfile(seg_gt_png_path):
                    os.remove(seg_gt_png_path) #png 파일삭제
                    
                    with open(json_file_path, "r+") as json_file: #json file 내용 삭제
                        new_json = json_file.readlines()
                        json_file.seek(0) #파일의 시작점으로 이동
                        for line in new_json:
                            json_data = json.loads(line)
                            if json_data['raw_file'].split('/')[-1] != jpg_image_name:
                                json_file.write(json.dumps(json_data)+'\n')
                        json_file.truncate() #현재 기록된 정보 이후로 전부 삭제

                    with open("./train_gt.txt", 'r+') as gt_txt:
                        new_gt_txt = gt_txt.readlines()
                        gt_txt.seek(0)
                        count = 0
                        for line in new_gt_txt:
                            tmp = line.split(' ')[0].split('/')[1]
                            if tmp != jpg_image_name:
                                gt_txt.write(line)
                            else :
                                index = count #train_gt에서 index 번호 확인
                            count+=1
                        gt_txt.truncate()
                    
                    with open("./train_cart_classes.txt", 'r+') as gt_classes_txt:
                        new_gt_classes_txt = gt_classes_txt.readlines()
                        gt_classes_txt.seek(0)
                        count = 0
                        for line in new_gt_classes_txt:
                            if count != index:
                                gt_classes_txt.write(line)
                            count+=1
                        gt_classes_txt.truncate()
                    print("사용자에 의해 {} labelling 정보 삭제됨".format(jpg_image_name))
                
            elif check == 2:
                print("삭제안됨")
                
        # 'q' (113) 나 'Q' (81) 누르면 while문에서 빠져나가도록
        elif waitKey == ord('q') or waitKey == ord('Q') or waitKey == 66:
            #json 저장 함수 호출 작성해야함!!!!!!!!!!!
            break
        
    cv2.destroyAllWindows()
    # f.close()
    
def messageBox(title, text, style): 
    # message box 띄우기 
    # 메시지 스타일 참고 : https://papazeros.tistory.com/m/3/
	return ctypes.windll.user32.MessageBoxW(None, text, title, style)

if __name__ == '__main__':
    ap = ArgParse()
    ap.add_argument('--imagenum', type=int, default=0)
    ap.add_argument('--auto_bright', type=int, default=0)
    #ap.add_argument('--labels', type=str, default='label_data_0313.json')

    args = ap.parse_args()

    #label_index.txt 파일에 적힌 번호 다음 번호를 받아와서 이미지를 로드함
    label_index = 0
    b_imgnum = False
    #생성된 png를 기준으로 다음 번호 받아와서 이미지를 로드함
    if len(png_images)==0:
        label_index = os.path.splitext(os.path.basename(jpg_images[0]))[0]
    else:
        label_index = os.path.splitext(os.path.basename(png_images[-1]))[0]

    if label_index != 0 and args.imagenum == 0:
        args.imagenum = label_index
        print('label_index.txt 파일에 의해 ' + str(label_index) + '번 이미지가 열림')
    #--imagenum argument로 사용자에게 이미지 번호를 받아서 로드함
    elif args.imagenum != 0:
        b_imgnum = True #arg의 imgnum의 flag
        print('사용자에 의해 ' + str(args.imagenum) + '번 이미지가 열림')

    left_lane_coordi = MyLane()
    right_lane_coordi = MyLane()

    pre_left_lane_coordi = MyLane()
    pre_right_lane_coordi = MyLane()

    labeling(args.imagenum, args.auto_bright)