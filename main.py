import random

import ray
import os
import glob
import cv2
import numpy as np
from Edge import *
import time
from VisionTool3D import *
import open3d as o3d
import numpy
lambda_hat =97
tau =25
sigma = 0.3
sample_point_len =30
param_mode = "Find dynamic"# "change_Lambda_tau", "Find dynamic"
test_image_set_idx = 7
flow_mode ="denseFlow"#"denseSIFT", "denseFlow", "OrbDesc", "OpticalFlow"
sigma_mode =True
##매칭되는 데이터셋을 가져옵니다.
fast_measure_mode = False
out_range = 0.2#평균적으로 out_range(m)만큼 거리차이가 나는 점이 많으면 다음과같이 구해집니다.
iou_get_mode = True

#1. get image and depth map


depth_names = glob.glob("depth/test" + str(test_image_set_idx) + "/*.png")
image_names = glob.glob("image/test" + str(test_image_set_idx) + "/*.png")
dmaps = [cv2.imread(depth_names[i],cv2.IMREAD_ANYDEPTH) for i in range(len(depth_names))]

images = [cv2.imread(image_names[i]) for i in range(len(image_names))]
for i, distorted_image in enumerate(images):
    fx = 544.2582548211519  # focal length x
    fy = 546.0878823951958  # focal length y
    cx = 326.8604521819424  # optical center x
    cy = 236.1210149172594  # optical center y
    camera_matrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    return_image = cv2.undistort(distorted_image,camera_matrix,np.array([0.0369,-0.0557,0,0]))
    images[i] =return_image

image_height, image_width, _ = images[0].shape
#2. get depth map cluster
#2-1 filter depth map
if test_image_set_idx < 6:
    filtered_dmaps = Depth_preprocess(dmaps)
else:
    filtered_dmaps = Depth_preprocess2(dmaps)
#2-2 get surface normal
phi_d_image, phi_c_image = Get_EdgeMap(image_height,image_width,filtered_dmaps[0])
if param_mode == "change_Lambda_tau":
    Edgemap_changer(phi_d_image, phi_c_image, filtered_dmaps[0])

#2-3 get Image Segmentation Infomation
cnt, labels = Get_SegmentedImage(image_height,image_width,phi_d_image,
                        phi_c_image,tau,lambda_hat, filtered_dmaps[0])
first_label_image = np.zeros((image_height,image_width,3),dtype=np.uint8)
first_label_image[labels == 0] = [0,0,0]
for i in range(cnt):
    if i==0:
        continue
    else:
        first_label_image[labels==i] = [int(j) for j in np.random.randint(0,255, 3)]
cv2.imshow("first label image",first_label_image)
#3 sample points up to 10 in each segment

component_edge_graph, edge_set, component_graph = Find_VoronoiEdge(image_height, image_width,labels,cnt)
second_label_image = np.zeros((image_height,image_width,3),dtype=np.uint8)
for i, row_list in enumerate(component_graph):
    if i == 0:
        continue
    for j, col_value in enumerate(row_list[1:i + 1], start=1):
        if component_graph[i][j] :
            color = [int(j) for j in np.random.randint(0,255, 3)]
            for point_idx in component_edge_graph[i][j]:
                second_label_image[edge_set[point_idx].y,edge_set[point_idx].x] = color
cv2.imshow("second label image",second_label_image)
#3-4 cluster get 각 cluster에서 점들을 추출한다.
bad_removed_labels =labels.copy()
# bad_removed_labels[dmaps[0]>3.5] = -1
bad_removed_labels[dmaps[0]==0] = -1

sampled_point = [[] for _ in range(cnt)]#0번은 배경에 대한 엣지입니다. range가 적절하지않으면 버립니다.
for clusterIdx in range(1,cnt):#0번은 배경 즉 엣지이기 때문에 넘어감
    clu_point= np.where(bad_removed_labels==clusterIdx)
    # clu_point_2 = np.where(labels == clusterIdx)
    point_size = len(clu_point[0])
    if point_size ==0:#원소가 없다? 이건 유효하지 않은 세그먼트라는 뜻이다. 그렇다면 샘플링하지않고 넘어감
        # print(clusterIdx)
        continue
    elif point_size >sample_point_len:#10개 이상이다. 그러면 샘플링해서 저장한다.
        randIdx = [i for i in range(point_size)]
        choiced_Idx = random.sample(randIdx,sample_point_len)
        for k in choiced_Idx:
            sampled_point[clusterIdx].append([clu_point[0][k],clu_point[1][k]])
    else: #10개 미만이다.
        for k in range(point_size):
            sampled_point[clusterIdx].append([clu_point[0][k],clu_point[1][k]])


#4 get feature and find next point

if flow_mode == "denseSIFT":
    from sift_flow_torch import SiftFlowTorch
    from MatchMaker import *
    import torch
    step_size =2
    sift_flow = SiftFlowTorch(cell_size=10,step_size=step_size,is_boundary_included=True,num_bins=16 ,cuda=True,fp16=True,return_numpy=False)
    torch.cuda.synchronize()
    start = time.perf_counter()
    descs = sift_flow.extract_descriptor(images)##이미지 두장에 대해서 feature를 뽑습니다.
    end = time.perf_counter()
    flow = find_local_matches(descs[0:1], descs[1:2],11)
    flow = flow.permute(1, 2, 0).detach().cpu().numpy()#두 이미지 상에 매칭을 저장합니다. 0번째는 x, 1번째는 y입니다.

elif flow_mode == "denseFlow":
    source = cv2.cvtColor(images[0],cv2.COLOR_BGR2GRAY)
    target = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)
    next_sampled_point = [[] for _ in range(cnt)]
    flow = cv2.calcOpticalFlowFarneback(source,target,0.0,0.5,3,14,10,5,1.2,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    view_image = images[0].copy()
    # correspondence_map.append(dense_flow)
    for y in range(0, image_height, 5):
        for x in range(0, image_width, 5):
            view_image = cv2.arrowedLine(view_image, (x, y),
                                         (int(x + flow[y, x][0]), int(y + flow[y, x][1])), (0, 0, 255), 1,
                                         8, 0)
elif flow_mode=="OrbDesc" :

    GFTT_mode = False
    show_image= images[0].copy()
    show_image2 = images[1].copy()
    source = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    target = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)
    # kernel = np.array([[0, -1, 0],
    #                    [-1, 5, -1],
    #                    [0, -1, 0]])
    # source = cv2.filter2D(src=source, ddepth=-1, kernel=kernel)
    # target = cv2.filter2D(src=target, ddepth=-1, kernel=kernel)
    cv2.imshow("src_sharp", source)
    cv2.imshow("tar_sharp", target)
    if not GFTT_mode:
        orb = cv2.ORB_create(
            nfeatures=30000
        )
        kp_source, des_source = orb.detectAndCompute(source,None)
        kp_target, des_target = orb.detectAndCompute(target,None)
    else :
        kp_source = cv2.goodFeaturesToTrack(source, 400, 0.01, 10)
        kp_target = cv2.goodFeaturesToTrack(target, 400, 0.01, 10)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_source, des_target,k=2)
    pts1 = []
    pts2 = []
    pts1_query = []
    pts2_query = []
    sampled_point = [[] for _ in range(cnt)]
    next_sampled_point = [[] for _ in range(cnt)]

    sampled_point_f = [[] for _ in range(cnt)]
    next_sampled_point_f = [[] for _ in range(cnt)]
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)
            pts1.append(kp_source[m.queryIdx].pt)
            pts2.append(kp_target[m.trainIdx].pt)
            pts1_query.append(kp_source[m.queryIdx])
            pts2_query.append(kp_target[m.trainIdx])
    print("매칭된 점의 수 :", len(pts1))
    for k, point in enumerate(pts1):
        p_x, p_y = int(point[0]), int(point[1])
        next_p_x, next_p_y = int(pts2[k][0]), int(pts2[k][1])
        if bad_removed_labels[p_y,p_x] != -1 and bad_removed_labels[p_y,p_x] != 0:
            sampled_point[bad_removed_labels[p_y,p_x]].append([point[1],point[0]])
            sampled_point_f[bad_removed_labels[p_y,p_x]].append([point[0],point[1]])
            next_sampled_point[bad_removed_labels[p_y,p_x]].append([pts2[k][1], pts2[k][0]])
            next_sampled_point_f[bad_removed_labels[p_y, p_x]].append([pts2[k][0], pts2[k][1]])
            # cv2.circle(show_image, (p_x, p_y), 3, (0, 0, 255),1)
            # cv2.circle(show_image2,(next_p_x,next_p_y), 3, (0, 0, 255),1)
            # match_image = cv2.drawMatches(source,kp_source,target,kp_target,[good_matches[k]],None, cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # cv2.imshow("match", match_image)
            # cv2.waitKey(0)
    #------------------------같은
    for i, seg_points in enumerate(sampled_point_f):
        if len(seg_points) <8:
            sampled_point_f[i] = []
            next_sampled_point_f[i] = []
            sampled_point[i] = []
            next_sampled_point[i] = []
            continue
        show_image = images[0].copy()
        show_image2 = images[1].copy()
        for k, point in enumerate(seg_points):
            cv2.circle(show_image, (int(point[0]), int(point[1])), 3, (255, 0, 0), -1)
            cv2.circle(show_image2, (int(next_sampled_point_f[i][k][0]), int(next_sampled_point_f[i][k][1])), 3, (255, 0, 0), -1)
        source_seg_points = np.float64(seg_points)
        target_seg_points = np.float64(next_sampled_point_f[i])
        F, mask = cv2.findFundamentalMat(source_seg_points, target_seg_points , cv2.FM_LMEDS)
        # print(i, "거리",len(seg_points))
        new_point =[]
        next_new_point = []
        for j, point in enumerate(seg_points):
            reddot_image = show_image.copy()
            reddot_image2 = show_image2.copy()
            next_point = next_sampled_point_f[i][j]
            homo_point = np.float64([point[0],point[1],1])
            homo_next_point = np.float64([next_point[0], next_point[1], 1])
            dist = cv2.sampsonDistance(homo_point,homo_next_point,F)
            if dist <3:
                # print(dist)
                new_point.append([point[1],point[0]])
                next_new_point.append([next_point[1],next_point[0]])
                # #
                cv2.circle(reddot_image, (int(point[0]), int(point[1])), 3, (0, 0, 255),-1)
                cv2.circle(reddot_image2,(int(next_point[0]),int(next_point[1])), 3, (0, 0, 255),-1)
                cv2.imshow("src", reddot_image)
                cv2.imshow('target', reddot_image2)
                cv2.waitKey(0)
        sampled_point_f[i] = new_point
        next_sampled_point_f[i] = next_new_point
        sampled_point[i] = new_point
        next_sampled_point[i] = next_new_point


    cv2.imshow("src", show_image)
    cv2.imshow('target',show_image2)
    cv2.waitKey(1)
elif flow_mode=="OpticalFlow" :
    source = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    target = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)
    new_sampled_point = [[] for _ in range(cnt)]
    next_sampled_point = [[] for _ in range(cnt)]
    sampled_point = [[] for _ in range(cnt)]
    pt1 = cv2.goodFeaturesToTrack(source, 500, 0.01,10)
    for i in range(pt1.shape[0]):
        p_x = int(pt1[i,0,0])
        p_y = int(pt1[i,0,1])
        if bad_removed_labels[p_y,p_x] != -1:
            sampled_point[bad_removed_labels[p_y,p_x]].append([p_y,p_x])

    # print(pt1.shape)
    # exit(0)
    for i, points in enumerate(sampled_point):
        if len(points) ==0:
            continue
        np_points = np.array(points)
        np_points = np.flip(np_points,axis=1)#x,y순서이어야함
        np_points = np_points.reshape(-1,1,2).astype(np.float32)
        # print(np_points.shape)
        # exit(0)
        pt2, status, err = cv2.calcOpticalFlowPyrLK(source, target, np_points, None)
        dst = cv2.addWeighted(images[0], 0.5, images[1], 0.5, 0)
        for j in range(pt2.shape[0]):
            if status[j, 0] == 0:  # status = 0인 것은 제외, 잘못 찾은 것을 의미
                continue
            if err[j,0] >20:#에러차이가 너무 심한 점
                continue
            if int(pt2[j,0,1]) <0 or int(pt2[j,0,1]) >=image_height-1 or int(pt2[j,0,0]) <0 or int(pt2[j,0,1]) >= image_width-1:
                continue
            cv2.circle(dst, (points[j][1],points[j][0]), 4, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(dst, (int(pt2[j,0,0]),int(pt2[j,0,1])), 4, (0, 0, 255), 2, cv2.LINE_AA)

            # pt1과 pt2를 이어주는 선 그리기
            cv2.arrowedLine(dst, (points[j][1],points[j][0]), (int(pt2[j,0,0]),int(pt2[j,0,1])), (0, 255, 0), 2)
            new_sampled_point[i].append([points[j][0],points[j][1]])
            next_sampled_point[i].append([pt2[j,0,1],pt2[j,0,0]])
        cv2.imshow('dst', dst)
        cv2.waitKey()
    sampled_point= new_sampled_point

#4-1 get point 3d each image
sampled_point_3d = [[] for _ in range(cnt)]#0번은 배경에 대한 엣지입니다. depth가 적절하지않으면 버립니다.
next_sampled_point_3d = [[] for _ in range(cnt)]#0번은 배경에 대한 엣지입니다. depth가 적절하지않으면 버립니다.

sample_point_check_image = np.zeros((image_height,image_width,3),dtype=np.uint8)
for i in range(cnt):
    # 레이블이 같은 영역에 랜덤한 색상 적용 ---②
    sample_point_check_image[labels==i] =  [int(j) for j in np.random.randint(0,255, 3)]
for idx, samp_point in enumerate(sampled_point):
    if idx ==0:
        continue
    good_point =[]
    for p_id, point in enumerate(samp_point):#샘플링된 포인트의 필터링된 3차원위치를 찾습니다.
        sample_point_check_image = cv2.circle(sample_point_check_image,(int(point[1]),int(point[0])),2,(0,0,255),1,0)
        if flow_mode == "denseSIFT":
            next_point_2d = [point[0]+flow[int(point[0]/step_size),int(point[1]/step_size),1]*step_size,point[1]+flow[int(point[0]/step_size),int(point[1]/step_size),0]*step_size]
        elif flow_mode == "denseFlow":
            next_point_2d = [point[0] + flow[int(point[0]),int(point[1]), 1],
                             point[1] + flow[int(point[0]),int(point[1]), 0]]
            if (next_point_2d[0] <0 or next_point_2d[1] <0) or (next_point_2d[0] >=image_height-1 or next_point_2d[1] >=image_width-1):
                continue#optical flow가
        else :##Orb를 사용하거나 optical flow를 사용하는경우
            next_point_2d = next_sampled_point[idx][p_id]
        point_3d = pixel2Point3D(point[0], point[1], dmaps[0][int(point[0]), int(point[1])])
        if dmaps[1][int(next_point_2d[0]), int(next_point_2d[1])] ==0:
            continue
        else:
            good_point.append(point)
            next_sampled_point[idx].append(next_point_2d)
            next_point_3d = pixel2Point3D(next_point_2d[0], next_point_2d[1],
                                          dmaps[1][int(next_point_2d[0]), int(next_point_2d[1])])#과연 0이나올까?
            sampled_point_3d[idx].append(point_3d)
            next_sampled_point_3d[idx].append(next_point_3d)
    sampled_point[idx] = good_point
    # print("-------",idx,"--------")
    # print("기존 2D점들 ", len(samp_point))
    # print("남은 3차원점들 ", len(next_sampled_point_3d[idx]))
# cv2.imshow("sample check image", sample_point_check_image)
# cv2.waitKey(0)

distance_graph = [[-1 for _ in range(cnt) ] for _ in range(cnt)]#두 요소사이에 존재하는 엣지 리스트
#4-2 원래프레임, 다음프레임 에서 이웃간에 거리비교
distance_edge_image = np.zeros((image_height,image_width,3), dtype=np.uint8)
sum_image = np.zeros((image_height,image_width), dtype=np.uint8)# 의미있는 섹션만 흰색으로 하는 라벨 이미지
section_image = np.ones((image_height,image_width,3), dtype=np.uint8)*255#최종적인 결과를 저장하기 위한 이미지
for i, row_list in enumerate(component_graph):
    if i==0:
        continue
    for j, col_value in enumerate(row_list[1:i+1],start=1):

        if col_value :#만약에 둘이 엣지관계라면 두 점들 사이의 관계를 행렬로 저장해둔다.
            i_point_length = len(sampled_point_3d[i])
            j_point_length = len(sampled_point_3d[j])
            if i_point_length ==0 or j_point_length ==0:
                continue
            node_1_point = sampled_point_3d[i]
            node_2_point = sampled_point_3d[j]
            next_node_1_point = next_sampled_point_3d[i]
            next_node_2_point = next_sampled_point_3d[j]

            distance_metric = np.zeros((i_point_length,j_point_length))
            distance_metric_2 = np.zeros((i_point_length, j_point_length))
            # print("그래프 상의 노드",i, j)
            # print("실제 점 좌표")
            # print(node_1_point)
            # print(node_2_point)
            for s_i in range(i_point_length):
                for s_j in range(j_point_length):
                    node_1_point_s_i = node_1_point[s_i]
                    node_2_point_s_j = node_2_point[s_j]
                    next_node_1_point_s_i = next_node_1_point[s_i]
                    next_node_2_point_s_j = next_node_2_point[s_j]
                    distance_metric[s_i,s_j] = np.sqrt(np.sum((node_1_point_s_i-node_2_point_s_j)**2))
                    distance_metric_2[s_i, s_j] = np.sqrt(np.sum((next_node_1_point_s_i - next_node_2_point_s_j)**2))#거리차이로는 애매한걸 수도 있다.
                    # print(i,j,"----------------")
                    # print(distance_metric[s_i,s_j]-distance_metric_2[s_i, s_j])
            # if not fast_measure_mode:
            #     if not sigma_mode:
            #         geometry_cost = np.mean(np.exp(-(distance_metric-distance_metric_2)**2/sigma))#거리가 크면 작아짐
            #     else :
            #         geometry_cost =-(distance_metric-distance_metric_2)**2
            # else :
            i_mean_diff = np.mean(np.abs(distance_metric - distance_metric_2), axis=1)
            j_mean_diff = np.mean(np.abs(distance_metric - distance_metric_2), axis=0)
            num_move_I = np.where(i_mean_diff>out_range)[0].shape[0]#5cm 이상 이동할 경우 동적이다.
            num_move_J = np.where(j_mean_diff>out_range)[0].shape[0] # 5cm 이상 이동할 경우 동적이다.
            geometry_cost =(num_move_J+num_move_I)/(i_point_length+j_point_length)#이게 일정이상 많으면 동적인 점이다.
            # print(geometry_cost)
            distance_graph[i][j] = geometry_cost#다른 이웃한 점과의 차이가 클 것이다.

#4-3 나중프레임에서 이웃간에 거리비교
#4-4 거리차이가 유지되는지 비교 (어떻게 비교할까?)
if not sigma_mode:
    for i, row_list in enumerate(distance_graph):
        if i==0:
            continue
        for j, col_value in enumerate(row_list[1:i+1],start=1):
            if col_value>0:
                color = [255-int(255*col_value),0,int(255*col_value)]#빨가면 유사도가 높다
                for k in component_edge_graph[i][j]:
                    distance_edge_image[edge_set[k].y,edge_set[k].x] = color
            elif len(component_edge_graph[i][j])>0:#매칭 관계는 있지만 바깥이랑 이어진경우 혹은 0인점이랑 이어진경우
                color = [0, 255, 0]
                for k in component_edge_graph[i][j]:
                    distance_edge_image[edge_set[k].y,edge_set[k].x] = color
    # cv2.imshow("sample point", sample_point_check_image)
    cv2.imshow("edge result", distance_edge_image)
    cv2.waitKey(0)
elif fast_measure_mode:
    cv2.namedWindow("rate_test_window")
    cv2.createTrackbar("rate", "rate_test_window", 90, 100, lambda x: x)
    past_cnt = 0
    component_graph_woRed= component_graph.copy()
    while cv2.waitKey(1) != ord('q'):
        rate = cv2.getTrackbarPos("rate", "rate_test_window")
        for i, row_list in enumerate(distance_graph):
            if i == 0:
                continue
            for j, col_value in enumerate(row_list[1:i + 1], start=1):
                #두 엣지쌍에 대해서 검사를 함.
                if len(sampled_point_3d[i])<3 or len(sampled_point_3d[j])<3:
                    #매칭쌍이 부족한 부분은 노란색으로 채움. 보면 배경은 매칭쌍이 부족해서 해당부분은 0으로 일단채움.
                    color = [0, 228, 255]
                    if len(sampled_point_3d[i]) < 3 and len(sampled_point_3d[j]) < 3:
                        continue  # 만약에 둘다 depth가 0이라면, 그냥 넘어간다.
                    elif len(sampled_point_3d[i]) >3:#만약에 i의 샘플링된 점이 많다면 이걸 색칠한다.
                        sum_image[labels==i] =255
                    else:
                        sum_image[labels == j] = 255
                    for k in component_edge_graph[i][j]:
                        distance_edge_image[edge_set[k].y, edge_set[k].x] = color
                    continue
                if col_value != -1:#col_value가 있다면
                    if col_value > rate/100:#만약에 동적인 비율이 일정 이상이라면 파란색
                        color = [255 - int(0), 0, int(0)]
                        draw_bound = True#파란색이면 경계표시를 해야한다.
                    else:
                        color = [255 - int(255), 0, int(255)]
                        draw_bound = False#빨간색이면 경계표시를 안해도 된다.
                        component_graph_woRed[i][j] = True#두 영역이 같다고 표시합니다.
                        component_graph_woRed[j][i] = True#두 영역이 같다고 표시합니다.
                    for k in component_edge_graph[i][j]:# 두 점사이의 distance edge를 칠한다.
                        distance_edge_image[edge_set[k].y, edge_set[k].x] = color
                        if not draw_bound:#파란색이면 경계표시를 해야한다.
                            sum_image[edge_set[k].y, edge_set[k].x] = 255
        cv2.imshow("sum iamge",sum_image)
        cnt, labels,stats, centroids = cv2.connectedComponentsWithStats(sum_image)
        if past_cnt !=cnt:
            past_cnt = cnt
            # labels[filtered_dmaps[0] == 0] = 0##stat는 안바뀜
            # labels[dmaps[0] > 3.5] = 0
            # section_image[dmaps[0] > 3.5] = -1
            section_image[dmaps[0] == 0] = -1
            for i in range(cnt):
                if i==0:
                    section_image[labels == i] = [0,0,0]
                else:
                    section_image[labels==i] = [int(j) for j in np.random.randint(0,255, 3)]
        #후보 섹션들을 각각 구해서 큐에 담아놔야한다. 위에서 라벨링 할때, 만약에 노랑 조건을 만족하는 점들이면 0으로한다.
        cv2.imshow("rate_test_window", distance_edge_image)
        # cv2.imshow("sample point", sample_point_check_image)
        cv2.imshow("section image",section_image)
    cv2.destroyAllWindows()

##find segment
past_cnt = 0
rate = 80
# distance_graph : 위(위)의 거리비교를 적용한 결과 그래프
for i, row_list in enumerate(distance_graph):
    if i == 0:#0번을 제외한 이유는 엣지값을 가진 클러스터이기 때문에
        continue
    for j, col_value in enumerate(row_list[1:i + 1], start=1):#0번을 제외하고 비교한면
        if len(sampled_point_3d[i])<3 or len(sampled_point_3d[j])<3:
            #샘플 갯수를 비교, 3보다 작은 경우는 denseflow의 경우에는 depth가 0인경우이다.
            #즉 두 섹션중에서 하나는 depth가 0이다.(혹은 둘다.)
            color = [0, 228, 255]
            if len(sampled_point_3d[i]) < 3 and len(sampled_point_3d[j]) < 3:
                continue  # 만약에 둘다 depth가 0이라면, 그냥 넘어간다.
            elif len(sampled_point_3d[i]) > 3:  # 만약에 i의 샘플링된 점이 많다면 이걸 색칠한다.
                sum_image[bad_removed_labels == i] = 255
            else:
                sum_image[bad_removed_labels == j] = 255
            for k in component_edge_graph[i][j]:
                distance_edge_image[edge_set[k].y, edge_set[k].x] = color
        if col_value != -1:#점사이의 거리관계 비율을 가지고있다면
            if col_value > rate/100:#만약에 동적인 비율이 일정 이상이라면 파란색
                color = [255 - int(0), 0, int(0)]
                draw_bound = True
            else:#서로 나뉘어져야하는부분들
                color = [255 - int(255), 0, int(255)]
                draw_bound = False
                #두 영역이 같을 때~ 경계도 같은 255로 해야하지않을까?
            sum_image[labels == i] = 255
            sum_image[labels == j] = 255
            for k in component_edge_graph[i][j]:
                distance_edge_image[edge_set[k].y, edge_set[k].x] = color
                if not draw_bound:  # 파란색이면 경계표시를 해야한다.
                    sum_image[edge_set[k].y, edge_set[k].x] = 255


new_cnt, labels_2,stats, centroids = cv2.connectedComponentsWithStats(sum_image)
section_image[dmaps[0] == 0] = [0,0,0]
seg_sizes = [0]
for i in range(new_cnt):
    if i==0:
        section_image[labels_2 == i] = [0,0,0]
    else:
        filtered_dmaps_for_Area = dmaps[0].copy()
        section_image[labels_2==i] = [int(j) for j in np.random.randint(0,255, 3)]#[255,0,0]#
        filtered_dmaps_for_Area[labels_2 !=i] = 0
        seg_size = np.sum(filtered_dmaps_for_Area)
        seg_sizes.append(seg_size)
max_ind = seg_sizes.index(max(seg_sizes))
section_image[labels_2==max_ind] =[0,0,255]#가장넓은 섹션을 붉게 색칠한다.


#후보 섹션들을 각각 구해서 큐에 담아놔야한다. 위에서 라벨링 할때, 만약에 노랑 조건을 만족하는 점들이면 0으로한다.
new_sam_point= [[] for _ in range(new_cnt)]
next_new_sam_point = [[] for _ in range(new_cnt)]
for seg_idx, points in enumerate(sampled_point):
    for inner_idx, point in enumerate(points):#세그먼트 내에 점들에 대해서
        new_idx = labels_2[int(point[0]),int(point[1])]
        if new_idx !=0:
            new_sam_point[new_idx].append([point[0],point[1]])
            # section_image = cv2.circle(section_image,(int(point[1]),int(point[0])),2,(255,0,0),1, cv2.LINE_AA)
            try :
                next_new_sam_point[new_idx].append([next_sampled_point[seg_idx][inner_idx][0],
                                                next_sampled_point[seg_idx][inner_idx][1]])
            except IndexError:
                print(new_idx)
                print(new_cnt)
                print("리스트 크기 : ",len(next_sampled_point))
                print("실제 작은 인덱스 크기 : ",len(sampled_point[seg_idx]))
                print("작은 인덱스 크기 : ",len(next_sampled_point[seg_idx]))
                print(inner_idx)
                exit(0)

sampled_point_3d = [[] for _ in range(new_cnt)]#0번은 배경에 대한 엣지입니다. depth가 적절하지않으면 버립니다.
next_sampled_point_3d = [[] for _ in range(new_cnt)]#0번은 배경에 대한 엣지입니다. depth가 적절하지않으면 버립니다.

for idx, samp_point in enumerate(new_sam_point):
    if idx ==0:
        continue
    for p_id, point in enumerate(samp_point):#샘플링된 포인트의 필터링된 3차원위치를 찾습니다.
        next_point_2d = next_new_sam_point[idx][p_id]
        point_3d = pixel2Point3D(point[0], point[1], dmaps[0][int(point[0]), int(point[1])])
        if dmaps[0][int(point[0]), int(point[1])] ==0:
            continue
        if dmaps[1][int(next_point_2d[0]), int(next_point_2d[1])] ==0:
            continue
        else:
            next_point_3d = pixel2Point3D(next_point_2d[0], next_point_2d[1],
                                          dmaps[1][int(next_point_2d[0]), int(next_point_2d[1])])#과연 0이나올까?
            sampled_point_3d[idx].append(point_3d)
            next_sampled_point_3d[idx].append(next_point_3d)

## 같은 부분을 찾는부분
for idx, samp_point in enumerate(sampled_point_3d):
    if idx == 0 or idx == max_ind:#0은 엣지입니다.~
        continue
    # print("max id : ",max_ind, " idx : ",idx)
    i_point_length = len(sampled_point_3d[max_ind])
    j_point_length = len(sampled_point_3d[idx])
    if i_point_length == 0 or j_point_length == 0:
        continue
    node_1_point = sampled_point_3d[max_ind]
    node_2_point = sampled_point_3d[idx]
    next_node_1_point = next_sampled_point_3d[max_ind]
    next_node_2_point = next_sampled_point_3d[idx]
    distance_metric = np.zeros((i_point_length, j_point_length))
    distance_metric_2 = np.zeros((i_point_length, j_point_length))
    for s_i in range(i_point_length):
        for s_j in range(j_point_length):
            node_1_point_s_i = node_1_point[s_i]
            node_2_point_s_j = node_2_point[s_j]
            next_node_1_point_s_i = next_node_1_point[s_i]
            next_node_2_point_s_j = next_node_2_point[s_j]
            distance_metric[s_i, s_j] = np.sqrt(np.sum((node_1_point_s_i - node_2_point_s_j) ** 2))
            distance_metric_2[s_i, s_j] = np.sqrt(np.sum((next_node_1_point_s_i - next_node_2_point_s_j) ** 2))  # 거리차이로는 애매한걸 수도 있다.
    # print(distance_metric- distance_metric_2)
    i_mean_diff = np.mean(np.abs(distance_metric - distance_metric_2), axis=1)
    j_mean_diff = np.mean(np.abs(distance_metric - distance_metric_2), axis=0)
    # print(np.abs(distance_metric - distance_metric_2).astype(np.float32))
    # print("--------평균----------")
    num_move_I = np.where(i_mean_diff > out_range)[0].shape[0]  # 5cm 이상 이동할 경우 동적이다.
    num_move_J = np.where(j_mean_diff > out_range)[0].shape[0]  # 5cm 이상 이동할 경우 동적이다.
    # print(i_point_length,j_point_length)
    # print(i_mean_diff)
    # print(j_mean_diff)
    geometry_cost = (num_move_J + num_move_I) / (i_point_length + j_point_length)  # 이게 일정이상 많으면 동적인 점이다.
    # geometry_cost = num_move_J/j_point_length
    # print(idx)
    # print(geometry_cost)
    if geometry_cost < rate/100: #같은지점이다.
    # # if idx ==4:
        section_image[labels_2 == idx] = [0, 0, 255]

cv2.imshow("sum image",sum_image)
cv2.imshow("third image", distance_edge_image)
cv2.imshow("forth image",section_image)
cv2.waitKey(0)
