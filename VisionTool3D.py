import numpy as np
import cv2
from Edge import *
def pixel2Point3D(i,j,z):#2차원 좌표를 뎁스를 기반으로 하여 3차원좌표로 바꿉니다.

    fx = 544.2582548211519  # focal length x
    fy = 546.0878823951958  # focal length y
    cx = 326.8604521819424  # optical center x
    cy = 236.1210149172594  # optical center y
    # cx = 320.1
    # cy = 247.6
    # fx = 535.4
    # fy = 539.2
    # radial distortion coefficient
    X = (j-cx)*z/fx
    Y = (i-cy) * z / fy
    Z =z


    return np.array([X, Y, Z])

def get_norm_vec(depth_image,i,j,max_height,max_width):
    if (i <1 or j <1) or(i >=max_height-1 or j >=max_width-1):
        return False, []
    else:
        return_i_vec = -((depth_image[i+1,j]) - (depth_image[i-1,j]))/2.# 5000값에 1meter
        return_j_vec = -((depth_image[i, j+1]) - (depth_image[i, j-1])) / 2.
        return_vec = np.array([return_j_vec,return_i_vec,1.0])# x, y, z
        # V_ver = pixel2Point3D(i,j-1,depth_image[i, j-1])-pixel2Point3D(i,j+1,depth_image[i, j+1])
        # V_hor = pixel2Point3D(i-1, j , depth_image[i-1, j]) - pixel2Point3D(i+1, j, depth_image[i+1, j])
        #
        # cross = np.cross(V_ver,V_hor)
        norm_vec = np.linalg.norm(return_vec)

        if norm_vec ==0:
            return True, (return_vec).tolist()
        # else:
        #     print(i, j)
        #     print(V_ver)
        #     print(V_hor)
        #     print(cross)
        #     print((cross/norm_vec))
        return True, (return_vec/norm_vec).tolist()

def is_edge(arg_i,arg_j,norm_vec,image_height,image_width, dmap):
    '''

    :param arg_i:
    :param arg_j:
    :param norm_vec:
    :param image_height:
    :param image_width:
    :param dmap:
    :return: phi_d : 거리차이가 많이 나는가? phi_c : 90도이상 꺽였는가?
    '''
    NeighborNum = 8
    N_Way = [[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1],[1,0],[1,1]]
    for i in range(NeighborNum):
        N_i = arg_i+ N_Way[i][0]
        N_j = arg_j +N_Way[i][1]
        if (N_i <1 or N_j <1) or(N_i >=image_height-1 or N_j >=image_width-1): #보통이러면 테두리점은 아님
            return False, 0, 0
    test_V = pixel2Point3D(arg_i,arg_j,dmap[arg_i,arg_j])
    test_N = norm_vec[arg_i,arg_j]
    phi_d = 0
    phi_c = 1
    for i in range(NeighborNum):
        N_i = arg_i + N_Way[i][0]
        N_j = arg_j + N_Way[i][1]
        test_d =(pixel2Point3D(N_i, N_j, dmap[N_i, N_j])-test_V)@test_N.T
        test_d_abs = abs(test_d)

        if phi_d < test_d_abs:
            phi_d = test_d_abs

        if test_d >0:
            test_c = 1
        else :
            test_c = test_N@norm_vec[N_i,N_j].T
        if test_c < phi_c:
            phi_c = test_c
    return True, phi_d, phi_c

def Depth_preprocess(raw_depth_image):
    '''
    뎁스이미지를 나누고, 3.5m밖에 Depth를 버립니다.

    '''
    for i in range(len(raw_depth_image)):
        raw_depth_image[i] = raw_depth_image[i].astype(np.float64)/5000
        raw_depth_image[i][raw_depth_image[i] > 3.5] = 0  # use metric under 3.5m
    filtered_dmaps = [cv2.bilateralFilter(image.astype(np.float32), 3, 4, 4) for image in raw_depth_image]
    # filtered_dmaps = [image  for image in raw_depth_image]
    return filtered_dmaps

def Depth_preprocess2(raw_depth_image):
    '''
    뎁스이미지를 나누고, 3.5m밖에 Depth를 버립니다.
    '''
    for i in range(len(raw_depth_image)):
        raw_depth_image[i] = raw_depth_image[i].astype(np.float64)/1000
        # raw_depth_image[i][raw_depth_image[i] > 5] = 0  # use metric under 3.5m
    filtered_dmaps = [cv2.bilateralFilter(image.astype(np.float32), 7, 0.1, 5) for image in raw_depth_image]
    # filtered_dmaps = [image  for image in raw_depth_image]
    return filtered_dmaps

def Edgemap_changer(phi_d_image,phi_c_image,filtered_dmaps):
    '''
    각 파라미터를 변경하여 이미지를 출력합니다.
    
    :param phi_d_image: 거리에 대한 파라미터
    :param phi_c_image: 왜곡에 대한 파라미터
    :param filtered_dmaps: 기존 depth이미지
    :return: 없음
    '''
    image_height, image_width=phi_d_image.shape
    cv2.namedWindow("test_window")
    label_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    cv2.createTrackbar("lamda", "test_window", 94, 100, lambda x: x)
    cv2.createTrackbar("tau", "test_window", 31, 500, lambda x: x)
    cv2.setTrackbarPos("lamda", "test_window", 94)
    cv2.setTrackbarPos("tau", "test_window", 31)
    past_cnt = 0
    while cv2.waitKey(1) != ord('q'):
        lambda_hat = cv2.getTrackbarPos("lamda", "test_window")
        tau = cv2.getTrackbarPos("tau", "test_window")
        temp_image = phi_d_image
        result_image = np.zeros((image_height, image_width), dtype=np.uint8)
        result_image[temp_image > tau / 1000] = 255
        result_image[phi_c_image < lambda_hat / 100] = 255
        # result_image[filtered_dmaps == 0] = 0
        cv2.imshow("test_window", result_image)
        result_image = 255 - result_image
        cnt, labels = cv2.connectedComponents(result_image)
        if past_cnt != cnt:
            past_cnt = cnt
            for i in range(cnt):
                label_image[labels == i] = [int(j) for j in np.random.randint(0, 255, 3)]
            cv2.imshow("label image", label_image)
    cv2.destroyAllWindows()
    exit(0)
    
def Get_EdgeMap(image_height,image_width,filtered_dmaps):
    '''
    엣지맵을 만들기 위한 파라미터 요소를 가져옵니다.
    :param image_height: 
    :param image_width: 
    :param filtered_dmaps: 3.5m밖에 요소가 필터링된 뎁스 이미지
    :return: 
    '''
    is_normal = np.zeros((image_height, image_width), dtype=bool)
    norm_vec = np.zeros((image_height, image_width, 3))
    phi_d_image = np.zeros((image_height, image_width))
    phi_c_image = np.zeros((image_height, image_width))
    for i in range(image_height):
        for j in range(image_width):
            norm_success, vector = get_norm_vec(filtered_dmaps,i,j,image_height,image_width)
            is_normal[i,j] = norm_success
            if norm_success:
                norm_vec[i,j] = vector
    for i in range(image_height):
        for j in range(image_width):
            edge, phi_d,phi_c = is_edge(i,j,norm_vec,image_height,image_width,filtered_dmaps)
            if edge:
                phi_d_image[i,j] = phi_d
                phi_c_image[i,j] = phi_c
    return phi_d_image, phi_c_image

def Get_SegmentedImage(image_height, image_width,phi_d_image,phi_c_image,tau,lambda_hat,filtered_dmaps):
    temp_image = phi_d_image  # +lambda_hat*phi_c_image
    result_image = np.zeros((image_height, image_width), dtype=np.uint8)

    result_image[temp_image > tau / 1000] = 255
    result_image[phi_c_image < lambda_hat / 100] = 255
    result_image[filtered_dmaps == 0] = 0
    result_image = 255 - result_image
    cnt, labels = cv2.connectedComponents(result_image)
    return cnt, labels

def Find_VoronoiEdge(image_height, image_width,labels,cnt):
    graph_idx_image = np.full((image_height, image_width), -2,dtype=np.int64)  # 엣지이면 0임
    graph_idx_image[labels != 0] = -1  # 엣지가 아니면 전부 -1로 정해짐
    component_edge_graph = [[[] for _ in range(cnt)] for _ in range(cnt)]  # 두 요소사이에 존재하는 엣지 리스트
    component_graph = [[False for _ in range(cnt)] for _ in range(cnt)]#일단 엣지요소 포함해서 만든다. 0번안쓰면되자나
    edge_point = np.where(labels == 0)
    edge_point_y = edge_point[0]
    edge_point_x = edge_point[1]
    edge_set = []
    global_idx = 0
    checking_edge_set = []
    boundary_image = np.zeros((image_height, image_width),
                              dtype=np.uint8)  # 라벨링이 된 점은 -1, 검사되어야하는 점은 0, 이미 인접정보가 확인된 점은 1
    voronoi_image = np.zeros((image_height,
                              image_width))  # 라벨링이 된 점은 -1, 검사되어야하는 점은 0, 이미 인접정보가 확인된 점은 1
    voronoi_image[labels != 0] = -1  # 엣지가 아니면 전부 -1로 정해짐
    for i in range(edge_point_y.shape[0]):
        y = edge_point_y[i]
        x = edge_point_x[i]
        N_Way = [[0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]]
        past_label = -1
        for j in range(8):#시작할때부터 이미 두 점에 대한 경계면 어떡하지?
            N_i = y + N_Way[j][0]
            N_j = x + N_Way[j][1]
            if (N_i < 1 or N_j < 1) or (N_i >= image_height - 1 or N_j >= image_width - 1):  # 보통이러면 테두리점은 아님
                continue
            if labels[N_i, N_j] > 0:  # 0보다 크다 혹은 요소이다.
                if past_label > 0:
                    component_graph[past_label][labels[N_i, N_j]] = True
                    component_graph[labels[N_i, N_j]][past_label] = True
                    component_edge_graph[labels[N_i, N_j]][past_label].append(global_idx-1)
                    component_edge_graph[past_label][labels[N_i, N_j]].append(global_idx-1)
                    continue
                edge_set.append(Edge_node(global_idx, -1, y, x, labels[N_i, N_j], True))
                checking_edge_set.append(global_idx)
                graph_idx_image[y, x] = global_idx
                global_idx += 1
                voronoi_image[y, x] = 1
                boundary_image[y, x] = 255

    # cv2.imshow("boundary",boundary_image)
    # cv2.waitKey(0)
    # 3-3 do voronoi search
    N_Way = [[0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]]
    while True:
        if len(checking_edge_set) == 0:
            break
        new_check_edge_set = []
        for edge_idx in checking_edge_set:
            c_y, c_x = edge_set[edge_idx].get_point()
            for j in range(8):  # 주변 이웃점에 대해서 검사합니다.
                N_i = c_y + N_Way[j][0]
                N_j = c_x + N_Way[j][1]

                if voronoi_image[N_i, N_j] == 1:  # 1이라는건 주변에 확인된 엣지가 존재한다는 것
                    if edge_set[graph_idx_image[N_i, N_j]].label == edge_set[edge_idx].label:  ##이웃점을 검사하니 같은 요소에서 나옴.
                        continue  # 일단 넘어감
                    elif edge_set[graph_idx_image[N_i, N_j]].label != edge_set[edge_idx].label:  ##이웃점을 검사하니 다른 요소에서 나옴
                        if edge_set[edge_idx].endlabel != -1:  #이미 현재 엣지가 다른 라벨에 이어졌을 경우
                            continue

                        edge_set[edge_idx].endlabel = edge_set[int(graph_idx_image[N_i, N_j])].label
                        # 요소간에 엣지관계를 등록합니다.
                        component_graph[edge_set[edge_idx].label][edge_set[int(graph_idx_image[N_i, N_j])].label] = True
                        component_graph[edge_set[int(graph_idx_image[N_i, N_j])].label][edge_set[edge_idx].label] = True

                        # 현재 점을 두 점사이의 엣지관계에 추가하고, 부모 엣지를 업데이트 및 추가합니다.
                        component_edge_graph[edge_set[edge_idx].label][edge_set[edge_idx].endlabel].append(edge_idx)
                        component_edge_graph[edge_set[edge_idx].endlabel][edge_set[edge_idx].label].append(edge_idx)
                        update_label = edge_set[edge_idx].idx
                        while True:
                            # 기존 노드의 부모를 가져옵니다.
                            is_have, parent_idx = edge_set[update_label].get_parentIdx()
                            if is_have:
                                if edge_set[parent_idx].endlabel != -1:
                                    break
                                edge_set[parent_idx].endlabel = edge_set[update_label].endlabel  # 반대편 라벨을 등록해줍니다.
                                component_edge_graph[edge_set[update_label].label][
                                    edge_set[update_label].endlabel].append(
                                    parent_idx)  # 해당되는 엣지점을 등록합니다. 여러번들어갈 수 있음.
                                component_edge_graph[edge_set[update_label].endlabel][
                                    edge_set[update_label].label].append(
                                    parent_idx)  # 해당되는 엣지점을 등록합니다.
                                update_label = parent_idx  # 이다음 업데이트할 부모노드를 찾습니다.
                            else:
                                break
                        if edge_set[int(graph_idx_image[N_i, N_j])].endlabel == -1:
                            edge_set[int(graph_idx_image[N_i, N_j])].endlabel = edge_set[edge_idx].label
                            component_edge_graph[edge_set[edge_idx].label][edge_set[edge_idx].endlabel].append(
                                graph_idx_image[N_i, N_j])
                            component_edge_graph[edge_set[edge_idx].endlabel][edge_set[edge_idx].label].append(
                                graph_idx_image[N_i, N_j])
                            update_label = edge_set[edge_idx].idx
                            while True:
                                # 기존 노드의 부모를 가져옵니다.
                                is_have, parent_idx = edge_set[update_label].get_parentIdx()
                                if is_have:
                                    if edge_set[parent_idx].endlabel != -1:
                                        break
                                    edge_set[parent_idx].endlabel = edge_set[update_label].endlabel  # 반대편 라벨을 등록해줍니다.
                                    component_edge_graph[edge_set[update_label].label][
                                        edge_set[update_label].endlabel].append(
                                        parent_idx)  # 해당되는 엣지점을 등록합니다. 여러번들어갈 수 있음.
                                    component_edge_graph[edge_set[update_label].endlabel][
                                        edge_set[update_label].label].append(
                                        parent_idx)  # 해당되는 엣지점을 등록합니다.
                                    update_label = parent_idx  # 이다음 업데이트할 부모노드를 찾습니다.
                                else:
                                    # print(edge_set[edge_idx].endlabel)
                                    break
                        # print("find")

                elif voronoi_image[N_i, N_j] == 0:  # 처음보는 엣지임
                    # print("새로운 엣지 등록")

                    voronoi_image[N_i, N_j] = 1  # 이제 검사한 엣지가 된거임
                    # 부모노드를 등록 및 노드 초기화
                    edge_set.append(Edge_node(global_idx, edge_idx, N_i, N_j, edge_set[edge_idx].label, False))
                    new_check_edge_set.append(global_idx)  # 현재 노드의 인덱스를 넣어버림

                    graph_idx_image[N_i, N_j] = global_idx
                    global_idx += 1
        checking_edge_set = new_check_edge_set.copy()
    return component_edge_graph,edge_set, component_graph
