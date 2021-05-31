import numpy as np
import cv2 as cv
import random

names = ['陈恩婷', '陈铭希', '陈禹翰', '陈至雪', '陈志和', \
         '方梓健', '傅子豪', '高飞扬', '郭羽翔', '何鸿荣', \
         '何雪萌', '花星宇', '黄玟瑜', '黄泽宇', '解元鸿', \
         '康文生', '兰琦函', '李明禧', '李鹏艳', '李雪堃', \
         '李钰', '梁飞垚', '梁恒中', '梁睿凯', '林雁纯', \
         '林云涛', '刘海景', '刘皓青', '刘祺翰', '刘拓', \
         '刘祥宇', '刘宇轩', '罗冠华', '罗浚艺', '罗雨童', \
         '吕文禧', '马浩宇', '马森洋', '潘思晗', '彭瑞洲', \
         '綦袁璨然', '秦一丹', '邱煜炜', '葉珺明', '苏达威',\
         '苏铁强', '苏妍文', '孙晨景', '唐晨轩', '汪妍', \
         '王昌远', '王晶', '王玺侗', '王筝', '王郅成', \
         '韦媛馨', '吴天', '伍仁杰', '夏丕浪', '向鼎', \
         '萧锘汶', '肖浩慧', '肖翎予', '谢扬', '熊俊泽', \
         '杨可', '杨明', '杨荣杰', '杨伟达', '杨欣', \
         '叶光', '叶汇亘', '易辉轩', '袁浩华', '张栋瑛', \
         '张钺奇', '张家豪', '张文琪', '张小仪', '张桢怡', \
         '张子玉', '郑梁翰', '钟芳婷', '钟裕茗', '钟泽森', \
         '仲义龙', '朱德朋']

number_list = list(range(1, 10))
train_number_list = random.sample(range(1, 9), 7)
test_number_list = list(set(number_list) ^ set(train_number_list))

people_number = len(names)  # 人的个数
train_img_number = 7        # 作为训练集的照片的个数
test_img_number = 2         # 作为测试集的照片的个数
eig_vec_number = 40         # 选取的特征向量的个数

def img_process():
    '''
        1. 将图像放缩到 256x256 大小
        2. 将图像灰度化、直方图均衡化
        3. 将处理好后的图片保存
    '''
    img_cnt = 0
    for name in names:
        for img_cnt in range(1, 10):
            # 读取原始图片
            img_raw = cv.imdecode(np.fromfile("rawface/" + name + str(img_cnt) + '.jpg', dtype=np.uint8), -1)
            # 将图片放缩为 256x256 大小
            img_resize = cv.resize(img_raw, (256, 256), interpolation=cv.INTER_NEAREST)
            # 对图像进行灰度化和直方图均衡化处理
            img_gray = cv.cvtColor(img_resize, cv.COLOR_BGR2GRAY)
            img_processed = cv.equalizeHist(img_gray)
            # 将图片保存到指定目录
            cv.imencode('.jpg', img_processed)[1].tofile("proface/" + name + str(img_cnt) + '.jpg')

def img2mat():
    '''
        读取处理后的图片，然后依次将每张 256x256 的图片转换为 65536x1 的列向量，
        作为 train_mat 的列向量，填充完整个 train_mat
    '''
    train_mat = np.zeros((65536, people_number * train_img_number))
    face_cnt = 0
    for name in names:
        for train_number in train_number_list:
            train_face = cv.imdecode(np.fromfile("proface/" + name + str(train_number) + '.jpg', dtype=np.uint8), -1)
            train_mat[:, face_cnt:face_cnt+1] = train_face.reshape(65536, 1)
            face_cnt += 1
    return train_mat

def normalize(train_mat):
    '''
        将 train_mat 的每一列减去均值向量 average
    '''
    # 计算 train_mat 每一行的均值，即每个维度 x_i 的均值
    average = np.mean(train_mat, axis = 1).reshape((65536, 1))
    # 保存平均脸
    averface = average.reshape((256, 256))
    cv.imencode('.jpg', averface)[1].tofile("averface.jpg")
    # 将每个数据点（人脸）减去均值
    for j in range(people_number * train_img_number):
        train_mat[:, j:j+1] -= average
    return train_mat, average

def eigenface(train_mat, eig_vec_number):
    '''
        1. 计算 A.T @ A 的特征值和特征向量，再用 A @ u 得到协方差矩阵的特征向量 v
        2. 对特征向量 v 组成的矩阵的每一列（即每个特征脸）进行单位化
        3. 返回 eigface_mat.T
    '''
    # 计算协方差矩阵的特征值和特征向量
    # 注意到先计算的是 train_mat^T @ train_mat
    cov_mat = train_mat.T @ train_mat
    eig_val, eig_vec = np.linalg.eig(cov_mat)

    # 得到特征值最大的前 eig_vec_number 个特征向量 top_eig_vecs 组成的矩阵
    sorted_indices = np.argsort(eig_val)
    top_eig_vecs = eig_vec[:, sorted_indices[:-eig_vec_number-1:-1]]

    # 得到特征脸矩阵，每一列是一个特征脸
    # 这里得到的才是协方差矩阵的特征向量
    eigface_mat = train_mat @ top_eig_vecs

    for i in range(eig_vec_number):
        # 将所有的特征向量单位化
        eigface_mat[:, i:i+1] = eigface_mat[:, i:i+1] / np.linalg.norm(eigface_mat[:, i:i+1])

        # 作灰度转换，并保存40个特征脸
        eigface = eigface_mat[:, i]
        eigface = eigface * (255 / np.max(eigface))
        eigface = eigface.reshape((256, 256))
        for row in range(256):
            for col in range(256):
                if eigface[row][col] < 0:
                    eigface[row][col] = 0
                elif eigface[row][col] > 255:
                    eigface[row][col] = 255
                else:
                    eigface[row][col] = np.uint8(eigface[row][col])
        cv.imencode('.jpg', eigface)[1].tofile("eigface/" + str(i+1) + '.jpg')
    
    # 返回的 eigface 的转置，方便显示特征脸
    return eigface_mat.T

def read_test_face(average, eigface_mat, path):
    '''
        将测试集人脸向量单位化，再投影到特征空间
    '''
    test_face = cv.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    test_face = test_face.reshape((65536, 1)) - average
    test_face_proj = eigface_mat @ test_face
    return test_face_proj

def KNN_classifier(train_face_proj, test_face_proj, k):
    dist = np.zeros(people_number * train_img_number)
    # 计算 train_face_proj 每个列向量与 test_face_proj 的欧氏距离
    for i in range(people_number * train_img_number):
        dist[i] = np.linalg.norm(train_face_proj[:, i:i+1] - test_face_proj)
    # 将 dist 从小到大排序，得到每张人脸
    sorted_indices = np.argsort(dist)
    classCount = {}
    for i in range(k):
        votename = names[int(sorted_indices[i] / train_img_number)]
        classCount[votename] = classCount.get(votename, 0) + 1
    sorted(classCount.items(), key=lambda item:item[1])
    print(classCount)
    
    maxCount = -1
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            ans = key
    return ans

def classifier(train_face_proj, test_face_proj):
    '''
        计算单个测试人脸向量、与训练集的每张人脸向量之间的欧氏距离
        认为与之距离最小的人的名字是识别出来的人，并返回它
    '''
    dist = np.zeros(people_number * train_img_number)
    # 计算 train_face_proj 每个列向量与 test_face_proj 的欧氏距离
    for i in range(people_number * train_img_number):
        dist[i] = np.linalg.norm(train_face_proj[:, i:i+1] - test_face_proj)
    # 将 dist 从小到大排序，得到每张人脸
    sorted_indices = np.argsort(dist)
    return names[int(sorted_indices[0] / train_img_number)]

