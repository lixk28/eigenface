from eigenface import *

def test3(average, eigface_mat, train_face_proj):
    '''
        测试函数3:
        测试单个人脸照片识别的结果
    '''
    for i in range(1, 9):
        test_face = cv.imdecode(np.fromfile('capface/' + 'us' + str(i) + '.jpg', dtype=np.uint8), -1)
        test_face = cv.resize(test_face, (256, 256), interpolation=cv.INTER_NEAREST)
        test_face = cv.cvtColor(test_face, cv.COLOR_BGR2GRAY)
        test_face = cv.equalizeHist(test_face)
        test_face = test_face.reshape((65536, 1)) - average
        test_face_proj = eigface_mat @ test_face
        recog_name = KNN_classifier(train_face_proj, test_face_proj, 5)
        print(recog_name)

if __name__ == "__main__":
    # 图像预处理
    img_process()
    # 读取处理后的图像，得到人脸列向量组成的矩阵和均值向量
    train_face = img2mat()
    train_face, average = normalize(train_face)

    # 计算特征脸，得到特征脸列向量组成的矩阵
    eigface_mat = eigenface(train_face, eig_vec_number)
    # 作投影，将人脸投影到特征空间
    train_face_proj = eigface_mat @ train_face
    # 用单个人脸照片进行测试
    test3(average, eigface_mat, train_face_proj)
    