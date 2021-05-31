from eigenface import *

def test2(average, eigface_mat, train_face_proj):
    '''
        测试函数2:
        对测试集的每张人脸进行检测，判断识别出的名字与其真实名字是否相同，
        相同则识别数加1，然后将识别选取的特征向量个数、识别正确的个数、识别率输出到文件中保存，
        最后将结果进行可视化
    '''
    # 识别正确的个数
    recog_number = 0
    for name in names:
        for i in test_number_list:
            test_face_proj = read_test_face(average, eigface_mat, 'proface/' + name + str(i) + '.jpg')
            recog_name = classifier(train_face_proj, test_face_proj)
            if recog_name == name:
                recog_number += 1
    # 识别率
    recog_rate = recog_number / (people_number * test_img_number)
    data2 = open("assets/data", 'a')
    data2.write("{}\t{}\t{:.5f}\n".format(eigface_mat.shape[0], recog_number, recog_rate))
    data2.close()

if __name__ == "__main__":
    # 图像预处理
    img_process()
    # 读取处理后的图像，得到人脸列向量组成的矩阵和均值向量
    train_face = img2mat()
    train_face, average = normalize(train_face)

    # 选取不同个数的特征向量，对识别结果的影响
    eig_vec_number_list = list(range(20, 80, 5))
    for eig_vec_number in eig_vec_number_list:
        # 计算特征脸，得到特征脸列向量组成的矩阵
        eigface_mat = eigenface(train_face, eig_vec_number)
        # 作投影，将人脸投影到特征空间
        train_face_proj = eigface_mat @ train_face
        # 用测试集人脸进行测试
        test2(average, eigface_mat, train_face_proj)
    