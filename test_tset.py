from eigenface import *

def test1(average, eigface_mat, train_face_proj):
    '''
        测试函数1:
        对测试集的每张人脸进行检测，判断识别出的名字与其真实名字是否相同，
        相同则识别数加1，不相同则输出错误识别的信息，最后输出识别率
    '''
    recog_number = 0
    for name in names:
        for i in test_number_list:
            test_face_proj = read_test_face(average, eigface_mat, 'proface/' + name + str(i) + '.jpg')
            recog_name = classifier(train_face_proj, test_face_proj)
            if recog_name == name:
                recog_number += 1
            else:
                print("mistake " + name + " for " + recog_name)
    # 输出识别率
    recog_rate = recog_number / (people_number * test_img_number)
    print("{:.2%}".format(recog_rate))

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
    # 用测试集人脸进行测试
    test1(average, eigface_mat, train_face_proj)
    