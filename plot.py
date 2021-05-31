import matplotlib.pyplot as plt

if __name__ == "__main__":
    data2 = open("assets/data", 'r')
    eig_vec_number = []
    # recog_number = []
    recog_rate = []
    for line in data2:
        data = line.strip().split()
        eig_vec_number.append(int(data[0]))
        # recog_number = int(data[1])
        recog_rate.append(float(data[2]))
    plt.plot(eig_vec_number, recog_rate)
    plt.savefig("assets/result.png")
    plt.show()