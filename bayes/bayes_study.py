def redictp(X, y, a, b):
    """
    :param X:当前天气对应的矩阵
    :param y: 对应后一天的天气
    :param a: 第二天天气的值
    :param b: 当天天气的值
    :return: 所求当天天气值为b时，第二天天气为a的概率
    """
    table = [x[0] for x in X]
    table.append(y[-1])
    result_kind = []
    for each in table:
        if each not in result_kind:
            result_kind.append(each)
    # print(type(table))
    pa = table.count(a)/len(table)
    # 降水概率为40%
    pb = table.count(b)/len(table)
    # 阴天的概率为20%
    # print(pa, pb)
    pba_count = 0
    for i in range(0,len(X)):
        if X[i][0] == b and y[i] == a:
            pba_count += 1
    if pa != 0:
        pba = pba_count/ (pa * len(table))
        # print(pba)
    else:
        pba = 0
    if pb != 0:
        pab = pba * pa / pb
    else:
        pab = 0
    pab = round(pab,4)
    print(pab)
    return pab

def bayes(X,y,b):
    table = [x[0] for x in X]
    table.append(y[-1])
    result_kind = []
    for each in table:
        if each not in result_kind:
            result_kind.append(each)
    predict_result = 0
    for each in result_kind:
        if redictp(X,y,each,1) > predict_result:
            predict_result = each
    return predict_result


if __name__=='__main__':
    # data_table = [["date", "weather"],
    #               [1, '晴'],
    #               [2, '阴'],
    #               [3, '降水'],
    #               [4, '阴'],
    #               [5, '降水'],
    #               [6, '晴'],
    #               [7, '晴'],
    #               [8, '多云'],
    #               [9, '降水'],
    #               [10, '降水']]

    from sklearn.naive_bayes import GaussianNB
    data_table = [["date", "weather"],
                  [1, 0],
                  [2, 1],
                  [3, 2],
                  [4, 1],
                  [5, 2],
                  [6, 0],
                  [7, 0],
                  [8, 3],
                  [9, 1],
                  [10, 1]]
    X = []
    for i in range(1, len(data_table) - 1):
        X.append([data_table[i][1]])
    y = []
    for i in range(2, len(data_table)):
        y.append(data_table[i][1])

    # 使用naive_bayes库进行运算
    redictp(X, y, '降水', '阴')
    clf = GaussianNB().fit(X, y)
    p = [[1]]
    print(clf.predict(p))


    # 使用自写redictp函数进行计算

    print(bayes(X,y,1))

