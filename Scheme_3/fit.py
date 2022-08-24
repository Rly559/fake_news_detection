import pickle

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

clf = MLPClassifier(max_iter=500, hidden_layer_sizes=(100,), solver='adam', alpha=0.0002)
with open('dataset/train.pkl', 'rb') as file:
    # 训练
    train_set = pickle.load(file)
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_set['X'])  # 标准化（MLP对此敏感）
    joblib.dump(scaler, 'model/scaler.model')
    train_label = train_set['y']
    clf.fit(train_data, train_label)

    # 计算准确率
    factcheckscore = clf.score(train_data, train_label)
    print(f'Fact-checkAccuracy Is {round(factcheckscore * 100, 2)}%')

    y = np.array([round(factcheckscore * 100, 2), 100 - round(factcheckscore * 100, 2)])
    fig = plt.figure()
    # 设置背景色
    rect = fig.patch
    rect.set_facecolor('white')
    # 绘制百分比结果图
    plt.pie(y,
            labels=['Predict success', 'Predict failed'],
            colors=["#d5695d", "#5d8ca8"],
            autopct='%.2f%%',
            )
    plt.title("Predict the outcome")
    plt.show()

    joblib.dump(clf, 'model/mlp.model')  # 保存模型
