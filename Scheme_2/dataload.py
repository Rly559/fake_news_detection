

class preprocess():
    def preprocessingdata_news(self):
        def clear(text) -> str:
            # print(re.findall('[\u4E00-\u9FFF]+|\d',text))
            if type(text) is float:
                return
            str1 = ""
            str2 = str1.join(text)
            def y(x): return re.findall('[\u4E00-\u9FFF]+', x)
            li = y(str2)
            for i in range(len(li)):
                li[i] = list(jieba.cut(li[i]))
            li2 = []
            for subli in li:
                for item in subli:
                    li2.append(item)
            return li2
        import pandas as pd
        import jieba
        import re
        import pickle
        import os
        datapath = ".\\data\\train.csv"
        testpath = ".\\data\\test.csv"
        retdata = "\data\train"
        rettest = "\data\test"
        if (os.path.exists(retdata)):
            dataset2 = open(retdata, 'rb')
            dataset = pickle.load(dataset2)
            dataset2.close()
        else:
            print("正在数据集删除空数据并分词....")
            dataset = pd.read_csv(datapath)
            dataset.columns = ['id', 'content', 'comment_all', 'label']
            dataset = dataset.set_index('id')
            for index, row in dataset.iterrows():
                dataset.at[index, 'content'] = clear(
                    dataset.at[index, 'content'])
                dataset.at[index, 'comment_all'] = clear(
                    dataset.at[index, 'comment_all'])

            dataset.to_pickle('.\\data\\train')
        if (os.path.exists(rettest)):
            testset2 = open(rettest, 'rb')
            testset = pickle.load((testset2))
            testset2.close()
        else:
            print("正在删除测试集空数据并分词....")
            test = pd.read_csv(testpath)
            test.columns = ['id', 'content', 'comment_all']
            test = test.set_index('id')
            testset = test
            for index, row in testset.iterrows():
                testset.at[index, 'content'] = clear(
                    testset.at[index, 'content'])
                testset.at[index, 'comment_all'] = clear(
                    testset.at[index, 'comment_all'])
            testset.to_pickle('.\\data\\test')
        return dataset, testset

    def processingdata_bayes(self, dataset, testset):
        def label_get(dataset):
            labelset = []
            for index, row in dataset.iterrows():
                labelset.append(dataset.loc[index, 'label'])
            return labelset

        def pd_li_transform(dataset):
            dataset_list = []
            for index, row in dataset.iterrows():
                str2 = " "
                if dataset.loc[index, 'comment_all'] is None:
                    dataset_list.append(dataset.loc[index, 'content'])
                else:
                    dataset_list.append(
                        dataset.loc[index, 'content']+dataset.loc[index, 'comment_all'])
            return dataset_list

        def li_str_transform(dataset_list):
            dataset2 = []
            for seq in dataset_list:
                str1 = ' '.join(seq)
                dataset2.append(str1)
            return dataset2
        labelset = label_get(dataset)
        testset_list = pd_li_transform(testset)
        dataset_list = pd_li_transform(dataset)
        dataset_str = li_str_transform(dataset_list)
        testset_str = li_str_transform(testset_list)
        return dataset_str, testset_str, labelset

    def processingbert(self, dataset, testset):
        def label_get(dataset):
            labelset = []
            for index, row in dataset.iterrows():
                labelset.append(dataset.loc[index, 'label'])
            return labelset

        def pd_li_transform(dataset):
            dataset_list = []
            for index, row in dataset.iterrows():
                str2 = " "
                if dataset.loc[index, 'comment_all'] is None:
                    dataset_list.append(dataset.loc[index, 'content'])
                else:
                    dataset_list.append(
                        dataset.loc[index, 'content'] + dataset.loc[index, 'comment_all'])
            return dataset_list

        def list_deal(dataset, labelset):
            labelset2 = []
            dataset2 = []
            for i in range(len(dataset)):
                if (dataset[i] != []):
                    dataset2.append(dataset[i])
                    labelset2.append(labelset[i])
            return dataset2, labelset2

        def list_onehot(labelset):
            newli = []
            for index in labelset:
                if (index == -1):
                    newli.append([-1.0, 0.0, 0.0])
                if (index == 0):
                    newli.append([0.0, -1.0, 0.0])
                if (index == 1):
                    newli.append([0.0, 0.0, -1.0])
            return newli

        def list_float(labelset):
            newli = []
            for index in labelset:
                if (index == -1):
                    newli.append(0.0)
                if (index == 0):
                    newli.append(1.0)
                if (index == 1):
                    newli.append(2.0)
            return newli

        labelset = label_get(dataset)
        testset_list = pd_li_transform(testset)
        dataset_list = pd_li_transform(dataset)
        dataset_list, labelset = list_deal(dataset_list, labelset)
        return dataset_list, labelset, testset_list


if __name__ == "__main__":
    pre = preprocess()
    dataset, testset = pre.preprocessingdata_news()
    print(dataset.head())
    print(testset.head())
