#coding=utf-8
from numpy import *
import xlrd
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt  
def loadDataSet(fileName):
    xls = xlrd.open_workbook(fileName)
    table = xls.sheets()[0]
    nrows = table.nrows
    data = []
    label = []
    for row in range(1, nrows):
        data.append([int(table.row_values(row)[3]), int(table.row_values(row)[4])])
    return data
    
#计算两个向量的距离，用的是欧几里得距离

def agglomerativeCluster(n):
    ac = AgglomerativeClustering(linkage='average',n_clusters = n)
    data = mat(loadDataSet(u'样本数据.xls'))

    predicted_labels = ac.fit_predict(data)
    return predicted_labels
  

def show(dataSet,k,predicted_labels):
    numSamples, dim = dataSet.shape  
    plt.xlabel(u"Height")
    plt.ylabel(u"Weight")
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  #设定图像数据点的颜色
    for i in xrange(numSamples):  
        markIndex = int(predicted_labels[i])  
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])  
    plt.show()
if __name__ == '__main__':
    data = mat(loadDataSet(u'样本数据.xls'))
    labels = agglomerativeCluster(2)
    show(data,2,labels)
    
    