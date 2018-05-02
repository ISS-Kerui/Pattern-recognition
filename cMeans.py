#coding=utf-8
from numpy import *
import xlrd
from matplotlib import pyplot as plt  
def loadDataSet(fileName):
    xls = xlrd.open_workbook(fileName)
    table = xls.sheets()[0]
    nrows = table.nrows
    data = []

    for row in range(1, nrows):
        data.append([int(table.row_values(row)[3]), int(table.row_values(row)[4])])
    return data
    
#计算两个向量的距离，用的是欧几里得距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #numpy.power 对x1中的每个元素求x2次方

#随机生成初始的质心
def randCenter(dataSet, k): # 初始化k个质心
    n = shape(dataSet)[1] #获取数据的类别数，这里有身高、体重两类
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j]) #获取身高、体重的最小值
        rangeJ = float(max(array(dataSet)[:,j]) - minJ) 
        centroids[:,j] = minJ + rangeJ * random.rand(k,1)
    
    return centroids

def fixCenter(dataSet,k):
    s =shape(dataSet)[0]
    n = shape(dataSet)[1] #获取数据的类别数，这里有身高、体重两类
    index = 0
    centroids = mat(zeros((k,n)))
    if k == 2:
        for i in range(s):

            if i == 0 or i == n/2+1:
                centroids[index,:] = dataSet[i,:]
                index = index+1
    if k == 3:
        for i in range(s):
            if i == 0 or i == n/3+1 or i == 2*n/3+1:
                centroids[index,:] = dataSet[i,:]
                index = index+1
   
    return centroids
    
def cMeans(dataSet, k, distMeas=distEclud, createCenter=fixCenter):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))                
    centroids = createCenter(dataSet,k)
    
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf # inf 无穷大值
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j

            if clusterAssment[i,0] != minIndex: 
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
       
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInClust, axis=0) 
    print "中心点为："+str(centroids)
    return centroids, clusterAssment

def show(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape  


    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  #设定图像数据点的颜色
    for i in xrange(numSamples):  
        markIndex = int(clusterAssment[i, 0])  
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])  
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  

    plt.xlabel(u"Height")
    plt.ylabel(u"Weight")
    for i in range(k):  
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)  
    plt.show()
      
def main():
    dataMat = loadDataSet(u'样本数据.xls')
    dataMat = mat(dataMat)
    myCentroids, clustAssing= cMeans(dataMat,3)
    show(dataMat, 3, myCentroids, clustAssing)  
    
    
if __name__ == '__main__':
    main()