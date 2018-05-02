 # -*- coding: utf-8 -*- 
import xlrd
import numpy as np
import math
from sympy import solve,symbols
from sympy import plot_implicit
from sympy.plotting import plot
from sympy.parsing.sympy_parser import parse_expr
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager, FontProperties  


def getChineseFont():  
    return FontProperties(fname='/System/Library/Fonts/PingFang.ttc')  

data = xlrd.open_workbook('作业数据_2017And2016.xls')
table = data.sheets()[0] 
nrows = table.nrows
man_h = []
man_w = []
woman_h = []
woman_w = []
for row in range(1,nrows):
	if table.row_values(row)[1] == 1:
		man_h.append(table.row_values(row)[3])
		man_w.append(table.row_values(row)[4])
	else:
		woman_h.append(table.row_values(row)[3])    	
		woman_w.append(table.row_values(row)[4])    
man_h_u=np.mean(man_h)
woman_h_u=np.mean(woman_h) 
man_w_u=np.mean(man_w)
woman_w_u=np.mean(woman_w) 
fig1 = plt.figure(1)
n, bins, patches = plt.hist(man_h, bins=10, normed=1,edgecolor='black',facecolor='blue',histtype='bar')  
plt.title(u'男生身高直方图',fontproperties=getChineseFont())
plt.xlabel(u'身高/cm',fontproperties=getChineseFont())
plt.ylabel(u'比例',fontproperties=getChineseFont())
plt.show()
fig2 = plt.figure(2)
n, bins, patches = plt.hist(woman_h, bins=10, normed=1,edgecolor='black',facecolor='red',histtype='bar')  
plt.title(u'女生身高直方图',fontproperties=getChineseFont())
plt.xlabel(u'身高/cm',fontproperties=getChineseFont())
plt.ylabel(u'比例',fontproperties=getChineseFont())
plt.show()
def Maximum_likelihood_estimate():
	print '\n\n极大似然估计法:'
	man_h_S = 0.0
	man_w_S = 0.0
	woman_h_S = 0.0
	woman_w_S = 0.0
	
	for i in range(len(man_h)):
		x = man_h[i]-man_h_u
		y = man_w[i]-man_w_u
		man_h_S = man_h_S + x*x
		man_w_S = man_w_S + y*y
	for j in range(len(woman_h)):
		x = woman_h[j]-woman_h_u
		y = woman_w[j]-woman_w_u
		woman_h_S = woman_h_S + x*x
		woman_w_S = woman_w_S + y*y
	sig_m_h = man_h_S/(len(man_h)-1)
	sig_m_w = man_w_S/(len(man_h)-1)
	sig_w_h = woman_h_S/(len(woman_h)-1)
	sig_w_w = woman_w_S/(len(woman_h)-1)
	print ('男生身高极大似然估计均值为：'+str(man_h_u)+' ,方差为：'+str(sig_m_h))
	print ('男生体重极大似然估计均值为：'+str(man_w_u)+' ,方差为：'+str(sig_m_w))
	print ('女生身高极大似然估计均值为：'+str(woman_h_u)+' ,方差为：'+str(sig_w_h))
	print ('女生体重极大似然估计均值为：'+str(woman_w_u)+' ,方差为：'+str(sig_w_w))
	return [sig_m_h,sig_m_w,sig_w_h,sig_w_w]

def Bayes_estimates(sigma):
	print '\n\n贝叶斯估计法:'
	sigmaX1=10     
	sigmaX2=10  
	sigmaX3=10  
	sigmaX4=10
	uN1=(sigma[0]*np.sum(man_h)+sigmaX1*man_h_u)/(sigma[0]*len(man_h)+sigmaX1);  
	uN2=(sigma[1]*np.sum(man_w)+sigmaX2*man_w_u)/(sigma[1]*len(man_w)+sigmaX2);  
	uN3=(sigma[2]*np.sum(woman_h)+sigmaX3*woman_h_u)/(sigma[2]*len(woman_h)+sigmaX3);  
	uN4=(sigma[3]*np.sum(woman_w)+sigmaX4*woman_w_u)/(sigma[3]*len(woman_w)+sigmaX4);    
	print ('设置均值u的先验分布方差均为10')
	print('最小错误率贝叶斯所估计出的均值为：'+str(uN1)+', '+str(uN2)+', '+str(uN3)+', '+str(uN4))

def draw_map(sigma):
	C1 = 0.0
	C2 = 0.0
	for i in range(len(man_h)):
		C1=C1+(man_h[i]-man_h_u)*(man_w[i]-man_w_u)
	for j in range(len(woman_h)):
		C2=C2+(woman_h[j]-woman_h_u)*(woman_w[j]-woman_w_u)
	sigma1 = C1/len(man_h)
	sigma2 = C2/len(woman_h)

	sigma_man = np.asmatrix([[sigma[0],sigma1],[sigma1,sigma[1]]])
	sigma_woman = np.asmatrix([[sigma[2],sigma2],[sigma2,sigma[3]]])

	p_m = len(man_h)/(len(man_h)+len(woman_h)+0.0)
	p_w = 1- p_m
	x1, x2 = symbols('height,weight')
	
	N_1 = np.asmatrix([x1-man_h_u,x2-man_w_u])
	N_2 = np.asmatrix([x1-woman_h_u,x2-woman_w_u])

	

	g = 0.5*N_1*(sigma_man**(-1))*np.matrix.getH(N_1)-0.5*N_2*(sigma_woman**(-1))*np.matrix.getH(N_2)+0.5*np.log(np.linalg.det(sigma_man)/np.linalg.det(sigma_woman))-np.log(p_m/p_w)
	g = np.asarray(g)
	function = str(g[0][0])
	
	ezplot = lambda exper: plot_implicit(parse_expr(exper),(x1,120,200),(x2,20,100),axis_center=(120,20))
	
	
	ezplot(function)
	

if __name__ == '__main__':
	sigma = Maximum_likelihood_estimate()
	Bayes_estimates(sigma)
	draw_map(sigma)
