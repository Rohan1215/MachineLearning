import numpy as np
def costFunction(theta):
    global X
    global Y
    global m
    hTheta=np.matmul(X,theta)
    diff=np.subtract(hTheta,Y)
    sq=np.square(diff)
    return np.sum(sq)/(2*(m))

fileInput=open("PopulationVsPriceData.txt",'r')
a=fileInput.readlines()
n=a[0].count(',')+1
m=len(a)
data=np.zeros((m,n))
for i in range(m):
    a[i]=a[i][:-1]
    x=a[i].split(",")
    for j in range(n):
        data[i][j]=float(x[j])

xOnes=np.ones((m,1))
xFeatures=data[:,:-1]
Y=data[:,n-1::1]
X=np.concatenate((xOnes,xFeatures),axis=1)

theta=np.zeros((n,1))
J=costFunction(theta)
print(J)


    



