import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
import random
density=np.array([0.697,0.774,0.634,0.556,0.403,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.639,0.657,0.360,0.593,0.719])
sugarrate=np.array([0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267,0.057,0.099,0.161,0.198,0.370,0.042,0.103])
data=np.vstack((density.T,sugarrate.T)).T
print(data)
RealLabel=np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])

#startpoint指初始的参数
def newton_method_2D(startpoint,data,label0):
    l=0
    weigh_d,weigh_s,y00=startpoint
    beta0=np.array([[weigh_d],[weigh_s],[y00]])
    v_df_d=1
    v_df_s=1
    v_df_s=1
    v_df_y0=1
    flag=1
    # wd = sy.symbols('w_d')
    # ws = sy.symbols('w_s')
    # y0 = sy.symbols('y0')
    # for i in range(0, 17):
    #     temp = wd * (data[i][0]) + ws * (data[i][1]) + y0
    #     # print(type(Hess))
    #     l += (-label0[i] * temp + sy.log(1 + sy.exp(temp)))
    # df_d=sy.diff(l,wd)
    # df_s=sy.diff(l,ws)
    # df_y0=sy.diff(l,y0)
    # v_df_d=df_d.subs({'w_d':weigh_d,'w_s':weigh_s,'y0':y00})
    # print(v_df_d)
    # print(df_y0,'\n')
    # v_df_s=df_s.subs({'w_d':weigh_d,'w_s':weigh_s,'y0':y00})
    # v_df_y0=df_y0.subs({'w_d':weigh_d,'w_s':weigh_s,'y0':y00})
#求偏导
    while (abs(v_df_d)>0.1 or abs(v_df_s)>0.1 or abs(v_df_y0)>0.1):

        wd=sy.symbols('w_d')
        ws=sy.symbols('w_s')
        y0=sy.symbols('y0')
        for i in range(0,17):
            temp=wd*(data[i][0])+ws*(data[i][1])+y0
            l+=(-label0[i]*temp+sy.log(1+sy.exp(temp)))
        df_d = sy.diff(l, wd)
        df_s = sy.diff(l, ws)
        df_y0 = sy.diff(l, y0)
        v_df_d = df_d.subs({'w_d': weigh_d, 'w_s': weigh_s, 'y0': y00})
        v_df_s = df_s.subs({'w_d': weigh_d, 'w_s': weigh_s, 'y0': y00})
        v_df_y0 = df_y0.subs({'w_d': weigh_d, 'w_s': weigh_s, 'y0': y00})
        temp=[[v_df_s],[v_df_d],[v_df_y0]]
        print(temp)
        h1=[sy.diff(df_d,wd).subs({'w_d':weigh_d,'w_s':weigh_s,'y0':y00}),sy.diff(df_s,wd).subs({'w_d':weigh_d,'w_s':weigh_s,'y0':y00}),sy.diff(df_y0,wd).subs({'w_d':weigh_d,'w_s':weigh_s,'y0':y00})]
        h2=[sy.diff(df_d,ws).subs({'w_d':weigh_d,'w_s':weigh_s,'y0':y00}),sy.diff(df_s,ws).subs({'w_d':weigh_d,'w_s':weigh_s,'y0':y00}),sy.diff(df_y0,ws).subs({'w_d':weigh_d,'w_s':weigh_s,'y0':y00})]
        h3=[sy.diff(df_d,y0).subs({'w_d':weigh_d,'w_s':weigh_s,'y0':y00}),sy.diff(df_s,y0).subs({'w_d':weigh_d,'w_s':weigh_s,'y0':y00}),sy.diff(df_y0,y0).subs({'w_d':weigh_d,'w_s':weigh_s,'y0':y00})]
        Hess=[h1,h2,h3]
        print(Hess)
        print(type(Hess[0][0]))
        Hess=np.array(Hess).astype('float')
        print(type(Hess[0][0]))
        print(beta0)
        beta0=beta0-np.dot(np.linalg.inv(Hess),np.array(temp))
        print(np.linalg.inv(Hess))
        print(temp)
        print(np.dot(np.linalg.inv(Hess),np.array(temp)))#矩阵乘法不是用*
        print(beta0[0][0])
        print(beta0[1][0])
        print(beta0[2][0])
        weigh_d=beta0[0][0]
        weigh_s=beta0[1][0]
        y00=beta0[2][0]
        print(beta0)
        if (abs(beta0[0][0])>50 and abs(beta0[1][0])>50):
            flag=0
            break
    return beta0,flag
def graddescent(startpoint,data,label0):
    a=0.001
    l=0
    weigh_d, weigh_s, y00 = startpoint
    v_df_d = 1
    v_df_s = 1
    v_df_s = 1
    v_df_y0 = 1
    wd=sy.symbols('w_d')
    ws=sy.symbols('w_s')
    y0=sy.symbols('y0')
    for i in range(0, 17):
        temp = wd * (data[i][0]) + ws * (data[i][1]) + y0
        l += (-label0[i] * temp + sy.log(1 + sy.exp(temp)))
    df_d = sy.diff(l, wd)
    df_s = sy.diff(l, ws)
    df_y0 = sy.diff(l, y0)
    while (abs(v_df_d)>0.001 or abs(v_df_s)>0.001 or abs(v_df_y0)>0.1):
        v_df_d = df_d.subs({'w_d': weigh_d, 'w_s': weigh_s, 'y0': y00})
        v_df_s = df_s.subs({'w_d': weigh_d, 'w_s': weigh_s, 'y0': y00})
        v_df_y0 = df_y0.subs({'w_d': weigh_d, 'w_s': weigh_s, 'y0': y00})
        v_wd=wd.subs({'w_d': weigh_d, 'w_s': weigh_s, 'y0': y00})
        v_ws=ws.subs({'w_d': weigh_d, 'w_s': weigh_s, 'y0': y00})
        v_y0=y0.subs({'w_d': weigh_d, 'w_s': weigh_s, 'y0': y00})
        weigh_d=weigh_d-a*v_df_d
        weigh_s=weigh_s-a*v_df_s
        y00=y00-a*v_df_y0
        print(v_df_d)
        print(v_df_s)
        print(v_df_y0)
    return weigh_d,weigh_s,y00
#画图
def print_result(data,RealLabel,beta):
    fig=plt.figure(figsize=(0.5,1.0))
    title=fig.suptitle("linear predict of watermelon")
    ax=fig.add_subplot(1,1,1)
    ax.set_xlabel("density")
    ax.set_ylabel("sugar")
    i=0
    for index in data:
        predict=index[0]*beta[0]+index[1]*beta[1]+beta[2]
        if(RealLabel[i]>=0.5):
            plt.scatter(index[0],index[1],c='r')
        else:
            plt.scatter(index[0],index[1],c='g')
        i=i+1
    x=np.linspace(0.3,0.7,1000)
    plt.plot(x,-beta[0]/beta[1]*x+(-beta[2]+0.5)/beta[1],c='b')
    plt.show()
#用循环来寻找合适的初始值
# for i in range(0,10000):
#     beta,flag=newton_method_2D([random.uniform(0,10),random.uniform(0,10),random.uniform(-5,0)],data,RealLabel)#牛顿法的迭代点怎么选
#     if(flag==1):
#         print_result(data,RealLabel,beta)
#         break
#     else:
#         continue
beta=graddescent([random.uniform(0,10),random.uniform(0,10),random.uniform(-5,0)],data,RealLabel)
print_result(data,RealLabel,beta)