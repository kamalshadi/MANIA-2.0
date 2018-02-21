# Curve fitting for distance bias
import numpy as num
from scipy.optimize import curve_fit
from sklearn.linear_model import TheilSenRegressor
# N is number of bins
# m is number of selected points per bin

from bokeh.plotting import output_notebook,figure, show
from bokeh.layouts import row,column,gridplot
from bokeh.models import Label

estimator = TheilSenRegressor(random_state=42)

def func(x, a, b):
    return a*x + b

def upit(a,w,h = None):
    if h:
        if (min(h)<2):
            return []
    else:
        if (min(a)<2):
            return []
    if (w<2):
        return a
    else:
        o = num.log2(pow(2, int(num.log2(w) + 0.5)))
    c = a*int(o)
    # c = c + num.random.rand(len(c))
    return c

def isdig(a):
    try:
        int(a)
        return True
    except:
        return False

def binit(x,ind,maxd=5,maxp = 10):
    l = len(x)
    if l==0:return
    np = max(int(num.ceil(l*maxp/100)),1)
    i1 = ind
    i2 = ind
    for i in range(1,l):
        tmp1 = i1 - i
        if tmp1 > 0 and (x[ind]-x[tmp1] <= maxd/2):
            i1 = tmp1
        else:
            break
    for i in range(1,l):
        tmp2 = i2 + i
        if tmp2 < l and (x[tmp2]-x[ind] <= maxd/2):
            i2 = tmp2
        else:
            break
    while True:
        tmp = i2-i1+1
        if tmp <= np:
            break
        else:
            dis1 = x[ind] - x[i1]
            dis2 = x[i2] - x[ind]
            if dis1 > dis2:
                i1 = i1+1
            else:
                i2 = i2-1
    return [(i1,i2),(x[i1],x[i2])]



def maxyout(x,y,bins):
    tmp = [float("-inf")]*len(y)
    np = 0
    l = len(x)
    for i,w in enumerate(x):
        for cur_bin in bins:
            if (w >= cur_bin[1][0] and w <= cur_bin[1][1]):
                break
        else:
            np = np + 1
            tmp[i] = y[i]
    if max(tmp) < 0:
        return
    if np < .2*l:
        flag=True
    else:
        flag = False
    return (num.argmax(tmp),flag)


def maxBin(x,y,N,m=1):
    z = zip(x,y)
    z = sorted(z)
    run = True
    if len(z) < N*m: # minimum number of points
        outs = [(xx,i) for i,xx in enumerate(z)]
        bins =[]
        run = False
        # return ([(xx,i) for i,xx in enumerate(z)],[])
    if run:
        x = [xx[0] for xx in z]
        y = [xx[1] for xx in z]
        outs = [] # output points
        bins =[]
        stopping = False
        flag = False
    while run:
        ind,flag = maxyout(x,y,bins) # get the maximum outside current bins
        if ind is None:
            break
        outs.append((z[ind],ind))
        # ind = num.argmax(y)
        cur_bin = binit(x,ind)
        bins.append(cur_bin)
        if (len(bins) >= N and flag):
            stopping=True
        if stopping: # see if K bins are formed
            break
    at = [xx[0][0] for xx in outs]
    bt = [xx[0][1] for xx in outs]
    popt, pcov = curve_fit(func, at, bt,maxfev = 10000)
    # a = sorted(list(set(a)))
    z = [func(xx,*popt) for xx in at]
    D = {}
    D['x'] = at
    D['y'] = bt
    D['z'] = z
    D['outs'] = outs
    D['bins'] = bins
    return D

def maxBin2(x,y,N,m=1):
    wp = 0.1 # KNN
    maxd = 5 #mm maximim distance
    z = zip(x,y)
    z = sorted(z)
    run = True
    if len(z) < N*m: # minimum number of points
        outs = [(xx,i) for i,xx in enumerate(z)]
        bins =[]
        run = False
        # return ([(xx,i) for i,xx in enumerate(z)],[])
    if run:
        x = [xx[0] for xx in z]
        y = [xx[1] for xx in z]
        outs = [] # output points
        bins =[]
        stopping = False
        flag = False
        l = len(x)
        w = int(num.ceil(l*wp))
        tail = False
        for i,cur in enumerate(x):
            bini = i + w + 1
            bini2 = int(i + w/4)
            if bini2 > l:
                break
            if bini>l:
                bini = l
            dmax = float(x[i]*15.0/100)+maxd
            if (x[bini-1] - x[i]>dmax):
                if num.max(y[i:l]) == y[i]:
                    outs.append([(cur,y[i]),(i,bini-1)])
                    cur_bin = (cur,x[bini-1])
                    bins.append(cur_bin)
                continue
            if num.max(y[i:bini]) == y[i]:
                outs.append([(cur,y[i]),(i,bini-1)])
                cur_bin = (cur,x[bini-1])
                bins.append(cur_bin)

    at = [xx[0][0] for xx in outs]
    bt = [xx[0][1] for xx in outs]
    popt, pcov = curve_fit(func, at, bt,maxfev = 10000)
    # a = sorted(list(set(a)))
    z = [func(xx,*popt) for xx in at]
    residuals = [(bt[q] - z[q])**2 for q in range(len(z))]
    ss_res = num.sum(residuals)
    ss_tot = num.sum((num.array(bt)-num.mean(bt))**2)
    r_squared = 1 - (ss_res / ss_tot)
    D = {}
    D['x'] = at
    D['y'] = bt
    D['z'] = z
    D['outs'] = outs
    D['bins'] = bins
    D['r'] = num.ceil(r_squared*100)
    return D

# softmax
def maxBin3(x,y,N,m=1):
    wp = 0.1 # KNN
    maxd = 5 #mm maximim distance
    z = zip(x,y)
    z = sorted(z)
    run = True
    if len(z) < N*m: # minimum number of points
        outs = [(xx,i) for i,xx in enumerate(z)]
        bins =[]
        run = False
        # return ([(xx,i) for i,xx in enumerate(z)],[])
    if run:
        x = [xx[0] for xx in z]
        y = [xx[1] for xx in z]
        outs = [] # output points
        bins =[]
        stopping = False
        flag = False
        l = len(x)
        w = int(num.ceil(l*wp))
        tail = False
        for i,cur in enumerate(x):
            if i >= (l - 3):
                break
            ql = y[i:]
            ax = num.percentile(ql,95)
            if y[i] >= ax:
                outs.append([(cur,y[i]),(i,l-1)])
                cur_bin = (cur,x[-1])
                bins.append(cur_bin)

    at = [xx[0][0] for xx in outs]
    alo = num.array(at)
    alo = alo.reshape(-1,1)
    bt = [xx[0][1] for xx in outs]
    blo = num.array(bt)
    blo = blo.reshape(-1,1)
    estimator.fit(alo, blo)
    z_pred = estimator.predict(alo)
    popt, pcov = curve_fit(func, at, bt,maxfev = 10000)
    # a = sorted(list(set(a)))
    z = [func(xx,*popt) for xx in at]
    residuals = [(bt[q] - z[q])**2 for q in range(len(z))]
    ss_res = num.sum(residuals)
    ss_tot = num.sum((num.array(bt)-num.mean(bt))**2)
    r_squared = 1 - (ss_res / ss_tot)
    D = {}
    D['params'] = popt
    D['x'] = at
    D['y'] = bt
    D['z'] = z
    D['z_pred'] = z_pred
    D['outs'] = outs
    D['bins'] = bins
    D['r'] = num.ceil(r_squared*100)
    return D


# hard max
def maxBin4(x,y,N,m=1):
    z = zip(x,y)
    z = sorted(z)
    run = True
    if len(z) < N*m: # minimum number of points
        outs = [(xx,i) for i,xx in enumerate(z)]
        bins =[]
        run = False
        # return ([(xx,i) for i,xx in enumerate(z)],[])
    elit = 1
    if run:
        q = 0
        while True:
            x = [xx[0] for xx in z]
            y = [xx[1] for xx in z]
            outs = [] # output points
            bins =[]
            stopping = False
            flag = False
            l = len(x)
            # w = int(num.ceil(l*wp))
            tail = False
            for i,cur in enumerate(x):
                if i >= (l - 3):
                    break
                ql = y[i:]
                tmp = 100 - q*elit
                ax = num.percentile(ql,tmp)
                if y[i] >= ax:
                    outs.append([(cur,y[i]),(i,l-1)])
                    cur_bin = (cur,x[-1])
                    bins.append(cur_bin)
            q = q + 1
            if len(bins)>= 5:
                break

    at = [xx[0][0] for xx in outs]
    # alo = num.array(at)
    # alo = alo.reshape(-1, 1)
    bt = [xx[0][1] for xx in outs]
    # blo = num.array(bt)
    # blo = blo.reshape(-1, 1)
    # estimator.fit(alo, blo)
    # z_pred = estimator.predict(alo)
    popt, pcov = curve_fit(func, at, bt,maxfev = 10000)
    # a = sorted(list(set(a)))
    z = [func(xx,*popt) for xx in at]
    residuals = [(bt[q] - z[q])**2 for q in range(len(z))]
    ss_res = num.sum(residuals)
    ss_tot = num.sum((num.array(bt)-num.mean(bt))**2)
    r_squared = 1 - (ss_res / ss_tot)
    D = {}
    D['params'] = popt
    D['x'] = at
    D['y'] = bt
    D['z'] = z
    D['popt'] = popt
    # D['z_pred'] = z_pred
    D['outs'] = outs
    D['bins'] = bins
    D['r'] = num.ceil(r_squared*100)
    return D


def regress(a,b, tit = ''):
    c = sorted(zip(a,b))
    d = [xx for xx in c if (xx[1]>100 and xx[0]>1)]
    a = [xx[0] for xx in d]
    b = [num.log(xx[1]) for xx in d]
    D = maxBin4(a,b,10)
    # D1 = maxBin3(a,b,10)
    outs = D['outs']
    bins = D['bins']
    _intercept = D['popt'][1] - num.log(5000)
    intercept = "{:5.2f}".format(_intercept)
    slope = num.ceil(-100*D['popt'][0])
    params = 'intercept ='+str(intercept)+' , '+'slope = '+str(slope)+'%'
    # normalize function
    fn = lambda x : x - num.log(5000)
    # outs1 = D1['outs']
    # bins1 = D1['bins']
    l = len(outs)
    # l1 = len(outs1)
    title = tit+'(R2%='+str(D['r'])+','+params+')'
    p1 = figure(plot_width=500, plot_height=500, title = title,y_axis_label = "Probability(log scale)",x_axis_label = "Distance")
    p1.circle(a,list(map(fn,b)))
    p1.line(D['x'],list(map(fn,D['z'])),legend="Max",color="red")
#     p1.line(D1['x'],D1['z'],legend="Soft-Max",color="blue")
#     p1.line(D1['x'],D1['z_pred'],legend="thiel-soft-Max",color="black")
#     p1.line(D['x'],D['z_pred'],legend="thiel-Max",color="cyan")
# #     p1.line(D['cxw'],D['czw'],legend="Donahue - weighted regression",color="blue")
    for i in range(l):
            p1.circle_cross(outs[i][0][0],fn(outs[i][0][1]),color='green',size=8)
    # for i in range(l1):
    #     p1.square(outs1[i][0][0],outs1[i][0][1],color='yellow',size=4,alpha=.8)
    return p1

max_nos = 5000    # probtrackx run parameter
noise_floor = num.log(100.0/max_nos)
min_r2 = 80
min_envelope_points = 5
min_points_on_right = 5

def pre_mania2(x,y):

    D = {} # regression model
    D['isSuccess'] = False
    D['reason'] = None
    D['popt'] = [0.0,0.0] # regression parameters
    D['slope'] = 0.0
    D['intercept'] = 0
    D['envelope'] = [] # envelope points
    D['r2'] = 0.0
    D['x'] = []
    D['z'] = []

    outs = [] # output points
    stopping = False
    l = len(x)
    # w = int(num.ceil(l*wp))
    tail = False
    for i,cur in enumerate(x):
        if i >= (l - min_points_on_right):
            break
        if y[i] <= noise_floor:
            continue
        ql = y[(i+1):]
        ax = num.max(ql)
        if y[i] >= ax:
            outs.append((cur,y[i]))

    D['envelope'] = outs

    if (len(outs) < min_envelope_points ):
        D['reason'] = 'envelope'
        return D

    at = [xx[0] for xx in outs]
    D['x'] = at
    bt = [xx[1] for xx in outs]

    popt, pcov = curve_fit(func, at, bt,maxfev = 10000)
    D['popt'] = popt
    # a = sorted(list(set(a)))
    z = [func(xx,*popt) for xx in at]
    D['z'] = z
    residuals = [(bt[q] - z[q])**2 for q in range(len(z))]
    ss_res = num.sum(residuals)
    ss_tot = num.sum((num.array(bt)-num.mean(bt))**2)
    r_squared = 1 - (ss_res / ss_tot)
    r_squared = r_squared * 100
    D['r2'] = r_squared
    if (r_squared <= min_r2):
        D['reason'] = 'r2'
        return D
    D['isSuccess'] = True
    D['intercept'] = popt[1]
    D['slope'] = -100*popt[0]
    return D

def regress_noplot(a,b, tit = ''):
    b [b==0] = 1 # for numerical purposes (log argumanet must be nonzero)
    b = num.log(b/max_nos)
    c = sorted(zip(a,b))
    a = [xx[0] for xx in c]
    b = [xx[1] for xx in c]
    D = pre_mania2(a,b)
    D['a'] = a
    D['b'] = b
    # D1 = maxBin3(a,b,10)
    return D

def binCollapse(x,y,N,m):
    # Generate bins
    minx = min(x)
    maxx = max(x)
    lx = len(x)
    nbins = int(min(lx/2,N))
    bins = num.linspace(minx,maxx,nbins)
    # Digitize
    dx = num.digitize(x,bins)  # digitized x
    D = {}
    # Select point per bin
    for i in range(lx):
        bini = dx[i]
        try:
            D[bini]
            D[bini]['x'].append(x[i])
            D[bini]['y'].append(y[i])
        except KeyError:
            D[bini] = {}
            D[bini]['x'] = [x[i]]
            D[bini]['y'] = [y[i]]
    for i in D.keys():
        try:
            ncur = len(D[i]['x'])
        except KeyError:
            D[i] = {}
            D[i]['x'] = []
            D[i]['y'] = []
            D[i]['_x'] = None
            D[i]['_y'] = None
            ncur = 0
            continue
        tmp = ncur/2
        if tmp<m:
            nsample = int(tmp) + 1
        else:
            nsample = m
        tmp1 = D[i]['x']
        tmp2 = D[i]['y']
        tmp3 = zip(tmp2,tmp1)
        tmp4 = (sorted(tmp3,reverse=True)[0:nsample])
        D[i]['w'] = len(D[i]['x'])

        D[i]['x'] = [xx[1] for xx in tmp4]
        D[i]['y'] = [xx[0] for xx in tmp4]
        D[i]['_x'] = num.median(D[i]['x'])
        D[i]['_y'] = num.median(D[i]['y'])
    # non-weighted
    la = [D[i]['x'] for i in D.keys()]
    lb = [D[i]['y'] for i in D.keys()]
    a = [item for sublist in la for item in sublist]
    b = [item for sublist in lb for item in sublist]
     # [D[i]['x'] for sublist in D.keys() for D[i]['x'] in sublist]
    popt2, pcov2 = curve_fit(func, a, b,maxfev = 10000)
    a = sorted(list(set(a)))
    z = [func(xx,*popt2) for xx in a]
    D['param'] = popt2
    D['cx'] = sorted(a)
    D['cz'] = z

    # weighted
    la = [upit(D[i]['x'],D[i]['w']) for i in D.keys() if isdig(i)]
    lb = [upit(D[i]['y'],D[i]['w'],D[i]['x'])  for i in D.keys() if isdig(i)]
    aa = [item for sublist in la for item in sublist]
    bb = [item for sublist in lb for item in sublist]
     # [D[i]['x'] for sublist in D.keys() for D[i]['x'] in sublist]
    popt2, pcov2 = curve_fit(func, aa, bb,maxfev = 10000)
    aa = sorted(list(set(aa)))
    zz = [func(xx,*popt2) for xx in aa]
    D['cxw'] = aa
    D['czw'] = zz

    D["bins"] = bins
    return D


# hard max
def correct(x,y,N,m=1):

    run = True
    # if len(y) < N*m: # minimum number of points
    #     return (x,[num.log(max(xx,1)) for i,xx in enumerate(z)])


    elit = 1
    q = 0

    while True:
        outs = [] # output points
        l = len(x)
        tail = False
        for i,cur in enumerate(x):
            if i >= (l - 3):
                break
            ql = y[i:]

            tmp = 100 - q*elit
            ax = num.percentile(ql,tmp)
            if y[i] >= ax:
                outs.append([(cur,y[i]),(i,l-1)])

        q = q + 1
        if len(outs)>= 5:
            break

    at = [xx[0][0] for xx in outs]
    bt = [num.log(xx[0][1]) for xx in outs]
    popt, pcov = curve_fit(func, at, bt,maxfev = 10000)
    # a = sorted(list(set(a)))
    y = [num.log(xx) for xx in y]
    zf = zip(x,y)
    tmp = [xx[1]-popt[1]*xx[0] for xx in zf]
    return tmp

# a = [1,5.6,6,7,7,8,9,11,11,12.5,13,13.2,13.3,24,24,31,31.2,34,41,51.1,51.2]
# b = [10,43,60,17,98,80,91,1,2,12.5,13,13.2,13.3,24,24,31,31.2,34,41,51.1,51.2]
# outs,bins = maxBin(a,b,10)
# print(outs)
# print(bins)
# b=[0,2,21,3,-1,2]
# D = binCollapse(a,b,3,2)
# print(D)


    # collapse x,y
    # return x,y collapsed and candidate
