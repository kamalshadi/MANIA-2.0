# Curve fitting for distance bias
import numpy as num
from scipy.optimize import curve_fit

# N is number of bins
# m is number of selected points per bin


def func(x, a, b):
    return a + b*x

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

# a = [1,4,3,5,7,4]
# b=[0,2,21,3,-1,2]
# D = binCollapse(a,b,3,2)
# print(D)


    # collapse x,y
    # return x,y collapsed and candidate
