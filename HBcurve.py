from bidict import bidict
from math import sqrt
from collections import deque
from matplotlib import pyplot as plt
from sys import getsizeof
import numpy as np

def exhaust(generator):
    deque(generator, maxlen=0)


def bet(func):
    b = (lambda f: (lambda x: x(x))(lambda y:
          f(lambda *args: lambda: y(y)(*args))))(func)
    def wrapper(*args):
        out = b(*args)
        while callable(out):
            out = out()
        return out
    return wrapper

# b = (lambda x: x(x))(lambda y: func(lambda *args: (lambda: y(y)(*args))))


# >>> fac = bet( lambda f: lambda n, a: a if not n else f(n-1,a*n) )
# >>> fac(5,1)
# 120


def HilbertCurve_TCO(w, h):
    print(f"\n## HBcurve")
    apn = list.append
    lsapn = lambda ls, *xs: exhaust(map(lambda x: apn(ls, x), xs))
    def coordmap(*ps):
        return {(p[-3], p[-2]) : p[-1] for p in ps}

    if h/w > sqrt(2):
        (todo, mapping) = ([(0, h, w, 0, 0, 0, 0)], bidict({(0, 0):0}))
        print("flipped!")
    else:
        (todo, mapping) = ([(w, 0, 0, h, 0, 0, 0)], bidict({(0, 0):0}))

    while todo:
        wdx, wdy, hdx, hdy, cx, cy, ind = todo.pop()
        hlfvec = lambda x: sign(x)*(abs(x) - abs(x)//2)
        # if Option.inFuncDrawing:
        #     draw_infunc(w, h, cx, cy, wdx, wdy, hdx, hdy, mapping)
        if abs(wdx*hdy) == 1 or abs(wdy*hdx) == 1:
            continue

        if abs(wdx+wdy) == 1:
            p2 = (wdx
                , wdy
                , hdx - sign(hdx)
                , hdy - sign(hdy)
                , cx + sign(hdx)
                , cy + sign(hdy)
                , ind + 1)
            lsapn(todo, p2)
            mapping.update(coordmap(p2))
            continue

        if abs((wdx+wdy)/(hdy+hdx)) > sqrt(2):
            wdx2, wdy2 = hlfvec(wdx), hlfvec(wdy)
            p1 = (wdx2
                , wdy2
                , hdx
                , hdy
                , cx
                , cy
                , ind)
            p2 = (wdx - wdx2
                , wdy - wdy2
                , hdx
                , hdy
                , cx + wdx2
                , cy + wdy2
                , ind + abs(wdx2*hdy) + abs(wdy2*hdx))
            lsapn(todo, p2, p1)
            mapping.update(coordmap(p2))
            continue

        if abs((hdx+hdy)/(wdx+wdy)) > sqrt(2):
            wdx2, wdy2 = hlfvec(wdx), hlfvec(wdy)
            hdx2, hdy2 = sign(hdx)*abs(wdy2), sign(hdy)*abs(wdx2)
            p1 = (hdx2
                , hdy2
                , wdx2
                , wdy2
                , cx
                , cy
                , ind)
            p2 = (wdx
                , wdy
                , hdx - hdx2
                , hdy - hdy2
                , cx + hdx2
                , cy + hdy2
                , ind + abs(wdx2*hdy2) + abs(wdy2*hdx2))
            p3 =(-hdx2
                ,-hdy2
                ,-(wdx - wdx2)
                ,-(wdy - wdy2)
                , cx + wdx - sign(wdx) + hdx2 - sign(hdx2)
                , cy + wdy - sign(wdy) + hdy2 - sign(hdy2)
                , ind + abs(wdx2*hdy2) + abs(wdy2*hdx2) \
                    + abs(wdx*(hdy - hdy2)) + abs(wdy*(hdx - hdx2)))
            lsapn(todo, p3, p2, p1)
            mapping.update(coordmap(p2, p3))
            continue

        wdx2, wdy2 = hlfvec(wdx), hlfvec(wdy)
        hdx2, hdy2 = hlfvec(hdx), hlfvec(hdy)
        p1 = (hdx2
            , hdy2
            , wdx2
            , wdy2
            , cx
            , cy
            , ind)
        p2 = (wdx2
            , wdy2
            , hdx - hdx2
            , hdy - hdy2
            , cx + hdx2
            , cy + hdy2
            , ind + abs(wdx2*hdy2) + abs(wdy2*hdx2))
        p3 = (wdx - wdx2
            , wdy - wdy2
            , hdx - hdx2
            , hdy - hdy2
            , cx + wdx2 + hdx2
            , cy + wdy2 + hdy2
            , ind + abs(wdx2*hdy2) + abs(wdy2*hdx2) \
                + abs(wdx2*(hdy - hdy2)) + abs(wdy2*(hdx - hdx2)))
        p4 = (-hdx2
            ,-hdy2
            ,-(wdx - wdx2)
            ,-(wdy - wdy2)
            , cx + wdx - sign(wdx) + hdx2 - sign(hdx2)
            , cy + wdy - sign(wdy) + hdy2 - sign(hdy2)
            , ind + abs(wdx*hdy) + abs(wdy*hdx) \
                - abs(hdx2*(wdy - wdy2)) - abs(hdy2*(wdx - wdx2)))
        lsapn(todo, p4, p3, p2, p1)
        mapping.update(coordmap(p2, p3, p4))

    return mapping

# recursive function that return a index that fills a*b grid,
# starts at bottom left end at bottom right
# odd side split to right/bottom
# work with accumulator
# Wrapper:
# (width, height) -> coord-index mapping (image x-y coord)
# Inner Rec:
# todo (list of (wdx, wdy, hdx, hdy, cx, cy, ind)), mapping (recorded index <-> coord)
# where c= bottom left corner, ind= c started index, width= "bottom" length
def sign(x):
    return int(abs(x)/x) if x!=0 else 0

count = {'#1':0, '#2':0, '#3':0, '#4':0, '#5':0}

class Option:
    inFuncDrawing = True

def hilbertCurve(w, h):
    apn = list.append
    lsapn = lambda ls, *xs: exhaust(map(lambda x: apn(ls, x), xs))
    def coordmap(*ps):
        return {(p[-3], p[-2]) : p[-1] for p in ps}

    def rec(todo, mapping):
        if not todo:
            return mapping

        wdx, wdy, hdx, hdy, cx, cy, ind = todo.pop()
        hlfvec = lambda x: sign(x)*(abs(x) - abs(x)//2)
        print(f"w: {w}, h: {h}")
        print(f"wdx: {wdx}\n, wdy: {wdy}\n, hdx: {hdx}\n, hdy: {hdy}\n, cx: {cx}\n, cy: {cy}\n, ind: {ind}")
        print(f"mapping: {mapping} \n")
        if Option.inFuncDrawing:
            draw_infunc(w, h, cx, cy, wdx, wdy, hdx, hdy, mapping)
        if abs(wdx*hdy) == 1 or abs(wdy*hdx) == 1:
            print("#1, Unit\n")
            count['#1']+=1
            return rec(todo, mapping)

        if abs(wdx+wdy) == 1: # B shape, create a jump!
            # hopefully don't run into n*1 shapes... (it will never be natutally, right? right?)
            # still, for generality, iterate through h:
            print("#2, B shape\n")
            count['#2']+=1
            p2 = (wdx
                , wdy
                , hdx - sign(hdx)
                , hdy - sign(hdy)
                , cx + sign(hdx)
                , cy + sign(hdy)
                , ind + 1)
            lsapn(todo, p2)
            mapping.update(coordmap(p2))

            return rec(todo, mapping)

        if abs((wdx+wdy)/(hdy+hdx)) > sqrt(2): # dx, dy one must be 0
            print("#3, flat\n")
            count['#3']+=1
            wdx2, wdy2 = hlfvec(wdx), hlfvec(wdy)
            # hdx2, hdy2 = hlfvec(hdx), hlfvec(hdy)
            p1 = (wdx2
                , wdy2
                , hdx
                , hdy
                , cx
                , cy
                , ind)
            p2 = (wdx - wdx2
                , wdy - wdy2
                , hdx
                , hdy
                , cx + wdx2
                , cy + wdy2
                , ind + abs(wdx2*hdy) + abs(wdy2*hdx))
            lsapn(todo, p2, p1)
            mapping.update(coordmap(p2))

            return rec(todo, mapping)

        if abs((hdx+hdy)/(wdx+wdy)) > sqrt(2):
            print("#4, tower\n")
            count['#4']+=1
            # top 1, bottom 2 squares
            # Exception of n*1 handled previously
            wdx2, wdy2 = hlfvec(wdx), hlfvec(wdy)
            hdx2, hdy2 = sign(hdx)*abs(wdy2), sign(hdy)*abs(wdx2) # special treatment
            p1 = (hdx2
                , hdy2
                , wdx2
                , wdy2
                , cx
                , cy
                , ind)
            p2 = (wdx
                , wdy
                , hdx - hdx2
                , hdy - hdy2
                , cx + hdx2
                , cy + hdy2
                , ind + abs(wdx2*hdy2) + abs(wdy2*hdx2))
            p3 =(-hdx2
                ,-hdy2
                ,-(wdx - wdx2)
                ,-(wdy - wdy2)
                , cx + wdx - sign(wdx) + hdx2 - sign(hdx2)
                , cy + wdy - sign(wdy) + hdy2 - sign(hdy2)
                , ind + abs(wdx2*hdy2) + abs(wdy2*hdx2) \
                    + abs(wdx*(hdy - hdy2)) + abs(wdy*(hdx - hdx2)))
            lsapn(todo, p3, p2, p1)
            mapping.update(coordmap(p2, p3))
            return rec(todo, mapping)

        print("#5, normal divided by 4\n")
        count['#5']+=1
        # rest of the situation:
        wdx2, wdy2 = hlfvec(wdx), hlfvec(wdy)
        hdx2, hdy2 = hlfvec(hdx), hlfvec(hdy)
        p1 = (hdx2
            , hdy2
            , wdx2
            , wdy2
            , cx
            , cy
            , ind)
        p2 = (wdx2
            , wdy2
            , hdx - hdx2
            , hdy - hdy2
            , cx + hdx2
            , cy + hdy2
            , ind + abs(wdx2*hdy2) + abs(wdy2*hdx2))
        p3 = (wdx - wdx2
            , wdy - wdy2
            , hdx - hdx2
            , hdy - hdy2
            , cx + wdx2 + hdx2
            , cy + wdy2 + hdy2
            , ind + abs(wdx2*hdy2) + abs(wdy2*hdx2) \
                + abs(wdx2*(hdy - hdy2)) + abs(wdy2*(hdx - hdx2)))
        p4 = (-hdx2
            ,-hdy2
            ,-(wdx - wdx2)
            ,-(wdy - wdy2)
            , cx + wdx - sign(wdx) + hdx2 - sign(hdx2)
            , cy + wdy - sign(wdy) + hdy2 - sign(hdy2)
            , ind + abs(wdx*hdy) + abs(wdy*hdx) \
                - abs(hdx2*(wdy - wdy2)) - abs(hdy2*(wdx - wdx2)))
        # abs(wdx2*hdy2) + abs(wdy2*hdx2) \
        #     + abs(wdx*(hdy - hdy2)) + abs(wdy*(hdx - hdx2)
        lsapn(todo, p4, p3, p2, p1)
        mapping.update(coordmap(p2, p3, p4))
        return rec(todo, mapping)
    # test if the initial height is > sqrt(2)*width, if so, alter the end points to optimize output
    if h/w > sqrt(2):
        return rec([(0, h, w, 0, 0, 0, 0)], bidict({(0, 0):0}))

    return rec([(w, 0, 0, h, 0, 0, 0)], bidict({(0, 0):0}))

def draw_infunc(w, h, cx, cy, wdx, wdy, hdx, hdy, mapping):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # plt.plot([0, w, w, 0, 0], [0, 0, h, h, 0], 'r')
    xs = []
    ys = []
    k = 0
    for i in range(int(w*h)):
        x, y = mapping.inverse[i]
        xs.append(x+0.5)
        ys.append(y+0.5)

    plt.plot(xs, ys, 'b')
    # highlight
    xs1 = [cx + 0.5 - sign(wdx+hdx)*0.5
        , cx + 0.5 - sign(wdx+hdx)*0.5 + (wdx+hdx)
        , cx + 0.5 - sign(wdx+hdx)*0.5 + (wdx+hdx)
        , cx + 0.5 - sign(wdx+hdx)*0.5
        , cx + 0.5 - sign(wdx+hdx)*0.5]
    ys1 = [cy + 0.5 - sign(wdx+hdx)*0.5
        , cy + 0.5 - sign(wdx+hdx)*0.5
        , cy + 0.5 - sign(wdx+hdx)*0.5 + (wdy+hdy)
        , cy + 0.5 - sign(wdx+hdx)*0.5 + (wdy+hdy)
        , cy + 0.5 - sign(wdx+hdx)*0.5]
    plt.plot(xs1, ys1, 'r--')

    plt.gca().set_aspect('equal', adjustable='box')
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, w+1, 1)
    minor_ticks = np.arange(0, h+1, 1)

    ax.set_xticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    plt.show()

def draw(w, h, mapping):
    draw_infunc(w, h, 0, 0, w, 0, 0, h, mapping)
