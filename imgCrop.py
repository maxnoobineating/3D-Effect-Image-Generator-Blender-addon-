import imageio.v3 as iio
from matplotlib import pyplot as plt
import numpy as np
from numpy import array as arr
import os
from math import sqrt


# arr in
# new arr out
# currently don't support shrinking (memory issue effect implementation)
def centeredResize(img, shape, filler):
    imgShape = img.shape
    imgCenter = arr((imgShape[0]/2, imgShape[1]/2))
    newimgCenter = arr((shape[0]/2, shape[1]/2))
    newimg = np.full(shape, filler)

    delta = (newimgCenter - imgCenter)

    it = np.nditer(img, flags=['multi_index'])
    with it:
        for v in it:
            shiftedP = tuple((arr(it.multi_index) + delta).astype('int'))
            try:
                newimg[shiftedP] = v
            except IndexError: pass
        del img # ...it's not like this, this only delete local binding, which is going to disappear anyway
    return newimg

def drawdot(diameter):
    n = diameter
    dot = {}
    for i in range(n):
        for j in range(n):
            if sqrt((i-n/2)**2+(j-n/2)**2)<=n/2:
                dot[(i, j)] = 0
    return dot

def all(f, a):
    if len(a) == 0:
        return True
    return f(a[0]) and all(f, a[1:])

def is_inside(start, end, p):
    m = np.vstack((start, end)).T
    xx = (min(m[0]), max(m[0]))
    yy = (min(m[1]), max(m[1]))
    return p[0]>=xx[0] and p[0]<=xx[1] \
            and p[1]>=yy[0] and p[1]<=yy[1]

def sign(x):
    return int(abs(x)/x) if x!=0 else 0

# start, end and width must be int
def drawLine(start, end, width):
    assert type(start[0]) == int \
        and type(end[0]) == int, "drawLine: only takes int arguments"

    lx = end[0] - start[0]
    ly = end[1] - start[1]
    if lx != 0:
        m = ly/lx
        line = {}
        for x in range(0, lx, sign(lx)):
            for i in range(int(width*sqrt(m**2+1))):
                px, py = arr((x, m*x+i-width/2))
                px = int(px)
                py = int(py)
                r = (start[0]+px, start[1]+py)
                line[r] = 0
        line[(end[0], end[1])] = 0

        return line
    else:
        line = {}
        for y in range(0, ly, sign(ly)):
            for i in range(int(width)):
                px, py = arr((0+i-width/2, y))
                px = int(px)
                py = int(py)
                r = (start[0]+px, start[1]+py)
                line[r] = 0
        line[(end[0], end[1])] = 0

        return line


# Dic in
# index value map shift
def shift(indV, dr):
    out_indV = {}
    for p in indV:
        out_indV[(dr[0]+p[0], dr[1]+p[1])] = indV[p]
    return out_indV.copy()

# Arr in
# mapping indexes to values
# return (p, shape)
def arr2dic(arr):
    indV = {}
    it = np.nditer(arr, flags=['multi_index'])
    with it:
        for v in it:
            indV[it.multi_index] = v
    return indV.copy(), arr.shape

# WARNING! pixels outside "shape" will be cropped out
def dic2arr(pic, filler):
    dic, shape = pic
    out = np.full(shape, filler)
    for p in dic:
        try:
            out[p] = dic[p]
        except IndexError: pass
    return out.copy()

int8Add = lambda x, y: x+y if x+y < 255 else 255


# loc denoted the upper left starting point
# arr, list of dicpics
def drawPixel(backimg, *patchpairs):
    for patchpair in patchpairs:
        loc, pic = patchpair
        for p, v in pic.items():
            try:
                backimg[tuple(arr(p)+arr(loc))] = v
            except IndexError: pass

# merge arr
def mergePixel(backimg, *imgpairs):
    for imgpair in imgpairs:
        loc, img = imgpair
        it = np.nditer(img, flags=['multi_index'])
        with it:
            for v in it:
                try:
                    backimg[tuple(arr(it.multi_index) + arr(loc))] = v
                except IndexError: pass
        del img

def decompose_img(img):
    RGB = img[...,:3]
    try:
        alpha = img[...,3]
    except IndexError:
        print("image file no alpha channel")
        alpha = None
    return RGB[..., 0], RGB[..., 1], RGB[..., 2], alpha

channels = 1
imgName = "condense"
subPath = r"G:\我的雲端硬碟\同步學習中心\二下\光學導論\小組\Code\MatLab_Ver\data\condense\condense_4"
on = False
path = r"G:\我的雲端硬碟\同步學習中心\二下\光學導論\小組\Code\MatLab_Ver\data"
config = {'attn_layers':8}

# Attenuation Layer Height x Width x Thickness:
picSize = (3.3, 4.75, 0.5)
imgRes = (333, 480)
if imgRes[0]<imgRes[1]:
    imgRes = (imgRes[1], imgRes[0])
    picSize = (picSize[1], picSize[0], picSize[2])
    pic_rotate = True
numlayers = 8
picThick = picSize[2]/(numlayers -1)

A4Size = (297, 210) # mm
actThick = 2.1 # mm
actPicSize = tuple(map(lambda x: x*actThick/picThick, picSize))

A4Res = tuple(map(lambda x: int(x*imgRes[0]/actPicSize[0]), A4Size))

pixelSize = actThick/picThick*picSize[0]/imgRes[0]

def size2p(size):
    return int(size*A4Res[0]/A4Size[0])

markLine_width = 0.6 # mm
markLineP = size2p(markLine_width)

markdot_dia = 1.5 # mm
markdotp = size2p(markdot_dia)

print(f'''

Image Resolution: {imgRes}
A4 Resolution: {A4Res}

''')


def main(config, imgName, subPath):
    # Attenuation Layer Height x Width x Thickness:
    actWidth = config['real_width']
    actHeight = config['real_height']
    actThick = config['real_thick'] # in mm

    ############################################################################
                        # Parameters initialization #
    actPicSize = (actHeight, actWidth, actThick)
    imgRes = (config['imgRes'][1], config['imgRes'][0])
    if imgRes[0]<imgRes[1]:
        imgRes = (imgRes[1], imgRes[0])
        actPicSize = (actPicSize[1], actPicSize[0], actPicSize[2])
        pic_rotate = True
    numlayers = config['attn_layers']

    A4Size = (297, 210) # mm

    A4Res = tuple(map(lambda x: int(x*imgRes[0]/actPicSize[0]), A4Size))

    pixelSize = actPicSize[0]/imgRes[0]

    def size2p(size):
        return int(size*A4Res[0]/A4Size[0])

    markLine_width = 0.6 # mm
    markLineP = size2p(markLine_width)

    markdot_dia = 1.5 # mm
    markdotp = size2p(markdot_dia)

    ############################################################################

    A4x, A4y = A4Res

    border = [((0, 0), drawLine((0, 0), (0, A4y), markLineP))
        , ((0, 0), drawLine((0, A4y), (A4x, A4y), markLineP))
        , ((0, 0), drawLine((A4x, A4y), (A4x, 0), markLineP))
        , ((0, 0), drawLine((A4x, 0), (0, 0), markLineP))
        , ((0, 0), drawLine((int(A4x/2), 0), (int(A4x/2), A4y), markLineP))
        , ((0, 0), drawLine((0, int(A4y/2)), (A4x, int(A4y/2)), markLineP))]

    for m in range((numlayers-1)//4+1):
        outputImgtup = tuple(np.zeros(A4Res) for i in range(channels))
        print('''
####################
cropping: pages
####################
        ''')

        for n in range(numlayers%4 if m>=numlayers//4 else 4):
            print('''
####################
cropping: quarter
####################
            ''')
            img = iio.imread(os.path.join(subPath
                , f"{imgName}-{4*m+n}.png"))
            if img.shape[0] < img.shape[1]:
                if channels == 1:
                    img = img.T
                    print
                else:
                    newimg = np.zeros((img.shape[1], img.shape[0], img.shape[2]))
                    for i in range(channels):
                        newimg[:, :, i] = img[:, :, i].T
                    img = newimg
            # plt.imshow(img)
            # plt.show()

            imgSize = img.shape # (512, 512, 3)
            if channels != 1:
                imgTup = decompose_img(img)[:channels]
            else:
                imgTup = (img,)

            dots = [((int(A4x/200*k), int(A4y/200)), drawdot(markdotp)) for k in range(4*m+n)]

            def quarterDraw(img):
                img = centeredResize(img, (int(A4x/2), int(A4y/2)), 255)
                drawPixel(img, *dots)
                if on:
                    plt.imshow(img)
                    plt.xlabel("quarterDraw")
                    plt.show()
                return img

            corners = [(0, 0), (int(A4x/2), 0), (int(A4x/2), int(A4y/2)), (0, int(A4y/2))]
            def fullpageDraw(backImg, img):
                mergePixel(backImg, (corners[n], img))
                if on:
                    plt.imshow(img)
                    plt.xlabel("fullpageDraw")
                    plt.show()

            print("############ quarterDraw ###################")
            imgtup = tuple(map(quarterDraw, imgTup))
            print("############ fullpageDraw ###################")
            list(map(fullpageDraw, outputImgtup, imgtup))

        def borderDraw(img):
            drawPixel(img, *border)
            if on:
                plt.imshow(img)
                plt.xlabel("borderDraw")
                plt.show()
        list(map(borderDraw, outputImgtup))

        canvas = np.full((A4x, A4y, channels), 0)
        for i in range(channels):
            canvas[:, :, i] = outputImgtup[i]
            if on:
                plt.imshow(canvas[:, :, i])
                plt.xlabel("borderDraw")
                plt.show()
        if channels == 1:
            finalPath = os.path.join(subPath, f"FinalPrint-{imgName}-{str(m+1)}.png")
            print(f'''

########## Final Path Save as {os.path.join(subPath, f"FinalPrint-{imgName}-{str(m+1)}.png")}

            ''')
            iio.imwrite(finalPath, canvas[:, :, 0].astype('uint8'))
        else:
            finalPath = os.path.join(subPath, imgName, f"FinalPrint-{imgName}-{str(m+1)}.png")
        print(f'''

########## Final Path Save as {os.path.join(subPath, imgName, f"FinalPrint-{imgName}-{str(m+1)}.png")}

        ''')
        iio.imwrite(finalPath, canvas.astype('uint8'))


if __name__ == "__main__":
    main(config, imgName, subPath)
