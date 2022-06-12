import scipy as sp
import numpy as np
from numpy import array as arr
import random as rdn
from scipy.sparse.linalg import svds, lsqr, spsolve, lsmr, minres
from scipy.sparse import dia_matrix, coo_matrix
from scipy.linalg import lstsq
from scipy.optimize import lsq_linear
from math import e, cos, sin, pi, log, tan, sqrt, exp
import time
import imageio.v3 as iio
import os
from bidict import bidict as bd
import json
from numba import njit, float64, int64
from collections import deque
from matplotlib import pyplot as plt
import matplotlib.animation as ani
import gc

# M = 15
# N = 10
# a = M/N
# b = 2
#
# M = 10000000
# N = 2000000
# a = M/N
# b = 100
# #   N
# # M 口
# #
# assert M >= N, "error! M<N"
#
# timesum = 0
# start_time = time.process_time()
# ###################
# ele = []
# coord = ([], [])
#
# for j in range(N):
#     lower = 0 if j*a<b else int(j*a-b)
#     upper = M if j*a>M-b else int(j*a+b)
#     for i in range(lower, upper):
#         ele.append(rdn.random())
#         coord[0].append(i)
#         coord[1].append(j)
#
# print("generation complete!")
# data = (ele, coord)
# cM = coo_matrix(data, shape=(M, N))
# # Mt = [[rdn.random() if abs(i-j*a) < b else 0 for j in range(N)] for i in range(M)]
#
# x0 = np.array([rdn.random() for i in range(N)])
# vec = cM.dot(x0)
# # vec = np.array([cos(i*0.0001*b/N)+cos(i*0.05*b/N)+cos(i*0.001*b/N)+cos(i*0.4*b/N) for i in range(M)])
#
#
# sq_cM = cM.T.dot(cM)
# sq_vec = cM.T.dot(vec)
# ####################
# timesum = time.process_time() - start_time
# print("prepared for:", timesum, "seconds!")
#
# print("## Start!")
#
# def run2(Mt, it, vec):
#     ms4 = minres(Mt, vec, maxiter=it)
#     m4 = ms4[0]
#     dx4 = sum(abs(Mt.dot(m4) - vec))/M
#     print("Deviation:", dx4)
#     return ms4
#
# x={}
# sq_dM = dia_matrix(sq_cM)
# for i in range(1):
#     print("Iter#",  int(e**i)+10, " sq_dM")
#     timesum = 0
#     start_time = time.process_time()
#     x[i] = run2(sq_dM, int(e**i)+10, sq_vec)
#     timesum = time.process_time() - start_time
#     print(str(i)+". ", timesum, "Seconds!")
#     print()
# from pympler.tracker import SummaryTracker

from .yCombinator import HilbertCurve_TCO as HBC

def HBcurve(w, h):
    # tracker = SummaryTracker()
    mp = HBC(w, h)
    # tracker.print_diff()
    return mp

def exhaust(generator):
    deque(generator, maxlen=0)

class Mapping:
    attnMap = {}
    LFMap = {}

    @classmethod
    # (x, y, z starting from upper/left/back corner)
    def attnMap2Index(cls, config, coord):
        layersNum = config['attn_layers']
        resY, resX = config['attnRes']
        if not cls.attnMap:
            mp = HBcurve(resX, resY)
            cls.attnMap = mp
        else:
            mp = cls.attnMap

        x, y, z = coord
        ind = mp[(x, y)]
        index = ind*layersNum + z
        return int(index)

    @classmethod
    def attnMap2Coord(cls, config, index):
        layersNum = config['attn_layers']
        resY, resX = config['attnRes']
        if not cls.attnMap:
            mp = HBcurve(resX, resY)
            cls.attnMap = mp
        else:
            mp = cls.attnMap

        z = index % layersNum
        ind = index//layersNum
        coord = mp.inverse[ind]
        return (coord[0], coord[1], z)

    @classmethod
    def LFMap2Index(cls, config, coord):
        angleResX = config['angleRes']
        angleResY = config['angleRes']
        resY, resX = config['imgRes']
        if not cls.LFMap:
            mp = HBcurve(resX, resY)
            cls.LFMap = mp
        else:
            mp = cls.LFMap

        x, y, phi, beta = coord
        ind = mp[(x, y)]
        index = ind*angleResX*angleResY + (beta*angleResX+phi)
        return int(index)

    @classmethod
    def LFMap2Coord(cls, config, index):
        angleResX = config['angleRes']
        angleResY = config['angleRes']
        resY, resX = config['imgRes']
        if not cls.LFMap:
            mp = HBcurve(resX, resY)
            cls.LFMap = mp
        else:
            mp = cls.LFMap

        ind = index//(angleResX*angleResY)
        x, y = mp.inverse[ind]
        phi = (index % (angleResX*angleResY)) % angleResX
        beta = (index % (angleResX*angleResY)) // angleResX
        return (x, y, phi, beta)



def decompose_img(img):
    print(f"\n## decompose_img")
    RGB = img[...,:3]
    try:
        alpha = img[...,3]
    except IndexError:
        print("image file no alpha channel")
        alpha = None
    return [RGB[..., 0], RGB[..., 1], RGB[..., 2], alpha] # reshaping -> copy

@njit
def delta_angle(fov, angleRes):
    return 2*tan(fov/2)/(angleRes-1)

# return coordinate (x, y) (ray center in length)/ tan(phi), tan(beta)
def ang_ind2coord(config, phi_i, beta_i):
    angleResX = config['angleRes']
    angleResY = config['angleRes']
    fovX = config['fov']
    fovY = config['fov']
    delta_X = delta_angle(fovX, angleResX)
    delta_Y = delta_angle(fovY, angleResY)

    ####
    tan_beta = tan(fovY/2) - beta_i*delta_Y
    tan_phi = -(tan(fovX/2) - phi_i*delta_X)
    return (tan_phi, tan_beta)

@njit
def ang_ind2coord_gutted(phi_i, beta_i, args):
    # angleResX = config['angleRes']
    # angleResY = config['angleRes']
    # fovX = config['fov']
    # fovY = config['fov']
    attn_z, layers, depth, pixel_scale, angleResX, angleResY, fovX, fovY, resX, resY = args
    delta_X = delta_angle(fovX, angleResX)
    delta_Y = delta_angle(fovY, angleResY)

    ####
    tan_beta = tan(fovY/2) - beta_i*delta_Y
    tan_phi = -(tan(fovX/2) - phi_i*delta_X)
    return (tan_phi, tan_beta)

def xy_ind2coord(config, lfc): 0/0

# iterator wrapping
def xy_iter(config):
    res_y, res_x = config['imgRes']
    for y_i in range(res_y):
        for x_i in range(res_x):
            yield (x_i, y_i)

def ang_iter(config):
    angleResX = config['angleRes']
    angleResY = config['angleRes']
    for beta_i in range(angleResY):
        for phi_i in range(angleResX):
            yield (phi_i, beta_i)



color_scale = (2**8-1) # 8-bit color
offset = 0
dimming = 10
contrast = 2 # exponential

def attnLayersMap(config, attnVec):
    attnResY, attnResX = config['attnRes']
    layers = config['attn_layers']
    attndim = attnResY*attnResX*layers
    attnVec = (np.exp(-attnVec)**contrast)*color_scale + offset
    if max(attnVec) > color_scale:
        print("@@ OHOH @@")
    attnVec = np.log(attnVec)
    maxV = max(attnVec)
    attnVec = attnVec/maxV*color_scale
    attnVec = attnVec.astype('uint8')
    imgs = np.zeros([attnResX, attnResY, layers], dtype='uint8')
    for index in range(attndim):
        coord = Mapping.attnMap2Coord(config, index)
        imgs[coord] = attnVec[index]
    for i in range(layers):
        if plotting:
            plt.imshow(imgs[:,:,i])
            plt.show()
            plt.pause(10)
            plt.close('all')
            plt.pause(2)
    return imgs

def imageWrite(config, imgs, currentSub):
    imgName = config['imgName']
    layers = config['attn_layers']
    for n in range(layers):
        imgPath = os.path.join(currentSub, f"{imgName}-{n}.png")
        iio.imwrite(imgPath, imgs[:, :, n])

def imgPreview(config, imgs, subPath):
    layers = config['attn_layers']
    resY, resX = config['imgRes']
    args = loadArgs(config)
    fig, ax = plt.subplots()
    ims = []
    reconImg = np.zeros([resX, resY], dtype='uint8')
    for phi, beta in ang_iter(config):
        for x, y in xy_iter(config):
            coords = intersect_attn(beta, phi, x, y, args)
            attnVal = 1
            for coord in coords:
                attnVal *= (imgs[coord]/color_scale)
            reconImg[(x, y)] = int(attnVal*color_scale)
        im = ax.imshow(reconImg, animated=True, cmap='gray', vmin=0, vmax=255)
        ims.append([im])
        print(f'''

### Progress ({phi}, {beta}) ###''')
    anime = ani.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    # To save the animation, use e.g.
    #
    # ani.save("movie.mp4")
    #
    # or
    #
    writer = ani.FFMpegWriter(
        fps=15, metadata=dict(artist='Me'), bitrate=1800)
    anime.save(os.path.join(subPath, "demo.mp4"), writer=writer)
    plt.show()



def light_normalized(lumi_mat, phi_tan, beta_tan):
    ang_attn = 1/sqrt(phi_tan**2+phi_tan**2+1)
    new_lumi_mat = -np.log(lumi_mat/(dimming*color_scale*ang_attn), dtype='float16')
    del lumi_mat
    gc.collect()
    return new_lumi_mat



channels = 1

# load, normalize, and turn LF into dictionary format:
def initialize_LF(config):
    filePath = config['filePath']
    imgName = config['imgName']
    angleResX = config['angleRes']
    LFdic = {}
    for phi_i, beta_i in ang_iter(config):
        imgPath = os.path.join(filePath, "img", "{0}-{1}.png".format(imgName, phi_i+beta_i*angleResX))
        img = iio.imread(imgPath)
        if channels == 1:
            matlst = [img]
        else:
            matlst = decompose_img(img)[:channels]
        del img

        assert matlst[-1].dtype is np.dtype('uint8'), "initialize_LF: loaded PNG file not unsigned 8-bit image"

        # print(f"## {(phi_i, beta_i)} ##")

        LFdic[(phi_i, beta_i)] = matlst

    return LFdic

# return iterator through color channels of dic[(phi index, beta index)] of img mat
def initialize_normalized_LF(config):
    print(f"\n## initialize_normalized_LF")
    # replace non-normalized uint8 with normalized float16 with unused array deleted channel by channel
    # channel.pop backwards: alpha -> B -> G -> R
    dic_LF = initialize_LF(config)

    for n in range(channels):
        dic_LF_normalized = {}
        for phi_i, beta_i in ang_iter(config):
            matlst = dic_LF[(phi_i, beta_i)]
            tan_phi, tan_beta = ang_ind2coord(config, phi_i, beta_i)
            dic_LF_normalized[(phi_i, beta_i)] = light_normalized(matlst.pop(), tan_phi, tan_beta)

        yield dic_LF_normalized
        del dic_LF_normalized

# return which pixel (x, y) is on
# attenuation pixels = mat[(height, width, depth)] starting upper-left-back corner
@njit
def onPixel(x, y, args):
    # layers = config['attn_layers']
    # pixel_scale = config['attnRes'][0]/config['imgRes'][0]
    attn_z, layers, depth, pixel_scale, angleResX, angleResY, fovX, fovY, resX, resY = args
    return (round(pixel_scale*x), round(pixel_scale*y))

@njit
def isOnFrame(x, y, resX, resY):
    within = lambda x, resX: (x<resX) & (x>=0)
    return within(x, resX) & within(y, resY)

# take in lightfield coord (tan_phi, tan_beta, x, y)
# return list of intersected attn coord
# assume lightfield x, y == attn x, y
@njit
def intersect_attn(ibeta, iphi, ix, iy, args):
    # attn_z = config['attn_z'] # attn displacement from origin
    # layers = config['attn_layers']
    # depth = config['attn_thick']
    attn_z, layers, depth, pixel_scale, angleResX, angleResY, fovX, fovY, resX, resY = args

    back_z = -attn_z-depth/2
    step_z = depth/(layers-1)
    tan_phi, tan_beta = ang_ind2coord_gutted(iphi, ibeta, args)

    for n in range(layers):
        z = back_z+step_z*n
        x, y = (ix-z*tan_beta, iy+z*tan_phi)
        x, y = onPixel(x, y, args)
        if not isOnFrame(x, y, resX, resY):
            continue
        yield (x, y, n)

def loadArgs(config):
    pixelUnit = (config['imgRes'][0]/config['attn_width'])
    attn_z = config['attn_z']*pixelUnit # attn displacement from origin
    layers = config['attn_layers']
    depth = config['attn_thick']*pixelUnit
    pixel_scale = config['attnRes'][0]/config['imgRes'][0]
    angleResX = config['angleRes']
    angleResY = config['angleRes']
    fovX = config['fov']
    fovY = config['fov']
    resY, resX = config['imgRes']
    args = (attn_z, layers, depth, pixel_scale, angleResX, angleResY, fovX, fovY, resX, resY)
    return args

# def calcSizeOfIntersect(args):
#     angleResX, angleResY, fovX, fovY
#     attn_z, layers, depth, pixel_scale, angleResX, angleResY, fovX, fovY = args
#     resX, resY
#     for n in range(layers):
#         for phi in range(angleResX//2):
#             for x in range(resX):
#                 intersect_attn(beta, phi, x)
#
#         for beta in range(angleResY//2):

# matrix mapping flattened attn vector to "effected" flattened lightfield vector
# uint8 sparse matrix
def generate_relation_mat(config):
    print("\n## generate_relation_mat")
    print("### Light Field Generating... ###")

    args = loadArgs(config)
    attn_z, layers, depth, pixel_scale, angleResX, angleResY, fovX, fovY, resX, resY = args
    # coordinates unit in LF pixels

    car = None
    apn = list.append
    if access_global:
        global xdata, ydata, count, attndim, LFdim
    attndim = config['attnRes'][0]*config['attnRes'][1]*config['attn_layers']
    LFdim = config['imgRes'][0]*config['imgRes'][1]*config['angleRes']*config['angleRes']

    xdata = np.zeros(LFdim*layers, dtype='int')
    ydata = np.zeros(LFdim*layers, dtype='int')
    count = 0
    for iphi, ibeta in ang_iter(config):
        start_time = time.process_time()
        t1 = 0
        t2 = 0
        # tracker = SummaryTracker()
        LFMap2Index = Mapping.LFMap2Index
        attnMap2Index = Mapping.attnMap2Index

        for ix, iy in xy_iter(config):

            dt1 = time.process_time()
            indexLF = LFMap2Index(config, (ix, iy, iphi, ibeta))
            dt1 = time.process_time() - dt1

            dt2 = time.process_time()
            for interCoord in intersect_attn(ibeta, iphi, ix, iy, args):
                # car = ((indexLF, attnMap2Index(config, interCoord)), car)
                xdata[count] = indexLF
                ydata[count] = attnMap2Index(config, interCoord)
                count += 1

            dt2 = time.process_time() - dt2
            t1 += dt1
            t2 += dt2

        # tracker.print_diff()

        print(f"(iphi, ibeta): ({iphi}, {ibeta}) in {time.process_time() - start_time} Seconds!")
        print(f"# it takes Mapping.LFMap2Index total {t1} seconds\n# it takes intersect_attn total {t2} seconds")
    xdata = xdata[:count]
    ydata = ydata[:count]
    values = np.ones(count, dtype='uint8')
    print(f'''
##########################
xdata, len({len(xdata)}): {xdata}
ydata, len({len(ydata)}): {ydata}
values, len({len(values)}): {values}
##########################
''')
    cMat = coo_matrix((values, (xdata, ydata)), shape= [LFdim, attndim], dtype='uint8')


    return cMat


def generate_relation_vec(config, LFdic):
    print("\n## generate_relation_vec")
    apn = list.append
    angleResX = config['angleRes']
    angleResY = config['angleRes']
    resY, resX = config['imgRes']
    if access_global:
        global vec
    LFveclen = resX*resY*angleResX*angleResY
    vec = []
    for index in range(LFveclen):
        x, y, phi, beta = Mapping.LFMap2Coord(config, index)
        value = LFdic[(phi, beta)][x, y]
        apn(vec, value)

    return arr(vec)

def generate_relation_mat_banded(config):
    print("\n## generate_relation_mat_banded")
    cMat = generate_relation_mat(config)
    cMat = dia_matrix(cMat)
    return cMat

def S_MINR(mat, vec):
    print("\n## S_MINR")
    # squarized+minres
    sqMat = mat.T.dot(mat)
    sqVec = mat.T.dot(vec)
    return minres(sqMat, sqVec, maxiter= 50, show=True)

def S_LSMR(mat, vec):
    print("\n## S_LSMR")
    # lsmr
    return lsmr(mat, vec, maxiter= 15)

def S_lsq(mat, vec):
    print("\n## S_lsq")
    sqMat = mat.T.dot(mat)
    sqVec = mat.T.dot(vec)

    return lsq_linear(sqMat, sqVec, bounds=(0, -log(1/255)), verbose=2)

def loadConfigs(imgName, path=''):
    with open(os.path.join(path, "light_field_config.json"), 'r') as fh:
        configs = json.loads(fh.read())
    config = configs[imgName]
    config['attnRes'] = config['imgRes']
    return config

def saveConfig(config, imgName, path=''):
    try:
        with open(os.path.join(path, "light_field_config.json"), "r") as fh:
            configs = json.laods(fh.read())
    except:
        configs = {}
    configs[imgName] = config
    with open(os.path.join(path, "light_field_config.json"), 'w') as fh:
        fh.write(json.dumps(configs))


def solver(config):
    mat = generate_relation_mat(config)
    print(f'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            {gc.collect()} Garbage Collected
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ''')
    # mat = generate_relation_mat_banded
    for LFdic in initialize_normalized_LF(config):
        print(f'''
## LightField Dict length: {len(LFdic)} ##
        ''')
        print(f'''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                {gc.collect()} Garbage Collected
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ''')
        vec = generate_relation_vec(config, LFdic)
        del LFdic
        print(f'''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                {gc.collect()} Garbage Collected
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ''')

        print(f'''

#########################################
mat dim: {mat.shape}
mat: {mat}

vec dim: {vec.shape}
vec: {vec}

#########################################

        ''')

        start_time = time.process_time()
        attnVec = S_MINR(mat, vec)
        print(f"#### minres {time.process_time() - start_time} Seconds!")
        del mat

        # start_time = time.process_time()
        # attnVec = S_LSMR(mat, vec)
        # print(f"#### LSMR {time.process_time() - start_time} Seconds!")
        # del mat

        return attnVec

# return path to the sub folder, mkdir if not exist and store in config['subPaths']
def createSubPath(config, subfldr):
    filePath = config['filePath']
    subPath = os.path.join(filePath, subfldr)
    try:
        os.mkdir(subPath)
    except FileExistsError:
        pass
    subPaths = config.get('subPaths', None)
    if not subPaths:
        subPaths = {}
    subPaths[subfldr] = subPath
    config['subPaths'] = subPaths
    return subPath

arrsNames = ['attnVec', 'imgs']
def loadArrs(config, subPath):
    arrs = {}
    for arrsName in arrsNames:
        arrs[arrsName] = np.load(os.path.join(subPath, arrsName+'.npy'))
    return arrs

red_black5 = ['#a91e25', '#7c1822', '#510a12', '#420a0d', '#0e0e10']
orange_brown5 = ['#c75d05', '#ff760c', '#ff9549', '#f1cb8c', '#915634']
light_darkBrown5 = ['#b58181', '#a36767', '#814d4d', '#683d3d', '#562b2b']
light_darkBlue5 = ['#5766bd', '#4b54a0', '#3b468a', '#374081', '#2e3979']
light_darkCyan5 = ['#4ef1ef', '#2acaea', '#34bdc6', '#02a9b9', '#0095a4']
light_darkGreen = ['#93a47d', '#93b47d', '#93c47d', '#93d47d', '#93e47d']

clrs = light_darkBlue5 + light_darkGreen

def analyzeResults(ax, lst, label, color):
    hist = Cnt(lst)
    xy = arr([(n, log(hist[n] if hist[n]!=0 else 1)/log(10)) for n in range(min(hist), max(hist)+1)])
    ax.plot(xy.T[0], xy.T[1], label=label, color= color)

from collections import Counter as Cnt
def imgsDist(config, ax, imgs):
    layers = config['attn_layers']
    for n in range(layers):
        analyzeResults(ax, imgs[:,:, n].flatten(), label=f'layers {n}', color=clrs[n])
    ax.legend()

def attnVecDist(config, ax, attnVec):
    analyzeResults(ax, (np.exp(-attnVec)*color_scale).astype('int'), '', 'b')

def massDraw(config):
    subPaths = config['subPaths']
    gNum = len(subPaths)
    for n, sp in enumerate(subPaths):
        arrs = loadArrs(config, sp)
        ax = plt.subplot(2, gNum, n+1)
        attnVec = arrs['attnVec']
        attnVecDist(config, ax, attnVec)
        ax = plt.subplot(2, gNum, n+1+gNum)
        imgs = arrs['imgs']
        imgsDist(config, ax, imgs)


access_global = False
config = None
plotting = False
if __name__ == "__main__":
    imgName = "demo1"
    plt.ion()
    config = loadConfigs(imgName)
    def task1(config, subPath):
        total_time = time.process_time()
        attnVec, status = solver(config)
        np.save(os.path.join(subPath, "attnVec"), attnVec)
        imgs = attnLayersMap(config, attnVec)
        np.save(os.path.join(subPath, "imgs"), imgs)
        imageWrite(config, imgs, subPath)
        gc.collect()

        imgPreview(config, imgs, subPath)
        gc.collect()
        print(f'''

    ######################################################

            Total {time.process_time() - total_time} seconds

    ######################################################
        ''')

    filePath = config['filePath']
    subfldr = f"demo1_{n}"
    subPath = createSubPath(config, subfldr)
    task1(config, subPath)
    # for n in range(5):
    #     subfldr = f"demo1_{n}"
    #     dimming = n*2+1
    #     subPath = createSubPath(config, subfldr)
    #     task1(config, subPath)
    #     gc.collect()

from .imgCrop import main as imgCrop

def main_generate(config, imgName, subfldr):
    config['attnRes'] = config['imgRes']
    subPath = createSubPath(config, subfldr)
    saveConfig(config, imgName, subPath)

    attnVec, status = solver(config)
    np.save(os.path.join(subPath, "attnVec"), attnVec)
    gc.collect()

    imgs = attnLayersMap(config, attnVec)
    np.save(os.path.join(subPath, "imgs"), imgs)

    imageWrite(config, imgs, subPath)
    gc.collect()

    imgCrop(config, imgName, subPath)
    gc.collect()


def main_preview(imgName, subPath):
    config = loadConfigs(imgName, subPath)
#     try:
#         subPath = config['subPaths'][subfldr]
#     except KeyError:
#         raise Excpetion('''
# attenuationLayersGeneration.main_preview:
#     This subsetting doesn't exist
# ''')
    try:
        arrs = loadArrs(config, subPath)
    except FileNotFoundError:
        raise Excpetion('''
attenuationLayersGeneration.main_preview:
    Preview before generation
        ''')
    imgs = arrs['imgs']
    gc.collect()

    imgPreview(config, imgs, subPath)
    gc.collect()
