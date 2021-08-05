#!python3
"""
Python 3 wrapper for identifying objects in images

Requires DLL compilation

Original *nix 2.7: https://github.com/pjreddie/darknet/blob/0f110834f4e18b30d5f101bf8f1724c34b7b83db/python/darknet.py
Windows Python 2.7 version: https://github.com/AlexeyAB/darknet/blob/fc496d52bf22a0bb257300d3c79be9cd80e722cb/build/darknet/x64/darknet.py

@author: Philip Kahn, Aymeric Dujardin
@date: 20180911
"""
# pylint: disable=R, W0401, W0614, W0703
import cv2
import pyzed.sl as sl
from ctypes import *
import math
import random
import os
import numpy as np
import statistics
import sys
import getopt
from random import randint
import time
import socket


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1


def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("darknet.so", RTLD_GLOBAL)
hasGPU = True
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                print("Flag value '"+tmp+"' not forcing CPU mode")
        except KeyError:
            # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError:
                pass
            # print(os.environ.keys())
            # print("FORCE_CPU flag undefined, proceeding with GPU")
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            print("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was
            # compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            print("Environment variables indicated a CPU run, but we didn't find `" +
                  winNoGPUdll+"`. Trying a GPU run anyway.")
else:
    #lib = CDLL("../libdarknet/libdarknet.so", RTLD_GLOBAL)

    #lib = CDLL(os.path.join(os.getcwd(), "libdarknet.so"), RTLD_GLOBAL)
    lib = CDLL(os.path.join(darknet_path, "libdarknet.so"), RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(
    c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            nameTag = meta.names[i]
        else:
            nameTag = altNames[i]
        res.append((nameTag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
    """
    Performs the detection
    """
    custom_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    custom_image = cv2.resize(custom_image, (lib.network_width(
        net), lib.network_height(net)), interpolation=cv2.INTER_LINEAR)
    im, arr = array_to_image(custom_image)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(
        net, image.shape[1], image.shape[0], thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    res = []
    if debug:
        print("about to range")
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    nameTag = meta.names[i]
                else:
                    nameTag = altNames[i]
                res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h), i))
    res = sorted(res, key=lambda x: -x[1])
    free_detections(dets, num)
    return res


netMain = None
metaMain = None
altNames = None


def getObjectDepth(depth, bounds):
    area_div = 2

    x_vect = []
    y_vect = []
    z_vect = []

    for j in range(int(bounds[0] - area_div), int(bounds[0] + area_div)):
        for i in range(int(bounds[1] - area_div), int(bounds[1] + area_div)):
            z = depth[i, j, 2]
            if not np.isnan(z) and not np.isinf(z):
                x_vect.append(depth[i, j, 0])
                y_vect.append(depth[i, j, 1])
                z_vect.append(z)
    try:
        x = statistics.median(x_vect)
        y = statistics.median(y_vect)
        z = statistics.median(z_vect)
    except Exception:
        x = -1
        y = -1
        z = -1
        pass

    return x, y, z


def generateColor(metaPath):
    random.seed(42)
    f = open(metaPath, 'r')
    content = f.readlines()
    class_num = int(content[0].split("=")[1])
    color_array = []
    for x in range(0, class_num):
        color_array.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    return color_array

def generateMap(xCoord,yCoord,xExtent,yExtent,width,height):
     # Note that coordinates may be out of bound
     if (xCoord+xExtent/2) < 0:
        xCoord_sc = 0.000001
     elif (xCoord+xExtent/2) > width:
        xCoord_sc = 0.999999
     else:
        xCoord_sc = (xCoord+xExtent/2)/width
     # Note that the y is flipped
     if (yCoord+yExtent/2) < 0:
        yCoord_sc = 0.999999
     elif (yCoord+yExtent/2) > height:
        yCoord_sc = 0.000001
     else:
        yCoord_sc = 1-(yCoord+yExtent/2)/height
     # Now compute the mapping
     mapx = math.floor(xCoord_sc * 5) + 1
     mapy = math.floor(yCoord_sc * 2) + 1
     #  print(mapx,mapy)
     return mapx, mapy

def main(argv):

    # TCP communication

    serverIp = 'localhost'
    tcpPort = 9998
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((serverIp, tcpPort))


    thresh = 0.25
    darknet_path="../libdarknet/"
    configPath = darknet_path + "cfg/yolov3-tiny.cfg"
    weightPath = "yolov3-tiny.weights"
    metaPath = "coco.data"
    svoPath = None

    help_str = 'darknet_zed.py -c <config> -w <weight> -m <meta> -t <threshold> -s <svo_file>'
    try:
        opts, args = getopt.getopt(
            argv, "hd:c:w:m:t:s:", ["darknet=","config=", "weight=", "meta=", "threshold=", "svo_file="])
    except getopt.GetoptError:
        print (help_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print (help_str)
            sys.exit()
        elif opt in ("-d", "--darknet"):
            darknet_path = arg
        elif opt in ("-c", "--config"):
            configPath = arg
        elif opt in ("-w", "--weight"):
            weightPath = arg
        elif opt in ("-m", "--meta"):
            metaPath = arg
        elif opt in ("-t", "--threshold"):
            thresh = float(arg)
        elif opt in ("-s", "--svo_file"):
            svoPath = arg

    init = sl.InitParameters()
    init.coordinate_units = sl.UNIT.METER
    if svoPath is not None:
        init.set_from_svo_file(svoPath)

    cam = sl.Camera()
    if not cam.is_opened():
        print("Opening ZED Camera...")
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    # Use STANDARD sensing mode
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD
    mat = sl.Mat()
    point_cloud_mat = sl.Mat()

    # Import the global variables. This lets us instance Darknet once, then just call performDetect() again without instancing again
    global metaMain, netMain, altNames  # pylint: disable=W0603
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = load_meta(metaPath.encode("ascii"))
    if altNames is None:
        # In thon 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    color_array = generateColor(metaPath)

    print("Running...")

    key = ''
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat, sl.VIEW.LEFT)
            image = mat.get_data()
            #img=cv2.imread(image)
            height,width,c=image.shape
            height_min=height//2
            width_min=width//2

            cam.retrieve_measure(
                point_cloud_mat, sl.MEASURE.XYZRGBA)
            depth = point_cloud_mat.get_data()

            # Do the detection
            detections = detect(netMain, metaMain, image, thresh)
            num_dec = len(detections)

            print(chr(27) + "[2J")

            #  print(chr(27) + "[2J"+"**** " +
            #      str(len(detections)) + " Results ****")
            if num_dec > 0:
                exp_map = []
                ind_dec = 0
                for detection in detections:
                    dec_tmp = []
                    label = detection[0]
                    confidence = detection[1]
                    pstring = label+": "+str(np.rint(100 * confidence))+"%"
                    print(pstring)
                    bounds = detection[2]
                    yExtent = int(bounds[3])
                    xEntent = int(bounds[2])
                    # Coordinates are around the center
                    xCoord = int(bounds[0] - bounds[2]/2)
                    yCoord = int(bounds[1] - bounds[3]/2)
                    mapx, mapy = generateMap(xCoord,yCoord,xEntent,yExtent,width,height)
                    boundingBox = [ [xCoord, yCoord], [xCoord, yCoord + yExtent], [xCoord + xEntent, yCoord + yExtent], [xCoord + xEntent, yCoord] ]
                    thickness = 1
                    x, y, z = getObjectDepth(depth, bounds)
                    if z == -1:
                        distance = 0.01
                    else:
                        distance = math.sqrt(x * x + y * y + z * z)
                        distance = "{:.2f}".format(distance)
                    exp_map.append(mapx)
                    exp_map.append(mapy)
                    exp_map.append(distance)
                    dec_tmp.append([mapx, mapy, distance])
                    print("Horizontal position:", dec_tmp[0][0])
                    print("Vertical position:", dec_tmp[0][1])
                    print("Distance:", dec_tmp[0][2])
                # cv2.rectangle(image, (xCoord-thickness, yCoord-thickness), (xCoord + xEntent+thickness, yCoord+(18 +thickness*4)), color_array[detection[3]], -1)
                # cv2.putText(image, label + " " +  (str(distance) + " m"), (xCoord+(thickness*4), yCoord+(10 +thickness*4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                # cv2.rectangle(image, (xCoord-thickness, yCoord-thickness), (xCoord + xEntent+thickness, yCoord + yExtent+thickness), color_array[detection[3]], int(thickness*2))
                    cv2.rectangle(image, (width*(dec_tmp[0][0]-1)//5,height*(2-dec_tmp[0][1])//2),(width*(dec_tmp[0][0])//5,height*(3-dec_tmp[0][1])//2),color_array[detection[3]],10)
                # print((width//5*(exp_map[0][0]-1),height//2*(exp_map[0][1]-1)))
                # print((width//5*(exp_map[0][0]),height//2*(exp_map[0][1])))
                # print(color_array[detection[3]])
                    ind_dec = ind_dec + 1
                msg = ''.join(str(exp_map))
                # msg = 'test'
                msg = msg.replace('[','')
                msg = msg.replace(']','')
                msg = msg.replace(',','')
                msg = msg.replace('\'','')
                msg = msg.replace(' ','')
                print(msg)
                s.sendall(msg.encode('utf-8'))
                time.sleep(0.03)
            # print(height,width,height_min,width_min)
            # print(exp_map)
            cv2.rectangle(image,(height_min,width_min),(height,width),(255,0,0),2)
            cv2.imshow("ZED", image)
            key = cv2.waitKey(5)
        else:
            key = cv2.waitKey(5)
    cv2.destroyAllWindows()


    s.close()

    cam.close()
    print("\nFINISH")

#img=cv2.imread(image)

	   # height,width,channels=img.shape
if __name__ == "__main__":
    main(sys.argv[1:])
