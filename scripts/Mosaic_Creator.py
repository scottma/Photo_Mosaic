#!/usr/bin/env python
import os, random, argparse
from PIL import Image
import numpy as np
import cv2
import math
from scipy import ndimage

parser = argparse.ArgumentParser(description='Creates a photomosaic from input images')
parser.add_argument('--target', dest='target', required=True, help="Image to create mosaic from")
parser.add_argument('--images', dest='images', required=True, help="Diectory of images")
parser.add_argument('--grid', nargs=2, dest='grid', required=True, help="Size of photo mosaic")
parser.add_argument('--output', dest='output', required=False)

args = parser.parse_args()

def nptoimg(nparry):
    return Image.fromarray(nparry, "RGBA")


cnt = 0

def deskew3(targetim, inputim, max_skew=1000):
    global cnt
    img_before = np.array(targetim)
    # img_after = np.array(inputim)
    
    # targetim.save("target{}.jpg".format(cnt))

    # img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    # ret, img_thresh = cv2.threshold(img_gray,0,255,0)
    img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.Canny(img_gray, 150, 150, apertureSize=3)

    cntrs = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cntrs[0]) == 0:
        return inputim

    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]    
    rotrect = cv2.minAreaRect(cntrs[0])

    box = cv2.boxPoints(rotrect)
    box = np.int0(box)
    # cv2.imwrite("img_thresh{}.jpg".format(cnt), img_thresh)
    img_thresh = cv2.drawContours(img_thresh,[box],0,(0,0,255),1)
    # cv2.imwrite("img_box{}.jpg".format(cnt), img_thresh)

    angle = rotrect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # cv2.imwrite("before{}.jpg".format(cnt), img_after)
    img_rot = inputim.rotate( angle ).convert('RGBA')
    # img_rotated = np.array(img_rot)
    # img_rotated = ndimage.rotate(img_after, angle)
    # cv2.imwrite("after{}.jpg".format(cnt), img_rotated)
    # print(f"Angle is {angle:.04f}")
    cnt = cnt + 1
    # img_rot.save("final{}.jpg".format(cnt))
    return img_rot



def deskew2(targetim, inputim, max_skew=1000):
    global cnt
#    targetim.save("target{}.jpg".format(cnt))

    img_before = np.array(targetim)
    img_after = np.array(inputim)

    img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100)

 #   cv2.imwrite("gray{}.jpg".format(cnt), img_gray)
 #   cv2.imwrite("edge{}.jpg".format(cnt), img_edges)


    cnt = cnt + 1


    if lines is None:
        return inputim

    angles = []

    for [[x1, y1, x2, y2]] in lines:
        cv2.line(img_before, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)
    img_rotated = ndimage.rotate(img_after, median_angle)
    print(f"Angle is {median_angle:.04f}")
    return nptoimg(img_rotated)

def deskew(targetim, inputim, max_skew=1000):
    
    t = np.array( targetim )

    height, width, channels = t.shape

    # Create a grayscale image and denoise it
    im_gs = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
    im_gs = cv2.fastNlMeansDenoising(im_gs, h=3)

    # Create an inverted B&W copy using Otsu (automatic) thresholding
    im_bw = cv2.threshold(im_gs, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Detect lines in this image. Parameters here mostly arrived at by trial and error.
    lines = cv2.HoughLinesP(
        im_bw, 1, np.pi / 180, 200, minLineLength=width / 12, maxLineGap=width / 150
    )

    # Collect the angles of these lines (in radians)
    angles = []

    if lines is None:
        return inputim


    #cprint( "l {}".format(len(lines)))

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angles.append(np.arctan2(y2 - y1, x2 - x1))

    # If the majority of our lines are vertical, this is probably a landscape image
    landscape = np.sum([abs(angle) > np.pi / 4 for angle in angles]) > len(angles) / 2

    # Filter the angles to remove outliers based on max_skew
    if landscape:
        angles = [
            angle
            for angle in angles
            if np.deg2rad(90 - max_skew) < abs(angle) < np.deg2rad(90 + max_skew)
        ]
    else:
        angles = [angle for angle in angles if abs(angle) < np.deg2rad(max_skew)]

    if len(angles) < 5:
        # Insufficient data to deskew
        return inputim

    # Average the angles to a degree offset
    angle_deg = np.rad2deg(np.median(angles))
    print("{}".format(angle_deg))
    i = np.array( inputim )

    # If this is landscape image, rotate the entire canvas appropriately
    if landscape:
        if angle_deg < 0:
            i = cv2.rotate(i, cv2.ROTATE_90_CLOCKWISE)
            angle_deg += 90
        elif angle_deg > 0:
            i = cv2.rotate(i, cv2.ROTATE_90_COUNTERCLOCKWISE)
            angle_deg -= 90

    # Rotate the image by the residual offset
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle_deg, 1)
    i = cv2.warpAffine(i, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return nptoimg(i)

def saveImg(img, filename, ext):
    if os.path.isfile(filename):
        os.remove( filename )
    img.save(filename, ext)

def getImages(images_directory):
    files = sorted(os.listdir(images_directory))
    images = []
    for file in files:
        filePath = os.path.abspath(os.path.join(images_directory, file))
        try:
            fp = open(filePath, "rb")
            im = Image.open(fp)
            im = im.convert('RGBA')
            images.append([im, file])
            im.load()
            fp.close()
        except:
            print("Invalid image: %s" % (filePath,))
    return (images)

def getAverageRGB(image):
    im = np.array(image)
    w, h, d = im.shape
    return (tuple(np.average(im.reshape(w * h, d), axis=0)))


def splitImage(image, size):
    W, H = image.size[0], image.size[1]
    m, n = size
    w, h = int(W / n), int(H / m)
    imgs = []
    for j in range(m):
        for i in range(n):
            imgs.append(image.crop((i * w, j * h, (i + 1) * w, (j + 1) * h)))
    return (imgs)


def getBestMatchIndex(input_avg, avgs):
    avg = input_avg
    index = 0
    min_index = 0
    min_dist = float("inf")
    for val in avgs:
        # if( val[0] == avg[0] and val[1] == avg[1] and val[2] == avg[2] ):
        #     return (min_index)
        dist = ((val[0] - avg[0]) * (val[0] - avg[0]) +
                (val[1] - avg[1]) * (val[1] - avg[1]) +
                (val[2] - avg[2]) * (val[2] - avg[2]))
        if dist < min_dist:
            min_dist = dist
            min_index = index
        index += 1

    # print( "input %s avg %s" % (str(input_avg), str(avgs[min_index])))

    return (min_index)


def createImageGrid(target_image, images, dims):

    m, n = dims
    multipler_width = max([img.size[0] for img in images])
    multipler_height = max([img.size[1] for img in images])

    width = round( multipler_width / resize_multipler )
    height = round( multipler_height / resize_multipler )

    width_diff = round((multipler_width-width)/2)
    height_diff = round((multipler_height-height)/2)

    print("{}x{} widthxheight {}x{} dim {}x{}".format(multipler_width, multipler_height, width, height, m, n))
    grid_img = Image.new('RGBA', (n * width, m * height))
    # target_image = target_image.convert('RGBA')
    grid_img.putalpha(1)
    grid_img = target_image
    for index in range(len(images)):
        row = int(index / n)
        col = index - n * row
        # images[index].putalpha(1)
        v = getAverageRGB(images[index])
        if v[0] == 255.0 and v[1] == 255.0 and v[2] == 255.0:
            continue
        if v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0:
            continue
        grid_img.paste(images[index], (round(col * width - width_diff), round(row * height - height_diff)), images[index].convert('RGBA'))
    return (grid_img)


def createPhotomosaic(target_image, input_images, grid_size,
                      reuse_images):

    target_small_images = splitImage(target_image, grid_size)
    print("splitted")
    output_images = []
    count = 0
    batch_size = int(len(target_small_images) / 10)
    avgs = []

    
    for arr in input_images:
        try:
            img = arr[0]
            v = getAverageRGB(img)
            avgs.append( v )
            # print(" %s  %s" % ( arr[1] ,str(v) ))
        except ValueError:
            continue

    for img in target_small_images:
        avg = getAverageRGB(img)
        match_index = getBestMatchIndex(avg, avgs)
        # output_images.append(input_images[match_index][0])
        output_images.append(deskew3(img, input_images[match_index][0]))
        # print("avg %s %s" % (str(avg),input_images[match_index][1]))
        # if count > 0 and batch_size > 10 and count % batch_size is 0:
        if count > 0 and batch_size > 10 and (count % batch_size) == 0:
            print('processed %d of %d...' % (count, len(target_small_images)))
        count += 1
        # remove selected image from input if flag set
        if not reuse_images:
            input_images.remove(match_index)

    mosaic_image = createImageGrid(target_image, output_images, grid_size)
    return (mosaic_image)


### ---------------------------------------------


target_image = Image.open(args.target)

# input images
print('reading input folder...')
input_images = getImages(args.images)

# check if any valid input images found
if input_images == []:
    print('No input images found in %s. Exiting.' % (args.images,))
    exit()

# shuffle list - to get a more varied output?
random.shuffle(input_images)

# size of grid
grid_size = (int(args.grid[0]), int(args.grid[1]))

# output
output_filename = 'mosaic.jpeg'
if args.output:
    output_filename = args.output

# re-use any image in input
reuse_images = True 

# resize the input to fit original image size?
resize_input = True

resize_multipler = 2.5

print('starting photomosaic creation...')

# if images can't be reused, ensure m*n <= num_of_images
if not reuse_images:
    if grid_size[0] * grid_size[1] > len(input_images):
        print('grid size less than number of images')
        exit()

# resizing input
if resize_input:
    print('resizing images...')
    # for given grid size, compute max dims w,h of tiles
    dims = (int(target_image.size[0] / grid_size[1] * resize_multipler),
            int(target_image.size[1] / grid_size[0] * resize_multipler))
    print("max tile dims: %s" % (dims,))
    # resize
    for arr in input_images:
        img = arr[0]
        img.thumbnail(dims)


# create photomosaic
mosaic_image = createPhotomosaic(target_image, input_images, grid_size, reuse_images)

# write out mosaic
if mosaic_image.height >= 65500 or mosaic_image.width >= 65500:
    if mosaic_image.height > mosaic_image.width:
        w = round(655000 * mosaic_image.width / mosaic_image.height)
        h = 65500
    else:
        h = round(655000 * mosaic_image.height / mosaic_image.width)
        w = 65500
    mosaic_image.resize( size=[w, h], resample=Image.NEAREST)

print( "{}x{}".format(mosaic_image.height,mosaic_image.width))
mosaic_image.save(output_filename, 'png')

print("saved output to %s" % (output_filename,))
print('done.')
