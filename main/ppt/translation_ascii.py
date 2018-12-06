import argparse
import numpy as np
from PIL import Image



# gray level
gscale1 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
gscale2 = "@%#*+=-:. "


def getAverageL(image):
    # return the average of rectangle grey value
    im = np.array(image)
    w, h = im.shape
    return np.average(im.reshape(w*h))


def covertImageToAscii(filename, cols=90, scale=0.43, moreLevels=False):
    # translating a color image to black and white(mode “L”),
    image = Image.open(filename).convert('L')
    W, H = image.size[0], image.size[1]
    print("input image dims: %d x %d " % (W, H))
    w = W/cols
    h = w/scale
    rows = int(H/h)

    print("cols: %d, rows: %d" % (cols, rows))
    print("tule dims: %d x %d" % (w, h))

    if cols > W or rows > H:
        print("Image too small for specified cols!")
        return []

    aimg = []
    for j in range(rows):
        y1 = int(j*h)
        y2 = int((j+1)*h)
        if j == rows-1:
            y2 = H

        aimg.append("")
        for i in range(cols):
            x1 = int(i*w)
            x2 = int((i+1)*w)

            if i == cols-1:
                x2 = W
            # cut a rectangle
            img = image.crop((x1, y1, x2, y2))
            # get the average grey levels of the rectangle
            avg = int(getAverageL(img))

            if moreLevels:
                gsval = gscale1[int((avg*69)/255)]
            else:
                gsval = gscale2[int((avg*9)/255)]

            aimg[j] += gsval
    return aimg

def main():
    descStr = "This program starts"
    parser = argparse.ArgumentParser(description=descStr)
    parser.add_argument('--file', dest='imgFile', required=True)
    parser.add_argument('--scale', dest='scale', required=False)
    parser.add_argument('--out', dest='outFile', required=False)
    parser.add_argument('--cols', dest='cols', required=False)
    parser.add_argument('--morelevels', dest='moreLevels', action='store_true')
    args = parser.parse_args()

    imgFile = args.imgFile
    outFile = 'out.txt'
    if args.outFile: outFile = args.outFile

    #the height of output ascii picture is longer than width
    scale = 0.43
    if args.scale: scale = int(args.scale)
    cols = 90
    if args.cols: cols = int(args.cols)
    print('generating ASCII art..')

    aimg = covertImageToAscii(imgFile, cols, scale, args.moreLevels)
    with open(outFile, 'w+') as f:
        for row in aimg:
            f.write(row + '\n')

    print('ASCII art written to %s' % outFile)


if __name__ == '__main__':
    main()
