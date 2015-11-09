from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from skimage import feature
from PIL import Image
import scipy
import math

#shows pointer location 
class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.001f}'.format(x, y, z)
        
def edgedetection (image, o):
    dim = image.shape
    edges = feature.canny(image, sigma=o)
    listEdges = []
    linecoords = []
    #scan across array of edges to detect the largest continuous edge around the foot and isolate this for floodfill
    for x in range (0,dim[0]):
        for y in range (0,dim[1]):
            #if one of the pixels is an edge start scan and black this pixel out
            if edges[x,y] == True:
                linecoords=np.asarray([x,y])
                edges[x,y]=False
                #loop over successive neighbouring pixels
                while True:
                    close = np.zeros((4,3))
                    far = np.zeros((4,3))
                    close[0,0], close[0,1], close[0,2] =  x-1, y, edges[x-1,y]
                    close[1,0], close[1,1], close[1,2] =  x+1, y, edges[x+1,y]
                    close[2,0], close[2,1], close[2,2] =  x, y-1, edges[x,y-1]
                    close[3,0], close[3,1], close[3,2] =  x, y+1, edges[x,y+1]
                    far[0,0], far[0,1], far[0,2] =  x+1, y+1, edges[x+1,y+1]
                    far[1,0], far[1,1], far[1,2] =  x+1, y-1, edges[x+1,y-1]
                    far[2,0], far[2,1], far[2,2] =  x-1, y+1, edges[x-1,y+1]
                    far[3,0], far[3,1], far[3,2] =  x-1, y-1, edges[x-1,y-1]
                    closeneighbour = np.argwhere(close[:,2]>0)
                    farneighbour = np.argwhere(far[:,2]>0)
                    if len(closeneighbour>0):
                        index = closeneighbour[0,0]
                        i,j = int(close[index,0]), int(close[index,1])
                        coords = np.asarray([i,j])
                        linecoords = np.vstack((linecoords, coords))
                        edges[i,j]=False
                        x,y=i,j
                    elif len(farneighbour>0):
                        index = farneighbour[0,0]
                        i,j = int(far[index,0]), int(far[index,1])
                        coords = np.asarray([i,j])
                        linecoords = np.vstack((linecoords, coords))
                        edges[i,j]=False
                        x,y=i,j
                    else:
                        listEdges.append(linecoords)
                        break
    
    #find the coordinates of the longest ledge that was found - hopefullly the outline of the foot
    maxlen,index = 0,0
    for i in range (0,len(listEdges)):
        length = len(listEdges[i])
        if length>maxlen:
            maxlen=length
            index=i
    outlinecoords = listEdges[index]  
    
    #make an image of this outline edge
    outline = np.zeros((dim[0],dim[1]))
    for i in range (0,len(outlinecoords)):
        x,y = outlinecoords[i,0], outlinecoords[i,1]
        outline[x,y]=1
        
    #make a cut-out image using this outline
    vals = []
    edgecutout = np.zeros((dim[0],dim[1]))
    for x in range (0,dim[0]):
        for y in range (0,dim[1]):  
            u,d,l,r = outline[x,y:],outline[x,:y],outline[:x,y],outline[x:,y]
            u,d,l,r = np.argwhere(u>0), np.argwhere(d>0), np.argwhere(l>0), np.argwhere(r>0)
            if len(u) > 0 and len(d) > 0 and len(l) > 0 and len(r) > 0:
                edgecutout[x,y] = image[x,y]
                vals.append(edgecutout[x,y])
            elif outline[x,y] == True:
                edgecutout[x,y] = image[x,y]
                vals.append(edgecutout[x,y])
    average = np.average(vals)
    stdev = np.std(vals)      
    return outline, edgecutout, average, stdev, vals

def registration (lToe, lHeel, rToe, rHeel, lImage, rImage): 
    ldx, ldy = float(lHeel[0]-lToe[0]), (lHeel[1]-lToe[1])
    rdx, rdy = float(rHeel[0]-rToe[0]), (rHeel[1]-rToe[1])
    #work out angle between vertical for both images
    leftAngle = math.atan(ldx/ldy)*(180.0/math.pi)
    rightAngle = math.atan(rdx/rdy)*(180.0/math.pi)
    print leftAngle, rightAngle
    #work out scale difference between both images
    ldist = math.sqrt((ldx*ldx)+(ldy*ldy))
    rdist = math.sqrt((rdx*ldx)+(rdy*rdy))
    #scaleFactor = float(1/(rdist/ldist))
    #rotate both images and scale right foot image to equal left foot image
    rImage = scipy.ndimage.interpolation.rotate(rImage,-rightAngle)
    lImage = scipy.ndimage.interpolation.rotate(lImage,-leftAngle)
    #rImage = scipy.misc.imresize(rImage, scaleFactor, mode=None)
    return lImage, rImage
#    sizerightx, sizerighty = rightrotated.shape[0], rightrotated.shape[1]
#    sizeleftx, sizelefty = edgecutoutleft.shape[0], edgecutoutleft.shape[1]
#    scaledleft = np.zeros((sizerightx,sizerighty))
#    scaledleftoutline = np.zeros((sizerightx,sizerighty))
#    minx = int(sizerightx/2) - int(sizeleftx/2) + xoffset
#    maxx = int(sizerightx/2) + int(sizeleftx/2) + xoffset
#    miny = int(sizerighty/2) - int(sizelefty/2) + yoffset
#    maxy = int(sizerighty/2) + int(sizelefty/2) + yoffset
#    scaledleft[minx:maxx,miny:maxy] = edgecutoutleft
#    scaledleftoutline[minx:maxx,miny:maxy] = outlineleft

def spotAve (image, x, y, r):
    vals = []
    xmin, xmax, ymin, ymax = x-r, x+r, y-r, y+r
    for i in range (xmin, xmax):
        for j in range (ymin, ymax):
            dx, dy = i-x, j-y
            if image[i,j] > 10:
                if math.sqrt((dx*dx) + (dy*dy)) <= r:
                    vals.append(image[i,j])    
    return np.sum(vals), len(vals)
    
def scanForSpots (image, threshold):
    dim = image.shape
    radii = [2,3,4,5]
    hotspots = np.asarray([0,0,0])
    # go through all pixels in image that are on the foot
    for x in range (0,dim[0]):
        for y in range (0,dim[1]):
            if image[x,y] > 10:
                for radius in radii:
                    innerSum, innerN = spotAve(image, x, y, radius)
                    outerSum, outerN = spotAve(image, x, y, radius+5)
                    annulusAve = (outerSum - innerSum)/(outerN - innerN)
                    innerAve = innerSum/innerN
                    if innerAve - annulusAve > threshold:
                        coords = [x, y, radius]
                        hotspots = np.vstack((hotspots, coords))
    return hotspots                
                    
                    
                        
            
                
#load txt array of image and split down the middle for left and right views
im = np.loadtxt('jp.txt')
left = im[:,:160]
right = im[:,160:]

sigma=4.5 
angle = -12
xoffset = 7
yoffset = 8     

outlineleft,edgecutoutleft, lave, lstd, lvals = edgedetection(left,sigma)  
outlineright,edgecutoutright, rave, rstd, rvals = edgedetection(right,sigma) 

mintemp = [np.min(lvals), np.min(rvals)] 
maxtemp = [np.max(lvals), np.max(rvals)]
mintemp = np.min(mintemp)
maxtemp = np.max(maxtemp)

leftedge = edges = feature.canny(left, sigma=sigma) 
rightedge = feature.canny(right, sigma=sigma)
flippedright=np.fliplr(edgecutoutright)
flippedrightoutline=np.fliplr(outlineright)

#rotate the image of the foot by number of degrees so it aligns with other foot
rightrotated = scipy.ndimage.interpolation.rotate(flippedright,angle)
outlinerightrotated = scipy.ndimage.interpolation.rotate(flippedrightoutline,angle)
#image = Image.fromarray(rightrotated)
#image.show()

xoffset, yoffset =10,-8

lToe = [90,21]
lHeel = [88,226]
rToe = [81,28]
rHeel = [23,221]
limage,rimage=registration(lToe,lHeel,rToe,rHeel,edgecutoutleft, edgecutoutright)
limage=np.fliplr(limage)
dxL, dyL, dxR, dyR = lToe[0] - lHeel[0], lToe[1] - lHeel[1], rToe[0] - rHeel[0], rToe[1] - rHeel[1]
distL = math.sqrt((dxL*dxL)+(dyL*dyL))
distR = math.sqrt((dxR*dxR)+(dyR*dyR))
factor = 1/(distL/distR)
#limage = scipy.misc.imresize(limage,factor)
print limage.max(), limage.min()

sizerightx, sizerighty = rimage.shape[0], rimage.shape[1]
sizeleftx, sizelefty = limage.shape[0], limage.shape[1]
scaledleft = np.zeros((sizerightx,sizerighty))
minx = int(sizerightx/2) - int(sizeleftx/2) + xoffset
maxx = int(sizerightx/2) + int(sizeleftx/2) + xoffset
miny = int(sizerighty/2) - int(sizelefty/2) + yoffset
maxy = int(sizerighty/2) + int(sizelefty/2) + yoffset
scaledleft[minx:maxx,miny:maxy] = limage


subtraction = rimage - scaledleft
#addoutlines = scaledleftoutline + outlinerightrotated

#fig, ax1 = plt.subplots(nrows=1, ncols=1)
#ax1,ax3,ax4,ax5 = axes.flat
#right=np.fliplr(left)
#print right.shape
#fig = plt.figure(frameon=False)
#fig.set_size_inches(5.0,7.5)
#ax = plt.Axes(fig, [0., 0., 1., 1.])
#ax.set_axis_off()
#fig.add_axes(ax)
#ax.imshow(right, interpolation = None)
#fig.savefig('jp.png')
#plt.show()

edgesl = feature.canny(left, sigma=sigma)
edgesr = feature.canny(right, sigma=sigma)
## display results
#fig, ax1 = plt.subplots(nrows=1, ncols=1)
#ax1,ax2,ax3,ax4,ax5 = axes.flat

plt.imshow(subtraction, interpolation = 'none', vmin = -3, vmax = 3)
plt.axis('off')
plt.colorbar()
#ax1.set_title('right', fontsize=20)
#for spot in hotspots:
#    y,x,r = spot
#    c = plt.Circle((x, y), r, color='k', linewidth=2, fill=False)
#    ax1.add_patch(c)

#ax2.imshow(edgecutoutleft, interpolation = 'none', vmin = mintemp, vmax = maxtemp)
#ax2.axis('off')
#
#ax3.imshow(edgecutoutright, interpolation = 'none', vmin = mintemp, vmax = maxtemp)
#ax3.axis('off')
#
#ax4.imshow(scaledleft, interpolation = 'none', vmin = mintemp, vmax = maxtemp)
#ax4.axis('off')
#
#ax5.imshow(rimage, interpolation = 'none', vmin = mintemp, vmax = maxtemp)
#ax5.axis('off')
#
#lnumbers = np.histogram(edgecutoutleft,bins=42,range=(15,36))[0]
#temps=np.arange(15.05,36.05,0.5)
#ax4.axvline(x=lave, color='b', linestyle='dashed', linewidth=2)
#ax4.axvline(x=(lave+(2*rstd)), color='r', linestyle='dashed', linewidth=2)
#ax4.axvline(x=(lave-(2*rstd)), color='r', linestyle='dashed', linewidth=2)
#ax4.text(lave, 2000, 'ave'+str(lave), fontsize=10, color='b')
#leftfoot = ax4.bar(temps, lnumbers, width=0.5, color='r')
#
#rnumbers = np.histogram(edgecutoutright,bins=42,range=(15,36))[0]
#temps=np.arange(15.05,36.05,0.5)
#ax5.axvline(x=rave, color='b', linestyle='dashed', linewidth=2)
#ax5.axvline(x=(rave-(2*rstd)), color='r', linestyle='dashed', linewidth=2)
#ax5.axvline(x=(rave+(2*rstd)), color='r', linestyle='dashed', linewidth=2)
#ax5.text(rave, 2000, 'ave'+str(rave), fontsize=10, color='b')
#rightfoot = ax5.bar(temps, rnumbers, width=0.5, color = 'y')


#ax4.imshow(subtraction, interpolation = 'none', vmin = -3, vmax = 3)
#ax4.axis('off')
#ax4.set_title('right', fontsize=20, )
#
#ax5.imshow(addoutlines, cmap=plt.cm.gray, interpolation = 'none')
#ax5.axis('off')
#ax5.set_title('left', fontsize=20, )
#
#ax6.imshow(outlineleft1, cmap=plt.cm.gray, interpolation = 'none')
#ax6.axis('off')
#ax6.set_title('right', fontsize=20, )
#plt.tight_layout()
plt.show()


#print edgecutoutright[87,108]

#print rimage[87,108]
#
#fig, (ax1,ax2) = plt.subplots(1,2)
#ax1.imshow(limage, interpolation = 'none', vmin = 17, vmax = 36)
#ax1.axis('off')
#ax1.set_title('left', fontsize=20)
#ax2.imshow(rimage, interpolation = 'none')
#ax2.axis('off')
#ax2.set_title('right', fontsize=20)
#plt.show()


#print hottest

#for x in range (0,leftdim[0]):
#    for y in range (0,leftdim[1]):  
#        
#
#for blobs, color, title in sequence:
#    fig, ax = plt.subplots(1, 1)
#    ax.set_title(title)
#    ax.imshow(image, interpolation='nearest')
#    for blob in blobs:
#        y, x, r = blob
#        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
#        ax.add_patch(c)
#
#plt.show()

#3d plot
#nx = edgecutoutleft.shape[0]
#ny = edgecutoutleft.shape[1]
#y = range(nx)
#x = range(ny)
#hf = plt.figure()
#ha = hf.add_subplot(111, projection='3d')
#X, Y = np.meshgrid(x, y)
#ha.plot_surface(X, Y, edgecutoutleft,cmap=cm.jet,linewidth=0.2, vmin = 17, vmax = 36)
#ha.set_zlim3d(20,36)
#plt.show()

