from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import distance
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import blob_log

from skimage.segmentation import felzenszwalb, mark_boundaries
import numpy as np
from sklearn.cluster import KMeans

white = (255, 255, 255)


def my_distance(a, b):
    return distance.euclidean(a, b)


img = io.imread("3.jpg")
image_gray = rgb2gray(img)
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
#
ax.set_title('Image')
ax.imshow(img)
ax = fig.add_subplot(1, 2, 2)
#
ax.set_title('GrayImage')
ax.imshow(image_gray)

# SLIC
segments = felzenszwalb(image_gray, scale=200,  sigma=0.5, min_size=100)
segments_ids = np.unique(segments)
print(segments_ids)

# centers
centers = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])
print(centers)
vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
plt.imshow(mark_boundaries(img, segments))
plt.scatter(centers[:, 1], centers[:, 0], c='y')

for i in range(bneighbors.shape[1]):
    y0, x0 = centers[bneighbors[0, i]]
    y1, x1 = centers[bneighbors[1, i]]

    l = Line2D([x0, x1], [y0, y1], alpha=0.5)
    ax.add_line(l)

#
def middle(a, b):
    color = []
    for i, j in zip(a, b):
        color.append((i + j) // 2)
    return color


dict_seg = {}
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        seg = segments[i, j]
        if seg not in dict_seg.keys():
            dict_seg[seg] = 1
            continue
        dict_seg[seg] += 1
max_l = max(dict_seg, key=dict_seg.get)
print(max_l)
blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.05)
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
#
ax.set_title('Laplacian of Gaussian')
ax.imshow(img)
count = 0
for blob in blobs_log:
    y, x, r = blob
    # print(max_l, end=" ")
    # print(dic_seg_claster[segments[int(y), int(x)]])
    if segments[int(y), int(x)] == max_l:
        c = plt.Circle((x, y), r, color='white', linewidth=2, fill=False)
        count += 1
        ax.add_patch(c)
print(count)
ax.set_axis_off()
ax = fig.add_subplot(1, 2, 2)
#
ax.set_title('Оригинал')
ax.imshow(img)
ax.set_axis_off()
plt.tight_layout()
plt.show()
