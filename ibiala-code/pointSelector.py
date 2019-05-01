import numpy as np
import matplotlib.pyplot as plt

project_data = np.load('../data/project_palace.npz')

try:
    coords = np.load('../data/coords_palace.npz')
    x1, y1 = list(coords['x1']), list(coords['y1'])
except:
    print('No previous coords saved...')
    x1, y1 = [], []

im1 = project_data['im1']

f, ax1 = plt.subplots(1, 1)
ax1.imshow(im1)
ax1.set_title('Select a point in this image')
ax1.set_axis_off()

for i in range(len(x1)):
    ax1.plot(x1[i], y1[i], '*', MarkerSize=6, linewidth=2)

while True:
    plt.sca(ax1)
    click_input = plt.ginput(1, mouse_stop=2)
    if not click_input:
        break
    x, y = click_input[0]

    xc = int(x)
    yc = int(y)
    ax1.plot(x, y, '*', MarkerSize=6, linewidth=2)
    print(x, y, xc, yc)
    x1.append(xc)
    y1.append(yc)

np.savez('../data/coords_palace.npz', x1=x1, y1=y1)
