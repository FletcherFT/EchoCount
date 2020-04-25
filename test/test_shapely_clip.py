from shapely.geometry import LineString
from shapely.geometry import box
import matplotlib.pyplot as plt
from shapely.ops import linemerge


def plot(ax, ob):
    try:
        x,y = ob.coords.xy
    except NotImplementedError:
        x,y = ob.exterior.xy
    plt.plot(x,y)

fig, ax = plt.subplots()
line = LineString([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])
plot(ax, line)
bb = box(0.5,1.1,2.9,1.5)
plot(ax, bb)
result = line.intersection(bb)
if len(result) > 1:
for r in result:
    plot(ax, r)

