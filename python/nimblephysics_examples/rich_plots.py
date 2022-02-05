import nimblephysics as nimble
from nimblephysics import NimbleGUI
import random

world = nimble.simulation.World()

gui = NimbleGUI(world)
gui.serve(8080)

# To create a rich plot
gui.nativeAPI().createRichPlot(
    "some_unique_key",  # key
    # initial position from top-left (plot can then be dragged with the mouse)
    [60, 90],
    [300, 300],  # size
    -5.0,  # min-x value
    20.0,  # max-x value
    -5.0,  # min-y value
    20.0,  # max-y value
    "Plot Title",  # title
    "X Axis",  # x axis label
    "Y Axis")  # y axis label

# To compare several data streams on your rich plot
gui.nativeAPI().setRichPlotData(
    "some_unique_key",  # key
    "Series 1",  # series name, displays in Legend, use same name to overwrite data
    "blue",  # color
    "line",  # plot-type, currently this only supports "line", someday we may support "scatter", "dotted", etc
    [i - 5.0 for i in range(25)],  # data point x values
    [i - 5.0 + random.random() for i in range(25)])  # data point y values
gui.nativeAPI().setRichPlotData(
    "some_unique_key",  # key
    "Series 2",
    "red",
    "line",
    [i - 5.0 for i in range(25)],  # data point x values
    [i*0.8 - 5.0 + random.random()*3 for i in range(25)])  # data point y values

# To change the bounds on displayed data
gui.nativeAPI().setRichPlotBounds(
    "some_unique_key",  # key
    -5.0,  # min-x value
    20.0,  # max-x value
    -5.0,  # min-y value
    25.0  # max-y value
)


gui.blockWhileServing()
