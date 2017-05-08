# call modules
from SimpleCV import *
from pybrain.structure import FeedForwardNetwork, LinearLayer, FullConnection
import pylab as plt

"""This is computer vision program mixed with neuronal network
    where try emulate neuronal plasticity of hidden layer"""
cam = Camera()  # call webcam
threshold = 5.0  # if mean exceeds this amount do something
plasticity = 100  # basic level neuron of hidden layer in ANN
multiplicator = 1  # basic factor to adapt the ANN
n = FeedForwardNetwork()  # call function of network type


# function plastic ann
def annplastic(adaptation):
    """Function who generate the plastic ann, received
        variable adaptation for give plasticity to
        network and take variable like argument"""
    # layers network
    inLayer = LinearLayer(100)
    hiddenLayer = LinearLayer(adaptation)
    outLayer = LinearLayer(10)
    # mount layers
    n.addInputModule(inLayer)
    n.addModule(hiddenLayer)
    n.addOutputModule(outLayer)
    # stabilise type confections
    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)
    # conecct the layers
    n.addConnection(in_to_hidden)
    n.addConnection(hidden_to_out)
    # start the module
    n.sortModules()


while True:
    """This loop try to get a stress threshold based on number of
     movements of visual elements then give a value who give a variable
     to recaulculate the number of hidden layer. If not detect activity or
     become low level input the ANN become to basic state."""
    previous = cam.getImage()  # grab a frame
    time.sleep(0.5)  # wait for half a second
    current = cam.getImage()  # grab another frame
    diff = current - previous
    matrix = diff.getNumpy()
    mean = matrix.mean()
    diff.show()
    # draw real time histogram of visual output
    peaks = diff.huePeaks()
    hist = diff.hueHistogram()
    plt.plot(hist)
    plt.pause(0.0001)
    plt.draw()
    """This block take the time diference and computate
        if is adaptable the netwotk ot not."""
    if mean >= threshold:
        # adaptation factor
        multiplicator += 1
        adaptation = plasticity * multiplicator
        print "Develop plasticity"  # simple advice
        print adaptation
        annplastic(adaptation)
    else:
        # not adapt
        # layers network
        adaptation = plasticity
        print "Not develop plasticity."  # it advice
        print adaptation
        annplastic(adaptation)
