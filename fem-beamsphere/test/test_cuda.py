import pycuda.driver as cuda
from six.moves import range

cuda.init()
print("%d device(s) found." % cuda.Device.count())

for ordinal in range(cuda.Device.count()):
    dev = cuda.Device(ordinal)
    print('Device #%d: %s' % (ordinal, dev.name()))
    print(' Compute Capability: %d.%d' % dev.compute_capability())
    print(' Total Memory: %s KB' % (dev.total_memory() // (1024)))

    atts = [(str(att), value) for att, value in list(dev.get_attributes().items())]
    atts.sort()

    for att, value in atts:
        print(' %s : %s' % (att, value))