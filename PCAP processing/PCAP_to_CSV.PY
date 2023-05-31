# This code takes every PCAP file, converts it to raw packet (byte value)
# then converts the raw byte to decimal and normalises it to fall between 0 - 1
# it pads every packet to have a max length of 1500 and finally converts it to CSV format

from scapy.all import *
import glob
import numpy as np
import pandas as pd
from scapy.compat import raw

path =  ''
files = glob.glob(path + "/*.pcap")

for f in files:
    z = []
    max = 1500
    p = rdpcap(f)
    new = (str(f))
    for i in p:
        t = np.frombuffer(raw(i), dtype=np.uint8)[0: max] / 255
        if len(t) <= max:
            pad_width = max - len(t)
            t = np.pad(t, pad_width=(0, pad_width), constant_values=0)
            z.append(t)
            
    df = pd.DataFrame(z)
    df.to_csv(new + '.csv', index=False, header=False)
    z.clear()

print("All done")
