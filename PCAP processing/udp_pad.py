#This code Pads UDP headers with zeros to match TCP header length of 20
from scapy.all import *
from scapy.utils import PcapWriter
import glob
from scapy.compat import raw
from scapy.layers.inet import IP, UDP
from scapy.packet import Padding

path = ""
files = glob.glob(path + "/*.pcap")

for f in files:
    p = rdpcap(f)
    new = PcapWriter(str(f))
    for i in p:
        if UDP in i:
            layer_after = i[UDP].payload.copy()

            pad = Padding()
            pad.load = '\x00' * 12

            layer_before = i.copy()
            layer_before[UDP].remove_payload()
            j = layer_before / raw(pad) / layer_after

            new.write(j)
        else:
            new.write(i)
    new.close()
print("ALL DONE")
