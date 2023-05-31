#This code removes the ethernet header of PCAP files
from scapy.all import *
from scapy.utils import PcapWriter
import glob
from scapy.layers.l2 import Ether

path = ""
files = glob.glob(path + "/*.pcap")

for f in files:
    p = rdpcap(f)
    new = PcapWriter(str(f))
    for i in p:
        if Ether in i:
            j = i[Ether].payload
            new.write(j)
        else:
            new.write(i)
    new.close()
print("ALL DONE")
