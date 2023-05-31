# This code will replace all IPv4 and IPv6 addresses with zeros
from scapy.all import *
from scapy.layers.inet6 import IPv6
from scapy.utils import PcapWriter
import glob
from scapy.layers.inet import IP, UDP

path = ""
files = glob.glob(path + "/*.pcap")

for f in files:
    p = rdpcap(f)
    new = PcapWriter(str(f))
    for i in p:
        if IPv6 in i:
            i[IPv6].dst = '0000::0:0'
            i[IPv6].src = '0000::0000:0000:0000:0000'
            new.write(i)
        if IP in i:
            i[IP].dst = '0.0.0.0'
            i[IP].src = '0.0.0.0'
            new.write(i)
    new.close()
print("ALL DONE")
