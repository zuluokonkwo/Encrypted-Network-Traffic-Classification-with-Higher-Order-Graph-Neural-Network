# Encrypted Network Traffic Classification with Higher Order Graph Neural Network
Encryption protects internet users’ data security and privacy but makes network traffic classification a much harder problem. Network traffic classification is essential for identifying and predicting user behaviour which is important for the overall task of network management. Deep learning methods used to tackle this problem have produced promising results. However, the conditions on which these experiments are carried out raise questions about their effectiveness when applied in the real world. We tackle this problem by modelling network traffic as graphs and applying deep learning for classification. We design a graph classifier based on higher order graph neural network with the aim of optimum generalisation. To demonstrate the robustness of our model, we cross validate it on the ISCXVPN and USTC-TFC datasets with varying input specifications. We use our model to demonstrate the impact of network data truncation on traffic classification and define benchmarks for input specifications.

# Approach
1. Strip Ethernet header off.
2. Mask Source and Destination IP addresses.
3. Pad UDP header with zeros to match TCP header size.
4. Convert traffic data to raw byte format (binary or hex).
5. Pad traffic packet to match MTU size (1500).
6. Normalise evry byte value to fall within the range of 0-1 by converting every byte to decimal and dividing by 255.

# Graph Generation
To make graph generation easy, we create four files.
1. Node attributes file: A one to one mapping of packets (nodes) and their raw byte info (attributes).
2.  Edge file: A one to one mapping of source to destination nodes.
3.  Graph to Label file: A one to one mapping of PCAP sessions (graphs) to ther class or application labels.
4.  Node to Graph file: A one to one mapping of packets (nodes) to their corresponding  sessions(graphs).![pcap to graph](https://github.com/zuluokonkwo/Encrypted-Network-Traffic-Classification-with-Higher-Order-Graph-Neural-Network/assets/106361071/a05aac98-2101-40d1-9534-50bdb4735bb1)

# Dataset
![image](https://github.com/zuluokonkwo/Encrypted-Network-Traffic-Classification-with-Higher-Order-Graph-Neural-Network/assets/106361071/7c1d8476-4baa-4b68-93fb-5518d7d9a3bc)

![image](https://github.com/zuluokonkwo/Encrypted-Network-Traffic-Classification-with-Higher-Order-Graph-Neural-Network/assets/106361071/8e31e0e7-a0d7-4ae4-87b8-89838b41b9db)

# Results
VPN classification result
![image](https://github.com/zuluokonkwo/Encrypted-Network-Traffic-Classification-with-Higher-Order-Graph-Neural-Network/assets/106361071/f3ef29c3-cd68-4c5b-84d2-65d311c79bc5)

non VPN Classification Result
![image](https://github.com/zuluokonkwo/Encrypted-Network-Traffic-Classification-with-Higher-Order-Graph-Neural-Network/assets/106361071/0e0429db-abfc-4562-9596-b8228f634b4e)

Benign classification result
![image](https://github.com/zuluokonkwo/Encrypted-Network-Traffic-Classification-with-Higher-Order-Graph-Neural-Network/assets/106361071/afa790ad-73e3-493c-b155-444a3f453dab)

Malware classification result
![image](https://github.com/zuluokonkwo/Encrypted-Network-Traffic-Classification-with-Higher-Order-Graph-Neural-Network/assets/106361071/7e318cac-023b-4a2e-a117-1d7e4259b99a)

# License
If you make use of our methodology in your work, please cite our paper below, as well as UNB's related research papers:

```
@inproceedings{okonkwo2023encrypted,
  title={Encrypted Network Traffic Classification with Higher Order Graph Neural Network},
  author={Okonkwo, Zulu and Foo, Ernest and Hou, Zhe and Li, Qinyi and Jadidi, Zahra},
  booktitle={Australasian Conference on Information Security and Privacy},
  pages={630--650},
  year={2023},
  organization={Springer}
}
```
