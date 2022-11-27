# MAST-GNN

The project is a Pytorch implementation of MAST-GCN proposed in the paper “XXXX”.

The paper can be visited at XXXXXX

## Requirement

1. cuda = 11.0
2. cudnn = 8.0
3. python = 3.7
4. pytorch = 1.7.0
5. numpy = 1.21.4
6. sklearn = 0.0
7. scikit-learn = 0.23.2
8. pandas = 1.1.3
9. scipy = 1.5.4
10. tqdm = 4.51.0
11. pillow = 8.0.1

## Data preprocessing

We implement and execute MAST-GCN on real world sector datasets. In our sector data, each sector contains 43 features and a label, of which 17 features are used. There are 126 sectors in total, and the data of each sector is saved in a csv file.

the following files should be prepared:

* sector_data/SectorFactor1002: This file represents the data of each sector
* ./data/adj_mx_geo_126.csv: This file is used to build the adjacency matrix of each sector

During model training, the data needs to be processed in npy format_ Make sure the data path is correct in the extract_data.py file, and run the following command:

* **python extract_data.py**
## Run the Project

### Run the command below to train the model

* **python main.py**

After running the training program, it would generate the following files.

* ./checkpoints/log/run.log: This file records information during model training
* ./checkpoints/log/bestmodel.pth: This file contains the best trained model
* ./checkpoints/log/tcnpredict.npy: This file contains the prediction results

### Run the command below to test the model

* **python main_test.py**

After running the testing program, it would generate the following files.

* ./checkpoints/log_test/run.log: This file records information during model testing
* ./checkpoints/log_test/tcnpredict.npy: This file contains the prediction results








