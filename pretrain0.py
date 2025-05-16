from model.featurisation import smiles2graph
from model.CL_model_vas_info import GNNModelWithNewLoss
import pandas as pd
import torch
from torch_geometric.data import DataLoader

def read_vsa_data(vsa_file):
    df = pd.read_csv(vsa_file)

    def parse_vsa(s):
        try:
            return list(map(float, s.strip('[]').split()))
        except:
            return []

    smr_arrays = df["SMR_VSA"].apply(parse_vsa).tolist()          
    slogp_arrays = df["SlogP_VSA"].apply(parse_vsa).tolist()     
    peoe_arrays = df["PEOE_VSA"].apply(parse_vsa).tolist()       

    properties = list(zip(smr_arrays, slogp_arrays, peoe_arrays))
    
    return df["SMILES"].tolist(), properties

def main():
    x_smiles, properties = read_vsa_data("./data/vsa_zinc.csv")
    data_list = smiles2graph(
        x_smiles, y=None, cluster=None, properties=properties, test=False
    )

    devices = ["cuda" if torch.cuda.is_available() else "cpu"]

    model1 = GNNModelWithNewLoss(
            num_node_features=data_list[0].x.shape[1],
            num_edge_features=data_list[0].edge_attr.shape[1],
            num_global_features=data_list[0].global_features.shape[0],
            hidden_dim=512,
            dropout_rate=0.1,
            property_index=0 ,
            save_path= 'premodels/0' 
        ).to(devices[0])

    model1.train_model(
        data_list,
    )
    model3 = GNNModelWithNewLoss(
            num_node_features=data_list[0].x.shape[1],
            num_edge_features=data_list[0].edge_attr.shape[1],
            num_global_features=data_list[0].global_features.shape[0],
            hidden_dim=512,
            dropout_rate=0.1,
            property_index=2,
            save_path= 'premodels/2' 
        ).to(devices[0])

    model3.train_model(
        data_list,
    )
    model2 = GNNModelWithNewLoss(
            num_node_features=data_list[0].x.shape[1],
            num_edge_features=data_list[0].edge_attr.shape[1],
            num_global_features=data_list[0].global_features.shape[0],
            hidden_dim=512,
            dropout_rate=0.1,
            property_index=1,
            save_path= 'premodels/1' 
        ).to(devices[0])

    model2.train_model(
        data_list,
    )

if __name__ == "__main__":
    main()