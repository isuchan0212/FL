import pandas as pd
import torch
import torch_geometric
from torch_geometric.utils import to_undirected
import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

class EllipticData():

    N_CLASSES = 3

    def __init__(self, args):
        self.args = args

    def get_data(self):
        df_edge = pd.read_csv('/Users/Suchan/Desktop/kaist/project/0611/data/elliptic_txs_edgelist.csv')
        df_class = pd.read_csv('/Users/Suchan/Desktop/kaist/project/0611/data/elliptic_txs_classes.csv')
        df_features = pd.read_csv('/Users/Suchan/Desktop/kaist/project/0611/data/elliptic_txs_features.csv', header=None)
        df_features.columns = ['id', 'time step'] + [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in range(72)]
        all_nodes = list(set(df_edge['txId1']).union(set(df_edge['txId2'])).union(set(df_class['txId'])).union(set(df_features['id'])))
        nodes_df = pd.DataFrame(all_nodes,columns=['id']).reset_index()
        df_edge = df_edge.join(nodes_df.rename(columns={'id':'txId1'}).set_index('txId1'),on='txId1',how='inner') \
            .join(nodes_df.rename(columns={'id':'txId2'}).set_index('txId2'),on='txId2',how='inner',rsuffix='2') \
            .drop(columns=['txId1','txId2']) \
            .rename(columns={'index':'txId1','index2':'txId2'})
        df_class = df_class.join(nodes_df.rename(columns={'id':'txId'}).set_index('txId'),on='txId',how='inner') \
                .drop(columns=['txId']).rename(columns={'index':'txId'})[['txId','class']]
        df_features = df_features.join(nodes_df.set_index('id'),on='id',how='inner') \
                .drop(columns=['id']).rename(columns={'index':'id'})
        df_features = df_features [ ['id']+list(df_features.drop(columns=['id']).columns) ]
        df_edge_time = df_edge.join(df_features[['id','time step']].rename(columns={'id':'txId1'}).set_index('txId1'),on='txId1',how='left',rsuffix='1') \
        .join(df_features[['id','time step']].rename(columns={'id':'txId2'}).set_index('txId2'),on='txId2',how='left',rsuffix='2')
        df_edge_time['is_time_same'] = df_edge_time['time step'] == df_edge_time['time step2']
        df_edge_time_fin = df_edge_time[['txId1','txId2','time step']].rename(columns={'txId1':'source','txId2':'target','time step':'time'})
        node_label = df_class.rename(columns={'txId':'nid','class':'label'})[['nid','label']].sort_values(by='nid').merge(df_features[['id','time step']].rename(columns={'id':'nid','time step':'time'}),on='nid',how='left')
        node_label['label'] =  node_label['label'].apply(lambda x: '3'  if x =='unknown' else x).astype(int)-1
        merged_nodes_df = node_label.merge(df_features.rename(columns={'id':'nid','time step':'time'}).drop(columns=['time']),on='nid',how='left')

        subgraphs = []
        for i in range(49):
                nodes_df_tmp=merged_nodes_df[merged_nodes_df['time']==i+1].reset_index()
                nodes_df_tmp['index']=nodes_df_tmp.index
                df_edge_tmp = df_edge_time_fin.join(nodes_df_tmp.rename(columns={'nid':'source'})[['source','index']].set_index('source'),on='source',how='inner')\
                        .join(nodes_df_tmp.rename(columns={'nid':'target'})[['target','index']].set_index('target'),on='target',how='inner',rsuffix='2') \
                        .drop(columns=['source','target']) \
                        .rename(columns={'index':'source','index2':'target'})
                
                X = nodes_df_tmp.sort_values(by='index').drop(columns=['index','nid','label'])
                edge_index = torch.tensor(np.array(df_edge_tmp[['source','target']]).T, dtype=torch.long)
                edge_index = to_undirected(edge_index)
                
                y=nodes_df_tmp['label']

                train_mask = y == 3
                test_mask = y == 3
                val_mask = y == 3

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

                train_indices = y_train.index
                test_indices = y_test.index
                val_indices = y_val.index
                unseen_indices = y[y==2].index

                train_mask[train_indices] = True
                train_mask[unseen_indices] = False
                test_mask[test_indices] = True
                test_mask[unseen_indices] = False
                val_mask[val_indices] = True
                val_mask[unseen_indices] = False

                train_mask = torch.tensor(np.array(train_mask))
                test_mask = torch.tensor(np.array(test_mask))
                val_mask = torch.tensor(np.array(val_mask))

                X=torch.tensor(np.array(X), dtype=torch.float)
                y=torch.tensor(np.array(y), dtype=torch.long)

                data = Data(x=X, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
                subgraphs.append(data)

        return subgraphs
    
        # df_edge = pd.read_csv('/Users/Suchan/Desktop/kaist/project/0611/data/elliptic_txs_edgelist.csv')
        # df_class = pd.read_csv('/Users/Suchan/Desktop/kaist/project/0611/data/elliptic_txs_classes.csv')
        # df_features = pd.read_csv('/Users/Suchan/Desktop/kaist/project/0611/data/elliptic_txs_features.csv', header=None)
        # df_features.columns = ['id', 'time step'] + [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in range(72)]

        # all_nodes = list(set(df_edge['txId1']).union(set(df_edge['txId2'])).union(set(df_class['txId'])).union(set(df_features['id'])))
        # nodes_df = pd.DataFrame(all_nodes,columns=['id']).reset_index()

        # Edge = df_edge.join(nodes_df.rename(columns={'id':'txId1'}).set_index('txId1'),on='txId1',how='inner') \
        # .join(nodes_df.rename(columns={'id':'txId2'}).set_index('txId2'),on='txId2',how='inner',rsuffix='2') \
        # .drop(columns=['txId1','txId2']) \
        # .rename(columns={'index':'txId1','index2':'txId2'})
            
        # Class = df_class.join(nodes_df.rename(columns={'id':'txId'}).set_index('txId'),on='txId',how='inner') \
        # .drop(columns=['txId']).rename(columns={'index':'txId'})[['txId','class']]

        # df_features = df_features.join(nodes_df.set_index('id'),on='id',how='inner') \
        # .drop(columns=['id']).rename(columns={'index':'id'})
        # Features = df_features [ ['id']+list(df_features.drop(columns=['id']).columns) ]

        # df_edge_time = df_edge.join(df_features[['id','time step']].rename(columns={'id':'txId1'}).set_index('txId1'),on='txId1',how='left',rsuffix='1') \
        # .join(df_features[['id','time step']].rename(columns={'id':'txId2'}).set_index('txId2'),on='txId2',how='left',rsuffix='2')
        # df_edge_time['is_time_same'] = df_edge_time['time step'] == df_edge_time['time step2']
        # Time = df_edge_time[['txId1','txId2','time step']].rename(columns={'txId1':'source','txId2':'target','time step':'time'})

        # node_label = Class.rename(columns={'txId':'nid','class':'label'})[['nid','label']].sort_values(by='nid').merge(Features[['id','time step']].rename(columns={'id':'nid','time step':'time'}),on='nid',how='left')
        # node_label['label'] =  node_label['label'].apply(lambda x: '3'  if x =='unknown' else x).astype(int)-1

        # merged_nodes_df = node_label.merge(Features.rename(columns={'id':'nid','time step':'time'}).drop(columns=['time']),on='nid',how='left')

        # merged_nodes_df['index']=merged_nodes_df.index    

        # df_edge_tmp = Time.join(merged_nodes_df.rename(columns={'nid':'source'})[['source','index']].set_index('source'),on='source',how='inner')\
        #     .join(merged_nodes_df.rename(columns={'nid':'target'})[['target','index']].set_index('target'),on='target',how='inner',rsuffix='2') \
        #     .drop(columns=['source','target']) \
        #     .rename(columns={'index':'source','index2':'target'})

        # edge_index = torch.tensor(np.array(df_edge_tmp[['source','target']]).T, dtype=torch.long)
        # edge_index = to_undirected(edge_index)

        # df_final = merged_nodes_df.sort_values(by='index')
        # final_graph = df_final.drop(columns=['index'])

        # df_seperate = [final_graph[final_graph['time']==i] for i in range(1, 50)]

        # dataset = []
        # for df in df_seperate:
        #     # X = df[df['label']!=2].drop(columns=['nid','label'])
        #     X = df[df['label']!=2].drop(columns=['nid','label'])
        #     y = df[df['label']!=2]['label']
        #     train_mask = y == 3
        #     test_mask = y == 3
        #     val_mask = y == 3

        #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        #     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

        #     train_indices = y_train.index
        #     test_indices = y_test.index
        #     val_indices = y_val.index

        #     train_mask[train_indices] = True
        #     test_mask[test_indices] = True
        #     val_mask[val_indices] = True

        #     train_mask = torch.tensor(np.array(train_mask))
        #     test_mask = torch.tensor(np.array(test_mask))
        #     val_mask = torch.tensor(np.array(val_mask))

        #     X=torch.tensor(np.array(X), dtype=torch.float)
        #     y=torch.tensor(np.array(y), dtype=torch.long)

        #     data = Data(x=X, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        #     dataset.append(data)
        # return dataset
        

        




        
        