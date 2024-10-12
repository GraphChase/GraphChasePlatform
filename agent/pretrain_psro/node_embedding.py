from networkx import Graph
import networkx as nx
import numpy as np
import copy
import torch
from agent.pretrain_psro.line import LINE
import pickle
import os
from graph.grid_graph import GridGraph


class NodeEmbedding(object):
    def __init__(self, graph: GridGraph, args):
        """
        Parameters:
            graph: Custom Graph Class
        """
        self.Graph = graph
        self.args = args
        self.graph:nx.graph = graph.graph
        self.total_node_number = graph.num_nodes
        self.exit_node = graph.exits

        self.node_information = np.zeros((self.total_node_number, len(self.exit_node)))
        self.information_proximity_matrix = np.zeros((self.total_node_number, self.total_node_number))

        self.save_path = os.path.join(args.save_path, "node_embedding_file")

        self.get_env_changestate_legal_action()

    def _assign_node_information(self,):
        
        # process node information
        if self.args.load_node_information_file:
            print("load_node_information......")
            self.node_information = torch.load(self.args.load_node_information_file)
        else:
            short_length = [[0 for _ in range(len(self.exit_node))] for _ in range(self.total_node_number)]

            for i in range(self.total_node_number):
                for j in range(len(self.exit_node)):
                    short_length[i][j] = nx.shortest_path_length(self.graph, source=i + 1, target=self.exit_node[j]) \
                        if nx.has_path(self.graph, source=i + 1, target=self.exit_node[j]) else 1000

            for i in range(self.total_node_number):
                if self.args.node_information_type == "min":
                    short_length_list = []
                    for idx, j in enumerate(self.change_state[i]):
                        short_length_list.append(min(short_length[j - 1]))
                else:
                    short_length_list = short_length[i]

                # normalize
                if self.args.node_information_normalize:
                    total_sum = sum(short_length_list)
                    self.node_information[i] = [p / total_sum for p in short_length_list]
                else:
                    self.node_information[i] = copy.deepcopy(short_length_list)                   
            self._save_node_information_file()


        # process proximity matrix
        if self.args.load_information_proximity_matrix:
            print("load information proximity matrix......")
            self.information_proximity_matrix = np.load(self.args.load_information_proximity_matrix)
        else:
            print("compute information proximity matrix......")
            for i in range(self.total_node_number):
                for j in range(self.total_node_number):
                    sim = self._compute_similarity(self.node_information[i], self.node_information[j])
                    self.information_proximity_matrix[i, j] = sim              

                total_sum = np.sum(self.information_proximity_matrix[i])
                if total_sum != 0:
                    normalized_information = self.information_proximity_matrix[i] / total_sum
                    for j in range(self.total_node_number):
                        self.information_proximity_matrix[i, j] = normalized_information[j] * self.total_node_number
            self._save_information_proximity_matrix()

        return self.information_proximity_matrix
    
    def get_env_changestate_legal_action(self,):
        """
        Accroding to the BaseGraph, get self.change_state and self.legal_action
        """

        self.change_state = [[i, i, i, i, i] for i in range(1, self.total_node_number + 1)]
        self.legal_action = [[0] for _ in range(1, self.total_node_number + 1)]        
        for node in range(1, self.total_node_number+1):  
            state = [node] * 5

            neighbors = self.Graph.adjlist.get(node, [])

            for neighbor in neighbors:
                if neighbor == node:
                    state[0] = neighbor
                elif neighbor == node - self.Graph.column:
                    state[1] = neighbor
                elif neighbor == node + self.Graph.column:
                    state[2] = neighbor
                elif neighbor == node - 1:
                    state[3] = neighbor
                elif neighbor == node + 1:
                    state[4] = neighbor
            
            self.change_state[node-1] = state

            for j, a in enumerate(self.change_state[node-1]):
                if a != node:
                    self.legal_action[node-1].append(j)      

    def train_embedding(self,) -> dict: 
        """
        1. Get information proximity matrix
        2. Use LINE method to train node_embeddings
        3. After traning got embeddings
        4. Save embeddings
        5. return embeddings: dict: {node: embedding, }
        """

        if self.args.load_node_embedding_model:
            print("load node embedding model.......")
            name = os.path.join(self.args.load_node_embedding_model, f"graphid_{self.args.graph_id}_node_embeddings.pkl")
            with open(name, 'rb') as f:
                self.n2v = pickle.load(f)
            return self.n2v
        
        if self.args.node_embedding_method == "line":
            self.information_proximity_matrix = [[0 for _ in range(self.total_node_number)] for _ in range(self.total_node_number)]
            print("train model embedding model......")
        else:
            if self.args.node_information_type == "all":
                self.node_information = [[0 for _ in range(len(self.exit_node))] for _ in range(self.total_node_number)]
            else:
                self.node_information = [[0 for _ in range(len(self.change_state[0]))] for _ in range(self.total_node_number)]
            print("assign node information......")
            self.information_proximity_matrix = self._assign_node_information()
            print("train model embedding model......")      

        model = LINE(self.graph, self.information_proximity_matrix, self.args.emb_size, self.args.line_batch_size, self.args.line_epochs, order=self.args.line_order)
        self.n2v = model.train() # dict: {node: embedding, }
        self._save_embeddings(self.n2v)
        
        return self.n2v  

    def load_embedding(self, path):
        with open(path, 'rb') as f:
            node_embs = pickle.load(f)

        return node_embs
    
    def _compute_similarity(self, vector_a: np.ndarray, vector_b: np.ndarray):
        if self.args.information_similarity == "cosine":
            num = np.dot(vector_a, vector_b)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            sim = num / denom
        else:
            sim = np.dot(vector_a, vector_b)
        return sim    
    
    def _save_node_information_file(self, ):
        os.makedirs(self.save_path, exist_ok=True)
        file_name = f"graphid_{self.args.graph_id}_node_information.npy"
        path = os.path.join(self.save_path, file_name)
        np.save(path, self.node_information)

    def _save_information_proximity_matrix(self, ):
        os.makedirs(self.save_path, exist_ok=True)
        file_name = f"graphid_{self.args.graph_id}_information_proximity_matrix.npy"
        path = os.path.join(self.save_path, file_name)
        np.save(path, self.information_proximity_matrix)

    def _save_embeddings(self, embeddings: dict):
        os.makedirs(self.save_path, exist_ok=True)
        file_name = f"graphid_{self.args.graph_id}_node_embeddings.pkl"
        path = os.path.join(self.save_path, file_name)        
        with open(path, 'wb') as f:
            pickle.dump(embeddings, f)
        # if self.args.save_node_embedding_model is not None:
        #     path = os.path.join(self.save_node_embedding_model, f"node_emb_model")
        #     if not os.path.exists(path):
        #         os.makedirs(path)
        #     file_name = os.path.join(path, 'embeddings.pkl')