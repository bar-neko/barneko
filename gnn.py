import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

class WerewolfGNN(nn.Module):
    def __init__(self, hidden_dim=64):
        super(WerewolfGNN, self).__init__()
        
        # Node feature dimensions
        self.num_agents = 5
        self.num_roles = 4  # villager, seer, werewolf, lunatic
        
        # GNN layers
        self.node_encoder = nn.Linear(self.num_roles, hidden_dim)
        self.edge_encoder = nn.Linear(1, hidden_dim)
        
        self.conv1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.conv2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.role_predictor = nn.Linear(hidden_dim, self.num_roles)
        
    def message_passing(self, x, edge_index, edge_attr):
        # Aggregate messages from neighbors
        row, col = edge_index
        edge_features = self.edge_encoder(edge_attr)
        
        # Combine node features with edge features
        message = torch.cat([x[row], edge_features], dim=1)
        message = self.conv1(message)
        
        # Aggregate messages for each node
        aggr_messages = torch.zeros(self.num_agents, message.size(1)).to(x.device)
        aggr_messages.index_add_(0, col, message)
        
        return aggr_messages
        
    def forward(self, x, edge_index, edge_attr):
        # Initial node embeddings
        x1 = self.node_encoder(x)
        print("x1.shape", x1.shape)
        
        # Two rounds of message passing
        m1 = self.message_passing(x1, edge_index, edge_attr)
        x2 = x1 + m1  # Skip connection
        print("x2.shape", x2.shape)
        
        m2 = self.message_passing(x2, edge_index, edge_attr)
        x3 = x2 + m2  # Skip connection
        print("x3.shape", x3.shape)
        
        # Predict role probabilities
        role_logits = self.role_predictor(x3)
        role_probs = F.softmax(role_logits, dim=-1)
        
        return role_probs

def create_edge_index():
    # Create fully connected graph (each agent connected to every other agent)
    edges = []
    for i in range(5):
        for j in range(5):
            if i != j:
                edges.append([i, j])
    return torch.tensor(edges).t()

if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = WerewolfGNN().to(device)
    
    # Create dummy data
    x = torch.rand(5, 4).to(device)  # Initial role probabilities for 5 agents
    edge_index = create_edge_index().to(device)
    print("x.shape", x.shape)
    print("edge_index.shape", edge_index.shape)
    edge_attr = torch.ones(20, 1).to(device)  # 20 edges (5*4) with single feature
    
    # Forward pass
    role_probs = model(x, edge_index, edge_attr)
    print("Role probabilities shape:", role_probs.shape)
    print("Role probabilities sum:", role_probs.sum(dim=1))  # Should sum to 1 for each agent
    
    # Visualize the computation graph, showing only parameter nodes
    dot = make_dot(role_probs, params=dict(model.named_parameters()), show_attrs=False, show_saved=False)
    dot.render('werewolf_gnn', format='png')
