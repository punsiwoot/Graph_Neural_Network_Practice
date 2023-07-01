from my_utils import accuracy,draw_graph,visual_embed_space
from model import simple_GCN
import torch
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx

#get data
dataset = KarateClub()
data = dataset[0]   
graph = to_networkx(data, to_undirected=True)
label = data.y
num_class = dataset.num_classes
num_fea = dataset.num_features

model = simple_GCN(
    in_channels= num_fea,
    hidden_channels=6,
    num_class=dataset.num_classes,
    activation_fcn="tanh"
)

loss_fcn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 400


for epoch in range(epochs):
    optimizer.zero_grad()
    out , embed = model(data.x, data.edge_index)
    loss = loss_fcn(out, label)
    loss.backward()
    optimizer.step()

    if (epoch  == 0):
        visual_embed_space(out,embed,data.y,epoch = epoch , loss = loss)
    if (epoch == epochs-1):
        visual_embed_space(out,embed,data.y,epoch = epoch , loss = loss)
        