import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def draw_graph(graph,label):
    nx.draw_networkx(graph, pos=nx.spring_layout(graph,seed=42),
                     with_labels=True,node_color = label,cmap = 'Set3')

def accuracy(pred,label):
    pred = np.asarray(pred.argmax(dim=1))
    label = np.asarray(label)
    return np.sum(pred == label)/len(label)

def visual_embed_space(pred , embed, label, epoch=None, loss=None):
    plt.figure(figsize=(8,8))
    embed = embed.detach().cpu().numpy()
    plt.scatter(embed[:,0], embed[:,1], s = 140, c = label)
    if epoch is not None and loss is not None:
         plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}, Accuracy: {accuracy(pred,label)}', fontsize=16)
    plt.show()