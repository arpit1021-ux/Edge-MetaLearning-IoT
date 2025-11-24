import torch
import torch.nn.functional as F

def euclidean(a,b):
    return ((a-b)**2).sum(dim=1)

def get_prototype(embeddings, labels):
    classes = labels.unique()
    prototypes = []
    for c in classes:
        prototypes.append(embeddings[labels==c].mean(0))
    return torch.stack(prototypes), classes

def proto_loss(model, sx, sy, qx, qy):
    s_emb = model(sx)
    q_emb = model(qx)
    proto, classes = get_prototype(s_emb, sy)
    dists = torch.cdist(q_emb, proto)
    return F.cross_entropy(-dists, qy)
