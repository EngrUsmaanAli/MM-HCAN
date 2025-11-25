import torch
import numpy as np
import torch.nn.functional as F

def build_hypergraph(features, k=10):

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    
    features_norm = F.normalize(features, p=2, dim=1)
    sim_matrix = torch.mm(features_norm, features_norm.t()))
    N = features.shape[0]
    _, indices = torch.topk(sim_matrix, k=k, dim=1)  
    H = torch.zeros(N, N, device=features.device)
    
    row_indices = torch.arange(N, device=features.device).unsqueeze(1).expand(-1, k)
    H[row_indices, indices] = 1.0
    
    D_v = torch.sum(H, dim=1) 
    D_e = torch.sum(H, dim=0) 
    
    inv_sqrt_Dv = torch.diag(torch.pow(D_v + 1e-8, -0.5))
    inv_De = torch.diag(torch.pow(D_e + 1e-8, -1.0)) 
    
    H_t = H.t()
    L = torch.eye(N, device=features.device) - inv_sqrt_Dv @ H @ inv_De @ H_t @ inv_sqrt_Dv
        
    return L.float(), H.float()
