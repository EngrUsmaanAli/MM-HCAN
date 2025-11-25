import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import os
import numpy as np

from models.temporal_model import TemporalFeatureExtractor
from models.spectral_model import SpectralFeatureExtractor
from models.hypergraph_builder import build_hypergraph
from models.hgnn_with_triplet import HGNNLayer, TripletLoss
from models.multihead_attention import MultiHeadAttentionFusion
from models.final_classifier import FinalClassifier
from utils.dataloader import get_dataloaders

def load_best_models(device):
    temporal_model = TemporalFeatureExtractor().to(device)
    temporal_model.eval()

    spectral_model = SpectralFeatureExtractor().to(device)
    spectral_model.eval()

    return temporal_model, spectral_model

def build_all_hypergraphs(temp_features, spec_features, concat_features, k=5):

    L_temp, _ = build_hypergraph(temp_features, k=k)
    L_spec, _ = build_hypergraph(spec_features, k=k)
    L_concat, _ = build_hypergraph(concat_features, k=k)

    return {
        'L_temp': L_temp,
        'L_spec': L_spec,
        'L_concat': L_concat
    }

def select_triplets(features, labels, device='cuda'):

    labels = labels.cpu().numpy()
    unique_labels = np.unique(labels)
    anchors, positives, negatives = [], [], []

    for label in unique_labels:
        idxs = np.where(labels == label)[0]
        if len(idxs) < 2: continue
        np.random.shuffle(idxs)
        anchor_idx = idxs[0]
        positive_idx = idxs[1]

        neg_labels = unique_labels[unique_labels != label]
        if len(neg_labels) == 0: continue
        neg_label = np.random.choice(neg_labels)
        neg_idxs = np.where(labels == neg_label)[0]
        if len(neg_idxs) == 0: continue 
        neg_idx = np.random.choice(neg_idxs)

        anchors.append(anchor_idx)
        positives.append(positive_idx)
        negatives.append(neg_idx)

    if not anchors: return None, None, None 
    return (features[anchors].to(device), features[positives].to(device), features[negatives].to(device))

def train_mm_hcan_with_triplet(
    temporal_model, spectral_model, ft_loader, 
    hg_temp, hg_spec, hg_concat, 
    fusion_layer, classifier, device, optimizer
):
    criterion_cls = CrossEntropyLoss()
    criterion_triplet = TripletLoss(margin=0.7)

    hg_temp.train(); hg_spec.train(); hg_concat.train()
    fusion_layer.train(); classifier.train()

    total_loss_total = 0; total_correct = 0; total_samples = 0

    for x_temp, x_spec, y in ft_loader:
        x_temp, x_spec, y = x_temp.to(device), x_spec.to(device), y.to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            f_temp = temporal_model(x_temp)
            f_spec = spectral_model(x_spec)
        
        f_concat = torch.cat((f_temp, f_spec), dim=1) 

        laps = build_all_hypergraphs(f_temp, f_spec, f_concat, k=5)
        L_t, L_s, L_c = laps['L_temp'].to(device), laps['L_spec'].to(device), laps['L_concat'].to(device)

        upd_temp = hg_temp(f_temp, L_t) 
        upd_spec = hg_spec(f_spec, L_s) 
        
        upd_concat = hg_concat(f_concat, L_c)

        fused = fusion_layer([upd_temp, upd_spec, upd_concat]) 

        anc, pos, neg = select_triplets(fused, y, device=device)
        loss_triplet = criterion_triplet(anc, pos, neg) if anc is not None else torch.tensor(0.0, device=device)

        logits = classifier(fused)
        loss_cls = criterion_cls(logits, y)

        total_loss = loss_cls + 0.7 * loss_triplet
        total_loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)
        total_loss_total += total_loss.item() * y.size(0)

    return total_loss_total / total_samples, total_correct / total_samples

def evaluate_mm_hcan(loader, temporal_model, spectral_model,
                     hg_temp, hg_spec, hg_concat,
                     fusion_layer, classifier, device):

    hg_temp.eval(); hg_spec.eval(); hg_concat.eval()
    fusion_layer.eval(); classifier.eval()

    correct = 0; total = 0; total_loss = 0
    criterion_cls = CrossEntropyLoss()

    with torch.no_grad():
        for x_temp, x_spec, y in loader:
            x_temp, x_spec, y = x_temp.to(device), x_spec.to(device), y.to(device)

            f_temp = temporal_model(x_temp)
            f_spec = spectral_model(x_spec)
            
            f_concat = torch.cat((f_temp, f_spec), dim=1) 

            laps = build_all_hypergraphs(f_temp, f_spec, f_concat, k=5)
            L_t, L_s, L_c = laps['L_temp'].to(device), laps['L_spec'].to(device), laps['L_concat'].to(device)

            upd_temp = hg_temp(f_temp, L_t)
            upd_spec = hg_spec(f_spec, L_s)
            upd_concat = hg_concat(f_concat, L_c)

            fused = fusion_layer([upd_temp, upd_spec, upd_concat])

            logits = classifier(fused)
            preds = torch.argmax(logits, dim=1)
            loss = criterion_cls(logits, y)

            total_loss += loss.item() * y.size(0)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total

def run_full_pipeline(root_dir="Dataset", epochs=50):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs("checkpoints", exist_ok=True)

    ft_loader, val_loader, test_loader = get_dataloaders(root_dir, 16)
    temporal_model, spectral_model = load_best_models(device)

    hg_temp = HGNNLayer(512, 512, 512).to(device)
    hg_spec = HGNNLayer(512, 512, 512).to(device)
    hg_concat = HGNNLayer(1024, 512, 512).to(device)
    
    fusion_layer = MultiHeadAttentionFusion(dim=512).to(device)
    classifier = FinalClassifier(input_dim=512).to(device)

    optimizer = torch.optim.Adam(
        list(hg_temp.parameters()) + list(hg_spec.parameters()) + list(hg_concat.parameters()) +
        list(fusion_layer.parameters()) + list(classifier.parameters()),
        lr=1e-4
    )

    best_val_acc = 0.0

    print("🧠 Training MM-HCAN (Multi-Branch)...")
    for epoch in range(epochs):
        train_loss, train_acc = train_mm_hcan_with_triplet(
            temporal_model, spectral_model, ft_loader, 
            hg_temp, hg_spec, hg_concat, 
            fusion_layer, classifier, device, optimizer
        )
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        val_loss, val_acc = evaluate_mm_hcan(
            val_loader, temporal_model, spectral_model,
            hg_temp, hg_spec, hg_concat,
            fusion_layer, classifier, device
        )
        print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'hg_layers': {
                    'temp': hg_temp.state_dict(),
                    'spec': hg_spec.state_dict(),
                    'concat': hg_concat.state_dict()
                },
                'fusion': fusion_layer.state_dict(),
                'classifier': classifier.state_dict()
            }, "checkpoints/best_val_model.pth")
            print(f"💾 Saved best val model")

    test_loss, test_acc = evaluate_mm_hcan(
        test_loader, temporal_model, spectral_model,
        hg_temp, hg_spec, hg_concat,
        fusion_layer, classifier, device
    )
    print(f"\nTest Result: Acc: {test_acc:.4f}")

if __name__ == "__main__":
    run_full_pipeline()
