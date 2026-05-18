import torch
import torch.nn as nn


class SurrogateRiskEstimator(nn.Module):
    """Pre-evaluation module: predicts per-sample cross-entropy loss (risk)."""

    def __init__(self, args):
        super().__init__()
        H = args.loss_hidden_size

        # Sequence feature extractor: two-layer MLP with mean pooling
        self.feature_extractor = nn.Sequential(
            nn.Linear(args.emb_size, H), nn.ReLU(), nn.Dropout(args.dropout),
            nn.Linear(H, H),             nn.ReLU(), nn.Dropout(args.dropout),
        )

        # Target representations
        self.target_embedding = nn.Embedding(args.labels_num, H)
        self.soft_tgt_projector = nn.Sequential(
            nn.Linear(args.labels_num, H), nn.ReLU()
        )

        # Loss prediction head: three-layer MLP
        dim = H * 3 + 2          # seq features + hard target + soft target + 2 hyper features
        self.loss_predictor = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.ReLU(), nn.Dropout(args.dropout),
            nn.Linear(dim // 2, dim // 2), nn.ReLU(),
            nn.Linear(dim // 2, 1),
        )
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, emb, tgt, soft_tgt, soft_alpha, soft_targets):
        # Sequence features with global mean pooling
        feat = self.dropout(torch.mean(self.feature_extractor(emb), dim=(1, 2)))
        # Target features
        tgt_emb = self.target_embedding(tgt.view(-1))
        soft_feat = self.soft_tgt_projector(soft_tgt)
        # Hyper-parameter features
        hyper = torch.cat((soft_alpha.unsqueeze(-1), soft_targets.unsqueeze(-1)), dim=-1)
        # Concatenate and predict loss
        combined = torch.cat((feat, tgt_emb, soft_feat, hyper), dim=-1)
        return self.loss_predictor(combined)


# ======================== PERO Training Loop ========================

# Initialize pre-evaluation module
risk_model = SurrogateRiskEstimator(args).to(device)
risk_optim = torch.optim.Adam(risk_model.parameters(), lr=args.riskmodel_lr)
risk_loss_fn = nn.MSELoss()

for src_batch, tgt_batch, seg_batch, soft_tgt_batch in batch_loader(
    candidate_batch_size, src, tgt, seg, soft_tgt
):
    # --- Step 1: Pre-evaluate candidate samples ---
    with torch.no_grad():
        emb = classifier.embedding(src_batch, seg_batch)

    predicted_risk = risk_model(emb, tgt_batch, soft_tgt_batch,
                                soft_alpha, soft_targets)

    # --- Step 2: Select top-B highest-risk samples ---
    _, index = torch.topk(predicted_risk.squeeze(), k=batch_size)
    src_batch = src_batch[index]
    tgt_batch = tgt_batch[index]
    seg_batch = seg_batch[index]
    soft_tgt_batch = soft_tgt_batch[index]
    emb = emb[index]

    # --- Step 3: Update classifier on selected samples ---
    classifier.zero_grad()
    loss, per_sample_loss, _ = classifier(src_batch, tgt_batch,
                                          seg_batch, soft_tgt_batch)
    loss.backward()
    classifier_optim.step()
    classifier_scheduler.step()

    # --- Step 4: Update pre-evaluation module ---
    risk_optim.zero_grad()
    risk_loss = risk_loss_fn(predicted_risk[index], per_sample_loss.detach())
    risk_loss.backward()
    risk_optim.step()
