import torch
import torch.nn as nn

class SurrogateRiskEstimator(nn.Module):
    """
    pre-evaluation module: predicting the cross-entropy loss (risk)
    """
    def __init__(self, args):
        super(SurrogateRiskEstimator, self).__init__()
        self.emb_size = args.emb_size
        self.labels_num = args.labels_num
        self.loss_hidden_size = args.loss_hidden_size
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.dropout = nn.Dropout(args.dropout)
        # Sequence Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.emb_size, self.loss_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.loss_hidden_size, self.loss_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
        )
        self.target_embedding = nn.Embedding(self.labels_num, self.loss_hidden_size)
        self.soft_tgt_projector = nn.Sequential(
            nn.Linear(self.labels_num, self.loss_hidden_size),
            nn.ReLU()
        )
        self.num_hyper_features = 2
        # Loss prediction header
        if self.soft_targets:
            final_feature_dim = (self.loss_hidden_size * 3) + self.num_hyper_features
        else:
            final_feature_dim = self.loss_hidden_size * 2
        self.loss_predictor = nn.Sequential(
            nn.Linear(final_feature_dim, final_feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(final_feature_dim//2, final_feature_dim // 2),
            nn.ReLU(),
            nn.Linear(final_feature_dim // 2, 1) # Output scalar prediction loss
        )

    def forward(self, emb, tgt, soft_tgt):
        features = self.feature_extractor(emb)
        pooled_features = torch.mean(features, dim=(1,2)) # Global Mean Pooling
        pooled_features = self.dropout(pooled_features)
        tgt_emb = self.target_embedding(tgt.view(-1))
        soft_tgt_features = self.soft_tgt_projector(soft_tgt)
        if soft_alpha.dim() == 1:
            soft_alpha = soft_alpha.unsqueeze(-1)
        if soft_targets.dim() == 1:
            soft_targets = soft_targets.unsqueeze(-1)
        hyper_features = torch.cat((soft_alpha, soft_targets), dim=-1)
        combined_features = torch.cat((pooled_features, tgt_emb, soft_tgt_features, hyper_features), dim=-1)
        predicted_loss = self.loss_predictor(combined_features)
        return predicted_loss

# --- Main Training Loop Optimization ---

for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(candidate_batch_size, src, tgt, seg, soft_tgt)):
    with torch.no_grad():
        src_batch = src_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        if soft_tgt_batch != None:
            soft_tgt_batch = soft_tgt_batch.to(args.device)
        if torch.cuda.device_count() > 1:
            emb = Classifier.module.embedding(src_batch,seg_batch)
        else:
            emb = Classifier.embedding(src_batch,seg_batch)
    # Predicting sample-wise risk
    riskmodel = SurrogateRiskEstimator(args)
    riskmodel = riskmodel.to(args.device)
    optim = torch.optim.Adam(riskmodel.parameters(),lr=args.riskmodel_learning_rate)
    risk_lossfn = nn.MSELoss()
    risk = riskmodel(emb,tgt_batch,soft_tgt_batch)
    # The predicted risk as acquisition score
    score = risk
    # Selecting top-B samples with highest risk
    risk, index = torch.topk(score.squeeze(),k=batch_size)
    src_batch = src_batch[index]
    tgt_batch = tgt_batch[index]
    seg_batch = seg_batch[index]
    emb = emb[index]
    if soft_tgt_batch != None:
        soft_tgt_batch = soft_tgt_batch[index]
    Classifier.zero_grad()
    # Updating the classifier with selected samples
    loss, loss_, _ = Classifier(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)
        loss_ = loss_.view(-1)
    loss.backward()
    args.optimizer.step()
    args.scheduler.step()
    total_loss += loss.item()
    optim.zero_grad()
    # Updating the pre-evaluation module with selected samples
    riskloss = risk_lossfn(risk,loss_.detach())
    riskloss.backward()
    optim.step()
