import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import utils
from torch.nn import Linear
from mlppredictor import MLPPredictor


class DglBaseModel(LightningModule):
    def __init__(self, encoder, params):
        super().__init__()
        self.save_hyperparameters(params)
        hyparams = self.hparams

        self.class_weight = torch.tensor([float(i) for i in self.hparams.class_weight.split('-')], dtype=torch.float32)
        # Loss
        self.loss = nn.CrossEntropyLoss(weight=self.class_weight)

        hidden_feats = self.hparams.hidden_channels

        self.model = encoder

        self.pred = MLPPredictor(self.hparams.out_channels, self.hparams.num_labels, hyparams.edge_dim)

        hidden_edims = hidden_feats
        self.lin0 = Linear(hidden_edims, hidden_feats * 2)
        self.lin1 = Linear(hidden_feats * 2, self.hparams.out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.model.reset_parameters()
        self.pred.reset_parameters()
        self.lin0.reset_parameters()
        self.lin1.reset_parameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=5e-4)
        return optimizer

    def forward(self, blocks, x, graph):

        row, col = graph.edges()
        edge_index_reshape = torch.stack([row, col], dim=0)
        edge_attr = graph.edata['h']
        h3 = self.model(x,  edge_index_reshape, edge_attr)

        h3 = F.relu(self.lin0(h3))
        h3 = F.relu(self.lin1(h3))

        pred = self.pred(graph, h3, edge_attr)
        return pred

    def training_step(self, batch, batch_idx):
        _, g, blocks = batch
        e_labels = g.edata['label']
        x = g.ndata["h"]
        pred = self(blocks, x, g)
        ce_loss = self.loss(pred, e_labels)
        logs = {
            'ce_loss': ce_loss,
            'batch_size': torch.tensor(g.num_edges(), dtype=torch.float32),
        }
        self.log_dict(logs, prog_bar=True, batch_size=self.hparams.batch_size)
        return ce_loss

    def validation_step(self, batch, batch_idx):
        _, g, blocks = batch
        e_labels = g.edata['label']
        x = g.ndata["h"]
        pred = self(blocks, x, g)
        ce_loss = self.loss(pred, e_labels)

        acc, f1_macro, f1_weighted = utils.calc_metrics(pred, e_labels, num_labels=self.hparams.num_labels)
        logs = {
            'ce_loss': ce_loss,
            'acc': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'batch_size': torch.tensor(g.num_edges(), dtype=torch.float32),
        }
        self.log_dict(logs, prog_bar=True, batch_size=self.hparams.batch_size)
        return logs

    def validation_epoch_end(self, val_step_outputs):
        avg_ce_loss = torch.stack([x['ce_loss'] for x in val_step_outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in val_step_outputs]).mean()
        avg_f1_macro = torch.stack([x['f1_macro'] for x in val_step_outputs]).mean()
        avg_f1_weighted = torch.stack([x['f1_weighted'] for x in val_step_outputs]).mean()
        logs = {
            'val_ce_loss': avg_ce_loss,
            'val_acc': avg_acc,
            'val_macro_f1': avg_f1_macro,
            'val_weighted_f1': avg_f1_weighted,
        }
        self.log_dict(logs, prog_bar=True, batch_size=self.hparams.batch_size)

    def predict(self, data):
        _, g, blocks = data
        e_labels = g.edata['label']
        x = g.ndata["h"]
        pred = self(blocks, x, g)
        probs = F.softmax(pred, 1)
        return probs, e_labels
