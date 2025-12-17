import torch

from eeg_ctm.models.ctm_core import CTMCoreConfig
from eeg_ctm.models.eeg_ctm_model import EEGCTM, EEGCTMConfig
from eeg_ctm.models.losses import TickLossConfig, tick_classification_loss
from eeg_ctm.models.pairs import PairBank, PairBankConfig
from eeg_ctm.models.tokenizer import TokenizerV1Config


def test_eeg_ctm_forward_backward_smoke():
    pair_bank = PairBank(PairBankConfig())
    model = EEGCTM(
        EEGCTMConfig(tokenizer=TokenizerV1Config(), ctm=CTMCoreConfig()),
        pair_bank=pair_bank,
    )
    x = torch.randn(4, 22, 1000)
    y = torch.tensor([0, 1, 2, 3])
    logits_ticks, certainty, _ = model(x)
    assert logits_ticks.shape == (4, 4, 12)
    assert certainty.shape == (4, 12)
    loss, _ = tick_classification_loss(logits_ticks, certainty, y, cfg=TickLossConfig(type="mean_ce"))
    loss.backward()

