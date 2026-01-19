from torch.utils.data import Dataset


class CyclingDataset(Dataset):
    def __init__(self, _samples, _targets, _stages, _riders=None):
        self.samples = _samples
        self.targets = _targets
        self.stages = _stages
        self.riders = _riders

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _samples = self.samples[idx]
        _targets = self.targets[idx]
        _stages = self.stages[idx]
        _riders = self.riders[idx] if self.riders is not None else None
        return {'samples': _samples, 'targets': _targets, 'stages': _stages, 'riders': _riders}