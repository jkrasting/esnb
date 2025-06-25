class VirtualDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def replace(self, new_dataset):
        self.dataset = new_dataset


def resolve_dataset_refs(obj):
    if isinstance(obj, VirtualDataset):
        return obj.dataset
    elif isinstance(obj, dict):
        return {k: resolve_dataset_refs(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_dataset_refs(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(resolve_dataset_refs(item) for item in obj)
    else:
        return obj
