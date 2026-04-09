"""Compatibility layer for shared dataset split definitions."""

from data_provider.data_split_recorder import split_recorder as _ProviderSplitRecorder


class split_recorder(_ProviderSplitRecorder):
    """Expose data-provider split definitions with legacy helper methods."""

    ZNion_train_files = _ProviderSplitRecorder.ZNcoin_train_files
    ZNion_val_files = _ProviderSplitRecorder.ZNcoin_val_files
    ZNion_test_files = _ProviderSplitRecorder.ZNcoin_test_files

    NA_train_files = _ProviderSplitRecorder.NAion_2024_train_files
    NA_val_files = _ProviderSplitRecorder.NAion_2024_val_files
    NA_test_files = _ProviderSplitRecorder.NAion_2024_test_files

    @staticmethod
    def get_split_lists(dataset_name):
        try:
            train_list = getattr(split_recorder, f"{dataset_name}_train_files")
            val_list = getattr(split_recorder, f"{dataset_name}_val_files")
            test_list = getattr(split_recorder, f"{dataset_name}_test_files")
            return train_list, val_list, test_list
        except AttributeError:
            pass

        if dataset_name.startswith("SNL_"):
            sub_type = dataset_name.replace("SNL_", "")
            if sub_type in ["LFP", "NCA", "NMC"]:
                base_train = getattr(split_recorder, "SNL_train_files", [])
                base_val = getattr(split_recorder, "SNL_val_files", [])
                base_test = getattr(split_recorder, "SNL_test_files", [])
                return (
                    [file_name for file_name in base_train if sub_type in file_name],
                    [file_name for file_name in base_val if sub_type in file_name],
                    [file_name for file_name in base_test if sub_type in file_name],
                )

        if dataset_name.startswith("Tongji_"):
            sub_type = dataset_name.replace("Tongji_", "")
            keyword_map = {
                "NCA": "Tongji1",
                "NMC": "Tongji2",
                "NCA_NMC": "Tongji3",
            }
            keyword = keyword_map.get(sub_type)
            if keyword:
                base_train = getattr(split_recorder, "Tongji_train_files", [])
                base_val = getattr(split_recorder, "Tongji_val_files", [])
                base_test = getattr(split_recorder, "Tongji_test_files", [])
                return (
                    [file_name for file_name in base_train if keyword in file_name],
                    [file_name for file_name in base_val if keyword in file_name],
                    [file_name for file_name in base_test if keyword in file_name],
                )

        if dataset_name.startswith("CALB_") and ("_LT" in dataset_name or "_HT" in dataset_name):
            parts = dataset_name.split("_")
            if len(parts) >= 3:
                version = parts[1]
                temp_mode = parts[-1]
                base_dataset = f"CALB_{version}"
                base_train = getattr(split_recorder, f"{base_dataset}_train_files", [])
                base_val = getattr(split_recorder, f"{base_dataset}_val_files", [])
                base_test = getattr(split_recorder, f"{base_dataset}_test_files", [])

                def filter_calb(file_list, mode):
                    if mode == "LT":
                        return [file_name for file_name in file_list if "_0_" in file_name]
                    if mode == "HT":
                        return [file_name for file_name in file_list if "_25_" in file_name or "_35_" in file_name]
                    return []

                return (
                    filter_calb(base_train, temp_mode),
                    filter_calb(base_val, temp_mode),
                    filter_calb(base_test, temp_mode),
                )

        return [], [], []
