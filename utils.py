def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = string.printable.index(string[c])
        except ValueError:
            continue
    return tensor

def load_data(data_dir):
    get_file = lambda x: os.path.join(data_dir, x)
    train_df = pd.read_csv(get_file("train.csv"))
    dev_df = pd.read_csv(get_file("dev.csv"))
    test_df = pd.read_csv(get_file("test.csv"))
    return train_df, dev_df, test_df
