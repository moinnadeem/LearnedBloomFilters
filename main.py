from argparse import ArgumentParser
from LBF import LearnedBloomFilter
import utils
from tqdm import tqdm
import pathlib
from sklearn.metrics import accuracy_score
from torch import nn
import string
import torch
import tqdm_logger as logger
from tqdm_logger.ansistyle import stylize, fg, bg, attr

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--savedir", default="trained_models/")
    parser.add_argument("--datadir", default="data/")
    return parser.parse_args()

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_df, dev_df, test_df = utils.load_data(args.datadir)
    INPUT_SIZE = len(string.printable) 
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 1
    N_LAYERS = 2
    model = LearnedBloomFilter(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, n_layers=N_LAYERS)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        print(stylize(f'Epoch {epoch}', fg('red'), attr('underlined')))

        # training phase
        running_loss = 0.0
        running_accuracy = 0.0
        steps = 0
        with tqdm(total=len(df)//args.batch_size) as pbar:
            for batch_offset in range(0, len(train_df), args.batch_size):
                # prepare the batch
                batch = [utils.char_tensor(i).to(device) for i in train_df[batch_offset:batch_offset+batch_size].url] 
                batch.sort(key=lambda x: len(x), reverse=True)
                lengths = [len(i) for i in batch]
                batch = nn.utils.rnn.pad_sequence(batch).to(device)
                Y = (df[batch_offset:batch_offset+batch_size].label=="bad").astype(int)

                # run it through the model
                output = model(batch) 
                loss = criterion(output, Y) 

                # update our metrics
                steps += 1
                running_loss += loss.item()
                preds = torch.argmax(output)
                accuracy = accuracy_score(preds, Y) 
                running_accuracy += accuracy

                # update the gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                pbar.update(1)
                if (batch_offset % 10) == 0:
                    batch_loss = running_loss / steps 
                    batch_acc = running_accuracy / steps 

                    desc = f'Loss: {batch_loss}, Accuracy: {batch_acc}'
                    logger.seclog(['train', 'blue'], desc, update=True)
                    
                    steps = 0
                    running_loss = 0.0
                    running_accuracy = 0.0
                            


        # dev phase
        running_loss = 0.0
        running_accuracy = 0.0
        steps = 0
        with tqdm(total=len(df)//args.batch_size) as pbar:
            for batch_offset in range(0, len(train_df), args.batch_size):
                # prepare the batch
                batch = [utils.char_tensor(i).to(device) for i in train_df[batch_offset:batch_offset+batch_size].url] 
                batch.sort(key=lambda x: len(x), reverse=True)
                lengths = [len(i) for i in batch]
                batch = nn.utils.rnn.pad_sequence(batch).to(device)
                Y = (df[batch_offset:batch_offset+batch_size].label=="bad").astype(int)

                # run it through the model
                output = model(batch) 
                loss = criterion(output, Y) 

                # update our metrics
                steps += 1
                running_loss += loss.item()
                preds = torch.argmax(output)
                accuracy = accuracy_score(preds, Y) 
                running_accuracy += accuracy

                # update the gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                pbar.update(1)
                if (batch_offset % 10) == 0:
                    batch_loss = running_loss / steps 
                    batch_acc = running_accuracy / steps 

                    desc = f'Loss: {batch_loss}, Accuracy: {batch_acc}'
                    logger.seclog(['train', 'blue'], desc, update=True)
                    
                    steps = 0
                    running_loss = 0.0
                    running_accuracy = 0.0

        print(stylize(f'Saving the model...', fg('green'), attr('bold')))

        # creating directory to save files in, if it doesn't exist.
        pathlib.Path(args.savedir).mkdir(parents=True, exist_ok=True)

        save_path = os.path.join(args.savedir, f'ep{epoch}_loss_{running_loss}_acc_{running_accuracy}.pth')
        torch.save(model.state_dict(), save_path)


if __name__=="__main__":
    args = parse_args()
    main(args)
