import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from model.split import split_to_jamo
from model.net import ConvRec
from model.data import Corpus, batchify
from model.utils import Tokenizer
from model.metric import evaluate, acc
from utils import Config, CheckpointManager, SummaryManager
from gluonnlp import Vocab

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing config.json of data")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing config.json of model")


if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    data_config = Config(json_path=data_dir / 'config.json')
    model_config = Config(json_path=model_dir / 'config.json')

    # tokenizer
    with open('data/vocab.pkl', mode='rb') as io:
        vocab = pickle.load(io)
    print(vocab)
    tokenizer = Tokenizer(vocab=vocab, split_fn=split_to_jamo)

    # model
    model = ConvRec(num_classes=model_config.num_classes, embedding_dim=model_config.embedding_dim,
                    hidden_dim=model_config.hidden_dim, vocab=vocab)

    # training
    tr_ds = Corpus(data_config.train, tokenizer.split_and_transform, min_length=model_config.min_length,
                   pad_val=vocab.to_indices(' '))
    tr_dl = DataLoader(tr_ds, batch_size=model_config.batch_size, shuffle=True, num_workers=4,
                       collate_fn=batchify, drop_last=True)
    val_ds = Corpus(data_config.validation, tokenizer.split_and_transform, min_length=model_config.min_length,
                    pad_val=vocab.to_indices(' '))
    val_dl = DataLoader(val_ds, batch_size=model_config.batch_size, collate_fn=batchify)

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(params=model.parameters(), lr=model_config.learning_rate)
    scheduler = ReduceLROnPlateau(opt, patience=5)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    writer = SummaryWriter('{}/runs'.format(model_dir))
    checkpoint_manager = CheckpointManager(model_dir)
    summary_manager = SummaryManager(model_dir)
    best_val_loss = 1e+10

    epochs_progress = tqdm(range(model_config.epochs), desc='epochs')
    for epoch in epochs_progress:

        tr_loss = 0
        tr_acc = 0

        model.train()
        steps_progress = tqdm(enumerate(tr_dl), desc='steps', total=len(tr_dl))
        for step, mb in steps_progress:
            x_mb, y_mb = map(lambda elm: elm.to(device), mb)

            opt.zero_grad()
            y_hat_mb = model(x_mb)
            mb_loss = loss_fn(y_hat_mb, y_mb)
            mb_loss.backward()
            opt.step()

            with torch.no_grad():
                mb_acc = acc(y_hat_mb, y_mb)

            tr_loss += mb_loss.item()
            tr_acc += mb_acc.item()

            if (epoch * len(tr_dl) + step) % model_config.summary_step == 0:
                val_loss = evaluate(model, val_dl, {'loss': loss_fn}, device)['loss']
                writer.add_scalars('loss', {'train': tr_loss / (step + 1),
                                            'val': val_loss}, epoch * len(tr_dl) + step)
                model.train()
        else:
            tr_loss /= (step + 1)
            tr_acc /= (step + 1)

            tr_summary = {'loss': tr_loss, 'acc': tr_acc}
            val_summary = evaluate(model, val_dl, {'loss': loss_fn, 'acc': acc}, device)
            scheduler.step(val_summary['loss'])
            tqdm.write('epoch : {}, tr_loss: {:.3f}, val_loss: '
                       '{:.3f}, tr_acc: {:.2%}, val_acc: {:.2%}'.format(epoch + 1, tr_summary['loss'],
                                                                        val_summary['loss'], tr_summary['acc'],
                                                                        val_summary['acc']))

            val_loss = val_summary['loss']
            is_best = val_loss < best_val_loss

            if is_best:
                state = {'epoch': epoch + 1,
                         'model_state_dict': model.state_dict(),
                         'opt_state_dict': opt.state_dict()}
                summary = {'train': tr_summary, 'validation': val_summary}

                summary_manager.update(summary)
                summary_manager.save('summary.json')
                checkpoint_manager.save_checkpoint(state, 'best.tar')

                best_val_loss = val_loss
