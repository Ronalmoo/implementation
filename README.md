# implementation
### 논문을 열심히 읽고 열심히 구현해봅시당


usage: main_last.py [-h] [--emb_size EMB_SIZE] [--n_layer N_LAYER]
                    [--batch_size BATCH_SIZE] [--hidden_size HIDDEN_SIZE]
                    [--output_size OUTPUT_SIZE] [--lr LR] [--epochs EPOCHS]
                    [--clip CLIP]

optional arguments:
  -h, --help            show this help message and exit
  --emb_size EMB_SIZE   Embedding size
  --n_layer N_LAYER     number of layer
  --batch_size BATCH_SIZE
                        batch size
  --hidden_size HIDDEN_SIZE
                        size of hidden layer
  --output_size OUTPUT_SIZE
                        size of output layer
  --lr LR               learning rate
  --epochs EPOCHS       epoch number of training
  --clip CLIP           learning rate
### Classification
+ Using the [Naver sentiment movie corpus v1.0](https://github.com/e9t/nsmc)
+ Hyper-parameter was arbitrarily selected.

|                  | Train ACC (120,000) | Validation ACC (30,000) | Test ACC (50,000) |
| :--------------- | :-------: | :------------: | :------: |
| Baseline (Feed Forward)         |  -  |     -     |  -  |
| SenCNN           |  92.3%  |     84.98%     |  84.42%  |
| CharCNN          | - | - | - |
| ConvRec          |  85.24% |  83.19  |  82.95%  |
| VDCNN            | - | - | - |
| SAN | - | - | - |
