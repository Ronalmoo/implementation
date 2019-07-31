# implementation
### 논문을 열심히 읽고 열심히 구현해봅시당

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