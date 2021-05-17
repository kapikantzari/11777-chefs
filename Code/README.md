To train MS-TCN for backbone of proposed method, put all video features and ground truth label files in `data` folder and use command

`python main.py --action=train --root_dir=../data`

To evaluate proposed methods on validation set, put additional checkpoints for MS-TCN and text-video matching in `data` folder and use command

`python main.py --action=eval --root_dir=../data --word2vec_cp=word2vec_cp.bin --howto100m_cp=howto100m_cp.pth --mstcn_cp=mstcn_cp.pth` 

To fine-tune pretrained HowTo100M model on EPIC-KITCHENS dataset, follow the instructions on [antoine77340/howto100m: Code for the HowTo100M paper (github.com)](https://github.com/antoine77340/howto100m) with our dataloader located in `proposed_methods/epic_dataloader.py` 