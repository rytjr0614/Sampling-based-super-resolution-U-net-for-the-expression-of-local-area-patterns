# Sampling based super resolution U-net for the expression of local area patterns


Install module
```bash
pip install -r requirements.txt
```

Add folder "result", "content". Add folder "Set5", "Set14", "bsd100", "Urban100", "images" in "content" and download dataset. "images" folder have "train", "val" folder each have BSD200, T91 dataset

Train model

```bash
python train.py
```

Test model

```bash
python test.py
```
