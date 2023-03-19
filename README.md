# DNN_watermark_whitebox
DNN whitebox watermark implemented by pytorch

This project is for adding a whitebox watermark to a classification network. The used dataset is cifar10.

cifar10.py is for data processing.

model.py is a simple CNN for the classification task.

train_wm_net.py is for traning while add watermark to the net simultaneously. We use a generator X to extract the ID information, where X
is simply randomrized. Details should be found in the code.

eval.py is for evaluating the water-marked network.

Extraction_wm.py is for extracting the ID information, which is defaultly all 1s bit string.

fine_tune.py and distill.py simulate the fine-tuning and distilling attack. You can use them to assess how well the method against normal attacks.

Others are the corresponding training results. You are free to use them.
