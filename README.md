# Few-Shot Learning with a Prototypical Network
Project for IMGS789 - Machine Learning for Difficult Data

[Project Paper Link](https://github.com/maxbak753/Few-Shot-Learning-IMGS789/blob/main/IMGS%20789%20Project%20Paper.pdf)

## Overview
Few-shot learning is a machine learning technique used when new classes need to be added to a classifier, and there is limited data available for the new classes. In this project, I aim to investigate two scenarios: when there are five samples (shots) per class and when only a single sample exists for each class.

To perform few-shot learning, there are many methods, but here I use a prototypical network. A prototypical network works by learning a latent space embedding that makes it possible to do a simple nearest-neighbor classification to predict the outputs. It works by splitting the training data into support and query sets, embedding the support set in the latent space, computing the average value of each class to use as a prototype, then comparing to the query data with the support prototypes and classifying the data as belonging to the closest class by some distance metric.

## Implementation
For my implementation I primarily used the PyTorch Python library with the Omniglot dataset. I chose a simple euclidean distance metric to classify the data. I used 10 epochs with 100 episodes of support-query training per epoch, and the ADAM optimizer. I chose 5-way 1-shot learning and 5-way 5-shot learning with 15 query samples per episode. The final test evaluation was also done with 5 classes, but with 1000 test iterations to increase the reliability of the results.

My convolutional neural network consists of 3 sets of convolution layer → max-pooling layer → ReLU, followed by average pooling, and then a linear fully connected layer to get the final embeddings. I originally started with LeNet, added an additional convolutional layer, then modified the end layer to only do embedding.

## Results

Overall, the model seemed to perform well, with an average accuracy of 94.47% and a fairly small standard deviation of 5.02%, for 5-shot learning. However, 1-shot learning performed worse, with 85.81% accuracy with a larger  9.82% standard deviation. The Matthews Correlation Coefficient had a similar pattern with a value of 0.8305 for 1-shot and 0.9325 for 5-shot. The results are depicted in table #1.

![Training Loss (1-shot)](https://github.com/maxbak753/Few-Shot-Learning-IMGS789/blob/main/images/1-shot%20training%20plot.PNG)
Figure #3: Training Loss (1-Shot)
![Training Loss (5-shot)](https://github.com/maxbak753/Few-Shot-Learning-IMGS789/blob/main/images/5-shot%20training%20plot.PNG)
Figure #2: Training Loss (5-shot)
![Results for 1-shot and 5-shot learning](https://github.com/maxbak753/Few-Shot-Learning-IMGS789/blob/main/images/result%20table%201-shot%205-shot.PNG)
Table #1: Results for 1-shot and 5-shot learning

## References
[1] IBM, “What is Few-Shot Learning?,” IBM Think. [Online]. Available: https://www.ibm.com/think/topics/few-shot-learning. [Accessed: Jul. 6, 2026].<br>
[2] PyTorch Contributors, “Torchvision Datasets,” PyTorch Documentation. [Online]. Available: https://docs.pytorch.org/vision/stable/datasets.html. [Accessed: Jul. 6, 2026].<br>
[3] H. Gharoun, F. Momenifar, F. Chen, and A. H. Gandomi, “Meta-learning Approaches for Few-Shot Learning: A Survey of Recent Advances,” ACM Computing Surveys, vol. 56, no. 12, pp. 294:1–294:41, 2024. doi: 10.1145/3659943.<br>
[4] M. Ali, “Prototypical Networks for Few-Shot Learning in PyTorch,” CodeGenes Blog. [Online]. Available: https://www.codegenes.net/blog/prototypical-networks-for-fewshot-learning-pytorch/. [Accessed: Jul. 6, 2026].<br>
[5] M. Grandini, E. Bagli, and G. Visani, “Metrics for Multi-Class Classification: An Overview,” arXiv preprint arXiv:2008.05756, 2020. [Online]. Available: https://arxiv.org/abs/2008.05756. [Accessed: Jul. 6, 2026].<br>
[6] B. M. Lake, “Omniglot: A Dataset for One-Shot Learning,” GitHub repository. [Online]. Available: https://github.com/brendenlake/omniglot. [Accessed: Jul. 6, 2026].<br>
