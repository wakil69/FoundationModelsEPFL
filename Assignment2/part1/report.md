### **Part 1 – Test-Time Adaptation Report**

For this part of the assignment, I first tried out different test-time adaptation (TTA) methods to improve how a pretrained CIFAR-10 ResNet-50 model performs on corrupted test images. The final TTA method I decided to use was TENT [1]. I also trained a new model starting from the pretrained weights by adding corrupted examples that the model predicted with high confidence, treating them as pseudo-labels.

### **Approach 1: Experimenting with NORM parameters**

The first method I tested was the normalization-based approach (NORM) provided in the starter code. The idea behind NORM is that the BatchNorm layers contain running mean and variance values learned from clean CIFAR-10 images, but these no longer match the corrupted test data. NORM replaces these running stats with fresh batch statistics computed directly from each corrupted batch, helping the model adjust to the new data distribution.

I experimented with different batch sizes, epsilon values, and momentum settings for BatchNorm. These hyperparameters matter because they affect how stable and reliable the batch statistics are. The best setup I found used a batch size of 3072, epsilon of 1e-5, and momentum of 0.1. With this configuration, the mean accuracy increased from 84.34% (default NORM) to 85.24%, an improvement of +0.9%. Even though this was helpful, I had the feeling that methods that actually update model parameters during test time would work better, so I moved on to TENT.

### **Approach 2: Implementing TENT as the main TTA method**

The main method I worked on, and the one I ended up submitting, is TENT (Test-Time Entropy Minimization). Unlike NORM, which only recalculates BatchNorm statistics, TENT also updates the affine BatchNorm parameters (gamma and beta) during test time while freezing the rest of the model. The idea is that corrupted images usually make the model’s predictions uncertain, so TENT reduces the entropy of the predictions to make them more confident.

To implement TENT, I disabled BatchNorm running statistics so that each batch computes its own mean and variance, which makes sense under distribution shift. I then enabled gradients only for gamma and beta, and updated them using SGD after each forward pass. I also experimented with weak data augmentations (random crop and horizontal flip). I ran a grid search over learning rates (1e-3 to 1e-5), momentums (0.7 to 0.9), and batch sizes (128 to 2048).

### **Training a new model from the pretrained weights**

After running TENT with the pretrained model on CIFAR10, the best performing configuration used a learning rate of 6e-4, momentum of 0.8, and a batch size of 128. With this setup, TENT reached a mean accuracy of 86.96%, which is a +2.62% improvement (compared to default NORM). However, I thought I could push the performance a bit further.

To do this, I used the pretrained model on ImageNet to predict labels on the exploratory dataset. I then selected corrupted images where the model was highly confident (confidence ≥ 0.85) and treated those predictions as pseudo-labels. Starting from the pretrained model, I trained it for 50 epochs using both the original data (clean CIFAR-10) and these high-confidence pseudo-labeled corrupted images. With this approach, I was able to reach **88.14% accuracy** (+3.80% compared to default NORM), which was the highest score I achieved.

---

### **Reproduction of results**

To reproduce the results: (all the scripts are located in the folder additional_scripts)

1. Run **01_plot_image_threshold.py** to generate a plot showing how many images are selected at different confidence thresholds. Pick a threshold when it starts decreasing. I personnally picked an interval [0.8,0.99] and tested many threshold values. I ended up picking 0.85. (plot is provided:threshold_vs_count.png)
2. Run **02_labelise_noisy_data.py** with the chosen threshold to produce the dataset of high-confidence pseudo-labeled corrupted images. (located in data/pseudo_labeled_corrupted.pkl)
3. Train the model with this newly generated dataset and CIFAR-10 clean using **03_train_from_pretrain_pseudo_label.py**.
4. Copy the generated model path into **submission.yaml**.
5. Run **04_generate_config** to generate many config files with different parameters (grid search).
6. Run the eval with all the config files and select the parameters with highest accuracy.

## References

[1] Dequan Wang et al., _"Tent: Fully Test-time Adaptation by Entropy Minimization"_, 2020.  
https://arxiv.org/abs/2006.10726
