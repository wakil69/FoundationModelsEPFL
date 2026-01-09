## **Part 2 – ROI Classification Approach**

### **Data Description**

For the second part of the assignment, the goal was to classify patients based on sets of pre-computed ROI patch embeddings. We have **3969 patients** which have **between 1 and 128 patches each**. The total number of patches is **104,089**.

One patient has one label, and there are **7 possible (unbalanced) labels**:
* **Label 0:** 429 patients (10.81%)
* **Label 1:** 705 patients (17.76%)
* **Label 2:** 673 patients (16.95%)
* **Label 3:** 568 patients (14.31%)
* **Label 4:** 403 patients (10.15%)
* **Label 5:** 757 patients (19.05%)
* **Label 6:** 435 patients (10.96%)

### **Model Choice: MLP**

Since our dataset contains a reasonably large number of patients, using neural network models becomes feasible and often more effective than traditional machine learning approaches such as random forests or SVMs. For this reason, I chose to use a **multilayer perceptron (MLP)** as the downstream classifier. Indeed, an MLP is lightweight, quick to train, and suited for fixed-size vector inputs (our patch embeddings are of dimension 3072).

### **Architecture Description**

The architecture I implemented is the following: two fully connected layers with ReLU activations project the original **3072-dimensional embeddings** into a **512-dimensional latent representation**. Each patch is processed independently through this network.

Since different patients can have very different numbers of patches, I used a **multi-instance learning approach** to construct a single patient-level feature vector: apply **mean pooling** over all patch embeddings belonging to the same patient. While more advanced pooling mechanisms exist (e.g., attention-based pooling or learnable pooling layers), mean pooling proved to be effective and robust, particularly given the imbalance in the number of patches across patients.

The resulting mean patient embedding is then passed through a final linear classifier that outputs the probability distribution over the seven diagnosis classes. During training, the model outputs raw logits to be compatible with the cross-entropy loss. During evaluation, we return softmax probabilities instead, as required by the ROC-AUC computation in the course evaluation pipeline.

### **Cross-Validation Strategy**

To improve generalization and reduce the risk of overfitting to a particular subset of patients, we trained the model using **5-fold cross-validation**, making sure the folds were split **at the patient level** to avoid information leakage. This choice is motivated by the relatively small and heterogeneous nature of the dataset. A single train/validation split might therefore give an inaccurate estimate of performance.

With 5-fold CV, the model is trained and evaluated on five different partitions, and we keep the best-performing model according to the validation F1 score. Our best fold achieved around **80.13% F1** on its validation set.

When evaluating the final model on the entire dataset using the provided evaluation script, the F1 score increased to **95.19%**. This is expected, as the evaluation includes all data, meaning the model has seen at least 80% of it during training. True generalization would need to be assessed on an external or held-out test set.

### **Conclusion**

In summary, the MLP-based approach, combined with mean pooling and 5-fold cross-validation, yielded **strong and stable results** for ROI-based patient classification.
