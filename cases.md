## Training cases

### Case 1 (Removed due to bad results)

- Modified the SAGPooling layer with `ratio` = 0.7

### Case 2 (Removed due to bad results)

- Modified the SAGPooling layer with `ratio` = 0.9

### Case 3 (Done)

- Modified the CoAttentionLayer not using tanh activation functions on equation (10)
- From this `e_scores = torch.tanh(e_activations) @ self.a` to this `e_scores = e_activations @ self.a`

```txt
Test Accuracy: 0.8833
Test ROC AUC: 0.9477
Test PRC AUC: 0.9383
```

### Case 4 (Removed due to bad results)

- Added the `Explict Valence` feature.
- Not using tanh activation functions (As the same in case 3)

### Case 5 (Removed due to bad results)

- Added the `Explict Valence` feature.

### Case 6 (Removed due to bad results)

- Using `ComplEx` instead of `RESCAL` (First version)

### Case 7 (Removed due to bad results)

- Using `ComplEx` instead of `RESCAL` (First version)
- Not using tanh activation functions (As the same in case 3)
- Added the `Explict Valence` feature.
- Adding exporting metrics function

### Case 8 (Removed due to bad results)

- Using `ComplEx` instead of `RESCAL` (First version)
- Not using tanh activation functions (As the same in case 3)

### Case 9 (Removed due to bad results)

- Using `ComplEx` instead of `RESCAL` (First version)
- Not using tanh activation functions (As the same in case 3)
- Added the `Explict Valence` feature.
- Adding exporting metrics function
- Modified the SAGPooling layer with `ratio` = 0.5

### Case 10 (Removed due to bad results)

- Modified the SAGPooling layer with `min_score` = 0.5

### Case 11 (Removed due to bad results)

- Modified the SAGPooling layer with `min_score` = 0.2

### Case 12 (Removed due to bad results)

- Using `ComplEx` instead of `RESCAL`
- Not using tanh activation functions (As the same in case 3)
- Using 5 GAT layers

### Case 13 (Done)

- Not using tanh activation functions (As the same in case 3)
- Using 5 GAT layers

```txt
Test Accuracy: 0.8938
Test ROC AUC: 0.9543
Test PRC AUC: 0.9462
```

### Case 14 (Removed due to bad results)

- Using updated version of CoAttentionLayer
- Using 5 GAT layers

```txt
Test Accuracy: 0.9016
Test ROC AUC: 0.9587
Test PRC AUC: 0.9519
```

### Case 15 (Removed due to bad results)

- Using updated version of CoAttentionLayer

```txt
Test Accuracy: 0.8838
Test ROC AUC: 0.9470
Test PRC AUC: 0.9364
```

### Case 16 (Removed due to bad results)

- Using updated version of CoAttentionLayer
- Using 5 GAT layers
- Using `ComplEx` instead of `RESCAL` (First version)

```txt
Test Accuracy: 0.8875
Test ROC AUC: 0.9477
Test PRC AUC: 0.9380
```

### Case 17 (Removed due to bad results)

- Using updated version of CoAttentionLayer
- Added the `Explict Valence` feature.
- Using 5 GAT layers

```txt
Test Accuracy: 0.8687
Test ROC AUC: 0.9368
Test PRC AUC: 0.9259
```

### Case 18 (Removed due to bad results)

- Using Multihead CoAttentionLayer (First version)
- Using 5 GAT layers

```txt
Test Accuracy: 0.8918
Test ROC AUC: 0.9524
Test PRC AUC: 0.9443
```

### Case 19 (Removed due to bad results)

Note : Result is not better in comparison with case 18.

- Using Multihead CoAttentionLayer with 4 heads (First version)
- Using 5 GAT layers

```txt
Test Accuracy: 0.8953
Test ROC AUC: 0.9531
Test PRC AUC: 0.9430
```

### Case 20 (Done)

- Using Multihead CoAttentionLayer
- Using 5 GAT layers

```txt
Test Accuracy: 0.9027
Test ROC AUC: 0.9596
Test PRC AUC: 0.9533
```

### Case 21 (Done)

- Using Multihead CoAttentionLayer
- Using `ComplEx` instead of `RESCAL` (Second version)
- Using 5 GAT layers

```txt
Test Accuracy: 0.9053
Test ROC AUC: 0.9605
Test PRC AUC: 0.9532
```

### Case 22 (Done)

- Using Multihead CoAttentionLayer
- Using 4 GAT layers

```txt
Test Accuracy: 0.8902
Test ROC AUC: 0.9518
Test PRC AUC: 0.9427
```

### Case 24 (Result is not good)

- Using Multihead CoAttentionLayer(nheads = 4)
- Using 5 GAT layers

```txt
Test Accuracy: 0.8956
Test ROC AUC: 0.9545
Test PRC AUC: 0.9459
```

### Case 26 (Done)

- Using `ComplEx` instead of `RESCAL` (Second version)
- Using 5 GAT layers

```txt
Test Accuracy: 0.8999
Test ROC AUC: 0.9573
Test PRC AUC: 0.9492
```

### Case 27 (Done)

- Using `ComplEx` instead of `RESCAL`  (Second version)
- Using 4 GAT layers

```txt
Test Accuracy: 0.8901
Test ROC AUC: 0.9502
Test PRC AUC: 0.9407
```

### Case 28 (Done)

- Using Multihead CoAttentionLayer
- Using `ComplEx` instead of `RESCAL`  (Second version)
- Using 4 GAT layers

```txt
Test Accuracy: 0.8955
Test ROC AUC: 0.9546
Test PRC AUC: 0.9467
```
