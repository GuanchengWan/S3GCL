# S3GCL: Spectral, Swift, Spatial Graph Contrastive Learning

>Guancheng Wan, Yijun Tian, Wenke Huang, Nitesh V Chawla, Mang Ye

## Abstract
Graph Contrastive Learning (GCL) has emerged as a highly effective self-supervised approach in graph representation learning. However, prevailing GCL methods confront two primary challenges: 1) They predominantly operate under homophily assumptions, focusing on low-frequency signals in node features while neglecting heterophilic edges that connect nodes with dissimilar features. 2) Their reliance on neighborhood aggregation for inference leads to scalability challenges and hinders deployment in real-time applications. In this paper, we introduce S3GCL,  an innovative framework designed to tackle these challenges. Inspired by spectral GNNs, we initially demonstrate the correlation between frequency and homophily levels. Then, we propose a novel cosine-parameterized Chebyshev polynomial as low/high-pass filters to generate biased graph views. To resolve the inference dilemma, we incorporate an MLP encoder and enhance its awareness of graph context by introducing structurally and semantically neighboring nodes as positive pairs in the spatial domain. Finally, we formulate a cross-pass GCL objective between full-pass MLP and biased-pass GNN filtered features, eliminating the need for augmentation. Extensive experiments on real-world tasks validate S3GCL proficiency in generalization to diverse homophily levels and its superior inference efficiency.


## Citation

``` latex
@inproceedings{S3GCL_ICML24,
    title={S3GCL: Spectral, Swift, Spatial Graph Contrastive Learning},
    author={Wan, Guancheng and Tian, Yijun and Huang, Wenke and Chawla, Nitesh V  and Ye, Mang},
    booktitle={ICML},
    year={2024}
}
```
