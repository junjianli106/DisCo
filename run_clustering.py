'''
python run_clustering.py \
    --dataset_name BLCA \
    --n_clusters 32 \
    --input_dir /home/lijunjian/data/TCGA/Feats/TCGA-BLCA/feats-m20-s448-conch_v1_5/pt_files/ \
    --gpu
'''
import os
import torch
import numpy as np
import faiss
import argparse

class FaissKmeans:
    def __init__(self, n_clusters, niter=20, gpu=True):
        """
        KMeans clustering using Faiss.
        Args:
            n_clusters (int): Number of clusters.
            niter (int): Number of iterations for KMeans.
            gpu (bool): Whether to use GPU for clustering.
        """
        self.n_clusters = n_clusters
        self.niter = niter
        self.gpu = gpu

    def train(self, data):
        """Train the KMeans model with Faiss."""
        n_samples, n_features = data.shape
        kmeans = faiss.Clustering(n_features, self.n_clusters)
        kmeans.niter = self.niter
        kmeans.verbose = True

        if self.gpu:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.device = 0  # 使用第一个 GPU
            index = faiss.GpuIndexFlatL2(res, n_features, flat_config)
        else:
            index = faiss.IndexFlatL2(n_features)

        kmeans.train(data, index)
        centroids = faiss.vector_to_array(kmeans.centroids).reshape(self.n_clusters, n_features)
        _, labels = index.search(data, 1)

        return centroids, labels.ravel()


def main():
    # 1. 设置参数解析器
    parser = argparse.ArgumentParser(description="Perform K-means clustering on feature vectors using Faiss.")
    parser.add_argument('--n_clusters', type=int, required=True,
                        help='The number of clusters for K-means.')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Name of the dataset for logging purposes.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Full path to the directory containing the input .pt files.')
    parser.add_argument('--output_dir', type=str, default='./cluster_centers',
                        help='Full path to the directory where the output cluster centers will be saved.')
    parser.add_argument('--gpu', action='store_true',
                        help='Set this flag to use GPU for clustering.')
    parser.add_argument('--n_files', type=int, default=20,
                        help='Number of .pt files to process from the input directory.')

    # 2. Parse the command-line arguments
    args = parser.parse_args()

    # 3. Use the parsed arguments
    print(f"Processing dataset: {args.dataset_name}")
    print(f"Number of clusters: {args.n_clusters}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Using GPU: {args.gpu}")
    print(f"Number of files to process: {args.n_files}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Get the specified number of .pt files
    pt_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".pt")])[:args.n_files]
    if not pt_files:
        print(f"Error: No .pt files found in directory {args.input_dir}.")
        return

    # Load all .pt files and merge the data
    all_data = []
    print(f"Loading the first {len(pt_files)} .pt files...")
    for pt_file in pt_files:
        pt_file_path = os.path.join(args.input_dir, pt_file)
        print(f"Loading file: {pt_file_path}")
        data = torch.load(pt_file_path, map_location=lambda storage, loc: storage)
        all_data.append(data)

    if not all_data:
        print("Error: No valid data could be loaded. Exiting.")
        return

    # 合并所有加载的数据
    combined_data = np.vstack(all_data).astype(np.float32)  # Faiss 要求 float32 类型
    print(f"Combined data shape: {combined_data.shape}")

    # Perform clustering
    print(f"Clustering data into {args.n_clusters} clusters using Faiss (GPU: {args.gpu})...")
    kmeans = FaissKmeans(n_clusters=args.n_clusters, niter=20, gpu=args.gpu)
    cluster_centers, labels = kmeans.train(combined_data)

    # 保存聚类中心为 .pt 文件
    output_path = os.path.join(args.output_dir, f"{args.dataset_name}_C_{args.n_clusters}_init.pt")
    torch.save(cluster_centers, output_path)
    print(f"Saved cluster centers to: {output_path}")
    print("Clustering complete.")


if __name__ == "__main__":
    main()