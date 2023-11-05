

def find_leftmost_cluster(cluster_centers):
    """
    Find the index of the leftmost cluster based on the x-value of cluster centers.
    
    :param cluster_centers: List of coordinates of cluster centers.
    :return: Index of the leftmost cluster and its x-value.
    """
    min_x = float('inf')  # Initialize with infinity
    leftmost_cluster_index = -1  # Initialize with an invalid index
    
    # Iterate over cluster centers to find the minimum x-value
    for i, center in enumerate(cluster_centers):
        if center[0] < min_x:
            min_x = center[0]
            leftmost_cluster_index = i
            
    return leftmost_cluster_index, min_x