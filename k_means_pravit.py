
def k_means_gray(image, K):
    (x, y) = image.shape
    x_one_d = image.copy().ravel()
    #Select 3 Centroids at random...
    centroids = image.copy().ravel()
    np.random.shuffle(centroids)
    centroids = centroids[:K]
    iterations = 0
    old_centroids = np.zeros(centroids.shape)
    ep = np.mean(abs(old_centroids - centroids))
    while(iterations < 50 and  ep > 0.2):
        print("For K= {} Iteration: {}".format(K,iterations))
        old_centroids = centroids.copy()
        iterations+=1
        closest_cent = compute_closest_centroids_g(image, centroids)
        new_centroids = []
        #Find New Centroids i.e. Mean of corresponding Cluster points:
        for i in range(K):
            ind_centroids = (closest_cent == i)
            points_closest_to_cluster = x_one_d[ind_centroids]
            average = points_closest_to_cluster.mean(axis=0)
            new_centroids.append(average)
        centroids = np.array(new_centroids)
        ep = np.mean(abs(old_centroids - centroids))
        print(ep)
        cluster_assignment = np.zeros(x_one_d.shape)
        
        for i in range(len(x_one_d)):
            cluster_assignment[i] = centroids[int(closest_cent[i])]
        cluster_assignment = cluster_assignment.reshape(image.shape)

    cv2.imwrite("Segmented K=5 Col Image.jpg", cluster_assignment)
    return cluster_assignment   

def k_means_color(image, K):
    (x, y, z) = image.shape
    #Select 3 Centroids at random...
    centroids = image.copy().reshape(x*y, z)
    np.random.shuffle(centroids)
    centroids = centroids[:K]
    iterations = 0
    old_centroids = np.zeros(centroids.shape)
    ep = np.mean(abs(old_centroids - centroids))
    print(ep)
    while(iterations < 40 and  ep > 0.3):
        print("Iteration: {}".format(iterations))
        old_centroids = centroids.copy()
        iterations+=1
        closest_cent =  compute_closest_centroids(image, centroids)
        print("Clossssses",closest_cent.shape)
        new_centroids = []
        #Find New Centroids i.e. Mean of corresponding Cluster points:
        for i in range(K):
            ind_centroids = (closest_cent == i)
            points_closest_to_cluster = image[ind_centroids]
            average = points_closest_to_cluster.mean(axis=0)
            new_centroids.append(average)
        centroids = np.array(new_centroids)
        ep = np.mean(abs(old_centroids - centroids))
        print(ep)
        cluster_assignment = np.zeros(image.shape)
        (m,n) = closest_cent.shape
        for i in range(m):
            for j in range(n):
                cluster_assignment[i][j] = centroids[closest_cent[i][j]] 

    cv2.imwrite("Segmented_K=9_Col_Image.jpg", cluster_assignment)
    return cluster_assignment   

def compute_closest_centroids( im, centroids ):

    differences = im - centroids[:, np.newaxis, np.newaxis]
    squareDifference = differences ** 2
    summedSquaredDifferences = squareDifference.sum(axis = 3)
    finalDistances = np.sqrt(summedSquaredDifferences)
    return np.argmin(finalDistances, axis=0)  

def compute_closest_centroids_g( im, centroids ):
    k = im.ravel()
    differences  = np.zeros(k.shape)
    for pt in range(k.shape[0]):
        differences[pt] = abs(k[pt] - centroids). argmin()
    return  differences



def part_1():
    img1 = cv2.imread('bird_col.jpg')
    img2 = cv2.imread('dog.jpg',0)
    K = 5
    k_means_color(img1, K)
    k_means_gray(img2,K)
    histr = []
    color_image = cv2.imread('bird_col.jpg')
    for channel_idx in range(3):
        hist_image = cv2.calcHist([color_image],[channel_idx],None,[128],[0,256])
        histr.append(hist_image)
    hist_array = np.asarray(histr).ravel()
    print(hist_array.shape)
    fig = plt.figure(2)
    plt.subplot(1,2,1)
    plt.imshow(color_image)
    plt.xlabel("Bird Color Image")
    plt.subplot(1,2,2)
    plt.plot(hist_array)
    plt.xlabel("R:0-127 G:128-255 B:256-380")
    plt.ylabel("Pixels")
    fig.tight_layout()
    plt.show()