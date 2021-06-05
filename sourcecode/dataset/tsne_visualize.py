import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
X = []

# glyph_file = 'data/glyph_onehot.txt'
# compress_out_file = 'data/glyph_tsne.csv'
glyph_file = 'data/component_vector.txt'
compress_out_file = 'data/component_tsne.csv'
n_clusters= 20

with open(glyph_file, 'r') as feature:
    for line in feature.readlines():
        line = line.split(" ")
        
        line = line[1:6]
        for i in range(len(line)):
            line[i] = int(line[i])
        X.append(line)
X = np.array(X)
print(X.shape)
X_embedded = TSNE(n_components=2).fit_transform(X)

with open(compress_out_file, 'w+') as out_file:
    for i in range(len(X_embedded)):
        out_file.write(str(X_embedded[i][0])+','+str(X_embedded[i][1])+ '\n')

print(X_embedded, X_embedded.shape)
step_num = n_clusters
start = [253,238,144]
end = [199,80,115]
step_R = (start[0] - end[0]) / step_num
step_G = (start[1] - end[1]) / step_num
step_B = (start[2] - end[2]) / step_num
step = [step_R, step_G, step_B]

kmeans = KMeans(n_clusters= n_clusters)
label = kmeans.fit_predict(X_embedded)
u_labels = np.unique(label)
for i in u_labels:
    color = tuple([(end[k] + step[k]*i)/255 for k in range(3)])
    plt.scatter(X_embedded[label == i , 0] , X_embedded[label == i , 1] , s=1.5, label = i, c=[color], alpha=0.7)
plt.show()