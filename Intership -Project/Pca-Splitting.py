#splitting into training and testing set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 42)

# Compute a PCA (eigenfaces) on the e face face dataset data ataset (treated as unlabeled #dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

print("Extracting the top %d eigenfaces from %d faces"% (n_components, x_train.shape[0]))

#Applying PCA
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(x_train)

#Generating eigenfaces
eigenfaces=pca.components_.reshape((n_components, h, w))

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()

print("Projecting the input data on the eigenfaces orthonormal basis")
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)
print(x_train_pca.shape,x_test_pca.shape)
#Compute Fisherfaces
lda = LinearDiscriminantAnalysis()
#Compute LDA of reduced data
lda.fit(x_train_pca, y_train)

x_train_lda = lda.transform(x_train_pca)
x_test_lda = lda.transform(x_test_pca)

print("Project done...")


