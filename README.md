# presentation-sarita
presentation-sarita created by GitHub Classroom
# Kohonen Network (Self-Organizing Map, SOM)

## Table of Contents

1. [Introduction](#introduction)
2. [History](#history)
3. [Significance](#significance)
4. [Architecture / Diagram / Network](#architecture--diagram--network) 
5. [Mathematical Model](#mathematical-model)
6. [Algorithm](#algorithm)
7. [Source Code / Example](#source-code--example)
8. [Application Areas / Real World Example](#application-areas--real-world-example)
9. [References](#references)

---

## Introduction
The Kohonen Network, also known as the Self-Organizing Map (SOM), is a type of artificial neural network developed by Professor Teuvo Kohonen in the 1980s. It is a powerful tool for visualizing and interpreting high-dimensional data by mapping it onto a lower-dimensional grid, typically two-dimensional. Unlike traditional neural networks, SOMs use unsupervised learning to categorize and cluster input data based on its inherent similarities. This makes them particularly useful for tasks such as data mining, pattern recognition, and feature extraction, where the goal is to discover underlying structures and relationships within the data without prior labeling.
 It follows an unsupervised learning approach and trained its network through a competitive learning algorithm. SOM is used for clustering and mapping (or dimensionality reduction) techniques to map multidimensional data onto lower-dimensional which allows people to reduce complex problems for easy interpretation. SOM has two layers, one is the Input layer and the other one is the Output layer. 

The architecture of the Self Organizing Map with two clusters and n input features of any sample is given below: 

![SOM](https://github.com/Nepal-College-of-Information-Technology/lab-works-saru12ita/assets/136351906/fdb77d9e-ea33-4163-883f-4e078877c19d)


## History

 The SOM algorithm grew out of early [neural network](http://www.scholarpedia.org/w/index.php?title=Neural_network&action=edit&redlink=1) models, especially models of [associative memory](http://www.scholarpedia.org/w/index.php?title=Associative_memory&action=edit&redlink=1) and [adaptive learning](http://www.scholarpedia.org/w/index.php?title=Adaptive_learning&action=edit&redlink=1) (cf. Kohonen 1984). A new incentive was to explain the spatial organization of the [brain](http://www.scholarpedia.org/article/Brain)'s functions, as observed especially in the cerebral [cortex](http://www.scholarpedia.org/w/index.php?title=Cortex&action=edit&redlink=1). Nonetheless the SOM was not the first step in that direction: one has to mention at least the spatially ordered line detectors of von der Malsburg (1973) and the [neural field](http://www.scholarpedia.org/w/index.php?title=Neural_field&action=edit&redlink=1) model of Amari (1980). However, the [self-organizing](http://www.scholarpedia.org/article/Self-organization) power of these early models was rather weak. The crucial invention of Kohonen was to introduce a system model that is composed of at least two interacting subsystems of different natures. One of these subsystems is a competitive [neural](http://www.scholarpedia.org/article/Neuron) network that implements the winner-take-all function, but there is also another subsystem that is controlled by the neural network and which modifies the local [synaptic plasticity](http://www.scholarpedia.org/w/index.php?title=Synaptic_plasticity&action=edit&redlink=1) of the neurons in learning. The learning is restricted spatially to the local neighborhood of the most active neurons. The plasticity-control subsystem could be based on nonspecific neural interactions, but more probably it is a chemical control effect. Only by means of the separation of the neural signal transfer and the plasticity control has it become possible to implement an effective and robust [self-organizing](http://www.scholarpedia.org/w/index.php?title=Self_Organization&action=edit&redlink=1) system.
Nonetheless, the SOM principle can also be expressed mathematically in a pure abstract form, without reference to any underlying neural or other components. The first application area of the SOM was [speech recognition](http://www.scholarpedia.org/w/index.php?title=Speech_recognition&action=edit&redlink=1) (see Figure [2](http://www.scholarpedia.org/article/Kohonen_network#fig:Phonemesom.png)). In its abstract form, the SOM has come into widespread use in data analysis and data exploration (Kaski et al. 1998, Oja et al. 2003, Pöllä et al. 2007).

###  Significance

**Data Visualization:** Self-Organizing Maps (SOMs) are invaluable for transforming high-dimensional data into more manageable, low-dimensional formats. This transformation allows researchers and analysts to visualize complex datasets in two-dimensional or three-dimensional spaces, making it easier to spot trends, clusters, and outliers. For example, in fields like genomics or customer segmentation, SOMs can help identify distinct groups or patterns that might not be immediately apparent in the raw, high-dimensional data.


**Unsupervised Learning:** One of the key strengths of SOMs is their ability to perform unsupervised learning. This means they can learn to categorize and cluster data based purely on the input data's structure, without requiring pre-labeled examples. This capability is particularly useful in exploratory data analysis and situations where labeled data is scarce or expensive to obtain. SOMs can autonomously discover underlying patterns and group similar data points together, providing insights that might be missed using traditional supervised methods.

**Pattern Recognition:** SOMs excel at recognizing patterns within data. By organizing input data into a structured map, they can highlight relationships and regularities. This makes them highly effective in applications such as image and speech recognition, where identifying consistent patterns within the data is crucial. For instance, in image recognition tasks, SOMs can group together similar visual features, helping to classify and identify objects or scenes within images.




**Feature Extraction:** In the process of mapping high-dimensional input data onto a lower-dimensional grid, SOMs naturally highlight the most relevant features of the data. This automatic feature extraction is beneficial in preprocessing stages of machine learning pipelines, where reducing the complexity of the data can improve the performance of subsequent models. By focusing on the most significant features, SOMs help in simplifying data without losing essential information.


**Dimensionality Reduction:** SOMs are highly effective tools for dimensionality reduction, a process that reduces the number of variables under consideration and can help in removing noise from the data. By projecting data onto a lower-dimensional space, SOMs can enhance the clarity and focus of the data, making it easier to analyze and interpret. This reduction is particularly useful in large datasets where many features may be irrelevant or redundant.



**Clustering:** The clustering capability of SOMs is one of their primary applications. They group similar data points together based on their inherent properties, facilitating the identification of natural clusters within the data. This is particularly useful in market segmentation, customer behavior analysis, and any other application where understanding groupings within the data is important. The clusters identified by SOMs can provide valuable insights into the structure and composition of the data.



**Versatility:** SOMs are versatile tools that have been successfully applied across a wide range of domains. In bioinformatics, they can be used to cluster genes or proteins with similar expressions. In finance, they can identify patterns in market data or group similar investment portfolios. In marketing, SOMs help segment customers based on purchasing behavior. Their broad applicability across different fields highlights their adaptability and utility in various types of data analysis tasks.




**Scalability:** SOMs are designed to handle large datasets effectively, making them suitable for real-world applications where data volume can be substantial. Their algorithmic efficiency allows them to scale with the size of the data, maintaining performance and providing meaningful insights even as the dataset grows. This scalability ensures that SOMs remain relevant and useful in modern data-intensive environments.




**Robustness:** The robustness of SOMs to noisy and incomplete data is another significant advantage. They can still perform well even when the input data is not perfect, making them reliable in practical scenarios where data quality might be an issue. This robustness ensures that SOMs can provide consistent and accurate results, enhancing their utility in a wide range of applications.


## Architecture / Diagram / Network
![Kohonen-network-architecture](https://github.com/Nepal-College-of-Information-Technology/lab-works-saru12ita/assets/136351906/21893a23-bef9-4a5b-a0f0-ef2094ba27df)

The architecture of a Kohonen Network, or Self-Organizing Map (SOM), typically consists of an input layer and a competitive layer, as depicted in the diagram above. The input layer receives the input data, which can be high-dimensional. Each input is connected to all the neurons in the competitive layer through weighted connections. The competitive layer, often arranged in a two-dimensional grid, is where the self-organization process occurs.

In this figure, we see that each neuron in the competitive layer is connected to every input node, illustrating the fully connected nature of the network. The neurons in the competitive layer compete to become the winning neuron for each input pattern. This competition is a fundamental aspect of the SOM, where only one neuron, or a small group of neurons, is activated for a given input, highlighting the best-matching unit (BMU).

The process of self-organization involves adjusting the weights of the connections based on the input data, gradually forming a structured map where similar inputs activate neighboring neurons. This results in a topological organization that preserves the spatial relationships of the input data, making SOMs useful for tasks like clustering and visualization. The arrows in the diagram indicate the direction of signal flow from the input layer to the competitive layer, demonstrating the interaction between the layers in forming the organized map.


## Mathematical Model
![TrainSOM](https://github.com/Nepal-College-of-Information-Technology/lab-works-saru12ita/assets/136351906/e0e6837c-b9e0-4056-af25-9f349b2a2e95)


**How do SOM works?**
Let’s say an input data of size (m, n) where m is the number of training examples and n is the number of features in each example. First, it initializes the weights of size (n, C) where C is the number of clusters. Then iterating over the input data, for each training example, it updates the winning vector (weight vector with the shortest distance (e.g Euclidean distance) from training example). Weight updation rule is given by : 

wij = wij(old) + alpha(t) *  (xik - wij(old))
where alpha is a learning rate at time t, j denotes the winning vector, i denotes the ith feature of training example and k denotes the kth training example from the input data. After training the SOM network, trained weights are used for clustering new examples. A new example falls in the cluster of winning vectors. 
![SOM](https://github.com/Nepal-College-of-Information-Technology/lab-works-saru12ita/assets/136351906/9bc4e548-31d8-4218-80a1-ed7aa1710e75)

**Algorithm
Training:**

Step 1: Initialize the weights wij random value may be assumed. Initialize the learning rate α.

Step 2: Calculate squared Euclidean distance.

                    D(j) = Σ (wij – xi)^2    where i=1 to n and j=1 to m

Step 3: Find index J, when D(j) is minimum that will be considered as winning index.

Step 4: For each j within a specific neighborhood of j and for all i, calculate the new weight.

                   wij(new)=wij(old) + α[xi – wij(old)]

Step 5: Update the learning rule by using :

                   α(t+1) = 0.5 * t

Step 6: Test the Stopping Condition.

### Numerical Example
![SOM1 drawio](https://github.com/Nepal-College-of-Information-Technology/lab-works-saru12ita/assets/136351906/4d6f33ba-0638-4c33-a4cb-da4cbb5e8aa8)

Q.construct KSOM to Cluster four given vectors.
X=>[0 0 1 1]
  =>[1 0 0 0]
  =>[0 1 1 0]
  =>[0 0 0 1]
no. of clusters to be formed are 2 .assumed initial learning rate as 0.5.
solution:
no .of input vectors (x)=4
no. of clusters (xi) =2
i= 1 to 4 and j= 1 to 2
initialize weight randomly between o & 1
![mat](https://github.com/Nepal-College-of-Information-Technology/lab-works-saru12ita/assets/136351906/d1db2b69-3b4a-43c6-b7ef-4f59b6c7e9d0)
first input vector: X=[0 0 1 1 ]
calculate Euclidian distance
![formu](https://github.com/Nepal-College-of-Information-Technology/lab-works-saru12ita/assets/136351906/597e388a-f941-4864-8344-17226acab772)
D(1)=(x1-w11)^2 +(x2-w21)^2 +(x3-w31)^2 +(x4-w41)^2
D(1)=(0-0.2)^2+(0-0.4)^2+(1-0.6)^2+(1-0.8)^2
D(1)=0.4
Similarly,
D(2)=(x1-w12)^2 +(x2-w22)^2 +(x3-w32)^2 +(x4-w42)^2
D(2)=(0-0.9)^2+(0-0.7)^2+(1-0.5)^2+(1-0.3)^2
D(2)=2.04
here, D(1)<D(2); therefore , wining cluster is j=1 i.e. Y1

Update weights of wining clusters i.e. j=1
![wincluster](https://github.com/Nepal-College-of-Information-Technology/lab-works-saru12ita/assets/136351906/4c4e18a6-69b2-4504-b3c0-6de634b6f084)
![updatew](https://github.com/Nepal-College-of-Information-Technology/lab-works-saru12ita/assets/136351906/270bfa06-fa32-4b0d-921a-a6c8a3212004)
![nwe](https://github.com/Nepal-College-of-Information-Technology/lab-works-saru12ita/assets/136351906/9b3086e8-a41b-48fe-a323-e301f70293f0)

similarly solve all the iterations and update the value of learning rate as:
alpha(t+1)=0.5 * alpha(t)
alpha(0+1)=0.5*alpha(0)       [since, alpha(0)=0.5]
alpha(1)=0.5*0.5=0.25

hence, with this learning rate we can proceed feature up to 100 iterations or till weight matrix reduces to very negligible value.

**Below is the implementation of the above approach: 
### Source Code

import math


**class SOM:**

	# Function here computes the winning vector
	# by Euclidean distance
	def winner(self, weights, sample):

		D0 = 0
		D1 = 0

		for i in range(len(sample)):

			D0 = D0 + math.pow((sample[i] - weights[0][i]), 2)
			D1 = D1 + math.pow((sample[i] - weights[1][i]), 2)

		# Selecting the cluster with smallest distance as winning cluster

		if D0 < D1:
			return 0
		else:
			return 1

	# Function here updates the winning vector
	def update(self, weights, sample, J, alpha):
		# Here iterating over the weights of winning cluster and modifying them
		for i in range(len(weights[0])):
			weights[J][i] = weights[J][i] + alpha * (sample[i] - weights[J][i])

		return weights

###  Driver code


def main():

	# Training Examples ( m, n )
	T = [[1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1]]

	m, n = len(T), len(T[0])

	# weight initialization ( n, C )
	weights = [[0.2, 0.6, 0.5, 0.9], [0.8, 0.4, 0.7, 0.3]]

	# training
	ob = SOM()

	epochs = 3
	alpha = 0.5

	for i in range(epochs):
		for j in range(m):

			# training sample
			sample = T[j]

			# Compute winner vector
			J = ob.winner(weights, sample)

			# Update winning vector
			weights = ob.update(weights, sample, J, alpha)

	# classify test sample
	s = [0, 0, 0, 1]
	J = ob.winner(weights, s)

	print("Test Sample s belongs to Cluster : ", J)
	print("Trained weights : ", weights)


if __name__ == "__main__":
	main()




## Application Areas / Real World Example

**Application Areas**

- Data Mining
- Pattern Recognition
- Image Processing
- Speech Recognition
- Market Segmentation
- Bioinformatics
- Finance
- Robotics
- Telecommunications
- Environmental Monitoring

**Real World Examples**
**Customer Segmentation in Marketing:** Companies use SOMs to segment customers based on purchasing behavior and demographics, enabling targeted marketing strategies and personalized promotions.

**Genomics in Bioinformatics:** SOMs help cluster genes or proteins with similar expression patterns, aiding in the identification of functional relationships and genetic markers.

**Image Compression:** SOMs can be used to reduce the size of image data while preserving important features, making storage and transmission more efficient.

**Speech Recognition:** SOMs assist in clustering phonemes and recognizing patterns in speech data, improving the accuracy of speech recognition systems.

**Financial Market Analysis:** Analysts use SOMs to identify patterns and trends in stock market data, helping in portfolio management and investment decision-making.

**Anomaly Detection in Telecommunications:** SOMs are employed to detect unusual patterns in network traffic, identifying potential security threats and network failures.

**Robotic Navigation:** SOMs enable robots to map and navigate their environment by organizing sensory inputs, facilitating tasks like pathfinding and obstacle avoidance.

**Environmental Monitoring:** SOMs are used to analyze environmental data, such as climate patterns or pollution levels, helping in the assessment and management of environmental issues.

**Medical Diagnosis:** In healthcare, SOMs assist in clustering patient data and identifying patterns that can lead to early diagnosis and personalized treatment plans.


## References

For more detailed information about Self-Organizing Maps (SOMs) and Kohonen Networks, you can refer to the following resource:

GeeksforGeeks: [Self Organizing Maps (Kohonen Maps)](https://www.geeksforgeeks.org/self-organising-maps-kohonen-maps/)
This comprehensive guide provides insights into the concepts, architecture, and applications of SOMs, offering a deeper understanding of how they function and their significance in various fields.
