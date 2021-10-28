# Pose Graph Tracking

The purpose of this package is to track humans being represented by pose graphs.  
In its current state, it predicts the next pose of a human given the current and the previous pose.  
This procedure can be applied sequentially to predict the poses following the next pose.  

The prediction step is planned to be part of a [Multi Object Tracking (MOT) pipeline](https://github.com/AIS-Bonn/multi_hypothesis_tracking).  
The MOT pipeline consists of:
1) a prediction step per tracked object 
2) an association step to correct the predictions using current data.
---
## Current State

The package consists of a Graph Neural Network (GNN) utilizing the previous and the current human pose graphs to predict the next human pose graph.  

![Overview of the pipeline](./docs/readme/pipeline_overview.png)

In the following the pipeline is described in detail. 

<details>
<summary><b>Input Data</b></summary>

The input pose graphs are generated from images of multiple cameras observing the same humans.  
The [Human3.6M data set](http://vision.imar.ro/human3.6m/description.php) provides such data and ground truth pose annotations.  
The images from the data set are processed using the [SmartEdgeSensor3DHumanPose](https://github.com/AIS-Bonn/SmartEdgeSensor3DHumanPose) package to generate human pose graphs.  

![Human pose graph projected into image from human3.6m](./docs/human_pose_graph_projected_into_human36m_image.png)

Figure 2: Exemplary generated human pose graph projected into an image from the Human3.6M data set.
Image source: [Real-Time Multi-View 3D Human Pose Estimation using Semantic Feedback to Smart Edge Sensors](https://www.ais.uni-bonn.de/papers/RSS_2021_Bultmann.pdf)

</details>


<details>
<summary><b>Data Preprocessing</b></summary>

Currently, two sequential poses of the same human are used as the input.  
The poses are normalized with respect to the first pose.  
Meaning, a normalization transformation is computed for the first pose and applied to every pose in the sequence.  

The assumptions in our use case are that the prediction of a single human is independent of:
- the global frame
  - e.g. a walking motion doesn't depend on the direction of walking or the exact spot in the room
- the height of the human
  - a tall person's walking motion is very similar to the walking motion of a smaller person  
  
These assumptions and the corresponding normalization allow the network to concentrate on learning how to predict a motion.  
The network doesn't need to know how predict the motion in every different setup for every human.   

The normalization applies a translation and rotation is such a way that:
- the mid hip position is in the origin of the local coordinate frame
  - that way all joint positions are relative to the mid hip joint, which lays in the middle of the right and left hip
- the human is facing in the x-direction of the local coordinate frame
  - this is achieved by rotating the pose around the z-axis of the local coordinate frame

![Visualization of a normalized pose](./docs/readme/normalization.png)
Additionally, the height of the human is estimated by summing the bone lengths from the one leg, the spine, the neck and the head.  
The pose is scaled by dividing it with the estimated height.  

Figure 3: Visualization of a normalized pose.

</details>


<details>
<summary><b>Graph Neural Network</b></summary>

---

#### Graph Neural Network in General

Remark: If you are new to Graph Neural Networks a good introduction can be found on [distill.pub](https://distill.pub/2021/gnn-intro/).

The task of the Graph Neural Network (GNN)  in our use case is to predict the next pose of a human given the previous poses.  
The assumption is that every joint can have an effect on every other joint.  
E.g. while walking, the right hand does not only move with respect to the right elbow but is also moving antagonistically to the left hand.  

--- 

#### Constructing the Input Graph

A GNN can only modify the node and edge features of one graph.  
Thus, the first thing to do is to construct one graph from the sequence of poses.  
For that every joint becomes one a node n<sub>i</sub> in the graph.  
To incorporate the assumption, every node from the previous pose at time step t-1 is connected to every node from the current pose at time step t via a directed edge e<sub>i,j</sub>.  
Every node and every edge can have an associated feature vector - e.g. joint position as node feature. 

![Conversion of the pose sequence to one graph for the GNN](./docs/readme/pose_sequence_to_graph_data_conversion.png)

Figure 4: A simplified example graph with only two joints per pose.

The GNN updates only features of nodes with incoming edges.  
It's task will be to update the node positions from time step t to generate a prediction for time step t+1. 

---

#### Edge Update

The GNN updates the edge features first.  
Each edge is updated in the same way using the same set of MLPs.  
For every edge e<sub>i,j</sub>:
- Concatenate the features of the connected nodes - e.g. positions - and the features of the edge - e.g. distance between nodes
- Use an MLP to estimate the effect *eff<sub>i,j</sub>* of the source node i onto the target node j 
- Use an MLP to estimate the extent of the effect *ext<sub>i,j</sub>* of the source node i onto the target node j 
- Multiply the effect and the extent to get the updated edge feature ẽ<sub>i,j</sub>

Exemplary for edge e<sub>1,3</sub>: 

![Exemplary graph for the edge update of a GNN](./docs/readme/gnn_edge_update_example_graph.png)
![Edge update of a GNN](./docs/readme/gnn_edge_update.png) 

Figure 5: Visualization of the graph and the submodules involved in the edge update for edge e<sub>1,3</sub>.

---

#### Node Update

After updating the edge features the node features are updated in a similar fashion.  
The main difference is that there can be a varying number of incoming edges to each node.  
This is addressed by an aggregation function combining the features of all incoming edges per node.  
Again, every node is updated in the same way using the same set of MLPs.  

For every node n<sub>j</sub> from time step t:
- Sum up the feature vectors of every incoming edge - representing how other joints affect this joint
- Concatenate the summed feature vector with the feature vector of the node itself
- Use an MLP to estimate how the node should be updated
- Add the result to the node state from time step t to get the predicted state

Exemplary for node n<sub>3</sub>:

![Exemplary graph for the node update of a GNN](./docs/readme/gnn_node_update.png)
![Node update of a GNN](./docs/readme/gnn_node_update_example_graph.png)

Figure 6: Visualization of the graph and the submodules involved in the node update for node n<sub>3</sub>.

</details>


<details>
<summary><b>Model Variants</b></summary>

After describing the utilized Graph Neural Network (GNN) in general, several tested variants are described in the following.  
For all variants, each node feature is encoded using the same MLP before being passed into the GNN.  
The same is done with the edge features using a different MLP.  
We noticed that the GNN is performing better with encoded features.  
After predicting the node features of the next time step an MLP is used on the node features to decode them back into the euclidean space.  

TODO: Describe model variants

</details>


<details>
<summary><b>Training</b></summary>

The models are trained on the Human3.6M data set.  
Poses are estimated on the image data.  
Noise is applied to the training data to prevent overfitting and increase the robustness of the model.  
The noise is applies to the joint positions.  
The amount of noise is sampled randomly between -n and n, with n being 5% of the link length the joint is connected by.  
The noisy estimated poses serve as the input to the model.  
The Mean Squared Error (MSE) between the normalized predicted joint positions and the normalized ground truth is used as the loss.  
Adam is utilized as the optimizer with a learning rate of 0.001.

</details>


<details>
<summary><b>Intermediate Results</b></summary>

TODO: insert gifs for qualitative results 

TODO: describe evaluation and evaluated model variants - insert schematic image per variant 

TODO: add links to branches containing the variants

</details>
