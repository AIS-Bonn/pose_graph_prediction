# Pose Graph Prediction

The purpose of this package is to predict humans represented by pose graphs.  
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

Figure 1: Pipeline overview.

In the following the pipeline is described in detail. 

<details>
<summary><b>Input Data</b></summary>

The input pose graphs are generated from images of multiple cameras observing the same humans.  
The [Human3.6M data set](http://vision.imar.ro/human3.6m/description.php) provides such data for single humans and ground truth pose annotations.  
The images from the data set are processed using the [SmartEdgeSensor3DHumanPose](https://github.com/AIS-Bonn/SmartEdgeSensor3DHumanPose) package to generate human pose graphs.

![Human pose graph projected into image from human3.6m](./docs/readme/human_pose_graph_projected_into_human36m_image.png)

Figure 2: Exemplary generated human pose graph projected into an image from the Human3.6M data set.
Image source: [Real-Time Multi-View 3D Human Pose Estimation using Semantic Feedback to Smart Edge Sensors](https://www.ais.uni-bonn.de/papers/RSS_2021_Bultmann.pdf)

</details>


<details>
<summary><b>Data Preprocessing</b></summary>

Currently, two sequential poses of the same human are used as the input.  
The poses are normalized with respect to the first pose.  
Meaning, a normalization transformation is computed for the first pose and applied to every pose in the sequence.  

The assumptions in our use case are that the prediction of a single human is independent of:
1) the global frame - e.g. a walking motion doesn't depend on the direction of walking or the exact spot in the room
2) the height of the human - a tall person's walking motion is very similar to the walking motion of a smaller person  
  
These assumptions and the corresponding normalization allow the network to concentrate on learning how to predict a motion.  
The network doesn't need to know how to predict the motion in every different setup for every different human.   

To address assumption 1, the normalization applies a translation and rotation is such a way that:
- the mid hip position is in the origin of the local coordinate frame
- and the front of the hip is facing in the x-direction (Fig. 3)
  - this is achieved by rotating the pose around the z-axis only

To address assumption 2, the height of the human is estimated by summing the bone lengths from one leg, the spine, the neck and the head.  
The pose is scaled by dividing it with the estimated height - resulting in a height of 1 (Fig. 3).  

![Visualization of a normalized pose](./docs/readme/normalization.png)

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
For that every joint becomes one a node in the graph.  
Every node n<sub>i</sub> from the previous pose at time step t-1 is connected to every node n<sub>j</sub> from the current pose at time step t via a directed edge e<sub>i,j</sub> (Fig. 4).  

![Conversion of the pose sequence to one graph for the GNN](./docs/readme/pose_sequence_to_graph_data_conversion.png)

Figure 4: A simplified example graph with only two joints per pose.

Every node and every edge can have an associated feature vector - e.g. joint position as node feature.  
The GNN's task will be to update the node positions from time step t to generate a prediction for time step t+1. 

---

#### Edge Update

The GNN updates the edge features first.  
Each edge is updated in the same way using the same set of MLPs.  
For every edge e<sub>i,j</sub>:
- Concatenate the features of the connected nodes n<sub>i</sub> and n<sub>j</sub> - e.g. positions - and the features of the edge - e.g. distance between nodes
- Use an MLP to estimate the effect *eff<sub>i,j</sub>* of the source node n<sub>i</sub> onto the target node n<sub>j</sub> 
- Use an MLP to estimate the extent of the effect *ext<sub>i,j</sub>* of the source node n<sub>i</sub> onto the target node n<sub>j</sub> 
- Multiply the effect and the extent to get the updated edge feature ???<sub>i,j</sub>

Example of the update process on edge e<sub>1,3</sub>:  

![Exemplary graph for the edge update of a GNN](./docs/readme/gnn_edge_update_example_graph.png)
![Edge update of a GNN](./docs/readme/gnn_edge_update.png) 

Figure 5: Visualization of the graph and the submodules involved in the edge update for edge e<sub>1,3</sub>.

---

#### Node Update

After updating all edge features the node features are updated in a similar fashion.  
Only nodes with incoming edges get updated.  
The main difference is that there can be a varying number of incoming edges to each node.  
This is addressed by an aggregation function combining the features of all incoming edges per node.  
Again, every node is updated in the same way using the same MLP.  

For every node n<sub>j</sub> from time step t:
- Sum up the updated feature vectors of every incoming edge - representing how other joints affect this joint
- Concatenate the summed feature vector with the feature vector of the node itself
- Use an MLP to estimate how the node should be updated
- Add the result to the node state from time step t to get the predicted state

Example of the update process on node n<sub>3</sub>:

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

The variants mostly differ in the way the data is presented to the GNN.  
Starting from the initial prototype, the input is adapted step by step to a homogeneous GNN - described above - that is compared to a heterogeneous GNN.  
The heterogeneous GNN differs by allowing to specify a type for each edge and node.  
Depending on the edge type a different set of MLPs is used and trained during the edge update.  
Edges of the same type use the same MLPs.  
The target node type defines the MLP used in the node update equivalently.  

__Model 1 - The Initial Prototype__

For the initial prototype each node feature vector consisted of the normalized 3D position of the corresponding joint and the specific node id, normalized to a range from -1.0 to 1.0.  
Each edge feature consisted of the difference between the normalized target and source joint positions.  

node_features<sub>j</sub> = [normalized(j), position<sub>j,t</sub>]  
edge_features<sub>i,j</sub> = [position<sub>j,t</sub> - position<sub>i,t-1</sub>]

The reasoning behind this is that the node feature encoder gets the option to encode the node positions depending on the node id.  
The edge feature encoder could transform the features into an equivalent latent space.  
The edge update gets the information about the connected nodes' ids, their positions and already their difference in the latent space.  
The node update gets the information about the node's id, its position and how the edges affect the node.  
Finally, the decoder's task is to transform from the latent to euclidean space. 

__Model 2 - The Corrected Prototype__

The corrected prototype differs to the initial prototype in the way the joint id is represented within the node features.  
Here, the joint id is encoded as a one hot vector.

node_features<sub>j</sub> = [one_hot_encoded(j), position<sub>j,t</sub>]  
edge_features<sub>i,j</sub> = [position<sub>j,t</sub> - position<sub>i,t-1</sub>]

The reasoning is the same as for the initial prototype. 

__Model 3 - The One Hot Encoded Edge Model__

For this model, the joint ids are encoded in the edge features.  
The combination of source and target node ids is encoded in a one hot vector.  
Each node feature consists of the corresponding joints' normalized 3D positions from time step t and t-1.  

node_features<sub>j</sub> = [position<sub>j,t-1</sub>, position<sub>j,t</sub>]  
edge_features<sub>i,j</sub> = [one_hot_encoded(i, j)]

The reasoning behind it is to separate the features by their domain.  
The node features are in the euclidean space and get encoded by the node feature encoder independently of their id.  
The edge encoder only gets the ids and could potentially steer the edge update depending on the ids.  
The node update on the other hand has to rely on the information from the updated edges, because it gets no information about the target node id from the encoded node features.  

A potential downside of this model is the combinatorial explosion of the joint id combinations encoded in the edges. 

__Model 4 - The No Initial Edge Features Model__

This model is similar to model 2.  
The difference is that the initial edge features are omitted and the node features are extended by the previous position of the corresponding joint.  
Therefore, the node features consist of the one hot encoded node id, the previous and the current position of the joint.  

node_features<sub>j</sub> = [one_hot_encoded(j), position<sub>j,t-1</sub>, position<sub>j,t</sub>]  
edge_features<sub>i,j</sub> = []

The reasoning here is that the model should easily be capable of computing the edge features of model 2 by itself in the edge update - if this would be of use.  
The input to the edge update still consists of the encoded ids and all positions of the source and target node.  
The potential upside to model 3 would be that the target node id is encoded in the node features.

__Model 5 - The Heterogeneous GNN Model__

This model is most closely comparable to model 3 and 4.  
Like in model 3, the initial edge features are empty.  
Like in model 4, the node features consist of the previous and current positions of the corresponding joint.  
The node ids define the node types.  
Similar to model three each node id combination defines an edge type. 

node_features<sub>j</sub> = [position<sub>j,t-1</sub>, position<sub>j,t</sub>]  
edge_features<sub>i,j</sub> = []

This way, each joint combination is processed by an own set of MLPs during the edge update.  
The MLPs have the same size but unique sets of weights.  
Similarly, each target node type gets an own MLP for the node update.  
Encoder and decoder are not affected by the types.  

The training and evaluation procedure are described in the following.  
The evaluation section contains the results of the models' comparison.  

</details>


<details>
<summary><b>Training</b></summary>

The models are trained on the Human3.6M data set.  
3D Poses are estimated on the image data.  
Noise is applied to the training data to prevent overfitting and increase the robustness of the model.  
The noise is added to the joint positions.  
The amount of noise is sampled randomly from a uniform distribution between -_n_ and _n_, with _n_ being 5% of the link length the joint is connected by.  
The noisy estimated poses serve as the input to the model.  
The Mean Squared Error (MSE) between the normalized predicted joint positions and the normalized ground truth is used as the loss.  
Adam is utilized as the optimizer with a learning rate of 0.001.

</details>


<details>
<summary><b>Evaluation</b></summary>

__Evaluation Procedure__

The training set of the Human3.6M data set is used for the evaluation.  
It consists of seven subjects (humans) performing the same set of actions.  

Leave-one-out cross-validation is utilized to evaluate the models.

In detail, the evaluation procedure performs the following steps:  
- For each subject s<sub>t</sub> used for testing:
  - For each other subject s<sub>v</sub> used for validation with v != t:
    - Initialize model weights randomly using [He initialization](https://openaccess.thecvf.com/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf)
    - For 50 epochs:
      - Train model variant on remaining subjects' data
      - Compute loss on withheld validation data from subject s<sub>v</sub>
      - Save model m<sub>best_on_v</sub> with best loss on validation data
    - Compute loss l<sub>best</sub> of m<sub>best_on_v</sub> on test data from subject s<sub>t</sub>
  - Calculate mean x??<sub>t</sub> of all l<sub>best</sub> for current test subject s<sub>t</sub>
- Calculate mean of all x??<sub>t</sub> to get the final score for the model variant

In other words train, validate and test on all possible combinations of the sequences and average the losses to get the score.

__Quantitative Results__


| Model | Score (lower is better) | Link to Commit |
|-------|------|---|
1 - The Initial Prototype | 0.000405 | 306ed84d1e3cd673c2c8803e224d98ee99701ee9
2 - The Corrected Prototype | 0.000542 | b5c4c542212a92b5aa1a9ae1cc4246da6e98d270
3 - The One Hot Encoded Edge Model | __0.000280__ | 5a6b53814d029b8150e7f9da2b482caf48673174
4 - The No Initial Edge Features Model | __0.000281__ | 7c338801102fb9b3597398dfc77d29ad6fdf739b
5 - The Heterogeneous GNN Model | 0.000548 | 8a0f108931a2751aa1366ecb0ccd0e440f5e8fe8

__Qualitative Results__

After the quantitative evaluation, the two best model variants - 3 and 4 - were trained for 2000 epochs.  
The data was split into a test set - consisting of subjects 9 and 11 - and a training set using data from the remaining subjects.  
This split is commonly used in the literature for training on the publicly available part of the Human3.6M data set.  

Model 4 achieved a lower loss of 0.000344 on the test set compared to model 3 with a loss of 0.000350.  
Model 4 was picked as the currently best model variant.  
It was trained further for a total of 8000 epochs, achieving its best loss of 0.000340.

The larger loss after training - compared to the one of the evaluation - results from the used data split.  
During evaluation all model variants achieved their largest loss on data of subject 9.  
This data contributes more to the loss of the test set during training, than to the overall mean loss during the evaluation.  
The following table shows an example of the mean losses per validation set generated during the evaluation of model 4.  

| Subject ID | Mean Loss on Validation Set |
|---|---|
1  | 0.000187
5  | 0.000288
6  | 0.000257
7  | 0.000273
8  | 0.000157
9  | 0.000582
11 | 0.000222

These [videos](https://drive.google.com/drive/folders/1Q9_9vGsIXRlS56VyWFc4pedRydc4iJ2g?usp=sharing) visualize the results of the initial prototype and best model.  
Frame by frame predictions use the estimated poses as the input and compute only the next pose visualized in green.  
The ground truth is visualized in blue.  
Sequential predictions use the output pose of the model as the next input pose.  
The estimated seed poses are visualized in grey while the predictions are visualized in green.  
The model was not trained to use its own output as an input.  

The approximate inference time for one pose are three to five milliseconds on an Intel i7-8550U Laptop CPU.  

</details>


<details>
<summary><b>Installation</b></summary>

#### Create virtual environment  

`virtualenv --python=python3.6 ./pose_graph_env`
  
#### Activate virtual environment by default  

Add this - or adapted line - to .bashrc and deactivate other virtual environments being activated by default  

`source /path_to_env/pose_graph_env/bin/activate`

Then source the bashrc to activate the virtual environment 

`source ~/.bashrc`

#### Clone repo 

`cd /path_to_env/pose_graph_env/`  
`git clone https://git.ais.uni-bonn.de/pose-graph-prediction/pose-graph-prediction.git`
  
#### Install requirements

`cd pose-graph-prediction`  
`pip install -r requirements.txt`
  
#### Install this package locally

`python -m pip install -e .`
  
#### Install pytorch geometric

Pytorch geometric refused to be installed properly using the requirements.txt.  
In any case, install the correct version for your cuda and torch setup.  
The official [install guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-via-pip-wheels) describes how.  
E.g. for torch version 1.7.0 and CUDA version 10.2, which were used while developing this package:

```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install torch-geometric
```

Tested also with torch version 1.10.0 and torch-geometric version 2.0.2. 
  
#### Download data from sciebo 

```
mkdir data
cd data 
wget --no-check-certificate --content-disposition "https://uni-bonn.sciebo.de/s/zOx3LNDhoxMzOsj/download"
tar -xf original.tar.gz
mkdir original
mv k* original/
```

</details>
