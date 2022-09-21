# Social-Recommendation
Models for social recommendation
*****
## 1. Graphrec

   A PyTorch implementation of the GraphRec model in Graph Neural Networks for Social Recommendation (Fan, Wenqi, et al. "Graph Neural Networks for Social Recommendation." The World Wide Web Conference. ACM, 2019).
    
   This code is modified on the basis of GraphRec_PyTorch (https://github.com/Wang-Shuo/GraphRec_PyTorch), so as to reproduce the results in the paper. 
    
   **Datasets**: Epinions and Ciao. 
    
   **preprocess_filter5.py**: users with more than 5 ratings are kept, as well as the corresponding items. 80%, 10%, 10% for training, validation and testing, respectively.
    
   ```  
   python preprocessing_filter5.py
   python main.py
   ```    
    
    
   **Warming**: 
    In data preprocessing, users or items in valid set or test set may not be appeared in training set and this case impacts the accuracy. So I'm not sure whether these users and items should be filtered in valid set and test set.


   **Comments**：I recommend you researchers should be careful with the GraphRec's performance, since although the performance is reproducted with some additional skills, it's still much worse than TrustSVD. So I hope all you guys can use TrustSVD as one of the baselines for Social Recommendation problem.


## 2. TrustSVD
  
  I find that TrustSVD is also really a strong baseline in social recommendation.
  
  So I add the config file with implementation of Librec.
  

## 3. S4Rec: Semantic and Structural view  Fusion Modeling  for Social Recommendation

  Recently we have proposed a new GNN-based framework **S4Rec** for rating prediction task in social recommendation. The paper is still under review.
  
  The framework is a combination of a GNN-based deep model and a wide shallow model (TrustSVD, TrustMF, SocialMF, etc.), and extensive experiments on three public datasets, Epinions, Ciao and yelp have demonstrated the effectiveness of the framework. 
  
  The source code is available in the file [S4Rec](https://github.com/lcwy220/Social-Recommendation/tree/master/S4Rec). The implementation details of the framework are shown as follows.
  
  ***
  
  The implementation of S4Rec consists of 4 steps: 
  
      1. preprocessing data 
      2. running deep graph model
      3. running wide shallow model 
      4. the final prediction fusion
  
  ### 3.1 Preprocessing data
  
  First, we need to preprocess the data with `preprocessing_filter5.py`. We retain the users with more than 5 ratings, and all items clicked by these retained users are also kept. 
  You can revise the **dataset_name** to test different dataset.
  
  Then, we extend more implicit relations with collective intelligence based strategy with `generate_implicit_relations.py`.
  
  
  ```
  cd dataset
  python preprocessing_filter5.py
  python generate_implicit_relations.py
  ```
  
  
  ### 3.2 Running deep graph model
  
  Before running the deep graph model, we need to pretrain the relational triplet constraint with TransH to obtain user and item embeddings.
  
      1. Switch to `S4Rec/tranh`, run `cross_sampling.py` to obtain triplets.
  
      2. Then, execute `pretrain_tranh.py` to obtain the pretrained user and item embeddings.
  
  
  Then, we switch to the upper data path `S4Rec` and execute `Main_DeppGraph.py`. When the training process is finished, we need to set `args.test=1` so that we can obtain the 'GNN_test.txt' and `GNN_vaild.txt`. Or we can directly use the file `test_best_predict_list.txt`.
  
  **Note: the parameter dataset_name needs revision.**
  
      1. python Main_DeppGraph.py
      
      2. set `args.test=1`, and python Main_S4Rec.py
      
  
  ### 3.3 Running wide shallow model
  
  Third, unzip the `librec-3.0.0.rar`, and copy the `dataset_name/new_*_set_filter5.txt` and 'trust_data.txt' to `librec-3.0.0/data/dataset_name`.
  
  The conf file of Librec is: `librec-3.0.0/core/target/classes/rec/context/rating/trustsvd-test.properties`. In this file, we can designate the input data dir and output result dir. You can revise it as you need.
  
  In windows, we can execute the java file: `librec-3.0.0/core/src/test/java/net/librec/MainTest.java` to obtain the trustsvd prediction. This java file uses the trustsvd-test.properties as the conf file.
  
  Takes Ciao as an example (run in Windows):
  
  ```
  unzip librec-3.0.0.rar
  cd dataset/Ciao
  cp new_* ../../librec-3.0.0/data/Ciao
  cp trust_data.txt ../../librec-3.0.0/data/Ciao
  # mkdir dir results
  execute librec-3.0.0/core/src/test/java/net/librec/MainTest.java
  ```
 
  Then the predicted results of TrustSVD are stored in the file 'librec-3.0.0/results'.
  
  
  ### 3.4 Final prediction fusion
  
  Lastly, we fuse the predictions from deep graph model and wide shallow model and obtain the final results through `fuse_loss.py`.
  
  In the file, paratemer `weight` can be adjusted to balance the metric of MAE and RMSE.
  
  ```
  python fuse_loss.py
  ```
  
  
  ******
  **Note: ** The fianl predictions rely on both deep and wide model. Specifically, the deep graph model needs to be well trained to get the best MAE prediction, so the training epoch must exceed 30. Besides, the result of TrustSVD affects the RMSE predition. 
  
  
