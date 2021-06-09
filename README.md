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
  
