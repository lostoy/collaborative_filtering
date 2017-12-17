## introduction
Recommender systems are one class of data analysis systems to discover patterns in historical
user preference data and provide recommendations for users' future interests. Such systems are of 
 vital importance in multiple industry fields ranging from only content providers, such as music/movie rental
 websites to E-commerce companies such as Amazon. Over the years, multiple algorithms have been proposed to tackle
 this problem and one of the most successful class of approaches is termed as "collaborative filtering"(CF). CF has proved
  to be quite effective in academic challenges and practical applications. CF is now playing a central role in working recommender systems.
   
  This project is focused on reviewing and benchmarking several popular CF methods, including Cosine Neighbour(CN), PureSVD and Matrix Factorization with Alternating 
   Least Squares(MF-ALS). Moreover, incremental update algorithm based on foldding-in technique is also discussed to simulate 
    a working online system when dealing with additional data.


## Collaborative filtering
In general, there are two primary approaches for collaborative filtering: 1) neighborhood based models and 2) latent factor models
Next, we discuss both of them and provide formulations in detail and defer to section [] for experimental comparison of them on a realistic dataset.
### neighborhood based models
#### SVD
##### missing data imputing
##### incremental update for SVD model
#### matrix factorization with ALS

## experiments
### dataset
In this project, the Jester dataset is selected to benchmark different CF algorithms. The Jester dataset is consited of [] of []. 
### evaluation metrics
Following standard in evaluation in recommendation systems, RMSE(Root Mean Squared Error), Recall-N and Precision-N are used to evaluate the performance of different algorithms.

``#### Recall-N =
#### Preicision-N =``

### Results
#### neighborhood based models, with different number of neighbours
```
cur_set: test	Knn: 25 	 recall: 0.387177020341 	prec: 0.790081255254 	rmse: 4.60166005756
cur_set: test	Knn: 50 	 recall: 0.390555246951 	prec: 0.794808469759 	rmse: 4.61197934807
cur_set: test	Knn: 100 	 recall: 0.392907764133 	prec: 0.798358884041 	rmse: 4.61733085424
cur_set: test	Knn: 200 	 recall: 0.394035259566 	prec: 0.800376255854 	rmse: 4.62006084333
cur_set: test	Knn: 400 	 recall: 0.39478508868 	prec: 0.801725173118 	rmse: 4.6214507199
```

================ recall and precision only on positive scored gts
```
cur_set: test	Knn: 25 	 recall: 0.492057082492 	prec: 0.613136933115 	rmse: 4.60166005756
cur_set: test	Knn: 50 	 recall: 0.499836530916 	prec: 0.621818836809 	rmse: 4.61197934807
cur_set: test	Knn: 100 	 recall: 0.504048989398 	prec: 0.62644998599 	rmse: 4.61733085424
cur_set: test	Knn: 200 	 recall: 0.505038521376 	prec: 0.627766881479 	rmse: 4.62006084333
cur_set: test	Knn: 400 	 recall: 0.505751721633 	prec: 0.627522715447 	rmse: 4.6214507199
```

====================== recall and precision only on score>3.0 gts
```cur_set: test	Knn: 25 	 recall: 0.570902423073 	prec: 0.447576351919 	rmse: 4.60166005756
cur_set: test	Knn: 50 	 recall: 0.579712839875 	prec: 0.454376976344 	rmse: 4.61197934807
cur_set: test	Knn: 100 	 recall: 0.585294781257 	prec: 0.458655886002 	rmse: 4.61733085424
cur_set: test	Knn: 200 	 recall: 0.586710471439 	prec: 0.459760637233 	rmse: 4.62006084333
cur_set: test	Knn: 400 	 recall: 0.586474383164 	prec: 0.45888804387 	rmse: 4.6214507199
```

#### SVD with different latent dimension
```
cur_set: test	k_dim: 5 	 recall: 0.369658283872 	prec: 0.763655285594 	rmse: 4.06385415998
cur_set: test	k_dim: 10 	 recall: 0.369338439223 	prec: 0.764980186527 	rmse: 4.06031519104
cur_set: test	k_dim: 25 	 recall: 0.363881213509 	prec: 0.758527798903 	rmse: 4.218244085
cur_set: test	k_dim: 50 	 recall: 0.363705107696 	prec: 0.754873313853 	rmse: 4.51843247222
cur_set: test	k_dim: 75 	 recall: 0.37601676167 	prec: 0.765476524036 	rmse: 4.72157445049
```

============= recall and precision only on positive scored gts
```
cur_set: test	k_dim: 2 	 recall: 0.478348233008 	prec: 0.59591322099 	rmse: 4.16749841919
cur_set: test	k_dim: 5 	 recall: 0.484712280543 	prec: 0.606556458392 	rmse: 4.06385415998
cur_set: test	k_dim: 10 	 recall: 0.490685236382 	prec: 0.608729936357 	rmse: 4.06031519104
cur_set: test	k_dim: 25 	 recall: 0.479965808197 	prec: 0.592739062563 	rmse: 4.218244085
cur_set: test	k_dim: 50 	 recall: 0.465339967991 	prec: 0.560669255093 	rmse: 4.51843247222
cur_set: test	k_dim: 75 	 recall: 0.479912330248 	prec: 0.559652563743 	rmse: 4.72157445049
cur_set: test	k_dim: 90 	 recall: 0.497565437955 	prec: 0.579446023296 	rmse: 4.77956747863
```

====================== recall and precision only on score>3.0 gts
```
cur_set: test	k_dim: 2 	 recall: 0.555281213915 	prec: 0.437161269663 	rmse: 4.16749841919
cur_set: test	k_dim: 5 	 recall: 0.572488561774 	prec: 0.450654445023 	rmse: 4.06385415998
cur_set: test	k_dim: 10 	 recall: 0.578724676355 	prec: 0.451919305127 	rmse: 4.06031519104
cur_set: test	k_dim: 25 	 recall: 0.562308445047 	prec: 0.436276668134 	rmse: 4.218244085
cur_set: test	k_dim: 50 	 recall: 0.534760568819 	prec: 0.40300604411 	rmse: 4.51843247222
cur_set: test	k_dim: 75 	 recall: 0.546810297293 	prec: 0.400396269463 	rmse: 4.72157445049
cur_set: test	k_dim: 90 	 recall: 0.561966974919 	prec: 0.414153624465 	rmse: 4.77956747863
```

#### SVD with incremental udpates
```
cur_set: test	k_dim: 10 	 update_percent: 90	recall: 0.365848758495 	prec: 0.759528479366 	rmse: 4.08748616017
cur_set: test	k_dim: 10 	 update_percent: 80	recall: 0.366725728165 	prec: 0.760833366689 	rmse: 4.07305698328
cur_set: test	k_dim: 10 	 update_percent: 70	recall: 0.367602295005 	prec: 0.762034183245 	rmse: 4.0673010362
cur_set: test	k_dim: 10 	 update_percent: 60	recall: 0.368576597138 	prec: 0.763707320978 	rmse: 4.06431742718
cur_set: test	k_dim: 10 	 update_percent: 50	recall: 0.369348504114 	prec: 0.764988191971 	rmse: 4.06291458323
cur_set: test	k_dim: 10 	 update_percent: 40	recall: 0.369436335594 	prec: 0.765020213745 	rmse: 4.06153550912
cur_set: test	k_dim: 10 	 update_percent: 30	recall: 0.369191484548 	prec: 0.764647960613 	rmse: 4.06077193902
cur_set: test	k_dim: 10 	 update_percent: 20	recall: 0.369178950125 	prec: 0.764720009607 	rmse: 4.06087146248
cur_set: test	k_dim: 10 	 update_percent: 10	recall: 0.369393600497 	prec: 0.765136292679 	rmse: 4.06087440155
cur_set: test	k_dim: 10 	 update_percent: 0	recall: 0.369338439223 	prec: 0.764980186527 	rmse: 4.06031519104
```

======================== recall and precision only on positive scored gts
```
cur_set: test	k_dim: 10 	 update_percent: 99	recall: 0.455704922307 	prec: 0.571356522435 	rmse: 4.19665053328
cur_set: test	k_dim: 10 	 update_percent: 95	recall: 0.478918325139 	prec: 0.597826522035 	rmse: 4.11029229901
cur_set: test	k_dim: 10 	 update_percent: 90	recall: 0.482850492249 	prec: 0.602201497018 	rmse: 4.08748616017
cur_set: test	k_dim: 10 	 update_percent: 80	recall: 0.485763824731 	prec: 0.604527078413 	rmse: 4.07305698328
cur_set: test	k_dim: 10 	 update_percent: 70	recall: 0.487614841636 	prec: 0.60605611816 	rmse: 4.0673010362
cur_set: test	k_dim: 10 	 update_percent: 60	recall: 0.489431301771 	prec: 0.607473081696 	rmse: 4.06431742718
cur_set: test	k_dim: 10 	 update_percent: 50	recall: 0.490284875842 	prec: 0.608321658728 	rmse: 4.06291458323
cur_set: test	k_dim: 10 	 update_percent: 40	recall: 0.490594548754 	prec: 0.608609854701 	rmse: 4.06153550912
cur_set: test	k_dim: 10 	 update_percent: 30	recall: 0.490371560126 	prec: 0.608545811152 	rmse: 4.06077193902
cur_set: test	k_dim: 10 	 update_percent: 20	recall: 0.49041660573 	prec: 0.608393707721 	rmse: 4.06087146248
cur_set: test	k_dim: 10 	 update_percent: 10	recall: 0.490680892763 	prec: 0.608625865589 	rmse: 4.06087440155
cur_set: test	k_dim: 10 	 update_percent: 0	recall: 0.490685236382 	prec: 0.608729936357 	rmse: 4.06031519104
```

====================== recall and precision only on score>3.0 gts
```
cur_set: test	k_dim: 10 	 update_percent: 99	recall: 0.535829286115 	prec: 0.420401873274 	rmse: 4.19665053328
cur_set: test	k_dim: 10 	 update_percent: 95	recall: 0.564871415887 	prec: 0.442961213625 	rmse: 4.11029229901
cur_set: test	k_dim: 10 	 update_percent: 90	recall: 0.570047578079 	prec: 0.446735780331 	rmse: 4.08748616017
cur_set: test	k_dim: 10 	 update_percent: 80	recall: 0.573362674241 	prec: 0.449017331786 	rmse: 4.07305698328
cur_set: test	k_dim: 10 	 update_percent: 70	recall: 0.575590334619 	prec: 0.449973982308 	rmse: 4.0673010362
cur_set: test	k_dim: 10 	 update_percent: 60	recall: 0.577567714118 	prec: 0.451070728095 	rmse: 4.06431742718
cur_set: test	k_dim: 10 	 update_percent: 50	recall: 0.577875905411 	prec: 0.451458992115 	rmse: 4.06291458323
cur_set: test	k_dim: 10 	 update_percent: 40	recall: 0.578750911812 	prec: 0.451787215306 	rmse: 4.06153550912
cur_set: test	k_dim: 10 	 update_percent: 30	recall: 0.57830767787 	prec: 0.451975343233 	rmse: 4.06077193902
cur_set: test	k_dim: 10 	 update_percent: 20	recall: 0.578810798184 	prec: 0.451895288796 	rmse: 4.06087146248
cur_set: test	k_dim: 10 	 update_percent: 10	recall: 0.57896977519 	prec: 0.451959332346 	rmse: 4.06087440155
cur_set: test	k_dim: 10 	 update_percent: 0	recall: 0.578724676355 	prec: 0.451919305127 	rmse: 4.06031519104
```

#### matrix factorization with different different dimension and different bias terms
TODO