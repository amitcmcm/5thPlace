there are two daily-living datasets processed and used in the code: 

- "daily_living" is the daily living data from the competition. in this code, it is preprocessed in the following way: 
1. using preprocess_dl, which attaches pseuodo-labels based on predictions from 1st, 3rd, and 5th place. 
2. using preprocess_dl_pt2_simple4h_new_new, which reduces the amount of data by selecting 4 hours based on fog frequency (more info in the code itself). 


- "fog_at_home" or "fog@home" is the labeled daily living data (from valeri's project). it is preprocessed using preprocess_fogathome_seperateannotsegments or preprocess_fogathome. 


*for fog at home, the original annotation files are not always consistent and tend to have issues, which is why i suggest using the already-preprocessed files (they are inside the folder fog@home_preprocessed). it is possible that more labeled fog@home data has been added, but i don't know if the amount of new data justifies dealing with the inconsistent label files and issues that could arise. if you decide to do it anyway, you will have to look at the fog at home preprocessing code carefully and look at the different cases and the data files they are meant to handle. you will also need access to the label files in the path listed below (ask eran). 
