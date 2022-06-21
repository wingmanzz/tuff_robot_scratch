Folder that holds different variations of modified training sets.

Non-deduped:

china_90_swap.7z: Training set not deduped, but 80-90, 90-100 reviewed swaps performed.

Deduped folders:

China_clean_no_swap.7z : Deduped, with no file swaps.

China_clean_90_swap.7z: Deduped, with only the reviewed 90-100 misclassified files swapped.

China_clean_80_swap.7z: Deduped, with both the reviewed 80-90 and 90-100 misclassified files swapped. 

Dealing with Interfolder Dupes: (Building off of internally deduped folders w/ 80-90 and 90-100 swaps).


china_removal.7z: All 660 possible interfolder training duplicates removed from the folder.

china_swap.7z: Predicting the outcome of the 660 interfolder dupes, then assigning them to a folder based on those predictions. 


