# Implementation of the Page Rank algorithm

We have implemented the PageRank algorithm to determine the ranking of the most visited pages on Wikipedia, based on general popularity or specific criteria such as the presence of chosen keywords.
This algorithm calculates a score for each node of a graph. In the case of wikipedia or web pages, it shows how popular a page is.

 ## Compiling the program:

- Type "python3 Baptiste_Buani_Big_Data_Projet/page_rank.py Baptiste_Buani_Big_Data_Projet/paths_finished.tsv " in the terminal
(if you're in the directory above our archive, as in the previous case, you can run the same command)

- if not, go to our archive directory and type python3 page_rank.py paths_finished.tsv 

Then follow the program instructions: <br>

1. choose the number of pages you want to display in descending pagerank order <br>

2. choose the *damping factor* <br>

3. choose the type of Page Rank you wish to apply : <br>
         - 1. Classic Page Rank <br>
         - 2.Custom Page Rank". <br>


4. if the second option is selected, you'll be able to give *importance to pages containing words you've typed*, separated by a space.
