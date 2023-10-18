import numpy as np
import pandas as pd
import sys

# Lire les arguments de ligne de commande
arguments = sys.argv

# Stocker le deuxième argument (qui est le path_finished saisi par l'utilisateur) dans un tableau Python
path_finished = arguments[1]
tab = pd.read_csv(path_finished, sep="\t", skiprows=15)
list_noeuds = list(tab.iloc[:, 3])
tab2 = []

print("Nous traitons votre demande, cela peut prendre plusieurs minutes \n")
#stockage des pages dans une liste tab2

for i in range(0, len(list_noeuds) - 1):
  tab2 = tab2 + list_noeuds[i].split(";")

#Permet de créer un dictionnaire (sommets->liste de successeurs) à partir d'une liste de chaines de caractères "tokens"
def dico2(tokens):
  successors = {}
  #numéro de la page visité
  page_visite = [0]
  #création d'une liste de successeurs vide pour le premier sommet
  successors[tokens[0]] = []
  i = 1
  #cette boucle permet de construire le chemin des pages visités
  while i < len(tokens):
    if tokens[i - 1] not in successors and tokens[i - 1] != '<':
      successors[tokens[i - 1]] = []
    if tokens[i] != '<':

      page_visite.append(i)
      k = 1
    else:
      j = i
      k = 0
      N = len(page_visite)
      while j < len(tokens) and tokens[j] == '<':
        k += 1
        j += 1
        page_visite.append(page_visite[N - k - 1])
    i = i + k

  pred = page_visite[0]
  #print("pred :", pred)
  
  #permet d'ajouter à partir du chemin les pages dans le dictionnaire successors 
  for i in range(1, len(page_visite)):
    #print("page visité",page_visite[i])
    if (pred < page_visite[i]):
      successors[tokens[page_visite[i - 1]]].append(tokens[page_visite[i]])
    pred = page_visite[i]
    if i == len(page_visite) - 1 and tokens[i] not in successors:
      successors[tokens[page_visite[i]]] = []
  return successors

#création du dictionnaire de successeurs 

dictionnaire_succ = dico2(tab2)

#création de la matrice d'incidence

n = len(dictionnaire_succ)
A = np.zeros((n, n), dtype=np.float64)


#permet de créer la matrice d'incidence à partir du dictionnaire
def adjacence(dico):
  noeuds = list(dico.keys())
  #print("lsite des sommets",noeuds)
  
  #on pourcourt tous les des dictionnaires et on stocke la liste de successeurs correspandantes
  for i, noeud in enumerate(noeuds):
    successeurs = dico[noeud]
    #on ajoute à la colonne j un 1 au successeur du sommet i
    for successeur in successeurs:
      j = noeuds.index(successeur)
      A[i, j] += 1
  return A


mat_adjacence = adjacence(dictionnaire_succ)

#print(mat_adjacence)


def transition(mat_adjacence):
  #normalisation
  degres = np.sum(mat_adjacence, axis=1, dtype=np.float64)
  matrice_normalisee = np.zeros_like(mat_adjacence, dtype=np.float64)
  for i in range(mat_adjacence.shape[0]):
    if degres[i] != 0:
      matrice_normalisee[i, :] = mat_adjacence[i, :] / degres[i]
  mat_transition = np.transpose(matrice_normalisee)
  return mat_transition


P = transition(mat_adjacence)

#print(P)


#extrait les indices des mots recherchés "mots_recherches" dans la liste tab

def perso(tab, mots_recherches):
  indices = []
  for i, nom in enumerate(tab):
    for j in mots_recherches:
      if j.lower() in nom.lower():
        indices.append(i)
  return indices


#permet d'effectuer le pagerank personnalisé pour une matrice de transiton P
#N le nombre de lignes de notre tableau de départs
#beta un réel correspondant au damping factor
#nom: l'ensemble de mots recherchés 
#tab le tableau où s'effectuera la recherche des mots

def pagerankperso2(P, N, beta,noms,tab):
  positions = perso(tab, noms)
  w = np.zeros(N)  # Crée un vecteur de N éléments initialisés à 0
  # Les positions où les éléments doivent être changés en 1
  w[positions] = 1  
  while np.sum(w)==0:
    noms2 = input("Pas de valeur trouvé,Entrez des mots séparés par des espaces : ")
    noms = noms2.split()
    positions = perso(tab, noms)
    w[positions] = 1
  q0=w/np.sum(w)
  iteration = 100
  s=len(positions)
  for i in range(iteration):
    qk = np.dot(P, q0)
    alpha=(1-beta)/s + np.sum(q0)
    qk = beta*qk +alpha*w 
    qk= qk/np.sum(qk)
    q0=qk
  return q0



def pagerank(P, N, beta):
  v = np.random.rand(N, 1)
  v = v / np.linalg.norm(v, 1)
  R = beta * P + (1 - beta) / N
  iterations = 100

  for i in range(iterations):
    v2 = np.dot(R, v)

    v = v2
  return v





# permet de donner le choix à l'utilisateur de chosir entre le pagerank personnalisé et le classique
choix_utilisateur = ""
while choix_utilisateur not in ["1", "2", "q", "Q"]:
  choix_utilisateur = input(
    "Veuillez saisir 1 pour un page_rank, 2 pour un page_rank personnalisé, ou q pour quitter : "
  )
  nb_pages = input("Combien de page voulez vous dans votre classement : ")
  beta1= float(input("Entrez le damping factor entre 0 et 1 : "))
  
  while beta1 >1 or beta1 <0:
      beta1=  float(input("Entrez le damping factor entre 0 et 1 : "))
  if choix_utilisateur == "1":
    print("Vous avez choisi l'option 1.")
    print("Classement des ", nb_pages,
          " meilleures pages avec avec un dumping facteur de ", beta1)
    scores = pagerank(P, n, beta1)
    dic_keys = list(dictionnaire_succ.keys())
    scores = list(scores)

    df = pd.DataFrame({
      "Page Rank": scores,
      "Nom_page": dic_keys,
    })

    #rank_afrique = df.iloc[df.iloc[:, 1] == 'Nature', 0].values[0]
    #print("Afrique",rank_afrique )

    df_trie = df.sort_values(by="Page Rank", ascending=False)
    df_trie.reset_index(inplace=True)
    print("Votre le classement des  :", nb_pages, "meilleures pages \n")
    print(df_trie.head(int(nb_pages)))
    
    
  elif choix_utilisateur == "2":
    print("Vous avez choisi l'option 2.Page rank Personnalisé")
    while True:
      try:
        mots_utilisateur = input("Entrez des mots séparés par des espaces : ")
        # Stocke les mots dans une liste
        liste_mots = mots_utilisateur.split()
        break
      except ValueError:
        print("Saisie invalide. Veuillez saisir un entier.")

    print("Vous avez choisis de personnaliser le noeud ", mots_utilisateur)

    print("Classement avec un dumping facteur de", beta1)

    scores = pagerankperso2(P, n, beta1,liste_mots,dictionnaire_succ.keys())
    dic_keys = list(dictionnaire_succ.keys())
    scores = list(scores)

    df = pd.DataFrame({
      "Page Rank": scores,
      "Nom_page": dic_keys,
    })

    

    df_trie = df.sort_values(by="Page Rank", ascending=False)
    #car l'ordre des indexes avaient changé
    df_trie.reset_index(inplace=True)
    print("Votre le classement des  :", nb_pages, "meilleures pages \n")
    print(df_trie.head(int(nb_pages)))
    #print(df_trie.iloc[0:0+int(nb_pages),])

  elif choix_utilisateur.lower() == "q":
    print("Au revoir !")
    break
  else:
    print(
      "Choix invalide. Veuillez saisir 1 pour option 1, 2 pour option 2, ou q pour quitter."
    )


