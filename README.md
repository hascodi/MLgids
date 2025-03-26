[//]: # (Open preview: Ctrl+Shift+V)
[//]: # (Open Preview to the Side: Ctrl+K V)
[//]: # (To HTML: Open md file, press F1 or Ctrl+Shift+P, Type export and select)

# Labo Gids ML Principles
### Author: Hassan Haddouchi 

Welkom bij de Labo Gids voor ML Principles. Deze gids is ontworpen om jullie te begeleiden bij het uitvoeren van de labo-opdrachten en om jullie te helpen bij het ontwikkelen van de vaardigheden die nodig zijn voor dit opleidingsonderdeel. 

## Doel van de gids

Het hoofddoel van deze gids is om jullie te helpen bij het succesvol afronden van de labo-opdrachten voor ML Principles. Naast de uitleg die ik geef tijdens de theorielessen en de labo contactmomenten, zal deze gids jullie voorzien van extra informatie, instructies en voorbeelden om de concepten die behandeld worden tijdens de labo's te begrijpen en toe te passen in de praktijk. Merk op dat de codevoorbeelden één manier representeren van hoe je een bepaalde taak kan uitvoeren. Een taak in Python kan op meerdere manieren worden uitgevoerd en het is belangrijk om de geselecteerde werkwijze ook in uw code werkt.

Met deze gids hebben jullie een handig hulpmiddel om jullie te begeleiden tijdens jullie reis door de labo's.

## Vereisten

**Python interpreter**: Python is de belangrijkste programmeertaal die wordt gebruikt voor het uitvoeren van de labo-opdrachten. Zorg ervoor dat je een recente stabiele versie van Python hebt geïnstalleerd op je computer. 

**Python libraries**: verschillende Python-bibliotheken worden gebruikt van data-analyse, machine learning en visualisatie. Enkele van de belangrijkste bibliotheken zijn:
    - `Pandas`: voor het inlezen, manipuleren en analyseren van gegevens.
    - `NumPy`: voor numerieke berekeningen en wiskundige operaties.
    - `Matplotlib`: voor het maken van visualisaties, zoals grafieken en diagrammen.
    - `scikit-learn`: voor machine learning-algoritmen en evaluatiemethoden. De bibliotheek biedt een breed scala aan algoritmen voor classificatie, regressie, clustering en dimensionaliteitsreductie, evenals hulpprogramma's voor modelselectie, validatie en prestatie-evaluatie.
    - `TensorFlow` en `Keras`: TensorFlow is een open-source machine learning-bibliotheek die wordt gebruikt voor het bouwen, trainen en implementeren van diepe neurale netwerken en andere machine learning-modellen. Keras is een high-level neurale netwerk-API die bovenop TensorFlow is gebouwd en het gemakkelijker maakt om diepe leermodellen te maken en te trainen.
    - `Seaborn` (optioneel): voor het maken van high-level visualisaties in combinatie met de bibliotheek matplotlib.
    - `Os` (optioneel): om je huidige werkfolder (working directory) op te vragen.
    - Jupyter Notebook (optioneel): voor interactieve ontwikkeling.

Merk op dat de bibliotheken die je zal en kan gebruiken zich niet beperken tot de lijst hierboven. Naargelang de operatie die je wil uitvoeren, kan je steeds een extra bibliotheek importeren. 
<div style="page-break-after: always"></div>

Zo importeer je in Python een bibliotheek:

```python
import pandas as pd
```

**Dataset: gegevensbestanden en -bronnen**: 

Voor het uitvoeren van experimenten en analyses met machine learning-modellen zijn gegevens nodig. Er zijn verschillende manieren om gegevens te verkrijgen en te gebruiken in Python:

- Externe datasets: datasets kunnen worden verkregen van externe bronnen zoals online repositories, openbare databases en dataverzamelingsplatforms. Deze datasets kunnen in verschillende formaten zijn, zoals CSV, Excel, JSON, SQL, enzovoort. Om deze datasets in te laden, kunnen bibliotheken zoals Pandas worden gebruikt voor het lezen en manipuleren van gestructureerde gegevens, terwijl bibliotheken zoals NumPy kunnen worden gebruikt voor het verwerken van numerieke gegevens.

Zo kan je een externe dataset inladen:
```python
import pandas as pd

data = pd.read_csv('dataset.csv')
```

- sklearn.datasets: de scikit-learn-bibliotheek biedt een ingebouwde verzameling van kleine standaarddatasets die handig zijn voor het oefenen en experimenteren met machine learning-algoritmen. Deze datasets kunnen direct worden geladen met behulp van functies zoals `load_boston`, `load_iris`, `load_digits`, `load_wine`, enzovoort.

Zo kan je de Iris dataset in Sklearn inladen:
```{python}
from sklearn.datasets import load_iris

iris = load_iris()
```

- Kaggle of UCI Machine Learning Repository datasets: Kaggle en UCI zijn populaire platformen voor datawetenschap en machine learning, waar gebruikers toegang hebben tot een uitgebreide verzameling van datasets die zijn geüpload door de community. Deze datasets kunnen worden gedownload van de website en vervolgens worden ingeladen en gebruikt in Python voor analyse en modelbouw.

<div style="page-break-after: always"></div>

**Ontwikkelomgeving**:
Gebruik de geïntegreerde ontwikkelomgeving (IDE) VS Code. Als je liever een andere IDE gebruikt, dan is dit op eigen risico met betrekking tot compatibileitsproblemen en troubleshooting. Zorg ervoor dat je IDE correct is geconfigureerd en up-to-date is.


## Veel-gebruikte functies

Functie: __read_csv__
Deze functie wordt gebruikt om een CSV-bestand in te lezen als een DataFrame in Python. Het wordt vaak gebruikt voor het werken met gestructureerde gegevens.
Library: pandas

Voorbeeld:
```python
import pandas as pd

# Lees een CSV-bestand in als een DataFrame
data = pd.read_csv('dataset.csv')
```


Functie: __train_test_split__
Deze functie wordt gebruikt om de dataset op te splitsen in een trainingsset en een testset. Dit is essentieel bij het bouwen en evalueren van machine learning-modellen om overfitting te voorkomen.
Library: `scikit-learn`

Voorbeeld:
```python
from sklearn.model_selection import train_test_split

# Split de dataset in trainings- en testset met de 80-20 regel
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


Functie: __Pipeline__
Een pipeline wordt gebruikt om een reeks gegevensverwerkingsstappen te definiëren die sequentieel worden toegepast op de gegevens. Het kan handig zijn om verschillende bewerkingen samen te voegen, zoals gegevensimputatie en modelbouw.
Library: `scikit-learn`

Voorbeeld:
```python
from sklearn.pipeline import Pipeline

# Definieer een pipeline met gegevensverwerkingsstappen en een model
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('regressor', LinearRegression())
])
```

<div style="page-break-after: always"></div>

Functie: __SimpleImputer__
Deze functie wordt gebruikt om ontbrekende waarden in de dataset te imputeren door ze te vervangen door een bepaalde strategie, zoals het gemiddelde of de mediane waarde van de kolom.
Library: `scikit-learn`

Voorbeeld:
```python
from sklearn.impute import SimpleImputer

# Maak een imputer object aan en specificeer de strategie voor het imputeren van ontbrekende waarden (in de geval de mediaan van de kolom)
imputer = SimpleImputer(strategy='mean')
```


Functie: __LinearRegression__
Dit is een machine learning-algoritme dat wordt gebruikt voor lineaire regressie. Het wordt gebruikt om de relatie tussen een afhankelijke variabele en een of meer onafhankelijke variabelen te modelleren.
Library: `scikit-learn`

Voorbeeld:
```python
from sklearn.linear_model import LinearRegression

# Maak een lineaire regressie model object aan
model = LinearRegression()
```


Functie: __mean_squared_error__
Deze functie wordt gebruikt om de mean squared error (MSE) te berekenen, een maat voor de nauwkeurigheid van een regressiemodel.
Library: `scikit-learn`

Voorbeeld:
```python
from sklearn.metrics import mean_squared_error

# Bereken de mean squared error tussen de voorspelde waarden en de echte waarden
mse = mean_squared_error(y_true, y_pred)
```


Functies: __Sequential__, __Dense__
Deze functies worden gebruikt voor het bouwen van neurale netwerkmodellen in TensorFlow met behulp van de Keras API. Sequential wordt gebruikt om een sequentieel model te maken en Dense wordt gebruikt om dicht verbonden lagen in het neurale netwerk te definiëren.
Libraries: `tensorflow`, `keras`

<div style="page-break-after: always"></div>

Voorbeeld:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Maak een sequentieel model aan
model = Sequential()

# Voeg een dicht verbonden laag toe aan het model met 65 neuronen en de ReLU-activatiefunctie
model.add(Dense(units=64, activation='relu'))

```
    Extra toelichting:
    _model.add(...)_: een methode om een nieuwe laag aan het model toe te voegen.
    _Dense(...)_: dit verwijst naar het type laag dat wordt toegevoegd. Een dicht verbonden laag is een van de meest voorkomende typen neurale netwerklagen.
    _units=64_: dit is het aantal neuronen in de dicht verbonden laag. In dit geval worden er 64 neuronen gedefinieerd.
    _activation='relu'_: dit is de activatiefunctie die wordt gebruikt in de neuronen van deze laag. 'relu' staat voor Rectified Linear Activation, een veelgebruikte activatiefunctie in neurale netwerken. Het helpt bij het introduceren van niet-lineariteit in het model en kan helpen bij het oplossen van niet-lineaire problemen.


Functie: __StandardScaler__
Deze functie wordt gebruikt voor het schalen van features door het verwijderen van de gemiddelde waarde en het schalen naar eenheidvariantie. Het is vaak belangrijk om features te standaardiseren voordat ze worden ingevoerd in een machine learning-model om ervoor te zorgen dat elke feature gelijk wordt behandeld.
Library: `scikit-learn`

Voorbeeld:
```python
from sklearn.preprocessing import StandardScaler

# Maak een scaler object aan en schaal de features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
<div style="page-break-after: always"></div>

    
    Extra toelichting:
    _scaler = StandardScaler()_: hier wordt een object van de StandardScaler-klasse aangemaakt. Dit object zal worden gebruikt om de gegevens te schalen, wat betekent dat het de gegevens zal transformeren zodat ze een gemiddelde van nul en een standaarddeviatie van één hebben.
    _X_scaled = scaler.fit_transform(X)_: hier worden de kenmerken (features) in de dataset X geschaald met behulp van de fit_transform-methode van de scaler. Deze methode schaalt de gegevens en past tegelijkertijd de schaaltransformatie toe op de gegevens. De methode fit_transform schaalt de gegevens op basis van de gegevens in de dataset X en retourneert de geschaalde gegevens in X_scaled.


## Concepten

**Lineaire regressie**

Lineaire regressie is een statistische techniek die wordt gebruikt om de relatie tussen een afhankelijke variabele (target) en een of meer onafhankelijke variabelen (features) te modelleren. In onze labo's passen we lineaire regressie toe (bijvoorbeeld labo's 5 en 6) om een voorspellend model te bouwen op basis van de kenmerken in onze dataset.

Bijvoorbeeld, in het labo met politiedata, hebben we gekeken naar kenmerken zoals leeftijd, geslacht en gevoel over het huishoudinkomen, en we willen het vertrouwen in de politie voorspellen. Lineaire regressie kan worden gebruikt om een model te maken dat het verband tussen deze kenmerken en het vertrouwen in de politie modelleren.

_Hoe werkt lineaire regressie?_
Bij lineaire regressie wordt aangenomen dat er een lineaire relatie bestaat tussen de onafhankelijke variabelen (features) en de afhankelijke variabele (target). Het model probeert een rechte lijn te vinden die het best past bij de gegevenspunten, waarbij de afwijkingen tussen de voorspelde waarden en de werkelijke waarden worden geminimaliseerd.

_Bouwen van het model voor lineaire regressie_
Om een lineair regressiemodel te bouwen, passen we een reeks wiskundige technieken toe om de parameters (coëfficiënten) van de rechte lijn te schatten, zodat deze het best past bij de gegevens. 

_Hoe interpreteer je een lineair regressiemodel?_
Nadat het model is gebouwd, kunnen we de coëfficiënten interpreteren om te begrijpen hoe elke feature bijdraagt aan de voorspelling van de target variabele. Een positieve coëfficiënt geeft aan dat een toename in die feature geassocieerd is met een toename in de target variabele, terwijl een negatieve coëfficiënt aangeeft dat een toename in die feature geassocieerd is met een afname in de target variabele. Zie labo's 5 en 6 voor een demonstratie met de coëfficiënten.



**Activatiefuncties**

Activatiefuncties zijn een integraal onderdeel van neurale netwerken en machine learning-modellen, waaronder die voor diepgaand leren (deep learning). In de context van onze labo's kunnen we activatiefuncties zien als de "niet-lineariteit" die wordt geïntroduceerd in de verborgen lagen van neurale netwerken.

Met "niet-lineariteit introduceren in het model" bedoel ik dat activatiefuncties ervoor zorgen dat de relatie tussen de invoer (input) en de uitvoer (output) van een neuron niet-lineair wordt. Dit is belangrijk omdat lineaire modellen, zoals we hebben gezien bij lineaire regressie, alleen lineaire relaties tussen de invoerkenmerken en de uitvoer kunnen modelleren.

Neurale netwerken bestaan uit meerdere lagen (inputlaag, verborgen lagen en uitvoerlaag), en activatiefuncties worden toegepast op de uitvoer van elke verborgen laag om niet-lineariteit in het model te introduceren.

_Hoe werken activatiefuncties?_
Activatiefuncties transformeren de gewogen som van de invoer (input) van een neurale netwerk naar de uitvoer (output) van de neuron. Ze voegen niet-lineariteit toe aan het model, wat cruciaal is voor het leren van complexe patronen in de data. Zonder activatiefuncties zouden alle verborgen lagen van het neurale netwerk gewoon lineaire transformaties uitvoeren, waardoor het model beperkt zou zijn tot het modelleren van lineaire relaties tussen de kenmerken.

_Belang van activatiefuncties_
Door het toevoegen van niet-lineariteit kunnen neurale netwerken complexe functies leren die lineaire modellen niet kunnen vastleggen. Dit stelt ons in staat om diepgaande neurale netwerken te gebruiken voor taken zoals beeldherkenning, natuurlijke taalverwerking en voorspellende modellering.

_Populaire activatiefuncties_
ReLU (Rectified Linear Unit): `f(x) = max(0, x)`
Sigmoid: `f(x) = 1 / (1 + e^(-x))`
Tanh (Hyperbolische tangent): `f(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

_Keuze van activatiefuncties_
De keuze van de activatiefunctie kan invloed hebben op het leervermogen van het model en de snelheid van het convergeren naar een optimale oplossing. In de praktijk wordt vaak ReLU gebruikt als activatiefunctie voor verborgen lagen vanwege de eenvoud en efficiëntie.



**Classificatie**

Classificatie is een machine learning-taak waarbij het doel is om invoergegevens in een of meer categorieën of klassen te categoriseren, op basis van bepaalde kenmerken of eigenschappen van de gegevens. Het is een van de meest gebruikte taken in machine learning en wordt vaak toegepast op verschillende domeinen, zoals medische diagnose, spamdetectie, beeldherkenning en sentimentanalyse. Denk aan het kat-en-hond-voorbeeld dat ik vaak in de les aanhaal.

_Hoe werkt classificatie?_
In het concept van classificatie worden gegevenspunten ingedeeld in vooraf gedefinieerde klassen op basis van de kenmerken die zijn geëxtraheerd uit de gegevens. Het doel is om een model te bouwen dat deze klassen nauwkeurig kan voorspellen voor nieuwe, niet-geziene gegevenspunten.

Er bestaan verschillende classificatietaken, zoals binair, multiklassen en multilabel classificatie. Bij binair-classificatietaken hebben we twee klassen (bijvoorbeeld kat en hond), terwijl bij multiklassen-classificatietaken meerdere klassen betrokken zijn. Multilabel-classificatie heeft betrekking op gevallen waarin een gegeven datapunt tot meerdere klassen tegelijkertijd kan behoren.

Er bestaan verschillende algoritmen voor classificatie, zoals logistieke regressie, ondersteunende vector machines (SVM), beslissingsbomen, k-nearest neighbors (KNN), en neurale netwerken (bijvoorbeeld voor diepgaande leerclassificatie). Deze algoritmen hebben elk hun eigen sterke en zwakke punten en zijn geschikt voor verschillende soorten problemen en datasets. Het kiezen van het juiste classificatiealgoritme hangt af van factoren zoals de aard van de gegevens, het aantal klassen, de grootte van de dataset en de gewenste prestaties. 



**Sequentieel model met Sequential**
"Sequential" verwijst naar een specifiek type model dat in Keras wordt gebruikt. Een Sequential-model is een lineaire stapel lagen waarin elke laag precies één inputtensor en één outputtensor heeft. Het is de eenvoudigste vorm van een model in Keras, waarbij lagen achtereenvolgens worden toegevoegd en verbonden (zie code hierboven). Dit betekent dat de output van elke laag de input van de volgende laag vormt, waardoor een reeks lagen wordt gecreëerd die samen het model vormen.

In termen van opbouw van het model, kun je Sequential gebruiken om een reeks lagen te specificeren, zoals Dense-lagen (volledig verbonden lagen), Convolutional-lagen (convolutienetwerklagen), Pooling-lagen, enzovoort. Je voegt deze lagen toe aan het model met behulp van de add() methode, waardoor ze sequentieel worden gestapeld.

Hier is een voorbeeld van hoe je een eenvoudig Sequential-model kunt maken in Keras:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Ik maak een Sequential model aan
model = Sequential()

# Ik voeg een Dense (volledig verbonden) laag toe aan het model
model.add(Dense(units=64, activation='relu', input_shape=(10,)))

# Ik voeg nog een Dense laag toe
model.add(Dense(units=32, activation='relu'))

# Ik voeg de laatste Dense laag toe met softmax activatie voor classificatie
model.add(Dense(units=10, activation='softmax'))

# Ik compileer het model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Ik bekijk de samenvatting van het model
model.summary()
```

In dit voorbeeld wordt een Sequential-model gemaakt met drie Dense-lagen: één invoerlaag met 64 neuronen, één verborgen laag met 32 neuronen en een uitvoerlaag met 10 neuronen voor classificatie. De activatiefunctie `relu` wordt gebruikt voor de verborgen lagen, terwijl `softmax` wordt gebruikt voor de uitvoerlaag bij classificatieproblemen.



## Tips & Tricks

**Efficiënt gebruik van resources**
Maak efficiënt gebruik van de resources die ik biedt, zoals deze gids en modeloplossingen. Gebruik a.u.b. de modeloplossingen niet als je er niet in bent geslaagd om de opdracht volledig op te lossen. Probeer eerst zelf iets te maken en gebruik mijn modeloplossing louter om een andere wijze van implementatie te bekijken.

**Vergeet jullie portfolio niet!**
Documenteer beknopt in uw portfolio hoe je uw opdracht hebt uitgevoerd en wat de belangrijkste inzichten zijn uit de theorieles.

## Enkele begrippen

Hieronder staan enkele begrippen uit de cursus. Merk op dat de begrippen die we leren zich niet enkel beperken tot onderstaande benamingen. De cursus bevat nog veel meer begrippen waarvan wordt verwacht dat studenten deze beheersen.

**Standaarddeviatie**
De standaarddeviatie is een maatstaf voor de spreiding van data rond het gemiddelde. Hoe groter de standaarddeviatie, hoe meer variatie er is in de dataset.

*Voorbeeld*

Stel dat je werkt met de woningprijzen in Boston (medv-kolom). Als de standaarddeviatie hoog is, betekent dit dat de prijzen sterk uiteenlopen, met zowel goedkope als zeer dure woningen. Als de standaarddeviatie laag is, zijn de prijzen relatief dicht bij elkaar gegroepeerd. Dit helpt bij het inschatten van hoe "gevarieerd" de data zijn.

**Mediaan**
De mediaan is de middelste waarde van een gesorteerde dataset. Het verdeelt de dataset in twee gelijke helften.

*Voorbeeld*

Als je de inkomens van huishoudens of gasprijzen per jaar analyseert, kan de mediaan een betere maatstaf zijn dan het gemiddelde. Dit komt doordat de mediaan minder gevoelig is voor extreme waarden (outliers). Als er bijvoorbeeld een paar extreem dure woningen in de dataset zitten, kan het gemiddelde misleidend zijn, terwijl de mediaan een beter beeld geeft van een "typisch" huis.

**Correlatiecoëfficiënt**
De correlatiecoëfficiënt meet de sterkte en richting van een lineaire relatie tussen twee variabelen. De waarde ligt tussen -1 en 1:

1 betekent een perfecte positieve correlatie (als de ene variabele stijgt, stijgt de andere ook).
0 betekent geen correlatie (de variabelen hebben geen verband).
-1 betekent een perfecte negatieve correlatie (als de ene variabele stijgt, daalt de andere).

*Voorbeeld*

In de Boston-huizenprijsdataset kunnen we bijvoorbeeld de correlatie bekijken tussen het aantal kamers (rm) en de huizenprijs (medv). Als de correlatiecoëfficiënt positief en hoog is, betekent dit dat huizen met meer kamers over het algemeen duurder zijn.
Als je kijkt naar ‘lstat’ (percentage lage inkomens) en ‘medv’ (woningprijzen), zal de correlatie waarschijnlijk negatief zijn: een hogere armoedegraad is vaak geassocieerd met lagere woningprijzen.

**Modus**
De modus is de meest voorkomende waarde in een dataset.

*Voorbeeld*
Als je kijkt naar de indeling van woningen in categorieën (bijvoorbeeld hoeveel huizen zich in een bepaalde prijsklasse bevinden), kan de modus aangeven welke prijsklasse het vaakst voorkomt. Dit kan handig zijn bij categorische data, bijvoorbeeld de meest voorkomende belastingklasse in een dataset met economische gegevens.


#### Minimum, mediaan, modus en maximum waarden in een dataset

Als je in één tabel de minimum, mediaan, modus en maximum waarden voor elke variabele in een dataset wil weergeven, dan doe je dat bijvoorbeeld zo:

```python
import pandas as pd

print(df.describe())
```

Een iets omslachtigere manier staat hieronder:

```python
summary = pd.DataFrame()
summary['min'] = df.min()
summary['median'] = df.median()
summary['max'] = df.max()
summary['mode'] = df.mode().iloc[0]
print(summary)
```

Merk op dat dit uiteraard enkel zal werken bij numerieke data. Als je een foutmelding krijgt omdat je data niet numeriek is, dan dien je eerst te controleren welke variabelen numeriek zijn. Vervolgens kan je overwegen om kolommen om te zetten naar getallen: `pd.to_numeric(..., errors='coerce')`. Doe dit enkel als je zeker weet wat je aan het doen bent.
