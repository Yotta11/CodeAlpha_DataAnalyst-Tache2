"""
EDA — World Happiness Report (2015-2019)
CodeAlpha — Tâche 2 : Exploratory Data Analysis
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

sns.set_theme(style="whitegrid")

df = pd.read_csv("world_happiness_combined.csv")

print(df.shape)                 
df.info()                      
df.describe()
df.head(10)
df.columns.tolist()



#analyse


df.isnull().sum()
# -> region : 8 valeurs manquantes (variations de noms de pays entre années)
# -> corruption : 1 valeur manquante (Émirats Arabes Unis, une année précise)

df.duplicated().sum()          

#for col in df.select_dtypes(include="object"):
#    print(col, df[col].nunique())
#

correction_pays = {
    "Taiwan Province of China": "Eastern Asia",
    "Hong Kong S.A.R., China": "Eastern Asia",
    "Trinidad & Tobago": "Latin America and Caribbean",
    "Northern Cyprus": "Western Europe",
    "North Macedonia": "Central and Eastern Europe",
    "Gambia": "Sub-Saharan Africa",
}
df["region"] = df.apply(
    lambda r: correction_pays.get(r["country"], r["region"]), axis=1
)


df["corruption"] = df.groupby("region")["corruption"].transform(
    lambda x: x.fillna(x.median())
)

print("Valeurs manquantes restantes :", df.isnull().sum().sum())

#graphiques

# Distribution du score de bonheur
plt.figure(figsize=(8, 5))
sns.histplot(df["happiness_score"], kde=True, color="steelblue")
plt.title("Distribution du score de bonheur (2015-2019)")
plt.xlabel("Happiness Score")
plt.tight_layout()
plt.savefig("graphiques/distribution_happiness_score.png", dpi=300)
plt.close()

# Détection d'outliers sur le PIB par habitant
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["gdp_per_capita"], color="salmon")
plt.title("Détection d'outliers — PIB par habitant")
plt.tight_layout()
plt.savefig("graphiques/boxplot_gdp.png", dpi=300)
plt.close()

# Répartition des pays par région
plt.figure(figsize=(10, 6))
df["region"].value_counts().plot(kind="barh", color="teal")
plt.title("Nombre d'observations par région (2015-2019)")
plt.xlabel("Nombre d'observations")
plt.tight_layout()
plt.savefig("graphiques/repartition_regions.png", dpi=300)
plt.close()

#analyse bivariee

# Matrice de corrélation entre variables numériques
variables_num = ["happiness_score", "gdp_per_capita", "social_support",
                  "life_expectancy", "freedom", "corruption", "generosity"]

matrice_corr = df[variables_num].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(matrice_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Corrélations entre variables du bonheur")
plt.tight_layout()
plt.savefig("graphiques/heatmap_correlations.png", dpi=300)
plt.close()

# Score de bonheur moyen par région
plt.figure(figsize=(10, 6))
ordre_regions = df.groupby("region")["happiness_score"].mean().sort_values(ascending=False).index
sns.boxplot(x="happiness_score", y="region", data=df, order=ordre_regions, palette="viridis")
plt.title("Score de bonheur par région")
plt.xlabel("Happiness Score")
plt.ylabel("")
plt.tight_layout()
plt.savefig("graphiques/happiness_par_region.png", dpi=300)
plt.close()

# Évolution du score moyen dans le temps
plt.figure(figsize=(9, 5))
df.groupby("year")["happiness_score"].mean().plot(marker="o", color="darkorange")
plt.title("Évolution du score de bonheur moyen (2015-2019)")
plt.xlabel("Année")
plt.ylabel("Score moyen")
plt.tight_layout()
plt.savefig("graphiques/evolution_temporelle.png", dpi=300)
plt.close()

# Relation PIB <-> Bonheur (nuage de points)
plt.figure(figsize=(8, 6))
sns.scatterplot(x="gdp_per_capita", y="happiness_score", hue="region", data=df, alpha=0.6, legend=False)
plt.title("PIB par habitant vs Score de bonheur")
plt.tight_layout()
plt.savefig("graphiques/gdp_vs_happiness.png", dpi=300)
plt.close()



# Hypothèse 1 : le score de bonheur moyen diffère significativement selon la région
groupes_region = [df[df["region"] == r]["happiness_score"] for r in df["region"].dropna().unique()]
f_stat, p_value = stats.f_oneway(*groupes_region)
print(f"ANOVA région -> F={f_stat:.2f}, p-value={p_value:.5f}")
# p-value < 0.05 -> la région a un effet statistiquement significatif sur le bonheur

# Hypothèse 2 : corrélation entre PIB par habitant et score de bonheur
corr_gdp, p_gdp = stats.pearsonr(df["gdp_per_capita"], df["happiness_score"])
print(f"Corrélation PIB/bonheur -> r={corr_gdp:.2f}, p-value={p_gdp:.5f}")

# Hypothèse 3 : le score de bonheur moyen a-t-il changé entre 2015 et 2019 ?
score_2015 = df[df["year"] == 2015]["happiness_score"]
score_2019 = df[df["year"] == 2019]["happiness_score"]
t_stat, p_t = stats.ttest_ind(score_2015, score_2019, equal_var=False)
print(f"Test t 2015 vs 2019 -> t={t_stat:.2f}, p-value={p_t:.5f}")

#conclusion
print("""

1. Le score de bonheur mondial moyen reste relativement stable entre 2015 et 2019.
2. La région d'appartenance a un effet statistiquement significatif sur le bonheur.
3. Le PIB par habitant est fortement corrélé au score de bonheur, mais n'explique pas tout
   (le soutien social et la liberté perçue jouent aussi un rôle important).
4. L'Europe du Nord/Ouest domine systématiquement le classement sur les 5 années.
5. Quelques pays présentent un score de bonheur supérieur à ce que leur PIB seul suggérerait
   (à investiguer : rôle du soutien social/liberté).
6. Limites : 8 régions ont dû être réattribuées manuellement (variations de noms de pays),
   et une valeur de corruption a été imputée par la médiane régionale.
""")
