from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
import pandas as pd


def hist_box(df, variable, descripcion:int, bins=None, agrupado=None):
    with open("data/features.json", "r") as file:
        info = json.load(file)
    unique_values = sorted(set(df[variable]))

    if bins == 1:
        bin_edges = [val - 0.5 for val in unique_values] + [max(unique_values) + 0.5]
    else:
        bin_edges = "auto"

    fig,axes = plt.subplots(2,1,figsize=(15,10),
                            sharex = True,
                            gridspec_kw={'height_ratios':[1,3]})
    sns.boxplot(ax = axes[0],
                data = df,
                x = variable,
                y = agrupado,
                color=sns.color_palette('pastel')[0]
                )
    if agrupado == None:
        sns.histplot(ax=axes[1],
                     data=df,
                     x=variable,
                     hue=agrupado,
                     kde=True,
                     bins=bin_edges,
                     color=sns.color_palette('pastel')[0]
                     )
    else:
        sns.histplot(ax=axes[1],
                     data=df,
                     x=variable,
                     hue=agrupado,
                     kde=True,
                     bins=bin_edges,
                     palette='pastel'
                     )
    # Si no hay agrupado, se agrega linea de media y mediana.
    if agrupado == None:
        plt.axvline(x=df[variable].mean(),
                    color = "red",
                    linestyle = "--",
                    label = "mean",
                    )
        plt.axvline(x=df[variable].median(),
                    color = "green",
                    linestyle = "--",
                    label = 'median')
        plt.legend(loc = 1)
        titulo = (f'HISTOGRAMA + BOXPLOT DE: {variable}')
    else:
        titulo = (f'HISTOGRAMA + BOXPLOT DE: {variable} SEGUN TARGET')
    # Si se queire agregar la descripcion, se levanta del json.
    if descripcion == 1:
        plt.suptitle(titulo + '\n' + info[variable],fontsize=14,y = 0.97)
    else:
        plt.suptitle(titulo,fontsize=16,y = 0.9)
    axes[0].set(xlabel=None, ylabel=None)
    fig.set_facecolor('white')
    return plt.show()


def graf_cat(df, variable, descripcion:int, agrupado):
    with open("data/features.json", "r") as file:
        info = json.load(file)
    titulo = (f'Barras variables no numericas: {variable}')
    fig,axes = plt.subplots(1,2,figsize=(12,8),
                            sharex = True)
    sns.countplot(ax = axes[0],
                data = df,
                x = variable,
                color=sns.color_palette('pastel')[7]
                )

    sns.histplot(ax = axes[1],
                 data  =df,
                 x = variable,
                 hue = agrupado,
                 palette='pastel',
                 multiple='fill',
                 discrete=True
                 )

    if descripcion == 1:
        plt.suptitle(titulo + '\n' + info[variable], fontsize=14, y = 0.97)
    else:
        plt.suptitle(titulo,fontsize=16,y = 0.9)
    axes[0].set_title('Histograma')
    axes[0].set(xlabel=variable, ylabel='count')
    axes[0].tick_params(axis='x', rotation=90)
    axes[1].set_title('Barras 100%')
    axes[1].set(xlabel=variable, ylabel='%')
    axes[1].tick_params(axis='x', rotation=90)
    fig.set_facecolor('white')
    return plt.show()


def mat_conf(y_test,y_pred):
  cf_matrix = confusion_matrix(y_test, y_pred)
  ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues',fmt='.1f')
  ax.set_title('Matriz de confusion con labels\n\n');
  ax.set_xlabel('\nValores predichos')
  ax.set_ylabel('Valores reales ');
  ax.xaxis.set_ticklabels(['False','True'])
  ax.yaxis.set_ticklabels(['False','True'])
  return(plt.show())