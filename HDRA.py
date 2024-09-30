#os is used to access the csv file 
import os
#Warnings are used to ignore the warnings prompted by pgmpy
import warnings

#Itertools is used to create combinations on data structures
from itertools import combinations, product

#Pandas is used to work with dataframes
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

#PyImpetus is used to get the most important features
warnings.simplefilter(action='ignore', category=UserWarning)
from PyImpetus import PPIMBC

#Matplotlib and networkx are used to plot the graph
import matplotlib.pyplot as plt
import networkx as nx

import numpy as np

#Pgmpy modules used throughout the code
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFWriter

#Sklearn modules used throughout the code to work with the data
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score

def clean_data(data: pd.DataFrame):
    """
    Cleans the given DataFrame by performing various data transformations.

    Args:
        data (pd.DataFrame): The input DataFrame to be cleaned.

    Returns:
        pd.DataFrame: The cleaned DataFrame.

    """
    data = data[~(data['Cholesterol'] == 0) & ~(data['RestingBP'] == 0)]
    df = data.copy()
    df["Age"] = pd.cut(x=data["Age"], bins=[20, 40, 50, 60, 70, np.Inf], labels=[
        "20-40", "40-50", "50-60", "60-70", "70+"])

    df["RestingBP"] = pd.cut(x=data["RestingBP"], bins=[90, 120, 140, np.Inf], labels=[
        "90-120", "120-140", "140+"])

    df["Cholesterol"] = pd.cut(x=data["Cholesterol"], bins=[
        -np.Inf, 200, 240, np.Inf], labels=["<=200", "200-240", "240+"])

    df["MaxHR"] = pd.qcut(x=data["MaxHR"], q=4, labels=[
                          "low", "medium", "high", "very-high"])  # binning using quartiles

    df["Oldpeak"] = pd.cut(x=data["Oldpeak"], bins=[-np.Inf, 0.5, 1, 2, np.Inf], labels=[
        "<=0.5", "0.5-1", "1-2", "2+"])
    df['FastingBS'] = df['FastingBS'].map({0: 'N', 1: 'Y'})
    return df

data = pd.read_csv(f'data{os.sep}heart.csv')
df = clean_data(data)
df.to_csv(f'data{os.sep}heart_cleaned.csv')

print("The dataset contains %s observations and %s attributes" % df.shape)

df.head()

df["HeartDisease"].value_counts().plot(kind="pie", autopct='%1.1f%%', startangle=90, explode=[
    0, 0.1], shadow=True, labels=['Normal', 'Disease'], label='', title="Heart Disease Distribution")
plt.tight_layout()
plt.show()

target_variable = "HeartDisease"
X, y = df.drop(columns=target_variable), df[target_variable]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

network = [(target_variable, x) for x in df.columns[:-1]]
naive_bayes = BayesianNetwork(network)

pos = nx.nx_agraph.graphviz_layout(naive_bayes, prog="dot")
nx.draw(naive_bayes, pos, with_labels=True, node_size=1000,
        font_size=8, arrowsize=20, alpha=0.8)

kfold = KFold(5, shuffle=True, random_state=42)

def bayesian_kfold(df, model, metric):
    score = []
    predictions = []
    for train, test in kfold.split(df):
        model.cpds = []
        model.fit(df.iloc[train, :], estimator=MaximumLikelihoodEstimator)
        y_pred = model.predict(df.drop(columns=target_variable, axis=1).iloc[test, :])
        score.append(
            metric(df[target_variable].iloc[test], y_pred[target_variable]))
        predictions.append(y_pred)
    return sum(score) / len(score), predictions[0]


roc_auc_value, _ = bayesian_kfold(df, naive_bayes, roc_auc_score)

print(f"The roc_auc score for the naive bayesian network is {roc_auc_value:.2f}")

scores = {} # Dictionary to store the roc_auc_score for each scoring method
networks = {} # Dictionary to store the network structure for each scoring method

for scoring in ['k2score', 'bdeuscore', 'bdsscore', 'bicscore', 'aicscore']:
    network = HillClimbSearch(df, use_cache=False).estimate(scoring_method=scoring)
    networks[scoring] = network    
    model = BayesianNetwork(network)
    scores[scoring], _ = bayesian_kfold(df, model, roc_auc_score)

    fig = plt.figure(figsize=(20,15))
i = 1
for scoring in networks:
    plt.subplot(3,2,i)
    pos = nx.nx_agraph.graphviz_layout(networks[scoring], prog="dot")
    nx.draw(networks[scoring], pos, with_labels=True, node_size=1000,
        font_size=8, arrowsize=20, alpha=0.8) 
    plt.title(scoring) 
    i += 1
plt.show()

pd.DataFrame(scores, index = ['ROC AUC'])

hc_unconst = HillClimbSearch(df, use_cache=False).estimate(scoring_method='bdeuscore')

pos = nx.nx_agraph.graphviz_layout(hc_unconst, prog="dot")
nx.draw(hc_unconst, pos, with_labels=True, node_size=1000,
        font_size=8, arrowsize=20, alpha=0.8)

black_list = [(target_variable, 'Cholesterol'), 
              ('Oldpeak', target_variable),
              ('ST_Slope', target_variable),
              ('ST_Slope', 'MaxHR'),
              ('ExerciseAngina', 'Cholesterol'),
              ('ST_Slope', 'Oldpeak'),
              ('ExerciseAngina', 'RestingECG'),
              ('ExerciseAngina', 'ChestPainType')] + [(x, 'Sex')for x in df.columns] + [(x, 'Age') for x in df.columns]

hc_const = HillClimbSearch(df, use_cache=False).estimate(
    scoring_method='BDeuScore', black_list=black_list)

pos = nx.nx_agraph.graphviz_layout(hc_const, prog="dot")
nx.draw(hc_const, pos, with_labels=True,
        node_size=1000, font_size=8, arrowsize=20, alpha=0.8)

hc_const.add_edge('Sex', 'Cholesterol')
hc_const.add_edge('Cholesterol', target_variable)
hc_const.add_edge('Oldpeak', 'ST_Slope')
hc_const.add_edge('RestingECG', target_variable)
hc_const.add_edge('RestingBP', target_variable)
hc_const.add_edge('Cholesterol', 'RestingBP')
hc_const.add_edge('FastingBS', target_variable)
hc_const.add_edge('Age', 'Cholesterol')

pos = nx.nx_agraph.graphviz_layout(hc_const, prog="dot")
nx.draw(hc_const, pos, with_labels=True,
        node_size=1000, font_size=8, arrowsize=20, alpha=0.8)

hc_const_model = BayesianNetwork(hc_const.edges())

hc_const_model.cpds = [] # Clear the cpds
hc_const_model.fit(train, estimator=MaximumLikelihoodEstimator)
hc_const_model.get_cpds()
assert hc_const_model.check_model()

accuracy, _ = bayesian_kfold(df, hc_const_model, accuracy_score)
roc_auc, _ = bayesian_kfold(df, hc_const_model, roc_auc_score)

print(f'ROC AUC: {roc_auc:.3f}')

print(f'Accuarcy: {accuracy:.3f}')
print(f'ROC AUC: {roc_auc:.3f}')

domain_kg_model = BayesianNetwork([
    ('Age', 'Cholesterol'),
    ('Age', 'RestingECG'),
    ('Age', 'MaxHR'),
    ('Age', 'RestingBP'),
    ('Age', 'FastingBS'),
    ('Age', target_variable),
    ('Sex', 'Cholesterol'),
    ('Sex', 'MaxHR'),
    ('Sex', 'ExerciseAngina'),
    ('Sex', target_variable),
    ('RestingECG', target_variable),
    ('MaxHR', 'ExerciseAngina'),
    ('Cholesterol', target_variable),
    ('Cholesterol', 'RestingBP'),
    ('RestingBP', 'FastingBS'),
    ('FastingBS', target_variable),
    ('ExerciseAngina', target_variable),
    ('ExerciseAngina', 'Oldpeak'),
    ('ExerciseAngina', 'ST_Slope'),
    (target_variable, 'Oldpeak'),
    (target_variable, 'ST_Slope'),
    (target_variable, 'ChestPainType'),
    ('Oldpeak', 'ST_Slope'),
]
)

pos = nx.nx_agraph.graphviz_layout(domain_kg_model, prog="dot")
nx.draw(domain_kg_model, pos, with_labels=True,
        node_size=1000, font_size=8, arrowsize=20, alpha=0.8)

domain_kg_model.cpds = []
domain_kg_model.fit(train, estimator=BayesianEstimator, prior_type="BDeu")
domain_kg_model.get_cpds()
assert domain_kg_model.check_model()


accuracy, _ = bayesian_kfold(df, domain_kg_model, accuracy_score)
roc_auc, y_pred = bayesian_kfold(df, domain_kg_model, roc_auc_score)

print(f'Accuarcy: {accuracy:.3f}')
print(f'ROC AUC: {roc_auc:.3f}')

if not os.path.isdir('model'):
    os.mkdir('model')
domain_kg_model.save(f'model{os.sep}heart_disease_model.bif')
writer = XMLBIFWriter(domain_kg_model)
writer.write_xmlbif(f'model{os.sep}heart_disease_model.xml')

if os.path.isdir('HeartDisease-Dashboard'):
    domain_kg_model.save(
        f'HeartDisease-Dashboard{os.sep}model{os.sep}heart_disease_model.bif')
    writer.write_xmlbif(
        f'HeartDisease-Dashboard{os.sep}model{os.sep}heart_disease_model.xml')
    
encoder = OrdinalEncoder().set_output(transform='pandas')

X_enc, _ = encoder.fit_transform(X), y

model_mb = PPIMBC(model=SVC(random_state=42, class_weight="balanced"), p_val_thresh=0.05, num_simul=100,
               simul_size=0.20, simul_type=1, sig_test_type="non-parametric", cv=5, random_state=42, n_jobs=-1, verbose=0)


model_mb.fit_transform(X_enc, y.values)

print(f"The most important features are: {model_mb.MB}")

model_mb.feature_importance()

all_nodes = model_mb.MB + [target_variable]
edges = [x for x in list(domain_kg_model.edges()) if x[0] in all_nodes and x[1] in all_nodes]

reduced_network = BayesianNetwork(edges)

pos = nx.nx_agraph.graphviz_layout(reduced_network, prog="dot")
nx.draw(reduced_network, pos, with_labels=True,
        node_size=1000, font_size=8, arrowsize=20, alpha=0.8)

reduced_network.cpds = []
reduced_network.fit(train[list(reduced_network.nodes)], estimator=BayesianEstimator, prior_type="BDeu")
reduced_network.get_cpds()
assert reduced_network.check_model()

removed_vars = list(reduced_network.nodes)
removed_vars.remove(target_variable)

accuracy, _ = bayesian_kfold(df.loc[:, list(reduced_network.nodes)], reduced_network, accuracy_score)
roc_auc, y_pred = bayesian_kfold(df.loc[:, list(reduced_network.nodes)], reduced_network, roc_auc_score)

print(f'Accuarcy: {accuracy:.3f}')
print(f'ROC AUC: {roc_auc:.3f}')

print(f'There can be made {len(domain_kg_model.get_independencies().get_assertions())}',
      'valid independence assertions with respect to the all possible given evidence.')
print('For instance, any node in the network is independent of its non-descendents given its parents:\n',
      f'\n{domain_kg_model.local_independencies(df.columns.tolist())}\n')

print('But we can also find some other independencies in the network given some evidence. For example:\n')

for node in df.columns.tolist():
    for assertion in domain_kg_model.get_independencies(latex=False, include_latents=True).get_assertions():
        if node in assertion.get_assertion()[0] and len(assertion.get_assertion()[2]) < 4 and len(assertion.get_assertion()[1]) < 3 and assertion not in domain_kg_model.local_independencies(df.columns.tolist()).get_assertions():
            print(assertion)
            break

print('Thanks to this library it is also possible to find automatically the Markov blanket of any node in the network.\n')


def markov_blanket(model, target_variable):
    return f"Markov blanket of \'{target_variable}\' is {domain_kg_model.get_markov_blanket(target_variable)}"


for column in df.columns:
    print(markov_blanket(domain_kg_model, column))

target = [target_variable]

def format_string(array):
    string = str(array[0])
    for item in array[1:]: 
        string += f', {item}'
    return string

def exact_inference(model, variables, evidence):
    inference = VariableElimination(model)
    result = inference.query(variables=variables, evidence=evidence)
    return result

def create_dictionary(df, columns):
    dictionary = {}
    for column in columns:
        dictionary[column] = df[column].unique().tolist()
    return dictionary

def get_all_combinations(dictionary):
    if len(dictionary) == 1:
        return [{list(dictionary.keys())[0]: value} for value in dictionary[list(dictionary.keys())[0]]]
    res = []
    for k1, k2 in combinations(dictionary.keys(), 2):
        for v1, v2 in product(dictionary[k1], dictionary[k2]):
            res.append({k1: v1, k2: v2})
    return res

def he_prob_analysis(model, target, knowledge, df):
    res = pd.DataFrame(columns=knowledge + ["Prob"]) 
    if len(knowledge) == 1:
        print(f'How does {format_string(knowledge)} affect the {format_string(target)} probability?')
    else:
        print(f'How the does combination of the variables {format_string(knowledge)} affect the {format_string(target)} probability?')

    all_values = create_dictionary(df, knowledge)
    all_queries = get_all_combinations(all_values)

    for query in all_queries:
        result = exact_inference(model, target, query)
        query["Prob"] = result.values[1]
        res.loc[len(res)] = query

    return res.sort_values(by=knowledge[0], ascending=False).reset_index(drop=True)

evidences = ["Age"]
he_prob_analysis(domain_kg_model, target, evidences, df)

evidences = ['RestingBP', 'FastingBS']
he_prob_analysis(domain_kg_model, target, evidences, df)

evidences =  ['Cholesterol', 'Sex']
he_prob_analysis(domain_kg_model, target, evidences, df)

evidences = ['ST_Slope', 'Oldpeak']
he_prob_analysis(domain_kg_model, target, evidences, df)

def prediction(model, query):
    labels = df.columns.tolist()
    labels.remove('HeartDisease')
    variables = []

    for label in labels:
        if label not in query.keys():
            variables.append(label)

    base_result = exact_inference(model, target, query)
    probs = base_result.values
    probs = np.round(probs * 100, 2)

    my_dict = {}
    for col in df.drop('HeartDisease', axis=1).columns:
        my_dict[col] = df[col].unique().tolist()
    exam_df = pd.DataFrame(columns=['exam', 'outcome', 'prob'])

    for var in variables:
        for val in my_dict[var]:
            query[var] = val
            result = exact_inference(model, target, query)
            exam_df.loc[len(exam_df)] = [var, val, round(result.values[1],2)]
            del query[var]

    exam_df.sort_values(by='prob', ascending=False,inplace=True)

    print(f'Given the evidence: ')
    for key, value in query.items():
        print(f'{key}: {value}')
    print()
    print(f'The probability of having a heart disease is {probs[1]}%')
    
    if probs[1] == 100:
            print(f'Exiting, probability has reached 100%')
            return

    print('The next exams suggested are:')
    display(exam_df.head(3))
    
query = {'Age': '20-40', 'Sex': 'M', 'ChestPainType' : 'ATA'}

prediction(domain_kg_model, query)
print('-'*20)

print('Now we are going to test the model with a new query adding the result of the suggested exams.')
query['Oldpeak'] = '2+'
prediction(domain_kg_model, query)
print('-'*20)

print('Now we are going to test the model with a new query adding the result of the suggested exams.')
query['ST_Slope'] = 'Up'
prediction(domain_kg_model, query)