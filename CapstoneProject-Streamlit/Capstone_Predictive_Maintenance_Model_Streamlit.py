import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
import streamlit as st
import plotly.graph_objects as go

print ("Start...")

mydata=pd.read_csv("predictive_maintenance.csv")

mydata1 = mydata.pop('failure') # remove column failure and store it in mydata1
mydata['failure']=mydata1 

#convert the non-numberical values to categorical values
for feature in mydata.columns: # Loop through all columns in the dataframe
	if mydata[feature].dtype == 'object': # Only apply for columns with categorical strings
		mydata[feature] = pd.Categorical(mydata[feature]).codes # Replace strings with an integer


# class count
##failure_count_0, failure_count_1 = mydata['failure'].value_counts()

# Separate class
##failure_0 = mydata[mydata['failure'] == 0]
##failure_1 = mydata[mydata['failure'] == 1]

# print the shape of the class
##print('failure 0:', failure_0.shape)
##print('failure 1:', failure_1.shape)

##failure_1_over = failure_1.sample(failure_count_0, replace=True)
##test_over = pd.concat([failure_1_over, failure_0], axis=0)
##print("Over Samling - total failure of 1 and 0 :\n", test_over['failure'].value_counts())

#split dataset in features and target variable
feature_cols = ['date', 'device', 'metric1', 'metric2','metric3','metric4', 'metric5','metric6','metric7', 'metric8', 'metric9']
target_cols = ['failure']

#X = mydata.drop(['date', 'device','failure'],axis='columns')
X = mydata.drop(['date', 'device','failure'],axis='columns')
Y = mydata['failure']

os=RandomOverSampler(sampling_strategy='minority')
X_train_res, y_train_res = os.fit_resample(X, Y)

print("RandomOverSampler - Total failure of 1 and 0 :\n", y_train_res.value_counts())


# split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(
	X_train_res, y_train_res, test_size=0.3, random_state=15, stratify=y_train_res
)


class StreamlitApp:
	def __init__(self):
		self.model = RandomForestClassifier()

	def train_data(self):
		self.model.fit(x_train, y_train)
		
		rfc_predict = self.model.predict(x_test)# check performance
		print ("Confustion Matrix")
		cnf_matrix = confusion_matrix(y_test, rfc_predict)
		print (cnf_matrix)
		print("Accuracy :", accuracy_score(y_test, rfc_predict))
		print("Precision:", precision_score(y_test, rfc_predict))
		print("Recall   :", recall_score(y_test, rfc_predict))
		print("F1       :", 2 * (precision_score(y_test, rfc_predict) * recall_score(y_test, rfc_predict)) / (precision_score(y_test, rfc_predict) + recall_score(y_test, rfc_predict)))
		print(classification_report(y_test,rfc_predict))
		return self.model
		
	def construct_sidebar(self):
		cols = [col for col in feature_cols]
		st.sidebar.markdown('<p class="header-style">Predictive Maintenance Classification</p>',unsafe_allow_html=True)
		form = st.sidebar.form(key='my_form')
		#pm_date = form.text_input(cols[0], "1/5/2015")
		#pm_device  = form.text_input(cols[1], "S1F0RRB1")
		pm_metric1 = form.text_input(cols[2], "48467332")
		pm_metric2 = form.text_input(cols[3], "64776")
		pm_metric3 = form.text_input(cols[4], "0")
		pm_metric4 = form.text_input(cols[5], "841")
		pm_metric5 = form.text_input(cols[6], "8")
		pm_metric6 = form.text_input(cols[7], "39267")
		pm_metric7 = form.text_input(cols[8], "56")
		pm_metric8 = form.text_input(cols[9], "56")
		pm_metric9 = form.text_input(cols[10], "1")
		submit_button = form.form_submit_button(label='Submit')
		
		#print ('pm_date')
		##pm_date_Val =pd.Categorical(pm_date).codes
		#pm_date_Val = time.mktime(datetime.datetime.strptime(pm_date, "%d/%m/%Y").timetuple())
		#print ( pm_date_Val)
		
		#print ('pm_device')
		#pm_device_val = pd.Categorical(pm_device).codes
		#print (pm_device_val[0])
		
		#values = [pm_date_Val[0], pm_device_val[0], pm_metric1, pm_metric2,pm_metric3,pm_metric4,pm_metric5,pm_metric6,pm_metric7,pm_metric8,pm_metric9]
		#values = [pm_metric1, pm_metric2,pm_metric3,pm_metric4,pm_metric5,pm_metric6,pm_metric7,pm_metric8,pm_metric9]
		
		values = [pm_metric1, pm_metric2, pm_metric3, pm_metric4, pm_metric5, pm_metric6, pm_metric7, pm_metric8, pm_metric9]
		print ('values')
		print(values)
		return values

	def plot_pie_chart(self, probabilities):
		fig = go.Figure(
			data=[go.Pie(
					labels=list('failure'),
					values=probabilities[0]
			)]
		)
		fig = fig.update_traces(
			hoverinfo='label+percent',
			textinfo='value',
			textfont_size=15
		)
		return fig
	
	def construct_app(self):
		self.train_data()
		values = self.construct_sidebar()
		values_to_predict = np.array(values).reshape(1, -1)
		print ('values_to_predict')
		print (values_to_predict)
		prediction = self.model.predict(values_to_predict)
		prediction_str = 'Failure'
		prediction_value = "NON-FAILURE"
		if prediction[0] == 1 :
			prediction_value = "FAILURE" 
		probabilities = self.model.predict_proba(values_to_predict)
		print('prediction')
		print(prediction[0])

		st.markdown(
			"""
			<style>
			.header-style {
				font-size:20px;
				font-family:sans-serif;
			}
			</style>
			""",
			unsafe_allow_html=True
		)

		st.markdown(
			"""
			<style>
			font-style {
				font-size:18px;
				font-family:sans-serif;
			}
			</style>
			""",
			unsafe_allow_html=True
		)
		st.markdown(
			'<p class="header-style"> Predictive Maintenance Data Predictions </p>',
			unsafe_allow_html=True
		)
		
		st.markdown(
			"""
			<style>
				.sidebar .sidebar-content {{
					width: 375px;
				}}
			</style>
			""",
			unsafe_allow_html=True
		)

		column_1, column_2, column_3 = st.columns(3)
		
		column_1.markdown(
			f'<p class="font-style" >Prediction Name </p>',
			unsafe_allow_html=True
		)
		column_1.write(f"{prediction_str}")
		
		column_2.markdown(
			f'<p class="font-style" >Prediction Value </p>',
			unsafe_allow_html=True
		)
		column_2.write(f"{prediction_value}")

		column_3.markdown(
			'<p class="font-style" >Probability </p>',
			unsafe_allow_html=True
		)
		column_3.write(f"{probabilities[0][prediction[0]]}")

		fig = self.plot_pie_chart(probabilities)
		st.markdown(
			'<p class="font-style" >Predictive Maintenance - Probability Distribution</p>',
			unsafe_allow_html=True
		)
		st.plotly_chart(fig, use_container_width=True)

		return self


sa = StreamlitApp()
sa.construct_app()
print ("END...")







