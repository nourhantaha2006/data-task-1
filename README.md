# data-task-1




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
df = pd.read_csv(r'dataset titanic/data tasks/Titanic-Dataset.csv')
df.head()
print(df.isnull().sum())

# تعويض القيم الناقصة في العمر بالمتوسط
df['Age'].fillna(df['Age'].median(), inplace=True)

# حذف عمود الكابينة لأنه يحتوي على الكثير من القيم المفقودة
df.drop(columns=['Cabin', 'Name', 'Ticket'], inplace=True)

# تعويض القيم الناقصة في الميناء بالقيمة الأكثر شيوعًا
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# تحويل الجنس إلى أرقام (ذكر = 0، أنثى = 1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# تحويل الميناء إلى أرقام
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print(df.head())
import matplotlib.pyplot as plt
import seaborn as sns

# مقارنة النجاة حسب الجنس
sns.barplot(x='Sex', y='Survived', data=df)
plt.title("life percentage according to sex")
plt.show()

# مقارنة النجاة حسب الدرجة
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("life percentage according to class")

plt.show()
X = df.drop(columns=['Survived'])  # كل البيانات ما عدا العمود المستهدف
y = df['Survived'] 
df.head()
rom sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression

# تدريب النموذج
model = LogisticRegression()
model.fit(X_train, y_train)

# التنبؤ على بيانات الاختبار
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

# حساب الدقة
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy_score {accuracy:.2f}")

# مصفوفة الارتباك
conf_matrix = confusion_matrix(y_test, y_pred)
print("confusion matrix")
print(conf_matrix)

new_passenger = pd.DataFrame([[3, 0, 22, 1, 0, 7.25, 0, 0, 0]],  # أضفنا عمودًا إضافيًا
                             columns=X.columns)
print(X.columns)
print(len(X.columns))  # عدد الأعمدة الفعلي

import numpy as np

# إنشاء راكب جديد بنفس عدد الأعمدة وبقيم عشوائية مقبولة
new_passenger = pd.DataFrame(np.array([[3, 0, 22, 1, 0, 7.25, 0, 0, 0]]), columns=X.columns)

# طباعة الشكل النهائي للبيانات
print(new_passenger)

df.head()

new_passenger = pd.DataFrame([[3, 0, 22, 1, 0, 7.25, 0, 0, 0]],
                             columns=X.columns)

prediction = model.predict(new_passenger)
print("survived" if prediction[0] == 1 else "not survived")
