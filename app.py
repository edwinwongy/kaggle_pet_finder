import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('input/train.csv')
train_df = train_df.assign(testtrain='train')
test_df = pd.read_csv('input/test.csv')
test_df = test_df.assign(testtrain='test')

# Combine into one big array
combined_df = train_df.append(test_df, sort=False)


def visualise_data(datset):
    quantitative_train_df = datset.drop(['Name', 'Description', 'PetID', 'RescuerID', 'State'], axis=1)
    columns: list = quantitative_train_df.columns.values.tolist()
    print(columns)
    age = train_df['Age']
    # Target

    # Displaying plots for all input
    for column in columns:
        print('Displaying Column:', column)
        sns.distplot(train_df[column], kde=False)
        plt.show()


def visualise_animal_type():
    combined_df['Type'] = combined_df['Type'].map({1: 'Dog', 2: 'Cat'})
    plt.xlabel('dataset_type')
    plt.ylabel('count')
    plt.title('No. of cats and dogs from respective dataset')

    # plt.bar(train_df['Type'], type_series)

    sns.countplot(data=combined_df, x='testtrain', hue='Type')
    plt.show()


def visualise_adoption_speed():
    plt.xlabel('dataset_type')
    plt.ylabel('count')
    plt.title('No. of cats and dogs from respective dataset')

    # plt.bar(train_df['Type'], type_series)

    sns.countplot(data=combined_df, x='testtrain', hue='Type')
    plt.show()


def visualise_count_plot(df, x_axis_column, title=None, split_by=None):
    # Plot title
    if title is not None:
        plt.title(title)

    sns.countplot(data=df, x=x_axis_column, hue=split_by)
    plt.show()


def visualise_histogram_plot(df, x_axis_column, title=None, kde=False):
    if title is not None:
        plt.title(title)
    sns.distplot(df[x_axis_column], kde=kde)
    plt.show()


# Perform EDA

# # Find gender ratio
# train_df['Gender'] = train_df['Gender'].map({1: 'Male', 2: 'Female', 3: 'Mixed'})
# train_df['Type'] = train_df['Type'].map({1: 'Dog', 2: 'Cat'})
#
# # Number of cats and dogs
# visualise_count_plot(train_df, 'Type', 'Type of pets')
# # Display Gender
# visualise_count_plot(train_df, 'Gender', 'Gender of Pets')
#
# # Display adoption speed of cats dogs
# visualise_count_plot(train_df, 'AdoptionSpeed', split_by='Type', title='Adoption speed of cats and dogs')
#
# # Display vaccinated
# visualise_count_plot(train_df, 'Vaccinated', 'Vaccinated pets')
#
# # Display distribution of age of pets
# visualise_histogram_plot(train_df, 'Age', title='Histogram of pets age')
# print('Age range: %s months to %s months' % (train_df['Age'].min(), train_df['Age'].max()))
#
# # Bin months into years
# years = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
# train_df['AgeInYear'] = pd.cut(train_df['Age'], 22,
#                                    labels=years)
#
# # visualise_histogram_plot(train_df, 'AgeInYear', 'Age in years')
#
# # Adoption rate for all animals
# visualise_count_plot(train_df, 'AdoptionSpeed', title='Adoption rate for all animals')
#
# # Display adoption speed by age
# visualise_count_plot(train_df, 'AdoptionSpeed', split_by='AgeInYear', title='Adoption speed based on age')

# Select features to use
features = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
            'Color3', 'MaturitySize', 'Vaccinated', 'Dewormed', 'AdoptionSpeed',
            'Sterilized', 'Health', 'Quantity', 'Fee']

train_df = train_df[[feature for feature in features if feature in train_df.columns]]
x_train =train_df.drop('AdoptionSpeed', axis=1)
y_df = train_df['AdoptionSpeed']

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0)
tree.fit(x_train, y_df)
print('Accuracy on training {}'.format(tree.score(x_train, y_df)))
print('Predict {}'.format(tree.predict(x_train)))
result_df = test_df[[feature for feature in features if feature in test_df.columns]]
test_df['Predicted'] = tree.predict(result_df)
submission = test_df[['PetID', 'Predicted']]
submission.to_csv('submission.csv', index=False)