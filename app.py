import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('data/train.csv')
print('Training data columns:', train_df.columns.values)
train_df = train_df.assign(testtrain='train')
test_df = pd.read_csv('data/test.csv')
print('Testing data columns:', test_df.columns.values)
test_df = test_df.assign(testtrain='test')

# Combine into one big array
combined_df = train_df.append(test_df, sort=False)


def visualise_data(datset):
    quantitative_train_df = datset.drop(['Name', 'Description', 'PetID', 'RescuerID', 'State'], axis=1)
    columns: list = quantitative_train_df.columns.values.tolist()
    print(columns)
    age = train_df['Age']
    # Target

    # Displaying plots for all data
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


def visualise_count_plot(df, x_axis_column, title=None, hue=None):
    if title is not None:
        plt.title(title)
    sns.countplot(data=df, x=x_axis_column, hue=hue)
    plt.show()


def visualise_histogram_plot(df, x_axis_column, title=None, kde=False):
    if title is not None:
        plt.title(title)
    sns.distplot(df[x_axis_column], kde=kde)
    plt.show()


# Perform EDA

# Find gender ratio
train_df['Gender'] = train_df['Gender'].map({1: 'Male', 2: 'Female', 3: 'Mixed'})
train_df['Type'] = train_df['Type'].map({1: 'Dog', 2: 'Cat'})


visualise_count_plot(train_df, 'Gender', 'Gender of Pets')

visualise_count_plot(train_df, 'AdoptionSpeed', hue='Type', title='Adoption speed of cats and dogs')

visualise_count_plot(train_df, 'AdoptionSpeed', hue='Age', title='Adoption speed based on age')

# Display distribution of age of pets
visualise_histogram_plot(train_df, 'Age', title='Histogram of pets age')

# TODO getting max min from panda series
print(train_df['Age'].max)
# print('Age range: %s months to %s months' % (train_df['Age'].min, train_df['Age'].max))
