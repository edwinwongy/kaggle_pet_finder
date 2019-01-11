import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plot_lib as plot_lib

train_df = pd.read_csv('data/train.csv')
train_df = train_df.assign(testtrain='train')
test_df = pd.read_csv('data/test.csv')
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


def visualise_count_plot(df, x_axis_column, title=None, primary_category=None, stack_category=None):
    if title is not None:
        plt.title(title)
    if primary_category is not None and stack_category is not None:
        # Unique data in category
        bar_categories = df[primary_category].unique()
        dfs = []
        # loop through category and create df for each
        for value in bar_categories:
            df = df.loc[df[primary_category] == value]
            print(df)
            dfs.append(df)
        plot_lib.plot_clustered_stacked(dfs, bar_categories)
    else:
        sns.countplot(data=df, x=x_axis_column, hue=primary_category)
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

# Display Gender
# visualise_count_plot(train_df, 'Gender', 'Gender of Pets')

# Display adoption speed of cats dogs
# visualise_count_plot(train_df, 'AdoptionSpeed', primary_category='Type', title='Adoption speed of cats and dogs')

# Display adoption speed by age
# visualise_count_plot(train_df, 'AdoptionSpeed', hue='Age', title='Adoption speed based on age')

# Display vaccinated
# visualise_count_plot(train_df, 'Vaccinated', 'Vaccinated pets')
# Display distribution of age of pets
# visualise_histogram_plot(train_df, 'Age', title='Histogram of pets age')
# print('Age range: %s months to %s months' % (train_df['Age'].min(), train_df['Age'].max()))

# Bin months into years
train_df['BinnedAgeYear'] = pd.cut(train_df['Age'], 22,
                                   labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                           10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])

# visualise_histogram_plot(train_df, 'BinnedAgeYear', 'Age in years')

visualise_count_plot(train_df, 'AdoptionSpeed',
                     primary_category='Type',
                     stack_category='Type2',
                     title='Adoption speed of cats and dogs')