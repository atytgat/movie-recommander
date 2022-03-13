import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import statistics
import matplotlib.pyplot as plt
import tikzplotlib


class User(object):

    def __init__(self, id, all_movies):
        self.id = id
        self.all_movies = all_movies
        self.ratings_given = []
        self.n_ratings = 0
        self.mean_rating = 0
        self.median_rating = 0

    # ratings_given[i] = [id_movie, movie_rating]
    def generate_ratings_given(self, movie_list):
        ratings_list = []
        for m in range(len(self.all_movies)):
            if self.all_movies[m] > 0:
                self.ratings_given.append([movie_list[m], self.all_movies[m]])
                self.mean_rating += self.all_movies[m]
                ratings_list.append(self.all_movies[m])
        self.n_ratings = len(self.ratings_given)
        self.mean_rating /= self.n_ratings
        self.median_rating = statistics.median(ratings_list)

class Movie(object):

    def __init__(self, id, all_users):
        self.id = id
        self.all_users = all_users
        self.ratings_recieved = []
        self.n_ratings = 0

    # ratings_recieved[i] = [id_user, movie_rating]
    def generate_ratings_recieved(self, user_list):
        for u in range(len(self.all_users)):
            if self.all_users[u] > 0:
                self.ratings_recieved.append([user_list[u], self.all_users[u]])
        self.n_ratings = len(self.ratings_recieved)

def itemrank(data_df, alpha, prec):
    """
    :param data: a dataframe of ratings of size (nuser, nmovie)
    :param alpha:
    :param prec: thresholf for convergence
    :return: numpy prediction matrix
    """
    data_df_copy = pd.DataFrame.copy(data_df)
    movie_list = []           # list of all the movies as object with initially only their id
    user_list = []            # list of all the users as object with initially only their id

    nuser, nmovie = data_df_copy.shape

    for m in range(nmovie):
        id = m+1
        all_users = data_df_copy[id].to_numpy()    # all ratings data (recieved or not) for movie_id = id
        movie = Movie(id, all_users)
        movie_list.append(movie)

    for u in range(nuser):
        id = u+1
        all_movies =  data_df_copy.T[id].to_numpy()    # all ratings data (given or not) for user_id = id
        user = User(id, all_movies)
        user_list.append(user)

    for m in movie_list:
        m.generate_ratings_recieved(user_list)

    for u in user_list:
        u.generate_ratings_given(movie_list)

    # Correlation Matrix CM computation
    CM = np.zeros((nmovie, nmovie))
    for m in movie_list:
        for u in m.ratings_recieved:
            for mrated in u[0].ratings_given:
                if mrated[0].id != m.id:
                    CM[m.id - 1][mrated[0].id - 1] += 1

    # divide each element by the sum of elements of its column
    # this way C is a stochastic matrix
    for c in range(len(CM[0,:])):
        if (sum(CM[:,c]) != 0):
            CM[:,c] /= sum(CM[:,c])

    # Prediction of movie ranking
    pred = np.zeros((nuser, nmovie))
    k = 0
    for u in user_list:
        "not normalized in algo, normalized in paper"
        d = u.all_movies
        d_rel = d - u.median_rating
        "ones in algo, ones normalized in paper"
        IR = np.ones(nmovie)
        converged = False
        ite = 0
        while not converged:
            ite += 1
            old_IR = IR
            IR = alpha * np.dot(CM, IR) + (1-alpha) * d_rel
            converged = ( abs(old_IR - IR) < prec ).all()

        pred[k] = IR
        k += 1

    # find the maximum and the minimum values of pred
    maxi = -1000
    mini = 1000
    for i in range(nuser):
        if max(pred[i]) > maxi:
            maxi = max(pred[i])
        if min(pred[i]) < mini:
            mini = min(pred[i])

    # transform the ranking values into integer ratings
    for i in range(nuser):
        for j in range(nmovie):
            pred[i][j] = transform_to_ratings_to_int(maxi, mini, pred[i][j])
            # pred[i][j] = transform_to_ratings(maxi, mini, pred[i][j])

    return pred

def transform_to_ratings(maximum, minimum, value):
    #  f(a)=c and f(b)=d
    # f(t) = c + (d-c)/(b-a) * (t-a)
    return 1 + 4/(maximum - minimum) * (value - minimum)

def transform_to_ratings_to_int(maximum, minimum, value):
    #  f(a)=c and f(b)=d
    # f(t) = c + (d-c)/(b-a) * (t-a)
    return int(round(1 + 4/(maximum - minimum) * (value - minimum)))

def transform_to_ratings_new(pred):

    nuser, nmovie = pred.shape

    mid = 3.5
    for i in range(nuser):
        maxi = max(pred[i])
        median = statistics.median(pred[i])
        mini = min(pred[i])
        for j in range(nmovie):
            if pred[i][j] <= median:
                pred[i][j] = int(round(1 + (mid-1)/(median - mini) * (pred[i][j] - mini)))
            else:
                pred[i][j] = int(round(mid + (5-mid)/(maxi - median) * (pred[i][j] - median)))

    return pred


def compute_MSE(np_ratings, np_preds):
    ratings_flat = np_ratings.flatten()
    preds_flat = np_preds.flatten()
    mse_tot = 0
    nb_ratings = 0

    for i in range(len(ratings_flat)):
        # if the rating is available
        if (ratings_flat[i] > 0):
            diff = (ratings_flat[i] - preds_flat[i])**2
            mse_tot += diff
            nb_ratings += 1

    return mse_tot/nb_ratings

def compute_MAE(np_ratings, np_preds):
    ratings_flat = np_ratings.flatten()
    preds_flat = np_preds.flatten()
    mae_tot = 0
    nb_ratings = 0

    for i in range(len(ratings_flat)):
        # if the rating is available
        if (ratings_flat[i] > 0):
            diff = abs(ratings_flat[i] - preds_flat[i])
            mae_tot += diff
            nb_ratings += 1

    return mae_tot/nb_ratings

def alpha_tuning():
    "This data set consists of:\
    * 100,000 ratings (1-5) from 943 users on 1682 movies.\
    * Each user has rated at least 20 movies. "
    links_df = pd.read_csv("ml-100k/u.data", sep="\t", header = 0, names = ["userId", "movieId", "rating", "timeStamp"], index_col=False)
    movie_ratings_df=links_df.pivot_table(index='movieId',columns='userId',values='rating').fillna(0).T
    k = 4  # number of folds for the cv

    alpha = np.arange(0.2, 1.2, 0.2)
    mean_mse_test = []
    mean_mae_test = []

    for a in alpha:
        print(a)

        mse_train = np.zeros(k)
        mae_train = np.zeros(k)
        mse_test = np.zeros(k)
        mae_test = np.zeros(k)

        kf = KFold(n_splits=k)
        i = 0
        for train, test in kf.split(links_df):
            print('fold ' + str(i+1))
            train_set_links = links_df.iloc[train] # select index of the training set
            test_set_links = links_df.iloc[test]   # select index of the test set

            # training set : create the rating matrix and add the missing columns and rows; missing values are replaced by 0
            train_set = train_set_links.pivot_table(index='movieId',columns='userId',values='rating').fillna(0).T
            train_set = train_set.reindex(list(range(movie_ratings_df.T.index.min(),movie_ratings_df.T.index.max()+1)),fill_value=0, axis='columns')
            train_set = train_set.reindex(list(range(movie_ratings_df.index.min(),movie_ratings_df.index.max()+1)),fill_value=0)
            train_set = train_set.astype(float)

            # test set : create the rating matrix and add the missing columns and rows; missing values are replaced by 0
            test_set = test_set_links.pivot_table(index='movieId',columns='userId',values='rating').fillna(0).T
            test_set = test_set.reindex(list(range(movie_ratings_df.T.index.min(),movie_ratings_df.T.index.max()+1)),fill_value=0, axis='columns')
            test_set = test_set.reindex(list(range(movie_ratings_df.index.min(),movie_ratings_df.index.max()+1)),fill_value=0)
            test_set = test_set.astype(float)

            # prediction
            train_df_copy = pd.DataFrame.copy(train_set)
            prediction = itemrank(train_df_copy, a, 10**-4)

            # performance evaluation on the training set
            mse_train[i] = compute_MSE(train_set.to_numpy(), prediction)
            mae_train[i] = compute_MAE(train_set.to_numpy(), prediction)

            # performance evaluation on the test set
            mse_test[i] = compute_MSE(test_set.to_numpy(), prediction)
            mae_test[i] = compute_MAE(test_set.to_numpy(), prediction)


            i += 1

        mean_mse_test.append(mse_test.mean())
        mean_mae_test.append(mae_test.mean())

    print(mean_mse_test)
    print(mean_mae_test)
    mean_mse_test = [2.575415482219289, 2.1176508012320494, 2.0177296675867034, 2.318582531701268, 2.812867477099084]
    mean_mae_test = [1.2923028713148526, 1.1584015304612183, 1.0974809464378574, 1.1603915216608665, 1.3499532909316372]
    alpha = [0.2, 0.4, 0.6, 0.8, 1]
    plt.plot(alpha, mean_mse_test, "-b", label="MSE")
    plt.plot(alpha, mean_mae_test, "-r", label="MAE")
    plt.xlabel('aplha values')
    plt.ylabel('MSE')
    plt.legend(loc="upper right")
    plt.title('alpha paraeter tuning')
    tikzplotlib.save('Latex/graph_alpha_tuning.tex')
    plt.show()


def full_prediction():
    "This data set consists of:\
    * 100,000 ratings (1-5) from 943 users on 1682 movies.\
    * Each user has rated at least 20 movies. "
    links_df = pd.read_csv("ml-100k/u.data", sep="\t", header = 0, names = ["userId", "movieId", "rating", "timeStamp"], index_col=False)
    movie_ratings_df=links_df.pivot_table(index='movieId',columns='userId',values='rating').fillna(0).T

    prediction = itemrank(movie_ratings_df, 0.6 , 10**-4)
    print(prediction)

    mse = compute_MSE(movie_ratings_df.to_numpy(), prediction)
    mae = compute_MAE(movie_ratings_df.to_numpy(), prediction)
    print(mse)
    print(mae)


if __name__ == '__main__':
    full_prediction()




