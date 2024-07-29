import project1 as p1
import utils
import numpy as np

#-------------------------------------------------------------------------------
# 2. Hinge Loss on One Data Sample
#-------------------------------------------------------------------------------

# feature_vector = np.array([1., 1.])
# label = 1.
# theta = np.array([-1., -1.])
# theta_0 = 0.
# hinge_loss = p1.hinge_loss_single(feature_vector, label, theta, theta_0)

#-------------------------------------------------------------------------------
# 2. The Complete Hinge Loss
#-------------------------------------------------------------------------------

# feature_matrix = np.array([[1., -1.], [-1., 1.]])
# labels = np.array([1., -1.])
# theta = np.array([[1., -1.], [-1., 1.]])
# theta_0 = np.array([1., 1.])
# hinge_loss = p1.hinge_loss_full(feature_matrix, labels, theta, theta_0)

#-------------------------------------------------------------------------------
# 3. Perceptron Single Step Update
#-------------------------------------------------------------------------------

# feature_vector = np.array([1., 2.])
# label = 1.
# current_theta = np.array([-1., -1.])
# current_theta_0 = 0.

# feature_vector = np.array([0.22782035, -0.40478725,  0.43807799,  0.42014005,  0.00329309,  0.19814613,
#                            0.03042681, -0.36701387, -0.4368788,  -0.21080741])
# label = -1
# current_theta = np.array([0.03323687,  0.14710972, -0.49769777, -0.33920351, -0.02404346,  0.14141823,
#                   -0.09854503,  0.36992849, -0.38952173,  0.13559639])
# current_theta_0 = 0.1455625480557411

# theta, theta_0 = p1.perceptron_single_step_update(
#         feature_vector,
#         label,
#         current_theta,
#         current_theta_0)

#-------------------------------------------------------------------------------
# random data
#-------------------------------------------------------------------------------

# n = 200  # need to be an even number
# sigma = 0.75
#
# # feature_matrix = np.random.rand(n,2)
# feature_matrix_pos_x = np.random.normal(random.uniform(-1.0, 1.0), sigma, size=int(n/2))
# feature_matrix_neg_x = np.random.normal(random.uniform(-1.0, 1.0), sigma, size=int(n/2))
# feature_matrix_x = np.concatenate((feature_matrix_pos_x, feature_matrix_neg_x))
#
# feature_matrix_pos_y = np.random.normal(random.uniform(-1.0, 1.0), sigma, size=int(n/2))
# feature_matrix_neg_y = np.random.normal(random.uniform(-1.0, 1.0), sigma, size=int(n/2))
# feature_matrix_y = np.concatenate((feature_matrix_pos_y, feature_matrix_neg_y))
#
# feature_matrix = np.concatenate((
#     feature_matrix_x.reshape(-1, 1),
#     feature_matrix_y.reshape(-1, 1)),
#     axis=1)  # column-wise concatenation
#
# #labels
# labels_pos = np.sign(np.random.normal(+0.5, 0.1, size=int(n/2)))  # skewed to negative
# labels_neg = np.sign(np.random.normal(-0.5, 0.1, size=int(n/2)))  # skewed to positive
# labels = np.concatenate((labels_pos, labels_neg))

#-------------------------------------------------------------------------------
# 3. Full Perceptron Algorithm
#-------------------------------------------------------------------------------

# feature_matrix = np.array(
#     [[1., 1.],
#      [2., 2.]])
# labels = np.array([1., -1.])
# theta = np.array([[1., -1.], [-1., 1.]])
# theta_0 = np.array([1., 1.])

# # number of cycles
# T = 5

# # full perceptron algorithm
# theta, theta_0 = p1.perceptron(feature_matrix, labels, T)

# # Average Perceptron Algorithm
# theta, theta_0 = p1.average_perceptron(feature_matrix, labels, T)

#-------------------------------------------------------------------------------
# 4. Pegasos Algorithm
#-------------------------------------------------------------------------------

# feature_vector = np.array([0.22782035, -0.40478725,  0.43807799,  0.42014005,  0.00329309,  0.19814613,
#                            0.03042681, -0.36701387, -0.4368788,  -0.21080741])
# label = -1

# feature_matrix = np.array(
#     [[1., 1.],
#      [-1.5, -0.5]])
# labels = np.array([1., -1.])
# theta = np.array([[1., -1.], [-1., 1.]])
# theta_0 = np.array([1., 1.])

# T = 100
# L = 1
# eta = 1
# current_theta = np.zeros(feature_matrix.shape)
# current_theta_0 = 0.

# # Pegasos single step update
# for i in range(labels.shape[0]):
#     theta, theta_0 = p1.pegasos_single_step_update(
#         feature_matrix[i],
#         labels[i],
#         L,
#         eta,
#         current_theta[i],
#         current_theta_0)

# # Pegasos full
# theta, theta_0 = p1.pegasos(feature_matrix, labels, T, L)

#-------------------------------------------------------------------------------
# plotting (only for 2 Parameters!)
#-------------------------------------------------------------------------------

# # figure
# fig, ax = plt.subplots()
#
# # 2D points
# for i in range(labels.size):
#     if labels[i] > 0:
#         ax.scatter(feature_matrix[i, 0], feature_matrix[i, 1], c='b', s=4)
#     else:
#         ax.scatter(feature_matrix[i, 0], feature_matrix[i, 1], c='r', s=4)
#
# # linear 2D classifier
# x = np.linspace(-4, 4, 100)
# y = (-theta[0]*x - theta_0)/theta[1]
# ax.plot(x, y, c='m', lw=0.4)
#
# # config and plot figure
# lim_x = 2
# lim_y = 2
# ax.set(xlabel='x_1', ylabel='x_2')
# ax.set(xlim=(-lim_x, lim_x), ylim=(-lim_y, lim_y))
# ax.grid(linestyle='--')
# ax.set_aspect('equal', 'box')
# plt.show()

#-------------------------------------------------------------------------------
# animation (optional)
#-------------------------------------------------------------------------------
#
# # create a line plot
# x = np.linspace(-4, 4, 100)
# y = np.zeros(x.shape)
# line, = ax.plot(x, y)
# point = ax.scatter(0, 0)
#
# # Create a text object for the legend
# legend_text = ax.text(0.05, 0.85, "", transform=ax.transAxes)
# # legend_text = fig.text(0.05, 0.9, "", transform=fig.transFigure)  # outside the plotting box
#
# # initialize characteristic vector
# global_theta = np.zeros(feature_matrix.shape[1])
# global_theta_0 = 0.
# global_random = np.random.permutation(200)
#
#
# # define the update function
# def update(frame, point, line, feature_matrix, labels):
#
#     # declare global parameters
#     global global_theta
#     global global_theta_0
#     global global_random
#
#     # init values
#     legend_text.set_text("frame: {:d}\ntheta: [{:.2f}, {:.2f}]\ntheta_0: {:.2f}"
#                          .format(frame, global_theta[0], global_theta[1], global_theta_0))
#
#     # workaround for several T iterations
#     if frame < 200:
#         i = frame
#     elif frame >= 200 and frame < 400:
#         i = frame - 200
#     elif frame >= 400 and frame < 600:
#         i = frame - 400
#
#     # Check if the current frame number exceeds the stopping point
#     if frame >= 199:
#         # Stop the animation
#         ani.event_source.stop()
#
#     # update the y-data of the line plot
#     # theta, theta_0 = p1.perceptron(feature_matrix, labels, 1)
#     theta, theta_0 = p1.perceptron_single_step_update(feature_matrix[global_random[i]],
#                                                       labels[global_random[i]],
#                                                       global_theta,
#                                                       global_theta_0)
#
#     # update global parameters
#     global_theta = theta
#     global_theta_0 = theta_0
#
#     point.set_offsets(feature_matrix[i])
#     line.set_ydata((-theta[0]*x - theta_0)/theta[1])
#     # legend_text.set_text("frame: {:d}\ntheta: [{:.2f}, {:.2f}]\ntheta_0: {:.2f}"
#     #                      .format(frame, theta[0], theta[1], theta_0))
#     return point, line, legend_text
#
#
# # create the animation
# ani = FuncAnimation(fig, update,
#                     fargs=(point, line, feature_matrix, labels),
#                     frames=n, blit=True, interval=50)
#
# ani.save('animation.gif', writer='pillow')
#
# fig.subplots_adjust(top=0.85)
# plt.show()

#-------------------------------------------------------------------------------
# Problem 5
#-------------------------------------------------------------------------------

# toy_features, toy_labels = toy_data = utils.load_toy_data('toy_data.tsv')
#
# T = 10
# L = 0.2
#
# x = np.linspace(0, 1, T)
# y1 = np.zeros(T)
# y2 = np.zeros(T)
# y3 = np.zeros(T)
#
# for i in range(1, T):
#     thetas_perceptron = p1.perceptron(toy_features, toy_labels, i)
#     thetas_avg_perceptron = p1.average_perceptron(toy_features, toy_labels, i)
#     thetas_pegasos = p1.pegasos(toy_features, toy_labels, i, L)
#
#     y1[i] = thetas_perceptron[0][0]
#     y2[i] = thetas_avg_perceptron[0][0]
#     y3[i] = thetas_pegasos[0][0]
#
# # plot the curves with different colors and labels
# plt.plot(x, y1, color='blue', label='thetas_perceptron')
# plt.plot(x, y2, color='green', label='thetas_avg_perceptron')
# plt.plot(x, y3, color='red', label='thetas_pegasos')
#
# # add a legend
# plt.legend()
#
# # show the plot
# plt.show()

# def plot_toy_results(algo_name, thetas):
#     print('theta for', algo_name, 'is', ', '.join(map(str,list(thetas[0]))))
#     print('theta_0 for', algo_name, 'is', str(thetas[1]))
#     # utils.plot_toy_data(algo_name, toy_features, toy_labels, thetas)
#
# plot_toy_results('Perceptron', thetas_perceptron)
# plot_toy_results('Average Perceptron', thetas_avg_perceptron)
# plot_toy_results('Pegasos', thetas_pegasos)

#-------------------------------------------------------------------------------
# Problem 7
#-------------------------------------------------------------------------------

# n = 200  # need to be an even number
# sigma = 0.75
#
# # feature_matrix = np.random.rand(n,2)
# feature_matrix_pos_x = np.random.normal(random.uniform(-1.0, 1.0), sigma, size=int(n/2))
# feature_matrix_neg_x = np.random.normal(random.uniform(-1.0, 1.0), sigma, size=int(n/2))
# feature_matrix_x = np.concatenate((feature_matrix_pos_x, feature_matrix_neg_x))
#
# feature_matrix_pos_y = np.random.normal(random.uniform(-1.0, 1.0), sigma, size=int(n/2))
# feature_matrix_neg_y = np.random.normal(random.uniform(-1.0, 1.0), sigma, size=int(n/2))
# feature_matrix_y = np.concatenate((feature_matrix_pos_y, feature_matrix_neg_y))
#
# feature_matrix = np.concatenate((
#     feature_matrix_x.reshape(-1, 1),
#     feature_matrix_y.reshape(-1, 1)),
#     axis=1)  # column-wise concatenation

# feature_matrix = np.array(
#     [[ 1. ,  1. , 1, 1, 1],
#      [-1.5, -0.5, 1, 1, 1],
#      [-1.5, -0.5, 1, 1, 1],
#      [-1.5, -0.5, 1, 1, 1],
#      [-1.5, -0.5, 1, 1, 1]])
# theta = np.array([1., -1., 1., -1., 1.])
# theta_0 = 0.
#
# labels = p1.classify(feature_matrix, theta, theta_0)

#-------------------------------------------------------------------------------
# Data loading. There is no need to edit code in this section.
#-------------------------------------------------------------------------------

train_data = utils.load_data('reviews_train.tsv')
val_data = utils.load_data('reviews_val.tsv')
test_data = utils.load_data('reviews_test.tsv')

train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))

# create list of stopwords
with open("stopwords.txt", "r") as f:
    stopwords = f.readlines()
stopwords = [word.strip() for word in stopwords]

# my_test_dict = p1.bag_of_words(train_texts[1], 'the')  # output is letter by letter as train_texts[1] is str
# my_test_dict = p1.bag_of_words(train_texts[:1], stopwords)  # output is word by word as train_texts[1] is tuple
dictionary = p1.bag_of_words(train_texts, stopwords, remove_stopword=True)

train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary, binarize=False)
val_bow_features = p1.extract_bow_feature_vectors(val_texts, dictionary, binarize=False)
test_bow_features = p1.extract_bow_feature_vectors(test_texts, dictionary, binarize=False)

T = 10
L = 0.01

# using Perceptron algorithm
pct_train_accuracy, pct_val_accuracy = p1.classifier_accuracy(
    p1.perceptron,
    train_bow_features,
    val_bow_features,
    train_labels,
    val_labels,
    T=T)
print("{} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
print("{} {:.4f}".format("Validation accuracy for perceptron:", pct_val_accuracy))

# using Average Perceptron algorithm
avg_pct_train_accuracy, avg_pct_val_accuracy = p1.classifier_accuracy(
    p1.average_perceptron,
    train_bow_features,
    val_bow_features,
    train_labels,
    val_labels,
    T=T)
print("{} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
print("{} {:.4f}".format("Validation accuracy for average perceptron:", avg_pct_val_accuracy))

# using Pegasos algorithm
avg_peg_train_accuracy, avg_peg_val_accuracy = p1.classifier_accuracy(
    p1.pegasos,
    train_bow_features,
    val_bow_features,
    train_labels,
    val_labels,
    T=T,
    L=L)
print("{} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
print("{} {:.4f}".format("Validation accuracy for Pegasos:", avg_peg_val_accuracy))

#-------------------------------------------------------------------------------
# Problem 8
#-------------------------------------------------------------------------------

data = (train_bow_features, train_labels, val_bow_features, val_labels)

# values of T and lambda to try
Ts = [1, 5, 10, 15, 25, 50]
Ls = [0.001, 0.01, 0.1, 1, 10]

pct_tune_results = utils.tune_perceptron(Ts, *data)
print('perceptron valid:', list(zip(Ts, pct_tune_results[1])))
print('best = {:.4f}, T={:.4f}'.format(np.max(pct_tune_results[1]), Ts[np.argmax(pct_tune_results[1])]))

avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)
print('avg perceptron valid:', list(zip(Ts, avg_pct_tune_results[1])))
print('best = {:.4f}, T={:.4f}'.format(np.max(avg_pct_tune_results[1]), Ts[np.argmax(avg_pct_tune_results[1])]))

# fix values for L and T while tuning Pegasos T and L, respective
fix_L = 0.01
peg_tune_results_T = utils.tune_pegasos_T(fix_L, Ts, *data)
print('Pegasos valid: tune T', list(zip(Ts, peg_tune_results_T[1])))
print('best = {:.4f}, T={:.4f}'.format(np.max(peg_tune_results_T[1]), Ts[np.argmax(peg_tune_results_T[1])]))

fix_T = Ts[np.argmax(peg_tune_results_T[1])]
peg_tune_results_L = utils.tune_pegasos_L(fix_T, Ls, *data)
print('Pegasos valid: tune L', list(zip(Ls, peg_tune_results_L[1])))
print('best = {:.4f}, L={:.4f}'.format(np.max(peg_tune_results_L[1]), Ls[np.argmax(peg_tune_results_L[1])]))

# utils.plot_tune_results('Perceptron', 'T', Ts, *pct_tune_results)
# utils.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)
# utils.plot_tune_results('Pegasos', 'T', Ts, *peg_tune_results_T)
# utils.plot_tune_results('Pegasos', 'L', Ls, *peg_tune_results_L)

#-------------------------------------------------------------------------------
# Use the best method (perceptron, average perceptron or Pegasos) along with
# the optimal hyperparameters according to validation accuracies to test
# against the test dataset. The test data has been provided as
# test_bow_features and test_labels.
#-------------------------------------------------------------------------------

test_train_accuracy, test_val_accuracy, best_theta = p1.classifier_accuracy_return_theta(
    p1.pegasos,
    train_bow_features,
    test_bow_features,
    train_labels,
    test_labels,
    T=25,
    L=0.01)

print("{} {:.4f}".format("Training accuracy for Pegasos using best Hyperparameters:", test_train_accuracy))
print("{} {:.4f}".format("Validation accuracy for Pegasos using best Hyperparameters", test_val_accuracy))

#-------------------------------------------------------------------------------
# Assign to best_theta, the weights (and not the bias!) learned by your most
# accurate algorithm with the optimal choice of hyperparameters.
#-------------------------------------------------------------------------------

#best_theta = None # obtained in the function classifier_accuracy_return_tetha
wordlist = [word for (idx, word) in sorted(zip(dictionary.values(), dictionary.keys()))]
sorted_word_features = utils.most_explanatory_word(best_theta, wordlist)
print("Most Explanatory Word Features")
print(sorted_word_features[:10])

#-------------------------------------------------------------------------------
# Problem 9
#-------------------------------------------------------------------------------

