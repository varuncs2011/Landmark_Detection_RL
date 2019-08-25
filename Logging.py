
class Logging:
    def __init__(self):
        pass

    def log_train_history(self, history_train, experiment_name):
        episode_count = 0
        with open('Train_' + experiment_name + '.txt', 'w') as f:
            for key, val in history_train.history.items():
                episode_count = len(val)
                for s in val:
                    f.write(key + ":" + str(s))
                    f.write("\n")
                f.write("******")
        return episode_count


    def log_test_history(self, history_test, experiment_name):
        with open('Test_' + experiment_name + '.txt', 'w') as f:
            for key, val in history_test.history.items():
                for s in val:
                    f.write(key + ":" + str(s))
                    f.write("\n")
                f.write("******")

    def log_Test_results(self, totalPass, totalFail, wander, nb_episodes,experiment_name):

        with open('Results_Test' + experiment_name + '.txt', 'w') as f:
            f.write("Total Pass:" + str(totalPass))
            f.write("\n")
            f.write("Total Fail:" + str(totalFail))
            f.write("\n")
            f.write("Total Wander:" + str(wander))
            f.write("\n")
            f.write("Total MaxEpisodeReached:" + str(nb_episodes - totalPass - totalFail - wander))







