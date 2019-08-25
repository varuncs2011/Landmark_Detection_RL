import multiprocessing as mp


class Parallel:
    def __init__(self):
        output = mp.Queue()


# define a example function
    def test(self, dqn, test_env, nb_episodes, nb_max_episode_steps, visualize):
        history_test = dqn.test(test_env, nb_episodes, nb_max_episode_steps, visualize)

    def createProcess(self,dqn, test_env, nb_episodes, nb_max_episode_steps, visualize):
        processes = [
            mp.Process(target=self.test, args=(dqn, test_env,300,100,False))
            for x in range(4)]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        results = [self.output.get() for p in processes]
        print(results)



