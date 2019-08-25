
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('A', 'B', 'C', 'D')

y_pos = np.arange(len(objects))

def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len(4))
    plt.bar(index, 4)
    plt.xlabel('Evaluation Type', fontsize=5)
    plt.ylabel('Percentage', fontsize=5)
    plt.xticks(index, la, fontsize=5, rotation=30)
    plt.title('Market Share for Each Genre 1995-2017')
    plt.show()

class three:

    performance_right = [1004, 181 , 773 , 230] # Right eye centre
    performance_left = [1008, 181 , 585 , 262] # Left eye centre
    performance_mouth = [1279, 5 , 657 , 243]   #
    performance_right_test = [656, 100 , 225 , 19] # Right eye centre
    performance_left_test = [669, 109 , 200 , 22] # Left eye centre
    performance_mouth_test = [795, 0 , 194 , 11]


class six:

    performance_right = [1580, 25, 702, 183]  # Right eye centre
    performance_left = [1589, 19, 557, 171]  # Left eye centre
    performance_mouth = [1568, 4, 615, 188]  #
    performance_right_test = [950, 10, 40, 0]  # Right eye centre
    performance_left_test = [933, 15, 51, 1]  # Left eye centre
    performance_mouth_test = [965, 4, 31, 0]


#plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of Episodes')
plt.title('Training Results for Right Eye-Centre point')

plt.xlabel("Episode Outcome")
plt.show()
