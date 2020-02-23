import matplotlib.pyplot as plt
import csv


def parse(path):
    reader = csv.reader(open(path), delimiter=',')
    line_count = 0
    episode = []
    avg_reward = []
    min_reward = []
    max_reward = []
    epislon = []
    for row in reader:
        if line_count == 0:
            line_count += 1
            continue
        line_count += 1
        episode.append(round(int(row[0]), 3))
        avg_reward.append(round(float(row[1]), 3))
        min_reward.append(round(float(row[2]), 3))
        max_reward.append(round(float(row[3]), 3))
    return (episode, avg_reward, min_reward, max_reward, epislon)


player1 = parse("out/test/player_1_stats.txt")
player2 = parse("out/test/player_2_stats.txt")

fig = plt.figure()

plt.subplot(2, 1, 1)
plt.plot(player1[0], player1[1],  label="avg_reward")
plt.plot(player1[0], player1[2],  label="min_reward")
plt.plot(player1[0], player1[3],  label="max_reward")
plt.legend(loc=1)

plt.subplot(2, 1, 2)
plt.plot(player2[0], player2[1],  label="avg_reward")
plt.plot(player2[0], player2[2],  label="min_reward")
plt.plot(player2[0], player2[3],  label="max_reward")
plt.show(block=True)

