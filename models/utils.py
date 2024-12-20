def distance(pos1, pos2):
    sum = 0
    for i in range(len(pos1)):
        sum += (pos1[i] - pos2[i]) ** 2
    return sum ** 0.5
