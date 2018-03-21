from Game import *
from ANN import *
import random as rand
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import Image


class Parent:
    def __init__(self):
        self.wolf1 = Animal()
        self.wolf2 = Animal()
        self.rabbit = Animal()

        self.score = 0
        self.index = -1


class GA:
    def __init__(self, generations=100, size=100, mutate=0.2, retain=0.1, game_size=10, game_rounds=10, game_turns=10,
                 game_exits=5, game_move=0.1, filename="attempt"):
        self.game_size = game_size
        self.game_rounds = game_rounds
        self.game_turns = game_turns
        self.game_exits = game_exits
        self.game_move = game_move
        self.generations = generations
        self.size = size
        self.wolf_tol = 0.5
        self.filename = filename + '_' + str(self.generations) + '_' + str(self.size)
        self.fit = 0.4 - retain
        while not(self.size % (1/retain) == 0):  # ensure that the population sizes fits with the parent pool
            self.size = self.size + 1
        self.mutate = mutate  # chance to have one mutation in one of the parents children
        self.retain = retain  # chance to retain a bad individual
        self.rabbits = [Parent() for _ in range(int(self.size * 0.4))]
        self.wolves = [Parent() for _ in range(int(self.size * 0.4))]
        # each 'individual' is composed of 3 ANNs -> i.e. a game
        # rabbits are chosen by rabbit score
        # wolves are chosen by combined wolf score, i.e. ability to work together
        self.file = open(name, 'w')
        contents = str(self.population[0].wolf1.ann.layers) + "\n" + str(self.population[0].wolf1.ann.layers) + "\n" + str(self.population[0].rabbit.ann.layers)
        self.file.write(contents)

        self.population = [Game(size=self.game_size, rounds=self.game_rounds, turns=self.game_turns,
                                num_exits=self.game_exits, move_thresh=self.game_move) for _ in range(self.size)]

    def gen_pop(self):
        self.population = [Game(size=self.game_size, rounds=self.game_rounds, turns=self.game_turns,
                                num_exits=self.game_exits, move_thresh=self.game_move) for _ in range(self.size)]

    def set_filename(self, filename):
        self.filename = filename + '_' + str(self.generations) + '_' + str(self.size)

    def fitness(self):
        # zero containing arrays
        self.rabbits = [Parent() for _ in range(int(self.size * self.fit))]
        self.wolves = [Parent() for _ in range(int(self.size * self.fit))]
        # score each individual game
        for i in range(self.size):
            print "--> Individual:", i
            self.population[i].start()
            add_wolf = False  # added this wolf yet
            add_rabbit = False  # added this rabbit yet
            # add to parents if good enough
            for j in range(len(self.rabbits)):
                if self.wolves[j].index == -1 and not add_wolf:  # no wolves added yet to this parent
                    self.wolves[j].index = i  # add parent location in original population
                    self.wolves[j].score = self.population[i].score[0]  # add wolf score to parent
                    self.wolves[j].wolf1 = self.population[i].wolf1  # add wolf1
                    self.wolves[j].wolf2 = self.population[i].wolf2  # add wolf2
                    add_wolf = True
                elif self.wolves[j].score < self.population[i].score[0] and not add_wolf:  # if individual has higher score, squeeze him instead
                    self.wolves[j:j] = [Parent()]
                    self.wolves[j].index = i  # add parent location in original population
                    self.wolves[j].score = self.population[i].score[0]  # add wolf score to parent
                    self.wolves[j].wolf1 = self.population[i].wolf1  # add wolf1
                    self.wolves[j].wolf2 = self.population[i].wolf2  # add wolf2
                    add_wolf = True
                elif self.wolves[j].score == self.population[i].score[0] and not add_wolf:  # if tied score add individual if lower rabbit score
                    if self.population[i].rabbit.score < self.population[self.wolves[j].index].rabbit.score:
                        self.wolves[j:j] = [Parent()]
                        self.wolves[j].index = i  # add parent location in original population
                        self.wolves[j].score = self.population[i].score[0]  # add wolf score to parent
                        self.wolves[j].wolf1 = self.population[i].wolf1  # add wolf1
                        self.wolves[j].wolf2 = self.population[i].wolf2  # add wolf2
                        add_wolf = True
                if self.rabbits[j].index == -1 and not add_rabbit:  # no rabbits added yet to this parent
                    self.rabbits[j].index = i  # add parent location in original population
                    self.rabbits[j].score = self.population[i].rabbit.score  # add rabbit score to parent
                    self.rabbits[j].rabbit = self.population[i].rabbit  # add rabbit
                    add_rabbit = True
                elif self.rabbits[j].score < self.population[i].rabbit.score and not add_rabbit:  # if individual has higher score, squeeze him in instead
                    self.rabbits[j:j] = [Parent()]
                    self.rabbits[j].index = i  # add parent location in original population
                    self.rabbits[j].score = self.population[i].rabbit.score  # add rabbit score to parent
                    self.rabbits[j].rabbit = self.population[i].rabbit  # add rabbit
                    add_rabbit = True
                elif self.rabbits[j].score == self.population[i].rabbit.score and not add_rabbit:  # if tied score add individual if lower wolf score
                    if self.population[i].score[0] < self.population[self.rabbits[j].index].score[0]:
                        self.rabbits[j:j] = [Parent()]
                        self.rabbits[j].index = i  # add parent location in original population
                        self.rabbits[j].score = self.population[i].rabbit.score  # add rabbit score to parent
                        self.rabbits[j].rabbit = self.population[i].rabbit  # add rabbit
                        add_rabbit = True
                if len(self.wolves) > (self.size * self.fit):  # too many wolf parents
                    self.wolves = self.wolves[0:int(self.size * self.fit)]  # crop parents
                if len(self.rabbits) > (self.size * self.fit):  # too many rabbit parents
                    self.rabbits = self.rabbits[0:int(self.size * self.fit)]  # crop parents

            # print "Population after run"
            # for j in range(self.size):
            #     print self.population[j].score
            # print "Parent wolf population after run"
            # for j in range(len(self.wolves)):
            #     print "Score:", self.wolves[j].score, "; Index:", self.wolves[j].index
            # print "Parent rabbit population after run"
            # for j in range(len(self.rabbits)):
            #     print "Score:", self.rabbits[j].score, "; Index:", self.rabbits[j].index

    def breed(self, p1=None, p2=None, mutate=0):
        p1 = p1 if not None else Animal()
        p2 = p2 if not None else Animal()

        P1 = p1.ann
        P2 = p2.ann
        children = [Animal(num_layers=P1.num_layers, bias=P1.bias, activation=P1.activation,
                           layers=P1.layers) for _ in range(3)]

        for i in range(P1.num_layers - 1):
            for j in range(len(P1.weights[i])):
                if mutate == 1:
                    rand_mutate = rand.uniform(0, 1)
                    if rand_mutate < self.mutate:
                        mutate_val = rand.gauss(0, 5)
                        p1.ann.weights[i][j] = p1.ann.weights[i][j] + mutate_val
                if mutate == 2:
                    rand_mutate = rand.uniform(0, 1)
                    if rand_mutate < self.mutate:
                        mutate_val = rand.gauss(0, 5)
                        p2.ann.weights[i][j] = p2.ann.weights[i][j] + mutate_val
                if mutate == 3:
                    rand_mutate = rand.uniform(0, 1)
                    if rand_mutate < self.mutate:
                        mutate_val = rand.gauss(0, 5)
                        p1.ann.weights[i][j] = p1.ann.weights[i][j] + mutate_val
                    rand_mutate = rand.uniform(0, 1)
                    if rand_mutate < self.mutate:
                        mutate_val = rand.gauss(0, 5)
                        p2.ann.weights[i][j] = p2.ann.weights[i][j] + mutate_val

        mutate_pos = -1
        mutate_val = 0
        mutate_child = 0
        rand_mutate = rand.uniform(0, 1)
        if rand_mutate < self.mutate:
            mutate_child = rand.randint(0, 2)
            mutate_val = rand.gauss(0, 5)  # mutate random number in gaussian distribution
            total_weights = -1
            for i in range(P1.num_layers - 1):  # run through all layers of weights
                total_weights = total_weights + len(P1.weights[i])
            mutate_pos = rand.randint(0, total_weights)  # mutate one weight amongst all children

        total_weights = -1
        for i in range(P1.num_layers - 1):  # run through all layers of weights
            for j in range(len(P1.weights[i])):  # run through all weights in the layer
                total_weights = total_weights + 1

                children[0].ann.weights[i][j] = (P1.weights[i][j] + P2.weights[i][j])/2  # first child is average of the weights

                rand_parent = rand.randint(0, 1)
                children[1].ann.weights[i][j] = P1.weights[i][j] if rand_parent == 0 else P2.weights[i][j]  # randomely choose a parent and take their weight

                min = np.min([P1.weights[i][j], P2.weights[i][j]])
                max = np.max([P1.weights[i][j], P2.weights[i][j]])
                children[2].ann.weights[i][j] = rand.uniform(min, max)  # generate a weight between parents weight values

                if mutate_pos == total_weights:  # mutate if necessary
                    children[mutate_child].ann.weights[i][j] = children[mutate_child].ann.weights[i][j] + mutate_val

        return children

    def optimize(self):
        name = self.filename + '.txt'
        self.file = open(name, 'w')
        self.file.close()

        wolf_score = []
        rabbit_score = []
        best_wolf_score = 0
        best_wolf_pop = 0
        best_rabbit_score = 0
        best_rabbit_pop = 0
        time_taken = 0
        for i in range(self.generations):
            print "\n-> Generation:", i
            sstart = timer()
            self.fitness()  # run the game for each individual and the calculate the parent population

            name = self.filename + '.txt'
            self.file = open(name, 'a')
            # add best ANN to file
            for j in range(self.rabbits[0].rabbit.ann.num_layers - 1):
                for weight in self.rabbits[0].rabbit.ann.weights[j]:
                    self.file.write("%f " % weight)
                self.file.write("\n")
            for j in range(self.wolves[0].wolf1.ann.num_layers - 1):
                for weight in self.wolves[0].wolf1.ann.weights[j]:
                    self.file.write("%f " % weight)
                self.file.write("\n")
            for j in range(self.wolves[0].wolf2.ann.num_layers - 1):
                for weight in self.wolves[0].wolf2.ann.weights[j]:
                    self.file.write("%f " % weight)
                self.file.write("\n")
            self.file.close()

            # add some weaker individuals back into the population
            for j in range(int(self.size * self.retain)):
                rabbit_parents = [self.rabbits[k].index for k in
                                  range(int(self.size * self.fit))]  # list of indices of taken parents
                wolves_parents = [self.wolves[k].index for k in
                                  range(int(self.size * self.fit))]  # list of indices of taken parents

                # add new rabbit
                rand_rabbit = rand.randint(0, self.size - 1)
                while rand_rabbit in rabbit_parents:  # not already used
                    rand_rabbit = rand.randint(0, self.size - 1)
                rabbit_parents.append(rand_rabbit)
                new_rabbit = Parent()
                new_rabbit.index = rand_rabbit
                new_rabbit.score = self.population[rand_rabbit].rabbit.score
                new_rabbit.rabbit = self.population[rand_rabbit].rabbit
                self.rabbits.append(new_rabbit)

                # add new wolf
                rand_wolves = rand.randint(0, self.size - 1)
                while rand_wolves in wolves_parents:  # not already used
                    rand_wolves = rand.randint(0, self.size - 1)
                wolves_parents.append(rand_wolves)
                new_wolves = Parent()
                new_wolves.index = rand_wolves
                new_wolves.score = self.population[rand_wolves].score[0]
                new_wolves.wolf1 = self.population[rand_wolves].wolf1
                new_wolves.wolf2 = self.population[rand_wolves].wolf2
                self.wolves.append(new_wolves)

            wolf_score.append(np.sum([self.population[j].score[0] for j in range(self.size)]))
            rabbit_score.append(np.sum([self.population[j].rabbit.score for j in range(self.size)]))
            if best_wolf_score < wolf_score[i]:
                best_wolf_score = wolf_score[i]
                best_wolf_pop = i
            if best_rabbit_score < rabbit_score[i]:
                best_rabbit_score = rabbit_score[i]
                best_rabbit_pop = i

            # create new population and populate it with parents and breed children
            for j in range(len(self.rabbits)):
                self.population.append(Game(size=self.game_size, rounds=self.game_rounds, turns=self.game_turns,
                                num_exits=self.game_exits, move_thresh=self.game_move))
                self.population[j].rabbit = self.rabbits[j].rabbit
                self.population[j].wolf1 = self.wolves[j].wolf1
                self.population[j].wolf2 = self.wolves[j].wolf2
            # start breeding parents
            r_used = []
            w_used = []
            for j in range(int(len(self.rabbits) / 2)):
                # randomely select parents to breed -> not necessarily 2 in a row
                # P_r = []
                # P_r_cumulate = 0.0
                # P_w = []
                # P_w_cumulate = 0.0
                #
                # score_r = 0
                # score_w = 0
                #
                # # run through all rabbits to get total current score
                # for k in range(int(len(self.rabbits))):
                #     if k not in r_used:
                #         score_r = score_r + self.rabbits[k].score
                #     if k not in w_used:
                #         score_w = score_w + self.wolves[k].score
                #
                # # run through all rabbits to compile a probability chart of unused scores
                # for k in range(int(len(self.rabbits))):
                #     if k not in r_used:
                #         p_r = self.rabbits[k].score/float(score_r)
                #         P_r_cumulate = P_r_cumulate + p_r
                #         P_r.append([P_r_cumulate, k])  # add cumulative probability and wolf index
                #     if k not in w_used:
                #         p_w = self.wolves[k].score/float(score_w)
                #         P_w_cumulate = P_w_cumulate + p_w
                #         P_w.append([P_w_cumulate, k])
                #
                # # randomly select two parents using the cumulative probability
                # rand_r1 = rand.uniform(0, 1)
                # rand_r2 = rand.uniform(0, 1)
                # rand_w1 = rand.uniform(0, 1)
                # rand_w2 = rand.uniform(0, 1)
                # index_r1 = -1
                # index_r2 = -2
                # index_w1 = -1
                # index_w2 = -2
                #
                # # retrieve the indices of the parents
                # while index_r1 == index_r2 or index_w1 == index_w2:
                #     if index_r1 == index_r2:
                #         rand_r2 = rand.uniform(0, 1)
                #         index_r2 = -2
                #     if index_w1 == index_w2:
                #         rand_w2 = rand.uniform(0, 1)
                #         index_w2 = -2
                #     # find the parents
                #     for k in range(len(P_r)):
                #         if rand_r1 < P_r[k][0] and index_r1 == -1:
                #             index_r1 = P_r[k][1]
                #         if rand_r2 < P_r[k][0] and index_r2 == -2:
                #             index_r2 = P_r[k][1]
                #         if rand_w1 < P_w[k][0] and index_w1 == -1:
                #             index_w1 = P_w[k][1]
                #         if rand_w2 < P_w[k][0] and index_w2 == -2:
                #             index_w2 = P_w[k][1]
                #
                # # these indices have now been used
                # r_used.append(index_r1)
                # r_used.append(index_r2)
                # w_used.append(index_w1)
                # w_used.append(index_w2)
                #
                # mutate = 0
                # # print self.wolves[j].wolf2.score
                # # print self.wolves[j].wolf1.score
                # if float(1 + self.wolves[index_w1].wolf2.score)/(1 + self.wolves[index_w1].wolf1.score) < self.wolf_tol:
                #     self.wolves[index_w1].wolf2 = self.wolves[index_w1].wolf1
                #     mutate = mutate + 1
                # elif float(1 + self.wolves[index_w1].wolf1.score)/(1 + self.wolves[index_w1].wolf2.score) < self.wolf_tol:
                #     self.wolves[index_w1].wolf1 = self.wolves[index_w1].wolf2
                #     mutate = mutate + 1
                # if float(1 + self.wolves[index_w2].wolf2.score)/(1 + self.wolves[index_w2].wolf1.score) < self.wolf_tol:
                #     self.wolves[index_w2].wolf2 = self.wolves[index_w2].wolf1
                #     mutate = mutate + 2
                # elif float(1 + self.wolves[index_w2].wolf1.score) / (1 + self.wolves[index_w2].wolf2.score) < self.wolf_tol:
                #     self.wolves[index_w2].wolf1 = self.wolves[index_w2].wolf2
                #     mutate = mutate + 2
                #
                # children1 = self.breed(self.rabbits[index_r1].rabbit, self.rabbits[index_r2].rabbit)
                # children2 = self.breed(self.wolves[index_w1].wolf1, self.wolves[index_w2].wolf1)
                # children3 = self.breed(self.wolves[index_w1].wolf2, self.wolves[index_w2].wolf2, mutate)

                mutate = 0
                # print self.wolves[j].wolf2.score
                # print self.wolves[j].wolf1.score
                if float(1 + self.wolves[j].wolf2.score) / (
                        1 + self.wolves[j].wolf1.score) < self.wolf_tol:
                    self.wolves[j].wolf2 = self.wolves[j].wolf1
                    mutate = mutate + 1
                elif float(1 + self.wolves[j].wolf1.score) / (
                        1 + self.wolves[j].wolf2.score) < self.wolf_tol:
                    self.wolves[j].wolf1 = self.wolves[j].wolf2
                    mutate = mutate + 1
                if float(1 + self.wolves[j + 1].wolf2.score) / (
                        1 + self.wolves[j + 1].wolf1.score) < self.wolf_tol:
                    self.wolves[j + 1].wolf2 = self.wolves[j + 1].wolf1
                    mutate = mutate + 2
                elif float(1 + self.wolves[j + 1].wolf1.score) / (
                        1 + self.wolves[j + 1].wolf2.score) < self.wolf_tol:
                    self.wolves[j + 1].wolf1 = self.wolves[j + 1].wolf2
                    mutate = mutate + 2

                children1 = self.breed(self.rabbits[j].rabbit, self.rabbits[j + 1].rabbit)
                children2 = self.breed(self.wolves[j].wolf1, self.wolves[j + 1].wolf1)
                children3 = self.breed(self.wolves[j].wolf2, self.wolves[j + 1].wolf2, mutate)
                for k in range(3):
                    self.population[len(self.rabbits) + j * 3 + k].rabbit = children1[k]
                    self.population[len(self.rabbits) + j * 3 + k].wolf1 = children2[k]
                    self.population[len(self.rabbits) + j * 3 + k].wolf2 = children3[k]

            eend = timer()
            print "Time taken for generation:", (eend - sstart), "s"
            time_taken = time_taken + (eend - sstart)
            print "Total time taken:", time_taken, "s"

            # print "Wolf score after run:", wolf_score[i], "    ;    Rabbit score after run:", rabbit_score[i]
            # print "Population after run"
            # for j in range(self.size):
            #     print self.population[j].score
            # print "Parent wolf population after run"
            # for j in range(len(self.wolves)):
            #     print "Score:", self.wolves[j].score, "; Index:", self.wolves[j].index
            # print "Parent rabbit population after run"
            # for j in range(len(self.rabbits)):
            #     print "Score:", self.rabbits[j].score, "; Index:", self.rabbits[j].index
        print "\nBest wolf score was:", best_wolf_score, "in generation", best_wolf_pop
        print "Best rabbit score was:", best_rabbit_score, "in generation", best_rabbit_pop
        print "Average time taken per generation was:", time_taken/self.generations, "s"

        plt.figure()
        x = np.arange(0, self.generations)
        line1, = plt.plot(x, wolf_score, label="Wolves")
        line2, = plt.plot(x, rabbit_score, label="Rabbit")
        plt.xlabel("Generation")
        plt.ylabel("Score")
        plt.legend(handles=[line1, line2])
        name = self.filename + '.png'
        plt.savefig(name)
        # plt.show()


ga = GA(generations=100, size=50, game_rounds=50, game_exits=4, game_move=0, filename="attempt")
start = timer()

times = 5
for i in range(0, times):
    print "<=====================> Starting ANN set number:", i, "<=====================>"
    ga.set_filename("ANN_21_03_0\Attempt_" + str(i))
    ga.gen_pop()
    ga.optimize()

    end = timer()
    print "Set time taken:", (end - start), "s"

end = timer()
print "Total time taken:", (end - start), "s"






















