import os
import sys
import time
import random
import math

def read_file_into_string(input_file, from_ord, to_ord):
    # take a file "input_file", read it character by character, strip away all unwanted
    # characters with ord < "from_ord" and ord > "to_ord" and return the concatenation
    # of the file as the string "output_string"
    the_file = open(input_file,'r')
    current_char = the_file.read(1)
    output_string = ""
    while current_char != "":
        if ord(current_char) >= from_ord and ord(current_char) <= to_ord:
            output_string = output_string + current_char
        current_char = the_file.read(1)
    the_file.close()
    return output_string

def stripped_string_to_int(a_string):
    # take a string "a_string" and strip away all non-numeric characters to obtain the string
    # "stripped_string" which is then converted to an integer with this integer returned
    a_string_length = len(a_string)
    stripped_string = "0"
    if a_string_length != 0:
        for i in range(0,a_string_length):
            if ord(a_string[i]) >= 48 and ord(a_string[i]) <= 57:
                stripped_string = stripped_string + a_string[i]
    resulting_int = int(stripped_string)
    return resulting_int

def get_string_between(from_string, to_string, a_string, from_index):
    # look for the first occurrence of "from_string" in "a_string" starting at the index
    # "from_index", and from the end of this occurrence of "from_string", look for the first
    # occurrence of the string "to_string"; set "middle_string" to be the sub-string of "a_string"
    # lying between these two occurrences and "to_index" to be the index immediately after the last
    # character of the occurrence of "to_string" and return both "middle_string" and "to_index"
    middle_string = ""              # "middle_string" and "to_index" play no role in the case of error
    to_index = -1                   # but need to initialized to something as they are returned
    start = a_string.find(from_string,from_index)
    if start == -1:
        flag = "*** error: " + from_string + " doesn't appear"
        #trace_file.write(flag + "\n")
    else:
        start = start + len(from_string)
        end = a_string.find(to_string,start)
        if end == -1:
            flag = "*** error: " + to_string + " doesn't appear"
            #trace_file.write(flag + "\n")
        else:
            middle_string = a_string[start:end]
            to_index = end + len(to_string)
            flag = "good"
    return middle_string,to_index,flag

def string_to_array(a_string, from_index, num_cities):
    # convert the numbers separated by commas in the file-as-a-string "a_string", starting from index "from_index",
    # which should point to the first comma before the first digit, into a two-dimensional array "distances[][]"
    # and return it; note that we have added a comma to "a_string" so as to find the final distance
    # distance_matrix = []
    if from_index >= len(a_string):
        flag = "*** error: the input file doesn't have any city distances"
        #trace_file.write(flag + "\n")
    else:
        row = 0
        column = 1
        row_of_distances = [0]
        flag = "good"
        while flag == "good":
            middle_string, from_index, flag = get_string_between(",", ",", a_string, from_index)
            from_index = from_index - 1         # need to look again for the comma just found
            if flag != "good":
                flag = "*** error: there aren't enough cities"
                # trace_file.write(flag + "\n")
            else:
                distance = stripped_string_to_int(middle_string)
                row_of_distances.append(distance)
                column = column + 1
                if column == num_cities:
                    distance_matrix.append(row_of_distances)
                    row = row + 1
                    if row == num_cities - 1:
                        flag = "finished"
                        row_of_distances = [0]
                        for i in range(0, num_cities - 1):
                            row_of_distances.append(0)
                        distance_matrix.append(row_of_distances)
                    else:
                        row_of_distances = [0]
                        for i in range(0,row):
                            row_of_distances.append(0)
                        column = row + 1
        if flag == "finished":
            flag = "good"
    return flag

def make_distance_matrix_symmetric(num_cities):
    # make the upper triangular matrix "distance_matrix" symmetric;
    # note that there is nothing returned
    for i in range(1,num_cities):
        for j in range(0,i):
            distance_matrix[i][j] = distance_matrix[j][i]

# read input file into string

#######################################################################################################
############ now we read an input file to obtain the number of cities, "num_cities", and a ############
############ symmetric two-dimensional list, "distance_matrix", of city-to-city distances. ############
############ the default input file is given here if none is supplied via a command line   ############
############ execution; it should reside in a folder called "city-files" whether it is     ############
############ supplied internally as the default file or via a command line execution.      ############
############ if your input file does not exist then the program will crash.                ############

input_file = "AISearchfile058.txt"

#######################################################################################################

# you need to worry about the code below until I tell you; that is, do not touch it!

if len(sys.argv) == 1:
    file_string = read_file_into_string("../city-files/" + input_file,44,122)
else:
    input_file = sys.argv[1]
    file_string = read_file_into_string("../city-files/" + input_file,44,122)
file_string = file_string + ","         # we need to add a final comma to find the city distances
                                        # as we look for numbers between commas
print("I'm working with the file " + input_file + ".")
                                        
# get the name of the file

name_of_file,to_index,flag = get_string_between("NAME=", ",", file_string, 0)

if flag == "good":
    print("I have successfully read " + input_file + ".")
    # get the number of cities
    num_cities_string,to_index,flag = get_string_between("SIZE=", ",", file_string, to_index)
    num_cities = stripped_string_to_int(num_cities_string)
else:
    print("***** ERROR: something went wrong when reading " + input_file + ".")
if flag == "good":
    print("There are " + str(num_cities) + " cities.")
    # convert the list of distances into a 2-D array
    distance_matrix = []
    to_index = to_index - 1             # ensure "to_index" points to the comma before the first digit
    flag = string_to_array(file_string, to_index, num_cities)
if flag == "good":
    # if the conversion went well then make the distance matrix symmetric
    make_distance_matrix_symmetric(num_cities)
    print("I have successfully built a symmetric two-dimensional array of city distances.")
else:
    print("***** ERROR: something went wrong when building the two-dimensional array of city distances.")

#######################################################################################################
############ end of code to build the distance matrix from the input file: so now you have ############
############ the two-dimensional "num_cities" x "num_cities" symmetric distance matrix     ############
############ "distance_matrix[][]" where "num_cities" is the number of cities              ############
#######################################################################################################

# now you need to supply some parameters ...

#######################################################################################################
############ YOU NEED TO INCLUDE THE FOLLOWING PARAMETERS:                                 ############
############ "my_user_name" = your user-name, e.g., mine is dcs0ias                        ############

my_user_name = "clvp22"

############ "my_first_name" = your first name, e.g., mine is Iain                         ############

my_first_name = "Alex"

############ "my_last_name" = your last name, e.g., mine is Stewart                        ############

my_last_name = "Goodall"

############ "alg_code" = the two-digit code that tells me which algorithm you have        ############
############ implemented (see the assignment pdf), where the codes are:                    ############
############    BF = brute-force search                                                    ############
############    BG = basic greedy search                                                   ############
############    BS = best_first search without heuristic data                              ############
############    ID = iterative deepening search                                            ############
############    BH = best_first search with heuristic data                                 ############
############    AS = A* search                                                             ############
############    HC = hilling climbing search                                               ############
############    SA = simulated annealing search                                            ############
############    GA = genetic algorithm                                                     ############

alg_code = "GA"

############ you can also add a note that will be added to the end of the output file if   ############
############ you like, e.g., "in my basic greedy search, I broke ties by always visiting   ############
############ the first nearest city found" or leave it empty if you wish                   ############

added_note = "This GA algorithm consists of indiviuals that represent tours (their encodings are lists of n cities).\n The improved mating strategy is a less naive single point crossover method, the mutation process consists of randomised 4-Opt moves with probabalistic acceptance criterion.\n A 3-Opt local search optimization algorithm is applied to the encodings of the children after the mating process.\n This algorithm is signifigantly more computationally expensive than the basic version so the 3-Opt optimization is omitted for larger city sets.\n The population is also generated from greedy / nearest neighbour tours, and is only evolved once, because experimentally population size is more important than number of generations for GAs."

############ the line below sets up a dictionary of codes and search names (you need do    ############
############ nothing unless you implement an alternative algorithm and I give you a code   ############
############ for it when you can add the code and the algorithm to the dictionary)         ############

codes_and_names = {'BF' : 'brute-force search',
                   'BG' : 'basic greedy search',
                   'BS' : 'best_first search without heuristic data',
                   'ID' : 'iterative deepening search',
                   'BH' : 'best_first search with heuristic data',
                   'AS' : 'A* search',
                   'HC' : 'hilling climbing search',
                   'SA' : 'simulated annealing search',
                   'GA' : 'genetic algorithm'}

#######################################################################################################
############    now the code for your algorithm should begin                               ############
#######################################################################################################

class Individual(object):
    # the indiviual class represents a solution the the TSP as a list of cities
    def __init__(self, encoding, fitness):
        self.encoding = encoding # encoding is a list of cities that represent a tour
        self.fitness = fitness # the fitness of the individual

    # define how to compare individuals
    def __lt__(self, other):
        return self.f() < other.f()

    def __gt__(self,other):
        return self.f() > other.f()
    
    def __eq__(self,other):
        return self.f() == other.f()

    # f / fitness function - the value of of the tour encoded by the individual
    def f(self):
        if self.fitness == None:
            # compute the fitness value of the individual
            tour = self.encoding
            tour_length = tour_length_calc(self.encoding)
            self.fitness = tour_length
            return tour_length
        else:
            # if the fitness has already been computed return the value
            return self.fitness

class Generation(object):
    # the generation class represents a group of individuals from the same generation
    def __init__(self, number, population):
        # the population size corresponds to the number of greedy tours we generate
        self.number = number # the number of the generation (1st generation or 2nd ...)
        self.population = population # the population its self is a list of individuals
        self.N = num_cities # the number of individuals in the population

    def new(self):
        # create a newly generated population
        # we start from greedy tours instead of random tours
        # these yield better end tours
        # we create N greedy tours each starting at different cities
        self.number = 1
        # nearest neighbour / greedy completion
        for i in range(0, self.N):
            # for each i
            # create the unvisited list
            unvisited = list(range(0, num_cities))
            # remove i
            unvisited.remove(i)
            # create the visited list
            visited = [i]
            j = i
            G = 0
            while unvisited != []:
                k = j # k is the last visited city
                Z = distance_matrix[j][unvisited[0]]
                j = unvisited[0]
                for l in range(1, len(unvisited)):
                    if distance_matrix[k][unvisited[l]] < Z:
                        Z = distance_matrix[k][unvisited[l]]
                        j = unvisited[l]
                # j is the nearest neighbour to k
                # j has been visited and is now the last visited city
                unvisited.remove(j)
                visited.append(j)
                G += Z
            G += distance_matrix[visited[-1]][visited[0]]
            # append the greedy tour to the population
            self.population.append(Individual(visited, G))
    

    def next(self):
        # evolve the current generation creating a new population from the current population
        # create a probability distribution ...
        maximum = max(self.population).f()
        # calculate the value of the least fit individual in the population
            
        weight=[]
        # fix the weight so fittest get chosen
        # the smaller an individuals f value is the fitter it is
        # a smaller f value is a smaller tour which is what we wnat
        for individual in self.population:
            weight.append(maximum - individual.f()) # weight = max - fitness 

        # increment the generation number    
        self.number += 1
        # init the new population
        new_population = []
        # repeat this N times until we have a new population of N children
        for _ in range(self.N):
            # randomly select 2 individuals from the population
            # fitter ones are more likely to get picked
            lst = random.choices(self.population, weights=weight, k=2)
            parentX = lst[0]
            parentY = lst[1]
            # reproduce a child
            child = reproduce(parentX, parentY)
            # append the child to the new population
            new_population.append(child)
        # set the new population
        self.population = new_population
        
    def get_best(self):
        # return the fittest individual in the population
        return min(self.population)
            
def reproduce(X, Y):
    # create an offspring from 2 parents
    # improved crossover/mating strategy
    # use 4-Opt moves for mutations
    # use simulated annealing for randomised acceptance criterion
    # if mutated child is worse than parent accept with probability e^delta/T
    # apply a local optimisation algoritm Opt3 on offspring

    # choose a random crossover point
    crossover = random.randint((num_cities//10), ((num_cities-1)//2))
    donor = X.encoding # one parent is a donor
    receiver = Y.encoding # one parent is a receiver
    lst = donor[:crossover] # lst is our child encoding
    unvisited = donor[crossover:] # list of unvisited cities

    while unvisited != []: # while we have some unvisited cities
        c = lst[-1] # c is the last visited city in our encoding
        index = receiver.index(c)
        neighbourX = receiver[(index+1)%(num_cities)] # right neighbour of c in receiver
        neighbourY = receiver[(index-1)%(num_cities)] # left neighbour of c in receiver
        # if either of the neighbours of c can be found the encoding of receiver we append one of them
        if neighbourX in unvisited:
            # right neighbour wins in tie breaker
            lst.append(neighbourX)
            unvisited.remove(neighbourX)
            continue
        elif neighbourY in unvisited:
            lst.append(neighbourY)
            unvisited.remove(neighbourY)
            continue
        neighbourX = donor[(index+1)%(num_cities)] # right neighbour of c in donor
        neighbourY = donor[(index-1)%(num_cities)] # left neighbour of c in donor
        # if either of the neighbours of c can be found the encoding of receiver we append one of them
        if neighbourX in unvisited:
            # right neighbour wins in tie breaker
            lst.append(neighbourX)
            unvisited.remove(neighbourX)
            continue
        elif neighbourY in unvisited:
            lst.append(neighbourY)
            unvisited.remove(neighbourY)
            continue
        # otherwise we add the next unvisited node w.r.t the donors encoding
        lst.append(unvisited.pop(0))

    # improved mutation strategy
    best_parent_fitness = max([X,Y]).f() # fitness of of fittest parent
    temperature = 1.0 # set temperature to 1.0
    child_fitness = tour_length_calc(lst) # calculate the child's fitness at the moment
    while True:
        # we perform 4-Opt moves for mutation
        # that is we remove 4 edges and try to reconnect the 4 resulting paths in some other way
        # randomly choose 4 edges
        MI = random.sample(range(0, len(lst)), 4)
        # sort the 4 edges
        MI.sort()
        # create the 4 paths s1, s2, s3 and s4
        i, j, k, l = MI[0], MI[1], MI[2], MI[3]
        s1 = lst[l:]+lst[:i]
        s2 = lst[i:j]
        s3 = lst[j:k]
        s4 = lst[k:l]
        new_lst = []
        # randomly choose a 4-Opt move and perform it
        # recalculate the length of the new tour / fitness of the new child encoding
        # this part is directly coded for efficiency
        mutation_process = random.randint(0, 17)
        length = child_fitness - distance_matrix[s1[-1]][s2[0]] - distance_matrix[s2[-1]][s3[0]] - distance_matrix[s3[-1]][s4[0]] - distance_matrix[s4[-1]][s1[0]]
        if mutation_process == 0:#
            new_lst = s1 + s3 + s2 + s4[::-1]
            length = length + distance_matrix[s1[-1]][s3[0]] + distance_matrix[s3[-1]][s2[0]] + distance_matrix[s2[-1]][s4[-1]] + distance_matrix[s4[0]][s1[0]]
        if mutation_process == 1:#
            new_lst = s1 + s3 + s2[::-1] + s4[::-1]
            length = length + distance_matrix[s1[-1]][s3[0]] + distance_matrix[s3[-1]][s2[-1]] + distance_matrix[s2[0]][s4[-1]] + distance_matrix[s4[0]][s1[0]]
        if mutation_process == 2:#
            new_lst = s1 + s3 + s4[::-1] + s2
            length = length + distance_matrix[s1[-1]][s3[0]] + distance_matrix[s3[-1]][s4[-1]] + distance_matrix[s4[0]][s2[0]] + distance_matrix[s2[-1]][s1[0]]
        if mutation_process == 3:#
            new_lst = s1 + s2[::-1] + s3[::-1] + s4[::-1]
            length = length + distance_matrix[s1[-1]][s2[-1]] + distance_matrix[s2[0]][s3[-1]] + distance_matrix[s3[0]][s4[-1]] + distance_matrix[s4[0]][s1[0]]
        if mutation_process == 4:#
            new_lst = s1 + s2[::-1] + s4[::-1] + s3[::-1]
            length = length + distance_matrix[s1[-1]][s2[-1]] + distance_matrix[s2[0]][s4[-1]] + distance_matrix[s4[0]][s3[-1]] + distance_matrix[s3[0]][s1[0]]
        if mutation_process == 5:#
            new_lst = s1 + s2[::-1] + s3 + s4[::-1]
            length = length + distance_matrix[s1[-1]][s2[-1]] + distance_matrix[s2[0]][s3[0]] + distance_matrix[s3[-1]][s4[-1]] + distance_matrix[s4[0]][s1[0]]
        if mutation_process == 6:#
            new_lst = s1 + s3[::-1] + s2 + s4[::-1]
            length = length + distance_matrix[s1[-1]][s3[-1]] + distance_matrix[s3[0]][s2[0]] + distance_matrix[s2[-1]][s4[-1]] + distance_matrix[s4[0]][s1[0]]
        if mutation_process == 7:#
            new_lst = s1 + s3[::-1] + s4 + s2
            length = length + distance_matrix[s1[-1]][s3[-1]] + distance_matrix[s3[0]][s4[0]] + distance_matrix[s4[-1]][s2[0]] + distance_matrix[s2[-1]][s1[0]]
        if mutation_process == 8:#
            new_lst = s1 + s3[::-1] + s4[::-1] + s2
            length = length + distance_matrix[s1[-1]][s3[-1]] + distance_matrix[s3[0]][s4[-1]] + distance_matrix[s4[0]][s2[0]] + distance_matrix[s2[-1]][s1[0]]
        if mutation_process == 9:#
            new_lst = s1 + s4 + s3 + s2
            length = length + distance_matrix[s1[-1]][s4[0]] + distance_matrix[s4[-1]][s3[0]] + distance_matrix[s3[-1]][s2[0]] + distance_matrix[s2[-1]][s1[0]]
        if mutation_process == 10:#
            new_lst = s1 + s4 + s3[::-1] + s2
            length = length + distance_matrix[s1[-1]][s4[0]] + distance_matrix[s4[-1]][s3[-1]] + distance_matrix[s3[0]][s2[0]] + distance_matrix[s2[-1]][s1[0]]
        if mutation_process == 11:#
            new_lst = s1 + s4 + s2 + s3[::-1]
            length = length + distance_matrix[s1[-1]][s4[0]] + distance_matrix[s4[-1]][s2[0]] + distance_matrix[s2[-1]][s3[-1]] + distance_matrix[s3[0]][s1[0]]
        if mutation_process == 12:#
            new_lst = s1 + s4 + s2[::-1] + s3
            length = length + distance_matrix[s1[-1]][s4[0]] + distance_matrix[s4[-1]][s2[-1]] + distance_matrix[s2[0]][s3[0]] + distance_matrix[s3[-1]][s1[0]]
        if mutation_process == 13:#
            new_lst = s1 + s4 + s2[::-1] + s3[::-1]
            length = length + distance_matrix[s1[-1]][s4[0]] + distance_matrix[s4[-1]][s2[-1]] + distance_matrix[s2[0]][s3[-1]] + distance_matrix[s3[0]][s1[0]]
        if mutation_process == 14:#
            new_lst = s1 + s4[::-1] + s3 + s2
            length = length + distance_matrix[s1[-1]][s4[-1]] + distance_matrix[s4[0]][s3[0]] + distance_matrix[s3[-1]][s2[0]] + distance_matrix[s2[-1]][s1[0]]
        if mutation_process == 15:#
            new_lst = s1 + s4[::-1] + s2 + s3[::-1]
            length = length + distance_matrix[s1[-1]][s4[-1]] + distance_matrix[s4[0]][s2[0]] + distance_matrix[s2[-1]][s3[-1]] + distance_matrix[s3[0]][s1[0]]
        if mutation_process == 16:#
            new_lst = s1 + s4[::-1] + s2[::-1] + s3
            length = length + distance_matrix[s1[-1]][s4[-1]] + distance_matrix[s4[0]][s2[-1]] + distance_matrix[s2[0]][s3[0]] + distance_matrix[s3[-1]][s1[0]]
        if mutation_process == 17:#
            new_lst = s1 + s4[::-1] + s2[::-1] + s3[::-1]
            length = length + distance_matrix[s1[-1]][s4[-1]] + distance_matrix[s4[0]][s2[-1]] + distance_matrix[s2[0]][s3[-1]] + distance_matrix[s3[0]][s1[0]]
        # we implement randomised acceptance criterion for the child encoding
        delta =  best_parent_fitness - length # delta is the fitness of the fittest parent minus the length of the new tour / child encoding
        if delta > 0:
            # if delta is positive then our child is fitter than both its parents and we have found an improved tour so accept
            lst = new_lst
            child_fitness = length
            break
        else:
            if random.random() < math.e**(delta/temperature):
                # if delta is negative then accept the child encoding with probability e^delta/T
                lst = new_lst
                child_fitness = length
                break
            else:
                # if we fail in accept the child encoding then we try another random 4-Opt move at randomly selected edges
                # we increase the temperature
                temperature = temperature*2
    # finally init and return the individual
    # but first we apply a 3-Opt local optimization algorithm on the encoding
    if num_cities <= 90:
        # if N is greater than 90 the 3-Opt local search algorithm is too computationally expensive so we ommit it
        lst, child_fitness = local_search(lst, child_fitness)
    return Individual(lst,child_fitness)

def local_search(tour, tour_length):
    # this function performs the 3-Opt local search optimization algorithm on a given tour / encoding
    # 3-Opt is works by removing 3 edges and then reconnecting the 3 paths in a better way
    # we try this for every possible configuration of edges, this is O(n^3) ...
    # calculate the tour length of the current tour
    for i in range(1, len(tour) - 2):
        for j in range(i+1, len(tour)-1):
            for k in range(j+1, len(tour)):
                # create the 3 paths s1, s2, s3
                s1 = tour[k:]+tour[:i]
                s2 = tour[i:j]
                s3 = tour[j:k]
                # set the base length for quick calculation
                # try all possible 3-Opt moves in turn
                base_length = tour_length - distance_matrix[s1[-1]][s2[0]] - distance_matrix[s2[-1]][s3[0]] - distance_matrix[s3[-1]][s1[0]]
                new_tour = s1 + s3 + s2
                new_tour_length = base_length + distance_matrix[s1[-1]][s3[0]] + distance_matrix[s3[-1]][s2[0]] + distance_matrix[s2[-1]][s1[0]]
                if new_tour_length < tour_length:
                    # if we find a better tour set it to the current best tour
                    tour = new_tour
                    tour_length = new_tour_length
                # this part is directly coded for efficiency
                new_tour = s1 + s3 + s2[::-1]
                new_tour_length = base_length + distance_matrix[s1[-1]][s3[0]] + distance_matrix[s3[-1]][s2[-1]] + distance_matrix[s2[0]][s1[0]]
                
                if new_tour_length < tour_length:
                    tour = new_tour
                    tour_length = new_tour_length
                    
                new_tour = s1 + s2[::-1] + s3[::-1]
                new_tour_length = base_length + distance_matrix[s1[-1]][s2[-1]] + distance_matrix[s2[0]][s3[-1]] + distance_matrix[s3[0]][s1[0]]
                
                if new_tour_length < tour_length:
                    tour = new_tour
                    tour_length = new_tour_length
                    
                new_tour = s1 + s3[::-1] + s2
                new_tour_length = base_length + distance_matrix[s1[-1]][s3[-1]] + distance_matrix[s3[0]][s2[0]] + distance_matrix[s2[-1]][s1[0]]
                
                if new_tour_length < tour_length:
                    tour = new_tour
                    tour_length = new_tour_length
    # return the best tour
    return tour, tour_length

def tour_length_calc(tour):
    # function that calculates the length of a tour represented by a list / encoding
    tour_length = 0
    for i in range(0, num_cities-1):
        tour_length += distance_matrix[tour[i]][tour[i+1]]
    tour_length += distance_matrix[tour[num_cities-1]][tour[0]]
    return tour_length
        
def search():
    # init the population
    population = Generation(1, [])
    # create a new population
    population.new()
    # evolve the population N times
    N = 1
    for i in range(0, N):
        population.next()
    # select the best individual from the newest generation to be the answer
    return population.get_best()

solution = search()
# write the solution to the appropriate variables
if solution is not None:
    tour = solution.encoding
    tour_length = solution.f()
else:
    tour = ''
    tour_length = -1
    
#######################################################################################################
############ the code for your algorithm should now be complete and you should have        ############
############ computed a tour held in the list "tour" of length "tour_length"               ############
#######################################################################################################

# you do not need to worry about the code below; that is, do not touch it

#######################################################################################################
############ start of code to verify that the constructed tour and its length are valid    ############
#######################################################################################################

check_tour_length = 0
for i in range(0,num_cities-1):
    check_tour_length = check_tour_length + distance_matrix[tour[i]][tour[i+1]]
check_tour_length = check_tour_length + distance_matrix[tour[num_cities-1]][tour[0]]
flag = "good"
if tour_length != check_tour_length:
    flag = "bad"
if flag == "good":
    print("Great! Your tour-length of " + str(tour_length) + " from your " + codes_and_names[alg_code] + " is valid!")
else:
    print("***** ERROR: Your claimed tour-length of " + str(tour_length) + "is different from the true tour length of " + str(check_tour_length) + ".")

#######################################################################################################
############ start of code to write a valid tour to a text (.txt) file of the correct      ############
############ format; if your tour is not valid then you get an error message on the        ############
############ standard output and the tour is not written to a file                         ############
############                                                                               ############
############ the name of file is "my_user_name" + mon-dat-hr-min-sec (11 characters);      ############
############ for example, dcs0iasSep22105857.txt; if dcs0iasSep22105857.txt already exists ############
############ then it is overwritten                                                        ############
#######################################################################################################

if flag == "good":
    local_time = time.asctime(time.localtime(time.time()))   # return 24-character string in form "Tue Jan 13 10:17:09 2009"
    output_file_time = local_time[4:7] + local_time[8:10] + local_time[11:13] + local_time[14:16] + local_time[17:19]
                                                             # output_file_time = mon + day + hour + min + sec (11 characters)
    output_file_name = my_user_name + output_file_time + ".txt"
    f = open(output_file_name,'w')
    f.write("USER = " + my_user_name + " (" + my_first_name + " " + my_last_name + ")\n")
    f.write("ALGORITHM = " + alg_code + ", FILENAME = " + name_of_file + "\n")
    f.write("NUMBER OF CITIES = " + str(num_cities) + ", TOUR LENGTH = " + str(tour_length) + "\n")
    f.write(str(tour[0]))
    for i in range(1,num_cities):
        f.write("," + str(tour[i]))
    if added_note != "":
        f.write("\nNOTE = " + added_note)
    f.close()
    print("I have successfully written the tour to the output file " + output_file_name + ".")
    
    











    


