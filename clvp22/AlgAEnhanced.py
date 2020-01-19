import os
import sys
import time
import random
import copy
import heapq

start_time = time.time()

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

input_file = "AISearchfile175.txt"

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

alg_code = "AS"

############ you can also add a note that will be added to the end of the output file if   ############
############ you like, e.g., "in my basic greedy search, I broke ties by always visiting   ############
############ the first nearest city found" or leave it empty if you wish                   ############

added_note = "For this AS algorithm the node states represent lists of selected edges, the heuristic function is the greedy selection of edges so that a valid tour is produced. This algorithm improves of the pervious one because the greedy selection of edges heuristic in general gets closer to the optimal solution that the nearest neighbour heuristic. However, this algorithm is more computationally expensive, so there is an enforced early termination after 90 seconds of searching the state space."

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

# this version of A* search models the problem as a list of selected edges
# the state of the nodes are a list of tuples that represent edges between two cities

# create the edge matrix from the distance matrix
edge_matrix = []
for i in range(0, num_cities):
    for j in range(i+1, num_cities):
        edge_matrix.append((i, j, distance_matrix[i][j]))
# sort the edge matrix in ascending order of edge weight
sorted_edges = sorted(edge_matrix, key=lambda tup: tup[2])
# sorted_edges should be a list of tuples [(1,2,40), (1,3,43), ....]
# where (1,2,40) means an edge from city 1 to city 2 costs 40

class Node(object):
    # class for nodes that represent a list of selected edges
    # the selected edges represent a forest, or a tour if they are goal nodes
    def __init__(self, state, cost, node_degrees, valid_edges):
        self.state = state # list of selected edges
        self.cost = cost # the cost of selected edges
        self.node_degrees = node_degrees # degrees of nodes should not exceed 2
        self.heuristic = None # heurisitic value of state
        self.valid_edges = valid_edges # list of valid edges that if added do not violate any constraints

    # define comaprison between objects
    def __lt__(self, other):
        if self.f()  == other.f():
            return len(self.state) > len(other.state)
        return self.f() < other.f() 

    def __gt__(self,other):
        if self.f() == other.f():
           return len(self.state) < len(other.state)
        return self.f() > other.f() 
    
    def __eq__(self,other):
        return self.f()  == other.f()
    
    # evaluation funtcion - f function
    def f(self):
        return self.h() + self.g()
    
    # heuristic function - h function
    def h(self):
        if self.heuristic == None:
            edge_stack = copy.copy(self.valid_edges) # list of valid edges to try - already sorted
            edges = copy.copy(self.state) # the current list of edges for this node
            degrees = copy.copy(self.node_degrees) # the degrees of the cities for this node
            G = 0
            # value of the greedy edge selection / greedy completion (to create a valid tour)
            while len(edges) != num_cities:
                new_edge = edge_stack.pop(0)
                if valid_edge_insertion(edges, degrees, new_edge):
                    # if the edge is a valid insertion - select it
                    degrees[new_edge[0]] += 1
                    degrees[new_edge[1]] += 1
                else:
                    continue
                # append new edge if its insertion does not violate any constraints
                edges.append(new_edge)
                # sum the value
                G += new_edge[2]
            # save the heuristic value for later and return it
            self.heuristic = G
            return G
        else:
            # return the heuristic value if it has been computed already
            return self.heuristic
        
    # g function
    def g(self):
        return self.cost

    def complete(self):
        # this function greedily completes the node state - through greedy edge selection
        while len(self.state) != num_cities:
            new_edge = self.valid_edges.pop(0) # the list of valid edges is already sorted
            if valid_edge_insertion(self.state, self.node_degrees, new_edge):
                # if the edge is a valid insertion - select it
                self.node_degrees[new_edge[0]] += 1
                self.node_degrees[new_edge[1]] += 1
            else:
                continue
            # append new edge if its insertion does not violate any constraints
            self.state.append(new_edge)
            # sum the cost
            self.cost += new_edge[2]

    def isGoalNode(self):
        return len(self.state) == num_cities

    def toList(self):
        # this function return a list of cities / tour represented by the list of selected edges
        lst = []
        # start at the first edge
        nodeX = (self.state[0])[0]
        edges = copy.copy(self.state)
        # append the first two cities
        lst.append(nodeX)
        # node Y is our current node
        while True:
            # traverse along the edges
            next_edge = None
            for edge in edges:
                if edge[0] == nodeX:
                    # if node X is in some edge
                    # traverse that edge
                    next_edge = edge
                    nodeX = edge[1]
                    lst.append(nodeX)
                    break
                if edge[1] == nodeX:
                    # if node X is in some edge
                    # traverse that edge
                    next_edge = edge
                    nodeX = edge[0]
                    lst.append(nodeX)
                    break
            if next_edge == None:
                # if there are no more edges then stop traversing
                break
            # remove the used edge
            edges.remove(next_edge)
        return lst[:-1]
            
def valid_edge_insertion(state, node_degrees, edge):
    # this function checks if adding some given edge to some given state is a valid move
    edges = copy.copy(state)
    N = len(state) + 1
    # if the given edge has already been selected then this is not a valid move
    if edge in edges:
        return False
    
    # check degrees violation
    # if the degree of any of the cities exceeds 2 this not a valid move
    if node_degrees[edge[0]] == 2:
        return False
    if node_degrees[edge[1]] == 2:
        return False

    # check for cycles
    # if the insertion of this edge gives us a cycle this is not a valid move
    # unless the cycle is a goal state
    if node_degrees[edge[0]] == 0 or node_degrees[edge[1]] == 0:
        return not N == num_cities
                    
    cycle = False # assume no cycles
    
    nodeX = edge[0]
    nodeY = edge[1]
    
    while True:
        # we check if there is an other way to get from nodeX to nodeY
        # if there is this implies there is a cycle
        # we traverse from nodeX along our selected edges
        next_edge = None
        for edge in edges:
            if edge[0] == nodeX:
                next_edge = edge
                nodeX = edge[1]
                break
            if edge[1] == nodeX:
                next_edge = edge
                nodeX = edge[0]
                break
        if next_edge == None:
            # if there are now more edges to traverse then we do not have a cycle
            cycle = False
            break
        if nodeX == nodeY:
            # if we find nodeY while traversing then we have a cycle
            cycle = True
            break
        edges.remove(next_edge)
        
    if N == num_cities:
        # if we have a goal state we want to have a cycle
        return cycle
    else:
        # otherwise we don't want to have a cycle
        return not cycle

def get_valid_edges(node):
    # this function take a node state and returns its list of valid edges
    # it tries every edge in its valid edges list 
    valid_edges = []
    for edge in node.valid_edges:
        if valid_edge_insertion(node.state, node.node_degrees, edge):
            valid_edges.append(edge)
    return valid_edges
    
class PriorityQueue(object):
    # priority queue class used as the fringe
    # uses heapq for better performance
    def __init__(self):
        self.Q = []

    def __str__(self):
        return "".join([str(i) for i in self.Q])

    def isEmpty(self):
        return len(self.Q) == 0

    def push(self, obj):
        heapq.heappush(self.Q, obj)
            
    def pop(self):
        if (not self.isEmpty()):
            return heapq.heappop(self.Q)
        else:
            return None

def AStarSearch():
    # init the start node
    # (state = list of visited nodes, cost = cost of selected edges, node_degrees = the degrees of the cities, valid_edges = list of sorted valid edges)
    startNode = Node([],0, [0]*num_cities, sorted_edges)
    # init the fringe priority queue
    fringe = PriorityQueue()
    # is the start node a goal node? - not likely
    if startNode.isGoalNode():
        return startNode
    else:
        # push the start node to the fringe
        fringe.push(startNode)
        # while fringe is not empty
        while not fringe.isEmpty():
            # pop the next node
            node = fringe.pop()
            if node.isGoalNode():
                # if it is a goal node return - it must be minimal among the fringe
                # A* search says we terminate if our node is a goal node and minimal among all other nodes on the fringe
                return node
            if (time.time() - start_time) > 90:
                # if the search has been running longer than 90 seconds we enforce early termination
                # greedily complete the current node and return it
                node.complete()
                return node
            # get the list of valid edge selections for this node
            valid_edges = get_valid_edges(node)
            # iterate through the valid edges create all possible child nodes
            for edge in valid_edges:
                # get the cost of selecting this edge
                weight = edge[2]
                # ensure weight is atleast zero - negative weights can't exist in the problem definition
                if weight >= 0:
                    newState = copy.copy(node.state)
                    newDegrees = copy.copy(node.node_degrees)
                    newState.append(edge)
                    newDegrees[edge[0]] += 1
                    newDegrees[edge[1]] += 1
                    # init the new child node
                    child = Node(newState, node.cost + weight, newDegrees, valid_edges)
                    fringe.push(child)
                    
    # if no goal node is found we have exhausted the fringe and return None
    return None
                
solution = AStarSearch()
# write the solution to the appropriate variables
if solution is not None:
    tour = solution.toList()
    tour_length = solution.cost
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
print(time.time()-start_time)
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
    
    











    


